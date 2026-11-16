"""Fixed-pose, captured-colour gsplat trial. Work/cache never enters the pack.

CPU smoke uses a small differentiable reference renderer, NOT gsplat CUDA.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import holes

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mesh_check', HERE / 'splat_mesh_check.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
P = M.P
sys.modules.setdefault('run_splat', sys.modules[__name__])
DEPTH_REV = 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'
WEIGHTS = 'https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth'
SH_C0 = .28209479177387814
TRAINING = {
    'version': 'S3', 'max_steps': 15000, 'budget_seconds': 3300,
    'finish_reserve_seconds': 90, 'checkpoint_reserve_multiplier': 2., 'checkpoint_every': 250,
    'gaussian_cap': 2500000, 'scene_scale': 1., 'initial_opacity': .1,
    'strategy': {'absgrad': True, 'grow_grad2d': 8e-4, 'grow_scale3d': .02,
                 'prune_scale3d': .20, 'prune_opa': .005, 'refine_scale2d_stop_iter': 0,
                 'refine_start_iter': 500, 'refine_stop_iter': 8000, 'refine_every': 100,
                 'reset_every': 3000, 'pause_refine_after_reset': 414},
    'learning_rates': {'means': 1.6e-4, 'quats': 1e-3, 'scales': 2e-3,
                      'opacities': .02, 'sh0': .0025, 'shN': .0025 / 20},
    'means_final_lr': 1.6e-6, 'adam_eps': 1e-15, 'adam_betas': [.9, .999],
    'scale_bounds_metres': [.001, .15], 'opacity_logit_bounds': [-8., 8.],
    'position_limit_metres': None, 'freeze_means_steps': 0, 'camera_optimization': False,
    'sh_degree': 1, 'sh_activation_step': 1000,
    'exposure': {'lr': .001, 'adam_eps': 1e-8, 'adam_betas': [.9, .999],
                 'regularization': .001, 'log_gain_bound': .1,
                 'bias_bound': .03, 'gauge': 'per-channel zero mean over training panoramas',
                 'unit': 'panorama', 'heldout_walk_export': 'identity'},
    'loss': {'L1': .8, 'DSSIM': .2, 'mono_start': .02, 'mono_end': .005,
             'mono_decay_end': 3000, 'mesh_start': .02, 'mesh_decay_start': 5000,
             'mesh_decay_end': 12000},
    'structure': {'normal_tolerance_degrees': 15., 'raster_size': 256,
                  'depth_match_absolute': .001, 'depth_match_relative': .001,
                  'agreement_absolute_metres': .2, 'agreement_relative': .2,
                  'method': 'visible near-horizontal/vertical mesh triangles, Z up; orientation is a structural proxy, not semantics'},
    'checks': {'every': 2500, 'faces_each_split': 3, 'appearance': 'identity',
               'selection': 'first/middle/last panorama, faces 0/2/4'},
    'data_cache_faces': 4, 'seed': 20261114, 'rasterization': 'antialiased RGB+ED, packed, absgrad',
    'near_plane': .05, 'far_plane': 100.,
}


def make_strategy():
    # Lazy import keeps the small CPU reference independent of CUDA/gsplat.
    from gsplat import DefaultStrategy
    from gsplat.strategy.ops import duplicate, split

    class CappedStrategy(DefaultStrategy):
        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            score = state['grad2d'] / state['count'].clamp_min(1)
            available = max(0, TRAINING['gaussian_cap'] - len(score))
            candidates = torch.where(score > self.grow_grad2d)[0]
            if len(candidates) > available:
                candidates = candidates[torch.topk(score[candidates], available).indices]
            chosen = torch.zeros_like(score, dtype=torch.bool)
            chosen[candidates] = True
            small = params['scales'].exp().amax(-1) <= self.grow_scale3d * state['scene_scale']
            clones, splits = chosen & small, chosen & ~small
            nc, ns = int(clones.sum()), int(splits.sum())
            # Both operations add exactly one net Gaussian per selected parent.
            if nc:
                duplicate(params, optimizers, state, clones)
            if ns:
                split(params, optimizers, state,
                      torch.cat((splits, splits.new_zeros(nc))), revised_opacity=self.revised_opacity)
            self.refine_events += 1
            self.grown += nc + ns
            return nc, ns

    result = CappedStrategy(**TRAINING['strategy'])
    result.refine_events = result.grown = 0
    return result


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_save(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def xyz(p):
    return np.array([p[k] for k in 'xyz'], dtype=np.float64)


def camera(p, face):
    r = P.rotation(p['rotation']) @ P.BASES[face][[1, 2, 0]].T
    v = np.eye(4)
    v[:3, :3] = r.T
    v[:3, 3] = -r.T @ xyz(p['position'])
    return v


def image(home, p, f, size):
    a = cv2.imread(str(home / 'faces_2k' / p['id'] / f'face{f}.jpg'))
    if a is None:
        raise RuntimeError(f"Missing captured face {p['id']}/{f}")
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)


def rays(p, f, uv, size):
    local = np.c_[2 * (uv[:, 0] + .5) / size - 1,
                  2 * (uv[:, 1] + .5) / size - 1, np.ones(len(uv))]
    return local @ camera(p, f)[:3, :3]


def sample_depth(depth, uv, size):
    x = np.clip(((uv[:, 0] + .5) * depth.shape[1] / size).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(((uv[:, 1] + .5) * depth.shape[0] / size).astype(int), 0, depth.shape[0] - 1)
    return depth[y, x].astype(np.float64)


class Trial:
    def __init__(self, args):
        self.a = args
        self.home = args.pack / args.sid
        self.work = args.root / (('smoke-' if args.smoke else '') + args.sid)
        self.out = args.root / 'send_back' / self.work.name
        self.proof = self.out / 'proof'
        self.cache = self.work / 'depth'
        for p in (self.work, self.out, self.proof, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        self.all = json.loads((self.home / 'poses.json').read_text())['panoramas']
        # Panorama split BEFORE any matching, depth or point initialization.
        self.test = self.all[7::8]
        self.train = [p for i, p in enumerate(self.all) if i % 8 != 7]
        if args.smoke:
            # Exactly two real panoramas, one training and one held-out.
            self.train = self.train[:1]
            c = xyz(self.train[0]['position'])
            self.test = [min(self.test, key=lambda p: np.linalg.norm(xyz(p['position']) - c))]
        self.panos = self.train + self.test
        self.device = 'cpu' if args.smoke else 'cuda'
        self.size = 64 if args.smoke else 2048
        self.iterations = 50 if args.smoke else args.iterations
        self.target = 256 if args.smoke else args.points
        self.rng = np.random.default_rng(TRAINING['seed'])
        torch.manual_seed(TRAINING['seed'])
        if args.smoke:
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        elif not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; use --smoke only for CPU plumbing proof.')
        config = dict(sid=args.sid, smoke=args.smoke, image_size=self.size,
                      iterations=self.iterations, target_points=self.target,
                      training=TRAINING,
                      depth_input_size=518, depth_revision=DEPTH_REV,
                      sources={p.name: digest(p) for p in [Path(__file__), HERE / 'web.js', HERE / 'web.html',
                               HERE / 'walk_path.py', HERE / 'holes.py', HERE / 'splat_mesh_check.py', HERE / 'splat_pose_check.py',
                               HERE / 'reuse_stages.py', HERE / 'retrain_smoke.py',
                               HERE.parent / 'setup_linux.sh', HERE.parent / 'run_splat.sh',
                               self.home / 'poses.json', self.home / 'rooms.json', self.home / 'mesh/mesh_500k.obj']},
                      captures={f"{p['id']}/{f}": digest(self.home / 'faces_2k' / p['id'] / f'face{f}.jpg')
                                for p in self.panos for f in range(6)})
        if os.environ.get('HP_SPLAT_DEPTH') == 'mast3r':
            import mast3r_depth
            config.update(depth_method='mast3r', depth_input_size=512,
                          depth_revision=mast3r_depth.REVISION, mast3r=mast3r_depth.settings())
            config['sources']['mast3r_depth.py'] = digest(HERE / 'mast3r_depth.py')
        imported = self.work / 'reuse_manifest.json'
        if imported.exists():
            from reuse_stages import validate_import
            validate_import(self.work, config)
            config['reused_stages'] = digest(imported)
        self.config = config
        self.sh_degree = 0
        cfg = self.work / 'config.json'
        if cfg.exists() and json.loads(cfg.read_text()) != config:
            raise RuntimeError(f'Configuration changed. Preserve old work then use a new HP_SPLAT_ROOT. See {cfg}')
        save_json(cfg, config)
        shutil.copy2(cfg, self.proof / 'config.json')
        save_json(self.proof / 'split.json', {'policy': 'API order, 1-based every eighth held out',
                  'training': [p['id'] for p in self.train], 'held_out': [p['id'] for p in self.test],
                  'smoke_override': args.smoke})
        self.vertices, self.triangles = M.read_obj(self.home / 'mesh/mesh_500k.obj')

    def stage(self, name, action):
        marker = self.work / (name + '.done.json')
        if marker.exists():
            saved = json.loads(marker.read_text())
            if all(Path(p).exists() and digest(p) == h for p, h in saved['outputs'].items()):
                print('Resume: completed', name, flush=True)
                return
            print('Rebuilding damaged stage', name, flush=True)
            if name in ('nadir', 'depth', 'calibration', 'fusion', 'surfaces', 'train'):
                raise RuntimeError(f'{name} outputs differ from recorded hashes. Preserve work and use a new HP_SPLAT_ROOT; refusing stale downstream results.')
        save_json(self.out / 'status.json', {'stage': name, 'state': 'running', 'utc': time.time()})
        start = time.monotonic()
        outputs = action()
        save_json(marker, {'seconds': time.monotonic() - start,
                          'outputs': {str(p): digest(p) for p in outputs}})
        print(f'Completed {name}: {time.monotonic() - start:.1f}s', flush=True)

    def depth_path(self, p, f):
        return self.cache / f"{p['id']}_{f}.npy"

    def depth(self):
        if os.environ.get('HP_SPLAT_DEPTH') == 'mast3r':
            from mast3r_depth import generate
            return generate(self)
        cache = Path(os.environ.get('HP_SPLAT_DEPTH_CACHE', str(Path.home() / '.cache/hp-splat/depth')))
        code = cache / ('Depth-Anything-V2-' + DEPTH_REV)
        if not (code / 'metric_depth/depth_anything_v2/dpt.py').exists():
            import tarfile
            cache.mkdir(parents=True, exist_ok=True)
            archive = cache / 'source.tar.gz'
            urllib.request.urlretrieve('https://codeload.github.com/DepthAnything/Depth-Anything-V2/tar.gz/' + DEPTH_REV, archive)
            with tarfile.open(archive) as tar:
                # Only source code; no datasets, links, binaries or arbitrary paths.
                for member in tar.getmembers():
                    parts = Path(member.name).parts
                    if len(parts) > 2 and parts[1] == 'metric_depth' and member.isfile() and '..' not in parts:
                        target = cache.joinpath(*parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src, target.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
            archive.unlink()
        weight = cache / 'depth_anything_v2_metric_hypersim_vits.pth'
        if not weight.exists():
            tmp = weight.with_suffix('.download')
            urllib.request.urlretrieve(WEIGHTS, tmp)
            tmp.replace(weight)
        sys.path.insert(0, str(code / 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
        model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
        model.eval().to(self.device)
        outputs = []
        for pi, p in enumerate(self.panos):
            for f in range(6):
                path = self.depth_path(p, f)
                if path.exists():
                    d = np.load(path, mmap_mode='r')
                    if d.shape != (2048, 2048) or not np.isfinite(d).all() or d.min() <= 0:
                        raise RuntimeError(f'Invalid cached depth: {path}')
                else:
                    raw = image(self.home, p, f, 2048)
                    # Official infer_image auto-selects MPS. Explicit tensor transfer keeps CPU smoke on CPU.
                    tensor, shape = model.image2tensor(raw, 518)
                    with torch.no_grad():
                        d = model(tensor.to(self.device))
                        d = F.interpolate(d[:, None], shape, mode='bilinear', align_corners=True)[0, 0]
                    d = d.cpu().numpy().astype(np.float16)
                    if not np.isfinite(d).all() or d.min() <= 0:
                        raise RuntimeError('Non-finite metric depth')
                    tmp = path.with_suffix('.tmp')
                    with tmp.open('wb') as out:
                        np.save(out, d)
                    tmp.replace(path)
                outputs.append(path)
            print(f'Depth {pi + 1}/{len(self.panos)} panoramas (6 native 2k outputs each)', flush=True)
        save_json(self.proof / 'depth_model.json', {
            'model': 'Depth Anything V2 Metric Hypersim Small (indoor, max 20m)',
            'source_revision': DEPTH_REV, 'weights_sha256': digest(weight),
            'native_output': [2048, 2048], 'network_input': 518,
            'note': '2k output from 518 network inference; geometry is estimated, no RGB synthesis',
            'device': self.device, 'pano_count': len(self.panos)})
        del model
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return outputs + [self.proof / 'depth_model.json']

    def calibration(self):
        # Match train-only faces. Triangulated optical depths are in metres because
        # translation comes from the supplied metric panorama positions.
        if self.a.smoke:
            save_json(self.proof / 'scale.json', {'scale': 1., 'validated': False,
                      'reason': 'Two-pano smoke has ONE training pano; held-out pano cannot calibrate geometry.'})
            return [self.proof / 'scale.json']
        sift = cv2.SIFT_create(nfeatures=2500)
        features = {}
        def feat(p, f):
            key = (p['id'], f)
            if key not in features:
                im = image(self.home, p, f, 768)
                k, d = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
                features[key] = (k, d)
            return features[key]
        centres = np.array([xyz(p['position']) for p in self.train])
        dist = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
        pairs = sorted({tuple(sorted((i, int(j)))) for i in range(len(centres))
                        for j in np.argsort(dist[i])[1:4] if .4 < dist[i, j] < 4.5})
        rows = []
        for pi, (i, j) in enumerate(pairs):
            p, q = self.train[i], self.train[j]
            t = centres[j] - centres[i]
            for f in range(1, 5):
                for g in range(1, 5):
                    k1, d1 = feat(p, f); k2, d2 = feat(q, g)
                    if d1 is None or d2 is None or min(len(d1), len(d2)) < 2:
                        continue
                    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
                    good = [m for m, n in matches if m.distance < .7 * n.distance]
                    if len(good) < 16:
                        continue
                    a = np.float64([k1[m.queryIdx].pt for m in good])
                    b = np.float64([k2[m.trainIdx].pt for m in good])
                    r1, r2 = rays(p, f, a, 768), rays(q, g, b, 768)
                    # Batched least squares for r1*z1 - r2*z2 = known baseline.
                    A = np.stack([r1, -r2], axis=-1)
                    z = np.linalg.solve(A.transpose(0, 2, 1) @ A + np.eye(2)[None] * 1e-9,
                                        A.transpose(0, 2, 1) @ np.broadcast_to(t, r1.shape)[..., None])[..., 0]
                    gap = np.linalg.norm(np.einsum('nij,nj->ni', A, z) - t, axis=1)
                    cos = np.sum(r1 * r2, axis=1) / np.linalg.norm(r1, axis=1) / np.linalg.norm(r2, axis=1)
                    keep = (gap < .025) & (z.min(1) > .3) & (z.max(1) < 18) & (cos < .998)
                    dp = sample_depth(np.load(self.depth_path(p, f)), a, 768)
                    dq = sample_depth(np.load(self.depth_path(q, g)), b, 768)
                    for n in np.flatnonzero(keep):
                        rows.append({'pair': [p['id'], q['id']], 'faces': [f, g],
                                     'metric_z': z[n].tolist(), 'pred_z': [dp[n], dq[n]],
                                     'rays': [r1[n].tolist(), r2[n].tolist()], 'baseline': t.tolist()})
            if pi % 10 == 0:
                print(f'Calibrate pairs {pi + 1}/{len(pairs)}, accepted {len(rows)}', flush=True)
        if len(rows) < 100:
            save_json(self.proof / 'scale_failure.json', {'matches': len(rows), 'reason': 'Need >=100 triangulated training matches'})
            raise RuntimeError('Insufficient scale correspondences. See proof/scale_failure.json; no uncalibrated full training.')
        # Split by PAIR, never by correlated keypoints from the same pair.
        pair_ids = sorted({tuple(r['pair']) for r in rows})
        validation = set(pair_ids[::5])
        fit = [r for r in rows if tuple(r['pair']) not in validation]
        val = [r for r in rows if tuple(r['pair']) in validation]
        if len(fit) < 50 or len(val) < 20:
            raise RuntimeError('Not enough independent calibration/validation pairs')
        ratios = np.array([np.array(r['metric_z']) / r['pred_z'] for r in fit]).ravel()
        scale = float(np.median(ratios))
        if os.environ.get('HP_SPLAT_DEPTH') == 'mast3r':
            print(f'S4 known-pose MASt3R calibration global scale={scale:.6f} (expected near 1)', flush=True)
        residual, depth_error, baseline_m = [], [], []
        for r in val:
            predicted = scale * (np.array(r['rays'][0]) * r['pred_z'][0] - np.array(r['rays'][1]) * r['pred_z'][1])
            baseline = np.array(r['baseline'])
            err = float(np.linalg.norm(predicted - baseline))
            baseline_m.append(err)
            residual.append(err / np.linalg.norm(baseline))
            depth_error.extend(abs(scale * np.array(r['pred_z']) / r['metric_z'] - 1))
        result = {'scale': scale, 'validated': True, 'matches_fit': len(fit), 'matches_validation': len(val),
                  'fit_pairs': len(pair_ids) - len(validation), 'validation_pairs': len(validation),
                  'validation_baseline_vector_relative_median': float(np.median(residual)),
                  'validation_baseline_vector_relative_p90': float(np.quantile(residual, .9)),
                  'validation_baseline_vector_error_metres_median': float(np.median(baseline_m)),
                  'validation_depth_relative_median': float(np.median(depth_error)),
                  'method': 'SIFT ratio .7; known-pose triangulation, ray gap<.025m, parallax>3.6deg; pair-disjoint validation',
                  'limitations': 'Estimated global scale; furniture, mirrors, repeated textures and local depth distortions remain.'}
        save_json(self.proof / 'scale.json', result)
        if not .1 < scale < 10 or result['validation_depth_relative_median'] > .5:
            raise RuntimeError('Metric scale failed sanity gate (factor .1..10, validation depth median<=50%). See scale.json.')
        return [self.proof / 'scale.json']

    def fusion(self):
        scale = json.loads((self.proof / 'scale.json').read_text())['scale']
        count = math.ceil(self.target / (len(self.train) * 6))
        positions, colours, sizes = [], [], []
        mesh_cache = self.work / 'mesh_depth'
        mesh_cache.mkdir(exist_ok=True)
        M.W = 128 if self.a.smoke else 256
        for pi, p in enumerate(self.train):
            # All six mesh depths are reused by every training iteration.
            mesh_paths = [mesh_cache / f"{p['id']}_{f}.npy" for f in range(6)]
            if not all(x.exists() for x in mesh_paths):
                for path, d in zip(mesh_paths, M.depths(self.vertices, self.triangles, p)):
                    tmp = path.with_suffix('.tmp')
                    with tmp.open('wb') as stream:
                        np.save(stream, np.where(np.isfinite(d), d, 0).astype(np.float32))
                    tmp.replace(path)
            for f in range(6):
                d = np.load(self.depth_path(p, f)).astype(np.float32) * scale
                uv = self.rng.integers(4, 2044, size=(count * 2, 2))
                z = d[uv[:, 1], uv[:, 0]]
                # Exclude depth discontinuities; estimated holes remain holes.
                grad = abs(d[uv[:, 1], uv[:, 0] + 2] - d[uv[:, 1], uv[:, 0] - 2])
                grad += abs(d[uv[:, 1] + 2, uv[:, 0]] - d[uv[:, 1] - 2, uv[:, 0]])
                captured = holes.valid_pixels(self, p, f, 2048)[uv[:, 1], uv[:, 0]]
                keep = np.flatnonzero(captured & (z > .25) & (z < 19.5 * scale) & (grad < np.maximum(.08, .05 * z)))[:count]
                uv, z = uv[keep], z[keep]
                positions.append(xyz(p['position']) + rays(p, f, uv, 2048) * z[:, None])
                colours.append(image(self.home, p, f, 2048)[uv[:, 1], uv[:, 0], ::-1].copy() / 255.)
                sizes.append(np.clip(z / 1024 * 1.5, .002, .035))
            print(f'Fuse {pi + 1}/{len(self.train)} training panoramas', flush=True)
        means = np.concatenate(positions).astype(np.float32)
        colors = np.concatenate(colours).astype(np.float32)
        scales = np.concatenate(sizes).astype(np.float32)
        # Metric voxel fusion prevents coincident points from different captured faces.
        _, ids = np.unique(np.floor(means / .008).astype(np.int32), axis=0, return_index=True)
        self.rng.shuffle(ids)
        ids = ids[:self.target]
        if not len(ids):
            raise RuntimeError('No finite points survived depth fusion')
        path = self.work / 'initial.npz'
        selected = means[ids].copy()
        if len(ids) >= 4:
            # Footprint alone is too small for a sparse 2M-point cloud. Use local
            # point spacing as in common 3DGS initialization to avoid pinholes.
            index = cv2.flann_Index(selected, dict(algorithm=1, trees=4))
            _, distances = index.knnSearch(selected, 4, params={'checks': 32})
            spacing = np.sqrt(np.maximum(distances[:, 1:].mean(1), 1e-8))
            initial_scale = np.maximum(scales[ids], np.clip(spacing * .7, .003, .08))
        else:
            initial_scale = scales[ids]
        np.savez(path, means=selected, colors=colors[ids], scales=initial_scale)
        save_json(self.proof / 'fusion.json', {'points': len(ids), 'target': self.target, 'voxel_metres': .008,
                  'source': 'scaled metric depth, known poses, captured 2k RGB; TRAINING panos only',
                  'depth_edge_rejection': True, 'cross_view_depth_consistency': False,
                  'gaussian_size': 'max(pixel footprint, .7 * local 3-neighbour RMS distance); clamped spacing term .003..08m',
                  'mesh': 'simple_planes, untextured; depth regularizer and collision only',
                  'people_masks': False})
        return [path, self.proof / 'fusion.json'] + list(mesh_cache.glob('*.npy'))

    def parameters(self):
        init = np.load(self.work / 'initial.npz')
        n = len(init['means'])
        def param(a):
            return torch.nn.Parameter(torch.as_tensor(a, dtype=torch.float32, device=self.device))
        return {'means': param(init['means']), 'quats': param(np.tile([1., 0, 0, 0], (n, 1))),
                'scales': param(np.repeat(np.log(init['scales'])[:, None], 3, axis=1)),
                'opacities': param(np.full(n, math.log(TRAINING['initial_opacity'] / (1 - TRAINING['initial_opacity'])))),
                'sh0': param(((init['colors'] - .5) / SH_C0)[:, None, :]),
                'shN': param(np.zeros((n, 3, 3)))}

    def render(self, params, view, width, height, f=None, with_info=False):
        f = f or width / 2
        v = torch.as_tensor(view, device=self.device, dtype=torch.float32)
        if not self.a.smoke:
            from gsplat import rasterization
            k = torch.tensor([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]], device=self.device)
            rgbd, alpha, info = rasterization(params['means'], F.normalize(params['quats'], dim=-1),
                       params['scales'].exp(), params['opacities'].sigmoid(), torch.cat((params['sh0'], params['shN']), dim=1),
                       v[None], k[None], width, height, packed=True, render_mode='RGB+ED',
                       rasterize_mode='antialiased', near_plane=TRAINING['near_plane'], far_plane=TRAINING['far_plane'],
                       sh_degree=self.sh_degree, absgrad=with_info)
            result = (rgbd[0, ..., :3], rgbd[0, ..., 3], alpha[0, ..., 0])
            return result + (info,) if with_info else result
        # Small CPU reference: depth-sorted front-to-back isotropic Gaussians.
        # It proves loss/checkpoint/data plumbing, not the gsplat rasterization path.
        xyzc = params['means'] @ v[:3, :3].T + v[:3, 3]
        order = torch.argsort(xyzc[:, 2].detach())
        xyzc = xyzc[order]
        z = xyzc[:, 2].clamp_min(.05)
        uv = xyzc[:, :2] / z[:, None] * f
        uv = uv + torch.tensor([(width - 1) / 2, (height - 1) / 2], device=self.device)
        yy, xx = torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing='ij')
        delta = (xx[None] - uv[:, 0, None, None]) ** 2 + (yy[None] - uv[:, 1, None, None]) ** 2
        sigma = (params['scales'].exp().mean(-1)[order] * f / z).clamp_min(.5)
        alpha = torch.exp(-delta / (2 * sigma[:, None, None] ** 2)) * params['opacities'].sigmoid()[order, None, None]
        alpha = (alpha * (xyzc[:, 2] > .05)[:, None, None]).clamp(max=.99)
        trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1 - alpha + 1e-8], dim=0), dim=0)[:-1]
        weight = trans * alpha
        cover = weight.sum(0)
        colors = .5 + SH_C0 * params['sh0'][:, 0]
        if self.sh_degree:
            centre = -v[:3, :3].T @ v[:3, 3]
            direction = F.normalize(params['means'] - centre, dim=-1)
            basis = .4886025119029199 * torch.stack((-direction[:, 1], direction[:, 2], -direction[:, 0]), dim=-1)
            colors = colors + (params['shN'] * basis[..., None]).sum(1)
        rgb = torch.einsum('nhw,nc->hwc', weight, colors.clamp_min(0)[order])
        depth = (weight * z[:, None, None]).sum(0) / cover.clamp_min(1e-6)
        result = (rgb, depth, cover)
        return result + ({},) if with_info else result

    def structure_mask(self, p, f):
        folder = self.work / 'structure_mask'
        folder.mkdir(exist_ok=True)
        path = folder / f"{p['id']}_{f}.npy"
        if not path.exists():
            tri = self.vertices[self.triangles]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            length = np.linalg.norm(normal, axis=1)
            nz = np.abs(normal[:, 2]) / np.maximum(length, 1e-12)
            cfg = TRAINING['structure']
            angle = math.radians(cfg['normal_tolerance_degrees'])
            structural = (length > 1e-12) & ((nz <= math.sin(angle)) | (nz >= math.cos(angle)))
            # Match the visible structural depth against the full mesh z-buffer;
            # an occluded wall cannot impose its depth on foreground furniture.
            old_size = M.W
            M.W = 128 if self.a.smoke else cfg['raster_size']
            try:
                depths = M.depths(self.vertices, self.triangles[structural], p)
            finally:
                M.W = old_size
            for face, structural_depth in enumerate(depths):
                mesh = np.load(self.work / 'mesh_depth' / f"{p['id']}_{face}.npy")
                keep = (mesh > 0) & np.isfinite(structural_depth)
                difference = np.full_like(structural_depth, np.inf)
                np.subtract(structural_depth, mesh, out=difference, where=keep)
                keep &= np.abs(difference) <= np.maximum(cfg['depth_match_absolute'], cfg['depth_match_relative'] * mesh)
                dst = folder / f"{p['id']}_{face}.npy"
                tmp = dst.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    np.save(stream, keep)
                tmp.replace(dst)
        return cv2.resize(np.load(path).astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(bool)

    def fixed_faces(self, split):
        panos = self.train if split == 'train' else self.test
        return [(panos[i], f) for i, f in zip((0, len(panos) // 2, len(panos) - 1), (0, 2, 4))]

    @torch.no_grad()
    def fixed_psnr(self, params):
        result = {}
        for split in ('train', 'heldout'):
            rows = []
            for p, f in self.fixed_faces(split):
                target = torch.as_tensor(image(self.home, p, f, self.size)[..., ::-1].copy(), device=self.device).float() / 255
                rgb, _, _ = self.render(params, camera(p, f), self.size, self.size)
                mask = torch.as_tensor(holes.valid_pixels(self, p, f, self.size), device=self.device)
                mse = float((rgb[mask] - target[mask]).square().mean())
                if not math.isfinite(mse):
                    raise RuntimeError('Non-finite fixed-face PSNR')
                rows.append({'pano': p['id'], 'face': f, 'PSNR': -10 * math.log10(max(mse, 1e-12))})
            result[split] = {'PSNR': float(np.mean([r['PSNR'] for r in rows])), 'faces': rows}
        return result

    def train_model(self, stop_after=None):
        self._budget_timer = None
        try:
            return self._train_model(stop_after)
        finally:
            if self._budget_timer is not None:
                self._budget_timer.cancel()

    def _train_model(self, stop_after):
        tick = time.monotonic()
        cp = self.work / 'checkpoint.pt'
        state = torch.load(cp, map_location=self.device, weights_only=False) if cp.exists() else None
        elapsed = state['seconds'] if state else 0.
        budget = TRAINING['budget_seconds']
        def hard_stop():
            print('Hard training budget expired; last atomic checkpoint preserved; train stage incomplete.', file=sys.stderr, flush=True)
            os._exit(124)
        remaining = budget - elapsed - (time.monotonic() - tick)
        if remaining <= 0:
            raise RuntimeError('Checkpoint exhausted training budget; inspect it before starting a new root')
        self._budget_timer = threading.Timer(remaining, hard_stop)
        self._budget_timer.daemon = True
        self._budget_timer.start()
        if state:
            if state.get('version') != TRAINING['version'] or state['config_digest'] != digest(self.work / 'config.json'):
                raise RuntimeError('Checkpoint provenance/schema mismatch; use a new root')
            # Construct at the SAVED size before creating/loading Adam state.
            params = {k: torch.nn.Parameter(v) for k, v in state['params'].items()}
        else:
            params = self.parameters()
        if not 0 < len(params['means']) <= TRAINING['gaussian_cap']:
            raise RuntimeError('Gaussian count is outside the configured cap')
        optimizers = {k: torch.optim.Adam([v], lr=TRAINING['learning_rates'][k],
                      eps=TRAINING['adam_eps'], betas=tuple(TRAINING['adam_betas'])) for k, v in params.items()}
        gamma = (TRAINING['means_final_lr'] / TRAINING['learning_rates']['means']) ** (1 / TRAINING['max_steps'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=gamma)
        exposure = {k: torch.nn.Parameter(torch.zeros((len(self.train), 3), device=self.device)) for k in ('log_gain', 'bias')}
        exp_optimizer = torch.optim.Adam(list(exposure.values()), lr=TRAINING['exposure']['lr'],
                        eps=TRAINING['exposure']['adam_eps'], betas=tuple(TRAINING['exposure']['adam_betas']))
        strategy = None if self.a.smoke else make_strategy()
        strategy_state = strategy.initialize_state(scene_scale=TRAINING['scene_scale']) if strategy else None
        series = {'train': [], 'heldout': []}
        start_step, step, last_loss, peak = 0, 0, None, 0
        if state:
            for k, opt in optimizers.items():
                opt.load_state_dict(state['optimizers'][k])
            scheduler.load_state_dict(state['scheduler'])
            with torch.no_grad():
                for k, v in exposure.items():
                    v.copy_(state['exposure'][k])
            exp_optimizer.load_state_dict(state['exposure_optimizer'])
            strategy_state = state['strategy_state']
            if strategy:
                strategy.refine_events, strategy.grown = state['refine_events'], state['grown']
            start_step = step = state['step']
            series, last_loss, peak = state['psnr_series'], state['last_loss'], state['peak_gpu_allocated_bytes']
            torch.set_rng_state(state['rng_cpu'].cpu())
            if self.device == 'cuda':
                torch.cuda.set_rng_state(state['rng_cuda'].cpu())
            print('Resume training at iteration', start_step, 'Gaussians', len(params['means']), flush=True)
        del state
        if strategy:
            strategy.check_sanity(params, optimizers)
        self.sh_degree = int(step >= TRAINING['sh_activation_step'])
        scale = json.loads((self.proof / 'scale.json').read_text())['scale']
        from functools import lru_cache
        @lru_cache(maxsize=TRAINING['data_cache_faces'])
        def data(i):
            p, f = self.train[i // 6], i % 6
            rgb = image(self.home, p, f, self.size)[..., ::-1].copy().astype(np.float32) / 255
            d = cv2.resize(np.load(self.depth_path(p, f)).astype(np.float32) * scale, (self.size, self.size))
            mesh = cv2.resize(np.load(self.work / 'mesh_depth' / f"{p['id']}_{f}.npy"),
                              (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            return [torch.as_tensor(x, device=self.device) for x in
                    (rgb, d, mesh, holes.valid_pixels(self, p, f, self.size), self.structure_mask(p, f))]
        def seconds():
            return elapsed + time.monotonic() - tick
        def check():
            scores = self.fixed_psnr(params)
            for split in series:
                series[split].append({'step': step, **scores[split]})
            print('Fixed faces', step, 'train', scores['train']['PSNR'], 'heldout', scores['heldout']['PSNR'], flush=True)
        save_seconds = 0.
        def checkpoint(reason):
            nonlocal save_seconds
            save_tick = time.monotonic()
            atomic_save(cp, {'version': TRAINING['version'], 'config_digest': digest(self.work / 'config.json'),
                'params': {k: v.detach() for k, v in params.items()},
                'optimizers': {k: opt.state_dict() for k, opt in optimizers.items()},
                'scheduler': scheduler.state_dict(), 'strategy_state': strategy_state,
                'refine_events': strategy.refine_events if strategy else 0, 'grown': strategy.grown if strategy else 0,
                'exposure': {k: v.detach() for k, v in exposure.items()}, 'exposure_optimizer': exp_optimizer.state_dict(),
                'step': step, 'seconds': seconds(), 'sh_degree': self.sh_degree, 'stop_reason': reason,
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state() if self.device == 'cuda' else None,
                'psnr_series': series, 'last_loss': last_loss, 'peak_gpu_allocated_bytes': current_peak()})
            save_seconds = max(save_seconds, time.monotonic() - save_tick)
        def current_peak():
            return max(peak, torch.cuda.max_memory_allocated() if self.device == 'cuda' else 0)
        limit = min(self.iterations, TRAINING['max_steps'], stop_after if stop_after is not None else TRAINING['max_steps'])
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        reason = 'requested_steps' if limit == self.iterations else 'smoke_boundary'
        recent_step_seconds = 0.
        for index in range(start_step, limit):
            if seconds() + max(TRAINING['finish_reserve_seconds'],
                               TRAINING['checkpoint_reserve_multiplier'] * save_seconds) + recent_step_seconds >= budget:
                reason = 'time_budget'
                break
            step_tick = time.monotonic()
            step = index + 1
            self.sh_degree = int(step >= TRAINING['sh_activation_step'])
            i = int(torch.randint(len(self.train) * 6, (1,)))
            p, f = self.train[i // 6], i % 6
            target, depth, mesh, captured, structure = data(i)
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            exp_optimizer.zero_grad(set_to_none=True)
            rgb, z, alpha, info = self.render(params, camera(p, f), self.size, self.size, with_info=True)
            if strategy:
                strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            appearance = (rgb * exposure['log_gain'][i // 6].exp() + exposure['bias'][i // 6]).clamp(0, 1)
            valid = captured & (depth > .25) & (alpha > .1)
            zero = rgb.sum() * 0
            loss_depth = ((z[valid] - depth[valid]).abs() / depth[valid]).mean() if valid.any() else zero
            cfg = TRAINING['structure']
            valid_mesh = valid & structure & (mesh > .25) & ((depth - mesh).abs() < torch.maximum(
                mesh * cfg['agreement_relative'], mesh.new_tensor(cfg['agreement_absolute_metres'])))
            loss_mesh = ((z[valid_mesh] - mesh[valid_mesh]).abs() / mesh[valid_mesh]).mean() if valid_mesh.any() else zero
            weights = TRAINING['loss']
            mono_weight = weights['mono_start'] + (weights['mono_end'] - weights['mono_start']) * min(index / weights['mono_decay_end'], 1)
            mesh_weight = weights['mesh_start'] * max(0., min(1., (weights['mesh_decay_end'] - index) /
                          (weights['mesh_decay_end'] - weights['mesh_decay_start'])))
            loss = weights['L1'] * (appearance[captured] - target[captured]).abs().mean()
            loss = loss + weights['DSSIM'] * (1 - ssim(appearance, target, captured)) + mono_weight * loss_depth + mesh_weight * loss_mesh
            loss = loss + TRAINING['exposure']['regularization'] * sum(v.square().mean() for v in exposure.values())
            if not torch.isfinite(loss):
                raise RuntimeError('Training loss is non-finite; previous checkpoint preserved')
            loss.backward()
            if strategy:
                strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
            for opt in optimizers.values():
                opt.step()
            exp_optimizer.step()
            scheduler.step()
            with torch.no_grad():
                params['scales'].clamp_(*(math.log(x) for x in TRAINING['scale_bounds_metres']))
                params['opacities'].clamp_(*TRAINING['opacity_logit_bounds'])
                for k, v in exposure.items():
                    v.sub_(v.mean(0, keepdim=True))
                    bound = TRAINING['exposure']['log_gain_bound' if k == 'log_gain' else 'bias_bound']
                    v.mul_((bound / v.abs().amax(0, keepdim=True).clamp_min(bound)).clamp_max(1))
            if not 0 < len(params['means']) <= TRAINING['gaussian_cap']:
                raise RuntimeError('Strategy produced an invalid Gaussian count')
            last_loss = float(loss.detach())
            if step % 50 == 0 or step == limit:
                print(f'Train {step}/{self.iterations}, loss {last_loss:.5f}, N {len(params["means"])}', flush=True)
            # Release graph/info before metrics, saving, or the next topology change.
            del rgb, z, alpha, info, appearance, loss, loss_depth, loss_mesh, zero
            if step % TRAINING['checks']['every'] == 0:
                check()
            if step % TRAINING['checkpoint_every'] == 0:
                checkpoint('running')
            recent_step_seconds = time.monotonic() - step_tick
        if not series['train'] or series['train'][-1]['step'] != step:
            check()
        checkpoint(reason)
        report = {'iterations': step, 'requested_iterations': self.iterations, 'stop_reason': reason, 'seconds': seconds(),
                  'resolution': self.size, 'gaussians': len(params['means']), 'settings': TRAINING,
                  'backend': 'CPU reference smoke' if self.a.smoke else 'gsplat 1.4.0 CUDA antialiased RGB+ED',
                  'peak_gpu_allocated_bytes': current_peak(), 'position_limit_metres': None, 'camera_optimization': False,
                  'exposure_compensation': True, 'densification': bool(strategy), 'sh_degree': self.sh_degree,
                  'refine_events': strategy.refine_events if strategy else 0, 'grown': strategy.grown if strategy else 0,
                  'psnr_series': series, 'last_loss': last_loss, 'max_checkpoint_seconds': save_seconds,
                  'nadir_mask': 'all losses exclude measured disc +15px; SSIM excludes overlapping 11x11 windows'}
        save_json(self.proof / 'training.json', report)
        return [cp, self.proof / 'training.json']

    def trained(self):
        state = torch.load(self.work / 'checkpoint.pt', map_location=self.device, weights_only=False)
        self.sh_degree = state['sh_degree']
        return state['params']

    @torch.no_grad()
    def evaluate(self):
        params = self.trained()
        results = []
        worst = []
        for p in self.test:
            for f in range(6):
                target = torch.tensor(image(self.home, p, f, self.size)[..., ::-1].copy() / 255., device=self.device, dtype=torch.float32)
                rgb, _, alpha = self.render(params, camera(p, f), self.size, self.size)
                captured = torch.tensor(holes.valid_pixels(self, p, f, self.size), device=self.device)
                mse = float(((rgb[captured] - target[captured]) ** 2).mean())
                score = -10 * math.log10(max(mse, 1e-12))
                results.append({'pano': p['id'], 'face': f, 'PSNR': score, 'SSIM': float(ssim(rgb, target, captured)),
                                'captured_fraction': float(captured.float().mean()),
                                'alpha_coverage_above_.1': float((alpha > .1).float().mean())})
                canvas = np.concatenate([target.cpu().numpy(), rgb.clamp(0, 1).cpu().numpy()], axis=1)
                worst.append((score, (canvas * 255).astype(np.uint8)))
                worst = sorted(worst, key=lambda x: x[0])[:6]
            print('Held-out evaluation', p['id'], flush=True)
        outputs = []
        for i, (_, a) in enumerate(worst):
            path = self.proof / f'heldout_worst_{i:02}.jpg'
            cv2.imwrite(str(path), a[..., ::-1]); outputs.append(path)
        save_json(self.proof / 'evaluation.json', {'resolution': self.size, 'faces': results,
                  'PSNR': float(np.mean([r['PSNR'] for r in results])),
                  'SSIM': float(np.mean([r['SSIM'] for r in results])),
                  'metric': 'nadir-masked captured RGB, sRGB, black background; SSIM only fully valid 11x11 windows, sigma1.5',
                  'quality_decision': 'Needs human inspection of held-out comparisons and walk; no automatic retry.'})
        return outputs + [self.proof / 'evaluation.json']

    @torch.no_grad()
    def walk(self):
        from walk_path import make_path
        vertices, triangles = M.read_obj(self.work / 'collision.obj')
        controls, report = make_path(self.home, self.panos, vertices, triangles)
        save_json(self.proof / 'walk_path.json', report)
        params = self.trained()
        width, height = (96, 64) if self.a.smoke else (1280, 720)
        fps = 24
        frames = 480
        # Smooth speed along a piecewise linear collision-checked path. No frame interpolation.
        length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(controls, axis=0), axis=1))]
        if length[-1] < .01:
            raise RuntimeError('No traversable walk path')
        def point(t):
            return np.array([np.interp(t, length, controls[:, k]) for k in range(3)])
        path = self.proof / 'walk.mp4'
        tmp = self.proof / 'walk.partial.mp4'
        process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '22',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
        candidates = []
        tick = time.monotonic()
        try:
            for i, t in enumerate(np.linspace(0, length[-1], frames)):
                origin = point(t)
                forward = point(min(length[-1], t + .3)) - point(max(0, t - .05))
                forward /= max(np.linalg.norm(forward), 1e-9)
                right = np.cross(forward, [0., 0., 1.]); right /= max(np.linalg.norm(right), 1e-9)
                down = np.cross(forward, right)
                r = np.column_stack([right, down, forward])
                view = np.eye(4); view[:3, :3] = r.T; view[:3, 3] = -r.T @ origin
                rgb, _, alpha = self.render(params, view, width, height, f=width / 2)
                a = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                process.stdin.write(a.tobytes())
                if i % 24 == 0:
                    candidates.append((float((alpha < .1).float().mean()), i, a))
                if i % 120 == 0:
                    print(f'Walk {i}/{frames}', flush=True)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed')
        tmp.replace(path)
        outputs = [path, self.proof / 'walk_path.json']
        for _, i, a in sorted(candidates, key=lambda x: x[0], reverse=True)[:6]:
            still = self.proof / f'walk_worst_{i:03}.jpg'
            cv2.imwrite(str(still), a[..., ::-1]); outputs.append(still)
        save_json(self.proof / 'walk_render.json', {'seconds_video': 20, 'fps_video': fps, 'frames': frames,
                  'resolution': [width, height], 'render_seconds': time.monotonic() - tick,
                  'worst_frame_selection': 'six sampled frames with highest uncovered alpha fraction, not a human quality rating'})
        return outputs + [self.proof / 'walk_render.json']

    def write_ply(self, ply):
        checkpoint = torch.load(self.work / 'checkpoint.pt', map_location='cpu', weights_only=False)
        state = checkpoint['params']
        n = len(state['means'])
        # Standard 3DGS: remaining coefficients are CHANNEL major (R's three,
        # then G's three, then B's three), not gsplat's [coefficient, channel].
        rest = state['shN'].numpy().transpose(0, 2, 1).reshape(n, 9)
        if checkpoint['sh_degree'] == 0:
            rest.fill(0)
        fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
        fields += [f'f_rest_{i}' for i in range(9)]
        fields += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        data = np.concatenate([state['means'].numpy(), np.zeros((n, 3)), state['sh0'][:, 0].numpy(), rest,
                               state['opacities'].numpy()[:, None], state['scales'].numpy(),
                               F.normalize(state['quats'], dim=-1).numpy()], axis=1).astype('<f4')
        with ply.open('wb') as f:
            header = ['ply', 'format binary_little_endian 1.0', 'comment PLYSplatRenderMode: mip', f'element vertex {n}']
            header += ['property float ' + s for s in fields] + ['end_header']
            f.write(('\n'.join(header) + '\n').encode()); f.write(data.tobytes())
        return n

    def export(self):
        ply = self.out / 'scene.ply'
        n = self.write_ply(ply)
        web = self.out / 'web'; web.mkdir(exist_ok=True)
        spz = web / 'scene.spz'
        subprocess.run(['splat-transform', str(ply), str(spz)], check=True)
        if spz.stat().st_size == 0:
            raise RuntimeError('Empty SPZ export')
        shutil.copy2(self.work / 'collision.obj', web / 'collision.obj')
        for asset in (self.work / 'holes').iterdir():
            shutil.copy2(asset, web / asset.name)
        shutil.copy2(self.proof / 'patches.json', web / 'patches.json')
        shutil.copy2(HERE / 'web.js', web / 'main.js')
        shutil.copy2(HERE / 'web.html', web / 'index.html')
        vendors = Path(os.environ.get('HP_SPLAT_WEB_DEPS', str(Path.home() / '.local/share/hp-splat-web')))
        vendor = web / 'vendor'; vendor.mkdir(exist_ok=True)
        sources = {'three.module.js': 'three/build/three.module.js', 'three.core.js': 'three/build/three.core.js',
                   'spark.module.js': '@sparkjsdev/spark/dist/spark.module.js',
                   'OBJLoader.js': 'three/examples/jsm/loaders/OBJLoader.js',
                   'Pass.js': 'three/examples/jsm/postprocessing/Pass.js',
                   'THREE-LICENSE.txt': 'three/LICENSE',
                   'SPARK-LICENSE.txt': '@sparkjsdev/spark/LICENSE'}
        for target, source in sources.items():
            shutil.copy2(vendors / 'node_modules' / source, vendor / target)
        p = self.train[0]
        save_json(web / 'scene.json', {'position': xyz(p['position']).tolist(), 'up': 'Z', 'units': 'metres',
                  'forward': (P.rotation(p['rotation']) @ P.BASES[1][0]).tolist(),
                  'smoke': self.a.smoke, 'mesh_warning': 'simple_planes plus captured-depth triangles; unknown geometry remains uncertain.'})
        save_json(self.proof / 'export.json', {'format': 'SPZ', 'bytes': spz.stat().st_size,
                  'spherical_harmonics': 1, 'exposure': 'identity; training panorama gains/biases omitted',
                  'gaussians': n, 'coordinate_transform': 'none; metric model frame, Z up',
                  'browser_fps': None, 'browser_verification': 'pending real desktop browser',
                  'web_start': 'cd web && python -m http.server 8080; open http://localhost:8080',
                  'SOG': 'SPZ selected; avoids headless WebGPU requirement for SOG compression'})
        return [ply, self.proof / 'export.json'] + [p for p in web.rglob('*') if p.is_file()]

    def finish(self):
        metrics = {name: json.loads((self.proof / (name + '.json')).read_text())
                   for name in ['scale', 'training', 'evaluation', 'fusion', 'export', 'walk_path', 'walk_render', 'nadir', 'seen_map', 'patches']}
        metrics['smoke_only'] = self.a.smoke
        metrics['limitations'] = ['No person masks. Training exposure compensation is dropped for held-out, walk and export.',
            'Structural mesh mask uses orientation and visibility; it cannot semantically distinguish furniture.',
            'Metric depth has local error; voxel fusion has no multi-view consistency filter.',
            'Collision uses simple_planes and captured-depth triangles; unknown walls cannot be guaranteed closed.',
            'Coverage is estimated, conditional on available geometry; flat mean-colour patches are labelled approximations.',
            'No automatic quality retry; inspect held-out comparisons and six worst walk frames.',
            'Browser FPS and Ubuntu/GPU proof require the real target.']
        save_json(self.proof / 'metrics.json', metrics)
        setup = Path.home() / 'hp_splat/send_back/proof'
        if setup.exists() and not self.a.smoke:
            dst = self.proof / 'setup'; dst.mkdir(exist_ok=True)
            for p in setup.iterdir():
                if p.is_file():
                    shutil.copy2(p, dst / p.name)
        for name in ['pose_metrics.json', 'pole_calibration.json']:
            shutil.copy2(self.home / 'proof' / name, self.proof / name)
        save_json(self.out / 'status.json', {'state': 'complete', 'smoke_only': self.a.smoke, 'utc': time.time()})
        files = sorted(p for p in self.out.rglob('*') if p.is_file() and p.name != 'files.tsv')
        (self.out / 'files.tsv').write_text('path\tbytes\n' + ''.join(f'{p.relative_to(self.out)}\t{p.stat().st_size}\n' for p in files))
        print('Send back:', self.out, '\nTotal bytes:', sum(p.stat().st_size for p in files), flush=True)


def ssim(a, b, mask=None):
    # Gaussian-window SSIM, RGB, data range 1; valid border (no padded pixels).
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    t = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-t * t / (2 * 1.5 ** 2)); g /= g.sum()
    kernel = (g[:, None] * g[None, :])[None, None].expand(3, 1, 11, 11)
    blur = lambda x: F.conv2d(x, kernel, groups=3)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    score = (((2 * ma * mb + .01 ** 2) * (2 * cov + .03 ** 2)) /
             ((ma * ma + mb * mb + .01 ** 2) * (va + vb + .03 ** 2)))
    if mask is None:
        return score.mean()
    valid = F.avg_pool2d(mask.float()[None, None], 11, stride=1) > .99999
    return score[valid.expand_as(score)].mean() if valid.any() else a.sum() * 0 + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid', choices=['VN4b1MGnnCR', 'HFwcH424n9w'])
    parser.add_argument('--pack', type=Path, default=HERE.parent)
    parser.add_argument('--root', type=Path, default=Path.home() / 'hp_splat')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--iterations', type=int, default=15000)
    parser.add_argument('--points', type=int, default=2000000)
    args = parser.parse_args()
    if not 1 <= args.iterations <= TRAINING['max_steps'] or not 1 <= args.points <= TRAINING['gaussian_cap']:
        parser.error('iterations must be 1..15000 and points 1..2500000')
    trial = None
    try:
        trial = Trial(args)
        for name, action in [('nadir', lambda: holes.masks(trial)), ('depth', trial.depth), ('calibration', trial.calibration),
                ('fusion', trial.fusion), ('surfaces', lambda: holes.surfaces(trial)), ('train', trial.train_model), ('evaluate', trial.evaluate),
                ('walk', trial.walk), ('export', trial.export)]:
            trial.stage(name, action)
        trial.finish()
    except BaseException as exc:
        if trial is not None:
            save_json(trial.out / 'status.json', {'state': 'failed', 'error': str(exc), 'type': type(exc).__name__, 'utc': time.time()})
        raise


if __name__ == '__main__':
    main()
