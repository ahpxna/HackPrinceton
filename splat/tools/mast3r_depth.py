"""S4: train-only MASt3R pairs, anchored to supplied metric cube-face cameras.

No downloads. Uses the existing F2 checkout/venv (HP_MAST3R_F2_ROOT,
default ~/hp_f2). Depth is camera-Z in metres, float16 2048-square .npy;
*_confidence.npy is float16 [0,1], with zero for filled/unobserved pixels.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REVISION = 'f5209afc300cec36239a7ac992263f36847bbba0'
CHECKPOINT_SHA = '0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb'
MODEL_REVISION = '06e7259f34c3060f322df5cb0c7b9094f57e41fc'
SIZE = 512
VRAM = 8 * 1024 ** 3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_array(path, value):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, value.astype(np.float16))
    tmp.replace(path)


def settings():
    return dict(method='mast3r_known_pose_pairs', source_revision=REVISION,
                model_revision=MODEL_REVISION, checkpoint_sha256=CHECKPOINT_SHA,
                input_size=SIZE, output_size=2048, neighbours=3, radius_metres=3.,
                max_vram_bytes=VRAM, budget_seconds=1800,
                fusion='confidence weighted median; 15% pair agreement; nearest fill has zero confidence')


@lru_cache(maxsize=1)
def pose_module():
    spec = importlib.util.spec_from_file_location('s4_pose', HERE / 'splat_pose_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def face_pose(pano, face):
    """Camera-to-world, using exactly run_splat.camera's right/down/forward basis."""
    P = pose_module()
    pose = np.eye(4)
    pose[:3, :3] = P.rotation(pano['rotation']) @ P.BASES[face][[1, 2, 0]].T
    pose[:3, 3] = [pano['position'][k] for k in 'xyz']
    return pose


def nearest_panoramas(panos):
    centres = np.array([[p['position'][k] for k in 'xyz'] for p in panos])
    dist = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    return [[int(j) for j in np.argsort(dist[i]) if .05 < dist[i, j] <= 3.][:3]
            for i in range(len(panos))]


def overlap(a, b):
    """Frustum overlap proxy at several indoor ranges; includes known translation."""
    if a[:3, 2] @ b[:3, 2] <= 0:
        return 0.
    uv = np.linspace(-.9, .9, 7)
    xx, yy = np.meshgrid(uv, uv)
    rays = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
    scores = []
    for z in (1., 2., 4., 8.):
        world = (rays * z) @ a[:3, :3].T + a[:3, 3]
        local = (world - b[:3, 3]) @ b[:3, :3]
        scores.append(np.mean((local[:, 2] > .1) &
                              (np.abs(local[:, :2]).max(1) < local[:, 2])))
    return float(max(scores))


def pair_graph(panos):
    """One best-overlapping face per each of the nearest three other panoramas.

    The undirected graph reuses both predictions, avoiding duplicate forwards.
    No held-out panorama can enter: callers supply the training list only.
    """
    poses = {(i, f): face_pose(p, f) for i, p in enumerate(panos) for f in range(6)}
    neighbours = nearest_panoramas(panos)
    pairs = set()
    for i in range(len(panos)):
        for f in range(6):
            for j in neighbours[i]:
                ranked = [(overlap(poses[i, f], poses[j, g]), g) for g in range(6)]
                score, g = max(ranked)
                if score >= .08:
                    pairs.add(tuple(sorted(((i, f), (j, g)))))
    pairs = sorted(pairs)
    degree = {key: 0 for key in poses}
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1
    missing = [f"{panos[i]['id']}/{f}" for (i, f), n in degree.items() if n == 0]
    if missing:
        raise RuntimeError('No overlapping training neighbour within 3m for: ' + ', '.join(missing))
    return pairs, poses, neighbours


def align_poses(predicted, known):
    """Similarity X_world = scale * X_pred @ rotation.T + translation.

    Fit rigid orientation from both camera frames; scale from the known baseline,
    never from mesh geometry or held-out data. Camera centres fix translation.
    """
    predicted, known = np.asarray(predicted), np.asarray(known)
    cov = sum(k[:3, :3] @ p[:3, :3].T for p, k in zip(predicted, known))
    u, _, vt = np.linalg.svd(cov)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    pc, kc = predicted[:, :3, 3], known[:, :3, 3]
    pb, kb = pc[1] - pc[0], kc[1] - kc[0]
    if min(np.linalg.norm(pb), np.linalg.norm(kb)) < 1e-5:
        raise ValueError('Degenerate pair baseline')
    scale = np.linalg.norm(kb) / np.linalg.norm(pb)
    translation = kc.mean(0) - scale * rotation @ pc.mean(0)
    baseline_error = np.linalg.norm(scale * rotation @ pb - kb) / np.linalg.norm(kb)
    angles = [np.degrees(np.arccos(np.clip((np.trace(k[:3, :3].T @ rotation @ p[:3, :3]) - 1) / 2, -1, 1)))
              for p, k in zip(predicted, known)]
    if not np.isfinite(scale) or baseline_error > .35 or max(angles) > 15:
        raise ValueError(f'Pair pose inconsistent: baseline residual={baseline_error:.3f}, angles={angles}')
    return float(scale), rotation, translation, dict(scale=float(scale),
            baseline_metres=float(np.linalg.norm(kb)), baseline_relative_residual=float(baseline_error),
            rotation_errors_degrees=list(map(float, angles)))


def grid(size):
    yy, xx = np.mgrid[:size, :size]
    return np.stack([xx, yy], -1).astype(np.float64)


def recover_pose(points, confidence):
    """Robust camera-to-prediction pose, known 90-degree square-face intrinsics."""
    size = points.shape[0]
    uv = grid(size)
    valid = np.isfinite(points).all(-1) & np.isfinite(confidence) & (confidence > 1.)
    valid[1::2] = False
    valid[:, 1::2] = False
    ids = np.flatnonzero(valid)
    if len(ids) < 64:
        raise ValueError('Too few finite confident PnP points')
    ids = ids[np.argsort(confidence.ravel()[ids])[-6000:]]
    K = np.array([[size / 2, 0, (size - 1) / 2], [0, size / 2, (size - 1) / 2], [0, 0, 1.]])
    xyz, pixels = points.reshape(-1, 3)[ids].astype(np.float64), uv.reshape(-1, 2)[ids]
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(xyz, pixels, K, None,
                iterationsCount=100, reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 64:
        raise ValueError('Pair camera PnP failed')
    keep = inliers.ravel()
    ok, rvec, tvec = cv2.solvePnP(xyz[keep], pixels[keep], K, None, rvec, tvec,
                                 useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('Pair PnP refinement failed')
    rotation = cv2.Rodrigues(rvec)[0]
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -rotation.T @ tvec.ravel()
    return pose


def anchored_depth(points, confidence, known, transform):
    scale, rotation, translation, _ = transform
    world = scale * points @ rotation.T + translation
    local = (world - known[:3, 3]) @ known[:3, :3]
    z = local[..., 2]
    size = len(z)
    rays = (grid(size) + .5) * 2 / size - 1
    residual = np.linalg.norm(local[..., :2] / np.maximum(z[..., None], 1e-6) - rays, axis=-1)
    valid = np.isfinite(local).all(-1) & np.isfinite(confidence) & (z > .05) & (z < 100) & (residual < .08)
    # MASt3R confidence is >=1, not a calibrated probability.
    weight = np.where(valid, np.clip(1 - 1 / np.maximum(confidence, 1), 0, 1) *
                      np.exp(-.5 * (residual / .03) ** 2), 0)
    return np.where(valid, z, 0).astype(np.float32), weight.astype(np.float32)


def fuse(depths, weights):
    """Weighted median + agreement average. Fill only for downstream positive-depth contract."""
    d, w = np.stack(depths), np.stack(weights)
    order = np.argsort(d, axis=0)
    sd, sw = np.take_along_axis(d, order, 0), np.take_along_axis(w, order, 0)
    index = (np.cumsum(sw, axis=0) < sw.sum(0)[None] / 2).sum(0).clip(0, len(d) - 1)
    median = np.take_along_axis(sd, index[None], 0)[0]
    agree = (np.abs(d - median) <= np.maximum(.05, .15 * median)) & (w > 0)
    w = w * agree
    total = w.sum(0)
    result = (d * w).sum(0) / np.maximum(total, 1e-9)
    valid = total > 0
    if valid.mean() < .1:
        raise RuntimeError(f'Only {valid.mean():.1%} valid depth coverage; refusing fabricated geometry')
    confidence = np.clip(total / len(depths), 0, 1)
    if not valid.all():
        from scipy.ndimage import distance_transform_edt
        indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        result[~valid] = result[tuple(indices[:, ~valid])]
    return result.astype(np.float32), confidence.astype(np.float32)


def f2_root():
    return Path(os.environ.get('HP_MAST3R_F2_ROOT', str(Path.home() / 'hp_f2'))).expanduser().resolve()


def load_model(root):
    """Reuse F2's exact local safetensors loader, with no from_pretrained/download path."""
    import torch
    source = root / 'experiments/furniture_scan'
    vendor = source / 'vendor/mast3r'
    checkpoint = source / 'weights/metric/model.safetensors'
    if not checkpoint.is_file() or digest(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError(f'Missing/mismatched existing F2 metric checkpoint: {checkpoint}; no downloads allowed')
    commit = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != REVISION:
        raise RuntimeError(f'F2 MASt3R revision mismatch: {commit} != {REVISION}')
    sys.path.insert(0, str(vendor))
    from mast3r.model import AsymmetricMASt3R
    spec = importlib.util.spec_from_file_location('s4_f2_scan', source / 'scan.py')
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    model = scan.load_metric_model(AsymmetricMASt3R, 'cuda')
    return model, checkpoint, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class PairRunner:
    def __init__(self, home, panos):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('S4 requires CUDA; --self-test and --plan-only work on CPU')
        free, total = torch.cuda.mem_get_info()
        if free < VRAM:
            raise RuntimeError('S4 needs 8 GiB free; wait for the other job, do not stop it')
        # Leave driver/cuBLAS headroom inside the approximate 8 GiB process budget.
        torch.cuda.set_per_process_memory_fraction((VRAM - 512 * 1024 ** 2) / total)
        torch.cuda.reset_peak_memory_stats()
        self.model, self.checkpoint, self.amp = load_model(f2_root())
        from dust3r.utils.image import load_images
        self.load_images = load_images
        self.home, self.panos = home, panos
        self.images = OrderedDict()

    def image(self, key):
        import torch
        if key not in self.images:
            i, f = key
            path = self.home / 'faces_2k' / self.panos[i]['id'] / f'face{f}.jpg'
            # square_ok=True is essential: upstream otherwise crops square faces to 512x384.
            view = self.load_images([str(path)], size=SIZE, square_ok=True, verbose=False)[0]
            if tuple(view['img'].shape[-2:]) != (SIZE, SIZE):
                raise ValueError(f'Expected square 2k face: {path}')
            view['img'] = view['img'].to('cuda')
            # Direct forwards bypass inference.py's collate: provide batch metadata too.
            view['true_shape'] = torch.as_tensor(view['true_shape'], dtype=torch.int32)
            view['idx'], view['instance'] = [i * 6 + f], [f'{i}/{f}']
            self.images[key] = view
            if len(self.images) > 24:
                self.images.popitem(last=False)
        self.images.move_to_end(key)
        return self.images[key]

    def pair(self, a, b):
        import torch
        with torch.inference_mode(), torch.autocast('cuda', dtype=self.amp):
            p, q = self.model(self.image(a), self.image(b))
        # Transfer just geometry/confidence; avoid dust3r.inference's CPU image-pair round trips.
        result = [(x[key][0].float().cpu().numpy(), x['conf'][0].float().cpu().numpy())
                  for x, key in ((p, 'pts3d'), (q, 'pts3d_in_other_view'))]
        del p, q
        return result


def run(home, panos, out, proof, budget=1800):
    import torch
    started = time.monotonic()
    pairs, poses, neighbours = pair_graph(panos)
    captures = {f"{p['id']}/{f}": digest(home / 'faces_2k' / p['id'] / f'face{f}.jpg')
                for p in panos for f in range(6)}
    poses_sha = digest(home / 'poses.json')
    out.mkdir(parents=True, exist_ok=True)
    proof.parent.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError(f'Depth directory must be empty; preserve partial results and use a new root: {out}')
    runner = PairRunner(home, panos)
    pending = {key: sum(key in pair for pair in pairs) for key in poses}
    accum, records, times, failures, outputs = {}, {}, {}, [], []
    per_face = []
    for number, (a, b) in enumerate(pairs, 1):
        if time.monotonic() - started > budget:
            raise RuntimeError(f'S4 exceeded {budget}s; no completed depth-stage marker may be written')
        tick = time.monotonic()
        predictions = runner.pair(a, b)
        try:
            predicted = [recover_pose(*item) for item in predictions]
            transform = align_poses(predicted, [poses[a], poses[b]])
            for key, (points, conf) in zip((a, b), predictions):
                d, w = anchored_depth(points, conf, poses[key], transform)
                accum.setdefault(key, []).append((d, w))
                records.setdefault(key, []).append(dict(pair=[f"{panos[i]['id']}/{f}" for i, f in (a, b)],
                                                        **transform[3]))
        except ValueError as exc:
            failures.append(dict(pair=[f"{panos[i]['id']}/{f}" for i, f in (a, b)], error=str(exc)))
        elapsed = time.monotonic() - tick
        del predictions
        for key in (a, b):
            times[key] = times.get(key, 0.) + elapsed / 2
            pending[key] -= 1
            if pending[key]:
                continue
            if key not in accum:
                raise RuntimeError(f'No pose-consistent predictions for {key}; rejected pairs: {failures[-6:]}')
            tick = time.monotonic()
            d, w = fuse(*zip(*accum.pop(key)))
            i, f = key
            path = out / f"{panos[i]['id']}_{f}.npy"
            confidence = path.with_name(path.stem + '_confidence.npy')
            metadata = path.with_suffix('.json')
            save_array(path, cv2.resize(d, (2048, 2048), interpolation=cv2.INTER_LINEAR))
            save_array(confidence, cv2.resize(w, (2048, 2048), interpolation=cv2.INTER_NEAREST))
            seconds = times.pop(key) + time.monotonic() - tick
            face_records = records.pop(key)
            row = dict(face=f"{panos[i]['id']}/{f}", seconds=seconds, valid_fraction=float((w > 0).mean()),
                       method=settings(), checkpoint=str(runner.checkpoint), pairs=face_records,
                       poses_sha256=poses_sha, capture_sha256=captures[f"{panos[i]['id']}/{f}"],
                       paired_capture_sha256={name: captures[name] for record in face_records
                                              for name in record['pair']},
                       depth_sha256=digest(path), confidence_sha256=digest(confidence),
                       confidence_note='Support score, not probability; nearest-filled pixels are zero')
            save_json(metadata, row)
            per_face.append(row)
            outputs.extend([path, confidence, metadata])
        if number % 12 == 0 or number == len(pairs):
            print(f'S4 pairs {number}/{len(pairs)}, faces {len(per_face)}/{len(poses)}, '
                  f'{time.monotonic() - started:.1f}s', flush=True)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    report = dict(method=settings(), training_panoramas=[p['id'] for p in panos],
                  neighbours={p['id']: [panos[j]['id'] for j in neighbours[i]] for i, p in enumerate(panos)},
                  pair_count=len(pairs), rejected_pairs=failures, faces=per_face,
                  checkpoint=str(runner.checkpoint), poses_sha256=poses_sha, captures=captures,
                  loader_sha256=digest(f2_root() / 'experiments/furniture_scan/scan.py'),
                  seconds=seconds, seconds_per_face=seconds / len(poses),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  outputs=[str(p) for p in outputs], global_scale_expected=1.,
                  limitations='512px learned geometry upsampled to 2k; CUDA runtime and accuracy must be measured')
    if seconds > budget or report['peak_reserved_bytes'] > VRAM:
        raise RuntimeError('S4 runtime/VRAM budget exceeded')
    save_json(proof, report)
    return report


def generate(trial):
    """Pipeline adapter: run inference in F2's existing environment, leave gsplat's env intact."""
    python = Path(os.environ.get('HP_MAST3R_PYTHON', str(f2_root() / '.venv-scan/bin/python')))
    if not python.is_file():
        raise RuntimeError(f'Existing F2 Python missing: {python}')
    proof = trial.proof / 'depth_model.json'
    subprocess.run([str(python), str(Path(__file__)), trial.a.sid, '--pack', str(trial.a.pack),
                    '--out', str(trial.cache), '--proof', str(proof),
                    '--panos', *[p['id'] for p in trial.train]], check=True, timeout=1800)
    report = json.loads(proof.read_text())
    return [Path(p) for p in report['outputs']] + [proof]


def self_test():
    """CPU synthetic points + known rotation, translation and scale; no model or weights."""
    rng = np.random.default_rng(4)
    rotation = cv2.Rodrigues(np.array([.23, -.31, .17]))[0]
    translation, scale = np.array([2.1, -1.7, .8]), 2.4
    known = np.repeat(np.eye(4)[None], 2, axis=0)
    known[0, :3, :3] = cv2.Rodrigues(np.array([.02, .04, -.03]))[0]
    known[1, :3, :3] = cv2.Rodrigues(np.array([.03, -.06, .01]))[0]
    known[1, :3, 3] = [.8, .15, .1]
    predicted = known.copy()
    predicted[:, :3, :3] = rotation.T @ known[:, :3, :3]
    predicted[:, :3, 3] = (known[:, :3, 3] - translation) @ rotation / scale
    recovered = align_poses(predicted, known)
    points = rng.normal(size=(1000, 3))
    np.testing.assert_allclose(recovered[0] * points @ recovered[1].T + recovered[2],
                               scale * points @ rotation.T + translation, atol=1e-10)
    for k, p in zip(known, predicted):
        rays = np.dstack([(grid(64) + .5) * 2 / 64 - 1, np.ones((64, 64))])
        depth = rng.uniform(2, 6, (64, 64))
        world = (rays * depth[..., None]) @ k[:3, :3].T + k[:3, 3]
        cloud = ((world - translation) @ rotation / scale).astype(np.float32)
        conf = np.full((64, 64), 10.)
        np.testing.assert_allclose(recover_pose(cloud, conf), p, atol=2e-6)
        got, weight = anchored_depth(cloud, conf, k, recovered)
        np.testing.assert_allclose(got, depth, atol=2e-6)
        assert (weight > .89).all()
    d = np.full((16, 16), 3., np.float32)
    got, confidence = fuse([d, d * 1.01, d * 4], [d / 3] * 3)
    np.testing.assert_allclose(got, 3.015, atol=1e-6)
    assert (confidence > 0).all()
    try:
        align_poses(np.repeat(predicted[:1], 2, axis=0), known)
    except ValueError:
        pass
    else:
        raise AssertionError('Degenerate baseline accepted')
    print('PASS: CPU similarity, two-camera PnP, known-pose optical depth, robust fusion and degenerate baseline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid', nargs='?', choices=['VN4b1MGnnCR', 'HFwcH424n9w'])
    parser.add_argument('--pack', type=Path, default=HERE.parent)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--panos', nargs='+')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.sid:
        parser.error('sid is required')
    home = args.pack / args.sid
    all_panos = json.loads((home / 'poses.json').read_text())['panoramas']
    training = all_panos if os.environ.get('HP_SPLAT_ALL_TRAIN') == '1' else [p for i, p in enumerate(all_panos) if i % 8 != 7]
    if args.panos:
        allowed = {p['id']: p for p in training}
        if len(set(args.panos)) != len(args.panos) or any(pid not in allowed for pid in args.panos):
            parser.error('--panos must be unique training IDs; held-out faces cannot be paired')
        training = [allowed[pid] for pid in args.panos]
    if args.plan_only:
        pairs, poses, neighbours = pair_graph(training)
        print(json.dumps(dict(panoramas=len(training), faces=len(poses), pairs=len(pairs),
                              neighbours_min=min(map(len, neighbours)), neighbours_max=max(map(len, neighbours)))))
        return
    if args.out is None or args.proof is None:
        parser.error('--out and --proof are required')
    run(home, training, args.out, args.proof)


if __name__ == '__main__':
    main()
