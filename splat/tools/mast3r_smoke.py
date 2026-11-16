"""Five-minute CUDA S4 check on three training panoramas, in an empty disposable root.

Compare against existing r2 mono depth on identical visible structural mesh pixels.
Does not download models, mutate r2, train splats, or mark pipeline stages complete.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

import mast3r_depth as S4


def select_three(panos):
    neighbours = S4.nearest_panoramas(panos)
    centres = np.array([[p['position'][k] for k in 'xyz'] for p in panos])
    for i, others in enumerate(neighbours):
        for n, j in enumerate(others):
            for k in others[n + 1:]:
                if .3 < np.linalg.norm(centres[j] - centres[k]) <= 3. and all(
                        np.linalg.norm(centres[i] - centres[x]) > .3 for x in (j, k)):
                    selected = [panos[x] for x in (i, j, k)]
                    try:
                        S4.pair_graph(selected)
                    except RuntimeError:
                        continue
                    return selected
    raise RuntimeError('No three training panoramas with mutually usable <=3m baselines')


def checked(path, marker):
    path = path.resolve()
    expected = marker.get('outputs', {}).get(str(path))
    if not expected or not path.is_file() or S4.digest(path) != expected:
        raise RuntimeError(f'Existing mono artifact missing/changed/unrecorded: {path}')
    return path


def worker(args):
    import torch
    import splat_mesh_check as M
    import holes
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run mast3r_depth.py --self-test for CPU geometry checks')
    home = args.pack / args.sid
    all_panos = json.loads((home / 'poses.json').read_text())['panoramas']
    training = all_panos if os.environ.get('HP_SPLAT_ALL_TRAIN') == '1' else [p for i, p in enumerate(all_panos) if i % 8 != 7]
    panos = select_three(training)
    donor = args.mono_root / args.sid
    donor_proof = args.mono_root / 'send_back' / args.sid / 'proof'
    config_path = donor / 'config.json'
    config = json.loads(config_path.read_text())
    if (config.get('sid') != args.sid or config.get('smoke') or config.get('image_size') != 2048
            or config.get('depth_input_size') != 518 or config.get('depth_method', 'mono') != 'mono'
            or config.get('sources', {}).get('poses.json') != S4.digest(home / 'poses.json')):
        raise RuntimeError('Comparison requires matching full-resolution poses and existing mono config')
    depth_marker = json.loads((donor / 'depth.done.json').read_text())
    scale_marker = json.loads((donor / 'calibration.done.json').read_text())
    nadir_marker = json.loads((donor / 'nadir.done.json').read_text())
    scale_path = checked(donor_proof / 'scale.json', scale_marker)
    mono_scale = float(json.loads(scale_path.read_text())['scale'])
    if not np.isfinite(mono_scale) or not .1 < mono_scale < 10:
        raise RuntimeError('Existing mono calibration scale invalid')
    mono_paths = {}
    for p in panos:
        checked(donor / 'nadir' / (p['id'] + '.png'), nadir_marker)
        for f in range(6):
            if config.get('captures', {}).get(f"{p['id']}/{f}") != S4.digest(home / 'faces_2k' / p['id'] / f'face{f}.jpg'):
                raise RuntimeError('Existing mono capture differs from current pack')
            path = checked(donor / 'depth' / f"{p['id']}_{f}.npy", depth_marker)
            value = np.load(path, mmap_mode='r')
            if value.shape != (2048, 2048) or not np.isfinite(value).all() or value.min() <= 0:
                raise RuntimeError(f'Invalid existing mono depth: {path}')
            mono_paths[p['id'], f] = path
    report = S4.run(home, panos, args.root / 'depth', args.root / 'depth_model.json', budget=270)
    vertices, triangles = M.read_obj(home / 'mesh/mesh_500k.obj')
    tri = vertices[triangles]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1)
    nz = np.abs(normals[:, 2]) / np.maximum(length, 1e-12)
    angle = np.deg2rad(15.)
    structural = (length > 1e-12) & ((nz <= np.sin(angle)) | (nz >= np.cos(angle)))
    errors, supported, comparisons = {'mast3r': [], 'mono_raw': [], 'mono_calibrated': []}, [], []
    trial = SimpleNamespace(work=donor)
    for p in panos:
        full = M.depths(vertices, triangles, p)
        structure = M.depths(vertices, triangles[structural], p)
        for f, (mesh, surface) in enumerate(zip(full, structure)):
            valid = np.isfinite(mesh) & (mesh > .25) & np.isfinite(surface)
            difference = np.full_like(mesh, np.inf)
            np.subtract(surface, mesh, out=difference, where=valid)
            valid &= np.abs(difference) <= np.maximum(.001, .001 * mesh)
            valid &= holes.valid_pixels(trial, p, f, M.W)
            path = args.root / 'depth' / f"{p['id']}_{f}.npy"
            depth = cv2.resize(np.load(path).astype(np.float32), (M.W, M.W), interpolation=cv2.INTER_NEAREST)
            conf = cv2.resize(np.load(path.with_name(path.stem + '_confidence.npy')).astype(np.float32),
                              (M.W, M.W), interpolation=cv2.INTER_NEAREST)
            mono = cv2.resize(np.load(mono_paths[p['id'], f]).astype(np.float32), (M.W, M.W),
                              interpolation=cv2.INTER_NEAREST)
            valid &= (depth > 0) & np.isfinite(depth) & (mono > 0) & np.isfinite(mono)
            count = int(valid.sum())
            row = dict(face=f"{p['id']}/{f}", structural_pixels=count,
                       supported_fraction=float((conf[valid] > 0).mean()) if count else None)
            for method, values in [('mast3r', depth), ('mono_raw', mono), ('mono_calibrated', mono * mono_scale)]:
                error = np.abs(values[valid] / mesh[valid] - 1)
                errors[method].extend(error.tolist())
                row[method + '_median_relative_error'] = float(np.median(error)) if count else None
            keep = valid & (conf > 0)
            supported.extend(np.abs(depth[keep] / mesh[keep] - 1).tolist())
            comparisons.append(row)
    if not errors['mast3r']:
        raise RuntimeError('No visible structural mesh pixels; cannot report depth agreement')
    result = dict(sid=args.sid, panoramas=[p['id'] for p in panos], faces=18,
                  mast3r_seconds=report['seconds'], seconds_per_face=report['seconds_per_face'],
                  face_runtimes=[dict(face=r['face'], seconds=r['seconds']) for r in report['faces']],
                  peak_allocated_bytes=report['peak_allocated_bytes'], peak_reserved_bytes=report['peak_reserved_bytes'],
                  structural_pixels=len(errors['mast3r']),
                  median_relative_error={k: float(np.median(v)) for k, v in errors.items()},
                  mast3r_supported_median_relative_error=float(np.median(supported)) if supported else None,
                  mono_global_scale=mono_scale, mono_root=str(args.mono_root),
                  mono_config_sha256=S4.digest(config_path),
                  mono_depth_marker_sha256=S4.digest(donor / 'depth.done.json'),
                  comparisons=comparisons, estimated_full_home_seconds=report['seconds_per_face'] * len(training) * 6,
                  expected_mast3r_global_scale=1.,
                  method='Same visible mesh pixels for both sources, 256px; nearest sampling; nadir excluded; pooled pixel median',
                  structural_proxy='Visible near-horizontal/vertical mesh triangles, 15 degrees; not semantic labels',
                  limitations='Coarse mesh is an agreement reference, not ground truth. Mono is cached r2 (raw and calibrated). '
                              'All-pixel metric includes zero-confidence nearest fills; supported-only also reported. '
                              'Three-pano timing extrapolation does not prove whole-home <=30min. '
                              'CUDA allocator peaks exclude driver context and other GPU processes; check nvidia-smi too.')
    S4.save_json(args.root / 'smoke.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid', choices=['VN4b1MGnnCR', 'HFwcH424n9w'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pack', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--mono-root', type=Path, default=Path.home() / 'hp_splat_r2')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.pack, args.mono_root = [p.expanduser().resolve() for p in (args.root, args.pack, args.mono_root)]
    if args.worker:
        worker(args)
        return
    if args.root.is_relative_to(args.mono_root) or args.mono_root.is_relative_to(args.root) or args.root.is_relative_to(args.pack) or args.pack.is_relative_to(args.root):
        parser.error('Disposable root must be separate from the pack and mono source')
    if args.root.exists() and (not args.root.is_dir() or any(args.root.iterdir())):
        parser.error('--root must be absent or empty; no existing output is overwritten')
    args.root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), args.sid, '--root', str(args.root),
               '--pack', str(args.pack), '--mono-root', str(args.mono_root), '--worker']
    child = subprocess.Popen(command, start_new_session=True)
    try:
        # Poll in short intervals so a timeout or user interrupt kills the entire worker group.
        while child.poll() is None:
            remaining = 300 - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, 300)
            try:
                child.wait(timeout=min(1., remaining))
            except subprocess.TimeoutExpired:
                pass
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        if isinstance(exc, KeyboardInterrupt):
            raise
        parser.exit(124, 'S4 smoke exceeded 300 seconds; killed worker group; partial root retained.\n')
    if child.returncode:
        parser.exit(child.returncode, 'S4 smoke failed; real roots unchanged.\n')


if __name__ == '__main__':
    main()
