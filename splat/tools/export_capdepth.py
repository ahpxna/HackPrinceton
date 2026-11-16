#!/usr/bin/env python3
"""Export run_splat's calibrated optical depth as CPU-only W9 capture ranges.

Training layout: ROOT/SID/depth/PANO_FACE.npy and
ROOT/send_back/SID/proof/scale.json. No training modules are imported.
Run --self-test for an analytic box-room round trip in that exact layout.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

SIZE = 256
CONVENTION = ('ray range from capture centre; face order and bases as poses.json '
              'face_bases_center_right_down; pixel centres (i+0.5)/N')
LABELS = {'mast3r', 'wide_baseline', 'mono_fallback', 'mono'}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe_name(name):
    if not isinstance(name, str) or name in ('', '.', '..') or Path(name).name != name:
        raise ValueError(f'Unsafe home/panorama id: {name!r}')
    return name


def world_basis(pano, bases):
    q = np.array([pano['rotation'][k] for k in 'xyzw'], dtype=np.float64)
    position = np.array([pano['position'][k] for k in 'xyz'], dtype=np.float64)
    if not np.isfinite(q).all() or not np.isfinite(position).all() or np.linalg.norm(q) == 0:
        raise ValueError(f"Invalid pose: {pano['id']}")
    x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return position, bases @ rotation.T


def ray(basis, px, py, size=SIZE):
    return basis[0] + (2*(px+.5)/size-1)*basis[1] + (2*(py+.5)/size-1)*basis[2]


def median_z(depth, scale):
    if depth.shape != (2048, 2048) or depth.dtype.kind not in 'fiu':
        raise ValueError(f'Expected native 2048x2048 numeric optical depth, got {depth.shape}/{depth.dtype}')
    # Match training's float32 * global-scale operation before any reduction.
    z = depth.astype(np.float32) * scale
    z[~np.isfinite(z) | (z <= 0)] = np.nan
    blocks = z.reshape(SIZE, 8, SIZE, 8).transpose(0, 2, 1, 3).reshape(SIZE, SIZE, 64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-invalid blocks stay invalid
        return np.nanmedian(blocks, axis=-1)


def encode(depth, scale, basis):
    z = median_z(depth, scale)
    uv = 2*(np.arange(SIZE, dtype=np.float64)+.5)/SIZE-1
    directions = (basis[0] + uv[None, :, None]*basis[1] + uv[:, None, None]*basis[2])
    # Optical z is the coefficient of the unnormalised pinhole ray in training.
    mm = np.rint(z * np.linalg.norm(directions, axis=-1) * 1000)
    valid = np.isfinite(mm) & (mm >= 1) & (mm <= 65535)
    return np.where(valid, mm, 0).astype('<u2')


def face_label(path, source):
    if source == 'mono_r2':
        return 'mono'
    metadata = path.with_suffix('.json')
    if not metadata.is_file():
        return 'mast3r'
    row = read_json(metadata)
    for key in ('source', 'depth_source', 'label', 'method'):
        value = row.get(key)
        if isinstance(value, str):
            if value in LABELS:
                return value
            # Preserve future explicit source labels instead of claiming MASt3R.
            if key != 'method':
                return value
    method = row.get('method', {})
    if isinstance(method, dict) and method.get('method') == 'mast3r_known_pose_pairs':
        return 'mast3r'
    return 'mast3r'


def export(sid, pack, root, out, source, archive=None):
    started = time.monotonic()
    safe_name(sid)
    poses_path = pack / sid / 'poses.json'
    poses = read_json(poses_path)
    if poses.get('units') != 'metres' or poses.get('quaternion_order') != 'xyzw':
        raise ValueError('poses.json must specify metres and xyzw quaternions')
    bases = np.asarray(poses['face_bases_center_right_down'], dtype=np.float64)
    if (bases.shape != (6, 3, 3) or not np.isfinite(bases).all()
            or not np.allclose(bases @ bases.transpose(0, 2, 1), np.eye(3), atol=1e-5)):
        raise ValueError('Expected six orthonormal center/right/down face bases')
    panos = poses['panoramas']
    ids = [safe_name(p['id']) for p in panos]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Expected nonempty, unique panoramas')
    depth_dir = root / sid / 'depth'
    paths = [depth_dir / f'{pid}_{face}.npy' for pid in ids for face in range(6)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)}/{len(paths)} depth files; first: {missing[0]}. '
                         'Wait for the complete depth stage; no export written.')
    scale_path = root / 'send_back' / sid / 'proof' / 'scale.json'
    if not scale_path.is_file():
        raise ValueError(f'Missing training calibration: {scale_path}; no uncalibrated export.')
    calibration = read_json(scale_path)
    scale = float(calibration['scale'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f'Invalid training scale in {scale_path}')
    target = out / 'capdepth'
    if target.exists():
        raise ValueError(f'Output already exists: {target}; use a fresh --out directory.')
    if archive and archive.exists():
        raise ValueError(f'Archive already exists: {archive}; use a fresh --tar path.')
    # Preflight all poses and labels before creating output.
    cameras = [world_basis(p, bases) for p in panos]
    labels = [face_label(path, source) for path in paths]
    counts = Counter(labels)
    index = dict(format='u16_range_mm', size=SIZE, source=source, convention=CONVENTION,
                 model_id=sid, filter='8x8 valid-only median optical z, then pixel-centre ray range',
                 invalid='0: nonfinite/nonpositive depth or range outside uint16 millimetres',
                 calibration=dict(path=str(scale_path), scale=scale,
                                  validated=calibration.get('validated')),
                 input_depth='optical_z_metres_before_global_scale',
                 panoramas={pid: labels[i*6:(i+1)*6] for i, pid in enumerate(ids)},
                 source_counts=dict(counts), check_points=[])
    # One point from first/middle/last capture, spread across three face directions.
    selections = [(0, 0, 61, 87), (len(panos)//2, 2, 171, 113), (len(panos)-1, 5, 139, 199)]
    for i, (pid, (position, world)) in enumerate(zip(ids, cameras)):
        folder = target / pid
        folder.mkdir(parents=True)
        for face in range(6):
            path = paths[i*6+face]
            try:
                data = encode(np.load(path, mmap_mode='r', allow_pickle=False), scale, world[face])
            except (ValueError, OSError) as exc:
                raise ValueError(f'{path}: {exc}') from exc
            data.tofile(folder / f'face{face}.u16')
            for pi, selected_face, px, py in selections:
                if pi != i or selected_face != face:
                    continue
                if data[py, px] == 0:
                    valid = np.argwhere(data > 0)
                    if not len(valid):
                        continue
                    py, px = map(int, valid[len(valid)//2])
                distance = int(data[py, px])/1000
                direction = ray(world[face], px, py)
                point = position + distance*direction/np.linalg.norm(direction)
                index['check_points'].append(dict(pano=pid, face=face, px=px, py=py,
                                                   range_m=distance, world_xyz=point.tolist()))
        print(f'{sid}: {i+1}/{len(panos)} captures', flush=True)
    if len(index['check_points']) != 3:
        raise ValueError('Cannot produce three valid check points; export is incomplete (no index/archive).')
    write_json(target / 'index.json', index)
    size = sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
    print(f'Export: {time.monotonic()-started:.2f}s, {size:,} bytes ({size/2**20:.2f} MiB); '
          f'faces by source: {dict(counts)}', flush=True)
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '-C', str(out), '-czf', str(archive), 'capdepth'], check=True)
        print(f'Archive: {archive} ({archive.stat().st_size:,} bytes); '
              f'total runtime {time.monotonic()-started:.2f}s', flush=True)
    return index


def self_test():
    # Independent analytic slab intersection, a translated camera and a full 3D
    # quaternion exercise scale, all face orientations, top-left and pixel centres.
    bases = [[[0, 0, 1], [0, -1, 0], [1, 0, 0]],
             [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
             [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
             [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
             [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
             [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]]
    position = np.array([.31, -.27, .19])
    axis = np.array([1., 2., -3.])/np.sqrt(14)
    angle = .37
    q = np.r_[axis*np.sin(angle/2), np.cos(angle/2)]
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3)*np.cos(angle) + (1-np.cos(angle))*np.outer(axis, axis) + np.sin(angle)*cross
    lo, hi = np.array([-3., -4., -2.]), np.array([4., 3., 3.])

    def intersection(direction):
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.maximum((lo-position)/direction, (hi-position)/direction).min(axis=-1)
        return position + t[..., None]*direction

    with tempfile.TemporaryDirectory(prefix='capdepth-self-test-') as tmp:
        tmp = Path(tmp)
        worst = 0.
        for sid, source in [('box_mono', 'mono_r2'), ('box_mast3r', 'mast3r_r5')]:
            pack, root = tmp/'pack', tmp/'root'
            home, depth = pack/sid, root/sid/'depth'
            home.mkdir(parents=True)
            depth.mkdir(parents=True)
            pano = dict(id='capture', position=dict(zip('xyz', position)), rotation=dict(zip('xyzw', q)))
            write_json(home/'poses.json', dict(units='metres', quaternion_order='xyzw',
                       face_bases_center_right_down=bases, panoramas=[pano]))
            proof = root/'send_back'/sid/'proof'
            proof.mkdir(parents=True)
            write_json(proof/'scale.json', dict(scale=1.7, validated=True))
            uv = 2*(np.arange(2048)+.5)/2048-1
            for face, basis in enumerate(np.array(bases) @ rotation.T):
                direction = basis[0] + uv[None, :, None]*basis[1] + uv[:, None, None]*basis[2]
                world = intersection(direction)
                z = ((world-position)*basis[0]).sum(axis=-1)
                np.save(depth/f'capture_{face}.npy', (z/1.7).astype(np.float16))
                if source == 'mast3r_r5' and face != 5:
                    write_json(depth/f'capture_{face}.json', dict(source=['mast3r', 'wide_baseline', 'mono_fallback'][face%3]))
            out, archive = tmp/sid, tmp/f'{sid}.tar.gz'
            result = export(sid, pack, root, out, source, archive)
            for face, basis in enumerate(np.array(bases) @ rotation.T):
                data = np.fromfile(out/'capdepth'/'capture'/f'face{face}.u16', dtype='<u2').reshape(SIZE, SIZE)
                for px, py in [(0, 0), (255, 255), (61, 87), (171, 113), (139, 199)]:
                    direction = basis[0] + (2*(px+.5)/SIZE-1)*basis[1] + (2*(py+.5)/SIZE-1)*basis[2]
                    actual = position + int(data[py, px])/1000*direction/np.linalg.norm(direction)
                    error = np.linalg.norm(actual-intersection(direction))
                    worst = max(worst, error)
                    assert error < .01, (face, px, py, error)
            for point in result['check_points']:
                basis = np.array(bases)[point['face']] @ rotation.T
                direction = basis[0] + (2*(point['px']+.5)/SIZE-1)*basis[1] + (2*(point['py']+.5)/SIZE-1)*basis[2]
                assert np.linalg.norm(np.array(point['world_xyz'])-intersection(direction)) < .01
            assert len(result['check_points']) == 3
            assert result['panoramas']['capture'] == (['mono']*6 if source == 'mono_r2' else
                       ['mast3r', 'wide_baseline', 'mono_fallback', 'mast3r', 'wide_baseline', 'mast3r'])
            members = subprocess.check_output(['tar', '-tzf', str(archive)], text=True).splitlines()
            assert all(name.startswith('capdepth/') for name in members)
            (depth/'capture_0.npy').unlink()
            try:
                export(sid, pack, root, tmp/f'{sid}-missing', source)
            except ValueError as exc:
                assert 'Missing 1/6 depth files' in str(exc)
                assert not (tmp/f'{sid}-missing').exists()
            else:
                raise AssertionError('Missing depth was accepted')
        invalid = np.zeros((2048, 2048), dtype=np.float32)
        invalid[:8, :8] = np.nan
        invalid[:8, 8:16] = np.inf
        invalid[:8, 16:24] = -1
        invalid[:8, 24:32] = 100
        assert not encode(invalid, 1., np.array(bases[0])).any()
        invalid.fill(2.)
        invalid[:8, :8] = 200
        invalid[:5, :8] = 2.
        assert median_z(invalid, 1.)[0, 0] == 2.  # median, never a mean
        print(f'PASS: analytic box, mono/r5 labels, archive, invalid/missing depths; '
              f'max round-trip error {worst*100:.3f} cm (<1 cm)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid', nargs='?')
    parser.add_argument('--pack', type=Path, default=Path.home()/'hp_splat/pack')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source', choices=('mono_r2', 'mast3r_r5'), default='mono_r2')
    parser.add_argument('--tar', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        try:
            os.nice(19)
        except OSError:
            if not args.self_test:
                raise
            print('Self-test: sandbox denied os.nice(19); using CPU only.', file=sys.stderr)
        if args.self_test:
            self_test()
        else:
            if not args.sid or args.root is None or args.out is None:
                parser.error('sid, --root and --out are required (or use --self-test)')
            export(args.sid, args.pack.expanduser(), args.root.expanduser(), args.out.expanduser(),
                   args.source, args.tar.expanduser() if args.tar else None)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'export_capdepth: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
