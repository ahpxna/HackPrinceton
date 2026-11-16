"""Verify and copy only the five immutable preprocessing stages into an empty root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

STAGES = ('nadir', 'depth', 'calibration', 'fusion', 'surfaces')
INPUT_SOURCES = ('holes.py', 'splat_mesh_check.py', 'splat_pose_check.py',
                 'poses.json', 'rooms.json', 'mesh_500k.obj')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_donor(root, sid):
    """Read markers and verify all outputs before copying or using any cache."""
    root = root.resolve()
    work = root / sid
    config_path = work / 'config.json'
    if not config_path.exists():
        config_path = work / 'donor_config.json'
    config = json.loads(config_path.read_text())
    if config.get('sid') != sid or config.get('smoke') or config.get('image_size') != 2048:
        raise RuntimeError('Donor must be the requested full-resolution, non-smoke scene')
    records, files = {}, {}
    for stage in STAGES:
        marker = work / f'{stage}.done.json'
        record = json.loads(marker.read_text())
        outputs = record.get('outputs')
        if not isinstance(outputs, dict) or not outputs:
            raise RuntimeError(f'Missing/empty outputs in {marker}')
        for name, sha in outputs.items():
            path = Path(name)
            if not path.is_absolute():
                raise RuntimeError(f'Non-absolute donor output: {name}')
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root)
            if not (resolved.is_relative_to(work) or resolved.is_relative_to(root / 'send_back' / sid)):
                raise RuntimeError(f'Output outside donor scene: {name}')
            if resolved.name in ('config.json', 'checkpoint.pt') or resolved.name.endswith('.done.json'):
                raise RuntimeError(f'Forbidden preprocessing output: {name}')
            if not resolved.is_file() or digest(resolved) != sha:
                raise RuntimeError(f'Donor hash mismatch: {name}')
            if relative in files and files[relative] != sha:
                raise RuntimeError(f'Conflicting donor hashes: {name}')
            files[relative] = sha
        records[stage] = {'record': record, 'marker_sha256': digest(marker)}
    return config, records, files


def validate_import(work, current):
    """Allow a new trainer but refuse changed captures or preprocessing inputs."""
    work = Path(work)
    manifest = json.loads((work / 'reuse_manifest.json').read_text())
    donor_path = work / 'donor_config.json'
    if digest(donor_path) != manifest['donor_config_sha256']:
        raise RuntimeError('Imported donor config hash mismatch')
    donor = json.loads(donor_path.read_text())
    for key in ('sid', 'smoke', 'image_size', 'target_points', 'depth_input_size', 'depth_revision', 'captures'):
        if donor.get(key) != current.get(key):
            raise RuntimeError(f'Reused preprocessing input changed: {key}')
    for name in INPUT_SOURCES:
        if donor['sources'].get(name) != current['sources'].get(name):
            raise RuntimeError(f'Reused preprocessing source changed: {name}')
    for stage in STAGES:
        marker = work / f'{stage}.done.json'
        if digest(marker) != manifest['rebased_marker_sha256'][stage]:
            raise RuntimeError(f'Imported stage marker changed: {stage}')


def reuse(sid, source, target):
    source, target = source.expanduser().resolve(), target.expanduser().resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise RuntimeError('Donor and target roots must be separate, non-nested directories')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise RuntimeError(f'Target exists and is not empty: {target}')
    _, records, files = checked_donor(source, sid)
    config_sha = digest(source / sid / 'config.json')
    target_existed = target.exists()
    target.mkdir(parents=True, exist_ok=target_existed)
    try:
        for relative, sha in files.items():
            dst = target / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, dst)
            if digest(dst) != sha:
                raise RuntimeError(f'Copied output hash mismatch: {dst}')
        work = target / sid
        work.mkdir(parents=True, exist_ok=True)
        donor = work / 'donor_config.json'
        # Store the exact original bytes, without installing a config.json.
        shutil.copy2(source / sid / 'config.json', donor)
        if digest(donor) != config_sha:
            raise RuntimeError('Donor config changed during reuse')
        rebased = {}
        for stage, saved in records.items():
            record = saved['record'].copy()
            record['outputs'] = {str(target / Path(path).resolve().relative_to(source)): sha
                                 for path, sha in record['outputs'].items()}
            marker = work / f'{stage}.done.json'
            write_json(marker, record)
            rebased[stage] = digest(marker)
        write_json(work / 'reuse_manifest.json', {
            'version': 1, 'sid': sid, 'source_root': str(source), 'target_root': str(target),
            'utc': time.time(), 'stages': list(STAGES), 'donor_config_sha256': config_sha,
            'donor_marker_sha256': {s: r['marker_sha256'] for s, r in records.items()},
            'rebased_marker_sha256': rebased, 'files': len(files),
            'policy': 'training-only code change; captures and preprocessing sources must still match donor',
        })
    except BaseException:
        # Only remove the new, exclusively owned destination, never donor work.
        shutil.rmtree(target)
        if target_existed:
            target.mkdir()
        raise
    print(f'Reused {len(files)} verified files, stages {", ".join(STAGES)}: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid', choices=['VN4b1MGnnCR', 'HFwcH424n9w'])
    parser.add_argument('--from', dest='source', type=Path, required=True)
    parser.add_argument('--to', dest='target', type=Path, required=True)
    args = parser.parse_args()
    try:
        reuse(args.sid, args.source, args.target)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
