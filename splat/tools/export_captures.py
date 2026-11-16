#!/usr/bin/env python3
"""Export real cube faces and their metric Z-up viewer-frame camera bases."""
import argparse
import json
import math
import shutil
from pathlib import Path


def rotate(vector, quaternion):
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx]


def export(sid, web_dir, size=1024):
    pack = Path(__file__).resolve().parents[1]
    if Path(sid).name != sid or sid in ('', '.', '..'):
        raise ValueError('sid must be a home directory name')
    home, web = pack / sid, Path(web_dir)
    poses = json.loads((home / 'poses.json').read_text())
    scene = json.loads((web / 'scene.json').read_text())
    if any(data.get('up') != 'Z' or data.get('units') != 'metres' for data in (poses, scene)):
        raise ValueError('Expected the untransformed metric Z-up viewer frame')
    if poses.get('quaternion_order') != 'xyzw':
        raise ValueError('Expected xyzw quaternions')
    bases = poses['face_bases_center_right_down']
    if len(bases) != 6 or any(len(face) != 3 or any(len(v) != 3 for v in face) for face in bases):
        raise ValueError('Expected six center/right/down face bases')
    captures = []
    for pano in poses['panoramas']:
        pid = pano['id']
        if Path(pid).name != pid or pid in ('', '.', '..'):
            raise ValueError('Unsafe panorama id')
        position = [pano['position'][k] for k in 'xyz']
        q = [pano['rotation'][k] for k in 'xyzw']
        norm = math.sqrt(sum(v * v for v in q))
        if not norm or not all(math.isfinite(v) for v in position + q):
            raise ValueError('Non-finite or zero camera pose')
        q = [v / norm for v in q]
        captures.append({'id': pid, 'position': position, 'rotation': q,
                         'face_bases_center_right_down': [[rotate(v, q) for v in face] for face in bases],
                         'faces': [f'captures/{pid}/face{face}.jpg' for face in range(6)]})
    if len({c['id'] for c in captures}) != len(captures):
        raise ValueError('Duplicate panorama id')
    # run_splat exports its first training capture without any coordinate transform.
    spawn = next((c for c in captures if math.dist(c['position'], scene['position']) < 1e-5), None)
    if spawn is None or math.dist(spawn['face_bases_center_right_down'][1][0], scene['forward']) > 1e-5:
        raise ValueError('scene.json spawn/forward does not match the supplied capture frame')
    sources = [(home / 'faces_2k' / c['id'] / f'face{face}.jpg', web / c['faces'][face])
               for c in captures for face in range(6)]
    for source, _ in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    # Four resident 1024px six-face textures use ~128 MiB including GPU mipmaps.
    # Keep original 2048px JPEGs byte-for-byte when --size 2048 is requested.
    if size != 2048:
        from PIL import Image
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        if size == 2048:
            shutil.copy2(source, target)
        else:
            with Image.open(source) as image:
                image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS).save(
                    target, quality=92, subsampling=0)
    result = {'model_id': sid, 'up': 'Z', 'units': 'metres', 'quaternion_order': 'xyzw',
              'frame': 'untransformed viewer/model frame; face bases already rotated into world space',
              'face_size': size, 'captures': captures}
    (web / 'captures.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{sid}: {len(captures)} captures, {len(sources)} faces at {size}px -> {web}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid')
    parser.add_argument('web_dir', type=Path)
    parser.add_argument('--size', type=int, choices=(1024, 2048), default=1024)
    args = parser.parse_args()
    export(args.sid, args.web_dir, args.size)
