"""Reconstruct conservative 10 cm obstacle cells from installed ray-range faces.

Run after the W13 navigation export to protect real baseline routes as well as
capture disks, measured doorway openings and the established stair corridors.
No RGB/depth input is modified. Output is reproducible from local assets only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from depth_consistency import reconcile

STEP = .1
MIN_POINTS = 12
PACK = Path(__file__).resolve().parents[1]
ROOT = Path('/Users/phanan/HackPrinceton')
OUT = ROOT / 'experiments/w13_collide'
STAIRS = {
    'VN4b1MGnnCR': [[[-8.3, 1], [-8.3, -2.3], [-9.4, -2.3], [-9.4, -1.3]],
                   [[.6, 9.65], [3.25, 9.65], [3.25, 10.95], [.65, 10.95]]],
    'HFwcH424n9w': [[[7.8, 1.4], [5.35, 1.4], [5.35, -.85]]],
}


def read_obj(filename):
    vertices, faces = [], []
    for line in filename.read_text().splitlines():
        if line.startswith('v '):
            vertices.append(list(map(float, line.split()[1:4])))
        elif line.startswith('f '):
            ids = [int(v.split('/')[0]) - 1 for v in line.split()[1:]]
            faces.extend([[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)])
    return np.asarray(vertices), np.asarray(faces)


def rectangle_distance(x, y, px, py):
    return np.hypot(max(x * STEP - px, 0, px - (x + 1) * STEP),
                    max(y * STEP - py, 0, py - (y + 1) * STEP))


def segment_distance(p, a, b):
    a, b, p = np.array(a), np.array(b), np.array(p)
    d = b - a
    t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-12), 0, 1)
    return float(np.linalg.norm(p - a - t * d))


def build(sid):
    folder = ROOT / 'data/splat/web' / sid
    metadata = json.loads((folder / 'captures.json').read_text())
    captures = metadata['captures']
    vertices, faces = read_obj(folder / 'walkmesh.obj')
    lower = np.floor(vertices[:, :2].min(axis=0) / STEP).astype(int) - 2
    upper = np.ceil(vertices[:, :2].max(axis=0) / STEP).astype(int) + 2
    width, height = upper - lower + 1
    atlas = {}
    # Same upward-face test as the viewer. Rasterize exact triangle heights.
    for triangle in vertices[faces]:
        a, b, c = triangle
        normal = np.cross(b - a, c - a)
        if normal[2] <= .6 * np.linalg.norm(normal):
            continue
        lo = np.floor(triangle[:, :2].min(axis=0) / STEP).astype(int)
        hi = np.floor(triangle[:, :2].max(axis=0) / STEP).astype(int)
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        x, y = (xx + .5) * STEP, (yy + .5) * STEP
        inside = np.ones(x.shape, bool)
        for p, q in [(a, b), (b, c), (c, a)]:
            inside &= (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]) >= -1e-7
        z = a[2] - (normal[0] * (x - a[0]) + normal[1] * (y - a[1])) / normal[2]
        for ix, iy, iz in zip(xx[inside], yy[inside], z[inside]):
            atlas.setdefault((int(ix), int(iy)), {})[round(float(iz) / .1)] = float(iz)
    layers = max(map(len, atlas.values()))
    floor_image = np.full((height, width, layers), -np.inf, dtype=np.float32)
    for (x, y), zs in atlas.items():
        floor_image[y - lower[1], x - lower[0], :len(zs)] = sorted(zs.values())
    evidence, sampled = {}, 0
    inputs = [folder / 'captures.json', folder / 'walkmesh.obj']
    axis = 2 * (np.arange(256) + .5) / 256 - 1
    uu, vv = np.meshgrid(axis, axis)
    for capture_index, capture in enumerate(captures):
        for face, basis in enumerate(capture['face_bases_center_right_down']):
            filename = folder / 'capdepth' / capture['id'] / f'face{face}.u16'
            if not filename.exists():
                continue
            inputs.append(filename)
            ranges = np.fromfile(filename, dtype='<u2').reshape(256, 256) / 1000
            centre, right, down = np.asarray(basis)
            ray = centre + uu[..., None] * right + vv[..., None] * down
            ray /= np.linalg.norm(ray, axis=2)[..., None]
            points = np.asarray(capture['position']) + ranges[..., None] * ray
            xy = np.floor(points[..., :2] / STEP).astype(int)
            ix, iy = xy[..., 0] - lower[0], xy[..., 1] - lower[1]
            valid = (ranges > 0) & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            p, cells = points[valid], xy[valid]
            floors = floor_image[iy[valid], ix[valid]]
            floors = np.where(floors <= p[:, 2, None], floors, -np.inf).max(axis=1)
            valid = np.isfinite(floors) & (p[:, 2] - floors >= .3) & (p[:, 2] - floors <= 1.8)
            p, cells, floors = p[valid], cells[valid], floors[valid]
            sampled += len(p)
            # Group before Python aggregation; each capture contributes one bit.
            keys = np.c_[cells, np.rint(floors / .2).astype(int)]
            unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            lows, highs = np.full(len(unique), np.inf), np.full(len(unique), -np.inf)
            np.minimum.at(lows, inverse, p[:, 2]); np.maximum.at(highs, inverse, p[:, 2])
            for key, count, low, high in zip(map(tuple, unique.tolist()), counts, lows, highs):
                value = evidence.setdefault(key, [0, 0, float('inf'), -float('inf')])
                value[0] += int(count); value[1] |= 1 << capture_index
                value[2] = min(value[2], float(low)); value[3] = max(value[3], float(high))
        if capture_index % 20 == 0:
            print(sid, 'captures', capture_index + 1, '/', len(captures), flush=True)
    seeds = {key: v for key, v in evidence.items() if v[0] >= MIN_POINTS and v[1].bit_count() >= 2}
    dilated = {}
    for (x, y, floor), value in seeds.items():
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (x + dx, y + dy, floor)
                v = dilated.setdefault(key, [value[2], value[3]])
                v[0] = min(v[0], value[2]); v[1] = max(v[1], value[3])
    doors = []
    rooms_file = PACK / sid / 'rooms.json'
    inputs.append(rooms_file)
    rooms = json.loads(rooms_file.read_text())
    for floor in rooms['floors']:
        for edge in floor['edges']:
            if len(edge.get('vertices', [])) != 2:
                continue
            a, b = [np.array([v['position']['x'], v['position']['y']]) for v in edge['vertices']]
            direction = b - a; direction /= max(np.linalg.norm(direction), 1e-10)
            for opening in edge.get('openings', []):
                if 'door' in opening['type']:
                    centre = a + (b - a) * opening['relativeCenter']
                    doors.append((centre - direction * opening['width'] / 2,
                                  centre + direction * opening['width'] / 2))
    removed = {'capture': 0, 'door': 0, 'stairs': 0, 'route': 0}
    nav_file = OUT / f'{sid}_navigation.json'
    if nav_file.exists():
        inputs.append(nav_file)
    routes = json.loads(nav_file.read_text()).get('protected', []) if nav_file.exists() else []
    route_index = {}
    for px, py, z in routes:
        route_index.setdefault((int(np.floor(px)), int(np.floor(py))), []).append((px, py, z))
    kept = []
    for (x, y, level), (low, high) in sorted(dilated.items()):
        p = [(x + .5) * STEP, (y + .5) * STEP]
        reason = None
        if any(rectangle_distance(x, y, *c['position'][:2]) < .35 for c in captures):
            reason = 'capture'
        elif any(segment_distance(p, a, b) < .38 for a, b in doors):
            reason = 'door'
        elif any(segment_distance(p, a, b) < .40 for route in STAIRS[sid] for a, b in zip(route, route[1:])):
            reason = 'stairs'
        elif any(low <= z + 1.6 and high >= z + .2 and rectangle_distance(x, y, px, py) < .36
                 for bx in range(int(np.floor(p[0])) - 1, int(np.floor(p[0])) + 2)
                 for by in range(int(np.floor(p[1])) - 1, int(np.floor(p[1])) + 2)
                 for px, py, z in route_index.get((bx, by), [])):
            reason = 'route'
        if reason:
            removed[reason] += 1
        else:
            kept.append([x, y, round(level * .2, 2), round(low, 3), round(high, 3)])
    result = {'version': 1, 'step': STEP, 'minPoints': MIN_POINTS, 'minCaptures': 2,
              'dilationCells': 1, 'columns': ['x', 'y', 'floor', 'low', 'high'], 'cells': kept}
    result, consistency = reconcile(folder, result)
    kept = result['cells']
    target = folder / 'obstacles.json'
    target.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    audit = {'sid': sid, 'captures': len(captures), 'retainedHeightPoints': sampled,
             'evidenceCells': len(evidence), 'multiCaptureSeeds': len(seeds), 'dilatedCells': len(dilated),
             'removed': removed, 'solidCells': len(kept), 'doorwaysProtected': len(doors),
             'minimumPoints': MIN_POINTS, 'minimumCaptures': 2,
             'depthConsistency': consistency,
             'inputsSha256': {str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f):
                              hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}}
    (OUT / f'{sid}_obstacles-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (OUT / f'{sid}_evidence.json').write_text(json.dumps([[*k, *v[:1], v[1].bit_count()] for k, v in seeds.items()], separators=(',', ':')))
    draw_map(sid, lower, upper, evidence, kept, captures, folder)
    print(json.dumps({k: v for k, v in audit.items() if k != 'inputsSha256'}), flush=True)


def draw_map(sid, lower, upper, evidence, kept, captures, folder):
    scale = 3
    size = tuple(((upper - lower + 1) * scale).astype(int))
    image = Image.new('RGB', (size[0], size[1] + 60), 'white'); draw = ImageDraw.Draw(image)
    def pt(x, y):
        return ((x / STEP - lower[0]) * scale, (upper[1] - y / STEP) * scale + 60)
    # Exact indexed footprint will be overlaid by check.mjs from collider audit.
    for x, y, _ in evidence:
        u, v = pt((x + .5) * STEP, (y + .5) * STEP)
        draw.rectangle((u - 1, v - 1, u + 1, v + 1), fill='#ffb8a8')
    for x, y, *_ in kept:
        u, v = pt((x + .5) * STEP, (y + .5) * STEP)
        draw.rectangle((u - 1, v - 1, u + 1, v + 1), fill='#c05030')
    collider = OUT / f'{sid}_collider-footprint.json'
    if collider.exists():
        for x, y in json.loads(collider.read_text()):
            u, v = pt(x, y); draw.point((u, v), fill='#2a4f95')
    for capture in captures:
        u, v = pt(*capture['position'][:2]); draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill='#008050')
    draw.text((10, 7), sid + ' | top-down, all storeys overlaid; grid 0.10 m; Z up', fill='black')
    draw.text((10, 25), 'Blue: existing collider | pale red: capdepth 0.3-1.8 m above floor', fill='black')
    draw.text((10, 42), 'Dark red: added cells | green: captures (0.35 m clear disks)', fill='black')
    image.save(OUT / f'{sid}_occupancy.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('sid', nargs='*', default=['VN4b1MGnnCR', 'HFwcH424n9w'])
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    for sid in args.sid:
        build(sid)
