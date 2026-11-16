#!/usr/bin/env python3
"""W14 CPU extraction: numpy geometry, stdlib I/O, macOS sips for real-photo crops.

python3 tools/extract_objects.py --web-root /Users/phanan/HackPrinceton/data/splat/web \
  --evidence /Users/phanan/HackPrinceton/experiments/w14_furniture
No learned models, image synthesis, viewer changes, or non-numpy Python packages.
"""
import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import zlib

import numpy as np

SIZE = 256
VOXEL = .04
BIAS = 1 << 20
MASK = (1 << 21) - 1
HOMES = ('VN4b1MGnnCR', 'HFwcH424n9w')


def pack(q):
    q = np.asarray(q, dtype=np.int64) + BIAS
    if np.any((q < 0) | (q > MASK)):
        raise ValueError('Voxel coordinates exceed 21-bit signed range')
    return (q[..., 0] << 42) | (q[..., 1] << 21) | q[..., 2]


def unpack(keys):
    return np.stack((keys >> 42, (keys >> 21) & MASK, keys & MASK), axis=-1) - BIAS


def read_obj(path):
    vertices, faces = [], []
    with path.open() as source:
        for line in source:
            if line.startswith('v '):
                vertices.append(tuple(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                ids = [int(s.split('/')[0]) for s in line.split()[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                faces.extend((ids[0], ids[i], ids[i+1]) for i in range(1, len(ids)-1))
    v, f = np.asarray(vertices), np.asarray(faces, dtype=np.int32)
    t = v[f]
    cross = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    length = np.linalg.norm(cross, axis=1)
    good = length > 1e-10
    return t[good], cross[good] / length[good, None], length[good] / 2


def faces(home, captures):
    uv = 2*(np.arange(SIZE) + .5)/SIZE - 1
    for capture in captures:
        position = np.asarray(capture['position'], dtype=float)
        bases = np.asarray(capture['face_bases_center_right_down'], dtype=float)
        for face, basis in enumerate(bases):
            path = home/'capdepth'/capture['id']/f'face{face}.u16'
            depth = np.fromfile(path, dtype='<u2')
            if depth.size != SIZE*SIZE:
                raise ValueError(f'Expected 256x256 u16: {path}')
            depth = depth.reshape(SIZE, SIZE)
            rays = basis[0] + uv[None, :, None]*basis[1] + uv[:, None, None]*basis[2]
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            valid = depth > 0
            points = position + rays[valid]*(depth[valid, None]/1000)
            yield capture['id'], face, valid, pack(np.floor(points/VOXEL).astype(np.int64))


class Floors:
    """Exact XY triangle raster at voxel centres; sorted multistorey height queries."""
    def __init__(self, triangles, normals, areas, captures):
        zcam = np.array([c['position'][2] for c in captures])
        bins = np.arange(math.floor(zcam.min()*5)/5, zcam.max()+.4, .2)
        hist, edges = np.histogram(zcam, bins)
        modes = []
        for i in np.argsort(-hist):
            height = (edges[i]+edges[i+1])/2
            if hist[i] < max(4, len(captures)*.08):
                continue
            if all(abs(height-m) > 1.2 for m in modes):
                modes.append(height)
        if not modes:
            raise ValueError('No supported capture-height storey')
        z = triangles.mean(axis=1)[:, 2]
        horizontal = np.abs(normals[:, 2]) >= .90
        flat = triangles[horizontal]
        a, b = flat[:, 1, :2]-flat[:, 0, :2], flat[:, 2, :2]-flat[:, 0, :2]
        determinant = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        levels = []
        for camera in sorted(modes):
            observed = []
            for capture in captures:
                position = np.asarray(capture['position'])
                if abs(position[2]-camera) > .45:
                    continue
                delta = position[:2]-flat[:, 0, :2]
                u = (delta[:, 0]*b[:, 1]-delta[:, 1]*b[:, 0])/determinant
                v = (a[:, 0]*delta[:, 1]-a[:, 1]*delta[:, 0])/determinant
                height = flat[:, 0, 2]+u*(flat[:, 1, 2]-flat[:, 0, 2])+v*(flat[:, 2, 2]-flat[:, 0, 2])
                beneath = (u >= 0) & (v >= 0) & (u+v <= 1)
                beneath &= (position[2]-height >= .9) & (position[2]-height <= 2.1)
                if beneath.any():
                    observed.append(float(height[beneath].max()))
            if len(observed) < 3:
                raise ValueError('Insufficient local walkmesh floors under capture group')
            levels.append(float(np.median(observed)))
        self.levels = np.array(sorted(levels))
        # The supplied walkmesh is unfiltered geometry: reject furniture tops,
        # ceilings and roofs using the pose-supported floor bands, not normal sign.
        select = horizontal & (np.min(abs(z[:, None]-self.levels), axis=1) <= .22)
        self.triangles = triangles[select]
        cells, heights = [], []
        for tri in self.triangles:
            lo = np.ceil(tri[:, :2].min(axis=0)/VOXEL-.5).astype(int)
            hi = np.floor(tri[:, :2].max(axis=0)/VOXEL-.5).astype(int)
            if np.any(hi < lo):
                continue
            xx, yy = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
            xy = np.column_stack((xx.ravel(), yy.ravel()))
            delta = (xy+.5)*VOXEL - tri[0, :2]
            a, b = tri[1, :2]-tri[0, :2], tri[2, :2]-tri[0, :2]
            det = a[0]*b[1]-a[1]*b[0]
            if abs(det) < 1e-12:
                continue
            u = (delta[:, 0]*b[1]-delta[:, 1]*b[0])/det
            v = (a[0]*delta[:, 1]-a[1]*delta[:, 0])/det
            inside = (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7)
            xy, u, v = xy[inside], u[inside], v[inside]
            if len(xy):
                cells.append(((xy[:, 0]+BIAS) << 21) | (xy[:, 1]+BIAS))
                heights.append(tri[0, 2]+u*(tri[1, 2]-tri[0, 2])+v*(tri[2, 2]-tri[0, 2]))
        if not cells:
            raise ValueError('No rasterized floors')
        xy = np.concatenate(cells)
        zmm = np.rint(np.concatenate(heights)*1000).astype(np.int64)
        # Separate structured fields avoid overflow from multiplying 42-bit XY keys.
        self.keys = np.unique(np.rec.fromarrays([xy, zmm], names='xy,z'))

    def beneath(self, points):
        q = np.floor(points[:, :2]/VOXEL).astype(np.int64) + BIAS
        xy = (q[:, 0] << 21) | q[:, 1]
        query = np.rec.fromarrays([xy, np.floor(points[:, 2]*1000).astype(np.int64)], names='xy,z')
        index = np.searchsorted(self.keys, query, side='right')-1
        safe = np.maximum(index, 0)
        exists = (index >= 0) & (self.keys.xy[safe] == xy)
        return np.where(exists, self.keys.z[safe]/1000, np.nan)


def wall_planes(triangles, normals, areas):
    select = (np.abs(normals[:, 2]) < .15) & (areas > 1e-8)
    tri, n, area = triangles[select], normals[select].copy(), areas[select]
    n[:, 2] = 0
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[n[:, 0] < 0] *= -1
    centers = tri.mean(axis=1)
    angle = np.arctan2(n[:, 1], n[:, 0])
    # Quantized Hough votes seed area-weighted fits; grow each seed to nearby
    # coplanar triangles. Bounds keep planes finite instead of slicing all rooms.
    anglebin = np.rint(angle/np.deg2rad(3)).astype(int)
    canonical = np.column_stack((np.cos(anglebin*np.deg2rad(3)), np.sin(anglebin*np.deg2rad(3)), np.zeros(len(n))))
    distance = np.einsum('ij,ij->i', centers, canonical)
    bins, inv = np.unique(np.column_stack((anglebin, np.rint(distance/.04).astype(int))), axis=0, return_inverse=True)
    votes = np.bincount(inv, weights=area)
    used = np.zeros(len(n), dtype=bool)
    planes = []
    for seed in np.argsort(-votes):
        if votes[seed] < .35:
            break
        chosen = (inv == seed) & ~used
        if area[chosen].sum() < .2:
            continue
        normal = np.average(n[chosen], axis=0, weights=area[chosen])
        normal /= np.linalg.norm(normal)
        offset = float(np.average(centers[chosen] @ normal, weights=area[chosen]))
        compatible = ~used & (n @ normal > math.cos(math.radians(6)))
        compatible &= np.abs(centers @ normal-offset) <= .04
        if area[compatible].sum() <= 1.5:
            continue
        pts = tri[compatible].reshape(-1, 3)
        tangent = np.array([-normal[1], normal[0], 0.])
        along = pts @ tangent
        planes.append(dict(normal=normal.tolist(), offset=offset, tangent=tangent.tolist(),
                           extent=[float(along.min()), float(along.max())],
                           z=[float(pts[:, 2].min()), float(pts[:, 2].max())],
                           area_m2=float(area[compatible].sum())))
        used |= compatible
    return planes


def remove_structure(points, planes, triangles, normals, areas, floors):
    remove = np.zeros(len(points), dtype=bool)
    for plane in planes:
        along = points @ np.array(plane['tangent'])
        near = np.abs(points @ np.array(plane['normal'])-plane['offset']) <= .08
        near &= (along >= plane['extent'][0]-.08) & (along <= plane['extent'][1]+.08)
        near &= (points[:, 2] >= plane['z'][0]-.08) & (points[:, 2] <= plane['z'][1]+.08)
        remove |= near
    wall_count = int(remove.sum())
    centers = triangles.mean(axis=1)
    horizontal = np.abs(normals[:, 2]) > .9
    # Ceilings are horizontal collision surfaces >=2.1 m above an actual local
    # floor, not low table tops. Rasterized XY patches bound ceiling rejection.
    candidates = np.flatnonzero(horizontal)
    local = floors.beneath(centers[candidates] - [0, 0, .10])
    candidates = candidates[(centers[candidates, 2]-local) >= 2.1]
    ceiling_xy, ceiling_z, ceiling_nz = [], [], []
    for tri in triangles[candidates]:
        lo = np.ceil(tri.min(axis=0)/VOXEL-.5).astype(int)
        hi = np.floor(tri.max(axis=0)/VOXEL-.5).astype(int)
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        a, b = tri[1, :2]-tri[0, :2], tri[2, :2]-tri[0, :2]
        det = a[0]*b[1]-a[1]*b[0]
        if abs(det) < 1e-12:
            continue
        d = (xy+.5)*VOXEL-tri[0, :2]
        u = (d[:, 0]*b[1]-d[:, 1]*b[0])/det
        v = (a[0]*d[:, 1]-a[1]*d[:, 0])/det
        z = tri[0, 2]+u*(tri[1, 2]-tri[0, 2])+v*(tri[2, 2]-tri[0, 2])
        inside = (u >= 0) & (v >= 0) & (u+v <= 1)
        if inside.any():
            selected = xy[inside]+BIAS
            ceiling_xy.append((selected[:, 0] << 21) | selected[:, 1])
            ceiling_z.append(z[inside])
            normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            ceiling_nz.append(np.full(inside.sum(), abs(normal[2])/np.linalg.norm(normal)))
    if ceiling_xy:
        xy, zz, nz = map(np.concatenate, (ceiling_xy, ceiling_z, ceiling_nz))
        ceiling = np.rec.fromarrays([xy, zz], names='xy,z')
        order = np.argsort(ceiling)
        ceiling, nz = ceiling[order], nz[order]
        q = np.floor(points[:, :2]/VOXEL).astype(np.int64)+BIAS
        xy = (q[:, 0] << 21) | q[:, 1]
        query = np.rec.fromarrays([xy, points[:, 2]], names='xy,z')
        above = np.searchsorted(ceiling, query)
        near = np.zeros(len(points), dtype=bool)
        for index in (above, above-1):
            safe = np.clip(index, 0, len(ceiling)-1)
            found = (index >= 0) & (index < len(ceiling)) & (ceiling.xy[safe] == xy)
            # At the same XY centre the exact point-to-plane distance is dz*nz.
            near |= found & (np.abs(ceiling.z[safe]-points[:, 2])*nz[safe] <= .08)
        remove |= near
    return remove, wall_count, int(remove.sum())-wall_count, len(candidates)


def components(keys):
    remaining = set(map(int, keys))
    offsets = [dx*(1 << 42)+dy*(1 << 21)+dz for dx in (-1, 0, 1)
               for dy in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dy or dz]
    while remaining:
        first = remaining.pop()
        queue, component = deque([first]), [first]
        while queue:
            key = queue.popleft()
            for offset in offsets:
                candidate = key+offset
                if candidate in remaining:
                    remaining.remove(candidate)
                    queue.append(candidate)
                    component.append(candidate)
        yield np.sort(np.asarray(component, dtype=np.int64))


def rle(q):
    lo, hi = q.min(axis=0), q.max(axis=0)
    shape = hi-lo+1
    local = q-lo
    linear = np.sort((local[:, 0]*shape[1]+local[:, 1])*shape[2]+local[:, 2])
    starts = np.r_[0, np.flatnonzero(np.diff(linear) != 1)+1]
    ends = np.r_[starts[1:], len(linear)]
    runs = np.column_stack((linear[starts], ends-starts)).ravel().tolist()
    return dict(encoding='rle_xyz_z_fastest', origin=lo.tolist(), shape=shape.tolist(), runs=runs)


def describe(keys, floors):
    q = unpack(keys)
    points = (q+.5)*VOXEL
    footprint = np.unique(q[:, :2], axis=0)
    xy = (footprint+.5)*VOXEL
    _, vectors = np.linalg.eigh(np.cov(xy.T))
    axis = vectors[:, -1]
    if axis[0] < 0:
        axis *= -1
    yaw = float(math.atan2(axis[1], axis[0]))
    rotation = np.array([[axis[0], -axis[1]], [axis[1], axis[0]]])
    rotated = xy @ rotation
    padding = VOXEL*(abs(axis[0])+abs(axis[1]))/2
    lo, hi = rotated.min(axis=0)-padding, rotated.max(axis=0)+padding
    center = ((lo+hi)/2) @ rotation.T
    zlo, zhi = q[:, 2].min()*VOXEL, (q[:, 2].max()+1)*VOXEL
    local = floors.beneath(points)
    floor = float(np.nanmedian(local))
    return dict(voxels=rle(q), voxel_count=len(q), footprint_m2=len(footprint)*VOXEL**2,
                obb=dict(centre=[*center.tolist(), (zlo+zhi)/2], size=[*(hi-lo).tolist(), zhi-zlo], yaw=yaw),
                base_height=float(zlo), floor_height=floor, storey=int(np.argmin(abs(floors.levels-floor))))


def png_bytes(rgb):
    h, w, _ = rgb.shape
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))
    raw = b''.join(b'\0'+row.tobytes() for row in rgb.astype(np.uint8))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))+chunk(b'IDAT', zlib.compress(raw))+chunk(b'IEND', b'')


DIGITS = ['111101101101111', '010110010010111', '111001111100111', '111001111001111',
          '101101111001001', '111100111001111', '111100111101111', '111001001001001',
          '111101111101111', '111101111001111']


def number(image, value, x, y, scale=3):
    for digit in str(value):
        bits = np.array(list(DIGITS[int(digit)]), dtype=int).reshape(5, 3)
        bits = np.repeat(np.repeat(bits, scale, axis=0), scale, axis=1).astype(bool)
        h, w = bits.shape
        if x >= 0 and y >= 0 and x+w <= image.shape[1] and y+h <= image.shape[0]:
            image[y:y+h, x:x+w][bits] = 0
        x += 4*scale


def sips(*args):
    subprocess.run(['sips', *map(str, args)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def bmp_rgb(path):
    data = path.read_bytes()
    offset = struct.unpack_from('<I', data, 10)[0]
    w, h, planes, bits, compression = struct.unpack_from('<iiHHI', data, 18)
    if bits not in (24, 32) or compression != 0 or planes != 1:
        raise ValueError(f'Unsupported sips BMP: {bits}/{compression}')
    stride = ((w*bits+31)//32)*4
    pixels = np.frombuffer(data, np.uint8, offset=offset, count=abs(h)*stride).reshape(abs(h), stride)
    rgb = pixels[:, :w*(bits//8)].reshape(abs(h), w, bits//8)[:, :, 2::-1]
    return rgb[::-1].copy() if h > 0 else rgb.copy()


class Outputs:
    def __init__(self, evidence):
        spec = importlib.util.spec_from_file_location('w14_undo', evidence/'undo.py')
        self.undo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.undo)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.undo.checkpoint(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
            out.write(data)
            temp = Path(out.name)
        temp.replace(path)
        self.undo.finish(path)

    def json(self, path, value):
        self.write(path, (json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n').encode())

    def remove(self, path):
        self.undo.checkpoint(path)
        path.unlink()
        self.undo.finish(path)


def thumbnails(home, objects, best, output, evidence):
    columns, tw, th = 8, 192, 180
    sheet = np.full((max(1, math.ceil(len(objects)/columns))*th, columns*tw, 3), 245, np.uint8)
    with tempfile.TemporaryDirectory(prefix='w14-photos-') as tmp:
        tmp = Path(tmp)
        for obj in objects:
            oid = obj['id']
            if oid not in best:
                raise ValueError(f'Object {oid} has no real-photo support')
            score, pano, face, bounds, count = best[oid]
            photo = home/'captures2k'/pano/f'face{face}.jpg'
            dims = subprocess.check_output(['sips', '-g', 'pixelWidth', '-g', 'pixelHeight', str(photo)], text=True)
            if 'pixelWidth: 2048' not in dims or 'pixelHeight: 2048' not in dims:
                raise ValueError(f'Expected real 2048 photo: {photo}')
            x0, y0, x1, y1 = bounds
            crop, small, bitmap = tmp/'crop.jpg', tmp/'small.jpg', tmp/'small.bmp'
            sips('--cropToHeightWidth', y1-y0, x1-x0, '--cropOffset', y0, x0, photo, '--out', crop)
            output.write(home/'objects'/f'{oid}.jpg', crop.read_bytes())
            sips('-Z', 152, crop, '--out', small)
            sips('-s', 'format', 'bmp', small, '--out', bitmap)
            rgb = bmp_rgb(bitmap)
            x, y = ((oid-1) % columns)*tw, ((oid-1)//columns)*th
            sheet[y:y+rgb.shape[0], x:x+rgb.shape[1]] = rgb
            number(sheet, oid, x+4, y+157, 3)
            obj['thumbnail'] = f'objects/{oid}.jpg'
            obj['photo'] = dict(path=f'captures2k/{pano}/face{face}.jpg', crop_xyxy=bounds,
                                supported_samples=count, selection_score=score)
        png = tmp/'contact.png'
        png.write_bytes(png_bytes(sheet))
        jpg = tmp/'contact.jpg'
        sips('-s', 'format', 'jpeg', png, '--out', jpg)
        output.write(evidence/f'{home.name}_contact.jpg', jpg.read_bytes())


def top_view(home, objects, component_keys, floors, output, evidence):
    n = len(floors.levels)
    bounds = floors.triangles[:, :, :2].reshape(-1, 2)
    lo, hi = bounds.min(axis=0)-.5, bounds.max(axis=0)+.5
    scale = min(1800/(hi-lo).max(), 60)
    width, height = np.ceil((hi-lo)*scale).astype(int)+1
    image = np.full((height, width*n, 3), 255, np.uint8)
    for storey in range(n):
        fk = floors.keys
        assigned = np.argmin(abs(fk.z[:, None]/1000-floors.levels), axis=1)
        xykey = fk.xy[assigned == storey]
        xy = np.column_stack(((xykey >> 21)-BIAS, (xykey & MASK)-BIAS))
        pixel = np.rint(((xy+.5)*VOXEL-lo)*scale).astype(int)
        image[height-1-pixel[:, 1], pixel[:, 0]+storey*width] = 225
        number(image, storey, storey*width+8, 8, 4)
    labels = []
    occupied = [(i*width, 0, i*width+30, 36) for i in range(n)]
    for obj, keys in zip(objects, component_keys):
        q = unpack(keys)
        pixel = np.rint(((q[:, :2]+.5)*VOXEL-lo)*scale).astype(int)
        storey = obj['storey']
        rgb = np.array([(obj['id']*71)%160+50, (obj['id']*113)%160+50, (obj['id']*157)%160+50])
        image[height-1-pixel[:, 1], pixel[:, 0]+storey*width] = rgb
        center = np.rint((np.array(obj['obb']['centre'][:2])-lo)*scale).astype(int)
        anchor = np.array([center[0]+storey*width, height-1-center[1]])
        label_width, label_height = len(str(obj['id']))*12+4, 21
        found = None
        for radius in range(20):
            offsets = [(0, 0)] if not radius else [
                (dx*22, dy*22) for dx in range(-radius, radius+1)
                for dy in range(-radius, radius+1) if max(abs(dx), abs(dy)) == radius]
            for dx, dy in offsets:
                x = int(np.clip(anchor[0]+dx, storey*width, (storey+1)*width-label_width))
                y = int(np.clip(anchor[1]+dy, 0, height-label_height))
                box = (x, y, x+label_width, y+label_height)
                if all(box[2]+2 < b[0] or box[0]-2 > b[2] or box[3]+2 < b[1] or box[1]-2 > b[3] for b in occupied):
                    found = box
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError('Could not place a readable object number')
        occupied.append(found)
        labels.append((obj['id'], anchor, found))
    # Draw leader lines first, then opaque label backgrounds, so labels remain
    # readable even when many small components share a room.
    for oid, anchor, box in labels:
        target = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
        count = int(np.linalg.norm(target-anchor))+1
        line = np.rint(np.linspace(anchor, target, count)).astype(int)
        image[line[:, 1], line[:, 0]] = 170
    for oid, anchor, (x0, y0, x1, y1) in labels:
        image[y0:y1, x0:x1] = 255
        number(image, oid, x0+2, y0+3, 3)
    output.write(evidence/f'{home.name}_objects_top.png', png_bytes(image))


def extract(home, evidence):
    started = time.monotonic()
    output = Outputs(evidence)
    capture_doc = json.loads((home/'captures.json').read_text())
    index = json.loads((home/'capdepth/index.json').read_text())
    if capture_doc.get('up') != 'Z' or capture_doc.get('units') != 'metres':
        raise ValueError('Expected untransformed Z-up metre captures')
    if index.get('format') != 'u16_range_mm' or index.get('size') != SIZE:
        raise ValueError('Expected 256x256 W9 ray ranges')
    captures = capture_doc['captures']
    ids = [c['id'] for c in captures]
    if len(ids) != len(set(ids)) or set(ids) != set(index['panoramas']):
        raise ValueError('Capture/depth index membership mismatch')
    for capture in captures:
        pid = capture['id']
        if not isinstance(pid, str) or pid in ('', '.', '..') or Path(pid).name != pid:
            raise ValueError('Unsafe capture id')
        basis = np.asarray(capture['face_bases_center_right_down'])
        if basis.shape != (6, 3, 3) or not np.allclose(basis @ basis.transpose(0, 2, 1), np.eye(3), atol=1e-5):
            raise ValueError(f'Invalid world-space bases: {pid}')
        for face in range(6):
            depth = home/'capdepth'/pid/f'face{face}.u16'
            if depth.stat().st_size != SIZE*SIZE*2 or not (home/'captures2k'/pid/f'face{face}.jpg').is_file():
                raise ValueError(f'Missing complete depth/photo: {pid}/{face}')
    errors = []
    byid = {c['id']: c for c in captures}
    for point in index['check_points']:
        c = byid[point['pano']]
        b = np.asarray(c['face_bases_center_right_down'][point['face']])
        ray = b[0]+(2*(point['px']+.5)/SIZE-1)*b[1]+(2*(point['py']+.5)/SIZE-1)*b[2]
        raw = np.fromfile(home/'capdepth'/c['id']/f"face{point['face']}.u16", dtype='<u2').reshape(SIZE, SIZE)
        pos = np.asarray(c['position'])+raw[point['py'], point['px']]/1000*ray/np.linalg.norm(ray)
        errors.append(float(np.linalg.norm(pos-np.asarray(point['world_xyz']))))
    if not errors or max(errors) > .02:
        raise ValueError('W9 world checkpoint error above 2 cm')
    print(f'{home.name}: back-projecting every sample from {len(captures)} captures', flush=True)
    batches, current, samples = [], [], 0
    for pid, face, valid, keys in faces(home, captures):
        current.append(keys)
        samples += len(keys)
        if face == 5:
            batches.append(np.unique(np.concatenate(current)))
            current = []
    keys, support = np.unique(np.concatenate(batches), return_counts=True)
    counts = dict(valid_samples=samples, observed_voxels=len(keys), supported_voxels=int((support >= 2).sum()))
    keys = keys[support >= 2]
    del batches, support
    print(f'{home.name}: {len(keys):,} voxels supported by >=2 distinct captures; loading floors', flush=True)
    floortri, floorn, floora = read_obj(home/'walkmesh.obj')
    floors = Floors(floortri, floorn, floora, captures)
    points = (unpack(keys)+.5)*VOXEL
    heights = floors.beneath(points)
    valid = np.isfinite(heights) & (points[:, 2]-heights >= .04) & (points[:, 2]-heights <= 2.3)
    keys, points = keys[valid], points[valid]
    counts['above_local_floor_voxels'] = len(keys)
    print(f'{home.name}: floors {floors.levels.tolist()}; {len(keys):,} floor-filtered voxels; loading collision', flush=True)
    del floortri, floorn, floora
    triangles, normals, areas = read_obj(home/'collision.obj')
    planes = wall_planes(triangles, normals, areas)
    removed, walls, ceilings, ceiling_triangles = remove_structure(points, planes, triangles, normals, areas, floors)
    keys = keys[~removed]
    counts.update(wall_planes=len(planes), wall_removed_voxels=walls, ceiling_removed_voxels=ceilings,
                  ceiling_triangles=ceiling_triangles, nonstructural_voxels=len(keys))
    del triangles, normals, areas, points
    print(f'{home.name}: {len(planes)} finite wall planes; {len(keys):,} residual voxels; 26-connectivity', flush=True)
    components_kept, tiny, huge = [], 0, 0
    for component in components(keys):
        area = len(np.unique(unpack(component)[:, :2], axis=0))*VOXEL**2
        if area < .04-1e-9:
            tiny += 1
        elif area > 6+1e-9:
            huge += 1
        else:
            components_kept.append(component)
    components_kept.sort(key=lambda c: int(c[0]))
    if len(components_kept) > 255:
        raise ValueError(f'{len(components_kept)} objects exceeds u8 capacity; refusing silent truncation')
    objects = [dict(id=i+1, **describe(c, floors)) for i, c in enumerate(components_kept)]
    labelkeys = np.concatenate(components_kept) if components_kept else np.array([], dtype=np.int64)
    labels = np.concatenate([np.full(len(c), i+1, np.uint8) for i, c in enumerate(components_kept)]) if objects else np.array([], np.uint8)
    order = np.argsort(labelkeys)
    labelkeys, labels = labelkeys[order], labels[order]
    best, mask_samples = {}, 0
    for pid, face, valid, keys in faces(home, captures):
        idx = np.searchsorted(labelkeys, keys)
        hit = idx < len(labelkeys)
        if len(labelkeys):
            hit &= labelkeys[np.minimum(idx, len(labelkeys)-1)] == keys
        mask = np.zeros((SIZE, SIZE), np.uint8)
        sample_labels = np.zeros(len(keys), np.uint8)
        sample_labels[hit] = labels[idx[hit]]
        mask[valid] = sample_labels
        output.write(home/'objmask'/pid/f'face{face}.u8', mask.tobytes())
        mask_samples += int(hit.sum())
        for oid in np.unique(sample_labels[hit]):
            yy, xx = np.nonzero(mask == oid)
            count = len(xx)
            x0, y0 = max(0, int(xx.min())*8-24), max(0, int(yy.min())*8-24)
            x1, y1 = min(2048, (int(xx.max())+1)*8+24), min(2048, (int(yy.max())+1)*8+24)
            score = count/(((x1-x0)*(y1-y0))**.3)
            if int(oid) not in best or score > best[int(oid)][0]:
                best[int(oid)] = (score, pid, face, [x0, y0, x1, y1], count)
    thumbnails(home, objects, best, output, evidence)
    for old in (home/'objects').glob('*.jpg'):
        if old.stem.isdigit() and 1 <= int(old.stem) <= 255 and int(old.stem) > len(objects):
            output.remove(old)
    top_view(home, objects, components_kept, floors, output, evidence)
    histogram, edges = np.histogram([o['footprint_m2'] for o in objects], bins=[.04, .1, .25, .5, 1, 2, 4, 6.000001])
    counts.update(tiny_components=tiny, huge_components=huge, objects=len(objects),
                  object_voxels=len(labelkeys), masked_samples=mask_samples, mask_faces=len(captures)*6)
    source_paths = ['captures.json', 'capdepth/index.json', 'walkmesh.obj', 'collision.obj']
    sources = {s: hashlib.sha256((home/s).read_bytes()).hexdigest() for s in source_paths}
    doc = dict(version=1, model_id=home.name, up='Z', units='metres', voxel_size=VOXEL,
               voxel_origin=[0, 0, 0], storeys=[dict(id=i, floor_height=float(z)) for i, z in enumerate(floors.levels)],
               mask=dict(size=SIZE, type='u8', none=0, path='objmask/<pano>/face<k>.u8', alignment='capdepth pixel centres'),
               parameters=dict(min_capture_support=2, height_above_local_floor=[.04, 2.3], wall_distance=.08,
                               min_wall_area=1.5, connectivity=26, footprint_range=[.04, 6], floor_band=.22),
               wall_planes=planes, objects=objects, counts=counts,
               footprint_histogram=dict(edges_m2=edges.tolist(), counts=histogram.tolist()),
               checkpoint_max_error_m=max(errors), sources=sources,
               runtime_seconds=round(time.monotonic()-started, 3))
    output.json(home/'objects.json', doc)
    output.json(evidence/f'{home.name}_stats.json', {k: v for k, v in doc.items() if k not in ('objects', 'wall_planes')})
    print(json.dumps(dict(home=home.name, counts=counts, histogram=doc['footprint_histogram'],
                          seconds=doc['runtime_seconds']), indent=2), flush=True)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web-root', type=Path, default=Path('/Users/phanan/HackPrinceton/data/splat/web'))
    parser.add_argument('--evidence', type=Path, default=Path('/Users/phanan/HackPrinceton/experiments/w14_furniture'))
    parser.add_argument('--sid', choices=HOMES, action='append')
    args = parser.parse_args()
    if not shutil.which('sips'):
        raise SystemExit('macOS sips is required for real-photo cropping; geometry needs only numpy')
    args.evidence.mkdir(parents=True, exist_ok=True)
    for sid in args.sid or HOMES:
        extract(args.web_root/sid, args.evidence)


if __name__ == '__main__':
    main()
