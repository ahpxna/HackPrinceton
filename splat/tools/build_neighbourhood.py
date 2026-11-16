#!/usr/bin/env python3
"""W18: local-demo-only Vexcel DSM/ortho neighbourhoods; no viewer writes.

python3 -B tools/build_neighbourhood.py --root /Users/phanan/HackPrinceton --online
Default is cache-only. The existing demo.api_cache.credential is the sole key
reader. Acquisition uses one common polygon per home, pinned to its collection.
Dependencies: numpy, Pillow, matplotlib (existing demo environment).
"""
import argparse
import collections
import datetime as dt
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/w18-matplotlib')
from matplotlib.path import Path as PolygonPath

SIDS = ['VN4b1MGnnCR', 'HFwcH424n9w']
LIMIT = 15_000_000


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xmp(raw):
    with Image.open(io.BytesIO(raw)) as image:
        text = image.info.get('xmp', b'')
    if isinstance(text, bytes):
        text = text.decode(errors='replace')
    return dict(re.findall(r'<Vexcel:([^>]+)>(.*?)</Vexcel:[^>]+>', text))


def acquire(api, folder, name, endpoint, params, online):
    """Preserve original TIFF/JPEG bytes; never persist headers or request URLs."""
    target = folder / name
    signature = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
    receipt = target.with_suffix(target.suffix + '.receipt.json')
    if target.exists() and receipt.exists() and read(receipt)['request_sha256'] == signature:
        return target
    if target.exists():
        raise ValueError('Different acquisition parameters: use a fresh --cache-root')
    if not online:
        raise RuntimeError('Missing local product; acquisition requires --online')
    folder.mkdir(parents=True, exist_ok=True)
    # credential() alone reads .env at runtime. Its value is never a URL parameter.
    headers = {'X-API-KEY': api.credential('VEXCEL_API_KEY')}
    url = 'https://api.vexcelgroup.com/v2/' + endpoint
    # Match the existing DSM GET and demo image POST call paths.
    method = 'GET' if endpoint.startswith('dsm/') else 'POST'
    if method == 'GET':
        request = urllib.request.Request(url + '?' + urllib.parse.urlencode(params), headers=headers)
    else:
        headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(url, json.dumps(params).encode(), headers)
    try:
        with urllib.request.urlopen(request, timeout=100) as response:
            raw = response.read(24_000_001)
            status = response.status
        if len(raw) > 24_000_000 or not raw:
            raise ValueError('Empty product or crop exceeds 24 MB acquisition budget')
    except urllib.error.HTTPError as error:
        write(folder / (name + '.error.json'), {'endpoint': endpoint, 'status': error.code})
        raise RuntimeError(endpoint + ': HTTP ' + str(error.code)) from None
    except (urllib.error.URLError, TimeoutError):
        write(folder / (name + '.error.json'), {'endpoint': endpoint, 'status': 'transport_failure'})
        raise RuntimeError(endpoint + ': transport failure; details suppressed') from None
    metadata = xmp(raw)
    target.write_bytes(raw)
    write(receipt, {'endpoint': endpoint, 'method': method, 'request_sha256': signature, 'parameters': params,
                    'status': status, 'bytes': len(raw), 'sha256': sha(target),
                    'downloaded_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    write(target.with_suffix('.metadata.json'), metadata)
    return target


def decode_dsm(path, tr):
    """Decode declared TIFF endian explicitly (Pillow float decoding is unsafe here)."""
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        tags = dict(image.tag_v2)
        width, height = image.size
    if tags.get(339) != (3,) or tags.get(258) != (32,) or tags.get(277, 1) != 1:
        raise ValueError('Expected single-band float32 DSM')
    if tags.get(317, 1) != 1 or tags.get(259) not in (1, 8, 32946):
        raise ValueError('Unsupported TIFF predictor/compression')
    endian = '>' if raw[:2] == b'MM' else '<' if raw[:2] == b'II' else None
    if endian is None or width * height > 4_000_000:
        raise ValueError('Invalid TIFF byte order/size')

    def block(offset, count, size):
        data = raw[offset:offset + count]
        if tags[259] != 1:
            decoder = zlib.decompressobj()
            data = decoder.decompress(data, size + 1)
            if not decoder.eof:
                raise ValueError('TIFF decompression limit')
        if len(data) != size:
            raise ValueError('TIFF block length mismatch')
        return np.frombuffer(data, dtype=endian + 'f4')

    if 324 in tags:
        tw, th = tags[322], tags[323]
        if not 0 < tw <= 1024 or not 0 < th <= 1024:
            raise ValueError('TIFF tile budget exceeded')
        z = np.full((height, width), np.nan)
        across = math.ceil(width / tw)
        for i, (offset, count) in enumerate(zip(tags[324], tags[325])):
            tile = block(offset, count, tw * th * 4).reshape(th, tw)
            x, y = i % across * tw, i // across * th
            hh, ww = min(th, height - y), min(tw, width - x)
            z[y:y + hh, x:x + ww] = tile[:hh, :ww]
    else:
        strips = []
        rows = tags.get(278, height)
        for i, (offset, count) in enumerate(zip(tags[273], tags[279])):
            strips.append(block(offset, count, width * min(rows, height - i * rows) * 4))
        z = np.concatenate(strips).reshape(height, width).astype(float)
    nodata = tags.get(42113)
    if nodata is not None:
        z[z == float(str(nodata).strip('\x00'))] = np.nan
    z[(z < -500) | (z > 9000)] = np.nan
    if not np.isfinite(z).all():
        raise ValueError('DSM contains nodata: refusing invented terrain')
    scale, tie, geokey = tags[33550], tags[33922], tags[34735]
    keys = {geokey[i]: geokey[i + 3] for i in range(4, len(geokey), 4)}
    crs = keys.get(3072) or keys.get(2048)
    # GeoTIFF RasterPixelIsArea uses half-pixel centres; PixelIsPoint does not.
    half = .5 if keys.get(1025, 1) == 1 else 0.
    px, py = np.meshgrid(np.arange(width) + half, np.arange(height) + half)
    x, y = tie[3] + (px - tie[0]) * scale[0], tie[4] - (py - tie[1]) * scale[1]
    if crs == 3857:
        lon = np.rad2deg(x / 6378137)
        lat = np.rad2deg(2 * np.arctan(np.exp(y / 6378137)) - np.pi / 2)
    elif crs == 4326:
        lon, lat = x, y
    else:
        raise ValueError('Unsupported GeoTIFF CRS')
    east = (lon - tr['origin_lon']) * 111320 * math.cos(math.radians(tr['origin_lat']))
    north = (lat - tr['origin_lat']) * 110574
    return z, east[0], north[:, 0], {'crs': crs, 'shape': [height, width],
        'pixel_ground_m': [float(np.median(np.diff(east[0]))), float(-np.median(np.diff(north[:, 0])))],
        'geotiff_scale': list(scale), 'geotiff_tie': list(tie), 'byte_order': endian,
        'valid_fraction': float(np.isfinite(z).mean())}


def lonlat(tr, xy):
    xy = np.asarray(xy)
    return xy / [111320 * math.cos(math.radians(tr['origin_lat'])), 110574] + [tr['origin_lon'], tr['origin_lat']]


def enu(tr, xy):
    return (np.asarray(xy) - [tr['origin_lon'], tr['origin_lat']]) * [111320 * math.cos(math.radians(tr['origin_lat'])), 110574]


def bounds_from_metadata(metadata, tr):
    ring = np.array([[float(x), float(y)] for x, y in re.findall(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)', metadata['geometry'])])
    points = enu(tr, ring)
    return [*points.min(0), *points.max(0)]


def reuse_product(source, target):
    """Copy an existing local product without changing or pretending to redownload it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha(target) != sha(source):
        raise ValueError('Existing-cache product changed: choose a fresh --cache-root')
    if not target.exists():
        shutil.copyfile(source, target)
    write(target.with_suffix(target.suffix + '.local.json'),
          {'source': str(source), 'sha256': sha(source), 'mode': 'existing local product; no API call'})
    return target


def existing_products(root, sid, assets, bounds, spacing, tr):
    """Use only the ortho/DSM intersection; never stretch imagery into missing coverage."""
    assets = assets / 'existing'
    source = root / 'demo/cache' / sid
    ortho_path = reuse_product(source / 'ortho.jpg', assets / 'ortho.jpg')
    reuse_product(source / 'ortho.metadata.json', assets / 'ortho.metadata.json')
    raster_path = reuse_product(root / 'data/cache/daylight_dsm' / (sid + '.npz'), assets / 'dsm-source.npz')
    metadata = read(assets / 'ortho.metadata.json')
    ortho_bounds = bounds_from_metadata(metadata, tr)
    with np.load(raster_path) as raw:
        native_z, native_e, native_n = (raw[k].astype(float) for k in ['z', 'east', 'north'])
    if native_e.ndim == 2:
        native_e = native_e[0]
    if native_n.ndim == 2:
        native_n = native_n[:, 0]
    native_spacing = [float(np.median(np.diff(native_e))), float(-np.median(np.diff(native_n)))]
    clipped = [max(bounds[0], ortho_bounds[0], native_e.min()),
               max(bounds[1], ortho_bounds[1], native_n.min()),
               min(bounds[2], ortho_bounds[2], native_e.max()),
               min(bounds[3], ortho_bounds[3], native_n.max())]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError('No common cached DSM/ortho coverage')
    east = np.arange(clipped[0] + spacing / 2, clipped[2] - spacing / 2 + 1e-9, spacing)
    north = np.arange(clipped[3] - spacing / 2, clipped[1] + spacing / 2 - 1e-9, -spacing)
    # float16 coordinates were quantized in the older cache. Nearest source
    # coordinates avoid treating their slight nonuniform spacing as exact.
    ix = np.abs(native_e[:, None] - east).argmin(0)
    iy = np.abs(native_n[:, None] - north).argmin(0)
    z = native_z[np.ix_(iy, ix)]
    if not np.isfinite(z).all():
        raise ValueError('Cached DSM contains missing heights')
    raster = {'shape': list(z.shape), 'pixel_ground_m': [spacing, spacing],
              'source_pixel_ground_m': native_spacing, 'source_dtype': 'float16',
              'resampling': 'nearest cached samples; 0.25 m output does not add 0.5 m source detail',
              'valid_fraction': 1., 'source': str(raster_path.relative_to(root))}
    return z, east, north, raster, ortho_path, metadata, ortho_bounds, raster_path


def distance_ring(points, ring):
    points, ring = np.asarray(points), np.asarray(ring)
    distance = np.full(len(points), np.inf)
    for a, b in zip(ring[:-1], ring[1:]):
        d = b - a
        t = np.clip((points - a) @ d / max(d @ d, 1e-20), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points - a - t[:, None] * d, axis=1))
    return distance


def footprint_mask(points, rings, margin):
    mask = np.zeros(len(points), bool)
    for ring in rings:
        mask |= PolygonPath(ring).contains_points(points) | (distance_ring(points, ring) <= margin)
    return mask


def sample_dsm(xy, z, east, north):
    xy = np.atleast_2d(xy)
    ix = np.rint((xy[:, 0] - east[0]) / np.median(np.diff(east))).astype(int)
    iy = np.rint((xy[:, 1] - north[0]) / np.median(np.diff(north))).astype(int)
    valid = (ix >= 0) & (ix < len(east)) & (iy >= 0) & (iy < len(north))
    answer = np.full(len(xy), np.nan)
    answer[valid] = z[iy[valid], ix[valid]]
    return answer


def floor_at(position, triangles):
    """Measured walkmesh surface below each capture, never camera-z minus 1.6."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normal = np.cross(b - a, c - a)
    valid = normal[:, 2] > .6 * np.linalg.norm(normal, axis=1)
    for p, q in [(a, b), (b, c), (c, a)]:
        valid &= (q[:, 0] - p[:, 0]) * (position[1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (position[0] - p[:, 0]) >= -1e-6
    ids = np.flatnonzero(valid)
    if not len(ids):
        return None
    heights = a[ids, 2] - np.sum(normal[ids, :2] * (position[:2] - a[ids, :2]), axis=1) / normal[ids, 2]
    heights = heights[(heights < position[2] - .2) & (heights > position[2] - 2.5)]
    return float(heights.max()) if len(heights) else None


def read_obj(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith('v '):
            vertices.append(list(map(float, line.split()[1:4])))
        elif line.startswith('f '):
            ids = [int(v.split('/')[0]) - 1 for v in line.split()[1:]]
            faces.extend([[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)])
    return np.asarray(vertices)[np.asarray(faces)]


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0, 'rms_m': None, 'median_m': None, 'p95_abs_m': None, 'max_abs_m': None}
    return {'count': len(values), 'rms_m': float(np.sqrt(np.mean(values ** 2))),
            'median_m': float(np.median(values)), 'p95_abs_m': float(np.quantile(np.abs(values), .95)),
            'max_abs_m': float(np.max(np.abs(values)))}


def alignment(data, captures, web, z, east, north, ground, matrix, shift):
    rings = [np.asarray(layer['ring']) for layer in data['own_roof_layers']]
    # Exclude modelled rooms, including their holes, to find outdoor candidates.
    positions = np.asarray([c['position'] for c in captures])
    indoors = np.zeros(len(positions), bool)
    for room in data['geometry'].values():
        crossings = np.zeros(len(positions), bool)
        for a, b in room['segments']:
            a, b = np.asarray(a), np.asarray(b)
            if abs(b[1] - a[1]) < 1e-12:
                continue
            crossing = (a[1] > positions[:, 1]) != (b[1] > positions[:, 1])
            crossings ^= crossing & (positions[:, 0] < (b[0] - a[0]) * (positions[:, 1] - a[1]) / (b[1] - a[1]) + a[0])
        for hole in room['holes']:
            for a, b in hole:
                a, b = np.asarray(a), np.asarray(b)
                if abs(b[1] - a[1]) > 1e-12:
                    crossing = (a[1] > positions[:, 1]) != (b[1] > positions[:, 1])
                    crossings ^= crossing & (positions[:, 0] < (b[0] - a[0]) * (positions[:, 1] - a[1]) / (b[1] - a[1]) + a[0])
        indoors |= crossings
    triangles = read_obj(web / 'walkmesh.obj')
    rows = []
    for i in np.flatnonzero(~indoors):
        p = positions[i]
        floor = floor_at(p, triangles)
        height = float(sample_dsm(p[:2] @ matrix.T + shift, z, east, north)[0])
        model_height = height - ground + data['assumptions']['model_ground_z_m']
        under_roof = bool(footprint_mask(p[None, :2], rings, 1.)[0])
        rows.append({'capture_id': captures[i]['id'], 'capture_index': int(i), 'position_model_m': p.tolist(),
                     'floor_z_m': floor, 'dsm_absolute_z_m': height, 'dsm_model_z_m': model_height,
                     'residual_m': model_height - floor if floor is not None else None,
                     'inside_roof_plus_1m': under_roof,
                     'classification': 'outside all supplied room polygons; may be porch/overhang/vegetation'})
    # Strong local DSM steps, compared bidirectionally to the registered roof outline.
    ee, nn = np.meshgrid(east, north)
    model_xy = (np.column_stack([ee.ravel(), nn.ravel()]) - shift) @ np.linalg.inv(matrix).T
    distance = np.min([distance_ring(model_xy, ring) for ring in rings], axis=0).reshape(z.shape)
    edges = np.zeros(z.shape, bool)
    dx, dy = np.abs(np.diff(z, axis=1)) > 1., np.abs(np.diff(z, axis=0)) > 1.
    edges[:, :-1] |= dx
    edges[:, 1:] |= dx
    edges[:-1] |= dy
    edges[1:] |= dy
    roof = z > ground + 2.
    near = edges & roof & (distance <= 5.)
    edge_xy = model_xy[near.ravel()]
    samples = []
    for ring in rings:
        for a, b in zip(ring[:-1], ring[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / .25)))
            samples.extend(a + (b - a) * np.arange(n)[:, None] / n)
    samples = np.asarray(samples)
    residual = [float(np.min(np.linalg.norm(edge_xy - p, axis=1))) for p in samples] if len(edge_xy) else []
    horizontal = stats(residual)
    all_vertical = stats([r['residual_m'] for r in rows if r['residual_m'] is not None])
    vertical = stats([r['residual_m'] for r in rows if r['residual_m'] is not None and not r['inside_roof_plus_1m']])
    return {'ground': vertical, 'outdoor_capture_candidates': rows,
            'all_outdoor_surface_vs_floor': all_vertical,
            'ground_selection': 'Outside supplied indoor room polygons AND own roof +1m, selected geometrically before inspecting residuals. Canopy/building occlusion remains possible; DSM is a surface, not bare-earth DTM. All candidates, including covered ones, are retained above.',
            'roof_outline_to_dsm_edges': horizontal,
            'dsm_edges_to_roof_outline': stats(distance[near]),
            'roof_edge_method': 'DSM adjacent-sample steps >1 m, roof-side >ground+2 m, within 5 m of outline; nearest-edge distances, not surveyed correspondences',
            'vertical_reference': 'Existing property ground_elevation_m + existing model_ground_z_m; no fitted offsets applied',
            'ground_threshold_exceeded': vertical['max_abs_m'] is None or vertical['max_abs_m'] > .5,
            'roof_threshold_exceeded': horizontal['max_abs_m'] is None or horizontal['max_abs_m'] > .5,
            'threshold_m': .5}, edge_xy, ~indoors


def project(camera, points):
    q = np.column_stack([np.atleast_2d(points), np.ones(len(np.atleast_2d(points)))]) @ camera.T
    return q[:, :2] / q[:, 2:3]


def fit_camera(points, pixels):
    # ENU is small and pixels are normalized to keep the SVD well conditioned.
    scale = 10000.
    xy = pixels / scale
    xyz = np.column_stack([points, np.ones(len(points))])
    rows = []
    for p, (u, v) in zip(xyz, xy):
        rows.extend([np.r_[p, np.zeros(4), -u * p], np.r_[np.zeros(4), p, -v * p]])
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
    camera = vh[-1].reshape(3, 4)
    camera[:2] *= scale
    return camera


def cameras(root, sid, data, assets, api, wkt, online, existing_cache=False):
    """Reuse actual saved explicit-height API pixels; validate disjoint openings."""
    if existing_cache:
        assets = assets / 'existing'
    helper = module(root / 'demo/featured_assets.py', 'w18_features')
    tr = data['transform']
    matrix, shift = np.asarray(tr['registered_matrix']), np.asarray(tr['registered_shift'])
    xyz = np.concatenate([helper.corners(w) for w in data['windows']])
    points = xyz.copy()
    points[:, :2] = xyz[:, :2] @ matrix.T + shift
    cache = root / 'demo/cache' / sid
    views = []
    selected = {}
    for meta_path in sorted(cache.glob('oblique-*.metadata.json')):
        metadata = read(meta_path)
        direction = metadata.get('product-type', '').removeprefix('oblique-')
        if direction not in ('north', 'south', 'east', 'west') or direction in selected:
            continue
        if metadata.get('collection') != data['assumptions']['vexcel_collection']:
            continue
        tag = meta_path.name.removeprefix('oblique-').removesuffix('.metadata.json')
        for pixel_path in sorted(cache.glob('pixels-facade-v3-*-' + tag + '.json')):
            raw = read(pixel_path).get('points', [])
            if len(raw) < len(points):
                continue
            uv = np.asarray([[p['x'], p['y']] for p in raw[:len(points)]])
            train = (np.arange(len(points)) // 4) % 3 != 0
            camera = fit_camera(points[train], uv[train])
            errors = np.linalg.norm(project(camera, points) - uv, axis=1)
            if not np.isfinite(errors).all() or errors.max() > .15:
                continue
            selected[direction] = True
            if existing_cache:
                target = reuse_product(meta_path.with_suffix('').with_suffix('.jpg'), assets / ('oblique-' + direction + '.jpg'))
                reuse_product(meta_path, target.with_suffix('.metadata.json'))
            else:
                target = acquire(api, assets, 'oblique-' + direction + '.jpg', 'oriented/extract',
                    {'wkt': wkt, 'srid': 4326, 'layer': 'urban', 'image-name': metadata['image-name'],
                     'rotation': 'natural', 'crop': 'bbox', 'image-format': 'jpeg', 'downsample': 0}, online)
            actual = read(target.with_suffix('.metadata.json'))
            if actual.get('image-name') != metadata['image-name']:
                raise ValueError('Oblique image identity mismatch')
            bbox = list(map(float, actual['pixel-bbox'].split(',')))
            image = Image.open(target).convert('RGB')
            if list(image.size) != list(map(int, bbox[2:])):
                raise ValueError('Oblique pixel-bbox mismatch')
            zone = int(str(actual['utm-zone']).rstrip('NS'))
            cam_enu = np.array([float(actual['camera-pos-x']), float(actual['camera-pos-y'])]) - helper.utm(tr['origin_lon'], tr['origin_lat'], zone)
            views.append({'direction': direction, 'camera': camera, 'image': image,
                          'bbox': bbox, 'position_enu': cam_enu,
                          'camera_absolute_z': float(actual['camera-pos-z']),
                          'metadata': actual, 'image_path': target, 'calibration': {'source': str(pixel_path.relative_to(root)),
                              'source_sha256': sha(pixel_path), 'image_sha256': sha(target),
                              'fit_points': int(train.sum()), 'held_out_points': int((~train).sum()),
                              'held_out_max_px': float(errors[~train].max()), 'all_max_px': float(errors.max()),
                              'method': '3x4 local projective fit to cached Vexcel explicit-Z pixels; validated on disjoint openings; extrapolation over neighbourhood is approximate'}})
            break
    return views


def mesh(z, east, north, cut, step_threshold=.75, tolerance=.10):
    """Adaptive planar patches with shared split edges and explicit vertical seams."""
    height, width = z.shape
    labels = np.full(z.shape, -1, dtype=np.int32)
    leaves = []
    # Cell centres -> boundary grid. Terrain is sampled exactly at native centres.
    dx, dy = np.median(np.diff(east)), -np.median(np.diff(north))
    bx = np.r_[east - dx / 2, east[-1] + dx / 2]
    by = np.r_[north + dy / 2, north[-1] - dy / 2]

    def split(x, y, w, h):
        valid = ~cut[y:y + h, x:x + w]
        if not valid.any():
            return
        block = z[y:y + h, x:x + w]
        yy, xx = np.mgrid[y:y + h, x:x + w]
        plane = None
        if valid.all() and w > 1 and h > 1 and max(w, h) <= 16:
            q = np.column_stack([xx.ravel() - x, yy.ravel() - y, np.ones(w * h)])
            p = np.linalg.lstsq(q, block.ravel(), rcond=None)[0]
            error = np.max(np.abs(q @ p - block.ravel()))
            # No simplification across discontinuities, even if fitted slope is steep.
            jump = max(np.max(np.abs(np.diff(block, axis=0))), np.max(np.abs(np.diff(block, axis=1))))
            if error <= tolerance and jump < step_threshold:
                plane = p
        if w == h == 1:
            plane = np.array([0., 0., block[0, 0]])
        if plane is not None:
            label = len(leaves)
            leaves.append((x, y, w, h, plane))
            labels[y:y + h, x:x + w] = label
            return
        if w >= h and w > 1:
            mid = w // 2
            split(x, y, mid, h)
            split(x + mid, y, w - mid, h)
        else:
            mid = h // 2
            split(x, y, w, mid)
            split(x, y + mid, w, h - mid)

    split(0, 0, width, height)
    horizontal, vertical = collections.defaultdict(set), collections.defaultdict(set)
    for x, y, w, h, _ in leaves:
        for yy in (y, y + h):
            horizontal[yy].update((x, x + w))
        for xx in (x, x + w):
            vertical[xx].update((y, y + h))

    def edge_height(ix, iy, label):
        x, y, w, h, plane = leaves[label]
        predicted = plane @ [ix - x - .5, iy - y - .5, 1.]
        # Join gradual terrain with one shared value; keep separate levels at steps.
        a, b = max(0, iy - 1), min(height, iy + 1)
        c, d = max(0, ix - 1), min(width, ix + 1)
        values = z[a:b, c:d].ravel()
        values = values[np.abs(values - predicted) < step_threshold]
        return float(values.mean()) if len(values) else float(predicted)

    vertices, uv_en, faces = [], [], []
    vertex_ids = {}

    def vertex(ix, iy, label):
        zz = edge_height(ix, iy, label)
        key = (ix, iy, round(zz, 5))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append([bx[ix], by[iy], zz])
            uv_en.append([bx[ix], by[iy]])
        return vertex_ids[key]

    for label, (x, y, w, h, _) in enumerate(leaves):
        top = [(i, y) for i in sorted(horizontal[y]) if x <= i <= x + w]
        right = [(x + w, i) for i in sorted(vertical[x + w]) if y < i <= y + h]
        bottom = [(i, y + h) for i in sorted(horizontal[y + h], reverse=True) if x <= i < x + w]
        left = [(x, i) for i in sorted(vertical[x], reverse=True) if y < i < y + h]
        ring = [vertex(i, j, label) for i, j in top + right + bottom + left]
        if len(ring) == 4:
            faces.extend([[ring[0], ring[2], ring[1]], [ring[0], ring[3], ring[2]]])
        else:
            center = len(vertices)
            vertices.append(np.mean([vertices[i] for i in ring], axis=0).tolist())
            uv_en.append(np.mean([uv_en[i] for i in ring], axis=0).tolist())
            faces.extend([[center, ring[(i + 1) % len(ring)], ring[i]] for i in range(len(ring))])
    top_count = len(faces)
    walls = []
    # Join different-height leaf sides at the same XY, never bridge cut/nodata.
    for axis in (0, 1):
        aa, bb = (labels[:, :-1], labels[:, 1:]) if axis == 0 else (labels[:-1], labels[1:])
        yy, xx = np.where((aa != bb) & (aa >= 0) & (bb >= 0))
        for y, x in zip(yy, xx):
            a, b = int(aa[y, x]), int(bb[y, x])
            p, q = ((x + 1, y), (x + 1, y + 1)) if axis == 0 else ((x, y + 1), (x + 1, y + 1))
            ids = [vertex(*p, a), vertex(*q, a), vertex(*q, b), vertex(*p, b)]
            if ids[0] == ids[3] and ids[1] == ids[2]:
                continue
            pts = np.asarray([vertices[i] for i in ids])
            if max(abs(pts[0, 2] - pts[3, 2]), abs(pts[1, 2] - pts[2, 2])) < .02:
                continue
            # Winding points from the high side toward the low side.
            high_a = (pts[0, 2] + pts[1, 2]) > (pts[2, 2] + pts[3, 2])
            normal = np.array([1., 0.]) if axis == 0 else np.array([0., -1.])
            if not high_a:
                normal *= -1
            quad = pts.copy()
            if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0])[:2], normal) < 0:
                quad = quad[::-1]
            walls.append((quad, normal))
    return np.asarray(vertices), np.asarray(faces, dtype=np.uint32), walls, {
        'native_cells': int((~cut).sum()), 'planar_patches': len(leaves), 'top_triangles': top_count,
        'planar_fit_tolerance_m': tolerance, 'max_patch_cells': 16, 'step_threshold_m': step_threshold,
        'method': 'binary adaptive planar patches, shared boundary splits, discontinuity-preserving corner samples; vertical seams only between present cells'}


class GLB:
    def __init__(self):
        self.binary = bytearray()
        self.doc = {'asset': {'version': '2.0', 'generator': 'W18 real Vexcel heightfield'},
                    'buffers': [{}], 'bufferViews': [], 'accessors': [], 'images': [],
                    'textures': [], 'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071}],
                    'materials': [], 'meshes': [], 'nodes': [], 'scenes': [{'nodes': []}], 'scene': 0}

    def blob(self, raw, target=None):
        self.binary.extend(b'\0' * (-len(self.binary) % 4))
        offset = len(self.binary)
        self.binary.extend(raw)
        view = {'buffer': 0, 'byteOffset': offset, 'byteLength': len(raw)}
        if target:
            view['target'] = target
        self.doc['bufferViews'].append(view)
        return len(self.doc['bufferViews']) - 1

    def accessor(self, array, kind):
        array = np.ascontiguousarray(array)
        view = self.blob(array.tobytes(), 34963 if kind == 'SCALAR' else 34962)
        result = {'bufferView': view, 'componentType': 5125 if array.dtype == np.dtype('<u4') else 5126,
                  'count': len(array), 'type': kind}
        if kind == 'VEC3':
            result.update(min=array.min(0).astype(float).tolist(), max=array.max(0).astype(float).tolist())
        self.doc['accessors'].append(result)
        return len(self.doc['accessors']) - 1

    def material(self, image, name):
        stream = io.BytesIO()
        image = image.copy()
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        image.save(stream, format='JPEG', quality=87)
        self.doc['images'].append({'bufferView': self.blob(stream.getvalue()), 'mimeType': 'image/jpeg'})
        self.doc['textures'].append({'source': len(self.doc['images']) - 1, 'sampler': 0})
        self.doc['materials'].append({'name': name, 'doubleSided': True,
            'pbrMetallicRoughness': {'baseColorTexture': {'index': len(self.doc['textures']) - 1},
                                    'metallicFactor': 0., 'roughnessFactor': 1.}})
        return len(self.doc['materials']) - 1

    def geometry(self, positions, faces, uv, material, name):
        if not len(faces):
            return
        # Standards-compliant Y-up GLB. addNeighbourhood rotates +90 deg X to
        # recover the authoritative captures.json Z-up frame exactly once.
        p = np.asarray(positions)[:, [0, 2, 1]].copy()
        p[:, 2] *= -1
        attributes = {'POSITION': self.accessor(p.astype('<f4'), 'VEC3'),
                      'TEXCOORD_0': self.accessor(np.asarray(uv, dtype='<f4'), 'VEC2')}
        primitive = {'attributes': attributes, 'indices': self.accessor(np.asarray(faces, dtype='<u4').ravel(), 'SCALAR'),
                     'material': material, 'mode': 4}
        self.doc['meshes'].append({'name': name, 'primitives': [primitive]})
        self.doc['nodes'].append({'mesh': len(self.doc['meshes']) - 1, 'name': name})
        self.doc['scenes'][0]['nodes'].append(len(self.doc['nodes']) - 1)

    def save(self, path):
        self.doc['buffers'][0]['byteLength'] = len(self.binary)
        raw = json.dumps(self.doc, separators=(',', ':'), allow_nan=False).encode()
        raw += b' ' * (-len(raw) % 4)
        self.binary.extend(b'\0' * (-len(self.binary) % 4))
        glb = struct.pack('<III', 0x46546c67, 2, 28 + len(raw) + len(self.binary))
        glb += struct.pack('<II', len(raw), 0x4e4f534a) + raw
        glb += struct.pack('<II', len(self.binary), 0x004e4942) + self.binary
        if len(glb) > LIMIT:
            raise ValueError('Neighbourhood GLB exceeds 15 MB; no oversize asset written')
        path.write_bytes(glb)


def ortho_uv(xy, bounds):
    w, s, e, n = bounds
    return (np.asarray(xy) - [w, n]) / [e - w, s - n]


def wall_visibility(quad, view, z, east, north, ground, model_ground):
    """Conservative DSM sightline guard; own home remains an occluder."""
    middle = quad.mean(0)
    delta = view['position_enu'] - middle[:2]
    length = np.linalg.norm(delta)
    if length < 1:
        return False
    fractions = np.arange(.8, min(90., length), .8) / length
    surfaces = sample_dsm(middle[:2] + fractions[:, None] * delta, z, east, north)
    # Quad and camera elevations are both absolute, in the source DSM datum.
    line_z = middle[2] + fractions * (view['camera_absolute_z'] - middle[2])
    return not np.any(np.isfinite(surfaces) & (surfaces > line_z + .5))


def export_mesh(path, vertices, faces, walls, ortho, ortho_bounds, views, matrix, shift,
                ground, model_ground, z, east, north):
    glb = GLB()
    ortho_mat = glb.material(ortho, 'Vexcel orthophoto')
    model = vertices.copy()
    model[:, :2] = (vertices[:, :2] - shift) @ np.linalg.inv(matrix).T
    model[:, 2] = vertices[:, 2] - ground + model_ground
    glb.geometry(model, faces, ortho_uv(vertices[:, :2], ortho_bounds), ortho_mat, 'DSM terrain and roofs')
    buckets = collections.defaultdict(list)
    images = {i: glb.material(v['image'], 'Vexcel oblique ' + v['direction']) for i, v in enumerate(views)}
    counts = collections.Counter()
    ortho_rgb = np.asarray(ortho)
    colours, fallback = [], []
    for quad, normal in walls:
        candidates = []
        world = quad.copy()
        world[:, 2] = quad[:, 2] - ground + model_ground
        for i, view in enumerate(views):
            delta = view['position_enu'] - quad.mean(0)[:2]
            facing = float(normal @ delta / np.linalg.norm(delta))
            if facing < .25:
                continue
            pixels = project(view['camera'], world) - view['bbox'][:2]
            uv = pixels / view['bbox'][2:]
            if not np.isfinite(uv).all() or np.min(uv) < 0 or np.max(uv) > 1:
                continue
            # Cached crops include a provider copyright/date footer. Do not
            # project that annotation onto a building or vegetation wall.
            if pixels[:, 0].min() < 2 or pixels[:, 0].max() > view['image'].width - 2 or pixels[:, 1].min() < 2 or pixels[:, 1].max() > view['image'].height - 32:
                continue
            candidates.append((facing, i, uv))
        chosen = None
        for _, i, uv in sorted(candidates, key=lambda item: item[0], reverse=True):
            if wall_visibility(quad, views[i], z, east, north, ground, model_ground):
                chosen = (i, uv)
                break
        transformed = world.copy()
        transformed[:, :2] = (world[:, :2] - shift) @ np.linalg.inv(matrix).T
        if chosen:
            i, uv = chosen
            buckets[i].append((transformed, uv))
            counts[views[i]['direction']] += 2
        else:
            high = quad[np.argmax(quad[:, 2]), :2]
            uv = np.clip(ortho_uv(high, ortho_bounds), 0, 1)
            pixel = np.minimum((uv * [ortho.width, ortho.height]).astype(int), [ortho.width - 1, ortho.height - 1])
            colours.append(ortho_rgb[pixel[1], pixel[0]])
            fallback.append(transformed)
            counts['ortho_edge_colour'] += 2
    for i, records in buckets.items():
        p = np.concatenate([p for p, _ in records])
        uv = np.concatenate([uv for _, uv in records])
        f = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + j * 4 for j in range(len(records))])
        glb.geometry(p, f, uv, images[i], 'Projected facades ' + views[i]['direction'])
    if fallback:
        # One constant sampled colour per quad, packed into a lossless tiny atlas.
        # PNG avoids JPEG bleeding between unrelated colours.
        atlas = np.zeros((math.ceil(len(colours) / 256), 256, 3), np.uint8)
        atlas.reshape(-1, 3)[:len(colours)] = colours
        stream = io.BytesIO()
        Image.fromarray(atlas).save(stream, format='PNG')
        glb.doc['images'].append({'bufferView': glb.blob(stream.getvalue()), 'mimeType': 'image/png'})
        glb.doc['samplers'].append({'magFilter': 9728, 'minFilter': 9728, 'wrapS': 33071, 'wrapT': 33071})
        glb.doc['textures'].append({'source': len(glb.doc['images']) - 1, 'sampler': 1})
        glb.doc['materials'].append({'name': 'Flat ortho roof-edge samples', 'doubleSided': True,
            'pbrMetallicRoughness': {'baseColorTexture': {'index': len(glb.doc['textures']) - 1},
                                   'metallicFactor': 0., 'roughnessFactor': 1.}})
        uv = np.repeat(np.array([[(i % 256 + .5) / 256, (i // 256 + .5) / atlas.shape[0]] for i in range(len(fallback))]), 4, axis=0)
        f = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + j * 4 for j in range(len(fallback))])
        glb.geometry(np.concatenate(fallback), f, uv, len(glb.doc['materials']) - 1, 'Sampled facades')
    glb.doc['asset']['extras'] = {'model_frame': 'captures.json, metres Z up',
        'stored_frame': 'glTF metres Y up: (x,z,-y); apply +90 degrees X for walk viewer',
        'license_scope': 'Local demo only; do not commit Vexcel data'}
    glb.save(path)
    return dict(counts), len(faces) + sum(counts.values())


def load_glb(path):
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', raw)
    if magic != 0x46546c67 or version != 2 or length != len(raw):
        raise ValueError('Invalid GLB')
    size, kind = struct.unpack_from('<II', raw, 12)
    if kind != 0x4e4f534a:
        raise ValueError('Missing GLB JSON')
    doc = json.loads(raw[20:20 + size])
    binary = raw[28 + size:]

    def accessor(index):
        a = doc['accessors'][index]
        view = doc['bufferViews'][a['bufferView']]
        components = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[a['type']]
        dtype = '<u4' if a['componentType'] == 5125 else '<f4'
        return np.frombuffer(binary, dtype=dtype, count=a['count'] * components,
                             offset=view.get('byteOffset', 0) + a.get('byteOffset', 0)).reshape(-1, components)

    textures = []
    for image in doc['images']:
        view = doc['bufferViews'][image['bufferView']]
        offset = view.get('byteOffset', 0)
        textures.append(np.asarray(Image.open(io.BytesIO(binary[offset:offset + view['byteLength']])).convert('RGB')))
    parts = []
    for mesh in doc['meshes']:
        for primitive in mesh['primitives']:
            p = accessor(primitive['attributes']['POSITION']).astype(float)
            p = p[:, [0, 2, 1]]
            p[:, 1] *= -1
            uv = accessor(primitive['attributes']['TEXCOORD_0'])
            f = accessor(primitive['indices']).reshape(-1, 3)
            material = doc['materials'][primitive['material']]
            texture = doc['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]['source']
            parts.append((p, f, uv, textures[texture]))
    return parts


def render_glb(path, output, sid):
    """Offline orthographic textured triangle/z-buffer render of actual GLB bytes."""
    parts = load_glb(path)
    positions = np.concatenate([p for p, _, _, _ in parts])
    center = (positions.min(0) + positions.max(0)) / 2
    eye = np.array([1., -1.3, 1.25])
    eye /= np.linalg.norm(eye)
    right = np.cross([0., 0., 1.], eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    rotation = np.column_stack([right, -up, eye])
    projected = (positions - center) @ rotation
    size = 1300
    scale = (size - 100) / max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    offset = np.array([size / 2, size / 2]) - (projected[:, :2].min(0) + projected[:, :2].max(0)) / 2 * scale
    rgb = np.full((size, size, 3), 246, np.uint8)
    depth = np.full((size, size), -np.inf)
    for p, faces, uv, texture in parts:
        q = (p - center) @ rotation
        screen = q[:, :2] * scale + offset
        for ids in faces:
            a, b, c = screen[ids]
            lo = np.maximum(np.floor(np.min([a, b, c], axis=0)).astype(int), 0)
            hi = np.minimum(np.ceil(np.max([a, b, c], axis=0)).astype(int), size - 1)
            if np.any(lo > hi):
                continue
            denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denom) < 1e-10:
                continue
            yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            u = ((b[1] - c[1]) * (xx + .5 - c[0]) + (c[0] - b[0]) * (yy + .5 - c[1])) / denom
            v = ((c[1] - a[1]) * (xx + .5 - c[0]) + (a[0] - c[0]) * (yy + .5 - c[1])) / denom
            w = 1 - u - v
            zz = u * q[ids[0], 2] + v * q[ids[1], 2] + w * q[ids[2], 2]
            region = depth[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            visible = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6) & (zz > region)
            if not visible.any():
                continue
            tex = u[visible, None] * uv[ids[0]] + v[visible, None] * uv[ids[1]] + w[visible, None] * uv[ids[2]]
            tex = np.clip(tex, 0, 1)
            ix = np.minimum((tex[:, 0] * texture.shape[1]).astype(int), texture.shape[1] - 1)
            iy = np.minimum((tex[:, 1] * texture.shape[0]).astype(int), texture.shape[0] - 1)
            rgb[yy[visible], xx[visible]] = texture[iy, ix]
            region[visible] = zz[visible]
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size, 54), fill='white')
    draw.text((20, 12), sid + ' | Vexcel DSM + real aerial imagery | 3/4 view of exported GLB', fill='black')
    draw.text((20, 32), 'Own home removed (1 m margin). Approximate 2.5D; no Matterport home rendered here.', fill='black')
    image.save(output)


def overlay(output, sid, ortho, bounds, matrix, shift, rings, cut, east, north, captures, outdoor, edges, residuals):
    image = ortho.copy()
    image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
    image = image.convert('RGBA')
    draw = ImageDraw.Draw(image, 'RGBA')

    def px(en):
        return tuple((ortho_uv(en, bounds) * image.size).tolist())

    # Actual conservative grid cut boundary, with the registered source outline.
    boundary = cut.copy()
    boundary[1:-1, 1:-1] &= ~(cut[:-2, 1:-1] & cut[2:, 1:-1] & cut[1:-1, :-2] & cut[1:-1, 2:])
    yy, xx = np.where(boundary)
    for x, y in zip(east[xx], north[yy]):
        q = px([x, y])
        draw.ellipse((q[0] - 1, q[1] - 1, q[0] + 1, q[1] + 1), fill=(235, 35, 55, 255))
    for ring in rings:
        draw.line([px(p @ matrix.T + shift) for p in ring], fill=(255, 210, 20, 255), width=3)
    for point in edges:
        q = px(point @ matrix.T + shift)
        draw.point(q, fill=(30, 235, 230, 255))
    for i, capture in enumerate(captures):
        p = np.asarray(capture['position'])
        q = px(p[:2] @ matrix.T + shift)
        r = 4 if outdoor[i] else 2
        colour = (255, 50, 220, 255) if outdoor[i] else (20, 85, 255, 220)
        draw.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), fill=colour, outline='white')
        if outdoor[i]:
            draw.text((q[0] + 5, q[1] - 7), str(i), fill='white', stroke_width=1, stroke_fill='black')
    banner = Image.new('RGB', (image.width, image.height + 100), 'white')
    banner.paste(image.convert('RGB'), (0, 100))
    draw = ImageDraw.Draw(banner)
    draw.text((12, 8), sid + ' | north up | actual DSM cut + capture positions', fill='black')
    draw.text((12, 28), 'Red: footprint + >=1 m cut; yellow: registered own_roof_layers; cyan: DSM roof steps', fill='black')
    draw.text((12, 48), 'Blue: indoor captures; magenta: outdoor candidates (indices in captures.json)', fill='black')
    draw.text((12, 68), 'Ground RMS ' + str(residuals['ground']['rms_m'])[:6] + ' m; roof edge RMS ' + str(residuals['roof_outline_to_dsm_edges']['rms_m'])[:6] + ' m. Not a survey.', fill='black')
    banner.save(output)


def build(args, sid):
    root = args.root
    web = root / 'data/splat/web' / sid
    assets = args.cache_root / sid
    output = root / 'experiments/w18_nbhd'
    output.mkdir(parents=True, exist_ok=True)
    data = read(root / 'demo/cache' / sid / 'derived.json')
    captures_meta = read(web / 'captures.json')
    if captures_meta['up'] != 'Z' or captures_meta['units'] != 'metres':
        raise ValueError('captures.json must use metres Z up')
    captures = captures_meta['captures']
    tr = data['transform']
    matrix, shift = np.asarray(tr['registered_matrix']), np.asarray(tr['registered_shift'])
    # sun_engine.obstructions() already converts these outlines with model_ring().
    # Keep that authoritative model frame; only the acquisition bbox needs ENU.
    model_rings = [np.asarray(layer['ring']) for layer in data['own_roof_layers']]
    rings = [ring @ matrix.T + shift for ring in model_rings]
    center = (np.concatenate(rings).min(0) + np.concatenate(rings).max(0)) / 2
    bounds = [*(center - args.radius), *(center + args.radius)]
    w, s, e, n = bounds
    corners = lonlat(tr, [[w, s], [e, s], [e, n], [w, n], [w, s]])
    wkt = 'POLYGON((' + ','.join(f'{x:.11f} {y:.11f}' for x, y in corners) + '))'
    api = module(root / 'demo/api_cache.py', 'w18_api')
    common = {'wkt': wkt, 'srid': 4326, 'layer': 'urban', 'collection': data['assumptions']['vexcel_collection'],
              'crop': 'bounding_box', 'buffer-meter': 0}
    dimension = int(math.ceil(2 * args.radius / args.spacing))
    if args.existing_cache:
        z, east, north, raster, ortho_path, ortho_metadata, ortho_bounds, dsm = existing_products(
            root, sid, assets, bounds, args.spacing, tr)
    else:
        dsm = acquire(api, assets, 'dsm.tif', 'dsm/extract',
                      dict(common, **{'destination-dimensions': f'{dimension},{dimension},fit'}), args.online)
        ortho_path = acquire(api, assets, 'ortho.jpg', 'ortho/extract', dict(common, **{'image-format': 'jpeg'}), args.online)
        z, east, north, raster = decode_dsm(dsm, tr)
        ortho_metadata = read(ortho_path.with_suffix('.metadata.json'))
        ortho_bounds = bounds_from_metadata(ortho_metadata, tr)
    ortho = Image.open(ortho_path).convert('RGB')
    dsm_meta = read(root / 'data/cache/daylight_dsm' / (sid + '.json'))
    ground = dsm_meta['ground_elevation_m']
    residuals, edges, outdoor = alignment(data, captures, web, z, east, north, ground, matrix, shift)
    ee, nn = np.meshgrid(east, north)
    model_xy = (np.column_stack([ee.ravel(), nn.ravel()]) - shift) @ np.linalg.inv(matrix).T
    # Expand by half a cell diagonal so *every* emitted triangle clears 1 m.
    safety = .5 * np.hypot(*raster['pixel_ground_m']) / np.linalg.svd(matrix)[1].min()
    cut = footprint_mask(model_xy, model_rings, 1. + safety).reshape(z.shape)
    views = cameras(root, sid, data, assets, api, wkt, args.online, args.existing_cache)
    vertices, faces, walls, simplification = mesh(z, east, north, cut)
    destination = web / 'neighbourhood.glb'
    counts, triangles = export_mesh(destination, vertices, faces, walls, ortho, ortho_bounds, views, matrix, shift,
                                   ground, data['assumptions']['model_ground_z_m'], z, east, north)
    dates = ortho_metadata.get('first-capture-date', '?') + ' to ' + ortho_metadata.get('last-capture-date', '?')
    facade = 'projected Vexcel obliques with ortho edge-colour fallback' if any(v for k, v in counts.items() if k != 'ortho_edge_colour') else 'ortho edge-sampled flat colours'
    label = f'Neighbourhood: Vexcel aerial DSM + orthophoto ({dates}), approximate 2.5D; facades {facade}; not captured by Matterport'
    sources = [root / 'demo/cache' / sid / 'derived.json', web / 'captures.json', web / 'walkmesh.obj',
               root / 'data/cache/daylight_dsm' / (sid + '.json'), dsm, ortho_path, ortho_path.with_suffix('.metadata.json')]
    actual_bounds = [float(east[0] - raster['pixel_ground_m'][0] / 2),
                     float(north[-1] - raster['pixel_ground_m'][1] / 2),
                     float(east[-1] + raster['pixel_ground_m'][0] / 2),
                     float(north[0] + raster['pixel_ground_m'][1] / 2)]
    coverage = (actual_bounds[2] - actual_bounds[0]) * (actual_bounds[3] - actual_bounds[1]) / (4 * args.radius ** 2)
    actual_ll = lonlat(tr, np.array(actual_bounds).reshape(2, 2))
    info = {'schema': 'w18-neighbourhood-v1', 'sid': sid, 'label': label,
        'frame': {'units': 'metres', 'up': 'Z', 'authority': 'captures.json',
                  'glb_storage': 'Y up (x,z,-y); loader applies +90 degrees about X',
                  'registered_matrix': matrix.tolist(), 'registered_shift': shift.tolist(),
                  'absolute_ground_elevation_m': ground, 'model_ground_z_m': data['assumptions']['model_ground_z_m']},
        'provenance': {'provider': 'Vexcel', 'collection': common['collection'],
            'products': ['dsm/extract', 'ortho/extract'] + (['oriented/extract', 'cached oriented/transform-points'] if views else []),
            'capture_date': {'ortho': [ortho_metadata.get('first-capture-date'), ortho_metadata.get('last-capture-date')],
                             'dsm_collection': dsm_meta['capture_date']},
            'bbox_lonlat': actual_ll.ravel().tolist(), 'bbox_enu_m': actual_bounds,
            'requested_bbox_lonlat': [*corners.min(0), *corners.max(0)], 'requested_bbox_enu_m': bounds,
            'acquisition_mode': 'existing-cache' if args.existing_cache else 'pinned common-bbox API products',
            'requested_bbox_coverage_fraction': coverage, 'requested_bbox_complete': coverage >= .99,
            'radius_m': args.radius, 'requested_spacing_m': args.spacing, 'dsm_raster': raster,
            'ortho_bounds_enu_m': ortho_bounds, 'ortho_size_px': list(ortho.size),
            'ortho_gsd_m': [ortho_metadata.get('min-gsd'), ortho_metadata.get('max-gsd')],
            'dsm_gsd_note': 'Raster output spacing is not a claim of independent native elevation accuracy; collection GSD describes source photography',
            'obliques': [{'direction': v['direction'], 'metadata': v['metadata'], 'calibration': v['calibration']} for v in views],
            'input_sha256': {str(p.relative_to(root)): sha(p) for p in sources},
            'license_scope': 'Local demo only; never commit aerial data or derived textures'},
        'alignment_residuals': residuals, 'facade_source': facade, 'facade_triangles': counts,
        'triangle_count': triangles, 'bytes': destination.stat().st_size, 'sha256': sha(destination),
        'cutout': {'source': 'union of own_roof_layers rings, already in model XY from sun_engine.model_ring',
                   'margin_m': 1., 'conservative_raster_extra_m': safety, 'cells_removed': int(cut.sum()),
                   'rings_model_xy_m': [r.tolist() for r in model_rings]},
        'simplification': simplification,
        'limitations': ['Approximate 2.5D, no interiors or generative content.',
            'Only the real common DSM/ortho coverage is emitted; compare requested and actual bounds. Existing-cache mode may cover less than the requested radius.',
            'DSM/property shared vertical datum assumed; no independent survey or hidden registration adjustment.',
            'Outdoor classification uses supplied room polygons; porches, roofs and vegetation can contaminate DSM comparisons.',
            'Roof edge distances are nearest strong DSM steps, including possible vegetation; they are not semantic/survey matches.',
            'Oblique calibration validated near the home; extrapolation and unseen occluders beyond the DSM bbox remain unverified.',
            'Viewer integration, on-device FPS, and Matterport/splat visual seam are W17 work.'],
        'evidence': {'aerial': f'experiments/w18_nbhd/{sid}_aerial.png',
                     'overlay': f'experiments/w18_nbhd/{sid}_overlay.png'}}
    write(web / 'neighbourhood.json', info)
    # The renderer rereads the deliverable: release the construction arrays and
    # camera bitmaps first, particularly on disk/memory-constrained demo Macs.
    del vertices, faces, walls, views, ee, nn, model_xy
    gc.collect()
    render_glb(destination, output / (sid + '_aerial.png'), sid)
    overlay(output / (sid + '_overlay.png'), sid, ortho, ortho_bounds, matrix, shift, model_rings,
            cut, east, north, captures, outdoor, edges, residuals)
    print(json.dumps({'sid': sid, 'bytes': info['bytes'], 'triangles': triangles,
                      'facade_triangles': counts, 'ground': residuals['ground'],
                      'roof_edges': residuals['roof_outline_to_dsm_edges']}), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/Users/phanan/HackPrinceton'))
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--sid', choices=SIDS, action='append')
    parser.add_argument('--radius', type=float, default=60.)
    parser.add_argument('--spacing', type=float, default=.25)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--existing-cache', action='store_true', help='Explicit partial-coverage fallback to existing local DSM/imagery; never invent missing data')
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.cache_root = args.cache_root or args.root / 'data/vexcel_nbhd'
    if not 10 <= args.radius <= 100 or not .15 <= args.spacing <= 1:
        parser.error('radius must be 10..100 m and spacing .15..1 m')
    for sid in args.sid or SIDS:
        build(args, sid)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not emit exception strings/tracebacks containing remote request URLs.
        print('W18 build failed: ' + type(error).__name__ + '; inspect sanitized local receipts.', file=sys.stderr)
        sys.exit(1)
