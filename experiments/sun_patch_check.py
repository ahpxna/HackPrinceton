"""Brief V. Acquisition is explicit; replay defaults to no network.

python3 -I experiments/sun_patch_check.py schema --acquire
python3 -I experiments/sun_patch_check.py select
python3 -I experiments/sun_patch_check.py acquire --acquire --batch 1
python3 -I experiments/sun_patch_check.py render --batch 1
python3 -I experiments/sun_patch_check.py measure
"""
import argparse
import ast
import base64
import concurrent.futures
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/sun_patch'
CACHE = ROOT / 'data/cache/sun_patch_20261114'
PROGRESS = Path('/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T14-32-05-670Z-56216-29204a2d-ccaf-4af4-9daf-dd75548e0812.md')
spec = importlib.util.spec_from_file_location('brief_v_projection', ROOT / 'experiments/sun_from_pano.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
NETWORK = False
AUTH_FAILURES = 0
LOCK = threading.Lock()
CALLS = 0
SITE = 'https://beta-app.matterport.com/2026-princeton-hackathon'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def progress(done, next_step):
    PROGRESS.write_text('# Objective\nBrief V renewed: two-home sunlight-patch measurement. Start 14:32 UTC; deadline 16:02 UTC, 2026-11-14. Prior interruption was a usage limit.\n\n# Done\n' + done + '\n\n# Next\n' + next_step + '\n')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if not any(t in k.lower() for t in ('token', 'secret', 'authorization', 'password', 'url'))}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'https?://[^\s<>\"\']+', '[URL omitted]', value)
    return value


def request(url, data=None, headers=None):
    global AUTH_FAILURES, CALLS
    if not NETWORK:
        raise RuntimeError('network_disabled')
    for attempt in range(5):
        with LOCK:
            if AUTH_FAILURES >= 2:
                raise RuntimeError('repeated_authentication_failure_stop')
            CALLS += 1
        req = urllib.request.Request(url, data=data, headers={'User-Agent': 'Mozilla/5.0', **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                with LOCK:
                    AUTH_FAILURES += 1
            if error.code == 429 and attempt < 4:
                time.sleep(min(32, 2 ** (attempt + 1)))
                continue
            raise RuntimeError('HTTP_' + str(error.code) + '_body_and_URL_suppressed') from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('transport_failure_URL_suppressed') from None
    raise RuntimeError('retry_exhausted')


def credentials():
    # Reuse the existing runtime parser. Its result is never printed or cached.
    return {**A.credentials(), **os.environ}


def gql(query):
    env = credentials()
    auth = base64.b64encode((env['MATTERPORT_API_KEY'] + ':' + env['MATTERPORT_API_TOKEN']).encode()).decode()
    raw = request('https://api.matterport.com/api/models/graph',
                  json.dumps({'query': query}).encode(),
                  {'Content-Type': 'application/json', 'Authorization': 'Basic ' + auth})
    result = json.loads(raw)
    if result.get('errors'):
        raise RuntimeError('GraphQL_error_body_suppressed')
    return result['data']


def site_headers():
    env = credentials()
    user, password = env.get('HACKATHON_SITE_USER'), env.get('HACKATHON_SITE_PASSWORD')
    if user and password:
        return {'Authorization': 'Basic ' + base64.b64encode((user + ':' + password).encode()).decode()}
    return {}


def schema():
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = gql('{__schema{types{name kind fields{name description type{kind name ofType{kind name ofType{kind name}}}}}}}')
    write(CACHE / 'schema.json', clean(result))
    relevant = [t for t in result['__schema']['types'] if
                any(v in t['name'].lower() for v in ('pano', 'scan', 'model'))]
    fields = [{'type': t['name'], 'field': f['name'], 'description': f.get('description'), 'return_type': f['type']}
              for t in relevant for f in t.get('fields') or [] if
              any(v in f['name'].lower() for v in ('capture', 'time', 'date', 'creat', 'scan', 'pano'))]
    write(OUT / 'capture_time_schema.json', {'started_utc': started,
          'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'fields': clean(fields),
          'relevant_type_names': [t['name'] for t in relevant]})
    print(json.dumps({'schema_types': len(result['__schema']['types']), 'time_related_fields': fields}, default=str))


def select():
    if (OUT / 'selection.json').exists():
        print('Selection already frozen; unchanged.')
        return
    eligible = []
    for row in csv.DictReader((ROOT / 'models_index.csv').open()):
        path = ROOT / 'data/sun' / row['sid'] / 'sun.json'
        if row['splat'] != 'yes' or row['segment'] != 'residential' or not path.exists():
            continue
        sun = read(path)
        if sun['registration'].get('poor') is not False:
            continue
        if sum(w['type'] == 'window' for w in sun['windows']) < 2:
            continue
        eligible.append(row)
    # Original index order, frozen before viewing patches or computing residuals.
    chosen = eligible[:30]
    write(OUT / 'selection.json', {'frozen_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
          'rule': 'First 30 in original models_index.csv order, residential, splat=yes, sun.json present, registration.poor=false, >=2 assigned windows. First 15; second 15 only if fewer than 3 qualify.',
          'eligible_count': len(eligible), 'rows': chosen})
    for row in chosen:
        dest = CACHE / row['sid']
        write(dest / 'sun_snapshot.json', read(ROOT / 'data/sun' / row['sid'] / 'sun.json'))
        write(dest / 'model_snapshot.json', read(ROOT / 'data/cache' / row['sid'] / 'matterport.json'))
    print(json.dumps({'eligible': len(eligible), 'selected': len(chosen)}))


def model(sid):
    return read(CACHE / sid / 'model_snapshot.json')['data']['model']


def frames(sid):
    m = model(sid)
    g = m['geocoordinates']
    lat, lon = float(g['p0']['lat']), float(g['p0']['long'])
    columns = []
    for key in ('px', 'py'):
        p = g[key]
        columns.append([(float(p['long']) - lon) * 111320 * math.cos(math.radians(lat)) / 20,
                        (float(p['lat']) - lat) * 110574 / 20])
    u, _, v = np.linalg.svd(np.array(columns).T)
    delivered = u @ v
    sun = read(CACHE / sid / 'sun_snapshot.json')
    theta = math.radians(sun['registration']['fix_rot_deg'])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return delivered, rot @ delivered


def date_info(sid):
    path = CACHE / sid / 'drone.json'
    if path.exists():
        return read(path)
    catalog_path = CACHE / sid / 'catalog_names.json'
    if catalog_path.exists():
        catalog = read(catalog_path)
    else:
        raw = request(SITE + '/api/models/' + sid, headers=site_headers())
        catalog = json.loads(raw)
        write(catalog_path, {'images': clean(catalog.get('images', []))})
    names = [str(v) for v in catalog.get('images', [])]
    dated = [(n, re.search(r'DJI_(\d{14})', n)) for n in names]
    dated = [(n, dt.datetime.strptime(match[1], '%Y%m%d%H%M%S')) for n, match in dated if match]
    dated.sort(key=lambda v: v[1])
    filename, stamp = dated[len(dated) // 2] if dated else (names[len(names)//2], None)
    head = request(SITE + '/resources/' + sid + '/drone-imagery/' + filename,
                   headers={**site_headers(), 'Range': 'bytes=0-49151'})
    # Extract numeric XMP only; JPEG headers and signed redirect URLs are not persisted.
    tags = dict(re.findall(rb'drone-dji:(\w+)="([^"]*)"', head[:49152]))
    date_source = 'DJI filename'
    if stamp is None:
        match = re.search(rb'xmp:CreateDate="([^"]*)"', head[:49152])
        if not match:
            raise RuntimeError('No_drone_capture_datetime_in_filename_or_XMP')
        stamp = dt.datetime.fromisoformat(match[1].decode().replace('Z', '+00:00'))
        date_source = 'XMP CreateDate; filenames are DJI_####.JPG without a timestamp'
    xmp = {}
    for key in ('GpsLatitude', 'GpsLongitude', 'AbsoluteAltitude', 'RelativeAltitude', 'GimbalYawDegree', 'GimbalPitchDegree'):
        if key.encode() in tags:
            try:
                xmp[key] = float(tags[key.encode()])
            except ValueError:
                pass
    geo = model(sid)['geocoordinates']
    tz = 'America/Los_Angeles' if float(geo['longitude']) < -110 else 'America/Chicago'
    value = {'photo_count': len(names), 'dated_count': len(dated), 'selected_file': filename,
             'first_filename_time': dated[0][1].isoformat() if dated else None, 'last_filename_time': dated[-1][1].isoformat() if dated else None,
             'filename_time': stamp.isoformat(), 'timezone': tz,
             'datetime_source': date_source,
             'filename_clock_assumption': 'DJI filename interpreted as local civil time; EXIF timezone not established.',
             'xmp': xmp, 'head_bytes_received': len(head), 'head_sha256': hashlib.sha256(head[:49152]).hexdigest()}
    write(path, value)
    return value


def engine_directions(sid, drone):
    sun = read(CACHE / sid / 'sun_snapshot.json')
    stamp = dt.datetime.fromisoformat(drone['filename_time']).replace(tzinfo=ZoneInfo(drone['timezone']))
    times = [stamp + dt.timedelta(minutes=v) for v in (-90, -45, 0, 45, 90)]
    g = model(sid)['geocoordinates']
    az, _, elevation = A.solar(times, float(g['latitude']), float(g['longitude']))
    matrix = frames(sid)[1]
    en = np.column_stack([np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(elevation)),
                          np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(elevation))])
    directions = np.column_stack([en @ matrix, np.sin(np.deg2rad(elevation))])
    return sun, times, directions


def pick(sid, panos, drone):
    sun, times, directions = engine_directions(sid, drone)
    rooms = {r['id']: r for r in sun['rooms']}
    raw_rooms = {r['id']: r for r in model(sid)['rooms']}
    # Import existing engine with all cache/font writes redirected to allowed cache.
    engine = load_engine()
    roofs = [{'height_model_z_m': r['height_model_z_m'], 'ring': np.array(r['ring'])} | {
             k: np.array(v) if isinstance(v, list) else v for k, v in r.items()
             if k not in ('height_model_z_m', 'ring')} for r in sun['own_roof_layers']]
    obstacles = [{**o, 'ring': np.array(o['ring_model_xy_m'])} for o in sun['obstructions']]
    candidates = []
    for w in sun['windows']:
        if w['type'] != 'window':
            continue
        r = rooms[w['room_id']]
        tangent = np.array(w['tangent'])
        n = np.array([-tangent[1], tangent[0]])
        seg = engine.segments(raw_rooms[r['id']].get('boundary'))
        inside_positive = engine.inside_segments((np.array(w['center']) + .15*n)[None], seg)[0]
        window = {**w, 'outward': -n if inside_positive else n}
        codes = engine.aperture_verdict(window, directions, roofs, obstacles)
        hits = np.count_nonzero(codes == 0, axis=1)
        for p in panos:
            pos = np.array([p['position'][k] for k in ('x', 'y', 'z')])
            if not (0.45 < pos[2] - r['floor_z_m'] < 2.6):
                continue
            if not engine.inside_segments(pos[None, :2], seg)[0]:
                continue
            distance = float(np.linalg.norm(pos[:2] - w['center']))
            if distance > 6:
                continue
            score = int(hits.max()) * 3 + int(hits[2]) * 2 + w['width'] * min(w['height'], 2) - distance
            candidates.append({'pano_id': p['id'], 'window_id': w['id'], 'room_id': r['id'],
                'floor_z_m': r['floor_z_m'], 'distance_m': distance, 'score': score,
                'clear_samples_near_drone_time': hits.tolist(), 'sample_times': [t.isoformat() for t in times]})
    candidates.sort(key=lambda v: (-v['score'], v['pano_id'], v['window_id']))
    selected = []
    for candidate in candidates:
        if candidate['pano_id'] in {v['pano_id'] for v in selected}:
            continue
        selected.append(candidate)
        if len(selected) == 3:
            break
    return {'candidates': candidates, 'selected': selected,
            'selection_basis': 'Existing engine 3x3 aperture occlusion at drone local filename time +/-90 minutes, room membership, window proximity. Nonpositive candidates retained explicitly if no clear candidates.'}


def load_engine():
    if 'brief_v_engine' in sys.modules:
        return sys.modules['brief_v_engine']
    previous = os.environ.get('MPLCONFIGDIR')
    spec = importlib.util.spec_from_file_location('brief_v_engine', ROOT / 'experiments/sun_engine.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['brief_v_engine'] = module
    # The engine hardcodes MPLCONFIGDIR; data/cache is an allowed output directory.
    os.environ['MPLCONFIGDIR'] = str(CACHE / 'matplotlib')
    tree = ast.parse((ROOT / 'experiments/sun_engine.py').read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Attribute) and target.value.attr == 'environ'
        for target in node.targets))]
    exec(compile(tree, str(ROOT / 'experiments/sun_engine.py'), 'exec'), module.__dict__)
    return module


def acquire(batch):
    rows = read(OUT / 'selection.json')['rows'][(batch-1)*15:batch*15]
    for number, row in enumerate(rows, start=1):
        sid = row['sid']
        folder = CACHE / sid
        try:
            drone = date_info(sid)
            pose_path = folder / 'panos.json'
            if pose_path.exists():
                panos = read(pose_path)
            else:
                data = gql('{model(id:"%s"){panoLocations{id label position{x y z} rotation{x y z w}}}}' % sid)
                panos = data['model']['panoLocations']
                write(pose_path, clean(panos))
            chosen_path = folder / 'chosen.json'
            if not chosen_path.exists():
                write(chosen_path, pick(sid, panos, drone))
            chosen = read(chosen_path)['selected']
            missing = [(c, f) for c in chosen for f in range(6)
                       if not (folder / c['pano_id'] / f'face{f}.jpg').exists()]
            if missing:
                data = gql('{model(id:"%s"){panoLocations{id skybox(resolution:"2k"){children}}}}' % sid)
                links = {p['id']: p['skybox']['children'] for p in data['model']['panoLocations'] if p.get('skybox')}
                def download(pair):
                    c, face = pair
                    raw = request(links[c['pano_id']][face])
                    image = Image.open(io.BytesIO(raw)).convert('RGB')
                    path = folder / c['pano_id'] / f'face{face}.jpg'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path, quality=95)
                    return {'pano_id': c['pano_id'], 'face': face, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                            'saved_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'resolution': list(image.size)}
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    downloads = list(pool.map(download, missing))
                write(folder / 'downloads.json', downloads)
            print(json.dumps({'batch': batch, 'number': number, 'sid': sid, 'panos': len(chosen), 'status': 'ok'}), flush=True)
        except RuntimeError as error:
            write(folder / 'failure.json', {'reason': str(error)})
            print(json.dumps({'batch': batch, 'number': number, 'sid': sid, 'status': str(error)}), flush=True)
            if AUTH_FAILURES >= 2:
                break
    progress(f'Frozen 30-home cohort; batch {batch} acquisition completed. Safe caches: {CACHE.relative_to(ROOT)}. Network calls this process: {CALLS}.', 'Render and visually inspect contact sheets; save marks before comparing alignments.')


def sample_image(vectors, pose, faces):
    q = A.rotation(pose['rotation'])
    face, uv = A.cube_coordinates(vectors, q)
    result = np.zeros((*face.shape, 3), dtype=np.uint8)
    for k, image in enumerate(faces):
        mask = face == k
        if not mask.any():
            continue
        a = np.asarray(image)
        xy = np.clip((uv[mask] * (image.width - 1)).astype(int), 0, image.width - 1)
        result[mask] = a[xy[:, 1], xy[:, 0]]
    return Image.fromarray(result)


def rectified(sid, candidate, pose, faces, size=800):
    room = next(r for r in model(sid)['rooms'] if r['id'] == candidate['room_id'])
    vertices = np.array([[v['position'][k] for k in ('x', 'y')]
                         for e in room['boundary']['edges'] for v in e['vertices']])
    lo, hi = vertices.min(0) - .1, vertices.max(0) + .1
    span = max(hi-lo)
    center = (hi+lo)/2
    lo, hi = center - span/2, center + span/2
    x, y = np.meshgrid(np.linspace(lo[0], hi[0], size), np.linspace(hi[1], lo[1], size))
    points = np.stack([x, y, np.full_like(x, candidate['floor_z_m'])], axis=-1)
    pos = np.array([pose['position'][k] for k in ('x', 'y', 'z')])
    image = sample_image(points - pos, pose, faces)
    return image, {'lo': lo.tolist(), 'hi': hi.tolist(), 'size': size, 'floor_z_m': candidate['floor_z_m'],
                   'pixel_to_model': 'x=lo[0]+u/(size-1)*(hi[0]-lo[0]); y=hi[1]-v/(size-1)*(hi[1]-lo[1])'}


def render(batch):
    rows = read(OUT / 'selection.json')['rows'][(batch-1)*15:batch*15]
    for row in rows:
        sid = row['sid']
        folder = CACHE / sid
        if not (folder / 'chosen.json').exists():
            continue
        panos = {p['id']: p for p in read(folder / 'panos.json')}
        chosen = read(folder / 'chosen.json')['selected']
        tiles = []
        for c in chosen:
            pid = c['pano_id']
            if not all((folder / pid / f'face{f}.jpg').exists() for f in range(6)):
                continue
            faces = [Image.open(folder / pid / f'face{f}.jpg').convert('RGB') for f in range(6)]
            # Cap basis is inferred from seams, independent of solar hypotheses.
            seams = A.check_cap_seams(faces)
            image, transform = rectified(sid, c, panos[pid], faces)
            path = OUT / sid
            path.mkdir(parents=True, exist_ok=True)
            image.save(path / f'{pid}_floor.jpg', quality=95)
            write(path / f'{pid}_projection.json', {'candidate': c, 'transform': transform, 'cap_seams': seams,
                   'basis': A.BASES.tolist(), 'pose': panos[pid]})
            # Window-facing perspective: shows both aperture and nearby floor.
            sun = read(folder / 'sun_snapshot.json')
            w = next(w for w in sun['windows'] if w['id'] == c['window_id'])
            pos = np.array([panos[pid]['position'][k] for k in ('x', 'y', 'z')])
            forward = np.array([w['center'][0]-pos[0], w['center'][1]-pos[1], -.6])
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 0, 1.]); right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            u, v = np.meshgrid(np.linspace(-1.3,1.3,800), np.linspace(-.8,1.2,650))
            perspective = sample_image(forward + u[...,None]*right + v[...,None]*down, panos[pid], faces)
            perspective.save(path / f'{pid}_view.jpg', quality=95)
            tile = Image.new('RGB', (1600,850), 'white')
            tile.paste(image,(0,50));tile.paste(perspective,(800,50))
            d = ImageDraw.Draw(tile)
            d.text((10,8), f'{sid} pano {panos[pid].get("label", "?")} {pid} | {c["window_id"]} | {c["room_id"]} | clear={c["clear_samples_near_drone_time"]}', fill='black',font=ImageFont.load_default(size=20))
            tiles.append(tile)
        if tiles:
            sheet = Image.new('RGB',(1600,850*len(tiles)),'white')
            for i,tile in enumerate(tiles):sheet.paste(tile,(0,850*i))
            sheet.save(OUT/sid/'contact_sheet.jpg',quality=91)
            print(json.dumps({'sid':sid,'tiles':len(tiles)}),flush=True)
    progress(f'Batch {batch} contact sheets and per-pano floor rectifications rendered in data/sun_patch/<sid>/. No comparisons computed.', 'Visually inspect every contact sheet; freeze observations and patch-corner marks.')


def window_corners(window):
    center = np.array(window['center'])
    half = np.array(window['tangent']) * window['width'] / 2
    low = window['lower_z_m']
    return np.array([[*(center-half), low], [*(center+half), low],
                     [*(center+half), low+window['height']],
                     [*(center-half), low+window['height']]])


def geometry_views():
    """Independent graph/panorama check, before any solar comparisons."""
    for sid in ('1ErEvp9NR7U', '2uxUNEWGeww'):
        selected = read(CACHE/sid/'chosen.json')['selected']
        windows = read(CACHE/sid/'sun_snapshot.json')['windows']
        for candidate in selected:
            pid = candidate['pano_id']
            projection = read(OUT/sid/(pid+'_projection.json'))
            A.BASES = np.array(projection['basis'])
            pose = projection['pose']
            position = np.array([pose['position'][k] for k in ('x','y','z')])
            q = A.rotation(pose['rotation'])
            faces = [Image.open(CACHE/sid/pid/f'face{i}.jpg').convert('RGB') for i in range(6)]
            sheet = Image.new('RGB', (2304,1536), 'white')
            relevant = [w for w in windows if w['type']=='window' and w['room_id']==candidate['room_id']]
            for index, face in enumerate(faces):
                d = ImageDraw.Draw(face)
                for w in relevant:
                    corners = window_corners(w)
                    for a,b in zip(corners,np.roll(corners,-1,axis=0)):
                        f,uv=A.cube_coordinates(np.linspace(a,b,100)-position,q)
                        for j in range(99):
                            if f[j]==index and f[j+1]==index:
                                d.line([tuple(uv[j]*2047),tuple(uv[j+1]*2047)],fill='magenta',width=5)
                    f,uv=A.cube_coordinates(corners-position,q)
                    for j in range(4):
                        if f[j]==index:
                            d.text(tuple(uv[j]*2047),w['id']+':'+str(j),fill='magenta',
                                   stroke_width=1,stroke_fill='black',font=ImageFont.load_default(size=28))
                d.text((20,20),f'{pid} face {index}',fill='yellow',font=ImageFont.load_default(size=36))
                face.save(OUT/sid/(pid+f'_geometry_face{index}.jpg'),quality=94)
                sheet.paste(face.resize((768,768)),((index%3)*768,(index//3)*768))
            sheet.save(OUT/sid/(pid+'_geometry_sheet.jpg'),quality=94)
    progress('Window geometry reprojection sheets produced offline for both provisional homes. No solar heading comparisons computed.',
             'Check aperture-corner identity and projection agreement; freeze marks.')


def wrap(angle):
    return (np.asarray(angle) + 180) % 360 - 180


def floor_point(uv, transform):
    lo, hi = np.array(transform['lo']), np.array(transform['hi'])
    xy = lo + np.array([uv[0], transform['size']-1-uv[1]]) / (transform['size']-1) * (hi-lo)
    return np.array([*xy, transform['floor_z_m']])


def floor_pixel(xyz, transform):
    lo, hi = np.array(transform['lo']), np.array(transform['hi'])
    uv = (np.asarray(xyz)[..., :2]-lo)/(hi-lo)*(transform['size']-1)
    uv[..., 1] = transform['size']-1-uv[..., 1]
    return uv


def aperture_points(patch, window, plane_offset=0):
    """Intersect independently marked image rays with supplied window wall plane."""
    p = read(OUT/patch['sid']/(patch['window_mark_pano_id']+'_projection.json'))
    basis = np.array(p['basis'])
    pose = p['pose']
    position = np.array([pose['position'][k] for k in ('x','y','z')])
    q = A.rotation(pose['rotation'])
    tangent = np.array(window['tangent'])
    normal = np.array([tangent[1], -tangent[0], 0.])
    plane = np.array([*window['center'], window['lower_z_m']]) + plane_offset*normal
    points = window_corners(window)
    for mark in patch['window_marks']:
        center, right, down = basis[mark['face']]
        uv = np.array(mark['uv']) / 2047 * 2 - 1
        ray = q @ (center + uv[0]*right + uv[1]*down)
        distance = np.dot(normal, plane-position)/np.dot(normal, ray)
        if distance <= 0:
            raise RuntimeError('window_ray_plane_behind_camera')
        points[mark['corner']] = position + distance*ray
    return points


SOLAR_TRACKS = {}


def solar_root(sid, elevations, day_offset=0, branch='morning'):
    """30-second NOAA/Meeus path interpolation, using apparent elevation."""
    key = (sid, day_offset, branch)
    if key not in SOLAR_TRACKS:
        drone = read(CACHE/sid/'drone.json')
        date = dt.datetime.fromisoformat(drone['filename_time']).date() + dt.timedelta(days=day_offset)
        midnight = dt.datetime.combine(date, dt.time(), ZoneInfo(drone['timezone']))
        times = [midnight + dt.timedelta(seconds=30*i) for i in range(2881)]
        g = model(sid)['geocoordinates']['p0']
        az, geo, apparent = A.solar(times, float(g['lat']), float(g['long']))
        peak = int(np.argmax(apparent))
        ix = np.arange(0, peak+1) if branch=='morning' else np.arange(2880,peak-1,-1)
        ix = ix[apparent[ix] >= 0]
        SOLAR_TRACKS[key] = (apparent[ix], az[ix], np.array([times[i].timestamp() for i in ix]))
    el, az, stamp = SOLAR_TRACKS[key]
    target = np.asarray(elevations)
    return (np.interp(target, el, az, left=np.nan, right=np.nan),
            np.interp(target, el, stamp, left=np.nan, right=np.nan))


def direction_stats(vectors):
    vectors = np.atleast_2d(vectors)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    mean = unit.mean(axis=0)
    az = float(np.degrees(np.arctan2(mean[0], mean[1])) % 360)
    el = float(np.degrees(np.arctan2(mean[2], np.linalg.norm(mean[:2]))))
    spread = float(np.degrees(np.arccos(np.clip(unit @ (mean/np.linalg.norm(mean)), -1, 1))).max())
    return az, el, spread


def evaluate(sid, vectors, day_offset=0, branch='morning'):
    az, el, spread = direction_stats(vectors)
    true_az, stamp = solar_root(sid, el, day_offset, branch)
    matrices = frames(sid)
    scan_xy = np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])
    compass = [float(np.degrees(np.arctan2(*(m@scan_xy))) % 360) for m in matrices]
    errors = [float(wrap(a-true_az)) for a in compass]
    return {'scan_azimuth_deg': az, 'scan_elevation_deg': el, 'corner_max_deviation_deg': spread,
            'true_sun_azimuth_deg': float(true_az), 'epoch': float(stamp),
            'delivered_compass_deg': compass[0], 'registered_compass_deg': compass[1],
            'delivered_error_deg': errors[0], 'registered_error_deg': errors[1]}


def numeric_record(record, timezone):
    result = dict(record)
    stamp = result.pop('epoch')
    result['implied_local_time'] = dt.datetime.fromtimestamp(stamp, ZoneInfo(timezone)).isoformat(timespec='seconds') if np.isfinite(stamp) else None
    return result


def error_envelope(sid, points, aperture, pairs, day_offsets, mode):
    """Corner-wise systematic envelope. Includes disagreement between corner rays."""
    vectors = []
    for i, pair in enumerate(pairs):
        angles = np.arange(32)*2*np.pi/32 if mode in ('floor', 'combined') else np.array([0.])
        for angle in angles:
            point = points[i].copy()
            if mode in ('floor', 'combined'):
                point[:2] += .05*np.array([math.cos(angle), math.sin(angle)])
            for height in (-.05,0.,.05) if mode in ('height', 'combined') else (0.,):
                w = aperture[pair['corner']].copy()
                w[2] += height
                vectors.append(w-point)
    vectors = np.array(vectors)
    scan_az = np.degrees(np.arctan2(vectors[:,0],vectors[:,1])) % 360
    elevations = np.degrees(np.arctan2(vectors[:,2],np.linalg.norm(vectors[:,:2],axis=1)))
    matrices = frames(sid)
    xy = np.column_stack([np.sin(np.radians(scan_az)), np.cos(np.radians(scan_az))])
    ranges = [[],[]]
    stamps = []
    for day in day_offsets:
        true, stamp = solar_root(sid, elevations, day)
        stamps.extend((stamp-day*86400).tolist())
        for i, m in enumerate(matrices):
            compass_xy = xy @ m.T
            compass = np.degrees(np.arctan2(compass_xy[:,0],compass_xy[:,1])) % 360
            ranges[i].extend(wrap(compass-true).tolist())
    failed = int(np.count_nonzero(~np.isfinite(ranges[0])))
    def limit(values, operation):
        finite=np.asarray(values)[np.isfinite(values)]
        return float(operation(finite)) if len(finite) else None
    return {'delivered_min_deg': limit(ranges[0],np.min), 'delivered_max_deg': limit(ranges[0],np.max),
            'registered_min_deg': limit(ranges[1],np.min), 'registered_max_deg': limit(ranges[1],np.max),
            'clock_min_epoch_on_nominal_date': limit(stamps,np.min), 'clock_max_epoch_on_nominal_date': limit(stamps,np.max),
            'no_above_horizon_root_count':failed, 'evaluated_count':len(ranges[0])}


def perturbation_effects(sid, vectors):
    """Changes of the patch mean, separate from baseline corner disagreement."""
    baseline=evaluate(sid,vectors)
    candidates={name:[] for name in ('one_floor_corner_5cm','one_window_corner_5cm_3d','common_sill_height_5cm','date_1day','date_7days')}
    for i in range(len(vectors)):
        for angle in np.arange(32)*2*np.pi/32:
            change=.05*np.array([math.cos(angle),math.sin(angle),0.])
            altered=vectors.copy();altered[i]-=change
            candidates['one_floor_corner_5cm'].append(evaluate(sid,altered))
            for tilt in np.linspace(-np.pi/2,np.pi/2,9):
                delta=np.array([change[0]*math.cos(tilt),change[1]*math.cos(tilt),.05*math.sin(tilt)])
                altered=vectors.copy();altered[i]+=delta
                candidates['one_window_corner_5cm_3d'].append(evaluate(sid,altered))
    for height in (-.05,.05):
        altered=vectors.copy();altered[:,2]+=height
        candidates['common_sill_height_5cm'].append(evaluate(sid,altered))
    for name,offsets in [('date_1day',(-1,1)),('date_7days',range(-7,8))]:
        candidates[name]=[evaluate(sid,vectors,day) for day in offsets]
    rows=[]
    for name, values in candidates.items():
        delta=[float(wrap(v['delivered_error_deg']-baseline['delivered_error_deg'])) for v in values]
        rows.append({'perturbation':name,'heading_change_min_deg':min(delta),'heading_change_max_deg':max(delta),
                     'maximum_abs_heading_change_deg':max(abs(x) for x in delta),
                     'elevation_change_min_deg':min(v['scan_elevation_deg']-baseline['scan_elevation_deg'] for v in values),
                     'elevation_change_max_deg':max(v['scan_elevation_deg']-baseline['scan_elevation_deg'] for v in values),
                     'implied_clock_change_min_minutes':min((v['epoch']-baseline['epoch'])/60 for v in values) if name.startswith('one_') or name.startswith('common_') else None,
                     'implied_clock_change_max_minutes':max((v['epoch']-baseline['epoch'])/60 for v in values) if name.startswith('one_') or name.startswith('common_') else None})
    return rows


def joint_scenarios(detail):
    """Paired alignment comparisons with common nuisances across a home's patches."""
    rows, summaries = [], []
    for sid in sorted({x['patch']['sid'] for x in detail}):
        items = [x for x in detail if x['patch']['sid']==sid and x['result']['usable_for_summary']]
        plane, angle, height = np.meshgrid([-.1,0.,.1], np.arange(32)*np.pi/16,
                                          [-.05,0.,.05], indexing='ij')
        plane, angle, height = plane.ravel(), angle.ravel(), height.ravel()
        floor_shift = np.column_stack([.05*np.cos(angle), .05*np.sin(angle), -height])
        azimuths, elevations = [], []
        for item in items:
            patch = item['patch']
            window = next(w for w in read(CACHE/sid/'sun_snapshot.json')['windows'] if w['id']==patch['window_id'])
            by_plane = {offset:aperture_points(patch,window,offset) for offset in (-.1,0.,.1)}
            corners = [p['corner'] for p in patch['pairs']]
            aperture = np.array([by_plane[offset][corners] for offset in plane])
            vectors = aperture-np.array(item['floor_points_xyz_m'])[None,:,:]-floor_shift[:,None,:]
            mean = (vectors/np.linalg.norm(vectors,axis=-1,keepdims=True)).mean(axis=1)
            azimuths.append(np.arctan2(mean[:,0],mean[:,1]))
            elevations.append(np.degrees(np.arctan2(mean[:,2],np.linalg.norm(mean[:,:2],axis=1))))
        azimuths, elevations = np.array(azimuths), np.array(elevations)
        xy = np.stack([np.sin(azimuths),np.cos(azimuths)],axis=-1)
        compass = []
        for matrix in frames(sid):
            en = xy @ matrix.T
            compass.append(np.degrees(np.arctan2(en[:,:,0],en[:,:,1])))
        home_rows = []
        for day in range(-7,8):
            true, stamp = solar_root(sid,elevations,day)
            absolute_errors = [np.median(np.abs(wrap(c-true)),axis=0) for c in compass]
            for i in range(len(plane)):
                row = {'sid':sid,'day_offset':day,'plane_offset_m':float(plane[i]),
                       'floor_mark_shift_azimuth_deg':float(np.degrees(angle[i])),
                       'common_window_height_error_m':float(height[i]),
                       'median_abs_delivered_error_deg':float(absolute_errors[0][i]),
                       'median_abs_registered_error_deg':float(absolute_errors[1][i]),
                       'registration_advantage_deg':float(absolute_errors[0][i]-absolute_errors[1][i])}
                home_rows.append(row)
        rows.extend(home_rows)
        summaries.append({'sid':sid,'scenario_count':len(home_rows),
                          'minimum_advantage':min(home_rows,key=lambda r:r['registration_advantage_deg']),
                          'maximum_advantage':max(home_rows,key=lambda r:r['registration_advantage_deg'])})
    with (OUT/'joint_scenarios.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'alignment_discrimination.json',{
        'interpretation':'Positive advantage means registered has smaller median absolute error. Shared perturbations apply to all retained patches in a home. Deterministic sensitivity grid, not a probability or confidence interval.',
        'nuisances':'All integer date offsets -7..7; floor marks shifted by 5cm in 32 common horizontal directions; common aperture height -5/0/+5cm; common wall plane -10/0/+10cm. No mark is revised. Other window-corner identity/pose errors remain unbounded.',
        'homes':summaries})


def measure():
    marks_path = OUT/'marks_v1.json'
    receipt = read(OUT/'marks_freeze_receipt.json')
    mark_hash = hashlib.sha256(marks_path.read_bytes()).hexdigest()
    if mark_hash != receipt['sha256'] or not receipt['comparison_outputs_absent']:
        raise RuntimeError('marks_freeze_receipt_mismatch')
    marks = read(marks_path)
    detail, rows, days, sensitivities, effects = [], [], [], [], []
    for patch in marks['patches']:
        sid = patch['sid']
        projection = read(OUT/sid/(patch['pano_id']+'_projection.json'))
        transform = projection['transform']
        w = next(w for w in read(CACHE/sid/'sun_snapshot.json')['windows'] if w['id']==patch['window_id'])
        aperture = aperture_points(patch,w)
        points = np.array([floor_point(p['floor_uv'], transform) for p in patch['pairs']])
        vectors = np.array([aperture[p['corner']]-points[i] for i,p in enumerate(patch['pairs'])])
        nominal = evaluate(sid,vectors)
        timezone = read(CACHE/sid/'drone.json')['timezone']
        per_corner = [numeric_record(evaluate(sid,[v]),timezone) for v in vectors]
        box = window_corners(w)
        box_result = evaluate(sid,np.array([box[p['corner']]-points[i] for i,p in enumerate(patch['pairs'])]))
        alternative = evaluate(sid,vectors,branch='afternoon')
        outward = np.array([w['tangent'][1],-w['tangent'][0]])
        branch_checks = {}
        for name,result in [('morning',nominal),('afternoon',alternative)]:
            en = np.array([math.sin(math.radians(result['true_sun_azimuth_deg'])), math.cos(math.radians(result['true_sun_azimuth_deg']))])
            branch_checks[name] = [float(np.dot(m@outward,en)) for m in frames(sid)]
        sens = {}
        for name, offsets, mode in [('corner_5cm',[0],'floor'),('height_5cm',[0],'height'),('date_1day',[-1,0,1],'none'),('date_7days',range(-7,8),'none'),('combined',range(-7,8),'combined')]:
            sens[name] = error_envelope(sid,points,aperture,patch['pairs'],offsets,mode)
        sens['required_joint_without_plane']=dict(sens['combined'])
        # Joint plane-location sensitivity is applied before any result is judged.
        for offset in (-.1,.1):
            extra = error_envelope(sid,points,aperture_points(patch,w,offset),patch['pairs'],range(-7,8),'combined')
            for key in sens['combined']:
                if key in ('no_above_horizon_root_count','evaluated_count'):
                    sens['combined'][key] += extra[key]
                elif extra[key] is not None:
                    sens['combined'][key] = (min if 'min' in key else max)(sens['combined'][key], extra[key])
        corner_offsets = [float(np.linalg.norm(aperture[p['corner']]-box[p['corner']])) for p in patch['pairs']]
        # Gross aperture/graph disagreement invalidates the inferred ray, regardless of heading.
        usable = max(corner_offsets) <= .25 and sens['combined']['no_above_horizon_root_count']==0
        record = {'patch_id':patch['patch_id'],'sid':sid,'pano_id':patch['pano_id'],
                  'scan_label':projection['pose']['label'],'room_id':w['room_id'],'window_id':w['id'],
                  'status':patch['status'],'corner_count':len(points), **numeric_record(nominal,timezone),
                  'usable_for_summary':usable,'max_glazing_to_box_corner_offset_m':max(corner_offsets),
                  'box_delivered_error_deg':box_result['delivered_error_deg'],
                  'box_registered_error_deg':box_result['registered_error_deg'],
                  'combined_delivered_min_deg':sens['combined']['delivered_min_deg'],
                  'combined_delivered_max_deg':sens['combined']['delivered_max_deg'],
                  'combined_registered_min_deg':sens['combined']['registered_min_deg'],
                  'combined_registered_max_deg':sens['combined']['registered_max_deg']}
        rows.append(record)
        for day in range(-7,8):
            days.append({'patch_id':patch['patch_id'],'sid':sid,'day_offset':day,
                         **numeric_record(evaluate(sid,vectors,day),timezone)})
        for name, limits in sens.items():
            sensitivities.append({'patch_id':patch['patch_id'],'sid':sid,'perturbation':name,**limits})
        effects.extend({'patch_id':patch['patch_id'],'sid':sid,**effect} for effect in perturbation_effects(sid,vectors))
        detail.append({'patch':patch, 'result':record,'window_aperture_xyz_m':aperture.tolist(),
                       'provided_box_xyz_m':box.tolist(),'floor_points_xyz_m':points.tolist(),
                       'glazing_to_box_corner_offsets_m':corner_offsets,
                       'geometry_qc':'Keep if paired glazing/model corner offsets <=25cm (five times requested mark error) and full joint sweep has solar roots. Rule introduced after first attempted run exposed gross geometry failure, independent of alignment residuals.',
                       'corner_results':per_corner,'box_result':numeric_record(box_result,timezone),
                       'afternoon_alternative':numeric_record(alternative,timezone),
                       'branch_outward_dot_delivered_registered':branch_checks,'sensitivity':sens})
    for filename, table in [('results.csv',rows),('date_sensitivity.csv',days),('sensitivity.csv',sensitivities),('perturbation_effects.csv',effects)]:
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write(OUT/'measurement_detail.json',{'marks_sha256':mark_hash,'solar_method':'Existing NOAA/Meeus implementation, apparent elevation; 30-second monotone interpolation', 'patches':detail})
    home_rows=[]
    for sid in sorted({r['sid'] for r in rows}):
        home=[r for r in rows if r['sid']==sid and r['usable_for_summary']]
        summary={'sid':sid,'patch_count':len(home),'corner_count':sum(r['corner_count'] for r in home),
                 'implied_local_time_min':min(r['implied_local_time'] for r in home),
                 'implied_local_time_max':max(r['implied_local_time'] for r in home)}
        for alignment in ('delivered','registered'):
            errors=[r[alignment+'_error_deg'] for r in home]
            summary.update({alignment+'_median_deg':float(np.median(errors)),alignment+'_min_deg':min(errors),alignment+'_max_deg':max(errors),alignment+'_spread_deg':max(errors)-min(errors) if len(errors)>1 else None,
                            alignment+'_combined_min_deg':min(r['combined_'+alignment+'_min_deg'] for r in home),alignment+'_combined_max_deg':max(r['combined_'+alignment+'_max_deg'] for r in home)})
        home_rows.append(summary)
    write(OUT/'home_summary.json',home_rows)
    joint_scenarios(detail)
    make_overlays(detail)
    make_window_marks(marks)
    A.solar_reference(OUT)
    progress('Steps 2–5 computed from frozen marks: 5 attempted patches / 9 rays; 4 conditional retained patches / 7 rays in 2 homes. H2_DOWN rejected for gross window geometry disagreement. results.csv, measurement_detail.json, home_summary.json, date_sensitivity.csv, sensitivity.csv, perturbation_effects.csv and both overlay.jpg and window_marks.jpg files written. Inputs remain frozen; no acquisition.',
             'Inspect overlays and corner consistency; run offline deterministic replay and existing seven-test guard; write honest report.')


def make_overlays(detail):
    font = ImageFont.load_default(size=15)
    for sid in sorted({d['patch']['sid'] for d in detail}):
        entries=[d for d in detail if d['patch']['sid']==sid]
        panos=list(dict.fromkeys(d['patch']['pano_id'] for d in entries))
        sheet=Image.new('RGB',(800*len(panos),900),'white')
        for column,pid in enumerate(panos):
            im=Image.open(OUT/sid/(pid+'_floor.jpg')).convert('RGB');draw=ImageDraw.Draw(im)
            t=read(OUT/sid/(pid+'_projection.json'))['transform']
            for item in [d for d in entries if d['patch']['pano_id']==pid]:
                patch=item['patch'];r=item['result'];aperture=np.array(item['provided_box_xyz_m'])
                # Engine aperture geometry stays fixed; hand glazing marks only infer the time.
                az,el=map(math.radians,(r['true_sun_azimuth_deg'],r['scan_elevation_deg']))
                en=np.array([math.sin(az),math.cos(az)])*math.cos(el)
                for matrix,color in zip(frames(sid),('red','lime')):
                    sun=np.array([*(matrix.T@en),math.sin(el)])
                    predicted=aperture-(aperture[:,2:3]-t['floor_z_m'])/sun[2]*sun
                    uv=floor_pixel(predicted,t)
                    draw.line([tuple(v) for v in np.vstack([uv,uv[0]])],fill=color,width=3)
                outline=[tuple(v) for v in patch['visible_outline']]
                if patch['outline_closed']:outline.append(outline[0])
                draw.line(outline,fill='cyan',width=3)
                for pair in patch['pairs']:
                    x,y=pair['floor_uv'];draw.ellipse((x-4,y-4,x+4,y+4),fill='cyan',outline='black',width=1)
                    draw.text((x+8,y),patch['patch_id']+':'+str(pair['corner']),fill='cyan',stroke_fill='black',stroke_width=1,font=font)
            sheet.paste(im,(column*800,100))
            sd=ImageDraw.Draw(sheet)
            sd.text((column*800+10,10),sid+'  pano '+pid,fill='black',font=font)
            sd.text((column*800+10,35),'Cyan: observed visible edges/corners; red: delivered; green: registered',fill='black',font=font)
            sd.text((column*800+10,60),'Engine aperture at patch-implied time; no roof/blind mask. Cyan clipped edges stay open.',fill='black',font=font)
            if any(not d['result']['usable_for_summary'] for d in entries if d['patch']['pano_id']==pid):
                sd.text((column*800+10,80),'FAILED aperture geometry: shown for diagnosis; excluded from usable summaries.',fill='red',font=font)
            elif any(d['patch']['status']=='single_corner_conditional' for d in entries if d['patch']['pano_id']==pid):
                sd.text((column*800+10,80),'Single visible corner: conditional; hidden aperture corners use the supplied box.',fill='black',font=font)
        sheet.save(OUT/sid/'overlay.jpg',quality=95)


def make_window_marks(marks):
    for sid in sorted({p['sid'] for p in marks['patches']}):
        patches=[p for p in marks['patches'] if p['sid']==sid]
        sheet=Image.new('RGB',(600*len(patches),900),'white')
        for column,patch in enumerate(patches):
            face=patch['window_marks'][0]['face']
            im=Image.open(CACHE/sid/patch['window_mark_pano_id']/f'face{face}.jpg').convert('RGB')
            d=ImageDraw.Draw(im);coords=[]
            for mark in patch['window_marks']:
                if mark['face']!=face:continue
                x,y=mark['uv'];coords.append([x,y]);d.ellipse((x-5,y-5,x+5,y+5),fill='cyan',outline='black',width=2)
                d.text((x+8,y),str(mark['corner']),fill='cyan',stroke_fill='black',stroke_width=1,font=ImageFont.load_default(size=30))
            xy=np.array(coords);lo=np.maximum(0,xy.min(axis=0)-100);hi=np.minimum(2048,xy.max(axis=0)+100)
            im=im.crop((*lo,*hi));im.thumbnail((590,740));sheet.paste(im,(column*600+(600-im.width)//2,140))
            d=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=16)
            d.text((column*600+10,10),patch['patch_id']+' '+patch['window_id']+' face '+str(face),fill='black',font=font)
            d.text((column*600+10,35),'Cyan: frozen hand aperture marks, raw 2048px coordinates',fill='black',font=font)
            for i,mark in enumerate(patch['window_marks']):
                d.text((column*600+10,60+i*18),f"{mark['corner']}: {mark['uv']}",fill='black',font=font)
        sheet.save(OUT/sid/'window_marks.jpg',quality=95)


def main():
    global NETWORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['schema', 'select', 'acquire', 'render', 'geometry', 'measure'])
    parser.add_argument('--acquire', action='store_true')
    parser.add_argument('--batch', type=int, default=1, choices=[1,2])
    args = parser.parse_args()
    NETWORK = args.acquire
    OUT.mkdir(parents=True, exist_ok=True)
    if not NETWORK:
        def blocked(*args, **kwargs):
            raise RuntimeError('offline_socket_guard')
        socket.socket = blocked
        socket.create_connection = blocked
    if args.command == 'schema': schema()
    elif args.command == 'select': select()
    elif args.command == 'acquire': acquire(args.batch)
    elif args.command == 'render': render(args.batch)
    elif args.command == 'geometry': geometry_views()
    elif args.command == 'measure': measure()
    print(json.dumps({'command': args.command, 'network_enabled': NETWORK,
                      'request_calls': CALLS,
                      'socket_guard': not NETWORK}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Untrusted remote values / signed URLs can be present in exception repr.
        print(json.dumps({'status': 'failed', 'exception_type': type(error).__name__,
                          'detail': str(error) if isinstance(error, RuntimeError) else 'details_suppressed'}))
        sys.exit(1)
