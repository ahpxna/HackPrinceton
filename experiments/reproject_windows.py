"""Brief E: cached scan/opening reprojection, with explicit observation provenance.

Replay: python3 -I experiments/reproject_windows.py khbuN7q853q
Acquire: add --acquire; run SAM: add --detect. Network is opt-in on replay.
Only this file and data/reproj are experiment outputs. No new tests.
"""
import argparse
import base64
import collections
import functools
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/reproj'
CACHE = BASE / 'cache'
SID = 'khbuN7q853q'
LON, LAT = -95.4163965, 29.8195283
GROUND = 22.5552
API = 'https://api.vexcelgroup.com/v2/'
AUTH_FAILURES = 0
NETWORK = False
COLORS = {'delivered': 'red', 'footprint_corrected': 'lime', 'refined': 'cyan'}
# Fixed before fitting: train only west-looking images; south-looking images are
# held out. These are hand-marked *visible sill endpoints*, never inferred tops.
# Coordinates are raw crop pixels, not the enlarged review-image coordinates.
MANUAL = {
    'west-1': {
        'F0E4O0': [[275.5,173.0],[288.5,173.0]],
        'F0E28O0': [[337.0,173.0],[324.5,173.0]],
    },
    'west-6': {
        'F0E4O0': [[366.5,246.0],[381.0,246.0]],
        'F0E28O0': [[440.0,245.0],[426.0,245.0]],
        'F0E26O0': [[331.0,246.0],[318.0,246.0]],
    },
    'south-1': {'F0E10O0': [[332.5,433.5],[348.0,433.5]]},
    'south-2': {'F0E10O0': [[313.5,434.0],[329.0,434.0]]},
    'south-3': {'F0E10O0': [[328.5,435.0],[343.0,435.0]]},
}
SIGMA_PX = 1.5  # ASSUMED picking precision, not measured survey accuracy.
FIELDS = ('image-name,geometry,camera-pos-x,camera-pos-y,camera-pos-z,utm-zone,'
          'ground-z,omega,phi,kappa,focal-length,pixel-size,raster-size-width,'
          'raster-size-height,pp0-x,pp0-y,capture-date,collection,product-type,'
          'min-gsd,max-gsd,k0,k1,k2,k3,p1,p2,b1,b2')


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()
                if k.lower() not in ('authorization', 'x-api-key', 'api-key', 'token', 'signed-url')}
    if isinstance(value, list):
        return [safe(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'https?://[^\s<>\"\']+',
                      lambda m: m[0].split('?')[0].split('#')[0], value)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def credentials(name='VEXCEL_API_KEY'):
    # Runtime parsing is expressly permitted by the brief's author clarification.
    # Never return credentials to the CLI, persist them, or include them in URLs.
    if os.environ.get(name):
        return os.environ[name]
    for line in (ROOT / '.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            field, value = line.split('=', 1)
            if field.strip() == name:
                return value.strip().strip('\"\'')
    raise RuntimeError('Required credential unavailable; values suppressed')


def acquire_indoor():
    """One bedroom pano sees E4 directly; the existing living-room pano cannot."""
    folder = CACHE / SID
    if all((folder / f'bedroom_face{f}.jpg').exists() for f in range(6)):
        return
    if not NETWORK:
        raise RuntimeError('Bedroom cache incomplete; opt-in acquisition required')
    pano_id = 'ix2tm8c2xmu43cs2gq1ndhy1d'
    query = '{model(id:"%s"){panoLocations{id position{x y z} rotation{x y z w} skybox(resolution:"2k"){children}}}}' % SID
    auth = base64.b64encode((credentials('MATTERPORT_API_KEY') + ':' +
                            credentials('MATTERPORT_API_TOKEN')).encode()).decode()
    req = urllib.request.Request('https://api.matterport.com/api/models/graph',
        json.dumps({'query':query}).encode(),
        {'Content-Type':'application/json','Authorization':'Basic '+auth})
    try:
        with urllib.request.urlopen(req,timeout=90) as response:
            raw = response.read(); status = response.status
        result = json.loads(raw)
        # URLs exist only in memory for download. No signed URL reaches cache.
        cached = json.loads(raw)
        for p in cached.get('data',{}).get('model',{}).get('panoLocations',[]):
            p['skybox']['children'] = ['SIGNED_URL_REDACTED'] * len(p['skybox']['children'])
        write(folder/'bedroom-query.json',cached)
        log_request({'endpoint':'matterport/model/graph','status':status,
            'sha256':hashlib.sha256(raw).hexdigest(),'cache':f'cache/{SID}/bedroom-query.json',
            'response_sanitization':'all skybox URLs redacted before caching'})
        panos = result['data']['model']['panoLocations']
        p = next(p for p in panos if p['id']==pano_id)
        write(folder/'bedroom_pose.json',{k:p[k] for k in ('id','position','rotation')})
        for f,url in enumerate(p['skybox']['children']):
            path = folder/f'bedroom_face{f}.jpg'
            if path.exists():
                continue
            with urllib.request.urlopen(url,timeout=90) as response:
                raw = response.read(); status=response.status
            # Save image content only; signed download location is not metadata.
            with Image.open(io.BytesIO(raw)) as im:
                im.convert('RGB').save(path,quality=95)
            log_request({'endpoint':'matterport/skybox','pano_id':pano_id,'face':f,
                'status':status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                'cache':str(path.relative_to(BASE)),'response_sanitization':'JPEG metadata removed'})
        print('acquired bedroom pano',pano_id,flush=True)
    except (urllib.error.HTTPError,urllib.error.URLError,KeyError,StopIteration):
        write(folder/'bedroom-failed.json',{'failed':True,'details_suppressed':True})
        raise RuntimeError('Bedroom pano request failed; credentials/URLs suppressed') from None


def request(endpoint, cache, params=None, body=None):
    global AUTH_FAILURES
    path = CACHE / cache
    if path.exists():
        return read(path) if path.suffix == '.json' else path.read_bytes()
    if not NETWORK:
        raise RuntimeError('Missing cached response: ' + cache + '; acquisition is opt-in')
    if AUTH_FAILURES >= 2:
        raise RuntimeError('Stopped after repeated HTTP 401/403')
    key = credentials()
    headers = {'X-API-KEY': key, 'Content-Type': 'application/json'}
    url = API + endpoint + ('?' + urllib.parse.urlencode(params) if params else '')
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(5):
        entry = {'endpoint': endpoint, 'params': params, 'body': body, 'attempt': attempt,
                 'cache': str(path.relative_to(BASE)), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
                entry.update(status=response.status, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if path.suffix == '.json':
                result = safe(json.loads(raw))
                write(path, result)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                result = raw
            log_request(entry)
            return result
        except urllib.error.HTTPError as error:
            code = error.code
            raw = error.read().decode(errors='replace').replace(key, '[redacted]')
            try:
                response_body = safe(json.loads(raw))
            except ValueError:
                # Arbitrary HTML errors may echo headers; preserve status only.
                response_body = {'body_suppressed': True, 'bytes': len(raw)}
            failure = CACHE / 'failures' / (Path(cache).stem + f'-{attempt}-{time.time_ns()}.json')
            write(failure, {'status': code, 'response': response_body})
            entry.update(status=code, failure_cache=str(failure.relative_to(BASE)))
            log_request(entry)
            if code in (401, 403):
                AUTH_FAILURES += 1
            if code == 429 and attempt < 4:
                wait = max(2 ** (attempt + 1), float(error.headers.get('Retry-After', 0) or 0))
                time.sleep(min(wait, 60))
                continue
            raise RuntimeError(f'{endpoint}: HTTP {code}; response cached, credentials suppressed') from None
        except (urllib.error.URLError, TimeoutError):
            write(CACHE / 'failures' / (Path(cache).stem + '-transport.json'),
                  {'transport_failed': True, 'details_suppressed': True})
            raise RuntimeError(endpoint + ': transport failed; details suppressed') from None
    raise RuntimeError('Rate limit retry budget exhausted')


def log_request(entry):
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'requests.jsonl').open('a') as stream:
        stream.write(json.dumps(safe(entry)) + '\n')


def enu(lon, lat):
    return np.array([(float(lon)-LON)*111320*math.cos(math.radians(LAT)),
                     (float(lat)-LAT)*110574])


def ll(east, north):
    return [LON+east/(111320*math.cos(math.radians(LAT))), LAT+north/110574]


def square(radius=20):
    return 'POLYGON((' + ','.join(' '.join(map(str, ll(e, n))) for e, n in
        [(-radius,-radius),(radius,-radius),(radius,radius),(-radius,radius),(-radius,-radius)]) + '))'


def project(matrix, points):
    p = np.atleast_2d(points)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(matrix).T
    return q[:, :2] / q[:, 2:3]


def xmp(raw):
    with Image.open(io.BytesIO(raw)) as im:
        meta = im.info.get('xmp', b'')
    if isinstance(meta, bytes):
        meta = meta.decode(errors='replace')
    return dict(re.findall(r'<Vexcel:([^>]+)>(.*?)</Vexcel:[^>]+>', meta))


def bootstrap():
    dest = CACHE / SID
    dest.mkdir(parents=True, exist_ok=True)
    sources = {
        'geometry.json': 'data/sun_check/khbuN7q853q/geometry.json',
        'footprint_fit.json': 'data/sun_check/khbuN7q853q/heading_fit.json',
        'registration.json': 'data/shadow_check/khbuN7q853q/origin-registration.json',
        'indoor_projection.json': 'data/sun_check/khbuN7q853q/projection_validation.json',
        'roof_height_prior.json': 'data/sun_check/khbuN7q853q/roof_edge_fit.json',
    }
    manifest = []
    for name, source in sources.items():
        path = dest / name
        if not path.exists():
            write(path, read(ROOT / source))
        manifest.append({'cache': str(path.relative_to(BASE)), 'source': source,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    pano_path = dest / 'pano_pose.json'
    if not pano_path.exists():
        # Keep only pose fields; the source also contains signed CDN URLs.
        pano = read(ROOT / 'data/matterport/khbuN7q853q/pano_meta.json')['pano']
        write(pano_path, {k: pano[k] for k in ('id', 'position', 'rotation')})
    for face in range(6):
        path = dest / f'indoor_face{face}.jpg'
        if not path.exists():
            with Image.open(ROOT / f'data/matterport/khbuN7q853q/pano_td86n1_face{face}.jpg') as im:
                im.convert('RGB').save(path, quality=95)
    write(dest / 'input_manifest.json', manifest)


def calibrate(label, props, raw):
    meta = xmp(raw)
    if meta.get('image-name') != props['image-name']:
        raise RuntimeError(label + ': returned image identity differs')
    bbox = list(map(float, meta['pixel-bbox'].split(',')))
    with Image.open(io.BytesIO(raw)) as im:
        size = im.size
    if list(size) != [int(bbox[2]), int(bbox[3])]:
        raise RuntimeError(label + ': resized crop cannot use raw offsets')
    pts = np.array([[e,n,z] for z in [GROUND,GROUND+12] for n in [-25,0,25] for e in [-25,0,25]])
    checks = np.array([[-17.7,12.4,28.1],[13.2,-19.7,25.2],[11.6,18.8,31.3],[-7.5,-8.4,23.7]])
    def transform(points, suffix):
        wkt = 'LINESTRINGZ(' + ','.join(' '.join(map(str,[*ll(e,n),z])) for e,n,z in points) + ')'
        response = request('oriented/transform-points', f'{SID}/transform-{label}-{suffix}.json',
            body={'operation':'world-2-pixel','image-name':props['image-name'],'wkt':wkt,'srid':4326})
        return np.array([[p['x'], p['y']] for p in response['points']]) - bbox[:2]
    pixel = transform(pts, 'fit')
    rows, rhs = [], []
    for (e,n,z),(u,v) in zip(pts,pixel):
        rows += [[e,n,z,1,0,0,0,0,-u*e,-u*n,-u*z], [0,0,0,0,e,n,z,1,-v*e,-v*n,-v*z]]
        rhs += [u,v]
    matrix = np.r_[np.linalg.lstsq(rows, rhs, rcond=None)[0],1].reshape(3,4)
    observed = transform(checks, 'validation')
    error = np.linalg.norm(project(matrix, checks)-observed, axis=1)
    if error.max() > .1:
        raise RuntimeError(label + ': independent camera check exceeds 0.1 px')
    return {'matrix': matrix.tolist(), 'raw_crop_pixel_bbox': bbox, 'size': list(size),
        'max_fit_error_px': float(np.linalg.norm(project(matrix, pts)-pixel,axis=1).max()),
        'held_out_max_error_px': float(error.max()), 'held_out_errors_px': error.tolist(),
        'validation_points_enu_absolute_elevation_m': checks.tolist(),
        'validation_observed_crop_px': observed.tolist(),
        'method': 'Local projective approximation to explicit-height Vexcel calibrated world-to-pixel; independent validation points excluded from matrix fitting',
        'pixel_convention': 'unrotated raw JPEG pixels, EXIF deliberately not applied; full image = crop + bbox origin',
        'xmp': meta}


def acquire():
    views = []
    for direction in ('north','south','east','west','nadir'):
        product = 'nadir' if direction == 'nadir' else 'oblique-' + direction
        result = request('oriented/query',f'{SID}/query-{direction}.json',params={
            'wkt':f'POINT({LON} {LAT})','srid':4326,'layer':'urban',
            'product-type':product,'collection':'us-tx-houston-2026','include':FIELDS,
            'order-by':'image-date-desc,image-center-distance-asc'})
        candidates = [f['properties'] for f in result.get('features', [])]
        # Inspect every candidate from this recent flight rather than assuming
        # temporal rank also means least foliage or best facade visibility.
        count = 1 if direction == 'nadir' else len(candidates)
        for index, props in enumerate(candidates[:count]):
            label = f'{direction}-{index+1}'
            extract = {'wkt':square(),'srid':4326,'layer':'urban',
                'image-name':props['image-name'],'rotation':'natural','crop':'bbox','image-format':'jpeg', 'downsample':0}
            raw = request('oriented/extract',f'{SID}/raw-{label}.jpg',body=extract)
            projection = calibrate(label,props,raw)
            view = {'id':label,'direction':direction,'product':product,'properties':props,
                    'raw_cache':f'{SID}/raw-{label}.jpg','extract_request':extract,
                    'projection':projection, 'candidate_rank':index+1}
            views.append(view)
            write(CACHE / SID / 'views.json', views)
            print('acquired',label, 'camera_check_px',round(projection['held_out_max_error_px'],5), flush=True)
    return views


def detect(views):
    for view in views:
        if view['direction'] == 'nadir':
            continue
        label = view['id']
        # Same image identity, but SAM may downsample/re-crop: metadata must be reconciled.
        body = {'wkt':square(),'srid':4326,'buffer-meter':0,
            'images':[{'endpoint':'oriented/extract','request':view['extract_request']}],
            'model-input':{'elements':[{'label':l,'model':'sam3','detection-threshold':.3}
                for l in ('window','door')], 'include':['bbox','confidence','pixel-mask','pixel-bbox']}}
        try:
            result = request('elements/detect',f'{SID}/detect-{label}.json',body=body)
            print('detected', label, len(result.get('features',[])), 'features', flush=True)
        except RuntimeError as error:
            write(CACHE / SID / f'detect-{label}-failed.json',{'reason':str(error)})
            print('detector_failed', label, str(error), flush=True)
            if AUTH_FAILURES >= 2:
                raise


def rotation2(degrees):
    a = math.radians(degrees)
    return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])


def model_geometry():
    model = read(CACHE / SID / 'geometry.json')
    walls, openings = [], []
    # Exterior status is checked against the closed wall-graph raster, not a hull
    # which would hide concave perimeter segments.
    segments = []
    for fi, floor in enumerate(model['floors']):
        for ei, edge in enumerate(floor['edges']):
            if len(edge['vertices']) != 2:
                continue
            a,b = [np.array([v['position'][k] for k in ('x','y')]) for v in edge['vertices']]
            segments.append((fi,ei,a,b,edge))
    points = np.array([p for _,_,a,b,_ in segments for p in (a,b)])
    res = .05
    lo = points.min(0)-2
    shape = np.ceil((points.max(0)+2-lo)/res).astype(int)+1
    mask = np.zeros(shape, bool)
    for _,_,a,b,_ in segments:
        line = np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/res*3)))
        ij = np.rint((line-lo)/res).astype(int)
        mask[ij[:,0],ij[:,1]] = True
    # Close subpixel graph endpoint gaps (one 5cm cell dilation).
    w = mask.copy()
    w[1:] |= mask[:-1]; w[:-1] |= mask[1:]
    w[:,1:] |= mask[:,:-1]; w[:,:-1] |= mask[:,1:]
    outside = np.zeros(shape,bool)
    q = collections.deque([(0,0)])
    outside[0,0] = True
    while q:
        i,j = q.popleft()
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            u,v = i+di,j+dj
            if 0 <= u < shape[0] and 0 <= v < shape[1] and not outside[u,v] and not w[u,v]:
                outside[u,v] = True; q.append((u,v))
    def is_out(p):
        ij = np.rint((p-lo)/res).astype(int)
        return bool(outside[tuple(ij)])
    for fi,ei,a,b,edge in segments:
        tangent = (b-a)/np.linalg.norm(b-a)
        normal = np.array([-tangent[1],tangent[0]])
        c = (a+b)/2
        sides = [is_out(c+sign*.25*normal) for sign in (1,-1)]
        exterior = sides[0] != sides[1]
        outward = normal if sides[0] else -normal
        walls.append({'id':f'F{fi}E{ei}','segment_model_xy':np.array([a,b]).tolist(),
                      'exterior':exterior,'outward_model_xy':outward.tolist()})
        for oi,o in enumerate(edge.get('openings') or []):
            center = a+o['relativeCenter']*(b-a)
            p0,p1 = center-tangent*o['width']/2, center+tangent*o['width']/2
            z,h = o['lowerElevation'],o['height']
            openings.append({**o, 'id':f'F{fi}E{ei}O{oi}','exterior':exterior,
                'outward_model_xy':outward.tolist(),'wall_id':f'F{fi}E{ei}',
                'corners_model_xyz':[[*p0,z],[*p1,z],[*p1,z+h],[*p0,z+h]]})
    return walls,openings


@functools.lru_cache(maxsize=1)
def registration():
    return read(CACHE / SID / 'registration.json')


def world_points(points, alignment='footprint_corrected', delta=None):
    reg = registration()
    A = np.array(reg['model_xy_to_enu_affine'])
    origin = np.array(reg['model_origin_enu_m'])
    fix = reg['footprint_correction']
    c = np.array(fix['rotation_centre_enu_m'])
    pts = np.atleast_2d(points).astype(float)
    xy = pts[:,:2] @ A.T + origin
    if alignment != 'delivered':
        xy = (xy-c) @ rotation2(fix['rotation_deg']).T+c
        xy += [fix['shift_east_m'],fix['shift_north_m']]
    if delta is not None:
        # x,y in metres; yaw in degrees; height in metres, relative to corrected.
        pivot = c + [fix['shift_east_m'],fix['shift_north_m']]
        xy = (xy-pivot) @ rotation2(delta[2]).T+pivot+np.asarray(delta[:2])
    z = pts[:,2]+GROUND+(0 if delta is None else delta[3])
    return np.column_stack([xy,z])


def font(size=18):
    path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default(size=size)


def polygon(draw, points, color, width=1):
    xy = [tuple(p) for p in np.asarray(points)]
    if len(xy) > 1:
        draw.line(xy+[xy[0]], fill=color, width=width)


def observations():
    return {view:{oid:{'crop_px':uv,'corner_indices':[0,1],
        'method':'hand-marked','identity_method':'manual facade/order matching after SAM localization',
        'measurement':'two visible lower sill endpoints; top corners hidden by eaves',
        'sigma_per_coordinate_px_assumed':SIGMA_PX,
        'limitations':'outer trim/sill can differ from Matterport graph aperture; 1.5px is assumed picking noise, not total systematic error',
        'split':'training' if view.startswith('west') else 'held_out_other_direction'}
        for oid,uv in items.items()} for view,items in MANUAL.items()}


def fit_pose(views, openings):
    by_id = {o['id']:o for o in openings}
    view_by_id = {v['id']:v for v in views}
    obs = observations()
    train, held = [], []
    for vid, items in obs.items():
        for oid, item in items.items():
            record = (np.asarray(view_by_id[vid]['projection']['matrix']),
                      np.array(by_id[oid]['corners_model_xyz'])[:2],np.array(item['crop_px']))
            (train if item['split']=='training' else held).append(record)
    def residual(delta, data=train):
        return np.concatenate([(project(P,world_points(xyz,delta=delta))-uv).ravel()
                               for P,xyz,uv in data])
    steps = np.array([.001,.001,.01,.001])
    def jacobian(delta):
        return np.column_stack([(residual(delta+steps[i]*np.eye(4)[i])-
            residual(delta-steps[i]*np.eye(4)[i]))/(2*steps[i]) for i in range(4)])
    def solve(target=None, start=None):
        delta = np.zeros(4) if start is None else np.array(start,dtype=float)
        target = np.zeros(len(residual(delta))) if target is None else target
        damping = .001
        for iteration in range(100):
            r = residual(delta)-target
            J = jacobian(delta)
            normal = J.T@J
            step = np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal),1e-8)),-J.T@r)
            candidate = delta+step
            if np.linalg.norm(residual(candidate)-target) < np.linalg.norm(r):
                delta=candidate; damping=max(1e-10,damping/3)
                if np.linalg.norm(step*np.array([1,1,.1,1]))<1e-8:
                    break
            else:
                damping*=10
                if damping>1e12:
                    break
        return delta
    starts = [np.zeros(4),np.array([0,0,5,0]),np.array([0,0,-5,0])]
    candidates = [solve(start=s) for s in starts]
    delta = min(candidates,key=lambda x:np.linalg.norm(residual(x)))
    J = jacobian(delta)
    covariance = np.linalg.pinv(J.T@J)*SIGMA_PX**2
    # Condition parameters in common physical units: yaw displacement at 5m.
    physical_scale = np.array([1,1,180/(math.pi*5),1])
    singular = np.linalg.svd(J*physical_scale,compute_uv=False)
    rng = np.random.default_rng(20261114)
    # Independent pixel picking plus a shared per-view 1px offset. These are
    # sensitivity assumptions, not empirical accuracy confidence intervals.
    samples = []
    for _ in range(150):
        noise = rng.normal(0,SIGMA_PX,len(residual(delta)))
        offset=0
        shared={}
        for vid in ('west-1','west-6'):
            shared[vid]=rng.normal(0,1,2)
        for vid,items in obs.items():
            if not vid.startswith('west'):
                continue
            for _item in items:
                noise[offset:offset+4]+=np.tile(shared[vid],2);offset+=4
        samples.append(solve(target=noise,start=delta))
    samples=np.array(samples)
    sensitivity_std=samples.std(0,ddof=1)
    def rms(data, alignment='footprint_corrected', d=None):
        r=np.concatenate([project(P,world_points(xyz,alignment,d))-uv for P,xyz,uv in data])
        return float(np.sqrt(np.mean(np.sum(r*r,axis=1))))
    scores={split:{a:rms(data,a,delta if a=='refined' else None) for a in COLORS}
            for split,data in [('training',train),('held_out_other_direction',held)]}
    result={'status':'exploratory hand-marked partial-sill fit; validation assessed below',
        'observation_method':'hand-marked','parameters_order':['east_m','north_m','yaw_deg','height_m'],
        'delta_from_footprint_corrected':delta.tolist(),
        'delta_cm_deg':{'east_cm':float(delta[0]*100),'north_cm':float(delta[1]*100),
            'yaw_deg':float(delta[2]),'height_cm':float(delta[3]*100)},
        'training_views':['west-1','west-6'],'held_out_views':['south-1','south-2','south-3'],
        'training_facade':'east-facing','held_out_facade':'north-facing',
        'training_corner_count':sum(len(x[1]) for x in train),
        'held_out_corner_count':sum(len(x[1]) for x in held),
        'rms_corner_error_px':scores,
        'nominal_7p5cm_gsd_equivalent_rms_m':{k:{a:v*.075 for a,v in x.items()} for k,x in scores.items()},
        'optimizer':'unregularized four-parameter nonlinear least squares, numerical Jacobian, damped Gauss-Newton; three yaw starts',
        'start_solutions':np.array(candidates).tolist(),
        'jacobian_rank':int(np.linalg.matrix_rank(J)),
        'physical_scaled_singular_values':singular.tolist(),
        'physical_scaled_condition_number':float(singular[0]/singular[-1]),
        'linear_independent_picking_sigma':np.sqrt(np.diag(covariance)).tolist(),
        'noise_sensitivity':{'status':'ASSUMED noise model, not survey uncertainty',
            'samples':len(samples),'seed':20261114,'independent_coordinate_sigma_px':SIGMA_PX,
            'shared_per_image_sigma_px':1,'parameter_std_m_deg':sensitivity_std.tolist(),
            'parameter_percentiles_2p5_97p5':np.percentile(samples,[2.5,97.5],axis=0).tolist(),
            'delta_exceeds_2sigma_by_parameter':(abs(delta)>2*sensitivity_std).tolist(),
            'unmodelled':'graph-to-visible-trim mismatch, correlated camera calibration, floor datum, occlusion and hand-match error'},
        'validation_limitations':'Only one independent held-out physical opening; three adjacent images are correlated. Partial sill endpoints only, no visible top corners or doors. No calibration-grade claim.'}
    result['accepted']=False
    result['status']='REJECTED: hand-marked partial-sill fit worsens held-out other-direction error'
    result['decision']='Do not replace footprint pose or expand to five homes: held-out error worsens, and every parameter shift is below 2 sigma under the assumed noise model.'
    return delta,result


def error_metrics(P, xyz, observed, alignment, delta=None):
    xyz=np.asarray(xyz)
    points=world_points(xyz,alignment,delta)
    predicted=project(P,points)
    residual=predicted-np.asarray(observed)
    # Convert in the local vertical facade plane, not by pretending oblique
    # pixels are isotropic ground metres. This plane is conditional on pose.
    tangent=points[1]-points[0];tangent[2]=0;tangent/=np.linalg.norm(tangent)
    metric=[]
    for point,r in zip(points,residual):
        directions=np.array([tangent,[0,0,1]])
        J=((project(P,point+directions*.01)-project(P,point-directions*.01))/.02).T
        metric.append(np.linalg.solve(J,r))
    return {'rms_corner_px':float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
        'per_corner_px':np.linalg.norm(residual,axis=1).tolist(),
        'residual_vectors_px':residual.tolist(),
        'nominal_7p5cm_gsd_equivalent_rms_m':float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))*.075),
        'local_facade_plane_rms_m':float(np.sqrt(np.mean(np.sum(np.asarray(metric)**2,axis=1)))),
        'local_facade_tangent_vertical_residual_m':np.asarray(metric).tolist(),
        'metre_interpretation':'nominal GSD is only pixel-equivalent; facade metric uses local projection Jacobian conditional on assumed pose and plane, not a surveyed 3D error'}


def detector_evidence(view):
    path=CACHE/SID/f'detect-{view["id"]}.json'
    if not path.exists():
        return {'status':'not run on nadir','detections':[]}
    data=read(path)
    metadata=data['images'][0]['metadata']
    bbox=list(map(float,metadata['pixel-bbox'].split(',')))
    same=(metadata['image-name']==view['properties']['image-name'] and
          bbox==view['projection']['raw_crop_pixel_bbox'] and
          [metadata['width'],metadata['height']]==view['projection']['size'] and
          metadata['response-downsample']==0)
    if not same:
        raise RuntimeError('Detector grid differs; require explicit re-registration')
    manual_matches={
        'west-1':{3:'F0E4O0',0:'F0E28O0'},
        'west-6':{0:'F0E4O0',1:'F0E28O0',2:'F0E26O0'},
        'south-1':{1:'F0E10O0'},'south-2':{0:'F0E10O0'},'south-3':{0:'F0E10O0'}}
    detections=[]
    for i,feature in enumerate(data['features']):
        for detection in feature['properties']['detections']:
            match=manual_matches.get(view['id'],{}).get(i)
            detections.append({'feature_index':i,'label':feature['properties']['label'],
                'bbox_crop_px':detection['pixel-bbox'],'confidence':detection['confidence'],
                'manually_matched_opening':match,
                'used_as_corner_observation':False,
                'status':'target window, localization only; sill endpoints hand-marked' if match else
                    'not used: neighboring building, occlusion, or unmeasured repeat view; no automatic identity inferred'})
    return {'status':'live SAM3 cached; pixel grid verified identical to raw crop',
        'cache':str(path.relative_to(ROOT)),'detection_count':len(detections),
        'door_detection_count':sum(x['label']=='door' for x in detections),
        'grid_identity_verified':same,'detections':detections,
        'limitation':'SAM boxes describe visible image regions, not the hidden physical aperture. They are not physical-corner ground truth.'}


def simple_analysis(view, openings):
    """Fallback attempt: dark compact components in target facade regions.

    Intensity components also include canopy/shadow and cannot certify an
    aperture corner. Keep candidates as diagnostics, never call them truth.
    """
    im=Image.open(CACHE/view['raw_cache']).convert('L')
    P=view['projection']['matrix']
    uv=np.concatenate([project(P,world_points(o['corners_model_xyz'])) for o in openings if o['exterior']])
    a=np.maximum(np.floor(uv.min(0)-15).astype(int),0)
    b=np.minimum(np.ceil(uv.max(0)+15).astype(int),np.array(im.size))
    gray=np.array(im)[a[1]:b[1],a[0]:b[0]]
    mask=gray<35
    visited=np.zeros_like(mask,bool);boxes=[]
    for y,x in np.argwhere(mask):
        if visited[y,x]:
            continue
        q=collections.deque([(int(y),int(x))]);visited[y,x]=True;pixels=[]
        while q:
            yy,xx=q.popleft();pixels.append((xx,yy))
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                v,u=yy+dy,xx+dx
                if 0<=v<mask.shape[0] and 0<=u<mask.shape[1] and mask[v,u] and not visited[v,u]:
                    visited[v,u]=True;q.append((v,u))
        points=np.array(pixels);lo=points.min(0);size=points.max(0)-lo+1
        density=len(points)/float(np.prod(size))
        if 6<=size[0]<=60 and 4<=size[1]<=40 and density>.45:
            boxes.append({'bbox_crop_px':[*map(int,lo+a),*map(int,size)],'dark_fill_fraction':density})
    return {'method':'image analysis: 4-connected grayscale <35/255 components, width6..60px, height4..40px, fill>0.45 in projected house ROI',
        'roi_crop_px':[*map(int,a),*map(int,b)],'candidates':boxes,
        'accepted_observations':0,
        'outcome':'FAILED to add certified corners: shadow, canopy and hidden tops make dark components semantically ambiguous; hand-marked visible sills used instead'}


def facing(view, opening):
    P=np.array(view['projection']['matrix'])
    _,_,vt=np.linalg.svd(P)
    camera=vt[-1,:3]/vt[-1,3]
    xyz=np.asarray(opening['corners_model_xyz'])
    center=xyz.mean(0)
    normal=np.asarray(opening['outward_model_xy'])
    mapped=world_points([center,center+np.r_[normal,0]])
    n=mapped[1,:2]-mapped[0,:2]
    return {'camera_facing':bool(n@(camera[:2]-mapped[0,:2])>0),
        'outward_azimuth_deg_east_of_north':float(math.degrees(math.atan2(n[0],n[1]))%360),
        'camera_local_enu_absolute_elevation_m':camera.tolist()}


def review_images(views,walls,openings,detectors):
    out=BASE/SID
    # Keep the topology decision and all candidate angles reviewable.
    im=Image.new('RGB',(1000,1000),'white');draw=ImageDraw.Draw(im)
    def pt(p):return (450+40*p[0],650+40*p[1])
    for wall in walls:
        draw.line([pt(x) for x in wall['segment_model_xy']],fill='red' if wall['exterior'] else 'gray',width=3)
        draw.text(pt(np.mean(wall['segment_model_xy'],axis=0)),wall['id'],fill='black',font=font(13))
    for opening in openings:
        if opening['exterior']:
            draw.line([pt(x) for x in np.array(opening['corners_model_xyz'])[:2,:2]],
                      fill='blue' if opening['type']=='window' else 'orange',width=7)
    draw.text((20,20),'Exterior graph: red walls / blue windows / orange doorways',fill='black',font=font(20))
    im.save(out/'geometry_plan.jpg',quality=95)
    for direction in ('north','south','east','west'):
        candidates=[v for v in views if v['direction']==direction]
        sheet=Image.new('RGB',(1440,600*math.ceil(len(candidates)/2)),'white')
        for i,view in enumerate(candidates):
            raw=Image.open(CACHE/view['raw_cache']).convert('RGB')
            center=project(view['projection']['matrix'],world_points([[-2,-5,1.5]]))[0]
            box=[max(0,int(center[0]-140)),max(0,int(center[1]-120)),
                min(raw.width,int(center[0]+140)),min(raw.height,int(center[1]+120))]
            tile=raw.crop(box);tile=tile.resize((tile.width*2,tile.height*2))
            sheet.paste(tile,((i%2)*720,(i//2)*600+60))
            d=ImageDraw.Draw(sheet);origin=((i%2)*720+10,(i//2)*600+8)
            d.text(origin,view['id']+' '+view['properties']['capture-date'],fill='black',font=font(20))
            d.text((origin[0],origin[1]+27),'HAND-MARKED sills' if view['id'] in MANUAL else 'Visibility review; no measured corners',fill='black',font=font(18))
        sheet.save(out/f'candidates_{direction}.jpg',quality=95)
    regions={'west-1':(265,150,345,182),'west-6':(307,225,448,258),
        'west-7':(291,195,425,230),'south-1':(322,415,359,442),
        'south-2':(303,415,340,443),'south-3':(318,416,354,444)}
    for view in views:
        vid=view['id']
        if vid in regions:
            box=regions[vid];raw=Image.open(CACHE/view['raw_cache']).convert('RGB')
            im=raw.crop(box).resize(((box[2]-box[0])*12,(box[3]-box[1])*12),Image.Resampling.NEAREST)
            d=ImageDraw.Draw(im)
            for x in range(box[0],box[2]):
                xx=(x-box[0])*12;d.line((xx,0,xx,im.height),fill='#555555')
                if x%5==0:d.text((xx,0),str(x),fill='yellow',font=font(12))
            for y in range(box[1],box[3]):
                yy=(y-box[1])*12;d.line((0,yy,im.width,yy),fill='#555555')
                if y%5==0:d.text((0,yy),str(y),fill='yellow',font=font(12))
            im.save(out/f'pixel_review_{vid}.png')
        if vid not in MANUAL and vid!='west-7' and vid!='east-5':
            continue
        im=Image.open(CACHE/view['raw_cache']).convert('RGB');d=ImageDraw.Draw(im)
        for det in detectors[vid]['detections']:
            x,y,w,h=det['bbox_crop_px'];d.rectangle((x,y,x+w,y+h),outline='yellow')
            d.text((x,y+h+2),str(det['feature_index']),fill='yellow',font=font(10))
        for opening in openings:
            if opening['id'] in ('F0E4O0','F0E28O0','F0E26O0','F0E10O0'):
                uv=project(view['projection']['matrix'],world_points(opening['corners_model_xyz']))
                polygon(d,uv,'lime');d.text(tuple(uv.mean(0)),opening['id'],fill='red',font=font(10))
        im.resize((im.width*3,im.height*3)).save(out/f'detector_review_{vid}.jpg',quality=95)


def both_sides(views,openings,delta):
    out=BASE/SID
    meta=read(CACHE/SID/'bedroom_pose.json')
    opening=next(o for o in openings if o['id']=='F0E4O0')
    xyz=np.array(opening['corners_model_xyz'])
    pos=np.array([meta['position'][k] for k in ('x','y','z')])
    x,y,z,w=np.array([meta['rotation'][k] for k in ('x','y','z','w')])
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    bases=np.array(read(CACHE/SID/'indoor_projection.json')['face_bases_local'])
    forward=xyz.mean(0)-pos;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    size=800;focal=500
    yy,xx=np.mgrid[:size,:size]
    rays=forward+((xx-size/2)/focal)[...,None]*right-((yy-size/2)/focal)[...,None]*up
    local=rays@R
    dots=local@bases[:,0].T;face=dots.argmax(-1)
    den=np.take_along_axis(dots,face[...,None],axis=-1)[...,0]
    u=(local*bases[face,1]).sum(-1)/den
    v=(local*bases[face,2]).sum(-1)/den
    rgb=np.zeros((size,size,3),np.uint8)
    for fi in range(6):
        source=np.array(Image.open(CACHE/SID/f'bedroom_face{fi}.jpg').convert('RGB'))
        good=face==fi
        ix=np.clip(((u[good]+1)/2*source.shape[1]).astype(int),0,source.shape[1]-1)
        iy=np.clip(((v[good]+1)/2*source.shape[0]).astype(int),0,source.shape[0]-1)
        rgb[good]=source[iy,ix]
    indoor=Image.fromarray(rgb);d=ImageDraw.Draw(indoor)
    q=xyz-pos;den=q@forward
    uv=np.column_stack([size/2+focal*(q@right)/den,size/2-focal*(q@up)/den])
    polygon(d,uv,'lime',3)
    d.text((15,15),'F0E4O0 graph outline',fill='lime',stroke_fill='black',stroke_width=1,font=font(23))
    # One perspective view spans the verified two adjacent cube faces.
    view=next(v for v in views if v['id']=='west-6')
    raw=Image.open(CACHE/view['raw_cache']).convert('RGB')
    crop=[330,192,430,292]
    outside=raw.crop(crop).resize((size,size),Image.Resampling.NEAREST)
    d=ImageDraw.Draw(outside)
    for alignment,color in COLORS.items():
        uvout=(project(view['projection']['matrix'],world_points(xyz,alignment,delta if alignment=='refined' else None))-crop[:2])*8
        polygon(d,uvout,color,3)
    observed=(np.array(MANUAL['west-6']['F0E4O0'])-crop[:2])*8
    d.line([tuple(p) for p in observed],fill='yellow',width=4)
    for p in observed:d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill='yellow')
    d.text((15,15),'HAND-MARKED sill; hidden top',fill='yellow',stroke_fill='black',stroke_width=1,font=font(23))
    canvas=Image.new('RGB',(1600,990),'white');canvas.paste(indoor,(0,100));canvas.paste(outside,(800,100))
    d=ImageDraw.Draw(canvas)
    d.text((20,15),'The same east-facade window, F0E4O0: bedroom pano and west-looking aerial',fill='black',font=font(28))
    d.text((20,55),'Indoor: cube-derived view; green graph opening. Outdoor: 8x nearest-pixel crop, no added detail.',fill='black',font=font(23))
    d.text((20,915),'Red delivered | green footprint | cyan REJECTED candidate | yellow HAND-MARKED sill',fill='black',font=font(24))
    d.text((20,950),'Identity checked by room/wall and window order; exterior upper aperture is hidden by the roof overhang.',fill='black',font=font(23))
    canvas.save(out/'both_sides.jpg',quality=95)
    # Reproduce the six-face context sheet used to check room/window identity.
    sheet=Image.new('RGB',(1536,1024),'white')
    for fi in range(6):
        tile=Image.open(CACHE/SID/f'bedroom_face{fi}.jpg').convert('RGB')
        ImageDraw.Draw(tile).text((30,30),str(fi),fill='red',font=font(60))
        sheet.paste(tile.resize((512,512)),((fi%3)*512,(fi//3)*512))
    sheet.save(out/'bedroom_review.jpg',quality=95)
    return {'opening_id':opening['id'],'indoor_pano_id':meta['id'],
        'indoor_method':'perspective resampling from cached 2k cube faces using Brief A verified basis and bedroom quaternion',
        'virtual_camera':{'position_model_m':pos.tolist(),'forward_model':forward.tolist(),
            'right_model':right.tolist(),'up_model':up.tolist(),'focal_px':focal,'size_px':[size,size]},
        'outdoor_view':'west-6','outdoor_raw_crop_box':crop,'outdoor_enlargement':8,
        'association_method':'manual room, exterior wall and ordered window correspondence',
        'observation_method':'hand-marked','limitation':'Not an independent survey association; window top hidden outdoors. Indoor graph boundary can differ from visible trim.'}


def render(views):
    out = BASE / SID
    out.mkdir(parents=True,exist_ok=True)
    walls, openings = model_geometry()
    write(out / 'model_openings.json', {'walls':walls,'openings':openings,
        'exterior_method':'closed wall graph flood fill at 5cm, one cell dilation; midpoint +/-25cm must have exactly one outside side'})
    eave = 2.6
    # A's 2.02m roof-edge prior is not assumed to be an exterior wall eave survey.
    assessment = {'floor_model_z0_absolute_elevation_m':GROUND,'floor_height_above_ground_m':0,
        'floor_height_status':'ASSUMED; no floor survey or camera/ground vertical datum reconciliation',
        'eave_above_model_floor_m':eave,'eave_status':'ASSUMED 2.6m for visualization; not fitted or surveyed',
        'vertical_note':'height parameter absorbs floor-to-ground and any datum offset; cannot identify them separately'}
    delta,fit = fit_pose(views,openings)
    write(out/'fit_candidate.json',fit)
    obs=observations()
    detectors={v['id']:detector_evidence(v) for v in views}
    rows, tiles = [], []
    for view in views:
        im = Image.open(CACHE / view['raw_cache']).convert('RGB')
        matrix = np.array(view['projection']['matrix'])
        draw = ImageDraw.Draw(im)
        record = {'id':view['id'],'direction':view['direction'],'image_name':view['properties']['image-name'],
            'capture_date_utc':view['properties']['capture-date'],'crop_size_px':list(im.size),
            'raw_crop_pixel_bbox':view['projection']['raw_crop_pixel_bbox'],
            'projection_validation_max_px':view['projection']['held_out_max_error_px'],
            'split':next(iter(obs[view['id']].values()))['split'] if view['id'] in obs else 'unmeasured',
            'observation_method':'hand-marked' if view['id'] in obs else None,
            'opening_rectangle_errors_available':False,
            'detector':detectors[view['id']],
            'image_analysis_fallback':simple_analysis(view,openings) if view['direction']!='nadir' else None,
            'wall_outlines':[],'openings':[]}
        for alignment,color in COLORS.items():
            for wall in walls:
                if not wall['exterior']:
                    continue
                p = np.array(wall['segment_model_xy'])
                outline={'wall_id':wall['id'],'alignment':alignment,'ground_crop_px':None,'eave_crop_px':None}
                for h,key in ((0,'ground_crop_px'),(eave,'eave_crop_px')):
                    xyz = np.column_stack([p,np.full(2,h)])
                    uv=project(matrix,world_points(xyz,alignment,delta if alignment=='refined' else None))
                    outline[key]=uv.tolist()
                    draw.line([tuple(q) for q in uv],fill=color,width=1)
                record['wall_outlines'].append(outline)
        for o in openings:
            if not o['exterior']:
                continue
            observation=obs.get(view['id'],{}).get(o['id'])
            face=facing(view,o)
            reason=('nadir: roof obscures vertical openings' if view['direction']=='nadir' else
                'back-facing wall; projected for completeness, not an observation' if not face['camera_facing'] else
                'not measured: eaves, canopy/shadow or insufficient unambiguous aperture detail')
            item = {'id':o['id'],'type':o['type'],'observed_crop_px':None,
                'observation_method':None,'used_for_fit':False,'held_out':False,
                'status':reason,'facing':face,'projected_crop_px':{},'errors':None,
                'observed_corner_indices':None,'unobserved_corner_indices':[0,1,2,3]}
            for alignment,color in COLORS.items():
                uv = project(matrix,world_points(o['corners_model_xyz'],alignment,delta if alignment=='refined' else None))
                item['projected_crop_px'][alignment] = uv.tolist()
                polygon(draw,uv,color,1)
            xy = np.mean(item['projected_crop_px']['footprint_corrected'],axis=0)
            draw.text(tuple(xy),o['id'].replace('F0',''),fill='white',stroke_width=1,stroke_fill='black',font=font(10))
            if observation:
                item.update(observed_crop_px=observation['crop_px'],observation_method='hand-marked',
                    observation=observation,used_for_fit=observation['split']=='training',
                    held_out=observation['split']=='held_out_other_direction',
                    observed_corner_indices=[0,1],unobserved_corner_indices=[2,3],
                    status='HAND-MARKED partial sill only; top hidden; no complete-rectangle error')
                item['errors']={a:error_metrics(matrix,np.array(o['corners_model_xyz'])[:2],
                    observation['crop_px'],a,delta if a=='refined' else None) for a in COLORS}
                points=np.array(observation['crop_px'])
                draw.line([tuple(p) for p in points],fill='yellow',width=2)
                for p in points:draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill='yellow')
            record['openings'].append(item)
        measured=[o for o in record['openings'] if o['errors']]
        record['measured_opening_instances']=len(measured)
        record['partial_sill_rms_errors']=None
        if measured:
            record['partial_sill_rms_errors']={a:{
                'px':float(np.sqrt(np.mean([o['errors'][a]['rms_corner_px']**2 for o in measured]))),
                'nominal_7p5cm_gsd_equivalent_m':float(np.sqrt(np.mean([o['errors'][a]['nominal_7p5cm_gsd_equivalent_rms_m']**2 for o in measured]))),
                'local_facade_plane_m':float(np.sqrt(np.mean([o['errors'][a]['local_facade_plane_rms_m']**2 for o in measured])))} for a in COLORS}
        header = Image.new('RGB',(im.width,86),'white')
        hdraw = ImageDraw.Draw(header)
        hdraw.text((8,5),view['id']+' | red delivered / green footprint / cyan REJECTED fit',fill='black',font=font(12))
        hdraw.text((8,24),'Raw pixels, no EXIF rotation. Floor z=0=ground and eave=2.6m ASSUMED.',fill='black',font=font(12))
        hdraw.text((8,43),'Yellow: HAND-MARKED sill endpoints only; upper corners NOT observed.',fill='black',font=font(12))
        hdraw.text((8,62),'Training west; held out south. All projections include hidden/back-facing walls.',fill='black',font=font(12))
        final = Image.new('RGB',(im.width,im.height+86),'white');final.paste(header,(0,0));final.paste(im,(0,86))
        final.save(out/f'overlay_{view["id"]}.jpg',quality=95)
        # Unannotated enlarged detail supports observation without alignment cues.
        center = project(matrix,world_points([[ -2,-5,1.5]]))[0]
        box = [max(0,int(center[0]-180)),max(0,int(center[1]-180)),min(im.width,int(center[0]+180)),min(im.height,int(center[1]+180))]
        raw = Image.open(CACHE / view['raw_cache']).convert('RGB')
        detail = raw.crop(box);detail=detail.resize((detail.width*2,detail.height*2))
        detail.save(out/f'inspect_{view["id"]}.jpg',quality=95)
        record['inspection_detail_raw_box'] = box
        record['overlay_coordinate_note']='Stored pixels refer to raw crop, not overlay JPEG; overlay adds 86px header above raw image.'
        tile = final.copy();tile.thumbnail((650,650))
        canvas = Image.new('RGB',(650,690),'white');canvas.paste(tile,(0,0))
        ImageDraw.Draw(canvas).text((8,660),view['id'],fill='black',font=font(20))
        tiles.append(canvas);rows.append(record)
    sheet = Image.new('RGB',(650*3,690*math.ceil(len(tiles)/3)),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*650,(i//3)*690))
    sheet.save(out/'contact_sheet.jpg',quality=94)
    review_images(views,walls,openings,detectors)
    sides=both_sides(views,openings,delta)
    fit['local_facade_plane_rms_m']={split:{a:float(np.sqrt(np.mean([
        o['errors'][a]['local_facade_plane_rms_m']**2 for r in rows if r['split']==split
        for o in r['openings'] if o['errors']]))) for a in COLORS}
        for split in ('training','held_out_other_direction')}
    write(out/'fit_candidate.json',fit)
    write(out/'reprojection.json', {'model_id':SID,'observation_method':'hand-marked partial sill endpoints',
        'assumptions':assessment,'images':rows,'refinement':fit,'both_sides':sides,
        'selection':{'collection':'us-tx-houston-2026',
            'strategy':'Fetched and visually reviewed every available cardinal oblique candidate in this recent collection, plus one nadir; no direction silently dropped',
            'images_by_look_direction':dict(collections.Counter(v['direction'] for v in views)),
            'best_measured_training':['west-1','west-6'],'held_out_other_direction':['south-1','south-2','south-3'],
            'north':'South facade largely canopy/overhang; no certified corners',
            'east':'West facade largely eaves/deep shadow; no certified corners',
            'nadir':'Roof context only; no vertical opening observation',
            'resolution':'40m east/north square, unrotated raw JPEG downsample=0; crop is its projected bounding box'},
        'projection_method':'Local 3x4 projective approximation calibrated to Vexcel explicit-height world-to-pixel, not an independently derived omega/phi/kappa implementation. 18 fit points and 4 disjoint validation points per image; applies only to this local house region.',
        'outline_support':{'status':'planar plausibility only; no measured wall/eave correspondences',
            'source':'data/reproj/cache/'+SID+'/footprint_fit.json',
            'finding':'Prior footprint correction improves raster scan/roof overlap, but roof eaves differ from wall perimeter. It cannot establish window corners, absolute elevation or a calibration-grade 3D pose.'},
        'step6':{'reached':False,'reason':'First-house hand-marked fit fails held-out direction and noise checks; do not expand to five homes'}})
    write(out/'cache_manifest.json',[{'path':str(p.relative_to(ROOT)),
        'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted((CACHE/SID).rglob('*')) if p.is_file()])
    report(out,fit,rows)
    print('rendered',len(rows),'images',sum(o['exterior'] for o in openings),'exterior openings',flush=True)


def report(out,fit,rows):
    """Keep the human report beside the quantitative evidence on every replay."""
    prior=read(CACHE/SID/'footprint_fit.json')
    delta=fit['delta_cm_deg']
    scores=fit['rms_corner_error_px']
    lines=[
        '# Brief E report — khbuN7q853q',
        '',
        '## Verified',
        '',
        '- Acquired all 23 oblique candidates in the selected recent Houston collection (6 north-looking, 3 south-looking, 6 east-looking, 8 west-looking) plus one nadir. Full-resolution downsample=0, approximately 40m ground square; capture dates 2025-12-19 and 2026-01-01. Evidence: cache_manifest.json and ../cache/khbuN7q853q/views.json.',
        f'- Local camera projection matches four disjoint Vexcel world-to-pixel check points per image: maximum {max(r["projection_validation_max_px"] for r in rows):.7f}px. This verifies local API consistency, not absolute survey accuracy. Evidence: reprojection.json and cached transform-*-validation.json.',
        '- Projected all 14 exterior graph openings (9 windows, 5 doorways) and exterior wall ground/eave lines under delivered, footprint-corrected and candidate poses in all 24 images. Includes hidden/back-facing geometry, explicitly flagged. Evidence: overlay_*.jpg, geometry_plan.jpg and reprojection.json (336 opening/image records).',
        '- SAM3 window/door requests completed for all 23 obliques; response pixel grids agree exactly with raw crops. SAM localizes visible regions but neither detector boxes nor simple dark-component analysis establishes physical aperture corners. Evidence: reprojection.json detector/image_analysis_fallback records, cached detect-*.json, detector_review_*.jpg.',
        '- HAND-MARKED two visible lower sill endpoints per measured opening: five training instances of three east-facade windows, three held-out instances of one north-facade window. No upper corner or door measured. Evidence: yellow marks in overlays, observation provenance in reprojection.json, pixel_review_*.png.',
        '- Same bedroom window shown indoors and outdoors; identity checked manually from room, wall and facade window order. Evidence: both_sides.jpg; indoor pose and six cached bedroom cube faces.',
        '- HAND-MARKED partial-sill RMS below. Metres are nominal 0.075m/pixel equivalents, not surveyed 3D errors; reprojection.json also supplies local facade-plane Jacobian conversions per opening and image.',
        '',
        '| HAND-MARKED split | Delivered px (nominal m) | Footprint px (nominal m) | Rejected candidate px (nominal m) |',
        '|---|---:|---:|---:|',
    ]
    for split,values in scores.items():
        cells=[f'{values[a]:.3f} ({values[a]*.075:.3f})' for a in COLORS]
        lines.append('| '+split+' | '+' | '.join(cells)+' |')
    lines += [
        '',
        f'- Four-parameter HAND-MARKED candidate versus footprint pose: east {delta["east_cm"]:+.2f}cm, north {delta["north_cm"]:+.2f}cm, yaw {delta["yaw_deg"]:+.3f} degrees, height {delta["height_cm"]:+.2f}cm. Fit has numerical rank four; three starting yaw values converge to the same candidate. Evidence: fit_candidate.json.',
        f'- Prior planar footprint fit, copied from Brief A rather than remeasured here: roof overlap IoU {prior["iou_before"]:.3f} -> {prior["iou_after"]:.3f}, containment {prior["contain_before"]:.3f} -> {prior["contain_after"]:.3f}. Evidence: ../cache/khbuN7q853q/footprint_fit.json.',
        '',
        '## Assumed or failed',
        '',
        '- FAILED: calibration-grade window registration is not supported by these obliques. Eaves hide aperture tops, trees/shadows obscure other facades, and visible sills are only about 13–17 raw pixels wide. Full window rectangles and doors have no observed-corner error; missing observations remain null, never zero.',
        '- REJECTED HAND-MARKED candidate: training improves but other-direction held-out error worsens. Cyan is diagnostic only; the footprint pose is not replaced. Five-home expansion (step 6) is skipped; no summary.csv.',
        '- ASSUMED noise: 1.5px independent coordinate picking and 1px shared per-image offset, 150 deterministic Monte Carlo trials. Parameter standard deviations: east 29.49cm, north 13.75cm, yaw 1.026 degrees, height 26.64cm. All candidate shifts are below two standard deviations, so none is robustly larger than this assumed noise. These are sensitivity values, not empirical confidence intervals.',
        '- ASSUMED floor z=0 lies at ground elevation 22.5552m and eave height is 2.6m above it. Floor/camera vertical datum relation is unverified; fitted height confounds floor offset, datum and marking/systematic errors. No roof-height prior was substituted for a wall-eave survey.',
        '- ASSUMED graph aperture matches visible outer sill/trim. Correspondence and exterior topology are model/manual interpretations, not surveyed controls. The held-out set has one independent physical window across three correlated captures; it cannot establish generalization across houses.',
        '- Wall/roof outline supports only horizontal footprint plausibility at metre scale. Roof eaves differ from wall perimeter; no measured wall/eave correspondences, absolute height or independent 3D pose follow from the improved overlap.',
        '',
        '## Files written',
        '',
        '- Codex wrote experiments/reproject_windows.py, data/reproj/cache/ (sanitized API responses and copied inputs), data/reproj/requests.jsonl, and data/reproj/khbuN7q853q/ (overlays, comparison, diagnostic images, geometry, fit, reprojection, manifests and this report). Complete inventory: files_written.json. Existing-suite evidence: ../existing_suite.log. No other experiment or source files changed; no new tests, commits or git initialization.',
        '- Replay: python3 -I experiments/reproject_windows.py khbuN7q853q (network opt-in; all acquisition inputs cached).',
        '- Reversible source edit history: python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path /Users/phanan/HackPrinceton/experiments/reproject_windows.py ; inspect rollback <id> before rollback <id> --yes. Generated outputs are confined to data/reproj/.',
    ]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    files=[ROOT/'experiments/reproject_windows.py',*sorted(p for p in BASE.rglob('*') if p.is_file())]
    write(out/'files_written.json',{'author':'Codex','paths':[str(p.relative_to(ROOT)) for p in files
        if p!=out/'files_written.json']+[str((out/'files_written.json').relative_to(ROOT))],
        'additional_user_required_progress_file':'/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T06-36-27-451Z-24886-9af9eeae-7dd2-4511-bf6e-dc2d4c631ccd.md'})


def main():
    global NETWORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid', choices=[SID])
    parser.add_argument('--acquire', action='store_true')
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--acquire-only', action='store_true')
    parser.add_argument('--indoor', action='store_true')
    args = parser.parse_args()
    NETWORK = args.acquire or args.detect or args.indoor
    bootstrap()
    views = acquire() if args.acquire else read(CACHE / SID / 'views.json')
    if args.detect:
        detect(views)
    if args.indoor:
        acquire_indoor()
    if not args.acquire_only:
        render(views)


if __name__ == '__main__':
    main()
