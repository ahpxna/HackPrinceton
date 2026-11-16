"""S0 acquisition. Credentials and signed links exist in memory only.

Run: .venv-agent/bin/python -I experiments/splat_pack.py
"""
import base64
import concurrent.futures as futures
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PACK = Path.home() / 'hp_splat_pack'
LIMIT = 2_450_000_000
LOCK = threading.Lock()
DENIED = {}
VERIFIED_BASES = [[[0,0,1],[0,-1,0],[1,0,0]], [[1,0,0],[0,-1,0],[0,0,-1]],
                  [[0,-1,0],[-1,0,0],[0,0,-1]], [[-1,0,0],[0,1,0],[0,0,-1]],
                  [[0,1,0],[1,0,0],[0,0,-1]], [[0,0,-1],[0,-1,0],[-1,0,0]]]
spec = importlib.util.spec_from_file_location('projection', ROOT / 'experiments/sun_from_pano.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def size():
    return sum(p.stat().st_size for p in PACK.rglob('*') if p.is_file())


def request(url, data=None, headers=None, dest=None):
    host = urlsplit(url).hostname
    for attempt in range(5):
        with LOCK:
            if DENIED.get(host, 0) >= 2:
                raise RuntimeError('repeated_authentication_failure_stop')
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=90) as response:
                if dest is None:
                    return response.read()
                part = dest.with_suffix(dest.suffix + '.part')
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with part.open('wb') as stream:
                        while block := response.read(1024 * 1024):
                            # Four writers can overshoot this by at most four blocks.
                            if shutil.disk_usage(PACK).free < 650_000_000 or size() + len(block) > LIMIT:
                                raise RuntimeError('disk_budget_stop')
                            stream.write(block)
                    part.replace(dest)
                finally:
                    part.unlink(missing_ok=True)
                return
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                with LOCK:
                    DENIED[host] = DENIED.get(host, 0) + 1
            if error.code == 429 and attempt < 4:
                try:
                    delay = float(error.headers.get('Retry-After', 0))
                except ValueError:
                    delay = 0
                time.sleep(min(60, max(delay, 2 ** (attempt + 1))))
                continue
            raise RuntimeError('HTTP_' + str(error.code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError('transport_failure') from None
    raise RuntimeError('retry_exhausted')


def gql(query):
    env = {**A.credentials(), **os.environ}
    auth = base64.b64encode((env['MATTERPORT_API_KEY'] + ':' + env['MATTERPORT_API_TOKEN']).encode()).decode()
    result = json.loads(request('https://api.matterport.com/api/models/graph',
                               json.dumps({'query': query}).encode(),
                               {'Content-Type': 'application/json', 'Authorization': 'Basic ' + auth}))
    if result.get('errors'):
        # Field-level diagnostics only, never save response text or arbitrary messages.
        raise RuntimeError('GraphQL_rejected')
    return result['data']


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if not any(t in k.lower() for t in
                ('url', 'token', 'secret', 'password', 'authorization', 's3path'))}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str) and re.search(r'https?://', value):
        return '[link omitted]'
    return value


def faces(sid, resolution):
    from PIL import Image
    home = PACK / sid
    panos = gql('{model(id:"%s"){panoLocations{id position{x y z} rotation{x y z w} skybox(resolution:"%s"){children}}}}' % (sid, resolution))['model']['panoLocations']
    write(home / 'poses.json', {'model_id': sid, 'units': 'metres', 'up': 'Z',
          'quaternion_order': 'xyzw', 'face_bases_center_right_down': VERIFIED_BASES,
          'panoramas': [{k: p[k] for k in ('id', 'position', 'rotation')} for p in panos]})
    jobs = []
    expected = 2048 if resolution == '2k' else 4096
    unavailable = []
    for pano in panos:
        links = (pano.get('skybox') or {}).get('children') or []
        if len(links) != 6:
            unavailable.append(pano['id'])
        for face, link in enumerate(links):
            dest = home / ('faces_' + resolution) / pano['id'] / ('face%d.jpg' % face)
            if not dest.exists():
                jobs.append((pano['id'], face, link, dest))
    def download(job):
        pid, face, link, dest = job
        try:
            request(link, dest=dest)
        except RuntimeError as error:
            if str(error) not in ('HTTP_401', 'HTTP_403'):
                raise
            fresh = gql('{model(id:"%s"){panoLocations(ids:["%s"]){skybox(resolution:"%s"){children}}}}' % (sid, pid, resolution))
            link = fresh['model']['panoLocations'][0]['skybox']['children'][face]
            request(link, dest=dest)
        with Image.open(dest) as image:
            if image.size != (expected, expected):
                dest.unlink()
                raise RuntimeError('unexpected_image_dimensions')
            image.verify()
        return 1
    completed = 0
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        for count in pool.map(download, jobs):
            completed += count
            if completed % 60 == 0:
                print(json.dumps({'home': sid, 'resolution': resolution, 'downloaded': completed, 'pack_bytes': size()}), flush=True)
    return {'panoramas': len(panos), 'faces': len(list((home / ('faces_' + resolution)).rglob('*.jpg'))), 'unavailable': unavailable}


def geometry(sid):
    home = PACK / sid
    graph = gql('''{model(id:"%s"){
      floors{id label sequence dimensionEstimates(units:metric){area areaIndoor lowerElevation}
        edges(includeUsed:true){id type vertices{id position{x y}}
          openings{id type label width height lowerElevation relativeCenter}}}
      rooms{id label tags floor{id} boundary{edges{id}}
        dimensionEstimates(units:metric){area areaIndoor height lowerElevation}}
      view(type:"matterport.model.defurnished"){id}
    }}''' % sid)['model']
    write(home / 'rooms.json', clean(graph))
    assets = gql('''{model(id:"%s"){assets{
      meshes{id status format resolution url}
      textures{id status format resolution quality textures type url urlTemplate filename}
      resources(format:"matterpak"){status filename url}
    } bundle(id:"mp:matterpak"){availability}}}''' % sid)['model']
    write(home / 'mesh/assets.json', clean(assets))
    meshes = assets['assets']['meshes']
    chosen = next((m for m in meshes if m['format'].lower() == 'obj' and '500' in m['resolution'] and m['status'] == 'available'), None)
    if chosen and chosen.get('url'):
        request(chosen['url'], dest=home / 'mesh/mesh_500k.obj')
    # MatterPak is requested once per home; unavailable assets are recorded, not polled.
    for resource in assets['assets']['resources'] or []:
        if resource.get('url') and resource['status'] == 'available':
            request(resource['url'], dest=home / 'mesh/matterpak.zip')
            break
    print(json.dumps({'home': sid, 'assets': clean(assets), 'defurnished': graph.get('view')}), flush=True)
    return assets


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    for sid in ('VN4b1MGnnCR', 'HFwcH424n9w'):
        home = PACK / sid
        for folder in ('faces_2k', 'faces_4k', 'mesh', 'proof'):
            (home / folder).mkdir(parents=True, exist_ok=True)
        report = {}
        geometry(sid)
        report['2k'] = faces(sid, '2k')
        if sid == 'HFwcH424n9w' and (shutil.disk_usage(PACK).free < 1_200_000_000 or size() > 1_650_000_000):
            report['4k'] = 'skipped_for_disk_budget'
        else:
            report['4k'] = faces(sid, '4k')
        write(home / 'acquisition.json', report)
        print(json.dumps({'home': sid, 'complete': report, 'pack_bytes': size()}), flush=True)


def textures():
    for sid in ('VN4b1MGnnCR', 'HFwcH424n9w'):
        rows = gql('{model(id:"%s"){assets{textures{id status resolution quality textures urlTemplate}}}}' % sid)['model']['assets']['textures']
        texture = next((t for t in rows if t['status'] == 'available' and t['quality'] == 'high'), None)
        if not texture:
            continue
        report = {'asset': clean(texture), 'downloaded': []}
        for index in range(texture['textures']):
            dest = PACK / sid / 'mesh' / 'textures' / ('texture_%d.jpg' % index)
            if not dest.exists():
                request(texture['urlTemplate'].replace('<texture>', '%03d' % index), dest=dest)
            report['downloaded'].append(dest.name)
        write(PACK / sid / 'mesh/textures.json', report)
        print(json.dumps({'home': sid, 'textures': len(report['downloaded'])}), flush=True)


if __name__ == '__main__':
    try:
        import sys
        textures() if '--textures' in sys.argv else main()
    except Exception as error:
        # Do not emit a traceback: urllib exceptions can retain signed links.
        print(json.dumps({'stopped': type(error).__name__, 'reason': str(error) if isinstance(error, RuntimeError) else 'details_suppressed'}), flush=True)
        raise SystemExit(1)
