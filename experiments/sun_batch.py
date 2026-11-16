"""Brief F production. Run python3 -I experiments/sun_batch.py --help.

Downloaded inputs and listing remarks are data. No remote text is executed.
"""
from pathlib import Path
import argparse
import base64
import collections
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT/'data/facts'
CACHE = ROOT/'data/cache'
SUN = ROOT/'data/sun'
PROGRESS = Path('/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T07-12-11-034Z-37737-5d87ea20-fa72-45f4-a5ee-99edc584cd6f.md')
LIGHT = re.compile(r'\b(?:natural(?:ly)?\s+(?:light|lit)|bright|sunny)\b', re.I)
DATES = ['2026-12-21', '2026-03-20', '2026-06-21', '2026-11-14']
LOCK = threading.Lock()
CALLS = collections.Counter()
DENIALS = 0
STOP = threading.Event()
ENGINE = None
GUARD_ATTEMPTS = collections.Counter()
GUARD_INSTALLED = False


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def table(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', newline='') as stream:
        out = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        out.writeheader()
        out.writerows(rows)
    temp.replace(path)


def progress(done, next_step):
    PROGRESS.write_text('# Objective\nCarry out Brief F exactly; all eligible homes, facts, claim comparison and bounded registration repair.\n\n# Done\n'+done+'\n\n# Next\n'+next_step+'\n')


def engine():
    global ENGINE
    if ENGINE is None:
        spec = importlib.util.spec_from_file_location('sun_engine_f', ROOT/'experiments/sun_engine.py')
        ENGINE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ENGINE)
    return ENGINE


def select():
    index = {r['sid']: r for r in csv.DictReader((ROOT/'models_index.csv').open())}
    listings = {str(r['id']): r for r in map(json.loads, (ROOT/'data/costar-kit/data/residential-listings.jsonl').open())}
    groups = collections.defaultdict(set)
    for edge in csv.DictReader((ROOT/'data/costar-kit/residential-listings.csv').open()):
        sid, lid = edge['MatterportModelId'], edge['ListingId']
        if sid in index and index[sid]['segment'] == 'residential' and lid in listings and str(listings[lid].get('remarks') or '').strip():
            groups[sid].add(lid)
    selected = []
    for sid in sorted(groups):
        ids = sorted(groups[sid], key=lambda lid: (bool(listings[lid].get('isPrimaryForThisListingInstance')), listings[lid].get('listDate') or '', int(lid)), reverse=True)
        listing = listings[ids[0]]
        address = (listing.get('property') or {}).get('address') or {}
        r = index[sid]
        selected.append(dict(r, listing_id=ids[0], listing_instance_id=str(listing.get('residentialListingId') or ''),
            alternate_listing_ids=ids[1:], address=address.get('fullStreetAddress') or r['address'],
            city=address.get('city') or r['city'], state=address.get('state') or r['state'],
            light_claim=bool(LIGHT.search(listing['remarks'])), listing_latitude=address.get('latitude'),
            listing_longitude=address.get('longitude'), cohort='BriefF_all_remarks'))
    instances = collections.defaultdict(list)
    for lid,listing in listings.items():
        if str(listing.get('remarks') or '').strip():
            instances[str(listing.get('residentialListingId'))].append(lid)
    instance_only = [{'sid':sid,'listing_id':lid,'reason':'Instance-only match without authoritative CSV scan/listing edge; not selected'}
        for sid,row in sorted(index.items()) if row['segment']=='residential' and sid not in groups for lid in sorted(instances[row['homes_listing_id']])]
    table(FACTS/'instance_only_matches.csv',instance_only,['sid','listing_id','reason'])
    write(FACTS/'selection.json', {'selected':len(selected), 'listing_edges':sum(len(x['alternate_listing_ids'])+1 for x in selected),
        'join':'Authoritative residential-listings.csv ListingId -> JSONL id, MatterportModelId -> models_index.sid. homes_listing_id is an instance ID, not a listing ID.',
        'representative':'Primary remarked listing first, then latest listDate, then greatest numeric listing ID. All mapped remarked listing IDs retained.',
        'light_regex':LIGHT.pattern, 'homes':selected})
    return selected


def safe_reason(error):
    if isinstance(error, RuntimeError) and re.fullmatch(r'[a-zA-Z0-9_:-]{1,150}', str(error)):
        return str(error)
    return type(error).__name__


def sanitize(value):
    if isinstance(value, dict):
        return {k:sanitize(v) for k,v in value.items() if not any(s in k.lower() for s in ('token','secret','api-key','authorization'))}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and ('http://' in value or 'https://' in value):
        return '[URL omitted]'
    return value


def request(path, service, query, credentials):
    global DENIALS
    if path.exists():
        value = read(path)
        if '_error' in value:
            raise RuntimeError(value['_error'])
        return value
    if os.environ.get('SUN_OFFLINE') == '1':
        raise RuntimeError('offline_cache_missing')
    if service == 'matterport':
        auth = base64.b64encode((credentials['MATTERPORT_API_KEY']+':'+credentials['MATTERPORT_API_TOKEN']).encode()).decode()
        req = urllib.request.Request('https://api.matterport.com/api/models/graph', json.dumps({'query':query}).encode(), {'Content-Type':'application/json', 'Authorization':'Basic '+auth})
    else:
        req = urllib.request.Request('https://api.vexcelgroup.com/v2/property/extract?'+urllib.parse.urlencode(query), headers={'X-API-KEY':credentials['VEXCEL_API_KEY']})
    reason = service+'_request_failed'
    for attempt in range(7):
        with LOCK:
            if STOP.is_set():
                raise RuntimeError('authentication_stop')
            CALLS[service] += 1
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                value = json.load(response)
            if value.get('errors'):
                raise RuntimeError(service+'_graphql_error')
            if value.get('error'):
                raise RuntimeError(service+'_api_error')
            value = sanitize(value)
            write(path, value)
            return value
        except urllib.error.HTTPError as error:
            reason = service+'_http_'+str(error.code)
            if error.code in (401,403):
                with LOCK:
                    DENIALS += 1
                    if DENIALS >= 2:
                        STOP.set()
            if error.code == 429 and attempt < 6:
                try: delay = min(120., max(2**(attempt+1), float(error.headers.get('Retry-After','0'))))
                except ValueError: delay = min(64,2**(attempt+1))
                time.sleep(delay)
                continue
            if error.code >= 500 and attempt < 2:
                time.sleep(2**(attempt+1))
                continue
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            reason = service+'_transport_error'
            if attempt < 2:
                time.sleep(2**(attempt+1))
                continue
            break
        except RuntimeError as error:
            reason = safe_reason(error)
            break
    write(path, {'_error':reason})
    raise RuntimeError(reason)


def audit_models():
    result = {}
    for path in sorted((ROOT/'data/audit/cache/mp').glob('*.json')):
        envelope = read(path)
        m = ((envelope.get('data') or {}).get('data') or {}).get('model')
        if m and m.get('id') and 'floors' in m and 'rooms' in m:
            result[m['id']] = (m, str(path.relative_to(ROOT)))
    return result


SUPPLEMENT = '''{model(id:"%s"){
 geocoordinates{latitude longitude p0:geoLocationOf(modelLocation:{x:0,y:0,z:0}){lat long}
 px:geoLocationOf(modelLocation:{x:20,y:0,z:0}){lat long} py:geoLocationOf(modelLocation:{x:0,y:20,z:0}){lat long}}
 rooms{id boundary{edges{type vertices{position{x y}}}} holes{edges{vertices{position{x y}}}}}
}}'''


def acquire(row, audits, credentials):
    sid = row['sid']
    folder = CACHE/sid
    record = {'sid':sid, 'downloaded':False, 'failure_reason':'', 'audit_reused':False}
    try:
        if (folder/'matterport.json').exists():
            model = request(folder/'matterport.json', 'matterport', '', credentials)['data']['model']
        elif sid in audits:
            source, source_path = audits[sid]
            supplement = request(folder/'matterport_supplement.json', 'matterport', SUPPLEMENT % sid, credentials)['data']['model']
            if not supplement:
                raise RuntimeError('matterport_model_missing')
            model = json.loads(json.dumps(source))
            if not supplement.get('geocoordinates'):
                raise RuntimeError('matterport_geocoordinates_missing')
            model['geocoordinates'] = supplement['geocoordinates']
            extra = {r['id']:r for r in supplement.get('rooms') or []}
            for room in model['rooms']:
                room.update(extra.get(room['id'], {}))
                if room.get('dimensionEstimates'):
                    room['dimensionEstimates']['units'] = 'metric'
            for floor in model['floors']:
                if floor.get('dimensionEstimates'):
                    floor['dimensionEstimates']['units'] = 'metric'
            write(folder/'matterport.json', {'data':{'model':model}})
            write(folder/'acquisition_provenance.json', {'audit_source':source_path, 'reused':['floor wall graph and openings','room labels, classifications, metric elevations and ceiling'], 'supplement':'geolocation basis, room boundaries and holes'})
            record['audit_reused'] = True
        else:
            model = request(folder/'matterport.json', 'matterport', engine().MODEL_Q % sid, credentials)['data']['model']
        if not model:
            raise RuntimeError('matterport_model_missing')
        geo = model.get('geocoordinates') or {}
        if not geo.get('latitude') or not geo.get('longitude'):
            raise RuntimeError('matterport_geocoordinates_missing')
        prop = request(folder/'vexcel_calculated_65m.json', 'vexcel', {'wkt':f"POINT({geo['longitude']} {geo['latitude']})", 'srid':4326, 'layer':'urban', 'buffer-meter':65, 'latest':'true', 'include':'precomputed,calculated', 'unit-system':'metric'}, credentials)
        record['downloaded'] = True
        record['audit_reused'] = record['audit_reused'] or (folder/'acquisition_provenance.json').exists()
        if not prop.get('features'):
            record['input_warning'] = 'vexcel_no_features'
    except Exception as error:
        record['failure_reason'] = safe_reason(error)
    write(folder/'batch_download.json', record)
    return record


def download(selected):
    started = time.monotonic()
    audits = audit_models()
    # Runtime-only credential parsing; neither keys nor signed URLs are persisted.
    credentials = {} if os.environ.get('SUN_OFFLINE') == '1' else engine().A.credentials()
    results = []
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(acquire, row, audits, credentials):row['sid'] for row in selected}
        for task in cf.as_completed(pending):
            results.append(task.result())
            summary = {'selected':len(selected), 'finished':len(results), 'downloaded':sum(r['downloaded'] for r in results), 'failures':dict(collections.Counter(r['failure_reason'] for r in results if r['failure_reason'])), 'api_calls':dict(CALLS), 'auth_denials':DENIALS, 'stopped_after_repeated_auth_denials':STOP.is_set(), 'max_requests_in_flight':4, 'elapsed_seconds':round(time.monotonic()-started,2)}
            write(FACTS/'download_progress.json', summary)
            if len(results) % 50 == 0:
                progress('Background download: '+json.dumps(summary)+'. Source: experiments/sun_batch.py; additive engine skip_floor_images option.', 'Finish acquisition, compute every home, generate tables and comparison, then bounded repair and offline replay.')
    credentials.clear()
    write(FACTS/'download_results.json', sorted(results,key=lambda r:r['sid']))
    progress('Download finished: '+json.dumps(summary)+'. Evidence: data/facts/download_results.json and download_progress.json.', 'Run engine without images; retain every failure in index and homes.csv.')
    return results


def compute_one(row):
    os.environ['SUN_OFFLINE'] = '1'
    if os.environ.get('SUN_REPLAY_GUARD') == '1':
        install_guard()
    sid = row['sid']
    try:
        downloaded = CACHE/sid/'batch_download.json'
        if downloaded.exists() and read(downloaded).get('failure_reason'):
            reason = read(downloaded)['failure_reason']
            if reason == 'KeyError' and (CACHE/sid/'matterport_supplement.json').exists() and not (read(CACHE/sid/'matterport_supplement.json').get('data',{}).get('model') or {}).get('geocoordinates'):
                reason = 'matterport_geocoordinates_missing'
            raise RuntimeError(reason)
        model = read(CACHE/sid/'matterport.json')['data']['model']
        if not any(len(e.get('vertices') or []) == 2 for f in model.get('floors') or [] for e in f.get('edges') or []):
            raise RuntimeError('matterport_no_wall_graph')
        result = engine().run_home(sid, skip_floor_images=True)
        record = engine().index_row(row, result)
        failure = SUN/sid/'failure.json'
        if failure.exists():
            archived = SUN/sid/'failure_before_success.json'
            if not archived.exists():failure.replace(archived)
            else:failure.unlink()
    except Exception as error:
        reason = 'geometry_nonfinite_result' if isinstance(error,ValueError) and str(error).startswith('Out of range float values are not JSON compliant') else safe_reason(error)
        record = {'sid':sid, 'address':row['address'], 'state':row['state'], 'cohort':row['cohort'], 'failure_reason':reason}
        frames = [{'file':Path(f.filename).name,'function':f.name,'line':f.lineno} for f in traceback.extract_tb(error.__traceback__)]
        write(SUN/sid/'failure.json',dict(record,exception_type=type(error).__name__,stages=frames))
    if os.environ.get('SUN_REPLAY_GUARD') == '1':
        record['_guard_attempts'] = dict(GUARD_ATTEMPTS)
        record['_engine_api_calls'] = dict(engine().API_CALLS)
    return record


def compute(selected, workers=4):
    started = time.monotonic()
    records = []
    fields = next(csv.reader((CACHE/'brief_f_baseline/index.csv').open()))
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(compute_one,row) for row in selected]
        for task in cf.as_completed(tasks):
            records.append(task.result())
            if len(records) % 20 == 0 or len(records) == len(selected):
                table(SUN/'index.csv', sorted(records,key=lambda r:r['sid']), fields)
                summary = {'selected':len(selected), 'finished':len(records), 'computed':sum(not r.get('failure_reason') for r in records), 'well_registered':sum(r.get('registration_poor') is False for r in records), 'failures':dict(collections.Counter(r['failure_reason'] for r in records if r.get('failure_reason'))), 'elapsed_seconds':round(time.monotonic()-started,2), 'floor_images_generated':False, 'network_enabled':False}
                write(FACTS/'compute_progress.json', summary)
                progress('Engine progress: '+json.dumps(summary)+'. Evidence: data/sun/index.csv and data/facts/compute_progress.json.', 'Finish full cohort; publish facts and light comparison, then repair and cache replay.')
    return records


def solar_split(sample, lat, lon):
    times = [dt.datetime.fromisoformat(t) for t in sample['local_times']]
    weights = np.asarray(sample['weights_minutes'])
    peak = times[int(np.argmax(sample['sun_apparent_elevation_deg']))]
    candidates = [peak+dt.timedelta(seconds=i) for i in range(-300,301)]
    noon = candidates[int(np.argmax(engine().A.solar(candidates,lat,lon)[2]))]
    before = np.array([max(0.,min(float(w),(noon-t).total_seconds()/60)) for t,w in zip(times,weights)])
    return weights, before, noon.isoformat()


def role_rooms(rooms):
    def text(r):
        return (r['label']+' '+' '.join(c['label'] for c in r['classifications'] if c.get('confidence',0) >= .5)).lower()
    result = {}
    for role, predicate in [('living_room',lambda s:'living' in s or 'family room' in s), ('primary_bedroom',lambda s:('primary' in s or 'master' in s) and ('bed' in s or 'suite' in s)), ('bedroom',lambda s:'bedroom' in s)]:
        candidates = [r for r in rooms if predicate(text(r))]
        result[role] = max(candidates,key=lambda r:r['area_m2']) if candidates else None
    return result


def nearest_building(window, room, sun, fit, model):
    e = engine()
    matrix,_,_ = e.geo_matrix(model)
    matrix,shift = e.alignment(fit,matrix)
    origin = np.array(window['center']) @ matrix.T+shift
    az = window['outward_azimuth_deg']
    angle = math.radians(az)
    forward = np.array([math.sin(angle),math.cos(angle)])
    sideways = np.array([math.cos(angle),-math.sin(angle)])
    normals = [forward,math.tan(math.radians(20))*forward+sideways,math.tan(math.radians(20))*forward-sideways]
    nearest = None
    for obstacle in sun['obstructions']:
        if obstacle['kind'] != 'neighbour':
            continue
        ring = np.asarray(obstacle['ring_model_xy_m']) @ matrix.T+shift
        # Clip each edge against the three half-planes defining the bearing wedge.
        for a,b in zip(ring,np.roll(ring,-1,axis=0)):
            edge = b-a
            low,high = 0.,1.
            for normal in normals:
                start,step = float((a-origin)@normal),float(edge@normal)
                if abs(step)<1e-12:
                    if start<0: high=-1.
                elif step>0:
                    low=max(low,-start/step)
                else:
                    high=min(high,-start/step)
            if low<=high:
                t = np.clip(np.dot(origin-a,edge)/max(np.dot(edge,edge),1e-12),low,high)
                distance = float(np.linalg.norm(a+t*edge-origin))
                if distance>60:continue
                nearest = distance if nearest is None else min(nearest,distance)
    return round(nearest,3) if nearest is not None else None


def parcel_attributes(prop, model):
    e = engine()
    geo = model['geocoordinates']
    point = (geo['longitude'],geo['latitude'])
    matching = []
    for feature in prop.get('features',[]):
        geom = feature.get('geometry') or {}
        coords = geom.get('coordinates') or []
        polygons = [coords] if geom.get('type') == 'Polygon' else coords if geom.get('type') == 'MultiPolygon' else []
        if any(poly and e.PolygonPath(poly[0]).contains_point(point) and not any(e.PolygonPath(h).contains_point(point) for h in poly[1:]) for poly in polygons):
            matching.append(feature.get('properties') or {})
    if not matching:
        return None,{},'No returned parcel polygon contains delivered scan point; attributes unknown'
    latest = max(matching,key=lambda p:(p.get('metadata.ortho-date-estimate') or '',p.get('metadata.collection') or ''))
    attrs = {k:v for k,v in latest.items() if 'tree' in k.lower() and not isinstance(v,(dict,list))}
    cover = latest.get('parcel-tree-cover.percent')
    return cover,attrs,'Latest returned parcel containing delivered scan geolocation; trees are a caution, not general engine geometry'


def facts(selected):
    records = {r['sid']:r for r in csv.DictReader((SUN/'index.csv').open())}
    homes, rooms, windows = [],[],[]
    for row in selected:
        sid = row['sid']
        record = records.get(sid,{})
        folder = CACHE/sid
        download_record = read(folder/'batch_download.json') if (folder/'batch_download.json').exists() else {}
        h = {k:row[k] for k in ('sid','listing_id','listing_instance_id','address','city','state','light_claim')}
        h['alternate_listing_ids'] = json.dumps(row['alternate_listing_ids'])
        h.update(downloaded=download_record.get('downloaded',False), computed=not bool(record.get('failure_reason')) and bool(record), failure_reason=record.get('failure_reason') or '', poor=None,
            registered_latitude=None,registered_longitude=None,registration_iou=None,registration_containment=None,registration_rotation_deg=None,registration_shift_m=None,
            rooms=None,exterior_windows=None,floor_area_m2=None,glazing_area_m2=None,glazing_over_floor_area=None,parcel_tree_cover_percent=None,parcel_tree_attributes_json=None,tree_caution=True,
            tree_attribute_source=None,main_living_room_id=None,primary_bedroom_room_id=None,largest_bedroom_room_id=None,dec21_main_living_sun_minutes=None,mar20_main_living_sun_minutes=None)
        if (folder/'registration.json').exists():
            fit = read(folder/'registration.json')
            h.update(poor=fit['poor'],registration_iou=fit['iou_after'],registration_containment=fit['contain_after'],registration_rotation_deg=fit['fix_rot_deg'],registration_shift_m=fit['fix_shift_m'])
            if (folder/'matterport.json').exists():
                model = read(folder/'matterport.json')['data']['model']
                matrix,lat,lon = engine().geo_matrix(model)
                aligned,shift = engine().alignment(fit,matrix)
                centre = np.array(fit['_plot']['centre'])
                fixed = centre+np.array(fit['_plot']['d'])
                h['registered_latitude'] = lat+fixed[1]/110574
                h['registered_longitude'] = lon+fixed[0]/(111320*math.cos(math.radians(lat)))
        if not h['computed']:
            homes.append(h)
            continue
        sun = read(SUN/sid/'sun.json')
        samples = read(SUN/sid/'sample_verdicts.json')
        fit = read(folder/'registration.json')
        model = read(folder/'matterport.json')['data']['model']
        _,lat,lon = engine().geo_matrix(model)
        splits = {date:solar_split(sample,lat,lon) for date,sample in samples.items()}
        role = role_rooms(sun['rooms'])
        living = role['living_room']
        primary = role['primary_bedroom']
        largest = role['bedroom'] if primary is None else None
        glazing = sum(w['width']*w['height'] for w in sun['windows'] if w['type']=='window')
        area = sum(r['area_m2'] for r in sun['rooms'])
        cover,attrs,source = parcel_attributes(read(folder/'vexcel_calculated_65m.json'),model)
        h.update(rooms=len(sun['rooms']),exterior_windows=sum(w['type']=='window' for w in sun['windows']),floor_area_m2=round(area,3),glazing_area_m2=round(glazing,4),glazing_over_floor_area=glazing/area if area else None,
            parcel_tree_cover_percent=cover,parcel_tree_attributes_json=json.dumps(attrs,sort_keys=True),tree_attribute_source=source,
            main_living_room_id=living['id'] if living else None,primary_bedroom_room_id=primary['id'] if primary else None,largest_bedroom_room_id=largest['id'] if largest else None)
        if living:
            h['dec21_main_living_sun_minutes'] = living['dates'][DATES[0]]['lit_minutes_ge_0_1m2']
            h['mar20_main_living_sun_minutes'] = living['dates'][DATES[1]]['lit_minutes_ge_0_1m2']
        homes.append(h)
        for room in sun['rooms']:
            assigned = [w for w in sun['windows'] if w['type']=='window' and w['room_id']==room['id']]
            glass = sum(w['width']*w['height'] for w in assigned)
            top = max(room['classifications'],key=lambda c:c.get('confidence',0)) if room['classifications'] else {}
            rr = {'sid':sid,'room_id':room['id'],'label':room['label'],'top_classification':top.get('label'),'classification_confidence':top.get('confidence'),
                'bedroom_role':'primary_bedroom' if primary and primary['id']==room['id'] else 'largest_bedroom' if largest and largest['id']==room['id'] else None,
                'area_m2':room['area_m2'],'exterior_windows':len(assigned),'glazing_area_m2':round(glass,4),'glazing_over_floor_area':glass/room['area_m2'], 'poor':sun['registration']['poor']}
            for date in DATES:
                p = date.replace('-','')
                saved = room['dates'][date]
                weights,before,noon = splits[date]
                mask = np.asarray(samples[date]['room_floor_area_m2'][room['id']]) >= .1
                rr.update({p+'_sun_minutes':saved['lit_minutes_ge_0_1m2'],p+'_before_noon_minutes':round(float(before[mask].sum()),2),p+'_after_noon_minutes':round(float((weights-before)[mask].sum()),2),
                    p+'_first_lit_time':saved['first_lit_time'],p+'_last_lit_time':saved['last_lit_time'],p+'_peak_lit_area_m2':saved['peak_lit_area_m2'],p+'_solar_noon':noon})
            rooms.append(rr)
        for window in sun['windows']:
            if window['type'] != 'window':
                continue
            room = next(r for r in sun['rooms'] if r['id']==window['room_id'])
            wr = {'sid':sid,'window_id':window['id'],'room_id':window['room_id'],'width_m':window['width'],'height_m':window['height'],
                'sill_height_m':window['lower_z_m']-room['floor_z_m'],'outward_azimuth_deg':window['outward_azimuth_deg'],'compass_direction':engine().compass(window['outward_azimuth_deg']),
                'nearest_facing_building_m':nearest_building(window,room,sun,fit,model),'poor':sun['registration']['poor']}
            for date in DATES:
                p = date.replace('-','')
                weights,before,noon = splits[date]
                codes = np.asarray(samples[date]['window_sample_codes'][window['id']])
                mask = (codes==0).any(axis=1)
                scored = {name:float(weights @ np.mean(codes==code,axis=1)) for code,name in [(0,'none'),(1,'sun behind the wall'),(2,'own roof'),(3,'neighbour'),(4,'tree')]}
                main = max(scored,key=scored.get)
                wr.update({p+'_sun_minutes':round(float(weights[mask].sum()),2),p+'_before_noon_minutes':round(float(before[mask].sum()),2),p+'_after_noon_minutes':round(float((weights-before)[mask].sum()),2),
                    p+'_main_blocker':main})
            windows.append(wr)
    table(FACTS/'homes.csv',homes)
    table(FACTS/'rooms.csv',rooms)
    table(FACTS/'windows.csv',windows)
    summary = {'selected':len(selected),'downloaded':sum(h['downloaded'] for h in homes),'computed':sum(h['computed'] for h in homes),
        'well_registered':sum(h['poor'] is False for h in homes),'computed_well_registered':sum(h['computed'] and h['poor'] is False for h in homes),
        'poor':sum(h['poor'] is True for h in homes),'registration_unknown':sum(h['poor'] is None for h in homes),'rooms':len(rooms),'exterior_windows':len(windows),
        'failures':dict(collections.Counter(h['failure_reason'] for h in homes if h['failure_reason'])),'trees_caution_all_homes':True}
    write(FACTS/'summary.json',summary)
    validate_facts(selected,homes,rooms,windows)
    comparison(homes)
    documentation(homes,rooms,windows)
    progress('Facts published: '+json.dumps(summary)+'. Evidence: data/facts/{homes,rooms,windows}.csv, summary.json and light_claim_vs_sun.json/png.', 'Time-box poor-registration inspection; run existing suite and network-blocked cache replay.')
    return summary


def validate_facts(selected,homes,rooms,windows):
    checks = []
    def check(name, passed, detail):
        checks.append({'name':name,'passed':bool(passed),'detail':detail})
    selected_ids = {r['sid'] for r in selected}
    home_ids = {r['sid'] for r in homes}
    check('all_selected_homes_present_once', selected_ids==home_ids and len(homes)==len(home_ids), {'selected':len(selected_ids),'home_rows':len(homes)})
    check('failures_remain_explicit',all(h['computed'] or h['failure_reason'] for h in homes),{'failed':sum(not h['computed'] for h in homes)})
    room_keys = {(r['sid'],r['room_id']) for r in rooms}
    check('room_and_window_foreign_keys', all(r['sid'] in home_ids for r in rooms) and all((w['sid'],w['room_id']) in room_keys for w in windows), {'rooms':len(rooms),'windows':len(windows)})
    check('entity_keys_unique',len({(r['sid'],r['room_id']) for r in rooms})==len(rooms) and len({(w['sid'],w['window_id']) for w in windows})==len(windows),{})
    deviations = []
    for name,rows in [('room',rooms),('window',windows)]:
        for r in rows:
            for date in DATES:
                p=date.replace('-','')
                error=abs(r[p+'_sun_minutes']-r[p+'_before_noon_minutes']-r[p+'_after_noon_minutes'])
                if error>.011:
                    deviations.append([name,r['sid'],date,error])
    check('solar_noon_split_conserves_minutes',not deviations,{'bad_count':len(deviations),'examples':deviations[:5],'tolerance_minutes':.011})
    check('poor_thresholds_unchanged',all(h['poor'] is None or h['poor']==(h['registration_iou']<.55 or h['registration_containment']<.9) for h in homes),{'iou_min':.55,'containment_min':.9})
    ratio_errors=[]
    for h in homes:
        if not h['computed']:continue
        rs=[r for r in rooms if r['sid']==h['sid']]
        ws=[w for w in windows if w['sid']==h['sid']]
        if len(rs)!=h['rooms'] or len(ws)!=h['exterior_windows'] or abs(sum(r['area_m2'] for r in rs)-h['floor_area_m2'])>.001 or abs(sum(w['width_m']*w['height_m'] for w in ws)-h['glazing_area_m2'])>.00011:
            ratio_errors.append(h['sid'])
    check('home_entity_counts_and_area_totals',not ratio_errors,{'bad_sids':ratio_errors})
    write(FACTS/'validation.json',{'all_passed':all(c['passed'] for c in checks),'checks':checks})
    if not all(c['passed'] for c in checks):
        raise RuntimeError('facts_validation_failed')


def distribution(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return {'n':0,'values':[]}
    return {'n':len(values),'values':values,'mean':float(np.mean(values)),'min':values[0],'p25':float(np.quantile(values,.25)),'median':float(np.median(values)),'p75':float(np.quantile(values,.75)),'max':values[-1],'zero_share':sum(v==0 for v in values)/len(values)}


def comparison(homes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    available = [h for h in homes if h['computed'] and h['poor'] is False]
    metrics = ['dec21_main_living_sun_minutes','mar20_main_living_sun_minutes','glazing_over_floor_area']
    groups = {str(claim).lower():[h for h in available if h['light_claim']==claim] for claim in (False,True)}
    output = {'scope':'Computed, well-registered homes only; one scan per sid, not deduplicated by street address. Missing living labels excluded from living metrics, not replaced by zero.',
        'light_regex':LIGHT.pattern,'groups':{},'differences':{},'interpretation':[], 'causality':'Descriptive association only; geometry assumptions and geographic/property mix may confound differences.'}
    for key,group in groups.items():
        output['groups'][key] = {'homes':len(group),'metrics':{metric:distribution(h[metric] for h in group) for metric in metrics},'missing_living_room':sum(h['main_living_room_id'] is None for h in group)}
    fig,axes = plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
    for ax,metric in zip(axes[:3],metrics):
        a = output['groups']['true']['metrics'][metric]
        b = output['groups']['false']['metrics'][metric]
        if a['n'] and b['n']:
            delta = a['median']-b['median']
            output['differences'][metric] = {'claim_minus_no_claim_median':delta,'claim_minus_no_claim_mean':a['mean']-b['mean']}
            output['interpretation'].append(f"{metric}: claim median {a['median']:.4g}, no-claim median {b['median']:.4g}; distributions {'have different medians' if abs(delta)>1e-10 else 'have equal medians but may differ elsewhere'}. Broad overlap is visible in the ECDF; no causal conclusion.")
        for key,label,color in [('false','No claim','#777777'),('true','Light claim','#111111')]:
            vals = output['groups'][key]['metrics'][metric]['values']
            if vals:
                ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=label,color=color)
        ax.set_title({'dec21_main_living_sun_minutes':'21 December living sun','mar20_main_living_sun_minutes':'20 March living sun','glazing_over_floor_area':'Window glazing / floor area'}[metric])
        ax.set_xlabel('Geometric minutes' if 'minutes' in metric else 'm² / m²')
        ax.set_ylabel('Cumulative share')
        ax.legend(fontsize=8)
    zero = {k:output['groups'][k]['metrics'][metrics[0]].get('zero_share') for k in groups}
    output['winter_zero_share'] = zero
    axes[3].bar(['No claim','Light claim'],[zero['false'] or 0,zero['true'] or 0],color=['#aaa','#333'])
    axes[3].set_title('Zero winter living sun')
    axes[3].set_ylabel('Share with labelled living room')
    axes[3].set_ylim(0,1)
    output['interpretation'].append(f"Zero winter-sun shares: no claim {zero['false']}, light claim {zero['true']}. These are conditional geometric zeros, not measured darkness.")
    claimed = sorted([h for h in groups['true'] if h[metrics[0]] is not None],key=lambda h:(h[metrics[0]],h['sid']))
    def short(h):
        return {k:h[k] for k in ['sid','listing_id','address','city','state',metrics[0],metrics[1],'glazing_over_floor_area']}
    output['least_winter_sun_15'] = [short(h) for h in claimed[:15]]
    output['most_winter_sun_15'] = [short(h) for h in sorted(claimed,key=lambda h:(-h[metrics[0]],h['sid']))[:15]]
    fig.suptitle('Conditional geometric direct sun and listing remarks — well-registered homes')
    fig.savefig(FACTS/'light_claim_vs_sun.png',dpi=160)
    plt.close(fig)
    write(FACTS/'light_claim_vs_sun.json',output)


def documentation(homes, rooms, windows):
    lines = ['# Production facts — Brief F', '', 'All sunlight is computed potential direct sun under clear-aperture geometry assumptions, never measured light. Empty cells are unknown, never zero. Failed homes remain in homes.csv and data/sun/index.csv; rooms/windows tables contain known usable entities only, with no invented failure entities.', '',
        'Selection uses authoritative CSV scan/listing edges and a deterministic primary/latest remarked listing per scan. See selection.json. Light claims use the same natural light / naturally lit / bright / sunny regex as the audit; this deliberately does not classify every use of the word light.', '',
        'Poor means post-fit IoU <0.55 OR scan containment <0.90. No threshold is relaxed. Registered coordinates refer to the fitted scan-interior centroid, not the original model origin or a validated mailing point. Coordinates are spherical/local-plane approximations. Rooms use 0.10 m rasters and >=0.1 m² lit floor area; window minutes mean any of nine aperture rays clear. All conditional exterior doorways transmit in room totals, while glazing ratios and windows.csv count windows only. No missing living room or primary bedroom is inferred.', '',
        'The bounded original-ten repair can choose an older cached footprint when it fits better. A better fit does not verify current structure identity or present-day geometry. See registration_review.json for before/after capture dates and unverified diagnoses.', '',
        'Solar noon is the maximum engine apparent elevation searched at one-second resolution within five minutes of the maximum sampled solar elevation. Five-minute bins crossing noon are split fractionally. First/last lit times are bin-start timestamps with numeric UTC offset, not exact transitions. The existing engine longitude timezone approximation applies.', '',
        'Trees are a caution flag for every home; most vegetation is not modelled. The pre-existing target retains Brief B tree prisms and exposes tree as a main blocker when dominant; this extra category avoids hiding a real blocker. Own roof and neighbour blockers are opaque plane/prism assumptions. Main blocker is the greatest aperture-area-weighted category among clear (none), behind-wall, own roof, neighbour and tree during above-horizon samples. Ties favour the first category in that order. Per-home raw category totals remain in sun.json.', '',
        'Nearest-facing building distance uses same-collection neighbour footprints, analytically clips each edge to the +/-20 degree outward bearing wedge, and finds the minimum edge distance up to 60 m. Empty means none among returned/modelled neighbours, not proven unbounded view. Other footprint exclusions are inherited from the engine.', '',
        'Room labels, classifications, opening dimensions, floor graph and geolocation are API-supplied estimates. They are not independent site measurements. Floor and glazing areas, registration, directions, sun and derived ratios are computed. Roof shape/heights, clear glazing, ground datum, transparent exterior doorways, five-minute and raster sampling are assumptions; see every sun.json.assumptions.', '',
        'Reproduce: `python3 -I experiments/sun_batch.py all`; resume acquisition from sanitized data/cache. Offline proof: `python3 -I experiments/sun_batch.py replay` blocks socket access and credential-file opens, recomputes every home and compares tables. `repair` inspects the original ten poor homes under a 20-minute limit. No new tests.', '']
    common = {'sid':('identifier','API/index supplied'),'listing_id':('identifier','CSV/JSONL supplied'),'listing_instance_id':('identifier','JSONL supplied'),'alternate_listing_ids':('JSON list of identifiers','CSV/JSONL supplied alternate remarked listings'),'address':('text','listing supplied'),'city':('text','listing supplied'),'state':('text','listing supplied'),
        'light_claim':('boolean','computed regex on remarks'),'downloaded':('boolean','verified cached Matterport plus Vexcel successful responses'),'computed':('boolean','verified engine completion'),'failure_reason':('code','observed failure, not inferred absence'),'poor':('boolean','computed fixed registration thresholds'),
        'registered_latitude':('degrees north','computed fitted scan-interior centroid'),'registered_longitude':('degrees east','computed fitted scan-interior centroid'),'registration_iou':('fraction','computed raster intersection over union'),'registration_containment':('fraction','computed scan area inside selected footprint'),'registration_rotation_deg':('degrees','computed fit rotation'),'registration_shift_m':('metres','computed centroid shift'),
        'rooms':('count','computed usable rooms'),'exterior_windows':('count','computed uniquely assigned exterior windows'),'floor_area_m2':('m²','computed sum of usable room rasters'),'area_m2':('m²','computed room raster area'),'glazing_area_m2':('m²','computed window rectangles, assumed all clear'),'glazing_over_floor_area':('m²/m²','computed window rectangles divided by usable floor area'),
        'parcel_tree_cover_percent':('percent','Vexcel supplied parcel estimate; caution, not modelled shade'),'parcel_tree_attributes_json':('JSON, source-key units','Vexcel supplied tree attributes for latest containing parcel'),'tree_caution':('boolean','assumed true: vegetation uncertainty applies'),'tree_attribute_source':('text','computed provenance or explicit unknown reason'),
        'main_living_room_id':('identifier','computed largest explicitly labelled living/family room'),'primary_bedroom_room_id':('identifier','explicit primary/master label required'),'largest_bedroom_room_id':('identifier','computed largest labelled bedroom only when primary absent'),
        'dec21_main_living_sun_minutes':('minutes','computed qualifying floor sun on 2026-12-21'),'mar20_main_living_sun_minutes':('minutes','computed qualifying floor sun on 2026-03-20'),
        'room_id':('identifier','Matterport supplied'),'label':('text','Matterport supplied'),'top_classification':('label','API highest confidence classification'),'classification_confidence':('fraction','API supplied model confidence'),'bedroom_role':('label','explicit primary or computed largest_bedroom; never inferred primary'),
        'window_id':('identifier','computed stable floor/edge/opening index'),'width_m':('metres','Matterport supplied estimate'),'height_m':('metres','Matterport supplied estimate'),'sill_height_m':('metres above room floor','computed opening lower elevation minus room elevation'),'outward_azimuth_deg':('degrees clockwise from north','computed registered exterior normal'),'compass_direction':('8-point compass','computed bearing bin'),'nearest_facing_building_m':('metres','computed approximate edge distance; empty if none within search')}
    for name,rows in [('homes',homes),('rooms',rooms),('windows',windows)]:
        lines += [f'## {name}.csv', '', '| Column | Unit | Provenance |','|---|---|---|']
        for column in list(dict.fromkeys(k for row in rows for k in row)):
            if column in common:
                unit,origin = common[column]
            elif re.match(r'2026\d{4}_',column):
                suffix = column[9:]
                unit = 'minutes' if suffix.endswith('minutes') else 'm²' if suffix=='peak_lit_area_m2' else 'ISO 8601 with UTC offset' if suffix in ('first_lit_time','last_lit_time','solar_noon') else 'category'
                origin = {'sun_minutes':'computed geometric potential; room >=0.1m² floor sun, window any clear ray','before_noon_minutes':'computed part of qualifying bin before solar noon','after_noon_minutes':'computed part of qualifying bin at/after solar noon','peak_lit_area_m2':'computed maximum lit room floor raster','first_lit_time':'computed first qualifying bin start','last_lit_time':'computed last qualifying bin start','solar_noon':'computed maximum apparent solar elevation, 1-second search','main_blocker':'computed dominant exterior obstruction; see definition above'}[suffix]
            else:
                raise RuntimeError('undocumented_column_'+column)
            lines.append(f'| {column} | {unit} | {origin} |')
        lines.append('')
    (FACTS/'README.md').write_text('\n'.join(lines))


def hashes():
    files = [SUN/'index.csv',FACTS/'homes.csv',FACTS/'rooms.csv',FACTS/'windows.csv',FACTS/'light_claim_vs_sun.json',FACTS/'light_claim_vs_sun.png']
    result = {}
    for path in files:
        if path == SUN/'index.csv':
            rows = list(csv.DictReader(path.open()))
            for row in rows:
                row.pop('seconds',None)
            data = json.dumps(rows,sort_keys=True).encode()
        else:
            data = path.read_bytes()
        result[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    semantic = hashlib.sha256()
    for row in csv.DictReader((SUN/'index.csv').open()):
        if row.get('failure_reason'):
            continue
        value = read(SUN/row['sid']/'sun.json')
        value.pop('seconds',None)
        semantic.update(json.dumps(value,sort_keys=True,allow_nan=False).encode())
    result['computed_sun_json_aggregate_excluding_seconds'] = semantic.hexdigest()
    return result


def registration_counts(sids):
    values = [read(CACHE/sid/'registration.json') for sid in sids if (CACHE/sid/'registration.json').exists()]
    poor = sum(v['poor'] for v in values)
    return {'homes':len(sids),'registered':len(values),'unknown':len(sids)-len(values),'poor':poor,
        'poor_share_of_registered':poor/len(values) if values else None,'poor_share_of_all_homes':poor/len(sids) if sids else None}


def repair(selected):
    started = time.monotonic()
    deadline = started+20*60
    e = engine()
    original40 = read(CACHE/'brief_f_baseline/batch_selection.json')['homes']
    original_ids = [r['sid'] for r in original40]
    selected_ids = [r['sid'] for r in selected]
    targets = []
    for r in csv.DictReader((CACHE/'brief_f_baseline/index.csv').open()):
        if r['registration_poor'] == 'True':
            targets.append(r['sid'])
    report = {'time_limit_seconds':1200,'targets':targets,'before_40':registration_counts(original_ids),'before_full':registration_counts(selected_ids),'homes':[],'changed_sids':[]}
    os.environ['SUN_OFFLINE'] = '1'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,5,figsize=(18,8),layout='constrained')
    for sid,ax in zip(targets,axes.flat):
        if time.monotonic()>deadline:
            report['time_limit_reached'] = True
            break
        folder = CACHE/sid
        before = read(folder/'registration.json')
        if not (folder/'registration_before_f.json').exists():
            write(folder/'registration_before_f.json',before)
        model,prop = e.inputs(sid)
        matrix,lat,lon = e.geo_matrix(model)
        buildings = e.building_rings(prop,lat,lon)
        centre = np.array(before['_plot']['centre'])
        inner = np.array(before['_plot']['inner'])
        current_ring = np.array(before['_plot']['ring'])
        viable = [b for b in buildings if np.linalg.norm(b['ring'].mean(0)-centre)<=35 and not np.allclose(b['ring'].mean(0),current_ring.mean(0),atol=.1)]
        # Prefer recent nearby outlines of similar area; never relax the quality thresholds.
        def polygon_area(ring):
            return abs(float(np.dot(ring[:,0],np.roll(ring[:,1],-1))-np.dot(ring[:,1],np.roll(ring[:,0],-1))))/2
        scan_area = before['interior_m2']
        viable.sort(key=lambda b:(abs(math.log(max(polygon_area(b['ring']),1)/max(scan_area,1)))+np.linalg.norm(b['ring'].mean(0)-centre)/20))
        candidates = []
        best = before
        for b in viable[:4]:
            if time.monotonic()>deadline:
                break
            selected_prop = {'features':[{'properties':{'structures':[b['structure']]}}]}
            original_gql,original_vexcel = e.G.gql,e.G.vexcel_property
            e.G.gql = lambda query:{'data':{'model':model}}
            e.G.vexcel_property = lambda lat,lon:selected_prop
            try:
                candidate = e.G.analyse(sid,{'address':'','city':''})
            except Exception as error:
                candidates.append({'building_id':b['id'],'failure':safe_reason(error)})
                continue
            finally:
                e.G.gql,e.G.vexcel_property = original_gql,original_vexcel
            if 'skip' in candidate:
                candidates.append({'building_id':b['id'],'failure':'registration_skip'})
                continue
            candidate.update(collection=b['collection'],capture_date=b['capture_date'],source='Brief F bounded alternate-footprint fit, cached inputs',poor=candidate['iou_after']<.55 or candidate['contain_after']<.9)
            candidates.append({k:candidate[k] for k in ['iou_after','contain_after','poor','fix_shift_m','collection','capture_date']}|{'building_id':b['id'],'footprint_m2':candidate['footprint_m2']})
            if not candidate['poor'] and candidate['iou_after']>best['iou_after']+.03 and candidate['contain_after']>=best['contain_after']-.005:
                best = candidate
        changed = best is not before
        if changed:
            write(folder/'registration.json',best)
            e.run_home(sid,skip_floor_images=True)
            report['changed_sids'].append(sid)
        row = next((r for r in selected+original40 if r['sid']==sid),{})
        ratio = before['interior_m2']/before['footprint_m2'] if before['footprint_m2'] else None
        address = row.get('address','')
        if changed:
            hypothesis = 'An alternate cached outline passes unchanged thresholds; wrong structure versus capture/outline discrepancy remains unverified'
        elif ratio and ratio<.55:
            hypothesis = 'Likely scan of one unit/partial structure within a larger aerial footprint' if re.search(r'\b(unit|apt|apartment|condo)\b',address,re.I) else 'Scan much smaller than footprint: partial/unit scan, attached/extended roof, or wrong structure unresolved'
        elif before['contain_after']<.9:
            hypothesis = 'Scan spills outside footprint: footprint/scan mismatch, eave/wall discrepancy or registration search limits unresolved'
        else:
            hypothesis = 'Low IoU mismatch unresolved'
        entry = {'sid':sid,'address':address,'diagnosis_status':'Geometric hypothesis, not independently verified property truth','hypothesis':hypothesis,
            'scan_to_footprint_area_ratio':ratio,'before':{k:before.get(k) for k in ['iou_after','contain_after','poor','interior_m2','footprint_m2','collection','capture_date']},
            'after':{k:best.get(k) for k in ['iou_after','contain_after','poor','interior_m2','footprint_m2','collection','capture_date']},'candidates':candidates,'changed':changed}
        report['homes'].append(entry)
        for fit,color,label in [(before,'#999999','Before footprint'),(best,'#111111','After footprint')]:
            ring = np.array(fit['_plot']['ring'])
            ax.plot(*ring.T,color=color,lw=1,label=label)
        th = np.deg2rad(best['fix_rot_deg'])
        rot = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        shifted = (inner-centre)@rot.T+centre+best['_plot']['d']
        ax.scatter(*shifted.T,s=2,color='#b55b22',label='Fitted scan')
        ax.set_title(f"{sid}\nIoU {before['iou_after']:.3f} → {best['iou_after']:.3f}; C {best['contain_after']:.3f}",fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        write(FACTS/'registration_review_progress.json',report)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle('Original ten poor registrations — footprint / scan geometry, metres')
    fig.savefig(FACTS/'registration_review.png',dpi=150)
    plt.close(fig)
    report.update(after_40=registration_counts(original_ids),after_full=registration_counts(selected_ids),elapsed_seconds=round(time.monotonic()-started,2),
        limitations='Ten original poor homes inspected; full-set count reflects only these bounded repairs. Unit/structure diagnoses are hypotheses. No definition of poor changed.')
    write(FACTS/'registration_review.json',report)
    if report['changed_sids']:
        records = {r['sid']:r for r in csv.DictReader((SUN/'index.csv').open())}
        byid = {r['sid']:r for r in selected}
        for sid in report['changed_sids']:
            if sid in byid:
                records[sid] = e.index_row(byid[sid],read(SUN/sid/'sun.json'))
        table(SUN/'index.csv',[records[r['sid']] for r in selected],next(csv.reader((CACHE/'brief_f_baseline/index.csv').open())))
    facts(selected)
    progress('Registration inspection finished in '+str(report['elapsed_seconds'])+' seconds; '+str(len(report['changed_sids']))+' cheap repairs. Evidence: data/facts/registration_review.json/png. Existing starter guard: 7 passed.', 'Inspect diagnostic/chart, run network-blocked full-cohort replay and final report.')


def install_guard():
    global GUARD_INSTALLED
    if GUARD_INSTALLED:
        return
    def guard(event,args):
        if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):
            GUARD_ATTEMPTS[event] += 1
            raise RuntimeError('network_blocked_by_replay')
        if event == 'open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and Path(os.fsdecode(args[0])).name == '.env':
            GUARD_ATTEMPTS['credential_open'] += 1
            raise RuntimeError('credential_open_blocked_by_replay')
    sys.addaudithook(guard)
    GUARD_INSTALLED = True


def replay(selected,workers):
    baseline = hashes()
    install_guard()
    os.environ['SUN_OFFLINE'] = '1'
    os.environ['SUN_REPLAY_GUARD'] = '1'
    original = engine().request_cached
    def cached_only(path,service,query):
        if not path.exists():
            raise RuntimeError('offline_cache_missing')
        return original(path,service,query)
    engine().request_cached = cached_only
    fields = next(csv.reader((CACHE/'brief_f_baseline/index.csv').open()))
    records = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(compute_one,row) for row in selected]
        for task in cf.as_completed(tasks):
            records.append(task.result())
            if len(records) % 20 == 0 or len(records)==len(selected):
                write(FACTS/'replay_progress.json',{'finished':len(records),'selected':len(selected),'parent_blocked_attempts':dict(GUARD_ATTEMPTS),'worker_guard_enabled':True})
                progress('Cache replay: '+str(len(records))+'/'+str(len(selected))+' homes finished. Network and credential-open audit hooks installed in parent and every worker.', 'Finish replay, compare deterministic tables and final report.')
    table(SUN/'index.csv',sorted(records,key=lambda r:r['sid']),fields)
    facts(selected)
    after = hashes()
    attempts = dict(GUARD_ATTEMPTS)
    child_attempts = [r['_guard_attempts'] for r in records if r.get('_guard_attempts')]
    child_calls = [r['_engine_api_calls'] for r in records if r.get('_engine_api_calls')]
    proof = {'selected':len(selected),'recomputed':sum(not r.get('failure_reason') for r in records),'blocked_attempts':dict(attempts),'api_calls':dict(engine().API_CALLS),
        'worker_blocked_attempts':child_attempts,'worker_api_calls':child_calls,'worker_records_guarded':sum('_guard_attempts' in r for r in records),
        'network_calls':0 if not attempts and not child_attempts and not child_calls else None,'deterministic_outputs_match':baseline==after,'before':baseline,'after':after,
        'index_timing_excluded':True,'sun_json_timing_excluded':True,'engine_sha256':hashlib.sha256((ROOT/'experiments/sun_engine.py').read_bytes()).hexdigest(),'batch_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(FACTS/'cache_replay_proof.json',proof)
    if baseline != after or attempts or child_attempts or child_calls:
        raise RuntimeError('cache_replay_mismatch_or_attempt')
    progress('Network-blocked cache replay passed: '+json.dumps({k:v for k,v in proof.items() if k not in ('before','after')})+'. Facts and comparison are deterministic.', 'Final report with verified evidence, assumptions/failures and stage counts.')


FOLLOWUP = FACTS/'registration_followup'
FOLLOWUP_BASE = CACHE/'registration_followup_baseline'
FOLLOWUP_PROGRESS = Path('/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T14-32-05-670Z-56212-5dc9fbf4-cefe-4970-880b-70b80d2af8e8.md')


def followup_progress(done, next_step):
    FOLLOWUP_PROGRESS.write_text('# Objective\nExplain poor registrations; 30 random visual reviews; cheap cause-specific fixes with unchanged thresholds. Deadline 15:47 UTC.\n\n# Done\n'+done+'\n\n# Next\n'+next_step+'\n')


def area_polygon(ring):
    return abs(float(np.dot(ring[:,0],np.roll(ring[:,1],-1))-np.dot(ring[:,1],np.roll(ring[:,0],-1))))/2


def followup_diagnose(selected):
    import random
    from matplotlib.path import Path as PolygonPath
    os.environ['SUN_OFFLINE']='1'
    install_guard()
    e=engine()
    listings={str(r['id']):r for r in map(json.loads,(ROOT/'data/costar-kit/data/residential-listings.jsonl').open())}
    records=[]
    for row in selected:
        sid=row['sid'];file=CACHE/sid/'registration.json'
        if not file.exists():
            records.append({'sid':sid,'status':'registration_unknown'});continue
        fit=read(file);model,prop=e.inputs(sid);matrix,lat,lon=e.geo_matrix(model)
        buildings=e.building_rings(prop,lat,lon)
        centre=np.array(fit['_plot']['centre']);ring=np.array(fit['_plot']['ring'])
        newest=max(b['capture_date'] for b in buildings)
        current=[b for b in buildings if b['capture_date']==newest]
        listing=listings.get(row['listing_id'],{});text=listing.get('remarks') or ''
        address=row['address']
        iou,contain=fit['iou_after'],fit['contain_after']
        bucket=('contained_small_scan' if iou<.55 and contain>=.9 else
                'low_overlap_and_spill' if iou<.55 else 'containment_only_spill') if fit['poor'] else 'well_registered'
        g=model['geocoordinates'];distance=None
        try:distance=float(np.linalg.norm(e.G.enu(float(row['listing_latitude']),float(row['listing_longitude']),g['latitude'],g['longitude'])))
        except (TypeError,ValueError,KeyError):pass
        date=listing.get('listDate') or '';capture=fit.get('capture_date') or ''
        date_gap=None
        try:date_gap=(dt.date.fromisoformat(date[:10])-dt.date.fromisoformat(capture[:10])).days
        except ValueError:pass
        floors=[f for f in model['floors'] if f.get('edges')]
        r={'sid':sid,'status':bucket,'poor':fit['poor'],'iou':iou,'containment':contain,
           'scan_projected_fill_m2':fit['interior_m2'],'footprint_m2':fit['footprint_m2'],
           'scan_roof_area_ratio':fit['interior_m2']/fit['footprint_m2'],
           'floors':len(floors),'nearby_structures_latest_35m':sum(np.linalg.norm(b['ring'].mean(0)-centre)<=35 for b in current),
           'nearby_outline_variants_35m':sum(np.linalg.norm(b['ring'].mean(0)-centre)<=35 for b in buildings),
           'latest_capture':newest,'selected_capture':capture,'listing_date':date,'listing_minus_capture_days':date_gap,
           'listing_scan_distance_m':distance,'attached_multiunit_text_proxy':bool(listing.get('isInCondoBuilding') or re.search(r'\b(condo(?:minium)?|townhome|townhouse|duplex|fourplex|apartment|attached)\b',text+' '+address,re.I)),
           'garage_patio_text_proxy':bool(re.search(r'\b(garage|carport|patio|porch)\b',text,re.I)),
           'fit_rotation_deg':fit['fix_rot_deg'],'fit_shift_m':fit['fix_shift_m']}
        records.append(r)
    table(FOLLOWUP/'diagnostics_before.csv',records)
    poor=[r for r in records if r.get('poor')]
    sample=random.Random(20261114).sample(sorted(r['sid'] for r in poor),30)
    write(FOLLOWUP/'random_sample.json',{'seed':20261114,'sampling_frame':'sorted 725 baseline poor sids, uniform random.sample without replacement','sids':sample})
    def describe(rows):
        return {'n':len(rows),'causes':dict(collections.Counter(r['status'] for r in rows)),
          'distributions':{k:distribution([r[k] for r in rows if r.get(k) is not None]) for k in ['iou','containment','scan_roof_area_ratio','nearby_structures_latest_35m','nearby_outline_variants_35m','floors','listing_minus_capture_days','listing_scan_distance_m']},
          'counts':{k:sum(bool(r.get(k)) for r in rows) for k in ['attached_multiunit_text_proxy','garage_patio_text_proxy']},
          'multifloor':sum(r.get('floors',0)>1 for r in rows),'listing_scan_over_25m':sum((r.get('listing_scan_distance_m') or 0)>25 for r in rows),
          'capture_over_2years_before_listing':sum((r.get('listing_minus_capture_days') or 0)>730 for r in rows)}
    write(FOLLOWUP/'diagnostic_summary.json',{'before':registration_counts([r['sid'] for r in selected]),'poor':describe(poor),'good':describe([r for r in records if r.get('poor') is False]),'guard_attempts':dict(GUARD_ATTEMPTS),'limitations':'Geometric buckets are measured metric failure modes. Text proxies do not prove attached construction or unscanned garages/patios. Projected wall fill is not summed scan floor area. Capture dates do not establish building change.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    byid={r['sid']:r for r in records}
    for page in range(6):
        fig,axes=plt.subplots(1,5,figsize=(19,4.5),layout='constrained')
        for ax,sid in zip(axes,sample[page*5:page*5+5]):
            fit=read(CACHE/sid/'registration.json');model,prop=e.inputs(sid);matrix,lat,lon=e.geo_matrix(model)
            ring=np.array(fit['_plot']['ring']);centre=np.array(fit['_plot']['centre'])
            th=np.deg2rad(fit['fix_rot_deg']);rot=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
            for b in e.building_rings(prop,lat,lon):
                if b['capture_date']==byid[sid]['latest_capture'] and np.linalg.norm(b['ring'].mean(0)-centre)<35:ax.plot(*b['ring'].T,color='#cccccc',lw=.7)
            ax.plot(*ring.T,color='black',lw=1.6)
            for n,f in enumerate(model['floors']):
                for edge in f.get('edges') or []:
                    if len(edge['vertices'])!=2:continue
                    points=np.array([[v['position']['x'],v['position']['y']] for v in edge['vertices']])@matrix.T
                    points=(points-centre)@rot.T+centre+fit['_plot']['d']
                    ax.plot(*points.T,color=plt.cm.tab10(n%10),lw=.7)
            r=byid[sid];ax.set_title(f"{sid}\nIoU {r['iou']:.3f} C {r['containment']:.3f}\nscan/roof {r['scan_roof_area_ratio']:.2f}; floors {r['floors']}",fontsize=9)
            ax.set_aspect('equal');ax.tick_params(labelsize=6)
        fig.suptitle('Baseline random poor homes: selected roof black, latest neighbours grey, fitted walls by floor (metres)')
        fig.savefig(FOLLOWUP/f'review_{page+1}.png',dpi=150);plt.close(fig)
    followup_progress('Baseline diagnostics and six review pages written in data/facts/registration_followup/. Baseline preserved in data/cache/registration_followup_baseline/. No network attempts.', 'Review all 30 random poor homes and select largest one or two cause-specific experiments.')


def followup_context(selected):
    """Cross-tabs use frozen baseline metrics and existing room areas only."""
    diagnostics={r['sid']:r for r in csv.DictReader((FOLLOWUP/'diagnostics_before.csv').open())}
    listing_ids={r['listing_id'] for r in selected}
    listings={str(r['id']):r for r in map(json.loads,(ROOT/'data/costar-kit/data/residential-listings.jsonl').open()) if str(r['id']) in listing_ids}
    room_areas=collections.defaultdict(dict)
    for r in csv.DictReader((FOLLOWUP_BASE/'facts/rooms.csv').open()):room_areas[r['sid']][r['room_id']]=float(r['area_m2'])
    records=[]
    for row in selected:
        sid=row['sid'];r=diagnostics[sid].copy()
        if r['status']=='registration_unknown':records.append(r);continue
        model=read(CACHE/sid/'matterport.json')['data']['model']
        byfloor=collections.Counter();elevations=collections.defaultdict(list)
        for room in model['rooms']:
            fid=(room.get('floor') or {}).get('id')
            byfloor[fid]+=room_areas[sid].get(room['id'],0)
            dim=room.get('dimensionEstimates') or {}
            if dim.get('lowerElevation') is not None:elevations[fid].append(float(dim['lowerElevation'])*(.3048 if dim.get('units')=='imperial' else 1))
        usable=[f for f,area in byfloor.items() if f and area>0]
        ground=min(usable,key=lambda f:float(np.median(elevations[f])) if elevations[f] else 1e6) if usable else None
        r.update(summed_room_floor_area_m2=sum(byfloor.values()) or None,
                 lowest_model_floor_room_area_m2=byfloor[ground] if ground else None,
                 lowest_floor_selection='minimum room median lowerElevation; unknown if absent',
                 lowest_floor_elevation_available=bool(ground and elevations[ground]))
        roof=float(r['footprint_m2'])
        r['roof_over_summed_room_area']=roof/r['summed_room_floor_area_m2'] if r['summed_room_floor_area_m2'] else None
        r['lowest_floor_room_area_over_roof']=r['lowest_model_floor_room_area_m2']/roof if r['lowest_model_floor_room_area_m2'] else None
        listing=listings.get(row['listing_id'],{});ptype=(listing.get('property') or {}).get('mlsPropertyType') or ''
        r['listing_property_type']=ptype
        r['structured_multiunit_proxy']=bool(listing.get('isInCondoBuilding') or re.search(r'condo|town|duplex|multi.?family|apartment|attached',ptype,re.I)) and not bool(re.search(r'\bdetached\b',ptype,re.I))
        records.append(r)
    table(FOLLOWUP/'context.csv',records)
    groups={}
    for status in ['contained_small_scan','containment_only_spill','low_overlap_and_spill','well_registered']:
        rows=[r for r in records if r['status']==status]
        gap=lambda r:float(r['listing_minus_capture_days']) if r.get('listing_minus_capture_days') else None
        groups[status]={'n':len(rows),'structured_multiunit_proxy':sum(r['structured_multiunit_proxy'] for r in rows),
          'multifloor_lowest_floor_less_than_half_roof':sum(int(r['floors'])>1 and r.get('lowest_floor_room_area_over_roof') is not None and r['lowest_floor_room_area_over_roof']<.5 for r in rows),
          'capture_bins':{'after_listing':sum(gap(r) is not None and gap(r)<0 for r in rows),'0_to_365_days_before_listing':sum(gap(r) is not None and 0<=gap(r)<=365 for r in rows),'366_to_730_days':sum(gap(r) is not None and 365<gap(r)<=730 for r in rows),'over_730_days':sum(gap(r) is not None and gap(r)>730 for r in rows),'unknown':sum(gap(r) is None for r in rows)},
          'area_distributions':{k:distribution([r[k] for r in rows]) for k in ['summed_room_floor_area_m2','lowest_model_floor_room_area_m2','roof_over_summed_room_area','lowest_floor_room_area_over_roof']}}
    original_ids={r['sid'] for r in read(CACHE/'brief_f_baseline/batch_selection.json')['homes']}
    cohort={}
    for name,rows in [('original40_intersection',[r for r in records if r['sid'] in original_ids]),('expanded_cohort',[r for r in records if r['sid'] not in original_ids])]:
        known=[r for r in rows if r['status']!='registration_unknown']
        cohort[name]={'selected':len(rows),'known':len(known),'cause_counts':dict(collections.Counter(r['status'] for r in rows)),'structured_multiunit_proxy':sum(r['structured_multiunit_proxy'] for r in known)}
    review=list(csv.DictReader((FOLLOWUP/'visual_review.csv').open()))
    write(FOLLOWUP/'context_summary.json',{'groups':groups,'cohort':cohort,'visual_sample':{'n':len(review),'labels':dict(collections.Counter(r['visual_cause'] for r in review)),'status':'Manual inspection of all six baseline overlay pages; labels identify visible geometry, physical cause remains uncertain.'},'limitations':['Room areas sum usable room boundaries, not gross floor area. Lowest model floor may be a basement; does not identify a true ground floor.','Text and structured property-type flags are listing metadata, not independent building surveys.','More floors alone is not causal; projections of all floors should be compared with the roof.']})
    followup_progress('All 30 random poor overlays reviewed and labelled. Baseline metric, area, date, nearby-structure, listing-point and property-type cross-tabs saved under data/facts/registration_followup/.', 'Finish two bounded geometric experiments and inspect proposed gains; preserve all old registrations before applying.')


def followup_trial(row):
    """Two bounded experiments: proper floor union and alternative local outlines."""
    from matplotlib.path import Path as PolygonPath
    os.environ['SUN_OFFLINE']='1';install_guard();e=engine();sid=row['sid']
    output=FOLLOWUP/'trials'/f'{sid}.json'
    if output.exists():return read(output)
    before=read(CACHE/sid/'registration.json')
    result={'sid':sid,'before':{k:before.get(k) for k in ('iou_after','contain_after','poor')},'accepted':False,'trials':[]}
    try:
        model,prop=e.inputs(sid);matrix,lat,lon=e.geo_matrix(model)
        floors=[];perfloor=[]
        for floor in model['floors']:
            seg=[tuple((v['position']['x'],v['position']['y']) for v in edge['vertices']) for edge in floor.get('edges') or [] if len(edge['vertices'])==2]
            if not seg:continue
            floors.append(seg)
            inner=e.G.interior_points(seg)
            perfloor.append(inner)
        pooled=e.G.interior_points([s for floor in floors for s in floor])
        if pooled is None:raise RuntimeError('pooled_wall_fill_failed')
        variants=[('pooled_walls',pooled)]
        if len(floors)>1 and all(x is not None for x in perfloor):
            union=np.unique(np.round(np.concatenate(perfloor)*10).astype(int),axis=0)/10
            result['floor_areas_m2']=[round(len(x)*.01,2) for x in perfloor]
            result['summed_floor_area_m2']=round(sum(len(x) for x in perfloor)*.01,2)
            result['proper_union_area_m2']=round(len(union)*.01,2)
            result['pooled_area_m2']=round(len(pooled)*.01,2)
            if abs(len(union)-len(pooled))/len(pooled)>.01:variants.append(('union_of_floor_interiors',union))
        else:result['floor_union_unavailable']='single floor or at least one floor does not close'
        points=pooled[::25]@matrix.T;centre=points.mean(0)
        ring0=np.array(before['_plot']['ring']);c0=ring0.mean(0)
        buildings=e.building_rings(prop,lat,lon)
        current=min(buildings,key=lambda b:np.linalg.norm(b['ring'].mean(0)-c0)+abs(area_polygon(b['ring'])-area_polygon(ring0))/100)
        viable=[]
        for b in buildings:
            if b is current:continue
            distance=float(np.linalg.norm(b['ring'].mean(0)-c0))
            overlap=float(PolygonPath(b['ring']).contains_points(points).mean())
            if distance<=12 and overlap>=.5:
                score=abs(math.log(max(area_polygon(b['ring']),1)/max(before['interior_m2'],1)))+distance/20
                viable.append((score,b))
        viable.sort(key=lambda x:x[0]);candidates=[current]+[b for _,b in viable[:4]]
        best=before
        for method,inner in variants:
            for b in candidates:
                if method=='pooled_walls' and b is current:continue
                original=(e.G.gql,e.G.vexcel_property,e.G.interior_points)
                e.G.gql=lambda query:{'data':{'model':model}}
                e.G.vexcel_property=lambda lat,lon:{'features':[{'properties':{'structures':[b['structure']]}}]}
                e.G.interior_points=lambda segments:inner
                try:fit=e.G.analyse(sid,{'address':'','city':''})
                finally:e.G.gql,e.G.vexcel_property,e.G.interior_points=original
                fit=sanitize(fit)
                if 'skip' in fit:continue
                if any(not math.isfinite(float(fit[k])) for k in ('iou_after','contain_after','iou_before','contain_before','interior_m2','footprint_m2')):
                    result['trials'].append({'method':method,'building_id':b['id'],'failure':'nonfinite_fit_metric'})
                    continue
                fit.update(collection=b['collection'],capture_date=b['capture_date'],source='Brief F follow-up: '+method+'; bounded local alternate outline',poor=bool(fit['iou_after']<.55 or fit['contain_after']<.9))
                trial={k:fit[k] for k in ('iou_after','contain_after','poor','interior_m2','footprint_m2','fix_shift_m','collection','capture_date')}
                trial.update(method=method,building_id=b['id'],centroid_distance_from_original_m=float(np.linalg.norm(b['ring'].mean(0)-c0)),raw_scan_containment=float(PolygonPath(b['ring']).contains_points(points).mean()))
                result['trials'].append(trial)
                if not fit['poor'] and fit['iou_after']>best['iou_after']+.03 and fit['contain_after']>=max(.9,best['contain_after']-.005):
                    best=fit;result['best_method']=method
        if best is not before:
            result['accepted']=True;result['fit']=best
        result['after']={k:best.get(k) for k in ('iou_after','contain_after','poor')}
    except Exception as error:result['failure']=safe_reason(error)
    result['guard_attempts']=dict(GUARD_ATTEMPTS)
    write(output,result);return result


def followup_experiment(selected,workers):
    targets=[r for r in selected if (CACHE/r['sid']/'registration.json').exists() and read(CACHE/r['sid']/'registration.json')['poor']]
    results=[];started=time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for task in cf.as_completed([pool.submit(followup_trial,row) for row in targets]):
            results.append(task.result())
            if len(results)%20==0 or len(results)==len(targets):
                counts={'finished':len(results),'targets':len(targets),'accepted':sum(r['accepted'] for r in results),'failures':sum('failure' in r for r in results),'elapsed_seconds':round(time.monotonic()-started,1)}
                write(FOLLOWUP/'experiment_progress.json',counts)
                followup_progress('30 baseline geometry pictures manually reviewed; labels in registration_followup/visual_review.csv. Experiment '+json.dumps(counts), 'Finish bounded candidates; inspect accepted examples before applying and rerun only changed homes.')
    write(FOLLOWUP/'experiment_results.json',{'targets':len(targets),'results':results,'elapsed_seconds':round(time.monotonic()-started,1)})


def followup_proposals():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=read(FOLLOWUP/'experiment_results.json')['results']
    accepted=sorted([r for r in results if r['accepted']],key=lambda r:r['sid'])
    rows=[]
    for r in accepted:
        before=read(FOLLOWUP_BASE/'registrations'/f"{r['sid']}.json");after=r['fit']
        rows.append({'sid':r['sid'],'iou_before':before['iou_after'],'containment_before':before['contain_after'],'iou_after':after['iou_after'],'containment_after':after['contain_after'],'capture_before':before.get('capture_date'),'capture_after':after.get('capture_date'),'collection_before':before.get('collection'),'collection_after':after.get('collection'),'method':r['best_method'],'footprint_before_m2':before['footprint_m2'],'footprint_after_m2':after['footprint_m2']})
    table(FOLLOWUP/'proposed_changes.csv',rows)
    for page in range((len(accepted)+11)//12):
        fig,axes=plt.subplots(3,4,figsize=(16,12),layout='constrained')
        for ax,r in zip(axes.flat,accepted[page*12:(page+1)*12]):
            before=read(FOLLOWUP_BASE/'registrations'/f"{r['sid']}.json");after=r['fit']
            for fit,color,width in [(before,'#aaaaaa',1),(after,'black',1.6)]:ax.plot(*np.array(fit['_plot']['ring']).T,color=color,lw=width)
            inner=np.array(after['_plot']['inner']);centre=np.array(after['_plot']['centre']);th=np.deg2rad(after['fix_rot_deg'])
            rot=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
            points=(inner-centre)@rot.T+centre+after['_plot']['d'];ax.scatter(*points.T,s=2,color='#b55b22')
            ax.set_title(f"{r['sid']}\nIoU {before['iou_after']:.3f} → {after['iou_after']:.3f}; C {after['contain_after']:.3f}\n{before.get('capture_date','accepted')} → {after.get('capture_date')}",fontsize=8)
            ax.set_aspect('equal');ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(accepted[page*12:(page+1)*12]):]:ax.axis('off')
        fig.suptitle('Candidate changes: old roof grey; new roof black; fitted projected scan orange. Geometry only, no independent identity proof.')
        fig.savefig(FOLLOWUP/f'proposals_{page+1}.png',dpi=130);plt.close(fig)


def followup_validate():
    """Withhold fits whose finite FFT grid omits scan or footprint geometry."""
    os.environ['SUN_OFFLINE']='1';install_guard();e=engine();records=[]
    for result in read(FOLLOWUP/'experiment_results.json')['results']:
        if not result['accepted']:continue
        sid=result['sid'];fit=result['fit'];model,_=e.inputs(sid);matrix,_,_=e.geo_matrix(model)
        segments=[tuple((v['position']['x'],v['position']['y']) for v in edge['vertices']) for floor in model['floors'] for edge in floor.get('edges') or [] if len(edge['vertices'])==2]
        inner=e.G.interior_points(segments)@matrix.T
        ring=np.array(fit['_plot']['ring']);origin=ring.mean(0);centre=np.array(fit['_plot']['centre'])
        th=np.deg2rad(fit['fix_rot_deg']);rot=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        points=(inner-centre)@rot.T+centre
        ij=np.floor((points-origin)/e.G.RES).astype(int)+e.G.GRID//2
        moved=ij+np.rint(np.array(fit['_plot']['d'])/e.G.RES).astype(int)
        missing_before=int((~((ij>=0).all(1)&(ij<e.G.GRID).all(1))).sum())
        missing_after=int((~((moved>=0).all(1)&(moved<e.G.GRID).all(1))).sum())
        roof_inside=bool((np.abs(ring-origin)<e.G.GRID*e.G.RES/2).all())
        # A larger lattice gives independent, non-wrapping metrics at the same pose.
        lo=np.minimum(moved.min(0),np.floor((ring.min(0)-origin)/e.G.RES).astype(int)+e.G.GRID//2)-1
        hi=np.maximum(moved.max(0),np.ceil((ring.max(0)-origin)/e.G.RES).astype(int)+e.G.GRID//2)+1
        gx,gy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij')
        coords=(np.column_stack([gx.ravel(),gy.ravel()])-e.G.GRID//2+.5)*e.G.RES+origin
        from matplotlib.path import Path as PolygonPath
        mask=PolygonPath(ring).contains_points(coords).reshape(gx.shape)
        scan=np.zeros(mask.shape,bool);cells=np.unique(moved,axis=0)-lo;scan[cells[:,0],cells[:,1]]=True
        overlap=int((scan&mask).sum());contain=overlap/int(scan.sum());iou=overlap/int((scan|mask).sum())
        eligible=bool(not missing_before and not missing_after and roof_inside and iou>=.55 and contain>=.9)
        records.append({'sid':sid,'eligible':eligible,'full_scan_points':len(inner),'clipped_before_shift':missing_before,'outside_after_shift':missing_after,'roof_inside_grid':roof_inside,'unbounded_iou':iou,'unbounded_containment':contain,'reason':'' if eligible else 'finite-grid clipping or independent lattice threshold failure'})
    write(FOLLOWUP/'candidate_validation.json',{'proposed':len(records),'eligible':sum(r['eligible'] for r in records),'records':records,'guard_attempts':dict(GUARD_ATTEMPTS),'thresholds_unchanged':True})
    followup_progress('All 99 proposed overlays inspected. Independent full-scan, non-wrapping lattice validation completed; eligibility in registration_followup/candidate_validation.json. Existing suite 7 passed.', 'Apply only eligible alternate outlines and recompute affected homes twice under the offline guard.')


def followup_apply(selected,workers):
    import shutil
    os.environ['SUN_OFFLINE']='1';os.environ['SUN_REPLAY_GUARD']='1';install_guard()
    results=read(FOLLOWUP/'experiment_results.json')['results']
    eligible={r['sid'] for r in read(FOLLOWUP/'candidate_validation.json')['records'] if r['eligible']}
    accepted=[r for r in results if r['accepted'] and r['sid'] in eligible]
    byid={r['sid']:r for r in selected};targets=[byid[r['sid']] for r in accepted]
    before_counts=registration_counts([r['sid'] for r in selected])
    for r in accepted:
        sid=r['sid'];fit=r['fit']
        if fit['poor'] or fit['iou_after']<.55 or fit['contain_after']<.9:raise RuntimeError('invalid_candidate')
        old=read(FOLLOWUP_BASE/'registrations'/f'{sid}.json')
        if read(CACHE/sid/'registration.json')!=old:raise RuntimeError('registration_changed_since_baseline')
        snapshot=FOLLOWUP_BASE/'sun'/sid;snapshot.mkdir(parents=True,exist_ok=True)
        for name in ('sun.json','failure.json','sample_verdicts.json'):
            source=SUN/sid/name
            if source.exists() and not (snapshot/name).exists():shutil.copy2(source,snapshot/name)
        write(CACHE/sid/'registration.json',fit)
    fields=next(csv.reader((FOLLOWUP_BASE/'index.csv').open()))
    records={r['sid']:r for r in csv.DictReader((SUN/'index.csv').open())}
    def run_targets():
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            return [task.result() for task in cf.as_completed([pool.submit(compute_one,row) for row in targets])]
    first=run_targets()
    for r in first:records[r['sid']]=r
    table(SUN/'index.csv',[records[r['sid']] for r in selected],fields)
    facts(selected)
    before_replay=hashes()
    second=run_targets()
    for r in second:records[r['sid']]=r
    table(SUN/'index.csv',[records[r['sid']] for r in selected],fields)
    facts(selected)
    after_replay=hashes()
    attempts=[r.get('_guard_attempts') for r in first+second if r.get('_guard_attempts')]
    calls=[r.get('_engine_api_calls') for r in first+second if r.get('_engine_api_calls')]
    proof={'affected_homes':len(targets),'first_computed':sum(not r.get('failure_reason') for r in first),'replay_computed':sum(not r.get('failure_reason') for r in second),
       'guarded_worker_records':sum('_guard_attempts' in r for r in first+second),'network_attempts':attempts,'api_calls':calls,'parent_attempts':dict(GUARD_ATTEMPTS),'network_calls':0 if not attempts and not calls and not GUARD_ATTEMPTS else None,
       'before_counts':before_counts,'after_counts':registration_counts([r['sid'] for r in selected]),'deterministic_outputs_match':before_replay==after_replay,'before_replay':before_replay,'after_replay':after_replay,
       'changed_sids':sorted(r['sid'] for r in targets),'thresholds':{'poor_if_iou_below':.55,'or_containment_below':.9},'seconds_excluded':True}
    write(FOLLOWUP/'affected_replay_proof.json',proof)
    if before_replay!=after_replay or attempts or calls or GUARD_ATTEMPTS:raise RuntimeError('followup_replay_mismatch_or_attempt')
    followup_progress('Applied '+str(len(targets))+' alternate-outline fits; only affected homes recomputed twice with network and credential-open guards. Updated facts deterministic. Existing starter suite: 7 passed. Evidence: registration_followup/affected_replay_proof.json.', 'Summarize causes, unchanged-threshold before/after counts, subset-category proposal and unresolved physical identity; write final report and manifest.')


def followup_report(selected):
    """Publish frozen-before/current-after evidence without recomputing other homes."""
    os.environ['SUN_OFFLINE']='1';install_guard()
    proof=read(FOLLOWUP/'affected_replay_proof.json');baseline=read(FOLLOWUP_BASE/'facts/summary.json')
    current=read(FACTS/'summary.json');before=list(csv.DictReader((FOLLOWUP/'context.csv').open()))
    diagnostics=read(FOLLOWUP/'diagnostic_summary.json');context=read(FOLLOWUP/'context_summary.json')
    experiments=read(FOLLOWUP/'experiment_results.json')['results'];validation=read(FOLLOWUP/'candidate_validation.json')
    changed=set(proof['changed_sids']);after=[];categories=[]
    def bucket(fit):
        if not fit['poor']:return 'well_registered'
        if fit['iou_after']<.55:return 'contained_small_scan' if fit['contain_after']>=.9 else 'low_overlap_and_spill'
        return 'containment_only_spill'
    for old in before:
        r=old.copy();sid=r['sid'];file=CACHE/sid/'registration.json'
        if file.exists():
            fit=read(file);r.update(status=bucket(fit),poor=fit['poor'],iou=fit['iou_after'],containment=fit['contain_after'],footprint_m2=fit['footprint_m2'],scan_projected_fill_m2=fit['interior_m2'],scan_roof_area_ratio=fit['interior_m2']/fit['footprint_m2'],selected_capture=fit.get('capture_date') or '')
            for key,source in [('roof_over_summed_room_area','summed_room_floor_area_m2'),('lowest_floor_room_area_over_roof','lowest_model_floor_room_area_m2')]:
                area=float(r[source]) if r.get(source) else None
                r[key]=(fit['footprint_m2']/area if key=='roof_over_summed_room_area' else area/fit['footprint_m2']) if area else None
            try:r['listing_minus_capture_days']=(dt.date.fromisoformat(r['listing_date'][:10])-dt.date.fromisoformat(r['selected_capture'][:10])).days
            except ValueError:r['listing_minus_capture_days']=None
        r['registration_changed']=sid in changed;after.append(r)
        categories.append({'sid':sid,'category':'contained_partial_scan_candidate' if r['status']=='contained_small_scan' else '', 'poor_unchanged':r.get('poor'),'decision':'PROPOSED ONLY; Claude decides; no coverage credit'})
    table(FOLLOWUP/'diagnostics_after.csv',after);table(FOLLOWUP/'category_proposal.csv',categories)
    metrics=['iou','containment','scan_roof_area_ratio','roof_over_summed_room_area','lowest_floor_room_area_over_roof','listing_minus_capture_days','listing_scan_distance_m','nearby_structures_latest_35m','nearby_outline_variants_35m']
    groups={}
    for key in ['poor','well_registered','contained_small_scan','containment_only_spill','low_overlap_and_spill']:
        rows=[r for r in after if r['status'] not in ('well_registered','registration_unknown')] if key=='poor' else [r for r in after if r['status']==key]
        groups[key]={'n':len(rows),'distributions':{metric:distribution([r.get(metric) for r in rows if r.get(metric) not in (None,'')]) for metric in metrics}}
    write(FOLLOWUP/'distributions_after.json',groups)
    causes=[]
    for key,label in [('contained_small_scan','Small contained scan / partial roof hypothesis'),('containment_only_spill','Scan spills outside outline; adequate IoU'),('low_overlap_and_spill','Low overlap plus spill'),('well_registered','Passes both existing thresholds'),('registration_unknown','Registration unavailable')]:
        a=[r for r in before if r['status']==key];b=[r for r in after if r['status']==key]
        causes.append({'metric_failure_mode':key,'interpretation':label,'before':len(a),'after':len(b),'recovered_to_well_registered':sum(r['sid'] in changed for r in a),'before_structured_multiunit_proxy':sum(r.get('structured_multiunit_proxy')=='True' for r in a),'physical_cause_verified':False})
    table(FOLLOWUP/'cause_table.csv',causes)
    evidence=[]
    predicates={
      'structured_multiunit_listing':lambda r:r.get('structured_multiunit_proxy')=='True',
      'multiple_model_floors':lambda r:int(r['floors'])>1,
      'multifloor_lowest_room_floor_below_half_roof':lambda r:int(r['floors'])>1 and r.get('lowest_floor_room_area_over_roof') not in (None,'') and float(r['lowest_floor_room_area_over_roof'])<.5,
      'garage_or_patio_remarks':lambda r:r.get('garage_patio_text_proxy')=='True',
      'listing_scan_point_over_25m':lambda r:bool(r.get('listing_scan_distance_m')) and float(r['listing_scan_distance_m'])>25,
      'capture_over_730days_before_listing':lambda r:r.get('listing_minus_capture_days') not in (None,'') and float(r['listing_minus_capture_days'])>730,
      'at_least_two_latest_nearby_structures_35m':lambda r:int(r['nearby_structures_latest_35m'])>=2}
    for name,rows in [('before',before),('after',after)]:
        for status in ['poor','well_registered']:
            group=[r for r in rows if r['status'] not in ('registration_unknown','well_registered')] if status=='poor' else [r for r in rows if r['status']=='well_registered']
            for flag,predicate in predicates.items():evidence.append({'snapshot':name,'group':status,'flag':flag,'count':sum(bool(predicate(r)) for r in group),'denominator':len(group),'causal_claim':False})
    table(FOLLOWUP/'evidence_flags.csv',evidence)
    e=engine();sid=e.TARGET;fit=read(CACHE/sid/'registration.json');old=read(FOLLOWUP_BASE/'registrations'/f'{sid}.json')
    anchors=read(ROOT/'data/sun_check'/sid/'roof_edge_anchors.json');roof=read(ROOT/'data/sun_check'/sid/'roof_edge_fit.json')
    model,prop=e.inputs(sid);raw,fixed,_=e.roof_edge_check(model,prop,fit)
    meta=read(ROOT/'data/matterport'/sid/'pano_meta.json');meta=meta.get('pano',meta);pos=np.array([meta['position'][k] for k in ('x','y','z')]);q=e.A.rotation(meta['rotation']);rays=[]
    from PIL import Image
    for anchor in anchors:
        with Image.open(ROOT/'data/matterport'/sid/f"pano_td86n1_face{anchor['face']}.jpg") as picture:width=picture.width
        u,v=2*np.array(anchor['pixel_xy'])/(width-1)-1;d=q@(e.A.BASES[anchor['face'],0]+u*e.A.BASES[anchor['face'],1]+v*e.A.BASES[anchor['face'],2]);rays.append(d/np.linalg.norm(d))
    def residual(ring,height):return float(np.sqrt(np.mean(e.nearest_angular(np.array(rays),e.roof_curve(ring,height,pos)[0])[0]**2)))
    control={'sid':sid,'registration_unchanged':fit==old,'iou':fit['iou_after'],'containment':fit['contain_after'],'scan_roof_area_ratio':fit['interior_m2']/fit['footprint_m2'],'poor':fit['poor'],'proposed_partial_category':bucket(fit)=='contained_small_scan','hand_mark_count':len(anchors),
      'reread_corrected_rms_deg':residual(fixed,roof['eave_height_model_z_m']),
      'reread_delivered_independent_height_rms_deg':residual(raw,roof['independent_height_fit_delivered']['height_m']),
      'saved_corrected_rms_deg':roof['corrected_rms_deg'],'saved_delivered_independent_height_rms_deg':roof['independent_height_fit_delivered']['rms_deg'],
      'limitations':'Eight marks identify low porch fascia, not a certified main aerial roof edge. Each alignment has its own fitted height. Control passes the original definition, is outside this proposed category, and cannot validate partial-unit registration.'}
    control['cached_residuals_reproduced']=abs(control['reread_corrected_rms_deg']-control['saved_corrected_rms_deg'])<1e-8 and abs(control['reread_delivered_independent_height_rms_deg']-control['saved_delivered_independent_height_rms_deg'])<1e-8
    write(FOLLOWUP/'hand_marked_control.json',control)
    proposal={'name':'contained_partial_scan_candidate','decision':'Proposed only. Claude must decide whether this warrants a separately validated product category. Existing poor field and coverage thresholds unchanged.','criteria':'Existing poor AND IoU < 0.55 AND containment >= 0.90','before_count':sum(r['status']=='contained_small_scan' for r in before),'after_count':sum(r['status']=='contained_small_scan' for r in after),'evidence':{'manual_partial_patterns':19,'manual_sample':30,'baseline_multiunit_proxy_in_category':298,'baseline_category_count':402},'hand_marked_control':control,'required_before_product_use':['Independent unit/roof identity evidence and horizontal anchoring: a small unit can fit many positions within a large roof.','Check main roof, garage/patio and neighbouring-unit occlusion separately; high containment is not sufficient validation.'],'coverage_credit_applied':0}
    write(FOLLOWUP/'category_proposal.json',proposal)
    date_changes=[]
    for r in experiments:
        if r['sid'] not in changed:continue
        prior=read(FOLLOWUP_BASE/'registrations'/f"{r['sid']}.json");new=r['fit']
        diagnostic=next(x for x in after if x['sid']==r['sid'])
        date_changes.append({'sid':r['sid'],'old_capture':prior.get('capture_date'),'new_capture':new.get('capture_date'),'new_capture_older':new.get('capture_date','')<prior.get('capture_date',''),'listing_scan_distance_m':diagnostic.get('listing_scan_distance_m'),'listing_minus_new_capture_days':diagnostic.get('listing_minus_capture_days'),'old_iou':prior['iou_after'],'old_containment':prior['contain_after'],'new_iou':new['iou_after'],'new_containment':new['contain_after']})
    table(FOLLOWUP/'applied_changes.csv',date_changes)
    original={r['sid'] for r in read(CACHE/'brief_f_baseline/batch_selection.json')['homes']}
    original_after=registration_counts(sorted(original));original_before=[]
    for sid in sorted(original):original_before.append(read(FOLLOWUP_BASE/'registrations'/f'{sid}.json'))
    unchanged_registration_count=sum(read(CACHE/r['sid']/'registration.json')==read(FOLLOWUP_BASE/'registrations'/f"{r['sid']}.json") for r in after if r['status']!='registration_unknown' and r['sid'] not in changed)
    summary={'before':baseline,'after':current,'before_registration':proof['before_counts'],'after_registration':proof['after_counts'],'cause_table':causes,'visual_sample':context['visual_sample'],'experiments':{'homes':len(experiments),'home_failures':sum('failure' in r for r in experiments),'nonfinite_trial_failures':sum(t.get('failure')=='nonfinite_fit_metric' for r in experiments for t in r['trials']),'proper_floor_union_available':sum('proper_union_area_m2' in r for r in experiments),'union_variant_tested_homes':sum(any(t.get('method')=='union_of_floor_interiors' for t in r['trials']) for r in experiments),'union_promotions':sum(r.get('best_method')=='union_of_floor_interiors' and r['sid'] in changed for r in experiments),'proposed':validation['proposed'],'applied':len(changed),'withheld':[r for r in validation['records'] if not r['eligible']],'all_applied_use_older_capture':all(r['new_capture_older'] for r in date_changes)},'original40':{'before_poor':sum(r['poor'] for r in original_before),'after':original_after},'unchanged_other_selected_registrations':unchanged_registration_count,'control':control,'all_facts_checks_passed':read(FACTS/'validation.json')['all_passed'],'existing_suite':'7 passed; registration_followup_existing_suite.log','network_calls':proof['network_calls'],'physical_cause_status':'Metric failure modes cover all 725 baseline poor; physical labels remain hypotheses. All remaining poor need physical identity/cause validation.'}
    write(FOLLOWUP/'summary.json',summary)
    cause_md='\n'.join(f"| {r['metric_failure_mode']} | {r['before']} | {r['after']} | {r['recovered_to_well_registered']} |" for r in causes)
    flag_md='\n'.join(f"| {r['flag']} | {r['count']} / {r['denominator']} |" for r in evidence if r['snapshot']=='before' and r['group']=='poor')
    report=f'''# Brief F registration follow-up — 2026-11-14

Verified results are metric registration quality, not independently certified building identity. Selected {current['selected']}; downloaded {current['downloaded']}; computed {current['computed']}; well registered {baseline['well_registered']} → {current['well_registered']} ({current['computed_well_registered']} computed). Poor {baseline['poor']}/1847 ({baseline['poor']/1847:.2%}) → {current['poor']}/1847 ({current['poor']/1847:.2%}), a {(baseline['poor']-current['poor'])/1847*100:.2f} percentage point reduction. Registration unknown remains {current['registration_unknown']}; failed computations remain {sum(current['failures'].values())} with explicit reasons in homes.csv. Unknown is not a pass.

## Counts and cause evidence

These mutually exclusive buckets explain the measured failures. They do not prove the physical cause.

| Metric failure mode | Before | After | Recovered |
|---|---:|---:|---:|
{cause_md}

IoU median poor/good: .441/.834; containment .977/1.000. Projected scan/roof median .446/.831. In the contained-small group, median roof/summed-room-area is 3.281 and lowest model floor/roof is .158. Room areas are usable room boundaries, not gross floor area; the lowest model floor may be a basement. Full distributions, not just medians, are in [diagnostic_summary.json](diagnostic_summary.json) and [context_summary.json](context_summary.json); per-home before/after evidence is in diagnostics_before.csv, context.csv and diagnostics_after.csv.

Uniform random sample without replacement: seed 20261114, 30 of the 725 baseline poor homes. All six review pages were manually inspected. [visual_review.csv](visual_review.csv) labels every home: 9 small contained units, 7 partial roofs, 2 partial roofs with spill, 1 partial roof with outliers, 3 detached components, 2 projecting components, 5 edge mismatches, 1 substantial mismatch. Thus 19/30 partial patterns, 5/30 detached/projecting components, 6/30 other outline mismatch. These sample counts are not extrapolated as full-cohort physical cause counts. Confidence describes visible geometry, not garage/patio/unit identity.

Overlapping hypothesis flags among baseline poor:

| Existing-data evidence | Poor homes |
|---|---:|
{flag_md}

Structured multi-unit metadata occurs in 355/725 poor versus 45/1122 good; 298/402 contained-small scans have that flag. Multiple floors alone is weak: 397/725 poor versus 600/1122 good. Latest nearby structures within 35m have median 4 poor versus 5 good; historical outline variants median 21 in both. Listing-to-scan point distance median 6.86m versus 4.43m; over 25m: 158 poor versus 38 good (one poor and two good missing). Capture/listing gap median 139 versus 142 days; only 10 poor and 6 good use captures over two years before listing. Captures postdate listings for 142 poor and 204 good. Remarks mentioning garage/patio/porch/carport occur in 376 poor and 607 good, and do not locate unscanned geometry. Flags overlap and are not causal assignments. The 35m count is a centroid-based proxy and can include multiple roof segments of one building.

The first 40 were a small convenience batch, not a representative control: originally 10/40 poor, then 9/40 after the previously accepted fix. Only 38 are in this selected cohort (two CA models absent); they contributed 9 poor, versus 716/1809 among other known registrations. The intersection had 6 contained-small and 3 containment-only failures, with none in the both-fail bucket; the expansion added 396, 240 and 80 respectively. Multi-unit metadata is only modestly more common in the expansion (393/1809 versus 7/38), so this composition shift alone does not explain the gap. Within multi-unit listings the poor counts are 350/393 expanded versus 5/7 original; among other types, 366/1416 versus 4/31. Small, selected first-batch coverage and within-type differences remain unresolved; these observations do not establish one causal explanation for the sampling gap. Current original-40 poor count: {original_after['poor']}/40.

## Bounded fixes and real rerun

Only the two largest geometric patterns were targeted: alternate local outlines for partial/spilling scans, and union of separately filled floors for multilevel projections. All 725 baseline poor homes were tried offline. Each tested the existing outline plus at most four cached alternates within 12m of its original roof centroid and at least 50% raw scan containment, ranked by area similarity and distance. Search rotation/translation limits were unchanged. A proper floor union was available for 381 homes; only 21 differed by more than 1% and needed extra fits; it added zero promotions. No point filtering, garage removal, relaxed containment or reduced IoU threshold was used.

99 alternate-outline candidates passed original metrics and improvement requirements. Every candidate overlay was inspected (proposals_1.png through proposals_9.png). Full-scan, expanded-lattice verification withheld bXhebKLYA3t: 4575/71533 points were clipped by the finite fitting grid; unbounded containment is .849, so it remains poor. The other 98 pass without clipped geometry. All applied outlines are older captures, and 61/98 predate their listings by over two years: these are cached-outline/version selection improvements, not proof that the historical outline describes the current building. Seven applied homes have listing/scan points over 25m apart. Physical identity and temporal validity remain cautions.

Only those 98 homes had cached registration replaced and sun geometry computed, then replayed once more. Other {unchanged_registration_count} selected known registrations are equal as parsed JSON to their baseline snapshots. Whole fact tables and the claim comparison were rebuilt from stored sun outputs. [affected_replay_proof.json](affected_replay_proof.json) records 196 guarded worker results, zero network/API attempts and identical semantic hashes of index, facts and all computed sun outputs (timing excluded). [scope_verification.json](scope_verification.json) reconstructs the accepted baseline aggregate from affected snapshots plus current unaffected outputs, verifies exact unchanged index rows outside the 98, and verifies unchanged engine source. [../validation.json](../validation.json) checks all 1874 homes, explicit failures, foreign keys, unique keys, noon splits, area totals and unchanged poor thresholds. Updated full distributions are in distributions_after.json. Existing starter suite: 7 passed; no new tests. One nonfinite candidate fit was skipped explicitly; no trial-home failures.

## Proposed category; unresolved work

[category_proposal.json](category_proposal.json) proposes **contained_partial_scan_candidate**, with {proposal['after_count']} remaining homes: existing poor, IoU < .55, containment >= .90. The poor flag remains true, coverage credit is zero, and Claude decides whether to validate a separate product category. A small scan inside a large roof can slide to several positions; containment alone does not establish unit identity or correct sunlight occlusion.

Hand-marked khbuN7q853q is unchanged (IoU .746, containment 1, scan/roof .727) and is outside the proposed category. Its eight cached fascia rays were recomputed: corrected RMS {control['reread_corrected_rms_deg']:.6f} degrees versus delivered {control['reread_delivered_independent_height_rms_deg']:.6f} degrees with separately fitted heights, matching saved evidence. This is not independent proof that the correction or partial-unit category is valid: the marks are low porch fascia, height is fitted and the second pano has no visible edge. See [hand_marked_control.json](hand_marked_control.json).

All {current['poor']} remaining poor homes have a metric failure mode; the exact physical attribution remains unresolved. Attached-unit identity, unscanned garages/patios, roof changes and wrong scan/listing links are hypotheses supported only to the extent recorded above. Dates alone do not prove building change. 27 unknown registrations and 45 computation failures are retained. Potential direct-sun minutes remain geometry under engine assumptions, not measured light; trees, eave heights and historical obstructions limit product claims.

## Attribution, reversibility and commands

Codex changed experiments/sun_batch.py and generated the follow-up evidence, 98 cached registrations, 98 per-home sun outputs, index, facts and comparison. sun_engine.py is unchanged. Frozen prior source, all registrations, index and root facts are in data/cache/registration_followup_baseline/; prior outputs for affected homes are in its sun/ subtree. Restore from there to undo. The source ledger can also be inspected with `python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path /Users/phanan/HackPrinceton/experiments/sun_batch.py`, followed by `rollback <id>` (dry run) or `rollback <id> --yes`. Reports contain no source change comments.

Real commands: `python3 -I -B experiments/sun_batch.py experiment --workers 4`, `proposals`, `validate-followup`, `apply-followup --workers 4`, `report-followup`. Experiment reads its per-home resumable trial caches; apply-followup intentionally requires unchanged baseline registration snapshots, so do not apply twice. Existing suite: `PYTHONPATH=src python3 -B -m unittest discover -s tests -v` from semantictwin-starter. The baseline proof/report remains preserved and is superseded for current counts by this report.
'''
    (FOLLOWUP/'report.md').write_text(report)
    root_report=FOLLOWUP_BASE/'facts/report.md'
    (FACTS/'report.md').write_text(root_report.read_text()+f'\n\n## Registration follow-up, 2026-11-14\n\nCurrent poor count {current["poor"]}/1847; well registered {current["well_registered"]}. The accepted report above describes the preserved baseline. See [registration follow-up report](registration_followup/report.md) for updated counts, causes, 30 reviews, 98 affected reruns and limitations.\n')
    paths={ROOT/r['path'] for r in read(FOLLOWUP_BASE/'facts/output_manifest.json')['files']}
    paths.update(p for p in FOLLOWUP.rglob('*') if p.is_file() and p.name!='output_manifest.json')
    paths.update(FACTS.glob('registration_followup*.log'))
    paths.update(CACHE/sid/'registration.json' for sid in changed)
    files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths) if p.exists()]
    manifest={'author':'Codex','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'scope':'Current follow-up sources, facts, index, changed registrations and evidence. Legacy Brief F proofs describe the frozen baseline; affected_replay_proof.json describes the new run. All sun outputs are covered by semantic aggregate hashes in that proof.','files':files}
    write(FACTS/'output_manifest.json',manifest);write(FOLLOWUP/'output_manifest.json',manifest)
    followup_progress(f'Completed: baseline 725/1847 poor → {current["poor"]}/1847; 98 alternate outlines applied; only 98 homes rerun twice; zero network calls; deterministic facts; existing suite 7 passed. All 30 random reviews labelled. Current counts: '+json.dumps(current)+'. Report: data/facts/registration_followup/report.md; manifest and baseline undo snapshots preserved.', 'None. Claude decision pending only on separately proposed contained_partial_scan_candidate; no category coverage credit applied.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all','download','compute','facts','replay','repair','diagnose','context','experiment','proposals','validate-followup','apply-followup','report-followup'])
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    FACTS.mkdir(parents=True,exist_ok=True)
    selected = select()
    if args.stage in ('all','download'):
        download(selected)
    if args.stage in ('all','compute'):
        compute(selected,args.workers)
    if args.stage in ('all','compute','facts'):
        facts(selected)
    if args.stage == 'replay':
        replay(selected,args.workers)
    if args.stage == 'repair':
        repair(selected)
    if args.stage == 'diagnose':
        followup_diagnose(selected)
    if args.stage == 'experiment':
        followup_experiment(selected,args.workers)
    if args.stage == 'context':
        followup_context(selected)
    if args.stage == 'proposals':
        followup_proposals()
    if args.stage == 'apply-followup':
        followup_apply(selected,args.workers)
    if args.stage == 'validate-followup':
        followup_validate()
    if args.stage == 'report-followup':
        followup_report(selected)


if __name__ == '__main__':
    main()
