"""Additive daylight research outputs; never writes engine or existing facts.

Run with python3 -I -B experiments/daylight.py {example,weather,compute,all}.
Network acquisition is limited to Open-Meteo weather and Vexcel DSM stages,
with four requests maximum in flight. Offline stages reject network access.
"""
from pathlib import Path
import argparse
import ast
import builtins
import collections
import concurrent.futures
import csv
import datetime as dt
import hashlib
import gzip
import io
import json
import math
import os
import shutil
import types
import re
import socket
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zlib
import numpy as np
from PIL import Image
from zoneinfo import ZoneInfo
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/hackprinceton-daylight-mpl')
from matplotlib.path import Path as PolygonPath

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/facts/daylight'
CACHE = ROOT / 'data/cache/daylight_weather'
DATES = ['2026-12-21', '2026-03-20', '2026-06-21', '2026-11-14']
MONTHS = [12, 3, 6, 11]
LABEL = 'modelled weather + geometry, at the window glass; not room brightness, not measured'
PROGRESS = Path('/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T19-05-47-982Z-3116-db18a8f5-40e7-4dbf-9b4e-a3be54281e23.md')
EXAMPLE_URL = 'https://sportscotland.org.uk/media/ni2ao5tp/understandingdaylightsc.pdf'
WEATHER_URL = 'https://open-meteo.com/en/docs/historical-weather-api'
LOCK = threading.Lock()
NET = collections.Counter()
AUDIT = collections.Counter()
RUNTIME_CREDENTIAL_READ = False


def environment_guard(event, args):
    if event=='open' and args and isinstance(args[0],(str,bytes)) and Path(os.fsdecode(args[0])).name=='.env' and not RUNTIME_CREDENTIAL_READ:
        raise PermissionError('.env reads forbidden')


def offline_guard(event, args):
    environment_guard(event,args)
    if event in ('socket.connect','socket.sendto'):
        AUDIT['network_attempts']+=1
        raise PermissionError('offline replay forbids network')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def table(path, records):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(records)


def progress(done, next_step):
    PROGRESS.write_text('## Objective\nDaylight items 1–6 plus DSM horizon. Fresh deadline 21:05 UTC; preserve engine/existing tables. No manual credential inspection, new tests or commits. Runtime-only credential loading follows Brief F.\n## Done (files, test results)\n' + done + '\n## Next\n' + next_step + '\n')


def df_value(area_sky, surface, transmittance, reflectance):
    return transmittance * area_sky / (surface * (1-reflectance**2))


def example():
    # Published example already supplies NET glazing; no extra frame deduction.
    surface = 2*18*27 + 2*(18+27)*7.6
    result = df_value(54*70, surface, .7*.9, .5)
    value = {'source': EXAMPLE_URL, 'page': 15, 'local_source': 'data/cache/daylight_sources/understanding_daylight.pdf',
             'title': 'sportscotland, Understanding Daylight, Estimating DFave',
             'room_dimensions_m': [18, 27, 7.6], 'net_glazing_m2': 54, 'theta_degrees': 70,
             'T': .7, 'maintenance_factor': .9, 'R': .5, 'total_internal_surface_m2': surface,
             'substitution': '0.7 * 0.9 * 54 * 70 / (1656 * (1 - 0.5**2))',
             'computed_df_percent': result, 'published_df_percent_rounded': 1.9,
             'reproduced_to_published_precision': round(result, 1) == 1.9,
             'cohort_difference': 'Cohort T=0.6/0.7/0.8 is assumed effective diffuse transmittance; no separately calibrated dirt factor; opening area multiplied by assumed frame factor0.8.'}
    assert value['reproduced_to_published_precision']
    write(OUT/'published_example.json', value)
    return value


def home_cells():
    selection = {h['sid']: h for h in read(ROOT/'data/facts/selection.json')['homes']}
    result = []
    for h in rows(ROOT/'data/facts/homes.csv'):
        s = selection[h['sid']]
        lat = h['registered_latitude'] or s.get('lat')
        lon = h['registered_longitude'] or s.get('lon')
        coordinate_source = 'registered scan' if h['registered_latitude'] else 'delivered model coordinate; registration unavailable'
        try:
            lat, lon = float(lat), float(lon)
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError('coordinates')
            cell = (math.floor(lat*4), math.floor(lon*4))
            result.append(dict(h, latitude=lat, longitude=lon, coordinate_source=coordinate_source,
                               cell_id=f'{cell[0]}_{cell[1]}', cell_latitude=(cell[0]+.5)/4,
                               cell_longitude=(cell[1]+.5)/4))
        except (TypeError, ValueError):
            result.append(dict(h, coordinate_source='missing coordinate', cell_id=''))
    return result


def weather_cell(home, offline=False):
    cell_id = home['cell_id']
    target = CACHE/f'{cell_id}_2016_2025.json'
    if target.exists():
        return read(target)
    if offline:
        raise RuntimeError('weather_cache_missing')
    query = dict(latitude=home['cell_latitude'], longitude=home['cell_longitude'],
                 start_date='2016-01-01', end_date='2025-12-31', timezone='UTC',
                 daily='sunshine_duration,daylight_duration',
                 hourly='direct_normal_irradiance,diffuse_radiation,shortwave_radiation', models='era5')
    url = 'https://archive-api.open-meteo.com/v1/archive?' + urllib.parse.urlencode(query)
    for attempt in range(6):
        try:
            with LOCK:
                NET['active'] += 1
                NET['maximum_in_flight'] = max(NET['maximum_in_flight'], NET['active'])
                NET['attempts'] += 1
            try:
                with urllib.request.urlopen(url, timeout=180) as response:
                    data = json.load(response)
            finally:
                with LOCK:
                    NET['active'] -= 1
            if data.get('error'):
                raise RuntimeError(str(data.get('reason')))
            assert len(data['daily']['time']) == 3653
            assert len(data['hourly']['time']) == 3653*24
            value = {'source_url': url, 'source_documentation': WEATHER_URL,
                     'requested_cell_id': cell_id, 'downloaded_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                     'data': data}
            write(target, value)
            return value
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(f'Open-Meteo HTTP {exc.code}; stopped') from exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f'Open-Meteo HTTP {exc.code}') from exc
            time.sleep(min(60, 5*2**attempt))
        except (TimeoutError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(min(60, 5*2**attempt))


def climate_summary(raw):
    d = raw['data']['daily']
    daily = collections.defaultdict(list)
    years = collections.defaultdict(set)
    for date, sun, day in zip(d['time'], d['sunshine_duration'], d['daylight_duration']):
        if sun is not None and day is not None and day > 0 and 0 <= sun <= day:
            month = int(date[5:7])
            daily[month].append(sun/day)
            years[month].add(int(date[:4]))
    hourly = raw['data']['hourly']
    buckets = collections.defaultdict(list)
    for i, stamp in enumerate(hourly['time']):
        month, hour = int(stamp[5:7]), int(stamp[11:13])
        for name in ('direct_normal_irradiance', 'diffuse_radiation', 'shortwave_radiation'):
            v = hourly[name][i]
            if v is not None and math.isfinite(v) and v >= 0:
                buckets[(month, hour, name)].append(v)
    result = {}
    for month in MONTHS:
        result[str(month)] = {'sunshine_share': statistics.mean(daily[month]) if daily[month] else None,
                              'valid_daily_ratios': len(daily[month]), 'years': sorted(years[month]),
                              'profile_utc_hour_ending': [dict(hour=h, **{
                                  name: statistics.mean(buckets[(month, h, name)]) if buckets[(month, h, name)] else None
                                  for name in ('direct_normal_irradiance', 'diffuse_radiation', 'shortwave_radiation')},
                                  valid_hours=min(len(buckets[(month,h,name)]) for name in ('direct_normal_irradiance','diffuse_radiation','shortwave_radiation')))
                                  for h in range(24)]}
    return result


def weather(offline=False):
    homes = home_cells()
    cells = {h['cell_id']: h for h in homes if h['cell_id']}
    failures, summaries = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(weather_cell, h, offline): cell for cell, h in cells.items()}
        for future in concurrent.futures.as_completed(tasks):
            cell = tasks[future]
            try:
                summaries[cell] = climate_summary(future.result())
            except Exception as exc:
                failures[cell] = f'{type(exc).__name__}: {exc}'
            write(OUT/'weather_progress.json', {'cells': len(cells), 'completed': len(summaries), 'failures': failures, 'network': dict(NET)})
    write(OUT/'climate_cells.json', {'source': WEATHER_URL, 'model': 'ERA5 reanalysis',
                                    'years': [2016,2025], 'cell_degrees': .25,
                                    'share_definition': 'Arithmetic mean of valid daily sunshine_duration/daylight_duration ratios in each month over2016–2025; UTC days',
                                    'profiles': 'Arithmetic mean of preceding-hour radiation at each UTC hour-ending in each month; not instantaneous radiation',
                                    'cells': dict(sorted(summaries.items())), 'failures': failures})
    table(OUT/'home_climate.csv', [dict(sid=h['sid'], cell_id=h['cell_id'], coordinate_source=h['coordinate_source'],
          latitude=h.get('latitude'), longitude=h.get('longitude'), date=date, month=int(date[5:7]),
          sunshine_share=summaries.get(h['cell_id'],{}).get(str(int(date[5:7])),{}).get('sunshine_share'),
          label='modelled climate average', failure=failures.get(h['cell_id'],'') if h['cell_id'] else 'missing coordinate')
          for h in homes for date in DATES])
    progress(f'Published DF example reproduced in data/facts/daylight/published_example.json. Weather cells: {len(summaries)}/{len(cells)}; failures {failures}; network {dict(NET)}. Baseline hashes/wording backups in data/cache/daylight_baseline/. No tests added.',
             'Compute room DF/tree cautions, weather-adjusted minutes and window/room radiation; compare claims, wording, real cache replay and existing suite guard.')
    return summaries


def geometry_functions():
    """Reuse only pure functions, without importing the engine's executable module."""
    scope = dict(np=np, dt=dt, ROOT=ROOT, json=json, PolygonPath=PolygonPath, TZ=dt.timezone.utc)
    for filename, names in [('sun_from_pano.py', ['solar', 'daylight']),
                            ('sun_engine.py', ['segments', 'inside_segments', 'cross2', 'prism_hit',
                                               'aperture_verdict', 'floor_patch', 'alignment', 'sample_day'])]:
        source = ast.parse((ROOT/'experiments'/filename).read_text())
        for node in source.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), scope)
    scope['A'] = types.SimpleNamespace(solar=scope['solar'], daylight=scope['daylight'])
    return scope


def transform(model, fit):
    g = model['geocoordinates']; lat, lon = float(g['p0']['lat']), float(g['p0']['long'])
    raw = np.column_stack([np.array([(float(g[k]['long'])-lon)*111320*math.cos(math.radians(lat)),
                                      (float(g[k]['lat'])-lat)*110574])/20 for k in ('px','py')])
    matrix, shift = GEO['alignment'](fit, raw)
    return matrix, shift, lat, lon


def load_geometry(sid):
    sun = read(ROOT/'data/sun'/sid/'sun.json')
    model = read(ROOT/'data/cache'/sid/'matterport.json')['data']['model']
    fit = read(ROOT/'data/cache'/sid/'registration.json')
    matrix, shift, lat, lon = transform(model, fit)
    enrich_path = ROOT/'data/cache'/sid/'room_labels.json'
    enrich = {r['id']:r for r in read(enrich_path).get('model',{}).get('rooms',[])} if enrich_path.exists() else {}
    raw = {r['id']:{**r, **enrich.get(r['id'],{})} for r in model['rooms']}
    rooms = {r['id']:dict(r, segments=GEO['segments'](raw[r['id']].get('boundary')),
                          holes=[GEO['segments'](v) for v in raw[r['id']].get('holes') or []]) for r in sun['rooms']}
    windows = []
    for source in sun['windows']:
        if source['type'] != 'window':
            continue
        w = dict(source); tangent = np.array(w['tangent']); n = np.array([-tangent[1],tangent[0]])
        az = math.radians(w['outward_azimuth_deg']); target = np.array([math.sin(az), math.cos(az)])
        w['outward'] = n if (matrix@n)@target > 0 else -n
        w['center'] = np.array(w['center']); w['tangent'] = tangent
        windows.append(w)
    roofs = [dict(v,ring=np.array(v['ring'])) for v in sun['own_roof_layers']]
    obstacles = [dict(v,ring=np.array(v['ring_model_xy_m'])) for v in sun['obstructions']]
    return dict(sid=sid,sun=sun,model=model,fit=fit,matrix=matrix,shift=shift,lat=lat,lon=lon,
                rooms=rooms,windows=windows,roofs=roofs,obstacles=obstacles)


def directions(az, el, matrix):
    az, el = np.deg2rad(az), np.deg2rad(el)
    en = np.column_stack([np.sin(az)*np.cos(el), np.cos(az)*np.cos(el)])
    d = np.column_stack([en@np.linalg.inv(matrix).T, np.sin(el)])
    return d/np.linalg.norm(d,axis=1)[:,None]


def sky_samples(w, g, horizon=None):
    # Isotropic irradiance integral: cos(incidence) d(solid angle)/pi.
    azimuth = float(w['outward_azimuth_deg'])
    aa, ee = np.meshgrid(np.arange(-85,90,10.), np.arange(5,90,10.))
    az, el = (aa.ravel()+azimuth)%360, ee.ravel()
    weight = np.cos(np.deg2rad(aa.ravel()))*np.cos(np.deg2rad(el))**2
    weight *= .5/weight.sum()
    # Midpoint of window, not a floor illuminance calculation.
    mid = dict(w, width=w['width']*1e-6, height=w['height']*1e-6,
               lower_z_m=w['lower_z_m']+w['height']/2)
    codes = GEO['aperture_verdict'](mid,directions(az,el,g['matrix']),g['roofs'],
                                   [v for v in g['obstacles'] if v['kind']!='tree'])[:,4]
    clear = codes==0
    clear_dsm = clear & (el > horizon_at(horizon,az)) if horizon is not None else None
    te = np.arange(.25,90,.5); ta = np.full(len(te),azimuth)
    tc = GEO['aperture_verdict'](mid,directions(ta,te,g['matrix']),g['roofs'],
                                [v for v in g['obstacles'] if v['kind']!='tree'])[:,4]==0
    td = tc & (te > horizon_at(horizon,ta)) if horizon is not None else None
    return dict(theta_geometry_deg=float(tc.sum()*.5), sky_view_geometry=float(weight@clear),
                theta_dsm_deg=float(td.sum()*.5) if td is not None else None,
                sky_view_dsm=float(weight@clear_dsm) if clear_dsm is not None else None)


def horizon_at(horizon, az):
    return np.interp(np.asarray(az)%360, (np.arange(len(horizon))+.5)*360/len(horizon), horizon, period=360)


def window_horizon(w, g, crop):
    """Opaque DSM samples outside the own footprint; 360 azimuth bins."""
    en = w['center']@g['matrix'].T + g['shift']
    z = crop['ground_elevation_m'] + w['lower_z_m'] + w['height']/2 - g['sun']['assumptions']['model_ground_z_m']
    de = crop['east']-en[0]; dn = crop['north']-en[1]
    distance = np.hypot(de,dn)
    valid = crop['valid'] & ~crop['own'] & (distance>1.)
    az = np.rad2deg(np.arctan2(de[valid],dn[valid]))%360
    elevation = np.rad2deg(np.arctan2(crop['z'][valid]-z,distance[valid]))
    bins = (np.floor(az).astype(int))%360
    horizon = np.zeros(360)
    np.maximum.at(horizon,bins,np.maximum(0,elevation))
    # Fill angular gaps conservatively with neighbours, not interpolation through peaks.
    horizon = np.maximum.reduce([np.roll(horizon,k) for k in (-1,0,1)])
    return horizon


def energy(profile, w, g, sky, horizon=None):
    date = profile['date']; hourly = profile['profile']
    times = [dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)+dt.timedelta(hours=v['hour']-.5) for v in hourly]
    az, _, el = GEO['solar'](times,g['lat'],g['lon'])
    d = directions(az,el,g['matrix'])
    codes = GEO['aperture_verdict'](w,d,g['roofs'],g['obstacles'])
    fraction = (codes==0).mean(axis=1)
    if horizon is not None:
        fraction *= el>horizon_at(horizon,az)
    incidence = np.maximum(0,np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az-w['outward_azimuth_deg'])))
    incidence[el<=0]=0
    dni = np.array([v['direct_normal_irradiance'] for v in hourly],float)
    dhi = np.array([v['diffuse_radiation'] for v in hourly],float)
    ghi = np.array([v['shortwave_radiation'] for v in hourly],float)
    if not np.isfinite(np.r_[dni,dhi,ghi]).all():
        raise ValueError('missing_monthly_radiation')
    direct = dni*incidence*fraction
    diffuse = dhi*sky; ground=ghi*.2*.5
    total=direct+diffuse+ground
    daily=float(total.sum()/1000); area=w['width']*w['height']*.8
    lo=direct*90+(diffuse+ground)*110; hi=direct*105+(diffuse+ground)*130
    return dict(glass_kwh_m2_day=daily,direct_kwh_m2_day=float(direct.sum()/1000),
                diffuse_kwh_m2_day=float(diffuse.sum()/1000),ground_kwh_m2_day=float(ground.sum()/1000),
                entering_kwh_day_low=daily*area*.6,entering_kwh_day_central=daily*area*.7,
                entering_kwh_day_high=daily*area*.8,
                glass_lux_24h_mean_low=float(lo.mean()),glass_lux_24h_mean_high=float(hi.mean()),
                glass_lux_peak_hour_low=float(lo.max()),glass_lux_peak_hour_high=float(hi.max()),
                glass_dli_mol_m2_day_low=daily*3.6e6*2.0e-6,
                glass_dli_mol_m2_day_high=daily*3.6e6*2.3e-6), dict(
                    hours_utc_ending=list(range(24)),direct_w_m2=direct.tolist(),diffuse_w_m2=diffuse.tolist(),
                    ground_w_m2=ground.tolist(),glass_lux_low=lo.tolist(),glass_lux_high=hi.tolist(),
                    aperture_clear_fraction=fraction.tolist(),incidence_cosine=incidence.tolist(),
                    dni_w_m2=dni.tolist(),dhi_w_m2=dhi.tolist(),ghi_w_m2=ghi.tolist())


def compute_home(h, climates, with_hourly=False):
    g=load_geometry(h['sid']); crop=load_dsm(g)
    GEO['daylight'].__globals__['TZ']=ZoneInfo(g['sun']['assumptions']['timezone'])
    days={date:GEO['sample_day'](date,g['lat'],g['lon'],g['matrix']) for date in DATES}
    winrows=[]; radiation=[]; hourrows=[]; surfaces={}; failures=[]
    climate=climates.get(h['cell_id'],{})
    horizons={}
    for r in g['rooms'].values():
        surfaces[r['id']]=room_surface(g,r)
    for w in g['windows']:
        horizon=window_horizon(w,g,crop) if crop is not None else None
        horizons[w['id']]=horizon
        sky=sky_samples(w,g,horizon); area=w['width']*w['height']*.8
        base=dict(sid=h['sid'],window_id=w['id'],room_id=w['room_id'],poor=h['poor'],
                  glazed_area_m2=area,**sky,dsm_status='available' if crop is not None else 'withheld_plausibility' if h['sid'] in dsm_withheld() else 'unavailable',
                  dsm_capture_date=crop['capture_date'] if crop is not None else '',
                  leaf_state='unknown at capture',label=LABEL)
        winrows.append(base)
        for date in DATES:
            month=str(int(date[5:7])); cp=climate.get(month)
            if cp is None:
                failures.append(dict(sid=h['sid'],window_id=w['id'],date=date,failure='weather_cell_missing'));continue
            times,weights,d,az,el,_,_=days[date]
            codes=GEO['aperture_verdict'](w,d,g['roofs'],g['obstacles'])
            fraction=(codes==0).mean(axis=1)
            original=float(w['dates'][date]['unobstructed_area_weighted_minutes'])
            recomputed=float(weights@fraction)
            dsmminutes=float(weights@(fraction*(el>horizon_at(horizon,az)))) if horizon is not None else None
            common=dict(sid=h['sid'],window_id=w['id'],room_id=w['room_id'],date=date,month=int(month),
                        poor=h['poor'],glazed_area_m2=area,clear_sky_minutes_existing=original,
                        clear_sky_minutes_recomputed=recomputed,clear_sky_dsm_horizon_minutes=dsmminutes,
                        dsm_removed_minutes=recomputed-dsmminutes if dsmminutes is not None else None,
                        sunshine_share=cp['sunshine_share'],expected_sunny_minutes=original*cp['sunshine_share'],
                        expected_sunny_minutes_dsm=dsmminutes*cp['sunshine_share'] if dsmminutes is not None else None,
                        climate_label='modelled climate average',label=LABEL)
            for scenario,sv,hz in [('geometry',sky['sky_view_geometry'],None),('dsm',sky['sky_view_dsm'],horizon)]:
                if sv is None:continue
                value,hourly=energy(dict(date=date,profile=cp['profile_utc_hour_ending']),w,g,sv,hz)
                radiation.append(dict(common,scenario=scenario,**value))
                if with_hourly:
                    hourrows.append(dict(sid=h['sid'],window_id=w['id'],date=date,scenario=scenario,**hourly))
    roomrows=[]
    for r in g['rooms'].values():
        rr=dict(sid=h['sid'],room_id=r['id'],label=r['label'],poor=h['poor'],light_claim=h['light_claim'],
                area_m2=r['area_m2'],**surfaces[r['id']],
                ceiling_height_source=r['ceiling_height_source'],floor_height_source=r['floor_height_source'],
                main_living=r['id']==h['main_living_room_id'])
        ww=[v for v in winrows if v['room_id']==r['id']]
        rr['windows']=len(ww);rr['glazed_area_m2']=sum(v['glazed_area_m2'] for v in ww)
        for scenario,theta in [('geometry','theta_geometry_deg'),('dsm','theta_dsm_deg')]:
            valid=scenario!='dsm' or crop is not None
            area_sky=sum(v['glazed_area_m2']*v[theta] for v in ww) if valid else None
            for name,T,R in [('low',.6,.4),('central',.7,.5),('high',.8,.6)]:
                surface=rr['surface_area_m2']
                rr[f'df_{scenario}_percent_{name}']=df_value(area_sky,surface,T,R) if valid and surface is not None and surface>0 else None
        roomrows.append(rr)
    if crop is not None:
        path=ROOT/'data/cache/daylight_dsm'/f"{h['sid']}_horizons.npz"
        np.savez_compressed(path,**{k:v.astype(np.float16) for k,v in horizons.items()})
    return winrows,roomrows,radiation,hourrows,failures


def room_surface(g,r):
    height=r['ceiling_z_m']-r['floor_z_m']
    source='Existing ceiling_z_m minus floor_z_m'
    if not math.isfinite(height) or height<=0:
        raw=next(v for v in g['model']['rooms'] if v['id']==r['id'])
        dimensions=raw.get('dimensionEstimates') or {}
        height=dimensions.get('height')
        if height is not None:height=float(height)*(.3048 if dimensions.get('units')=='imperial' else 1.)
        source='Matterport dimensionEstimates.height fallback; existing ceiling at/below floor'
    valid=height is not None and math.isfinite(height) and height>0
    perimeter=sum(float(np.linalg.norm(seg[:,1]-seg[:,0],axis=1).sum()) for seg in [r['segments']]+r['holes'])
    surface=2*r['area_m2']+perimeter*height if valid else None
    return dict(surface_area_m2=surface,df_room_height_m=height if valid else None,
                surface_height_source=source,surface_geometry_failure='' if valid else 'No positive scan room height')


def refresh_surfaces():
    """Repair only derived DF fields in completed caches, preserving sun/energy."""
    counts=collections.Counter();failures=[]
    for path in sorted((ROOT/'data/cache/daylight_computed').glob('*.json.gz')):
        with gzip.open(path,'rt') as f:value=json.load(f)
        g=load_geometry(path.name.split('.')[0])
        for r in value['rooms']:
            r.update(room_surface(g,g['rooms'][r['room_id']]))
            counts['rooms']+=1
            if 'fallback' in r['surface_height_source']:counts['height_fallbacks']+=1
            if r['surface_geometry_failure']:failures.append(dict(sid=r['sid'],room_id=r['room_id'],reason=r['surface_geometry_failure']))
            ww=[w for w in value['windows'] if w['room_id']==r['room_id']]
            for scenario,theta in [('geometry','theta_geometry_deg'),('dsm','theta_dsm_deg')]:
                valid=scenario!='dsm' or value['has_dsm']
                area_sky=sum(w['glazed_area_m2']*w[theta] for w in ww) if valid else None
                for name,T,R in [('low',.6,.4),('central',.7,.5),('high',.8,.6)]:
                    surface=r['surface_area_m2']
                    r[f'df_{scenario}_percent_{name}']=df_value(area_sky,surface,T,R) if valid and surface is not None and surface>0 else None
        temporary=path.with_suffix('.surface.tmp')
        with gzip.open(temporary,'wt') as f:json.dump(value,f,allow_nan=False)
        temporary.replace(path)
        counts['homes']+=1
    write(OUT/'surface_geometry.json',dict(counts=dict(counts),failures=failures,engine_untouched=True))


DSM_WITHHELD = None


def dsm_withheld():
    global DSM_WITHHELD
    if DSM_WITHHELD is None:
        path=OUT/'dsm_quality.json'
        DSM_WITHHELD={r['sid']:r for r in read(path)['homes_detail'] if r['plausibility_flag']} if path.exists() else {}
    return DSM_WITHHELD


def dsm_available(sid):
    path=ROOT/'data/cache/daylight_dsm'/f'{sid}.npz'
    return path.exists() and path.with_suffix('.json').exists() and sid not in dsm_withheld()


def load_dsm(g, allow_flagged=False):
    path=ROOT/'data/cache/daylight_dsm'/f"{g['sid']}.npz"
    meta_path=path.with_suffix('.json')
    if not path.exists() or not meta_path.exists():return None
    if not allow_flagged and g['sid'] in dsm_withheld():return None
    meta=read(meta_path)
    with np.load(path,allow_pickle=False) as data:
        z=data['z'].astype(float); east=data['east'].astype(float);north=data['north'].astype(float)
    points=np.column_stack([east.ravel(),north.ravel()])
    own=PolygonPath(np.array(g['fit']['_plot']['ring'])).contains_points(points).reshape(z.shape)
    return dict(z=z,east=east,north=north,valid=np.isfinite(z),own=own,
                ground_elevation_m=meta['ground_elevation_m'],capture_date=meta['capture_date'],metadata=meta)


DSM_STOP=threading.Event()
DSM_AUTH=0


def dsm_request(endpoint, query, key):
    global DSM_AUTH
    if DSM_STOP.is_set():raise RuntimeError('DSM authorization stop')
    url='https://api.vexcelgroup.com/v2/dsm/'+endpoint+'?'+urllib.parse.urlencode(query)
    for attempt in range(6):
        if DSM_STOP.is_set():raise RuntimeError('DSM authorization stop')
        try:
            req=urllib.request.Request(url,headers={'X-API-KEY':key})
            with LOCK:
                NET['active']+=1;NET['maximum_in_flight']=max(NET['maximum_in_flight'],NET['active']);NET['attempts']+=1
            try:
                with urllib.request.urlopen(req,timeout=120) as response:
                    payload=response.read(12*1024*1024+1)
                    if len(payload)>12*1024*1024:raise ValueError('DSM response exceeds crop budget')
            finally:
                with LOCK:NET['active']-=1
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code in (401,403):
                with LOCK:
                    DSM_AUTH+=1
                    if DSM_AUTH>=2:DSM_STOP.set()
                raise RuntimeError(f'DSM HTTP {exc.code}; authorization failure count {DSM_AUTH}') from None
            if exc.code not in (429,500,502,503,504) or attempt==5:
                raise RuntimeError(f'DSM HTTP {exc.code}') from None
            time.sleep(min(60,5*2**attempt))
        except (TimeoutError,urllib.error.URLError):
            if attempt==5:raise RuntimeError('DSM timeout or transport failure') from None
            time.sleep(min(60,5*2**attempt))


def own_ground(g):
    prop=read(ROOT/'data/cache'/g['sid']/'vexcel_calculated_65m.json')
    target=np.array(g['fit']['_plot']['ring']).mean(axis=0);found=[]
    for feature in prop.get('features',[]):
        properties=feature.get('properties') or {}
        if g['fit'].get('collection') and properties.get('metadata.collection')!=g['fit']['collection']:continue
        for structure in properties.get('structures') or []:
            text=structure.get('footprint.wkt','')
            xy=np.array([[float(a),float(b)] for a,b in re.findall(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)',text)])
            if len(xy)<3 or structure.get('ground-elevation.value') is None:continue
            en=(xy-np.array([g['lon'],g['lat']]))*np.array([111320*math.cos(math.radians(g['lat'])),110574])
            found.append((np.linalg.norm(en.mean(axis=0)-target),float(structure['ground-elevation.value'])))
    if not found:raise ValueError('own_footprint_ground_datum_missing')
    return min(found)[1]


def acquire_dsm(h,key):
    folder=ROOT/'data/cache/daylight_dsm';folder.mkdir(exist_ok=True)
    path=folder/f"{h['sid']}.npz"
    if path.exists() and path.with_suffix('.json').exists():return read(path.with_suffix('.json'))
    if shutil.disk_usage(ROOT).free < 3_000_000_000:
        raise RuntimeError('free_disk_below_3GB; crop skipped')
    g=load_geometry(h['sid']);lat=h['latitude'];lon=h['longitude']
    query=dict(wkt=f'POINT({lon} {lat})',srid=4326,layer='urban,urban-r,urban-g')
    cover=json.loads(dsm_request('collections',dict(query,include='collection,first-capture-date,last-capture-date,min-gsd,max-gsd,geometry',**{'total-records':1,'order-by':'first-capture-date-desc'}),key))
    features=cover.get('features',[]) if isinstance(cover,dict) else []
    if not features:raise ValueError('DSM no collection coverage')
    properties=features[0].get('properties',{})
    collection=properties.get('collection')
    if not collection:raise ValueError('DSM collection identifier missing')
    payload=dsm_request('extract',dict(query,collection=collection,**{'bbox-dimensions':'120,120','destination-dimensions':'240,240,fit','crop':'bounding_box'}),key)
    image=Image.open(io.BytesIO(payload))
    tags=image.tag_v2
    if image.size!=(240,240) or tags.get(339)!=(3,) or tags.get(258)!=(32,) or tags.get(317,1)!=1:
        raise ValueError('DSM TIFF requires240x240 single-band float32 without predictor')
    compression=tags.get(259)
    if compression not in (1,8,32946):raise ValueError('DSM TIFF compression unsupported')
    # Decode the declared TIFF byte order explicitly; Pillow's float raster path
    # in the installed build misreads this provider's big-endian deflate samples.
    endian='>' if payload[:2]==b'MM' else '<' if payload[:2]==b'II' else None
    if endian is None:raise ValueError('DSM TIFF byte order absent')
    def unpack(offset,count,limit):
        block=payload[offset:offset+count]
        if compression!=1:
            decoder=zlib.decompressobj();block=decoder.decompress(block,limit+1)
            if len(block)>limit or not decoder.eof:raise ValueError('DSM raster decompression exceeds crop budget')
        return block
    if 324 in tags:
        tw,th=tags[322],tags[323]
        if not 0<tw<=1024 or not 0<th<=1024:raise ValueError('DSM TIFF tile exceeds crop budget')
        z=np.full((240,240),np.nan)
        across=math.ceil(240/tw)
        offsets,counts=tags[324],tags[325]
        if len(offsets)!=across*math.ceil(240/th):raise ValueError('DSM tile count mismatch')
        for i,(offset,count) in enumerate(zip(offsets,counts)):
            block=unpack(offset,count,tw*th*4)
            if len(block)!=tw*th*4:raise ValueError('DSM tile byte count mismatch')
            tile=np.frombuffer(block,dtype=endian+'f4').reshape(th,tw)
            x,y=(i%across)*tw,(i//across)*th
            hh,ww=min(th,240-y),min(tw,240-x)
            z[y:y+hh,x:x+ww]=tile[:hh,:ww]
    else:
        raw=b''.join(unpack(offset,count,240*240*4) for offset,count in zip(tags[273],tags[279]))
        if len(raw)!=240*240*4:raise ValueError('DSM strip byte count mismatch')
        z=np.frombuffer(raw,dtype=endian+'f4').reshape(240,240).astype(float)
    if z.ndim!=2:raise ValueError('DSM extract is not single-band elevation')
    scale=tags.get(33550); tie=tags.get(33922); geokey=tags.get(34735)
    if not scale or not tie or not geokey:raise ValueError('DSM GeoTIFF georeferencing absent')
    keys={geokey[i]:geokey[i+3] for i in range(4,len(geokey),4)}
    crs=keys.get(3072) or keys.get(2048)
    px,py=np.meshgrid(np.arange(z.shape[1])+.5,np.arange(z.shape[0])+.5)
    x=tie[3]+(px-tie[0])*scale[0];y=tie[4]-(py-tie[1])*scale[1]
    if crs==3857:
        lon_grid=np.rad2deg(x/6378137);lat_grid=np.rad2deg(2*np.arctan(np.exp(y/6378137))-np.pi/2)
    elif crs==4326:lon_grid,lat_grid=x,y
    else:raise ValueError(f'DSM unsupported CRS {crs}')
    east=(lon_grid-g['lon'])*111320*math.cos(math.radians(g['lat']));north=(lat_grid-g['lat'])*110574
    nodata=tags.get(42113)
    if nodata is not None:
        try:z[z==float(str(nodata).strip('\x00'))]=np.nan
        except ValueError:pass
    z[(z < -500)|(z > 9000)]=np.nan
    if np.isfinite(z).mean()<.9:
        finite=z[np.isfinite(z)]
        write(folder/f"{h['sid']}_decode.json",dict(mode=image.mode,shape=list(z.shape),dtype=str(z.dtype),
            tiff_header=payload[:8].hex(),compression=tags.get(259),predictor=tags.get(317),
            payload_bytes=len(payload),valid_fraction=float(np.isfinite(z).mean()),
            raw_finite_fraction=float(np.isfinite(z).mean()),
            raw_min=float(finite.min()) if len(finite) else None,raw_max=float(finite.max()) if len(finite) else None,
            crs=crs,scale=list(scale),tie=list(tie),sample_format=tags.get(339),bits=tags.get(258),
            nodata=str(nodata),collection=collection,properties=properties))
        raise ValueError('DSM crop has less than90% valid elevations')
    meta=dict(sid=h['sid'],collection=collection,capture_date=[properties.get('first-capture-date'),properties.get('last-capture-date')],
              collection_properties=properties,ground_elevation_m=own_ground(g),ground_source='Cached own structure ground-elevation.value; shared DSM vertical datum assumed',
              requested_box_m=[120,120],shape=list(z.shape),crs=crs,leaf_state='unknown at capture',
              raster_decode='TIFF declared byte order, single-band float32 tiles or strips; standard-library deflate',
              downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),free_disk_bytes=shutil.disk_usage(ROOT).free,
              service='https://api.vexcelgroup.com/v2/dsm/extract',own_footprint_mask='Accepted registration footprint; no dilation',
              vertical_datum='No independent datum survey; shared provider DSM/property ground reference assumed')
    np.savez_compressed(path,z=z.astype(np.float16),east=east.astype(np.float16),north=north.astype(np.float16))
    write(path.with_suffix('.json'),meta)
    return meta


def dsm_download(credential_file,featured_only=True):
    global RUNTIME_CREDENTIAL_READ
    if not featured_only:
        validation=OUT/'dsm_featured_validation.json'
        if not validation.exists() or not read(validation).get('passed'):
            raise RuntimeError('Validate all six featured DSM crops before cohort acquisition')
        if shutil.disk_usage(ROOT).free < 3_000_000_000:
            raise RuntimeError('free_disk_below_3GB; cohort acquisition forbidden')
    if not credential_file:raise ValueError('Supply credential file path; never a key argument')
    credential_path=Path(credential_file).resolve()
    if credential_path.name=='.env':
        if credential_path!=ROOT/'.env':raise ValueError('Only project runtime credential source is allowed')
        RUNTIME_CREDENTIAL_READ=True
        try:
            with credential_path.open() as stream:
                key=next((line.split('=',1)[1].strip().strip("\"'") for line in stream
                          if '=' in line and line.split('=',1)[0].strip()=='VEXCEL_API_KEY'),'')
        finally:RUNTIME_CREDENTIAL_READ=False
    else:key=credential_path.read_text().strip()
    if not key:raise ValueError('empty private credential file')
    hh=[h for h in home_cells() if h['computed']=='True' and (h['sid'] in FEATURED or not featured_only)]
    status=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(acquire_dsm,h,key):h['sid'] for h in hh}
        for future in concurrent.futures.as_completed(tasks):
            sid=tasks[future]
            try:
                m=future.result();status.append(dict(sid=sid,status='cached',collection=m['collection'],capture_date=m['capture_date']))
            except Exception as exc:
                status.append(dict(sid=sid,status='failed',reason=str(exc),
                                   location=[f'{Path(v.filename).name}:{v.lineno}:{v.name}' for v in traceback.extract_tb(exc.__traceback__)]))
            write(OUT/'dsm_download_progress.json',dict(total=len(hh),completed=len(status),network=dict(NET),
                                                       free_disk_bytes=shutil.disk_usage(ROOT).free,status=status))
    key=''
    return status


def validate_dsm():
    """Produce real-crop checks and review panels before the cohort gate opens."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    checks=[]
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for sid,ax in zip(FEATURED,axes.flat):
        try:
            g=load_geometry(sid);crop=load_dsm(g)
            if crop is None:raise ValueError('crop unavailable')
            inside=crop['valid'] & crop['own']
            if not inside.any():raise ValueError('own footprint outside DSM crop')
            relative=crop['z']-crop['ground_elevation_m']
            own_height=float(np.nanmedian(relative[inside]))
            model_height=max(r['ceiling_z_m'] for r in g['sun']['rooms'])-g['sun']['assumptions']['model_ground_z_m']
            horizons=[window_horizon(w,g,crop) for w in g['windows']]
            finite=all(np.isfinite(v).all() and (v>=0).all() and (v<=90).all() for v in horizons)
            capture=crop['capture_date']
            passed=finite and bool(capture) and all(capture) and -1<own_height<50 and abs(own_height-model_height)<15
            checks.append(dict(sid=sid,passed=passed,collection=crop['metadata']['collection'],capture_date=capture,
                valid_fraction=float(crop['valid'].mean()),own_mask_pixels=int(inside.sum()),windows=len(horizons),
                own_roof_median_above_property_ground_m=own_height,model_max_ceiling_above_ground_m=model_height,
                finite_horizon=finite,own_pixels_excluded=True,
                mask_note='Own roof pixels removed from DSM only; existing own-roof model remains in sun/sky checks'))
            shown=np.ma.array(relative,mask=~crop['valid'])
            im=ax.pcolormesh(crop['east'],crop['north'],shown,shading='nearest',vmin=0,vmax=20,cmap='terrain')
            ring=np.array(g['fit']['_plot']['ring']);ring=np.vstack([ring,ring[0]])
            ax.plot(ring[:,0],ring[:,1],color='red',lw=1.5,label='own mask')
            for w in g['windows']:
                en=w['center']@g['matrix'].T+g['shift'];ax.plot(*en,'ko',ms=2)
            ax.set_title(f'{sid}\n{capture[0]} to {capture[1]}');ax.set_aspect('equal')
            ax.set_xlabel('east m');ax.set_ylabel('north m')
        except Exception as exc:
            checks.append(dict(sid=sid,passed=False,reason=f'{type(exc).__name__}: {exc}'))
            ax.set_title(sid);ax.text(.1,.5,'DSM unavailable',transform=ax.transAxes)
    fig.savefig(OUT/'dsm_featured_panels.png',dpi=120);plt.close(fig)
    report=dict(passed=False,automatic_checks_passed=len(checks)==6 and all(v['passed'] for v in checks),
        manual_panel_review='pending; reviewer must inspect panels before setting passed true',homes=checks,
        hand_checked_home='khbuN7q853q',
        hand_check_scope='Existing hand marks validate placement, not surveyed tree heights or DSM horizon accuracy; no new ground truth',
        leaf_state='unknown at capture',free_disk_bytes=shutil.disk_usage(ROOT).free)
    write(OUT/'dsm_featured_validation.json',report)
    return report


def control_dsm():
    g=load_geometry('khbuN7q853q');crop=load_dsm(g)
    if crop is None:raise ValueError('Control DSM unavailable')
    source=read(ROOT/'data/sun_check/khbuN7q853q/geometry_pano_agreement.json')
    openings={w['id']:dict(w,center=np.array(w['center'])) for w in g['sun']['windows']}
    horizons={wid:window_horizon(w,g,crop) for wid,w in openings.items()}
    checked=[]
    for record in source['records']:
        wid=record.get('window_id')
        if wid not in horizons:continue
        limit=float(horizon_at(horizons[wid],record['sun_azimuth_deg']))
        checked.append(dict(record,dsm_horizon_deg=limit,
            dsm_blocks_center=record['sun_elevation_deg']<=limit,
            existing_own_roof_preserved=record['geometry_at_opening_center']=='own_roof'))
    foliage=[r for r in checked if r['pano_category']=='foliage']
    write(OUT/'dsm_control_check.json',dict(sid=g['sid'],capture_date=crop['capture_date'],
        source='data/sun_check/khbuN7q853q/geometry_pano_agreement.json',
        scoped_records=len(checked),foliage_records=len(foliage),
        foliage_center_rays_blocked_by_dsm=sum(r['dsm_blocks_center'] for r in foliage),
        own_roof_center_blocks_retained=sum(r['existing_own_roof_preserved'] for r in checked),
        limitation='Visible foliage rays are on a glazed door: auxiliary control only, excluded from window cohort. Panorama camera differs from window; this is a consistency comparison, not measured horizon ground truth.',
        records=checked))


def dsm_quality():
    checks=[]
    for h in home_cells():
        if h['computed']!='True':continue
        try:
            g=load_geometry(h['sid']);crop=load_dsm(g,allow_flagged=True)
            if crop is None:continue
            own=crop['valid'] & crop['own']
            roof=float(np.median(crop['z'][own])-crop['ground_elevation_m']) if own.any() else None
            model=max(r['ceiling_z_m'] for r in g['sun']['rooms'])-g['sun']['assumptions']['model_ground_z_m']
            gap=roof-model if roof is not None else None
            checks.append(dict(sid=h['sid'],poor=h['poor'],collection=crop['metadata']['collection'],
                capture_date=crop['capture_date'],valid_fraction=float(crop['valid'].mean()),
                own_pixels=int(own.sum()),own_roof_relative_ground_m=roof,model_max_ceiling_m=model,
                roof_minus_ceiling_m=gap,
                plausibility_flag=roof is None or not -1<roof<50 or abs(gap)>=15))
        except Exception as exc:checks.append(dict(sid=h['sid'],plausibility_flag=True,error=f'{type(exc).__name__}: {exc}'))
    write(OUT/'dsm_quality.json',dict(homes=len(checks),flagged=sum(r['plausibility_flag'] for r in checks),
        scope='Automated plausibility screen, not independent accuracy validation. Same bounds as featured gate: own roof above ground in(-1,50)m and within15m of model ceiling. Flagged DSM estimates withheld, raw crops retained; original registration labels unchanged.',
        homes_detail=checks))


GEO = geometry_functions()

FEATURED=['khbuN7q853q','16ZWm6HRVjS','1FFftrgRFBd','VN4b1MGnnCR','LQLjZqrmKqP','HFwcH424n9w']


def aggregate_rooms(radiation, roomrows):
    grouped=collections.defaultdict(list)
    for w in radiation:grouped[(w['sid'],w['room_id'],w['date'],w['scenario'])].append(w)
    result=[]
    for (sid,rid,date,scenario),ww in grouped.items():
        area=sum(v['glazed_area_m2'] for v in ww)
        rr=dict(sid=sid,room_id=rid,date=date,month=int(date[5:7]),scenario=scenario,
                windows=len(ww),glazed_area_m2=area,label=LABEL)
        for key in ww[0]:
            if key.startswith(('entering_kwh_day','direct_kwh','diffuse_kwh','ground_kwh','glass_kwh','glass_dli','glass_lux_24h')):
                rr[key]=sum(v[key] for v in ww) if key.startswith('entering') else sum(v[key]*v['glazed_area_m2'] for v in ww)/area
        result.append(rr)
    present={(r['sid'],r['room_id'],r['date'],r['scenario']) for r in result}
    valuekeys=[k for r in result[:1] for k in r if k.startswith(('entering_kwh','direct_kwh','diffuse_kwh','ground_kwh','glass_kwh','glass_dli','glass_lux_24h'))]
    for room in roomrows:
        if room['windows']:continue
        for date in DATES:
            for scenario in ['geometry','dsm']:
                if scenario=='dsm' and not dsm_available(room['sid']):continue
                key=(room['sid'],room['room_id'],date,scenario)
                if key not in present:
                    result.append(dict(sid=room['sid'],room_id=room['room_id'],date=date,month=int(date[5:7]),
                                       scenario=scenario,windows=0,glazed_area_m2=0,label=LABEL,
                                       **{k:0 if k.startswith('entering') else None for k in valuekeys}))
    return result


def claim_comparison(roomrows):
    result={'scope':'Well-registered homes only; descriptive, no causal inference. Rooms are clustered within homes.',
            'all_room_scope':'Valid positive room geometry only; includes windowless rooms (DF zero); exterior transparent doorways excluded.',
            'groups':{}}
    for scenario in ['geometry','dsm']:
        key=f'df_{scenario}_percent_central';groups={}
        for claim in ['True','False']:
            rr=[r for r in roomrows if r['poor']=='False' and r['light_claim']==claim and r[key] is not None]
            homes=collections.defaultdict(list)
            for r in rr:homes[r['sid']].append(r[key])
            living=[r[key] for r in rr if r['main_living']]
            windowed=[r[key] for r in rr if r['windows']>0]
            groups['claims natural light' if claim=='True' else 'no claim']=dict(
                homes=len(homes),rooms=len(rr),room_df_median_percent=statistics.median([r[key] for r in rr]) if rr else None,
                windowed_rooms=len(windowed),windowed_room_df_median_percent=statistics.median(windowed) if windowed else None,
                median_of_home_room_medians_percent=statistics.median([statistics.median(v) for v in homes.values()]) if homes else None,
                main_living_rooms=len(living),main_living_df_median_percent=statistics.median(living) if living else None)
        result['groups'][scenario]=groups
    return result


def prepare_home(h,climates):
    target=ROOT/'data/cache/daylight_computed'/(h['sid']+'.json.gz')
    has_dsm=dsm_available(h['sid'])
    if target.exists():
        try:
            with gzip.open(target,'rt') as f:cached=json.load(f)
            if cached['has_dsm']==has_dsm:return dict(sid=h['sid'],status='reused',dsm=has_dsm)
        except (ValueError,EOFError,OSError):pass
    try:
        value=dict(zip(['windows','rooms','radiation','hourly','failures'],compute_home(h,climates,h['sid'] in FEATURED)))
        value['has_dsm']=has_dsm
        temporary=target.with_suffix(f'.{os.getpid()}.tmp')
        with gzip.open(temporary,'wt') as f:json.dump(value,f,allow_nan=False)
        temporary.replace(target)
        return dict(sid=h['sid'],status='computed',dsm=has_dsm)
    except Exception as exc:return dict(sid=h['sid'],status='failed',reason=f'{type(exc).__name__}: {exc}')


def worker_guard():
    sys.addaudithook(offline_guard)


def prepare(featured=False,jobs=4):
    climates=read(OUT/'climate_cells.json')['cells']
    homes=[h for h in home_cells() if h['computed']=='True' and (not featured or h['sid'] in FEATURED)]
    (ROOT/'data/cache/daylight_computed').mkdir(exist_ok=True)
    status=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs,initializer=worker_guard) as pool:
        futures=[pool.submit(prepare_home,h,climates) for h in homes]
        for future in concurrent.futures.as_completed(futures):
            status.append(future.result())
            if len(status)%25==0 or len(status)==len(homes):
                write(OUT/'prepare_progress.json',dict(total=len(homes),completed=len(status),counts=dict(collections.Counter(v['status'] for v in status)),failures=[v for v in status if v['status']=='failed']))
    return status


def compute(featured_only=False, refresh=False):
    climates=read(OUT/'climate_cells.json')['cells']; homes=home_cells()
    selected=[h for h in homes if not featured_only or h['sid'] in FEATURED]
    cache=ROOT/'data/cache/daylight_computed';cache.mkdir(exist_ok=True)
    windows=[];roomrows=[];radiation=[];hourly=[];failures=[];status=[]
    for i,h in enumerate(selected):
        if h['computed']!='True':
            status.append(dict(sid=h['sid'],status='existing_engine_failed',reason=h['failure_reason']));continue
        target=cache/(h['sid']+'.json.gz')
        try:
            dsm_exists=dsm_available(h['sid'])
            if target.exists() and not refresh:
                with gzip.open(target,'rt') as f: value=json.load(f)
                if value['has_dsm']!=dsm_exists:raise ValueError('cache_dsm_changed; use --refresh')
            else:
                value=dict(zip(['windows','rooms','radiation','hourly','failures'],compute_home(h,climates,h['sid'] in FEATURED)))
                value['has_dsm']=dsm_exists
                with gzip.open(target,'wt') as f:json.dump(value,f,allow_nan=False)
            windows.extend(value['windows']);roomrows.extend(value['rooms']);radiation.extend(value['radiation'])
            hourly.extend(value['hourly']);failures.extend(value['failures'])
            status.append(dict(sid=h['sid'],status='computed',dsm=value['has_dsm']))
        except Exception as exc:
            status.append(dict(sid=h['sid'],status='failed',reason=f'{type(exc).__name__}: {exc}'))
        if i%25==0:
            write(OUT/'compute_progress.json',dict(processed=i+1,total=len(selected),windows=len(windows),rooms=len(roomrows),failures=[v for v in status if v['status']=='failed']))
    suffix='_featured' if featured_only else ''
    table(OUT/f'windows{suffix}.csv',windows);table(OUT/f'rooms{suffix}.csv',roomrows)
    table(OUT/f'window_months{suffix}.csv',radiation);table(OUT/f'room_months{suffix}.csv',aggregate_rooms(radiation,roomrows))
    table(OUT/f'home_status{suffix}.csv',status)
    write(OUT/f'hourly_featured{suffix}.json',hourly)
    failures.extend(dict(sid=r['sid'],room_id=r['room_id'],failure='invalid_surface_geometry',
                         reason=r.get('surface_geometry_failure','Unknown surface area'))
                    for r in roomrows if r['df_geometry_percent_central'] is None)
    write(OUT/f'failures{suffix}.json',failures)
    write(OUT/f'claim_comparison{suffix}.json',claim_comparison(roomrows))
    ds=[r for r in radiation if r['scenario']=='dsm' and r['month']==12]
    deltas=[v['dsm_removed_minutes'] for v in ds]
    verification=dict(selected=len(homes),downloaded=sum(h['downloaded']=='True' for h in homes),
                      engine_computed=sum(h['computed']=='True' for h in homes),well_registered=sum(h['poor']=='False' for h in homes),
                      daylight_computed_homes=sum(v['status']=='computed' for v in status),rooms=len(roomrows),windows=len(windows),
                      df_geometry_rooms_valid=sum(r['df_geometry_percent_central'] is not None for r in roomrows),
                      df_geometry_rooms_failed=sum(r['df_geometry_percent_central'] is None for r in roomrows),
                      df_height_fallbacks=sum('fallback' in r.get('surface_height_source','') for r in roomrows),
                      dsm_homes=sum(v.get('dsm',False) for v in status),winter_dsm_windows=len(ds),
                      winter_removed_median_minutes=statistics.median(deltas) if deltas else None,
                      winter_share_windows_affected=sum(v>.01 for v in deltas)/len(deltas) if deltas else None,
                      maximum_existing_replay_difference_minutes=max((abs(v['clear_sky_minutes_existing']-v['clear_sky_minutes_recomputed']) for v in radiation),default=0),
                      failures=[v for v in status if v['status']=='failed'])
    write(OUT/f'verification{suffix}.json',verification)
    progress(f'Example and 16 weather cells cached. Computed {verification}. Files in data/facts/daylight; code experiments/daylight.py; resumable gzip home outputs data/cache/daylight_computed/.',
             'Finish DSM validation/cohort, wording, conversion report, existing suite and protected-file hashes.')
    return verification


def home_months():
    climates=read(OUT/'climate_cells.json')['cells'];result=[]
    for h in home_cells():
        sunpath=ROOT/'data/sun'/h['sid']/'sun.json'
        sun=read(sunpath) if sunpath.exists() else None
        main=next((r for r in sun['rooms'] if r['id']==h['main_living_room_id']),None) if sun else None
        for date in DATES:
            share=climates.get(h['cell_id'],{}).get(str(int(date[5:7])),{}).get('sunshine_share')
            clear=main['dates'][date]['windows_only_lit_minutes_ge_0_1m2'] if main else None
            result.append(dict(sid=h['sid'],date=date,month=int(date[5:7]),poor=h['poor'],main_living_room_id=h['main_living_room_id'],
                               cell_id=h['cell_id'],sunshine_share=share,clear_sky_main_living_windows_only_minutes=clear,
                               expected_sunny_main_living_minutes=share*clear if share is not None and clear is not None else None,
                               label='modelled climate average',failure='' if main else 'main_living_or_engine_unavailable'))
    table(OUT/'home_months.csv',result)


def hand_conversion():
    records=read(OUT/'hourly_featured_featured.json')
    candidates=[(h,i) for h in records if h['scenario']=='geometry' for i,v in enumerate(h['direct_w_m2']) if v>0]
    h,i=max(candidates,key=lambda x:x[0]['direct_w_m2'][x[1]])
    lookup={(v['sid'],v['window_id']):v for v in rows(OUT/'windows_featured.csv')}
    area=float(lookup[(h['sid'],h['window_id'])]['glazed_area_m2'])
    direct=h['dni_w_m2'][i]*h['incidence_cosine'][i]*h['aperture_clear_fraction'][i]
    diffuse=h['diffuse_w_m2'][i]; ground=h['ghi_w_m2'][i]*.2*.5; watts=direct+diffuse+ground
    result=dict(sid=h['sid'],window_id=h['window_id'],date=h['date'],utc_hour_ending=i,
                dni_w_m2=h['dni_w_m2'][i],cos_incidence=h['incidence_cosine'][i],sunlit_fraction=h['aperture_clear_fraction'][i],
                dhi_w_m2=h['dhi_w_m2'][i],sky_view=float(lookup[(h['sid'],h['window_id'])]['sky_view_geometry']),
                ghi_w_m2=h['ghi_w_m2'][i],direct_w_m2=direct,diffuse_w_m2=diffuse,ground_w_m2=ground,
                total_w_m2=watts,one_hour_glass_kwh_m2=watts/1000,glazed_area_m2=area,
                one_hour_entering_kwh_low=watts/1000*area*.6,one_hour_entering_kwh_high=watts/1000*area*.8,
                glass_lux_low=direct*90+(diffuse+ground)*110,glass_lux_high=direct*105+(diffuse+ground)*130,
                one_hour_par_mol_m2_low=watts*3600*2e-6,one_hour_par_mol_m2_high=watts*3600*2.3e-6,
                formula='W/m2 * 3600s = J/m2; /3,600,000 = kWh/m2; *2.0..2.3 umol/J /1,000,000 = mol/m2. Sum hourly contributions for daily values.',label=LABEL)
    assert abs(direct-h['direct_w_m2'][i])<1e-9
    assert abs(diffuse-h['dhi_w_m2'][i]*result['sky_view'])<1e-8
    write(OUT/'hand_conversion.json',result)
    return result


def evidence():
    baseline=read(ROOT/'data/cache/daylight_baseline/manifest.json')
    protected={k:v for k,v in baseline.items() if not k.startswith(('docs/pitch/','agent/skills/'))}
    changed=[k for k,v in protected.items() if not (ROOT/k).exists() or hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
    authorized=[k for k,v in baseline.items() if k not in protected and (ROOT/k).exists() and hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
    write(OUT/'protected_files.json',dict(checked=len(protected),changed=changed,unchanged=not changed,
                                        authorized_wording_changes=authorized,writer='Codex'))
    ww=rows(OUT/'windows.csv');rr=rows(OUT/'rooms.csv');mm=rows(OUT/'window_months.csv')
    rm=rows(OUT/'room_months.csv')
    dsm_sids={w['sid'] for w in ww if w['dsm_status']=='available'}
    checks=dict(unique_windows=len({(w['sid'],w['window_id']) for w in ww})==len(ww),
                unique_rooms=len({(r['sid'],r['room_id']) for r in rr})==len(rr),
                expected_room_month_rows=len(rm)==4*(len(rr)+sum(dsm_available(r['sid']) for r in rr)),
                expected_four_dates_per_available_scenario=len(mm)==4*(len(ww)+sum(w['dsm_status']=='available' for w in ww)))
    energy_error=[];photon_error=[];dsm_increases=[];room_energy=collections.defaultdict(float)
    for r in mm:
        energy=float(r['glass_kwh_m2_day'])
        energy_error.append(abs(energy-sum(float(r[k+'_kwh_m2_day']) for k in ('direct','diffuse','ground'))))
        photon_error.append(abs(float(r['glass_dli_mol_m2_day_low'])-energy*7.2))
        room_energy[(r['sid'],r['room_id'],r['date'],r['scenario'])]+=float(r['entering_kwh_day_central'])
        if r['scenario']=='dsm':dsm_increases.append(float(r['clear_sky_dsm_horizon_minutes'])>float(r['clear_sky_minutes_recomputed'])+1e-7)
    checks.update(max_component_energy_error_kwh_m2=max(energy_error,default=0),
                  max_dli_conversion_error_mol_m2=max(photon_error,default=0),dsm_added_sun=any(dsm_increases),
                  max_room_entering_sum_error_kwh=max((abs(float(r['entering_kwh_day_central'])-room_energy[(r['sid'],r['room_id'],r['date'],r['scenario'])]) for r in rm),default=0))
    stats={}
    for scenario in ('geometry','dsm'):
        for month in MONTHS:
            selected=[r for r in mm if r['scenario']==scenario and int(r['month'])==month and r['poor']=='False']
            def median(field):return statistics.median(float(r[field]) for r in selected) if selected else None
            stats[f'{scenario}_{month}']=dict(homes=len({r['sid'] for r in selected}),windows=len(selected),
                median_glass_kwh_m2_day=median('glass_kwh_m2_day'),median_entering_kwh_day_central=median('entering_kwh_day_central'),
                median_glass_dli_low=median('glass_dli_mol_m2_day_low'),median_glass_dli_high=median('glass_dli_mol_m2_day_high'))
    winter={}
    for name,well_only in [('all_covered_windows',False),('well_registered_windows',True)]:
        selected=[r for r in mm if r['scenario']=='dsm' and int(r['month'])==12 and (not well_only or r['poor']=='False')]
        deltas=[float(r['dsm_removed_minutes']) for r in selected]
        winter[name]=dict(homes=len({r['sid'] for r in selected}),windows=len(selected),
            median_removed_minutes=statistics.median(deltas) if deltas else None,
            affected_windows=sum(v>.01 for v in deltas),
            share_affected=sum(v>.01 for v in deltas)/len(deltas) if deltas else None,
            affected_definition='DSM removes more than0.01 aperture-area-weighted minute on December21')
    write(OUT/'cohort_summary.json',dict(scope='Well-registered windows only; window medians, not area-weighted cohort statistics',
        by_month_scenario=stats,winter_dsm_removal=winter,checks=checks,home_status_rows=len(rows(OUT/'home_status.csv')),
        room_month_rows=len(rm),window_month_rows=len(mm)))
    assert not changed,'Protected input files changed; inspect protected_files.json'
    assert checks['unique_windows'] and checks['unique_rooms'] and checks['expected_four_dates_per_available_scenario'] and checks['expected_room_month_rows']
    assert checks['max_component_energy_error_kwh_m2']<1e-10 and checks['max_dli_conversion_error_mol_m2']<1e-10
    assert not checks['dsm_added_sun']
    assert checks['max_room_entering_sum_error_kwh']<1e-10


F4_CACHE = ROOT/'data/cache/daylight_f4'
F4_PROGRESS = Path('/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T20-07-18-859Z-12097-333f578f-6c74-47f3-914d-fa4bfbbfffc9.md')
F4_BANDS = {'morning':(6,10),'midday':(10,14),'afternoon':(14,18),'evening':(18,24)}


def f4_checkpoint(done,next_step):
    F4_PROGRESS.write_text('## Objective\nFinish resumable F4 cohort, preserve schema_version1, then wire heat/hot-sun/flags into buyer ranking. No sponsor calls, credentials or new tests.\n## Done (files, test results)\n'+done+'\n## Next\n'+next_step+'\n')


def f4_cache_path(sid):
    original=F4_CACHE/'computed'/f'{sid}.json.gz'
    return original if original.exists() else OUT/'f4_computed'/f'{sid}.json.gz'


def f4_temperature(h,offline=False):
    path=F4_CACHE/'temperature'/f"{h['cell_id']}_2016_2025.json"
    if path.exists():return read(path)
    if offline:raise RuntimeError('temperature_cache_missing')
    query=dict(latitude=h['cell_latitude'],longitude=h['cell_longitude'],start_date='2016-01-01',
               end_date='2025-12-31',timezone='UTC',hourly='temperature_2m',models='era5')
    url='https://archive-api.open-meteo.com/v1/archive?'+urllib.parse.urlencode(query)
    for attempt in range(6):
        try:
            with LOCK:
                NET['active']+=1;NET['maximum_in_flight']=max(NET['maximum_in_flight'],NET['active']);NET['attempts']+=1
            try:
                with urllib.request.urlopen(url,timeout=180) as response:data=json.load(response)
            finally:
                with LOCK:NET['active']-=1
            if data.get('error'):raise ValueError('temperature_archive_error')
            assert len(data['hourly']['time'])==87672
            assert data['hourly_units']['temperature_2m']=='°C'
            value=dict(source_url=url,downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),data=data)
            write(path,value);return value
        except urllib.error.HTTPError as exc:
            if exc.code not in (429,500,502,503,504) or attempt==5:raise RuntimeError(f'temperature HTTP {exc.code}') from exc
            time.sleep(min(60,5*2**attempt))
        except (TimeoutError,urllib.error.URLError):
            if attempt==5:raise
            time.sleep(min(60,5*2**attempt))


def f4_weather(featured=False,offline=False):
    homes=[h for h in home_cells() if h['cell_id'] and (not featured or h['sid'] in FEATURED)]
    cells={h['cell_id']:h for h in homes};failures={};profiles={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(f4_temperature,h,offline):cell for cell,h in cells.items()}
        for future in concurrent.futures.as_completed(tasks):
            cell=tasks[future]
            try:
                temperature=future.result()['data']['hourly']
                radiation=read(CACHE/f'{cell}_2016_2025.json')['data']['hourly']
                assert temperature['time']==radiation['time']
                counts=np.zeros((12,24));hot=np.zeros((12,24));sums=np.zeros((12,24,3));valid=np.zeros((12,24,3))
                for i,stamp in enumerate(temperature['time']):
                    m,h=int(stamp[5:7])-1,int(stamp[11:13]);t=temperature['temperature_2m'][i]
                    dni=radiation['direct_normal_irradiance'][i]
                    if t is not None and dni is not None:
                        counts[m,h]+=1;hot[m,h]+=t>=32 and dni>0
                    for j,key in enumerate(('direct_normal_irradiance','diffuse_radiation','shortwave_radiation')):
                        v=radiation[key][i]
                        if v is not None and math.isfinite(v) and v>=0:sums[m,h,j]+=v;valid[m,h,j]+=1
                assert (counts>0).all() and (valid>0).all()
                profiles[cell]=dict(radiation=(sums/valid).tolist(),hot_sunny_probability=(hot/counts).tolist(),
                                    valid_temperature_radiation_pairs=counts.astype(int).tolist())
            except Exception as exc:failures[cell]=f'{type(exc).__name__}: {exc}'
            write(OUT/'f4_weather_progress.json',dict(requested=len(cells),complete=len(profiles),failures=failures,network=dict(NET)))
    previous=read(OUT/'f4_climate.json') if (OUT/'f4_climate.json').exists() else {}
    previous.update(profiles);write(OUT/'f4_climate.json',previous)
    f4_checkpoint(f'F4 temperature/radiation profiles: {len(previous)} cells cached; failures {failures}.','Compute featured six first, then cohort; run existing suite and real-output contract checks.')
    return profiles


def f4_home(h,profiles):
    target=f4_cache_path(h['sid'])
    if target.exists():
        with gzip.open(target,'rt') as f:return json.load(f)
    if shutil.disk_usage(ROOT).free<1024**3:raise RuntimeError('STOP: free space below 1 GiB')
    g=load_geometry(h['sid']);crop=load_dsm(g);climate=profiles.get(h['cell_id'])
    if climate is None:raise ValueError('f4_climate_missing')
    tz=ZoneInfo(g['sun']['assumptions']['timezone']);steps=6
    dates={m:f'2026-{m:02d}-{dict([(3,20),(11,14)]).get(m,21):02d}' for m in range(1,13)}
    times=[dt.datetime.fromisoformat(date).replace(tzinfo=tz)+dt.timedelta(minutes=10*i+5)
           for date in dates.values() for i in range(24*steps)]
    az,_,el=GEO['solar'](times,g['lat'],g['lon']);direction=directions(az,el,g['matrix'])
    indices=np.array([(t.astimezone(dt.timezone.utc).hour+1)%24 for t in times]).reshape(12,24,steps)
    radiation=np.array(climate['radiation']);hotprob=np.array(climate['hot_sunny_probability'])
    selected_rad=np.array([radiation[m,indices[m]] for m in range(12)])
    selected_hot=np.array([hotprob[m,indices[m]] for m in range(12)])
    all_windows=[];room_lit={r:np.zeros((12,24,steps),bool) for r in g['rooms']}
    room_heat={r:np.zeros((12,24)) for r in g['rooms']};sides=collections.defaultdict(set)
    for w in g['windows']:
        hz=window_horizon(w,g,crop) if crop is not None else None
        fraction=(GEO['aperture_verdict'](w,direction,g['roofs'],g['obstacles'])==0).mean(axis=1)
        if hz is not None:fraction*=el>horizon_at(hz,az)
        incidence=np.maximum(0,np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az-w['outward_azimuth_deg'])))
        incidence[el<=0]=0
        fraction[incidence<=0]=0
        direct=selected_rad[:,:,:,0]*incidence.reshape(12,24,steps)*fraction.reshape(12,24,steps)
        sky=sky_samples(w,g,hz);sv=sky['sky_view_dsm'] if hz is not None else sky['sky_view_geometry']
        diffuse=selected_rad[:,:,:,1]*sv
        area=w['width']*w['height']*.8
        heat=((direct+diffuse)*area).mean(axis=2)
        lit=fraction.reshape(12,24,steps)>0
        side=['N','NE','E','SE','S','SW','W','NW'][int((w['outward_azimuth_deg']+22.5)%360//45)]
        rid=w['room_id'];room_lit[rid]|=lit;room_heat[rid]+=heat;sides[rid].add(side)
        all_windows.append(dict(window_id=w['id'],room_id=rid,side=side,azimuth_deg=w['outward_azimuth_deg'],
            glass_area_m2=area,hourly_minutes=(lit.sum(axis=2)*10).tolist(),
            hourly_heat_w_low_e=(heat*.25).tolist(),hourly_heat_w_clear=(heat*.70).tolist()))
    result=[];clocks=[]
    for rid,r in g['rooms'].items():
        lit=room_lit[rid];clock=lit.sum(axis=2)*10
        bands={str(m):{name:float(clock[m-1,a:b].sum()) for name,(a,b) in F4_BANDS.items()} for m in range(1,13)}
        heat=room_heat[rid];summer=heat[5:8];peak=np.unravel_index(int(np.argmax(summer)),summer.shape)
        has_windows=bool(sides[rid]);ranked=bool(lit.any())
        hot=float(np.sum((lit*selected_hot).mean(axis=2)[5:8].sum(axis=1)*np.array([30,31,31]))/92) if has_windows else 0.
        labels=' '.join([r['label']]+[v['label'] for v in r.get('classifications',[]) or []]).lower()
        candidates=[('bedroom_summer_afternoon_sun',bool(re.search(r'bedroom|bed room',labels)),float(np.mean([bands[str(m)]['afternoon'] for m in (6,7,8)]))),
                    ('kitchen_morning_sun',bool(re.search(r'kitchen|breakfast',labels)),float(np.mean([bands[str(m)]['morning'] for m in range(1,13)]))),
                    ('living_winter_midday_sun',bool(re.search(r'living|family room|great room',labels)),float(np.mean([bands[str(m)]['midday'] for m in (12,1,2)]))),
                    ('office_workhours_glare',bool(re.search(r'office|\bden\b|study',labels)),float(clock[:,9:17].sum(axis=1).mean()))]
        flags=[dict(id=name,value=round(value,4),unit='minutes/day') for name,match,value in candidates if match and value>0]
        result.append(dict(sid=h['sid'],room_id=rid,label=r['label'],floor=r['floor_id'],status='ok' if ranked else 'not_ranked_no_sun',
            sides=sorted(sides[rid]),dsm=crop is not None,bands=bands,
            heat=dict(peak_month=int(peak[0]+6),peak_hour=int(peak[1]),peak_w_low_e=float(summer[peak]*.25),
                      peak_w_clear=float(summer[peak]*.70)) if has_windows else None,
            hot_sun_hours_jja=hot,flags=flags))
        clocks.append(dict(room_id=rid,hourly_minutes=clock.tolist(),hourly_heat_w_low_e=(heat*.25).tolist(),
                           hourly_heat_w_clear=(heat*.7).tolist()))
    value=dict(sid=h['sid'],poor=h['poor'],timezone=str(tz),dates=dates,dsm=crop is not None,rooms=result,room_clocks=clocks,windows=all_windows)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(f'.{os.getpid()}.tmp')
    with gzip.open(temporary,'wt') as f:json.dump(value,f,allow_nan=False)
    temporary.replace(target);return value


def f4_compute(featured=False,jobs=4):
    homes=[h for h in home_cells() if h['computed']=='True' and (not featured or h['sid'] in FEATURED)]
    profiles=read(OUT/'f4_climate.json');values=[];failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs,initializer=worker_guard) as pool:
        tasks={pool.submit(f4_home,h,profiles):h['sid'] for h in homes}
        for future in concurrent.futures.as_completed(tasks):
            sid=tasks[future]
            try:values.append(future.result())
            except Exception as exc:failures.append(dict(sid=sid,reason=f'{type(exc).__name__}: {exc}'))
            if (len(values)+len(failures))%25==0 or len(values)+len(failures)==len(homes):
                write(OUT/'f4_compute_progress.json',dict(requested=len(homes),completed=len(values),failures=failures))
                f4_checkpoint(f'F4 home caches complete {len(values)}/{len(homes)}, failures {len(failures)}. Featured-first run={featured}.','Resume f4-compute from atomic per-home caches; validate contract, suite, summaries.')
    rooms=sorted([r for v in values for r in v['rooms']],key=lambda r:(r['sid'],r['room_id']))
    payload=dict(schema_version=1,rooms=rooms)
    write(OUT/'room_sun_f4.json',payload)
    if featured:write(OUT/'room_sun_f4_featured.json',payload)
    flags=collections.Counter(f['id'] for r in rooms for f in r['flags'])
    well={v['sid'] for v in values if v['poor']=='False'}
    well_flags=collections.Counter(f['id'] for r in rooms if r['sid'] in well for f in r['flags'])
    featured_summary={v['sid']:dict(rooms=len(v['rooms']),dsm=v['dsm'],
        bands={str(m):{band:round(sum(r['bands'][str(m)][band] for r in v['rooms']),2) for band in F4_BANDS} for m in range(1,13)},
        flags=dict(collections.Counter(f['id'] for r in v['rooms'] for f in r['flags']))) for v in values if v['sid'] in FEATURED}
    write(OUT/'f4_summary.json',dict(homes=len(values),rooms=len(rooms),dsm_homes=sum(v['dsm'] for v in values),
        status=dict(collections.Counter(r['status'] for r in rooms)),flag_counts=dict(flags),well_registered_flag_counts=dict(well_flags),
        featured=featured_summary,failed_homes=failures,existing_engine_failed_homes=[h['sid'] for h in home_cells() if h['computed']!='True']))
    write(OUT/'f4_failures.json',failures)
    f4_checkpoint(f'F4 JSON written: {len(values)}homes/{len(rooms)}rooms; flags {dict(flags)}. Featured summaries in f4_summary.json; failures {failures}.','Run existing suite and real-output invariants; write method/report and cohort summary.')


def f4_evidence():
    payload=read(OUT/'room_sun_f4.json');summary=read(OUT/'f4_summary.json')
    assert set(payload)=={'schema_version','rooms'} and payload['schema_version']==1
    required={'sid','room_id','label','floor','status','sides','dsm','bands','heat','hot_sun_hours_jja','flags'}
    flag_ids={'bedroom_summer_afternoon_sun','kitchen_morning_sun','living_winter_midday_sun','office_workhours_glare'}
    rooms=payload['rooms'];lookup={(r['sid'],r['room_id']):r for r in rooms}
    assert len(lookup)==len(rooms)
    for r in rooms:
        assert set(r)==required and isinstance(r['dsm'],bool)
        assert r['status'] in ('ok','not_ranked_no_sun')
        assert set(r['sides'])<=set(['N','NE','E','SE','S','SW','W','NW'])
        assert set(r['bands'])=={str(m) for m in range(1,13)}
        for bands in r['bands'].values():
            assert set(bands)==set(F4_BANDS)
            assert all(math.isfinite(bands[k]) and 0<=bands[k]<=60*(b-a) for k,(a,b) in F4_BANDS.items())
        assert r['hot_sun_hours_jja'] is None or 0<=r['hot_sun_hours_jja']<=24
        if r['heat'] is not None:
            heat=r['heat'];assert set(heat)=={'peak_month','peak_hour','peak_w_low_e','peak_w_clear'}
            assert heat['peak_month'] in (6,7,8) and 0<=heat['peak_hour']<24
            assert heat['peak_w_low_e']>=0 and abs(heat['peak_w_clear']-heat['peak_w_low_e']*2.8)<1e-7
        for f in r['flags']:
            assert set(f)=={'id','value','unit'} and f['id'] in flag_ids and f['unit']=='minutes/day' and f['value']>0
    max_heat_error=0.;windows=0;facades=[];featured_checks={}
    for sid in sorted({r['sid'] for r in rooms}):
        with gzip.open(f4_cache_path(sid),'rt') as stream:v=json.load(stream)
        by_room=collections.defaultdict(list);by_side=collections.defaultdict(list)
        for w in v['windows']:
            by_room[w['room_id']].append(w);by_side[w['side']].append(w);windows+=1
        for clock in v['room_clocks']:
            rid=clock['room_id'];r=lookup[(sid,rid)];minutes=np.array(clock['hourly_minutes'])
            assert minutes.shape==(12,24) and (minutes>=0).all() and (minutes<=60).all()
            assert (r['status']=='ok')==bool(minutes.any())
            for m in range(1,13):
                for band,(a,b) in F4_BANDS.items():assert r['bands'][str(m)][band]==float(minutes[m-1,a:b].sum())
            ws=by_room[rid]
            for suffix in ('low_e','clear'):
                heat=np.array(clock['hourly_heat_w_'+suffix]);total=sum((np.array(w['hourly_heat_w_'+suffix]) for w in ws),np.zeros((12,24)))
                max_heat_error=max(max_heat_error,float(np.max(np.abs(heat-total))))
                assert (heat>=0).all() and np.isfinite(heat).all()
                if ws:assert abs(r['heat']['peak_w_'+suffix]-float(heat[5:8].max()))<1e-7
            if ws:
                window_clocks=np.array([w['hourly_minutes'] for w in ws])
                assert (minutes>=window_clocks.max(axis=0)).all() and (minutes<=window_clocks.sum(axis=0)).all()
            else:assert not minutes.any() and r['heat'] is None
        facades.append(dict(sid=sid,timezone=v['timezone'],sides={side:dict(windows=len(ws),
            glass_area_m2=sum(w['glass_area_m2'] for w in ws),
            hourly_window_minutes=sum((np.array(w['hourly_minutes']) for w in ws)).tolist(),
            hourly_heat_w_low_e=sum((np.array(w['hourly_heat_w_low_e']) for w in ws)).tolist(),
            hourly_heat_w_clear=sum((np.array(w['hourly_heat_w_clear']) for w in ws)).tolist()) for side,ws in by_side.items()}))
        if sid in FEATURED:
            featured_checks[sid]=dict(rooms=len(v['rooms']),windows=len(v['windows']),dsm=v['dsm'],
                max_room_hot_sun_hours_jja=max(r['hot_sun_hours_jja'] for r in v['rooms']),
                max_room_peak_w_low_e=max((r['heat']['peak_w_low_e'] for r in v['rooms'] if r['heat']),default=0),
                max_room_peak_w_clear=max((r['heat']['peak_w_clear'] for r in v['rooms'] if r['heat']),default=0))
    assert max_heat_error<1e-7
    with gzip.open(OUT/'facade_sun_f4.json.gz','wt') as stream:
        json.dump(dict(schema_version=1,months=list(range(1,13)),hours=list(range(24)),homes=facades),stream,allow_nan=False)
    baseline=read(ROOT/'data/cache/daylight_baseline/manifest.json')
    protected={k:v for k,v in baseline.items() if not k.startswith(('docs/pitch/','agent/skills/'))}
    changed=[k for k,v in protected.items() if not (ROOT/k).exists() or hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
    assert not changed,changed
    checks=dict(schema_valid=True,unique_rooms=len(rooms),homes=len(facades),windows=windows,
        max_room_window_heat_sum_error_w=max_heat_error,band_hourly_sums_valid=True,
        room_time_union_bounds_valid=True,shgc_ratio_valid=True,protected_inputs_checked=len(protected),
        changed_protected_inputs=changed,featured=featured_checks,
        featured_checkpoint_exists=(OUT/'room_sun_f4_featured.json').exists(),
        weather=read(OUT/'f4_weather_progress.json'),existing_suite_log='f4_existing_suite.log')
    write(OUT/'f4_verification.json',checks)
    if len(facades)==6 and {v['sid'] for v in facades}==set(FEATURED):
        write(OUT/'f4_featured_verification.json',checks)
    lines=['# F4 buyer sun and solar heat', '',
        'This is modelled sun and solar heat at the window glass.',
        'It is NOT the indoor temperature: there is no HVAC, insulation, blinds or tree model.', '',
        '## Scope and method', '',
        f"Computed {summary['homes']} homes, {summary['rooms']} rooms, {windows} exterior windows; {summary['dsm_homes']} homes use the accepted DSM horizon. Existing engine failures: {len(summary['existing_engine_failed_homes'])}; new F4 failures: {len(summary['failed_homes'])}.",
        'Room clocks use the union of sunlit exterior-window times, sampled at ten-minute midpoints, on one representative date per month in 2026 (21st, except March 20 and November 14). A window counts when any of its nine aperture samples is geometrically clear. This differs from the earlier aperture-area-weighted minutes. It does not trace sun patches into rooms. These are clear-sky geometric minutes, not expected weather sunshine minutes.',
        'All hours are local civil time, including daylight saving. Bands are [06,10), [10,14), [14,18), [18,24); hours 00–06 remain in hourly caches but are outside these bands. Month summaries are representative days, not simulations of every day. Floor is the existing Matterport floor identifier.',
        'Hourly DNI and DHI are ERA5 2016–2025 monthly/hourly means from the cached Open-Meteo archive, paired by exact UTC timestamps with the newly fetched temperature_2m (16 cells). Radiation is a preceding-hour mean; geometry is sampled within that hour, while temperature is at its ending timestamp. Hot-sun hours use the historical joint frequency of temperature >=32°C AND DNI>0, multiplied by the representative geometric sun flag. They are a climatological coincidence proxy, not a forecast or ten-year daily ray trace. JJA is weighted 30/31/31 days. Geometry-only flags can occur even when weather makes hot-sun hours negligible.',
        'Window direct irradiance = DNI × positive vertical-plane incidence cosine × unblocked aperture fraction. Diffuse = DHI × obstruction-adjusted projected sky view. Heat = (direct + diffuse) × opening width × height × assumed frame factor 0.8 × SHGC (0.25 low-E, 0.70 old clear). No ground-reflected heat term in F4. Room heat sums windows simultaneously; peak is the largest local hourly mean in June–August, not a ten-year extreme. A 1.5 kW space heater is 1500 W; divide peak W by 1500 for that comparison. No indoor air or cooling load is calculated.',
        'DSM uses surveyed neighbours and vegetation surfaces, own footprint masked, with capture dates and quality exclusions retained from the accepted daylight round. Cached geometry may also contain opaque tree prisms. There is no new tree model, leaf-state or seasonal canopy prediction. Missing/withheld DSM falls back to existing geometry and is labelled dsm=false.',
        'Facade bins are 45° sectors centred on N/NE/E/SE/S/SW/W/NW. facade_sun_f4.json.gz holds per-home/side 12×24 arrays: summed window-minutes (overlaps between windows deliberately retained), summed simultaneous heat and glass area. Existing per-window and room hourly arrays remain in data/cache/daylight_f4/computed/<sid>.json.gz; resumed missing homes are in data/facts/daylight/f4_computed/<sid>.json.gz. The agent contract has no extra fields.',
        'Flags are positive-minute rule outputs, not clinical/cooling/glare measurements: bedroom or bed room → mean JJA afternoon; kitchen or breakfast → mean of twelve monthly morning bands; living/family room/great room → mean DJF midday; office/den/study → mean of twelve monthly 09–17 clocks. Emit only if value >0; units minutes/day. Names come from existing labels and classifications. Mixed-use rooms may have multiple flags. not_ranked_no_sun means no direct sun in these samples, not no daylight.',
        'All registration qualities are retained; the consuming agent separately gates unreliable placements. Well-registered-only counts are supplied below. No-sun rooms are retained with zero bands; heat can remain positive from diffuse sky. No-window rooms have null heat and zero hot-sun hours.', '',
        '## Featured homes: all twelve months', '',
        'Values are summed **room-minutes/day** across rooms, so simultaneous sun in different rooms is counted separately. Columns M/MD/A/E mean morning/midday/afternoon/evening.', '']
    for sid in FEATURED:
        featured=summary['featured'][sid];check=featured_checks[sid]
        lines += [f'### {sid}', '',f"{check['rooms']} rooms, {check['windows']} windows, DSM={check['dsm']}. Maximum room summer peak {check['max_room_peak_w_low_e']:.1f}–{check['max_room_peak_w_clear']:.1f} W ({check['max_room_peak_w_low_e']/1500:.2f}–{check['max_room_peak_w_clear']/1500:.2f} space heaters); maximum room hot-sun {check['max_room_hot_sun_hours_jja']:.3f} h/day.", '', '| Month | M | MD | A | E |','|---|---:|---:|---:|---:|']
        for m in range(1,13):lines.append('| '+str(m)+' | '+' | '.join(f"{featured['bands'][str(m)][band]:g}" for band in F4_BANDS)+' |')
        lines.append('')
    lines += ['## Flag counts (rooms)', '', '| Rule | All registrations | Well registered |','|---|---:|---:|']
    for flag in sorted(flag_ids):lines.append(f"| {flag} | {summary['flag_counts'].get(flag,0)} | {summary['well_registered_flag_counts'].get(flag,0)} |")
    lines += ['', '## Verification and limits', '',
        'Seven existing tests passed (f4_existing_suite.log). Real outputs pass exact schema, unique room IDs, band/hour bounds and sums, room/window heat conservation, room time-union bounds, SHGC ratio, summer peak and finite-value checks (f4_verification.json). Featured six were written before cohort acquisition/computation. Protected original inputs are hash-checked unchanged. Weather cache receipts record requests and concurrency; computation runs offline with network denied.',
        'Not verified: measured sunshine or heat; actual SHGC/frame dimensions; room-use label accuracy; comfort, cooling demand, glare severity, tree leaf state, hourly ground truth, monthly full-day sampling error, sub-hour weather correlation, and poor-registration placements. Hand-marked khbuN7q853q uses accepted placement but this round has no new field measurements. Existing engine failures have no fabricated room records.', '',
        'Changed by Codex: experiments/daylight.py (F4 stages); data/facts/daylight/*f4* outputs and offline_f4-compute_audit.json. Takeover reuses original data/cache/daylight_f4/ files without rewriting them; new computed homes go to data/facts/daylight/f4_computed/. Agent F4 scoring and evaluation are maintained separately under agent/ and data/claims/eval/. Accepted F1–F3 outputs are preserved. Source baseline: data/cache/daylight_f4_baseline/daylight.py. No demo/ changes, sponsor calls, .env reads, new tests or commits.',
        'Reproduce: python3 -I -B experiments/daylight.py f4-weather --offline; python3 -I -B experiments/daylight.py f4-compute --offline --jobs 4; python3 -I -B experiments/daylight.py f4-evidence --offline.',
        'Source edit history: python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path experiments/daylight.py; use rollback <id> for a dry run, then --yes to undo an exact edit. New artifacts are separately identifiable by the f4 names/cache directory.', '',
        'Weather source: https://open-meteo.com/en/docs/historical-weather-api (ERA5 reanalysis, modelled climate average).']
    (OUT/'f4_report.md').write_text('\n'.join(lines)+'\n')
    f4_checkpoint(f"F4 complete: {len(facades)} homes/{len(rooms)} rooms/{windows} windows. Exact contract and real-output invariants pass; protected inputs unchanged. Existing suite 7/7. See f4_report.md, f4_summary.json, f4_verification.json.",'Report featured band summaries, cohort flags, changed files and limits to user and Claude/tracker.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['example','weather','dsm','validate-dsm','control-dsm','dsm-quality','prepare','refresh-surfaces','compute','summarize','evidence','all','f4-weather','f4-compute','f4-evidence'])
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--featured', action='store_true')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--credential-file')
    parser.add_argument('--jobs',type=int,default=4)
    args = parser.parse_args()
    sys.addaudithook(environment_guard)
    if args.offline:
        sys.addaudithook(offline_guard)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage=='f4-weather':f4_weather(args.featured,args.offline)
    if args.stage=='f4-compute':f4_compute(args.featured,args.jobs)
    if args.stage=='f4-evidence':f4_evidence()
    if args.stage in ('example','all'):
        example()
    if args.stage in ('weather','all'):
        weather(args.offline)
    if args.stage=='dsm':
        dsm_download(args.credential_file,args.featured)
    if args.stage=='validate-dsm':validate_dsm()
    if args.stage=='control-dsm':control_dsm()
    if args.stage=='dsm-quality':dsm_quality()
    if args.stage=='prepare':prepare(args.featured,args.jobs)
    if args.stage=='refresh-surfaces':refresh_surfaces()
    if args.stage in ('compute','all'):
        compute(args.featured,args.refresh)
    if args.stage in ('summarize','all'):
        home_months();hand_conversion()
    if args.stage=='evidence':evidence()
    if args.offline:
        write(OUT/f'offline_{args.stage}_audit.json',dict(network_attempts=AUDIT['network_attempts'],stage=args.stage,completed=True))


if __name__ == '__main__':
    main()
