"""Read-only bridge to Brief C. All derived output lives under demo/."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.dontwritebytecode = True
os.environ['MPLCONFIGDIR'] = str(ROOT/'demo/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
spec = importlib.util.spec_from_file_location('brief_c_demo', ROOT/'experiments/sun_engine.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
os.environ['MPLCONFIGDIR'] = str(ROOT/'demo/cache/matplotlib')
SOURCE_SHA = hashlib.sha256((ROOT/'experiments/sun_engine.py').read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2).tolist()

def build(sid):
    source = ROOT/'data/sun'/sid
    sun = read(source/'sun.json')
    samples = read(source/'sample_verdicts.json')
    model = read(ROOT/'data/cache'/sid/'matterport.json')['data']['model']
    fit = read(ROOT/'data/cache'/sid/'registration.json')
    raw, lat, lon = engine.geo_matrix(model)
    matrix, shift = engine.alignment(fit, raw)
    rooms, _ = engine.room_data(sid, model)
    windows, _ = engine.exterior_openings(model, rooms, matrix)
    windows = {w['id']:w for w in windows}
    engine.A.TZ = ZoneInfo(sun['assumptions']['timezone'])
    geometry = {}
    for room in rooms:
        geometry[room['id']] = {'segments':np.round(room['segments'], 5).tolist(),
            'holes':[np.round(h, 5).tolist() for h in room['holes']],
            'grid':np.round(room['grid'], 5).tolist()}
    max_error = 0.
    for date, sample in samples.items():
        times, weights, directions, az, el, rise, sunset = engine.sample_day(date, lat, lon, matrix)
        if len(times) != len(sample['local_times']):
            raise RuntimeError('Engine/source sampling changed: '+sid)
        if not np.allclose(az, sample['sun_azimuth_deg'], atol=1e-7):
            raise RuntimeError('Engine/source solar positions changed: '+sid)
        # Find solar noon using the engine's solar function, not a second model.
        peak = times[int(np.argmax(el))]
        candidates = [peak+dt.timedelta(seconds=i) for i in range(-300, 301)]
        noon = candidates[int(np.argmax(engine.A.solar(candidates, lat, lon)[2]))]
        sample['solar_noon'] = noon.astimezone(engine.A.TZ).isoformat()
        sample['solar_noon_definition'] = 'Engine solar-elevation maximum, one-second resolution'
        sample['room_lit_runs'] = {}
        for room in rooms:
            mask = np.zeros((len(times), len(room['grid'])), dtype=bool)
            for w in sun['windows']:
                if w['room_id'] != room['id']: continue
                if w['id'] not in windows:
                    raise RuntimeError('Engine/source opening changed: '+sid+'/'+w['id'])
                codes = np.asarray(sample['window_sample_codes'][w['id']], dtype=np.uint8)
                mask |= engine.floor_patch(windows[w['id']], room, directions, codes)
            areas = mask.sum(1)*.01
            saved = np.asarray(sample['room_floor_area_m2'][room['id']])
            error = float(np.max(np.abs(areas-saved)))
            max_error = max(max_error, error)
            if error > 1e-8: raise RuntimeError('Engine cells differ from published areas: '+sid)
            sample['room_lit_runs'][room['id']] = [runs(m) for m in mask]
            published = next(r for r in sun['rooms'] if r['id']==room['id'])['dates'][date]
            qualifying = areas >= .1
            before = np.array([max(0., min(float(w), (noon-t).total_seconds()/60)) for t,w in zip(times,weights)])
            published['before_solar_noon_minutes'] = round(float(before[qualifying].sum()), 2)
            published['after_solar_noon_minutes'] = round(float((weights-before)[qualifying].sum()), 2)
            if abs(float(weights[qualifying].sum())-published['lit_minutes_ge_0_1m2']) > .011:
                raise RuntimeError('Engine minutes differ from published totals: '+sid)
    for w in sun['windows']:
        room = next(r for r in sun['rooms'] if r['id']==w['room_id'])
        w['sill_above_room_floor_m'] = round(w['lower_z_m']-room['floor_z_m'], 4)
        w['floor_id'] = room['floor_id']
    sun.update(geometry=geometry, days=samples,
        transform={'delivered_matrix':raw.tolist(), 'registered_matrix':matrix.tolist(),
            'registered_shift':shift.tolist(), 'origin_lat':lat, 'origin_lon':lon},
        engine_provenance={'source':'experiments/sun_engine.py','sha256':SOURCE_SHA,
            'functions':['room_data','exterior_openings','sample_day','floor_patch','A.solar'],
            'max_cell_area_difference_m2':max_error,'cells_checked_against':'sample_verdicts.json, every room and five-minute daylight bin',
            'aperture_codes':'Published engine sample_verdicts.json, 3x3 rays per opening'})
    return sun, model
