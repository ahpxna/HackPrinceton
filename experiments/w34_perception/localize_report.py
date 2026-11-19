#!/usr/bin/env python3
"""Contact sheets, storey maps and acceptance checks for the W34 3D OWLv2 objects (objects_hf/ from localize.py).

Checks: the down-ray floor test, localize's exact depths against the older 5 cm voxel march, SAM 3 doors seen
from two captures (correspondence by projection, not 3D proximity), reprojection of confirmed centroids, and the
expected outcomes for this home.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'w34_navigation'))
import navgrid  # noqa: E402
from localize import Caster, cast as voxel_cast  # noqa: E402

TILE = 320
T_MIN, T_MAX = 0.15, 40.0
GRID = 5
DOOR_BIT, MASK = 6, 2048
CODES = {'refrigerator': 'RF', 'dishwasher': 'DW', 'oven': 'OV', 'stove': 'ST', 'microwave': 'MW', 'range hood': 'RH',
         'washing machine': 'WM', 'clothes dryer': 'CD', 'ceiling fan': 'CF', 'fireplace': 'FP', 'air conditioner': 'AC',
         'water heater': 'WH', 'bathtub': 'BT', 'shower': 'SH', 'toilet': 'TO', 'sink': 'SK', 'kitchen island': 'KI',
         'wine cooler': 'WC', 'television': 'TV', 'smoke detector': 'SD', 'thermostat': 'TH', 'chandelier': 'CH',
         'swimming pool': 'PL', 'hot tub': 'HT', 'solar panel': 'SP', 'garage door': 'GD', 'bed': 'BD', 'sofa': 'SF'}


def face_rays(cap, face, xy):
    """Unit world rays through normalised pixels (x right, y down) of a cube face (W33 convention)."""
    c, r, d = np.asarray(cap['face_bases_center_right_down'], float)[face]
    out = c + (2 * xy[:, :1] - 1) * r + (2 * xy[:, 1:2] - 1) * d
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def project(cap, p):
    """World point -> (face, x, y) on the face it falls in, or None."""
    basis = np.asarray(cap['face_bases_center_right_down'], float)
    delta = np.asarray(p) - np.asarray(cap['position'])
    for k, (c, r, d) in enumerate(basis):
        z = delta @ c
        if z <= 0:
            continue
        u, v = delta @ r / z, delta @ d / z
        if abs(u) <= 1 and abs(v) <= 1:
            return k, (u + 1) / 2, (v + 1) / 2
    return None


def project_on(cap, face, p):
    c, r, d = np.asarray(cap['face_bases_center_right_down'], float)[face]
    delta = np.asarray(p) - np.asarray(cap['position'])
    z = delta @ c
    return None if z <= 0 else ((delta @ r / z + 1) / 2, (delta @ d / z + 1) / 2)


def stats(a):
    a = np.asarray(a, float)
    return dict(n=int(a.size), median=round(float(np.median(a)), 4), p90=round(float(np.quantile(a, .9)), 4),
                max=round(float(a.max()), 4)) if a.size else dict(n=0)


def tile(faces, cap, face, box, caption, colour):
    im = Image.open(faces / cap / f'face{face}.jpg').convert('RGB').resize((TILE, TILE), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle([v * TILE for v in box], outline=colour, width=3)
    for i, line in enumerate(caption):
        d.rectangle([0, 13 * i, 6 * len(line) + 4, 13 * i + 13], fill=(0, 0, 0))
        d.text((2, 13 * i), line, fill=colour)
    return im


def sheets(objects, faces, out):
    by = defaultdict(list)
    for o in objects:
        by[o['label']].append(o)
    made = []
    colour = dict(confirmed=(60, 255, 60), candidate=(255, 190, 0), rejected=(255, 70, 70))
    for label, os_ in sorted(by.items()):
        good = [o for o in os_ if o['status'] != 'rejected']
        if not good:
            continue
        pick = sorted(good, key=lambda o: (o['status'] != 'confirmed', -o['score_max']))[:16]
        pick += sorted([o for o in os_ if o['status'] == 'rejected'], key=lambda o: -o['score_max'])[:4]
        tiles = []
        for o in pick:
            b = o['best_view']
            cap = [f'{o["id"]} {o["status"]} s{o["score_max"]:.2f}',
                   f'sup {o["support"]}/{o["n_captures"]} {o["scope"]}',
                   f'size {o["size_m"]:.2f} m  h {o["h_m"]:.2f} m  st{o["storey"]}']
            if o['status'] == 'rejected' and o['flags']:
                cap.append(o['flags'][0][:50])
            tiles.append(tile(faces, b['capture'], b['face'], b['box'], cap, colour[o['status']]))
        sheet = Image.new('RGB', (4 * TILE, ((len(tiles) + 3) // 4) * TILE), (20, 20, 20))
        for i, im in enumerate(tiles):
            sheet.paste(im, ((i % 4) * TILE, (i // 4) * TILE))
        path = out / 'sheets' / f'{label.replace(" ", "_")}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, quality=85)
        made.append(path.name)
    return made


def maps(objects, caps, nav, navdir, out, scale=2):
    lo, res = np.array(nav['grid']['origin'], float), nav['params']['res']
    H = nav['grid']['cells_xy'][1]
    px = lambda x, y: ((x - lo[0]) / res * scale, (H - (y - lo[1]) / res) * scale)  # noqa: E731
    floor_of = {r['id']: r['storey'] for r in nav['captures']}
    made = []
    for s in range(len(nav['storeys_floor_z'])):
        im = Image.open(navdir / f'map_band{s}.png').convert('RGB')
        im = Image.blend(im, Image.new('RGB', im.size, (255, 255, 255)), 0.45)
        d = ImageDraw.Draw(im)
        for cid, cap in caps.items():
            if floor_of.get(cid) == s:
                x, y = px(*cap['position'][:2])
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(90, 90, 90))
        for o in sorted(objects, key=lambda o: o['status'] == 'confirmed'):
            if o['storey'] != s or o['status'] == 'rejected':
                continue
            x, y = px(*o['centroid'][:2])
            col = (0, 150, 0) if o['status'] == 'confirmed' else (230, 140, 0)
            rad = 6 if o['status'] == 'confirmed' else 4
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col, outline=(0, 0, 0))
            if o['status'] == 'confirmed':
                d.text((x + 7, y - 6), CODES.get(o['label'], '??'), fill=(0, 0, 0))
        legend = ['green = confirmed (code), orange = candidate, grey = capture',
                  '  '.join(f'{c}={k}' for k, c in list(CODES.items())[:14]),
                  '  '.join(f'{c}={k}' for k, c in list(CODES.items())[14:])]
        for i, line in enumerate(legend):
            d.rectangle([4, 4 + 14 * i, 8 + 6 * len(line), 18 + 14 * i], fill=(255, 255, 255))
            d.text((6, 5 + 14 * i), line, fill=(0, 0, 0))
        path = out / f'map_storey{s}.png'
        im.save(path)
        made.append(path.name)
    return made


def check_floor(caster, caps, nav):
    floor = {r['id']: r['floor_z'] for r in nav['captures']}
    rows = []
    for cid, cap in caps.items():
        basis = np.asarray(cap['face_bases_center_right_down'], float)
        down = int(np.argmin(basis[:, 0, 2]))
        t = caster.cast(cap['position'], face_rays(cap, down, np.array([[0.5, 0.5]])))[0]
        rows.append(dict(capture=cid, camera_height=round(cap['position'][2] - floor[cid], 3),
                         exact_down_hit=None if np.isnan(t) else round(float(t), 4)))
    err = [abs(r['exact_down_hit'] - r['camera_height']) for r in rows if r['exact_down_hit'] is not None]
    worst = sorted(rows, key=lambda r: -abs((r['exact_down_hit'] or 0) - r['camera_height']))[:5]
    return dict(hits=len(err), captures=len(rows), abs_error_m=stats(err),
                within_5cm=sum(e <= 0.05 for e in err), worst=worst)


def check_depth(caster, caps, dets, occ, lo, res, n=300, seed=7):
    """localize v5 depth (exact) against the v4 method (voxel march) on a seeded sample, and a recompute."""
    pool = [d for d in dets if 'depth_m' in d and not d['in_mirror']]
    rng = np.random.default_rng(seed)
    pick = [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]
    g = 0.2 + 0.6 * np.arange(GRID) / (GRID - 1)
    gx, gy = np.meshgrid(g, g)
    inner = np.c_[gx.ravel(), gy.ravel()]
    rows = []
    for d in sorted(pick, key=lambda d: d['capture']):
        x0, y0, x1, y1 = d['box']
        xy = np.c_[x0 + inner[:, 0] * (x1 - x0), y0 + inner[:, 1] * (y1 - y0)]
        o, D = np.asarray(caps[d['capture']]['position'], float), face_rays(caps[d['capture']], d['face'], xy)
        h_exact, h_vox = caster.cast(o, D), voxel_cast(np.repeat(o[None], len(D), 0), D, occ, lo, res)
        q = lambda h: float(np.nanquantile(h, .25)) if np.isfinite(h).mean() >= 0.3 else None  # noqa: E731
        rows.append(dict(id=d['id'], label=d['label'], localize=d['depth_m'], recomputed=q(h_exact), voxel=q(h_vox)))
    rec = [abs(r['localize'] - r['recomputed']) for r in rows if r['recomputed'] is not None]
    diff = [r['voxel'] - r['localize'] for r in rows if r['voxel'] is not None]
    return dict(sample=len(rows), recompute_abs_m=stats(rec), voxel_minus_exact_m=stats(diff), voxel_abs_m=stats(np.abs(diff)),
                voxel_within_10cm=int(sum(abs(x) <= 0.10 for x in diff)), voxel_compared=len(diff),
                worst=sorted([r for r in rows if r['voxel'] is not None], key=lambda r: -abs(r['voxel'] - r['localize']))[:5])


def check_doors(caster, caps, result, min_score=0.6):
    """Each strong SAM 3 door instance -> 3D point (mask centroid ray, exact hit). Correspondence between
    captures comes from projecting that point into the other capture's face and finding the SAM 3 door box
    that contains it (with a line-of-sight test), never from 3D proximity of the two estimates."""
    inst = []
    for jf in sorted((result / 'sam3').glob('*/face*.json')):
        d = json.loads(jf.read_text())
        doors = [i for i in d['instances'] if i['prompt'] == 'door' and i['score'] >= min_score]
        if not doors:
            continue
        bits = (np.asarray(Image.open(jf.with_suffix('.png'))) >> DOOR_BIT) & 1
        for i in doors:
            x0, y0, x1, y1 = (np.clip(np.array(i['box']) * MASK, 0, MASK - 1)).astype(int)
            ys, xs = np.nonzero(bits[y0:y1 + 1, x0:x1 + 1])
            if xs.size < 50:
                continue
            cx, cy = (xs.mean() + x0 + .5) / MASK, (ys.mean() + y0 + .5) / MASK
            inst.append(dict(capture=d['capture'], face=d['face'], box=i['box'], score=i['score'], xy=(cx, cy),
                             border=bool(min(i['box'][:2]) < 0.01 or max(i['box'][2:]) > 0.99)))
    for i in sorted(inst, key=lambda i: i['capture']):
        cap = caps[i['capture']]
        t = caster.cast(cap['position'], face_rays(cap, i['face'], np.array([i['xy']])))[0]
        i['point'] = None if np.isnan(t) else (np.asarray(cap['position']) + face_rays(cap, i['face'], np.array([i['xy']]))[0] * t)
    by_face = defaultdict(list)
    for k, i in enumerate(inst):
        by_face[(i['capture'], i['face'])].append(k)
    pairs, seen = [], set()
    for a, A in enumerate(inst):
        if A['point'] is None:
            continue
        for cid, cap in caps.items():
            if cid == A['capture'] or np.linalg.norm(np.asarray(cap['position']) - A['point']) > 10:
                continue
            pr = project(cap, A['point'])
            if pr is None:
                continue
            face, x, y = pr
            hits = [b for b in by_face.get((cid, face), [])
                    if inst[b]['box'][0] <= x <= inst[b]['box'][2] and inst[b]['box'][1] <= y <= inst[b]['box'][3]]
            if len(hits) != 1 or inst[hits[0]]['point'] is None or (min(a, hits[0]), max(a, hits[0])) in seen:
                continue
            ray = A['point'] - np.asarray(cap['position'])
            dist = np.linalg.norm(ray)
            t = caster.cast(cap['position'], (ray / dist)[None], t_max=dist + 1)[0]
            if np.isnan(t) or abs(t - dist) > 0.15:          # occluded: the box belongs to another door
                continue
            b = hits[0]
            seen.add((min(a, b), max(a, b)))
            B = inst[b]
            pairs.append(dict(a=f'{A["capture"]}:{A["face"]}', b=f'{B["capture"]}:{B["face"]}',
                              dist_m=round(float(np.linalg.norm(A['point'] - B['point'])), 3),
                              both_unclipped=not (A['border'] or B['border']),
                              point_a=[round(float(v), 3) for v in A['point']],
                              point_b=[round(float(v), 3) for v in B['point']]))
    full = [p['dist_m'] for p in pairs if p['both_unclipped']]
    allp = [p['dist_m'] for p in pairs]
    example = min((p for p in pairs if p['both_unclipped']), key=lambda p: abs(p['dist_m'] - np.median(full)), default=None)
    return dict(instances=len(inst), pairs=len(pairs), all_pairs_m=stats(allp),
                within_0_3m_all=int(sum(x <= 0.3 for x in allp)), unclipped_pairs_m=stats(full),
                within_0_3m_unclipped=int(sum(x <= 0.3 for x in full)), median_example=example)


def check_reprojection(objects, caps, dets_by_id, rules):
    inside = grown = total = 0
    split = {'at_or_above_thr': [0, 0], 'below_thr': [0, 0]}
    worst = []
    for o in objects:
        if o['status'] != 'confirmed':
            continue
        for m in o['members']:
            d = dets_by_id[m]
            total += 1
            pr = project_on(caps[d['capture']], d['face'], o['centroid'])
            if pr is None:
                worst.append((o['id'], m, 'behind camera'))
                continue
            x, y = pr
            x0, y0, x1, y1 = d['box']
            mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
            inside += x0 <= x <= x1 and y0 <= y <= y1
            ok = x0 - mx <= x <= x1 + mx and y0 - my <= y <= y1 + my
            grown += ok
            k = 'at_or_above_thr' if d['score'] >= rules[o['label']]['thr'] else 'below_thr'
            split[k][0] += ok
            split[k][1] += 1
            if not ok:
                worst.append((o['id'], m, f'({x:.3f},{y:.3f}) vs {d["box"]}'))
    return dict(members=total, inside_box=int(inside), inside_fraction=round(inside / max(total, 1), 3),
                inside_grown_25pct=int(grown), grown_fraction=round(grown / max(total, 1), 3),
                grown_by_view={k: dict(inside=int(a), of=n, fraction=round(a / max(n, 1), 3)) for k, (a, n) in split.items()},
                misses=worst[:12])


def check_outcomes(objects):
    conf = [o for o in objects if o['status'] == 'confirmed']
    by = lambda label: [o for o in conf if o['label'] == label]  # noqa: E731
    tubs, gds, sds, ths = by('bathtub'), by('garage door'), by('smoke detector'), by('thermostat')
    near = 0
    for i, a in enumerate(sds):
        for b in sds[i + 1:]:
            near += a['storey'] == b['storey'] and np.linalg.norm(np.subtract(a['centroid'][:2], b['centroid'][:2])) < 1.0
    return {
        'television_confirmed_0': dict(got=len(by('television')), ok=len(by('television')) == 0),
        'solar_home_confirmed_0': dict(got=sum(o['scope'].startswith('home') for o in by('solar panel')),
                                       ok=not any(o['scope'].startswith('home') for o in by('solar panel'))),
        'bathtub_2_to_4_indoor_low': dict(got=[dict(id=o['id'], h_m=o['h_m'], covered_fraction=o['covered_fraction'],
                                                    storey=o['storey'], centroid=o['centroid']) for o in tubs],
                                          ok=2 <= len(tubs) <= 4 and all(o['covered'] and o['h_m'] <= 0.8 for o in tubs)),
        'garage_door_home': dict(got=[dict(id=o['id'], scope=o['scope'], covered_fraction=o['covered_fraction'],
                                           above_m=o['above_m'], sam3_agree=o.get('sam3_agree_fraction'),
                                           centroid=o['centroid']) for o in gds],
                                 ok=len(gds) >= 1 and all(o['scope'].startswith('home') for o in gds)),
        'thermostat_wall_or_rejected': dict(got=len(ths), heights=[o['h_m'] for o in ths],
                                            ok=all(1.0 <= o['h_m'] <= 1.9 for o in ths)),
        'smoke_detector_ceiling_or_rejected': dict(got=len(sds), above_m=stats([o['above_m'] for o in sds]),
                                                   pairs_closer_than_1m=int(near),
                                                   ok=all(o['above_m'] is not None and o['above_m'] <= 0.35 for o in sds)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--objects', type=Path, default=Path('experiments/w34_perception/objects_hf'))
    ap.add_argument('--result', type=Path, default=Path('data/w34/hf-perception-v1'))
    ap.add_argument('--captures', type=Path, default=Path('data/splat/web/HFwcH424n9w/captures.json'))
    ap.add_argument('--faces', type=Path, default=Path('data/splat/web/HFwcH424n9w/captures'))
    ap.add_argument('--mesh', type=Path, default=Path('experiments/w28_scene_reconstruction/matterport_export/assets'))
    ap.add_argument('--nav', type=Path, default=Path('experiments/w34_navigation/out_hf'))
    args = ap.parse_args()
    t0 = time.time()
    data = json.loads((args.objects / 'objects.json').read_text())
    objects = data['objects']
    dets = [json.loads(l) for l in (args.objects / 'detections.jsonl').open()]
    caps = {c['id']: c for c in json.loads(args.captures.read_text())['captures']}
    nav = json.loads((args.nav / 'nav.json').read_text())
    timing = {}
    made = dict(sheets=sheets(objects, args.faces, args.objects), maps=maps(objects, caps, nav, args.nav, args.objects))
    timing['sheets_maps'] = round(time.time() - t0, 1)
    v, f = navgrid.load_obj(args.mesh)
    caster = Caster(v, f)
    p = navgrid.Params()
    occ, _, lo, _ = navgrid.voxelize(v, f, p)
    checks = {}
    for name, fn in (('a_floor_exact', lambda: check_floor(caster, caps, nav)),
                     ('a_depth_exact_vs_voxel', lambda: check_depth(caster, caps, dets, occ, lo, p.res)),
                     ('b_door_two_view', lambda: check_doors(caster, caps, args.result)),
                     ('c_reprojection', lambda: check_reprojection(objects, caps, {d['id']: d for d in dets}, data['meta']['rules']['class_rules'])),
                     ('d_outcomes', lambda: check_outcomes(objects))):
        t = time.time()
        checks[name] = fn()
        timing[name] = round(time.time() - t, 1)
        print(name, json.dumps({k: v for k, v in checks[name].items() if k not in ('worst', 'misses')}, default=str)[:900], flush=True)
    timing['total'] = round(time.time() - t0, 1)
    out = dict(localize_version=data['meta']['version'], mesh_triangles=int(len(f)), made=made, timing_s=timing, checks=checks)
    (args.objects / 'acceptance.json').write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    print('timing', timing)


if __name__ == '__main__':
    main()
