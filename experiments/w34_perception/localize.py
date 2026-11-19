#!/usr/bin/env python3
"""Place W34 OWLv2 detections in 3D on the HF mesh, merge views into objects, label scope and status.

Per detection: rays through the box hit the provider mesh exactly (ray-triangle),
giving depth, a 3D point, metric size, storey, height above floor, whether it is under a roof, and its
distance outside the area the cameras covered. SAM 3 masks flag reflections (mirror) and cross-check
classes both models know. Detections of one label are then merged across captures into objects, and
each object is checked against documented plausibility priors (CLASS_RULES), which are assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'w34_navigation'))
import navgrid  # noqa: E402

VERSION = 'w34-localize-7'
STEP, T0, TMAX = 0.02, 0.15, 40.0          # ray marching, metres
GRID = 5                                   # GRID x GRID rays over the central 60% of each box
MIN_HIT_FRACTION = 0.3
HULL_HOME, HULL_FAR = 2.5, 5.0             # metres outside the camera hull: home / seen from home
MIRROR_BIT, MASK_SIZE = 3, 2048
SAM3_PAIRS = {'television': 'television', 'garage door': 'door'}
SAM3_MIN_AGREE = 0.3                       # paired classes: share of views where SAM 3 sees the same thing
CANDIDATE_MIN = 0.2                        # below this best score an object is noise, not a candidate
OUTDOOR_MAX_COVERED = 0.25                 # outdoor objects: share of views allowed to have a surface above
ZSCALE = np.array([1.0, 1.0, 0.5])         # clustering metric: vertical gaps count half (tile views see parts of tall things)
SUPPORT_MIN = 0.3                          # a view supports an object (3D consistency) at this score
# `confirmed` needs support from 2 captures plus one view reaching the class `thr` (the class evidence).

# Plausibility priors, not facts. where: indoor (under a roof), outdoor (open sky above), any.
# h: object-centre height above its storey floor (m). size: larger box side in metres.
# ceiling: surface straight above within this many metres. thr: a view at or above it supports the object.
# confirmable=False: Claude's look at the best views (2026-11-18) found OWLv2 confusing the class
# systematically, and agreement across views cannot fix a confusion every view shares, so such objects
# stay candidates. note: what was reviewed; classes without a note were not checked by eye (thr 0.3 default).
# The thresholds were tuned on this same home, so they are a calibration for it, not a validated setting.
R = lambda where, h, size, thr=0.3, ceiling=None, ground_storey=False, confirmable=True, note=None: dict(  # noqa: E731
    where=where, h=h, size=size, thr=thr, ceiling=ceiling, ground_storey=ground_storey,
    confirmable=confirmable, note=note)
CLASS_RULES = {
    'refrigerator': R('indoor', (0.4, 1.4), (0.55, 2.2), confirmable=False,
                      note='4 confirmed (0.41-0.56) were walls and cabinets; the house is vacant'),
    'dishwasher': R('indoor', (0.1, 0.8), (0.4, 0.9), confirmable=False,
                    note='5 confirmed: 1 dishwasher (0.76), 3 bathroom vanity cabinets (one 0.77), 1 floor'),
    'oven': R('indoor', (0.1, 1.6), (0.45, 1.1), confirmable=False,
              note='3 confirmed (0.49-0.65): a microwave, a door, 1 unclear'),
    'stove': R('indoor', (0.4, 1.1), (0.45, 1.2), thr=0.6,
               note='0.61 the kitchen range; 0.59 the patio grill, 0.39 a cabinet'),
    'microwave': R('indoor', (0.3, 2.0), (0.3, 0.9), thr=0.55,
                   note='0.91 and 0.55 the over-range microwave; 0.39-0.42 wrong'),
    'range hood': R('indoor', (1.3, 2.5), (0.45, 1.4), confirmable=False,
                    note='kitchen has an over-range microwave; 2 confirmed were a hallway light and a small box'),
    'washing machine': R('indoor', (0.1, 1.3), (0.45, 1.2)),
    'clothes dryer': R('indoor', (0.1, 1.3), (0.45, 1.2)),
    'ceiling fan': R('indoor', (1.9, 7.0), (0.6, 1.9), ceiling=1.0, thr=0.7,
                     note='7 fans at 0.76-0.90 (2 under patio and gazebo roofs); 0.65 pendant light, 0.56 unclear'),
    'fireplace': R('indoor', (0.0, 1.6), (0.5, 2.6), note='1 confirmed (0.76): real living-room fireplace'),
    'air conditioner': R('any', (-0.2, 2.6), (0.35, 1.6), confirmable=False,
                         note='3 confirmed (0.36-0.45): window edge, ceiling register, wall vent'),
    'water heater': R('indoor', (0.2, 1.6), (0.35, 1.9)),
    'bathtub': R('indoor', (-0.1, 0.8), (0.9, 2.4), thr=0.7,
                 note='best views >= 0.75 were 3 real tubs; 0.66 a toilet, 0.40 and 0.37 bare floor'),
    'shower': R('indoor', (0.0, 2.3), (0.5, 2.6), confirmable=False,
                note='top 4: 2 shower heads, 1 marble floor, 1 ceiling'),
    'toilet': R('indoor', (0.0, 0.8), (0.3, 0.95), thr=0.85,
                note='0.86-0.91 real toilets; 0.76 and 0.66 tiny far boxes, 0.39 floor'),
    'sink': R('indoor', (0.5, 1.2), (0.25, 1.3), confirmable=False,
              note='top 4 (0.70-0.73): 2 sinks, 1 light switch, 1 unclear'),
    'kitchen island': R('indoor', (0.1, 1.1), (0.9, 4.5), thr=0.55,
                        note='0.56-0.60 the island (probably one island split in parts); 0.41 a vanity, 0.38 tiny'),
    'wine cooler': R('indoor', (0.0, 1.1), (0.3, 0.9)),
    'television': R('indoor', (0.4, 2.6), (0.45, 2.3)),
    'smoke detector': R('indoor', (1.9, 7.0), (0.05, 0.35), ceiling=0.35,
                        note='top 8 (0.73-0.92): small ceiling or high-wall discs, 1 clear detector, none clearly wrong'),
    'thermostat': R('indoor', (1.25, 1.75), (0.05, 0.3), thr=0.78,     # US mounting ~52-60 in
                    note='top 7 (0.66-0.80): light switches up to 0.75; only 0.80 looked like a thermostat'),
    'chandelier': R('indoor', (1.4, 7.0), (0.25, 1.6), ceiling=3.0, thr=0.6,
                    note='0.62-0.74 pendants and a wagon-wheel chandelier; 0.39-0.45 bathroom ceiling and vanity lights'),
    'swimming pool': R('outdoor', (-1.5, 0.5), (2.0, 25.0), ground_storey=True),
    'hot tub': R('outdoor', (-0.8, 1.2), (1.4, 3.6), ground_storey=True),
    'solar panel': R('outdoor', (2.2, 12.0), (0.8, 25.0), ground_storey=True),
    'garage door': R('any', (0.2, 2.5), (1.8, 6.0), ground_storey=True,
                     note='top 3: 2 street views of this home\'s garage (x~7.6, SAM 3 door on 5/9 and 6/6 views), '
                          '1 tiny box 18-23 m away (SAM 3 door on 0/9 views, unclear); needs SAM3_MIN_AGREE'),
    'bed': R('indoor', (0.0, 1.0), (0.9, 2.6), confirmable=False,
             note='house is vacant; the 1 confirmed bed (0.34) was a stair railing seen from above'),
    'sofa': R('indoor', (0.0, 1.0), (0.7, 3.6)),
}


def sha(path: Path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rays(cap, face, xy):
    """World directions through normalised face pixels (x right, y down), as in refine_atlas_v3.sample."""
    c, r, d = np.asarray(cap['face_bases_center_right_down'])[face]
    u, v = 2 * xy[:, 0] - 1, 2 * xy[:, 1] - 1
    out = c[None] + u[:, None] * r[None] + v[:, None] * d[None]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def cast(origins, dirs, occ, lo, res, t0=T0, tmax=TMAX, batch=40_000, chunk=50):
    """First occupied voxel along each ray (distance in m, NaN if none before leaving the grid or tmax)."""
    hit = np.full(len(origins), np.nan)
    shape = np.array(occ.shape[::-1])
    for b0 in range(0, len(origins), batch):
        o, dr = origins[b0:b0 + batch], dirs[b0:b0 + batch]
        active, t = np.arange(len(o)), t0
        while active.size and t < tmax:
            ts = t + STEP * np.arange(chunk)
            pts = o[active, None] + dr[active, None] * ts[None, :, None]
            ijk = np.floor((pts - lo) / res).astype(np.int64)
            inside = ((ijk >= 0) & (ijk < shape)).all(-1)
            occv = np.zeros(inside.shape, bool)
            sel = ijk[inside]
            occv[inside] = occ[sel[:, 2], sel[:, 1], sel[:, 0]] > 0
            has = occv.any(1)
            hit[b0 + active[has]] = ts[occv[has].argmax(1)]
            active = active[~has & inside[:, -1]]                     # a ray that left the box never returns
            t += STEP * chunk
    return hit


class Caster:
    """Exact first-hit distances. Triangles are culled per ray bundle by bounding sphere vs. cone."""

    def __init__(self, v, f):
        self.v0 = v[f[:, 0]]
        self.e1, self.e2 = v[f[:, 1]] - self.v0, v[f[:, 2]] - self.v0
        self.c = v[f].mean(1)
        self.r = np.linalg.norm(v[f] - self.c[:, None], axis=-1).max(1)
        self._o = None

    def _origin(self, o):
        if self._o is None or not np.array_equal(self._o, o):
            vec = self.c - o
            self.dist = np.linalg.norm(vec, axis=1)
            self.unit = vec / np.maximum(self.dist, 1e-9)[:, None]
            self.ang_r = np.arcsin(np.clip(self.r / np.maximum(self.dist, 1e-9), 0, 1))
            self._o = o.copy()

    def cast(self, o, D, t_min=T0, t_max=TMAX):
        o = np.asarray(o, float)
        self._origin(o)
        axis = D.mean(0)
        axis /= np.linalg.norm(axis)
        half = float(np.arccos(np.clip(D @ axis, -1, 1)).max())
        ang = np.arccos(np.clip(self.unit @ axis, -1, 1))
        keep = np.flatnonzero(((ang - self.ang_r) <= half + 1e-3) | (self.dist <= self.r))
        keep = keep[self.dist[keep] - self.r[keep] <= t_max]
        out = np.full(len(D), np.nan)
        for s in range(0, len(keep), 200_000):
            k = keep[s:s + 200_000]
            v0, e1, e2 = self.v0[k], self.e1[k], self.e2[k]
            pvec = np.cross(D[:, None], e2[None])
            det = (e1[None] * pvec).sum(-1)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
            tvec = o - v0
            u = (tvec[None] * pvec).sum(-1) * inv
            qvec = np.cross(tvec, e1)
            w = (D @ qvec.T) * inv
            t = (e2 * qvec).sum(-1)[None] * inv
            hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > t_min) & (t < t_max)
            t = np.where(hit, t, np.inf).min(1)
            out = np.fmin(out, np.where(np.isfinite(t), t, np.nan))
        return out


def hull_distance(points_xy, hull_xy):
    """Horizontal distance outside the polygon (0 inside)."""
    a, b = hull_xy, np.roll(hull_xy, -1, 0)
    ab = b - a
    t = np.clip(((points_xy[:, None] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0, 1)
    dist = np.linalg.norm(points_xy[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)
    cross = ab[None, :, 0] * (points_xy[:, None, 1] - a[None, :, 1]) - ab[None, :, 1] * (points_xy[:, None, 0] - a[None, :, 0])
    inside = (cross >= 0).all(1) | (cross <= 0).all(1)
    return np.where(inside, 0.0, dist)


def storey_of(z, cam_storey, floors):
    """The camera's storey if the point lies in its height range, else the highest floor below the point."""
    lo_ok = z >= floors[cam_storey] - 0.4
    hi_ok = z <= (floors[cam_storey + 1] - 0.4 if cam_storey + 1 < len(floors) else np.inf)
    if lo_ok and hi_ok:
        return cam_storey
    below = [i for i, f in enumerate(floors) if z >= f - 0.4]
    return below[-1] if below else 0


def hard_flags(flags):
    """Flags that reject an object; the rest only keep it a candidate."""
    return [x for x in flags if not x.startswith(('support ', 'scope unknown', 'class not confirmable', 'sam3 weak'))]


def check_rule(rule, obj):
    """Reasons an object breaks its class priors (empty list = plausible). Size uses the 80th percentile of
    the member views, because a box only shows the extent across the line of sight (a tub seen end-on)."""
    if rule is None:
        return ['no plausibility rule for this label']
    bad = []
    if rule['where'] == 'indoor' and obj['covered_fraction'] < 0.5:
        bad.append(f'expected indoors, open sky above in {1 - obj["covered_fraction"]:.0%} of views')
    if rule['where'] == 'outdoor' and obj['covered_fraction'] > OUTDOOR_MAX_COVERED:
        bad.append(f'expected outdoors with open sky above, roof above in {obj["covered_fraction"]:.0%} of views')
    if not rule['h'][0] <= obj['h_m'] <= rule['h'][1]:
        bad.append(f'height {obj["h_m"]:.2f} m outside {rule["h"]}')
    if not rule['size'][0] <= obj['size_m'] <= rule['size'][1]:
        bad.append(f'size {obj["size_m"]:.2f} m outside {rule["size"]}')
    if rule['ceiling'] is not None and not (obj['above_m'] is not None and obj['above_m'] <= rule['ceiling']):
        bad.append(f'not on/near a ceiling (above {obj["above_m"]} m)')
    if rule['ground_storey'] and obj['storey'] != 0:
        bad.append('expected on the ground storey')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--result', type=Path, default=Path('data/w34/hf-perception-v1'))
    ap.add_argument('--captures', type=Path, default=Path('data/splat/web/HFwcH424n9w/captures.json'))
    ap.add_argument('--mesh', type=Path, default=Path('experiments/w28_scene_reconstruction/matterport_export/assets'))
    ap.add_argument('--nav', type=Path, default=Path('experiments/w34_navigation/out_hf/nav.json'))
    ap.add_argument('--out', type=Path, default=Path('experiments/w34_perception/objects_hf'))
    args = ap.parse_args()
    t_start = time.time()
    caps = {c['id']: c for c in json.loads(args.captures.read_text())['captures']}
    nav = json.loads(args.nav.read_text())
    floors = nav['storeys_floor_z']
    cam_floor = {r['id']: r['floor_z'] for r in nav['captures']}
    cam_storey = {k: int(np.argmin([abs(v - f) for f in floors])) for k, v in cam_floor.items()}

    p = navgrid.Params()
    v, f = navgrid.load_obj(args.mesh)
    occ, _, lo, _ = navgrid.voxelize(v, f, p)
    exact = Caster(v, f)
    print(f'voxels {occ.shape} in {time.time() - t_start:.1f}s', flush=True)

    hull_pts = np.array([c['position'][:2] for c in caps.values()])
    hull_xy = hull_pts[ConvexHull(hull_pts).vertices]

    # Geometry check: the down-face centre ray should meet the floor at camera height.
    geo = []
    for cid, cap in caps.items():
        down = int(np.argmin(np.asarray(cap['face_bases_center_right_down'])[:, 0, 2]))
        d = rays(cap, down, np.array([[0.5, 0.5]]))
        t = exact.cast(np.asarray(cap['position']), d)[0]
        geo.append(dict(capture=cid, camera_height=round(cap['position'][2] - cam_floor[cid], 3),
                        down_hit=None if np.isnan(t) else round(float(t), 3)))
    err = np.array([abs(g['down_hit'] - g['camera_height']) for g in geo if g['down_hit'] is not None])
    print(f'down-ray floor check: {len(err)}/{len(geo)} hit, |error| median {np.median(err):.3f} m, '
          f'p90 {np.quantile(err, .9):.3f} m, max {err.max():.3f} m', flush=True)

    # SAM 3 instances per face and home-wide prompt counts.
    sam3, sam3_total = {}, defaultdict(int)
    for jf in (args.result / 'sam3').glob('*/face*.json'):
        d = json.loads(jf.read_text())
        sam3[(d['capture'], d['face'])] = d['instances']
        for i in d['instances']:
            sam3_total[i['prompt']] += 1

    # Detections -> rays.
    dets, origins, dirs, owner = [], [], [], []
    g = (0.2 + 0.6 * np.arange(GRID) / (GRID - 1))
    gx, gy = np.meshgrid(g, g)
    inner = np.c_[gx.ravel(), gy.ravel()]
    for jf in sorted((args.result / 'owl').glob('*/face*.json')):
        d = json.loads(jf.read_text())
        cap, face = caps[d['capture']], d['face']
        for k, det in enumerate(d['detections']):
            x0, y0, x1, y1 = det['box']
            pts = np.r_[np.c_[x0 + inner[:, 0] * (x1 - x0), y0 + inner[:, 1] * (y1 - y0)],
                        [[(x0 + x1) / 2, (y0 + y1) / 2]]]
            dets.append(dict(id=f'{d["capture"]}:{face}:{k}', capture=d['capture'], face=face, label=det['label'],
                             score=det['score'], box=det['box'], view=det['view'], samples=pts))
            dd = rays(cap, face, pts)
            origins.append(np.repeat(np.asarray(cap['position'])[None], len(pts), 0))
            dirs.append(dd)
            owner.append(np.full(len(pts), len(dets) - 1))
    origins, dirs, owner = np.concatenate(origins), np.concatenate(dirs), np.concatenate(owner)
    t0 = time.time()
    per = GRID * GRID + 1                                      # rays of one detection share an origin
    hits = np.concatenate([exact.cast(origins[i * per], dirs[i * per:(i + 1) * per]) for i in range(len(dets))])
    print(f'{len(dets)} detections, {len(hits)} rays cast in {time.time() - t0:.1f}s', flush=True)

    # Per-detection geometry.
    per = GRID * GRID + 1
    hits = hits.reshape(len(dets), per)
    dirs = dirs.reshape(len(dets), per, 3)
    points, ok = np.full((len(dets), 3), np.nan), np.zeros(len(dets), bool)
    for i, det in enumerate(dets):
        h = hits[i, :GRID * GRID]
        frac = float(np.isfinite(h).mean())
        det['hit_fraction'] = round(frac, 3)
        if frac < MIN_HIT_FRACTION:
            continue
        depth = float(np.nanquantile(h, 0.25))                       # nearest consistent surface
        cap = caps[det['capture']]
        centre = dirs[i, -1]
        points[i] = np.asarray(cap['position']) + centre * depth
        ok[i] = True
        c = np.asarray(cap['face_bases_center_right_down'])[det['face'], 0]
        zc = depth * float(centre @ c)                               # distance along the face axis
        x0, y0, x1, y1 = det['box']
        det.update(depth_m=round(depth, 3), point=[round(float(a), 3) for a in points[i]],
                   width_m=round(2 * (x1 - x0) * zc, 3), height_m=round(2 * (y1 - y0) * zc, 3))
        det['size_m'] = max(det['width_m'], det['height_m'])

    # Straight-up ray from just in front of each point: roof/ceiling above?
    idx = np.flatnonzero(ok)
    start = points[idx] - dirs[idx, -1] * 0.10
    above = cast(start, np.tile([0.0, 0.0, 1.0], (len(idx), 1)), occ, lo, p.res, t0=0.05, tmax=12.0)
    dist_out = hull_distance(points[idx, :2], hull_xy)
    mirror_png = {}
    for j, i in enumerate(idx):
        det = dets[i]
        det['above_m'] = None if np.isnan(above[j]) else round(float(above[j]), 3)
        det['covered'] = det['above_m'] is not None and det['above_m'] <= 8.0
        det['outside_hull_m'] = round(float(dist_out[j]), 2)
        s = storey_of(points[i, 2], cam_storey[det['capture']], floors)
        det['storey'], det['h_m'] = s, round(float(points[i, 2] - floors[s]), 3)

    for i, det in enumerate(dets):
        key = (det['capture'], det['face'])
        if any(x['prompt'] == 'mirror' for x in sam3.get(key, [])):
            if key not in mirror_png:
                mirror_png[key] = np.asarray(Image.open(args.result / 'sam3' / key[0] / f'face{key[1]}.png'))
            px = np.clip((det['samples'] * MASK_SIZE).astype(int), 0, MASK_SIZE - 1)
            det['in_mirror'] = bool(((mirror_png[key][px[:, 1], px[:, 0]] >> MIRROR_BIT) & 1).mean() > 0.5)
        else:
            det['in_mirror'] = False
        prompt = SAM3_PAIRS.get(det['label'])
        if prompt:
            x0, y0, x1, y1 = det['box']
            area = max((x1 - x0) * (y1 - y0), 1e-9)
            cover = [max(0, min(x1, b[2]) - max(x0, b[0])) * max(0, min(y1, b[3]) - max(y0, b[1])) / area
                     for b in (x['box'] for x in sam3.get(key, []) if x['prompt'] == prompt)]
            det['sam3_agrees'] = bool(cover and max(cover) >= 0.3)
        else:
            det['sam3_agrees'] = None
        if not ok[i]:
            det['scope'] = 'seen_from_home'                           # rays find no surface: beyond the meshed area
        elif det['outside_hull_m'] > HULL_FAR:
            det['scope'] = 'seen_from_home'
        elif det['outside_hull_m'] <= HULL_HOME:
            det['scope'] = 'home_indoor' if det['covered'] else 'home_outdoor'
        else:
            det['scope'] = 'unknown'
        del det['samples']

    # Merge views per label by leader clustering: the strongest unassigned detection claims every unassigned
    # detection within `radius` of it. Unlike single linkage it cannot chain one room's sightings into the next.
    objects = []
    for label in sorted({d['label'] for d in dets}):
        rule = CLASS_RULES.get(label)
        radius = float(np.clip(0.5 * (rule['size'][1] if rule else 1.0), 0.25, 1.2))
        ids = np.array([i for i, d in enumerate(dets) if d['label'] == label and ok[i] and not d['in_mirror']])
        if not ids.size:
            continue
        ids = ids[np.argsort([-dets[i]['score'] for i in ids], kind='stable')]
        tree, free = cKDTree(points[ids] * ZSCALE), np.ones(len(ids), bool)
        for k in range(len(ids)):
            if not free[k]:
                continue
            near = [j for j in tree.query_ball_point(points[ids[k]] * ZSCALE, radius) if free[j]]
            free[near] = False
            mem = [int(ids[j]) for j in near]
            rows = [dets[m] for m in mem]
            w = np.array([r['score'] for r in rows])
            best = rows[int(w.argmax())]
            centroid = np.median(points[mem], 0)
            scopes = defaultdict(float)
            for r in rows:
                scopes[r['scope']] += r['score']
            thr = rule['thr'] if rule else 1.0
            sizes = [r['size_m'] for r in rows]
            obj = dict(id=f'{label.replace(" ", "_")}_{len(objects)}', label=label,
                       centroid=[round(float(a), 3) for a in centroid],
                       storey=int(np.bincount([r['storey'] for r in rows]).argmax()),
                       h_m=round(float(np.median([r['h_m'] for r in rows])), 3),
                       size_m=round(float(np.quantile(sizes, 0.8)), 3), size_median_m=round(float(np.median(sizes)), 3),
                       width_m=round(float(np.median([r['width_m'] for r in rows])), 3),
                       height_m=round(float(np.median([r['height_m'] for r in rows])), 3),
                       above_m=None, covered_fraction=round(float(np.mean([r['covered'] for r in rows])), 3),
                       scope=max(scopes, key=scopes.get), n_captures=len({r['capture'] for r in rows}),
                       support=len({r['capture'] for r in rows if r['score'] >= SUPPORT_MIN}),
                       n_detections=len(rows), score_max=round(float(w.max()), 4), score_mean=round(float(w.mean()), 4),
                       best_view=dict(capture=best['capture'], face=best['face'], box=best['box'], score=best['score']),
                       members=[r['id'] for r in rows])
            obj['covered'] = obj['covered_fraction'] >= 0.5
            ab = [r['above_m'] for r in rows if r['above_m'] is not None]
            obj['above_m'] = round(float(np.median(ab)), 3) if ab else None
            flags = check_rule(rule, obj)
            if SAM3_PAIRS.get(label):
                agree = [r['sam3_agrees'] for r in rows]
                obj['sam3_agree_fraction'] = round(float(np.mean(agree)), 3)
                if label != 'television' and obj['sam3_agree_fraction'] < SAM3_MIN_AGREE:
                    flags.append(f'sam3 weak: a SAM 3 {SAM3_PAIRS[label]} overlaps {obj["sam3_agree_fraction"]:.0%} of views')
                if label == 'television' and sam3_total['television'] == 0:
                    flags.append('contradicted_by_sam3: SAM 3 found no television anywhere in the home')
            if obj['score_max'] < CANDIDATE_MIN:
                flags.append(f'weak: score_max < {CANDIDATE_MIN}')
            if obj['support'] < 2:
                flags.append(f'support {obj["support"]} capture(s) with score >= {SUPPORT_MIN}')
            if obj['score_max'] < thr:
                flags.append(f'support weak: best score {obj["score_max"]:.2f} below class thr {thr}')
            if obj['scope'] == 'unknown':
                flags.append('scope unknown')
            if rule and not rule['confirmable']:
                flags.append('class not confirmable from OWLv2 alone (reviewed confusions)')
            if hard_flags(flags):
                obj['status'] = 'rejected'
            elif (obj['support'] >= 2 and obj['score_max'] >= thr and obj['scope'] != 'unknown' and rule['confirmable']
                  and not any(x.startswith('sam3 weak') for x in flags)):
                obj['status'] = 'confirmed'
            else:
                obj['status'] = 'candidate'
            obj['flags'] = flags
            for r in rows:
                r['object'] = obj['id']
            objects.append(obj)
    for i, det in enumerate(dets):
        if det['in_mirror']:
            det['object'] = None

    # Summary.
    summary = {}
    for label in sorted({d['label'] for d in dets}):
        ds = [d for d in dets if d['label'] == label]
        os_ = [o for o in objects if o['label'] == label]
        summary[label] = dict(
            detections_ge_0_1=len(ds), detections_ge_0_3=sum(d['score'] >= 0.3 for d in ds),
            in_mirror=sum(d['in_mirror'] for d in ds), no_surface=sum('point' not in d for d in ds), objects=len(os_),
            **{s: sum(o['status'] == s for o in os_) for s in ('confirmed', 'candidate', 'rejected')},
            confirmed_by_scope={sc: sum(o['status'] == 'confirmed' and o['scope'] == sc for o in os_)
                                for sc in ('home_indoor', 'home_outdoor', 'seen_from_home')})
    args.out.mkdir(parents=True, exist_ok=True)
    meta = dict(version=VERSION, seconds=round(time.time() - t_start, 1),
                inputs={str(args.captures): sha(args.captures), str(args.nav): sha(args.nav),
                        'owl_summary': sha(args.result / 'owl_summary.json'), 'sam3_summary': sha(args.result / 'sam3_summary.json')},
                raycast=f'exact ray-triangle hits on the provider mesh (Moller-Trumbore, cone-culled); depth = lower quartile of '
                        f'{GRID}x{GRID} central hits; the covered up-ray marches {STEP} m through navgrid 5 cm voxels',
                rules=dict(min_hit_fraction=MIN_HIT_FRACTION, hull_home_m=HULL_HOME, hull_far_m=HULL_FAR,
                           covered='surface straight above within 8 m', class_rules=CLASS_RULES,
                           confirm='plausible + support (captures with score >= SUPPORT_MIN) >= 2 + score_max >= class thr + scope known + not contradicted', support_min=SUPPORT_MIN, candidate_min=CANDIDATE_MIN, sam3_min_agree=SAM3_MIN_AGREE, outdoor_max_covered=OUTDOOR_MAX_COVERED, clustering='leader per label, radius clamp(0.5 * size_max, 0.25, 1.2) m, vertical distances halved'),
                geometry_check=dict(down_ray_hits=len(err), captures=len(geo), median_abs_error_m=round(float(np.median(err)), 3),
                                    p90_abs_error_m=round(float(np.quantile(err, .9)), 3), max_abs_error_m=round(float(err.max()), 3),
                                    rows=geo),
                sam3_home_counts=dict(sam3_total))
    (args.out / 'objects.json').write_text(json.dumps(dict(meta=meta, objects=objects), indent=1))
    with open(args.out / 'detections.jsonl', 'w') as fh:
        for d in dets:
            fh.write(json.dumps(d) + '\n')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(f'done in {time.time() - t_start:.1f}s')
    for label, s in summary.items():
        print(f'{label:16s} det>=.3 {s["detections_ge_0_3"]:4d} mirror {s["in_mirror"]:3d} objects {s["objects"]:3d} '
              f'confirmed {s["confirmed"]:2d} {s["confirmed_by_scope"]} candidate {s["candidate"]:3d} rejected {s["rejected"]:3d}')


if __name__ == '__main__':
    main()
