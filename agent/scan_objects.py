"""Publish confirmed W34 scan objects into the home graph as "located in 3D scan" features.

Input: experiments/w34_perception/objects_hf/objects.json, produced by localize.py. Each object is a
cluster of OWLv2 boxes whose rays were cast onto the Matterport mesh and merged across captures.
Only status=confirmed objects with a home* scope are published. The room link uses the object's own
3D point, unlike panorama features, where the room is only the camera location.
Run: .venv-agent/bin/python -m agent.scan_objects
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import defaultdict
from statistics import median
from agent.tools import ROOT, store

GRAPH = ROOT/'data/claims/graph'
OUT = GRAPH/'scan_objects'
SOURCE_KIND = 'located in 3D scan'
SOURCES = {'HFwcH424n9w': 'experiments/w34_perception/objects_hf/objects.json'}
MODEL = 'google/owlv2-base-patch16-ensemble'
PULL_M = 0.3  # fallback when the hit sits on a boundary: probe the room from the side the camera saw.
# Fixed fixtures must sit in a matching room type; otherwise the polygon is likely incomplete and the room stays unknown.
ROOM_TYPES = {'bathtub': ('bath',), 'toilet': ('bath',), 'shower': ('bath',), 'sink': ('bath', 'kitchen', 'laundry'),
              'stove': ('kitchen',), 'oven': ('kitchen',), 'microwave': ('kitchen',), 'kitchen island': ('kitchen',),
              'range hood': ('kitchen',), 'dishwasher': ('kitchen',), 'refrigerator': ('kitchen',), 'garage door': ('garage',),
              'washing machine': ('laundry',), 'clothes dryer': ('laundry',)}
# Claude's best-view review of this exact run. This is an AI review, not a human one.
# Each entry is keyed by version + id + centroid, so a rerun with new ids cannot misapply it.
REVIEW = {'w34-localize-7': [
    dict(id='kitchen_island_582', centroid=(0.07, -4.86, 0.85), action='exclude',
         reason='Best view shows the perimeter counter beside the stove, not an island (Claude best-view review, 2026-11-18).'),
]}
CAVEATS = [
    'OWLv2 zero-shot detections placed on the scan mesh and merged across captures; not human verified.',
    'object_count counts detector clusters; it is an upper bound, and one physical object can appear as several clusters.',
    'Room link uses the object point (or, on a boundary, the point pulled 0.3 m toward its best camera) in cached Matterport room polygons; open polygons and room-type conflicts leave the room unknown.',
    'covered_by_roof means a surface lies above the object; it does not mean indoors (garage doors, a gazebo).',
    'Missing objects do not establish absence. Class thresholds were tuned on this same home.',
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, separators=(',', ':'), ensure_ascii=False))
    tmp.replace(path)


def room_locator(sid):
    """Closed cached room polygons with a floor/ceiling interval; heights fall back to the floor's other rooms."""
    model, model_file = store()._audit_graph(sid)
    graph = json.loads((GRAPH/f'{sid}.json').read_text())
    rooms = {n['room_id']: n for n in graph['nodes'] if n['kind'] == 'room'}
    floor_of = {e['target'].split(':')[-1]: e['source'].split(':')[-1] for e in graph['edges'] if e['kind'] == 'floor-room'}
    sunfile = ROOT/f'data/sun/{sid}/sun.json'
    heights = {r['id']: r for r in (json.loads(sunfile.read_text()) if sunfile.exists() else {}).get('rooms', [])}
    by_floor = defaultdict(list)
    for rid, h in heights.items():
        if rid in floor_of and h.get('floor_z_m') is not None and h.get('ceiling_z_m') is not None:
            by_floor[floor_of[rid]].append((h['floor_z_m'], h['ceiling_z_m']))
    floor_z = {f: (median(v[0] for v in vals), median(v[1] for v in vals)) for f, vals in by_floor.items()}
    edges = {e['id']: e for f in (model or {}).get('floors', []) for e in f.get('edges', []) if e.get('id')}
    polygons = []
    for r in (model or {}).get('rooms', []):
        if r['id'] not in rooms:
            continue
        segments = []
        for edge in (r.get('boundary') or {}).get('edges', []):
            vertices = edges.get(edge.get('id'), edge).get('vertices', [])
            if len(vertices) == 2:
                segments.append(tuple((v['position']['x'], v['position']['y']) for v in vertices))
        ends = defaultdict(int)
        for seg in segments:
            for x, y in seg:
                ends[(round(x, 2), round(y, 2))] += 1
        if len(segments) < 3 or any(n != 2 for n in ends.values()):
            continue  # open boundary: never guess the nearest room
        h = heights.get(r['id'], {})
        if h.get('floor_z_m') is not None and h.get('ceiling_z_m') is not None:
            lo, hi, how = h['floor_z_m'], h['ceiling_z_m'], 'room floor/ceiling'
        elif floor_of.get(r['id']) in floor_z:
            (lo, hi), how = floor_z[floor_of[r['id']]], "floor/ceiling assumed from the same floor's rooms"
        else:
            continue
        polygons.append((r['id'], segments, lo, hi, how))

    def inside(x, y, segments):
        hit = False
        for (ax, ay), (bx, by) in segments:
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                hit = not hit
        return hit

    def locate(p, floor_level, what='object point'):
        found = [(rid, how) for rid, segs, lo, hi, how in polygons if lo - 0.35 < p[2] < hi + 0.15 and inside(p[0], p[1], segs)]
        if len(found) == 1:
            rid, how = found[0]
            return dict(room_id=rid, room=rooms[rid]['type'], floor_id=floor_of.get(rid),
                        room_mapping=f'{what} inside closed cached XY room boundary; {how} interval')
        floor = min(floor_z, key=lambda f: abs(floor_z[f][0] - floor_level)) if floor_z else None
        return dict(room_id=None, room='room unknown', floor_id=floor,
                    room_mapping='outside or ambiguous among closed cached room polygons; floor chosen by the nearest floor height')
    return locate, model_file


def accepted(sid):
    """Confirmed home objects grouped per (label, room); review exclusions recorded, not silently dropped."""
    source = ROOT/SOURCES[sid]
    doc = json.loads(source.read_text())
    meta = doc['meta']
    caps = json.loads((ROOT/f'data/splat/web/{sid}/captures.json').read_text())
    caps = {c['id']: c['position'] for c in (caps if isinstance(caps, list) else caps['captures'])}
    locate, model_file = room_locator(sid)
    reviews = REVIEW.get(meta['version'], [])
    excluded, groups = [], defaultdict(list)
    for o in doc['objects']:
        if o['status'] != 'confirmed' or not o['scope'].startswith('home'):
            continue
        review = next((r for r in reviews if r['id'] == o['id'] and math.dist(r['centroid'], o['centroid']) < 0.05), None)
        if review:
            excluded.append(dict(id=o['id'], label=o['label'], reason=review['reason']))
            continue
        c, cam = o['centroid'], caps[o['best_view']['capture']]
        place = locate(c, c[2] - o['h_m'], 'object point')
        if place['room_id'] is None:
            d = math.dist(c[:2], cam[:2])
            t = min(PULL_M, d) / d if d > 1e-6 else 0
            place = locate([c[0] + t * (cam[0] - c[0]), c[1] + t * (cam[1] - c[1]), c[2]], c[2] - o['h_m'],
                           f'object point pulled {PULL_M} m toward its best camera')
        wanted = ROOM_TYPES.get(o['label'])
        if place['room_id'] and wanted and not any(w in place['room'].casefold() for w in wanted):
            place = dict(place, room_id=None, polygon_room=place['room'], room='room unknown',
                         room_mapping=f"polygon room type ({place['room']}) conflicts with the object class; cached room polygons may be incomplete")
        groups[(o['label'], place['room_id'] or 'floor:' + str(place['floor_id']))].append((o, place))
    notes = meta['rules']['class_rules']
    features = []
    for (label, key), members in sorted(groups.items()):
        best, place = max(members, key=lambda m: m[0]['score_max'])
        objects = [dict(id=o['id'], position_m=o['centroid'], height_above_floor_m=o['h_m'], size_m=o['size_m'],
                        support_captures=o['support'], n_captures=o['n_captures'], score_max=o['score_max'],
                        covered_by_roof=o['covered'], best_view=o['best_view'],
                        image_file=f"data/splat/web/{sid}/captures/{o['best_view']['capture']}/face{o['best_view']['face']}.jpg")
                   for o, _ in sorted(members, key=lambda m: -m[0]['score_max'])]
        ident = hashlib.sha256(f'{sid}:{label}:{key}'.encode()).hexdigest()[:20]
        features.append(dict(
            id=f'feature:{sid}:scan:{ident}', kind='feature', text=label, label=label, source_kind=SOURCE_KIND,
            source=f"located in 3D scan: {len(objects)} cluster(s); best view {best['best_view']['capture']} face {best['best_view']['face']}",
            source_file=SOURCES[sid], home=sid, **place,
            confidence=best['score_max'], confidence_kind='OWLv2 maximum detection score over views; not a probability',
            model=MODEL, object_count=len(objects),
            count_kind='detector clusters after multi-view merging; upper bound, duplicates possible',
            objects=objects, image_file=objects[0]['image_file'], human_reviewed=False, spatially_verified=False,
            spatially_located=True, location_method=meta['raycast'],
            review_note=(notes.get(label) or {}).get('note')))
    inputs = {SOURCES[sid]: sha256(source), f'data/splat/web/{sid}/captures.json': sha256(ROOT/f'data/splat/web/{sid}/captures.json')}
    if model_file:
        inputs[model_file] = sha256(ROOT/model_file)
    return dict(sid=sid, source_version=meta['version'], inputs=inputs, features=features, excluded=excluded, caveats=CAVEATS)


def publish():
    from agent.home_graph import attach_scan_objects, graph_search
    summary = []
    for sid in sorted(SOURCES):
        doc = accepted(sid)
        save(OUT/f'accepted_{sid}.json', doc)
        graphpath = GRAPH/f'{sid}.json'
        save(graphpath, attach_scan_objects(json.loads(graphpath.read_text())))
        summary.append(dict(sid=sid, features=len(doc['features']), objects=sum(f['object_count'] for f in doc['features']),
                            in_room=sum(1 for f in doc['features'] if f['room_id']), excluded=len(doc['excluded'])))
    indexpath = GRAPH/'index.json'
    index = json.loads(indexpath.read_text())
    counts = defaultdict(int)
    for file in index['files']:
        for node in json.loads((GRAPH/file).read_text())['nodes']:
            counts[node['kind']] += 1
    index['counts'] = dict(counts)
    index['bytes'] = sum((GRAPH/f).stat().st_size for f in index['files'])
    index['scan_objects'] = dict(homes=summary, source_kind=SOURCE_KIND, caveats=CAVEATS)
    indexpath.write_text(json.dumps(index, indent=2))
    queries = [dict(feature='bathtub', source_kind=SOURCE_KIND, limit=5),
               dict(room='bedroom', feature='ceiling fan', source_kind=SOURCE_KIND, limit=5),
               dict(feature='garage door', source_kind=SOURCE_KIND, limit=5)]
    proof = [dict(query=q, matches=(r := graph_search(q))['matches'],
                  rooms=[(x['sid'], x['room'].get('type'), [f['text'] for f in x['features']]) for x in r['results']]) for q in queries]
    save(ROOT/'data/claims/eval/scan_objects_proof.json', dict(summary=summary, queries=proof))
    print(json.dumps(dict(summary=summary, queries=[(p['query'].get('room'), p['query']['feature'], p['matches']) for p in proof])))


if __name__ == '__main__':
    publish()
