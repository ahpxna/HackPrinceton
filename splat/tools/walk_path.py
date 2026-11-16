"""Collision-checked capture graph with doorway-crossing detours, metres Z-up."""
import heapq
import json
import numpy as np


def segment_clear(a, b, vertices, triangles, radius=.12):
    delta = b - a
    length = np.linalg.norm(delta)
    if length < 1e-6:
        return True
    direction = delta / length
    side = np.cross(direction, [0., 0., 1.])
    side /= max(np.linalg.norm(side), 1e-9)
    tri = vertices[triangles]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    # Floor/ceiling-like triangles (fused depth noise near the floor) cannot block a level walk.
    normal = np.cross(e1, e2)
    upright = abs(normal[:, 2]) <= .7 * np.maximum(np.linalg.norm(normal, axis=1), 1e-12)
    tri, e1, e2 = tri[upright], e1[upright], e2[upright]
    if not len(tri):
        return True
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum('ij,ij->i', e1, h)
    ok = abs(det) > 1e-9
    inv = np.divide(1., det, out=np.zeros_like(det), where=ok)
    for z in [0., -.6, -1.1]:
        for lateral in [-radius, 0., radius]:
            s = a + [0., 0., z] + lateral * side - tri[:, 0]
            u = inv * np.einsum('ij,ij->i', s, h)
            q = np.cross(s, e1)
            v = inv * (q @ direction)
            t = inv * np.einsum('ij,ij->i', e2, q)
            if np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > .015) & (t < length + radius)):
                return False
    return True


def make_path(home, panos, vertices, triangles):
    capture = np.array([[p['position'][k] for k in 'xyz'] for p in panos])
    nodes = list(capture.copy())
    graph = {i: [] for i in range(len(nodes))}
    distances = np.linalg.norm(capture[:, None] - capture[None], axis=-1)
    pairs = sorted({tuple(sorted((i, int(j)))) for i in range(len(nodes))
                    for j in np.argsort(distances[i])[1:7] if distances[i, j] < 4})
    def link(i, j):
        graph.setdefault(i, []).append(j); graph.setdefault(j, []).append(i)
    for i, j in pairs:
        if abs(capture[i, 2] - capture[j, 2]) < .7 and segment_clear(capture[i], capture[j], vertices, triangles):
            link(i, j)
    # Room graph uses model-frame XY. Door relativeCenter is along the edge;
    # capture height is used rather than assuming floor elevations absent in some exports.
    rooms = json.loads((home / 'rooms.json').read_text())
    doors = []
    for floor in rooms.get('floors', []):
        for edge in floor.get('edges', []):
            if len(edge.get('vertices', [])) != 2:
                continue
            a, b = [np.array([v['position'][k] for k in 'xy']) for v in edge['vertices']]
            d = b - a
            if np.linalg.norm(d) < .01:
                continue
            normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            for o in edge.get('openings', []):
                if o.get('type') not in ('door', 'doorway') or o.get('width', 0) < .5:
                    continue
                centre = a + float(o.get('relativeCenter', .5)) * d
                nearest = np.argsort(np.linalg.norm(capture[:, :2] - centre, axis=1))[:6]
                height = float(np.median(capture[nearest[:2], 2]))
                mid = np.r_[centre, height]
                before, after = mid + np.r_[normal * .45, 0], mid - np.r_[normal * .45, 0]
                if not segment_clear(before, after, vertices, triangles):
                    continue
                attachments = []
                for point in (before, after):
                    choices = [int(i) for i in nearest if np.linalg.norm(capture[i] - point) < 3
                               and segment_clear(capture[i], point, vertices, triangles)]
                    attachments.append(choices[:2])
                if not all(attachments):
                    continue
                ids = list(range(len(nodes), len(nodes) + 3))
                nodes.extend([before, mid, after])
                link(ids[0], ids[1]); link(ids[1], ids[2])
                for end, matches in zip([ids[0], ids[2]], attachments):
                    for i in matches:
                        link(end, i)
                doors.append({'id': o['id'], 'nodes': ids, 'centre': mid.tolist()})
    nodes = np.array(nodes)
    def shortest(start, target):
        todo = [(0., start, [start])]; visited = set()
        while todo:
            cost, node, route = heapq.heappop(todo)
            if node == target:
                return route
            if node in visited:
                continue
            visited.add(node)
            for nxt in graph[node]:
                if nxt not in visited:
                    heapq.heappush(todo, (cost + np.linalg.norm(nodes[node] - nodes[nxt]), nxt, route + [nxt]))
        return None
    route = [0]; used = []
    for door in doors:
        approach = shortest(route[-1], door['nodes'][0])
        if approach:
            route += approach[1:] + door['nodes'][1:]
            used.append(door)
        if len(used) == 2:
            break
    if len(route) == 1:
        # Smoke may have just two captures. A clear connection still tests interpolation.
        reachable = [shortest(0, i) for i in range(1, len(capture))]
        reachable = [r for r in reachable if r]
        if reachable:
            route = max(reachable, key=len)
        else:
            # Small verified local motion instead of passing through a known wall: any capture, 8 directions.
            for step in (.3, .15):
                for start in capture:
                    for angle in np.arange(8) * np.pi / 4:
                        end = start + [step * np.cos(angle), step * np.sin(angle), 0.]
                        if segment_clear(start, end, vertices, triangles):
                            points = np.array([start, end])
                            return points, {'controls': points.tolist(), 'doorways_crossed': [],
                                'requirements_met': False, 'reason': 'No connected capture route: short local diagnostic only'}
            # Do not block the web export on a diagnostic video: label it unverified instead.
            points = np.array([capture[0], capture[0] + [.15, 0., 0.]])
            return points, {'controls': points.tolist(), 'doorways_crossed': [], 'requirements_met': False,
                            'collision_verified': False,
                            'reason': 'Collision mesh blocked every local segment; 15 cm unverified diagnostic only'}
    # Explicit capture midpoints make interpolation failures visible.
    controls = [nodes[route[0]]]
    for i, j in zip(route[:-1], route[1:]):
        controls.extend([(nodes[i] + nodes[j]) / 2, nodes[j]])
    points = np.array(controls)
    return points, {'controls': points.tolist(), 'doorways_crossed': [d['id'] for d in used],
                    'requirements_met': len(used) >= 2,
                    'collision_method': '9 swept rays: 12cm lateral radius, eye/torso/knee, upright triangles only',
                    'limitations': 'Missing mesh surfaces cannot collide. Path checks do not establish captured RGB coverage.'}
