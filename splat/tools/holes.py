"""Capture masks and conservative, geometry-conditioned surface support.

Flat ring-colour patches are separate labelled display geometry, never training
data. No texture extrapolation, neural fill, or triangulation across depth jumps.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def detect_nadir(bgr):
    """Measure first sustained texture return along 180 rays from the nadir.

    The supplied captures have a blurred/patched central region. A radial
    texture boundary is a heuristic, not a semantic tripod detector. Fail closed
    on weak/oversized detections rather than admitting an unmeasured fixed disc.
    """
    native = bgr.shape[0]
    gray = cv2.cvtColor(cv2.resize(bgr, (512, 512)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    texture = cv2.boxFilter(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), -1, (9, 9))
    radius = np.arange(12, 241, dtype=np.float32)
    angle = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    xx = 255.5 + np.cos(angle)[:, None] * radius
    yy = 255.5 + np.sin(angle)[:, None] * radius
    profile = cv2.remap(texture, xx.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR)
    centre = float(np.median(profile[:, :8]))
    outside = float(np.median(profile[:, -30:]))
    threshold = max(.7, centre * 3, outside * .15)
    high = profile > threshold
    sustained = cv2.boxFilter(high.astype(np.float32), -1, (9, 1), normalize=True)
    crossings = sustained > .8
    has = crossings.any(1)
    boundaries = radius[np.argmax(crossings, axis=1)]
    good = has & (boundaries > 20)
    fraction = float(good.mean())
    measured = float(np.quantile(boundaries[good], .98)) if good.any() else 0.
    # The circumscribing disc excludes the rectangular/rounded blur's corners.
    # Missing radial boundaries expand to the outer measured search limit.
    if fraction < 1 or outside < max(.9, centre * 2) or not 25 < measured < 225:
        raise RuntimeError('Nadir boundary ambiguous: inspect bottom face; refusing an arbitrary radius. '
                           f'boundary_fraction={fraction:.3f}, centre={centre:.3f}, outer={outside:.3f}, radius512={measured:.1f}')
    measured = max(measured, float(boundaries[good].max()))
    r = measured * native / 512
    dilated = math.ceil(r + 15)
    y, x = np.ogrid[:native, :native]
    excluded = (x - (native - 1) / 2) ** 2 + (y - (native - 1) / 2) ** 2 <= dilated ** 2
    return excluded, {'measured_radius_px': r, 'dilation_px': 15, 'excluded_radius_px': dilated,
                      'centre_px': [(native - 1) / 2] * 2, 'boundary_ray_fraction': fraction,
                      'texture_centre': centre, 'texture_outer': outside,
                      'ray_radii_px': (boundaries * native / 512).round(2).tolist(),
                      'excluded_fraction_bottom': float(excluded.mean()),
                      'method': 'central low-texture patch; 180 radial sustained texture crossings; conservative circumscribed disc',
                      'confidence': 'heuristic; inspect saved overlay; not a semantic guarantee'}


def fallback_nadir(bgr, reason):
    """Ambiguous detection: mask the detector's whole radial search disc (conservative over-mask)."""
    native = bgr.shape[0]
    r = 240 * native / 512
    dilated = math.ceil(r + 15)
    y, x = np.ogrid[:native, :native]
    excluded = (x - (native - 1) / 2) ** 2 + (y - (native - 1) / 2) ** 2 <= dilated ** 2
    return excluded, {'measured_radius_px': None, 'fallback_radius_px': r, 'dilation_px': 15,
                      'excluded_radius_px': dilated, 'centre_px': [(native - 1) / 2] * 2,
                      'excluded_fraction_bottom': float(excluded.mean()),
                      'method': 'fallback: detector ambiguous; full radial search disc (240/512 of face) plus 15 px',
                      'detector_error': reason, 'confidence': 'conservative over-mask; inspect overlay'}


def masks(trial):
    folder = trial.work / 'nadir'; folder.mkdir(exist_ok=True)
    proof = trial.proof / 'nadir'; proof.mkdir(exist_ok=True)
    report = []; outputs = []
    for p in trial.panos:
        bgr = cv2.imread(str(trial.home / 'faces_2k' / p['id'] / 'face5.jpg'))
        if bgr is None:
            raise RuntimeError('Missing bottom capture: ' + p['id'])
        try:
            excluded, record = detect_nadir(bgr)
        except RuntimeError as exc:
            excluded, record = fallback_nadir(bgr, str(exc))
        name = folder / (p['id'] + '.png')
        cv2.imwrite(str(name), excluded.astype(np.uint8) * 255)
        overlay = bgr.copy(); overlay[excluded] = (overlay[excluded] * .4 + np.array([0, 0, 255]) * .6).astype(np.uint8)
        cv2.circle(overlay, (1024, 1024), record['excluded_radius_px'], (0, 255, 255), 4)
        view = proof / (p['id'] + '.jpg'); cv2.imwrite(str(view), cv2.resize(overlay, (768, 768)))
        report.append(dict(pano=p['id'], **record)); outputs.extend([name, view])
        print('Nadir mask', p['id'], 'measured px', record['measured_radius_px'], flush=True)
    path = trial.proof / 'nadir.json'
    write_json(path, {'panoramas': report, 'losses': 'L1, SSIM-window support, depth and mesh losses exclude mask',
                     'fusion': 'excluded before point sampling and fused triangulation',
                     'held_out': 'evaluation excludes nadir; visibility excludes nadir in every capture',
                     'limitations': 'Blur detector can over-mask natural smooth floor. Ambiguous detections stop; inspect overlays.'})
    return outputs + [path]


def valid_pixels(trial, pano, face, size):
    if face != 5:
        return np.ones((size, size), dtype=bool)
    mask = cv2.imread(str(trial.work / 'nadir' / (pano['id'] + '.png')), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError('Nadir stage missing')
    # Area resampling excludes ANY contaminated subpixel at training resolution.
    return cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) < .01


def surface_class(normal, centre, cameras):
    z = abs(normal[2])
    if z < .3:
        return 'wall'
    if z > .85:
        near = cameras[np.argmin(np.linalg.norm(cameras[:, :2] - centre[:2], axis=1))]
        return 'floor' if centre[2] < near[2] - .35 else 'ceiling' if centre[2] > near[2] + .35 else 'other'
    return 'other'


def fused_collision(trial):
    """Triangulate sampled, train-only metric depth without bridging silhouettes.

    A 48x48 grid bounds browser collision cost; source depth still native 2k.
    Faces whose edges or depth jump too far are discarded, not filled.
    """
    from run_splat import rays, xyz
    scale = json.loads((trial.proof / 'scale.json').read_text())['scale']
    n = 24 if trial.a.smoke else 48
    axis = np.linspace(1, 2046, n).astype(int)
    yy, xx = np.meshgrid(axis, axis, indexing='ij'); uv = np.c_[xx.ravel(), yy.ravel()]
    a = np.arange(n * n).reshape(n, n)
    triangles = np.concatenate([np.stack([a[:-1, :-1], a[:-1, 1:], a[1:, :-1]], -1).reshape(-1, 3),
                                np.stack([a[:-1, 1:], a[1:, 1:], a[1:, :-1]], -1).reshape(-1, 3)])
    vertices = [trial.vertices]; faces = [trial.triangles]; offset = len(trial.vertices)
    estimated = []
    for p in trial.train:
        for f in range(6):
            d = np.load(trial.depth_path(p, f)).astype(np.float32) * scale
            z = d[uv[:, 1], uv[:, 0]]
            v = xyz(p['position']) + rays(p, f, uv, 2048) * z[:, None]
            captured = valid_pixels(trial, p, f, 2048)
            # Keep a whole grid-cell margin so triangle interiors cannot bridge the excluded disc.
            if f == 5:
                margin = math.ceil(2048 / (n - 1))
                captured = cv2.erode(captured.astype(np.uint8), np.ones((margin * 2 + 1,) * 2, np.uint8)).astype(bool)
            valid = captured[uv[:, 1], uv[:, 0]] & (z > .25) & (z < 19.5 * scale)
            t = triangles[valid[triangles].all(1)]
            if not len(t):
                continue
            # Edge checks reject large triangles and depth discontinuities.
            length = np.linalg.norm(v[t] - np.roll(v[t], 1, axis=1), axis=2)
            jump = np.ptp(z[t], axis=1)
            t = t[(length.max(1) < .65) & (jump < np.maximum(.15, .06 * z[t].mean(1)))]
            vertices.append(v); faces.append(t + offset); offset += len(v)
            if len(t):
                cross = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
                area = np.linalg.norm(cross, axis=1) / 2
                normals = cross / np.maximum(2 * area[:, None], 1e-9)
                estimated.append((v[t].mean(1), normals, area, v[t]))
    path = trial.work / 'collision.obj'
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        for v in vertices:
            for row in v:stream.write('v %.5f %.5f %.5f\n' % tuple(row))
        for t in faces:
            for row in t + 1:stream.write('f %d %d %d\n' % tuple(row))
    temp.replace(path)
    return path, estimated, sum(len(x) for x in faces) - len(trial.triangles)


class Visibility:
    def __init__(self, trial):
        self.t = trial
        from run_splat import xyz
        self.cameras = np.array([xyz(p['position']) for p in trial.panos])
        self.scale = json.loads((trial.proof / 'scale.json').read_text())['scale']
        self.train_ids = {p['id'] for p in trial.train}

    def observe(self, points, normal=None, colour=False):
        from run_splat import camera, image, xyz
        seen = np.zeros(len(points), bool)
        # Ring colours are TRAIN-only even though coverage diagnoses all captures.
        rgb = np.zeros((len(points), 3), np.float32); votes = np.zeros(len(points), np.uint16)
        for p in self.t.panos:
            centre = xyz(p['position'])
            local = (points - centre) @ self.t_rotation(p)
            faces = np.argmax(local @ self.t_bases()[:, 0].T, axis=1)
            for f in range(6):
                ids = np.flatnonzero(faces == f)
                if not len(ids):continue
                v = camera(p, f); q = points[ids] @ v[:3, :3].T + v[:3, 3]
                z = q[:, 2]
                uv = np.rint(q[:, :2] / np.maximum(z[:, None], .001) * 1024 + 1023.5).astype(int)
                valid = (z > .25) & (uv.min(1) >= 0) & (uv.max(1) < 2048)
                if normal is not None:
                    direction = centre - points[ids]
                    selected_normal = normal[ids] if np.ndim(normal) == 2 else normal
                    cos = np.abs(np.sum(direction * selected_normal, axis=-1)) / np.maximum(np.linalg.norm(direction, axis=1), 1e-9)
                    valid &= cos > .15
                clipped = np.clip(uv, 0, 2047)
                mask = valid_pixels(self.t, p, f, 2048)
                valid &= mask[clipped[:, 1], clipped[:, 0]]
                depth = np.load(self.t.depth_path(p, f), mmap_mode='r')
                predicted = depth[clipped[:, 1], clipped[:, 0]].astype(np.float32) * self.scale
                valid &= np.isfinite(predicted) & (np.abs(z - predicted) < np.maximum(.10, .05 * z))
                # Coarse mesh is an additional occluder, never a source of colour.
                mesh = self.mesh_depth(p, f)
                u = np.clip(((uv + .5) * mesh.shape[0] / 2048).astype(int), 0, mesh.shape[0] - 1)
                mz = mesh[u[:, 1], u[:, 0]]
                valid &= ~np.isfinite(mz) | (z <= mz + np.maximum(.10, .05 * z))
                ids = ids[valid]; pixels = uv[valid]
                seen[ids] = True
                if colour and p['id'] in self.train_ids and len(ids):
                    bgr = image(self.t.home, p, f, 2048)
                    rgb[ids] += bgr[pixels[:, 1], pixels[:, 0], ::-1] / 255.
                    votes[ids] += 1
        rgb /= np.maximum(votes[:, None], 1)
        return seen, rgb, votes

    @staticmethod
    def t_rotation(p):
        from run_splat import P
        return P.rotation(p['rotation'])

    @staticmethod
    def t_bases():
        from run_splat import P
        return P.BASES

    def mesh_depth(self, p, f):
        folder = self.t.work / 'visibility_mesh';folder.mkdir(exist_ok=True)
        path = folder / f"{p['id']}_{f}.npy"
        if not path.exists():
            from run_splat import M
            M.W = 128 if self.t.a.smoke else 256
            for i, d in enumerate(M.depths(self.t.vertices, self.t.triangles, p)):
                dest = folder / f"{p['id']}_{i}.npy";tmp = dest.with_suffix('.tmp')
                with tmp.open('wb') as stream:np.save(stream, d.astype(np.float32))
                tmp.replace(dest)
        return np.load(path, mmap_mode='r')


def plane_groups(vertices, triangles):
    groups = {}
    for tri in triangles:
        v = vertices[tri];cross = np.cross(v[1] - v[0], v[2] - v[0]);norm = np.linalg.norm(cross)
        if norm < 1e-6:continue
        n = cross / norm
        if n[np.argmax(np.abs(n))] < 0:n = -n
        offset = float(n @ v[0])
        key = tuple(np.round(n, 3)) + (round(offset / .02),)
        group = groups.setdefault(key, {'normal': n, 'offset': offset, 'triangles': []})
        # Only merge truly coplanar triangles; no extension beyond their footprint.
        if np.max(abs(v @ group['normal'] - group['offset'])) < .005:
            group['triangles'].append(v)
        else:
            groups[key + (len(groups),)] = {'normal': n, 'offset': offset, 'triangles': [v]}
    return list(groups.values())


def surfaces(trial):
    collision, estimated, added_triangles = fused_collision(trial)
    assets = trial.work / 'holes';assets.mkdir(exist_ok=True)
    vis = Visibility(trial); planes = []; patches = []; outputs = [collision]
    totals = {k: {'area_m2': 0., 'never_seen_m2': 0.} for k in ['floor','wall','ceiling','other']}
    for group in plane_groups(trial.vertices, trial.triangles):
        n = group['normal'];v = np.array(group['triangles'])
        u = np.cross(n, [0, 0, 1] if abs(n[2]) < .85 else [0, 1, 0]);u /= np.linalg.norm(u)
        w = np.cross(n, u);origin = n * group['offset']
        xy = np.stack([(v - origin) @ u, (v - origin) @ w], -1)
        lo = xy.reshape(-1, 2).min(0);hi = xy.reshape(-1, 2).max(0)
        step = max(.05, math.sqrt(np.prod(hi - lo) / 250000))
        shape = np.maximum(1, np.ceil((hi - lo) / step).astype(int)) + 1
        inside = np.zeros((shape[1], shape[0]), np.uint8)
        for t in xy:cv2.fillConvexPoly(inside, np.rint((t - lo) / step).astype(np.int32), 1)
        yy, xx = np.nonzero(inside)
        if not len(xx):continue
        base = origin + (lo[0] + step * .5) * u + (lo[1] + step * .5) * w
        points = base + xx[:, None] * step * u + yy[:, None] * step * w
        seen, colours, votes = vis.observe(points, n, colour=True)
        kind = surface_class(n, points.mean(0), vis.cameras)
        seen_grid = np.zeros_like(inside);seen_grid[yy, xx] = seen
        train_grid = np.zeros_like(inside);train_grid[yy, xx] = votes > 0
        colour_grid = np.zeros((*inside.shape, 3), np.float32);colour_grid[yy, xx] = colours
        unseen = inside & (1 - seen_grid)
        total = len(xx) * step ** 2;never = int(unseen.sum()) * step ** 2
        totals[kind]['area_m2'] += total;totals[kind]['never_seen_m2'] += never
        tinted = np.zeros((*inside.shape, 4), np.uint8)
        tinted[unseen > 0] = [220, 45, 190, 170]
        patch_texture = np.zeros_like(tinted)
        labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(unseen, 8)
        pid = len(planes)
        for label in range(1, labels_n):
            component = labels == label;area = stats[label, cv2.CC_STAT_AREA] * step ** 2
            # Small ENCLOSED holes only, never an unseen room or silhouette edge.
            if kind not in ('floor','wall','ceiling') or area > .25 or area < step ** 2:continue
            expanded = cv2.dilate(component.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)
            ring = expanded & ~component
            if np.any(ring & ~inside.astype(bool)) or component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():continue
            if not ring.any() or np.mean(seen_grid[ring]) < .95 or np.mean(train_grid[ring]) < .95:continue
            ring_rgb = colour_grid[ring & train_grid.astype(bool)]
            mean = ring_rgb.mean(0)
            patch_id = f'plane{pid}_hole{label}'
            patch_texture[component] = [*np.rint(mean * 255).astype(int), 255]
            tinted[component] = [255, 160, 0, 200]
            cells = np.argwhere(component)
            patches.append({'id': patch_id, 'plane_id': pid, 'surface': kind, 'area_m2': float(area),
                            'normal': n.tolist(), 'mean_rgb_srgb': mean.tolist(),
                            'cells_yx': cells.tolist(), 'ring_cells': int(ring.sum()),
                            'ring_mean_only': True, 'source': 'visible ring pixels, training panoramas only',
                            'label': 'not captured by the scan', 'measured_geometry': 'simple_planes footprint; estimated visibility',
                            'not_capture_truth': True})
        overlay = assets / f'coverage_{pid}.png';flat = assets / f'patch_{pid}.png'
        cv2.imwrite(str(overlay), cv2.cvtColor(tinted, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(str(flat), cv2.cvtColor(patch_texture, cv2.COLOR_RGBA2BGRA))
        support = assets / f'support_{pid}.png';cv2.imwrite(str(support), seen_grid * 255)
        outputs.extend([overlay, flat, support])
        lower = base - .5 * step * (u + w)
        planes.append({'id': pid, 'surface': kind, 'origin': lower.tolist(), 'u': u.tolist(), 'v': w.tolist(),
                       'normal': n.tolist(), 'step': step, 'width': int(shape[0]), 'height': int(shape[1]),
                       'area_m2': float(total), 'never_seen_m2': float(never),
                       'overlay': overlay.name, 'patch': flat.name, 'support': support.name})
        if pid % 10 == 0:print('Seen-map planes', pid + 1, flush=True)
    for row in totals.values():row['never_seen_share'] = row['never_seen_m2'] / row['area_m2'] if row['area_m2'] else None
    # Estimated surfaces overlap simple_planes and each other; report separately.
    estimated_stats = {k: {'sampled_triangle_area_m2': 0., 'never_seen_m2': 0.} for k in totals}
    unseen_geometry = []
    for centres, normals, area, geometry in estimated:
        seen, _, _ = vis.observe(centres, normals)
        unseen_geometry.extend(geometry[~seen])
        for i, centre in enumerate(centres):
            row = estimated_stats[surface_class(normals[i], centre, vis.cameras)]
            row['sampled_triangle_area_m2'] += float(area[i])
            if not seen[i]:row['never_seen_m2'] += float(area[i])
    for row in estimated_stats.values():row['never_seen_share'] = row['never_seen_m2'] / row['sampled_triangle_area_m2'] if row['sampled_triangle_area_m2'] else None
    unseen_obj = assets / 'unseen.obj'
    with unseen_obj.open('w') as stream:
        for triangle in unseen_geometry:
            for vertex in triangle:stream.write('v %.5f %.5f %.5f\n' % tuple(vertex))
        for i in range(len(unseen_geometry)):stream.write('f %d %d %d\n' % (3*i+1, 3*i+2, 3*i+3))
    outputs.append(unseen_obj)
    report = {'simple_planes': totals, 'estimated_depth_surfaces': estimated_stats,
              'capture_count': len(trial.panos), 'all_capture_positions': not trial.a.smoke,
              'held_out_usage': 'coverage diagnostic only; ring colours and fused collision TRAIN-only; never used for training',
              'occlusion': 'known-pose cube projection, scaled depth agreement max(.10m,5%), mesh z-buffer occlusion; nadir excluded',
              'never_seen_definition': 'no usable capture pixel supports this sampled surface under the estimated visibility model',
              'denominator': 'known simple_planes footprint; estimated-depth surface area separate, includes overlap',
              'limitations': ['Not a whole-home completeness guarantee; absent geometry has no measurable area.',
                             'Depth/mesh error can mark captured surface unsupported or occluded surface seen.',
                             'Grid area estimate at >=5cm; sample thresholds and nadir detector are conservative heuristics.'],
              'collision': {'simple_planes_triangles': len(trial.triangles), 'fused_depth_triangles': added_triangles,
                            'depth_grid': 24 if trial.a.smoke else 48, 'max_triangle_edge_m': .65,
                            'policy': 'combined geometry; no triangles across masked pixels or depth jumps; viewer refuses unsupported floor',
                            'guarantee': 'Incomplete scans cannot guarantee all walls; unknown floor is blocked, not invented.'},
              'patch_count': len(patches)}
    write_json(trial.proof / 'seen_map.json', report)
    write_json(trial.proof / 'patches.json', {'patches': patches, 'max_hole_area_m2': .25,
               'rule': 'enclosed component on known plane; >=95% seen and train-coloured ring; constant sRGB ring mean',
               'not_capture_truth': True, 'label': 'not captured by the scan', 'training_usage': 'none; viewer-only separate overlay geometry'})
    write_json(assets / 'holes.json', {'planes': planes, 'patch_count': len(patches), 'label': 'not captured by the scan'})
    return outputs + [trial.proof / 'seen_map.json', trial.proof / 'patches.json', assets / 'holes.json']
