#!/usr/bin/env python3
"""Walkable-space navigation on a scanned home mesh (HackPrinceton W34).

Ported from robot-as-code-mac-mvp:
  * ``mapping.Grid``: world <-> cell transforms;
  * ``navigation.Costmap`` / ``plan_path``: EDT inflation, Dijkstra on an
    8-connected grid, line-of-sight shortcut.

That code rasterises MuJoCo primitives on one flat floor. This module works on
a real triangle mesh instead:
  * the surface is voxelised at 5 cm, with all triangles plus up-facing ones
    kept separately;
  * each storey band gets a 2.5D floor layer with headroom and step checks, so
    stairs link storeys while table tops and furniture never count as floor;
  * clearance is an exact distance transform rather than a fixed inflation
    radius, so one graph answers queries for any body radius;
  * bottleneck widths (maximin clearance) are computed between capture
    points, along with person and wheelchair accessibility.

Run ``build`` once per home; ``route`` then answers queries from ``nav.npz``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

VERSION = 'w34-navgrid-1'


@dataclass
class Params:
    res: float = 0.05            # cell size in x, y and z, metres
    spacing: float = 0.025       # surface sampling spacing (half a cell)
    walk_nz: float = 0.8         # floor triangles face up within ~37 degrees
    foot: float = 0.25           # obstacles lower than this above the floor are stepped over
    head: float = 1.90           # headroom a standing person needs
    step: float = 0.25           # largest walkable rise between neighbouring cells (5 cm quantisation)
    storey_min_captures: int = 3
    band_low: float = 1.0        # a storey band starts this far below its floor
    band_overlap: float = 0.8    # neighbouring bands overlap (about 4 treads) so stair treads link them
    centre_weight: float = 2.0   # routing prefers the middle of passages
    centre_sigma: float = 0.15


# Radius is half the body or chair width. ADA: a wheelchair is 0.76 m wide and
# needs a 1.52 m turning circle; it cannot climb steps.
PROFILES = {'person': dict(radius=0.20, step=0.25),
            'wheelchair': dict(radius=0.38, step=0.05, turning_radius=0.76)}
BOTTLENECK_LEVELS = np.round(np.arange(0.10, 1.501, 0.025), 3)


# --------------------------------------------------------------------------- mesh

def load_obj(folder: Path):
    """Positions and triangles only, from the Matterpak OBJ (any face arity)."""
    obj, = Path(folder).glob('*.obj')
    vertices, faces = [], []
    for line in obj.open():
        if line.startswith('v '):
            vertices.append(line.split()[1:4])
        elif line.startswith('f '):
            idx = [int(c.split('/')[0]) - 1 for c in line.split()[1:]]
            faces.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError('invalid OBJ indices')
    return v, f


def voxelize(v: np.ndarray, f: np.ndarray, p: Params):
    """Mark every 5 cm voxel touched by the surface (``occ``) and by floor-like triangles (``up``).

    Each triangle is sampled on a lattice spanned by its two shortest edges.
    The angle between them is at least 60 degrees, so sample spacing stays at
    or below ``p.spacing`` with few wasted points, even for slivers.
    """
    lo = v.min(0) - p.res
    shape = np.ceil((v.max(0) + p.res - lo) / p.res).astype(int)            # x, y, z
    occ = np.zeros((shape[2], shape[1], shape[0]), np.uint8)
    up = np.zeros_like(occ)
    P = v[f]                                                                # T, 3, 3
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-30)
    opposite = np.stack([np.linalg.norm(P[:, 1] - P[:, 2], axis=1),
                         np.linalg.norm(P[:, 2] - P[:, 0], axis=1),
                         np.linalg.norm(P[:, 0] - P[:, 1], axis=1)], 1)
    i0 = opposite.argmax(1)                                                 # vertex facing the longest edge
    ar = np.arange(len(f))
    P0, P1, P2 = P[ar, i0], P[ar, (i0 + 1) % 3], P[ar, (i0 + 2) % 3]
    e1, e2 = P1 - P0, P2 - P0
    n1 = np.clip(np.ceil(np.linalg.norm(e1, axis=1) / p.spacing), 1, 400).astype(np.int64)
    n2 = np.clip(np.ceil(np.linalg.norm(e2, axis=1) / p.spacing), 1, 400).astype(np.int64)
    key = n1 * 1000 + n2
    order = np.argsort(key, kind='stable')
    starts = np.flatnonzero(np.r_[True, key[order][1:] != key[order][:-1]])
    stops = np.r_[starts[1:], len(order)]
    floor_like = nz > p.walk_nz
    samples = 0
    for s0, s1 in zip(starts, stops):
        tris = order[s0:s1]
        a, b = n1[tris[0]], n2[tris[0]]
        ii, jj = np.meshgrid(np.arange(a + 1), np.arange(b + 1), indexing='ij')
        keep = ii * b + jj * a <= a * b
        u, w = ii[keep] / a, jj[keep] / b
        per = max(1, int(3_000_000 // len(u)))
        for c0 in range(0, len(tris), per):
            t = tris[c0:c0 + per]
            pts = P0[t, None] + u[None, :, None] * e1[t, None] + w[None, :, None] * e2[t, None]
            vox = np.floor((pts - lo) / p.res).astype(np.int64)
            occ[vox[..., 2], vox[..., 1], vox[..., 0]] = 1
            fl = floor_like[t]
            if fl.any():
                vf = vox[fl]
                up[vf[..., 2], vf[..., 1], vf[..., 0]] = 1
            samples += pts.shape[0] * pts.shape[1]
    return occ, up, lo, samples


# ------------------------------------------------------------------ floor layers

def capture_floors(top: np.ndarray, lo: np.ndarray, captures: list, p: Params):
    """Floor voxel under each capture: the highest floor-like surface top at least 0.4 m below the lens."""
    out = []
    for cap in captures:
        x, y, z = cap['position']
        ix, iy = int((x - lo[0]) / p.res), int((y - lo[1]) / p.res)
        best = None
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                jy, jx = iy + dy, ix + dx
                if not (0 <= jy < top.shape[1] and 0 <= jx < top.shape[2]):
                    continue
                ks = np.flatnonzero(top[:, jy, jx])
                ks = ks[lo[2] + (ks + .5) * p.res < z - 0.4]
                if len(ks) and (best is None or ks[-1] > best):
                    best = int(ks[-1])
        out.append(best)
    return out


def storeys_from(floor_voxels: list, p: Params):
    """Storey floor levels: heights shared by at least ``storey_min_captures`` captures (stair captures are not)."""
    h = np.array([k for k in floor_voxels if k is not None])
    radius = int(round(0.15 / p.res))
    levels, used = [], np.zeros(len(h), bool)
    while True:
        counts = np.array([((np.abs(h - x) <= radius) & ~used).sum() for x in h])
        if not len(h) or counts.max() < p.storey_min_captures:
            break
        centre = int(np.median(h[(np.abs(h - h[counts.argmax()]) <= radius) & ~used]))
        levels.append(centre)
        used |= np.abs(h - centre) <= int(round(1.5 / p.res))
    return sorted(levels)


def floor_layers(occ, up, lo, captures, p: Params):
    nzv = occ.shape[0]
    foot_v, head_v = int(round(p.foot / p.res)), int(round(p.head / p.res))
    step_v = int(round(p.step / p.res))
    upb = up.astype(bool)
    top = upb & ~np.concatenate([upb[1:], np.zeros((1,) + upb.shape[1:], bool)], 0)
    cs = np.zeros((nzv + 1,) + occ.shape[1:], np.int16)
    np.cumsum(occ, axis=0, dtype=np.int16, out=cs[1:])
    k = np.arange(nzv)
    clear = np.empty(occ.shape, bool)
    for z in range(nzv):                                                     # per slice: bounded memory
        a, b = min(z + foot_v, nzv), min(z + head_v + 1, nzv)
        clear[z] = cs[b] == cs[a]
    del cs
    cand = top & clear
    floors = capture_floors(top, lo, captures, p)
    levels = storeys_from(floors, p)
    if not levels:
        raise ValueError('no storey found under the captures')
    bands = []
    for j, level in enumerate(levels):
        z0 = 0 if j == 0 else level - int(round(p.band_low / p.res))
        z1 = nzv if j == len(levels) - 1 else levels[j + 1] - int(round((p.band_low - p.band_overlap) / p.res))
        bands.append((max(0, z0), min(nzv, z1)))
    layers = []
    for z0, z1 in bands:
        m = cand[z0:z1]
        has = m.any(0)
        h = np.where(has, m.argmax(0) + z0, -1).astype(np.int32)
        cliff = np.zeros_like(has)
        for ax in (0, 1):                                                    # 4-neighbour rises taller than a step
            a = [slice(None)] * 2; b = [slice(None)] * 2
            a[ax] = slice(0, -1); b[ax] = slice(1, None)
            both = has[tuple(a)] & has[tuple(b)]
            big = both & (np.abs(h[tuple(a)] - h[tuple(b)]) > step_v)
            cliff[tuple(a)] |= big
            cliff[tuple(b)] |= big
        free = has & ~cliff
        # Distance from a cell centre to the centre of the nearest blocked cell. The blocking surface
        # lies somewhere inside that cell, so this is unbiased to +-res/2 (subtracting res/2 would
        # shrink every 0.8 m door by up to 5 cm and fail wheelchairs on quantisation alone).
        clearance = (ndimage.distance_transform_edt(free) * p.res).astype(np.float32)
        clearance[~free] = 0
        layers.append(dict(band=(z0, z1), height=h, free=free, cliff=cliff & has, clearance=clearance))
    return layers, levels, floors


# ------------------------------------------------------------------------ graph

class Nav:
    """Walk graph over all floor layers; nodes are free cells, edges join neighbours within a step."""

    def __init__(self, layers, lo, p: Params):
        self.p, self.lo, self.layers = p, np.asarray(lo, float), layers
        ny, nx = layers[0]['free'].shape
        self.ids, offset, rows = [], 0, []
        for b, L in enumerate(layers):
            ids = np.full((ny, nx), -1, np.int64)
            iy, ix = np.nonzero(L['free'])
            ids[iy, ix] = offset + np.arange(len(iy))
            offset += len(iy)
            self.ids.append(ids)
            rows.append((np.full(len(iy), b), iy, ix, L['height'][iy, ix], L['clearance'][iy, ix]))
        self.band, self.iy, self.ix, self.h, self.clear = (np.concatenate(c) for c in zip(*rows))
        self.n = offset
        u, v, length, dh = [], [], [], []
        for b, L in enumerate(layers):
            ids, free, h = self.ids[b], L['free'], L['height']
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                ya, yb = slice(0, ny - dy), slice(dy, ny)
                xa = slice(max(0, -dx), nx - max(0, dx)); xb = slice(max(0, dx), nx - max(0, -dx))
                ok = free[ya, xa] & free[yb, xb]
                if dy and dx:                                                # no corner cutting
                    ok &= free[ya, xb] & free[yb, xa]
                d = np.abs(h[ya, xa] - h[yb, xb])
                u.append(ids[ya, xa][ok]); v.append(ids[yb, xb][ok]); dh.append(d[ok])
                planar = math.hypot(dx, dy) * p.res
                length.append(np.sqrt(planar ** 2 + (d[ok] * p.res) ** 2))
        for b in range(len(layers) - 1):                                     # stair treads seen by two bands
            A, B = layers[b], layers[b + 1]
            same = A['free'] & B['free'] & (A['height'] == B['height'])
            u.append(self.ids[b][same]); v.append(self.ids[b + 1][same])
            dh.append(np.zeros(same.sum(), np.int64)); length.append(np.full(same.sum(), 1e-6))
        self.u, self.v = np.concatenate(u), np.concatenate(v)
        self.length, self.dh = np.concatenate(length), np.concatenate(dh)
        self.cmin = np.minimum(self.clear[self.u], self.clear[self.v])
        self.portals = []                                                    # see find_portals

    # -- geometry
    def world(self, nodes):
        nodes = np.asarray(nodes)
        x = self.lo[0] + (self.ix[nodes] + .5) * self.p.res
        y = self.lo[1] + (self.iy[nodes] + .5) * self.p.res
        z = self.lo[2] + (self.h[nodes] + 1) * self.p.res                    # top of the floor voxel
        return np.stack([x, y, z], -1)

    def edge_mask(self, radius, step):
        return (self.cmin >= radius) & (self.dh * self.p.res <= step + 1e-6)

    def portal_edges(self, radius, step):
        """Bridge each usable portal: the nearest node fitting ``radius`` on either side of the thin surface.

        A portal is usable when its floor step is within ``step`` and, if its opening width was measured
        (``width_m``, e.g. from door masks), the body fits; an unmeasured width counts for walkers only,
        so callers pass ``walker=True`` through ``graph``.
        """
        u, v, w, ids = [], [], [], []
        for q in self.portals:
            if q['step_m'] > step + 1e-6 or (q.get('width_m') is not None and q['width_m'] < 2 * radius):
                continue
            d, mid = np.asarray(q['dir'], float), np.asarray(q['xy'], float)
            ends = []
            for sign, z in ((-1, q['floor_z'][0]), (1, q['floor_z'][1])):
                x, y = mid + sign * d * (q['thickness_m'] / 2 + radius + self.p.res)
                hit = self.snap((x, y, z), 0.4, band=q['band'], min_clear=radius)
                if hit is None or (self.world([hit[0]])[0][:2] - mid) @ d * sign <= 0:
                    break
                ends.append(hit[0])
            if len(ends) == 2:
                u.append(ends[0]); v.append(ends[1]); ids.append(q['id'])
                w.append(float(np.linalg.norm(self.world([ends[0]])[0] - self.world([ends[1]])[0])))
        return np.array(u, np.int64), np.array(v, np.int64), np.array(w), ids

    def graph(self, radius, step, weighted=True, portals=False, walker=True):
        m = self.edge_mask(radius, step)
        u, v, w = self.u[m], self.v[m], self.length[m]
        if weighted:
            w = w * (1 + self.p.centre_weight * np.exp(-(self.cmin[m] - radius) / self.p.centre_sigma))
        if portals:
            saved = self.portals
            if not walker:                                                   # unmeasured openings: walkers only
                self.portals = [q for q in saved if q.get('width_m') is not None]
            pu, pv, pw, _ = self.portal_edges(radius, step)
            self.portals = saved
            u, v = np.r_[u, pu], np.r_[v, pv]
            w = np.r_[w, pw * (1 + self.p.centre_weight if weighted else 1)]
        return coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(self.n, self.n)).tocsr()

    def components(self, radius, step, portals=False, walker=True):
        g = self.graph(radius, step, weighted=False, portals=portals, walker=walker)
        _, labels = connected_components(g, directed=False)
        labels = labels.copy()
        labels[self.clear < radius] = -1
        return labels

    def snap(self, xyz, max_dist=0.6, band=None, min_clear=0.0):
        """Nearest free node with at least ``min_clear`` clearance, on a floor within 0.3 m of the given height."""
        x, y, z = xyz
        ix, iy = int((x - self.lo[0]) / self.p.res), int((y - self.lo[1]) / self.p.res)
        r = int(math.ceil(max_dist / self.p.res))
        best = None
        for b, ids in enumerate(self.ids):
            if band is not None and b != band:
                continue
            y0, y1 = max(0, iy - r), min(ids.shape[0], iy + r + 1)
            x0, x1 = max(0, ix - r), min(ids.shape[1], ix + r + 1)
            win = ids[y0:y1, x0:x1]
            cand = win[win >= 0]
            if not len(cand):
                continue
            fz = self.lo[2] + (self.h[cand] + 1) * self.p.res
            ok = (np.abs(fz - z) <= 0.3) & (self.clear[cand] >= min_clear)
            if not ok.any():
                continue
            cand = cand[ok]
            d = np.hypot(self.lo[0] + (self.ix[cand] + .5) * self.p.res - x, self.lo[1] + (self.iy[cand] + .5) * self.p.res - y)
            j = int(np.argmin(d - 1e-3 * np.minimum(self.clear[cand], 0.3)))
            if d[j] <= max_dist and (best is None or d[j] < best[1]):
                best = (int(cand[j]), float(d[j]))
        return best

    # -- routes
    def route(self, a, b, profile='person', radius=None, portals=False):
        prof = PROFILES[profile]
        radius = prof['radius'] if radius is None else radius
        walker = profile == 'person'
        g = self.graph(radius, prof['step'], portals=portals, walker=walker)
        dist, pred = dijkstra(g, indices=a, return_predecessors=True)
        if not np.isfinite(dist[b]):
            return None
        path = [b]
        while path[-1] != a:
            path.append(pred[path[-1]])
        path = np.array(path[::-1])
        used = []
        if portals:
            saved = self.portals
            if not walker:
                self.portals = [q for q in saved if q.get('width_m') is not None]
            pu, pv, _, pid = self.portal_edges(radius, prof['step'])
            self.portals = saved
            bridge = {frozenset((int(x), int(y))): i for x, y, i in zip(pu, pv, pid)}
            used = [bridge[k] for k in (frozenset((int(x), int(y))) for x, y in zip(path[:-1], path[1:])) if k in bridge]
        pts = self.shortcut(path, radius, prof['step'])
        pts = pts[np.r_[True, np.abs(np.diff(pts, axis=0)).max(1) > 1e-9]]          # stair links repeat a point
        seg = np.diff(pts, axis=0)
        return dict(profile=profile, radius=radius, points=np.round(pts, 3).tolist(),
                    length_m=round(float(np.linalg.norm(seg, axis=1).sum()), 2),
                    narrowest_clearance_m=round(float(self.clear[path].min()), 3),
                    storeys=sorted(set(int(x) for x in self.band[path])),
                    unverified_portals=[i for i in used if next(q for q in self.portals if q['id'] == i).get('width_m') is None],
                    portals=used)

    def _clear_line(self, a, b, radius, step):
        """Straight walk a->b stays within one band, keeps clearance and never rises more than a step."""
        if self.band[a] != self.band[b]:
            return False
        L = self.layers[self.band[a]]
        n = max(2, int(math.hypot(self.ix[b] - self.ix[a], self.iy[b] - self.iy[a]) * 2) + 1)
        t = np.linspace(0, 1, n)
        yy = np.round(self.iy[a] + (self.iy[b] - self.iy[a]) * t).astype(int)
        xx = np.round(self.ix[a] + (self.ix[b] - self.ix[a]) * t).astype(int)
        if not L['free'][yy, xx].all() or (L['clearance'][yy, xx] < radius).any():
            return False
        return bool((np.abs(np.diff(L['height'][yy, xx])) * self.p.res <= step + 1e-6).all())

    def shortcut(self, path, radius, step):
        """Keep a far visible node at each step (binary search, O(log n) line checks per kept node)."""
        keep, i, last = [path[0]], 0, len(path) - 1
        while i < last:
            if self._clear_line(path[i], path[last], radius, step):
                j = last
            else:
                lo, hi = i + 1, last
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    lo, hi = (mid, hi) if self._clear_line(path[i], path[mid], radius, step) else (lo, mid)
                j = lo
            keep.append(path[j]); i = j
        return self.world(np.array(keep))


# --------------------------------------------------------------------- portals

def neighbour_pairs(path):
    """Matterport sweep links (``captures[].neighbors`` in the world build) as sorted id pairs."""
    caps = json.loads(Path(path).read_text())['captures']
    known = {c['id'] for c in caps}
    return sorted({tuple(sorted((c['id'], j))) for c in caps for j in c.get('neighbors', []) if j in known})


def find_portals(nav: 'Nav', captures, floors, pairs, p: Params, max_thick=0.20):
    """Thin surfaces (at most ``max_thick``) cutting the straight line between two linked captures.

    Matterport links two sweeps when one is visible from the other, so their line is open, or seen
    through glass. One short blocked run with floor within a step on both sides is most likely a door
    that was shut during the scan, a glass door, or a glass wall. The mesh cannot tell these apart,
    so every portal stays ``unverified`` and has no width until a door mask measures it.
    """
    node = {}
    for cap, fv in zip(captures, floors):
        if fv is not None:
            hit = nav.snap((cap['position'][0], cap['position'][1], nav.lo[2] + (fv + 1) * p.res), 0.6)
            if hit:
                node[cap['id']] = hit[0]
    portals = []
    for a, b in pairs:
        na, nb = node.get(a), node.get(b)
        if na is None or nb is None or nav.band[na] != nav.band[nb]:
            continue
        band = int(nav.band[na])
        L = nav.layers[band]
        pa, pb = nav.world([na, nb])
        length = float(np.hypot(*(pb - pa)[:2]))
        if length < 2 * p.res:
            continue
        t = np.linspace(0, 1, int(length / p.res * 4) + 2)
        iy = ((pa[1] + (pb[1] - pa[1]) * t - nav.lo[1]) / p.res).astype(int)
        ix = ((pa[0] + (pb[0] - pa[0]) * t - nav.lo[0]) / p.res).astype(int)
        keep = np.r_[True, (np.diff(iy) != 0) | (np.diff(ix) != 0)]
        iy, ix = iy[keep], ix[keep]
        blocked = np.flatnonzero(~L['free'][iy, ix])
        if not len(blocked) or blocked[-1] - blocked[0] + 1 != len(blocked):   # exactly one blocked run
            continue
        s0, e0 = blocked[0], blocked[-1]
        if s0 == 0 or e0 == len(iy) - 1:
            continue
        ya, xa, yb, xb = iy[s0 - 1], ix[s0 - 1], iy[e0 + 1], ix[e0 + 1]
        thick = math.hypot(yb - ya, xb - xa) * p.res - p.res
        ha, hb = int(L['height'][ya, xa]), int(L['height'][yb, xb])
        if thick > max_thick or abs(ha - hb) * p.res > PROFILES['person']['step'] + 1e-6:
            continue
        mid = nav.lo[:2] + (np.array([(xa + xb) / 2, (ya + yb) / 2]) + .5) * p.res
        near = next((q for q in portals if q['band'] == band and math.hypot(*(np.array(q['xy']) - mid)) < 0.6), None)
        if near:
            near['pairs'].append([a, b])
            continue
        portals.append(dict(id=f'portal_{len(portals)}', band=band, xy=mid.round(3).tolist(),
                            dir=((pb - pa)[:2] / length).round(4).tolist(), thickness_m=round(thick, 3),
                            step_m=round(abs(ha - hb) * p.res, 2),
                            floor_z=[round(float(nav.lo[2] + (h + 1) * p.res), 3) for h in (ha, hb)],
                            pairs=[[a, b]], width_m=None,
                            status='unverified: thin surface between linked captures (shut door, glass door or glass wall)'))
    return portals


# ------------------------------------------------------------------- analysis

def analyse(nav: Nav, captures, floors, levels, p: Params):
    """Per capture: floor, storey, snapped node. Per profile: reachability. Per pair: widest passage, walk."""
    k = len(captures)
    floor_xyz = [None if fv is None else (c['position'][0], c['position'][1], nav.lo[2] + (fv + 1) * p.res)
                 for c, fv in zip(captures, floors)]

    def snap_all(min_clear):
        hits = [nav.snap(xyz, 0.6, min_clear=min_clear) if xyz else None for xyz in floor_xyz]
        return np.array([h[0] if h else -1 for h in hits]), hits

    _, hits = snap_all(0.0)
    rows = []
    for cap, fv, hit in zip(captures, floors, hits):
        storey = None
        if fv is not None:
            gaps = [abs(fv - lv) for lv in levels]
            storey = int(np.argmin(gaps)) if min(gaps) * p.res <= 0.3 else None
        rows.append(dict(id=cap['id'], floor_z=None if fv is None else round(float(nav.lo[2] + (fv + 1) * p.res), 2),
                         storey=storey, snapped_m=None if not hit else round(hit[1], 3),
                         clearance_m=None if not hit else round(float(nav.clear[hit[0]]), 3)))
    out, nodes = dict(captures=rows), {}
    runs = [(name, prof, False) for name, prof in PROFILES.items()]
    if nav.portals:
        runs.append(('person_via_portals', PROFILES['person'], True))
    for name, prof, use in runs:
        cn, phits = snap_all(prof['radius'])                                   # a body of this radius fits there
        nodes[name] = cn
        lab = nav.components(prof['radius'], prof['step'], portals=use, walker=name != 'wheelchair')
        cl = np.where(cn >= 0, lab[np.maximum(cn, 0)], -1)
        ids, counts = np.unique(cl[cl >= 0], return_counts=True)
        main = int(ids[counts.argmax()]) if len(ids) else None
        for r, c, h in zip(rows, cl, phits):
            r[f'{name}_component'] = int(c)
            r[f'{name}_main'] = bool(main is not None and c == main)
            r[f'{name}_snap_m'] = None if not h else round(h[1], 3)
        cells = lab == main if main is not None else np.zeros(nav.n, bool)
        per = lambda m: {f'storey_{j}': round(float(((nav.band == j) & m).sum() * p.res ** 2), 1) for j in range(len(levels))}
        summary = dict(radius_m=prof['radius'], max_step_m=prof['step'], unverified_portals=use,
                       captures_in_main=int(sum(r[f'{name}_main'] for r in rows)),
                       captures_snapped=int((cn >= 0).sum()), components_with_captures=int(len(ids)),
                       main_area_m2=per(cells))
        if 'turning_radius' in prof:
            summary['turning_space_m2'] = per(cells & (nav.clear >= prof['turning_radius']))
        out[name] = summary
    out['person']['captures_not_reachable'] = [r['id'] for r in rows if not r['person_main']]
    # Widest passage between two captures: the largest radius whose component holds both (walker steps).
    step = PROFILES['person']['step']
    best = np.zeros((k, k), np.float32)
    for t in BOTTLENECK_LEVELS:
        cn, _ = snap_all(float(t))
        lab = nav.components(float(t), step)
        cl = np.where(cn >= 0, lab[np.maximum(cn, 0)], -1)
        best[(cl[:, None] == cl[None, :]) & (cl[:, None] >= 0)] = t
    width = np.round(2 * best, 3)
    # Walking distances on the person graph (plain lengths, not the centre-weighted cost).
    def walks(portals):
        pn = nodes['person']
        g = nav.graph(PROFILES['person']['radius'], step, weighted=False, portals=portals)
        dist = np.full((k, k), np.nan, np.float32)
        src_all = np.flatnonzero(pn >= 0)
        for c0 in range(0, len(src_all), 12):
            src = src_all[c0:c0 + 12]
            dist[np.ix_(src, src_all)] = dijkstra(g, indices=pn[src])[:, pn[src_all]]
        dist[~np.isfinite(dist)] = np.nan
        return dist
    dist = walks(False)
    out['walk_via_portals'] = walks(True) if nav.portals else None
    # Captures a walker reaches but a wheelchair does not, with the likely reason.
    main = [i for i, r in enumerate(rows) if r['wheelchair_main']]
    chair = 2 * PROFILES['wheelchair']['radius']
    blocked = []
    for i, r in enumerate(rows):
        if r['person_main'] and not r['wheelchair_main']:
            j = max(main, key=lambda j: width[i, j]) if main else None
            w = float(width[i, j]) if j is not None else 0.0
            main_storeys = {rows[m]['storey'] for m in main}
            cause = ('other storey: stairs only' if r['storey'] not in main_storeys else
                     'no chair-wide floor within 0.6 m of the capture' if nodes['wheelchair'][i] < 0 else
                     'step over 5 cm' if w >= chair else 'passage narrower than the chair')
            blocked.append(dict(id=r['id'], storey=r['storey'], widest_passage_to_main_m=w, likely_cause=cause,
                                walk_m=None if j is None or np.isnan(dist[i, j]) else round(float(dist[i, j]), 2)))
    out['wheelchair']['captures_not_reachable'] = blocked
    return out, nodes, width, dist


# ---------------------------------------------------------------------- outputs

def palette(c):
    """Clearance (m) -> RGB: red under a wheelchair radius, amber under 0.76 m, teal above."""
    t = np.clip(c / 1.2, 0, 1)
    rgb = np.stack([np.interp(t, [0, .3, .63, 1], [180, 230, 70, 30]),
                    np.interp(t, [0, .3, .63, 1], [50, 170, 170, 120]),
                    np.interp(t, [0, .3, .63, 1], [50, 60, 150, 200])], -1)
    return rgb.astype(np.uint8)


def draw_maps(nav: Nav, rows, cap_nodes, routes, out: Path, scale=2):
    from PIL import Image, ImageDraw
    files = []
    for b, L in enumerate(nav.layers):
        img = np.full(L['free'].shape + (3,), 24, np.uint8)
        img[L['cliff']] = (120, 20, 120)
        img[L['free']] = palette(L['clearance'][L['free']])
        im = Image.fromarray(img[::-1]).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        H = img.shape[0]
        px = lambda ix, iy: ((ix + .5) * scale, (H - iy - .5) * scale)
        zlo, zhi = nav.lo[2] + np.array(L['band']) * nav.p.res + np.array([0, 1e-6])
        inside = lambda z: zlo < z <= zhi                                     # floor tops of this band
        for route in routes:
            col = (255, 255, 255) if route['profile'] == 'person' else (40, 120, 255)
            P = route['points']
            for (xa, ya, za), (xb, yb, zb) in zip(P[:-1], P[1:]):
                if inside(za) and inside(zb):
                    d.line([px((xa - nav.lo[0]) / nav.p.res - .5, (ya - nav.lo[1]) / nav.p.res - .5),
                            px((xb - nav.lo[0]) / nav.p.res - .5, (yb - nav.lo[1]) / nav.p.res - .5)], fill=col, width=3)
        for q in nav.portals:
            if q['band'] == b:
                x, y = px((q['xy'][0] - nav.lo[0]) / nav.p.res - .5, (q['xy'][1] - nav.lo[1]) / nav.p.res - .5)
                d.rectangle([x - 6, y - 6, x + 6, y + 6], outline=(0, 0, 0), fill=(255, 220, 0))
        for r, node in zip(rows, cap_nodes):
            if node < 0 or nav.band[node] != b:
                continue
            x, y = px(nav.ix[node], nav.iy[node])
            col = (255, 255, 255) if r['wheelchair_main'] else (255, 60, 60)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(0, 0, 0), fill=col)
        path = out / f'map_band{b}.png'
        im.save(path)
        files.append(path.name)
    return files


def build(args):
    t0 = time.time()
    p = Params()
    captures = json.loads(Path(args.captures).read_text())['captures']
    v, f = load_obj(Path(args.mesh))
    occ, up, lo, samples = voxelize(v, f, p)
    nzv = occ.shape[0]
    t1 = time.time()
    layers, levels, floors = floor_layers(occ, up, lo, captures, p)
    del occ, up
    nav = Nav(layers, lo, p)
    if args.neighbours:
        nav.portals = find_portals(nav, captures, floors, neighbour_pairs(args.neighbours), p)
    t2 = time.time()
    summary, nodes, width, dist = analyse(nav, captures, floors, levels, p)
    dist_portals = summary.pop('walk_via_portals')
    cap_nodes, chair_nodes = nodes['person'], nodes['wheelchair']
    t3 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = summary['captures']
    routes = []
    valid = np.flatnonzero(cap_nodes >= 0)
    if len(valid) >= 2:                                                        # longest person walk, and a wheelchair example
        dp = dist if dist_portals is None else dist_portals
        d = np.where(np.isnan(dp), -1, dp)
        i, j = np.unravel_index(int(np.argmax(d)), d.shape)
        r = nav.route(int(cap_nodes[i]), int(cap_nodes[j]), 'person', portals=dist_portals is not None)
        if r:
            routes.append(dict(r, from_id=rows[i]['id'], to_id=rows[j]['id']))
        d = np.where(np.isnan(dist), -1, dist)
        main = [k for k in valid if rows[k]['wheelchair_main']]
        if len(main) >= 2:
            dm = d[np.ix_(main, main)]
            a, b = np.unravel_index(int(np.argmax(dm)), dm.shape)
            r = nav.route(int(chair_nodes[main[a]]), int(chair_nodes[main[b]]), 'wheelchair')
            if r:
                routes.append(dict(r, from_id=rows[main[a]]['id'], to_id=rows[main[b]]['id']))
    maps = draw_maps(nav, rows, cap_nodes, routes, out)
    np.savez_compressed(out / 'nav.npz', lo=nav.lo, res=p.res, levels=np.array(levels),
                        bands=np.array([L['band'] for L in layers]),
                        height=np.stack([L['height'] for L in layers]).astype(np.int16),
                        free=np.stack([L['free'] for L in layers]),
                        cliff=np.stack([L['cliff'] for L in layers]),
                        clearance=np.stack([L['clearance'] for L in layers]).astype(np.float16),
                        params=json.dumps(asdict(p)), portals=json.dumps(nav.portals))
    ids = [c['id'] for c in captures]
    (out / 'pairs.json').write_text(json.dumps(dict(
        capture_ids=ids,
        walk_m=[[None if np.isnan(x) else round(float(x), 2) for x in row] for row in dist],
        walk_via_portals_m=None if dist_portals is None else
        [[None if np.isnan(x) else round(float(x), 2) for x in row] for row in dist_portals],
        widest_passage_m=width.tolist(),
        note='walk_m: shortest walk for a 0.40 m wide person (steps up to 0.25 m). '
             'widest_passage_m: widest body (2 x radius) that can travel between the two captures with '
             'walker steps, 0.20 to 3.00 m in 5 cm steps, +-5 cm from the 5 cm grid; 0 means no connection '
             'even at 0.20 m. Ends snap to the nearest floor that fits the body within 0.6 m of the capture.')))
    report = dict(version=VERSION, model=args.model, mesh=str(args.mesh), params=asdict(p), profiles=PROFILES,
                  grid=dict(cells_xy=list(layers[0]['free'].shape[::-1]), voxels_z=int(nzv),
                            origin=nav.lo.round(3).tolist(), surface_samples=int(samples)),
                  storeys_floor_z=[round(float(lo[2] + (lv + 1) * p.res), 2) for lv in levels],
                  bands_z=[[round(float(lo[2] + z * p.res), 2) for z in L['band']] for L in layers],
                  nodes=int(nav.n), edges=int(len(nav.u)),
                  free_area_m2=[round(float(L['free'].sum() * p.res ** 2), 1) for L in layers],
                  timings_s=dict(voxelize=round(t1 - t0, 1), layers_graph=round(t2 - t1, 1), analyse=round(t3 - t2, 1)),
                  maps=maps, example_routes=routes, portals=nav.portals, **summary)
    (out / 'nav.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ['storeys_floor_z', 'bands_z', 'nodes', 'edges', 'free_area_m2', 'timings_s']}))
    for name in [n for n in ('person', 'wheelchair', 'person_via_portals') if n in report]:
        print(name, json.dumps({k: v for k, v in report[name].items() if k != 'captures_not_reachable'}))
    print('portals:', len(nav.portals), 'wheelchair blocked captures:', len(report['wheelchair']['captures_not_reachable']))
    for r in routes:
        print('route', r['profile'], r['from_id'][:6], '->', r['to_id'][:6], r['length_m'], 'm, storeys', r['storeys'],
              'narrowest', r['narrowest_clearance_m'], 'portals', r['portals'])
    return report


def load(out: Path) -> Nav:
    z = np.load(Path(out) / 'nav.npz')
    p = Params(**json.loads(str(z['params'])))
    layers = [dict(band=tuple(b), height=h.astype(np.int32), free=fr, cliff=cl, clearance=c.astype(np.float32))
              for b, h, fr, cl, c in zip(z['bands'], z['height'], z['free'], z['cliff'], z['clearance'])]
    nav = Nav(layers, z['lo'], p)
    nav.portals = json.loads(str(z['portals'])) if 'portals' in z else []
    return nav


def point(text, nav: Nav, floors, radius):
    """A capture id (its floor from the build) or ``x,y,floor_z``, snapped to a node that fits ``radius``."""
    if text in floors:
        x, y, z = floors[text]
        if z is None:
            return None
    else:
        x, y, z = map(float, text.split(','))
    return nav.snap((x, y, z), 0.6, min_clear=radius)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--mesh', required=True); b.add_argument('--captures', required=True)
    b.add_argument('--out', required=True); b.add_argument('--model', default='HFwcH424n9w')
    b.add_argument('--neighbours', help='world.json whose captures[].neighbors are Matterport sweep links')
    r = sub.add_parser('route', help='route between two capture ids or x,y,floor_z points')
    r.add_argument('--nav', required=True); r.add_argument('--captures', required=True)
    r.add_argument('--from', dest='src', required=True); r.add_argument('--to', dest='dst', required=True)
    r.add_argument('--profile', choices=sorted(PROFILES), default='person')
    r.add_argument('--portals', action='store_true', help='allow unverified portals (walkers only)')
    args = ap.parse_args()
    if args.cmd == 'build':
        build(args)
        return
    nav = load(Path(args.nav))
    built = {r['id']: r['floor_z'] for r in json.loads((Path(args.nav) / 'nav.json').read_text())['captures']}
    floors = {c['id']: (*c['position'][:2], built.get(c['id']))
              for c in json.loads(Path(args.captures).read_text())['captures']}
    radius = PROFILES[args.profile]['radius']
    a, b = point(args.src, nav, floors, radius), point(args.dst, nav, floors, radius)
    if not a or not b:
        sys.exit(f'start or goal has no floor that fits a {args.profile} within 0.6 m')
    res = nav.route(a[0], b[0], args.profile, portals=args.portals)
    print(json.dumps(res if res else dict(profile=args.profile, reachable=False)))


if __name__ == '__main__':
    main()
