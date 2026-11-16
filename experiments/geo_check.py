"""Go/no-go: how well does Matterport's own geo-alignment of a home scan match
the Vexcel aerial building footprint, and how much does a refinement improve it?

For N residential models:
  1. Matterport: wall graph (all floors) + geoLocationOf() for 3 reference points
     -> affine map model(x, y) -> local East/North metres.
  2. Vexcel property/extract at the model's lat/lon -> structure footprint polygon.
  3. Rasterise both, measure IoU / containment / heading offset as delivered,
     then search rotation (+-20 deg) and shift (+-8 m) for the best IoU.

Usage: python3 -I experiments/geo_check.py [N]
Reads keys from .env in the project root. Writes data/geo_check/.
"""
import base64
import collections
import csv
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from matplotlib.path import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "geo_check")
ENV = dict(l.strip().split("=", 1) for l in open(os.path.join(ROOT, ".env")) if "=" in l)
MP_AUTH = base64.b64encode(
    f"{ENV['MATTERPORT_API_KEY']}:{ENV['MATTERPORT_API_TOKEN']}".encode()).decode()
VX = "https://api.vexcelgroup.com/v2/"

RES = 0.2          # metres per cell for the comparison grid
GRID = 384         # cells per side (76.8 m)
MAX_SHIFT = 40     # cells (8 m)
ROT_RANGE = np.arange(-20, 20.01, 0.5)


def gql(query):
    req = urllib.request.Request(
        "https://api.matterport.com/api/models/graph",
        json.dumps({"query": query}).encode(),
        {"Content-Type": "application/json", "Authorization": "Basic " + MP_AUTH})
    try:
        return json.load(urllib.request.urlopen(req, timeout=90))
    except urllib.error.HTTPError as e:
        return {"errors": [{"message": f"HTTP {e.code} {e.read().decode()[:200]}"}]}


def vexcel_property(lat, lon):
    url = VX + "property/extract?" + urllib.parse.urlencode({
        "wkt": f"POINT({lon} {lat})", "srid": 4326, "layer": "urban",
        "latest": "true", "include": "precomputed", "unit-system": "metric"})
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"X-API-KEY": ENV["VEXCEL_API_KEY"]})
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            return {"error": e.code}
    return {"error": 429}


MODEL_Q = """{ model(id:"%s"){
  floors{ edges(includeUsed:true){ type vertices{position{x y}} openings{type} } }
  geocoordinates{ latitude longitude source
    p0: geoLocationOf(modelLocation:{x:0,y:0,z:0}){lat long}
    px: geoLocationOf(modelLocation:{x:20,y:0,z:0}){lat long}
    py: geoLocationOf(modelLocation:{x:0,y:20,z:0}){lat long} } } }"""


def enu(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = float(lat), float(lon), float(lat0), float(lon0)
    return np.array([(lon - lon0) * 111320.0 * math.cos(math.radians(lat0)),
                     (lat - lat0) * 110574.0])


def interior_points(segments):
    """Fill the wall graph: cells not reachable from outside = the scanned interior.
    Returns Nx2 model-frame points (0.1 m spacing) or None if the walls leak."""
    res = 0.1
    pts = np.array([p for s in segments for p in s])
    lo = pts.min(0) - 1.0
    shape = np.ceil((pts.max(0) + 1.0 - lo) / res).astype(int) + 1
    if shape.max() > 1500:
        return None
    for dilate in (1, 2, 3):
        wall = np.zeros(shape, bool)
        for a, b in segments:
            n = max(2, int(np.hypot(*(np.subtract(b, a))) / (res / 2)) + 1)
            line = np.linspace(a, b, n)
            ij = ((line - lo) / res).astype(int)
            wall[ij[:, 0], ij[:, 1]] = True
        for _ in range(dilate):
            w = wall.copy()
            w[1:] |= wall[:-1]; w[:-1] |= wall[1:]
            w[:, 1:] |= wall[:, :-1]; w[:, :-1] |= wall[:, 1:]
            wall = w
        outside = np.zeros(shape, bool)
        queue = collections.deque([(0, 0)])
        outside[0, 0] = True
        while queue:
            i, j = queue.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u, v = i + di, j + dj
                if 0 <= u < shape[0] and 0 <= v < shape[1] and not outside[u, v] and not wall[u, v]:
                    outside[u, v] = True
                    queue.append((u, v))
        inside = ~outside
        if inside.sum() * res * res > 30:      # > 30 m2 -> the walls closed
            ij = np.argwhere(inside)
            return ij * res + lo + res / 2
    return None


def dominant_angle(segments):
    """Length-weighted dominant wall direction, modulo 90 degrees."""
    acc = 0j
    for a, b in segments:
        d = np.subtract(b, a)
        acc += np.hypot(*d) * np.exp(4j * math.atan2(d[1], d[0]))
    return math.degrees(np.angle(acc)) / 4


def raster(points, origin):
    ij = np.floor((points - origin) / RES).astype(int) + GRID // 2
    ok = (ij >= 0).all(1) & (ij < GRID).all(1)
    m = np.zeros((GRID, GRID), bool)
    m[ij[ok, 0], ij[ok, 1]] = True
    return m


def parse_ring(wkt):
    ring = re.search(r"\(\(([^)]+)\)", wkt).group(1)
    return np.array([[float(v) for v in p.split()[:2]] for p in ring.split(",")])


def analyse(sid, row):
    r = gql(MODEL_Q % sid)
    if r.get("errors"):
        return {"sid": sid, "skip": "matterport: " + r["errors"][0]["message"][:80]}
    m = r["data"]["model"]
    segs = [tuple((v["position"]["x"], v["position"]["y"]) for v in e["vertices"])
            for f in m["floors"] for e in f["edges"] if len(e["vertices"]) == 2]
    if len(segs) < 8:
        return {"sid": sid, "skip": "no wall graph"}
    windows = sum(1 for f in m["floors"] for e in f["edges"]
                  for o in (e.get("openings") or []) if o["type"] == "window")
    g = m["geocoordinates"]
    lat0, lon0 = g["p0"]["lat"], g["p0"]["long"]
    ex = enu(g["px"]["lat"], g["px"]["long"], lat0, lon0) / 20
    ey = enu(g["py"]["lat"], g["py"]["long"], lat0, lon0) / 20
    A = np.column_stack([ex, ey])                 # model xy -> ENU metres
    inner = interior_points(segs)
    if inner is None:
        return {"sid": sid, "skip": "walls do not close"}
    inner_enu = inner @ A.T
    segs_enu = [tuple(np.array(p) @ A.T for p in s) for s in segs]

    prop = vexcel_property(g["latitude"], g["longitude"])
    if "error" in prop or not prop.get("features"):
        return {"sid": sid, "skip": f"vexcel: {prop.get('error', 'no parcel')}"}
    structs = prop["features"][0]["properties"].get("structures") or []
    rings = []
    for s in structs:
        if s.get("footprint.wkt"):
            ll = parse_ring(s["footprint.wkt"])
            rings.append(np.array([enu(la, lo, lat0, lon0) for lo, la in ll]))
    if not rings:
        return {"sid": sid, "skip": "no footprint"}
    centre = inner_enu.mean(0)
    ring = max(rings, key=lambda rg: Path(rg).contains_points(inner_enu[::25]).mean()
               - 1e-3 * np.hypot(*(rg.mean(0) - centre)))
    origin = ring.mean(0)

    ax = (np.arange(GRID) - GRID // 2 + 0.5) * RES
    gx, gy = np.meshgrid(ax + origin[0], ax + origin[1], indexing="ij")
    F = Path(ring).contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(GRID, GRID)
    aF = F.sum()
    fF = np.fft.rfft2(F.astype(float))

    def score(theta_deg):
        t = math.radians(theta_deg)
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        M = raster((inner_enu - centre) @ R.T + centre, origin)
        corr = np.fft.irfft2(fF * np.conj(np.fft.rfft2(M.astype(float))), s=F.shape)
        return M, corr

    M0, corr0 = score(0.0)
    ov0 = corr0[0, 0]
    iou0 = ov0 / (aF + M0.sum() - ov0)
    cont0 = ov0 / M0.sum()

    best = (iou0, 0.0, 0, 0, cont0)
    shifts = np.r_[0:MAX_SHIFT + 1, -MAX_SHIFT:0]
    for th in ROT_RANGE:
        M, corr = score(th)
        sub = corr[np.ix_(shifts, shifts)]
        iou = sub / (aF + M.sum() - sub)
        k = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[k] > best[0]:
            best = (float(iou[k]), float(th), int(shifts[k[0]]), int(shifts[k[1]]),
                    float(sub[k] / M.sum()))
    iou1, th, dx, dy, cont1 = best

    ring_segs = list(zip(ring[:-1], ring[1:]))
    heading = (dominant_angle(segs_enu) - dominant_angle(ring_segs) + 45) % 90 - 45
    return {
        "sid": sid, "address": f"{row['address']}, {row['city']}", "windows": windows,
        "walls": len(segs), "interior_m2": round(float(len(inner) * 0.01), 1),
        "footprint_m2": round(float(aF * RES * RES), 1),
        "iou_before": round(float(iou0), 3), "contain_before": round(float(cont0), 3),
        "iou_after": round(iou1, 3), "contain_after": round(cont1, 3),
        "fix_rot_deg": th, "fix_shift_m": round(math.hypot(dx, dy) * RES, 2),
        "heading_offset_deg": round(heading, 2),
        "_plot": {"ring": ring.tolist(), "inner": inner_enu[::40].tolist(),
                  "centre": centre.tolist(), "th": th, "d": [dx * RES, dy * RES]},
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "models_index.csv")))
            if r["segment"] == "residential" and int(r["pano_count"] or 0) >= 15 and r["lat"]]
    random.seed(7)
    random.shuffle(rows)
    done, skipped = [], collections.Counter()
    for row in rows:
        if len(done) >= n or len(done) + sum(skipped.values()) >= n * 3:
            break
        try:
            res = analyse(row["sid"], row)
        except Exception as e:                       # keep going, record why
            res = {"sid": row["sid"], "skip": f"error: {type(e).__name__}: {e}"[:100]}
        if "skip" in res:
            skipped[res["skip"].split(":")[0]] += 1
            print("skip", res["sid"], res["skip"], flush=True)
            continue
        done.append(res)
        print({k: v for k, v in res.items() if not k.startswith("_")}, flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 5
    fig, axes = plt.subplots(math.ceil(len(done) / cols), cols, figsize=(4 * cols, 4 * math.ceil(len(done) / cols)))
    for ax_, res in zip(np.atleast_1d(axes).ravel(), done):
        p = res.pop("_plot")
        ring, inner, c = np.array(p["ring"]), np.array(p["inner"]), np.array(p["centre"])
        t = math.radians(p["th"])
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        fixed = (inner - c) @ R.T + c + np.array(p["d"])
        ax_.fill(ring[:, 0], ring[:, 1], color="#cccccc", label="Vexcel footprint")
        ax_.scatter(inner[:, 0], inner[:, 1], s=2, c="#d62728", label="Matterport as delivered")
        ax_.scatter(fixed[:, 0], fixed[:, 1], s=2, c="#2ca02c", label="after refinement")
        ax_.set_title(f"{res['sid']}  IoU {res['iou_before']:.2f} -> {res['iou_after']:.2f}\n"
                      f"rot {res['fix_rot_deg']:+.1f} deg, shift {res['fix_shift_m']:.1f} m", fontsize=9)
        ax_.set_aspect("equal"); ax_.set_xticks([]); ax_.set_yticks([])
    for ax_ in np.atleast_1d(axes).ravel()[len(done):]:
        ax_.axis("off")
    np.atleast_1d(axes).ravel()[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overlay.png"), dpi=70)
    json.dump({"homes": done, "skipped": dict(skipped)}, open(os.path.join(OUT, "results.json"), "w"), indent=1)

    def med(k):
        return float(np.median([abs(d[k]) for d in done])) if done else float("nan")
    print(f"\nN={len(done)} skipped={dict(skipped)}")
    print(f"median IoU before {med('iou_before'):.3f} -> after {med('iou_after'):.3f}")
    print(f"median containment before {med('contain_before'):.3f} -> after {med('contain_after'):.3f}")
    print(f"median |rotation fix| {med('fix_rot_deg'):.1f} deg, median shift {med('fix_shift_m'):.2f} m")
    print(f"median |heading offset| (walls vs footprint edges) {med('heading_offset_deg'):.2f} deg")


if __name__ == "__main__":
    main()
