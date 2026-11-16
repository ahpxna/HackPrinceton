"""Reject obstacle boxes contradicted by complete, independent depth views.

All coordinates are the installed viewer's Z-up metres. Unknown pixels and
boxes crossing cube-face boundaries provide no free-space evidence.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def free_boxes(folder, rows, captures, depth_cache, inputs):
    rows = np.asarray(rows, dtype=float).reshape(-1, 5)
    positions = np.asarray([c['position'] for c in captures])
    low = np.c_[rows[:, :2] * .1, rows[:, 3]]
    high = np.c_[(rows[:, :2] + 1) * .1, rows[:, 4]]
    corners = np.where(np.asarray(list(itertools.product([0, 1], repeat=3)))[None, :, :],
                       high[:, None, :], low[:, None, :])
    middle = (low + high) / 2
    radius = np.linalg.norm(high - low, axis=1) / 2
    distances = np.linalg.norm(middle[:, None, :] - positions[None, :, :], axis=2)
    nearest = np.argsort(distances, axis=1)[:, :8]
    free = np.zeros((len(rows), len(captures)), dtype=bool)
    support = np.zeros(len(rows), dtype=bool)
    for ci, capture in enumerate(captures):
        indices = np.flatnonzero(np.any(nearest == ci, axis=1) &
                                 (distances[:, ci] >= .3) & (distances[:, ci] <= 8))
        if not len(indices):
            continue
        bases = np.asarray(capture['face_bases_center_right_down'])
        delta = middle[indices] - positions[ci]
        face_ids = np.argmax(delta @ bases[:, 0].T, axis=1)
        for face, (centre, right, down) in enumerate(bases):
            selected = indices[face_ids == face]
            filename = folder / 'capdepth' / capture['id'] / f'face{face}.u16'
            if not len(selected) or not filename.exists():
                continue
            if filename not in depth_cache:
                raw = filename.read_bytes()
                if len(raw) != 256 * 256 * 2:
                    raise ValueError(f'Invalid depth size: {filename}')
                inputs[filename] = hashlib.sha256(raw).hexdigest()
                depth_cache[filename] = np.frombuffer(raw, dtype='<u2').reshape(256, 256) / 1000
            ranges = depth_cache[filename]
            # A plausible midpoint surface vetoes removal even when the full
            # box straddles a face boundary and cannot provide free evidence.
            direction = middle[selected] - positions[ci]
            uv_mid = np.stack([direction @ right, direction @ down], axis=-1) / (direction @ centre)[:, None]
            mid_pixels = np.rint((uv_mid + 1) * 128 - .5).astype(int)
            for dx, dy in itertools.product([-1, 0, 1], repeat=2):
                px, py = mid_pixels[:, 0] + dx, mid_pixels[:, 1] + dy
                inside = (px >= 0) & (py >= 0) & (px < 256) & (py < 256)
                ids = selected[inside]
                values = ranges[py[inside], px[inside]]
                support[ids] |= (values > 0) & (np.abs(values - distances[ids, ci]) <= .2 + radius[ids])
            vectors = corners[selected] - positions[ci]
            front = vectors @ centre
            uv = np.stack([vectors @ right, vectors @ down], axis=-1) / np.maximum(front[..., None], 1e-9)
            valid = np.all(front > .05, axis=1) & np.all(np.abs(uv) < .98, axis=(1, 2))
            pixels = (uv + 1) * 128 - .5
            for local in np.flatnonzero(valid):
                index = selected[local]
                # Expand by a pixel: an edge or a depth hole must not carve a box.
                xmin, ymin = np.floor(pixels[local].min(axis=0)).astype(int) - 1
                xmax, ymax = np.ceil(pixels[local].max(axis=0)).astype(int) + 1
                if xmin < 0 or ymin < 0 or xmax >= 256 or ymax >= 256:
                    continue
                patch = ranges[ymin:ymax + 1, xmin:xmax + 1]
                farthest = np.linalg.norm(vectors[local], axis=1).max()
                if np.all(patch > 0) and patch.min() > farthest + .3 + .08 * farthest:
                    free[index, ci] = True
    witnesses = {}
    for index in np.flatnonzero((free.sum(axis=1) >= 3) & ~support):
        views = np.flatnonzero(free[index])
        # Three different IDs alone are insufficient when centres coincide.
        triple = next((ids for ids in itertools.combinations(views, 3)
                       if all(np.linalg.norm(positions[a] - positions[b]) >= .4
                              for a, b in itertools.combinations(ids, 2))), None)
        if triple is None:
            continue
        witnesses[int(index)] = list(map(int, triple))
    return witnesses


def reconcile(folder, obstacle_data):
    folder = Path(folder)
    metadata_path = folder / 'captures.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('up') != 'Z' or metadata.get('units') != 'metres':
        raise ValueError('Expected installed Z-up metric captures')
    if obstacle_data.get('version') != 1 or obstacle_data.get('step') != .1:
        raise ValueError('Expected version 1, 10 cm obstacle cells')
    captures, cells = metadata['captures'], obstacle_data['cells']
    inputs = {metadata_path: hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
    depth_cache, kept, removed = {}, [], []
    total_slabs, changed_cells = 0, 0
    for start in range(0, len(cells), 2048):
        slabs, owners = [], []
        for index in range(start, min(start + 2048, len(cells))):
            x, y, floor, low, high = cells[index]
            edges = [low] + [round(z / 10, 3) for z in range(math.floor(low * 10) + 1, math.ceil(high * 10))] + [high]
            for bottom, top in zip(edges, edges[1:]):
                slabs.append([x, y, floor, bottom, top])
                owners.append(index)
        witnesses = free_boxes(folder, slabs, captures, depth_cache, inputs)
        total_slabs += len(slabs)
        changed_cells += len({owners[i] for i in witnesses})
        previous_owner = None
        for index, (owner, slab) in enumerate(zip(owners, slabs)):
            if index in witnesses:
                removed.append([owner, slab[3], slab[4], *witnesses[index]])
                previous_owner = None
            elif previous_owner == owner:
                kept[-1][4] = slab[4]
            else:
                kept.append(slab)
                previous_owner = owner
        print(folder.name, 'depth consistency', min(start + 2048, len(cells)), '/', len(cells), flush=True)
    result = {**obstacle_data, 'cells': kept}
    audit = {'algorithm': 'complete-slab-free-space-v2', 'beforeCells': len(cells),
             'slabHeightMetres': .1, 'beforeSlabs': total_slabs,
             'changedCells': changed_cells, 'removedSlabs': len(removed), 'afterCells': len(kept),
             'minimumViews': 3, 'minimumBaselineMetres': .4,
             'rangeMarginMetres': .3, 'relativeRangeMargin': .08,
             'nearestCaptures': 8, 'maximumDistanceMetres': 8,
             'inputsSha256': {str(p.relative_to(folder)): digest for p, digest in sorted(inputs.items())},
             'captureIds': [c['id'] for c in captures],
             'removedColumns': ['originalCellIndex', 'low', 'high', 'captureA', 'captureB', 'captureC'],
             'removed': removed}
    return result, audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    source = args.input or args.folder / 'obstacles.json'
    if source.resolve() == args.output.resolve():
        parser.error('Write a separate candidate first; source must remain available for review')
    result, audit = reconcile(args.folder, json.loads(source.read_text()))
    audit['obstaclesSha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    audit['outputSha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.audit.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('inputsSha256', 'removed', 'captureIds')}))
