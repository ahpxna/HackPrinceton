#!/usr/bin/env python3
"""Review a W34 perception result: verify and unpack it, count per prompt/label, draw contact sheets."""
import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TILE = 320


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unpack(archive: Path, expect: str, dest: Path) -> Path:
    if expect and sha(archive) != expect:
        raise SystemExit(f'{archive} sha256 does not match {expect}')
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter='data')
    return dest / 'hf-perception-v1'


def face_image(faces: Path, cap, k):
    return Image.open(faces / cap / f'face{k}.jpg').convert('RGB').resize((TILE, TILE), Image.BILINEAR)


def stats(values):
    v = np.asarray(values, float)
    return dict(n=len(v), min=round(float(v.min()), 3), median=round(float(np.median(v)), 3),
                max=round(float(v.max()), 3)) if len(v) else dict(n=0)


def sheet(rows, path: Path, cols=4):
    if not rows:
        return
    out = Image.new('RGB', (cols * TILE, ((len(rows) + cols - 1) // cols) * TILE), (20, 20, 20))
    for i, im in enumerate(rows):
        out.paste(im, ((i % cols) * TILE, (i // cols) * TILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=85)


def draw_box(im, box, text, colour):
    d = ImageDraw.Draw(im)
    b = [v * TILE for v in box]
    d.rectangle(b, outline=colour, width=2)
    d.rectangle([b[0], b[1], b[0] + 7 * len(text), b[1] + 12], fill=(0, 0, 0))
    d.text((b[0] + 1, b[1]), text, fill=colour)


def review_sam3(result: Path, faces: Path, report: Path, picks):
    inst = []
    for f in sorted((result / 'sam3').glob('*/face*.json')):
        d = json.loads(f.read_text())
        inst += [dict(i, capture=d['capture'], face=d['face']) for i in d['instances']]
    by = defaultdict(list)
    for i in inst:
        by[i['prompt']].append(i)
    summary = {}
    for prompt, rows in sorted(by.items()):
        summary[prompt] = dict(instances=len(rows), faces=len({(r['capture'], r['face']) for r in rows}),
                               captures=len({r['capture'] for r in rows}), score=stats([r['score'] for r in rows]),
                               area=stats([r['area'] for r in rows]))
        rows = sorted(rows, key=lambda r: -r['score'])
        chosen = rows[:picks - 4] + rows[-4:] if len(rows) > picks else rows      # strongest plus the weakest kept
        tiles = []
        for r in chosen:
            im = face_image(faces, r['capture'], r['face'])
            bits = np.asarray(Image.open(result / 'sam3' / r['capture'] / f'face{r["face"]}.png'))
            m = ((bits >> r['bit']) & 1).astype(np.uint8) * 255
            m = np.asarray(Image.fromarray(m).resize((TILE, TILE), Image.NEAREST)) > 0
            a = np.asarray(im).copy()
            a[m] = (0.5 * a[m] + 0.5 * np.array([255, 0, 160])).astype(np.uint8)
            im = Image.fromarray(a)
            draw_box(im, r['box'], f'{r["score"]:.2f} {r["capture"][:6]} f{r["face"]}', (255, 255, 0))
            tiles.append(im)
        sheet(tiles, report / f'sam3_{prompt.replace(" ", "_")}.jpg')
    return summary


def review_owl(result: Path, faces: Path, report: Path, picks, show=0.2):
    dets = []
    for f in sorted((result / 'owl').glob('*/face*.json')):
        d = json.loads(f.read_text())
        dets += [dict(x, capture=d['capture'], face=d['face']) for x in d['detections']]
    by = defaultdict(list)
    for x in dets:
        by[x['label']].append(x)
    summary = {}
    for label, rows in sorted(by.items()):
        rows = sorted(rows, key=lambda r: -r['score'])
        summary[label] = {f'>={t}': sum(r['score'] >= t for r in rows) for t in (0.1, 0.2, 0.3, 0.5)}
        summary[label]['captures>=0.3'] = len({r['capture'] for r in rows if r['score'] >= 0.3})
        tiles = []
        for r in [r for r in rows if r['score'] >= show][:picks]:
            im = face_image(faces, r['capture'], r['face'])
            draw_box(im, r['box'], f'{r["score"]:.2f} {r["capture"][:6]} f{r["face"]}', (0, 255, 0))
            tiles.append(im)
        sheet(tiles, report / f'owl_{label.replace(" ", "_")}.jpg')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path)
    ap.add_argument('--sha256', default='')
    ap.add_argument('--result', type=Path, help='already unpacked hf-perception-v1 folder')
    ap.add_argument('--unpack-to', type=Path, default=Path('data/w34'))
    ap.add_argument('--faces', type=Path, default=Path('data/splat/web/HFwcH424n9w/captures'))
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--picks', type=int, default=16)
    args = ap.parse_args()
    result = args.result or unpack(args.archive, args.sha256, args.unpack_to)
    out = dict(result=str(result))
    for name, fn in (('sam3', review_sam3), ('owl', review_owl)):
        if (result / name).is_dir():
            out[name] = fn(result, args.faces, args.report, args.picks)
    for name in ('sam3_summary.json', 'owl_summary.json', 'run_status.json'):
        if (result / name).exists():
            out[name] = json.loads((result / name).read_text())
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / 'review.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k in ('sam3', 'owl')}, indent=1))


if __name__ == '__main__':
    main()
