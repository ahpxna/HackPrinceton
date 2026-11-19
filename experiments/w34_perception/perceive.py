#!/usr/bin/env python3
"""SAM 3 masks and OWLv2 boxes on the HF cube faces in one job (W34; runs on a CUDA GPU machine).

stage ``sam3``: for each face, a 16-bit PNG at 2048 px whose bit k marks SAM 3 instances of
    MASK_PROMPTS[k], plus a JSON of instances (prompt, score, normalised box, area share).
stage ``owl``: for each face, OWLv2 boxes for APPLIANCE_QUERIES over the whole face and a 2x2 tile
    grid (small things such as thermostats), merged by per-label NMS.

Boxes are normalised ``[x0, y0, x1, y1]`` on the face, so they hold at any face resolution.
Each face is written atomically, JSON last; a rerun skips faces whose JSON exists.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = 'w34-perception-1'
SIZE = 2048
MASK_PROMPTS = ['person', 'tripod', 'shadow', 'mirror', 'window', 'glass door', 'door',
                'television', 'staircase', 'railing']
MASK_THRESHOLD = 0.4
APPLIANCE_QUERIES = ['refrigerator', 'dishwasher', 'oven', 'stove', 'microwave', 'range hood',
                     'washing machine', 'clothes dryer', 'ceiling fan', 'fireplace', 'air conditioner',
                     'water heater', 'bathtub', 'shower', 'toilet', 'sink', 'kitchen island', 'wine cooler',
                     'television', 'smoke detector', 'thermostat', 'chandelier', 'swimming pool', 'hot tub',
                     'solar panel', 'garage door', 'bed', 'sofa']
OWL_MODEL = 'google/owlv2-base-patch16-ensemble'
OWL_THRESHOLD = 0.10
OWL_KEEP = 15


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def list_faces(root: Path):
    return [(cap.name, k, cap / f'face{k}.jpg') for cap in sorted(p for p in root.iterdir() if p.is_dir())
            for k in range(6) if (cap / f'face{k}.jpg').exists()]


def load(path: Path) -> Image.Image:
    im = Image.open(path)
    im.draft('RGB', (SIZE, SIZE))                                  # JPEG DCT scaling: 4096 decodes at 2048
    im = im.convert('RGB')
    return im if im.size == (SIZE, SIZE) else im.resize((SIZE, SIZE), Image.BICUBIC)


def atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def to_np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def auth_problem(e: BaseException) -> bool:
    s = f'{type(e).__name__}: {e}'
    return any(k in s for k in ('GatedRepo', 'gated', '401', '403', 'Access to model', 'restricted',
                                'not have access', 'LocalEntryNotFoundError'))


class Progress:
    def __init__(self, name, total):
        self.name, self.total, self.done, self.t0 = name, total, 0, time.time()

    def step(self):
        self.done += 1
        if self.done % 25 == 0 or self.done == self.total:
            rate = self.done / max(time.time() - self.t0, 1e-6)
            log(f'{self.name}: {self.done}/{self.total} faces, {rate * 60:.1f}/min, '
                f'eta {(self.total - self.done) / max(rate, 1e-6) / 60:.0f} min')


# ----------------------------------------------------------------------- SAM 3

def sam3_backend(device):
    """The image API under bf16 autocast (how SAM 3's own examples run it); the video API as fallback.

    CAROECT-D crashed with BFloat16/Float matmuls because it disabled autocast around set_image.
    """
    import torch
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        proc = Sam3Processor(build_sam3_image_model(device=device), device=device,
                             confidence_threshold=MASK_THRESHOLD)

        def segment(im):                                            # set_image has no inference_mode of its own
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                state = proc.set_image(im)
                out = []
                for k, prompt in enumerate(MASK_PROMPTS):
                    proc.reset_all_prompts(state)
                    o = proc.set_text_prompt(prompt=prompt, state=state)
                    out += [(k, float(s), m[0] > 0.5) for s, m in zip(to_np(o['scores']), to_np(o['masks']))]
            return out
        segment(Image.new('RGB', (SIZE, SIZE), (128, 128, 128)))   # surfaces dtype errors before the loop
        return 'image', segment
    except Exception as e:                                          # noqa: BLE001 - reported, then fallback
        if auth_problem(e):
            raise
        log(f'sam3 image API failed ({type(e).__name__}: {str(e)[:200]}); trying the video API')
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor()
    tmp = Path(os.environ.get('TMPDIR', '/tmp')) / f'w34_sam3_{os.getpid()}.jpg'

    def segment(im):
        im.save(tmp, quality=95)
        sid = predictor.handle_request(request=dict(type='start_session', resource_path=str(tmp)))['session_id']
        out = []
        try:
            for k, prompt in enumerate(MASK_PROMPTS):
                o = predictor.handle_request(request=dict(type='add_prompt', session_id=sid,
                                                          frame_index=0, text=prompt))['outputs']
                for s, m in zip(to_np(o['out_probs']), to_np(o['out_binary_masks'])):
                    if s >= MASK_THRESHOLD:
                        out.append((k, float(s), m > 0.5))
                predictor.handle_request(request=dict(type='reset_session', session_id=sid))
        finally:
            predictor.handle_request(request=dict(type='close_session', session_id=sid))
        return out
    return 'video', segment


def run_sam3(items, out: Path, device):
    todo = [x for x in items if not (out / 'sam3' / x[0] / f'face{x[1]}.json').exists()]
    log(f'sam3: {len(items)} faces, {len(todo)} to do')
    if not todo:
        return dict(stage='sam3', status='complete', faces=len(items))
    try:
        backend, segment = sam3_backend(device)
    except Exception as e:                                          # noqa: BLE001
        if auth_problem(e):
            log('SAM 3 checkpoint facebook/sam3 is gated on Hugging Face and this machine cannot download it.')
            log('Fix on the GPU machine (your own account): accept the licence at https://huggingface.co/facebook/sam3,')
            log('run `hf auth login` (or `huggingface-cli login`), then run the same command again.')
            sys.exit(3)
        raise
    log(f'sam3 backend: {backend}')
    prog, counts = Progress('sam3', len(todo)), {p: 0 for p in MASK_PROMPTS}
    for cap, k, path in todo:
        bits, inst = np.zeros((SIZE, SIZE), np.uint16), []
        for bit, score, mask in segment(load(path)):
            if mask.shape != (SIZE, SIZE):
                mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.NEAREST)) > 127
            if not mask.any():
                continue
            bits[mask] |= np.uint16(1 << bit)
            ys, xs = np.nonzero(mask)
            inst.append(dict(prompt=MASK_PROMPTS[bit], bit=bit, score=round(score, 4), area=round(float(mask.mean()), 6),
                             box=[round(xs.min() / SIZE, 5), round(ys.min() / SIZE, 5),
                                  round((xs.max() + 1) / SIZE, 5), round((ys.max() + 1) / SIZE, 5)]))
            counts[MASK_PROMPTS[bit]] += 1
        buf = io.BytesIO()
        Image.fromarray(bits).save(buf, 'PNG')
        atomic(out / 'sam3' / cap / f'face{k}.png', buf.getvalue())
        atomic(out / 'sam3' / cap / f'face{k}.json', json.dumps(dict(
            capture=cap, face=k, size=SIZE, backend=backend, threshold=MASK_THRESHOLD,
            bits={p: i for i, p in enumerate(MASK_PROMPTS)}, instances=inst)).encode())
        prog.step()
    return dict(stage='sam3', status='complete', backend=backend, faces=len(items), new=len(todo), instances=counts)


# ----------------------------------------------------------------------- OWLv2

def nms(boxes, scores, iou=0.5):
    order, keep = np.argsort(-scores), []
    while len(order):
        i = order[0]
        keep.append(i)
        b, r = boxes[i], boxes[order[1:]]
        w = np.clip(np.minimum(b[2], r[:, 2]) - np.maximum(b[0], r[:, 0]), 0, None)
        h = np.clip(np.minimum(b[3], r[:, 3]) - np.maximum(b[1], r[:, 1]), 0, None)
        inter = w * h
        union = (b[2] - b[0]) * (b[3] - b[1]) + (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) - inter
        order = order[1:][inter / np.maximum(union, 1e-9) <= iou]
    return keep


def run_owl(items, out: Path, device):
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    todo = [x for x in items if not (out / 'owl' / x[0] / f'face{x[1]}.json').exists()]
    log(f'owl: {len(items)} faces, {len(todo)} to do')
    if not todo:
        return dict(stage='owl', status='complete', faces=len(items))
    proc = Owlv2Processor.from_pretrained(OWL_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(device).eval()
    if device == 'cuda':
        model = model.half()
    half = SIZE // 2
    offsets = [(0.0, 0.0, 1.0)] + [(x / SIZE, y / SIZE, 0.5) for y in (0, half) for x in (0, half)]
    prog, counts = Progress('owl', len(todo)), {q: 0 for q in APPLIANCE_QUERIES}
    for cap, k, path in todo:
        im = load(path)
        crops = [im] + [im.crop((x, y, x + half, y + half)) for y in (0, half) for x in (0, half)]
        inputs = proc(text=[APPLIANCE_QUERIES] * len(crops), images=crops, return_tensors='pt')
        inputs = {n: v.to(device) for n, v in inputs.items()}
        if device == 'cuda':
            inputs['pixel_values'] = inputs['pixel_values'].half()
        with torch.inference_mode():
            o = model(**inputs)
        probs, boxes = to_np(o.logits.sigmoid()), to_np(o.pred_boxes)        # (B, P, Q), (B, P, 4) cxcywh
        rows = []
        for b, (ox, oy, sc) in enumerate(offsets):
            label, score = probs[b].argmax(1), probs[b].max(1)
            for p in np.flatnonzero(score >= OWL_THRESHOLD):
                cx, cy, w, h = boxes[b, p]
                x0, y0, x1, y1 = np.clip([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0, 1)
                rows.append((int(label[p]), float(score[p]), [ox + x0 * sc, oy + y0 * sc, ox + x1 * sc, oy + y1 * sc], b))
        dets = []
        for q in sorted({r[0] for r in rows}):
            sel = [r for r in rows if r[0] == q]
            bx, sc_ = np.array([r[2] for r in sel]), np.array([r[1] for r in sel])
            for i in nms(bx, sc_)[:OWL_KEEP]:
                dets.append(dict(label=APPLIANCE_QUERIES[q], score=round(float(sc_[i]), 4),
                                 box=[round(float(v), 5) for v in bx[i]], view='full' if sel[i][3] == 0 else f'tile{sel[i][3]}'))
                counts[APPLIANCE_QUERIES[q]] += 1
        atomic(out / 'owl' / cap / f'face{k}.json', json.dumps(dict(
            capture=cap, face=k, model=OWL_MODEL, threshold=OWL_THRESHOLD, detections=dets)).encode())
        prog.step()
    return dict(stage='owl', status='complete', faces=len(items), new=len(todo), detections_at_0_1=counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage', choices=['sam3', 'owl'], required=True)
    ap.add_argument('--faces', required=True, help='folder of <capture_id>/face{0..5}.jpg')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='first N faces only (smoke test)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        log('CUDA is not available to this Python; stopping.')
        sys.exit(4)
    items = list_faces(Path(args.faces))
    if args.limit:
        items = items[:args.limit]
    out = Path(args.out)
    t0 = time.time()
    summary = (run_sam3 if args.stage == 'sam3' else run_owl)(items, out, args.device)
    summary.update(version=VERSION, seconds=round(time.time() - t0, 1), python=sys.version.split()[0],
                   torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    atomic(out / f'{args.stage}_summary.json', json.dumps(summary, indent=1).encode())
    log(json.dumps(summary))


if __name__ == '__main__':
    main()
