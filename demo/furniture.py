"""Local, ephemeral furniture lookup and the separately installed F2 scan adapter."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from email.parser import BytesParser
from email.policy import default

ROOT = Path(__file__).resolve().parents[1]
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
# Typical W/D/H centimetres, with broad category ranges; never measurements.
PRESETS = [
    ('2-seat sofa', (170, 90, 85), (140, 75, 70), (200, 110, 105)),
    ('3-seat sofa', (213, 97, 84), (190, 80, 70), (250, 115, 105)),
    ('sectional sofa', (280, 180, 85), (230, 140, 70), (360, 240, 110)),
    ('armchair', (85, 85, 90), (65, 65, 65), (115, 110, 120)),
    ('twin bed', (100, 200, 100), (97, 190, 50), (115, 215, 140)),
    ('full bed', (145, 200, 100), (137, 190, 50), (160, 220, 150)),
    ('queen bed', (160, 215, 100), (152, 203, 50), (180, 235, 150)),
    ('king bed', (200, 215, 100), (193, 203, 50), (220, 240, 160)),
    ('dresser', (140, 50, 85), (80, 40, 70), (180, 60, 130)),
    ('nightstand', (50, 40, 55), (30, 30, 40), (70, 55, 80)),
    ('wardrobe', (150, 60, 200), (80, 50, 170), (240, 70, 240)),
    ('4-person dining table', (120, 80, 75), (90, 70, 70), (150, 100, 80)),
    ('6-person dining table', (180, 90, 75), (150, 80, 70), (220, 110, 80)),
    ('desk', (120, 60, 75), (80, 45, 65), (180, 90, 85)),
    ('bookshelf', (90, 30, 180), (45, 20, 90), (150, 45, 240)),
    ('TV stand', (150, 40, 55), (90, 30, 35), (220, 60, 85)),
    ('coffee table', (110, 60, 40), (60, 40, 25), (150, 90, 55)),
    ('5 by 8 foot rug', (152, 244, 1), (145, 235, 0.5), (160, 250, 3)),
    ('8 by 10 foot rug', (244, 305, 1), (235, 295, 0.5), (250, 315, 3)),
    ('upright piano', (150, 60, 120), (140, 50, 100), (165, 75, 140)),
    ('grand piano', (150, 180, 100), (145, 150, 95), (170, 280, 110)),
    ('dining chair', (50, 55, 90), (40, 40, 75), (65, 70, 110)),
    ('ottoman', (70, 55, 40), (40, 40, 30), (120, 90, 55)),
    ('bench', (120, 40, 45), (80, 30, 40), (180, 60, 65)),
    ('recliner', (95, 100, 105), (75, 80, 85), (120, 180, 120)),
    ('television', (120, 25, 75), (70, 15, 45), (190, 45, 120)),
]
_engine = None
_lock = threading.Lock()


def upload(handler, limit):
    size = int(handler.headers.get('Content-Length', '0'))
    kind = handler.headers.get('Content-Type', '')
    if not 0 < size <= limit or not kind.startswith('multipart/form-data;'):
        raise ValueError('Upload a file within the size limit')
    raw = handler.rfile.read(size)
    if len(raw) != size:
        raise ValueError('Incomplete upload')
    message = BytesParser(policy=default).parsebytes(
        b'Content-Type: ' + kind.encode('ascii') + b'\r\nMIME-Version: 1.0\r\n\r\n' + raw)
    fields = {}
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if name in ('file', 'known_dim'):
            fields[name] = part.get_payload(decode=True)
    if not fields.get('file'):
        raise ValueError('Choose a photo or video')
    return fields


def lookup(raw):
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(raw)) as source:
        if source.width * source.height > 25_000_000:
            raise ValueError('Photo exceeds 25 megapixels')
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((1024, 1024))
    small = image.resize((64, 64)).quantize(colors=5)
    index = max(small.getcolors(), key=lambda row: row[0])[1]
    rgb = small.getpalette()[index*3:index*3+3]
    global _engine
    with _lock:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        # Exact existing snapshot; no hub calls, even on a cache miss.
        snapshot = ROOT / 'data/matterport/_feature_model/hub/models--openai--clip-vit-base-patch32/snapshots' / REVISION
        if not snapshot.is_dir():
            raise RuntimeError('Cached photo model unavailable; enter exact size manually')
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            if _engine is None:
                torch.set_num_threads(4)
                model = CLIPModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=False).eval()
                processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
                inputs = processor(text=['a photo of a ' + row[0] for row in PRESETS], return_tensors='pt', padding=True)
                with torch.inference_mode():
                    text = model.get_text_features(**inputs)
                    text = text if isinstance(text, torch.Tensor) else text.pooler_output
                    text = text / text.norm(dim=-1, keepdim=True)
                _engine = model, processor, text
            model, processor, text = _engine
            with torch.inference_mode():
                features = model.get_image_features(**processor(images=image, return_tensors='pt'))
                features = features if isinstance(features, torch.Tensor) else features.pooler_output
                scores = ((features / features.norm(dim=-1, keepdim=True)) @ text.T)[0]
        best = scores.topk(3).indices.tolist()
        return {'categories': [dict(category=PRESETS[i][0], cosine_score=round(scores[i].item(), 5),
                 size_cm=PRESETS[i][1], range_cm={'min': PRESETS[i][2], 'max': PRESETS[i][3]}) for i in best],
                'color': '#%02x%02x%02x' % tuple(rgb), 'model_revision': REVISION,
                'label': 'typical size for this category, enter your exact size'}


def openings(sid):
    if not re.fullmatch(r'[A-Za-z0-9]{8,32}', sid):
        raise ValueError('Invalid home')
    model = json.loads((ROOT / 'data/cache' / sid / 'matterport.json').read_text())['data']['model']
    zones = []
    for floor in model['floors']:
        for edge in floor.get('edges', []):
            vertices = edge.get('vertices', [])
            if len(vertices) != 2:
                continue
            a, b = [[v['position'][k] for k in ('x', 'y')] for v in vertices]
            length = math.dist(a, b)
            if length <= 0:
                continue
            for opening in edge.get('openings', []):
                if opening['type'] == 'window':
                    continue
                center = [a[i] + (b[i]-a[i])*opening['relativeCenter'] for i in range(2)]
                zones.append(dict(floor_id=floor['id'], center=center, width=opening['width'],
                                  tangent=[(b[i]-a[i])/length for i in range(2)]))
    return {'zones': zones, 'available': True, 'clearance_zone_m': 0.9}


def scan(fields):
    python = ROOT / '.venv-scan/bin/python'
    cli = ROOT / 'experiments/furniture_scan/scan.py'
    if not python.is_file() or not cli.is_file():
        return {'error': 'Video scan not installed on this machine'}, 503
    known = fields.get('known_dim', b'').decode('ascii').strip()
    if known and not re.fullmatch(r'width=(?:\d+(?:\.\d*)?|\.\d+)', known):
        raise ValueError('Known dimension must be width in metres')
    if known and not 0 < float(known.split('=')[1]) <= 20:
        raise ValueError('Known width must be between 0 and 20 metres')
    with tempfile.TemporaryDirectory(prefix='furniture-') as temp:
        folder = Path(temp)
        video = folder / 'input.mp4'
        video.write_bytes(fields['file'])
        output = folder / 'out'
        command = [str(python), str(cli), '--input', str(video), '--out', str(output)]
        if known:
            command += ['--known-dim', known]
        try:
            subprocess.run(command, cwd=ROOT, timeout=240, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result = json.loads((output / 'result.json').read_text())
            for key in ('width_m', 'depth_m', 'height_m', 'uncertainty_pct', 'frames_used', 'seconds'):
                value = result[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid scan result')
            if any(not 0 < result[key] <= 20 for key in ('width_m', 'depth_m', 'height_m')):
                raise ValueError('Invalid scan dimensions')
            # preview_jpg names a deleted ephemeral file; never return a local path.
            clean = {k: result[k] for k in ('width_m', 'depth_m', 'height_m', 'anchor', 'uncertainty_pct', 'frames_used', 'seconds')}
            return clean, 200
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return {'error': 'Video scan failed; enter exact size manually'}, 422
