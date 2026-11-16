"""Round 2: official public checkpoints, cached photographs only.

Run with .venv-agent/bin/python -I experiments/sun_patch_round2.py --model MODEL
Models: clipseg, dino_sam2, silt; then --score and --gallery.
Acquisition receipts and pinned revisions live in data/sun_patch/round2/.
"""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/sun_patch/round2'
PROGRESS = Path('/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T21-09-57-516Z-20198-b4e5abd3-f1cc-4c82-964d-c3e2ae4ce2a3.md')
spec = importlib.util.spec_from_file_location('sp_round1', ROOT/'experiments/sun_patch_consistency.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
V = S.V
PROMPT = 'patch of sunlight on the floor'
MODELS = ['dino_sam2', 'clipseg', 'silt']
PARAMS = {'prompt': PROMPT, 'dino_box_threshold': .25, 'dino_text_threshold': .25,
          'clipseg_probability_threshold': .5, 'silt_shadow_threshold': .7,
          'silt_crop': False, 'silt_filter_threshold': .1, 'face_size': 512,
          'min_floor_pixels': 60, 'max_floor_fraction': .25,
          'recall_min_coverage': .5, 'scale_gate_hits': 3, 'threads': 4}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def progress(done, next_step):
    PROGRESS.write_text('# Objective\nRound 2 public detectors, fixed 15-home pilot / four retained marks; scale only >=3/4.\n\n# Done\n'+done+'\n\n# Next\n'+next_step+'\n')


def inventory():
    rows = []
    for home in V.read(V.OUT/'selection.json')['rows'][:15]:
        for path in sorted((V.OUT/home['sid']).glob('*_projection.json')):
            rows.append((home['sid'], path.name.removesuffix('_projection.json'), V.read(path)))
    reference = {(m['sid'], m['pano_id']) for m in V.read(V.OUT/'marks_v1.json')['patches']}
    return sorted(rows, key=lambda r: ((r[0], r[1]) not in reference, r[0], r[1]))


def load_detector(name, device='cpu'):
    from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                              Sam2Processor, Sam2Model, CLIPSegProcessor,
                              CLIPSegForImageSegmentation)
    weights = OUT/'models'
    if name == 'clipseg':
        path = weights/'clipseg-rd64-refined'
        processor = CLIPSegProcessor.from_pretrained(path, local_files_only=True)
        model = CLIPSegForImageSegmentation.from_pretrained(path, local_files_only=True).eval()

        def detect(image):
            inputs = processor(text=[PROMPT], images=[image], return_tensors='pt', padding=True)
            logits = model(**inputs).logits.reshape(1, 1, 352, 352)
            prob = F.interpolate(logits, size=(512,512), mode='bilinear', align_corners=False).sigmoid()[0,0]
            return prob.numpy(), {'threshold': .5}
        return detect
    if name == 'dino_sam2':
        path = weights/'grounding-dino-tiny'
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).eval()
        sp = Sam2Processor.from_pretrained(weights/'sam2-hiera-tiny', local_files_only=True)
        sam = Sam2Model.from_pretrained(weights/'sam2-hiera-tiny', local_files_only=True).eval()

        def detect(image):
            inputs = processor(images=image, text=PROMPT+'.', return_tensors='pt')
            result = processor.post_process_grounded_object_detection(
                model(**inputs), inputs.input_ids, threshold=.25, text_threshold=.25,
                target_sizes=[(image.height,image.width)])[0]
            boxes = result['boxes'].tolist()
            mask = np.zeros((512,512),np.float32)
            qualities = []
            if boxes:
                sam_inputs = sp(images=image, input_boxes=[boxes], return_tensors='pt')
                output = sam(**sam_inputs, multimask_output=True)
                masks = sp.post_process_masks(output.pred_masks, sam_inputs['original_sizes'])[0]
                for i, scores in enumerate(output.iou_scores[0]):
                    best = int(scores.argmax())
                    mask = np.maximum(mask, masks[i,best].numpy().astype(np.float32))
                    qualities.append(float(scores[best]))
            return mask, {'boxes_xyxy': boxes, 'scores': result['scores'].tolist(),
                          'labels': result.get('text_labels',result.get('labels')), 'sam_iou_scores': qualities}
        return detect
    if name == 'silt':
        vendor = OUT/'vendor/hanyangclarence-SILT-c888520'
        sys.path.insert(0, str(vendor))
        from model.network import Network
        # Official author-linked checkpoint contains OmegaConf config objects.
        checkpoint = torch.load(weights/'PVTb5_SBU.pth', map_location='cpu', weights_only=False)
        cfg = checkpoint['configs']['training_config']
        model = Network(input_resolution=cfg['data']['resolution'], backbone=cfg['backbone']).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(device)
        from torchvision.transforms import ToTensor

        def detect(image):
            x = ToTensor()(image).unsqueeze(0).to(device)
            # Official infer.py's crop=False path; full 512px image, no normalization.
            shadow = model(x)[0,0]
            return shadow.cpu().numpy(), {'kind': 'shadow_probability', 'threshold': .7,
                                    'strict_state_dict': True, 'official_crop': False}
        return detect
    raise ValueError(name)


def run_model(name, limit=None):
    torch.set_num_threads(PARAMS['threads'])
    torch.manual_seed(20261114)
    socket.socket = S.blocked
    socket.create_connection = S.blocked
    detector = load_detector(name)
    rows = inventory()
    if limit: rows = rows[:limit]
    started = time.monotonic()
    for pano_index, (sid, pid, projection) in enumerate(rows):
        folder = OUT/name/sid/pid
        folder.mkdir(parents=True,exist_ok=True)
        for face in range(6):
            metadata_path = folder/f'face{face}.json'
            if metadata_path.exists(): continue
            path = V.CACHE/sid/pid/f'face{face}.jpg'
            image = Image.open(path).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
            tick = time.monotonic()
            with torch.inference_mode(): prob, meta = detector(image)
            # Raw probability is retained; shadow complement is only a lit-region candidate.
            binary = prob <= .7 if name == 'silt' else prob >= .5
            Image.fromarray(np.rint(np.clip(prob,0,1)*65535).astype(np.uint16)).save(folder/f'face{face}_prob.png')
            Image.fromarray(binary.astype(np.uint8)*255).save(folder/f'face{face}_mask.png')
            write(metadata_path, {**meta, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'seconds': time.monotonic()-tick, 'params': PARAMS,
                                  'probability_kind': 'shadow' if name=='silt' else 'sunlight',
                                  'mask_fraction': float(binary.mean())})
            print(json.dumps({'model':name, 'pano':pano_index+1, 'panos':len(rows), 'face':face,
                              'seconds':round(time.monotonic()-tick,2)}),flush=True)
        progress(f'{name}: {pano_index+1}/{len(rows)} pilot panoramas complete; resumable per-face masks/probabilities and inference receipts in data/sun_patch/round2/{name}/. Elapsed {time.monotonic()-started:.0f}s.',
                 'Finish all three detectors; score same four marks; no scale before >=3/4; export comparison and report.')


def candidates(name):
    output = []
    rejected = []
    for sid,pid,p in inventory():
        folder = OUT/name/sid/pid
        if not all((folder/f'face{f}_mask.png').exists() for f in range(6)): continue
        V.A.BASES = np.array(p['basis'])
        t = p['transform']; n=t['size']
        xx,yy=np.meshgrid(np.linspace(t['lo'][0],t['hi'][0],n),np.linspace(t['hi'][1],t['lo'][1],n))
        points=np.stack([xx,yy,np.full_like(xx,t['floor_z_m'])],-1)
        position=np.array([p['pose']['position'][k] for k in ('x','y','z')])
        faces=[Image.open(folder/f'face{f}_mask.png').convert('RGB') for f in range(6)]
        raw=np.asarray(V.sample_image(points-position,p['pose'],faces))[:,:,0]>127
        room=S.room_mask(sid,p['candidate']['room_id'],t)
        Image.fromarray(raw.astype(np.uint8)*255).save(folder/'floor_raw.png')
        count, labels, stats, _=cv2.connectedComponentsWithStats((raw&room).astype(np.uint8),8)
        for label in range(1,count):
            area=int(stats[label,cv2.CC_STAT_AREA]);fraction=float(area/max(1,room.sum()))
            if area<60 or fraction>.25:
                rejected.append({'sid':sid,'pano_id':pid,'label':label,'area_pixels':area,'floor_fraction':fraction})
                continue
            mask=labels==label
            path=folder/f'patch{label:03d}.png'
            Image.fromarray(mask.astype(np.uint8)*255).save(path)
            output.append({'sid':sid,'pano_id':pid,'patch_id':f'{name}/{sid}/{pid}/{label:03d}',
                           'mask_path':str(path.relative_to(ROOT)), 'mask':mask,'area_pixels':area,
                           'floor_fraction':fraction})
    return output,rejected


def score():
    result={'parameters':PARAMS,'models':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in MODELS:
        found,rejected=candidates(name)
        rows=S.reference_recall(found)
        # Report pre-filter coverage too, so rejection is distinguishable from model failure.
        for row,mark in zip(rows,V.read(V.OUT/'marks_v1.json')['patches']):
            raw_path=OUT/name/mark['sid']/mark['pano_id']/'floor_raw.png'
            row['inference_complete']=raw_path.exists()
            if raw_path.exists():
                polygon=np.zeros((800,800),np.uint8)
                cv2.fillPoly(polygon,[np.array(mark['visible_outline'],np.int32)],1)
                raw=np.array(Image.open(raw_path))>127
                row['raw_union_polygon_coverage']=float(np.count_nonzero(raw&(polygon>0))/max(1,polygon.sum()))
        retained=[r for r in rows if r['retained']]
        completed=sum(all((OUT/name/sid/pid/f'face{f}.json').exists() for f in range(6)) for sid,pid,p in inventory())
        complete=completed==45
        hits=sum(r['hit'] for r in retained)
        result['models'][name]={'panos_complete':completed,'pilot_complete':complete,'reference_recall':rows,
                               'retained_hits':hits,'retained_total':len(retained),
                               'raw_union_hits':sum(r.get('raw_union_polygon_coverage',0)>=.5 for r in retained),
                               'gate_passed': complete and hits>=3 and len(retained)==4,
                               'candidates':[{k:v for k,v in c.items() if k!='mask'} for c in found],
                               'rejected_components':rejected}
        print(name,completed,'panos',hits,'/4',flush=True)
    write(OUT/'scores.json',result)
    return result


def report():
    result=V.read(OUT/'scores.json')
    provenance=V.read(OUT/'provenance.json')
    result['provenance']=provenance
    result['scale_eligible_models']=[name for name,v in result['models'].items() if v['gate_passed']]
    result['precision']=None
    result['scope']='Detector evaluation only; round-1 geometry rows are baseline, not recomputed learned-detector evidence.'
    result['comparison']='data/sun_patch/auto/compare/index.html'
    scale_folder=S.AUTO/'round2_scale'
    if (scale_folder/'run_state.json').exists():
        result['scale_status']=V.read(scale_folder/'run_state.json')
    if (scale_folder/'results.json').exists():
        result['scale']=V.read(scale_folder/'results.json')
        result['scope']='Pilot detector evaluation and separate cache-only SILT scale geometry/null experiment.'
    base=V.read(OUT/'checkpoint/sun_patch_consistency.json')
    base['round2']=result
    write(S.EVAL/'sun_patch_consistency.json',base)
    names=['Grounding DINO tiny IDEA-Research/grounding-dino-tiny@a2bb814d + SAM 2 tiny facebook/sam2-hiera-tiny@7c218bea',
           'CLIPSeg CIDAS/clipseg-rd64-refined@999e0328', 'SILT hanyangclarence/SILT@c888520e (PVTb5_SBU.pth)']
    lines=['# Round 2 | '+' | '.join(names),
           'Official checkpoints/repo revisions and SHA256: `data/sun_patch/round2/provenance.json`; torch '+provenance['versions']['torch']+', transformers '+provenance['versions']['transformers']+'.',
           'Public sources: [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny), [SAM 2](https://huggingface.co/facebook/sam2-hiera-tiny), [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined), [SILT](https://github.com/hanyangclarence/SILT).',
           'SILT checkpoint: [official README-linked public Drive](https://drive.google.com/file/d/1CvO6xoXdUw72xGFyhHfroi4LjGyEjBKD/view); downloaded without login; strict state_dict load matched all 1,108 keys.',
           'SDDNet alternative not run: SILT downloaded successfully. SAM 3 remains gated; no workaround.',
           'CPU inference; same cached 15 homes / 45 panos / 270 faces per model. No .env reads or Matterport/Vexcel calls; socket guard during inference.',
           'Prompt: "patch of sunlight on the floor" (DINO adds terminal period); DINO box/text thresholds 0.25/0.25; highest predicted-IoU SAM 2 mask per box.',
           'CLIPSeg sigmoid threshold 0.5. SILT official crop=False path, RGB [0,1], 512px, shadow threshold 0.7; lit candidate = shadow probability <=0.7.',
           'Five-crop SILT smoke outputs preserved separately; not mixed into scored full-image run. No label-based prompt/threshold tuning.',
           'Cube masks use frozen basis/pose projection. Room erosion 10cm; connected components retained at 60px..25% of room, as round-1 area limits; no luminance/contrast/straight-edge gate.',
           'Recall unchanged: >=50% visible-outline polygon coverage OR all marked corners inside ONE retained component. Four retained marks; H2_DOWN excluded as before.',
           '| Model | Pilot complete | Candidate recall | Raw union coverage >=50% | Scale gate >=3/4 |', '|---|---:|---:|---:|---|']
    for name,v in result['models'].items():
        lines.append(f'| {name} | {v["panos_complete"]}/45 panos | {v["retained_hits"]}/4 | {v["raw_union_hits"]}/4 | {"PASS" if v["gate_passed"] else "FAIL" if v["pilot_complete"] else "INCOMPLETE"} |')
    lines += ['| Hand mark | DINO+SAM2 coverage | CLIPSeg coverage | SILT coverage |', '|---|---:|---:|---:|']
    for i,mark in enumerate(V.read(V.OUT/'marks_v1.json')['patches']):
        cols=[]
        for name in MODELS:
            row=result['models'][name]['reference_recall'][i]
            cols.append(f'{row["max_polygon_coverage"]:.3f} ({"hit" if row["hit"] else "miss"})')
        lines.append('| '+mark['patch_id']+(' (excluded)' if not result['models'][MODELS[0]]['reference_recall'][i]['retained'] else '')+' | '+' | '.join(cols)+' |')
    lines += ['Per-patch hits, raw-union coverage, retained/rejected area components and paths: `data/sun_patch/round2/scores.json`; raw 16-bit probabilities and binary masks alongside inference receipts.',
              'Raw union coverage counts broad/disconnected masks too; it does not establish localized patch recall. DINO often covers the floor; its marked regions fail the unchanged component-area limit. Scale gate uses candidate recall.',
              'Scale-eligible models: '+(', '.join(result['scale_eligible_models']) or 'none')+'. Scale execution is reported separately; recall is not precision.',
              'SILT finds three marks in one bedroom (H1); the retained H2 mark is missed: effectively 1/2 retained homes. Its non-shadow complement can include furniture/diffuse light; human precision remains null.',
              'Comparison: `data/sun_patch/auto/compare/index.html`; 270 seven-column Round-2 JPGs, original Round-1 JPGs preserved. Yellow frozen outlines; raw neural masks before floor/area filtering.',
              'Human review priority: six hand-marked faces listed in `data/sun_patch/auto/compare/manifest_round2.json`, then unlabelled homes; no false-positive count inferred from missing marks.',
              'Existing guard suite: 7/7 PASS (`data/sun_patch/round2/existing_tests.log`); real inference receipts are the runtime evidence. No new tests.',
              'Reproduce: `.venv-agent/bin/python -I experiments/sun_patch_round2.py --model clipseg` (also `dino_sam2`, `silt`); then `--score --gallery --report`.',
              'Source authored/reviewed by Codex: `experiments/sun_patch_round2.py`; source SHA256 in Round-2 JSON. Models loaded locally; completed face receipts make reruns resumable.',
              'Undo: reports/index before-copies in `data/sun_patch/round2/checkpoint/`; source ledger: `python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path experiments/sun_patch_round2.py`, then `rollback <id>` dry run / `--yes`.',
              '## Round 1 (preserved baseline result)',
              'SP-luminance-edge-v1 / OpenCV 5.0.0, no trained weights: 15 homes / 45 panos, 339 candidates, retained recall 0/4; SAM 3 HTTP401; no trained shadow model ran.',
              'Geometry baseline only: 90/339 IoU>=0.3; 38 in annual band + cached prism horizon; mean 1,000-north-rotation null chance 0.25076. All 15 homes well registered; no poor group.',
              'Full DSM unavailable; cached neighbour/tree prisms omit own roof. 1deg horizon grid; directions fitted independently of solar band. Furniture/occlusion remain unmodelled.',
              'Round-1 JSON rows retained in same report JSON; exact original MD/JSON and source provenance preserved under `round2/checkpoint/`. Original 50-candidate sheet unchanged; human count pending.',
              'Interpretation only: direction implied by a detected patch is consistent with the possible sun path. No sun-minutes validation; Round 2 does not establish geometry validity.']
    if 'scale' in result:
        scale=result['scale']
        lines.insert(-6,f'SILT scale complete: 169 homes / 1,014 panoramas / 5,070 faces; {scale["candidates"]} candidates / {scale["iou_accepted"]} IoU-accepted. Details and well/poor null split: `data/sun_patch/auto/round2_scale/results.json`.')
        lines.insert(-6,'Scale human sheet: `data/claims/graph/sun_patch_auto_sheet.jpg`; 100-face gallery: `data/sun_patch/auto/round2_scale/gallery/index.html`; precision pending.')
    elif 'scale_status' in result:
        lines.insert(-6,'SILT scale RUNNING: '+result['scale_status']['phase']+'; 169 homes / 1,014 cached panoramas / 5,070 non-ceiling faces. Per-face status: `data/sun_patch/auto/round2_scale/inference_progress.json`; resumable command: `.venv-agent/bin/python -I experiments/sun_patch_round2_scale.py --run`.')
    assert len(lines)<=50
    (S.EVAL/'sun_patch_consistency.md').write_text('\n'.join(lines)+'\n')
    progress('Round-2 report/JSON updated alongside preserved Round-1 results. '+ '; '.join(f'{n}: {v["retained_hits"]}/4, {v["panos_complete"]}/45 panos' for n,v in result['models'].items()),
             'Finish any remaining inference/scale work, inspect gallery and finalize evidence.')


def gallery():
    folder=S.AUTO/'compare'
    entries=V.read(folder/'manifest.json')
    colors={'dino_sam2':np.array([255,75,40]),'clipseg':np.array([20,150,255]),'silt':np.array([190,65,255])}
    font=ImageFont.load_default(size=14)
    for row in entries:
        sid,pid,face=row['sid'],row['pano_id'],row['face']
        original=Image.open(V.CACHE/sid/pid/f'face{face}.jpg').convert('RGB').resize((384,384))
        # Preserve the complete round-1 comparison JPEG, including frozen hand outlines.
        old=Image.open(folder/row['file']).convert('RGB')
        canvas=Image.new('RGB',(7*384,444),'white')
        canvas.paste(old,(0,0))
        projection=V.read(V.OUT/sid/f'{pid}_projection.json')
        V.A.BASES=np.array(projection['basis'])
        position=np.array([projection['pose']['position'][k] for k in ('x','y','z')])
        rotation=V.A.rotation(projection['pose']['rotation'])
        segments=[]
        for mark in V.read(V.OUT/'marks_v1.json')['patches']:
            if (mark['sid'],mark['pano_id'])!=(sid,pid):continue
            outline=mark['visible_outline']
            edges=list(zip(outline,outline[1:]))
            if mark['outline_closed']:edges.append((outline[-1],outline[0]))
            for a,b in edges:
                points=np.array([V.floor_point(uv,projection['transform']) for uv in np.linspace(a,b,150)])
                face_ids,uv=V.A.cube_coordinates(points-position,rotation)
                for j in range(len(points)-1):
                    if face_ids[j]==face and face_ids[j+1]==face:
                        segments.append((tuple(uv[j]*384),tuple(uv[j+1]*384)))
        for index,name in enumerate(MODELS,4):
            path=OUT/name/sid/pid/f'face{face}_mask.png'
            panel=np.array(original).copy()
            status='NOT RUN'
            if path.exists():
                mask=np.array(Image.open(path).resize((384,384),Image.Resampling.NEAREST))>127
                panel[mask]=(.55*panel[mask]+.45*colors[name]).astype(np.uint8)
                status='lit complement' if name=='silt' else 'sunlight mask'
            panel=Image.fromarray(panel)
            for segment in segments:ImageDraw.Draw(panel).line(segment,fill='yellow',width=3)
            canvas.paste(panel,(index*384,60))
            ImageDraw.Draw(canvas).text((index*384+5,5),f'{name}: {status}',fill='black',font=font)
        target=folder/(Path(row['file']).stem+'_round2.jpg')
        canvas.save(target,quality=88)
        row['round2_file']=target.name
    write(folder/'manifest_round2.json',entries)
    result=V.read(OUT/'scores.json')
    summary='; '.join(f'{name}: {v["retained_hits"]}/4 ({v["panos_complete"]}/45 panos)' for name,v in result['models'].items())
    cards=''.join(f'<figure><a href="{html.escape(r["round2_file"])}"><img loading="lazy" src="{html.escape(r["round2_file"])}"></a><figcaption>{html.escape(r["file"])}{" — HAND MARK" if r["hand_marked"] else ""}</figcaption></figure>' for r in entries)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SP rounds 1–2</title><style>body{font:14px sans-serif;background:white;color:black}figure{margin:24px 0}img{width:2688px;max-width:none}figure{overflow-x:auto}figcaption{overflow-wrap:anywhere;position:sticky;left:0}</style><h1>Same 15-home pilot: rounds 1–2</h1><p>'+html.escape(summary)+'</p><p>Columns: original | SAM 3 (NOT RUN: gated) | round-1 shadow (NOT RUN) | raw luminance baseline | Grounding DINO + SAM 2 | CLIPSeg | SILT non-shadow complement. Yellow: frozen visible hand outline. Coloured face masks are raw, before room/area filtering; scores use floor candidates (60 pixels to 25% of room). SILT complement is not proof of direct sunlight. Marked faces first. Scroll each row horizontally. Human precision remains pending.</p>'+cards)
    progress('Three-model comparison exported for '+str(len(entries))+' faces to data/sun_patch/auto/compare/index.html. '+summary,
             'Update same report/JSON; apply >=3/4 scale gate; run existing guard suite and inspect rendered marked panels.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--limit-panos',type=int)
    parser.add_argument('--score',action='store_true')
    parser.add_argument('--gallery',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.model:run_model(args.model,args.limit_panos)
    if args.score:score()
    if args.gallery:gallery()
    if args.report:report()
