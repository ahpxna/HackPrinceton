"""Resumable Model API acquisition and local CLIP crop classification.

Never persists Model API URLs. Scores are cosine similarities, not probabilities.
"""
from __future__ import annotations
import argparse
from collections import Counter
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import re
import random
from pathlib import Path
import shutil
import sys
import sysconfig
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont

GRAPH = ROOT / 'data/claims/graph'
OUT = GRAPH / 'panorama_features'
MODEL = 'openai/clip-vit-base-patch32'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
BASELINE_FREE = 7667765248
VOCAB = ('kitchen island', 'fireplace', 'exposed beams or rafters', 'skylight',
         'vaulted ceiling', 'hardwood floor', 'carpet', 'tile floor', 'ceiling fan',
         'built-in shelves', 'window seat', 'french doors', 'pool', 'patio')
_disk_cache = (0, 0)


def task_disk_bytes():
    """Conservative allocation bound; includes preexisting graph/dependencies too.

    Other active sessions write this volume, so free-space delta is not task usage.
    """
    global _disk_cache
    if time.monotonic()-_disk_cache[0]<20:return _disk_cache[1]
    site=Path(sysconfig.get_paths()['purelib'])
    packages=('torch','torchgen','functorch','transformers','huggingface_hub','safetensors','tokenizers',
              'sympy','mpmath','networkx','fsspec','hf_xet','typer','rich','markdown_it','mdurl',
              'shellingham','filelock','requests','regex','packaging','yaml','tqdm','numpy','PIL')
    roots=[GRAPH,ROOT/'data/matterport/_feature_model']+[p for p in (ROOT/'data/matterport').glob('*/panorama_features')]
    roots.extend(p for p in site.iterdir() if any(p.name==name or p.name.startswith(name+'-') for name in packages))
    total=0;seen=set()
    for root in roots:
        for file in root.rglob('*') if root.is_dir() else [root]:
            try:
                stat=file.lstat();key=(stat.st_dev,stat.st_ino)
                if key not in seen:total+=stat.st_blocks*512;seen.add(key)
            except FileNotFoundError:continue
    _disk_cache=(time.monotonic(),total)
    return total


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def guard():
    free = shutil.disk_usage(ROOT).free
    if free < 1024**3: raise RuntimeError('STOP: free disk below 1 GiB')
    if task_disk_bytes() > 3_000_000_000:
        raise RuntimeError('STOP: attributable allocation exceeds 3 GB budget')
    return free


def helper():
    spec = importlib.util.spec_from_file_location('existing_pano_api', ROOT/'experiments/sun_from_pano.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def homes():
    index = json.loads((ROOT/'demo/site/data/index.json').read_text())
    first = list(dict.fromkeys(index['featured'] + index['heavy']))
    rest = [v if isinstance(v,str) else v['sid'] for v in index['homes']]
    return list(dict.fromkeys(first + rest))


def beam_listing_homes():
    """Text selects an acquisition cohort, never establishes a visual feature."""
    from agent.tools import store
    s=store()
    pattern=r'exposed.{0,16}(?:beam|rafter)|beamed|wood.beam|beam.ceiling|rafters'
    return sorted(sid for sid in s.homes if re.search(pattern,s.listing(sid)['remarks'],re.I))


def room_mapper(sid):
    from agent.tools import store
    model, _ = store()._audit_graph(sid)
    graph = json.loads((GRAPH/f'{sid}.json').read_text())
    rooms = {n['room_id']: n for n in graph['nodes'] if n['kind']=='room'}
    sunfile = ROOT/f'data/sun/{sid}/sun.json'
    sun = json.loads(sunfile.read_text()) if sunfile.exists() else {}
    heights = {r['id']: r for r in sun.get('rooms',[])}
    edges = {e['id']: e for f in (model or {}).get('floors',[]) for e in f.get('edges',[]) if e.get('id')}
    polygons = []
    for r in (model or {}).get('rooms',[]):
        if r['id'] not in rooms: continue
        segments = []
        for edge in (r.get('boundary') or {}).get('edges',[]):
            vertices = edges.get(edge.get('id'),edge).get('vertices',[])
            if len(vertices)==2:
                segments.append(tuple((v['position']['x'],v['position']['y']) for v in vertices))
        # Reject open boundaries rather than assigning the nearest room.
        counts=Counter((round(x,2),round(y,2)) for seg in segments for x,y in seg)
        if len(segments)<3 or any(n!=2 for n in counts.values()): continue
        z=heights.get(r['id'],{})
        if z.get('floor_z_m') is None or z.get('ceiling_z_m') is None: continue
        polygons.append((r['id'],segments,z['floor_z_m'],z['ceiling_z_m']))
    def inside(x,y,segments):
        hit=False
        for (ax,ay),(bx,by) in segments:
            if (ay>y)!=(by>y) and x < ax+(y-ay)*(bx-ax)/(by-ay): hit=not hit
        return hit
    def locate(p):
        pos=p['position']; found=[rid for rid,segs,lo,hi in polygons
            if lo+0.15<pos['z']<hi+0.15 and inside(pos['x'],pos['y'],segs)]
        if len(found)!=1: return dict(room_id=None,room='room unknown',room_mapping='unknown or ambiguous cached polygon')
        rid=found[0]
        return dict(room_id=rid,room=rooms[rid]['type'],room_mapping='camera inside closed cached XY room boundary and floor/ceiling interval; visible content may extend through openings')
    return locate


def acquire(sid):
    dest=ROOT/f'data/matterport/{sid}/panorama_features'
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'.acquire.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return acquire_locked(sid)


def acquire_locked(sid):
    guard(); dest=ROOT/f'data/matterport/{sid}/panorama_features'; meta=dest/'panoramas.json'
    if meta.exists(): return json.loads(meta.read_text())
    raw=helper().gql('{model(id:"%s"){panoLocations{id position{x y z} rotation{x y z w} skybox(resolution:"2k"){children}}}}'%sid)
    panos=(raw.get('model') or {}).get('panoLocations') or []
    locate=room_mapper(sid)
    candidates=[dict(p,**locate(p)) for p in panos if len((p.get('skybox') or {}).get('children') or [])==6]
    candidates.sort(key=lambda p:(not any(v in p['room'].lower() for v in ('bedroom','kitchen','living')),p['id']))
    selected=[]; seen=set()
    for p in candidates:
        if p['room_id'] and p['room_id'] not in seen:
            selected.append(p); seen.add(p['room_id'])
            if len(selected)==6: break
    # Unmapped / duplicate-room views are only used after all distinct mapped rooms.
    remaining=[p for p in candidates if p not in selected]
    while len(selected)<6 and remaining:
        def distance(p):
            return min(sum((p['position'][k]-q['position'][k])**2 for k in 'xyz') for q in selected) if selected else 0
        p=max(remaining,key=distance);selected.append(p);remaining.remove(p)
    dest.mkdir(parents=True,exist_ok=True)
    def download(job):
        pid,face,url=job
        guard(); path=dest/pid/f'face{face}.jpg';path.parent.mkdir(exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url,timeout=60) as response:
                image=Image.open(io.BytesIO(response.read())).convert('RGB')
            image.thumbnail((640,640)); image.save(path,quality=86)
        return str(path.relative_to(ROOT))
    jobs=[(p['id'],i,url) for p in selected for i,url in enumerate(p['skybox']['children'])]
    # Exceptions are caught by the CLI with class names only, never signed request URLs.
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: images=list(pool.map(download,jobs))
    clean=[]
    for i,p in enumerate(selected):
        clean.append({k:p[k] for k in ('id','position','rotation','room_id','room','room_mapping')})
        clean[-1]['faces']=images[i*6:(i+1)*6]
    result=dict(sid=sid,api='Matterport Model API via experiments/sun_from_pano.py:gql',panoramas=clean,
                available_panoramas=len(panos),selection='one per distinct mapped room first, then spatially separated views; maximum six',signed_urls_saved=False)
    save(meta,result); return result


def local_model():
    guard()
    os.environ.update(HF_HOME=str(ROOT/'data/matterport/_feature_model'),HF_HUB_DISABLE_XET='1',
                      HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
    for name in ('huggingface_hub','httpx','httpcore','transformers'): logging.getLogger(name).setLevel(logging.CRITICAL)
    import torch
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(min(6,os.cpu_count() or 2))
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    # Download diagnostics can contain redirect URLs; suppress them, not just secrets.
    with open(os.devnull,'w') as null, contextlib.redirect_stderr(null), contextlib.redirect_stdout(null):
        from huggingface_hub import snapshot_download
        snapshot=snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.txt','pytorch_model.bin'])
        model=CLIPModel.from_pretrained(snapshot,use_safetensors=False,local_files_only=True).to(device).eval()
        processor=CLIPProcessor.from_pretrained(snapshot,local_files_only=True)
    prompts=[]
    for label in VOCAB:
        words='exposed wooden ceiling beams and rafters' if label=='exposed beams or rafters' else label
        prompts.extend([f'a real estate photograph of {words}.',f'a photograph of a room with {words}.'])
    with torch.inference_mode():
        result=model.get_text_features(**processor(text=prompts,return_tensors='pt',padding=True).to(device))
        vectors=result if isinstance(result,torch.Tensor) else result.pooler_output
        vectors=vectors/vectors.norm(dim=-1,keepdim=True)
        vectors=vectors.reshape(len(VOCAB),2,-1).mean(dim=1);vectors=vectors/vectors.norm(dim=-1,keepdim=True)
    return model,processor,vectors,device


def score(sid, engine):
    path=OUT/f'{sid}.json'
    if path.exists(): return json.loads(path.read_text())
    # Scoring never acquires images; separate stages prevent duplicate API calls.
    meta=json.loads((ROOT/f'data/matterport/{sid}/panorama_features/panoramas.json').read_text())
    model,processor,text,device=engine
    import torch
    rows=[]; images=[]; info=[]
    for pano in meta['panoramas']:
        for face,file in enumerate(pano['faces']):
            with Image.open(ROOT/file) as source:
                image=source.convert('RGB')
            boxes=[(0,0,1,1)] if face in (0,5) else [(0,0,1,1),(0,0,1,.65),(0,.35,1,1)]
            for box in boxes:
                images.append(image.crop(tuple(round(v*(image.width if k%2==0 else image.height)) for k,v in enumerate(box))))
                info.append(dict(home=sid,room=pano['room'],room_id=pano['room_id'],room_mapping=pano['room_mapping'],
                                 pano_id=pano['id'],face=face,image_file=file,crop=list(box)))
    for start in range(0,len(images),16):
        with torch.inference_mode():
            value=model.get_image_features(**processor(images=images[start:start+16],return_tensors='pt').to(device))
            vec=value if isinstance(value,torch.Tensor) else value.pooler_output
            sims=((vec/vec.norm(dim=-1,keepdim=True)) @ text.T).cpu().tolist()
        for crop_meta,values in zip(info[start:start+16],sims):
            for label,value in zip(VOCAB,values):
                rows.append(dict(crop_meta,label=label,confidence=round(value,6),confidence_kind='CLIP image/text cosine similarity, not a probability',model=MODEL,model_revision=REVISION,
                    crop_definition='Normalized source crop before CLIP resize-shortest-edge and center crop to 224px; not an object bounding box'))
    # One observation per label/face, retaining the strongest crop, not an object count.
    best={}
    for r in rows:
        key=(r['pano_id'],r['face'],r['label'])
        if key not in best or r['confidence']>best[key]['confidence']: best[key]=r
    result=dict(sid=sid,model=MODEL,device=device,panoramas=len(meta['panoramas']),candidates=list(best.values()))
    save(path,result); return result


def render_sheet(chosen, name):
    tilew,tileh=350,300; canvas=Image.new('RGB',(tilew*5,tileh*((len(chosen)+4)//5)),'white')
    draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=16)
    for i,r in enumerate(chosen):
        with Image.open(ROOT/r['image_file']) as source:
            box=tuple(round(v*(source.width if k%2==0 else source.height)) for k,v in enumerate(r['crop']))
            crop=source.crop(box).convert('RGB');crop.thumbnail((tilew,215))
        x,y=(i%5)*tilew,(i//5)*tileh;canvas.paste(crop,(x,y))
        draw.text((x+4,y+218),f"{i+1:02} {r['label']} {r['confidence']:.3f}",fill='black',font=font)
        draw.text((x+4,y+241),r['home']+' face '+str(r['face']),fill='black',font=font)
        draw.text((x+4,y+264),r['room'][:37],fill='black',font=font)
    canvas.save(GRAPH/(name+'.jpg'),quality=90)


def random_sheet(seed=20261114):
    """Stratified probability sample; rare-label strata are complete censuses."""
    existing=GRAPH/'panorama_random_sheet.json'
    if existing.exists() and json.loads(existing.read_text()).get('review_status','pending')!='pending':
        raise ValueError('Reviewed random sheet exists; preserve its JPG and JSON before regenerating')
    calibration=json.loads((GRAPH/'panorama_features_calibration.json').read_text())
    rows=[n for file in json.loads((GRAPH/'index.json').read_text())['files']
          for n in json.loads((GRAPH/file).read_text())['nodes'] if n.get('source_kind')=='seen in panorama']
    groups={label:sorted((r for r in rows if r['label']==label),key=lambda r:r['id'])
            for label,cutoff in calibration['label_thresholds'].items() if cutoff is not None}
    assert sum(map(len,groups.values()))==len(rows), 'Published disabled labels require investigation'
    rare={'exposed beams or rafters','built-in shelves','fireplace'}
    allocations={label:len(group) for label,group in groups.items() if label in rare}
    remaining=40-sum(allocations.values())
    if remaining<0:raise ValueError('Rare-label census exceeds 40; do not silently omit mandatory nodes')
    common={label:group for label,group in groups.items() if label not in rare and group}
    total=sum(map(len,common.values()));budget=min(remaining,total)
    quotas={label:budget*len(group)/total for label,group in common.items()} if total else {}
    allocations.update({label:int(q) for label,q in quotas.items()})
    for label in sorted(quotas,key=lambda label:(-(quotas[label]-int(quotas[label])),label))[:budget-sum(int(q) for q in quotas.values())]:
        allocations[label]+=1
    rng=random.Random(seed);chosen=[]
    for label,group in sorted(groups.items()):chosen.extend(rng.sample(group,allocations.get(label,0)))
    rng.shuffle(chosen)
    name='panorama_random_sheet';render_sheet(chosen,name)
    save(GRAPH/(name+'.json'),dict(seed=seed,selection='Seeded uniform random sample without replacement within enabled label; complete beam, shelf and fireplace strata; proportional largest-remainder allocation for remaining labels',
        unit='Published observation node, not unique physical object; views can duplicate an object',
        population_size=len(rows),population_sha256=hashlib.sha256(json.dumps(sorted(rows,key=lambda r:r['id']),sort_keys=True).encode()).hexdigest(),
        calibration_sha256=hashlib.sha256((GRAPH/'panorama_features_calibration.json').read_bytes()).hexdigest(),
        strata={label:dict(population=len(group),sample=allocations.get(label,0),inclusion_probability=allocations.get(label,0)/len(group) if group else None) for label,group in groups.items()},
        estimator='Population-weighted precision = sum(label_population * label_correct / label_sample) / total_population. Do not use the unweighted sample ratio as population precision; rare labels are oversampled.',
        review_status='pending',crops=[dict(r,tile=i+1) for i,r in enumerate(chosen)]))
    return len(chosen)


def sheet():
    rows=[r for p in sorted(OUT.glob('*.json')) for r in json.loads(p.read_text()).get('candidates',[])]
    chosen=[]
    # Top and boundary examples per label expose failure modes rather than estimate prevalence.
    for label in VOCAB:
        ranked=sorted((r for r in rows if r['label']==label),key=lambda r:-r['confidence'])
        if ranked:
            chosen.append(ranked[0])
            if len(ranked)>1: chosen.append(ranked[min(len(ranked)-1,max(1,len(ranked)//20))])
    for r in sorted(rows,key=lambda r:-r['confidence']):
        if len(chosen)>=40: break
        if r not in chosen and r['label']=='exposed beams or rafters' and sum(v['home']==r['home'] and v['label']==r['label'] for v in chosen)<2:
            chosen.append(r)
    render_sheet(chosen,'panorama_features_sheet')
    save(GRAPH/'panorama_features_sheet.json',dict(selection='Per label highest score and 95th-percentile score, plus top beam examples (maximum two per home); not random and not representative precision evidence',crops=chosen))
    return len(chosen)


def publish():
    from agent.home_graph import attach_panorama_features,graph_search
    calibration=json.loads((GRAPH/'panorama_features_calibration.json').read_text())
    threshold=calibration['threshold'];counts=Counter();processed=[];manifest=[]
    claims={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/claims').glob('*.json')}
    for path in sorted(OUT.glob('*.json')):
        if path.name.startswith('accepted_'):continue
        guard();doc=json.loads(path.read_text());sid=doc['sid'];accepted=[]
        for row in doc['candidates']:
            cutoff=calibration.get('label_thresholds',{}).get(row['label'],threshold)
            if cutoff is None or row['confidence']<cutoff:continue
            ident=hashlib.sha256(f"{row['pano_id']}:{row['face']}:{row['label']}".encode()).hexdigest()[:20]
            accepted.append(dict(row,id=f'feature:{sid}:pano:{ident}',kind='feature',text=row['label'],model_revision=REVISION,
                crop_definition='Normalized source crop before CLIP resize-shortest-edge and center crop to 224px; not an object bounding box',
                source_kind='seen in panorama',source=f"seen in panorama {row['pano_id']} face {row['face']}",
                source_file=row['image_file'],human_reviewed=False,spatially_verified=False,
                threshold=cutoff,calibration_file='data/claims/graph/panorama_features_calibration.json'))
            counts[row['label']]+=1
        save(OUT/f'accepted_{sid}.json',dict(sid=sid,features=accepted))
        graphpath=GRAPH/f'{sid}.json'; graph=attach_panorama_features(json.loads(graphpath.read_text()))
        save(graphpath,graph);processed.append(dict(sid=sid,panoramas=doc['panoramas'],observations=len(accepted)))
        meta=ROOT/f'data/matterport/{sid}/panorama_features/panoramas.json'
        files=[meta,path]+[ROOT/f for p in json.loads(meta.read_text())['panoramas'] for f in p['faces']]
        for file in files:
            manifest.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    indexpath=GRAPH/'index.json';index=json.loads(indexpath.read_text())
    index['caveats'][0]='Listing text is attributed as listing says. Seen-in-panorama features are uncalibrated local CLIP crop classifications, not human-verified objects. Room is the camera location; content can extend through openings.'
    index['panorama_features']=dict(homes=len(processed),observations=sum(counts.values()),model=MODEL,threshold=threshold)
    # Recount derived index after replacement, so rebuild/publish remain idempotent.
    kinds=Counter()
    for file in index['files']:
        for node in json.loads((GRAPH/file).read_text())['nodes']:kinds[node['kind']]+=1
    index['counts']=dict(kinds);index['bytes']=sum((GRAPH/f).stat().st_size for f in index['files']);save(indexpath,index)
    queries=[dict(feature='beam|rafter',source_kind='seen in panorama',limit=5),
             dict(room='bedroom',feature='beam|rafter',source_kind='seen in panorama',limit=5),
             dict(room='primary bedroom',feature='beam|rafter',source_kind='seen in panorama',date='2026-12-21',min_sun_minutes=60,limit=5)]
    proof=[graph_search(q) for q in queries]
    if not proof[0]['matches']:
        label=counts.most_common(1)[0][0] if counts else 'fireplace'
        proof.insert(0,graph_search(dict(feature=label,source_kind='seen in panorama',min_feature_confidence=threshold,limit=5)))
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/claims').glob('*.json')}
    result=dict(model=MODEL,model_revision=REVISION,devices=sorted({json.loads(p.read_text())['device'] for p in OUT.glob('*.json') if not p.name.startswith('accepted_')}),
        versions={p:importlib.metadata.version(p) for p in ('torch','transformers','pillow','numpy','huggingface-hub','safetensors','tokenizers')},
        threshold=threshold,label_thresholds=calibration.get('label_thresholds',{}),
        processed=processed,skipped=[row for p in GRAPH.glob('panorama_score*run.json') for row in json.loads(p.read_text()).get('skipped',[]) if row['sid'] not in {v['sid'] for v in processed}],
        nodes_per_label={label:counts[label] for label in VOCAB},queries=proof,per_home_claims_unchanged=claims==after,
        confidence_definition='Cosine image/text similarity, not a probability. Observations can duplicate a physical feature across views.',
        room_caveat='Camera room only; object may be visible through a doorway. Largest bedroom is not assumed primary.',
        precision=calibration['precision_estimate'],baseline_free_bytes=BASELINE_FREE,free_bytes=guard(),conservative_task_allocated_bytes=task_disk_bytes())
    for file in (ROOT/'data/matterport/_feature_model').rglob('*'):
        if file.is_file() and not file.is_symlink():
            manifest.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    for file in (GRAPH/'panorama_features_calibration.json',GRAPH/'panorama_features_sheet.jpg',GRAPH/'panorama_features_sheet.json',
                 GRAPH/'panorama_features_calibration36_sheet.jpg',GRAPH/'panorama_features_calibration36_sheet.json',
                 ROOT/'experiments/sun_from_pano.py',ROOT/'agent/panorama_features.py',ROOT/'agent/home_graph.py',ROOT/'agent/requirements-panorama.txt'):
        manifest.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    save(ROOT/'data/claims/eval/panorama_features_proof.json',result)
    save(GRAPH/'panorama_features_input_manifest.json',manifest)
    print(json.dumps(dict(homes=len(processed),nodes_per_label=result['nodes_per_label'],queries=[p['matches'] for p in proof],free_bytes=result['free_bytes'])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['download-model','acquire','score','sheet','random-sheet','publish']);parser.add_argument('--limit',type=int,default=36)
    parser.add_argument('--seed',type=int,default=20261114)
    parser.add_argument('--beam-listings',action='store_true',help='After featured cohort, inspect cached beam/rafter listing candidates; text is only an acquisition hint')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);guard()
    if args.command=='download-model':
        engine=local_model();print(json.dumps(dict(model=MODEL,device=engine[-1],free_bytes=guard())));return
    if args.command=='sheet':print(json.dumps(dict(crops=sheet())));return
    if args.command=='random-sheet':print(json.dumps(dict(crops=random_sheet(args.seed),seed=args.seed)));return
    if args.command=='publish':publish();return
    engine=local_model() if args.command=='score' else None
    report=dict(baseline_free_bytes=BASELINE_FREE,model_revision=REVISION,versions={p:importlib.metadata.version(p) for p in ('torch','transformers','pillow','numpy','huggingface-hub','safetensors','tokenizers')},processed=[],skipped=[])
    cohort=beam_listing_homes() if args.beam_listings else homes()[:args.limit]
    for sid in cohort:
        try:
            result=score(sid,engine) if engine else acquire(sid)
            row=dict(sid=sid,panoramas=result['panoramas'] if isinstance(result['panoramas'],int) else len(result['panoramas']))
            report['processed'].append(row);print(json.dumps(row),flush=True)
        except Exception as error:
            if isinstance(error,RuntimeError) and str(error).startswith('STOP:'): raise
            row=dict(sid=sid,reason=type(error).__name__);report['skipped'].append(row);print(json.dumps(row),flush=True)
        report['free_bytes']=guard();report['conservative_task_allocated_bytes']=task_disk_bytes()
        suffix='_beam' if args.beam_listings else ''
        save(GRAPH/f'panorama_{args.command}{suffix}_run.json',report)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print(json.dumps(dict(status='interrupted',detail='Resume the same command; completed per-home artifacts are reused.')),flush=True)
        sys.exit(130)
    except Exception as error:
        # Never render remote exceptions, request objects or HTTP response bodies.
        print(json.dumps(dict(error=type(error).__name__,detail=str(error) if isinstance(error,RuntimeError) and str(error).startswith('STOP:') else 'remote diagnostics suppressed')),flush=True)
        sys.exit(1)
