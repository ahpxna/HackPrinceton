"""Resume SILT's gated, cache-only 169-home expansion; no acquisition functions.

Run: .venv-agent/bin/python -I experiments/sun_patch_round2_scale.py --run
Per-face and per-panorama receipts permit restarting that exact command.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('round2', ROOT/'experiments/sun_patch_round2.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
V, S = R.V, R.S
OUT = ROOT/'data/sun_patch/auto/round2_scale'
INITIAL_BASES = V.A.BASES.copy()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    path = OUT/'manifest.json'
    if path.exists():
        return V.read(path)
    rows = []
    for path in sorted((ROOT/'data/matterport').glob('*/panorama_features/panoramas.json')):
        metadata = V.read(path)
        sid = metadata['sid']
        sun = ROOT/'data/sun'/sid/'sun.json'
        model = ROOT/'data/cache'/sid/'matterport.json'
        # These are already cached inputs, copied to isolate this run from other work.
        for source, name in [(sun,'sun_snapshot.json'),(model,'model_snapshot.json')]:
            target = OUT/'cache'/sid/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copy2(source,target)
        for pose in metadata['panoramas']:
            folder = path.parent/pose['id']
            assert all((folder/f'face{f}.jpg').exists() for f in range(6))
            rows.append({'sid':sid,'pano_id':pose['id'],'pose':pose,
                         'source_folder':str(folder.relative_to(ROOT)),
                         'sun_sha256':digest(sun) if sun.exists() else None,
                         'model_sha256':digest(model) if model.exists() else None,
                         'metadata_sha256':digest(path)})
    assert len(rows)==1014 and len({r['sid'] for r in rows})==169
    result = {'rows':rows,'homes':169,'panoramas':1014,'faces':5070,
              'included_faces':[1,2,3,4,5],'excluded_face':0,
              'ceiling_basis_local':INITIAL_BASES[0].tolist(),
              'source_sha256':digest(Path(__file__)), 'pilot_params':R.PARAMS,
              'checkpoint_sha256':digest(R.OUT/'models/PVTb5_SBU.pth')}
    R.write(OUT/'manifest.json',result)
    return result


def select_device(rows):
    receipt = OUT/'device.json'
    if receipt.exists():
        return V.read(receipt)['selected']
    result = {'mps_built':torch.backends.mps.is_built(),
              'mps_available':torch.backends.mps.is_available(), 'selected':'cpu'}
    if result['mps_available']:
        try:
            cpu = R.load_detector('silt','cpu')
            gpu = R.load_detector('silt','mps')
            checks = []
            for f in range(1,6):
                image = Image.open(ROOT/rows[0]['source_folder']/f'face{f}.jpg').convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
                with torch.inference_mode():
                    a,_ = cpu(image)
                    b,_ = gpu(image)
                checks.append({'face':f,'mask_disagreement_pixels':int(np.count_nonzero((a<=.7)!=(b<=.7))),
                               'max_probability_difference':float(np.max(abs(a-b)))})
            result['parity_checks'] = checks
            if all(c['mask_disagreement_pixels']==0 for c in checks):
                result['selected']='mps'
        except Exception as error:
            result['mps_error']=type(error).__name__+': '+str(error)[:500]
    else:
        result['reason']='PyTorch reports no available MPS device in this execution environment.'
    R.write(receipt,result)
    return result['selected']


def infer(data):
    rows = data['rows']
    device = select_device(rows)
    detector = R.load_detector('silt',device)
    for i,row in enumerate(rows):
        folder = OUT/row['sid']/row['pano_id']
        folder.mkdir(parents=True, exist_ok=True)
        for face in data['included_faces']:
            meta = folder/f'face{face}.json'
            source = ROOT/row['source_folder']/f'face{face}.jpg'
            if meta.exists():
                old=V.read(meta)
                assert old['source_sha256']==digest(source)
                assert old['params']==R.PARAMS
                assert all((folder/f'face{face}_{kind}.png').exists() for kind in ('prob','mask'))
                continue
            tick = time.monotonic()
            image = Image.open(source).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
            with torch.inference_mode(): probability, details = detector(image)
            Image.fromarray(np.rint(probability.clip(0,1)*65535).astype(np.uint16)).save(folder/f'face{face}_prob.png')
            Image.fromarray((probability<=.7).astype(np.uint8)*255).save(folder/f'face{face}_mask.png')
            R.write(meta,{**details,'source_sha256':digest(source),'params':R.PARAMS,'device':device,
                          'seconds':time.monotonic()-tick,'mask_fraction':float(np.mean(probability<=.7))})
        R.write(OUT/'inference_progress.json',{'panos_complete':i+1,'panos_total':len(rows),
                                             'last_sid':row['sid'],'last_pano':row['pano_id']})
        print(json.dumps({'phase':'infer','pano':i+1,'total':len(rows),'sid':row['sid']}),flush=True)
        if (i+1)%6==0:
            R.progress(f'Pilot scores/report/gallery complete; SILT scale inference {i+1}/{len(rows)} panos, {5*(i+1)}/5070 faces. Device {device}. Per-face checkpoints under data/sun_patch/auto/round2_scale/.',
                       'Finish cached 169-home inference, geometry/null and human-review artifacts; resume with experiments/sun_patch_round2_scale.py --run.')


def project(row):
    sid,pid = row['sid'],row['pano_id']
    folder = OUT/sid/pid
    path = folder/'projection.json'
    if path.exists():
        return V.read(path)
    if row['sun_sha256'] is None or row['model_sha256'] is None:
        R.write(folder/'projection_unavailable.json',{'reason':'Cached sun geometry/model unavailable; no acquisition.'})
        return None
    sun = V.read(V.CACHE/sid/'sun_snapshot.json')
    room = next((r for r in sun['rooms'] if r['id']==row['pose']['room_id']),None)
    if room is None:
        R.write(folder/'projection_unavailable.json',{'reason':'Pano room absent from cached sun rooms; no floor height invented.'})
        return None
    faces = [Image.open(ROOT/row['source_folder']/f'face{f}.jpg').convert('RGB') for f in range(6)]
    V.A.BASES = INITIAL_BASES.copy()
    seams = V.A.check_cap_seams(faces)
    candidate = {'room_id':room['id'],'floor_z_m':room['floor_z_m']}
    image, transform = V.rectified(sid,candidate,row['pose'],faces)
    image.save(folder/'floor.jpg',quality=92)
    result = {'candidate':candidate,'transform':transform,'basis':V.A.BASES.tolist(),
              'pose':row['pose'],'cap_seams':seams,'floor_elevation_uncertain':room.get('floor_elevation_uncertain')}
    R.write(path,result)
    return result


def geometry(data):
    V.CACHE = OUT/'cache'
    all_rows = []
    for i,row in enumerate(data['rows']):
        sid,pid = row['sid'],row['pano_id']
        folder = OUT/sid/pid
        receipt = folder/'geometry.json'
        if receipt.exists():
            all_rows.extend(V.read(receipt)['rows'])
            continue
        projection = project(row)
        if projection is None:
            R.write(receipt,{'rows':[],'status':'cached_room_unavailable'})
            continue
        V.A.BASES = np.array(projection['basis'])
        t = projection['transform']; n=t['size']
        xx,yy=np.meshgrid(np.linspace(t['lo'][0],t['hi'][0],n),np.linspace(t['hi'][1],t['lo'][1],n))
        points=np.stack([xx,yy,np.full_like(xx,t['floor_z_m'])],-1)
        position=np.array([projection['pose']['position'][k] for k in ('x','y','z')])
        # The excluded ceiling contributes zero. Its raw photograph is used only for seam calibration.
        masks=[Image.new('RGB',(512,512))]+[Image.open(folder/f'face{f}_mask.png').convert('RGB') for f in range(1,6)]
        raw=np.array(V.sample_image(points-position,projection['pose'],masks))[:,:,0]>127
        room=S.room_mask(sid,projection['candidate']['room_id'],t)
        count,labels,stats,_=cv2.connectedComponentsWithStats((raw&room).astype(np.uint8),8)
        found=[];rejected=[]
        for label in range(1,count):
            area=int(stats[label,cv2.CC_STAT_AREA]);fraction=float(area/max(1,room.sum()))
            if area<60 or fraction>.25:
                rejected.append({'label':label,'area':area,'fraction':fraction})
                continue
            mask=labels==label
            patch=folder/f'patch{label:03d}.png'
            Image.fromarray(mask.astype(np.uint8)*255).save(patch)
            c={'sid':sid,'pano_id':pid,'patch_id':f'silt_scale/{sid}/{pid}/{label:03d}',
               'room_id':projection['candidate']['room_id'],'mask':mask,'projection':projection,
               'mask_path':str(patch.relative_to(ROOT)),'area_pixels':area,'floor_fraction':fraction}
            fit_path=folder/f'patch{label:03d}_fit.json'
            if fit_path.exists():
                result=V.read(fit_path)
            else:
                fit=S.fit(c)
                result={k:v for k,v in c.items() if k not in ('mask','projection')}
                result['fit']=fit
                result['band']=S.band_test(c,fit) if fit and fit['accepted'] else None
                result['registration_quality']='poor' if V.read(V.CACHE/sid/'sun_snapshot.json')['registration']['poor'] else 'well'
                R.write(fit_path,result)
            found.append(result)
        R.write(receipt,{'rows':found,'rejected_components':rejected,'status':'complete'})
        all_rows.extend(found)
        print(json.dumps({'phase':'geometry','pano':i+1,'total':len(data['rows']),'candidates':len(found)}),flush=True)
        R.progress(f'SILT scale inference complete (5070 faces); geometry/null complete for {i+1}/1014 panos; per-patch fit checkpoints retained.',
                   'Finish geometry; export random 50 accepted-patch contact sheet and 100-face gallery; refresh report.')
    return all_rows


def artifacts(data,rows):
    accepted=[r for r in rows if r['fit'] and r['fit']['accepted']]
    summary={'status':'complete','homes':169,'panoramas':1014,'faces':5070,'candidates':len(rows),
             'iou_accepted':len(accepted),'precision':None,'registration_groups':{},
             'missing_geometry_panos':sum(V.read(OUT/r['sid']/r['pano_id']/'geometry.json')['status']!='complete' for r in data['rows']),
             'rows':rows,'source_sha256':digest(Path(__file__)),
             'interpretation':'Direction implied by a detected patch is consistent with the possible sun path; not a validation of sun minutes.',
             'caveat':'Non-shadow complement is not a sunlight detector. Three pilot hits in one home; second retained home missed. Repeated views are correlated.',
             'horizon':'Cached neighbour/tree prism proxy only; full DSM unavailable; own roof omitted.'}
    for quality in ('well','poor'):
        group=[r for r in accepted if r['registration_quality']==quality]
        homes={r['sid'] for r in data['rows'] if r['sun_sha256'] is not None and ('poor' if V.read(OUT/'cache'/r['sid']/'sun_snapshot.json')['registration']['poor'] else 'well')==quality}
        summary['registration_groups'][quality]={'homes_total':len(homes),'homes_with_accepted':len({r['sid'] for r in group}),
            'accepted':len(group),'in_band':sum(r['band']['in_annual_band_and_proxy_horizon'] for r in group),
            'in_band_fraction':float(np.mean([r['band']['in_annual_band_and_proxy_horizon'] for r in group])) if group else None,
            'mean_null_chance':float(np.mean([r['band']['null_chance'] for r in group])) if group else None}
    R.write(OUT/'results.json',summary)
    rng=np.random.default_rng(20261114)
    picks=[accepted[int(i)] for i in rng.choice(len(accepted),min(50,len(accepted)),replace=False)] if accepted else []
    sheet=Image.new('RGB',(1500,max(1,math.ceil(len(picks)/5))*260+55),'white')
    draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=13)
    draw.text((8,8),f'SILT Round 2: {len(picks)} random IoU>=0.3 candidates; seed 20261114; human precision pending.',fill='black',font=font)
    sheet_manifest=[]
    for i,c in enumerate(picks):
        folder=OUT/c['sid']/c['pano_id']
        image=np.array(Image.open(folder/'floor.jpg').convert('RGB'))
        mask=np.array(Image.open(ROOT/c['mask_path']))>127
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,(0,255,0),3)
        x,y=(i%5)*300,55+(i//5)*260
        sheet.paste(Image.fromarray(image).resize((240,240)),(x,y))
        draw.text((x,y+242),f'{i+1}: {c["sid"]} IoU={c["fit"]["iou"]:.2f}',fill='black',font=font)
        sheet_manifest.append({'tile':i+1,'patch_id':c['patch_id']})
    graph=ROOT/'data/claims/graph/sun_patch_auto_sheet.jpg'
    if graph.exists() and not (R.OUT/'checkpoint/sun_patch_auto_sheet.round1.jpg').exists():
        shutil.copy2(graph,R.OUT/'checkpoint/sun_patch_auto_sheet.round1.jpg')
    sheet.save(graph,quality=94)
    R.write(OUT/'sheet_manifest.json',sheet_manifest)
    faces=[(r,f) for r in data['rows'] for f in range(1,6)]
    chosen=[faces[int(i)] for i in rng.choice(len(faces),100,replace=False)]
    gallery=OUT/'gallery';gallery.mkdir(exist_ok=True)
    cards=[];manifest_rows=[]
    for i,(r,f) in enumerate(chosen):
        image=np.array(Image.open(ROOT/r['source_folder']/f'face{f}.jpg').convert('RGB').resize((384,384)))
        mask=np.array(Image.open(OUT/r['sid']/r['pano_id']/f'face{f}_mask.png').resize((384,384),Image.Resampling.NEAREST))>127
        overlay=image.copy();overlay[mask]=(.55*overlay[mask]+.45*np.array([190,65,255])).astype(np.uint8)
        canvas=Image.new('RGB',(768,416),'white');canvas.paste(Image.fromarray(image),(0,32));canvas.paste(Image.fromarray(overlay),(384,32))
        ImageDraw.Draw(canvas).text((5,5),f'{i+1}: {r["sid"]} / {r["pano_id"]} / face{f} | raw non-shadow mask',fill='black',font=font)
        filename=f'{i:03d}.jpg';canvas.save(gallery/filename,quality=92)
        cards.append(f'<figure><img loading="lazy" src="{filename}" width="768" height="416"><figcaption>{i+1}: {r["sid"]} / {r["pano_id"]} / face{f}</figcaption></figure>')
        manifest_rows.append({'sample':i+1,'sid':r['sid'],'pano_id':r['pano_id'],'face':f,'file':filename})
    R.write(gallery/'manifest.json',manifest_rows)
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SILT 169-home sample</title><style>body{font:14px sans-serif;background:white;color:black}img{max-width:100%;height:auto}figure{margin:20px 0}</style><h1>SILT 169-home expansion: 100 random faces</h1><p>Seed 20261114. Original | raw non-shadow complement. Ceiling excluded. No human precision estimate.</p>'+''.join(cards))
    # Preserve pilot and round-1 rows; add completed scale receipts to the same report.
    target=S.EVAL/'sun_patch_consistency.json'
    report=V.read(target);report['round2']['scale']=summary
    report['round2']['scope']='Completed pilot detector evaluation and separate cache-only SILT scale geometry/null experiment.'
    R.write(target,report)
    target=S.EVAL/'sun_patch_consistency.md'
    lines=[line for line in target.read_text().splitlines() if not line.startswith(('SILT scale complete:', 'Scale well/poor results ', 'Scale per-patch distances/null:'))]
    insertion=[f'SILT scale complete: 169 homes / 1,014 cached panoramas / 5,070 faces (ceiling excluded); {len(rows)} candidates, {len(accepted)} IoU>=0.3; {summary["missing_geometry_panos"]} panoramas lack cached floor geometry.',
               'Scale well/poor results (accepted, in-band fraction, random-north mean): '+json.dumps(summary['registration_groups'],separators=(',',':'))+'.',
               'Scale per-patch distances/null: `data/sun_patch/auto/round2_scale/results.json`; 50 accepted-patch human sheet `data/claims/graph/sun_patch_auto_sheet.jpg`; 100-face gallery `data/sun_patch/auto/round2_scale/gallery/index.html`.']
    index=next(i for i,l in enumerate(lines) if l.startswith('## Round 1'))
    lines[index:index]=insertion
    assert len(lines)<=50
    target.write_text('\n'.join(lines)+'\n')
    R.progress(f'Completed Round 2 pilot and SILT scale:169 homes/1014panos/5070faces; {len(rows)} candidates/{len(accepted)} IoU-accepted. Reports, null split, contact sheet and sampled gallery exported.',
               'Human review: count true sun patches on 50-candidate sheet; precision is null until then.')


def run():
    scores=V.read(R.OUT/'scores.json')
    assert scores['models']['silt']['gate_passed'], '169-home inference requires completed pilot with >=3/4 recall'
    torch.set_num_threads(4);torch.manual_seed(20261114)
    socket.socket=S.blocked;socket.create_connection=S.blocked
    data=manifest()
    assert data['pilot_params']==R.PARAMS
    assert data['checkpoint_sha256']==digest(R.OUT/'models/PVTb5_SBU.pth')
    for sid in {row['sid'] for row in data['rows']}:
        row=next(row for row in data['rows'] if row['sid']==sid)
        for key,name in [('sun_sha256','sun_snapshot.json'),('model_sha256','model_snapshot.json')]:
            if row[key] is not None:
                assert row[key]==digest(OUT/'cache'/sid/name)
    state={'pid':os.getpid(),'cwd':str(ROOT),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'command':'.venv-agent/bin/python -I experiments/sun_patch_round2_scale.py --run',
           'source_sha256':digest(Path(__file__)),'phase':'inference'}
    R.write(OUT/'run_state.json',state)
    infer(data)
    state['phase']='geometry';R.write(OUT/'run_state.json',state)
    rows=geometry(data)
    artifacts(data,rows)
    state['phase']='complete';state['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    R.write(OUT/'run_state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true')
    if parser.parse_args().run:
        try:
            run()
        except Exception as error:
            path=OUT/'run_state.json'
            state=V.read(path) if path.exists() else {}
            state.update({'phase':'failed','error_type':type(error).__name__,'error':str(error)[:500]})
            R.write(path,state)
            raise
