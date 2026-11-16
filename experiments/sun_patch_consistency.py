"""SP pilot. Offline luminance/straight-edge baseline, NOT a trained shadow model.

Run: .venv-agent/bin/python -I experiments/sun_patch_consistency.py
No credentials, acquisition, SAM licence bypass, or source-data mutation.
"""
import csv
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / 'data/sun_patch/auto'
EVAL = ROOT / 'data/claims/eval'
PROGRESS = Path('/var/folders/x3/hfjp6k_96z50j37_q9hftwjr0000gp/T/oab-codex-runs/progress-2026-11-14T20-49-11-903Z-19107-19646880-17d8-46c0-b169-e55c75c50b06.md')
SPEC = importlib.util.spec_from_file_location('sp_projection', ROOT / 'experiments/sun_patch_check.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
PARAMS = dict(version='SP-luminance-edge-v1', face_size=512, min_luma=185,
              percentile=78, blur_sigma=2, min_floor_pixels=60,
              max_floor_fraction=.25, min_contrast=12, min_solidity=.65,
              polygon_epsilon=.025, max_polygon_vertices=10,
              min_sharp_boundary_fraction=.18, sharp_gradient=8,
              direction_step_deg=4, refinement_step_deg=1, min_iou=.3,
              north_rotations=1000, seed=20261114)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def progress(done, next_step):
    PROGRESS.write_text('# Objective\nSP cached 15-home pilot on this Mac (CPU); scale only after reference detector gate.\n\n# Done\n' + done + '\n\n# Next\n' + next_step + '\n')


def blocked(*args, **kwargs):
    raise RuntimeError('SP_offline_socket_guard')


def room_mask(sid, room_id, transform):
    room = next(r for r in V.model(sid)['rooms'] if r['id'] == room_id)
    n = transform['size']
    yy, xx = np.mgrid[:n, :n]
    x = transform['lo'][0] + xx/(n-1)*(transform['hi'][0]-transform['lo'][0])
    y = transform['hi'][1] - yy/(n-1)*(transform['hi'][1]-transform['lo'][1])
    inside = np.zeros((n,n), bool)
    for edge in room['boundary']['edges']:
        a,b = [v['position'] for v in edge['vertices'][:2]]
        if abs(b['y']-a['y']) < 1e-10:
            continue
        inside ^= ((a['y'] > y) != (b['y'] > y)) & (x < (b['x']-a['x'])*(y-a['y'])/(b['y']-a['y'])+a['x'])
    # Exclude a 10cm wall strip; cannot exclude furniture without a depth map.
    px = max(1, round(.10/(transform['hi'][0]-transform['lo'][0])*(n-1)))
    return cv2.erode(inside.astype(np.uint8), np.ones((px*2+1,px*2+1),np.uint8)).astype(bool)


def detect(sid, pid, projection):
    """Threshold cube faces first, project masks with the frozen cube conventions."""
    folder = AUTO / sid / pid
    folder.mkdir(parents=True, exist_ok=True)
    V.A.BASES = np.array(projection['basis'])
    face_masks = []
    face_metadata = []
    for face in range(6):
        path = V.CACHE / sid / pid / f'face{face}.jpg'
        im = Image.open(path).convert('RGB').resize((512,512), Image.Resampling.LANCZOS)
        gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        smooth = cv2.GaussianBlur(gray.astype(np.float32),(0,0),PARAMS['blur_sigma'])
        threshold = max(PARAMS['min_luma'], float(np.percentile(smooth,PARAMS['percentile'])))
        mask = (smooth >= threshold).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        Image.fromarray(mask).save(folder / f'face{face}_bright.png')
        face_masks.append(Image.fromarray(mask).convert('RGB'))
        face_metadata.append({'face':face, 'threshold':threshold, 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    t = projection['transform']
    n = t['size']
    xx,yy = np.meshgrid(np.linspace(t['lo'][0],t['hi'][0],n),np.linspace(t['hi'][1],t['lo'][1],n))
    points = np.stack([xx,yy,np.full_like(xx,t['floor_z_m'])],-1)
    pose = projection['pose']
    position = np.array([pose['position'][k] for k in ('x','y','z')])
    bright = np.asarray(V.sample_image(points-position,pose,face_masks))[:,:,0] > 127
    rgb = np.array(Image.open(V.OUT / sid / f'{pid}_floor.jpg').convert('RGB'))
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),2)
    room = room_mask(sid,projection['candidate']['room_id'],t)
    binary = (bright & room).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary,8)
    grad = np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))
    candidates = []
    rejects = []
    for label in range(1,count):
        area = int(stats[label,cv2.CC_STAT_AREA])
        if area < PARAMS['min_floor_pixels']:
            continue
        mask = (labels==label).astype(np.uint8)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours,key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = min(1., area/max(1.,hull_area))
        approx = cv2.approxPolyDP(c,PARAMS['polygon_epsilon']*cv2.arcLength(c,True),True)
        ring = (cv2.dilate(mask,np.ones((25,25),np.uint8)) > 0) & ~bright & room
        contrast = float(np.median(gray[mask>0])-np.median(gray[ring])) if ring.any() else 0.
        boundary = (mask-cv2.erode(mask,np.ones((3,3),np.uint8))) > 0
        sharp = float(np.mean(grad[boundary] >= PARAMS['sharp_gradient'])) if boundary.any() else 0.
        metrics = dict(area_pixels=area, contrast_luma=contrast, solidity=solidity, polygon_vertices=len(approx),sharp_boundary_fraction=sharp)
        reasons = []
        if area > room.sum()*PARAMS['max_floor_fraction']: reasons.append('too_large')
        if contrast < PARAMS['min_contrast']: reasons.append('low_surrounding_contrast')
        if solidity < PARAMS['min_solidity']: reasons.append('noncompact')
        if not 3 <= len(approx) <= PARAMS['max_polygon_vertices']: reasons.append('not_straight_polygon')
        if sharp < PARAMS['min_sharp_boundary_fraction']: reasons.append('soft_edges')
        if reasons:
            rejects.append({**metrics,'reasons':reasons})
            continue
        index = len(candidates)
        path = folder / f'patch{index:03d}.png'
        Image.fromarray(mask*255).save(path)
        candidates.append({'sid':sid,'pano_id':pid,'patch_id':f'{sid}/{pid}/{index:03d}',
                           'room_id':projection['candidate']['room_id'],'mask_path':str(path.relative_to(ROOT)),
                           'metrics':metrics,'mask':mask.astype(bool),'projection':projection})
    write(folder/'detection.json',{'face_metadata':face_metadata,'candidate_count':len(candidates),'rejected':rejects})
    return candidates


def reference_recall(candidates):
    usable = {r['patch_id'] for r in csv.DictReader((V.OUT/'results.csv').open()) if r['usable_for_summary']=='True'}
    rows = []
    for mark in V.read(V.OUT/'marks_v1.json')['patches']:
        polygon = np.zeros((800,800),np.uint8)
        cv2.fillPoly(polygon,[np.array(mark['visible_outline'],np.int32)],1)
        matches = []
        for c in candidates:
            if (c['sid'],c['pano_id']) != (mark['sid'],mark['pano_id']):continue
            coverage = float(np.count_nonzero(c['mask'] & (polygon>0))/max(1,polygon.sum()))
            # One-pixel mask sampling; no expansion tuned to marked corners.
            corners = all(c['mask'][int(p['floor_uv'][1]),int(p['floor_uv'][0])] for p in mark['pairs'])
            matches.append((coverage,bool(corners),c['patch_id']))
        hits = [v for v in matches if v[0]>=.5 or v[1]]
        rows.append({'reference_id':mark['patch_id'],'retained':mark['patch_id'] in usable,
                     'visible_polygon_closed':mark['outline_closed'],
                     'max_polygon_coverage':max((v[0] for v in matches),default=0),
                     'hit':bool(hits),'matched_candidate_ids':[v[2] for v in hits]})
    return rows


def fit(c):
    """Geometry-only grid fit. Solar path/north are never used to select d."""
    sid = c['sid']; p = c['projection']; t = p['transform']
    sun = V.read(V.CACHE/sid/'sun_snapshot.json')
    windows = [w for w in sun['windows'] if w['type']=='window' and w['room_id']==c['room_id']]
    small = 160
    target = cv2.resize(c['mask'].astype(np.uint8),(small,small),interpolation=cv2.INTER_NEAREST)>0
    room = cv2.resize(room_mask(sid,c['room_id'],t).astype(np.uint8),(small,small),interpolation=cv2.INTER_NEAREST)>0
    best = None
    def evaluate(w, azimuths, elevations):
        nonlocal best
        corners = V.window_corners(w)
        if corners[:,2].min() <= t['floor_z_m']:return
        for el in elevations:
            for az in azimuths:
                a,e = np.deg2rad([az,el])
                d = np.array([math.sin(a)*math.cos(e),math.cos(a)*math.cos(e),math.sin(e)])
                projected = corners-(corners[:,2:3]-t['floor_z_m'])/d[2]*d
                uv = V.floor_pixel(projected,t)*(small-1)/(t['size']-1)
                # Clip extreme coordinates before rasterization; 2deg minimum avoids infinity.
                uv = np.clip(np.rint(uv),-10000,10000).astype(np.int32)
                pred = np.zeros((small,small),np.uint8)
                cv2.fillPoly(pred,[uv],1)
                pred = (pred>0)&room
                union = np.count_nonzero(pred|target)
                iou = float(np.count_nonzero(pred&target)/union) if union else 0.
                if best is None or iou>best['iou']:
                    best = {'iou':iou,'window_id':w['id'],'model_azimuth_deg':float(az%360),
                            'elevation_deg':float(el),'direction_model_xyz':d.tolist()}
    for w in windows:evaluate(w,range(0,360,4),range(2,90,4))
    if best:
        az,el = best['model_azimuth_deg'],best['elevation_deg']
        w = next(w for w in windows if w['id']==best['window_id'])
        evaluate(w,np.arange(az-4,az+4.1,1),np.arange(max(1,el-4),min(89,el+4)+.1,1))
        best['accepted'] = best['iou']>=PARAMS['min_iou']
    return best


def horizon_proxy(sun, origin):
    """Azimuth horizon from cached neighbour/tree prisms; no full DSM claim."""
    az = np.deg2rad(np.arange(360))
    rays = np.column_stack([np.sin(az),np.cos(az)])
    heights = np.zeros(360)
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for o in sun['obstructions']:
        dz = o['top']-origin[2]
        if dz<=0:continue
        ring = np.array(o['ring_model_xy_m'])
        nearest = np.full(360,np.inf)
        for a,b in zip(ring,np.roll(ring,-1,axis=0)):
            edge=b-a; delta=a-origin[:2];den=cross(rays,edge)
            safe=np.abs(den)>1e-10
            den=np.where(safe,den,1)
            distance=cross(delta,edge)/den; u=cross(delta,rays)/den
            valid=safe&(distance>1e-5)&(u>=0)&(u<=1)
            nearest=np.minimum(nearest,np.where(valid,distance,np.inf))
        heights=np.maximum(heights,np.rad2deg(np.arctan2(dz,nearest)))
    return heights


def band_test(c,fit_result):
    sid=c['sid']; d=np.array(fit_result['direction_model_xyz']);sun=V.read(V.CACHE/sid/'sun_snapshot.json')
    frame=V.frames(sid)[1]
    en=frame@d[:2]; geo=V.model(sid)['geocoordinates'];lat=math.radians(float(geo['latitude']))
    north=np.array([0,math.cos(lat),math.sin(lat)])
    true=np.array([*en,d[2]])
    decl=math.degrees(math.asin(np.clip(true@north,-1,1)))
    exact_distance=max(0.,abs(decl)-23.44)
    t=c['projection']['transform'];xy=np.mean(np.argwhere(c['mask']),axis=0)[::-1]
    origin=V.floor_point(xy,t)
    h=horizon_proxy(sun,origin)
    az=np.arange(360); el=np.arange(0,90.01,1)
    aa,ee=np.meshgrid(np.deg2rad(az),np.deg2rad(el))
    local=np.stack([np.sin(aa)*np.cos(ee),np.cos(aa)*np.cos(ee),np.sin(ee)],-1)
    en_grid=local[...,:2]@frame.T
    world=np.concatenate([en_grid,local[...,2:3]],axis=-1)
    dec=np.rad2deg(np.arcsin(np.clip(world@north,-1,1)))
    valid=(abs(dec)<=23.44)&(np.rad2deg(ee)>h[None,:])
    samples=local[valid]
    proxy_distance=float(np.rad2deg(np.arccos(np.clip(np.max(samples@d),-1,1)))) if len(samples) else None
    modelaz=fit_result['model_azimuth_deg']
    above=fit_result['elevation_deg']>h[int(round(modelaz))%360]
    in_band=abs(decl)<=23.44 and above
    if in_band:proxy_distance=0.
    # Same per-row north rotations, preserving home correlation and horizon geometry.
    rotations=np.random.default_rng(PARAMS['seed']).uniform(0,2*np.pi,PARAMS['north_rotations'])
    rotated_n=en[1]*np.cos(rotations)-en[0]*np.sin(rotations)
    null_dec=np.rad2deg(np.arcsin(np.clip(math.cos(lat)*rotated_n+math.sin(lat)*d[2],-1,1)))
    null=abs(null_dec)<=23.44
    return {'latitude_deg':float(geo['latitude']),'declination_deg':decl,
            'annual_band_distance_no_obstructions_deg':exact_distance,
            'cached_prism_horizon_deg':float(h[int(round(modelaz))%360]),
            'above_cached_prism_horizon':bool(above),'in_annual_band_and_proxy_horizon':bool(in_band),
            'distance_to_proxy_visible_annual_band_deg_1deg_grid':proxy_distance,
            'null_north_rotation_hits':int(np.count_nonzero(null) if above else 0),
            'null_north_rotation_trials':1000,'null_chance':float(np.mean(null) if above else 0),
            'dsm_horizon_distance_deg':None,'dsm_status':'Full DSM horizon unavailable; cached neighbour/tree prisms are a proxy, exclude own roofs.'}


def contact_sheet(candidates, accepted):
    folder=ROOT/'data/claims/graph';folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(PARAMS['seed'])
    indices=rng.choice(len(accepted),min(50,len(accepted)),replace=False) if accepted else []
    picks=[accepted[int(i)] for i in indices]
    sheet=Image.new('RGB',(1500,max(1,math.ceil(len(picks)/5))*255+65),'white')
    draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=13)
    draw.text((10,8),f'SP baseline: {len(picks)} random IoU-accepted candidates; seed {PARAMS["seed"]}. Human count pending.',fill='black',font=font)
    draw.text((10,30),'Green contour: detected mask. Candidate IDs in sheet manifest. No precision estimate yet.',fill='black',font=font)
    manifest=[]
    for i,c in enumerate(picks):
        im=np.array(Image.open(V.OUT/c['sid']/f'{c["pano_id"]}_floor.jpg').convert('RGB'))
        contours,_=cv2.findContours(c['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im,contours,-1,(0,255,0),3)
        im=Image.fromarray(im).resize((230,230))
        x=(i%5)*300;y=(i//5)*255+65
        sheet.paste(im,(x,y));draw.text((x+2,y+231),f'{i+1}: {c["sid"]} IoU={c["fit"]["iou"]:.2f}',fill='black',font=font)
        manifest.append({'tile':i+1,'patch_id':c['patch_id']})
    if not picks:draw.text((10,75),'No IoU-accepted candidates.',fill='black',font=font)
    sheet.save(folder/'sun_patch_auto_sheet.jpg',quality=92)
    write(AUTO/'sheet_manifest.json',manifest)


def export_comparisons():
    """Unavailable model panels are labelled empty, never fabricated masks."""
    folder = AUTO / 'compare'
    folder.mkdir(parents=True, exist_ok=True)
    marks = V.read(V.OUT/'marks_v1.json')['patches']
    entries = []
    size = 384
    font = ImageFont.load_default(size=16)
    for row in V.read(V.OUT/'selection.json')['rows'][:15]:
        sid = row['sid']
        for path in sorted((V.OUT/sid).glob('*_projection.json')):
            pid = path.name.removesuffix('_projection.json')
            if not (AUTO/sid/pid/'detection.json').exists(): continue
            p = V.read(path)
            V.A.BASES = np.array(p['basis'])
            pos = np.array([p['pose']['position'][k] for k in ('x','y','z')])
            q = V.A.rotation(p['pose']['rotation'])
            segments = {f: [] for f in range(6)}
            for mark in marks:
                if (mark['sid'],mark['pano_id']) != (sid,pid): continue
                outline = mark['visible_outline']
                # Open outlines have no invented closing edge.
                edges = list(zip(outline,outline[1:]))
                if mark['outline_closed']: edges.append((outline[-1],outline[0]))
                for a,b in edges:
                    points = np.array([V.floor_point(uv,p['transform']) for uv in np.linspace(a,b,150)])
                    faces,uv = V.A.cube_coordinates(points-pos,q)
                    for j in range(len(points)-1):
                        if faces[j] == faces[j+1]:
                            segments[int(faces[j])].append((tuple(uv[j]*size),tuple(uv[j+1]*size)))
            for f in range(6):
                original = Image.open(V.CACHE/sid/pid/f'face{f}.jpg').convert('RGB').resize((size,size))
                baseline = np.array(original).copy()
                mask = np.array(Image.open(AUTO/sid/pid/f'face{f}_bright.png').resize((size,size),Image.Resampling.NEAREST))>0
                baseline[mask] = (.55*baseline[mask]+.45*np.array([0,255,100])).astype(np.uint8)
                panels = [original,original.copy(),original.copy(),Image.fromarray(baseline)]
                labels = ['Original','SAM 3: NOT RUN (HTTP 401)','Shadow model: NOT RUN','Luminance baseline (raw)']
                canvas = Image.new('RGB',(4*size,size+60),'white')
                draw = ImageDraw.Draw(canvas)
                for col,panel in enumerate(panels):
                    pd = ImageDraw.Draw(panel)
                    for segment in segments[f]: pd.line(segment,fill='yellow',width=3)
                    canvas.paste(panel,(col*size,60))
                    draw.text((col*size+5,5),labels[col],fill='black',font=font)
                draw.text((5,30),f'{sid} / {pid} / face {f} | yellow: frozen visible outline; green: threshold mask',fill='black',font=font)
                name = f'{sid}_{pid}_{f}.jpg'
                canvas.save(folder/name,quality=85)
                entries.append({'file':name,'sid':sid,'pano_id':pid,'face':f,'hand_marked':bool(segments[f])})
    entries.sort(key=lambda r: (not r['hand_marked'],r['file']))
    write(folder/'manifest.json',entries)
    cards = ''.join(f'<figure><a href="{html.escape(r["file"])}"><img loading="lazy" src="{html.escape(r["file"])}"></a><figcaption>{html.escape(r["file"])}{" — HAND MARK" if r["hand_marked"] else ""}</figcaption></figure>' for r in entries)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SP pilot comparisons</title><style>body{font:14px sans-serif;background:white;color:black}figure{margin:20px 0}img{width:100%;max-width:1536px}figcaption{overflow-wrap:anywhere}</style><h1>15-home pilot: visual comparison</h1><p>SAM 3 and shadow model did not run. Their panels contain the original with status labels, not predictions. Green is the untrained raw luminance mask before floor filtering; yellow is a frozen hand-marked visible outline. Marked faces appear first. Precision is pending human review.</p>'+cards)
    return entries


def record_comparisons(entries):
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = EVAL/'sun_patch_consistency.md'
    # Keep the numerical-run source hash separate when exporting saved outputs.
    base = report.read_text().split('Comparison: data/sun_patch/auto/compare/index.html;')[0]
    extra = ['Comparison: data/sun_patch/auto/compare/index.html; '+str(len(entries))+' face JPGs; unavailable model panels explicitly labelled NOT RUN.',
             'Faces needing reference review: '+', '.join(r['file'] for r in entries if r['hand_marked'])+'.',
             'Visual QA by Codex: sheet contains bright furniture, wall and window-frame matches; no human precision count inferred.',
             'Comparison exporter SHA256: `'+digest+'`; original numerical-run source hash above.']
    report.write_text(base+'\n'.join(extra)+'\n')
    result = V.read(EVAL/'sun_patch_consistency.json')
    result['comparisons'] = {'index':'data/sun_patch/auto/compare/index.html','faces':len(entries),'hand_marked_faces':[r['file'] for r in entries if r['hand_marked']], 'exporter_script_sha256':digest}
    result['visual_qa'] = 'Codex inspected sheet: bright furniture, wall and window-frame matches are present. No human precision count inferred.'
    write(EVAL/'sun_patch_consistency.json',result)


def main():
    socket.socket=blocked; socket.create_connection=blocked
    AUTO.mkdir(parents=True,exist_ok=True);EVAL.mkdir(parents=True,exist_ok=True)
    write(AUTO/'parameters.json',PARAMS)
    sids=[r['sid'] for r in V.read(V.OUT/'selection.json')['rows'][:15]]
    all_candidates=[]; inventory=[]
    for sid in sids:
        projections=sorted((V.OUT/sid).glob('*_projection.json'))
        home=[]
        for path in projections:
            pid=path.name.removesuffix('_projection.json')
            if not all((V.CACHE/sid/pid/f'face{k}.jpg').exists() for k in range(6)):continue
            home+=detect(sid,pid,V.read(path))
        all_candidates+=home
        inventory.append({'sid':sid,'panos':len(projections),'candidate_count':len(home),
                          'registration_quality':'poor' if V.read(V.CACHE/sid/'sun_snapshot.json')['registration']['poor'] else 'well'})
        print(json.dumps(inventory[-1]),flush=True)
        progress(f'Detection completed {len(inventory)}/15 homes, {len(all_candidates)} baseline candidates; masks and detection metadata saved. No learned shadow checkpoint available; SAM3 HTTP401.','Complete pilot/reference recall, then geometry fits and null baseline. No scale before gate.')
    recall=reference_recall(all_candidates)
    write(AUTO/'reference_recall.json',recall)
    progress(f'15-home detection finished. Reference results: {json.dumps(recall)}','Fit all pilot baseline candidates independently of solar path; report limitations and no scale on failed gate.')
    accepted=[]
    for i,c in enumerate(all_candidates):
        c['fit']=fit(c)
        if c['fit'] and c['fit']['accepted']:
            c['band']=band_test(c,c['fit']);accepted.append(c)
        print(json.dumps({'fitted':i+1,'total':len(all_candidates),'accepted':len(accepted)}),flush=True)
        if (i+1)%10==0:progress(f'15-home detection done; fitted {i+1}/{len(all_candidates)}, {len(accepted)} IoU>=0.3 candidates.','Finish geometry and annual band/null analysis; write report/contact sheet.')
    contact_sheet(all_candidates,accepted)
    comparison_entries = export_comparisons()
    serial=[{k:v for k,v in c.items() if k not in ('mask','projection')} for c in all_candidates]
    retained=[r for r in recall if r['retained']]
    gate=all(r['hit'] for r in retained) and len(retained)==4
    reference_panos={(p['sid'],p['pano_id']) for p in V.read(V.OUT/'marks_v1.json')['patches'] if p['patch_id'] in {r['reference_id'] for r in retained}}
    matched={p for r in retained for p in r['matched_candidate_ids']}
    unmatched=[c['patch_id'] for c in all_candidates if (c['sid'],c['pano_id']) in reference_panos and c['patch_id'] not in matched]
    groups={}
    for quality in ('well','poor'):
        group=[c for c in accepted if ('poor' if V.read(V.CACHE/c['sid']/'sun_snapshot.json')['registration']['poor'] else 'well')==quality]
        groups[quality]={'accepted_patches':len(group),'homes':len({c['sid'] for c in group}),
                         'in_proxy_band':sum(c['band']['in_annual_band_and_proxy_horizon'] for c in group),
                         'mean_null_chance':float(np.mean([c['band']['null_chance'] for c in group])) if group else None}
    result={'detector':'SP-luminance-edge-v1; OpenCV '+cv2.__version__+'; no learned checkpoint',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parameters':PARAMS,
            'sam3':{'repo':'meta/sam3','status':'gated config request HTTP401; not run'},
            'shadow_model':{'status':'SILT/SDDNet public model/space searches returned no checkpoint; not run. This is an untrained baseline.'},
            'device':'CPU (OpenCV); torch MPS unavailable in this execution environment',
            'network':'Inference guarded offline; only pip/HuggingFace model discovery used; no Matterport/Vexcel calls or .env reads',
            'pilot_inventory':inventory,'reference_recall':recall,'reference_gate_passed':gate,
            'reference_recalled':sum(r['hit'] for r in retained),'reference_total':len(retained),
            'scale_to_169':False,'scale_reason':'Learned detector unavailable; baseline pilot is diagnostic only, not approved as reliable auto-labels.',
            'unmatched_candidates_on_reference_panos':unmatched,
            'false_positive_count':None,'false_positive_status':'Incomplete positive marks; unmatched candidates require human check and cannot all be called false positives.',
            'precision':None,'precision_status':'Human count pending; sheet samples up to 50 IoU-accepted candidates, not verified patches.',
            'registration_groups':groups,'rows':serial}
    write(EVAL/'sun_patch_consistency.json',result)
    lines=[f'# SP-luminance-edge-v1 | OpenCV {cv2.__version__} | checkpoint: none (untrained baseline)',
           f'Source: experiments/sun_patch_consistency.py SHA256 `{result["script_sha256"]}`.',
           'SAM 3 (`meta/sam3`) config returned HTTP 401; no licence workaround; SAM 3 not run.',
           'No public SILT/SDDNet checkpoint found through bounded HuggingFace model/space discovery; neither trained shadow model ran.',
           'CPU run; torch MPS unavailable. Offline socket guard; no .env reads or Matterport/Vexcel calls.',
           f'Pilot: 15 cached homes, {sum(r["panos"] for r in inventory)} panos, {len(all_candidates)} filtered baseline masks.',
           f'Retained reference recall: {sum(r["hit"] for r in retained)}/4; gate {"PASS for baseline only" if gate else "FAIL"}.',
           'Recall: >=50% of visible corner polygon covered OR all marked corners inside one candidate mask; outlines are incomplete.',
           *[f'- {r["reference_id"]}: {"hit" if r["hit"] else "miss"}; max visible-polygon coverage {r["max_polygon_coverage"]:.3f}; retained={r["retained"]}.' for r in recall],
           f'Unmatched candidates on retained-reference panos: {len(unmatched)}; not verified false positives (positive marks are incomplete).',
           'No measured false-positive count or precision: requires human review; seven previously unreviewed homes remain unlabeled.',
           'No expansion to 169 homes: learned detector unavailable; these baseline candidates are diagnostic.',
           f'Geometry: {len(accepted)}/{len(all_candidates)} masks IoU >=0.3; all windows in same room searched independently of sun band.',
           'Grid: model azimuth 0..356 by 4deg, elevation 2..86 by 4deg; best window refined at 1deg; room-clipped aperture masks at 160px.',
           'Directions are unconstrained by solar path. Foreground furniture and hidden/clipped aperture edges remain unmodelled.',
           'Analytic annual declination band +/-23.44deg; cached prism horizon proxy uses neighbour/tree silhouettes at the patch centroid.',
           'Full DSM horizon unavailable: DSM-specific distance is null. Proxy omits own roof; no full DSM claim.',
           'Proxy distance uses 1deg sky grid (up to ~1.5deg discretization); no-obstruction distance is analytic.',
           'Null: same 1,000 seeded uniform north rotations per row; horizon fixed in model frame; correlated views are not independent homes.',
           *[f'{q}: {g["accepted_patches"]} accepted candidates / {g["homes"]} homes; {g["in_proxy_band"]} in proxy band; mean null chance {g["mean_null_chance"]}.' for q,g in groups.items()],
           'Pilot selection was previously restricted to well-registered homes; poor-registration comparison is unavailable.',
           f'Contact sheet: data/claims/graph/sun_patch_auto_sheet.jpg, {min(50,len(accepted))} randomly selected candidates (no repeats); manifest under auto/.',
           'Human count pending: precision remains null; a sheet of fewer than 50 means fewer candidates passed IoU.',
           'Interpretation only: direction implied by a detected patch is consistent with the possible sun path.',
           'No sun-minutes validation. Detection gate/model and DSM limitations prevent a geometry-validation conclusion.',
           'Reproduce: `.venv-agent/bin/python -I experiments/sun_patch_consistency.py`.',
           'Original sun_patch_check.py, frozen marks, demo/ and agent/ unchanged. New source authored by Codex.',
           'Undo source: `python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path experiments/sun_patch_consistency.py`; rollback ID dry run, then --yes.',
           'Guard suite results: see data/sun_patch/auto/existing_tests.log.']
    (EVAL/'sun_patch_consistency.md').write_text('\n'.join(lines)+'\n')
    record_comparisons(comparison_entries)
    progress(f'Pilot complete: 15 homes, {len(all_candidates)} candidates, {len(accepted)} IoU-accepted; recall {sum(r["hit"] for r in retained)}/4. Reports data/claims/eval/sun_patch_consistency.md/.json; masks data/sun_patch/auto/; contact sheet data/claims/graph/sun_patch_auto_sheet.jpg.','Run existing guard suite and deterministic replay; inspect contact sheet; report limitations. No 169-home scale.')


if __name__=='__main__':
    if '--export-comparisons-only' in sys.argv:
        socket.socket=blocked; socket.create_connection=blocked
        entries = export_comparisons()
        record_comparisons(entries)
        progress('Pilot complete; comparison face JPGs/index exported. Existing guard suite: 7 passed. Recall 0/4; no expansion.','Visually inspect generated artifacts and verify report consistency.')
    else:
        main()
