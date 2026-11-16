"""python3 -I demo/build_data.py [--online]. Default: cache-only, no network."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode=True
sys.dontwritebytecode=True
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
SITE=BASE/'site'

def module(name):
    spec=importlib.util.spec_from_file_location('demo_'+name,BASE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

api=module('api_cache')
adapter=module('engine_adapter')

def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False))

def listing_index():
    result={}
    for line in (ROOT/'data/costar-kit/data/residential-listings.jsonl').open():
        obj=json.loads(line)
        for key in ('id','residentialListingId'):
            if obj.get(key) is not None: result[str(obj[key])]=obj
    return result

def choose_featured(rows,models,listings):
    eligible=[]
    for row in rows:
        sid=row['sid']; sunfile=ROOT/'data/sun'/sid/'sun.json'
        if not sunfile.exists() or row.get('registration_poor')!='False': continue
        if not listings.get(models[sid]['homes_listing_id'],{}).get('remarks'): continue
        if not row.get('living_room_dec_minutes') or not row.get('living_room_jun_minutes'): continue
        sun=read(sunfile)
        if not any(w['type']=='window' for w in sun['windows']):continue
        eligible.append(row)
    selected=['khbuN7q853q']; why={'khbuN7q853q':'Required: independent hand-marked sill check'}
    rules=[('lowest winter minutes',lambda r:float(r['living_room_dec_minutes'])),
        ('highest winter minutes',lambda r:-float(r['living_room_dec_minutes'])),
        ('highest summer minutes',lambda r:-float(r['living_room_jun_minutes'])),
        ('largest summer-minus-winter gain',lambda r:float(r['living_room_dec_minutes'])-float(r['living_room_jun_minutes'])),
        ('nearest cohort median winter minutes',lambda r:abs(float(r['living_room_dec_minutes'])-float(np.median([float(e['living_room_dec_minutes']) for e in eligible]))))]
    for label,key in rules:
        pool=[r for r in eligible if r['sid'] not in selected]
        if not pool:break
        chosen=min(pool,key=lambda r:(key(r),r['sid']))
        selected.append(chosen['sid']);why[chosen['sid']]=label
    return selected,why

def ortho(sid,data,model):
    tr=data['transform'];lat=tr['origin_lat'];lon=tr['origin_lon']
    m=np.asarray(tr['registered_matrix']);shift=np.asarray(tr['registered_shift'])
    grid=np.concatenate([np.asarray(g['grid']) for g in data['geometry'].values()])
    center=(grid.min(0)+grid.max(0))/2@m.T+shift
    bounds=[float(center[0]-30),float(center[1]-30),float(center[0]+30),float(center[1]+30)]
    ll=lambda e,n:(lon+e/(111320*np.cos(np.deg2rad(lat))),lat+n/110574)
    ring=[ll(bounds[i],bounds[j]) for i,j in [(0,1),(2,1),(2,3),(0,3),(0,1)]]
    wkt='POLYGON(('+','.join(f'{x:.11f} {y:.11f}' for x,y in ring)+'))'
    image=BASE/'cache'/sid/'ortho.jpg'
    api.request(Path(sid)/'ortho.jpg','ortho/extract',body={'wkt':wkt,'srid':4326,'layer':'urban','crop':'bounding_box','image-format':'jpeg','buffer-meter':0})
    metadata=read(image.with_suffix('.metadata.json')) if image.with_suffix('.metadata.json').exists() else {}
    requested_bounds=bounds[:]
    if metadata.get('geometry'):
        actual=np.array([adapter.engine.G.enu(la,lo,lat,lon) for lo,la in adapter.engine.G.parse_ring(metadata['geometry'])])
        bounds=[float(actual[:,0].min()),float(actual[:,1].min()),float(actual[:,0].max()),float(actual[:,1].max())]
    target=SITE/'img'/sid;target.mkdir(parents=True,exist_ok=True)
    with Image.open(image) as im:
        im.convert('RGB').save(target/'aerial.jpg',quality=92)
        size=list(im.size)
        thumb=im.convert('RGB')
    # Crop is explicitly north-up, requested bounding box, no buffer or rotation.
    ground={'image':f'img/{sid}/aerial.jpg','bounds_enu_m':bounds,'bounds_lonlat':ring,
        'size_px':size,'north_up':True,'origin':'transform.origin_lat/lon',
        'requested_bounds_enu_m':requested_bounds,
        'bounds_source':'Returned Vexcel XMP geometry (pixel-snapped extent), EPSG:4326 north-up bounding box' if metadata.get('geometry') else 'Explicit EPSG:4326 crop polygon, bounding_box, buffer-meter=0',
        'pixel_mapping':'u=(east-west)/(east_bound-west)*width; v=(north_bound-north)/(north_bound-south)*height',
        'request_wkt':wkt,'metadata':metadata}
    draw=ImageDraw.Draw(thumb)
    def px(p):
        e,n=np.asarray(p)@m.T+shift
        return ((e-bounds[0])/(bounds[2]-bounds[0])*size[0],(bounds[3]-n)/(bounds[3]-bounds[1])*size[1])
    for room in data['geometry'].values():
        for a,b in room['segments']:draw.line([px(a),px(b)],fill='#11e09b',width=max(2,round(size[0]/350)))
    thumb.thumbnail((420,420));thumb.save(target/'card.jpg',quality=88)
    return ground

def method():
    camera=read(ROOT/'data/shadow_check/khbuN7q853q/camera.json')
    geo=read(ROOT/'data/geo_check/results.json')['homes']
    reproj=read(ROOT/'data/reproj/khbuN7q853q/reprojection.json')
    ref=reproj['refinement']
    metrics={'camera_held_out_px':max(i['projection_held_out_error_px'] for i in camera['images']),
        'camera_fit_px':max(i['projection_max_error_px'] for i in camera['images']),
        'footprint_homes':len(geo),'overlap_before':float(np.median([h['iou_before'] for h in geo])),
        'overlap_after':float(np.median([h['iou_after'] for h in geo])),
        'median_shift_m':float(np.median([h['fix_shift_m'] for h in geo])),
        'median_rotation_deg':float(np.median([abs(h['heading_offset_deg']) for h in geo])),
        'median_applied_grid_rotation_deg':float(np.median([abs(h['fix_rot_deg']) for h in geo])),
        'hand_marked_points':ref['training_corner_count']+ref['held_out_corner_count'],
        'hand_marked_held_out_points':ref['held_out_corner_count'],
        'hand_marked_rms_m':ref['nominal_7p5cm_gsd_equivalent_rms_m']['held_out_other_direction'],
        'refinement_accepted':ref['accepted'],
        'sources':['data/shadow_check/khbuN7q853q/camera.json','data/geo_check/results.json','data/reproj/khbuN7q853q/reprojection.json']}
    imgs=[]
    for name in ['overlay_west-6.jpg','overlay_south-1.jpg','overlay_east-1.jpg','both_sides.jpg']:
        src=ROOT/'data/reproj/khbuN7q853q'/name
        if src.exists():
            dest=SITE/'img/method'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);imgs.append('img/method/'+name)
    metrics['images']=imgs
    write(SITE/'data/method.json',metrics)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--online',action='store_true')
    default_cohort=BASE/'cache/cohort_index.csv'
    parser.add_argument('--cohort',type=Path,default=default_cohort,help='Input index CSV; defaults to the frozen demo cohort. The required Sara Rose home is always retained.')
    args=parser.parse_args()
    api.ONLINE=args.online
    # The producer updates its index during a run. Keep this demo's cohort
    # stable unless a different input is explicitly requested. External input
    # files are read-only; only the default cache snapshot can be created.
    cohort=args.cohort.resolve();default_cohort=default_cohort.resolve()
    if not cohort.exists() and cohort==default_cohort:
        cohort.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'data/sun/index.csv',cohort)
        write(BASE/'cache/cohort_provenance.json',{'source':'data/sun/index.csv','capture':'Direct file snapshot'})
    if not cohort.is_file():parser.error('Cohort index does not exist')
    rows=list(csv.DictReader(cohort.open()))
    required_home_appended=not any(r['sid']=='khbuN7q853q' for r in rows)
    if required_home_appended:
        frozen=list(csv.DictReader(default_cohort.open())) if default_cohort.exists() else []
        required=next((r for r in frozen if r['sid']=='khbuN7q853q'),None)
        if required is None:required=read(SITE/'data/khbuN7q853q.json')['selection']
        rows.append(required)
    models={r['sid']:r for r in csv.DictReader((ROOT/'models_index.csv').open())}
    listings=listing_index();featured,why=choose_featured(rows,models,listings)
    print('cohort',len(rows),'featured',','.join(featured),flush=True)
    cards=[];failures=[]
    for row in rows:
        sid=row['sid'];cache=BASE/'cache'/sid
        try:
            inputs=[ROOT/'data/sun'/sid/f for f in ['sun.json','sample_verdicts.json']]+[ROOT/'data/cache'/sid/f for f in ['matterport.json','registration.json']]
            signature=hashlib.sha256((adapter.SOURCE_SHA+'|'+(BASE/'engine_adapter.py').read_text()).encode()+b''.join(p.read_bytes() for p in inputs)).hexdigest()
            derived=cache/'derived.json';manifest=cache/'derived.sha256'
            if derived.exists() and manifest.exists() and manifest.read_text()==signature:
                data=read(derived);model=read(inputs[2])['data']['model']
            else:
                data,model=adapter.build(sid);write(derived,data);manifest.write_text(signature)
            listing=listings.get(models[sid]['homes_listing_id'],{})
            remarks=listing.get('remarks') or ''
            data['listing']={'id':models[sid]['homes_listing_id'],'address':models[sid]['address'],
                'price':listing.get('currentPrice'),'currency':listing.get('monetaryUnit') or 'USD','remarks':remarks,
                'status':'present' if listing else 'No matching listing in supplied residential-listings.jsonl'}
            data['claims']=None
            claimpath=ROOT/'data/claims'/f'{sid}.json'
            if claimpath.exists():
                claim=read(claimpath)
                if claim.get('sid')==sid and str(claim.get('listing_id'))==data['listing']['id']:
                    data['claims']=claim
                else: failures.append({'sid':sid,'stage':'claims','reason':'SID/listing mismatch; omitted'})
            data['featured']=sid in featured;data['featured_reason']=why.get(sid)
            data['aerial']=ortho(sid,data,model)
            data['selection']=row
            out=SITE/'data'/f'{sid}.json'
            # Preserve previously generated local crops when rebuilding the data.
            asset=cache/'window_assets.json'
            if asset.exists():data['window_assets']=read(asset)
            write(out,data)
            main_id=row.get('living_room_room_id')
            main=next((r for r in data['rooms'] if r['id']==main_id),None)
            cards.append({'sid':sid,'address':data['listing']['address'],'price':data['listing']['price'],
                'featured':data['featured'],'featured_reason':why.get(sid),'poor':bool(data['registration']['poor']),
                'registration_iou':data['registration']['iou_after'],'shift_m':data['registration']['fix_shift_m'],
                'rotation_deg':data['registration']['fix_rot_deg'],'mentions_light':bool(re.search(r'\b(light|bright|sunny|sunlight|sunlit)\b',remarks,re.I)),
                'main_room_id':main_id or None,'winter':main['dates']['2026-12-21']['lit_minutes_ge_0_1m2'] if main else None,
                'summer':main['dates']['2026-06-21']['lit_minutes_ge_0_1m2'] if main else None,
                'rooms':[{k:r[k] for k in ['id','label','classifications','dates']} for r in data['rooms']],
                'image':f'img/{sid}/card.jpg','data':f'data/{sid}.json'})
            print('ready',sid,len(data['rooms']),'rooms',flush=True)
        except Exception as error:
            reason=str(error)
            if any(x in reason.lower() for x in ['https:','http:','key','token','authorization']): reason=type(error).__name__+'; details suppressed'
            failures.append({'sid':sid,'stage':'build','reason':reason});print('unavailable',sid,reason,flush=True)
            if api.AUTH_FAILURES>=2: break
    selection_rule='From the supplied index: non-poor registration, listing remarks, main-living winter and summer totals, at least one actual window. Without replacement: lowest winter, highest winter, highest summer, largest summer-minus-winter gain, nearest eligible-cohort median winter; SID breaks ties. Plus required khbuN7q853q.'
    write(SITE/'data/index.json',{'homes':cards,'dates':adapter.engine.DATES,'featured':featured,'selection_rule':selection_rule,'engine_sha256':adapter.SOURCE_SHA,'failures':failures})
    method()
    assets=module('featured_assets')
    coverage=[]
    for sid in featured:
        path=SITE/'data'/f'{sid}.json'
        if not path.exists():continue
        data=read(path)
        try:
            data['window_assets']=assets.build(sid,data,api,adapter.engine)
            write(path,data)
            views=data['window_assets']
            coverage.append({'sid':sid,'openings':len(data['windows']),
                'inside':sum(bool(views.get(w['id'],{}).get('inside')) for w in data['windows']),
                'outside':sum(bool(views.get(w['id'],{}).get('outside')) for w in data['windows']),
                'gaps':[{'window_id':w['id'],'view':view,'reason':views.get(w['id'],{}).get(view+'_status','Unavailable')}
                    for w in data['windows'] for view in ['inside','outside'] if not views.get(w['id'],{}).get(view)]})
        except Exception as error:
            # Do not echo network exception strings, which can contain signed URLs.
            failures.append({'sid':sid,'stage':'featured assets','reason':type(error).__name__+'; inspect local status manifest'})
            print('featured asset gap',sid,type(error).__name__,flush=True)
    write(BASE/'build_report.json',{'homes':len(cards),'expected_cohort':len(rows),'featured':[{'sid':s,'address':models[s]['address'],'rule':why[s]} for s in featured],
        'selection_rule':selection_rule,'cohort_snapshot':str(cohort.relative_to(ROOT)) if cohort.is_relative_to(ROOT) else str(cohort),
        'cohort_sha256':hashlib.sha256(cohort.read_bytes()).hexdigest(),'required_home_appended':required_home_appended,
        'failures':failures,'featured_asset_coverage':coverage,'network_calls':api.CALLS,'engine_sha256':adapter.SOURCE_SHA,
        'missing_engine_api':'No consolidated frame accessor; adapter composes existing room_data, exterior_openings, sample_day and floor_patch; no engine edits.'})
    print('done',len(cards),'homes; network calls',api.CALLS,'gaps',len(failures),flush=True)

if __name__=='__main__':
    sys.path.insert(0,str(BASE))
    import round2_builder
    round2_builder.main()
