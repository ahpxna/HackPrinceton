"""Fresh facts and claims on every run; expensive engine cells cached separately."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import re
import time
import build_data as old
from round3_data import contained, export_replay

BASE,ROOT,SITE=old.BASE,old.ROOT,old.SITE
DATES=['2026-12-21','2026-03-20','2026-06-21','2026-11-14']
FEATURED=['khbuN7q853q','16ZWm6HRVjS','1FFftrgRFBd','VN4b1MGnnCR','LQLjZqrmKqP','HFwcH424n9w']
TYPES=['any','main','living','bedroom','primary','kitchen','dining','bathroom']
VERDICTS=['supported','contradicted','unclear','not_checkable']
EXCLUDE=re.compile(r'void|stair|closet|garage|patio|porch|balcony|deck|outdoor|yard|terrace|storage|attic|crawl',re.I)

def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))
def num(value):
    try:
        x=float(value);return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None
def write(path,value):old.write(path,value)
def objects(name):
    return {str(x['id']):x for x in map(json.loads,(ROOT/f'data/costar-kit/data/{name}.jsonl').open())}

def geometry_job(sid):
    paths=[ROOT/'data/sun'/sid/f for f in ['sun.json','sample_verdicts.json']]+[ROOT/'data/cache'/sid/f for f in ['matterport.json','registration.json']]
    signature=hashlib.sha256((old.adapter.SOURCE_SHA+(BASE/'engine_adapter.py').read_text()).encode()+b''.join(p.read_bytes() for p in paths)).hexdigest()
    path=BASE/'cache'/sid/'round2-cells.json.gz';stamp=path.with_suffix('.sha256')
    try:
        hit=path.exists() and stamp.exists() and stamp.read_text()==signature
        if not hit:
            data,_=old.adapter.build(sid)
            path.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(path,'wt',compresslevel=2) as stream:json.dump(data,stream,separators=(',',':'),allow_nan=False)
            stamp.write_text(signature)
        return sid,signature,None,hit
    except Exception as error:return sid,signature,type(error).__name__+': '+str(error)[:150],False

def listing(row,listings,features):
    lid=str(row['listing_id']);source=listings.get(lid,{})
    feature=features.get(lid,{})
    prop=source.get('property') or {}
    return {'id':lid,'address':row['address'],'city':row['city'],'price':source.get('currentPrice'),
        'remarks':source.get('remarks') or '',
        'highlights':[x['text'] for x in feature.get('highlights',[]) if x.get('text')],
        'features':[{k:x[k] for k in ['name','doubleValue','textValue','boolValue'] if k in x} for x in feature.get('features',[])],
        'fields':{k:prop.get(k) for k in ['bedsMin','bedsMax','bathsMin','bathsMax','bathsFull','bathsHalf','sqftMin','sqftMax']}}

def choose_heavy(summary,computed):
    chosen=list(FEATURED);reasons={s:'Original featured home; accepted Round 1 selection retained.' for s in chosen}
    topics=list(summary['topics'])
    priority=['ceilings','light','open_plan','bedrooms','garage','pool']
    topics=priority+[t for t in topics if t not in priority]
    # Preserve each supplied clearest-case order, round-robin across topics and verdicts.
    for rank in range(20):
        for verdict in ['contradicted','supported']:
            for topic in topics:
                cases=summary['topics'][topic].get('clearest',{}).get(verdict,[])
                if rank>=len(cases):continue
                sid=cases[rank]['sid']
                if sid not in computed or sid in chosen:continue
                chosen.append(sid);reasons[sid]=f'Supplied clearest {verdict} case for {topic}, rank {rank+1}.'
                if len(chosen)==36:return chosen,reasons
    return chosen,reasons

def ceiling_index():
    result={}
    for r in rows(ROOT/'data/audit/matterport_rooms.csv'):
        classes=[x.get('label','') for x in json.loads(r.get('all_classifications') or '[]')]
        label=r.get('label') or ' / '.join(classes) or r.get('classification') or ''
        h=num(r.get('ceiling_max_m'));h=h if h is not None else num(r.get('height_m'))
        if classes and any(c not in ['Other','Void'] for c in classes) and not EXCLUDE.search(label) and h is not None and 1.8<=h<=8:
            result[r['sid']]=max(result.get(r['sid'],0),h)
    return result

def sun_index(row,rooms):
    poor=row['poor']!='False' or row['computed']!='True'
    result={}
    if poor:return result
    for typ in TYPES:
        selected=[r for r in rooms if typ=='any' or (typ=='main' and r['room_id']==row['main_living_room_id']) or
            (typ=='primary' and r['bedroom_role']=='primary_bedroom') or
            (typ=='living' and re.search(r'living|family|great room',(r['label']+' '+r['top_classification']),re.I)) or
            (typ not in ['main','primary','any','living'] and typ in (r['label']+' '+r['top_classification']).lower())]
        if not selected:continue
        values=[]
        for date in DATES:
            for period in ['sun_minutes','before_noon_minutes','after_noon_minutes']:
                numbers=[num(r.get(date.replace('-','')+'_'+period)) for r in selected]
                numbers=[n for n in numbers if n is not None]
                values.append(round(max(numbers),2) if numbers else None)
        result[typ]=values
    return result

def claim_outlines(sid,data,audit):
    """Keep real scan boundaries for evidence rooms omitted by the sun calculation."""
    data['audit_rooms']=audit
    needed={e['room_id'] for c in (data.get('claims') or {}).get('claims',[]) for e in c.get('evidence',[]) if e.get('room_id')}
    missing=needed-set(data.get('geometry',{}));outlines={}
    if missing and data.get('computed'):
        model=old.read(ROOT/'data/cache'/sid/'matterport.json')['data']['model']
        for room in model.get('rooms') or []:
            if room['id'] not in missing:continue
            segments=old.adapter.engine.segments(room.get('boundary'))
            floor=(room.get('floor') or {}).get('id')
            if len(segments)<3 or not floor:continue
            labels=[r['label'] for r in audit if r['id']==room['id']]
            outlines[room['id']]={'segments':segments.round(5).tolist(),
                'holes':[old.adapter.engine.segments(h).round(5).tolist() for h in room.get('holes') or []],
                'label':labels[0] if labels else room.get('label') or 'Room', 'floor_id':floor}
    data['claim_room_outlines']=outlines
    data['plan_room_ids']=list(data.get('geometry',{}))+list(outlines)

def main():
    start=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online',action='store_true');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--skip-assets',action='store_true',help='Refresh all facts, claims and cells without photo acquisition.')
    args=parser.parse_args();old.api.ONLINE=args.online
    input_paths=[ROOT/p for p in ['data/facts/homes.csv','data/facts/rooms.csv','data/facts/windows.csv','data/facts/summary.json','data/claims/summary.json','data/audit/matterport_rooms.csv','data/costar-kit/data/residential-listings.jsonl','data/costar-kit/data/residential-listing-features.jsonl']]
    input_digests={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    homes=rows(ROOT/'data/facts/homes.csv');room_rows=defaultdict(list)
    for r in rows(ROOT/'data/facts/rooms.csv'):room_rows[r['sid']].append(r)
    audit=defaultdict(list)
    for r in rows(ROOT/'data/audit/matterport_rooms.csv'):audit[r['sid']].append({'id':r['room_id'],'label':r['label']})
    summary=old.read(ROOT/'data/claims/summary.json');facts=old.read(ROOT/'data/facts/summary.json')
    listings=objects('residential-listings');features=objects('residential-listing-features');ceilings=ceiling_index()
    computed={r['sid'] for r in homes if r['computed']=='True' and (ROOT/'data/sun'/r['sid']/'sun.json').exists()}
    heavy,reasons=choose_heavy(summary,computed);topics=list(summary['topics']);cards=[];failures=[];hits=0
    print('selected',len(homes),'computed',len(computed),'heavy',len(heavy),flush=True)
    signatures={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for i,(sid,signature,error,hit) in enumerate(pool.map(geometry_job,sorted(computed),chunksize=4)):
            if error:failures.append({'sid':sid,'stage':'engine cells','reason':error})
            else:signatures[sid]=signature
            hits+=int(hit)
            if i%100==0:print('cells',i+1,'cached',hits,'failures',len(failures),flush=True)
    cells_seconds=time.monotonic()-start
    coverage=[];export_hits=0
    build_digest=hashlib.sha256(Path(__file__).read_bytes()+(BASE/'build_data.py').read_bytes()).hexdigest()
    for i,row in enumerate(homes):
        sid=row['sid'];lst=listing(row,listings,features);claimpath=ROOT/'data/claims'/f'{sid}.json'
        claims=old.read(claimpath) if claimpath.exists() else None
        if claims and (claims.get('sid')!=sid or str(claims.get('listing_id'))!=lst['id']):
            failures.append({'sid':sid,'stage':'claims','reason':'Primary listing mismatch; omitted'});claims=None
        status='poor' if row['poor']=='True' else 'well' if row['poor']=='False' else 'unknown'
        packed=SITE/'data'/f'{sid}.json.gz';export_stamp=BASE/'cache'/sid/'round2-export.sha256'
        export_digest=hashlib.sha256(json.dumps([build_digest,row,lst,claims,audit[sid],signatures.get(sid)],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        reuse=sid not in heavy and packed.exists() and export_stamp.exists() and export_stamp.read_text()==export_digest
        export_hits+=int(reuse)
        data={'sid':sid,'rooms':[],'windows':[],'geometry':{},'days':{},'computed':sid in signatures}
        if sid in signatures and not reuse:
            with gzip.open(BASE/'cache'/sid/'round2-cells.json.gz','rt') as stream:data=json.load(stream)
            data['computed']=True
        data.update(listing=lst,claims=claims,featured=sid in FEATURED,heavy_assets=sid in heavy,
            featured_reason=reasons.get(sid),placement_status=status,selection=row,aerial=None,contained_partial=contained(row))
        if sid in heavy and sid in signatures:
            try:data['aerial']=old.ortho(sid,data,None)
            except Exception as error:failures.append({'sid':sid,'stage':'aerial','reason':type(error).__name__})
            asset=BASE/'cache'/sid/'window_assets.json'
            if asset.exists():data['window_assets']=old.read(asset)
            if not args.skip_assets and data['aerial'] and old.api.AUTH_FAILURES<2:
                # A changed placement invalidates projected exterior crops, never indoor crops.
                stamp=BASE/'cache'/sid/'photo-placement.sha256'
                photo_hash=hashlib.sha256(json.dumps(data['transform'],sort_keys=True).encode()).hexdigest()
                if not stamp.exists() or stamp.read_text()!=photo_hash:
                    records=data.get('window_assets',{})
                    for record in records.values():
                        for k in list(record):
                            if k.startswith('outside') or k=='outdoor_projection':record.pop(k)
                    write(asset,records)
                assets=old.module('featured_assets')
                data['window_assets']=assets.build(sid,data,old.api,old.adapter.engine)
                stamp.write_text(photo_hash)
            views=data.get('window_assets',{})
            coverage.append({'sid':sid,'openings':len(data['windows']),
                'inside':sum(bool(views.get(w['id'],{}).get('inside')) for w in data['windows']),
                'outside':sum(bool(views.get(w['id'],{}).get('outside')) for w in data['windows'])})
        # Keep all audited room identities, including rooms omitted by the sun engine.
        if not reuse:
            claim_outlines(sid,data,audit[sid])
            temporary=packed.with_suffix('.gz.tmp')
            with gzip.open(temporary,'wb',compresslevel=5) as stream:stream.write(json.dumps(data,separators=(',',':'),allow_nan=False).encode())
            temporary.replace(packed)
            export_stamp.parent.mkdir(parents=True,exist_ok=True);export_stamp.write_text(export_digest)
            # The lossless packed copy replaces only this generated home's old export.
            (SITE/'data'/f'{sid}.json').unlink(missing_ok=True)
        verdicts={}
        for c in (claims or {}).get('claims',[]):
            if c.get('topic') in topics and c.get('verdict') in VERDICTS:
                t=str(topics.index(c['topic']));verdicts[t]=verdicts.get(t,0)|(1<<VERDICTS.index(c['verdict']))
        price=lst['price'];price=price.get('amount') if isinstance(price,dict) else price
        parcel_path=ROOT/'data/cache'/sid/'vexcel_calculated_65m.json'
        parcels=[x.get('properties',{}) for x in old.read(parcel_path).get('features',[]) if x.get('properties',{}).get('request-information.inside-parcel') is True] if parcel_path.exists() else []
        parcel=max(parcels,key=lambda x:x.get('metadata.ortho-date-estimate') or '') if parcels else {}
        pool=parcel.get('pool.detected')
        card={'sid':sid,'address':row['address'],'city':row['city'],'price':num(price),'placement':status,
            'computed':sid in signatures,'contained_partial':contained(row),'ceiling':round(ceilings[sid],3) if sid in ceilings else None,
            'pool':pool,'claims':verdicts,'sun':sun_index(row,room_rows[sid])}
        if sid in heavy and data['aerial']:card['image']=f'img/{sid}/card.jpg'
        cards.append(card)
        if i%100==0:print('pages',i+1,'elapsed',round(time.monotonic()-start),flush=True)
    write(SITE/'data/index.json',{'homes':cards,'dates':DATES,'room_types':TYPES,'topics':topics,'verdicts':VERDICTS,
        'featured':FEATURED,'heavy':heavy,'selection_rule':'The accepted six featured homes, plus 30 distinct computed homes from the supplied clearest supported and contradicted cases, alternating topics and verdicts in supplied rank order.'})
    old.method();method=old.read(SITE/'data/method.json');method['full_counts']=facts
    method['actual_counts']={'listed':len(homes),'computed':len(computed),'well_placed_computed':sum(r['sid'] in computed and r['poor']=='False' for r in homes),
        'poor_computed':sum(r['sid'] in computed and r['poor']=='True' for r in homes),'unknown_computed':sum(r['sid'] in computed and r['poor'] not in ['True','False'] for r in homes)}
    write(SITE/'data/method.json',method);write(SITE/'data/claims-summary.json',summary)
    report={'homes':len(homes),'computed':len(computed),'pages_with_cells':len(signatures),'heavy':[{'sid':s,'address':next(r['address'] for r in homes if r['sid']==s),'rule':reasons[s]} for s in heavy],
        'featured':FEATURED,'failures':failures,'asset_coverage':coverage,'network_calls':old.api.CALLS,'cached_cells':hits,'reused_home_exports':export_hits,
        'cells_seconds':round(cells_seconds,2),'full_build_seconds':round(time.monotonic()-start,2),'engine_sha256':old.adapter.SOURCE_SHA,'home_input_signatures':signatures,
        'fresh_inputs':input_digests}
    write(BASE/'build_report.json',report)
    export_replay()
    print('done',len(cards),'homes;',len(signatures),'computed pages;',report['full_build_seconds'],'seconds; failures',len(failures),flush=True)

if __name__=='__main__':main()
