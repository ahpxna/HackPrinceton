"""Derived local property graph. Text evidence never becomes a measured feature."""
from __future__ import annotations
import hashlib
import itertools
import json
import re
import shutil
from pathlib import Path
from agent.tools import ROOT, store, number

GRAPH = ROOT/'data/claims/graph'
DATES = ('2026-12-21','2026-03-20','2026-06-21','2026-11-14')


def attach_panorama_features(doc):
    """Merge accepted observations from the reproducible sidecar, including on rebuild."""
    path=GRAPH/'panorama_features'/('accepted_'+doc['sid']+'.json')
    if not path.exists():return doc
    accepted=json.loads(path.read_text())
    old={n['id'] for n in doc['nodes'] if n.get('source_kind')=='seen in panorama'}
    doc['nodes']=[n for n in doc['nodes'] if n['id'] not in old]
    doc['edges']=[e for e in doc['edges'] if e['source'] not in old and e['target'] not in old]
    roomids={n['room_id']:n['id'] for n in doc['nodes'] if n['kind']=='room'}
    for feature in accepted['features']:
        doc['nodes'].append(feature)
        rid=feature.get('room_id');source=roomids.get(rid,'home:'+doc['sid'])
        doc['edges'].append(dict(kind='room-feature' if rid in roomids else 'home-feature',source=source,target=feature['id'],
            scope='Local zero-shot crop classification, not human verified; room is the camera location, not an object location.'))
    return doc


def build_graph():
    """Materialize compact JSON from cached sources only; never modify upstream claims."""
    s=store();GRAPH.mkdir(parents=True,exist_ok=True);files=[];sources=set();counts={}
    def source(path):
        if path:sources.add(path)
        return path
    for sid in sorted(s.homes):
        if shutil.disk_usage(ROOT).free<1024**3:raise RuntimeError('STOP: free space below 1 GiB')
        listing=s.listing(sid);rooms=s.rooms(sid)['rooms'];sun=s.suns.get(sid,{})
        reliable=s.placement_reliable(sid);nodes=[];edges=[]
        hid='home:'+sid
        def node(kind,key,**attrs):
            ident=kind+':'+sid+':'+str(key) if kind!='home' else hid
            nodes.append(dict(id=ident,kind=kind,**attrs));return ident
        def edge(kind,a,b,**attrs):edges.append(dict(kind=kind,source=a,target=b,**attrs))
        node('home',sid,sid=sid,city=listing['city'],price=number(listing['price']),
             placement_reliable=reliable,caveats=s.light_caveats(sid) if reliable else [s.placement_caveat(sid)],
             source_file=source(listing['source_file']))
        sunrooms={r['id']:r for r in sun.get('rooms',[])};rids={r['room_id'] for r in rooms}
        floors={}
        for room in rooms:
            fid=room['floor_id']
            if fid not in floors:
                floors[fid]=node('floor',fid,floor_id=fid,source_file=source('data/audit/matterport_rooms.csv'))
                edge('home-floor',hid,floors[fid])
            rid='room:'+sid+':'+room['room_id'];sr=sunrooms.get(room['room_id'],{})
            durations={d:sr.get('dates',{}).get(d,{}).get('windows_only_lit_minutes_ge_0_1m2') if reliable else None for d in DATES}
            envelopes={d:{k:sr.get('dates',{}).get(d,{}).get(k) if reliable else None for k in ('first_lit_time','last_lit_time')} for d in DATES}
            node('room',room['room_id'],room_id=room['room_id'],type=room['room'],classifications=room['classifications'],
                 bedroom_role=s.fact_rooms[sid].get(room['room_id'],{}).get('bedroom_role') or None,
                 area_m2=room['area_m2'],ceiling_height_m=room['ceiling_max_m'] or room['height_m'],
                 sun_minutes=durations,lit_hours=envelopes,lit_hours_definition='First/last room envelope including conditional doorways; not continuous windows-only intervals.',
                 source_file=source('data/audit/matterport_rooms.csv'),sun_source=source(f'data/sun/{sid}/sun.json') if sun else None)
            edge('home-room',hid,rid);edge('floor-room',floors[fid],rid)
            feature=node('feature','label:'+room['room_id'],text=room['room'],source_kind='Matterport room label',source_file='data/audit/matterport_rooms.csv',room_id=room['room_id'])
            edge('room-feature',rid,feature)
        for w in sun.get('windows',[]):
            if w['type']!='window' or w['room_id'] not in rids:continue
            wid=node('window',w['id'],window_id=w['id'],room_id=w['room_id'],compass_azimuth_deg=w.get('outward_azimuth_deg'),
                     area_m2=w['width']*w['height'],sun_minutes={d:w.get('dates',{}).get(d,{}).get('unobstructed_any_aperture_minutes') if reliable else None for d in DATES},
                     source_file=source(f'data/sun/{sid}/sun.json'))
            edge('room-window','room:'+sid+':'+w['room_id'],wid)
        model,path=s._audit_graph(sid)
        if model:
            source(path);byedge={};emap={e['id']:e for f in model.get('floors',[]) for e in f.get('edges',[]) if e.get('id')}
            for r in model.get('rooms',[]):
                for e in (r.get('boundary') or {}).get('edges',[]):byedge.setdefault(e.get('id'),set()).add(r['id'])
            for eid,adjacent in byedge.items():
                wall=emap.get(eid,{});openings=[o for o in wall.get('openings',[]) if str(o.get('type','')).lower() in ('door','doorway','opening')]
                if not openings:continue
                for a,b in itertools.combinations(sorted(adjacent & rids),2):
                    edge('room-room','room:'+sid+':'+a,'room:'+sid+':'+b,via='cached doorway/opening',edge_id=eid,source_file=path)
        # Sentence-level text stays home-scoped unless the sentence explicitly names a room type.
        for i,match in enumerate(re.finditer(r'[^.!?\n]+[.!?]?',listing['remarks'])):
            text=match.group().strip()
            if not text:continue
            start=match.start()+len(match.group())-len(match.group().lstrip())
            feat=node('feature','listing:'+str(i),text=text,source_kind='listing says',source_file=listing['source_file'],start=start,end=start+len(text),spatially_verified=False)
            edge('home-feature',hid,feat)
            for j,clausematch in enumerate(re.finditer(r'[^,;]+',match.group())):
                clause=clausematch.group().strip()
                scope_text=re.sub(r'\b(?:\d+|one|two|three|four|five|six)[ -]+(?:bedrooms?|baths?|bathrooms?)\b','',clause,flags=re.I)
                kinds=[k for k in ('bedroom','kitchen','living','dining','office','garage','bathroom') if re.search(r'\b'+k+r'\b',scope_text,re.I)]
                # Ambiguous multi-room clauses remain home-scoped; never spread a feature across them.
                if len(kinds)!=1:continue
                clause_start=match.start()+clausematch.start()+len(clausematch.group())-len(clausematch.group().lstrip())
                cf=node('feature',f'clause:{i}:{j}',text=clause,source_kind='listing says',source_file=listing['source_file'],
                        start=clause_start,end=clause_start+len(clause),spatially_verified=False)
                for room in rooms:
                    label=room['room'].casefold();primary=bool(re.search(r'\b(primary|master)\s+bedroom\b',scope_text,re.I))
                    role=s.fact_rooms[sid].get(room['room_id'],{}).get('bedroom_role','').casefold()
                    if kinds[0] in label and (not primary or role in ('primary','master')):
                        edge('room-feature','room:'+sid+':'+room['room_id'],cf,scope='Listing clause names this room type; room identity and feature are unverified.')
        claimpath=ROOT/f'data/claims/{sid}.json'
        if claimpath.exists():
            source(str(claimpath.relative_to(ROOT)))
            doc=json.loads(claimpath.read_text())
            for i,c in enumerate(doc.get('claims',[])):
                cid=node('claim',str(i),text=c['text'],verdict=c['verdict'],topic=c['topic'],source_file=str(claimpath.relative_to(ROOT)))
                edge('home-claim',hid,cid)
                for rid in sorted({e.get('room_id') for e in c.get('evidence',[]) if e.get('room_id') in rids}):edge('claim-about-room',cid,'room:'+sid+':'+rid)
        doc=attach_panorama_features(dict(sid=sid,nodes=nodes,edges=edges));nodes=doc['nodes']
        for feature in nodes:
            if feature.get('source_kind')=='seen in panorama':
                source(feature['source_file']);source(f'data/claims/graph/panorama_features/accepted_{sid}.json')
        path=GRAPH/(sid+'.json');path.write_text(json.dumps(doc,separators=(',',':'),ensure_ascii=False))
        files.append(path.name)
        for n in nodes:counts[n['kind']]=counts.get(n['kind'],0)+1
    manifest=[]
    for relative in sorted(sources):
        p=ROOT/relative
        if p.is_file():
            digest=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
            manifest.append(dict(path=relative,sha256=digest.hexdigest(),bytes=p.stat().st_size))
    (GRAPH/'input_manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
    result=dict(schema_version=1,files=files,counts=counts,bytes=sum((GRAPH/f).stat().st_size for f in files),
                caveats=['Listing sentences are unverified text. Panorama observations, when present, are local model classifications, not human-verified objects; camera-room mapping does not establish object location.',
                         'Missing room links, sun, wall openings or features mean unknown, never absent. Claim verdicts are from the existing per-home claims, not recomputed.',
                         'Source hashes describe files observed during build; concurrent upstream writes are not a transactional snapshot.'])
    (GRAPH/'index.json').write_text(json.dumps(result,indent=2));return result


def visual_display(feature):
    """Keep detector vocabulary reproducible while avoiding unsupported material claims."""
    result=dict(feature)
    result['display_label']='wood-look floor' if feature['label']=='hardwood floor' else feature['label']
    if feature['label']=='hardwood floor':result['material_verified']=False
    return result


def panorama_evidence(sid: str) -> dict:
    """Read published visual observations for one home, separately from listing claims."""
    store().home(sid)
    calibration=json.loads((GRAPH/'panorama_features_calibration.json').read_text())
    path=GRAPH/f'{sid}.json';doc=json.loads(path.read_text()) if path.exists() else {'nodes':[]}
    rooms={n['room_id']:n for n in doc['nodes'] if n['kind']=='room'}
    features=[dict(visual_display(n),camera_room=rooms.get(n.get('room_id'))) for n in doc['nodes'] if n.get('source_kind')=='seen in panorama']
    return dict(status='found' if features else 'no panorama evidence',features=features,
        labels={label:dict(status='published observation' if any(n['label']==label for n in features) else 'no panorama evidence',
                           enabled=cutoff is not None,threshold=cutoff,
                           reason='label disabled after calibration' if cutoff is None else 'published observations only; missing never means absent')
                for label,cutoff in calibration['label_thresholds'].items()},
        caveats=['Local CLIP cosine scores are not probabilities or verified objects.',
                 'Room is the camera location; visible content may extend through openings. A Bedroom label or largest_bedroom role does not establish primary bedroom.',
                 'Disabled labels answer no panorama evidence. Missing evidence does not establish absence.'])


def graph_search(constraints: dict) -> dict:
    """Combine structured home/room/sun constraints with literal text alternatives (feature='beam|rafter'). Text stays attributed, never measured."""
    from agent.buyer_tools import matches,validate_needs
    if not isinstance(constraints,dict) or set(constraints)-{'needs','room','feature','source_kind','date','min_sun_minutes','min_area_m2','min_ceiling_height_m','min_feature_confidence','limit'}:raise ValueError('Unknown graph constraint')
    needs=validate_needs(constraints.get('needs',{}));limit=constraints.get('limit',10)
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=50:raise ValueError('Limit must be 1 to 50')
    date=constraints.get('date','2026-12-21')
    if date not in DATES:raise ValueError('Use a computed date')
    for k in ('min_sun_minutes','min_area_m2','min_ceiling_height_m','min_feature_confidence'):
        if k in constraints and (isinstance(constraints[k],bool) or not isinstance(constraints[k],(int,float)) or number(constraints[k]) is None or constraints[k]<0):raise ValueError('Invalid numeric graph constraint')
    for k in ('room','feature','source_kind'):
        if k in constraints and (not isinstance(constraints[k],str) or len(constraints[k])>200):raise ValueError('Invalid text constraint')
    pattern=constraints.get('feature','').casefold();terms=[v.strip() for v in pattern.split('|') if v.strip()]
    terms=['hardwood floor' if t in ('wood-look floor','wood look floor') else t for t in terms]
    if pattern and not terms:raise ValueError('Empty feature alternatives')
    if not (GRAPH/'index.json').exists():raise ValueError('Graph missing; run python -m agent.home_graph first')
    index=json.loads((GRAPH/'index.json').read_text());results=[];total=0
    for file in index['files']:
        sid=Path(file).stem
        if not matches(sid,needs):continue
        doc=json.loads((GRAPH/file).read_text());nodes={n['id']:n for n in doc['nodes']};features={}
        for e in doc['edges']:
            if e['kind']=='room-feature':features.setdefault(e['source'],[]).append((nodes[e['target']],e.get('scope')))
            elif e['kind']=='home-feature' and nodes[e['target']].get('source_kind')=='seen in panorama':
                features.setdefault('unknown',[]).append((nodes[e['target']],e.get('scope')))
        rooms=[n for n in doc['nodes'] if n['kind']=='room']
        if features.get('unknown') and not any(k in constraints for k in ('room','min_sun_minutes','min_area_m2','min_ceiling_height_m')):
            rooms.append(dict(id='unknown',room_id=None,type='room unknown',sun_minutes={},area_m2=None,ceiling_height_m=None))
        for room in rooms:
            requested=constraints.get('room','').casefold()
            if requested in ('primary bedroom','master bedroom'):
                if not ('bedroom' in room['type'].casefold() and (room.get('bedroom_role') in ('primary','master') or re.search(r'\b(primary|master)\b',room['type'],re.I))):continue
            elif requested not in room['type'].casefold():continue
            values={'min_sun_minutes':room['sun_minutes'].get(date),'min_area_m2':room['area_m2'],'min_ceiling_height_m':room['ceiling_height_m']}
            if any(k in constraints and (v is None or v<constraints[k]) for k,v in values.items()):continue
            feats=[dict({k:f[k] for k in ('text','source_kind','source_file','source','label','confidence','confidence_kind','model','model_revision','home','room','room_mapping','pano_id','face','image_file','crop','crop_definition','human_reviewed','spatially_verified','threshold') if k in f},scope=scope)
                   for f,scope in features.get(room['id'],[]) if (not terms or any(t in f['text'].casefold() for t in terms))
                   and (not constraints.get('source_kind') or f['source_kind']==constraints['source_kind'])
                   and ('min_feature_confidence' not in constraints or f.get('confidence',-1)>=constraints['min_feature_confidence'])]
            feats=[visual_display(f) if f.get('source_kind')=='seen in panorama' else f for f in feats]
            if (terms or constraints.get('source_kind') or 'min_feature_confidence' in constraints) and not feats:continue
            total+=1
            if len(results)<limit:results.append(dict(sid=sid,room=room,features=feats,graph_file='data/claims/graph/'+file))
    result=dict(status='found' if results else 'empty',constraints=constraints,matches=total,results=results,caveats=index['caveats'])
    if constraints.get('source_kind')=='seen in panorama':
        calibration=json.loads((GRAPH/'panorama_features_calibration.json').read_text())
        result['disabled_labels']=[label for label,cutoff in calibration['label_thresholds'].items() if cutoff is None and (not terms or any(t in label for t in terms))]
        result['panorama_evidence_status']='published observations' if total else 'no panorama evidence'
        result['panorama_evidence_scope']='All requested constraints must hold; disabled/missing observations do not establish absence.'
    return result


if __name__=='__main__':
    result=build_graph();print(json.dumps({k:v for k,v in result.items() if k!='files'}))
