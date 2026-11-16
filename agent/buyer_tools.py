"""MCP data tools for buyers. All measurements originate in local source tables."""
from __future__ import annotations
import hashlib
import json
import re
import threading
from pathlib import Path
from agent.tools import ROOT, store, number
from agent.rules import check_listing, decide, skipped
from agent.grounding import audit_explanation

STATE = ROOT/'agent/state'
PROFILE_LOCK = threading.Lock()


def profile_path(profile_id='default'):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', profile_id):
        raise ValueError('Invalid local profile id')
    return STATE/(profile_id+'.json')


def profile_read(profile_id: str = 'default') -> dict:
    """Read only needs and home feedback explicitly supplied by this local buyer."""
    path=profile_path(profile_id)
    return json.loads(path.read_text()) if path.exists() else {'needs':{},'feedback':[]}


def validate_preferences(needs, feedback):
    needs=validate_needs(needs)
    if not isinstance(feedback,list) or len(feedback)>1000:raise ValueError('Invalid feedback list')
    for item in feedback:
        if not isinstance(item,dict) or set(item)!={'sid','thumb','reason'}:raise ValueError('Feedback requires sid, thumb and reason')
        store().home(item['sid'])
        if item['thumb'] not in ('up','down') or not isinstance(item['reason'],str) or not 0<len(item['reason'].strip())<=2000:raise ValueError('Invalid feedback')
    return needs


def commit_preferences(needs: dict, feedback: list[dict], profile_id: str = 'default') -> dict:
    """Persist buyer preferences. Call only behind the graph's save_preferences approval interrupt; not exposed over MCP."""
    needs=validate_preferences(needs,feedback)
    with PROFILE_LOCK:
        previous=profile_read(profile_id)
        votes={f['sid']:f for f in previous.get('feedback',[])}
        votes.update({f['sid']:f for f in feedback})
        result=dict(previous,needs=needs,feedback=list(votes.values()))
        path=profile_path(profile_id);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    return dict(status='saved_approved_preferences',profile_id=profile_id,preferences=result)


def validate_needs(needs):
    if not isinstance(needs,dict):raise ValueError('Needs must be an object')
    allowed={'city','max_price','min_price','min_bedrooms','min_bathrooms','light','min_ceiling_height_m','pool','must_haves','nice_to_haves'}
    if set(needs)-allowed:raise ValueError('Unknown structured need')
    out=dict(needs)
    for key in ('max_price','min_price','min_bedrooms','min_bathrooms','min_ceiling_height_m'):
        if key in out and (isinstance(out[key],bool) or not isinstance(out[key],(int,float)) or number(out[key]) is None or number(out[key])<0):raise ValueError('Invalid numeric need')
    if 'city' in out and (not isinstance(out['city'],str) or not out['city'].strip()):raise ValueError('Invalid city')
    if 'min_price' in out and 'max_price' in out and out['min_price']>out['max_price']:raise ValueError('Invalid price range')
    if 'pool' in out and not isinstance(out['pool'],bool):raise ValueError('Pool must be boolean')
    if not isinstance(out.get('light',[]),list) or not isinstance(out.get('nice_to_haves',[]),list):raise ValueError('Expected preference list')
    for light in out.get('light',[]):
        if not isinstance(light,dict):raise ValueError('Invalid light need')
        if set(light)-{'room','date','time','min_minutes'}:raise ValueError('Unknown light need')
        if light.get('time','any') not in ('any','before','after'):raise ValueError('Use any, before or after solar noon')
        if light.get('date','2026-03-20') not in ('2026-03-20','2026-06-21','2026-12-21','2026-11-14'):raise ValueError('Use a computed sun date')
        if not isinstance(light.get('min_minutes'),(int,float)) or isinstance(light['min_minutes'],bool) or number(light['min_minutes']) is None or light['min_minutes']<0:raise ValueError('Invalid light duration')
    fields=[]
    for item in out.get('nice_to_haves',[]):
        if not isinstance(item,dict):raise ValueError('Invalid nice-to-have')
        if set(item)!={'field','value'} or item['field'] not in allowed-{'nice_to_haves','must_haves','light'}:raise ValueError('Unsupported nice-to-have')
        if item['field'] in fields:raise ValueError('Duplicate nice-to-have')
        fields.append(item['field']);validate_needs({item['field']:item['value']})
    if out.get('must_haves'):raise ValueError('Express must-haves as the structured hard fields; unsupported free-text hard requirements cannot be verified')
    return out


def candidate_facts(sid: str) -> dict:
    """Get listing price, scan room counts, ceiling, intrinsic light and placement caveats."""
    s=store();listing=s.listing(sid);rooms=s.rooms(sid);light=s.light_summary(sid)
    h=s.ceiling_heights(sid)
    return dict(sid=sid,address=listing['address'],city=listing['city'],price=number(listing['price']),
        bedrooms=rooms['totals']['bedrooms'],bathrooms=rooms['totals']['bathrooms'],
        max_ceiling_height_m=h['max_height_m'],light=light,
        pool=s.aerial(sid)['pool'],source_files=[listing['source_file'],rooms['totals_source_file'],h['source_file'],light['source_file']],
        caveats=list(dict.fromkeys(rooms['caveats']+light['caveats']+h['caveats'])))


def matches(sid,needs):
    s=store();listing=s.listing(sid);price=number(listing['price'])
    if needs.get('city') and str(listing['city']).casefold()!=str(needs['city']).casefold():return False
    for key,op in [('max_price',lambda x,y:x<=y),('min_price',lambda x,y:x>=y)]:
        if key in needs and (price is None or not op(price,needs[key])):return False
    counts=s.rooms(sid)['totals']
    for key,field in [('min_bedrooms','bedrooms'),('min_bathrooms','bathrooms')]:
        if key in needs and (counts[field] is None or counts[field]<needs[key]):return False
    if 'min_ceiling_height_m' in needs:
        h=s.ceiling_heights(sid)['max_height_m']
        if h is None or h<needs['min_ceiling_height_m']:return False
    if 'pool' in needs and s.aerial(sid)['pool'] is not needs['pool']:return False
    for need in needs.get('light',[]):
        if not s.placement_reliable(sid):return False
        selected=[r for r in s.suns[sid]['rooms'] if str(need.get('room','living')).casefold() in r['label'].casefold()]
        key=need.get('time','any');fact_name='windows_only_lit_minutes_ge_0_1m2' if key=='any' else key+'_noon_minutes'
        vals=[e['value'] for r in selected for e in s.sun_for_room(sid,r['id'],need.get('date','2026-03-20'))['evidence'] if e['fact']==fact_name]
        if not vals or max(vals)<need['min_minutes']:return False
    return True


def buyer_search(needs: dict, profile_id: str = 'default', relax_nice: str = '') -> dict:
    """Search all hard needs and stated nice-to-haves; relax exactly one named nice-to-have only after an empty search."""
    needs=validate_needs(needs);profile=profile_read(profile_id)
    hard={k:v for k,v in needs.items() if k not in ('nice_to_haves','must_haves')}
    missing=[k for k in ('city','max_price') if k not in hard]
    if missing:return dict(status='clarification',question='Which city should I search?' if 'city' in missing else 'What is your maximum budget?',missing=missing[0],needs=needs)
    nice=needs.get('nice_to_haves',[])
    def find(relaxed):
        return [sid for sid in sorted(store().homes) if matches(sid,hard) and all(n['field']==relaxed or matches(sid,{n['field']:n['value']}) for n in nice)]
    initial=find('')
    if relax_nice:
        if initial or relax_nice not in [n['field'] for n in nice]:raise ValueError('Relaxation requires empty search and a stated nice-to-have')
    ids=find(relax_nice) if relax_nice else initial
    # Only explicitly typed, approved feedback affects ranking. Hard needs always filter first.
    votes={f['sid']:f for f in profile.get('feedback',[])}
    remembered=profile.get('needs',{})
    soft={k:v for k,v in remembered.items() if k not in hard and k not in ('must_haves','nice_to_haves')}
    base=list(ids)
    def preference_score(sid):
        return (1 if votes.get(sid,{}).get('thumb')=='up' else -1 if votes.get(sid,{}).get('thumb')=='down' else 0,
                sum(matches(sid,{k:v}) for k,v in soft.items()))
    ids.sort(key=lambda sid:(tuple(-v for v in preference_score(sid)),sid))
    changes=[]
    if ids!=base:
        changes=[dict(sid=sid,reason=votes[sid]['reason'],thumb=votes[sid]['thumb']) for sid in ids if sid in votes]
        if soft:changes.append(dict(remembered_needs=soft))
    homes=[candidate_facts(sid) for sid in ids[:3]]
    result=dict(status='found' if homes else 'empty',needs=needs,homes=homes,matches=len(ids),relaxed_nice_to_have=relax_nice or None,
                remembered_preferences_changing_order=changes,profile=profile)
    result['needs_md']='Buyer-stated structured needs (USD, rooms, metres and minutes):\n'+json.dumps(needs,indent=2)+'\nApproved remembered preferences:\n'+json.dumps(profile,indent=2)
    result['candidates_json']=json.dumps({k:v for k,v in result.items() if k not in ('needs_md','profile')},indent=2)
    return result


def evidence_ids(doc):
    for i,claim in enumerate(doc['claims']):
        for j,e in enumerate(claim['evidence']):e['id']=f'{doc["sid"]}:c{i}:e{j}'
    return doc


def inspect_claims(sid: str, additional_claims: list[dict] | None = None) -> dict:
    """Check listing claims with measured tools; optionally discover exact extra spans missed by patterns. Returns audited dossier text and evidence ids."""
    s=store();doc=check_listing(s,sid);listing=s.listing(sid)
    for claim in additional_claims or []:
        if claim.get('source')!='remarks' or not isinstance(claim.get('start'),int) or not isinstance(claim.get('end'),int):raise ValueError('Exact remark span required')
        if listing['remarks'][claim['start']:claim['end']]!=claim.get('text') or claim['start']<0 or claim['end']<=claim['start']:raise ValueError('Invalid claim span')
        if skipped(listing['remarks'],claim['start'],claim['end'],claim['topic']):continue
        if not any(c['start']==claim['start'] and c['end']==claim['end'] and c['source']=='remarks' for c in doc['claims']):
            doc['claims'].append(decide(s,sid,claim))
    evidence_ids(doc)
    for claim in doc['claims']:audit_explanation(claim)
    text='# '+str(listing['address'])+'\n\n'
    for claim in doc['claims']:
        text+='Claim: '+claim['text']+'\nVerdict: '+claim['verdict']+'\n'+claim['explanation']+'\nEvidence: '+', '.join(e['id'] for e in claim['evidence'])+'\nCaveats: '+'; '.join(claim['caveats'])+'\n\n'
    from agent.home_graph import panorama_evidence
    visual=panorama_evidence(sid);doc['panorama_evidence']=visual
    text+='Panorama observations (local model, separate from listing claims):\n'
    for feature in visual['features']:
        text+=f"{feature['display_label']} | room label: {feature['room']} | confidence (cosine, not probability): {feature['confidence']} | source: {feature['source']} | model: {feature['model']} | image: {feature['image_file']}\n"
    for label,status in visual['labels'].items():
        if status['status']=='no panorama evidence':text+=f"{label}: no panorama evidence ({status['reason']}).\n"
    text+='Caveats: '+'; '.join(visual['caveats'])+'\n'
    return dict(sid=sid,document=doc,dossier_md=text,facts=candidate_facts(sid),method='Local geometry and attributed listing claims; panorama observations are separate unverified CLIP classifications; no measured indoor brightness.')


def sun_analysis(sid: str, date: str = '2026-03-20') -> dict:
    """Get reliable room/window direct-sun durations and room lit-time envelope; withhold all timing for poor placement."""
    s=store();s.home(sid)
    if not s.placement_reliable(sid):return dict(sid=sid,available=False,rooms=[],windows=[],intrinsic=s.light_summary(sid),caveats=[s.placement_caveat(sid),'Sun timing could not be computed reliably.'])
    sun=s.suns[sid];rooms=[];windows=[]
    dates=sorted({d for r in sun['rooms'] for d in r.get('dates',{})})
    if date not in dates:return dict(sid=sid,available=False,date=date,available_dates=dates,rooms=[],windows=[],caveats=['This date was not computed; no timing is inferred.'])
    for room in sun['rooms']:
        value=room.get('dates',{}).get(date,{})
        rooms.append(dict(room_id=room['id'],room=room['label'],date=date,**{k:value.get(k) for k in ('windows_only_lit_minutes_ge_0_1m2','first_lit_time','last_lit_time')},evidence=s.sun_for_room(sid,room['id'],date)['evidence']))
    for window in sun['windows']:
        if window['type']!='window':continue
        v=window.get('dates',{}).get(date,{})
        windows.append(dict(window_id=window['id'],room_id=window['room_id'],date=date,direct_sun_minutes=v.get('unobstructed_any_aperture_minutes'),center=window['center'],z_m=window.get('lower_z_m'),height_m=window['height'],source_file=f'data/sun/{sid}/sun.json'))
    return dict(sid=sid,available=True,available_dates=dates,rooms=rooms,windows=windows,source_file=f'data/sun/{sid}/sun.json',caveats=s.light_caveats(sid)+['Room first/last times include conditional doorways and describe the room envelope, not a windows-only interval.','Window output contains duration only; exact window lit intervals are unavailable.'])


def comparison_packet(sids: list[str], needs: dict, profile_id: str = 'default') -> dict:
    """Build a grounded finalist table and neutral question/viewing drafts for all unclear or contradicted claims."""
    if len(sids)>3 or len(set(sids))!=len(sids):raise ValueError('Up to three distinct finalists')
    needs=validate_needs(needs);hard={k:v for k,v in needs.items() if k not in ('nice_to_haves','must_haves')}
    if any(not matches(sid,hard) for sid in sids):raise ValueError('Finalist fails a hard need')
    rows=[];questions=['DRAFT — for buyer review; never sent.','Please help us understand these observations from the local scan geometry, labelled rooms and aerial data. Scan coverage and labels may be incomplete.'];view=['Viewing plan','Confirm scan coverage, room labels and the placement caveats on site.']
    dossiers={}
    for sid in sids:
        packet=inspect_claims(sid);f=packet['facts'];dossiers[sid]=packet
        def value(x):return 'unavailable' if x is None else str(x)
        light=f['light'];ratio=light['ratios'].get('home');sun=light['minutes'].get('2026-03-20')
        rows.append('| '+' | '.join([str(f['address']),value(f['price']),value(f['bedrooms']),value(f['bathrooms']),value(f['max_ceiling_height_m']),value(ratio),value(sun),'; '.join(f['caveats'])])+' |')
        for claim in packet['document']['claims']:
            if claim['verdict'] in ('unclear','contradicted'):
                questions.append(f"At {f['address']}, the listing says “{claim['text']}”. {claim['explanation']} Caveats: {'; '.join(claim['caveats'])} Could you clarify the scope, scan coverage or provide supporting records?")
        analysis=sun_analysis(sid,'2026-11-14')
        timing=max((r for r in analysis['rooms'] if r.get('windows_only_lit_minutes_ge_0_1m2') is not None),key=lambda r:r['windows_only_lit_minutes_ge_0_1m2'],default=None)
        view.append(str(f['address'])+': '+(f"On {timing['date']}, inspect {timing['room']} near the room's geometric lit envelope {timing['first_lit_time']} to {timing['last_lit_time']}; this envelope includes conditional doorways, is not a windows-only appointment or a weather forecast, and does not imply continuously lit intervals." if timing and timing['first_lit_time'] and timing['last_lit_time'] else 'Sun timing for the viewing date is unavailable; ask about orientation and observe each main room during the viewing.'))
        view.append('Check the ceiling height and window area in the scan against the actual rooms; ask each drafted question and confirm any aerial pool detection on site.')
    table='| Home | Listing price USD | Scan bedrooms | Scan bathrooms | Highest main ceiling m | Window/floor fraction | Best main-room March sun min | Caveats |\n|---|---|---|---|---|---|---|---|\n'+'\n'.join(rows)
    return dict(sids=sids,needs=needs,comparison_md=table,questions_md='\n\n'.join(questions),viewing_plan_md='\n\n'.join(view),dossiers=dossiers)


from agent.home_graph import graph_search
BUYER_TOOLS=[profile_read,candidate_facts,buyer_search,inspect_claims,sun_analysis,comparison_packet,graph_search]
