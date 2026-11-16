"""Deterministic conservative checks over local measurements; no model required."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from .tools import DATES, ROOT, Store, fact, number, quantiles
from .explanations import explain, display
from .grounding import audit_explanation

PATTERNS = {
    'light': r'\b(?:(?:full of|flooded with|abundant|plenty of|lots of|ample)\s+)?(?:natural\s+light|naturally\s+(?:lit|bright)|sun[\s-]?(?:lit|filled|drenched)|light[\s-]filled|bright|sunny)\b',
    'ceilings': r"\b(?:(?:\d+(?:\.\d+)?)[\s-]*(?:ft\.?|feet|foot|['′])\s*(?:[\s-]*(?:high|tall))?[\s-]*ceilings?|(?:high|tall|soaring|vaulted|cathedral|two[\s-]stor(?:y|ey))\s+(?:[\w-]+\s+){0,2}ceilings?)\b",
    'open_plan': r'\bopen[\s-]+(?:concept|floor\s*plan|plan)\b',
    'garage': r'\b(?:[1234]|one|two|three|four)[\s-]*car\s+(?:(?:attached|detached|tandem|oversized)\s+)?garage\b',
    'pool': r'\b(?:(?:private|in[\s-]ground|backyard|swimming)\s+)?pool\b',
    'solar': r'\bsolar[\s-]+(?:panels?|photovoltaic)\b',
    'roof_age': r'\b(?:new(?:er)?\s+roof|roof\s+(?:replaced|replacement|age)|(?:replaced|updated)\s+roof)\b',
    'renovations_finishes': r'\b(?:remodel(?:ed|led|ing)?|renovat(?:ed|ion)|updated|quartz|granite|hardwood|tile\s+floor(?:ing)?|luxury\s+finishes)\b',
    'appliances': r'\b(?:appliances?|refrigerator|dishwasher|oven|washer|dryer|range)\b',
    'schools_location': r'\b(?:schools?|school\s+district|zoned|walking\s+distance|located|minutes\s+(?:from|to)|close\s+to|convenient(?:ly)?|trails|neighborhood)\b',
}
COMPILED = {k:re.compile(v,re.I) for k,v in PATTERNS.items()}
LIMITS = {'roof_age':'The aerial image cannot tell when the roof was installed.', 'renovations_finishes':'Room measurements cannot tell when work was done or verify materials and finish quality.',
          'appliances':'Room measurements cannot verify which appliances are included or their condition.', 'schools_location':'Room measurements and aerial images cannot verify school assignments, travel times or neighbourhood quality.'}
VERDICTS = ('supported','contradicted','unclear','not_checkable')


def clause(text, start, end):
    left=max([text.rfind(x,0,start) for x in '.;!?\n']+[-1])+1
    stops=[text.find(x,end) for x in '.;!?\n']
    right=min([x for x in stops if x>=0]+[len(text)])
    return text[left:right],start-left,end-left


def skipped(text, start, end, topic):
    line,a,b=clause(text,start,end)
    prefix=line[:a]
    # Negation/hypothesis belongs to the same local clause, not another sentence.
    local=re.split(r',|\bbut\b|\band\b',prefix,flags=re.I)[-1]
    if re.search(r"\b(?:no|not|without|never|lack(?:s|ing)?|could|would|might|may|potential|possib\w*|option(?:al)?|room\s+for|space\s+for|ready\s+for|perfect\s+for|ideal\s+for|add(?:ing)?)\b",local,re.I):return True
    suffix=line[b:b+55]
    if re.match(r'\s+(?:could|would|may|might|not\b|is\s+not|can\s+be\s+added)',suffix,re.I):return True
    if topic=='pool':
        local_context=line[max(0,a-65):min(len(line),b+40)]
        if re.search(r'\bcommunity|communal|association|neighbou?rhood|clubhouse|pool[-\s]+(?:bath|house|table|sized)|access\s+to',local_context,re.I):return True
    return False


def light_scope(text,start,end):
    line,a,b=clause(text,start,end)
    # Scope the nearest explicitly named room in this clause.
    aliases=[('Kitchen',r'\bkitchen\b'),('Living Room',r'\bliving\s+(?:room|space|area)\b'),('Family Room',r'\bfamily\s+room\b'),
             ('Great Room',r'\bgreat\s+room\b'),('Dining',r'\bdining(?:\s+room)?\b'),('Bedroom',r'\bbedroom\b'),('Bathroom',r'\bbathroom\b'),('Office',r'\boffice\b')]
    matches=[(min(abs(m.start()-b),abs(m.end()-a)),label) for label,pattern in aliases for m in re.finditer(pattern,line,re.I)]
    return min(matches)[1] if matches else 'main living spaces'


def extract(listing):
    claims=[];seen=set()
    sources=[('remarks',listing['remarks'])]+[('feature',text) for text in listing['highlights']]
    for source,text in sources:
        for topic,pattern in COMPILED.items():
            for match in pattern.finditer(text):
                if skipped(text,match.start(),match.end(),topic):continue
                # One out-of-reach entry per family, even across remarks/highlights.
                key=topic if topic in LIMITS else (topic,source,match.group().casefold(),match.start() if source=='remarks' else None)
                if key in seen:continue
                seen.add(key)
                scope=light_scope(text,match.start(),match.end()) if topic=='light' else 'home'
                claims.append(dict(text=match.group(),source=source,start=match.start() if source=='remarks' else None,end=match.end() if source=='remarks' else None,topic=topic,scope=scope))
    return claims


def decide(store, sid, claim):
    topic=claim['topic'];text=claim['text'];scope=claim['scope']
    verdict='not_checkable';evidence=[];caveats=[];cars=None
    if topic in LIMITS:
        explanation=LIMITS[topic]
    elif topic=='light':
        tool=store.light_summary(sid,scope);evidence=tool['evidence'];caveats=tool['caveats']
        if tool['registration_poor'] is True:
            verdict='unclear'
        elif not tool['available'] or tool['registration_poor'] is not False:
            pass
        else:
            ratio=tool['ratios'].get('scope');thresholds=tool['thresholds'].get('scope_ratio',{})
            march=tool['minutes'].get(DATES[1]);winter=tool['minutes'].get(DATES[0])
            if (march is not None and march>=120) or (ratio is not None and thresholds.get('p67') is not None and ratio>=thresholds['p67']):
                verdict='supported'
            elif march==0 and winter==0 and ratio is not None and ratio<=thresholds.get('p25',-1) and tool['scope_complete']:
                # Unknown doors could transmit real sun; do not contradict from windows-only zeros.
                conditional=[x['value'] for x in evidence if x['fact']=='conditional_door_direct_sun_minutes' and x.get('date') in DATES[:2]]
                if conditional and max(conditional)==0:
                    verdict='contradicted'
                else:
                    verdict='unclear'
            else:
                verdict='unclear'
    elif topic=='ceilings':
        tool=store.ceiling_heights(sid);evidence=tool['evidence'];caveats=tool['caveats'];height=tool['max_height_m']
        numeric=re.search(r"(\d+(?:\.\d+)?)\s*[- ]*(?:ft|feet|foot|['′])",text,re.I)
        if numeric:
            target=float(numeric.group(1))*.3048
            # Stated number is sourced to this exact listing phrase, not invented measurement.
            evidence.append(fact('listing_stated_ceiling_height',float(numeric.group(1)),'ft','data/costar-kit/data/residential-listings.jsonl' if claim['source']=='remarks' else 'data/costar-kit/data/residential-listing-features.jsonl'))
            high,low=target-.5*.3048,target-1*.3048
        elif re.search(r'soaring|vaulted|cathedral|two[\s-]stor',text,re.I):high,low=3.35,2.75
        else:high,low=2.70,2.50
        if height is not None:
            if height>=high:verdict='supported'
            elif (height<low if numeric else height<=low) and tool['complete']:
                verdict='contradicted'
            else:verdict='unclear'
    elif topic=='open_plan':
        tool=store.layout(sid);caveats=tool['caveats']
        evidence=[fact('shared_kitchen_living_classifications',tool['shared_kitchen_living'],'boolean',tool['source_file'])] if tool['available'] else []
        for conn in tool['connections']:
            evidence.append(fact('kitchen_connection_width',conn['width_m'],'m',conn['source_file'],type=conn['type'],rooms=conn['rooms'],edge_id=conn['edge_id']))
        if tool['shared_kitchen_living'] or any(c['type']=='divider' or (c['width_m'] is not None and c['width_m']>=1.8) for c in tool['connections']):
            verdict='supported'
        elif tool['complete'] and tool['kitchen_connections'] and all(c['type'] in ('door','doorway') and c['width_m'] is not None and c['width_m']<=1.0 for c in tool['kitchen_connections']):
            evidence=[fact('kitchen_doorway_width',c['width_m'],'m',c['source_file'],edge_id=c['edge_id'],rooms=c['rooms']) for c in tool['kitchen_connections']]
            verdict='contradicted'
    elif topic=='garage':
        tool=store.garage(sid);evidence=tool['evidence'];caveats=tool['caveats']
        token=re.search(r'\b([1234]|one|two|three|four)',text,re.I).group(1).lower()
        cars={'one':1,'two':2,'three':3,'four':4}.get(token,int(token) if token.isdigit() else 0)
        needed={1:15,2:30,3:50}.get(cars)
        area=tool['area_m2']
        if area is not None and needed is not None:
            verdict='supported' if area>=needed else 'contradicted' if area<.8*needed else 'unclear'
    elif topic in ('pool','solar'):
        tool=store.aerial(sid);caveats=tool['caveats'];key='pool' if topic=='pool' else 'solar_panels'
        evidence=[x for x in tool['evidence'] if x['fact'] in (key,'image_age_at_listing_days')]
        if tool['available'] and tool[key] is not None:
            verdict='supported' if tool[key] else 'unclear'
    result=dict(claim,verdict=verdict,evidence=evidence,caveats=caveats)
    result['explanation']=explanation if topic in LIMITS else explain(result,tool,evidence,cars)
    if topic=='light':result['light_status_reason']=tool.get('unavailable_reason') or 'measured'
    return result


def count_claims(store,sid,listing):
    scan=store.counts(sid);fields=listing['fields'];result=[]
    for topic,field,measured,unit in [('bedrooms','bedsMin','bedrooms','rooms'),('bathrooms','bathsMin','bathrooms','rooms'),('living_area','sqftMin','living_area_sqft','sqft')]:
        stated=number(fields.get(field));observed=scan[measured]
        if stated is None:continue
        caveats=list(scan['caveats']);comparison=stated
        if topic=='bathrooms':
            # A fractional bath total cannot reveal the physical full + half count.
            half=number(fields.get('bathsHalf'));full=number(fields.get('bathsFull'))
            if half is not None and full is not None:comparison=full+half
            elif stated%1 or (full is not None and full!=stated):comparison=None
        evidence=[fact('listing_'+topic,stated,unit,listing['source_file'],source_field='property.'+field)]
        if comparison is not None and comparison!=stated:evidence.append(fact('listing_physical_bathrooms',comparison,'rooms',listing['source_file'],source_field='property.bathsFull + property.bathsHalf'))
        if observed is not None:evidence.append(fact('scan_'+topic,observed,unit,scan['source_file']))
        if observed is None:
            verdict='not_checkable';tail='the scan has no usable total for comparison'
        elif comparison is None:
            verdict='unclear';tail='the listing’s bathroom total and the scan’s separate bathroom spaces count different things'
        elif observed==comparison or comparison>0 and abs(observed-comparison)/comparison<=.10:
            verdict='supported';tail='the totals are close enough to support the listing'
        elif topic=='bedrooms' and comparison-observed>=2:
            # Fail closed when missing labels or room boundaries explain fewer rooms.
            rr=store.rooms(sid)['rooms']
            complete=all(r['boundary_complete'] and r['classifications'] and not all(c in ('Other','Void') for c in r['classifications']) for r in rr if r['indoor_area_m2'] and r['indoor_area_m2']>0)
            verdict='contradicted' if complete else 'unclear'
            tail='the fully labelled scan shows substantially fewer bedrooms' if complete else 'missing room labels or walls could explain the lower bedroom count'
        else:
            verdict='unclear';tail='missing rooms, labels or differing area definitions could explain the difference'
        noun={'bedrooms':'bedrooms','bathrooms':'bathrooms','living_area':'square feet of living area'}[topic]
        explanation=f'The listing gives {stated:g} {noun}'
        if comparison is not None and comparison!=stated:explanation+=f' ({comparison:g} separate bathroom spaces)'
        if observed is not None:
            measured_fact=next(e for e in evidence if e['fact']=='scan_'+topic)
            observed_text=display(evidence,measured_fact,round(observed,1),unit,'round(value, 1)')
            explanation+=f'; the scan records about {observed_text} {noun}'
        explanation+='; '+tail+'.'
        result.append(dict(text='property.'+field,source='listing_field',start=None,end=None,topic=topic,scope='home',verdict=verdict,evidence=evidence,explanation=explanation,caveats=caveats))
    return result


def check_listing(store, sid):
    listing=store.listing(sid)
    claims=[decide(store,sid,c) for c in extract(listing)]+count_claims(store,sid,listing)
    for claim in claims:audit_explanation(claim)
    return dict(sid=sid,listing_id=listing['listing_id'],generated_by='rules_v1',registration_poor=store.suns.get(sid,{}).get('registration',{}).get('poor'),claims=claims)


def distributions(store):
    values=defaultdict(list)
    for sid in sorted(store.homes):
        light=store.raw_light(sid)
        if light.get('registration_poor') is False:
            for key,v in light['ratios'].items():values['well_registered_'+key+'_window_floor_ratio'].append(v)
            for date,v in light['minutes'].items():values['well_registered_best_main_room_windows_only_sun_'+date].append(v)
        values['max_main_ceiling_m'].append(store.ceiling_heights(sid)['max_height_m'])
        values['garage_area_m2'].append(store.garage(sid)['area_m2'])
        values['home_window_floor_ratio_audit'].append(number(store.models.get(sid,{}).get('window_floor_ratio')))
        scan=store.counts(sid)
        for key in ('bedrooms','bathrooms','living_area_sqft'):values['scan_'+key].append(scan[key])
        layout=store.layout(sid)
        values['kitchen_living_widest_opening_m'].append(max((c['width_m'] for c in layout['connections'] if c['width_m'] is not None),default=None))
        values['shared_kitchen_living_classification'].append(int(layout['shared_kitchen_living']) if layout['available'] else None)
        aerial=store.aerial(sid)
        for key in ('pool','solar_panels','tree_cover_percent'):
            v=aerial[key];values['aerial_'+key].append(int(v) if isinstance(v,bool) else v)
    out={k:quantiles(v) for k,v in sorted(values.items())}
    scopes=sorted({'main living spaces'}|{c['scope'] for sid in store.homes for c in extract(store.listing(sid)) if c['topic']=='light'})
    out['light_by_scope']={scope:{k:quantiles(v) for k,v in sorted(store.light_population(scope).items())} for scope in scopes}
    return out


def summary(store,documents):
    topics=defaultdict(lambda:dict(listings=set(),verdict_counts=Counter(),cases=[]))
    claims_by_sid={d['sid']:d['claims'] for d in documents}
    for doc in documents:
        for claim in doc['claims']:
            topic=topics[claim['topic']];topic['listings'].add(doc['listing_id']);topic['verdict_counts'][claim['verdict']]+=1
            topic['cases'].append(dict(sid=doc['sid'],listing_id=doc['listing_id'],address=store.homes[doc['sid']].get('address'),**claim))
    out={}
    for name,topic in sorted(topics.items()):
        def strength(c):
            nums=[abs(x['value']) for x in c['evidence'] if isinstance(x.get('value'),(int,float)) and not isinstance(x['value'],bool)]
            # Evidence completeness first; stable measured magnitude only breaks ties.
            return (-len(c['caveats']),len(c['evidence']),max(nums,default=0))
        clearest={}
        for verdict in ('contradicted','supported'):
            cases=sorted((c for c in topic['cases'] if c['verdict']==verdict),key=lambda c:(tuple(-x for x in strength(c)),c['sid'],c['text']))
            selected=[];seen=set()
            for c in cases:
                if c['listing_id'] not in seen:selected.append(c);seen.add(c['listing_id'])
            clearest[verdict]=selected[:10]
        out[name]=dict(listings_making_claim=len(topic['listings']),verdict_counts={v:topic['verdict_counts'][v] for v in VERDICTS},clearest=clearest)
    base_rates={}
    for topic in ('light','ceilings','open_plan'):
        groups=defaultdict(lambda:defaultdict(list))
        seen_listings=set()
        for sid in sorted(store.homes):
            if not store.listing(sid)['remarks']:continue
            listing_id=store.homes[sid]['listing_id']
            if listing_id in seen_listings:continue
            seen_listings.add(listing_id)
            has=any(c['topic']==topic for c in claims_by_sid.get(sid,[]));group='claim' if has else 'no_claim'
            if topic=='light':
                value=store.raw_light(sid)
                if value.get('registration_poor') is False:
                    groups[group]['window_floor_ratio'].append(value['ratios'].get('scope'))
                    groups[group]['march_windows_only_best_main_room_minutes'].append(value['minutes'].get(DATES[1]))
            elif topic=='ceilings':groups[group]['max_main_ceiling_m'].append(store.ceiling_heights(sid)['max_height_m'])
            else:
                value=store.layout(sid)
                if value['available']:
                    groups[group]['shared_kitchen_living_fraction'].append(int(value['shared_kitchen_living']))
                    groups[group]['widest_kitchen_living_opening_m'].append(max((c['width_m'] for c in value['connections'] if c['width_m'] is not None),default=None))
        comparison={g:{k:quantiles(v) for k,v in quantities.items()} for g,quantities in groups.items()}
        differences={}
        for key,claim_values in groups.get('claim',{}).items():
            no_values=groups.get('no_claim',{}).get(key,[])
            a=quantiles(claim_values);b=quantiles(no_values)
            if a['n'] and b['n']:differences[key]=a['median']-b['median']
        positive=any(d>0 for d in differences.values())
        uninformative=[k for k,v in differences.items() if v==0]
        interpretation='Descriptive differences only; no predictive accuracy or causal information was established.' if positive else 'The wording carries no positive information in these measured cohort medians; this is descriptive, not a statistical equivalence test.'
        if uninformative:interpretation+=' The wording carries no information in the identical medians for: '+', '.join(uninformative)+'. Means are also reported; equal medians do not establish identical distributions.'
        base_rates[topic]=dict(groups=comparison,median_differences_claim_minus_no_claim=differences,interpretation=interpretation)
    light_groups=defaultdict(lambda:dict(entries=0,homes=set(),verdict_counts=Counter(),with_ratio_and_position=0))
    for doc in documents:
        for c in doc['claims']:
            if c['topic']!='light':continue
            group=light_groups[c['light_status_reason']]
            group['entries']+=1;group['homes'].add(doc['sid']);group['verdict_counts'][c['verdict']]+=1
            names={e['fact'] for e in c['evidence']}
            if any({p+'_window_floor_ratio',p+'_ratio_cohort_percentile'}<=names for p in ('scope','home')):group['with_ratio_and_position']+=1
    coverage={k:dict(v,homes=len(v['homes']),verdict_counts=dict(v['verdict_counts'])) for k,v in sorted(light_groups.items())}
    return dict(homes_with_claims_file=len(documents),homes_with_listing=sum(bool(d['listing_id']) for d in documents),unique_primary_listings=len({d['listing_id'] for d in documents if d['listing_id']}),homes_with_sun=len(store.suns),light_coverage=coverage,
                topics=out,base_rates=base_rates,notes=['Verdict counts are claim entries across scan files; listings_making_claim deduplicates primary listing ids across all scans.',
                'Base rates use one scan per primary listing (lexicographically first sid); means and quantiles exclude missing measurements. No independence or predictive accuracy claim is made.',
                'Clearest cases are ranked by fewer caveats, then more evidence, then measured magnitude, with deterministic ties; this is not a dishonesty ranking.',
                'Sun populations use the files present when the run started; rerun as Brief F produces more.',
                'No model, sponsor API or measured illuminance was used.'])
