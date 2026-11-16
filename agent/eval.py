"""Fixed local evaluation. Dry-run is a scripted harness check, never model quality."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.tools import ROOT,store,TOOL_FUNCTIONS
from agent.rules import check_listing as rules_check
from agent.agent import check_listing,configured_model,RejectedClaim
from agent.buyer_tools import buyer_search,candidate_facts

OUT=ROOT/'data/claims/eval'
REQUESTS=[{'city':'Houston','max_price':600000},
 {'city':'Houston','max_price':300000,'min_bedrooms':3},
 {'city':'Houston','max_price':500000,'min_bathrooms':2},
 {'city':'Houston','max_price':900000,'min_ceiling_height_m':3.35},
 {'city':'Houston','max_price':800000,'pool':True},
 {'city':'Houston','max_price':600000,'light':[{'room':'bedroom','date':'2026-12-21','min_minutes':60}]},
 {'city':'Houston','max_price':750000,'light':[{'room':'living','date':'2026-03-20','min_minutes':120}]},
 {'city':'Houston','max_price':600000,'min_price':250000},
 {'city':'Houston','max_price':1},
 {'city':'Houston','max_price':600000,'min_bedrooms':3,'min_bathrooms':2}]


def fixture_model(sid,claims,bad=False):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    class Fixture(FakeMessagesListChatModel):
        def _get_ls_params(self,**kwargs):return {'ls_provider':'local_fake','ls_model_name':'eval-fixture'}
        def bind_tools(self,tools,**kwargs):return self
    functions={f.__name__:f for f in TOOL_FUNCTIONS}
    topics={'light':'light_summary','ceilings':'ceiling_heights','open_plan':'layout','garage':'garage','pool':'aerial_facts','solar':'aerial_facts','bedrooms':'rooms','bathrooms':'rooms','living_area':'rooms'}
    def call(name,args,i):return dict(name=name,args=args,id='fixture_'+str(i))
    responses=[AIMessage(content='',tool_calls=[call('listing',{'sid':sid},0)])]
    proposed=[]
    for i,c in enumerate(claims):
        name=topics.get(c['topic'],'listing');args={'sid':sid}
        if name=='light_summary':args['scope']=c['scope']
        responses.append(AIMessage(content='',tool_calls=[call(name,args,i+1)]))
        p={k:c[k] for k in ('text','source','start','end','topic','scope','verdict')}
        p['evidence_call_ids']=['call_'+str(i+2)]
        p['explanation']='The recorded measurements and coverage caveats determine this check.'
        if bad and i==0:p['explanation']='The measured value is 987654321987654321.'
        proposed.append(p)
    responses.append(AIMessage(content=json.dumps({'claims':proposed})))
    return Fixture(responses=responses)


def verify_home(sid,needs):
    f=candidate_facts(sid);fail=[]
    if f['city'].casefold()!=needs['city'].casefold():fail.append('city')
    for key,field,op in [('max_price','price',lambda v,n:v<=n),('min_price','price',lambda v,n:v>=n),
                         ('min_bedrooms','bedrooms',lambda v,n:v>=n),('min_bathrooms','bathrooms',lambda v,n:v>=n),
                         ('min_ceiling_height_m','max_ceiling_height_m',lambda v,n:v>=n)]:
        if key in needs and (f[field] is None or not op(f[field],needs[key])):fail.append(key)
    if 'pool' in needs and f['pool'] is not needs['pool']:fail.append('pool')
    for need in needs.get('light',[]):
        s=store();vals=[r.get('dates',{}).get(need['date'],{}).get('windows_only_lit_minutes_ge_0_1m2') for r in s.suns.get(sid,{}).get('rooms',[]) if need['room'].casefold() in r['label'].casefold()]
        if not s.placement_reliable(sid) or not any(v is not None and v>=need['min_minutes'] for v in vals):fail.append('light')
    return fail


def main(live=False):
    OUT.mkdir(parents=True,exist_ok=True);fixed=OUT/'fixed_cases.json'
    if fixed.exists():cases=json.loads(fixed.read_text())
    else:
        cases=dict(sids=sorted(store().homes)[:50],requests=REQUESTS)
        fixed.write_text(json.dumps(cases,indent=2))
    rows=[];topics=defaultdict(Counter);model,name=configured_model() if live else (None,'scripted-rules-fixture')
    for sid in cases['sids']:
        baseline=rules_check(store(),sid);claims=baseline['claims']
        doc=check_listing(sid,model if live else fixture_model(sid,claims),name,OUT/('live_listings' if live else 'scripted_listings'))
        reference={(c['source'],c['start'],c['end'],c['text'],c['topic']):c for c in claims};extra=[]
        for c in doc['claims']:
            old=reference.get((c['source'],c['start'],c['end'],c['text'],c['topic']))
            if old:
                topics[c['topic']]['compared']+=1;topics[c['topic']]['agreed']+=int(c['verdict']==old['verdict'])
            else:extra.append(c)
        rejected=None
        if not live and claims:
            try:check_listing(sid,fixture_model(sid,claims,bad=True),name,OUT/'should_never_exist')
            except RejectedClaim:rejected=True
            else:raise AssertionError('Unsupported numeric probe was accepted')
        rows.append(dict(sid=sid,claims=len(doc['claims']),extra_claims=extra,unsupported_number_probe_rejected=rejected,
                         tool_calls=doc['telemetry']['tool_calls'],tokens=doc['telemetry'] if live else {'input_tokens':0,'output_tokens':0},
                         accounting='Provider usage only when returned; scripted calls use zero provider tokens.'))
    requests=[]
    for needs in cases['requests']:
        result=buyer_search(needs,profile_id='eval-no-feedback')
        homes=[dict(sid=h['sid'],hard_need_failures=verify_home(h['sid'],needs)) for h in result['homes']]
        if any(h['hard_need_failures'] for h in homes):raise AssertionError('Returned home fails hard needs')
        requests.append(dict(needs=needs,status=result['status'],homes=homes,matches=result['matches']))
    panorama_cases=[];sun_fit_cases=[]
    if not live:
        from agent.buyer_agent import panorama_dry_run, sun_fit_dry_run
        panorama_cases.append(panorama_dry_run(workspace_base=OUT/'workspaces'))
        sun_fit_cases.append(sun_fit_dry_run())
    output=dict(passed=True,real_model_run=live,model=name,listing_count=len(rows),buyer_request_count=len(requests),panorama_cases=panorama_cases,sun_fit_cases=sun_fit_cases,
                agreement_per_topic=dict(topics),listings=rows,buyer_requests=requests,
                interpretation='Scripted fixtures replay rules verdicts through actual Deep Agents tools and grounding. Agreement is tautological: harness proof only, not model quality.' if not live else 'Live listing-model results; buyer needs are prestructured fixtures, not natural-language extraction quality.',
                limitations=['Missing live accounting is null, not zero. No claim of recall without human labels. Empty results do not establish search quality.'])
    if live:
        import os,uuid
        if os.getenv('LANGSMITH_API_KEY'):
            try:
                from langsmith import Client
                from datetime import datetime,timezone
                client=Client();ident=uuid.uuid4()
                client.create_run(name='sunproof-fixed-evaluation',run_type='chain',id=ident,inputs={'cases_file':'data/claims/eval/fixed_cases.json'},outputs=output,end_time=datetime.now(timezone.utc))
                output['langsmith']={'status':'submitted','run_id':str(ident)}
            except Exception:output['langsmith']={'status':'failed','reason':'Upload failed; provider details withheld'}
    path=OUT/('evaluation_live.json' if live else 'evaluation_dry_run.json');path.write_text(json.dumps(output,indent=2))
    print(json.dumps(dict(passed=True,report=str(path.relative_to(ROOT)),listings=len(rows),requests=len(requests),real_model_run=live)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');parser.add_argument('--dry-run',action='store_true')
    main(parser.parse_args().live)
