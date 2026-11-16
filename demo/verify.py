"""Check the real HTTP-served build, source joins, offsets, identifiers and assets."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
import hashlib
import gzip
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
sys.dontwritebytecode=True
ROOT=Path(__file__).resolve().parents[1];SITE=ROOT/'demo/site';BASE='http://localhost:8765/'
sys.path.insert(0,str(ROOT/'demo'))
from round3_data import contained
DISK_ONLY='--disk-only' in sys.argv

class References(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v and not v.startswith(('#','http:','https:','mailto:')):self.refs.append(v.split('?')[0])

def main():
    start=time.monotonic();failures=[];counts=Counter();assets=set();claim_room_gaps=Counter()
    index=json.loads((SITE/'data/index.json').read_text());homeids={h['sid'] for h in index['homes']}
    facts={r['sid']:r for r in csv.DictReader((ROOT/'data/facts/homes.csv').open())}
    build=json.loads((ROOT/'demo/build_report.json').read_text())
    for name,digest in build['fresh_inputs'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:failures.append({'path':name,'reason':'Input refreshed since build'})
    engine_digest=hashlib.sha256((ROOT/'experiments/sun_engine.py').read_bytes()).hexdigest()
    signature_prefix=(engine_digest+(ROOT/'demo/engine_adapter.py').read_text()).encode()
    assert homeids==set(facts)
    assert len(index['heavy'])==36 and len(index['featured'])==6
    for h in index['homes']:
        sid=h['sid'];path=SITE/'data'/f'{sid}.json.gz'
        with gzip.open(path,'rt') as stream:raw=stream.read()
        data=json.loads(raw)
        if re.search(r'"(?:authorization|access_token|api_key|x-api-key|secret)"\s*:',raw,re.I) or re.search(r'https?://[^\s"<>]*[?&](?:token|signature|key|access_token|sig|authorization)=',raw,re.I):
            failures.append({'path':sid,'reason':'Forbidden credential field or signed transport URL'})
        counts['homes']+=1
        assert data['listing']['id']==str(facts[sid]['listing_id'])
        assert data['selection']==facts[sid],(sid,'facts changed since build')
        if h['placement']!='well':assert h['sun']=={}
        partial=contained(facts[sid])
        assert bool(data.get('contained_partial'))==partial and bool(h.get('contained_partial'))==partial
        if partial:counts['contained_partial']+=1;assert h['placement']!='well' and not h['sun']
        if data.get('computed'):counts['computed_pages']+=1
        if data.get('computed') and build.get('home_input_signatures'):
            inputs=[ROOT/'data/sun'/sid/f for f in ['sun.json','sample_verdicts.json']]+[ROOT/'data/cache'/sid/f for f in ['matterport.json','registration.json']]
            digest=hashlib.sha256(signature_prefix+b''.join(p.read_bytes() for p in inputs)).hexdigest()
            if digest!=build['home_input_signatures'].get(sid):failures.append({'path':sid,'reason':'Sun/scan/placement refreshed since build'})
        if data.get('aerial'):
            assert sid in index['heavy'];assets.add(data['aerial']['image']);assets.add(f'img/{sid}/card.jpg');counts['aerial_homes']+=1
        source_claim=ROOT/'data/claims'/f'{sid}.json'
        if source_claim.exists() and data.get('claims')!=json.loads(source_claim.read_text()):counts['claims_changed_since_build']+=1
        plan=set(data.get('geometry',{}))|set(data.get('claim_room_outlines',{}));audit={r['id'] for r in data.get('audit_rooms',[])};windows={w['id'] for w in data.get('windows',[])}
        for c in (data.get('claims') or {}).get('claims',[]):
            counts['claims']+=1
            assert c['verdict'] in index['verdicts']
            if c['source']=='remarks':assert data['listing']['remarks'][c['start']:c['end']]==c['text'],(sid,'claim offset')
            for e in c.get('evidence',[]):
                if e.get('room_id'):
                    rid=e['room_id']
                    if rid not in plan:
                        if rid in audit:claim_room_gaps['audited_room_without_plan_outline']+=1
                        else:claim_room_gaps['room_not_in_facts_plan_or_audit']+=1
                if e.get('window_id') and e['window_id'] not in windows:claim_room_gaps['window_not_in_plan']+=1
        for views in data.get('window_assets',{}).values():
            for key in ['inside','outside']:
                if views.get(key):assets.add(views[key]);counts[key+'_images']+=1
        for day in data.get('days',{}).values():
            assert all(len(frames)==len(day['local_times']) for frames in day['room_lit_runs'].values())
    parser=References()
    for path in SITE.glob('*.html'):parser.feed(path.read_text())
    replay=json.loads((SITE/'data/agent-replay.json').read_text())
    assert replay['passed'] is True and replay['real_model_run'] is False
    assert hashlib.sha256((ROOT/replay['source']).read_bytes()).hexdigest()==replay['source_sha256'],'Replay source refreshed; rerun demo/round3_data.py'
    counts['replay_events']=len(replay['events']);counts['replay_files']=len(replay['files'])
    paths={str(p.relative_to(SITE)) for p in SITE.glob('*.html')}|{str(p.relative_to(SITE)) for p in SITE.glob('*.js')}|{'style.css','agent.css','data/agent-replay.json','data/index.json','data/method.json','data/claims-summary.json'}|{f'data/{sid}.json.gz' for sid in homeids}|assets|set(parser.refs)|{v['image'] for v in replay['files'].values() if 'image' in v}
    method=json.loads((SITE/'data/method.json').read_text());paths.update(method.get('images',[]))
    def fetch(path):
        try:
            expected=hashlib.sha256((SITE/path).read_bytes()).digest()
            if DISK_ONLY:return None
            with urllib.request.urlopen(BASE+path,timeout=60) as response:
                served=response.read();assert response.status==200
                if path=='home.html':
                    # Runtime-only SDK config is intentionally absent from disk.
                    assert response.headers.get('Cache-Control')=='no-store'
                    served=re.sub(rb'<script type="application/json" id="tour-config">.*?</script>',b'',served)
                actual=hashlib.sha256(served).digest()
            assert actual==expected;return None
        except Exception as error:return {'path':path,'reason':type(error).__name__}
    with ThreadPoolExecutor(max_workers=8) as pool:failures.extend(x for x in pool.map(fetch,sorted(paths)) if x)
    for path in SITE.glob('*.js'):
        result=subprocess.run(['node','--check',str(path)],capture_output=True,text=True)
        if result.returncode:failures.append({'path':path.name,'reason':result.stderr})
    if not DISK_ONLY:
        with urllib.request.urlopen(BASE+'agent/status',timeout=5) as response:
            assert response.status==200
            assert isinstance(json.load(response)['configured'],bool)
            counts['agent_status_http']=200
    # Report only counts, never credential values or untrusted transport URLs.
    forbidden=[]
    for path in list(SITE.glob('*.html'))+list(SITE.glob('*.js')):
        text=path.read_text()
        if re.search(r'https?://(?!127\.0\.0\.1:8770)',text):forbidden.append(path.name)
    assert not forbidden,forbidden
    if counts['claims_changed_since_build']:failures.append({'path':'data/claims/','reason':'Inputs refreshed since build; rebuild required'})
    report={'passed':not failures,'disk_only':DISK_ONLY,'counts':dict(counts),'index_bytes':(SITE/'data/index.json').stat().st_size,'disk_files_checked':len(paths),'http_files_checked':0 if DISK_ONLY else len(paths),'http_failures':failures,
        'claim_room_reference_gaps':dict(claim_room_gaps),'js_syntax_modules':len(list(SITE.glob('*.js'))),'seconds':round(time.monotonic()-start,2),
        'dom_review':'Run browser-review.html in a real browser; result is in window.demoReview and the page.'}
    name=('round5-verification.json' if DISK_ONLY else 'round5-http-full.json') if '--round5' in sys.argv else (('round4-verification.json' if DISK_ONLY else 'round4-http-full.json') if '--round4' in sys.argv else ('round3-verification.json' if DISK_ONLY else 'verification.json'))
    (ROOT/'demo'/name).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
