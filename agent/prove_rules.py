"""Audit real rule-command output and byte determinism, without a test suite."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.tools import ROOT, Store
from agent.run_rules import OUT, write
from agent.rules import LIMITS, VERDICTS
from agent.grounding import audit_explanation


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def inputs():
    paths=[ROOT/p for p in ('data/facts/selection.json','data/facts/rooms.csv',
        'data/costar-kit/data/residential-listings.jsonl','data/costar-kit/data/residential-listing-features.jsonl',
        'data/costar-kit/residential-listings.csv','data/audit/matterport_rooms.csv',
        'data/audit/matterport_openings.csv','data/audit/matterport_models.csv','demo/site/data/index.json')]
    paths+=list((ROOT/'data/audit/cache/mp').glob('*.json'))
    paths+=list((ROOT/'agent').glob('*.py'))
    homes=json.loads((ROOT/'data/facts/selection.json').read_text())['homes']
    for sid in sorted({h['sid'] for h in homes}|{'khbuN7q853q','16ZWm6HRVjS','1FFftrgRFBd','VN4b1MGnnCR','LQLjZqrmKqP','HFwcH424n9w'}):
        paths += [ROOT/f'data/sun/{sid}/sun.json',ROOT/f'data/cache/{sid}/vexcel_calculated_65m.json',ROOT/f'data/cache/{sid}/matterport.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths)) if p.is_file()}


def outputs():
    cohort=json.loads((OUT/'cohort.json').read_text())
    paths=[OUT/f"{h['sid']}.json" for h in cohort['homes']]
    paths += [OUT/f for f in ('cohort.json','distributions.json','summary.json','summary.md')]
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def main():
    before=inputs()
    manifests=[]
    for run in (1,2):
        with (OUT/f'rerun-{run}.log').open('w') as log:
            subprocess.run([sys.executable,'-I',str(ROOT/'agent/run_rules.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifests.append(outputs())
    after=inputs()
    changes=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
    assert not changes,'Inputs changed during proof: '+repr(changes)
    assert manifests[0]==manifests[1],'Reruns differ'
    store=Store();count=0;claims=0
    for sid in sorted(store.homes):
        doc=json.loads((OUT/f'{sid}.json').read_text());listing=store.listing(sid)
        assert doc['sid']==sid and doc['listing_id']==listing['listing_id'] and doc['generated_by']=='rules_v1'
        families={}
        for c in doc['claims']:
            claims+=1
            audit_explanation(c)
            assert c['verdict'] in VERDICTS
            assert all(k in c for k in ('text','source','start','end','topic','scope','evidence','explanation','caveats'))
            if c['source']=='remarks':assert listing['remarks'][c['start']:c['end']]==c['text']
            else:
                assert c['start'] is None and c['end'] is None
                if c['source']=='feature':assert any(c['text'] in h for h in listing['highlights'])
                else:assert c['source']=='listing_field' and c['text'].removeprefix('property.') in listing['fields']
            for e in c['evidence']:
                assert all(k in e for k in ('fact','value','unit','source_file'))
                assert (ROOT/e['source_file']).is_file(),e['source_file']
            if c['topic']=='light' and doc['registration_poor'] is True:
                assert c['verdict']=='unclear'
                assert not any(e['unit']=='min' or e['fact'].startswith('2026-') for e in c['evidence'])
                assert any('Sun timing could not be computed' in caveat for caveat in c['caveats'])
                for prefix in ('scope','home'):
                    if any(e['fact']==prefix+'_window_floor_ratio' for e in c['evidence']):
                        assert any(e['fact']==prefix+'_ratio_cohort_percentile' for e in c['evidence'])
            elif c['topic']=='light' and doc['registration_poor'] is not False:assert c['verdict']=='not_checkable'
            if c['topic'] in LIMITS:families[c['topic']]=families.get(c['topic'],0)+1
        assert all(n==1 for n in families.values())
        count+=1
    assert {'khbuN7q853q','16ZWm6HRVjS','1FFftrgRFBd','VN4b1MGnnCR','LQLjZqrmKqP','HFwcH424n9w'}<=set(store.homes)
    try:audit_explanation(dict(evidence=[],explanation='The room gets 987654321 hours of sun.'))
    except ValueError:pass
    else:raise AssertionError('Unsupported explanation number accepted')
    write(OUT/'input_manifest.json',before)
    write(OUT/'output_manifest.json',manifests[1])
    proof=dict(passed=True,command=f'{sys.executable} -I agent/run_rules.py',real_full_runs=2,
        inputs_stable=True,input_files_hashed=len(before),byte_identical_outputs=len(manifests[1]),
        homes_with_claims_file=count,claim_entries=claims,exact_spans_and_sources_checked=True,
        evidence_sources_exist=True,poor_registration_intrinsic_only=True,all_explanation_numbers_in_evidence=True,
        buyer_jargon_absent=True,display_conversions_checked=True,unsupported_number_rejected=True,out_of_reach_families_unique=True,
        six_demo_homes_present=True,input_manifest='data/claims/input_manifest.json',output_manifest='data/claims/output_manifest.json')
    write(OUT/'rules_proof.json',proof)
    print(json.dumps(proof))


if __name__=='__main__':main()
