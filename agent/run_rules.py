"""Run from the workspace: python3 -I agent/run_rules.py [--sid SID]."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.tools import ROOT, Store
from agent.rules import check_listing, distributions, summary

OUT=ROOT/'data/claims'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    # No timestamps: equal inputs produce equal bytes. Atomic replacement for demo readers.
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(data)
    temp.replace(path)


def run(sid=None):
    store=Store()
    dist=distributions(store)
    print(json.dumps(dist,sort_keys=True),flush=True)
    write(OUT/'distributions.json',dist)
    write(OUT/'cohort.json',dict(homes=[store.homes[s] for s in sorted(store.homes)],demo_sids=store.demo_sids,
                               sun_sids=sorted(store.suns),selection_source='data/facts/selection.json'))
    sids=[sid] if sid else sorted(store.homes)
    documents=[]
    for selected in sids:
        doc=check_listing(store,selected)
        write(OUT/f'{selected}.json',doc)
        documents.append(doc)
    if not sid:
        report=summary(store,documents)
        write(OUT/'summary.json',report)
        lines=['# Listing claims — offline rule run','',f"Homes with claims files: {len(documents)}; sun outputs in this snapshot: {len(store.suns)}.",'',
               '| Topic | Listings | Supported | Contradicted | Unclear | Not checkable |','|---|---:|---:|---:|---:|---:|']
        for topic,row in report['topics'].items():
            c=row['verdict_counts'];lines.append(f"| {topic} | {row['listings_making_claim']} | {c['supported']} | {c['contradicted']} | {c['unclear']} | {c['not_checkable']} |")
        lines += ['', '## Light coverage', '', 'Counts are claim entries; homes deduplicate scan ids within each reason. Poor placement retains intrinsic window shares, with unclear verdicts and no sun timing.', '```json',json.dumps(report['light_coverage'],indent=2),'```']
        lines += ['', '## Base rates', '', 'Only homes with remarks and the relevant measurement enter the comparison. Claims include remarks and highlights. Missing measurements are excluded.']
        for topic,row in report['base_rates'].items():
            lines += ['',f'### {topic}',row['interpretation'],'```json',json.dumps(row,indent=2),'```']
        lines += ['', '## Clearest cases', '', 'Up to ten of each verdict per topic; fewer means no further qualifying cases. Full room-level numbers and caveats are in summary.json.']
        for topic,row in report['topics'].items():
            for verdict,cases in row['clearest'].items():
                lines += ['',f'### {topic}: {verdict}']
                for c in cases:
                    values=[{k:v for k,v in e.items() if k in ('fact','value','unit','date','room')} for e in c['evidence']]
                    lines.append(f"- {c['sid']} — {c['address']} — {c['text']!r}: {json.dumps(values,ensure_ascii=False)}")
                if not cases:lines.append('None.')
        lines += ['', '## Limits', '', *report['notes']]
        (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps({k:report[k] for k in ('homes_with_claims_file','homes_with_listing','homes_with_sun')}),flush=True)
    return documents


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sid')
    args=parser.parse_args()
    run(args.sid)
