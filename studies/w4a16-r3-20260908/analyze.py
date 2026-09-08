#!/usr/bin/env python3
"""Freeze from exploratory sweeps; independently aggregate fresh-worker repeats."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
SHAPES=[(5120,5120,1),(17408,5120,1),(5120,17408,1),(5120,5120,4)]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
def config(data):return tuple(int(data[k]) for k in ('method','R','S'))
def shape(data):return tuple(int(data[k]) for k in ('M','K','N'))
def runs(mode):
    inventory=json.loads((ROOT/'build/inventory.json').read_text())
    for name,digest in inventory['hashes'].items():assert sha(ROOT/name)==digest,(name,'changed')
    out=[]
    for path in sorted((ROOT/'gpu-results').glob('*/result.json')):
        rec=json.loads(path.read_text())
        if mode not in rec['argv'] or rec['sanitizer']:continue
        assert all(rec['hashes'].get(k)==v for k,v in inventory['hashes'].items()),path
        assert rec['status']=='PASS' and rec['returncode']==0,path
        text=(path.parent/'stdout.txt').read_text()
        assert text.endswith('BENCH_PASS\n') and not (path.parent/'stderr.txt').read_text().strip(),path
        checks={};times=collections.defaultdict(list);shapes=set()
        for line in text.splitlines():
            data=fields(line)
            if line.startswith('CHECK '):
                key=config(data);assert key not in checks
                assert data['operation_order_bitexact']=='1' and data['baseline_changed']=='0' and data['S']=='32'
                checks[key]=data
            elif line.startswith('TIME '):
                shapes.add(shape(data));times[config(data)].append((int(data['round']),float(data['pipeline_us'])))
        assert len(shapes)==1 and checks.keys()==times.keys(),path
        for key,samples in times.items():assert sorted(r for r,_ in samples)==list(range(4 if mode=='sweep' else 9)),(path,key)
        out.append(dict(run=path.parent.name,shape=shapes.pop(),record=rec,checks=checks,times=times,
            medians={key:statistics.median(t for r,t in samples if r>0) for key,samples in times.items()}))
    return out
def freeze():
    sweeps=runs('sweep');assert len(sweeps)==4 and {r['shape'] for r in sweeps}==set(SHAPES)
    # Preserve both controls and fixed component comparisons even when they lose.
    chosen={(14,2,32),(36,2,32),(100,2,32),(102,2,32),(103,2,32),(113,2,32),(202,2,32),(212,2,32)}
    selections=[]
    for run in sweeps:
        candidate=min((key for key in run['medians'] if key[0]>=100),key=run['medians'].get)
        chosen.add(candidate);control=(14 if run['shape'][2]==1 else 36,2,32)
        selections.append(dict(shape=run['shape'],run=run['run'],candidate=candidate,
            candidate_us=run['medians'][candidate],control_us=run['medians'][control]))
    path=ROOT/'frozen-configs.txt';assert not path.exists(),'Do not overwrite frozen configs'
    path.write_text(''.join('%d %d %d\n'%key for key in sorted(chosen)))
    manifest=dict(configs=sorted(chosen),configuration_sha256=sha(path),selection=selections,
        inventory_sha256=sha(ROOT/'build/inventory.json'),method='one sweep per shape; winners frozen before repetitions')
    (ROOT/'frozen-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
def aggregate():
    manifest=json.loads((ROOT/'frozen-manifest.json').read_text())
    assert sha(ROOT/'frozen-configs.txt')==manifest['configuration_sha256']
    assert sha(ROOT/'build/inventory.json')==manifest['inventory_sha256']
    data=runs('bench');by_shape=collections.defaultdict(list)
    configs={tuple(c) for c in manifest['configs']}
    for run in data:
        assert run['record']['gpu']==2 and run['record']['hashes']['config_file']==manifest['configuration_sha256']
        assert set(run['checks'])==configs
        by_shape[run['shape']].append(run)
    assert set(by_shape)==set(SHAPES) and all(len(reps)==3 for reps in by_shape.values()),'Need 3 fresh workers per shape'
    rows=[]
    for s in SHAPES:
        for key in sorted(configs):
            reps=by_shape[s];values=[run['medians'][key] for run in reps]
            assert all(run['checks'][key]==reps[0]['checks'][key] for run in reps),'Numerical reference changed'
            rows.append(dict(M=s[0],K=s[1],N=s[2],method=key[0],R=key[1],S=key[2],
                median_us=statistics.median(values),min_worker_us=min(values),max_worker_us=max(values),
                worker_us=values,runs=[run['run'] for run in reps],baseline_changed=0))
    comparisons=[]
    for selected in manifest['selection']:
        s=tuple(selected['shape']);candidate=tuple(selected['candidate']);control=(14 if s[2]==1 else 36,2,32)
        c=next(row for row in rows if shape(row)==s and config(row)==candidate)
        b=next(row for row in rows if shape(row)==s and config(row)==control)
        pairs=[100*(1-run['medians'][candidate]/run['medians'][control]) for run in by_shape[s]]
        comparisons.append(dict(M=s[0],K=s[1],N=s[2],candidate=candidate,baseline_us=b['median_us'],candidate_us=c['median_us'],
            latency_reduction_pct=100*(1-c['median_us']/b['median_us']),speedup=b['median_us']/c['median_us'],paired_reduction_pct=pairs))
    summary=dict(manifest=manifest,worker_count=len(data),configuration_checks=len(data)*len(configs),
        retained_event_samples=len(data)*len(configs)*8,rows=rows,comparisons=comparisons)
    (ROOT/'aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (ROOT/'aggregate.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('AGGREGATE_PASS workers=%d checks=%d retained=%d'%(len(data),summary['configuration_checks'],summary['retained_event_samples']))
    print(json.dumps(comparisons,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','aggregate']);args=parser.parse_args()
    (freeze if args.mode=='freeze' else aggregate)()
