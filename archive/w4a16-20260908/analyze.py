#!/usr/bin/env python3
"""Strict provenance checks; freeze sweep choices, then aggregate fresh workers."""
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
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
def key(d):return tuple(int(d[k]) for k in ('method','R','S'))
def shape(d):return tuple(int(d[k]) for k in ('M','K','N'))
def read_runs(mode):
    inv=json.loads((ROOT/'build/inventory.json').read_text())
    for file,digest in inv['hashes'].items():assert sha(ROOT/file)==digest,(file,'changed')
    out=[]
    for path in sorted((ROOT/'gpu-results').glob('*/result.json')):
        rec=json.loads(path.read_text())
        if mode not in rec['argv'] or rec['sanitizer']:continue
        if any(rec['hashes'].get(k)!=v for k,v in inv['hashes'].items()):continue
        assert rec['status']=='PASS',path
        log=(path.parent/'stdout.txt').read_text()
        assert 'BENCH_PASS' in log and not (path.parent/'stderr.txt').read_text().strip(),path
        times=collections.defaultdict(list);checks={};shapes=set();packs=[]
        for line in log.splitlines():
            d=fields(line)
            if line.startswith('TIME '):
                shapes.add(shape(d));times[key(d)].append((int(d['round']),float(d['pipeline_us'])))
            elif line.startswith('CHECK '):
                assert key(d) not in checks and int(d['operation_order_bitexact'])==1,path
                checks[key(d)]=d
                if int(d['method'])%5!=4 and int(d['S'])==32:assert int(d['baseline_changed'])==0,path
            elif line.startswith('PACK '):packs.append(d)
        assert len(shapes)==1 and times.keys()==checks.keys(),path
        rounds=4 if mode=='sweep' else 9
        for k,values in times.items():assert sorted(r for r,_ in values)==list(range(rounds)),(path,k)
        out.append(dict(run=path.parent.name,shape=shapes.pop(),record=rec,checks=checks,packs=packs,
                        times=times,medians={k:statistics.median(v for r,v in t if r>0) for k,t in times.items()}))
    return out
def freeze():
    runs=read_runs('sweep');byshape=collections.defaultdict(list)
    for r in runs:byshape[r['shape']].append(r)
    assert set(byshape)==set(SHAPES) and all(len(v)==1 for v in byshape.values()),'Need one current-build sweep per shape'
    # Fixed conservative controls, then only fastest direct/LUT sweep choices.
    chosen={(m,1,32) for m in (0,1,5,6)}
    chosen|={(m,r,32) for m in (2,7) for r in (1,2,4)}
    chosen|={(m,4,32) for m in (3,4,8,9)}
    selections=[]
    for s in SHAPES:
        run=byshape[s][0];med=run['medians']
        winners={}
        for family in ('direct','lut'):
            eligible=[k for k in med if k[0]%5==(2 if family=='direct' else 4)]
            win=min(eligible,key=med.get);chosen.add(win);winners[family]=dict(config=win,us=med[win])
        selections.append(dict(shape=s,run=run['run'],winners=winners))
    target=ROOT/'frozen-configs.txt';assert not target.exists(),'Do not overwrite frozen selection'
    target.write_text(''.join('%d %d %d\n'%k for k in sorted(chosen)))
    manifest=dict(configuration_sha256=sha(target),configs=sorted(chosen),selection=selections,
                  inventory_sha256=sha(ROOT/'build/inventory.json'),method='one sweep per shape; choices fixed before repetitions')
    (ROOT/'frozen-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
def aggregate():
    manifest=json.loads((ROOT/'frozen-manifest.json').read_text())
    assert sha(ROOT/'frozen-configs.txt')==manifest['configuration_sha256']
    assert sha(ROOT/'build/inventory.json')==manifest['inventory_sha256']
    runs=read_runs('bench');byshape=collections.defaultdict(list)
    configs={tuple(k) for k in manifest['configs']}
    for r in runs:
        assert r['record']['hashes']['config_file']==manifest['configuration_sha256'],r['run']
        assert set(r['medians'])==configs,r['run']
        byshape[r['shape']].append(r)
    assert set(byshape)==set(SHAPES) and all(len(v)==3 for v in byshape.values()),'Need exactly three fresh workers per shape'
    rows=[];allchecks=0;retained=0
    for s in SHAPES:
        for k in sorted(configs):
            reps=byshape[s];medians=[r['medians'][k] for r in reps];checks=[r['checks'][k] for r in reps]
            assert all(c==checks[0] for c in checks),'Seed/correctness changed across repetitions'
            allchecks+=3;retained+=24
            rows.append(dict(M=s[0],K=s[1],N=s[2],method=k[0],R=k[1],S=k[2],
                median_us=statistics.median(medians),min_worker_us=min(medians),max_worker_us=max(medians),
                worker_us=medians,runs=[r['run'] for r in reps],baseline_changed=int(checks[0]['baseline_changed']),
                outputs=int(checks[0]['outputs']),max_abs=float(checks[0]['max_abs']),
                max_L1_relative=float(checks[0]['max_L1_relative']),relative_L2=float(checks[0]['relative_L2'])))
    result=dict(manifest=manifest,worker_count=len(runs),configuration_checks=allchecks,retained_event_samples=retained,
                samples='median of 3 fresh-worker medians; each median has 8 post-warmup rounds, 3 pipelines/event',rows=rows,
                packing=[dict(run=r['run'],shape=r['shape'],records=r['packs']) for r in runs])
    (ROOT/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'aggregate.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('AGGREGATE_PASS workers=%d configuration_checks=%d retained_samples=%d'%(len(runs),allchecks,retained))
    for s in SHAPES:
        cells=[r for r in rows if (r['M'],r['K'],r['N'])==s]
        print('SHAPE',s)
        for r in sorted(cells,key=lambda r:r['median_us']):
            print('  m%d R%d S%d %.3f [%.3f, %.3f] changed=%d'%(r['method'],r['R'],r['S'],r['median_us'],r['min_worker_us'],r['max_worker_us'],r['baseline_changed']))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','aggregate']);args=parser.parse_args()
    (freeze if args.mode=='freeze' else aggregate)()
