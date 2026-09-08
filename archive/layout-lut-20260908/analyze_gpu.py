#!/usr/bin/env python3
"""Summarize the completed frozen three-fresh-worker series, excluding tuning."""
import csv
import hashlib
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((ROOT/'gpu-build.json').read_text())
config_hash=sha(ROOT/'frozen-configs.txt')
fields=['W','M','K','N','F','method','R','S','old']
groups=defaultdict(list)
runs=[]
not_launched=[]
for result_path in sorted((ROOT/'gpu-results').glob('layout-*/result.json')):
    record=json.loads(result_path.read_text())
    if record.get('hashes',{}).get('config_file')!=config_hash:continue
    argv=record['argv']
    if 'bench' not in argv or int(argv[argv.index('--m')+1])<5120:continue
    if not (result_path.parent/'prelaunch.json').exists():
        assert 'returncode' not in record and 'pid' not in record,result_path
        not_launched.append(dict(run=result_path.parent.name,status=record['status']))
        continue
    assert record['status']=='PASS',result_path
    assert record['hashes']['worker']==build['worker_sha256'],result_path
    assert record['hashes']['baseline_cubin']==build['baseline_cubin_sha256'],result_path
    text=(result_path.parent/'stdout.txt').read_text()
    assert text.endswith('BENCH_PASS\n'),result_path
    per_run=defaultdict(dict)
    for line in text.splitlines():
        if not line.startswith('TIME '):continue
        data=dict(re.findall(r'(\w+)=([^ ]+)',line))
        key=tuple(int(data[f]) for f in fields)
        round_id=int(data['round'])
        assert round_id not in per_run[key]
        per_run[key][round_id]=float(data['pipeline_us'])
    for key,values in per_run.items():
        assert set(values)==set(range(9)),(result_path,key)
        samples=[values[i] for i in range(1,9)]
        groups[key].append(dict(run=result_path.parent.name,median_us=st.median(samples),samples_us=samples))
    runs.append(dict(run=result_path.parent.name,before=record['before'],after=record['after'],hashes=record['hashes']))
assert len(runs)==24,len(runs)
rows=[]
for key,values in sorted(groups.items()):
    assert len(values)==3,(key,len(values))
    medians=[v['median_us'] for v in values]
    rows.append(dict(zip(fields,key),median_us=st.median(medians),min_worker_us=min(medians),max_worker_us=max(medians),workers=values))
assert len(rows)==104,len(rows)
summary=dict(scope='Synthetic signed W2/W4 x A8 G32, full prepared-device pipeline',
             config_sha256=config_hash,build=build,discard_round=0,samples_per_worker=8,
             authorization_history='Final W4 repetition initially policy-blocked; user explicitly removed the sub-Q8 ban, GPU1 was freshly reserved, and layout-1788869038900603625 completed it with unchanged worker/cubins/configs.',
             pipelines_per_sample=3,runs=runs,not_launched=not_launched,rows=rows)
(ROOT/'GPU-SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'GPU-SUMMARY.csv').open('w') as out:
    writer=csv.DictWriter(out,fieldnames=fields+['median_us','min_worker_us','max_worker_us'])
    writer.writeheader()
    for row in rows:writer.writerow({k:v for k,v in row.items() if k!='workers'})
for shape in sorted(set(tuple(r[f] for f in fields[:4]) for r in rows)):
    subset=[r for r in rows if tuple(r[f] for f in fields[:4])==shape]
    print('SHAPE',shape)
    for name,f,m in [('VMAD',1,0),('SAD',1,1),('regLUT',3,5),('shared',3,3),('replicated',3,4),('oldVMAD',0,11),('oldmasked',0,12),('oldendpoint',0,17)]:
        r=min((r for r in subset if r['F']==f and r['method']==m),key=lambda r:r['median_us'])
        print(f"  {name:12} {r['median_us']:9.3f} us [{r['min_worker_us']:.3f},{r['max_worker_us']:.3f}] R{r['R']}S{r['S']}")
print('PASS log analysis: 24 fresh workers, 104 aggregates, 2496 retained samples; all cells have three fresh workers')
