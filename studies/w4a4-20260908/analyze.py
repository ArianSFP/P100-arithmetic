"""Audit retained builds/raw workers and aggregate frozen paired comparisons."""
import csv
import hashlib
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
selection=json.loads((ROOT/'frozen-selection.json').read_text())
for f,h in selection['hashes'].items():assert sha(ROOT/'frozen'/f)==h
builds={}
for path in [ROOT/'build-manifest.json',*ROOT.glob('*-build/build-manifest.json')]:
 manifest=json.loads(path.read_text())
 for name,h in manifest['hashes'].items():
  local=path.parent/name
  if not local.is_file():local=ROOT/name
  assert sha(local)==h,(path,name)
 builds[sha(path)]=manifest
runs=[];grouped=defaultdict(list);checks=0;values=0
for path in sorted((ROOT/'gpu-results').glob('*/result.json')):
 result=json.loads(path.read_text());assert result['status']=='PASS',path
 assert result['manifest_sha256'] in builds,path
 text=(path.parent/'stdout.txt').read_text()
 assert re.search(r'^PASS$',text,re.M),path
 checklines=re.findall(r'^CHECK (\w+) PASS(?: values=(\d+))?$',text,re.M)
 checks+=len(checklines);values+=sum(int(n) for _,n in checklines if n)
 for state in ['before','after']:
  row=[x.strip() for x in result[state]['device'].strip().split(',')]
  assert row[1]=='GPU-4868830a-c1cf-90bd-8018-2360c55293b8' and int(row[3])<=64 and int(row[6])==0
 run={'id':path.parent.name,'checks':len(checklines),'manifest':result['manifest_sha256'],'frozen':'frozen' in result['argv']}
 runs.append(run)
 if not run['frozen']:continue
 shape='-'.join(re.search(r'DATA M=(\d+) K=(\d+) N=(\d+)',text).groups())
 assert result['config_sha256']==selection['hashes'][shape+'.txt']
 times=defaultdict(list)
 for rnd,name,us in re.findall(r'TIME round=(\d+) config=(\w+) us=([\d.]+)',text):
  if int(rnd) in selection['retained_rounds']:times[name].append(float(us))
 assert set(times)==set(selection['sets'][shape])|{'sealed_a16','current_a16_r3'}
 assert all(len(v)==8 for v in times.values())
 medians={name:st.median(v) for name,v in times.items()}
 grouped[shape].append({'run':run['id'],'medians_us':medians,'samples':dict(times),
  'relative_l2':float(re.search(r'QUANTIZATION relative_L2=([\deE.+-]+)',text)[1])})
rows=[]
for shape,workers in grouped.items():
 assert len(workers)==3,(shape,len(workers))
 chosen=selection['selected'][shape]
 med={name:st.median(w['medians_us'][name] for w in workers) for name in workers[0]['medians_us']}
 fp32=min((k for k in med if k.startswith('w4a4_h0_')),key=lambda k:med[k])
 half=min((k for k in med if k.startswith('w4a4_h1_')),key=lambda k:med[k])
 ratios=[w['medians_us'][chosen]/w['medians_us']['current_a16_r3'] for w in workers]
 rows.append({'shape':shape,'selected':chosen,'a16_r2_us':med['sealed_a16'],'a16_r3_us':med['current_a16_r3'],
 'a4_selected_us':med[chosen],'a4_fp32_us':med[fp32],'a4_half2_us':med[half],
 'a4_slowdown_vs_r3_pct':(med[chosen]/med['current_a16_r3']-1)*100,
 'paired_slowdown_min_pct':(min(ratios)-1)*100,'paired_slowdown_max_pct':(max(ratios)-1)*100,
 'a4_gain_vs_r2_pct':(1-med[chosen]/med['sealed_a16'])*100,
 'a4_gain_vs_fp32_pct':(1-med[chosen]/med[fp32])*100,'synthetic_relative_l2':workers[0]['relative_l2']})
assert len(rows)==4
out={'rows':rows,'workers':dict(grouped),'audit':{'successful_workers':len(runs),'pipeline_checks':checks,'explicit_checked_values':values,
 'frozen_workers':sum(len(v) for v in grouped.values()),'retained_samples':sum(len(v) for ws in grouped.values() for w in ws for v in w['samples'].values()),'builds_verified':len(builds)},'runs':runs}
(ROOT/'aggregate.json').write_text(json.dumps(out,indent=2)+'\n')
with (ROOT/'aggregate.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({'rows':rows,'audit':out['audit']},indent=2))
