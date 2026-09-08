#!/usr/bin/env python3
import json,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[]
for run in sorted(R.glob('run-*')):
 for p in sorted(run.glob('*.jsonl')):
  try:data=[json.loads(l) for l in p.read_text().splitlines()]
  except json.JSONDecodeError:continue
  if not data:continue
  dev=data[0];checks={(r['mode'],r['algo']):r for r in data if r['type']=='check'}
  stages=[r for r in data if r['type']=='stages']
  groups=collections.defaultdict(list)
  for r in data:
   if r['type']=='timing':groups[r['mode'],r['algo']].append(r)
  if not stages or any(len(x)!=7 for x in groups.values()):continue
  row=dict(file=str(p.relative_to(R)),M=dev['M'],N=dev['N'],K=dev['K'],stages={k:statistics.median(r[k] for r in stages) for k in ['weight_us','activation_us','finish_us']},configs=[])
  for key,vals in groups.items():row['configs'].append(dict(mode=key[0],algo=key[1],different=checks[key]['different'],relL2=checks[key]['relL2'],gemm_us=statistics.median(x['gemm_us'] for x in vals),pipeline_us=statistics.median(x['pipeline_us'] for x in vals)))
  rows.append(row)
(R/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for shape in sorted({(r['M'],r['N'],r['K']) for r in rows}):
 rr=[r for r in rows if (r['M'],r['N'],r['K'])==shape]
 cs={}
 for key in {(c['mode'],c['algo']) for r in rr for c in r['configs']}:
  cc=[c for r in rr for c in r['configs'] if (c['mode'],c['algo'])==key]
  cs[key]={k:statistics.median(c[k] for c in cc) for k in ['gemm_us','pipeline_us','different','relL2']};cs[key]['exact']=all(c['different']==0 for c in cc)
 stage={k:statistics.median(r['stages'][k] for r in rr) for k in rr[0]['stages']}
 print('SHAPE',shape,'workers',len(rr),'STAGES',stage)
 for mode in [0,1]:
  default=(mode,99 if mode else -1);best=min((k for k in cs if k[0]==mode and cs[k]['exact']),key=lambda k:cs[k]['pipeline_us'])
  print('DEFAULT',default,cs[default],'BEST EXACT',best,cs[best])
 print('DEFAULT RATIOS GEMM',cs[0,-1]['gemm_us']/cs[1,99]['gemm_us'],'PIPE',cs[0,-1]['pipeline_us']/cs[1,99]['pipeline_us'])
 print('TOP HALF',sorted(((k,c['pipeline_us'],c['exact']) for k,c in cs.items() if k[0]==1),key=lambda x:x[1])[:8])
