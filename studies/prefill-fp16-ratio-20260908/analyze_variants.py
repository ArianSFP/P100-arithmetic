#!/usr/bin/env python3
import json,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent
workers=[]
for p in sorted(R.glob('run-*/*.jsonl')):
 try:d=[json.loads(l) for l in p.read_text().splitlines()]
 except json.JSONDecodeError:continue
 vv=[r for r in d if r['type']=='prepvariant']
 if len(vv)!=90:continue
 dev=d[0];row=dict(file=str(p.relative_to(R)),M=dev['M'],N=dev['N'],K=dev['K'],configs=[])
 for mode in [0,1]:
  for v in [0,2,4,8,-1]:
   cc=[r for r in vv if r['mode']==mode and r['variant']==v]
   assert len(cc)==9
   row['configs'].append(dict(mode=mode,variant=v,**{k:statistics.median(c[k] for c in cc) for k in ['pipeline_us','weight_us']}))
 workers.append(row)
summary=[]
for shape in sorted({(r['M'],r['N'],r['K']) for r in workers}):
 rr=[r for r in workers if (r['M'],r['N'],r['K'])==shape];cs={}
 for mode in [0,1]:
  for v in [0,2,4,8,-1]:
   cc=[c for r in rr for c in r['configs'] if c['mode']==mode and c['variant']==v]
   cs[mode,v]={k:statistics.median(c[k] for c in cc) for k in ['pipeline_us','weight_us']}
 print('SHAPE',shape,'workers',len(rr))
 for v in [0,2,4,8,-1]:
  a,b=cs[0,v]['pipeline_us'],cs[1,v]['pipeline_us']
  print('VAR',v,'weight',cs[1,v]['weight_us'],'FP32',a,'FP16',b,'FAIR RATIO',a/b,'VS ORIGINAL F32',cs[0,0]['pipeline_us']/b)
 summary.append(dict(M=shape[0],N=shape[1],K=shape[2],workers=len(rr),configs=[dict(mode=k[0],variant=k[1],**v) for k,v in cs.items()]))
(R/'variant-summary.json').write_text(json.dumps(dict(workers=workers,summary=summary),indent=2)+'\n')
