#!/usr/bin/env python3
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
workers=[]
for p in sorted(R.glob('frozen-*/*.jsonl')):
 try:d=[json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
 except json.JSONDecodeError:continue
 timings=[x for x in d if x['type']=='frozen']
 if len(timings)!=88:continue
 checks=[x for x in d if x['type']=='frozen_check'];assert len(checks)==8 and all(x['different']==0 for x in checks)
 row={k:d[0][k] for k in ['M','N','K']};row['file']=str(p.relative_to(R));row['configs']=[]
 for mode in [0,1]:
  for arm in range(4):
   cc=[x for x in timings if x['mode']==mode and x['arm']==arm];assert len(cc)==11
   row['configs'].append(dict(mode=mode,arm=arm,algo=cc[0]['algo'],variant=cc[0]['variant'],**{k:statistics.median(c[k] for c in cc) for k in ['pipeline_us','gemm_us']}))
 workers.append(row)
summary=[]
for shape in sorted({(r['M'],r['N'],r['K']) for r in workers}):
 rr=[r for r in workers if (r['M'],r['N'],r['K'])==shape];cs={}
 for mode in [0,1]:
  for arm in range(4):
   cc=[c for r in rr for c in r['configs'] if c['mode']==mode and c['arm']==arm]
   cs[mode,arm]=dict(algo=cc[0]['algo'],variant=cc[0]['variant'],**{k:statistics.median(c[k] for c in cc) for k in ['pipeline_us','gemm_us']},worker_pipeline_us=[c['pipeline_us'] for c in cc])
 print('SHAPE',shape,'workers',len(rr))
 for arm in range(4):
  a,b=cs[0,arm],cs[1,arm]
  print('ARM',arm,'F32',round(a['pipeline_us'],2),'F16',round(b['pipeline_us'],2),'RATIO',round(a['pipeline_us']/b['pipeline_us'],4),'VS ORIGINAL F32',round(cs[0,0]['pipeline_us']/b['pipeline_us'],4),'GEMM RATIO',round(a['gemm_us']/b['gemm_us'],4),'ALGOS',a['algo'],b['algo'])
 summary.append(dict(M=shape[0],N=shape[1],K=shape[2],workers=len(rr),configs=[dict(mode=k[0],arm=k[1],**v) for k,v in cs.items()]))
(R/'frozen-summary.json').write_text(json.dumps(dict(workers=workers,summary=summary),indent=2)+'\n')
