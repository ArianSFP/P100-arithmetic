#!/usr/bin/env python3
import json,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent
workers=[]
for p in sorted(R.glob('large-*/*/stdout.jsonl')):
 try:d=[json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
 except json.JSONDecodeError:continue
 ts=[x for x in d if x['type']=='confirmation']
 if not ts:continue
 checks=[x for x in d if x['type']=='confirmation_check'];assert len(checks)==2 and all(x['different']==0 for x in checks)
 groups=collections.defaultdict(list)
 for x in ts:groups[x['mode'],x['arm']].append(x)
 assert len(ts)==108 and all(len(g)==18 for g in groups.values()),p
 row={k:d[0][k] for k in ['M','N','K']};row['file']=str(p.relative_to(R));row['configs']=[]
 for (m,a),gg in groups.items():
  # Each ABBA's two same-arm legs form one paired sample before aggregation.
  paired=[statistics.mean(g['pipeline_us'] for g in gg if g['sample']==s) for s in range(9)]
  gs=[x['gemm_us'] for x in d if x['type']=='confirmation_gemm' and (x['mode'],x['arm'])==(m,a)]
  row['configs'].append(dict(mode=m,arm=a,algo=gg[0]['algo'],layout=gg[0]['layout'],pipeline_us=statistics.median(paired),gemm_us=statistics.median(gs),paired_samples=paired))
 workers.append(row)
summary=[]
for shape in sorted({(r['M'],r['N'],r['K']) for r in workers}):
 rr=[r for r in workers if (r['M'],r['N'],r['K'])==shape];cs={}
 for mode in [0,1]:
  for arm in range(3):
   cc=[c for r in rr for c in r['configs'] if c['mode']==mode and c['arm']==arm]
   cs[mode,arm]=dict(algo=cc[0]['algo'],layout=cc[0]['layout'],**{k:statistics.median(c[k] for c in cc) for k in ['pipeline_us','gemm_us']},worker_pipeline_us=[c['pipeline_us'] for c in cc])
 print('SHAPE',shape,'workers',len(rr))
 for arm in range(3):
  a,b=cs[0,arm],cs[1,arm];print('ARM',arm,'F32',round(a['pipeline_us'],2),'F16',round(b['pipeline_us'],2),'RATIO',round(a['pipeline_us']/b['pipeline_us'],4),'F16 VS TUNED TN',round(cs[1,1]['pipeline_us']/b['pipeline_us'],4),'GEMM RATIO',round(a['gemm_us']/b['gemm_us'],4))
 summary.append(dict(M=shape[0],N=shape[1],K=shape[2],workers=len(rr),configs=[dict(mode=k[0],arm=k[1],**v) for k,v in cs.items()]))
(R/'confirmation-summary.json').write_text(json.dumps(dict(workers=workers,summary=summary),indent=2)+'\n')
