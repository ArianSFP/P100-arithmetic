#!/usr/bin/env python3
import json,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent;rows=[]
for p in sorted(R.glob('large-*/*/stdout.jsonl')):
 try:d=[json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
 except json.JSONDecodeError:continue
 ts=[x for x in d if x['type']=='isolated']
 if not ts:continue
 if len(ts)!=31 or len([x for x in d if x['type']=='confirmation_check'])!=2:continue
 row={k:d[0][k] for k in ['M','N','K']};row.update(mode=ts[0]['mode'],arm=ts[0]['arm'],pipeline_us=statistics.median(x['pipeline_us'] for x in ts),samples=[x['pipeline_us'] for x in ts],file=str(p.relative_to(R)));rows.append(row)
summary=[]
for shape in sorted({(r['M'],r['N'],r['K']) for r in rows}):
 rr=[r for r in rows if (r['M'],r['N'],r['K'])==shape];modes={}
 for mode in [0,1]:
  cc=[r for r in rr if r['mode']==mode]
  if cc:modes[str(mode)]=dict(workers=len(cc),median_us=statistics.median(r['pipeline_us'] for r in cc),worker_medians=[r['pipeline_us'] for r in cc])
 out=dict(M=shape[0],N=shape[1],K=shape[2],modes=modes)
 if len(modes)==2:out['ratio']=modes['0']['median_us']/modes['1']['median_us']
 summary.append(out);print(out)
(R/'isolated-summary.json').write_text(json.dumps(dict(workers=rows,summary=summary),indent=2)+'\n')
