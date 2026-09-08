#!/usr/bin/env python3
import json,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[]
for p in sorted(R.glob('large-*/*/stdout.jsonl')):
 try:d=[json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
 except json.JSONDecodeError:continue
 ts=[x for x in d if x['type']=='layout']
 if not ts or not any(x['type']=='post_timing_check' for x in d):continue
 groups=collections.defaultdict(list)
 for x in ts:groups[x['mode'],x['algo'],x['layout']].append(x)
 assert all(len(g)==5 for g in groups.values()),p
 row={k:d[0][k] for k in ['M','N','K']};row['file']=str(p.relative_to(R));row['configs']=[]
 for (m,a,l),gg in groups.items():row['configs'].append(dict(mode=m,algo=a,layout=l,**{k:statistics.median(g[k] for g in gg) for k in ['gemm_us','pipeline_us']}))
 rows.append(row)
 print('SHAPE',row['M'],row['N'],row['K'],'exact configs',len(groups))
 for mode in [0,1]:
  cc=sorted((c for c in row['configs'] if c['mode']==mode),key=lambda c:c['pipeline_us']);print('MODE',mode,'TOP',cc[:4]);print('BEST TN',next(c for c in cc if c['layout']==0))
 a=min(c['pipeline_us'] for c in row['configs'] if c['mode']==0);b=min(c['pipeline_us'] for c in row['configs'] if c['mode']==1);print('BEST RATIO',a/b)
(R/'large-screen-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
