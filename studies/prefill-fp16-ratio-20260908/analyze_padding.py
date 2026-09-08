#!/usr/bin/env python3
import json,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[]
for p in sorted(R.glob('large-*/*/stdout.jsonl')):
 try:d=[json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
 except json.JSONDecodeError:continue
 ts=[x for x in d if x['type']=='padding']
 if not ts or not any(x['type']=='post_timing_check' for x in d):continue
 groups=collections.defaultdict(list)
 for x in ts:groups[x['mode'],x['layout'],x['pad']].append(x)
 assert all(len(g)==9 for g in groups.values()),p
 row={k:d[0][k] for k in ['M','N','K']};row['file']=str(p.relative_to(R));row['configs']=[]
 for (m,l,pad),gg in groups.items():row['configs'].append(dict(mode=m,algo=3 if m else 6,layout=l,pad=pad,**{k:statistics.median(g[k] for g in gg) for k in ['gemm_us','pipeline_us']},samples=[g['pipeline_us'] for g in gg]))
 rows.append(row)
 print('SHAPE',row['M'],row['N'],row['K'])
 for mode in [0,1]:
  cc=sorted((c for c in row['configs'] if c['mode']==mode),key=lambda c:c['pipeline_us']);print('MODE',mode,'TOP',[(c['layout'],c['pad'],round(c['pipeline_us'],2),round(min(c['samples']),2),round(max(c['samples']),2)) for c in cc[:5]])
(R/'padding-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
