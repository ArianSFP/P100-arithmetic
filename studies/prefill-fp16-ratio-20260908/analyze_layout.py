#!/usr/bin/env python3
import json,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[]
for p in sorted(R.glob('layout-*/*.jsonl')):
 try:d=[json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
 except json.JSONDecodeError:continue
 if not d:continue
 ts=[r for r in d if r['type']=='layout'];checks=[r for r in d if r['type']=='layout_check']
 if not ts:continue
 groups=collections.defaultdict(list)
 for r in ts:groups[r['mode'],r['algo'],r['layout']].append(r)
 if any(len(g)!=7 for g in groups.values()):continue
 row={k:d[0][k] for k in ['M','N','K']};row['file']=str(p.relative_to(R));row['configs']=[]
 for (m,a,l),gg in groups.items():row['configs'].append(dict(mode=m,algo=a,layout=l,**{k:statistics.median(g[k] for g in gg) for k in ['gemm_us','pipeline_us']}))
 rows.append(row)
 print('SHAPE',row['M'],row['N'],row['K'],'exact configs',len(groups),'/',len(checks))
 for mode in [0,1]:
  cc=[c for c in row['configs'] if c['mode']==mode];cc.sort(key=lambda c:c['pipeline_us']);print('MODE',mode,'TOP',cc[:5])
  orig=[c for c in cc if c['layout']==0];print('BEST TN',orig[0])
 a=min(c['pipeline_us'] for c in row['configs'] if c['mode']==0);b=min(c['pipeline_us'] for c in row['configs'] if c['mode']==1);print('BEST RATIO',a/b)
(R/'layout-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
