#!/usr/bin/env python3
"""Aggregate retained event samples; keep geometry and precision comparisons separate."""
from pathlib import Path
import json,statistics,collections
r=Path(__file__).resolve().parent
out=[]
for p in sorted(r.glob('run-*/*/stdout.jsonl')):
    meta=json.loads((p.parent/'result.json').read_text()) if (p.parent/'result.json').exists() else {}
    if meta.get('status')!='PASS':continue
    rows=[json.loads(s) for s in p.read_text().splitlines() if s.startswith('{')]
    dat=next(x for x in rows if x.get('type')=='data')
    prep=next(x['included'] for x in rows if x.get('type')=='preparation')
    checks={x['name']:x for x in rows if x.get('type')=='check'}
    times=collections.defaultdict(list)
    for x in rows:
        if x.get('type')=='timing' and x['retained']:times[x['name']].append(x['us'])
    configs={n:dict(us=statistics.median(v),min=min(v),max=max(v),samples=v,check=checks[n]) for n,v in times.items()}
    x=dict(path=str(p.parent.relative_to(r)),shape=[dat['M'],dat['K'],dat['N']],seed=dat['seed'],family=dat['family'],prep=prep,sanitizer='compute-sanitizer' in ' '.join(meta.get('command',[])),configs=configs)
    out.append(x)
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out:
 if x['sanitizer']:continue
 cs=x['configs'];best={}
 for mode in [-1,0,1,2]:
  names=[n for n,c in cs.items() if (c['check']['mode'] if c['check']['mode']<0 else c['check']['mode']%3)==mode and not c['check']['nonfinite']]
  if names:best[mode]=min(names,key=lambda n:cs[n]['us'])
 print(x['path'],x['shape'],'prep',x['prep'],'family',x['family'])
 for mode,n in best.items():
  c=cs[n];b=min((cs[name]['us'] for m,name in best.items() if m<=0),default=None)
  ratio=f"{b/c['us']:.3f}x" if b is not None else "no paired F32 arm"
  print(f"  mode {mode}: {n:18s} {c['us']:9.3f}us vs best F32 {ratio}, L2 {c['check']['finite_relL2']:.6g}")
