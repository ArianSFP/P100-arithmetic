#!/usr/bin/env python3
from pathlib import Path
import json,statistics,collections
r=Path(__file__).resolve().parent
out=[]
for p in sorted(r.glob('run-*/*/result.json')):
 meta=json.loads(p.read_text())
 if meta.get('status')!='PASS':continue
 rows=[json.loads(s) for s in (p.parent/'stdout.jsonl').read_text().splitlines() if s.startswith('{')]
 data=next(x for x in rows if x['type']=='data');checks={x['name']:x for x in rows if x['type']=='check'};geometry={x['name']:x for x in rows if x['type']=='geometry'};timing=collections.defaultdict(list)
 for x in rows:
  if x['type']=='timing' and x['retained']:timing[x['name']].append(x['us'])
 cs={name:dict(us=statistics.median(v),samples=v,check=checks[name],geometry=geometry[name]) for name,v in timing.items()}
 x=dict(path=str(p.parent.relative_to(r)),shape=[data['M'],data['K'],data['N']],family=data['family'],seed=data['seed'],configs=cs,sanitizer='compute-sanitizer' in ' '.join(meta['command']))
 out.append(x)
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out:
 if x['sanitizer']:continue
 cs=x['configs'];print(x['path'],x['shape'],'family',x['family'])
 exact=[n for n,c in cs.items() if c['check']['mode']%3==0 or c['check']['mode']<0]
 exact=[n for n in exact if cs[n]['check']['different']==0]
 best32=min(exact,key=lambda n:cs[n]['us']) if exact else None
 for kind in ['exact32','halfgroup','halflocal']:
  choices=exact if kind=='exact32' else [n for n,c in cs.items() if c['check']['mode']>=0 and c['check']['mode']%3==(1 if kind=='halfgroup' else 2) and not c['check']['nonfinite']]
  if not choices:continue
  name=min(choices,key=lambda n:cs[n]['us']);ratio=cs[best32]['us']/cs[name]['us'] if best32 else None
  print(kind,name,round(cs[name]['us'],3),'us','vs best exact32',round(ratio,3) if ratio else 'unpaired')
