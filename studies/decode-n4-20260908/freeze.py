#!/usr/bin/env python3
from pathlib import Path
import json
r=Path(__file__).resolve().parent;xs=json.loads((r/'summary.json').read_text());plans=[]
for x in xs:
 if '/screen4-' not in x['path'] or x['sanitizer']:continue
 cs=x['configs'];exact=[n for n,c in cs.items() if c['check']['different']==0 and (c['check']['mode']<0 or c['check']['mode']%3==0)]
 f=min(exact,key=lambda n:cs[n]['us']);h=[]
 for mode in [1,2]:h.append(min((n for n,c in cs.items() if c['check']['mode']>=0 and c['check']['mode']%3==mode and not c['check']['nonfinite']),key=lambda n:cs[n]['us']))
 names=list(dict.fromkeys(['r3b4',f]+h+['m4r2b4w8s16','m5r2b4w8s16','g4r1b4w4s32','g5r1b4w4s32']))
 plans.append(dict(shape=x['shape'],screen=x['path'],fp32=f,fp16_group=h[0],fp16_local=h[1],names=names))
(r/'plans.json').write_text(json.dumps(plans,indent=2)+'\n')
jobs=[]
for rep in range(3):
 for p in plans[rep:]+plans[:rep]:
  m,k,n=p['shape'];jobs.append(dict(tag=f'final-{m}-{k}-{n}-r{rep}',binary='worker',args=list(map(str,[m,k,n,610908+rep,0]))+[','.join(p['names']),'9']))
(r/'final-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(json.dumps(plans,indent=2))
