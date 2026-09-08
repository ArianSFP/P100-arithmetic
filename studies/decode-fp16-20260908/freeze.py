#!/usr/bin/env python3
from pathlib import Path
import json
r=Path(__file__).resolve().parent
summary=json.loads((r/'summary.json').read_text())
plans=[]
for x in summary:
 if not x['path'].split('/')[-1].startswith('v4-') or x['sanitizer']:continue
 cs=x['configs'];names=list(cs)
 f32=min((n for n in names if cs[n]['check']['mode']<0 or cs[n]['check']['mode']%3==0),key=lambda n:cs[n]['us'])
 h1=min((n for n in names if cs[n]['check']['mode']%3==1 and cs[n]['check']['mode']>=0),key=lambda n:cs[n]['us'])
 h2=min((n for n in names if cs[n]['check']['mode']%3==2 and cs[n]['check']['mode']>=0),key=lambda n:cs[n]['us'])
 def match(n):return n[0]+str(int(n[1])-int(n[1])%3)+n[2:]
 selected=list(dict.fromkeys([n for n in names if n.startswith('r3')]+[f32,h1,h2,match(h1),match(h2)]))
 plans.append(dict(shape=x['shape'],screen=x['path'],fp32=f32,fp16_group=h1,fp16_local=h2,matched_group_fp32=match(h1),matched_local_fp32=match(h2),configs=selected,accuracy='EXPERIMENTAL: PPL/KLD gate not passed; not production dispatch'))
(r/'plans.json').write_text(json.dumps(plans,indent=2)+'\n')
jobs=[]
for rep in range(3):
 for p in (plans[rep:]+plans[:rep]):
  m,k,n=p['shape'];jobs.append(dict(tag=f'confirm-{m}-{k}-{n}-r{rep}',binary='worker',args=list(map(str,[m,k,n,610908+rep,0]))+[','.join(p['configs']),'9']))
(r/'confirmation-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
print('Froze',len(plans),'shape plans;',len(jobs),'fresh paired workers')
