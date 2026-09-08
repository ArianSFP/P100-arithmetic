#!/usr/bin/env python3
from pathlib import Path
import json,statistics
r=Path(__file__).resolve().parent;xs=json.loads((r/'summary.json').read_text());plans=json.loads((r/'plans.json').read_text());out=[]
for p in plans:
 ws=[x for x in xs if x['shape']==p['shape'] and '/final-' in x['path']]
 if len(ws)!=3:continue
 meds={n:statistics.median(w['configs'][n]['us'] for w in ws) for n in p['names']}
 valid=[n for n in meds if ws[0]['configs'][n]['check']['different']==0 and (ws[0]['configs'][n]['check']['mode']<0 or ws[0]['configs'][n]['check']['mode']%3==0)]
 f=min(valid,key=lambda n:meds[n]);g=p['fp16_group'];h=p['fp16_local']
 out.append(dict(shape=p['shape'],fp32=f,group=g,local=h,median_us=meds,ratio_old_group=meds['r3b4']/meds[g],ratio_old_local=meds['r3b4']/meds[h],ratio_best_group=meds[f]/meds[g],ratio_best_local=meds[f]/meds[h],worker_paths=[w['path'] for w in ws],samples={n:[w['configs'][n]['us'] for w in ws] for n in meds}))
(r/'confirmation-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out:print(x['shape'],x['median_us'],'vs old',x['ratio_old_local'],'vs strongest',x['ratio_best_local'])
