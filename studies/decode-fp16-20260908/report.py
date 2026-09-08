#!/usr/bin/env python3
from pathlib import Path
import json,statistics
r=Path(__file__).resolve().parent
summary=json.loads((r/'summary.json').read_text());plans=json.loads((r/'plans.json').read_text())
results=[]
for p in plans:
 workers=[x for x in summary if x['shape']==p['shape'] and x['path'].split('/')[-1].startswith('final-')]
 if len(workers)!=3:continue
 names=p['configs'];med={n:statistics.median(x['configs'][n]['us'] for x in workers) for n in names}
 f32=min((n for n in names if workers[0]['configs'][n]['check']['mode']<0 or workers[0]['configs'][n]['check']['mode']%3==0),key=lambda n:med[n])
 h1=p['fp16_group'];h2=p['fp16_local'];base=med[f32]
 row=dict(shape=p['shape'],fp32=f32,fp32_us=base,group_fp16=h1,group_us=med[h1],group_ratio=base/med[h1],local_fp16=h2,local_us=med[h2],local_ratio=base/med[h2],local_matched_ratio=med[p['matched_local_fp32']]/med[h2],worker_paths=[x['path'] for x in workers],ranges={n:[min(x['configs'][n]['us'] for x in workers),max(x['configs'][n]['us'] for x in workers)] for n in [f32,h1,h2]},local_relL2=[x['configs'][h2]['check']['finite_relL2'] for x in workers],local_different=[x['configs'][h2]['check']['different'] for x in workers])
 results.append(row)
(r/'confirmation-summary.json').write_text(json.dumps(results,indent=2)+'\n')
print('| M | K | Tokens | FP32 us | FP16 group us | Gain | FP16 local us | Gain | Same-mapping ratio |')
print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for x in sorted(results,key=lambda x:(x['shape'][2],x['shape'][0],x['shape'][1])):
 m,k,n=x['shape'];print(f"| {m} | {k} | {n} | {x['fp32_us']:.3f} | {x['group_us']:.3f} | {x['group_ratio']:.3f}x | {x['local_us']:.3f} | {x['local_ratio']:.3f}x | {x['local_matched_ratio']:.3f}x |")
print('Completed confirmation groups',len(results),'of',len(plans))
