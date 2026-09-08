#!/usr/bin/env python3
from pathlib import Path
import json,statistics
r=Path(__file__).resolve().parent
xs=json.loads((r/'summary.json').read_text());plans=json.loads((r/'plans.json').read_text());out=[]
for p in plans:
 shape=p['shape'];workers=[x for x in xs if x['shape']==shape and '/isolated-' in x['path']]
 if not workers:continue
 arm={}
 for key in ['fp32','fp16_local']:
  name=p[key];v=[x['configs'][name]['us'] for x in workers if name in x['configs']];assert len(v)==2
  kernel=[x['configs'][name]['us'] for x in xs if x['shape']==shape and '/kernelonly-' in x['path'] and name in x['configs']];assert len(kernel)==1
  arm[key]=dict(name=name,worker_us=v,median_us=statistics.median(v),kernel_only_us=kernel[0])
 out.append(dict(shape=shape,arms=arm,ratio=arm['fp32']['median_us']/arm['fp16_local']['median_us'],kernel_only_ratio=arm['fp32']['kernel_only_us']/arm['fp16_local']['kernel_only_us']))
(r/'isolated-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out:print(x['shape'],f"isolated {x['ratio']:.3f}x, kernel-only {x['kernel_only_ratio']:.3f}x")
