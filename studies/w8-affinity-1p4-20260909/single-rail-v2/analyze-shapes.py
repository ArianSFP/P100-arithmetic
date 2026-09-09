from pathlib import Path
import json,re,statistics as st,hashlib
p=Path(__file__).resolve().parent;rows=[]
for m in [64,128,256,512]:
 for proj in ['gu','down']:
  workers=[]
  for r in range(3):
   base=p;tag=f'{proj}-r{r}' if m==256 else f'm{m}-{proj}-r{r}'
   raw=(base/f'{tag}.out').read_text();meta=json.loads((base/f'{tag}.meta.json').read_text());assert meta['exit_code']==0 and 'PASS' in raw
   for f,h in meta['build']['hashes'].items():assert hashlib.sha256((base/f).read_bytes()).hexdigest()==h
   ck=re.findall(r'CHECK A(\d+) outputs=(\d+) samples=(\d+) bad=(\d+) nonfinite=(\d+) changed=(\d+) rel_l2=([\d.e+-]+)',raw);assert len(ck)==6 and all(int(c[3])==int(c[4])==0 for c in ck)
   d={}
   for b,rn,u in re.findall(r'TIME A(\d+) prep=1 rep=(\d+) us=([\d.]+)',raw):
    if int(rn)>0:d.setdefault(b,[]).append(float(u))
   assert all(len(v)==6 for v in d.values());med={b:st.median(v) for b,v in d.items()};ref=min(med['16'],med['32'])
   workers.append({'tag':tag,'base_us':ref,'us':med,'speedup':{b:ref/med[b] for b in ['64','128','256','512']},'rel_l2':{c[0]:float(c[-1]) for c in ck}})
  row={'m':m,'projection':proj,'base_us':st.median(w['base_us'] for w in workers),'speedup':{b:st.median(w['speedup'][b] for w in workers) for b in ['64','128','256','512']},'workers':workers};rows.append(row)
(p/'shape-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['m'],r['projection'],round(r['base_us'],2),{k:round(v,4) for k,v in r['speedup'].items()})
