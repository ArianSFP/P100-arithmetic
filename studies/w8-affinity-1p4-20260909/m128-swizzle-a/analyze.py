from pathlib import Path
import re,json,statistics as st,hashlib
p=Path(__file__).resolve().parent
manifest=json.loads((p/'manifest.json').read_text())
for name,h in manifest['hashes'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h
rows=[]
for proj in ['gu','down']:
 for r in range(3):
  tag=f'{proj}-r{r}';raw=(p/f'{tag}.out').read_text();meta=json.loads((p/f'{tag}.meta.json').read_text())
  assert meta['exit_code']==0 and meta['build']==manifest and 'PASS' in raw
  checks=re.findall(r'CHECK A(\d+) outputs=(\d+) samples=(\d+) bad=(\d+) nonfinite=(\d+) changed=(\d+) rel_l2=([\d.e+-]+)',raw)
  assert len(checks)==8 and all(int(c[3])==int(c[4])==0 for c in checks)
  samples={}
  for b,prep,rep,us in re.findall(r'TIME A(\d+) prep=(\d+) rep=(\d+) us=([\d.]+)',raw):
   if int(rep)>0:samples.setdefault(b,[]).append(float(us))
  assert len(samples)==8 and all(len(x)==6 for x in samples.values())
  med={b:st.median(v) for b,v in samples.items()};base=min(med['16'],med['32'])
  rows.append(dict(tag=tag,device=meta['device'],us=med,base_us=base,speedup={b:base/v for b,v in med.items()},rel_l2={c[0]:float(c[-1]) for c in checks}))
summary={}
for proj in ['gu','down']:
 rr=[r for r in rows if r['tag'].startswith(proj)]
 summary[proj]={'base_us':st.median(r['base_us'] for r in rr),'candidates':{b:{'us':st.median(r['us'][b] for r in rr),'speedup':st.median(r['speedup'][b] for r in rr),'rel_l2':rr[0]['rel_l2'][b]} for b in ['64','128','256','512','4096','8192']}}
(p/'summary.json').write_text(json.dumps({'workers':rows,'summary':summary},indent=2)+'\n');print(json.dumps(summary,indent=2))
