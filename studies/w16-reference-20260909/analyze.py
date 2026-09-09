from pathlib import Path
import json,re,hashlib,statistics as st
p=Path(__file__).resolve().parent
manifest=json.loads((p/'manifest.json').read_text())
for f,h in manifest['hashes'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
rows=[]
for proj in ['gu','down']:
 for r in range(3):
  tag=f'{proj}-r{r}';raw=(p/f'{tag}.out').read_text();meta=json.loads((p/f'{tag}.meta.json').read_text())
  assert meta['exit_code']==0 and meta['build']==manifest and 'PASS' in raw
  checks=re.findall(r'CHECK A(\d+) outputs=(\d+) samples=(\d+) bad=(\d+) nonfinite=(\d+) changed=(\d+) rel_l2=([\d.e+-]+)',raw)
  assert len(checks)==5
  for b,n,s,bad,nf,c,l in checks:
   assert int(bad)==int(nf)==0
   if b in ['32','16']:assert int(c)==0
  samples={}
  for b,prep,rep,us in re.findall(r'TIME A(\d+) prep=(\d+) rep=(\d+) us=([\d.]+)',raw):
   if int(rep)>0:samples.setdefault(f'A{b}_prep{prep}',[]).append(float(us))
  assert len(samples)==7 and all(len(v)==6 for v in samples.values())
  med={k:st.median(v) for k,v in samples.items()};best=min(med['A32_prep1'],med['A16_prep1'])
  rows.append(dict(tag=tag,device=meta['device'],median_us=med,best_q8_us=best,w16_speedup=best/med['A64_prep1'],w16_rel_l2=float(next(x[-1] for x in checks if x[0]=='64'))))
summary={}
for proj in ['gu','down']:
 rr=[x for x in rows if x['tag'].startswith(proj)]
 summary[proj]={'q8_us':st.median(x['best_q8_us'] for x in rr),'w16_us':st.median(x['median_us']['A64_prep1'] for x in rr),'w16_speedup':st.median(x['w16_speedup'] for x in rr),'speedup_range':[min(x['w16_speedup'] for x in rr),max(x['w16_speedup'] for x in rr)],'w16_rel_l2':rr[0]['w16_rel_l2']}
(p/'summary.json').write_text(json.dumps({'workers':rows,'summary':summary},indent=2)+'\n')
print(json.dumps(summary,indent=2))
