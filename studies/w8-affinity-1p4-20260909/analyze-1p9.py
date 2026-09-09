from pathlib import Path
import hashlib,json,re,statistics as st
root=Path(__file__).resolve().parent
rows=[]
names=[p.name for p in root.glob('dual-*')]+['virtual-raw-f16','virtual-raw-f16-only','planned-raw-f16','compact-plan-raw','compact-plan-dimensions','compact-plan-f32']
for name in sorted(names):
 p=root/name
 for proj in ['gu','down']:
  f=p/(proj+'-r0.out')
  if not f.exists() or 'PASS' not in f.read_text():continue
  meta=json.loads((p/(proj+'-r0.meta.json')).read_text());assert meta['exit_code']==0
  for item,h in meta['build']['hashes'].items():assert hashlib.sha256((p/item).read_bytes()).hexdigest()==h,(name,item)
  samples={}
  for b,rep,us in re.findall(r'TIME A(\d+) prep=\d+ rep=(\d+) us=([\d.]+)',f.read_text()):
   if int(rep)>0:samples.setdefault(b,[]).append(float(us))
  assert all(len(v)==6 for v in samples.values())
  med={k:st.median(v) for k,v in samples.items()};base=min(med['16'],med['32'])
  rows.append(dict(variant=name,projection=proj,baseline_us=base,candidate_us=med['512'],speedup=base/med['512'],device=meta['device']))
(root/'target-1p9-screen.json').write_text(json.dumps(rows,indent=2)+'\n')
selected={}
for name in ['compact-plan-f32','compact-plan-dimensions','dual-raw-f16']:
 p=root/name;manifest=json.loads((p/'manifest.json').read_text());summary=json.loads((p/'summary.json').read_text())
 for f,h in manifest['hashes'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h,(name,f)
 for row in summary['workers']:
  tag=row['tag'];meta=json.loads((p/(tag+'.meta.json')).read_text());assert meta['build']==manifest and meta['exit_code']==0
  matches=re.findall(r'MATCH_PREVIOUS A\d+ values=\d+ bad=(\d+)',(p/(tag+'.out')).read_text());assert len(matches)==4 and all(int(n)==0 for n in matches)
 selected[name]=summary['summary']
(root/'target-1p9-confirmed.json').write_text(json.dumps(selected,indent=2)+'\n')
print('Audited',len(rows)//2,'screened variants and',len(selected),'confirmed variants')
