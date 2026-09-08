#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,re
r=Path(__file__).resolve().parent
hashes={}
for p in r.rglob('*'):
 if p.is_file() and (p.suffix in ['.cu','.cuh','.hpp'] or p.name=='worker'):
  hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(),[]).append(str(p.relative_to(r)))
workers=[];checks=outputs=oracle=0;sanitizers=[];nonfinite=[]
for p in sorted(r.glob('run-*/*/result.json')):
 x=json.loads(p.read_text());assert x['status']=='PASS',(p,x)
 assert x['returncode']==0
 for key in ['binary_sha256','source_sha256']:assert x[key] in hashes,(p,key,x[key])
 for h in x['other_sources'].values():assert h in hashes,(p,h)
 rows=[json.loads(s) for s in (p.parent/'stdout.jsonl').read_text().splitlines() if s.startswith('{')]
 assert rows[-1]==dict(type='done',status='PASS'),p
 data=next(x for x in rows if x['type']=='data')
 cs=[x for x in rows if x['type']=='check']
 for c in cs:
  checks+=1;outputs+=c['outputs'];oracle+=c['oracle_checks'];assert c['local_bytes']==0
  if c['nonfinite']:nonfinite.append(dict(path=str(p.parent.relative_to(r)),shape=[data['M'],data['K'],data['N']],family=data['family'],name=c['name'],nonfinite=c['nonfinite'],outputs=c['outputs']))
  if data['family']==0:assert c['nonfinite']==0
  if c['mode']<0:assert c['different']==0
 if 'compute-sanitizer' in ' '.join(x['command']):
  combined=(p.parent/'stdout.jsonl').read_text()+(p.parent/'stderr.txt').read_text()
  assert re.search(r'(ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\))',combined),p
  sanitizers.append(str(p.parent.relative_to(r)))
 workers.append(str(p.parent.relative_to(r)))
expected=json.loads((r/'final-jobs.json').read_text());done={Path(x).name for x in workers}
extra=json.loads((r/'extra-jobs.json').read_text())
missing=[j['tag'] for j in expected+extra if j['tag'] not in done]
for d in r.glob('run-*'):
 if (d/'final-health.txt').exists():assert ', 5,' in (d/'final-health.txt').read_text()
result=dict(status='PASS' if not missing else 'INCOMPLETE',workers=len(workers),config_checks=checks,compared_outputs=outputs,independent_cpu_dots=oracle,sanitizers=sanitizers,final_workers=sum(Path(x).name.startswith('final-') for x in workers),missing_final_workers=missing,nonfinite_diagnostics=nonfinite,all_source_and_binary_hashes_retained=True,final_health_present=all((d/'final-health.txt').exists() for d in r.glob('run-*')))
(r/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],len(workers),'workers;',checks,'configuration checks;',oracle,'CPU dots;',len(sanitizers),'clean sanitizer workers; final',result['final_workers'],'/',len(expected),'released',result['final_health_present'])
