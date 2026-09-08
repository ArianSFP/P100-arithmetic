from pathlib import Path
import json,hashlib,re,statistics
r=Path(__file__).resolve().parent
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
retained={h(p) for p in r.rglob('*') if p.is_file() and p.name in ['worker','worker.cu','kernels.cuh','r3-baseline.cuh','common.hpp']}
counts=dict(workers=0,checks=0,outputs=0,oracle_checks=0,anchors=0,sanitizers=0)
for p in r.glob('run-*/*/result.json'):
 m=json.loads(p.read_text());assert m['status']=='PASS' and m['returncode']==0
 for v in [m['binary_sha256'],m['source_sha256'],*m['other_sources'].values()]:assert v in retained,(p,v)
 counts['workers']+=1
 if 'compute-sanitizer' in ' '.join(m['command']):
  counts['sanitizers']+=1;e=(p.parent/'stderr.txt').read_text()+(p.parent/'stdout.jsonl').read_text();assert re.search(r'(ERROR|RACECHECK) SUMMARY: 0',e),e[-1000:]
 for line in (p.parent/'stdout.jsonl').read_text().splitlines():
  if not line.startswith('{'):continue
  x=json.loads(line)
  if x['type']=='check':
   counts['checks']+=1;counts['outputs']+=x['outputs'];counts['oracle_checks']+=x['oracle_checks']
   if (x['mode']<=0 or x['mode']%12==9) and x['split']==32:assert x['different']==0
  if x['type']=='geometry' and x['anchor_bitidentical']:counts['anchors']+=1
parent=r.parent/'decode-fp16-20260908';manifest=json.loads((parent/'ARTIFACT-MANIFEST.json').read_text())
for n,digest in manifest['files'].items():assert h(parent/n)==digest,n
for p in [r/'compile.log',*r.glob('snapshots/*/compile.log')]:
 for a,b,c in re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',p.read_text()):assert (a,b,c)==('0','0','0')
xs=json.loads((r/'summary.json').read_text());ps=json.loads((r/'plans.json').read_text());isolated=[]
assert len([x for x in xs if '/final-' in x['path']])==9
for p in ps:
 d={n:[x['configs'][n]['us'] for x in xs if '/isolated-' in x['path'] and x['shape']==p['shape'] and n in x['configs']] for n in [p['fp32'],p['fp16_local']]}
 assert all(len(v)==2 for v in d.values());isolated.append(dict(shape=p['shape'],samples=d,ratio=statistics.median(d[p['fp32']])/statistics.median(d[p['fp16_local']])))
run=r/'run-1788906374375597879';con=[json.loads(s) for s in (run/'concurrency.jsonl').read_text().splitlines()];gpu3=sum(any(line.startswith('3,') and int(line.split(',')[3])>0 for line in x['gpu']['stdout'].splitlines()) for x in con if x['gpu']['returncode']==0)
out=dict(status='PASS',counts=counts,parent_manifest_files_verified=len(manifest['files']),isolated=isolated,monitor_samples=len(con),gpu3_busy_samples=gpu3,final_health=(run/'final-health.txt').read_text())
(r/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
