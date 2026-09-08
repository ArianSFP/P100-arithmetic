#!/usr/bin/env python3
"""Audit completed artifact sets without opening a CUDA context."""
import hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
versions={}
for p in R.iterdir():
 if p.is_file():versions.setdefault(sha(p),[]).append(p.name)
report=[]
for run in sorted(p for p in R.iterdir() if p.is_dir() and (p/'manifest.json').exists()):
 m=json.loads((run/'manifest.json').read_text());entry=dict(run=run.name,final_health=(run/'final-health.txt').exists(),sources={},workers=[])
 for name,digest in m['hashes'].items():
  matches=versions.get(digest,[])
  entry['sources'][name]=dict(sha256=digest,retained_as=matches)
  assert matches, (run,name,'no retained source/binary matches manifest')
 for p in sorted(run.glob('*.jsonl')):
  rows=[];errors=[]
  for line in p.read_text().splitlines():
   if line.startswith('{'):rows.append(json.loads(line))
   elif line.startswith('=========') and 'ERROR SUMMARY:' in line:errors.append(line)
  assert rows and rows[0]['type']=='device',p
  for row in rows:
   for k,v in row.items():
    if k.endswith('_us'):assert math.isfinite(v) and v>=0,(p,row)
  checks=[r for r in rows if r['type'] in ['frozen_check','fused_input_check']]
  assert all(c['different']==0 for c in checks),p
  assert all('ERROR SUMMARY: 0 errors' in e for e in errors),p
  err=p.with_suffix('.err');assert not err.read_text().strip(),err
  entry['workers'].append(dict(file=p.name,rows=len(rows),exact_checks=len(checks),memcheck_clean=bool(errors)))
 report.append(entry)
for run in sorted(p for p in R.glob('large-*') if p.is_dir()):
 entry=dict(run=run.name,final_health=(run/'final-health.txt').exists(),workers=[])
 for d in sorted(p for p in run.iterdir() if p.is_dir()):
  result=d/'result.json'
  assert result.exists(), (d,'worker incomplete')
  m=json.loads(result.read_text());assert m['status']=='PASS' and m['returncode']==0,(d,m)
  for key in ['source_sha256','binary_sha256']:assert m[key] in versions,(d,key,'no matching retained file')
  rows=[];mem=False
  for line in (d/'stdout.jsonl').read_text().splitlines():
   if line.startswith('{'):rows.append(json.loads(line))
   elif 'ERROR SUMMARY:' in line:assert 'ERROR SUMMARY: 0 errors' in line;mem=True
  assert not (d/'stderr.txt').read_text().strip(),d
  final=[x for x in rows if x['type'] in ['post_timing_check','confirmation_check']]
  assert final and all(x['different']==0 for x in final),d
  entry['workers'].append(dict(file=str(d.relative_to(R)),rows=len(rows),memcheck_clean=mem))
 report.append(entry)
(R/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS',sum(len(r['workers']) for r in report),'workers;',sum(w['memcheck_clean'] for r in report for w in r['workers']),'clean memchecks; all completed source/binary hashes retained')
