#!/usr/bin/env python3
"""Read recorded workers and hashes; write a new final audit, never launch CUDA."""
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    inventory=json.loads((ROOT/'build/inventory.json').read_text())
    for name,digest in inventory['hashes'].items():assert sha(ROOT/name)==digest,name
    frozen=json.loads((ROOT/'frozen-manifest.json').read_text())
    assert sha(ROOT/'build/inventory.json')==frozen['inventory_sha256']
    assert sha(ROOT/'frozen-configs.txt')==frozen['configuration_sha256']
    aggregate=json.loads((ROOT/'aggregate.json').read_text())
    assert aggregate['worker_count']==12 and aggregate['configuration_checks']==120 and aggregate['retained_event_samples']==960
    counts=collections.Counter();checks=outputs=full_outputs=0;maxnorm=0;runids=[]
    reference={}
    for path in sorted((ROOT/'gpu-results').glob('*/result.json')):
        rec=json.loads(path.read_text());pre=json.loads((path.parent/'prelaunch.json').read_text())
        assert rec['status']=='PASS' and rec['returncode']==0,path
        assert rec['gpu']==2 and rec['gpu_uuid']=='GPU-2aa85c85-bc04-0ac3-fc1b-4827d8303d81'
        assert rec['supervisor_sha256']==sha(ROOT/'supervise.py')
        for name,digest in inventory['hashes'].items():assert rec['hashes'][name]==digest,path
        for name in ('argv','hashes','gpu','gpu_uuid','token','supervisor_sha256','coordination_sha256','shared_sha256','sanitizer'):
            assert rec[name]==pre[name],(path,name)
        assert not (path.parent/'stderr.txt').read_text().strip(),path
        text=(path.parent/'stdout.txt').read_text()
        category=rec['sanitizer'] or next(mode for mode in ('validate','sweep','bench') if mode in rec['argv'])
        counts[category]+=1;runids.append(path.parent.name)
        if rec['sanitizer']:
            assert 'SMOKE_PASS' in text
            assert ('0 hazards displayed (0 errors, 0 warnings)' if category=='racecheck' else 'ERROR SUMMARY: 0 errors') in text
        elif category=='validate':assert 'VALIDATE_PASS' in text and 'EXHAUSTIVE_FINITE_ACTIVATIONS_PASS' in text and 'CONVERSION_PASS patterns=65536' in text
        else:assert text.endswith('BENCH_PASS\n')
        local=0
        for line in text.splitlines():
            if not line.startswith('CHECK '):continue
            data=dict(re.findall(r'(\w+)=([^ ]+)',line))
            assert data['operation_order_bitexact']=='1' and data['baseline_changed']=='0' and data['S']=='32'
            key=tuple(data[k] for k in ('M','K','N','method','R','S','family'))
            assert key not in reference or reference[key]==data
            reference[key]=data;checks+=1;local+=1;outputs+=int(data['outputs'])
            if category=='validate':full_outputs+=int(data['outputs'])
            maxnorm=max(maxnorm,float(data['max_L1_relative']))
        assert local==(507 if category=='validate' else 10 if category=='bench' else 39),(path,local)
    assert dict(counts)==dict(validate=1,memcheck=1,synccheck=1,racecheck=1,sweep=4,bench=12)
    assert checks==900 and full_outputs==388128 and outputs==3712821
    exported=json.loads((REPO/'EXPORT-MANIFEST.json').read_text());expected=set()
    for entry in exported['files']:
        expected.add(entry['path']);path=REPO/'archive'/entry['path']
        assert path.stat().st_size==entry['size'] and sha(path)==entry['sha256'],path
    actual={p.relative_to(REPO/'archive').as_posix() for p in (REPO/'archive').rglob('*') if p.is_file()}
    extras=sorted(actual-expected)
    final_health=json.loads((ROOT/'final-health.json').read_text())
    assert final_health['status']=='PASS' and final_health['gpu']==2
    released='- FINAL RELEASE: w4a16-r3 GPU2 token='+final_health['token']
    assert released in (REPO/'bench/COORDINATION-20260908.md').read_text().splitlines()
    assert released in Path('/home/arian/llama.cpp-qwen38-p100/bench/COORDINATION-20260908.md').read_text().splitlines()
    report=dict(status='PASS',gpu=2,reservation_released=True,final_health_sha256=sha(ROOT/'final-health.json'),
        worker_count=len(runids),worker_categories=dict(counts),runids=runids,
        all_pipeline_checks=checks,all_pipeline_outputs=outputs,full_validation_outputs=full_outputs,
        baseline_bitidentical=True,max_L1_relative=maxnorm,build_hashes=inventory['hashes'],
        frozen_configuration_sha256=frozen['configuration_sha256'],aggregate_sha256=sha(ROOT/'aggregate.json'),
        supervisor_sha256=sha(ROOT/'supervise.py'),report_sha256=sha(ROOT/'RESULTS.md'),
        archived_manifest_entries_verified=len(expected),unmanifested_archive_files=extras,
        strict_whole_archive_status='FAIL_EXTRA_FILES' if extras else 'PASS',
        note='Manifested archive bytes unchanged. Unmanifested Python caches observed during concurrent study work are left untouched.')
    (ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('AUDIT_PASS workers=%d checks=%d outputs=%d archive_manifest_hashes=%d'%(len(runids),checks,outputs,len(expected)))
    print('Archive extras:',extras)
if __name__=='__main__':main()
