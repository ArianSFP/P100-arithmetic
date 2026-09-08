#!/usr/bin/env python3
"""Offline build, CPU checks and SASS inventory. Does not initialize CUDA."""
from pathlib import Path
import collections
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)

def run(args):
    p=subprocess.run([str(x) for x in args],text=True,capture_output=True,timeout=60)
    return {'argv':[str(x) for x in args],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

build=run(['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14','-std=c++17','-O3',
           '-arch=sm_60','-lineinfo','-Xptxas=-v',ROOT/'endpoint_sad.cu','-o',ROOT/'endpoint-sad'])
(OUT/'build.json').write_text(json.dumps(build,indent=2)+'\n')
if build['returncode']: raise SystemExit(build['stderr'])
cpu=run([ROOT/'endpoint-sad','--cpu'])
(OUT/'cpu.json').write_text(json.dumps(cpu,indent=2)+'\n')
print(cpu['stdout'])
if cpu['returncode']: raise SystemExit(cpu['stderr'] or 'CPU test failed')
sass=run(['/usr/local/cuda-12.8/bin/cuobjdump','-sass',ROOT/'endpoint-sad'])
if sass['returncode']: raise SystemExit(sass['stderr'])
(OUT/'endpoint-sad.sass').write_text(sass['stdout'])
functions={}
for name,section in re.findall(r'Function\s*:\s*(\S+)\n(.*?)(?=Function\s*:|\Z)',sass['stdout'],re.S):
    ins=[s.strip().lstrip('{').strip() for s in re.findall(r'/\*[0-9a-f]+\*/\s+([^;\n]+);',section)]
    ins=[re.sub(r'^@!?P\d+\s+', '', s) for s in ins]
    counts=collections.Counter(s.split()[0].split('.')[0] for s in ins)
    functions[name]={'counts_including_setup':dict(counts),'instructions':ins}
    if 'dot_probe' in name:
        print(name, ' '.join(f'{op}={counts[op]}' for op in ('VABSDIFF4','PRMT','LOP','LOP3','LOP32I','POPC','VMAD','XMAD','SHR','SHL')))
record={'source_sha256':hashlib.sha256((ROOT/'endpoint_sad.cu').read_bytes()).hexdigest(),
        'binary_sha256':hashlib.sha256((ROOT/'endpoint-sad').read_bytes()).hexdigest(),
        'functions':functions,'scope':'SASS inventory only; no GPU execution'}
resources={}
for name,stack,stores,loads,registers in re.findall(
    r"Compiling entry function '([^']+)'.*?(\d+) bytes stack frame, (\d+) bytes spill stores, "
    r"(\d+) bytes spill loads.*?Used (\d+) registers",build['stderr'],re.S):
    resources[name]={'registers':int(registers),'stack_bytes':int(stack),
                     'spill_store_bytes':int(stores),'spill_load_bytes':int(loads)}
assert len(functions)==37 and set(resources)==set(functions), 'Missing compiled kernel'
assert all(not r['spill_store_bytes'] and not r['spill_load_bytes'] for r in resources.values()), 'Spill gate failed'
for bits in (2,4):
    for method in (2,3,4,5,7,8):
        name=f'_Z9dot_probeILi{bits}ELi{method}EEvPKjS1_PKsS3_Pii'
        body=functions[name]['instructions']
        counts=functions[name]['counts_including_setup']
        assert counts['VABSDIFF4']==8*bits and counts['PRMT']==8*bits, name
        assert sum(s.startswith('VABSDIFF4.U8.U8.ACC ') for s in body)==8*bits, name
        if method in (7,8):
            logic=[s for s in body if s.startswith('LOP')]
            assert len(logic)==8 and all(s.startswith('LOP32I.XOR ') and s.endswith('0x80808080') for s in logic), name
        if method in (2,4):
            assert sum(s.startswith('LOP.AND ') for s in body)==8*bits, name
record['resources']=resources
record['offline_gates']='PASS: CPU math/layout, native unsigned SAD, direct endpoint logical-op count, zero spills'
(OUT/'inventory.json').write_text(json.dumps(record,indent=2)+'\n')
print(record['offline_gates'])
