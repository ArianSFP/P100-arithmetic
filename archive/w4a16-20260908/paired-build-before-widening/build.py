#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'build';OUT.mkdir(exist_ok=True)
CUDA=Path('/usr/local/cuda-12.8/bin')
records=[]
def run(argv,name):
    p=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,text=True)
    records.append(dict(argv=list(map(str,argv)),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    (OUT/(name+'.json')).write_text(json.dumps(records[-1],indent=2)+'\n')
    if p.returncode:raise SystemExit(p.stdout+p.stderr)
    return p.stdout
host=['/usr/bin/g++-14','-O3','-std=c++17','-ffp-contract=off','-Wall','-Wextra']
run([*host,'cpu_tests.cpp','-o',OUT/'cpu-tests'],'cpu-build')
print(run([OUT/'cpu-tests'],'cpu-test'),flush=True)
run([*host,'-fsanitize=undefined','-fno-sanitize-recover=all','cpu_tests.cpp','-o',OUT/'cpu-tests-ubsan'],'ubsan-build')
print(run([OUT/'cpu-tests-ubsan'],'ubsan-test'),flush=True)
run([CUDA/'nvcc','-ccbin','/usr/bin/g++-14','-std=c++17','-O3','-arch=sm_60','--ftz=false','-lineinfo','-Xptxas=-v','-cubin','kernels.cu','-o',OUT/'kernels.cubin'],'cuda-build')
sass=run([CUDA/'cuobjdump','--dump-sass',OUT/'kernels.cubin'],'sass')
(OUT/'kernels.sass').write_text(sass)
assert not re.search(r'\b(?:HFMA2|HMUL2)\b',sass),'Unexpected FP16 multiplication/accumulation'
conversions=re.findall(r'\bHADD2[^;]+',sass)
assert all(re.fullmatch(r'HADD2\.F32 R\d+, R\d+\.H0_H0(?:\.reuse)?, -RZ\.H0_H0\s*',s) for s in conversions),'Unexpected HADD2 use beyond exact half widening'
compile_log=records[-2]['stderr']
assert not any(int(x) for x in re.findall(r'(\d+) bytes spill (?:stores|loads)',compile_log)),'Spills found'
assert not any(int(x) for x in re.findall(r'(\d+) bytes stack frame',compile_log)),'Local stack allocation found'
assert not re.search(r'\b(?:LDL|STL)(?:\.|\s)',sass),'Local-memory instruction found'
run([*host,'-I/usr/local/cuda-12.8/include','worker.cpp','-L/usr/local/cuda-12.8/lib64/stubs','-lcuda','-o',OUT/'worker'],'worker-build')
files=['common.hpp','cpu_tests.cpp','kernels.cu','worker.cpp','build/kernels.cubin','build/worker']
inventory=dict(hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
    kernel_count=len(re.findall(r'Function :',sass)),half_multiply_accumulate_instructions=0,
    half_to_float_conversion_instructions=len(conversions),spills=0,commands=records)
(OUT/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print('BUILD_PASS kernels='+str(inventory['kernel_count'])+' no_half_multiply_accumulate=1 no_spills=1; no GPU initialized')
