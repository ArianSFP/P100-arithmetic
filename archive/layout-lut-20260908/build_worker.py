#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import subprocess
ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'endpoint-sad-20260908/endpoint_sad.cu'
baseline_args=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14','-std=c++17','-O3','-arch=sm_60','-lineinfo','-cubin',str(source),'-o',str(ROOT/'gpu-baseline.cubin')]
bp=subprocess.run(baseline_args,capture_output=True,text=True,timeout=60)
if bp.returncode:raise SystemExit(bp.stderr)
argv=['/usr/bin/g++-14','-O3','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Wno-unknown-pragmas',
      '-I/usr/local/cuda-12.8/include',str(ROOT/'gpu_worker.cpp'),'-o',str(ROOT/'gpu-worker'),
      '-L/usr/local/cuda-12.8/lib64/stubs','-lcuda']
p=subprocess.run(argv,capture_output=True,text=True,timeout=60)
record=dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
            source_sha256=hashlib.sha256((ROOT/'gpu_worker.cpp').read_bytes()).hexdigest(),
            baseline_argv=baseline_args,baseline_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            baseline_cubin_sha256=hashlib.sha256((ROOT/'gpu-baseline.cubin').read_bytes()).hexdigest())
if not p.returncode:record['worker_sha256']=hashlib.sha256((ROOT/'gpu-worker').read_bytes()).hexdigest()
(ROOT/'gpu-build.json').write_text(json.dumps(record,indent=2)+'\n')
print(p.stderr or 'PASS host worker compilation; no GPU initialized')
raise SystemExit(p.returncode)
