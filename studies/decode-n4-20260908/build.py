#!/usr/bin/env python3
"""Build only while this study owns the rig lock; no CUDA initialization."""
from pathlib import Path
import subprocess,json,hashlib,sys
r=Path(__file__).resolve().parent
cmd=['/usr/bin/taskset','--cpu-list','0-11','/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14','-std=c++17','-O3','-arch=sm_60','--ftz=false','-lineinfo','-Xptxas=-v','-Xcompiler=-ffp-contract=off',str(r/'worker.cu'),'-o',str(r/'worker')]
p=subprocess.run(cmd,capture_output=True,text=True)
(r/'compile.log').write_text(p.stdout+p.stderr)
(r/'compile.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode),indent=2)+'\n')
if p.returncode:print(p.stderr);sys.exit(p.returncode)
p=subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(r/'worker')],capture_output=True,text=True,check=True)
(r/'worker.sass').write_text(p.stdout)
files=['worker.cu','kernels.cuh','r3-baseline.cuh','common.hpp','worker','worker.sass','build.py']
(r/'build-manifest.json').write_text(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files},indent=2)+'\n')
print('BUILD PASS')
