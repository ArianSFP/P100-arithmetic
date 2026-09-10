"""CPU/compiler only. Never initializes CUDA or queries a GPU."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NVCC='/usr/local/cuda-12.8/bin/nvcc'
commands=[
 [NVCC,'-ccbin','/usr/bin/g++-14','-arch=sm_60','-std=c++17','-O3','--fmad=false','-Xptxas=-v','-cubin',str(ROOT/'kernels.cu'),'-o',str(ROOT/'kernels.cubin')],
 ['/usr/bin/g++-14','-std=c++17','-O2','-ffp-contract=off','-I/usr/local/cuda-12.8/include',str(ROOT/'worker.cpp'),'-L/usr/local/cuda-12.8/lib64/stubs','-lcuda','-o',str(ROOT/'worker')],
 ['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(ROOT/'kernels.cubin')],
 ['python3',str(ROOT/'cpu_checks.py')],
]
for cmd,name in zip(commands,['build.log','worker-build.log','kernels.sass','cpu.log']):
 p=subprocess.run(cmd,capture_output=True,text=True)
 (ROOT/name).write_text(p.stdout+p.stderr)
 if p.returncode:raise SystemExit(f'Build failed: {name}')
text=(ROOT/'kernels.sass').read_text();out={}
for part in text.split('Function : ')[1:]:
 name=part.splitlines()[0].strip();counts={}
 for op in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]+)',part):
  base=op.split('.')[0];counts[base]=counts.get(base,0)+1
 out[name]=counts
assert len(out)==14
assert all(not c.get('LDL') and not c.get('STL') for c in out.values())
assert out['w4a4_h1_r2_b1']['HFMA2']==32
assert out['w4a4_h1_r2_b4']['HFMA2']==128
assert all(int(x)==0 for x in re.findall(r'(\d+) bytes (?:stack frame|spill stores|spill loads)',(ROOT/'build.log').read_text()))
(ROOT/'sass-counts.json').write_text(json.dumps(out,indent=2)+'\n')
files=['kernels.cu','kernels.cubin','kernels.sass','worker.cpp','worker','cpu_checks.py','cpu-results.json','build.py']
files+=['../../archive/w4a16-r2-20260908/common.hpp','../../archive/w4a16-r2-20260908/build/kernels.cubin']
manifest={'commands':commands,'hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},'gpu_executed':False}
(ROOT/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PASS: CPU arithmetic, 14 compiled entry points, SASS and spill checks; no GPU execution')
