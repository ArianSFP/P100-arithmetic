#!/usr/bin/env python3
"""Prepare by default. Compiler only with explicit --compile after parent slot."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'qwen35-q4-t64-20260908/affinity-wave.cu'
EXPECTED='fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def block(s,start):
    a=s.index(start);b=s.index('{',a)+1;depth=1
    while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[a:b]
def main():
    if sys.argv[1:] not in ([],['--compile']):raise SystemExit('use no arguments or --compile')
    assert sha(SOURCE)==EXPECTED
    s=SOURCE.read_text();parts=[]
    parts.append(s[s.index('__host__ __device__ static inline int aw_q4_tag'):s.index('struct aw_work_desc {')])
    for name in ['aw_work_desc','aw_tile_desc','aw_smem_m64_n128_256']:parts.append(block(s,'struct '+name+' {')+';')
    for name in ['aw_bf16_to_float','aw_float_to_bf16']:
        start=s.rfind('__device__',0,s.index(name+'('));parts.append(block(s,s[start:s.index(name+'(',start)+len(name)+1]))
    start=s.rfind('__device__',0,s.index('aw_load_bf16x8('));parts.append(block(s,s[start:s.index('aw_load_bf16x8(',start)+len('aw_load_bf16x8(')]))
    parts.append(block(s,'template<bool BF16_INPUT>\n__global__ __launch_bounds__(256, 2)\nstatic void aw_q8_service_m64_n128_halfpipe_sync('))
    out=ROOT/'build';out.mkdir(exist_ok=True)
    (out/'current.inc').write_text('\n'.join(parts))
    command=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14','-O3','-std=c++17','-arch=sm_60','-lineinfo','-Xptxas=-v','-I'+str(out),str(ROOT/'worker.cu'),'-o',str(out/'worker')]
    if '--compile' in sys.argv:
        r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/'compile.log').write_text(r.stdout);print(r.stdout[-2000:]);r.check_returncode()
        r=subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','-sass',str(out/'worker')],capture_output=True,text=True,check=True)
        (out/'worker.sass').write_text(r.stdout)
        assert 'HFMA2' not in r.stdout and 'HMUL2' not in r.stdout
    paths=[SOURCE,ROOT/'build.py',ROOT/'worker.cu',ROOT/'kernels.cuh',ROOT/'supervise.py',ROOT.parents[1]/'w4a16-t64-r1-20260908/supervise.py',out/'current.inc']
    if '--compile' in sys.argv:paths.append(out/'worker')
    (out/'manifest.json').write_text(json.dumps({'hashes':{str(p):sha(p) for p in paths},'command':command,'compiled':'--compile' in sys.argv,'gpu_tested':False},indent=2)+'\n')
    print('Prepared; CUDA not initialized. Compile requested:', '--compile' in sys.argv)
if __name__=='__main__':main()
