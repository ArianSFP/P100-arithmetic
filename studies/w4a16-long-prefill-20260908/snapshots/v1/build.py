#!/usr/bin/env python3
"""Isolated current-service control and Q4_0 specialization, no CUDA execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'w4a16-delivery-leads-20260908'
SOURCE=ROOT.parent/'qwen35-q4-t64-20260908/affinity-wave.cu'
EXPECTED='fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
def block(s,start):
    a=s.index(start);b=s.index('{',a)+1;depth=1
    while depth:
        depth+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[a:b]

def main():
    assert digest(SOURCE)==EXPECTED
    old=json.loads((BASE/'build/manifest.json').read_text())
    for name,h in old['hashes'].items():assert digest(Path(name))==h,name
    source=SOURCE.read_text()
    helpers=source[source.index('__host__ __device__ static inline int aw_q4_tag'):source.index('struct aw_work_desc {')]
    shared=block(source,'struct aw_smem_m64_n128_256 {')+';'
    kernel=block(source,'template<bool BF16_INPUT>\n__global__ __launch_bounds__(256, 2)\nstatic void aw_q8_service_m64_n128_halfpipe_sync(')
    specialized=kernel.replace('aw_q8_service_m64_n128_halfpipe_sync','aw_special_q4')
    specialized=specialized.replace('aw_q4_tag(desc.weight)','1').replace('((uintptr_t)desc.weight & 4u)','true')
    specialized=specialized.replace('aw_q4_load16(', 'known_load16(').replace('aw_q4_scale(', 'known_scale(').replace('aw_q4_offset(', 'known_offset(')
    fixed='''
__device__ __forceinline__ const char* known_stage(const char*p,int row,int g,int k){
    return (const char*)((uintptr_t)p & ~(uintptr_t)7)+((size_t)(row/64)*(k/32)+g)*1152;
}
__device__ __forceinline__ uint4 known_load16(const char*p,int row,int g,int k){return __ldg((const uint4*)(known_stage(p,row,g,k)+128+(row%64)*16));}
__device__ __forceinline__ float known_scale(const char*p,int row,int g,int k){return __half2float(*(const half*)(known_stage(p,row,g,k)+(row%64)*2));}
__device__ __forceinline__ float known_offset(const char*,int,int,int){return 0.f;}
'''
    build=ROOT/'build';build.mkdir(exist_ok=True)
    for name in ('aw_control.inc','aw_q4_control.inc'):(build/name).write_bytes((BASE/'build'/name).read_bytes())
    (build/'current_control.inc').write_text(helpers+'\n'+shared+'\n'+kernel+'\n'+fixed+'\n'+specialized)
    s=(BASE/'build/worker.cu').read_text()
    s=once(s,'#include "aw_q4_control.inc"','#include "aw_q4_control.inc"\n#include "current_control.inc"\nstatic aw_work_desc *current_desc;')
    s=once(s,'    if(c.kind==1){launch_delivery','    if(c.kind==-4){aw_q8_service_m64_n128_halfpipe_sync<false><<<grid,256>>>(current_desc,tiles,ntiles,n,k);return;}\n    if(c.kind==-5){aw_special_q4<false><<<grid,256>>>(current_desc,tiles,ntiles,n,k);return;}\n    if(c.kind==1){launch_delivery')
    s=once(s,'    std::vector<Config>cfg=', '''    auto current=d32;
    for(int e=0;e<experts;++e)current[e].weight=(const char*)((uintptr_t)(dw4.p+size_t(e)*n*groups*18)|5u);
    Device<aw_work_desc> dc(current.size());dc.put(current);current_desc=dc.p;
    std::vector<Config>cfg=''')
    s=once(s,'    #define ADD(M,N,B,S,D,U)', '''    cfg.push_back({64,128,256,2,0,16,-4,"aw_current_q4"});
    cfg.push_back({64,128,256,2,0,16,-5,"aw_special_q4"});
    #define ADD(M,N,B,S,D,U)''')
    s=once(s,'if(prep&&c.kind!=-3&&','if(prep&&c.kind!=-3&&c.kind!=-4&&c.kind!=-5&&')
    (build/'worker.cu').write_text(s)
    if sys.argv[1:]==['--prepare-only']:
        print('Prepared current control and Q4_0 specialization; no compiler or CUDA execution')
        return
    if sys.argv[1:]:raise SystemExit('Expected no arguments or --prepare-only')
    command=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14','-O3','-std=c++17','-arch=sm_60','-lineinfo','-Xptxas=-v','-I'+str(build),str(build/'worker.cu'),'-o',str(build/'worker')]
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (build/'compile.log').write_text(result.stdout);print(result.stdout[-2000:]);result.check_returncode()
    with (build/'worker.sass').open('w') as f:subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','-sass',str(build/'worker')],stdout=f,check=True)
    sass=(build/'worker.sass').read_text();assert 'HFMA2' not in sass and 'HMUL2' not in sass
    hashes=old['hashes']|{str(p):digest(p) for p in [SOURCE,Path(__file__),ROOT/'supervise.py',build/'worker.cu',build/'current_control.inc',build/'aw_control.inc',build/'aw_q4_control.inc',build/'worker']}
    (build/'manifest.json').write_text(json.dumps(dict(hashes=hashes,command=command,control_source_sha256=EXPECTED,scope='Equal-M screening, not ragged/full-model acceptance'),indent=2)+'\n')

if __name__=='__main__':main()
