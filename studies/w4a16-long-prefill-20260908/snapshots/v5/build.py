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
    s=once(s,'    if(c.kind==1){launch_delivery','''    if(c.kind==1&&c.column>=5000){
        if(c.column==5016)delivery_lead<4,1,16,0><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==5032)delivery_lead<4,1,32,0><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else throw std::runtime_error("unknown long shape");
        return;
    }
    if(c.kind==1){launch_delivery''')
    s=once(s,'    if(!only.empty())cfg.erase', '''    for(int u:{16,32})cfg.push_back({64,64,128,1,1,u,1,"delivery_fused_m64u"+std::to_string(u),5000+u});
    if(!only.empty())cfg.erase''')
    s=once(s,'c.name!="q4_m64n64b128s1d1u8r";', 'c.name!="q4_m64n64b128s1d1u8r"&&c.name!="delivery_fused_u32";')
    s=once(s,'#include "current_control.inc"','#include "current_control.inc"\n#include "'+str(ROOT/'staging.cuh')+'"\nstatic half *stagedw;')
    s=once(s,'    if(c.kind==1&&c.column>=5000){','''    if(c.kind==1&&c.column>=6000){
        if(c.column==6016)staged_f32<16><<<grid,128>>>(leadraw,stagedw,out,t,n,k,e);
        else if(c.column==6032)staged_f32<32><<<grid,128>>>(leadraw,stagedw,out,t,n,k,e);
        else throw std::runtime_error("unknown staged shape");
        return;
    }
    if(c.kind==1&&c.column>=5000){''')
    s=once(s,'    auto current=d32;', '    Device<half> dsw(wc);stagedw=dsw.p;\n    printf("WEIGHT_SCRATCH bytes=%zu prep1_charged_every_run=1\\n",wc*2);\n    auto current=d32;')
    s=once(s,'    if(!only.empty())cfg.erase', '''    for(int u:{16,32})cfg.push_back({32,64,128,1,1,u,1,"staged_u"+std::to_string(u),6000+u});
    if(!only.empty())cfg.erase''')
    s=once(s,'auto run=[&](const Config&c,bool prep){','''auto run=[&](const Config&c,bool prep){
        if(prep&&c.kind==1&&c.column>=6000)stage_q4_weights<<<std::min<size_t>((wc+255)/256,4096),256>>>(dw4.p,stagedw,wc);''')
    s=once(s,'#include "current_control.inc"','#include "current_control.inc"\n#include "'+str(ROOT/'wide.cuh')+'"')
    s=once(s,'    if(c.kind==1&&c.column>=6000){','''    if(c.kind==1&&c.column>=7000){
        if(c.column==7008)wide_q4<8,false><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==7016)wide_q4<16,false><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==7032)wide_q4<32,false><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==8008)wide_q4<8,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==8016)wide_q4<16,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==8032)wide_q4<32,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==9032)wide_q4<32,true,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==10032)wide_q4<32,false,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else throw std::runtime_error("unknown wide shape");
        return;
    }
    if(c.kind==1&&c.column>=6000){''')
    s=once(s,'if(prep&&c.kind==1&&c.column>=6000)stage_q4_weights', 'if(prep&&c.kind==1&&c.column>=6000&&c.column<7000)stage_q4_weights')
    s=once(s,'    if(!only.empty())cfg.erase', '''    for(int packed:{0,1})for(int u:{8,16,32})cfg.push_back({64,128,128,1,1,u,1,"wide_"+std::string(packed?"packed":"f32")+"_u"+std::to_string(u),7000+packed*1000+u});
    cfg.push_back({64,128,128,2,1,32,1,"wide_packed_split_u32",9032});
    cfg.push_back({64,128,128,2,1,32,1,"wide_f32_split_u32",10032});
    if(!only.empty())cfg.erase''')
    s=once(s,'        if(c.kind==1){\n            if(sequential.empty())', '        if(c.kind==1&&c.sk==1){\n            if(sequential.empty())')
    s=once(s,'uint32_t rng(uint32_t &s)', '''void verify_dequant(){
    Device<unsigned> result(65536u*16);
    decode_check_kernel<<<4096,256>>>(result.p);CU(cudaGetLastError());CU(cudaDeviceSynchronize());
    const auto got=result.get();size_t checked=0,bad=0;
    for(unsigned code=0;code<16;++code)for(unsigned scale=0;scale<65536;++scale){
        if((scale&0x7c00)==0x7c00)continue;
        unsigned short raw=(unsigned short)scale;half d;memcpy(&d,&raw,2);
        const float f=__half2float(d);
        half lo=__float2half_rn(float(int(code)-8)*f),hi=__float2half_rn(float(7-int(code))*f);
        unsigned short lb,hb;memcpy(&lb,&lo,2);memcpy(&hb,&hi,2);
        const unsigned expected=unsigned(lb)|(unsigned(hb)<<16);
        if(got[code*65536+scale]!=expected){
            if(bad<3)printf("DEQUANT_MISMATCH code=%u scale=%04x got=%08x expected=%08x\\n",code,scale,got[code*65536+scale],expected);
            ++bad;
        }
        ++checked;
    }
    printf("DEQUANT_CHECK finite_scale_code_pairs=%zu bad=%zu\\n",checked,bad);
    if(bad)throw std::runtime_error("packed dequantization failed");
    puts("PASS");
}
uint32_t rng(uint32_t &s)''')
    s=once(s,'    std::string only;', '    int quantcheck=0;\n    std::string only;')
    s=once(s,'if(key=="--gpu-approved")approved=v;', 'if(key=="--gpu-approved")approved=v;else if(key=="--quant-check")quantcheck=v;')
    s=once(s,'    int groups=k/32,grid=', '    if(quantcheck){if(quantcheck!=1)throw std::runtime_error("invalid quant check");verify_dequant();return 0;}\n    int groups=k/32,grid=')
    s=once(s,'    int quantcheck=0;', '    int quantcheck=0,scalestress=0;')
    s=once(s,'#include "'+str(ROOT/'wide.cuh')+'"', '#include "'+str(ROOT/'wide.cuh')+'"\n#include "'+str(ROOT/'compact.cuh')+'"')
    s=once(s,'    if(c.kind==1&&c.column>=7000){', '''    if(c.kind==1&&c.column>=11000){
        if(c.column==11032)compact_q4<32,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==12032)compact_q4<32,false><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else throw std::runtime_error("unknown compact shape");
        return;
    }
    if(c.kind==1&&c.column>=7000){''')
    s=once(s,'    if(!only.empty())cfg.erase', '''    cfg.push_back({32,64,128,2,1,32,1,"compact_packed_u32",11032});
    cfg.push_back({32,64,128,2,1,32,1,"compact_scalar_u32",12032});
    if(!only.empty())cfg.erase''')
    s=once(s,'else if(key=="--quant-check")quantcheck=v;', 'else if(key=="--quant-check")quantcheck=v;else if(key=="--scale-stress")scalestress=v;')
    s=once(s,'    cudaDeviceProp prop{};', '''    if(quantcheck&&scalestress)throw std::runtime_error("quant-check and scale-stress are separate tests");
    if(scalestress&&(scalestress!=1||rawwitness||
        (only!="wide_packed_split_u32"&&only!="wide_f32_split_u32"&&only!="compact_packed_u32"&&only!="compact_scalar_u32")))
        throw std::runtime_error("scale-stress requires an explicit supported split-wide candidate");
    cudaDeviceProp prop{};''')
    s=once(s,'scales[bg]=scale;', '''
        if(scalestress){
            unsigned short raw=(unsigned short)(0x1800u+rng(rs)%0x2000u);
            if(rng(rs)&1)raw|=0x8000u;
            memcpy(&scale,&raw,2);
        }
        scales[bg]=scale;''')
    s=once(s,'    std::vector<float>reference,sequential;', '''    if(scalestress){
        if(scalestress!=1||rawwitness)throw std::runtime_error("invalid scale stress");
        cfg.erase(std::remove_if(cfg.begin(),cfg.end(),[](const Config&c){
            return c.kind!=-4&&c.kind!=-5&&c.column!=9032&&c.column!=10032&&c.column!=11032&&c.column!=12032;
        }),cfg.end());
        if(std::none_of(cfg.begin(),cfg.end(),[&](const Config&c){return c.name==only;}))
            throw std::runtime_error("requested scale-stress candidate missing");
        puts("SCALE_STRESS correctness_only=1 half_rounded_weights=1");
    }
    std::vector<float>reference,sequential;''')
    s=once(s,'if(c.kind==-3)reference=got;', 'if(c.kind==-3||(scalestress&&c.kind==-4))reference=got;')
    s=once(s,'split[rail]=std::fma(x,q*scale,split[rail]);', 'split[rail]=std::fma(x,scalestress?__half2float(__float2half_rn(q*scale)):q*scale,split[rail]);')
    s=once(s,'    cudaEvent_t start,end;', '    if(scalestress){puts("PASS");return 0;}\n    cudaEvent_t start,end;')
    (build/'worker.cu').write_text(s)
    if sys.argv[1:]==['--prepare-only']:
        print('Prepared current control and Q4_0 specialization; no compiler or CUDA execution')
        return
    if sys.argv[1:]:raise SystemExit('Expected no arguments or --prepare-only')
    command=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14','-O3','-std=c++17','-arch=sm_60','-lineinfo','-Xptxas=-v','-I'+str(build),str(build/'worker.cu'),'-o',str(build/'worker')]
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (build/'compile.log').write_text(result.stdout);print(result.stdout[-2000:]);result.check_returncode()
    with (build/'worker.sass').open('w') as f:subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','-sass',str(build/'worker')],stdout=f,check=True)
    sass=(build/'worker.sass').read_text();assert 'HFMA2' not in sass
    import re
    sections=re.split(r'Function\s*:\s*(\S+)',sass)
    for name,body in zip(sections[1::2],sections[2::2]):
        if 'HMUL2' in body:assert 'decode_check_kernel' in name or (re.search(r'(?:wide|compact)_q4ILi\d+ELb1E',name) and 'FFMA' in body),name
    hashes=old['hashes']|{str(p):digest(p) for p in [SOURCE,Path(__file__),ROOT/'staging.cuh',ROOT/'wide.cuh',ROOT/'compact.cuh',ROOT/'supervise.py',build/'worker.cu',build/'current_control.inc',build/'aw_control.inc',build/'aw_q4_control.inc',build/'worker']}
    (build/'manifest.json').write_text(json.dumps(dict(hashes=hashes,command=command,control_source_sha256=EXPECTED,scope='Equal-M screening, not ragged/full-model acceptance'),indent=2)+'\n')

if __name__=='__main__':main()
