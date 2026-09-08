#!/usr/bin/env python3
"""CPU-only generation/compilation. Does not load CUDA or execute the worker."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'w4a16-t64-f32-20260908'

def replace_once(s,old,new):
    assert s.count(old)==1,old
    return s.replace(old,new)

def main():
    old=json.loads((BASE/'build/manifest.json').read_text())
    for name,h in old['hashes'].items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==h,name
    build=ROOT/'build';build.mkdir(exist_ok=True)
    for name in ['aw_control.inc','aw_q4_control.inc']:
        (build/name).write_bytes((BASE/'build'/name).read_bytes())
    s=(BASE/'worker.cu').read_text()
    s=replace_once(s,'#include "kernels.cuh"',f'#include "{BASE}/kernels.cuh"\n#include "{ROOT}/leads.cuh"\nstatic half *lead16; static float *lead32; static unsigned char *leadw; static const float *leadraw;')
    a=s.index('#define CANDIDATES(X)');b=s.index('\nvoid launch(',a)
    s=s[:a]+'#define CANDIDATES(X) X(64,64,128,1,1,8)\n'+s[b:]
    s=replace_once(s,'    if(c.kind==-3){', '''    if(c.kind==1){launch_delivery(c.column,a,lead16,lead32,w,leadw,out,t,n,k,e,grid,leadraw);return;}
    if(c.kind==-3){''')
    s=replace_once(s,'    std::string only;', '    int approved=0,vectorprep=0,prepgrid=0,rawwitness=0;\n    std::string only;')
    s=replace_once(s,'        if(key=="--tokens")t=v;', '        if(key=="--gpu-approved")approved=v;else if(key=="--raw-witness")rawwitness=v;else if(key=="--prep-grid")prepgrid=v;else if(key=="--vector-prep")vectorprep=v;else if(key=="--tokens")t=v;')
    s=replace_once(s,'    cudaDeviceProp prop{};', '    if(approved!=1)throw std::runtime_error("GPU execution disabled: requires released GPUs and coordinated --gpu-approved 1");\n    cudaDeviceProp prop{};')
    s=replace_once(s,'    da.put(a);dh.put(ha);dw4.put(w4);dw8.put(w8);', '''    if(rawwitness<0||rawwitness>1)throw std::runtime_error("invalid raw witness");
    std::vector<float> raw(a);
    if(rawwitness){
        const float witnesses[]={1.f+0x1p-11f,1.f+3*0x1p-11f,-1.f-0x1p-11f,-1.f-3*0x1p-11f,
            0x1p-25f,3*0x1p-25f,0x1p-14f+0x1p-25f,-0x1p-14f-0x1p-25f};
        size_t changed=0;
        for(size_t i=0;i<ac;++i){
            raw[i]=i%16<8?witnesses[i%8]:a[i]+std::ldexp(std::max(std::abs(a[i]),0x1p-14f),-12);
            ha[i]=__float2half_rn(raw[i]);a[i]=__half2float(ha[i]);
            changed+=bits(raw[i])!=bits(a[i]);
        }
        if(!changed)throw std::runtime_error("empty conversion witness");
        printf("RAW_WITNESS changed=%zu count=%zu accuracy_only=1\\n",changed,ac);
    }
    Device<float> draw(rawwitness?ac:1);
    if(rawwitness)draw.put(raw);
    da.put(a);dh.put(ha);dw4.put(w4);dw8.put(w8);
    std::vector<unsigned char> wtile(w4.size());
    for(size_t b=0;b<w4.size();b+=1152){
        memcpy(wtile.data()+b,w4.data()+b,128);
        for(int row=0;row<64;++row)for(int chunk=0;chunk<4;++chunk)
            memcpy(wtile.data()+b+128+(chunk*64+row)*4,w4.data()+b+128+row*16+chunk*4,4);
    }
    const size_t padded=size_t(experts)*((t+63)/64)*64*k;
    Device<half> dl16(padded);Device<float> dl32(padded);Device<unsigned char>dwt(wtile.size());
    dwt.put(wtile);lead16=dl16.p;lead32=dl32.p;leadw=dwt.p;leadraw=rawwitness?draw.p:da.p;
    printf("SCRATCH tiled_half_bytes=%zu tiled_float_bytes=%zu weight_layout_bytes=%zu\\n",padded*2,padded*4,wtile.size());''')
    s=replace_once(s,'    #undef ADD', '''    #undef ADD
    for(int layout=0;layout<4;++layout)for(int bw=0;bw<2;++bw)for(int u:{8,16,32})
        cfg.push_back({64,64,128,1,1,u,1,"delivery_a"+std::to_string(layout)+"b"+std::to_string(bw)+"u"+std::to_string(u),layout*100+bw*10+u});
    for(int layout=0;layout<4;++layout){
        cfg.push_back({64,64,128,1,1,4,1,"delivery_a"+std::to_string(layout)+"b1u4",layout*100+14});
        for(int u:{4,8})cfg.push_back({64,64,128,1,1,u,1,"delivery_a"+std::to_string(layout)+"b1u"+std::to_string(u)+"_4x8",1000+layout*100+10+u});
    }
    for(int u:{4,8,16,32})cfg.push_back({32,64,128,1,1,u,1,"delivery_m32u"+std::to_string(u),2010+u});
    for(int u:{8,16,32})cfg.push_back({16,64,128,1,1,u,1,"delivery_m16u"+std::to_string(u),3010+u});
    for(int u:{16,32})cfg.push_back({32,64,128,1,1,u,1,"delivery_fused_u"+std::to_string(u),4000+u});''')
    s=s.replace('return c.kind==0&&c.name!=only;', 'return c.kind>=0&&c.name!=only;')
    s=replace_once(s,'auto run=[&](const Config&c,bool prep){if(prep&&c.kind!=-3)prepare_a16<<<std::min<size_t>((ac+255)/256,4096),256>>>(da.p,dh.p,ac);', '''auto run=[&](const Config&c,bool prep){
        if(prep&&c.kind!=-3&&!(c.kind==1&&c.column>=4000)) {
            if(c.kind==1&&(c.column/100==1||c.column/100==3))prepare_tiled<false><<<std::min<size_t>(padded/1024,4096),256>>>(da.p,lead16,t,k,experts);
            else if(c.kind==1&&c.column/100==2)prepare_tiled<true><<<std::min<size_t>(padded/1024,4096),256>>>(da.p,lead32,t,k,experts);
            else if(vectorprep)prepare_a16_vector<<<std::min<size_t>((ac/8+255)/256,prepgrid?prepgrid:4096),256>>>(da.p,dh.p,ac);
            else prepare_a16<<<std::min<size_t>((ac+255)/256,4096),256>>>(da.p,dh.p,ac);
        }''')
    # Require every new output word to match the previously validated sequential
    # FP32 winner, not only the spot-checked host oracle.
    s=replace_once(s,'    std::vector<float>reference;', '    std::vector<float>reference,sequential;')
    s=replace_once(s,'        if(c.kind==-3)reference=got;', '''        if(c.kind==-3)reference=got;
        if(c.kind==0)sequential=got;
        if(c.kind==1){
            if(sequential.empty())throw std::runtime_error("missing frozen winner");
            for(size_t i=0;i<oc;++i)if(bits(got[i])!=bits(sequential[i]))throw std::runtime_error("delivery differs from frozen sequential winner");
        }''')
    # Retain the frozen winner even when selecting just one delivery lead.
    s=s.replace('return c.kind>=0&&c.name!=only;', 'return c.kind>=0&&c.name!=only&&c.name!="q4_m64n64b128s1d1u8r";')
    s=s.replace('c.column/100','(c.column%1000)/100')
    s=replace_once(s,'    cudaDeviceProp prop{};', '    if(vectorprep<0||vectorprep>1||prepgrid<0||prepgrid>4096)throw std::runtime_error("invalid vector preparation flag/grid");\n    printf("VECTOR_PREP %d GRID %d\\n",vectorprep,prepgrid);\n    cudaDeviceProp prop{};')
    (build/'worker.cu').write_text(s)
    cmd=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14','-O3','-std=c++17',
         '-arch=sm_60','-lineinfo','-Xptxas=-v','-I'+str(build),str(build/'worker.cu'),'-o',str(build/'worker')]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (build/'compile.log').write_text(p.stdout);print(p.stdout[-2000:]);p.check_returncode()
    with (build/'worker.sass').open('w') as f:
        subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','-sass',str(build/'worker')],stdout=f,check=True)
    sass=(build/'worker.sass').read_text()
    assert 'HFMA2' not in sass and 'HMUL2' not in sass
    paths=[Path(__file__),ROOT/'leads.cuh',ROOT/'test_offline.py',ROOT/'supervise.py',build/'worker.cu',build/'worker',build/'aw_control.inc',build/'aw_q4_control.inc',BASE/'worker.cu',BASE/'kernels.cuh',BASE/'supervise.py',ROOT.parent/'w4a16-t64-r1-20260908/supervise.py']
    manifest=dict(base_manifest=old,command=cmd,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},gpu_executed=False)
    (build/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    entries=[]
    for m in re.finditer(r"Compiling entry function '([^']+)'.*?Used (\d+) registers.*?(\d+) bytes smem",p.stdout,re.S):
        entries.append(dict(symbol=m[1],registers=int(m[2]),shared_bytes=int(m[3])))
    (build/'resources.json').write_text(json.dumps(entries,indent=2)+'\n')

if __name__=='__main__':main()
