#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
constexpr int AW_KSTAGE=32,AW_T64_ROWS=64,AW_T64_STAGE_BYTES=2176;
#include "current.inc"
#include "kernels.cuh"
#define CU(x) do{auto z=(x);if(z!=cudaSuccess)throw std::runtime_error(std::string(#x)+cudaGetErrorString(z));}while(0)
template<class T>struct D{T*p;size_t n;D(size_t n):n(n){CU(cudaMalloc(&p,n*sizeof(T)));}~D(){cudaFree(p);}void put(const std::vector<T>&v){if(n!=v.size())throw std::runtime_error("size");CU(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>v(n);CU(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
unsigned rng(unsigned&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
unsigned bits(float x){unsigned b;memcpy(&b,&x,4);return b;}
float hround(float x){return __half2float(__float2half_rn(x));}
int main(int argc,char**argv)try{
    int n=512,k=2048,reps=7,iters=5,approved=0,warmup=1000,gridmul=5,pattern=0,seed=3911;
    std::string counts="33,65",only="all";
    for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("argument");std::string key=argv[i],v=argv[i+1];
        if(key=="--counts")counts=v;else if(key=="--only")only=v;else{int z=std::stoi(v);
            if(key=="--n")n=z;else if(key=="--k")k=z;else if(key=="--reps")reps=z;else if(key=="--iters")iters=z;
            else if(key=="--gpu-approved")approved=z;else if(key=="--warmup-ms")warmup=z;else if(key=="--grid")gridmul=z;
            else if(key=="--pattern")pattern=z;else if(key=="--seed")seed=z;
            else throw std::runtime_error("unknown argument");}}
    std::vector<int>ms;std::stringstream ss(counts);std::string v;size_t rows=0;
    while(std::getline(ss,v,',')){int m=std::stoi(v);if(m<0||m>8192)throw std::runtime_error("M range");ms.push_back(m);rows+=m;}
    if(approved!=1||ms.empty()||ms.size()>64||!rows||rows>65536||n<128||n>2048||n%128||k<32||k>2048||k%32||reps<1||reps>15||iters<1||iters>20||warmup<0||warmup>5000||gridmul<1||gridmul>8||pattern<0||pattern>1)throw std::runtime_error("outside approved limits");
    if(std::fegetround()!=FE_TONEAREST)throw std::runtime_error("CPU oracle requires RN-even");
    if(only!="all"&&only!="popc2"&&only!="popc4")throw std::runtime_error("only all/popc2/popc4");
    int visible=0;CU(cudaGetDeviceCount(&visible));if(visible!=1)throw std::runtime_error("one visible GPU required");
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));
    const unsigned char uuid[16]={0x48,0x68,0x83,0x0a,0xc1,0xcf,0x90,0xbd,0x80,0x18,0x23,0x60,0xc5,0x52,0x93,0xb8};
    if(prop.major!=6||prop.minor!=0||memcmp(prop.uuid.bytes,uuid,16))throw std::runtime_error("physical GPU3 P100 only");
    int ne=ms.size(),ng=k/32;size_t wg=size_t(ne)*n*ng;
    std::vector<float>a(rows*k);std::vector<unsigned char>w(wg*18,0);std::vector<int8_t>wq(wg*32);std::vector<float>wd(wg);
    unsigned rs=unsigned(seed);
    for(auto&x:a)x=hround((int(rng(rs)%4097)-2048)/1024.f);
    if(pattern){
        const float ties[]={7,-7,.5f,1.5f,2.5f,-.5f,-1.5f,-2.5f};
        const float raw[]={1+0x1p-11f,1+3*0x1p-11f,1+0x1p-11f-0x1p-23f,1+0x1p-11f+0x1p-23f,-1-0x1p-11f,0x1p-25f,3*0x1p-25f,-0x1p-25f};
        const float outliers[]={65504,-65504,0x1p-24f,-0x1p-24f,0x1p-14f,-0x1p-14f,1,-1};
        const float tiny[]={0x1p-24f,3*0x1p-24f,0x1p-14f,-0x1p-24f,-3*0x1p-24f,-0x1p-14f,0.f,-0.f};
        for(size_t i=0;i<a.size();++i){int family=(i/32)%5;
            a[i]=family==0?(i%2?-0.f:0.f):family==1?ties[i%8]:family==2?raw[i%8]:family==3?outliers[i%8]:tiny[i%8];}
    }
    std::vector<float>a16(a.size());size_t changed=0;
    for(size_t i=0;i<a.size();++i){a16[i]=hround(a[i]);if(!std::isfinite(a16[i]))throw std::runtime_error("nonfinite after A16 conversion unsupported");changed+=bits(a16[i])!=bits(a[i]);}
    printf("FIXTURE pattern=%d seed=%d raw_to_half_changed=%zu\n",pattern,seed,changed);
    for(int e=0;e<ne;++e)for(int r=0;r<n;++r)for(int g=0;g<ng;++g){
        size_t bg=(size_t(e)*n+r)*ng+g,base=(size_t(e)*(n/64)*ng+size_t(r/64)*ng+g)*1152;
        half d=__float2half_rn(float(1+rng(rs)%127)/1024.f);
        if(pattern){const unsigned short scales[]={0x0001,0x03ff,0x3555,0xb555,0x3c01,0xbc01,0x0401,0x8001};unsigned short raw=scales[bg%8];memcpy(&d,&raw,2);}
        wd[bg]=__half2float(d);memcpy(w.data()+base+2*(r%64),&d,2);
        for(int b=0;b<32;++b){int q=int(rng(rs)%16)-8;wq[bg*32+b]=q;w[base+128+(r%64)*16+b%16]|=(q+8)<<(b<16?0:4);}
    }
    std::vector<PlaneGroup>ap(rows*ng);std::vector<int8_t>aq(rows*k);
    for(size_t bg=0;bg<rows*ng;++bg){float ma=0;for(int b=0;b<32;++b)ma=std::max(ma,std::abs(a16[bg*32+b]));
        float d=ma/7.f;ap[bg].d=d;
        for(int b=0;b<32;++b){int q=d==0?0:std::max(-7,std::min(7,int(std::nearbyint(a16[bg*32+b]/d))));aq[bg*32+b]=q;
            for(int p=0;p<4;++p)ap[bg].p[p]|=unsigned((q>>p)&1)<<b;}}
    D<float>da(a.size()),out(rows*n);D<unsigned char>dw(w.size());D<PlaneGroup>dap(ap.size()),dwp(wg);
    da.put(a);dw.put(w);
    std::vector<PlaneJob>jobs;std::vector<aw_work_desc>aw;std::vector<aw_tile_desc>at;std::vector<PlaneTile>p2,p4;std::vector<size_t>offset;
    size_t off=0;
    for(int e=0;e<ne;++e){offset.push_back(off);
        jobs.push_back({da.p+off*k,dw.p+size_t(e)*n*ng*18,dap.p+off*ng,dwp.p+size_t(e)*n*ng,out.p+off*n,ms[e]});
        aw.push_back({da.p+off*k,(const char*)((uintptr_t)(dw.p+size_t(e)*n*ng*18)|5u),out.p+off*n,nullptr,0,ms[e],0,e});
        for(int m=0;m<ms[e];m+=64)at.push_back({e,m,std::min(64,ms[e]-m)});
        for(int m=0;m<ms[e];m+=16)p2.push_back({e,m});for(int m=0;m<ms[e];m+=32)p4.push_back({e,m});off+=ms[e];
    }
    D<PlaneJob>dj(jobs.size());D<aw_work_desc>dd(aw.size());D<aw_tile_desc>dt(at.size());D<PlaneTile>d2(p2.size()),d4(p4.size());
    dj.put(jobs);dd.put(aw);dt.put(at);d2.put(p2);d4.put(p4);
    auto prep=[&](){quantize_planes<<<dim3(112,ne),128>>>(dj.p,ne,k);weight_planes<<<dim3(112,ne),128>>>(dj.p,ne,n,k);};
    auto run=[&](int c,bool preparation){if(c==0){aw_q8_service_m64_n128_halfpipe_sync<false><<<112,256>>>(dd.p,dt.p,at.size(),n,k);return;}
        if(preparation)prep();int grid=gridmul*prop.multiProcessorCount;
        if(c==1)plane_gemm<2><<<grid,128>>>(dj.p,d2.p,p2.size(),n,k);
        if(c==2)plane_gemm<4><<<grid,128>>>(dj.p,d4.p,p4.size(),n,k);
        if(c==3)plane_gemm<4,false><<<grid,128>>>(dj.p,d4.p,p4.size(),n,k);
    };
    prep();CU(cudaGetLastError());CU(cudaDeviceSynchronize());
    auto gotap=dap.get(),gotwp=dwp.get();size_t bad=0;
    for(size_t i=0;i<ap.size();++i){bad+=bits(ap[i].d)!=bits(gotap[i].d);for(int p=0;p<4;++p)bad+=ap[i].p[p]!=gotap[i].p[p];}
    for(size_t i=0;i<wg;++i){bad+=bits(wd[i])!=bits(gotwp[i].d);for(int p=0;p<4;++p){unsigned expected=0;for(int b=0;b<32;++b)expected|=unsigned((wq[i*32+b]>>p)&1)<<b;bad+=expected!=gotwp[i].p[p];}}
    printf("PREP_CHECK activation_groups=%zu weight_groups=%zu bad=%zu\n",ap.size(),wg,bad);if(bad)throw std::runtime_error("preparation mismatch");
    const char*names[]={"current_q4_T64_A16","popc2","popc4","matched_A4_F32"};std::vector<int>cfg={0,1,2,3};
    if(only=="popc2")cfg={0,1};if(only=="popc4")cfg={0,2};
    std::vector<float>reference,a4reference;double err=0,norm=0;
    for(int c:cfg){CU(cudaMemset(out.p,0xff,out.n*sizeof(float)));run(c,true);CU(cudaGetLastError());CU(cudaDeviceSynchronize());auto result=out.get();
        for(auto x:result)if(!std::isfinite(x))throw std::runtime_error("nonfinite output");
        if(c==0)reference=result;
        else if(a4reference.empty())a4reference=result;
        else for(size_t i=0;i<result.size();++i)if(bits(a4reference[i])!=bits(result[i]))throw std::runtime_error("full A4 cross-configuration mismatch");
        size_t checks=0;unsigned rr=8211;
        for(int e=0;e<ne;++e){size_t ec=size_t(ms[e])*n;if(!ec)continue;size_t nc=rows*n<=16384?ec:std::min<size_t>(256,ec);
            for(size_t z=0;z<nc;++z){size_t oi=nc==ec?z:rng(rr)%ec;int m=oi/n,r=oi%n;float expected=0,s0=0,s1=0;
                for(int g=0;g<ng;++g){size_t ai=((offset[e]+m)*ng+g)*32,wi=((size_t(e)*n+r)*ng+g)*32;
                    if(c==0){for(int b=0;b<16;++b)s0=std::fma(a16[ai+b],hround(float(wq[wi+b])*wd[wi/32]),s0);
                        for(int b=16;b<32;++b)s1=std::fma(a16[ai+b],hround(float(wq[wi+b])*wd[wi/32]),s1);
                    }else{int dot=0;for(int b=0;b<32;++b)dot+=int(aq[ai+b])*int(wq[wi+b]);float scale=ap[ai/32].d*wd[wi/32];expected=std::fma(float(dot),scale,expected);}
                }
                if(c==0)expected=s0+s1;
                size_t outi=offset[e]*n+oi;if(bits(expected)!=bits(result[outi])){printf("MISMATCH %s e=%d m=%d r=%d got=%a expected=%a\n",names[c],e,m,r,result[outi],expected);throw std::runtime_error("CPU oracle mismatch");}++checks;
            }
        }
        if(c){err=norm=0;for(size_t i=0;i<result.size();++i){double d=double(result[i])-reference[i];err+=d*d;norm+=double(reference[i])*reference[i];}}
        printf("CHECK %s cpu_samples=%zu finite_outputs=%zu rel_l2=%.8g quality_qualified=0\n",names[c],checks,result.size(),c?sqrt(err/std::max(norm,1e-100)):0.);
    }
    auto start=std::chrono::steady_clock::now();do{for(int c:cfg)run(c,true);CU(cudaGetLastError());CU(cudaDeviceSynchronize());}while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count()<warmup);
    cudaEvent_t begin,end;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));
    printf("WORKLOAD counts=%s N=%d K=%d baseline_grid=112 candidate_grid=%d weight_prep_every_pipeline=1 activation_prep_every_pipeline=1 plane_scratch_bytes=%zu\n",counts.c_str(),n,k,gridmul*prop.multiProcessorCount,(ap.size()+wg)*sizeof(PlaneGroup));
    for(int rep=0;rep<reps;++rep)for(size_t ci=0;ci<cfg.size();++ci){int c=cfg[(ci+rep)%cfg.size()];CU(cudaEventRecord(begin));for(int i=0;i<iters;++i)run(c,true);CU(cudaEventRecord(end));CU(cudaEventSynchronize(end));CU(cudaGetLastError());float ms;CU(cudaEventElapsedTime(&ms,begin,end));printf("TIME rep=%d config=%s pipeline_us=%.6f\n",rep,names[c],ms*1000/iters);}
    CU(cudaEventDestroy(begin));CU(cudaEventDestroy(end));puts("PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
