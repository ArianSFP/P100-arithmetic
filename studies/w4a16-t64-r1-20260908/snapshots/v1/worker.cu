#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernels.cuh"
constexpr int AW_KSTAGE=32, AW_Q8_BLOCK_BYTES=34, AW_T64_ROWS=64, AW_T64_STAGE_BYTES=2176;
#include "aw_control.inc"

#define CU(x) do {cudaError_t e_=(x);if(e_!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e_));}while(0)
template<class T> struct Device {
    T *p; size_t count;
    explicit Device(size_t n):count(n){CU(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    void put(const std::vector<T>&v){if(v.size()!=count)throw std::runtime_error("upload size");CU(cudaMemcpy(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T>v(count);CU(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
uint32_t rng(uint32_t &s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
uint32_t bits(float x){uint32_t b;memcpy(&b,&x,4);return b;}
struct Config {int r,nw,nt;std::string name;};

void launch(const Config&c,const half*a,const unsigned char*w,float*out,int t,int n,int k,int e,int grid,
            aw_work_desc*d,aw_tile_desc*tiles,int ntiles) {
    if(c.r==-1){aw_q8_service_m64<false,true><<<grid,256>>>(d,tiles,ntiles,n,k);return;}
    #define RUN(R,NW,NT) if(c.r==R&&c.nw==NW&&c.nt==NT){q4_token_pairs<R,NW,NT><<<grid,NW*32>>>(a,w,out,t,n,k,e);return;}
    #define TILE(NW,NT) RUN(0,NW,NT) RUN(1,NW,NT) RUN(2,NW,NT) RUN(4,NW,NT) RUN(8,NW,NT)
    TILE(4,64) TILE(8,64) TILE(8,128)
    #undef TILE
    #undef RUN
    throw std::runtime_error("unknown config");
}

int main(int argc,char**argv) try {
    int t=64,n=512,k=2048,experts=8,reps=9,iters=8,seed=3911,pattern=0;
    std::string only;
    for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing argument");std::string key=argv[i];
        if(key=="--only"){only=argv[i+1];continue;}
        int v=std::stoi(argv[i+1]);
        if(key=="--tokens")t=v;else if(key=="--n")n=v;else if(key=="--k")k=v;
        else if(key=="--experts")experts=v;else if(key=="--reps")reps=v;else if(key=="--iters")iters=v;
        else if(key=="--seed")seed=v;else if(key=="--pattern")pattern=v;else throw std::runtime_error("unknown argument");
    }
    if(t<1||t>512||n<128||n>2048||n%128||k<32||k>2048||k%32||experts<1||experts>64||reps<1||reps>30||iters<1||iters>30||pattern<0||pattern>3)throw std::runtime_error("outside reviewed shape limits");
    cudaDeviceProp prop{};int count=0;CU(cudaGetDeviceCount(&count));if(count!=1)throw std::runtime_error("exactly one visible GPU required");
    CU(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor!=0)throw std::runtime_error("SM60 required");
    int groups=k/32,grid=2*prop.multiProcessorCount;
    size_t ac=size_t(experts)*t*k,oc=size_t(experts)*t*n,wc=size_t(experts)*n*k;
    std::vector<float>a(ac);std::vector<half>ha(ac);std::vector<int8_t>codes(wc);std::vector<half>scales(wc/32);
    std::vector<unsigned char>w4(wc/32*18,0),w8(wc/32*34,0);
    uint32_t rs=seed;
    for(size_t i=0;i<ac;++i){float x=(int(rng(rs)%4097)-2048)/1024.f;
        if(pattern==1)x=std::ldexp(x,int(rng(rs)%17)-8);
        if(pattern==2)x=(i%2?-1.f:1.f)*(1.f+(rng(rs)%8)/1024.f);
        if(pattern==3)x=2048.f;
        ha[i]=__float2half_rn(x);a[i]=__half2float(ha[i]);}
    for(int e=0;e<experts;++e)for(int row=0;row<n;++row)for(int g=0;g<groups;++g){
        size_t bg=(size_t(e)*n+row)*groups+g;
        half scale=__float2half_rn(pattern==3?1/1024.f:((rng(rs)%127)+1)/1024.f);scales[bg]=scale;
        size_t b4=(size_t(e)*(n/64)*groups+size_t(row/64)*groups+g)*1152;
        size_t b8=(size_t(e)*(n/64)*groups+size_t(row/64)*groups+g)*2176;
        memcpy(w4.data()+b4+2*(row%64),&scale,2);memcpy(w8.data()+b8+2*(row%64),&scale,2);
        for(int j=0;j<32;++j){int q=pattern==3?-8:int(rng(rs)%16)-8;codes[bg*32+j]=int8_t(q);
            w4[b4+128+(j%16)*64+row%64]|=uint8_t(q+8)<<(j<16?0:4);
            w8[b8+128+(row%64)*32+j]=uint8_t(int8_t(q));}
    }
    Device<float> da(ac),out(oc);Device<half> dh(ac);Device<unsigned char>dw4(w4.size()),dw8(w8.size());
    da.put(a);dh.put(ha);dw4.put(w4);dw8.put(w8);
    std::vector<aw_work_desc> descs;std::vector<aw_tile_desc> tiles;
    for(int e=0;e<experts;++e){descs.push_back({da.p+size_t(e)*t*k,reinterpret_cast<const char*>(dw8.p)+size_t(e)*n*groups*34,out.p+size_t(e)*t*n,nullptr,0,t,0,e});
        for(int row=0;row<t;row+=64)tiles.push_back({e,row,std::min(64,t-row)});}
    Device<aw_work_desc> dd(descs.size());Device<aw_tile_desc>dt(tiles.size());dd.put(descs);dt.put(tiles);
    std::vector<Config>cfg={{-1,8,64,"aw_t64_m64_sk2"}};
    for(auto shape:std::vector<std::pair<int,int>>{{4,64},{8,64},{8,128}})for(int r:{0,1,2,4,8})
        cfg.push_back({r,shape.first,shape.second,"q4_r"+std::to_string(r)+"_w"+std::to_string(shape.first)+"_n"+std::to_string(shape.second)});
    if(!only.empty())cfg.erase(std::remove_if(cfg.begin(),cfg.end(),[&](const Config&c){return c.name!="aw_t64_m64_sk2"&&c.name!=only;}),cfg.end());
    if(cfg.size()<2)throw std::runtime_error("no candidate");
    auto run=[&](const Config&c,bool prep){if(prep&&c.r>=0)prepare_a16<<<std::min<size_t>((ac+255)/256,4096),256>>>(da.p,dh.p,ac);
        launch(c,dh.p,dw4.p,out.p,t,n,k,experts,grid,dd.p,dt.p,int(tiles.size()));CU(cudaGetLastError());};
    printf("META tokens=%d n=%d k=%d experts=%d seed=%d pattern=%d sm=%d\n",t,n,k,experts,seed,pattern,prop.multiProcessorCount);fflush(stdout);
    std::vector<float>reference;
    for(const auto&c:cfg){CU(cudaMemset(out.p,0xff,oc*4));run(c,true);CU(cudaDeviceSynchronize());auto got=out.get();
        if(c.r==-1)reference=got;
        double err2=0,ref2=0,maxabs=0;size_t nonfinite=0,bad=0;
        for(size_t i=0;i<oc;++i){if(!std::isfinite(got[i])){nonfinite++;continue;}double d=double(got[i])-reference[i];err2+=d*d;ref2+=double(reference[i])*reference[i];maxabs=std::max(maxabs,std::abs(d));}
        uint32_t sample=uint32_t(seed)^9921;
        for(int ix=0;ix<std::min<size_t>(oc,256);++ix){size_t index=oc<=256?ix:rng(sample)%oc;
            int col=index%n, row=(index/n)%t,e=index/(size_t(n)*t);float expected=0,split[2]={};
            for(int g=0;g<groups;++g){float scale=__half2float(scales[(size_t(e)*n+col)*groups+g]);float sum=0;
                if(c.r<0){for(int j=0;j<32;++j){float q=codes[((size_t(e)*n+col)*groups+g)*32+j];float x=a[(size_t(e)*t+row)*k+g*32+j];split[j/16]=std::fma(x,q*scale,split[j/16]);}}
                else if(c.r==0){for(int j=0;j<32;++j)sum=std::fma(float(codes[((size_t(e)*n+col)*groups+g)*32+j]),a[(size_t(e)*t+row)*k+g*32+j],sum);}
                else {half h[8];for(int r=0;r<c.r;++r)h[r]=__float2half_rn(0);
                    for(int j=0;j<32;++j){double q=codes[((size_t(e)*n+col)*groups+g)*32+j];double x=a[(size_t(e)*t+row)*k+g*32+j];h[j%c.r]=__double2half(q*x+double(__half2float(h[j%c.r])));}
                    sum=__half2float(h[0]);for(int r=1;r<c.r;++r)sum+=__half2float(h[r]);}
                if(c.r>=0)expected=std::fma(sum,scale,expected);
            }
            if(c.r<0)expected=split[0]+split[1];
            if(bits(got[index])!=bits(expected)&&!(std::isnan(got[index])&&std::isnan(expected))){if(bad<3)printf("MISMATCH %s %zu %.9g %.9g\n",c.name.c_str(),index,got[index],expected);bad++;}
        }
        printf("CHECK %s outputs=%zu samples=%zu bad=%zu nonfinite=%zu rel_l2=%.10g maxabs=%.10g\n",c.name.c_str(),oc,std::min<size_t>(oc,256),bad,nonfinite,std::sqrt(err2/std::max(ref2,1e-300)),maxabs);fflush(stdout);
        if(bad||(nonfinite&&pattern!=3))throw std::runtime_error("arithmetic gate failed");
    }
    if(pattern==3){puts("EXPECTED_OVERFLOW_WITNESS_COMPLETE");return 0;}
    cudaEvent_t start,end;CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));
    for(const auto&c:cfg)for(int i=0;i<3;++i)run(c,true);CU(cudaDeviceSynchronize());
    for(int rep=0;rep<reps;++rep)for(size_t j=0;j<cfg.size();++j){const auto&c=cfg[(j+rep)%cfg.size()];
        for(int prep=0;prep<=1;++prep){CU(cudaEventRecord(start));for(int it=0;it<iters;++it)run(c,prep);CU(cudaEventRecord(end));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));
            printf("TIME %s prep=%d rep=%d us=%.9g\n",c.name.c_str(),prep,rep,ms*1000/iters);fflush(stdout);}}
    CU(cudaEventDestroy(start));CU(cudaEventDestroy(end));puts("PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"STOP: %s\n",e.what());return 2;}
