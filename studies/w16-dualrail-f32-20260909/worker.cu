#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int AW_KSTAGE=32, AW_T64_ROWS=64, AW_T64_STAGE_BYTES=2176;
#include "q8.inc"
#include "packed.inc"

#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)

template<class T> struct Device {
    T *p;
    size_t n;
    explicit Device(size_t count):p(nullptr),n(count){ CU(cudaMalloc(&p,n*sizeof(T))); }
    ~Device(){ if(p) cudaFree(p); }
    void put(const std::vector<T>&v){ CU(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n); CU(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
};

uint32_t rng(uint32_t&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
uint32_t bits(float f){uint32_t x;memcpy(&x,&f,4);return x;}

float half_round(double x){
    double a=std::abs(x);if(a>=65520)return std::copysign(INFINITY,x);if(a==0)return float(x);
    int exp;std::frexp(a,&exp);int shift=a<std::ldexp(1.,-14)?24:11-exp;
    double r=std::ldexp(std::nearbyint(std::ldexp(a,shift)),-shift);return std::copysign(float(r),x);
}

int main(int argc,char**argv) try {
    if(argc!=8||std::string(argv[1])!="--gpu-approved"||std::string(argv[2])!="1")
        throw std::runtime_error("usage: --gpu-approved 1 M N K experts grid-multiplier");
    const int m=atoi(argv[3]),n=atoi(argv[4]),k=atoi(argv[5]),ex=atoi(argv[6]),gridmul=atoi(argv[7]);
    if(m<1||m>512||n%128||n<128||n>2048||k%32||k<32||k>2048||ex<1||ex>64||gridmul<1||gridmul>6)
        throw std::runtime_error("shape bounds");
    int devices;CU(cudaGetDeviceCount(&devices));if(devices!=1)throw std::runtime_error("one visible GPU required");
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor!=0)throw std::runtime_error("SM60 required");

    const size_t ac=size_t(ex)*m*k,oc=size_t(ex)*m*n,wc=size_t(ex)*n*k;
    const int kg=k/AW_KSTAGE;
    std::vector<float>a(ac);std::vector<half>w16(wc);
    std::vector<int8_t>w(wc);
    std::vector<half>sc(wc/AW_KSTAGE);
    std::vector<unsigned char>packed(wc/AW_KSTAGE*34);
    uint32_t seed=3911;
    for(size_t i=0;i<ac;++i){
        float x=(int(rng(seed)%4097)-2048)/1024.f;
        if((i/32)%97==0)x=0;if((i/32)%101==0&&i%32==0)x=16;
        a[i]=x;
    }
    for(int e=0;e<ex;++e)for(int r=0;r<n;++r)for(int g=0;g<kg;++g){
        const size_t si=(size_t(e)*n+r)*kg+g;
        const half d=__float2half_rn(((rng(seed)%127)+1)/1024.f);sc[si]=d;
        const size_t qb=((size_t(e)*(n/64)+r/64)*kg+g)*2176;
        memcpy(packed.data()+qb+(r%64)*2,&d,2);
        for(int j=0;j<32;++j){
            const int q=int(rng(seed)%256)-128;w[si*32+j]=q;
            packed[qb+128+(r%64)*32+j]=uint8_t(q);
            const size_t hdst=((size_t(e)*(n/64)+r/64)*kg+g)*2048+(r%64)*32+j;
            w16[hdst]=__float2half_rn(float(q)*__half2float(d));
        }
    }

    Device<float>da(ac);Device<half>dw16(wc);Device<unsigned char>dw8(packed.size());Device<float>out(oc);
    da.put(a);dw16.put(w16);dw8.put(packed);
    std::vector<aw_work_desc>d8,d16;std::vector<aw_tile_desc>tiles;
    for(int e=0;e<ex;++e){
        const void *input=(const void*)(da.p+size_t(e)*m*k);
        aw_work_desc d={input,(const char*)dw8.p+size_t(e)*n*kg*34,out.p+size_t(e)*m*n,nullptr,0,m,0,e};
        d8.push_back(d);d.weight=(const char*)(dw16.p+size_t(e)*n*k);d16.push_back(d);
        for(int r=0;r<m;r+=64)tiles.push_back({e,r,std::min(64,m-r)});
    }
    Device<aw_work_desc>dd8(d8.size()),dd16(d16.size());dd8.put(d8);dd16.put(d16);
    Device<aw_tile_desc>dt(tiles.size()),dtpairs((tiles.size()+1)/2),dtleft(tiles.size());dt.put(tiles);
    Device<int>dcounts(2);
    const int pair_grid=prop.multiProcessorCount*gridmul, fallback_grid=prop.multiProcessorCount*3;
    const int original_grid=prop.multiProcessorCount*2;

    auto run=[&](int mode){
        if(mode==0){
            aw_q8_service_m64_n128_halfpipe_sync<false><<<original_grid,256>>>(dd8.p,dt.p,tiles.size(),n,k);
        } else {
            w8_make_pairs<<<1,256>>>(dt.p,tiles.size(),dtpairs.p,dtleft.p,dcounts.p);
            if(mode==8){
                w8_fallback_half<false,0><<<fallback_grid,256>>>(dd8.p,dtleft.p,tiles.size(),n,k,dcounts.p+1);
                if(n==512&&k==2048)w8_pair_half<false,0,512,2048><<<pair_grid,256>>>(dd8.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
                else if(n==2048&&k==512)w8_pair_half<false,0,2048,512><<<pair_grid,256>>>(dd8.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
                else w8_pair_half<false,0><<<pair_grid,256>>>(dd8.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
            } else {
                w16_fallback_half<false,0><<<fallback_grid,256>>>(dd16.p,dtleft.p,tiles.size(),n,k,dcounts.p+1);
                if(n==512&&k==2048)w16_pair_half<false,0,512,2048><<<pair_grid,256>>>(dd16.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
                else if(n==2048&&k==512)w16_pair_half<false,0,2048,512><<<pair_grid,256>>>(dd16.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
                else w16_pair_half<false,0><<<pair_grid,256>>>(dd16.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
            }
        }
        CU(cudaGetLastError());
    };

    std::vector<float>w8_result;
    for(int mode:{0,8,16}){
        CU(cudaMemset(out.p,0xff,oc*sizeof(float)));run(mode);CU(cudaDeviceSynchronize());auto got=out.get();
        size_t nonfinite=0,bad=0,diff=0;for(float v:got)nonfinite+=!std::isfinite(v);
        if(mode==8)w8_result=got;
        if(mode==16)for(size_t i=0;i<oc;++i)diff+=bits(got[i])!=bits(w8_result[i]);
        if(mode>=8){
            const size_t samples=std::min<size_t>(oc,256);uint32_t sample_seed=731;
            for(size_t z=0;z<samples;++z){
                const size_t ix=rng(sample_seed)%oc;const int col=ix%n,row=(ix/n)%m,e=ix/(size_t(m)*n);
                float sum[2]={};
                for(int g=0;g<kg;++g)for(int j=0;j<32;++j){
                    const float av=__half2float(__float2half_rn(a[(size_t(e)*m+row)*k+g*32+j]));
                    const float bv=__half2float(w16[((size_t(e)*(n/64)+col/64)*kg+g)*2048+(col%64)*32+j]);
                    sum[j/16]=half_round(double(av)*double(bv)+double(sum[j/16]));
                }
                bad+=bits(got[ix])!=bits(sum[0]+sum[1]);
            }
        }
        printf("CHECK mode=%s outputs=%zu samples=%zu bad=%zu nonfinite=%zu diff_vs_w8=%zu\n",
                mode==0?"original-q8":mode==8?"w8-dualrail":"w16-dualrail",oc,mode>=8?std::min<size_t>(oc,256):0,bad,nonfinite,diff);
        if(nonfinite||bad||(mode==16&&diff))throw std::runtime_error("semantic mismatch");
    }
    printf("META M=%d N=%d K=%d experts=%d seed=3911 wire=f32 pair_grid=%d all_dispatch_launches_timed=1\n",m,n,k,ex,pair_grid);

    auto warm_end=std::chrono::steady_clock::now()+std::chrono::seconds(ex==64?2:0);
    do{for(int mode:{0,8,16})run(mode);CU(cudaDeviceSynchronize());}while(std::chrono::steady_clock::now()<warm_end);
    cudaEvent_t st,en;CU(cudaEventCreate(&st));CU(cudaEventCreate(&en));
    const int reps=ex==64?7:1,iters=ex==64?6:1,modes[3]={0,8,16};
    for(int r=0;r<reps;++r)for(int i=0;i<3;++i){
        const int mode=modes[(i+r)%3];CU(cudaEventRecord(st));for(int j=0;j<iters;++j)run(mode);
        CU(cudaEventRecord(en));CU(cudaEventSynchronize(en));float ms;CU(cudaEventElapsedTime(&ms,st,en));
        printf("TIME mode=%s rep=%d us=%.9g\n",mode==0?"original-q8":mode==8?"w8-dualrail":"w16-dualrail",r,ms*1000/iters);
    }
    puts("PASS");return 0;
} catch(const std::exception&e){fprintf(stderr,"STOP: %s\n",e.what());return 1;}
