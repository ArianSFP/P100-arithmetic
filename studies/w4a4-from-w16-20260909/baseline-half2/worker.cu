#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cfenv>
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
#include "w4-pair.inc"

#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)

template<class T> struct Device {
    T *p;
    size_t n;
    explicit Device(size_t count):p(nullptr),n(count){ CU(cudaMalloc(&p,n*sizeof(T))); }
    ~Device(){ if(p) cudaFree(p); }
    void put(const std::vector<T>&v){ if(v.size()!=n)throw std::runtime_error("device upload size");CU(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n);CU(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v; }
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
    if(std::fegetround()!=FE_TONEAREST)throw std::runtime_error("CPU oracle requires RN-even");
    const int m=atoi(argv[3]),n=atoi(argv[4]),k=atoi(argv[5]),ex=atoi(argv[6]),gridmul=atoi(argv[7]);
    const int n_ngroups=n/128;
    if(m<1||m>512||n<128||n>2048||n%128||!n_ngroups||(n_ngroups&(n_ngroups-1))||
       k<32||k>2048||k%32||ex<1||ex>64||gridmul<1||gridmul>6)
        throw std::runtime_error("shape bounds");
    int devices;CU(cudaGetDeviceCount(&devices));if(devices!=1)throw std::runtime_error("one visible GPU required");
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor!=0)throw std::runtime_error("SM60 required");

    const int kg=k/AW_KSTAGE;
    const size_t ac=size_t(ex)*m*k, oc=size_t(ex)*m*n, wc=size_t(ex)*n*k;
    const size_t ag=size_t(ex)*m*kg, wg=size_t(ex)*n*kg;
    std::vector<half>a(ac),w16(wc),weight_scales(wg);
    std::vector<int8_t>wq(wc),aq(ac);
    std::vector<float>activation_scales(ag);
    std::vector<unsigned char>expected_a4(ag*16),packed_w4(size_t(ex)*(n/64)*kg*W4_STAGE_BYTES,0);

    uint32_t seed=3911;
    for(size_t i=0;i<ac;++i){
        float x=(int(rng(seed)%4097)-2048)/1024.f;
        if((i/32)%97==0)x=0;
        if((i/32)%101==0&&i%32==0)x=16;
        a[i]=__float2half_rn(x);
    }
    for(size_t group=0;group<ag;++group){
        float maximum=0;
        for(int j=0;j<32;++j)maximum=std::max(maximum,std::abs(__half2float(a[group*32+j])));
        const float scale=maximum/7.0f;activation_scales[group]=scale;
        for(int j=0;j<32;++j){
            const float value=__half2float(a[group*32+j]);
            const int q=scale==0.0f?0:std::max(-7,std::min(7,(int)std::nearbyint(value/scale)));
            aq[group*32+j]=(int8_t)q;
            expected_a4[group*16+j/2]|=(unsigned char)((q&15)<<((j&1)*4));
        }
    }

    for(int e=0;e<ex;++e)for(int r=0;r<n;++r)for(int g=0;g<kg;++g){
        const size_t si=(size_t(e)*n+r)*kg+g;
        const half d=__float2half_rn(((rng(seed)%127)+1)/1024.f);
        weight_scales[si]=d;
        const size_t stage=((size_t(e)*(n/64)+r/64)*kg+g)*W4_STAGE_BYTES;
        memcpy(packed_w4.data()+stage+(r%64)*sizeof(half),&d,sizeof(half));
        for(int j=0;j<32;++j){
            const int q=int(rng(seed)%16)-8;
            const size_t wi=si*32+j;wq[wi]=(int8_t)q;
            const int ph=j/16,word=(j%16)/8,within=j%8;
            const size_t byte=stage+128+ph*512+word*256+(r%64)*4+within/2;
            packed_w4[byte]|=(unsigned char)((q&15)<<((within&1)*4));
            const size_t hdst=((size_t(e)*(n/64)+r/64)*kg+g)*2048+(r%64)*32+j;
            w16[hdst]=__float2half_rn(float(q)*__half2float(d));
        }
    }

    Device<half>da(ac),dw16(wc);Device<unsigned char>dw4(packed_w4.size()),da4(ag*16);
    Device<float>das(ag),out(oc);da.put(a);dw16.put(w16);dw4.put(packed_w4);
    std::vector<aw_work_desc>d16;std::vector<w4_work_desc>d4;std::vector<aw_tile_desc>tiles;
    for(int e=0;e<ex;++e){
        const void *input=(const void*)((uintptr_t)(da.p+size_t(e)*m*k)|uintptr_t(1));
        d16.push_back({input,(const char*)(dw16.p+size_t(e)*n*k),out.p+size_t(e)*m*n,nullptr,0,m,0,e});
        d4.push_back({da4.p+size_t(e)*m*kg*16,das.p+size_t(e)*m*kg,
                (const char*)dw4.p+size_t(e)*(n/64)*kg*W4_STAGE_BYTES,
                out.p+size_t(e)*m*n,nullptr,m,e});
        for(int row=0;row<m;row+=64)tiles.push_back({e,row,std::min(64,m-row)});
    }
    Device<aw_work_desc>dd16(d16.size());dd16.put(d16);Device<w4_work_desc>dd4(d4.size());dd4.put(d4);
    Device<aw_tile_desc>dt(tiles.size()),dtpairs((tiles.size()+1)/2),dtleft(tiles.size());dt.put(tiles);
    Device<int>dcounts(2);
    const int pair_grid=prop.multiProcessorCount*gridmul;
    const int fallback_grid=prop.multiProcessorCount*3;
    const int quant_grid=prop.multiProcessorCount*2;

    auto plan=[&](){w8_make_pairs<<<1,256>>>(dt.p,tiles.size(),dtpairs.p,dtleft.p,dcounts.p);};
    auto launch_w4=[&](const aw_tile_desc*list,int capacity,const int*count){
        if(n==512&&k==2048)w4_pair_exact_g32<512,2048><<<pair_grid,256>>>(dd4.p,list,capacity,n,k,count);
        else if(n==2048&&k==512)w4_pair_exact_g32<2048,512><<<pair_grid,256>>>(dd4.p,list,capacity,n,k,count);
        else w4_pair_exact_g32<<<pair_grid,256>>>(dd4.p,list,capacity,n,k,count);
    };
    auto run=[&](int mode){
        if(mode==4){
            w4_quantize_a4<<<quant_grid,256>>>(da.p,da4.p,das.p,ag);
            plan();
            launch_w4(dtleft.p,tiles.size(),dcounts.p+1);
            launch_w4(dtpairs.p,(tiles.size()+1)/2,dcounts.p);
        }else{
            plan();
            w16_fallback_half<true,0><<<fallback_grid,256>>>(dd16.p,dtleft.p,tiles.size(),n,k,dcounts.p+1);
            if(n==512&&k==2048)w16_pair_half<true,0,512,2048><<<pair_grid,256>>>(dd16.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
            else if(n==2048&&k==512)w16_pair_half<true,0,2048,512><<<pair_grid,256>>>(dd16.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
            else w16_pair_half<true,0><<<pair_grid,256>>>(dd16.p,dtpairs.p,(tiles.size()+1)/2,n,k,dcounts.p);
        }
        CU(cudaGetLastError());
    };

    // Validate the device quantizer in full before testing either GEMM path.
    w4_quantize_a4<<<quant_grid,256>>>(da.p,da4.p,das.p,ag);CU(cudaGetLastError());CU(cudaDeviceSynchronize());
    const auto got_codes=da4.get();
    const auto got_scales=das.get();size_t quant_bad=0;
    for(size_t i=0;i<got_codes.size();++i)quant_bad+=got_codes[i]!=expected_a4[i];
    for(size_t i=0;i<got_scales.size();++i)quant_bad+=bits(got_scales[i])!=bits(activation_scales[i]);
    printf("QUANT_CHECK groups=%zu code_bytes=%zu bad=%zu\n",ag,got_codes.size(),quant_bad);
    if(quant_bad)throw std::runtime_error("device quantizer mismatch");

    // Validate planner coverage, including an odd/partial final M64 tile.
    plan();CU(cudaGetLastError());CU(cudaDeviceSynchronize());
    const auto counts=dcounts.get();
    const auto pairs=dtpairs.get();
    const auto left=dtleft.get();
    std::vector<int>covered(tiles.size());
    auto find_tile=[&](const aw_tile_desc&t){
        for(size_t i=0;i<tiles.size();++i)if(tiles[i].work==t.work&&tiles[i].row==t.row)return (int)i;
        return -1;
    };
    size_t planner_bad=0;
    for(int i=0;i<counts[0];++i){int p=find_tile(pairs[i]);planner_bad+=p<0;if(p>=0){++covered[p];if(p+1<(int)tiles.size()&&tiles[p+1].work==tiles[p].work&&tiles[p+1].row==tiles[p].row+64)++covered[p+1];else ++planner_bad;}}
    for(int i=0;i<counts[1];++i){int p=find_tile(left[i]);planner_bad+=p<0;if(p>=0)++covered[p];}
    for(int x:covered)planner_bad+=x!=1;
    printf("PLANNER_CHECK input_tiles=%zu pairs=%d leftovers=%d bad=%zu\n",tiles.size(),counts[0],counts[1],planner_bad);
    if(planner_bad)throw std::runtime_error("planner coverage mismatch");

    for(int mode:{16,4}){
        CU(cudaMemset(out.p,0xff,oc*sizeof(float)));run(mode);CU(cudaDeviceSynchronize());const auto got=out.get();
        size_t nonfinite=0,bad=0;for(float v:got)nonfinite+=!std::isfinite(v);
        const size_t samples=std::min<size_t>(oc,256);uint32_t sample_seed=731;
        for(size_t z=0;z<samples;++z){
            const size_t ix=rng(sample_seed)%oc;const int col=ix%n,row=(ix/n)%m,e=ix/(size_t(m)*n);
            float expected=0;
            if(mode==16){
                float sum[2]={};
                for(int g=0;g<kg;++g)for(int j=0;j<32;++j){
                    const float av=__half2float(a[(size_t(e)*m+row)*k+g*32+j]);
                    const float bv=__half2float(w16[((size_t(e)*(n/64)+col/64)*kg+g)*2048+(col%64)*32+j]);
                    sum[j/16]=half_round(double(av)*double(bv)+double(sum[j/16]));
                }
                expected=sum[0]+sum[1];
            }else{
                for(int g=0;g<kg;++g){
                    int dot=0;
                    const size_t ai=((size_t(e)*m+row)*kg+g)*32;
                    const size_t wi=((size_t(e)*n+col)*kg+g)*32;
                    for(int j=0;j<32;++j)dot+=int(aq[ai+j])*int(wq[wi+j]);
                    const float combined=activation_scales[ai/32]*__half2float(weight_scales[wi/32]);
                    expected=std::fma(float(dot),combined,expected);
                }
            }
            bad+=bits(got[ix])!=bits(expected);
        }
        printf("CHECK mode=%s outputs=%zu samples=%zu bad=%zu nonfinite=%zu\n",
                mode==16?"w16-dualrail":"w4a4-exact-g32",oc,samples,bad,nonfinite);
        if(nonfinite||bad)throw std::runtime_error("semantic mismatch");
    }
    printf("META M=%d N=%d K=%d experts=%d seed=3911 pair_grid=%d quant_grid=%d planner_timed=1 quantizer_timed_w4=1\n",m,n,k,ex,pair_grid,quant_grid);

    auto warm_end=std::chrono::steady_clock::now()+std::chrono::seconds(ex==64?2:0);
    do{for(int mode:{16,4}){run(mode);CU(cudaDeviceSynchronize());}}while(std::chrono::steady_clock::now()<warm_end);
    cudaEvent_t st,en;CU(cudaEventCreate(&st));CU(cudaEventCreate(&en));
    const int reps=ex==64?7:1,iters=ex==64?6:1,modes[2]={16,4};
    for(int r=0;r<reps;++r)for(int i=0;i<2;++i){
        const int mode=modes[(i+r)&1];CU(cudaEventRecord(st));for(int j=0;j<iters;++j)run(mode);
        CU(cudaEventRecord(en));CU(cudaEventSynchronize(en));float ms;CU(cudaEventElapsedTime(&ms,st,en));
        printf("TIME mode=%s rep=%d us=%.9g\n",mode==16?"w16-dualrail":"w4a4-exact-g32",r,ms*1000/iters);
    }
    puts("PASS");return 0;
} catch(const std::exception&e){fprintf(stderr,"STOP: %s\n",e.what());return 1;}
