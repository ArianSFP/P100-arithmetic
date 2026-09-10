#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)

__device__ __forceinline__ uint32_t half2_bits(half2 x) {
    __half2_raw r=x;
    return uint32_t(r.x)|(uint32_t(r.y)<<16);
}

__device__ __forceinline__ uint32_t mad_wide_s16(int16_t a,int16_t b,uint32_t c) {
    uint32_t result;
    asm volatile("mad.wide.s16 %0, %1, %2, %3;"
                 : "=r"(result) : "h"(a),"h"(b),"r"(c));
    return result;
}

extern "C" __global__ __launch_bounds__(256,2)
void bench_hfma2(const uint32_t *in,uint64_t *out,int iters) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const float t=float((in[id]&31u)+1u)*(1.0f/32768.0f);
    const half2 a=__floats2half2_rn(t,-t);
    const half2 b=__floats2half2_rn(0.5f,0.25f);
    half2 c0=__float2half2_rn(0.0f),c1=c0,c2=c0,c3=c0,c4=c0,c5=c0,c6=c0,c7=c0;
    #pragma unroll 1
    for(int i=0;i<iters;++i) {
        c0=__hfma2(a,b,c0);c1=__hfma2(a,b,c1);
        c2=__hfma2(a,b,c2);c3=__hfma2(a,b,c3);
        c4=__hfma2(a,b,c4);c5=__hfma2(a,b,c5);
        c6=__hfma2(a,b,c6);c7=__hfma2(a,b,c7);
    }
    out[id]=uint64_t(half2_bits(c0)^half2_bits(c1)^half2_bits(c2)^half2_bits(c3))|
            (uint64_t(half2_bits(c4)^half2_bits(c5)^half2_bits(c6)^half2_bits(c7))<<32);
}

extern "C" __global__ __launch_bounds__(256,2)
void bench_xmad_s16(const uint32_t *in,uint64_t *out,int iters) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const int16_t a=int16_t(int(in[id]&15u)-8);
    const int16_t b=int16_t(2048+4096*(int((in[id]>>4)&15u)-8)+(int((in[id]>>8)&15u)-8));
    uint32_t c0=1,c1=3,c2=5,c3=7,c4=11,c5=13,c6=17,c7=19;
    #pragma unroll 1
    for(int i=0;i<iters;++i) {
        c0=mad_wide_s16(a,b,c0);c1=mad_wide_s16(a,b,c1);
        c2=mad_wide_s16(a,b,c2);c3=mad_wide_s16(a,b,c3);
        c4=mad_wide_s16(a,b,c4);c5=mad_wide_s16(a,b,c5);
        c6=mad_wide_s16(a,b,c6);c7=mad_wide_s16(a,b,c7);
    }
    out[id]=uint64_t(c0^c1^c2^c3)|(uint64_t(c4^c5^c6^c7)<<32);
}

extern "C" __global__ __launch_bounds__(256,2)
void bench_fp32_pack2(const uint32_t *in,uint64_t *out,int iters) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const float a=float(int(in[id]&15u)-8)*(1.0f/65536.0f);
    const float b=float(int((in[id]>>4)&15u)-8)+4096.0f*float(int((in[id]>>8)&15u)-8)+2048.0f;
    float c0=1,c1=3,c2=5,c3=7,c4=11,c5=13,c6=17,c7=19;
    #pragma unroll 1
    for(int i=0;i<iters;++i) {
        c0=__fmaf_rn(a,b,c0);c1=__fmaf_rn(a,b,c1);
        c2=__fmaf_rn(a,b,c2);c3=__fmaf_rn(a,b,c3);
        c4=__fmaf_rn(a,b,c4);c5=__fmaf_rn(a,b,c5);
        c6=__fmaf_rn(a,b,c6);c7=__fmaf_rn(a,b,c7);
    }
    uint64_t lo=uint32_t(__float_as_uint(c0)^__float_as_uint(c1)^__float_as_uint(c2)^__float_as_uint(c3));
    uint64_t hi=uint32_t(__float_as_uint(c4)^__float_as_uint(c5)^__float_as_uint(c6)^__float_as_uint(c7));
    out[id]=lo|(hi<<32);
}

extern "C" __global__ __launch_bounds__(256,2)
void bench_dfma_pack4(const uint32_t *in,uint64_t *out,int iters) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const double a=double(int(in[id]&15u)-8)*(1.0/1048576.0);
    const double b=double(int((in[id]>>4)&15u)-8)+4096.0*double(int((in[id]>>8)&15u)-8)+16777216.0*double(int((in[id]>>12)&15u)-8)+68719476736.0*double(int((in[id]>>16)&15u)-8);
    double c0=1,c1=3,c2=5,c3=7;
    #pragma unroll 1
    for(int i=0;i<iters;++i) {
        c0=__fma_rn(a,b,c0);c1=__fma_rn(a,b,c1);
        c2=__fma_rn(a,b,c2);c3=__fma_rn(a,b,c3);
    }
    out[id]=uint64_t(__double_as_longlong(c0)^__double_as_longlong(c1)^
                     __double_as_longlong(c2)^__double_as_longlong(c3));
}

extern "C" __global__ __launch_bounds__(256,2)
void bench_mixed_hfma2_dfma(const uint32_t *in,uint64_t *out,int iters) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const float tf=float((in[id]&31u)+1u)*(1.0f/32768.0f);
    const half2 ha=__floats2half2_rn(tf,-tf),hb=__floats2half2_rn(0.5f,0.25f);
    half2 h0=__float2half2_rn(0),h1=h0,h2=h0,h3=h0,h4=h0,h5=h0,h6=h0,h7=h0;
    const double da=double(int(in[id]&15u)-8)*(1.0/1048576.0);
    const double db=double(int((in[id]>>4)&15u)-8)+4096.0*double(int((in[id]>>8)&15u)-8)+16777216.0*double(int((in[id]>>12)&15u)-8)+68719476736.0*double(int((in[id]>>16)&15u)-8);
    double d0=1,d1=3,d2=5,d3=7;
    #pragma unroll 1
    for(int i=0;i<iters;++i) {
        h0=__hfma2(ha,hb,h0);d0=__fma_rn(da,db,d0);
        h1=__hfma2(ha,hb,h1);h2=__hfma2(ha,hb,h2);d1=__fma_rn(da,db,d1);
        h3=__hfma2(ha,hb,h3);h4=__hfma2(ha,hb,h4);d2=__fma_rn(da,db,d2);
        h5=__hfma2(ha,hb,h5);h6=__hfma2(ha,hb,h6);d3=__fma_rn(da,db,d3);
        h7=__hfma2(ha,hb,h7);
    }
    uint64_t hv=half2_bits(h0)^half2_bits(h1)^half2_bits(h2)^half2_bits(h3)^
                half2_bits(h4)^half2_bits(h5)^half2_bits(h6)^half2_bits(h7);
    out[id]=hv^uint64_t(__double_as_longlong(d0)^__double_as_longlong(d1)^
                        __double_as_longlong(d2)^__double_as_longlong(d3));
}

int main(int argc,char **argv) try {
    if(argc!=4||std::string(argv[1])!="--gpu-approved"||std::string(argv[2])!="1")
        throw std::runtime_error("usage: --gpu-approved 1 iterations");
    const int iters=atoi(argv[3]);if(iters<256||iters>1048576)throw std::runtime_error("iteration bounds");
    int devices=0;CU(cudaGetDeviceCount(&devices));if(devices!=1)throw std::runtime_error("one visible GPU required");
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor!=0)throw std::runtime_error("SM60 required");
    const int block=256,grid=prop.multiProcessorCount*2,count=grid*block;
    std::vector<uint32_t> input(count);uint32_t s=3911;
    for(auto &v:input){s^=s<<13;s^=s>>17;s^=s<<5;v=s;}
    uint32_t *din=nullptr;uint64_t *dout=nullptr;CU(cudaMalloc(&din,count*sizeof(*din)));CU(cudaMalloc(&dout,count*sizeof(*dout)));
    CU(cudaMemcpy(din,input.data(),count*sizeof(*din),cudaMemcpyHostToDevice));
    struct Mode{const char *name;int useful;} modes[]={
        {"hfma2",16},{"xmad-s16-pack2",16},{"fp32-pack2",16},
        {"dfma-pack4",16},{"mixed-hfma2-dfma",32}};
    auto launch=[&](int mode){
        switch(mode){
            case 0:bench_hfma2<<<grid,block>>>(din,dout,iters);break;
            case 1:bench_xmad_s16<<<grid,block>>>(din,dout,iters);break;
            case 2:bench_fp32_pack2<<<grid,block>>>(din,dout,iters);break;
            case 3:bench_dfma_pack4<<<grid,block>>>(din,dout,iters);break;
            default:bench_mixed_hfma2_dfma<<<grid,block>>>(din,dout,iters);break;
        }
        CU(cudaGetLastError());
    };
    auto warm_end=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    do{for(int mode=0;mode<5;++mode)launch(mode);CU(cudaDeviceSynchronize());}while(std::chrono::steady_clock::now()<warm_end);
    cudaEvent_t begin,end;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));
    constexpr int rounds=9,repeats=3;
    for(int r=0;r<rounds;++r)for(int j=0;j<5;++j){
        const int mode=(j+2*r)%5;CU(cudaEventRecord(begin));for(int q=0;q<repeats;++q)launch(mode);CU(cudaEventRecord(end));CU(cudaEventSynchronize(end));
        float ms=0;CU(cudaEventElapsedTime(&ms,begin,end));const double us=ms*1000.0/repeats;
        const double macs=double(count)*iters*modes[mode].useful;
        printf("TIME mode=%s rep=%d us=%.9g useful_tmac_s=%.9g\n",modes[mode].name,r,us,macs/(us*1e6));
    }
    std::vector<uint64_t> output(count);CU(cudaMemcpy(output.data(),dout,count*sizeof(*dout),cudaMemcpyDeviceToHost));
    uint64_t checksum=0;for(uint64_t v:output)checksum^=v;
    printf("META grid=%d block=%d threads=%d iterations=%d seed=3911 checksum=%llu\n",grid,block,count,iters,(unsigned long long)checksum);
    CU(cudaFree(dout));CU(cudaFree(din));puts("PASS");return 0;
} catch(const std::exception &e){fprintf(stderr,"STOP: %s\n",e.what());return 1;}
