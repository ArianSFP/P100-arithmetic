// Compile-only probes, not performance benchmarks. Not compiled in this session.
// nvcc -O3 -std=c++17 -arch=sm_60 -cubin compile_probes.cu -o probes.cubin
// cuobjdump --dump-sass probes.cubin > probes.sass
#include "packed_int4.cuh"
extern "C" __global__ void probe_shared32(const int16_t* p,const int8_t* a,
                                          int2* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    auto r=packed_int4::shared_dot32(p+32*i,a+32*i);
    out[i]=make_int2(r.first,r.second);
}
extern "C" __global__ void probe_independent8(const int16_t* p,const int16_t* q,
                                             int2* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    auto r=packed_int4::independent_dots8(p+8*i,q+8*i);
    out[i]=make_int2(r.first,r.second);
}
extern "C" __global__ void probe_half2(const __half2* w,const __half* a,
                                      float2* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    out[i]=packed_int4::half2_integer_dot32(w+32*i,a+32*i);
}
