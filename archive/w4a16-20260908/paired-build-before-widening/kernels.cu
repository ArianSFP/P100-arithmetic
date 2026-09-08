#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.hpp"

__device__ __forceinline__ float widen(uint16_t v) {return __half2float(__ushort_as_half(v));}
#define ARGS const Q4Block *native,const uint32_t *words,const uint16_t *scales,const uint16_t *a,float *out,int rows,int groups
#define PASS native,words,scales,a,out,rows,groups

template<bool Repacked> __device__ __forceinline__ void baseline(ARGS) {
    int lane=threadIdx.x&31,row=(blockIdx.x*blockDim.x+threadIdx.x)/32,batch=blockIdx.y;
    if(row>=rows)return;
    float acc=0;
    for(int g=lane;g<groups;g+=32) {
        uint32_t w[4];
        if constexpr(Repacked) {
            #pragma unroll
            for(int p=0;p<4;++p)w[p]=words[word_address(row,g,p,groups,true)];
        }
        float dot=0;
        #pragma unroll
        for(int j=0;j<32;++j) {
            int q=Repacked ? unpack_nibble(w[j/8],j) : native_code(native[row*groups+g],j);
            dot=__fmaf_rn(float(q),widen(a[(batch*groups+g)*32+j]),dot);
        }
        uint16_t d=Repacked ? scales[row*groups+g] : native[row*groups+g].d;
        acc=__fmaf_rn(dot,widen(d),acc);
    }
    #pragma unroll
    for(int offset=16;offset;offset/=2)acc=__fadd_rn(acc,__shfl_down_sync(0xffffffff,acc,offset));
    if(lane==0)out[batch*rows+row]=acc;
}
extern "C" __global__ void native_direct(ARGS) {baseline<false>(PASS);}
extern "C" __global__ void old_repacked(ARGS) {baseline<true>(PASS);}

template<int R,int Method> __device__ __forceinline__ void row_body(ARGS) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int base=(blockIdx.x*4+warp)*32*R+lane,batch=blockIdx.y;
    float acc[R]={};
    for(int g=blockIdx.z;g<groups;g+=gridDim.z) {
        uint32_t w[R][4];float dot[R]={},sums[R][4]={};
        #pragma unroll
        for(int r=0;r<R;++r) {
            #pragma unroll
            for(int p=0;p<4;++p)w[r][p]=base+32*r<rows ? words[word_address(base+32*r,g,p,groups)] : 0;
        }
        if constexpr(Method<2) {
            #pragma unroll
            for(int j=0;j<32;++j) {
                float av=widen(a[(batch*groups+g)*32+j]);
                #pragma unroll
                for(int r=0;r<R;++r) {
                    int q;
                    if constexpr(Method==0)q=unpack_nibble(w[r][j/8],j);
                    else {int code=0;
                        #pragma unroll
                        for(int p=0;p<4;++p)code|=((w[r][p]>>j)&1)<<p;
                        q=(code^8)-8;
                    }
                    dot[r]=__fmaf_rn(float(q),av,dot[r]);
                }
            }
        } else {
            #pragma unroll
            for(int q=0;q<8;++q) {
                float entry=0;
                #pragma unroll
                for(int j=0;j<4;++j)if(lane&(1<<j))entry=__fadd_rn(entry,widen(a[(batch*groups+g)*32+4*q+j]));
                #pragma unroll
                for(int r=0;r<R;++r) {
                    #pragma unroll
                    for(int p=0;p<4;++p) {
                        unsigned idx=(w[r][p]>>(4*q))&15;
                        sums[r][p]=__fadd_rn(sums[r][p],__shfl_sync(0xffffffff,entry,idx));
                    }
                }
            }
            #pragma unroll
            for(int r=0;r<R;++r) {
                #pragma unroll
                for(int p=0;p<4;++p)dot[r]=__fmaf_rn(float(p==3 ? -8 : 1<<p),sums[r][p],dot[r]);
            }
        }
        #pragma unroll
        for(int r=0;r<R;++r) {
            float d=base+32*r<rows ? widen(scales[scale_address(base+32*r,g,groups)]) : 0;
            acc[r]=__fmaf_rn(dot[r],d,acc[r]);
        }
    }
    #pragma unroll
    for(int r=0;r<R;++r)if(base+32*r<rows)out[(batch*gridDim.z+blockIdx.z)*rows+base+32*r]=acc[r];
}
#define ROW(NAME,R,M) extern "C" __global__ void NAME(ARGS) {row_body<R,M>(PASS);}
ROW(direct_r1,1,0) ROW(direct_r2,2,0) ROW(direct_r4,4,0)
ROW(plane_direct_r1,1,1) ROW(plane_direct_r2,2,1) ROW(plane_direct_r4,4,1)
ROW(lut_r1,1,2) ROW(lut_r2,2,2) ROW(lut_r4,4,2)

template<int Offset,int S> __device__ __forceinline__ void reduce_tree(float (&v)[S]) {
    #pragma unroll
    for(int i=0;i<Offset;++i)v[i]=__fadd_rn(v[i],v[i+Offset]);
    if constexpr(Offset>1)reduce_tree<Offset/2>(v);
}
template<int S> __device__ __forceinline__ void reduce_body(const float *partial,float *out,int rows) {
    int row=blockIdx.x*blockDim.x+threadIdx.x,batch=blockIdx.y;
    if(row>=rows)return;
    float v[S];
    #pragma unroll
    for(int i=0;i<S;++i)v[i]=partial[(batch*S+i)*rows+row];
    reduce_tree<S/2>(v);
    out[batch*rows+row]=v[0];
}
#define REDUCE(S) extern "C" __global__ void reduce_##S(const float *p,float *out,int rows) {reduce_body<S>(p,out,rows);}
REDUCE(2) REDUCE(4) REDUCE(8) REDUCE(16) REDUCE(32) REDUCE(64)
extern "C" __global__ void convert_all(float *out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<65536)out[i]=widen(uint16_t(i));
}
