#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ half2 bits_half2(unsigned bits) {
    __half2_raw v;v.x=uint16_t(bits);v.y=uint16_t(bits>>16);return half2(v);
}
__device__ __forceinline__ unsigned half2_bits(half2 h) {
    __half2_raw v=h;return unsigned(v.x)|(unsigned(v.y)<<16);
}

// Two two's-complement nibbles -> two exact half integers. Input 0..255.
// 0x6400 is binary16 1024; XOR maps signed nibbles to offset binary.
__device__ __forceinline__ half2 decode_pair(unsigned q) {
    unsigned bits=0x64006400u | ((q&15u)^8u) | ((((q>>4)&15u)^8u)<<16);
    return __hsub2(bits_half2(bits),__float2half2_rn(1032.f));
}
__device__ __forceinline__ float widen(uint16_t h) {return __half2float(__ushort_as_half(h));}

// One warp per group, 4 warps per block. Round-to-nearest-even symmetric A4,
// [-7,7], FP32 scale maxabs/7, all-zero scale=0. Source is finite binary16.
extern "C" __global__ void quantize(const uint16_t *a,unsigned *packed,float *scale,
                                    unsigned groups,unsigned batches) {
    unsigned lane=threadIdx.x&31, bg=blockIdx.x*4+(threadIdx.x>>5);
    if(bg>=groups*batches)return;
    float v=widen(a[bg*32+lane]),mx=fabsf(v);
    for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));
    float d=mx/7.f;
    int q=mx ? max(-7,min(7,__float2int_rn(v/d))) : 0;
    if(lane==0)scale[bg]=d;
    unsigned word=(unsigned(q)&15u)<<(4*(lane&7));
    for(int o=4;o;o>>=1)word|=__shfl_xor_sync(0xffffffff,word,o);
    if((lane&7)==0)packed[bg*4+lane/8]=word;
}

#define ARGS const unsigned *__restrict__ w,const uint16_t *__restrict__ ds,const unsigned *__restrict__ a,const float *__restrict__ da,float *__restrict__ out,unsigned rows,unsigned groups,unsigned batches
#define PASS w,ds,a,da,out,rows,groups,batches
// H2 accumulates even/odd K positions separately. G32 -> each half lane has
// <=16*64=1024 magnitude; their FP32 sum is the exact G32 integer dot.
template<bool H2,int R,int B>
__device__ __forceinline__ void body(ARGS) {
    unsigned lane=threadIdx.x&31,tile=(blockIdx.x*4+(threadIdx.x>>5))*R,b0=blockIdx.y*B;
    float acc[R][B]={};
    for(unsigned g=blockIdx.z;g<groups;g+=32) {
        unsigned full[R][4];
        #pragma unroll
        for(int r=0;r<R;r++) {
            #pragma unroll
            for(int p=0;p<4;p++) full[r][p]=(tile+r)*32+lane<rows ? w[((tile+r)*groups+g)*128+p*32+lane] : 0;
        }
        float ad[B];unsigned shared_a[B];
        #pragma unroll
        for(int b=0;b<B;b++) {
            unsigned bg=(b0+b)*groups+g;
            ad[b]=b0+b<batches ? da[bg] : 0;
            if constexpr(H2) {
                unsigned q=b0+b<batches ? a[bg*4+(lane&15)/4] : 0;
                shared_a[b]=half2_bits(decode_pair((q>>(8*(lane&3)))&255));
            } else {
                unsigned q=b0+b<batches ? a[bg*4+lane/8] : 0;
                float f=float(int(((q>>(4*(lane&7)))&15)^8)-8);
                shared_a[b]=__float_as_uint(f);
            }
        }
        float dot[R][B]={};
        if constexpr(H2) {
            half2 hd[R][B];
            #pragma unroll
            for(int r=0;r<R;r++) {
                #pragma unroll
                for(int b=0;b<B;b++)hd[r][b]=__float2half2_rn(0.f);
            }
            #pragma unroll
            for(int j=0;j<16;j++) {
                half2 av[B];
                #pragma unroll
                for(int b=0;b<B;b++)av[b]=bits_half2(__shfl_sync(0xffffffff,shared_a[b],j));
                #pragma unroll
                for(int r=0;r<R;r++) {
                    half2 q=decode_pair((full[r][j/4]>>(8*(j&3)))&255);
                    #pragma unroll
                    for(int b=0;b<B;b++)hd[r][b]=__hfma2(q,av[b],hd[r][b]);
                }
            }
            #pragma unroll
            for(int r=0;r<R;r++) {
                #pragma unroll
                for(int b=0;b<B;b++) {float2 z=__half22float2(hd[r][b]);dot[r][b]=z.x+z.y;}
            }
        } else {
            #pragma unroll
            for(int j=0;j<32;j++) {
                float av[B];
                #pragma unroll
                for(int b=0;b<B;b++)av[b]=__uint_as_float(__shfl_sync(0xffffffff,shared_a[b],j));
                #pragma unroll
                for(int r=0;r<R;r++) {
                    float q=float(int(((full[r][j/8]>>(4*(j&7)))&15)^8)-8);
                    #pragma unroll
                    for(int b=0;b<B;b++)dot[r][b]=__fmaf_rn(q,av[b],dot[r][b]);
                }
            }
        }
        #pragma unroll
        for(int r=0;r<R;r++) {
            float d=(tile+r)*32+lane<rows ? widen(ds[((tile+r)*groups+g)*32+lane]) : 0;
            #pragma unroll
            for(int b=0;b<B;b++)acc[r][b]=__fmaf_rn(dot[r][b],__fmul_rn(d,ad[b]),acc[r][b]);
        }
    }
    #pragma unroll
    for(int r=0;r<R;r++) {
        unsigned row=(tile+r)*32+lane;
        #pragma unroll
        for(int b=0;b<B;b++)if(row<rows && b0+b<batches)out[((b0+b)*32+blockIdx.z)*rows+row]=acc[r][b];
    }
}
#define ONE(H,R,B) extern "C" __global__ void w4a4_h##H##_r##R##_b##B(ARGS) {body<H,R,B>(PASS);}
#define ROWS(H,B) ONE(H,1,B) ONE(H,2,B) ONE(H,4,B)
ROWS(0,1) ROWS(1,1) ROWS(0,4) ROWS(1,4)
template<int O> __device__ __forceinline__ void tree(float (&v)[32]) {
    #pragma unroll
    for(int i=0;i<O;i++)v[i]+=v[i+O];
    if constexpr(O>1)tree<O/2>(v);
}
extern "C" __global__ void reduce32(const float *p,float *out,unsigned rows) {
    unsigned row=blockIdx.x*128+threadIdx.x;if(row>=rows)return;
    float v[32];
    #pragma unroll
    for(int s=0;s<32;s++)v[s]=p[(blockIdx.y*32+s)*rows+row];
    tree<16>(v);
    out[blockIdx.y*rows+row]=v[0];
}
