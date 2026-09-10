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
template<int Mode> __device__ __forceinline__ void quant_body(const uint16_t *a,unsigned *packed,unsigned *planes,float *scale,
                                    unsigned groups,unsigned batches) {
    unsigned lane=threadIdx.x&31, bg=blockIdx.x*4+(threadIdx.x>>5);
    if(bg>=groups*batches)return;
    float v=widen(a[bg*32+lane]),mx=fabsf(v);
    for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));
    float d=mx/7.f;
    int q=mx ? max(-7,min(7,__float2int_rn(v/d))) : 0;
    if constexpr(Mode!=0) {
    #pragma unroll
    for(int p=0;p<4;p++) {
        unsigned bits=__ballot_sync(0xffffffff,(unsigned(q)>>p)&1u);
        if(lane==unsigned(p))planes[bg*4+p]=bits;
    }
    }
    if(lane==0)scale[bg]=d;
    if constexpr(Mode!=1) {
    unsigned word=(unsigned(q)&15u)<<(4*(lane&7));
    for(int o=4;o;o>>=1)word|=__shfl_xor_sync(0xffffffff,word,o);
    if((lane&7)==0)packed[bg*4+lane/8]=word;
    }
}
#define QUANT(NAME,MODE) extern "C" __global__ void NAME(const uint16_t *a,unsigned *packed,unsigned *planes,float *scale,unsigned groups,unsigned batches) {quant_body<MODE>(a,packed,planes,scale,groups,batches);}
QUANT(quantize,2) QUANT(quantize_nibbles,0) QUANT(quantize_planes,1)

#define ARGS const unsigned *__restrict__ w,const uint16_t *__restrict__ ds,const unsigned *__restrict__ a,const float *__restrict__ da,float *__restrict__ out,unsigned rows,unsigned groups,unsigned batches
#define PASS w,ds,a,da,out,rows,groups,batches
// H2 accumulates even/odd K positions separately. G32 -> each half lane has
// <=16*64=1024 magnitude; their FP32 sum is the exact G32 integer dot.
template<int H2,int R,int B,int NW=4,bool Fuse=false>
__device__ __forceinline__ void body(ARGS) {
    extern __shared__ float scratch[];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;
    unsigned tile=Fuse ? blockIdx.x*R : (blockIdx.x*NW+warp)*R,b0=blockIdx.y*B;
    for(unsigned slot=0;slot<(Fuse ? 32/NW:1);slot++) {
    unsigned stripe=Fuse ? warp+slot*NW : blockIdx.z;
    float acc[R][B]={};
    for(unsigned g=stripe;g<groups;g+=32) {
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
            if constexpr(H2==1) {
                unsigned q=b0+b<batches ? a[bg*4+(lane&15)/4] : 0;
                shared_a[b]=half2_bits(decode_pair((q>>(8*(lane&3)))&255));
            } else if constexpr(H2==0) {
                unsigned q=b0+b<batches ? a[bg*4+lane/8] : 0;
                float f=float(int(((q>>(4*(lane&7)))&15)^8)-8);
                shared_a[b]=__float_as_uint(f);
            }
        }
        float dot[R][B]={};
        if constexpr(H2==1) {
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
        } else if constexpr(H2==2) {
            unsigned planes[B][4];
            #pragma unroll
            for(int b=0;b<B;b++) {
                #pragma unroll
                for(int p=0;p<4;p++)planes[b][p]=b0+b<batches ? a[((b0+b)*groups+g)*4+p] : 0;
            }
            #pragma unroll
            for(int r=0;r<R;r++) {
                #pragma unroll
                for(int b=0;b<B;b++) {
                    int sum=0;
                    #pragma unroll
                    for(int p=0;p<4;p++) {
                        int inner=0;
                        #pragma unroll
                        for(int q=0;q<4;q++)inner+=(q==3 ? -8 : 1<<q)*__popc(full[r][p]&planes[b][q]);
                        sum+=(p==3 ? -8 : 1<<p)*inner;
                    }
                    dot[r][b]=float(sum);
                }
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
        for(int b=0;b<B;b++) {
            if constexpr(Fuse)scratch[((stripe*B+b)*R+r)*32+lane]=acc[r][b];
            else if(row<rows && b0+b<batches)out[((b0+b)*32+stripe)*rows+row]=acc[r][b];
        }
    }
    }
    if constexpr(Fuse) {
        __syncthreads();
        #pragma unroll
        for(int offset=16;offset;offset>>=1) {
            for(unsigned i=threadIdx.x;i<unsigned(offset*B*R*32);i+=NW*32)scratch[i]+=scratch[i+offset*B*R*32];
            __syncthreads();
        }
        for(unsigned i=threadIdx.x;i<unsigned(B*R*32);i+=NW*32) {
            unsigned b=i/(R*32),row=tile*32+i%(R*32);
            if(row<rows && b0+b<batches)out[(b0+b)*rows+row]=scratch[i];
        }
    }
}
#define ONE(H,R,B) extern "C" __global__ void w4a4_h##H##_r##R##_b##B(ARGS) {body<H,R,B>(PASS);}
#define ROWS(H,B) ONE(H,1,B) ONE(H,2,B) ONE(H,4,B)
ROWS(0,1) ROWS(1,1) ROWS(0,4) ROWS(1,4) ROWS(2,1) ROWS(2,4)
ONE(2,8,1) ONE(2,8,4) ONE(2,16,1)
#define ALT(R,B,W) extern "C" __global__ void w4a4_h2_r##R##_b##B##_w##W(ARGS) {body<2,R,B,W>(PASS);}
#define GEOM(R,B) ALT(R,B,1) ALT(R,B,2) ALT(R,B,8)
GEOM(2,1) GEOM(4,1) GEOM(8,1) GEOM(1,4) GEOM(2,4) GEOM(4,4)
#define FUSED(H,R,B,W) extern "C" __global__ void w4a4_h##H##_r##R##_b##B##_w##W##_f(ARGS) {body<H,R,B,W,true>(PASS);}
#define FGEOM(H,R) FUSED(H,R,1,8) FUSED(H,R,1,16) FUSED(H,R,1,32)
FGEOM(0,1) FGEOM(0,2) FGEOM(2,1) FGEOM(2,2)
FUSED(0,1,4,8) FUSED(0,1,4,16) FUSED(2,1,4,8) FUSED(2,1,4,16)
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
