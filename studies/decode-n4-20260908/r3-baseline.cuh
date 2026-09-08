#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.hpp"

__device__ __forceinline__ float widen(uint16_t v) {return __half2float(__ushort_as_half(v));}
#define ARGS const Q4Block *native,const uint32_t *__restrict__ words,const uint16_t *__restrict__ scales,const uint16_t *__restrict__ a,float *__restrict__ out,unsigned rows,unsigned groups,unsigned batches
#define PASS native,words,scales,a,out,rows,groups,batches

template<int R,bool Pair,int B,int NW,bool Fuse>
__device__ __forceinline__ void body(ARGS) {
    extern __shared__ float scratch[];
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const unsigned tile=Fuse ? blockIdx.x*R : (blockIdx.x*NW+warp)*R;
    const unsigned batch0=blockIdx.y*B;
    for(unsigned slot=0;slot<(Fuse ? 32/NW : 1);++slot) {
        const unsigned stripe=Fuse ? warp+slot*NW : blockIdx.z;
        float acc[R][B]={};
        for(unsigned g=stripe;g<groups;g+=32) {
            float dot[R][B]={};uint32_t full[R][4],lane_pair[B];float lane_a[B];
            #pragma unroll
            for(int b=0;b<B;++b) {
                if constexpr(Pair)lane_pair[b]=batch0+b<batches ? reinterpret_cast<const uint32_t *>(a)[((batch0+b)*groups+g)*16+(lane&15)] : 0;
                else lane_a[b]=batch0+b<batches ? widen(a[((batch0+b)*groups+g)*32+lane]) : 0;
            }
            #pragma unroll
            for(int r=0;r<R;++r) {
                #pragma unroll
                for(int p=0;p<4;++p)full[r][p]=(tile+r)*32+lane<rows ? words[((tile+r)*groups+g)*128+p*32+lane] : 0;
            }
            #pragma unroll
            for(int chunk=0;chunk<4;++chunk) {
                float av[B][8];
                #pragma unroll
                for(int b=0;b<B;++b) {
                    if constexpr(Pair) {
                        #pragma unroll
                        for(int j=0;j<4;++j) {
                            uint32_t v=__shfl_sync(0xffffffff,lane_pair[b],chunk*4+j);
                            av[b][2*j]=widen(uint16_t(v));av[b][2*j+1]=widen(uint16_t(v>>16));
                        }
                    } else {
                        #pragma unroll
                        for(int j=0;j<8;++j)av[b][j]=__shfl_sync(0xffffffff,lane_a[b],chunk*8+j);
                    }
                }
                #pragma unroll
                for(int j=0;j<8;++j) {
                    #pragma unroll
                    for(int r=0;r<R;++r) {
                        float q=float(unpack_nibble(full[r][chunk],j));
                        #pragma unroll
                        for(int b=0;b<B;++b)dot[r][b]=__fmaf_rn(q,av[b][j],dot[r][b]);
                    }
                }
            }
            #pragma unroll
            for(int r=0;r<R;++r) {
                float d=(tile+r)*32+lane<rows ? widen(scales[((tile+r)*groups+g)*32+lane]) : 0;
                #pragma unroll
                for(int b=0;b<B;++b)acc[r][b]=__fmaf_rn(dot[r][b],d,acc[r][b]);
            }
        }
        #pragma unroll
        for(int r=0;r<R;++r) {
            const unsigned row=(tile+r)*32+lane;
            #pragma unroll
            for(int b=0;b<B;++b) {
                if constexpr(Fuse)scratch[((b*32+stripe)*R+r)*32+lane]=acc[r][b];
                else if(row<rows && batch0+b<batches)out[((batch0+b)*32+stripe)*rows+row]=acc[r][b];
            }
        }
    }
    if constexpr(Fuse) {
        __syncthreads();
        #pragma unroll
        for(int offset=16;offset>0;offset/=2) {
            for(unsigned i=threadIdx.x;i<B*offset*R*32;i+=NW*32) {
                unsigned row=i%(R*32),s=(i/(R*32))%offset,b=i/(offset*R*32);
                unsigned index=(b*32+s)*R*32+row;
                scratch[index]=__fadd_rn(scratch[index],scratch[index+offset*R*32]);
            }
            __syncthreads();
        }
        for(unsigned i=threadIdx.x;i<B*R*32;i+=NW*32) {
            unsigned row=tile*32+i%(R*32),b=i/(R*32);
            if(row<rows && batch0+b<batches)out[(batch0+b)*rows+row]=scratch[b*32*R*32+i%(R*32)];
        }
    }
}

#define ONE(ID,R,P,B,NW,F) extern "C" __global__ void r3_##ID##_r##R(ARGS) {body<R,P,B,NW,F>(PASS);}
#define THREE(ID,P,B,NW) ONE(ID,1,P,B,NW,false) ONE(ID,2,P,B,NW,false) ONE(ID,4,P,B,NW,false)
#define TWO(ID,P,B,NW,F) ONE(ID,1,P,B,NW,F) ONE(ID,2,P,B,NW,F)
THREE(100,false,1,1)
THREE(101,false,1,2)
THREE(102,false,1,8)
THREE(103,true,1,4)
THREE(104,true,1,2)
TWO(110,false,4,1,false)
TWO(111,false,4,2,false)
TWO(112,false,4,8,false)
TWO(113,true,4,4,false)
TWO(114,true,4,2,false)
TWO(200,false,1,8,true)
TWO(201,false,1,16,true)
TWO(202,false,1,32,true)
TWO(210,false,4,8,true)
TWO(211,false,4,16,true)
TWO(212,false,4,32,true)
