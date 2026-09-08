#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.hpp"

__device__ __forceinline__ float widen(uint16_t v) {return __half2float(__ushort_as_half(v));}
#define ARGS const Q4Block *native,const uint32_t *__restrict__ words,const uint16_t *__restrict__ scales,const uint16_t *__restrict__ a,float *__restrict__ out,unsigned rows,unsigned groups,unsigned batches
#define PASS native,words,scales,a,out,rows,groups,batches

// All array indices are bounded below 2^32 by the host's shape checks.
template<int R,int AV,bool Stream,bool VecW,int B,int NW>
__device__ __forceinline__ void body(ARGS) {
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;
    unsigned tile=(blockIdx.x*NW+warp)*R,batch0=blockIdx.y*B;
    float acc[R][B]={};
    for(unsigned g=blockIdx.z;g<groups;g+=32) {
        float dot[R][B]={};uint32_t full[R][4];float lane_a[B];
        if constexpr(AV==3) {
            #pragma unroll
            for(int b=0;b<B;++b)lane_a[b]=batch0+b<batches ? widen(a[((batch0+b)*groups+g)*32+lane]) : 0;
        }
        if constexpr(!Stream) {
            #pragma unroll
            for(int r=0;r<R;++r) {
                unsigned row=(tile+r)*32+lane;
                if constexpr(VecW) {
                    uint4 v=make_uint4(0,0,0,0);
                    if(row<rows)v=reinterpret_cast<const uint4 *>(words)[((tile+r)*groups+g)*32+lane];
                    full[r][0]=v.x;full[r][1]=v.y;full[r][2]=v.z;full[r][3]=v.w;
                } else {
                    #pragma unroll
                    for(int p=0;p<4;++p)full[r][p]=row<rows ? words[((tile+r)*groups+g)*128+p*32+lane] : 0;
                }
            }
        }
        #pragma unroll
        for(int chunk=0;chunk<4;++chunk) {
            uint32_t w[R];float av[B][8];
            #pragma unroll
            for(int r=0;r<R;++r) {
                if constexpr(Stream)w[r]=(tile+r)*32+lane<rows ? words[((tile+r)*groups+g)*128+chunk*32+lane] : 0;
                else w[r]=full[r][chunk];
            }
            #pragma unroll
            for(int b=0;b<B;++b) {
                unsigned off=((batch0+b)*groups+g)*32+chunk*8;
                if constexpr(AV==0) {
                    #pragma unroll
                    for(int j=0;j<8;++j)av[b][j]=batch0+b<batches ? widen(a[off+j]) : 0;
                } else if constexpr(AV==1) {
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        uint32_t v=batch0+b<batches ? reinterpret_cast<const uint32_t *>(a)[off/2+j] : 0;
                        av[b][2*j]=widen(uint16_t(v));av[b][2*j+1]=widen(uint16_t(v>>16));
                    }
                } else if constexpr(AV==2) {
                    uint4 v=make_uint4(0,0,0,0);
                    if(batch0+b<batches)v=reinterpret_cast<const uint4 *>(a)[off/8];
                    uint32_t parts[4]={v.x,v.y,v.z,v.w};
                    #pragma unroll
                    for(int j=0;j<4;++j){av[b][2*j]=widen(uint16_t(parts[j]));av[b][2*j+1]=widen(uint16_t(parts[j]>>16));}
                } else {
                    #pragma unroll
                    for(int j=0;j<8;++j)av[b][j]=__shfl_sync(0xffffffff,lane_a[b],chunk*8+j);
                }
            }
            #pragma unroll
            for(int j=0;j<8;++j) {
                #pragma unroll
                for(int r=0;r<R;++r) {
                    float q=float(unpack_nibble(w[r],j));
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
        unsigned row=(tile+r)*32+lane;
        #pragma unroll
        for(int b=0;b<B;++b)if(row<rows && batch0+b<batches)out[((batch0+b)*32+blockIdx.z)*rows+row]=acc[r][b];
    }
}
#define ONE(ID,R,AV,ST,VW,B,NW) extern "C" __global__ void r2_##ID##_r##R(ARGS) {body<R,AV,ST,VW,B,NW>(PASS);}
#define ROWS(ID,AV,ST,VW,B,NW) ONE(ID,2,AV,ST,VW,B,NW) ONE(ID,4,AV,ST,VW,B,NW) ONE(ID,6,AV,ST,VW,B,NW) ONE(ID,8,AV,ST,VW,B,NW)
ROWS(10,0,false,false,1,4)
ROWS(11,1,false,false,1,4)
ROWS(12,2,false,false,1,4)
ROWS(13,0,true,false,1,4)
ROWS(14,3,false,false,1,4)
ROWS(15,1,true,false,1,4)
ROWS(16,0,false,true,1,4)
ROWS(17,1,false,true,1,4)
ROWS(18,1,false,false,1,1)
ROWS(19,1,false,false,1,2)
ROWS(20,1,false,false,1,8)
ROWS(21,0,false,false,1,1)
ROWS(22,0,false,false,1,2)
ROWS(23,0,false,false,1,8)
#define BATCH(ID,AV,VW,B,NW) ONE(ID,1,AV,false,VW,B,NW) ONE(ID,2,AV,false,VW,B,NW) ONE(ID,4,AV,false,VW,B,NW)
BATCH(30,1,false,4,4)
BATCH(31,1,false,2,4)
BATCH(32,1,false,4,2)
BATCH(33,1,false,4,1)
BATCH(34,1,true,4,4)
BATCH(35,0,false,4,4)
BATCH(36,3,false,4,4)
