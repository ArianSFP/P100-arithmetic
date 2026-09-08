#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__global__ void prepare_a16(const float *src, half *dst, size_t count) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<count;
         i+=size_t(gridDim.x)*blockDim.x) dst[i]=__float2half_rn(src[i]);
}

// Inputs stay A16 in global memory. Dequantization and all accumulation are
// FP32. Payload: scales[64] followed by packed Q4 bytes[64][16] per K32 stage.
// SK2 follows the original T64 split order, enabling a byte-identical control.
template<int MT, int NT, int BT, int SK, int SINGLE, int CHUNK=8, int COLUMN=0>
__global__ __launch_bounds__(BT)
void q4_f32_tile(const half *__restrict__ a, const unsigned char *__restrict__ w,
                 float *__restrict__ out, int tokens, int n, int k, int experts) {
    constexpr int NW=BT/32, WN=2, WM=NW/(WN*SK);
    constexpr int RM=MT/(WM*4), CN=NT/(WN*8);
    constexpr int NA=MT*4/BT, NB=NT*4/BT, BUFS=SINGLE?1:2;
    static_assert(MT%(WM*4)==0 && RM%2==0 && CN%4==0);
    static_assert(NA>0 && NB>0 && 32%CHUNK==0);
    union Shared {
        struct {float a[BUFS][32][MT]; float b[BUFS][32][NT];} stage;
        float reduction[SK==2?MT:1][NT];
    };
    __shared__ Shared sm;
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
    const int kg=warp/(NW/SK), wm=(warp%(NW/SK))/WN, wn=warp%WN;
    const int ar=wm*(MT/WM)+(lane/8)*RM, bc=wn*(NT/WN)+(lane%8)*CN;
    const int tm=(tokens+MT-1)/MT, ng=n/NT, groups=k/32;
    for(int work=blockIdx.x;work<experts*tm*ng;work+=gridDim.x) {
        const int e=work/(tm*ng), mt=(work/ng)%tm, col=work%ng*NT;
        const size_t abase=size_t(e)*tokens*k;
        const size_t wbase=size_t(e)*(n/64)*groups*1152;
        uint4 pa[NA]; uint32_t pb[NB]; float pd[NB];
        auto fetch=[&](int g) {
            #pragma unroll
            for(int l=0;l<NA;++l) {
                const int at=tid+l*BT, row=mt*MT+at/4, kk=8*(at%4);
                pa[l]=row<tokens?*reinterpret_cast<const uint4 *>(a+abase+size_t(row)*k+32*g+kk):uint4{};
            }
            #pragma unroll
            for(int l=0;l<NB;++l) {
                const int bt=tid+l*BT, row=col+bt/4, kk=4*(bt%4);
                const size_t b=wbase+(size_t(row/64)*groups+g)*1152;
                pd[l]=__half2float(reinterpret_cast<const half *>(w+b)[row%64]);
                pb[l]=*reinterpret_cast<const uint32_t *>(w+b+128+(row%64)*16+kk);
            }
        };
        auto stage=[&](int buf) {
            #pragma unroll
            for(int l=0;l<NA;++l) {
                const int at=tid+l*BT, row=at/4, kk=8*(at%4);
                const half *v=reinterpret_cast<const half *>(&pa[l]);
                #pragma unroll
                for(int i=0;i<8;++i) sm.stage.a[buf][kk+i][row]=__half2float(v[i]);
            }
            #pragma unroll
            for(int l=0;l<NB;++l) {
                const int bt=tid+l*BT, row=bt/4, kk=4*(bt%4);
                #pragma unroll
                for(int i=0;i<4;++i) {
                    sm.stage.b[buf][kk+i][row]=float(int((pb[l]>>(8*i))&15)-8)*pd[l];
                    sm.stage.b[buf][kk+i+16][row]=float(int((pb[l]>>(8*i+4))&15)-8)*pd[l];
                }
            }
        };
        float acc[RM][CN]={};
        fetch(0); stage(0); __syncthreads(); int buf=0;
        for(int g=0;g<groups;++g) {
            if(g+1<groups)fetch(g+1);
            #pragma unroll 1
            for(int chunk=0;chunk<32/SK/CHUNK;++chunk) {
                #pragma unroll
                for(int step=0;step<CHUNK;++step) {
                    const int kk=kg*(32/SK)+chunk*CHUNK+step;
                    float av[RM], bv[CN];
                    #pragma unroll
                    for(int i=0;i<RM;i+=2) *reinterpret_cast<float2 *>(&av[i])=*reinterpret_cast<const float2 *>(&sm.stage.a[buf][kk][ar+i]);
                    #pragma unroll
                    for(int j=0;j<CN;j+=4) *reinterpret_cast<float4 *>(&bv[j])=*reinterpret_cast<const float4 *>(&sm.stage.b[buf][kk][bc+j]);
                    #pragma unroll
                    for(int outer=0;outer<(COLUMN?CN:RM);++outer) {
                        #pragma unroll
                        for(int inner=0;inner<(COLUMN?RM:CN);++inner) {
                            const int i=COLUMN?inner:outer, j=COLUMN?outer:inner;
                            acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
                        }
                    }
                }
            }
            if constexpr(SINGLE) __syncthreads();
            if(g+1<groups)stage(SINGLE?0:buf^1);
            __syncthreads(); if constexpr(!SINGLE)buf^=1;
        }
        if constexpr(SK==2) {
            if(kg==0) {
                #pragma unroll
                for(int i=0;i<RM;++i) {
                    #pragma unroll
                    for(int j=0;j<CN;++j) sm.reduction[ar+i][bc+j]=acc[i][j];
                }
            }
            __syncthreads();
        }
        if(kg==SK-1) {
            #pragma unroll
            for(int i=0;i<RM;++i) {
                const int row=mt*MT+ar+i;
                #pragma unroll
                for(int j=0;j<CN;++j) {
                    float value=acc[i][j];
                    if constexpr(SK==2)value=sm.reduction[ar+i][bc+j]+value;
                    if(row<tokens)out[(size_t(e)*tokens+row)*n+col+bc+j]=value;
                }
            }
        }
        __syncthreads();
    }
}
