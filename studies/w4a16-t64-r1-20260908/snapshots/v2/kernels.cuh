#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Q4_0 T64: half scales[64], packed native bytes[16][64]. No expansion.
__global__ void prepare_a16(const float *src, half *dst, size_t count) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<count;
         i+=size_t(gridDim.x)*blockDim.x) dst[i]=__float2half_rn(src[i]);
}

template<int R, int NW, int NT>
__global__ __launch_bounds__(NW*32)
void q4_token_pairs(const half *__restrict__ a, const unsigned char *__restrict__ w,
                    float *__restrict__ out, int tokens, int n, int k, int experts) {
    constexpr int MT=NW*8, J=NT/16, BT=NW*32, BLOAD=16*NT/BT;
    __shared__ half sa[2][32][MT];
    __shared__ half2 sb[2][32][NT];
    __shared__ half sd[2][NT];
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
    const int ar=tid/4, ak=8*(tid%4);
    const int br=tid%NT, bk=BLOAD*(tid/NT);
    const int rm=(warp/2)*8+(lane/8)*2, cn=(warp%2)*(NT/2)+(lane%8)*J;
    const int tm=(tokens+MT-1)/MT, ng=n/NT, groups=k/32;
    for(int work=blockIdx.x;work<experts*tm*ng;work+=gridDim.x) {
        const int e=work/(tm*ng), mt=(work/ng)%tm, col=(work%ng)*NT;
        const int r0=mt*MT+ar;
        const size_t abase=size_t(e)*tokens*k;
        const size_t wbase=size_t(e)*(n/64)*groups*1152;
        float2 acc[2][J]={};
        auto fetch=[&](int g, uint4 &av, unsigned char (&bv)[BLOAD], half &d) {
            av=r0<tokens?*reinterpret_cast<const uint4 *>(a+abase+size_t(r0)*k+g*32+ak):uint4{};
            const size_t b=wbase+(size_t((col+br)/64)*groups+g)*1152;
            d=reinterpret_cast<const half *>(w+b)[br%64];
            #pragma unroll
            for(int i=0;i<BLOAD;++i) bv[i]=w[b+128+(bk+i)*64+br%64];
        };
        auto stage=[&](int buf, const uint4 &av, const unsigned char (&bv)[BLOAD], half d) {
            const half *ah=reinterpret_cast<const half *>(&av);
            #pragma unroll
            for(int i=0;i<8;++i) sa[buf][ak+i][ar]=ah[i];
            #pragma unroll
            for(int i=0;i<BLOAD;++i) {
                sb[buf][bk+i][br]=__half2half2(__int2half_rn(int(bv[i]&15)-8));
                sb[buf][bk+i+16][br]=__half2half2(__int2half_rn(int(bv[i]>>4)-8));
            }
            if(bk==0) sd[buf][br]=d;
        };
        uint4 av; unsigned char bv[BLOAD]; half d;
        fetch(0,av,bv,d); stage(0,av,bv,d); __syncthreads();
        int buf=0;
        for(int g=0;g<groups;++g) {
            if(g+1<groups) fetch(g+1,av,bv,d);
            if constexpr(R==0) {
                float2 sums[2][J]={};
                #pragma unroll
                for(int kk=0;kk<32;++kk) {
                    const float2 x[2]={__half22float2(*reinterpret_cast<const half2 *>(&sa[buf][kk][2*rm])),__half22float2(*reinterpret_cast<const half2 *>(&sa[buf][kk][2*rm+2]))};
                    #pragma unroll
                    for(int j=0;j<J;++j) {
                        const float q=__half2float(__low2half(sb[buf][kk][cn+j]));
                        #pragma unroll
                        for(int i=0;i<2;++i) {
                            sums[i][j].x=fmaf(x[i].x,q,sums[i][j].x);
                            sums[i][j].y=fmaf(x[i].y,q,sums[i][j].y);
                        }
                    }
                }
                #pragma unroll
                for(int j=0;j<J;++j) {
                    const float s=__half2float(sd[buf][cn+j]);
                    #pragma unroll
                    for(int i=0;i<2;++i) {
                        acc[i][j].x=fmaf(sums[i][j].x,s,acc[i][j].x);
                        acc[i][j].y=fmaf(sums[i][j].y,s,acc[i][j].y);
                    }
                }
            } else {
                half2 h[R][2][J];
                #pragma unroll
                for(int r=0;r<R;++r) {
                    #pragma unroll
                    for(int i=0;i<2;++i) {
                        #pragma unroll
                        for(int j=0;j<J;++j) h[r][i][j]=__float2half2_rn(0);
                    }
                }
                #pragma unroll
                for(int kk=0;kk<32;++kk) {
                    const half2 x[2]={*reinterpret_cast<const half2 *>(&sa[buf][kk][2*rm]),*reinterpret_cast<const half2 *>(&sa[buf][kk][2*rm+2])};
                    half2 bvec[J];
                    #pragma unroll
                    for(int j=0;j<J;j+=4) *reinterpret_cast<uint4 *>(&bvec[j])=*reinterpret_cast<const uint4 *>(&sb[buf][kk][cn+j]);
                    #pragma unroll
                    for(int j=0;j<J;++j) {
                        const half2 q=bvec[j];
                        #pragma unroll
                        for(int i=0;i<2;++i) h[kk%R][i][j]=__hfma2(q,x[i],h[kk%R][i][j]);
                    }
                }
                #pragma unroll
                for(int j=0;j<J;++j) {
                    const float s=__half2float(sd[buf][cn+j]);
                    #pragma unroll
                    for(int i=0;i<2;++i) {
                        float2 f=__half22float2(h[0][i][j]);
                        #pragma unroll
                        for(int r=1;r<R;++r) {
                            const float2 v=__half22float2(h[r][i][j]);
                            f.x+=v.x; f.y+=v.y;
                        }
                        acc[i][j].x=fmaf(f.x,s,acc[i][j].x);
                        acc[i][j].y=fmaf(f.y,s,acc[i][j].y);
                    }
                }
            }
            if(g+1<groups) stage(buf^1,av,bv,d);
            __syncthreads(); buf^=1;
        }
        #pragma unroll
        for(int i=0;i<2;++i) {
            const int row=mt*MT+2*(rm+i);
            #pragma unroll
            for(int j=0;j<J;++j) {
                if(row<tokens) out[(size_t(e)*tokens+row)*n+col+cn+j]=acc[i][j].x;
                if(row+1<tokens) out[(size_t(e)*tokens+row+1)*n+col+cn+j]=acc[i][j].y;
            }
        }
        __syncthreads();
    }
}
