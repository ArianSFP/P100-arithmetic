#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Self-contained copy of the verified packed Q4_0 weight decoder. These half
// operations prepare weights ONLY; every GEMM product and sum below is FP32.
namespace p100_direct_stage_detail {
__device__ __forceinline__ half2 decode_pair(unsigned byte,half d){
    const unsigned bits=0x64006400u|(byte&15)|((byte>>4)<<16);
    const half2 q=__hsub2(*reinterpret_cast<const half2*>(&bits),__float2half2_rn(1032.f));
    return __hmul2(q,__halves2half2(d,d));
}
}

// Contract: 256 threads; positive M/experts, N%64==0, K%32==0 with K>=32;
// A32 row-major, Q4_0 word-major 1152B stages, nonaliasing output, aligned bases.
// One bounded cap4/unroll32 candidate. No next-stage registers or extra scratch.
__global__ __launch_bounds__(256,4)
void tile256_direct_stage_q4(const float *a,const unsigned char *w,float *out,
                             int t,int n,int k,int experts){
    __shared__ float sa[32][64],sb[32][64];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int ar=(warp/2)*16+(lane/8)*4,bc=(warp%2)*32+(lane%8)*4;
    const int mt=(t+63)/64,ng=n/64,groups=k/32;
    for(int job=blockIdx.x;job<experts*mt*ng;job+=gridDim.x){
        const int e=job/(mt*ng),tm=(job/ng)%mt,col=job%ng;
        float lo[4][4]={},hi[4][4]={};
        for(int g=0;g<groups;++g){
            // No producer touches shared memory until every prior consumer
            // has finished. Uniform even at g=0; no warp-specific barrier.
            __syncthreads();
            {
                const int row=tm*64+tid/4,ks=(tid%4)*8;
                const size_t ai=(size_t(e)*t+row)*k+g*32+ks;
                // Keep this tiny producer loop rolled: only one raw float4
                // and its immediate conversions need be live at a time.
                #pragma unroll 1
                for(int vec=0;vec<2;++vec){
                    const float4 v=row<t?*reinterpret_cast<const float4*>(a+ai+4*vec):float4{};
                    sa[ks+4*vec][tid/4]=__half2float(__float2half_rn(v.x));
                    sa[ks+4*vec+1][tid/4]=__half2float(__float2half_rn(v.y));
                    sa[ks+4*vec+2][tid/4]=__half2float(__float2half_rn(v.z));
                    sa[ks+4*vec+3][tid/4]=__half2float(__float2half_rn(v.w));
                }
            }
            {
                const size_t wb=((size_t(e)*ng+col)*groups+g)*1152;
                const unsigned word=*reinterpret_cast<const unsigned*>(w+wb+128+tid*4);
                const half d=reinterpret_cast<const half*>(w+wb)[tid%64];
                const int row=tid%64,ks=(tid/64)*4;
                // Do not retain four decoded pairs: convert/store each before
                // the next byte. Rolling limits compiler load-ahead opportunity.
                #pragma unroll 1
                for(int byte=0;byte<4;++byte){
                    const float2 v=__half22float2(p100_direct_stage_detail::decode_pair((word>>(8*byte))&255,d));
                    sb[ks+byte][row]=v.x;sb[ks+byte+16][row]=v.y;
                }
            }
            __syncthreads();
            #pragma unroll
            for(int kk=0;kk<32;++kk){
                const float4 va=*reinterpret_cast<const float4*>(&sa[kk][ar]);
                const float4 vb=*reinterpret_cast<const float4*>(&sb[kk][bc]);
                const float *av=reinterpret_cast<const float*>(&va);
                const float *bv=reinterpret_cast<const float*>(&vb);
                #pragma unroll
                for(int i=0;i<4;++i){
                    #pragma unroll
                    for(int j=0;j<4;++j){
                        if(kk<16)lo[i][j]=__fmaf_rn(av[i],bv[j],lo[i][j]);
                        else hi[i][j]=__fmaf_rn(av[i],bv[j],hi[i][j]);
                    }
                }
            }
        }
        #pragma unroll
        for(int i=0;i<4;++i)if(tm*64+ar+i<t){
            #pragma unroll
            for(int j=0;j<4;++j)
                out[(size_t(e)*t+tm*64+ar+i)*n+col*64+bc+j]=__fadd_rn(lo[i][j],hi[i][j]);
        }
        // Before any warp begins producing the next persistent tile.
        __syncthreads();
    }
}
