#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace p100_one_rail512_detail {
// Byte-for-byte body of the verified parent Q4_0 decoder. Half arithmetic
// is confined to weight preparation, not activation products/accumulation.
__device__ __forceinline__ half2 decode_pair(unsigned byte,half d){
    const unsigned bits=0x64006400u|(byte&15)|((byte>>4)<<16);
    const half2 q=__hsub2(*reinterpret_cast<const half2*>(&bits),__float2half2_rn(1032.f));
    return __hmul2(q,__halves2half2(d,d));
}
union __align__(16) Shared {
    struct {float a[32][64],b[32][64];} stage;
    float low[64][64];
};
static_assert(sizeof(Shared)==16384,"staging/reduction must share16KiB");
}

// Exactly512 threads, cap2. Uniform expert-M interface/format as tile256.
// Positive M/experts, N%64==0, Kpositive multiple32, aligned nonaliasing arrays.
__global__ __launch_bounds__(512,2)
void tile512_one_rail_q4(const float *a,const unsigned char *w,float *out,
                        int t,int n,int k,int experts){
    __shared__ p100_one_rail512_detail::Shared sm;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int rail=warp/8,local_warp=warp%8;
    const int ar=(local_warp/2)*16+(lane/8)*4;
    const int bc=(local_warp%2)*32+(lane%8)*4;
    const int mt=(t+63)/64,ng=n/64,groups=k/32;
    for(int job=blockIdx.x;job<experts*mt*ng;job+=gridDim.x){
        const int e=job/(mt*ng),tm=(job/ng)%mt,col=job%ng;
        // Only16 accumulators/thread: low and high rails have different owners.
        float acc[4][4]={};
        for(int g=0;g<groups;++g){
            __syncthreads();
            {
                const int row=tm*64+tid/8,ks=(tid%8)*4;
                const size_t ai=(size_t(e)*t+row)*k+g*32+ks;
                const float4 v=row<t?*reinterpret_cast<const float4*>(a+ai):float4{};
                sm.stage.a[ks][tid/8]=__half2float(__float2half_rn(v.x));
                sm.stage.a[ks+1][tid/8]=__half2float(__float2half_rn(v.y));
                sm.stage.a[ks+2][tid/8]=__half2float(__float2half_rn(v.z));
                sm.stage.a[ks+3][tid/8]=__half2float(__float2half_rn(v.w));
            }
            // Only first256 threads load/decode B; no duplicate rail decode.
            // This predicate is warp-uniform, and no barrier is inside it.
            if(tid<256){
                const size_t wb=((size_t(e)*ng+col)*groups+g)*1152;
                const unsigned word=*reinterpret_cast<const unsigned*>(w+wb+128+tid*4);
                const half d=reinterpret_cast<const half*>(w+wb)[tid%64];
                const int row=tid%64,ks=(tid/64)*4;
                #pragma unroll 1
                for(int byte=0;byte<4;++byte){
                    const float2 v=__half22float2(p100_one_rail512_detail::decode_pair((word>>(8*byte))&255,d));
                    sm.stage.b[ks+byte][row]=v.x;sm.stage.b[ks+byte+16][row]=v.y;
                }
            }
            __syncthreads();
            #pragma unroll
            for(int s=0;s<16;++s){
                const int kk=rail*16+s;
                const float4 va=*reinterpret_cast<const float4*>(&sm.stage.a[kk][ar]);
                const float4 vb=*reinterpret_cast<const float4*>(&sm.stage.b[kk][bc]);
                const float *av=reinterpret_cast<const float*>(&va);
                const float *bv=reinterpret_cast<const float*>(&vb);
                #pragma unroll
                for(int i=0;i<4;++i){
                    #pragma unroll
                    for(int j=0;j<4;++j)acc[i][j]=__fmaf_rn(av[i],bv[j],acc[i][j]);
                }
            }
        }
        // All A/B consumers must finish before aliasing the entire union.
        __syncthreads();
        if(rail==0){
            #pragma unroll
            for(int i=0;i<4;++i){
                #pragma unroll
                for(int j=0;j<4;++j)sm.low[ar+i][bc+j]=acc[i][j];
            }
        }
        __syncthreads();
        if(rail==1){
            #pragma unroll
            for(int i=0;i<4;++i)if(tm*64+ar+i<t){
                #pragma unroll
                for(int j=0;j<4;++j){
                    const float low=sm.low[ar+i][bc+j];
                    out[(size_t(e)*t+tm*64+ar+i)*n+col*64+bc+j]=__fadd_rn(low,acc[i][j]);
                }
            }
        }
        // No next-job staging can overwrite another warp's low-partial reads.
        __syncthreads();
    }
}
