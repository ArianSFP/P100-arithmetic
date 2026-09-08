#pragma once
// Include verified parent wide.cuh first. Frozen tile256.cuh is unchanged.
// Exactly two cap4 experiments: no prefetch, or four-register half2 A prefetch.
template<bool HALF_PREFETCH>
__global__ __launch_bounds__(256,4)
void tile256_liveness_q4(const float *a,const unsigned char *w,float *out,
                         int t,int n,int k,int experts){
    __shared__ float sa[32][64],sb[32][64];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int ar=(warp/2)*16+(lane/8)*4,bc=(warp%2)*32+(lane%8)*4;
    const int mt=(t+63)/64,ng=n/64,groups=k/32;
    for(int job=blockIdx.x;job<experts*mt*ng;job+=gridDim.x){
        const int e=job/(mt*ng),tm=(job/ng)%mt,col=job%ng;
        // The unused representation is eliminated by constexpr branches.
        float4 pa0,pa1;half2 pha[4];unsigned pb;half pd;
        auto fetch=[&](int g){
            const int row=tm*64+tid/4;
            const size_t ai=(size_t(e)*t+row)*k+g*32+(tid%4)*8;
            const float4 v0=row<t?*reinterpret_cast<const float4*>(a+ai):float4{};
            const float4 v1=row<t?*reinterpret_cast<const float4*>(a+ai+4):float4{};
            if constexpr(HALF_PREFETCH){
                pha[0]=__floats2half2_rn(v0.x,v0.y);pha[1]=__floats2half2_rn(v0.z,v0.w);
                pha[2]=__floats2half2_rn(v1.x,v1.y);pha[3]=__floats2half2_rn(v1.z,v1.w);
            }else{pa0=v0;pa1=v1;}
            const size_t wb=((size_t(e)*ng+col)*groups+g)*1152;
            pd=reinterpret_cast<const half*>(w+wb)[tid%64];
            pb=*reinterpret_cast<const unsigned*>(w+wb+128+tid*4);
        };
        auto stage=[&](){
            if constexpr(HALF_PREFETCH){
                #pragma unroll
                for(int i=0;i<4;++i){
                    const float2 v=__half22float2(pha[i]);
                    sa[(tid%4)*8+2*i][tid/4]=v.x;
                    sa[(tid%4)*8+2*i+1][tid/4]=v.y;
                }
            }else{
                const float *v0=reinterpret_cast<const float*>(&pa0);
                const float *v1=reinterpret_cast<const float*>(&pa1);
                #pragma unroll
                for(int i=0;i<4;++i){
                    sa[(tid%4)*8+i][tid/4]=__half2float(__float2half_rn(v0[i]));
                    sa[(tid%4)*8+i+4][tid/4]=__half2float(__float2half_rn(v1[i]));
                }
            }
            const int row=tid%64,kk=(tid/64)*4;
            #pragma unroll
            for(int i=0;i<4;++i){
                const float2 v=__half22float2(decode_q4_pair((pb>>(8*i))&255,pd));
                sb[kk+i][row]=v.x;sb[kk+i+16][row]=v.y;
            }
        };
        float lo[4][4]={},hi[4][4]={};
        fetch(0);stage();__syncthreads();
        for(int g=0;g<groups;++g){
            if constexpr(HALF_PREFETCH){if(g+1<groups)fetch(g+1);}
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
            // Register-only fetch is safe before the barrier: no thread writes
            // the shared stage until every warp has finished consuming it.
            if constexpr(!HALF_PREFETCH){if(g+1<groups)fetch(g+1);}
            __syncthreads();if(g+1<groups)stage();__syncthreads();
        }
        #pragma unroll
        for(int i=0;i<4;++i)if(tm*64+ar+i<t){
            #pragma unroll
            for(int j=0;j<4;++j)
                out[(size_t(e)*t+tm*64+ar+i)*n+col*64+bc+j]=__fadd_rn(lo[i][j],hi[i][j]);
        }
        __syncthreads();
    }
}
