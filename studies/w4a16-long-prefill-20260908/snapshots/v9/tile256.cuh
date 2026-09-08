#pragma once
// Include the parent's verified wide.cuh first for decode_q4_pair().
// Q4_0 word-major 1152B stages, A32 -> A16 -> FP32, FP32 products/rails.
// MIN_BLOCKS is a compilation experiment, not proof of achieved occupancy.
template<int U,int MIN_BLOCKS>
__global__ __launch_bounds__(256,MIN_BLOCKS)
void tile256_q4(const float *a,const unsigned char *w,float *out,
                int t,int n,int k,int experts){
    static_assert(U==16||U==32,"bounded unroll choices only");
    static_assert(MIN_BLOCKS==3||MIN_BLOCKS==4,"bounded residency choices only");
    __shared__ float sa[32][64],sb[32][64];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int ar=(warp/2)*16+(lane/8)*4;
    const int bc=(warp%2)*32+(lane%8)*4;
    const int mt=(t+63)/64,ng=n/64,groups=k/32;
    for(int job=blockIdx.x;job<experts*mt*ng;job+=gridDim.x){
        const int e=job/(mt*ng),tm=(job/ng)%mt,col=job%ng;
        float4 pa0,pa1;unsigned pb;half pd;
        auto fetch=[&](int g){
            const int row=tm*64+tid/4;
            const size_t ai=(size_t(e)*t+row)*k+g*32+(tid%4)*8;
            pa0=row<t?*reinterpret_cast<const float4*>(a+ai):float4{};
            pa1=row<t?*reinterpret_cast<const float4*>(a+ai+4):float4{};
            const size_t wb=((size_t(e)*ng+col)*groups+g)*1152;
            pd=reinterpret_cast<const half*>(w+wb)[tid%64];
            pb=*reinterpret_cast<const unsigned*>(w+wb+128+tid*4);
        };
        auto stage=[&](){
            const float *v0=reinterpret_cast<const float*>(&pa0);
            const float *v1=reinterpret_cast<const float*>(&pa1);
            #pragma unroll
            for(int i=0;i<4;++i){
                sa[(tid%4)*8+i][tid/4]=__half2float(__float2half_rn(v0[i]));
                sa[(tid%4)*8+i+4][tid/4]=__half2float(__float2half_rn(v1[i]));
            }
            const int row=tid%64,kk=(tid/64)*4;
            #pragma unroll
            for(int i=0;i<4;++i){
                const float2 v=__half22float2(decode_q4_pair((pb>>(8*i))&255,pd));
                sb[kk+i][row]=v.x;sb[kk+i+16][row]=v.y;
            }
        };
        // Two separate rails persist across all G32 groups exactly as T64:
        // lo sees k%32=0..15; hi sees16..31. Only final output sums the rails.
        float lo[4][4]={},hi[4][4]={};
        fetch(0);stage();__syncthreads();
        for(int g=0;g<groups;++g){
            if(g+1<groups)fetch(g+1);
            #pragma unroll 1
            for(int chunk=0;chunk<32/U;++chunk){
                #pragma unroll
                for(int s=0;s<U;++s){
                    const int kk=chunk*U+s;
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
