#pragma once
// Q4_0 only. Packed arithmetic is confined to half-rounded weight decoding;
// GEMM products and both persistent K-half accumulation rails are FP32.
template<int U,bool PACKED,bool VSTORE=false,int NSPEC=0,int KSPEC=0,int MAP=0>
__global__ __launch_bounds__(128,5)
void compact_q4(const float *a,const unsigned char *w,float *out,int t,int n,int k,int experts){
    __shared__ float sa[32][32],sb[32][64];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int ar=(warp/2)*16+(lane/8)*4,bc=(warp%2)*32+(lane%8)*4;
    const int nn=NSPEC?NSPEC:n,kkdim=KSPEC?KSPEC:k;
    const int mt=(t+31)/32,ng=nn/64,groups=kkdim/32;
    static_assert(MAP>=0&&MAP<=2);
    const int total=experts*mt*ng;
    const int begin=MAP==1?int((static_cast<long long>(total)*blockIdx.x)/gridDim.x):int(blockIdx.x);
    const int end=MAP==1?int((static_cast<long long>(total)*(blockIdx.x+1))/gridDim.x):total;
    for(int job=begin;job<end;job+=MAP==1?1:gridDim.x){
        const int e=job/(mt*ng);
        int tm=(job/ng)%mt,col=job%ng;
        if constexpr(MAP==2){
            const int local=job%(mt*ng),slab=local/(2*ng);
            const int height=min(2,mt-2*slab),rem=local%(2*ng);
            tm=2*slab+(rem/2)%height;
            col=2*(rem/(2*height))+rem%2;
        }
        float4 pa[2];unsigned pb[2];half pd[2];
        auto fetch=[&](int g){
            const int row=tm*32+tid/4;
            const size_t ai=(size_t(e)*t+row)*kkdim+g*32+(tid%4)*8;
            pa[0]=row<t?*reinterpret_cast<const float4*>(a+ai):float4{};
            pa[1]=row<t?*reinterpret_cast<const float4*>(a+ai+4):float4{};
            const size_t wb=((size_t(e)*ng+col)*groups+g)*1152;
            #pragma unroll
            for(int l=0;l<2;++l){
                const int at=tid+l*128;
                pd[l]=reinterpret_cast<const half*>(w+wb)[at%64];
                pb[l]=*reinterpret_cast<const unsigned*>(w+wb+128+at*4);
            }
        };
        auto stage=[&](){
            const float *av=reinterpret_cast<const float*>(pa);
            #pragma unroll
            for(int i=0;i<8;++i)sa[(tid%4)*8+i][tid/4]=__half2float(__float2half_rn(av[i]));
            #pragma unroll
            for(int l=0;l<2;++l){
                const int at=tid+l*128,row=at%64,kk=(at/64)*4;
                #pragma unroll
                for(int i=0;i<4;++i){
                    if constexpr(PACKED){
                        const float2 v=__half22float2(decode_q4_pair((pb[l]>>(8*i))&255,pd[l]));
                        sb[kk+i][row]=v.x;sb[kk+i+16][row]=v.y;
                    }else{
                        const float d=__half2float(pd[l]);
                        sb[kk+i][row]=__half2float(__float2half_rn(__fmul_rn(float(int((pb[l]>>(8*i))&15)-8),d)));
                        sb[kk+i+16][row]=__half2float(__float2half_rn(__fmul_rn(float(int((pb[l]>>(8*i+4))&15)-8),d)));
                    }
                }
            }
        };
        float lo[4][4]={},hi[4][4]={};fetch(0);stage();__syncthreads();
        for(int g=0;g<groups;++g){
            if(g+1<groups)fetch(g+1);
            #pragma unroll 1
            for(int chunk=0;chunk<32/U;++chunk){
                #pragma unroll
                for(int s=0;s<U;++s){
                    const int kk=chunk*U+s;
                    const float4 va=*reinterpret_cast<const float4*>(&sa[kk][ar]);
                    const float4 vb=*reinterpret_cast<const float4*>(&sb[kk][bc]);
                    const float *av=reinterpret_cast<const float*>(&va),*bv=reinterpret_cast<const float*>(&vb);
                    #pragma unroll
                    for(int i=0;i<4;++i){
                        #pragma unroll
                        for(int j=0;j<4;++j){
                            if(kk<16)lo[i][j]=fmaf(av[i],bv[j],lo[i][j]);
                            else hi[i][j]=fmaf(av[i],bv[j],hi[i][j]);
                        }
                    }
                }
            }
            __syncthreads();if(g+1<groups)stage();__syncthreads();
        }
        #pragma unroll
        for(int i=0;i<4;++i)if(tm*32+ar+i<t){
            const size_t oi=(size_t(e)*t+tm*32+ar+i)*nn+col*64+bc;
            if constexpr(VSTORE){
                float4 result;
                float *r=reinterpret_cast<float*>(&result);
                #pragma unroll
                for(int j=0;j<4;++j)r[j]=__fadd_rn(lo[i][j],hi[i][j]);
                *reinterpret_cast<float4*>(out+oi)=result;
            }else{
                #pragma unroll
                for(int j=0;j<4;++j)out[oi+j]=lo[i][j]+hi[i][j];
            }
        }
        __syncthreads();
    }
}
