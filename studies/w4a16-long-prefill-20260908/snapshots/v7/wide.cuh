#pragma once
__device__ __forceinline__ half2 decode_q4_pair(unsigned byte,half d){
    const unsigned bits=0x64006400u|(byte&15)|((byte>>4)<<16);
    const half2 q=__hsub2(*reinterpret_cast<const half2*>(&bits),__float2half2_rn(1032.f));
    return __hmul2(q,__halves2half2(d,d));
}
__global__ void decode_check_kernel(unsigned *out){
    const unsigned x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=65536u*16)return;
    const unsigned code=x/65536;
    const half2 v=decode_q4_pair(code|((15-code)<<4),__ushort_as_half(x&65535));
    out[x]=*reinterpret_cast<const unsigned*>(&v);
}
// W4_0 only. Both modes reproduce half-rounded dequantized weights;
// PACKED changes weight preparation, NEVER the FP32 GEMM accumulation.
template<int U,bool PACKED,bool SPLIT=false,int MT=64>
__global__ __launch_bounds__(128,MT==32?3:1)
void wide_q4(const float *a,const unsigned char *w,float *out,int t,int n,int k,int experts){
    static_assert(MT==32||MT==64);
    constexpr int RM=MT/8;
    __shared__ float sa[32][MT],sb[32][128];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int ar=(warp/2)*(MT/2)+(lane/8)*RM,bc=(warp%2)*64+(lane%8)*8;
    const int mt=(t+MT-1)/MT,ng=n/128,groups=k/32;
    for(int job=blockIdx.x;job<experts*mt*ng;job+=gridDim.x){
        const int e=job/(mt*ng),tm=(job/ng)%mt,col=job%ng;
        float4 pa[4];unsigned pb[4];half pd[4];
        auto fetch=[&](int g){
            #pragma unroll
            for(int l=0;l<4;++l){
                const int at=tid+l*128;
                if(l<MT/32){
                    const int row=tm*MT+at/4;
                    const size_t ai=(size_t(e)*t+row)*k+g*32+(at%4)*8;
                    pa[2*l]=row<t?*reinterpret_cast<const float4*>(a+ai):float4{};
                    pa[2*l+1]=row<t?*reinterpret_cast<const float4*>(a+ai+4):float4{};
                }
                const int row=at%128,chunk=at/128;
                const size_t wb=((size_t(e)*(n/64)+col*2+row/64)*groups+g)*1152;
                pd[l]=reinterpret_cast<const half*>(w+wb)[row%64];
                pb[l]=*reinterpret_cast<const unsigned*>(w+wb+128+(chunk*64+row%64)*4);
            }
        };
        auto stage=[&](){
            #pragma unroll
            for(int l=0;l<4;++l){
                const int at=tid+l*128;
                if(l<MT/32){
                    const float *av=reinterpret_cast<const float*>(&pa[2*l]);
                    #pragma unroll
                    for(int i=0;i<8;++i)sa[(at%4)*8+i][at/4]=__half2float(__float2half_rn(av[i]));
                }
                const int row=at%128,kk=(at/128)*4;
                #pragma unroll
                for(int i=0;i<4;++i){
                    if constexpr(PACKED){
                        const unsigned byte=(pb[l]>>(8*i))&255;
                        const float2 v=__half22float2(decode_q4_pair(byte,pd[l]));
                        sb[kk+i][row]=v.x;sb[kk+i+16][row]=v.y;
                    }else{
                        const float d=__half2float(pd[l]);
                        sb[kk+i][row]=__half2float(__float2half_rn(__fmul_rn(float(int((pb[l]>>(8*i))&15)-8),d)));
                        sb[kk+i+16][row]=__half2float(__float2half_rn(__fmul_rn(float(int((pb[l]>>(8*i+4))&15)-8),d)));
                    }
                }
            }
        };
        float acc[RM][8]={},acc_hi[SPLIT?RM:1][SPLIT?8:1]={};fetch(0);stage();__syncthreads();
        for(int g=0;g<groups;++g){
            if(g+1<groups)fetch(g+1);
            #pragma unroll 1
            for(int chunk=0;chunk<32/U;++chunk){
                #pragma unroll
                for(int s=0;s<U;++s){
                    const int kk=chunk*U+s;float av[RM],bv[8];
                    #pragma unroll
                    for(int j=0;j<RM;j+=4){
                        *reinterpret_cast<float4*>(av+j)=*reinterpret_cast<const float4*>(&sa[kk][ar+j]);
                    }
                    #pragma unroll
                    for(int j=0;j<8;j+=4){
                        *reinterpret_cast<float4*>(bv+j)=*reinterpret_cast<const float4*>(&sb[kk][bc+j]);
                    }
                    #pragma unroll
                    for(int i=0;i<RM;++i){
                        #pragma unroll
                        for(int j=0;j<8;++j){
                            if constexpr(SPLIT){
                                if(kk<16)acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
                                else acc_hi[i][j]=fmaf(av[i],bv[j],acc_hi[i][j]);
                            }else acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
                        }
                    }
                }
            }
            __syncthreads();if(g+1<groups)stage();__syncthreads();
        }
        #pragma unroll
        for(int i=0;i<RM;++i)if(tm*MT+ar+i<t){
            #pragma unroll
            for(int j=0;j<8;++j){
                if constexpr(SPLIT)out[(size_t(e)*t+tm*MT+ar+i)*n+col*128+bc+j]=acc[i][j]+acc_hi[i][j];
                else out[(size_t(e)*t+tm*MT+ar+i)*n+col*128+bc+j]=acc[i][j];
            }
        }
        __syncthreads();
    }
}
