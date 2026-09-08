#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// All reviewed worker counts are multiples of32 (K is a multiple of32).
__global__ void prepare_a16_vector(const float *src,half *dst,size_t count) {
    for(size_t v=size_t(blockIdx.x)*blockDim.x+threadIdx.x;v<count/8;
        v+=size_t(gridDim.x)*blockDim.x) {
        const float4 a=*reinterpret_cast<const float4*>(src+v*8);
        const float4 b=*reinterpret_cast<const float4*>(src+v*8+4);
        const half2 h[4]={__floats2half2_rn(a.x,a.y),__floats2half2_rn(a.z,a.w),
                         __floats2half2_rn(b.x,b.y),__floats2half2_rn(b.z,b.w)};
        *reinterpret_cast<uint4*>(dst+v*8)=*reinterpret_cast<const uint4*>(h);
    }
}

// Conversion and tiled transpose are one timed pass. Output layout is
// [expert][token_tile64][K32_group][k32][token64], with zero-filled token tails.
template<bool FLOAT_OUT>
__global__ void prepare_tiled(const float *a, void *dst,int t,int k,int experts) {
    __shared__ float tile[32][33];
    const int x=threadIdx.x%32,y=threadIdx.x/32,mt=(t+63)/64,g=k/32;
    for(int job=blockIdx.x;job<experts*mt*g*2;job+=gridDim.x) {
        const int half_tile=job%2, group=(job/2)%g, token_tile=(job/(2*g))%mt,e=job/(2*g*mt);
        #pragma unroll
        for(int i=0;i<32;i+=8) {
            const int row=token_tile*64+half_tile*32+y+i;
            tile[y+i][x]=row<t?a[(size_t(e)*t+row)*k+group*32+x]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<32;i+=8) {
            const size_t out=((size_t(e)*mt+token_tile)*g+group)*2048+(y+i)*64+half_tile*32+x;
            const half h=__float2half_rn(tile[x][y+i]);
            if constexpr(FLOAT_OUT)reinterpret_cast<float*>(dst)[out]=__half2float(h);
            else reinterpret_cast<half*>(dst)[out]=h;
        }
        __syncthreads();
    }
}

// L=0 row-major A16, L=1 tiled A16, L=2 tiled widened A16,
// L=3 tiled A16 with four-token rather than eight-token staging vectors.
// B=1 uses [four-byte chunk][row][byte] instead of [row][16 bytes].
// FP32 sequential K order and weight operand preparation are unchanged.
template<int L,int B,int U,int SWAP=0>
__global__ __launch_bounds__(128)
void delivery_lead(const void *ap,const unsigned char *w,float *out,int t,int n,int k,int experts) {
    constexpr int MT=SWAP==3?16:(SWAP==2?32:64);
    static_assert(SWAP<2||L==0);
    __shared__ float sa[32][MT],sb[32][64];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    constexpr int RM=SWAP==3?2:(SWAP?4:8),CN=SWAP==1?8:4;
    const int ar=SWAP==1?warp*16+(lane/8)*4:(warp/2)*(MT/2)+(lane/8)*RM;
    const int bc=SWAP==1?(lane%8)*8:(warp%2)*32+(lane%8)*4;
    const int mt=(t+MT-1)/MT,ng=n/64,groups=k/32;
    for(int job=blockIdx.x;job<experts*mt*ng;job+=gridDim.x) {
        const int e=job/(mt*ng),tm=(job/ng)%mt,col=job%ng;
        uint4 pa[2];float4 pf[4];uint32_t pb[2];float pd[2];
        auto fetch=[&](int g) {
            #pragma unroll
            for(int l=0;l<2;++l) {
                const int at=tid+128*l;
                if(at<MT*4) {
                if constexpr(L==0) {
                    const int row=tm*MT+at/4;
                    pa[l]=row<t?*reinterpret_cast<const uint4*>(reinterpret_cast<const half*>(ap)+(size_t(e)*t+row)*k+g*32+8*(at%4)):uint4{};
                } else if constexpr(L==3) {
                    const size_t ai=((size_t(e)*mt+tm)*groups+g)*2048+at*4;
                    const uint2 lo=*reinterpret_cast<const uint2*>(reinterpret_cast<const half*>(ap)+ai);
                    const uint2 hi=*reinterpret_cast<const uint2*>(reinterpret_cast<const half*>(ap)+ai+1024);
                    pa[l]={lo.x,lo.y,hi.x,hi.y};
                } else {
                    const size_t ai=((size_t(e)*mt+tm)*groups+g)*2048+at*8;
                    if constexpr(L==1)pa[l]=*reinterpret_cast<const uint4*>(reinterpret_cast<const half*>(ap)+ai);
                    else {
                        pf[2*l]=*reinterpret_cast<const float4*>(reinterpret_cast<const float*>(ap)+ai);
                        pf[2*l+1]=*reinterpret_cast<const float4*>(reinterpret_cast<const float*>(ap)+ai+4);
                    }
                }
                }
                const int row=B?at%64:at/4,chunk=B?at/64:at%4;
                const size_t wb=((size_t(e)*ng+col)*groups+g)*1152;
                pd[l]=__half2float(reinterpret_cast<const half*>(w+wb)[row]);
                pb[l]=*reinterpret_cast<const uint32_t*>(w+wb+128+(B?(chunk*64+row)*4:row*16+chunk*4));
            }
        };
        auto stage=[&]() {
            #pragma unroll
            for(int l=0;l<2;++l) {
                const int at=tid+128*l;
                if(at<MT*4) {
                if constexpr(L==0) {
                    const half *v=reinterpret_cast<const half*>(&pa[l]);
                    #pragma unroll
                    for(int i=0;i<8;++i)sa[8*(at%4)+i][at/4]=__half2float(v[i]);
                } else {
                    float av[8];
                    if constexpr(L==1||L==3) {
                        const half *v=reinterpret_cast<const half*>(&pa[l]);
                        #pragma unroll
                        for(int i=0;i<8;++i)av[i]=__half2float(v[i]);
                    } else {
                        *reinterpret_cast<float4*>(av)=pf[2*l];
                        *reinterpret_cast<float4*>(av+4)=pf[2*l+1];
                    }
                    if constexpr(L==3) {
                        *reinterpret_cast<float4*>(&sa[at/16][4*(at%16)])=*reinterpret_cast<const float4*>(av);
                        *reinterpret_cast<float4*>(&sa[at/16+16][4*(at%16)])=*reinterpret_cast<const float4*>(av+4);
                    } else {
                        *reinterpret_cast<float4*>(&sa[at/8][8*(at%8)])=*reinterpret_cast<const float4*>(av);
                        *reinterpret_cast<float4*>(&sa[at/8][8*(at%8)+4])=*reinterpret_cast<const float4*>(av+4);
                    }
                }
                }
                const int row=B?at%64:at/4,kk=4*(B?at/64:at%4);
                #pragma unroll
                for(int i=0;i<4;++i) {
                    sb[kk+i][row]=float(int((pb[l]>>(8*i))&15)-8)*pd[l];
                    sb[kk+i+16][row]=float(int((pb[l]>>(8*i+4))&15)-8)*pd[l];
                }
            }
        };
        float acc[RM][CN]={};fetch(0);stage();__syncthreads();
        for(int g=0;g<groups;++g) {
            if(g+1<groups)fetch(g+1);
            #pragma unroll 1
            for(int chunk=0;chunk<32/U;++chunk) {
                #pragma unroll
                for(int s=0;s<U;++s) {
                    const int kk=chunk*U+s;
                    float a[RM],b[CN];
                    #pragma unroll
                    for(int i=0;i<RM;i+=((SWAP==1||SWAP==2)?4:2)) {
                        if constexpr(SWAP==1||SWAP==2)*reinterpret_cast<float4*>(a+i)=*reinterpret_cast<const float4*>(&sa[kk][ar+i]);
                        else *reinterpret_cast<float2*>(a+i)=*reinterpret_cast<const float2*>(&sa[kk][ar+i]);
                    }
                    #pragma unroll
                    for(int j=0;j<CN;j+=4)*reinterpret_cast<float4*>(b+j)=*reinterpret_cast<const float4*>(&sb[kk][bc+j]);
                    #pragma unroll
                    for(int i=0;i<RM;++i) {
                        #pragma unroll
                        for(int j=0;j<CN;++j)acc[i][j]=fmaf(a[i],b[j],acc[i][j]);
                    }
                }
            }
            __syncthreads();if(g+1<groups)stage();__syncthreads();
        }
        #pragma unroll
        for(int i=0;i<RM;++i) {
            const int row=tm*MT+ar+i;
            #pragma unroll
            for(int j=0;j<CN;++j)if(row<t)out[(size_t(e)*t+row)*n+col*64+bc+j]=acc[i][j];
        }
        __syncthreads();
    }
}

inline void launch_delivery(int code,const half *row,const half *tile16,const float *tile32,
                            const unsigned char *w,const unsigned char *wt,float *out,
                            int t,int n,int k,int e,int grid) {
    #define LD(L,B,U) if(code==L*100+B*10+U){delivery_lead<L,B,U><<<grid,128>>>(L==0?static_cast<const void*>(row):(L==2?static_cast<const void*>(tile32):static_cast<const void*>(tile16)),B?wt:w,out,t,n,k,e);return;}
    #define LS(L,B) LD(L,B,8) LD(L,B,16) LD(L,B,32)
    LS(0,0) LS(0,1) LS(1,0) LS(1,1) LS(2,0) LS(2,1) LS(3,0) LS(3,1)
    LD(0,1,4) LD(1,1,4) LD(2,1,4) LD(3,1,4)
    #define ALT(L,U) if(code==1000+L*100+10+U){delivery_lead<L,1,U,1><<<grid,128>>>(L==0?static_cast<const void*>(row):(L==2?static_cast<const void*>(tile32):static_cast<const void*>(tile16)),wt,out,t,n,k,e);return;}
    ALT(0,4) ALT(0,8) ALT(1,4) ALT(1,8) ALT(2,4) ALT(2,8) ALT(3,4) ALT(3,8)
    #undef ALT
    #define SMALL(U) if(code==2010+U){delivery_lead<0,1,U,2><<<grid,128>>>(row,wt,out,t,n,k,e);return;}
    SMALL(4) SMALL(8) SMALL(16) SMALL(32)
    #undef SMALL
    #define TINY(U) if(code==3010+U){delivery_lead<0,1,U,3><<<grid,128>>>(row,wt,out,t,n,k,e);return;}
    TINY(8) TINY(16) TINY(32)
    #undef TINY
    #undef LS
    #undef LD
    throw std::runtime_error("unknown delivery code");
}
