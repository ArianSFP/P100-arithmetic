#include "r3-baseline.cuh"

__device__ __forceinline__ half2 pair_bits(unsigned u) { half2 h;memcpy(&h,&u,4);return h; }
__device__ __forceinline__ half2 nibble_pair(unsigned w) {
    unsigned u=w^0x88;
    unsigned h=0x64006400u | (u&15) | ((u&240)<<12);
    return __hsub2(pair_bits(h),pair_bits(0x64086408u));
}
__device__ __forceinline__ half2 separated_pair(unsigned w) {
 unsigned magic;asm("lop3.b32 %0, %1, %2, %3, 0x6a;" : "=r"(magic) : "r"(w),"r"(0x000f000fu),"r"(0x64086408u));
 return __hsub2(pair_bits(magic),pair_bits(0x64086408u));
}
// Mode 0: FP32 sequential dot; 1: half pair dot, FP32 scaled group sums;
// 2: half pair dot and scaled group sums. All have final FP32 CTA reduction.
template<int Mode,int R,int B,int NW,int S>
__global__ void decode(const Q4Block *native,const uint32_t *__restrict__ words,const uint16_t *__restrict__ scales,const uint16_t *__restrict__ a,float *__restrict__ out,unsigned rows,unsigned groups,unsigned batches) {
    extern __shared__ float sm[];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,tile=blockIdx.x*R,batch0=blockIdx.y*B;
    for(unsigned stripe=warp;stripe<S;stripe+=NW){
        float acc[R][B]={};half2 total[R][B];
        #pragma unroll
        for(int r=0;r<R;r++)for(int b=0;b<B;b++)total[r][b]=pair_bits(0);
        for(unsigned g=stripe;g<groups;g+=S){
            float dot[R][B]={};half2 hd[R][B];uint32_t wa[R][4],ap[B];
            #pragma unroll
            for(int r=0;r<R;r++)for(int b=0;b<B;b++)hd[r][b]=pair_bits(0);
            #pragma unroll
            for(int b=0;b<B;b++){
                if constexpr(Mode<3)ap[b]=batch0+b<batches ? reinterpret_cast<const unsigned*>(a)[((batch0+b)*groups+g)*16+(lane&15)] : 0;
                else{unsigned i=((batch0+b)*groups+g)*32+(lane&15)/4*8+(lane&3);ap[b]=batch0+b<batches ? unsigned(a[i])|(unsigned(a[i+4])<<16) : 0;}
            }
            #pragma unroll
            for(int r=0;r<R;r++)for(int c=0;c<4;c++)wa[r][c]=(tile+r)*32+lane<rows ? words[((tile+r)*groups+g)*128+c*32+lane]:0;
            #pragma unroll
            for(int c=0;c<4;c++){
                #pragma unroll
                for(int j=0;j<4;j++){
                    half2 av[B];
                    #pragma unroll
                    for(int b=0;b<B;b++)av[b]=pair_bits(__shfl_sync(0xffffffff,ap[b],c*4+j));
                    #pragma unroll
                    for(int r=0;r<R;r++){
                        half2 q=Mode<3 ? nibble_pair(wa[r][c]>>(j*8)) : separated_pair(wa[r][c]>>(j*4));
                        #pragma unroll
                        for(int b=0;b<B;b++){
                            if constexpr(Mode%3==0){
                                dot[r][b]=__fmaf_rn(__low2float(q),__low2float(av[b]),dot[r][b]);
                                dot[r][b]=__fmaf_rn(__high2float(q),__high2float(av[b]),dot[r][b]);
                            }else hd[r][b]=__hfma2(q,av[b],hd[r][b]);
                        }
                    }
                }
            }
            #pragma unroll
            for(int r=0;r<R;r++){
                half d=__ushort_as_half((tile+r)*32+lane<rows ? scales[((tile+r)*groups+g)*32+lane]:0);
                #pragma unroll
                for(int b=0;b<B;b++){
                    if constexpr(Mode%3==2)total[r][b]=__hfma2(hd[r][b],__halves2half2(d,d),total[r][b]);
                    else {float v=Mode%3==0 ? dot[r][b] : __low2float(hd[r][b])+__high2float(hd[r][b]);acc[r][b]=__fmaf_rn(v,__half2float(d),acc[r][b]);}
                }
            }
        }
        #pragma unroll
        for(int r=0;r<R;r++)for(int b=0;b<B;b++)sm[((b*S+stripe)*R+r)*32+lane]=Mode%3==2 ? __low2float(total[r][b])+__high2float(total[r][b]):acc[r][b];
    }
    __syncthreads();
    #pragma unroll
    for(int off=S/2;off;off/=2){
        for(unsigned i=threadIdx.x;i<B*off*R*32;i+=NW*32){unsigned row=i%(R*32),s=(i/(R*32))%off,b=i/(off*R*32);unsigned ix=(b*S+s)*R*32+row;sm[ix]=__fadd_rn(sm[ix],sm[ix+off*R*32]);}
        __syncthreads();
    }
    for(unsigned i=threadIdx.x;i<B*R*32;i+=NW*32){unsigned row=tile*32+i%(R*32),b=i/(R*32);if(row<rows&&batch0+b<batches)out[(batch0+b)*rows+row]=sm[b*S*R*32+i%(R*32)];}
}
__global__ void reduce32(const float *p,float *out,unsigned m){unsigned row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=m)return;float v[32];for(int s=0;s<32;s++)v[s]=p[(blockIdx.y*32+s)*m+row];for(int o=16;o;o/=2)for(int s=0;s<o;s++)v[s]+=v[s+o];out[blockIdx.y*m+row]=v[0];}

__global__ void prepare(const float *raw,uint16_t *a,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]=__half_as_ushort(__float2half_rn(raw[i]));}
