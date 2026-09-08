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
template<int Mode,int R,int B,int NW,int S,bool Fuse=true>
__global__ void decode(const Q4Block *native,const uint32_t *__restrict__ words,const uint16_t *__restrict__ scales,const uint16_t *__restrict__ a,float *__restrict__ out,unsigned rows,unsigned groups,unsigned batches) {
    extern __shared__ float sm[];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,tile=(Fuse?blockIdx.x:blockIdx.x*NW+warp)*R,batch0=blockIdx.y*B;
    for(unsigned stripe=Fuse?warp:blockIdx.z;stripe<S;stripe+=Fuse?NW:S){
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
        for(int r=0;r<R;r++)for(int b=0;b<B;b++){
            float v=Mode%3==2 ? __low2float(total[r][b])+__high2float(total[r][b]):acc[r][b];
            if constexpr(Fuse)sm[((b*S+stripe)*R+r)*32+lane]=v;
            else if((tile+r)*32+lane<rows&&batch0+b<batches)out[((batch0+b)*S+stripe)*rows+(tile+r)*32+lane]=v;
        }
    }
    if constexpr(!Fuse)return;
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

__global__ void r3_batch8(ARGS){body<1,false,8,4,false>(PASS);}
// Split each output row over L lanes, reducing the per-thread dot length.
template<int Mode,int L,int B,int NW,int S>
__global__ void decode_lanes(ARGS){
 extern __shared__ float sm[];
 constexpr int ROWS=32/L;
 unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*ROWS+lane%ROWS,part=lane/ROWS,b0=blockIdx.y*B;
 for(unsigned stripe=warp;stripe<S;stripe+=NW){
  float acc[B]={};half2 total[B];
  #pragma unroll
  for(int b=0;b<B;b++)total[b]=pair_bits(0);
  for(unsigned g=stripe;g<groups;g+=S){
   float dot[B]={};half2 hd[B];unsigned ap[B],ww[4/L];
   #pragma unroll
   for(int b=0;b<B;b++){
    hd[b]=pair_bits(0);unsigned i=((b0+b)*groups+g)*32+(lane&15)/4*8+(lane&3);
    ap[b]=b0+b<batches ? unsigned(a[i])|(unsigned(a[i+4])<<16):0;
   }
   #pragma unroll
   for(int c=0;c<4/L;c++)ww[c]=row<rows?words[((row/32)*groups+g)*128+(part*(4/L)+c)*32+row%32]:0;
   #pragma unroll
   for(int c=0;c<4/L;c++){
    #pragma unroll
    for(int j=0;j<4;j++){
     half2 q=separated_pair(ww[c]>>(j*4));
     #pragma unroll
     for(int b=0;b<B;b++){
      half2 av=pair_bits(__shfl_sync(0xffffffff,ap[b],(part*(4/L)+c)*4+j));
      if constexpr(Mode%3==0){dot[b]=__fmaf_rn(__low2float(q),__low2float(av),dot[b]);dot[b]=__fmaf_rn(__high2float(q),__high2float(av),dot[b]);}
      else hd[b]=__hfma2(q,av,hd[b]);
     }
    }
   }
   half d=__ushort_as_half(row<rows?scales[((row/32)*groups+g)*32+row%32]:0);
   #pragma unroll
   for(int b=0;b<B;b++){
    if constexpr(Mode%3==2)total[b]=__hfma2(hd[b],__halves2half2(d,d),total[b]);
    else{float v=Mode%3==0?dot[b]:__low2float(hd[b])+__high2float(hd[b]);acc[b]=__fmaf_rn(v,__half2float(d),acc[b]);}
   }
  }
  #pragma unroll
  for(int b=0;b<B;b++)sm[(b*S+stripe)*32+lane]=Mode%3==2?__low2float(total[b])+__high2float(total[b]):acc[b];
 }
 __syncthreads();
 #pragma unroll
 for(int off=S/2;off;off/=2){for(unsigned i=threadIdx.x;i<B*off*32;i+=NW*32){unsigned lane0=i%32,s=(i/32)%off,b=i/(off*32),ix=(b*S+s)*32+lane0;sm[ix]=__fadd_rn(sm[ix],sm[ix+off*32]);}__syncthreads();}
 for(unsigned i=threadIdx.x;i<B*ROWS;i+=NW*32){unsigned b=i/ROWS,row0=i%ROWS;float v[L];
  #pragma unroll
  for(int p=0;p<L;p++)v[p]=sm[b*S*32+p*ROWS+row0];
  #pragma unroll
  for(int off=L/2;off;off/=2)for(int p=0;p<off;p++)v[p]=__fadd_rn(v[p],v[p+off]);
  if(b0+b<batches&&blockIdx.x*ROWS+row0<rows)out[(b0+b)*rows+blockIdx.x*ROWS+row0]=v[0];
 }
}

// Coarse K partitions preserve the parent stripe reduction tree.
template<int Mode,int R,int B,int NW,int S,int P>
__global__ void decode_parts(const Q4Block *native,const uint32_t *__restrict__ words,const uint16_t *__restrict__ scales,const uint16_t *__restrict__ a,float *__restrict__ out,unsigned rows,unsigned groups,unsigned batches) {
    extern __shared__ float sm[];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,tile=blockIdx.x*R,batch0=blockIdx.y*B;
    for(unsigned stripe=warp;stripe<S/P;stripe+=NW){
        float acc[R][B]={};half2 total[R][B];
        #pragma unroll
        for(int r=0;r<R;r++)for(int b=0;b<B;b++)total[r][b]=pair_bits(0);
        for(unsigned g=stripe*P+blockIdx.z;g<groups;g+=S){
            float dot[R][B]={};half2 hd[R][B];uint32_t wa[R][4],ap[B];float lane_a[B];
            #pragma unroll
            for(int r=0;r<R;r++)for(int b=0;b<B;b++)hd[r][b]=pair_bits(0);
            #pragma unroll
            for(int b=0;b<B;b++){
                if constexpr(Mode==9)lane_a[b]=batch0+b<batches ? widen(a[((batch0+b)*groups+g)*32+lane]):0;
                else if constexpr(Mode<3)ap[b]=batch0+b<batches ? reinterpret_cast<const unsigned*>(a)[((batch0+b)*groups+g)*16+(lane&15)] : 0;
                else{unsigned i=((batch0+b)*groups+g)*32+(lane&15)/4*8+(lane&3);ap[b]=batch0+b<batches ? unsigned(a[i])|(unsigned(a[i+4])<<16) : 0;}
            }
            #pragma unroll
            for(int r=0;r<R;r++)for(int c=0;c<4;c++)wa[r][c]=(tile+r)*32+lane<rows ? words[((tile+r)*groups+g)*128+c*32+lane]:0;
            if constexpr(Mode==9){
                #pragma unroll
                for(int c=0;c<4;c++){
                    float av[B][8];
                    #pragma unroll
                    for(int b=0;b<B;b++)for(int j=0;j<8;j++)av[b][j]=__shfl_sync(0xffffffff,lane_a[b],c*8+j);
                    #pragma unroll
                    for(int j=0;j<8;j++)for(int r=0;r<R;r++){
                        float q=float(unpack_nibble(wa[r][c],j));
                        #pragma unroll
                        for(int b=0;b<B;b++)dot[r][b]=__fmaf_rn(q,av[b][j],dot[r][b]);
                    }
                }
            }else{
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
        for(int r=0;r<R;r++)for(int b=0;b<B;b++){
            float v=Mode%3==2 ? __low2float(total[r][b])+__high2float(total[r][b]):acc[r][b];
            sm[((b*(S/P)+stripe)*R+r)*32+lane]=v;
        }
    }
    __syncthreads();
    #pragma unroll
    for(int off=S/P/2;off;off/=2){
        for(unsigned i=threadIdx.x;i<B*off*R*32;i+=NW*32){unsigned row=i%(R*32),s=(i/(R*32))%off,b=i/(off*R*32);unsigned ix=(b*(S/P)+s)*R*32+row;sm[ix]=__fadd_rn(sm[ix],sm[ix+off*R*32]);}
        __syncthreads();
    }
    for(unsigned i=threadIdx.x;i<B*R*32;i+=NW*32){unsigned row=tile*32+i%(R*32),b=i/(R*32);if(row<rows&&batch0+b<batches)out[((batch0+b)*P+blockIdx.z)*rows+row]=sm[b*(S/P)*R*32+i%(R*32)];}
}

template<int P>__global__ void reduce_parts(const float *in,float *out,unsigned rows){
 unsigned row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;float v[P];
 #pragma unroll
 for(int p=0;p<P;p++)v[p]=in[(blockIdx.y*P+p)*rows+row];
 #pragma unroll
 for(int off=P/2;off;off/=2)for(int p=0;p<off;p++)v[p]=__fadd_rn(v[p],v[p+off]);
 out[blockIdx.y*rows+row]=v[0];
}
