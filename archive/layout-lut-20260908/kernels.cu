// Compilation-only study. No host launcher, CUDA initialization or GPU execution.
#include <cuda_runtime.h>
#include "common.hpp"

// M0=VMAD, M1=SAD, M2=register LUT, M3=64-byte shared LUT,
// M4=replicated shared LUT; M5=register LUT with int8-cast table construction.
// Four warps per CTA, 32*R rows/warp. grid.z splits K into scale-group stripes.
template<int W,int F,int M,int R,bool OLD>
__global__ void rows(const uint32_t *w,const uint32_t *a,const int16_t *as,
                     const float *ws,const float *asc,float *out,int nrows,int groups) {
    static_assert(M!=1 || F!=3,"SAD requires a byte-sign layout");
    constexpr int warps=4;
    constexpr int table_words=M==3 ? 16 : M==4 ? 16*32 : 1;
    __shared__ int tables[warps*table_words];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int base=(int(blockIdx.x)*warps+warp)*32*R+lane;
    int batch=blockIdx.y;
    float accum[R]={};
    // All lanes stay live, including masked tail rows, at every shuffle/sync.
    for(int g=blockIdx.z;g<groups;g+=gridDim.z) {
        uint32_t planes[R][W];
        int sums[R][W]={},integer_sum[R]={};
        #pragma unroll
        for(int r=0;r<R;++r) {
            #pragma unroll
            for(int p=0;p<W;++p) planes[r][p]=base+32*r<nrows ? w[weight_address(base+32*r,g,p,groups,W,OLD)] : 0;
        }
        #pragma unroll
        for(int q=0;q<8;++q) {
            uint32_t av=a[(batch*groups+g)*8+q];
            int entry=0;
            if constexpr(M==2) entry=table_entry(av,lane&15);
            if constexpr(M==3 || M==5) entry=table_entry_cast(av,lane&15);
            if constexpr(M==3) {
                if(lane<16) tables[warp*16+lane]=entry;
                __syncwarp();
            }
            if constexpr(M==4) {
                #pragma unroll
                for(int e=0;e<16;++e) tables[warp*512+32*e+lane]=table_entry_cast(av,e);
                __syncwarp();
            }
            #pragma unroll
            for(int r=0;r<R;++r) {
                if constexpr(M==0) integer_sum[r]=vmad4(decode4<W,F>(planes[r],q),av,integer_sum[r]);
                else {
                    #pragma unroll
                    for(int p=0;p<W;++p) {
                        if constexpr(M==1) {
                            uint32_t ref=sign_bytes(planes[r][p]<<(F==0 ? 7-q : q));
                            sums[r][p]=sad4(av^0x80808080u,ref,sums[r][p]);
                        } else {
                            unsigned idx=index4<F>(planes[r][p],q);
                            int value;
                            if constexpr(M==2 || M==5) value=__shfl_sync(0xffffffff,entry,idx);
                            if constexpr(M==3) value=tables[warp*16+idx];
                            if constexpr(M==4) value=tables[warp*512+32*idx+lane];
                            sums[r][p]+=value;
                        }
                    }
                }
            }
            if constexpr(M==3 || M==4) __syncwarp();
        }
        int asum=0;
        if constexpr(M==1 || F==2) asum=as[batch*groups+g];
        #pragma unroll
        for(int r=0;r<R;++r) {
            int dot;
            if constexpr(M==0) dot=F==2 ? -asum-integer_sum[r] : integer_sum[r];
            else if constexpr(M==1) dot=finish_sad<W,F>(planes[r],sums[r],asum);
            else dot=F==2 ? -asum-combine(sums[r]) : combine(sums[r]);
            float scale=base+32*r<nrows ? ws[scale_address(base+32*r,g,groups,OLD)]*asc[batch*groups+g] : 0;
            accum[r]=fmaf(float(dot),scale,accum[r]);
        }
    }
    #pragma unroll
    for(int r=0;r<R;++r) if(base+32*r<nrows) out[(batch*gridDim.z+blockIdx.z)*nrows+base+32*r]=accum[r];
}

extern "C" __global__ void reduce_splits(const float *partials,float *out,int rows,int splits) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows) return;
    int batch=blockIdx.y;
    float sum=0;
    for(int s=0;s<splits;++s) sum+=partials[(batch*splits+s)*rows+row];
    out[batch*rows+row]=sum;
}

extern "C" __global__ void prepare_asums(const uint32_t *a,int16_t *as,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    int sum=0;
    #pragma unroll
    for(int q=0;q<8;++q) sum=sad4(a[i*8+q]^0x80808080u,0,sum);
    as[i]=int16_t(sum-128*32);
}

// Single-group probes keep setup/loop costs from hiding the extraction counts.
template<int W,int F> __global__ void sad_group(const uint32_t *w,const uint32_t *a,const int16_t *as,int *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t planes[W]; int sums[W]={};
    #pragma unroll
    for(int p=0;p<W;++p) planes[p]=w[i*W+p];
    #pragma unroll
    for(int q=0;q<8;++q) {
        uint32_t av=a[i*8+q]^0x80808080u;
        #pragma unroll
        for(int p=0;p<W;++p) sums[p]=sad4(av,sign_bytes(planes[p]<<(F==0 ? 7-q : q)),sums[p]);
    }
    out[i]=finish_sad<W,F>(planes,sums,as[i]);
}

// Separate W-only experiment. FP32 table and accumulation, not an A8 substitute.
template<int W,int R> __global__ void rows_f32(const uint32_t *w,const float *a,const float *ws,float *out,int nrows,int groups) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int base=(int(blockIdx.x)*4+warp)*32*R+lane;
    float accum[R]={};
    for(int g=blockIdx.z;g<groups;g+=gridDim.z) {
        uint32_t planes[R][W]; float sums[R][W]={};
        #pragma unroll
        for(int r=0;r<R;++r) {
            #pragma unroll
            for(int p=0;p<W;++p) planes[r][p]=base+32*r<nrows ? w[weight_address(base+32*r,g,p,groups,W,false)] : 0;
        }
        #pragma unroll
        for(int q=0;q<8;++q) {
            float entry=0;
            #pragma unroll
            for(int j=0;j<4;++j) if(lane&(1<<j)) entry+=a[g*32+4*q+j];
            #pragma unroll
            for(int r=0;r<R;++r) {
                #pragma unroll
                for(int p=0;p<W;++p) sums[r][p]+=__shfl_sync(0xffffffff,entry,index4<3>(planes[r][p],q));
            }
        }
        #pragma unroll
        for(int r=0;r<R;++r) {
            float dot=0;
            #pragma unroll
            for(int p=0;p<W;++p) dot=fmaf(float(coefficient(p,W)),sums[r][p],dot);
            if(base+32*r<nrows) accum[r]=fmaf(ws[scale_address(base+32*r,g,groups,false)],dot,accum[r]);
        }
    }
    #pragma unroll
    for(int r=0;r<R;++r) if(base+32*r<nrows) out[blockIdx.z*nrows+base+32*r]=accum[r];
}

#define ROW(W,F,M,R,O) template __global__ void rows<W,F,M,R,O>(const uint32_t*,const uint32_t*,const int16_t*,const float*,const float*,float*,int,int);
#define REUSE(W,F,M) ROW(W,F,M,1,false) ROW(W,F,M,2,false) ROW(W,F,M,4,false)
#define WIDTH(W) \
 ROW(W,0,0,1,false) ROW(W,0,1,1,false) \
 REUSE(W,1,0) REUSE(W,1,1) \
 ROW(W,2,0,1,false) ROW(W,2,1,1,false) \
 REUSE(W,3,0) REUSE(W,3,2) REUSE(W,3,5) ROW(W,1,2,1,false) \
 REUSE(W,3,3) REUSE(W,3,4) \
 ROW(W,1,0,1,true) ROW(W,1,1,1,true) \
 ROW(W,3,0,1,true) ROW(W,3,2,1,true)
WIDTH(2)
WIDTH(4)
#define GROUP(W,F) template __global__ void sad_group<W,F>(const uint32_t*,const uint32_t*,const int16_t*,int*);
GROUP(2,0) GROUP(2,1) GROUP(2,2) GROUP(4,0) GROUP(4,1) GROUP(4,2)
#define FLOATROW(W,R) template __global__ void rows_f32<W,R>(const uint32_t*,const float*,const float*,float*,int,int);
FLOATROW(2,1) FLOATROW(2,2) FLOATROW(2,4) FLOATROW(4,1) FLOATROW(4,2) FLOATROW(4,4)
