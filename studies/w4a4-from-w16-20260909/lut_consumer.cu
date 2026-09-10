#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)

__device__ __forceinline__ uint32_t half2_bits(half2 x) {
    __half2_raw r=x;return uint32_t(r.x)|(uint32_t(r.y)<<16);
}

// Optimistic resident control: exactly the current 16-row/4-column ownership,
// 32 KiB shared footprint and 2-CTA/SM geometry, but no nibble decode or G32
// scale finish. Each loop represents 16*4*32 = 2048 useful INT4 MACs/thread.
template<int ROWS>
__global__ __launch_bounds__(256,2)
void resident_half2(const uint32_t *seed,uint64_t *out,int groups) {
    __shared__ uint32_t table[256][32];
    const int tid=threadIdx.x,lane=tid&31,id=blockIdx.x*blockDim.x+tid;
    for(int i=tid;i<256*32;i+=blockDim.x)table[i/32][i&31]=seed[(i+id)&8191];
    __syncthreads();
    const uint32_t v=table[(seed[id&8191]>>24)&255][lane];
    const float x=float(int(v&15)-8),y=float(int((v>>4)&15)-8);
    const half2 a=__floats2half2_rn(x,y),b=__floats2half2_rn(y,x);
    half2 acc[ROWS][4];
    #pragma unroll
    for(int r=0;r<ROWS;++r)for(int c=0;c<4;++c)acc[r][c]=__float2half2_rn(0);
    #pragma unroll 1
    for(int g=0;g<groups;++g) {
        #pragma unroll
        for(int k=0;k<16;++k) {
            #pragma unroll
            for(int r=0;r<ROWS;++r) {
                #pragma unroll
                for(int c=0;c<4;++c)acc[r][c]=__hfma2((k^c)&1?a:b,(k^r)&1?b:a,acc[r][c]);
            }
        }
    }
    uint32_t lo=0,hi=0;
    #pragma unroll
    for(int r=0;r<ROWS;++r)for(int c=0;c<4;++c){uint32_t z=half2_bits(acc[r][c]);if(r<ROWS/2)lo^=z;else hi^=z;}
    out[id]=uint64_t(lo)|(uint64_t(hi)<<32);
}

// Level-3 optimistic lower bound. table[mask][lane] is bank-owned because its
// bank index is lane. The table is prebuilt and resident; construction, code
// loading, field widening, corrections and G32 scaling are intentionally absent.
// Each loop performs 256 LDS + 256 packed IADD for the same 2048 useful MACs.
template<int ROWS>
__global__ __launch_bounds__(256,2)
void resident_lut_divergent(const uint32_t *seed,uint64_t *out,int groups) {
    __shared__ uint32_t table[256][32];
    const int tid=threadIdx.x,lane=tid&31,id=blockIdx.x*blockDim.x+tid;
    for(int i=tid;i<256*32;i+=blockDim.x)table[i/32][i&31]=seed[(i+id)&8191];
    __syncthreads();
    volatile uint32_t (*lookup)[32]=table;
    uint32_t acc[ROWS][4];
    #pragma unroll
    for(int r=0;r<ROWS;++r)for(int p=0;p<4;++p)acc[r][p]=uint32_t(r*17+p*13+lane);
    uint32_t state=seed[id&8191];
    #pragma unroll 1
    for(int g=0;g<groups;++g) {
        state=state*1664525u+1013904223u;
        #pragma unroll
        for(int r=0;r<ROWS;++r) {
            #pragma unroll
            for(int slice=0;slice<4;++slice) {
                #pragma unroll
                for(int plane=0;plane<4;++plane) {
                    const unsigned mask=(state+r*29+slice*53+plane*71)&255;
                    acc[r][plane]+=lookup[mask][lane];
                }
            }
        }
    }
    uint32_t lo=state,hi=0;
    #pragma unroll
    for(int r=0;r<ROWS;++r)for(int q=0;q<4;++q){if(r<ROWS/2)lo^=acc[r][q];else hi^=acc[r][q];}
    out[id]=uint64_t(lo)|(uint64_t(hi)<<32);
}

// Primary optimistic LUT issue test. In the proposed GEMM, every lane in a
// warp consumes the same token's activation mask while lane selects one of 32
// distinct four-row tables. Keeping the mask warp-uniform models that access.
// Slice-major order models consuming a complete mu8 table before rebuilding it.
template<int ROWS,int MIN_BLOCKS>
__global__ __launch_bounds__(256,MIN_BLOCKS)
void resident_lut_uniform(const uint32_t *seed,uint64_t *out,int groups) {
    __shared__ uint32_t table[256][32];
    const int tid=threadIdx.x,lane=tid&31,id=blockIdx.x*blockDim.x+tid;
    for(int i=tid;i<256*32;i+=blockDim.x)table[i/32][i&31]=seed[(i+id)&8191];
    __syncthreads();
    volatile uint32_t (*lookup)[32]=table;
    uint32_t acc[ROWS][4];
    #pragma unroll
    for(int r=0;r<ROWS;++r)for(int p=0;p<4;++p)acc[r][p]=uint32_t(r*17+p*13+lane);
    uint32_t state=seed[(blockIdx.x*8+(tid>>5))&8191];
    #pragma unroll 1
    for(int g=0;g<groups;++g) {
        state=state*1664525u+1013904223u;
        #pragma unroll
        for(int slice=0;slice<4;++slice) {
            #pragma unroll
            for(int r=0;r<ROWS;++r) {
                #pragma unroll
                for(int plane=0;plane<4;++plane) {
                    const unsigned mask=(state+r*29+slice*53+plane*71)&255;
                    acc[r][plane]+=lookup[mask][lane];
                }
            }
        }
    }
    uint32_t lo=state,hi=0;
    #pragma unroll
    for(int r=0;r<ROWS;++r)for(int q=0;q<4;++q){if(r<ROWS/2)lo^=acc[r][q];else hi^=acc[r][q];}
    out[id]=uint64_t(lo)|(uint64_t(hi)<<32);
}

int main(int argc,char **argv) try {
    if(argc!=4||std::string(argv[1])!="--gpu-approved"||std::string(argv[2])!="1")
        throw std::runtime_error("usage: --gpu-approved 1 groups");
    const int groups=atoi(argv[3]);if(groups<8||groups>4096)throw std::runtime_error("group bounds");
    int devices=0;CU(cudaGetDeviceCount(&devices));if(devices!=1)throw std::runtime_error("one visible GPU required");
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor!=0)throw std::runtime_error("SM60 required");
    const int grid=prop.multiProcessorCount*2,block=256,count=grid*block;
    std::vector<uint32_t> input(8192);uint32_t s=3911;
    for(auto &v:input){s^=s<<13;s^=s>>17;s^=s<<5;v=s;}
    uint32_t *din=nullptr;uint64_t *dout=nullptr;CU(cudaMalloc(&din,input.size()*sizeof(*din)));CU(cudaMalloc(&dout,count*sizeof(*dout)));
    CU(cudaMemcpy(din,input.data(),input.size()*sizeof(*din),cudaMemcpyHostToDevice));
    auto launch=[&](int mode){
        if(mode==0)resident_half2<16><<<grid,block>>>(din,dout,groups);
        else if(mode==1)resident_lut_uniform<16,1><<<grid,block>>>(din,dout,groups);
        else if(mode==2)resident_half2<8><<<grid,block>>>(din,dout,groups);
        else if(mode==3)resident_lut_uniform<8,2><<<grid,block>>>(din,dout,groups);
        else if(mode==4)resident_lut_divergent<8><<<grid,block>>>(din,dout,groups);
        else if(mode==5)resident_half2<4><<<grid,block>>>(din,dout,groups);
        else resident_lut_uniform<4,2><<<grid,block>>>(din,dout,groups);
        CU(cudaGetLastError());
    };
    auto warm_end=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    do{for(int mode=0;mode<7;++mode)launch(mode);CU(cudaDeviceSynchronize());}while(std::chrono::steady_clock::now()<warm_end);
    cudaEvent_t begin,end;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));
    constexpr int rounds=9,repeats=3;const char *names[]={"resident-half2-r16","prebuilt-uniform-lut-r16-1cta","resident-half2-r8","prebuilt-uniform-lut-r8","prebuilt-divergent-lut-r8","resident-half2-r4","prebuilt-uniform-lut-r4"};
    const int useful[]={2048,2048,1024,1024,1024,512,512};
    for(int r=0;r<rounds;++r)for(int j=0;j<7;++j){int mode=(j+2*r)%7;CU(cudaEventRecord(begin));for(int q=0;q<repeats;++q)launch(mode);CU(cudaEventRecord(end));CU(cudaEventSynchronize(end));float ms=0;CU(cudaEventElapsedTime(&ms,begin,end));double us=ms*1000.0/repeats;double macs=double(count)*groups*useful[mode];printf("TIME mode=%s rep=%d us=%.9g useful_tmac_s=%.9g\n",names[mode],r,us,macs/(us*1e6));}
    std::vector<uint64_t> output(count);CU(cudaMemcpy(output.data(),dout,count*sizeof(*dout),cudaMemcpyDeviceToHost));uint64_t checksum=0;for(auto v:output)checksum^=v;
    printf("META grid=%d block=%d threads=%d groups=%d seed=3911 checksum=%llu prebuilt_table=1 omitted_finish=1\n",grid,block,count,groups,(unsigned long long)checksum);
    CU(cudaFree(dout));CU(cudaFree(din));puts("PASS");return 0;
} catch(const std::exception &e){fprintf(stderr,"STOP: %s\n",e.what());return 1;}
