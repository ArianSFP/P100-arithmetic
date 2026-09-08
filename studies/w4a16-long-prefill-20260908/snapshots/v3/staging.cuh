#pragma once
// Temporary dequantization is charged on EVERY prep=1 invocation, not cached
// across timed pipelines. Output [expert][N64tile][K32group][k32][n64].
__global__ void stage_q4_weights(const unsigned char *src,half *dst,size_t count){
    for(size_t x=size_t(blockIdx.x)*blockDim.x+threadIdx.x;x<count;x+=size_t(gridDim.x)*blockDim.x){
        const size_t tile=x/2048;const int row=x%64,kk=(x%2048)/64;
        const unsigned char *p=src+tile*1152;
        const float d=__half2float(reinterpret_cast<const half*>(p)[row]);
        const int q=int((p[128+row*16+kk%16]>>(kk>=16?4:0))&15)-8;
        dst[x]=__float2half_rn(__fmul_rn(d,float(q)));
    }
}

template<int U>
__global__ __launch_bounds__(128)
void staged_f32(const float *a,const half *w,float *out,int t,int n,int k,int experts){
    __shared__ float sa[32][32],sb[32][64];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int ar=(warp/2)*16+(lane/8)*4,bc=(warp%2)*32+(lane%8)*4;
    const int mt=(t+31)/32,ng=n/64,groups=k/32;
    for(int job=blockIdx.x;job<experts*mt*ng;job+=gridDim.x){
        const int e=job/(mt*ng),tm=(job/ng)%mt,col=job%ng;
        float4 pa[2];uint4 pw[2];
        auto fetch=[&](int g){
            const int row=tm*32+tid/4;
            const size_t ai=(size_t(e)*t+row)*k+g*32+(tid%4)*8;
            pa[0]=row<t?*reinterpret_cast<const float4*>(a+ai):float4{};
            pa[1]=row<t?*reinterpret_cast<const float4*>(a+ai+4):float4{};
            #pragma unroll
            for(int l=0;l<2;++l)pw[l]=*reinterpret_cast<const uint4*>(w+((size_t(e)*ng+col)*groups+g)*2048+(tid+l*128)*8);
        };
        auto stage=[&](){
            const float *av=reinterpret_cast<const float*>(pa);
            #pragma unroll
            for(int i=0;i<8;++i)sa[(tid%4)*8+i][tid/4]=__half2float(__float2half_rn(av[i]));
            #pragma unroll
            for(int l=0;l<2;++l){
                const int at=tid+l*128;const half *v=reinterpret_cast<const half*>(&pw[l]);float b[8];
                #pragma unroll
                for(int i=0;i<8;++i)b[i]=__half2float(v[i]);
                *reinterpret_cast<float4*>(&sb[at/8][(at%8)*8])=*reinterpret_cast<float4*>(b);
                *reinterpret_cast<float4*>(&sb[at/8][(at%8)*8+4])=*reinterpret_cast<float4*>(b+4);
            }
        };
        float acc[4][4]={};fetch(0);stage();__syncthreads();
        for(int g=0;g<groups;++g){
            if(g+1<groups)fetch(g+1);
            #pragma unroll 1
            for(int chunk=0;chunk<32/U;++chunk){
                #pragma unroll
                for(int s=0;s<U;++s){
                    const int kk=chunk*U+s;float av[4],bv[4];
                    *reinterpret_cast<float4*>(av)=*reinterpret_cast<const float4*>(&sa[kk][ar]);
                    *reinterpret_cast<float4*>(bv)=*reinterpret_cast<const float4*>(&sb[kk][bc]);
                    #pragma unroll
                    for(int i=0;i<4;++i){
                        #pragma unroll
                        for(int j=0;j<4;++j)acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
                    }
                }
            }
            __syncthreads();
            if(g+1<groups)stage();
            __syncthreads();
        }
        #pragma unroll
        for(int i=0;i<4;++i)if(tm*32+ar+i<t){
            #pragma unroll
            for(int j=0;j<4;++j)out[(size_t(e)*t+tm*32+ar+i)*n+col*64+bc+j]=acc[i][j];
        }
        __syncthreads();
    }
}
