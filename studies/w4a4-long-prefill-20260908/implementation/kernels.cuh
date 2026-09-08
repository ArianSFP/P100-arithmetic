#pragma once
// All math after exact integer group dots is binary32. No half accumulation.
struct PlaneGroup { unsigned p[4]; float d; };
struct PlaneJob { const float *a; const unsigned char *w; PlaneGroup *ap,*wp; float *out; int m; };
struct PlaneTile { int e,m; };
__global__ void quantize_planes(const PlaneJob* jobs,int ne,int k) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    for(int e=blockIdx.y;e<ne;e+=gridDim.y){
        auto j=jobs[e];
        for(int g=blockIdx.x*(blockDim.x/32)+warp;g<j.m*(k/32);g+=gridDim.x*(blockDim.x/32)){
            float v=__half2float(__float2half_rn(j.a[size_t(g)*32+lane]));
            float a=fabsf(v);
            for(int s=16;s;s/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,s));
            float d=__fdiv_rn(a,7.f);
            int q=d==0?0:max(-7,min(7,__float2int_rn(__fdiv_rn(v,d))));
            unsigned p[4];
            #pragma unroll
            for(int b=0;b<4;++b)p[b]=__ballot_sync(0xffffffff,(q>>b)&1);
            if(lane==0){for(int b=0;b<4;++b)j.ap[g].p[b]=p[b];j.ap[g].d=d;}
        }
    }
}
__global__ void weight_planes(const PlaneJob* jobs,int ne,int n,int k) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    for(int e=blockIdx.y;e<ne;e+=gridDim.y){
        auto j=jobs[e];int ng=k/32;
        for(int bg=blockIdx.x*(blockDim.x/32)+warp;bg<n*ng;bg+=gridDim.x*(blockDim.x/32)){
            int r=bg/ng,g=bg%ng;
            const unsigned char *s=j.w+(size_t(r/64)*ng+g)*1152;
            int q=((s[128+(r%64)*16+lane%16]>>(lane<16?0:4))&15)-8;
            unsigned p[4];
            #pragma unroll
            for(int b=0;b<4;++b)p[b]=__ballot_sync(0xffffffff,(q>>b)&1);
            if(lane==0){for(int b=0;b<4;++b)j.wp[bg].p[b]=p[b];j.wp[bg].d=__half2float(*(const half*)(s+2*(r%64)));}
        }
    }
}
template<int RM,bool INTEGER=true>
__global__ __launch_bounds__(128) void plane_gemm(const PlaneJob* jobs,const PlaneTile* tiles,int nt,int n,int k){
    constexpr int MT=8*RM,NT=64;
    __shared__ unsigned ap[4][MT],wp[4][NT];
    __shared__ float ad[MT],wd[NT];
    int tid=threadIdx.x,m0=(tid/16)*RM,n0=(tid%16)*4;
    for(int task=blockIdx.x;task<nt*(n/NT);task+=gridDim.x){
        auto tile=tiles[task/(n/NT)];auto j=jobs[tile.e];int no=(task%(n/NT))*NT;
        float acc[RM][4]={};
        for(int g=0;g<k/32;++g){
            if(tid<MT){PlaneGroup v{};if(tile.m+tid<j.m)v=j.ap[(tile.m+tid)*(k/32)+g];
                #pragma unroll
                for(int p=0;p<4;++p)ap[p][tid]=v.p[p];ad[tid]=v.d;
            }
            if(tid<NT){auto v=j.wp[(no+tid)*(k/32)+g];
                #pragma unroll
                for(int p=0;p<4;++p)wp[p][tid]=v.p[p];wd[tid]=v.d;
            }
            __syncthreads();
            unsigned av[RM][4],wv[4][4];float as[RM],ws[4];
            #pragma unroll
            for(int i=0;i<RM;++i){as[i]=ad[m0+i];for(int p=0;p<4;++p)av[i][p]=ap[p][m0+i];}
            #pragma unroll
            for(int l=0;l<4;++l){ws[l]=wd[n0+l];for(int p=0;p<4;++p)wv[l][p]=wp[p][n0+l];}
            #pragma unroll
            for(int i=0;i<RM;++i){
                #pragma unroll
                for(int l=0;l<4;++l){
                    float dot;
                    if constexpr(INTEGER){int sum=0;
                        #pragma unroll
                        for(int p=0;p<4;++p){
                            #pragma unroll
                            for(int q=0;q<4;++q){constexpr int c[4]={1,2,4,-8};sum+=c[p]*c[q]*__popc(av[i][p]&wv[l][q]);}
                        }dot=float(sum);
                    }else{dot=0;
                        #pragma unroll
                        for(int b=0;b<32;++b){int a=0,w=0;
                            #pragma unroll
                            for(int p=0;p<4;++p){constexpr int c[4]={1,2,4,-8};a+=c[p]*((av[i][p]>>b)&1);w+=c[p]*((wv[l][p]>>b)&1);}
                            dot=__fmaf_rn(float(a),float(w),dot);
                        }
                    }
                    acc[i][l]=__fmaf_rn(dot,__fmul_rn(as[i],ws[l]),acc[i][l]);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(int i=0;i<RM;++i)if(tile.m+m0+i<j.m){
            #pragma unroll
            for(int l=0;l<4;++l)j.out[size_t(tile.m+m0+i)*n+no+n0+l]=acc[i][l];
        }
        __syncthreads();
    }
}
