#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{auto s=(x);if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(3);}}while(0)
#define BL(x) do{auto s=(x);if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"BLAS %d\n",s);exit(3);}}while(0)
struct Q8{half d;signed char q[32];};static_assert(sizeof(Q8)==34);
__global__ void prepw(const Q8*q,half*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)w[i]=__float2half_rn(__half2float(q[i/32].d)*q[i/32].q[i%32]);}
__global__ void prepx(const float*x,half*h,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)h[i]=__float2half_rn(x[i]);}
__global__ void finish(const half*h,float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=__half2float(h[i]);}
template<int V> __global__ void prepvec(const Q8*q,half*w,int n){
 int i=(blockIdx.x*blockDim.x+threadIdx.x)*V;if(i>=n)return;
 half2 d=__halves2half2(q[i/32].d,q[i/32].d);
 #pragma unroll
 for(int j=0;j<V;j+=2){half2 v=__floats2half2_rn(float(q[i/32].q[i%32+j]),float(q[i/32].q[i%32+j+1]));reinterpret_cast<half2*>(w)[(i+j)/2]=__hmul2(d,v);}
}

template<class T>__global__ void trans(const T*in,T*out,int rows,int cols){
 __shared__ T t[32][33];int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)if(x<cols&&y+j<rows)t[threadIdx.y+j][threadIdx.x]=in[size_t(y+j)*cols+x];
 __syncthreads();x=blockIdx.y*32+threadIdx.x;y=blockIdx.x*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)if(x<rows&&y+j<cols)out[size_t(y+j)*rows+x]=t[threadIdx.x][threadIdx.y+j];
}
__global__ void transwide(const half*in,float*out,int rows,int cols){
 __shared__ half t[32][33];int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)if(x<cols&&y+j<rows)t[threadIdx.y+j][threadIdx.x]=in[size_t(y+j)*cols+x];
 __syncthreads();x=blockIdx.y*32+threadIdx.x;y=blockIdx.x*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)if(x<rows&&y+j<cols)out[size_t(y+j)*rows+x]=__half2float(t[threadIdx.x][threadIdx.y+j]);
}

__global__ void prepwt(const Q8*q,half*out,int M,int K,int LD){
 __shared__ half t[32][33];int k=blockIdx.x*32+threadIdx.x,m=blockIdx.y*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)if(k<K&&m+j<M){int i=(m+j)*K+k;t[threadIdx.y+j][threadIdx.x]=__float2half_rn(__half2float(q[i/32].d)*q[i/32].q[i%32]);}
 __syncthreads();m=blockIdx.y*32+threadIdx.x;k=blockIdx.x*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)if(m<M&&k+j<K)out[size_t(k+j)*LD+m]=t[threadIdx.x][threadIdx.y+j];
}
__global__ void prepxt(const float*in,half*out,int N,int K,int LD){
 __shared__ half t[32][33];int k=blockIdx.x*32+threadIdx.x,n=blockIdx.y*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)if(k<K&&n+j<N)t[threadIdx.y+j][threadIdx.x]=__float2half_rn(in[size_t(n+j)*K+k]);
 __syncthreads();n=blockIdx.y*32+threadIdx.x;k=blockIdx.x*32+threadIdx.y;
 #pragma unroll
 for(int j=0;j<32;j+=8)if(n<N&&k+j<K)out[size_t(k+j)*LD+n]=t[threadIdx.x][threadIdx.y+j];
}

// Native aligned Q8_0 conversion, transcribed from ggml convert.cu via
// llama.cpp-qwen38-dense-aw/bench/large-gemm-half2/probe.cu (read-only donor).
__global__ void prep_native(const Q8*q,half*w){
 __shared__ int vals[544];const int*src=reinterpret_cast<const int*>(q)+size_t(blockIdx.x)*544;
 #pragma unroll
 for(int j=threadIdx.x;j<544;j+=32)vals[j]=src[j];
 __syncthreads();half2*dst=reinterpret_cast<half2*>(w+size_t(blockIdx.x)*2048);
 #pragma unroll
 for(int j=0;j<2048;j+=64){const half*bb=reinterpret_cast<const half*>(vals)+17*((j+2*threadIdx.x)/32);char2 v=reinterpret_cast<const char2*>(bb+1)[threadIdx.x%16];dst[j/2+threadIdx.x]=__hmul2(__floats2half2_rn(float(v.x),float(v.y)),__half2half2(*bb));}
}
int main(int argc,char**argv){
 if(argc!=5&&argc!=6)return 2;int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]);unsigned rng=atoi(argv[4]);
 if(M<=0||N<=0||K<=0||M>8192||N>8192||K>8192||K%32)return 2;
 cudaDeviceProp pr;CK(cudaGetDeviceProperties(&pr,0));if(pr.major!=6||pr.minor!=0)return 2;
 auto next=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
 std::vector<Q8> q(size_t(M)*K/32);for(auto&g:q){g.d=__float2half_rn(float(next()%128+1)/65536);for(auto&v:g.q)v=int(next()%255)-127;}
 std::vector<float>x(size_t(N)*K);for(auto&v:x)v=float(int(next()%8193)-4096)/4096;
 Q8*dq;half*dw,*dx,*dh;float*di,*out;
 CK(cudaMalloc(&dq,q.size()*34));CK(cudaMalloc(&dw,size_t(M)*K*2));CK(cudaMalloc(&dx,x.size()*2));CK(cudaMalloc(&di,x.size()*4));CK(cudaMalloc(&dh,size_t(M)*N*2));CK(cudaMalloc(&out,size_t(M)*N*4));
 CK(cudaMemcpy(dq,q.data(),q.size()*34,cudaMemcpyHostToDevice));CK(cudaMemcpy(di,x.data(),x.size()*4,cudaMemcpyHostToDevice));
 cublasHandle_t b;BL(cublasCreate(&b));BL(cublasSetMathMode(b,CUBLAS_DEFAULT_MATH));int ver;BL(cublasGetVersion(b,&ver));printf("{\"type\":\"device\",\"sm\":%d,\"blas\":%d,\"M\":%d,\"N\":%d,\"K\":%d}\n",pr.multiProcessorCount,ver,M,N,K);
 half ha=__float2half(1),hb=__float2half(0);float fa=1,fb=0;
 int prepvariant=0;
 auto weights=[&](){
  if(prepvariant==0)prepw<<<(M*K+255)/256,256>>>(dq,dw,M*K);
  if(prepvariant==2)prepvec<2><<<(M*K/2+255)/256,256>>>(dq,dw,M*K);
  if(prepvariant==4)prepvec<4><<<(M*K/4+255)/256,256>>>(dq,dw,M*K);
  if(prepvariant==8)prepvec<8><<<(M*K/8+255)/256,256>>>(dq,dw,M*K);
 };
 auto prep=[&](){weights();prepx<<<(N*K+255)/256,256>>>(di,dx,N*K);};
 auto wide=[&](){finish<<<(M*N+255)/256,256>>>(dh,out,M*N);};
 auto gemm=[&](int mode,int algo){return cublasGemmEx(b,CUBLAS_OP_T,CUBLAS_OP_N,M,N,K,mode?(void*)&ha:(void*)&fa,dw,CUDA_R_16F,K,dx,CUDA_R_16F,K,mode?(void*)&hb:(void*)&fb,mode?(void*)dh:(void*)out,mode?CUDA_R_16F:CUDA_R_32F,M,mode?CUBLAS_COMPUTE_16F:CUBLAS_COMPUTE_32F,(cublasGemmAlgo_t)algo);};
 cudaEvent_t st,en;CK(cudaEventCreate(&st));CK(cudaEventCreate(&en));
 auto time=[&](auto f,int reps){CK(cudaEventRecord(st));for(int i=0;i<reps;i++)f();CK(cudaEventRecord(en));CK(cudaEventSynchronize(en));float ms;CK(cudaEventElapsedTime(&ms,st,en));return 1000.*ms/reps;};
 prep();BL(gemm(0,-1));CK(cudaDeviceSynchronize());std::vector<float>ref(size_t(M)*N),got(ref.size());CK(cudaMemcpy(ref.data(),out,ref.size()*4,cudaMemcpyDeviceToHost));
 // Independent complete-dot CPU spot checks for the FP32 cuBLAS comparator.
 for(int j=0;j<32;j++){int m=next()%M,n=next()%N;double exact=0;for(int k=0;k<K;k++){auto g=q[(size_t(m)*K+k)/32];float w=__half2float(__float2half_rn(__half2float(g.d)*g.q[k%32]));float a=__half2float(__float2half_rn(x[size_t(n)*K+k]));exact+=double(w)*a;}if(fabs(ref[size_t(n)*M+m]-exact)>0.0001*(1+fabs(exact)))return 4;}
 BL(gemm(1,99));wide();CK(cudaDeviceSynchronize());std::vector<float>href(ref.size());CK(cudaMemcpy(href.data(),out,href.size()*4,cudaMemcpyDeviceToHost));
 // All decoded weight bytes must match the original preparation kernel.
 std::vector<half> wb(size_t(M)*K),wc(wb.size());CK(cudaMemcpy(wb.data(),dw,wb.size()*2,cudaMemcpyDeviceToHost));
 for(int v:{2,4,8}){prepvariant=v;weights();CK(cudaDeviceSynchronize());CK(cudaMemcpy(wc.data(),dw,wc.size()*2,cudaMemcpyDeviceToHost));if(memcmp(wb.data(),wc.data(),wb.size()*2))return 5;}
 prepvariant=0;


 int pad=0,ldw=M,ldx=N;auto setpad=[&](int v){pad=v;ldw=M+v;ldx=N+v;};
 half *wt,*xt;float*tmp;CK(cudaMalloc(&wt,size_t(M+512)*K*2));CK(cudaMalloc(&xt,size_t(N+512)*K*2));CK(cudaMalloc(&tmp,size_t(M)*N*4));
 auto pw=[&](){prep_native<<<M*K/2048,32>>>(dq,dw);};
 auto px=[&](){prepx<<<(N*K+255)/256,256>>>(di,dx,N*K);};
 auto tw=[&](){prepwt<<<dim3((K+31)/32,(M+31)/32),dim3(32,8)>>>(dq,wt,M,K,ldw);};
 auto tx=[&](){prepxt<<<dim3((K+31)/32,(N+31)/32),dim3(32,8)>>>(di,xt,N,K,ldx);};
 auto prepare=[&](int layout){if(layout&1)tw();else pw();if(layout&2)tx();else px();};
 auto output=[&](int mode,int layout){if(layout&4){if(mode)transwide<<<dim3((N+31)/32,(M+31)/32),dim3(32,8)>>>(dh,out,M,N);else trans<<<dim3((N+31)/32,(M+31)/32),dim3(32,8)>>>(tmp,out,M,N);}else if(mode)wide();};
 auto multiply=[&](int mode,int algo,int layout){
  bool wtrans=layout&1,xtrans=layout&2,swap=layout&4;
  const half *a=swap?(xtrans?xt:dx):(wtrans?wt:dw),*bb=swap?(wtrans?wt:dw):(xtrans?xt:dx);
  auto ta=(swap?xtrans:wtrans)?CUBLAS_OP_N:CUBLAS_OP_T;
  auto tb=(swap?wtrans:xtrans)?CUBLAS_OP_T:CUBLAS_OP_N;
  int lda=swap?(xtrans?ldx:K):(wtrans?ldw:K),ldb=swap?(wtrans?ldw:K):(xtrans?ldx:K),mm=swap?N:M,nn=swap?M:N;
  return cublasGemmEx(b,ta,tb,mm,nn,K,mode?(void*)&ha:(void*)&fa,a,CUDA_R_16F,lda,bb,CUDA_R_16F,ldb,mode?(void*)&hb:(void*)&fb,mode?(void*)dh:(swap?(void*)tmp:(void*)out),mode?CUDA_R_16F:CUDA_R_32F,mm,mode?CUBLAS_COMPUTE_16F:CUBLAS_COMPUTE_32F,(cublasGemmAlgo_t)algo);
 };


 pw();CK(cudaDeviceSynchronize());CK(cudaMemcpy(wc.data(),dw,wc.size()*2,cudaMemcpyDeviceToHost));if(memcmp(wb.data(),wc.data(),wb.size()*2))return 9;
 // Exhaustive finite input-bit preservation for both fused preparation layouts.
 tw();tx();CK(cudaDeviceSynchronize());CK(cudaMemcpy(wc.data(),wt,wc.size()*2,cudaMemcpyDeviceToHost));
 for(int m=0;m<M;m++)for(int k=0;k<K;k++)if(memcmp(&wb[size_t(m)*K+k],&wc[size_t(k)*M+m],2))return 9;
 std::vector<half> xc(size_t(N)*K);CK(cudaMemcpy(xc.data(),xt,xc.size()*2,cudaMemcpyDeviceToHost));
 for(int n=0;n<N;n++)for(int k=0;k<K;k++){half v=__float2half_rn(x[size_t(n)*K+k]);if(memcmp(&v,&xc[size_t(k)*N+n],2))return 9;}
 printf("{\"type\":\"fused_input_check\",\"weights\":%zu,\"activations\":%zu,\"different\":0}\n",wb.size(),xc.size());

 struct Config{int mode,layout,pad;};std::vector<Config>cs;
 for(int v:{0,8,32,128,256,512})for(int layout:{2,6})for(int mode=0;mode<2;mode++){
  setpad(v);prepare(layout);BL(multiply(mode,mode?3:6,layout));output(mode,layout);CK(cudaDeviceSynchronize());CK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));auto&base=mode?href:ref;
  size_t diff=0;for(size_t i=0;i<got.size();i++)diff+=memcmp(&got[i],&base[i],4)!=0;
  printf("{\"type\":\"pad_check\",\"mode\":%d,\"layout\":%d,\"pad\":%d,\"different\":%zu}\n",mode,layout,v,diff);
  if(!diff)cs.push_back({mode,layout,v});
 }
 if(argc==5){
  for(int i=0;i<3;i++)for(auto c:cs){setpad(c.pad);prepare(c.layout);BL(multiply(c.mode,c.mode?3:6,c.layout));output(c.mode,c.layout);}CK(cudaDeviceSynchronize());
  for(int sample=0;sample<9;sample++)for(size_t j=0;j<cs.size();j++){
   auto c=cs[(j+sample*7)%cs.size()];setpad(c.pad);prepare(c.layout);
   double g=time([&](){BL(multiply(c.mode,c.mode?3:6,c.layout));},3);
   double p=time([&](){prepare(c.layout);BL(multiply(c.mode,c.mode?3:6,c.layout));output(c.mode,c.layout);},3);
   printf("{\"type\":\"padding\",\"sample\":%d,\"mode\":%d,\"layout\":%d,\"pad\":%d,\"gemm_us\":%.6f,\"pipeline_us\":%.6f}\n",sample,c.mode,c.layout,c.pad,g,p);
  }
 }
 for(auto c:cs){setpad(c.pad);prepare(c.layout);BL(multiply(c.mode,c.mode?3:6,c.layout));output(c.mode,c.layout);CK(cudaDeviceSynchronize());CK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));auto&base=c.mode?href:ref;if(memcmp(base.data(),got.data(),got.size()*4))return 10;}
 printf("{\"type\":\"post_timing_check\",\"configs\":%zu,\"different\":0}\n",cs.size());
 CK(cudaFree(wt));CK(cudaFree(xt));CK(cudaFree(tmp));
 CK(cudaGetLastError());BL(cublasDestroy(b));CK(cudaFree(dq));CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(di));CK(cudaFree(dh));CK(cudaFree(out));CK(cudaEventDestroy(st));CK(cudaEventDestroy(en));
}
