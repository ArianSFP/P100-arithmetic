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
int main(int argc,char**argv){
 if(argc!=5)return 2;int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]);unsigned rng=atoi(argv[4]);
 if(M<=0||N<=0||K<=0||M>8192||N>4096||K>8192||K%32)return 2;
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
 struct Config{int mode,algo;};std::vector<Config>cs{{0,-1},{1,99},{1,-1}};
 for(int mode=0;mode<2;mode++)for(int a=0;a<(N==256?24:7);a++)cs.push_back({mode,a});
 std::vector<Config>ok;
 for(auto c:cs){auto status=gemm(c.mode,c.algo);if(status==CUBLAS_STATUS_NOT_SUPPORTED){printf("{\"type\":\"unsupported\",\"mode\":%d,\"algo\":%d}\n",c.mode,c.algo);continue;}BL(status);if(c.mode)wide();CK(cudaDeviceSynchronize());CK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));double e=0,r=0;size_t diff=0;auto&base=c.mode?href:ref;for(size_t i=0;i<got.size();i++){if(!std::isfinite(got[i]))return 4;double d=double(got[i])-ref[i];e+=d*d;r+=double(ref[i])*ref[i];diff+=memcmp(&got[i],&base[i],4)!=0;}printf("{\"type\":\"check\",\"mode\":%d,\"algo\":%d,\"different\":%zu,\"relL2\":%.9g}\n",c.mode,c.algo,diff,sqrt(e/r));ok.push_back(c);}
 // Warm the GPU and all accepted paths before retained, rotated measurements.
 for(int i=0;i<3;i++)for(auto c:ok){prep();BL(gemm(c.mode,c.algo));if(c.mode)wide();}CK(cudaDeviceSynchronize());
 for(int s=0;s<7;s++){
  double w=time([&](){prepw<<<(M*K+255)/256,256>>>(dq,dw,M*K);},5),a=time([&](){prepx<<<(N*K+255)/256,256>>>(di,dx,N*K);},5),f=time(wide,5);
  printf("{\"type\":\"stages\",\"sample\":%d,\"weight_us\":%.6f,\"activation_us\":%.6f,\"finish_us\":%.6f}\n",s,w,a,f);
  for(size_t j=0;j<ok.size();j++){auto c=ok[(j+s*7)%ok.size()];double g=time([&](){BL(gemm(c.mode,c.algo));},3);double p=time([&](){prep();BL(gemm(c.mode,c.algo));if(c.mode)wide();},3);printf("{\"type\":\"timing\",\"sample\":%d,\"mode\":%d,\"algo\":%d,\"gemm_us\":%.6f,\"pipeline_us\":%.6f}\n",s,c.mode,c.algo,g,p);}
 }
 // Alternate preparation only with the unchanged default GEMMs; cache variant
 // excludes weight preparation, but still charges activation and output conversion.
 for(int s=0;s<9;s++)for(int j=0;j<10;j++){
  int order=(j+s)%10,mode=order%2;int variants[]={0,2,4,8,-1};prepvariant=variants[order/2];
  double w=prepvariant<0?0:time(weights,5);
  double p=time([&](){prep();BL(gemm(mode,mode?99:-1));if(mode)wide();},5);
  CK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));auto& base=mode?href:ref;
  if(memcmp(base.data(),got.data(),got.size()*4))return 6;
  printf("{\"type\":\"prepvariant\",\"sample\":%d,\"mode\":%d,\"variant\":%d,\"weight_us\":%.6f,\"pipeline_us\":%.6f}\n",s,mode,prepvariant,w,p);
 }
 CK(cudaGetLastError());BL(cublasDestroy(b));CK(cudaFree(dq));CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(di));CK(cudaFree(dh));CK(cudaFree(out));CK(cudaEventDestroy(st));CK(cudaEventDestroy(en));
}
