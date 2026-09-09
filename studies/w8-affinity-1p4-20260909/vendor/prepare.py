from pathlib import Path
import shutil
p=Path(__file__).resolve().parent;src=p.parent/'r1'
for f in ['q8.inc','lowact.inc','worker.cu','build.py','run.py','screen.py']:shutil.copyfile(src/f,p/f)
s=(p/'worker.cu').read_text().replace('#include <cuda_fp16.h>','#include <cuda_fp16.h>\n#include <cublas_v2.h>').replace('#include "packed.inc"','')
idx=s.index('template<class T>struct Device')
helpers='''__global__ void decode_weights(const unsigned char* packed,half*out,size_t count,int n,int k){
 for(size_t ix=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*4;ix<count;ix+=size_t(gridDim.x)*blockDim.x*4){
  size_t row=ix/k,e=row/n;int r=row%n,kk=ix%k;
  size_t base=((e*(n/64)+r/64)*(k/32)+kk/32)*2176;
  float d=__half2float(*(const half*)(packed+base+(r%64)*2));
  unsigned q=__ldg((const unsigned*)(packed+base+128+(r%64)*32+kk%32));
  ((half2*)(out+ix))[0]=__floats2half2_rn(float((signed char)(q&255))*d,float((signed char)((q>>8)&255))*d);
  ((half2*)(out+ix))[1]=__floats2half2_rn(float((signed char)((q>>16)&255))*d,float((signed char)((q>>24)&255))*d);
 }
}
__global__ void widen_output(const half*in,float*out,size_t count){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)out[i]=__half2float(in[i]);}
'''
s=s[:idx]+helpers+s[idx:]
a=s.index(' auto run=');b=s.index('\n std::vector<float>ref;',a)
s=s[:a]+''' Device<half>decoded(wc),hout(oc);cublasHandle_t handle;if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublasCreate");
 const half alpha16=__float2half_rn(1.f),beta16=__float2half_rn(0.f);const float alpha32=1,beta32=0;
 auto run=[&](int b,bool prep){
  if(b==32)aw_q8_service_m64_n128_halfpipe_sync<false><<<grid,256>>>(dd.p,dt.p,ts.size(),n,k);
  else if(b==16)aw_q8_service_m64_n128_halfpipe_sync<true><<<grid,256>>>(dd16.p,dt.p,ts.size(),n,k);
  else{
   decode_weights<<<4096,256>>>(dw.p,decoded.p,wc,n,k);
   cublasStatus_t st=cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,b==1024?(const void*)&alpha16:(const void*)&alpha32,decoded.p,CUDA_R_16F,k,(long long)n*k,dh.p,CUDA_R_16F,k,(long long)m*k,b==1024?(const void*)&beta16:(const void*)&beta32,b==1024?(void*)hout.p:(void*)out.p,b==1024?CUDA_R_16F:CUDA_R_32F,n,(long long)m*n,ex,b==1024?CUBLAS_COMPUTE_16F:CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
   if(st!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublasGemmStridedBatchedEx status="+std::to_string(int(st)));
   if(b==1024)widen_output<<<4096,256>>>(hout.p,out.p,oc);
  }CU(cudaGetLastError());};'''+s[b:]
s=s.replace('{32,16,64,128,256}','{32,16,1024,2048}').replace('i<5;++i){int b=bs[(i+r)%5]','i<4;++i){int b=bs[(i+r)%4]')
a=s.index('float sum[2]={},total=0;');b=s.index('bad+=bits(got[ix])!=bits(expected);',a)+len('bad+=bits(got[ix])!=bits(expected);')
s=s[:a]+'''float sum[2]={};double exact=0,l1=0;for(int g=0;g<kg;++g){float d=__half2float(sc[(size_t(e)*n+col)*kg+g]);for(int j=0;j<32;++j){float weight=float(w[((size_t(e)*n+col)*kg+g)*32+j])*d;if(b>=1024)weight=__half2float(__float2half_rn(weight));float activation=aa[(size_t(e)*m+row)*k+g*32+j];sum[j/16]=std::fma(activation,weight,sum[j/16]);exact+=double(activation)*weight;l1+=std::abs(double(activation)*weight);}}if(b>=1024)bad+=std::abs(double(got[ix])-exact)>(b==1024?0.01:0.000002)*std::max(1.,l1);else bad+=bits(got[ix])!=bits(sum[0]+sum[1]);'''+s[b:]
# Validate the complete decoder output independently before timings.
s=s.replace(' printf("META M=', ''' auto dwcheck=decoded.get();size_t decode_bad=0;for(size_t i=0;i<wc;++i){half expected=__float2half_rn(float(w[i])*__half2float(sc[i/32]));decode_bad+=__half_as_ushort(dwcheck[i])!=__half_as_ushort(expected);}printf("WEIGHT_DECODE values=%zu bad=%zu\\n",wc,decode_bad);if(decode_bad)throw std::runtime_error("weight decoder mismatch");
 printf("META M=''')
(p/'worker.cu').write_text(s)
s=(p/'build.py').read_text().replace("'packed.inc',",'').replace("str(p/'worker')]","str(p/'worker'),'-lcublas']");(p/'build.py').write_text(s)
s=(p/'run.py').read_text().replace("p.parents[1]/'bench/COORDINATION-20260908.md'","p.parents[2]/'bench/COORDINATION-20260908.md'").replace('w8-affinity-1p4-','w8-affinity-vendor-').replace('W8A16 packed-half partial accumulation candidates versus original Q8','Q8 decode plus batched vendor GEMM against unchanged AffinityWave')
(p/'run.py').write_text(s)
