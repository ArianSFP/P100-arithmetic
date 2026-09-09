#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
constexpr int AW_KSTAGE=32,AW_T64_ROWS=64,AW_T64_STAGE_BYTES=2176;
#include "q8.inc"
#define CU(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+cudaGetErrorString(e));}while(0)
template<int BITS>__global__ void quantize(const half*a,unsigned char*out,size_t groups){
 const int lane=threadIdx.x&31;constexpr int BYTES=4+BITS*4,QMAX=(1<<(BITS-1))-1;
 for(size_t g=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;g<groups;g+=size_t(gridDim.x)*blockDim.x/32){
  float x=__half2float(a[g*32+lane]),mx=fabsf(x);
  #pragma unroll
  for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  float scale=mx/QMAX,inv=mx?QMAX/mx:0.f;
  int q=max(-QMAX,min(QMAX,__float2int_rn(x*inv)));
  unsigned char*p=out+g*BYTES;if(lane==0)*(float*)p=scale;
  if constexpr(BITS==8)((signed char*)(p+4))[lane]=q;
  else{int next=__shfl_down_sync(0xffffffff,q,1);if(!(lane&1))p[4+lane/2]=(q&15)|((next&15)<<4);}
 }
}
template<int BITS>__device__ __forceinline__ void load_activation(const void*raw,size_t index,float4&lo,float4&hi){
 constexpr int BYTES=4+BITS*4;
 const unsigned char*p=(const unsigned char*)raw+(index/32)*BYTES;
 float d=__ldg((const float*)p);int off=index%32;float vals[8];
 if constexpr(BITS==8){
  const unsigned w0=__ldg((const unsigned*)(p+4+off)),w1=__ldg((const unsigned*)(p+8+off));
  #pragma unroll
  for(int j=0;j<8;++j)vals[j]=float((signed char)((j<4?w0:w1)>>((j%4)*8)))*d;
 }else{
  const unsigned w=__ldg((const unsigned*)(p+4+off/2));
  #pragma unroll
  for(int j=0;j<8;++j){int q=(w>>(j*4))&15;q=(q^8)-8;vals[j]=float(q)*d;}
 }
 lo=make_float4(vals[0],vals[1],vals[2],vals[3]);hi=make_float4(vals[4],vals[5],vals[6],vals[7]);
}
#include "lowact.inc"
#include "packed.inc"
#include "old-packed.inc"
template<class T>struct Device{T*p;size_t n;Device(size_t x):n(x){CU(cudaMalloc(&p,n*sizeof(T)));}~Device(){cudaFree(p);}void put(const std::vector<T>&v){CU(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>v(n);CU(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
uint32_t rng(uint32_t&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
uint32_t bits(float f){uint32_t x;memcpy(&x,&f,4);return x;}
float half_round(double x){
 double a=std::abs(x);if(a>=65520)return std::copysign(INFINITY,x);if(a==0)return float(x);
 int exp;std::frexp(a,&exp);int shift=a<std::ldexp(1.,-14)?24:11-exp;
 double r=std::ldexp(std::nearbyint(std::ldexp(a,shift)),-shift);return std::copysign(float(r),x);
}
int main(int argc,char**argv)try{
 if(argc!=8||std::string(argv[1])!="--gpu-approved"||std::string(argv[2])!="1")throw std::runtime_error("usage: --gpu-approved 1 M N K experts");
 int m=atoi(argv[3]),n=atoi(argv[4]),k=atoi(argv[5]),ex=atoi(argv[6]);
 if(m<1||m>512||n%128||n<128||n>2048||k%32||k<32||k>2048||ex<1||ex>64)throw std::runtime_error("shape bounds");
 int devices;CU(cudaGetDeviceCount(&devices));if(devices!=1)throw std::runtime_error("one visible GPU required");cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor!=0)throw std::runtime_error("SM60 required");
 size_t ac=size_t(ex)*m*k,oc=size_t(ex)*m*n,wc=size_t(ex)*n*k;int kg=k/32;
 std::vector<float>a(ac);std::vector<int8_t>w(wc);std::vector<half>sc(wc/32);std::vector<unsigned char>packed(wc/32*34);
 uint32_t seed=3911;
 for(size_t i=0;i<ac;++i){float x=(int(rng(seed)%4097)-2048)/1024.f;if((i/32)%97==0)x=0;if((i/32)%101==0&&i%32==0)x=16;a[i]=__half2float(__float2half_rn(x));}
 for(int e=0;e<ex;++e)for(int r=0;r<n;++r)for(int g=0;g<kg;++g){size_t si=(size_t(e)*n+r)*kg+g;half d=__float2half_rn(((rng(seed)%127)+1)/1024.f);sc[si]=d;size_t b=((size_t(e)*(n/64)+r/64)*kg+g)*2176;memcpy(packed.data()+b+(r%64)*2,&d,2);for(int j=0;j<32;++j){int q=int(rng(seed)%256)-128;w[si*32+j]=q;packed[b+128+(r%64)*32+j]=uint8_t(q);}}
 std::vector<half>ha(ac);for(size_t i=0;i<ac;++i)ha[i]=__float2half_rn(a[i]);Device<half>dh(ac);dh.put(ha);
 Device<float> da(ac),out(oc);Device<unsigned char>dw(packed.size()),aq8(ac/32*36),aq4(ac/32*20);da.put(a);dw.put(packed);
 int candidate_grid=prop.multiProcessorCount*atoi(argv[7]);if(candidate_grid<prop.multiProcessorCount||candidate_grid>prop.multiProcessorCount*6)throw std::runtime_error("grid bound");
 int qgrid=std::min<size_t>((ac/32+7)/8,4096),grid=prop.multiProcessorCount*2;
 auto quant=[&](int b){if(b==8)quantize<8><<<qgrid,256>>>(dh.p,aq8.p,ac/32);else quantize<4><<<qgrid,256>>>(dh.p,aq4.p,ac/32);CU(cudaGetLastError());};
 quant(8);quant(4);CU(cudaDeviceSynchronize());auto p8=aq8.get(),p4=aq4.get();std::vector<float>d8(ac),d4(ac);
 for(int b:{8,4}){auto&pk=b==8?p8:p4;auto&dq=b==8?d8:d4;int bytes=b==8?36:20,qmax=(1<<(b-1))-1;size_t bad=0;
  for(size_t g=0;g<ac/32;++g){float mx=0;for(int j=0;j<32;++j)mx=std::max(mx,std::abs(a[g*32+j]));float ds=mx/qmax,inv=mx?qmax/mx:0,got;memcpy(&got,pk.data()+g*bytes,4);bad+=bits(ds)!=bits(got);
   for(int j=0;j<32;++j){int q=b==8?(int)(int8_t)pk[g*bytes+4+j]:int((pk[g*bytes+4+j/2]>>((j%2)*4))&15);if(b==4)q=(q^8)-8;int expected=std::clamp(int(std::nearbyint(a[g*32+j]*inv)),-qmax,qmax);bad+=q!=expected;dq[g*32+j]=q*got;}}
  printf("QUANT bits=%d values=%zu bad=%zu\n",b,ac,bad);if(bad)throw std::runtime_error("quantizer mismatch");}
 std::vector<aw_work_desc>ds,ds16,ds8,ds4;std::vector<aw_tile_desc>ts,ts128;
 for(int e=0;e<ex;++e){aw_work_desc d={da.p+size_t(e)*m*k,(const char*)dw.p+size_t(e)*n*kg*34,out.p+size_t(e)*m*n,nullptr,0,m,0,e};ds.push_back(d);d.input=(const void*)((uintptr_t)(dh.p+size_t(e)*m*k)|1u);ds16.push_back(d);d.input=aq8.p+size_t(e)*m*kg*36;ds8.push_back(d);d.input=aq4.p+size_t(e)*m*kg*20;ds4.push_back(d);for(int r=0;r<m;r+=64)ts.push_back({e,r,std::min(64,m-r)});}
 for(int e=0;e<ex;++e)for(int r=0;r<m;r+=128)ts128.push_back({e,r,std::min(128,m-r)});Device<aw_tile_desc>dt128(ts128.size());dt128.put(ts128);
 Device<aw_work_desc>dd(ds.size()),dd16(ds16.size()),dd8(ds8.size()),dd4(ds4.size());Device<aw_tile_desc>dt(ts.size());dd.put(ds);dd16.put(ds16);dd8.put(ds8);dd4.put(ds4);dt.put(ts);
 Device<aw_tile_desc>dtplan((ts.size()+1)/2),dtleft(ts.size());Device<int>dcounts(2);
 auto run=[&](int b,bool prep){if(prep&&b<16)quant(b);if(b>=64)w8_make_pairs<<<1,256>>>(dt.p,ts.size(),dtplan.p,dtleft.p,dcounts.p);if(b==64){w8_fallback_half<true,1><<<prop.multiProcessorCount*3,256>>>(dd16.p,dtleft.p,ts.size(),n,k,dcounts.p+1);w8_pair_half<true,1><<<candidate_grid,256>>>(dd16.p,dtplan.p,(ts.size()+1)/2,n,k,dcounts.p);}else if(b==128){w8_fallback_half<true,2><<<prop.multiProcessorCount*3,256>>>(dd16.p,dtleft.p,ts.size(),n,k,dcounts.p+1);w8_pair_half<true,2><<<candidate_grid,256>>>(dd16.p,dtplan.p,(ts.size()+1)/2,n,k,dcounts.p);}else if(b==256){w8_fallback_half<true,4><<<prop.multiProcessorCount*3,256>>>(dd16.p,dtleft.p,ts.size(),n,k,dcounts.p+1);w8_pair_half<true,4><<<candidate_grid,256>>>(dd16.p,dtplan.p,(ts.size()+1)/2,n,k,dcounts.p);}else if(b==512){w8_fallback_half<true,0><<<prop.multiProcessorCount*3,256>>>(dd16.p,dtleft.p,ts.size(),n,k,dcounts.p+1);{if(n==512&&k==2048)w8_pair_half<true,0,512,2048><<<candidate_grid,256>>>(dd16.p,dtplan.p,(ts.size()+1)/2,n,k,dcounts.p);else if(n==2048&&k==512)w8_pair_half<true,0,2048,512><<<candidate_grid,256>>>(dd16.p,dtplan.p,(ts.size()+1)/2,n,k,dcounts.p);else w8_pair_half<true,0><<<candidate_grid,256>>>(dd16.p,dtplan.p,(ts.size()+1)/2,n,k,dcounts.p);}}else if(b==32)aw_q8_service_m64_n128_halfpipe_sync<false><<<grid,256>>>(dd.p,dt.p,ts.size(),n,k);else if(b==16)aw_q8_service_m64_n128_halfpipe_sync<true><<<grid,256>>>(dd16.p,dt.p,ts.size(),n,k);else if(b==8)w8_lowact<8><<<grid,256>>>(dd8.p,dt.p,ts.size(),n,k);else w8_lowact<4><<<grid,256>>>(dd4.p,dt.p,ts.size(),n,k);CU(cudaGetLastError());};
 std::vector<float>ref;
 for(int b:{32,16,64,128,256,512}){CU(cudaMemset(out.p,0xff,oc*4));run(b,true);CU(cudaDeviceSynchronize());auto got=out.get();if(b==32)ref=got;auto&aa=b>=16?a:b==8?d8:d4;size_t nonfinite=0,changed=0,bad=0;double err=0,norm=0;for(size_t i=0;i<oc;++i){nonfinite+=!std::isfinite(got[i]);changed+=bits(got[i])!=bits(ref[i]);double d=double(got[i])-ref[i];err+=d*d;norm+=double(ref[i])*ref[i];}
  size_t samples=std::min<size_t>(oc,256);uint32_t sr=731;for(size_t z=0;z<samples;++z){size_t ix=rng(sr)%oc;int col=ix%n,row=(ix/n)%m,e=ix/(size_t(m)*n);float sum[2]={},total=0;int flush=b==64?1:b==128?2:b==256?4:0;for(int g=0;g<kg;++g){float d=__half2float(sc[(size_t(e)*n+col)*kg+g]);for(int j=0;j<32;++j){float weight=float(w[((size_t(e)*n+col)*kg+g)*32+j])*d;float activation=aa[(size_t(e)*m+row)*k+g*32+j];if(b>=64){weight=__half2float(__float2half_rn(weight));sum[j/16]=half_round(double(activation)*double(weight)+double(sum[j/16]));}else sum[j/16]=std::fma(activation,weight,sum[j/16]);}if(flush&&((g+1)%flush==0||g+1==kg)){total+=sum[0]+sum[1];sum[0]=sum[1]=0;}}float expected=flush?total:sum[0]+sum[1];bad+=bits(got[ix])!=bits(expected);}
  printf("CHECK A%d outputs=%zu samples=%zu bad=%zu nonfinite=%zu changed=%zu rel_l2=%.9g\n",b,oc,samples,bad,nonfinite,changed,std::sqrt(err/std::max(norm,1e-300)));fflush(stdout);if(bad||nonfinite||((b==32||b==16)&&changed))throw std::runtime_error("arithmetic mismatch");if(b>=64){
   if(b==64)w8_old_half<true,1><<<candidate_grid,256>>>(dd16.p,dt128.p,ts128.size(),n,k);
   else if(b==128)w8_old_half<true,2><<<candidate_grid,256>>>(dd16.p,dt128.p,ts128.size(),n,k);
   else if(b==256)w8_old_half<true,4><<<candidate_grid,256>>>(dd16.p,dt128.p,ts128.size(),n,k);
   else w8_old_half<true,0><<<candidate_grid,256>>>(dd16.p,dt128.p,ts128.size(),n,k);
   CU(cudaGetLastError());CU(cudaDeviceSynchronize());auto previous=out.get();size_t diff=0;for(size_t ix=0;ix<oc;++ix)diff+=bits(previous[ix])!=bits(got[ix]);printf("MATCH_PREVIOUS A%d values=%zu bad=%zu\n",b,oc,diff);if(diff)throw std::runtime_error("buffer transformation changed output");
  }}
 printf("META M=%d N=%d K=%d experts=%d quantization_timed=1 full_q8_codes=1\n",m,n,k,ex);
 auto end=std::chrono::steady_clock::now()+std::chrono::seconds(ex==64&&k>=512?2:0);do{for(int b:{32,16,64,128,256,512})run(b,true);CU(cudaDeviceSynchronize());}while(std::chrono::steady_clock::now()<end);
 cudaEvent_t st,en;CU(cudaEventCreate(&st));CU(cudaEventCreate(&en));int reps=ex==64&&k>=512?7:1,iters=ex==64&&k>=512?6:1;int bs[]={32,16,64,128,256,512};
 for(int r=0;r<reps;++r)for(int i=0;i<6;++i){int b=bs[(i+r)%6];for(int prep=1;prep>=0;--prep){if(!prep&&b>=16)continue;CU(cudaEventRecord(st));for(int j=0;j<iters;++j)run(b,prep);CU(cudaEventRecord(en));CU(cudaEventSynchronize(en));float ms;CU(cudaEventElapsedTime(&ms,st,en));printf("TIME A%d prep=%d rep=%d us=%.9g\n",b,prep,r,ms*1000/iters);}}
 puts("PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"STOP: %s\n",e.what());return 1;}
