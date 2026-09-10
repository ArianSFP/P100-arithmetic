#include <cuda.h>
#include "../../archive/w4a16-r2-20260908/common.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <cfenv>
#include <map>

void ck(CUresult e,int line) {if(e){const char *s="?";cuGetErrorString(e,&s);fprintf(stderr,"CUDA_ERROR %d %s line=%d\n",int(e),s,line);exit(3);}}
#define CK(x) ck(x,__LINE__)
struct Buffer {
 CUdeviceptr p;size_t size;
 Buffer(size_t n):size(n){CK(cuMemAlloc(&p,n));}
 ~Buffer(){cuMemFree(p);}
 template<class T>void upload(const std::vector<T>&v){require(v.size()*sizeof(T)==size,"buffer size");CK(cuMemcpyHtoD(p,v.data(),size));}
};
CUfunction fn(CUmodule m,const char*n){
 static std::map<std::pair<CUmodule,std::string>,CUfunction> cache;
 auto key=std::make_pair(m,std::string(n));auto it=cache.find(key);if(it!=cache.end())return it->second;
 CUfunction f;CK(cuModuleGetFunction(&f,m,n));cache[key]=f;return f;
}
void launch(CUfunction f,unsigned x,unsigned y,unsigned z,void **args){CK(cuLaunchKernel(f,x,y,z,128,1,1,0,0,args,0));}
struct Config {int h,r,b;std::string name()const{return "w4a4_h"+std::to_string(h)+"_r"+std::to_string(r)+"_b"+std::to_string(b);}};
std::vector<Config> configs(){std::vector<Config>v;for(int h:{0,1,2})for(int r:{1,2,4})for(int b:{1,4})v.push_back({h,r,b});return v;}
struct Data {
 unsigned m,k,n,g,padded;
 std::vector<uint16_t> a,sc;
 std::vector<unsigned>w,aq,wp,ap;
 std::vector<float>ad,ref;
 Buffer dw,ds,din,daq,dad,partial,out,dwp,dap;
 Data(unsigned rows,unsigned cols,unsigned batches,int family):m(rows),k(cols),n(batches),g(k/32),padded((m+31)/32*32),
 a(n*k),sc(size_t(padded)*g),w(size_t(padded)*g*4),aq(n*g*4),wp(w.size()),ap(aq.size()),ad(n*g),ref(size_t(n)*m),
 dw(w.size()*4),ds(sc.size()*2),din(a.size()*2),daq(aq.size()*4),dad(ad.size()*4),partial(size_t(n)*m*32*4),out(ref.size()*4),dwp(wp.size()*4),dap(ap.size()*4) {
 uint32_t seed=0xa4040001u^m^(k*17)^(n*131)^(family*65537);
 for(size_t i=0;i<a.size();i++) {
  unsigned u=random32(seed);
  a[i]=uint16_t((u&0x8000)|((10+u%8)<<10)|((u>>16)&1023));
  if(family==1)a[i]=uint16_t((i&1 ? 0x8000:0)|0x3c00);
  if(family==2)a[i]=i%32==0 ? 0x7bff : uint16_t(i&1 ? 1 : 0x8000);
  if(family==3)a[i]=uint16_t(i&1 ? 0x8000:0);
 }
 for(unsigned row=0;row<m;row++)for(unsigned j=0;j<g;j++) {
  unsigned u=random32(seed);sc[scale_address(row,j,g)]=uint16_t((u&0x8000)|((8+u%10)<<10)|((u>>16)&1023));
  for(int p=0;p<4;p++)w[word_address(row,j,p,g)]=row<16 ? unsigned(row)*0x11111111u : random32(seed);
 }
 for(unsigned bg=0;bg<n*g;bg++) {
  float mx=0;for(int j=0;j<32;j++)mx=std::max(mx,std::abs(half_float(a[bg*32+j])));
  ad[bg]=mx/7.f;
  for(int j=0;j<32;j++) {int q=mx ? std::max(-7,std::min(7,int(std::nearbyint(half_float(a[bg*32+j])/ad[bg])))) : 0;aq[bg*4+j/8]|=(unsigned(q)&15)<<(4*(j&7));}
 }
 for(unsigned row=0;row<m;row++)for(unsigned group=0;group<g;group++)for(int j=0;j<32;j++) {
  unsigned q=unsigned(unpack_nibble(w[word_address(row,group,j/8,g)],j))&15;
  for(int p=0;p<4;p++)wp[word_address(row,group,p,g)]|=((q>>p)&1u)<<j;
 }
 dw.upload(w);ds.upload(sc);din.upload(a);daq.upload(aq);dad.upload(ad);dwp.upload(wp);pack_activation_planes();
 printf("DATA M=%u K=%u N=%u family=%d seed=0xa4040001_xor_shape_family\n",m,k,n,family);
 }
 void pack_activation_planes() {
  std::fill(ap.begin(),ap.end(),0);
  for(unsigned bg=0;bg<n*g;bg++)for(int j=0;j<32;j++) {
   unsigned q=(aq[bg*4+j/8]>>(4*(j&7)))&15;
   for(int p=0;p<4;p++)ap[bg*4+p]|=((q>>p)&1u)<<j;
  }
  dap.upload(ap);
 }
 void reference(bool endpoint=false) {
  if(endpoint){
   for(size_t i=0;i<aq.size();i++)aq[i]=i%3==0 ? 0x88888888u : i%3==1 ? 0x77777777u : 0x78787878u;
   daq.upload(aq);pack_activation_planes(); // Independent packed endpoint gate, bypass quantizer.
  }
  for(unsigned b=0;b<n;b++)for(unsigned row=0;row<m;row++) {
   float sums[32]={};
   for(unsigned s=0;s<32;s++)for(unsigned j=s;j<g;j+=32) {
    int dot=0;
    for(int i=0;i<32;i++)dot+=unpack_nibble(w[word_address(row,j,i/8,g)],i)*unpack_nibble(aq[(b*g+j)*4+i/8],i);
    float scale=half_float(sc[scale_address(row,j,g)])*ad[b*g+j];
    sums[s]=std::fma(float(dot),scale,sums[s]);
   }
   ref[b*m+row]=tree_sum(sums,32);
  }
 }
 void quant(CUmodule mod) {void *args[]={&din.p,&daq.p,&dap.p,&dad.p,&g,&n};launch(fn(mod,"quantize"),(g*n+3)/4,1,1,args);}
 void check_quant(CUmodule mod) {
  quant(mod);std::vector<unsigned> got(aq.size());std::vector<float> scales(ad.size());
  CK(cuMemcpyDtoH(got.data(),daq.p,got.size()*4));CK(cuMemcpyDtoH(scales.data(),dad.p,scales.size()*4));
  require(got==aq,"quantized codes mismatch");
  CK(cuMemcpyDtoH(got.data(),dap.p,got.size()*4));require(got==ap,"activation planes mismatch");
  for(size_t i=0;i<ad.size();i++)require(float_bits(scales[i])==float_bits(ad[i]),"activation scale mismatch");
 }
 void run(CUmodule mod,Config c,bool prep) {
  if(prep)quant(mod);
  CUdeviceptr weights=c.h==2 ? dwp.p:dw.p,acts=c.h==2 ? dap.p:daq.p;
  void *args[]={&weights,&ds.p,&acts,&dad.p,&partial.p,&m,&g,&n};
  launch(fn(mod,c.name().c_str()),(m+128*c.r-1)/(128*c.r),(n+c.b-1)/c.b,32,args);
  void *red[]={&partial.p,&out.p,&m};launch(fn(mod,"reduce32"),(m+127)/128,n,1,red);
 }
 void validate(CUmodule mod,Config c) {
  run(mod,c,false);std::vector<float> got(ref.size());CK(cuMemcpyDtoH(got.data(),out.p,got.size()*4));
  for(size_t i=0;i<ref.size();i++)if(float_bits(got[i])!=float_bits(ref[i])) {
   fprintf(stderr,"MISMATCH %s at %zu got %.9g expected %.9g\n",c.name().c_str(),i,got[i],ref[i]);exit(4);
  }
  printf("CHECK %s PASS values=%zu\n",c.name().c_str(),ref.size());
 }
 void a16(CUmodule old) {
  CUdeviceptr native=0;void *args[]={&native,&dw.p,&ds.p,&din.p,&partial.p,&m,&g,&n};
  launch(fn(old,n==1 ? "r2_14_r2":"r2_36_r2"),(m+255)/256,n==1?n:(n+3)/4,32,args);
 }
};
template<class F>float timed(F f,int iters) {
 CUevent start,stop;CK(cuEventCreate(&start,0));CK(cuEventCreate(&stop,0));
 CK(cuEventRecord(start,0));for(int i=0;i<iters;i++)f();CK(cuEventRecord(stop,0));CK(cuEventSynchronize(stop));
 float ms;CK(cuEventElapsedTime(&ms,start,stop));CK(cuEventDestroy(start));CK(cuEventDestroy(stop));return ms*1000/iters;
}
int main(int argc,char**argv) {try {
 require(argc==3 || argc==6,"usage: worker kernels.cubin check | worker kernels.cubin bench M K N");
 require(std::getenv("CUDA_VISIBLE_DEVICES") && std::string(std::getenv("CUDA_VISIBLE_DEVICES"))=="GPU-4868830a-c1cf-90bd-8018-2360c55293b8","GPU3 UUID required");
 std::fesetround(FE_TONEAREST);CK(cuInit(0));int count;CK(cuDeviceGetCount(&count));require(count==1,"exactly one visible GPU required");
 CUdevice dev;CK(cuDeviceGet(&dev,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,dev));CUmodule mod;CK(cuModuleLoad(&mod,argv[1]));
 if(std::string(argv[2])=="check") {
  for(int family=0;family<4;family++)for(auto shape:std::vector<std::vector<unsigned>>{{33,96,1},{129,32,3},{513,544,4}}) {
   Data d(shape[0],shape[1],shape[2],family);d.check_quant(mod);d.reference();for(auto c:configs())d.validate(mod,c);
   if(family==0){d.reference(true);for(auto c:configs())d.validate(mod,c);}
  }
 }else{
  unsigned m=std::stoul(argv[3]),k=std::stoul(argv[4]),n=std::stoul(argv[5]);
  require((m==5120||m==17408)&&(k==5120||k==17408)&&(n==1||n==4),"bounded benchmark shapes");
  Data d(m,k,n,0);d.check_quant(mod);d.reference();for(auto c:configs())d.validate(mod,c);
  CUmodule old;CK(cuModuleLoad(&old,"archive/w4a16-r2-20260908/build/kernels.cubin"));
  auto a16=[&]{d.a16(old);void *red[]={&d.partial.p,&d.out.p,&d.m};launch(fn(mod,"reduce32"),(m+127)/128,n,1,red);};
  // Sealed A16 control on ORIGINAL inputs; arithmetic comparison is only
  // between the W4A4 candidates. A16 outputs have separate quantization error.
  a16();std::vector<float> baseline(d.ref.size());CK(cuMemcpyDtoH(baseline.data(),d.out.p,baseline.size()*4));
  for(unsigned b=0;b<n;b++)for(unsigned row=0;row<m;row++) {
   float sums[32]={};
   for(unsigned stripe=0;stripe<32;stripe++)for(unsigned group=stripe;group<d.g;group+=32) {
    float dot=0;
    for(int j=0;j<32;j++)dot=std::fma(float(unpack_nibble(d.w[word_address(row,group,j/8,d.g)],j)),half_float(d.a[(b*d.g+group)*32+j]),dot);
    sums[stripe]=std::fma(dot,half_float(d.sc[scale_address(row,group,d.g)]),sums[stripe]);
   }
   require(float_bits(baseline[b*m+row])==float_bits(tree_sum(sums,32)),"sealed A16 oracle mismatch");
  }
  puts("CHECK sealed_a16 PASS");
  long double err=0,energy=0;for(size_t i=0;i<baseline.size();i++){long double e=baseline[i]-d.ref[i];err+=e*e;energy+=(long double)baseline[i]*baseline[i];}
  printf("QUANTIZATION relative_L2=%.9g model_accuracy_validated=0\n",double(sqrt(err/energy)));
  auto cs=configs();for(auto c:cs)for(int i=0;i<8;i++)d.run(mod,c,true);for(int i=0;i<8;i++)a16();CK(cuCtxSynchronize());
  for(int round=0;round<9;round++) {
   // Rotate both control and candidates through first/last positions.
   for(unsigned j=0;j<=cs.size();j++) {unsigned idx=(j+round)%(cs.size()+1);if(idx==cs.size())printf("TIME round=%d config=sealed_a16 us=%.6f\n",round,timed(a16,12));
    else {auto c=cs[idx];printf("TIME round=%d config=%s us=%.6f\n",round,c.name().c_str(),timed([&]{d.run(mod,c,true);},12));}}
  }
  CK(cuModuleUnload(old));
 }
 CK(cuModuleUnload(mod));CK(cuCtxDestroy(ctx));puts("PASS");return 0;
 }catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 2;}}
