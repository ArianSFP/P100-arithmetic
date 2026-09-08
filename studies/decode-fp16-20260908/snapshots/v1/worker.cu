#include "kernels.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(3);}}while(0)
struct Cfg {std::string name;int mode,r,b,nw,s;const void* kernel;bool fused;};
std::vector<Cfg> cfgs;
template<int R,int B,int NW,int S> void add(){
#define ADD(M) cfgs.push_back({"m"+std::to_string(M)+"r"+std::to_string(R)+"b"+std::to_string(B)+"w"+std::to_string(NW)+"s"+std::to_string(S),M,R,B,NW,S,(void*)decode<M,R,B,NW,S>,true})
ADD(0);ADD(1);ADD(2);
#undef ADD
}
template<int B> void sets(){add<1,B,4,8>();add<1,B,8,8>();add<1,B,8,16>();add<1,B,8,32>();add<1,B,16,32>();add<2,B,8,16>();add<2,B,8,32>(); if constexpr(B==1){add<2,B,16,32>();add<2,B,32,32>();}}
uint16_t rh(double v){
 if(std::isnan(v))return 0x7e00;unsigned sign=std::signbit(v)?0x8000:0;v=std::abs(v);if(v>=65520)return sign|0x7c00;if(v==0)return sign;
 int e;std::frexp(v,&e);e--;if(e<-14)return sign|unsigned(std::nearbyint(std::ldexp(v,24)));
 unsigned m=unsigned(std::nearbyint(std::ldexp(v,10-e)));if(m==2048){e++;m=1024;}return sign|((e+15)<<10)|(m-1024);
}
float hf(double v){return half_float(rh(v));}
struct Data {
 unsigned m,k,n,g,padded;uint32_t rng;int family;
 bool prep=true;std::vector<float>raw;float*draw=nullptr;
 std::vector<Q4Block> q;std::vector<uint16_t>a;std::vector<uint32_t>w;std::vector<uint16_t>s;
 Q4Block*dq=nullptr;unsigned*dw=nullptr;uint16_t*ds=nullptr,*da=nullptr;float*out=nullptr,*partial=nullptr;
 Data(unsigned M,unsigned K,unsigned N,unsigned seed,int fam):m(M),k(K),n(N),g(K/32),padded((M+31)/32*32),rng(seed),family(fam),raw(n*k),q(size_t(m)*g),a(n*k),w(size_t(padded)*g*4),s(size_t(padded)*g){
  for(size_t ai=0;ai<a.size();ai++){auto&v=a[ai];unsigned u=random32(rng);double x=(int(u%8193)-4096)/4093.;if(family==1)x=(int(u%65521)-32760)*2.;if(family==2)x=(u&1?1:-1)*(1.+double(u%1024)/1024);if(family==3)x=(int(u%2049)-1024)*std::ldexp(1.,-24);raw[ai]=float(x);v=rh(raw[ai]);}
  for(unsigned row=0;row<m;row++)for(unsigned z=0;z<g;z++){auto&v=q[size_t(row)*g+z];unsigned u=random32(rng);v.d=rh((u&1?-1:1)*(0.003+double(u%12001)/1000000));for(auto&b:v.qs)b=random32(rng);unsigned v4[4];pack_block(v,0,v4);for(int c=0;c<4;c++)w[word_address(row,z,c,g)]=v4[c];s[scale_address(row,z,g)]=v.d;for(int j=0;j<32;j++)require(unpack_nibble(v4[j/8],j%8)==native_code(v,j),"repack");}
#define ALLOC(P,V) CK(cudaMalloc(&P,V.size()*sizeof(V[0]))); CK(cudaMemcpy(P,V.data(),V.size()*sizeof(V[0]),cudaMemcpyHostToDevice))
 ALLOC(draw,raw);ALLOC(dq,q);ALLOC(dw,w);ALLOC(ds,s);ALLOC(da,a);
#undef ALLOC
 CK(cudaMalloc(&out,size_t(n)*m*4));CK(cudaMalloc(&partial,size_t(n)*m*32*4));
 }
 ~Data(){cudaFree(draw);cudaFree(dq);cudaFree(dw);cudaFree(ds);cudaFree(da);cudaFree(out);cudaFree(partial);}
 void launch(const Cfg&c){require(c.nw*32<=1024&&(!c.fused||c.s*c.r*c.b*32*4<=49152),"launch bounds");if(prep)prepare<<<(n*k+255)/256,256>>>(draw,da,n*k);float*dst=c.fused?out:partial;void*args[]={&dq,&dw,&ds,&da,&dst,&m,&g,&n};unsigned tile=c.r*32*(c.fused?1:c.nw);CK(cudaLaunchKernel(c.kernel,dim3((m+tile-1)/tile,(n+c.b-1)/c.b,c.fused?1:c.s),dim3(c.nw*32),args,c.fused?c.s*c.r*c.b*32*4:0));if(!c.fused)reduce32<<<dim3((m+127)/128,n),128>>>(partial,out,m);CK(cudaGetLastError());}
 std::vector<float> get(const Cfg&c){launch(c);std::vector<float>v(size_t(m)*n);CK(cudaMemcpy(v.data(),out,v.size()*4,cudaMemcpyDeviceToHost));return v;}
 float oracle(const Cfg&c,unsigned row,unsigned b){float acc[32]={};for(int stripe=0;stripe<c.s;stripe++){float h0=0,h1=0;for(unsigned z=stripe;z<g;z+=c.s){auto&v=q[size_t(row)*g+z];float dot=0,x0=0,x1=0;for(int j=0;j<32;j++){float x=half_float(a[b*k+z*32+j]),weight=native_code(v,j);if(c.mode<=0)dot=std::fma(weight,x,dot);else if(j&1)x1=hf(double(weight)*x+x1);else x0=hf(double(weight)*x+x0);}float d=half_float(v.d);if(c.mode==2){h0=hf(double(x0)*d+h0);h1=hf(double(x1)*d+h1);}else{if(c.mode==1)dot=x0+x1;acc[stripe]=std::fma(dot,d,acc[stripe]);}}if(c.mode==2)acc[stripe]=h0+h1;}return tree_sum(acc,c.s);}
};
int main(int argc,char**argv){
 if(argc<7)return 2;unsigned m=atoi(argv[1]),k=atoi(argv[2]),n=atoi(argv[3]),seed=atoi(argv[4]);int fam=atoi(argv[5]);std::string select=argv[6];int rounds=argc>7?atoi(argv[7]):9;
 if(!m||m>20000||!k||k>20000||k%32||!n||n>8||rounds<1||rounds>25)return 2;
 cudaDeviceProp pr;CK(cudaGetDeviceProperties(&pr,0));if(pr.major!=6||pr.minor!=0)return 2;
 cfgs.push_back({"r3r1",-1,1,1,8,32,(void*)r3_200_r1,true});cfgs.push_back({"r3r2",-1,2,1,32,32,(void*)r3_202_r2,true});cfgs.push_back({"r3b4",-1,2,4,8,32,(void*)r3_112_r2,false});
 sets<1>();if(n>1)sets<4>();if(n>4){add<1,8,4,8>();add<1,8,8,16>();add<1,8,8,32>();add<2,8,8,8>();add<2,8,8,16>();}
 if(select!="all"){std::vector<Cfg>sel;std::stringstream ss(select);std::string name;while(std::getline(ss,name,',')){auto it=std::find_if(cfgs.begin(),cfgs.end(),[&](const Cfg&c){return c.name==name;});if(it==cfgs.end())return 2;sel.push_back(*it);}cfgs=sel;}
 Data d(m,k,n,seed,fam);d.prep=argc>8?atoi(argv[8])!=0:true;printf("{\"type\":\"preparation\",\"included\":%s}\n",d.prep?"true":"false");Cfg base={"r3r1",-1,1,1,8,32,(void*)r3_200_r1,true};auto ref=d.get(base);printf("{\"type\":\"data\",\"M\":%u,\"K\":%u,\"N\":%u,\"seed\":%u,\"family\":%d,\"weight_bytes\":%zu,\"activation_bytes\":%zu}\n",m,k,n,seed,fam,d.q.size()*18,d.a.size()*2);
 std::vector<std::vector<float>> original;
 for(auto&c:cfgs){auto v=d.get(c);original.push_back(v);size_t changed=0,nonfinite=0;double err2=0,ref2=0,maxabs=0;
  for(size_t i=0;i<v.size();i++){changed+=float_bits(v[i])!=float_bits(ref[i]);nonfinite+=!std::isfinite(v[i]);if(std::isfinite(v[i])){double er=double(v[i])-ref[i];err2+=er*er;maxabs=std::max(maxabs,std::abs(er));}ref2+=double(ref[i])*ref[i];}
  unsigned checks=m*n<=2048?m*n:64;
  for(unsigned j=0;j<checks;j++){unsigned ix=checks==m*n?j:random32(d.rng)%(m*n);float expected=d.oracle(c,ix%m,ix/m);if((std::isnan(expected)&&std::isnan(v[ix]))||float_bits(expected)==float_bits(v[ix]))continue;fprintf(stderr,"ORACLE FAIL %s ix%u GPU %a CPU %a\n",c.name.c_str(),ix,v[ix],expected);return 5;}
  if(c.mode<=0&&c.s==32&&changed){fprintf(stderr,"Exact baseline failure\n");return 6;}
  cudaFuncAttributes at;CK(cudaFuncGetAttributes(&at,c.kernel));printf("{\"type\":\"check\",\"name\":\"%s\",\"mode\":%d,\"split\":%d,\"different\":%zu,\"outputs\":%zu,\"nonfinite\":%zu,\"finite_relL2\":%.9g,\"max_abs\":%.9g,\"oracle_checks\":%u,\"registers\":%d,\"local_bytes\":%zu}\n",c.name.c_str(),c.mode,c.s,changed,v.size(),nonfinite,ref2?sqrt(err2/ref2):0,maxabs,checks,at.numRegs,at.localSizeBytes);
 }
 auto t=std::chrono::steady_clock::now();unsigned warm=0;do{d.launch(cfgs[warm++%cfgs.size()]);CK(cudaDeviceSynchronize());}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()<0.3);
 cudaEvent_t st,en;CK(cudaEventCreate(&st));CK(cudaEventCreate(&en));
 for(int r=0;r<rounds+1;r++)for(size_t j=0;j<cfgs.size();j++){auto&c=cfgs[(j+r*7)%cfgs.size()];CK(cudaEventRecord(st));for(int i=0;i<12;i++)d.launch(c);CK(cudaEventRecord(en));CK(cudaEventSynchronize(en));float ms;CK(cudaEventElapsedTime(&ms,st,en));printf("{\"type\":\"timing\",\"round\":%d,\"retained\":%s,\"name\":\"%s\",\"us\":%.6f}\n",r,r?"true":"false",c.name.c_str(),ms*1000/12);}
 for(size_t j=0;j<cfgs.size();j++){auto v=d.get(cfgs[j]);if(memcmp(v.data(),original[j].data(),v.size()*4)){fprintf(stderr,"repeatability failure\n");return 7;}}
 CK(cudaEventDestroy(st));CK(cudaEventDestroy(en));puts("{\"type\":\"done\",\"status\":\"PASS\"}");
}
