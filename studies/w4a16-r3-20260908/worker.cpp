#include <cuda.h>
#include "common.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

void ck(CUresult e,int line) {if(e!=CUDA_SUCCESS){const char *s="?";cuGetErrorString(e,&s);std::fprintf(stderr,"CUDA_ERROR %d %s line=%d\n",int(e),s,line);std::exit(3);}}
#define CK(x) ck((x),__LINE__)
using Clock=std::chrono::steady_clock;
CUmodule module,original_module,round2_module;
std::map<std::string,CUfunction> functions;
CUfunction function(const std::string &name) {
    auto it=functions.find(name);if(it!=functions.end())return it->second;
    CUmodule owner=name.rfind("r3_",0)==0 ? module : name.rfind("r2_",0)==0 ? round2_module : original_module;
    CUfunction f;CK(cuModuleGetFunction(&f,owner,name.c_str()));functions[name]=f;return f;
}
struct Buffer {
    CUdeviceptr p=0;size_t bytes;
    explicit Buffer(size_t n):bytes(n){CK(cuMemAlloc(&p,n));}
    ~Buffer(){if(p)cuMemFree(p);}
    template<class T>void upload(const std::vector<T> &v){require(v.size()*sizeof(T)<=bytes,"upload bounds");CK(cuMemcpyHtoD(p,v.data(),v.size()*sizeof(T)));}
};
struct Config {
    int method,reuse,split;
    int base()const{return method>=10 ? 2 : method%5;}
    bool wide()const{return method>=5 && method<10;}
    int format()const{return method==16 || method==17 || method==34 ? 2 : 0;}
    bool fused()const{return method>=200;}
    int warps()const {
        if(fused())return method%10==0 ? 8 : method%10==1 ? 16 : 32;
        if(method>=100)return method%10==0 ? 1 : method%10==1 || method%10==4 ? 2 : method%10==2 ? 8 : 4;
        return 4;
    }
    int batch_reuse()const{return method==36 || (method>=110 && method<200) || method>=210 ? 4 : 1;}
    bool old()const{return base()==1;}
    bool lut()const{return base()==4;}
    std::string name()const {
        if(method>=100)return "r3_"+std::to_string(method)+"_r"+std::to_string(reuse);
        if(method>=10)return "r2_"+std::to_string(method)+"_r"+std::to_string(reuse);
        std::string prefix=base()==0 ? "native_direct" : base()==1 ? "old_repacked" :
            std::string(base()==2 ? "direct_r" : base()==3 ? "plane_direct_r" : "lut_r")+std::to_string(reuse);
        return prefix+(wide() ? "_wide" : "");
    }
};
std::vector<Config> configs(bool sweep) {
    (void)sweep;std::vector<Config> out={{14,2,32},{36,2,32}};
    for(int m=100;m<=104;++m)for(int r:{1,2,4})out.push_back({m,r,32});
    for(int m=110;m<=114;++m)for(int r:{1,2})out.push_back({m,r,32});
    for(int m:{200,201,202,210,211,212})for(int r:{1,2})out.push_back({m,r,32});
    return out;
}
struct Packed {Buffer words,scales;Packed(size_t groups):words(groups*16),scales(groups*2){}};
struct Dataset {
    int m,k,n,groups,padded,family;
    std::vector<Q4Block> native;
    std::vector<uint16_t> activation;
    std::vector<float> af,gd,gl,sd;
    std::vector<int64_t> fixed;
    std::vector<long double> exact,l1;
    std::map<int,std::unique_ptr<Packed>> packed;
    std::map<int,std::vector<float>> expected;
    Buffer dn,da,wide_a,partial,output;
    Dataset(int rows,int cols,int batch,int kind):m(rows),k(cols),n(batch),groups(k/32),padded((m+31)/32*32),family(kind),
        native(size_t(m)*groups),activation(n*k),af(n*k),gd(size_t(n)*m*groups),gl(gd.size()),sd(size_t(m)*groups),fixed(n*k),exact(size_t(n)*m),l1(exact.size()),
        dn(native.size()*18),da(activation.size()*2),wide_a(activation.size()*4),partial(size_t(n)*m*64*4),output(size_t(n)*m*4) {
        uint32_t rng=0x4166b000u^uint32_t(m*31+k*17+n+family*65537);
        const uint16_t edge[]={0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x3c00,0xbc00,0x7bff,0xfbff,0x3555,0xb555,0x3c01,0xbc01};
        for(size_t i=0;i<activation.size();++i) {
            unsigned u=random32(rng),bits;
            if(family==1)bits=(u&0x8000)|((u%31)<<10)|((u>>16)&1023);
            else if(family==2)bits=edge[i%16];
            else if(family==3)bits=i%31 ? (i&1 ? 0x8000 : 0) : edge[(i/31)%16];
            else if(family==4){bits=i&65535;if(((bits>>10)&31)==31)bits=0;}
            else bits=(u&0x8000)|((7+(u%17))<<10)|((u>>16)&1023);
            activation[i]=uint16_t(bits);af[i]=half_float(bits);fixed[i]=half_fixed(bits);
        }
        for(int row=0;row<m;++row)for(int g=0;g<groups;++g) {
            auto &b=native[size_t(row)*groups+g];
            unsigned u=random32(rng);
            b.d=uint16_t((u&0x8000)|((7+(u%9))<<10)|((u>>16)&1023));
            if(family==1)b.d=uint16_t((u&0x8000)|((u%31)<<10)|((u>>16)&1023));
            if(family==2)b.d=edge[(row+g)%16];
            sd[size_t(row)*groups+g]=half_float(b.d);
            for(int j=0;j<16;++j)b.qs[j]=uint8_t(row<16 ? row|(row<<4) : random32(rng));
        }
        std::vector<float> tables(size_t(n)*groups*8*16);
        for(int batch=0;batch<n;++batch)for(int g=0;g<groups;++g)for(int q=0;q<8;++q)for(int idx=0;idx<16;++idx) {
            float t=0;for(int j=0;j<4;++j)if(idx&(1<<j))t+=af[(batch*groups+g)*32+4*q+j];
            tables[((batch*groups+g)*8+q)*16+idx]=t;
        }
        for(int batch=0;batch<n;++batch)for(int row=0;row<m;++row) {
            __int128 total=0;unsigned __int128 absolute=0;
            for(int g=0;g<groups;++g) {
                const auto &b=native[size_t(row)*groups+g];float direct=0,sums[4]={};int64_t group=0;uint64_t mass=0;
                for(int j=0;j<32;++j) {
                    int code=native_code(b,j),aindex=(batch*groups+g)*32+j;
                    // FP16 times signed W4 is exact in FP32; addition rounds once.
                    direct+=float(code)*af[aindex];
                    int64_t term=code*fixed[aindex];group+=term;mass+=uint64_t(term<0 ? -term : term);
                }
                for(int q=0;q<8;++q)for(int p=0;p<4;++p) {
                    int idx=0;for(int j=0;j<4;++j)idx|=((signed_nibble(native_code(b,4*q+j))>>p)&1)<<j;
                    sums[p]+=tables[((batch*groups+g)*8+q)*16+idx];
                }
                float lookup=0;for(int p=0;p<4;++p)lookup=std::fma(float(p==3 ? -8 : 1<<p),sums[p],lookup);
                size_t pos=(size_t(batch)*m+row)*groups+g;gd[pos]=direct;gl[pos]=lookup;
                int64_t scale=half_fixed(b.d);total+=__int128(group)*scale;absolute+=(unsigned __int128)mass*uint64_t(scale<0 ? -scale : scale);
            }
            exact[batch*m+row]=std::ldexp((long double)total,-48);
            l1[batch*m+row]=std::ldexp((long double)absolute,-48);
        }
        dn.upload(native);da.upload(activation);
        std::printf("DATA M=%d K=%d N=%d family=%d activation=FP16 activation_bytes=%zu native_weight_bytes=%zu seed_formula=0x4166b000_xor_shape_family\n",m,k,n,family,activation.size()*2,native.size()*18);
    }
    Packed &get(int format,bool old) {
        int key=format*2+old;auto it=packed.find(key);if(it!=packed.end())return *it->second;
        auto start=Clock::now();std::vector<uint32_t> wp(size_t(padded)*groups*4,0);std::vector<uint16_t> sp(size_t(padded)*groups,0);
        for(int row=0;row<m;++row)for(int g=0;g<groups;++g) {
            const auto &b=native[size_t(row)*groups+g];uint32_t w[4];pack_block(b,0,w);
            for(int j=0;j<32;++j)require(unpack_block(w,0,j)==native_code(b,j),"lossless repack check");
            for(int p=0;p<4;++p) {
                size_t index=format==2 ? ((size_t(row/32)*groups+g)*32+row%32)*4+p : word_address(row,g,p,groups,old);
                wp[index]=w[p];
            }
            sp[scale_address(row,g,groups,old)]=b.d;
        }
        double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
        auto p=std::make_unique<Packed>(size_t(padded)*groups);p->words.upload(wp);p->scales.upload(sp);
        std::printf("PACK F=%d old=%d cpu_us=%.1f bytes=%zu includes_host_roundtrip_check=1\n",format,int(old),us,wp.size()*4+sp.size()*2);
        auto &result=*p;packed[key]=std::move(p);return result;
    }
    const std::vector<float> &reference(bool lut,int split) {
        int key=split+128*lut;auto it=expected.find(key);if(it!=expected.end())return it->second;
        std::vector<float> ref(size_t(n)*m);
        for(int batch=0;batch<n;++batch)for(int row=0;row<m;++row) {
            float v[64]={};
            for(int s=0;s<split;++s)for(int g=s;g<groups;g+=split) {
                size_t pos=(size_t(batch)*m+row)*groups+g;
                v[s]=std::fma(lut ? gl[pos] : gd[pos],sd[size_t(row)*groups+g],v[s]);
            }
            ref[batch*m+row]=tree_sum(v,split);
        }
        return expected.emplace(key,std::move(ref)).first->second;
    }
    void launch(const Config &c) {
        CUdeviceptr w=0,s=0,dst=c.fused() || c.base()<2 || c.split==1 ? output.p : partial.p;
        CUdeviceptr act=c.wide() ? wide_a.p : da.p;
        if(c.base()) {auto &p=get(c.format(),c.old());w=p.words.p;s=p.scales.p;}
        if(c.wide()) {
            int count=n*k;void *wargs[]={&da.p,&wide_a.p,&count};
            CK(cuLaunchKernel(function("widen_inputs"),(count+255)/256,1,1,256,1,1,0,nullptr,wargs,nullptr));
        }
        void *args[]={&dn.p,&w,&s,&act,&dst,&m,&groups,&n};
        if(c.base()<2)CK(cuLaunchKernel(function(c.name()),(m+7)/8,n,1,256,1,1,0,nullptr,args,nullptr));
        else {
            int threads=c.warps()*32;
            int tile=c.fused() ? c.reuse*32 : threads*c.reuse;
            unsigned smem=c.fused() ? 32*c.reuse*c.batch_reuse()*32*sizeof(float) : 0;
            require(threads<=1024 && smem<=48*1024,"launch resource bounds");
            CK(cuLaunchKernel(function(c.name()),(m+tile-1)/tile,(n+c.batch_reuse()-1)/c.batch_reuse(),c.fused() ? 1 : c.split,threads,1,1,smem,nullptr,args,nullptr));
            if(c.split>1 && !c.fused()){void *rargs[]={&partial.p,&output.p,&m};CK(cuLaunchKernel(function("reduce_"+std::to_string(c.split)),(m+127)/128,n,1,128,1,1,0,nullptr,rargs,nullptr));}
        }
    }
    void validate(const Config &c) {
        launch(c);std::vector<float> got(size_t(n)*m);CK(cuMemcpyDtoH(got.data(),output.p,got.size()*4));
        const auto &ref=reference(c.lut(),c.split),&baseline=reference(false,32);
        size_t changed=0;long double maxerr=0,maxnorm=0,square=0,reference_square=0;
        for(size_t i=0;i<got.size();++i) {
            if(float_bits(got[i])!=float_bits(ref[i])) {
                std::fprintf(stderr,"MISMATCH %s S%d family%d index%zu got=%a ref=%a\n",c.name().c_str(),c.split,family,i,double(got[i]),double(ref[i]));std::exit(5);
            }
            changed+=float_bits(got[i])!=float_bits(baseline[i]);
            long double err=std::fabs((long double)got[i]-exact[i]);maxerr=std::max(maxerr,err);
            long double norm=l1[i] ? err/l1[i] : err;maxnorm=std::max(maxnorm,norm);square+=err*err;reference_square+=exact[i]*exact[i];
            long double budget=(64+(groups+c.split-1)/c.split+c.split)*std::ldexp(4.0L,-24);
            require(std::isfinite(got[i]) && norm<=budget,"rational forward-error bound");
        }
        if(!c.lut() && c.split==32)require(changed==0,"strict baseline byte identity");
        std::printf("CHECK method=%d R=%d S=%d M=%d K=%d N=%d family=%d operation_order_bitexact=1 baseline_changed=%zu outputs=%zu max_abs=%.9Le max_L1_relative=%.9Le relative_L2=%.9Le\n",c.method,c.reuse,c.split,m,k,n,family,changed,got.size(),maxerr,maxnorm,reference_square ? std::sqrt(square/reference_square) : 0.0L);
    }
};

void conversion_test() {
    Buffer out(65536*4);void *args[]={&out.p};CK(cuLaunchKernel(function("convert_all"),256,1,1,256,1,1,0,nullptr,args,nullptr));
    std::vector<float> got(65536);CK(cuMemcpyDtoH(got.data(),out.p,got.size()*4));
    for(unsigned i=0;i<65536;++i) {
        float want=half_float(i);
        if(std::isnan(want))require(std::isnan(got[i]),"NaN conversion classification");
        else require(float_bits(want)==float_bits(got[i]),"GPU exact half conversion including signed zero/subnormals");
    }
    std::puts("CONVERSION_PASS patterns=65536 finite_and_infinity_bitexact=1 NaN_classification=1");
}
void correctness(bool smoke) {
    conversion_test();auto cfg=configs(false);
    for(int family=0;family<(smoke ? 1 : 4);++family)for(int m:smoke ? std::vector<int>{33} : std::vector<int>{33,129,513}) {
        Dataset d(m,m==33 ? 96 : m==129 ? 32 : 544,m==33 ? 1 : m==129 ? 3 : 4,family);
        for(const auto &c:cfg)if(!smoke || c.base()<2 || c.split==32 || c.split==4)d.validate(c);
    }
    if(!smoke){Dataset d(16,16384,4,4);for(const auto &c:cfg)d.validate(c);std::puts("EXHAUSTIVE_FINITE_ACTIVATIONS_PASS finite_patterns=63488 weight_codes=16");}
    std::puts(smoke ? "SMOKE_PASS" : "VALIDATE_PASS");
}
void benchmark(int m,int k,int n,bool sweep,const std::string &path) {
    std::vector<Config> cfg;
    if(sweep)cfg=configs(true);
    else {std::ifstream in(path);require(bool(in),"configuration file");Config c;while(in>>c.method>>c.reuse>>c.split)cfg.push_back(c);}
    require(!cfg.empty() && cfg.size()<=100,"configuration count");
    auto allowed=configs(false);
    for(const auto &c:cfg)require(std::any_of(allowed.begin(),allowed.end(),[&](Config x){return x.method==c.method && x.reuse==c.reuse && x.split==c.split;}),"configuration bounds");
    Dataset d(m,k,n,0);for(const auto &c:cfg)d.validate(c);
    auto begin=Clock::now();int warmups=0;
    do {d.launch(cfg[warmups%cfg.size()]);CK(cuCtxSynchronize());++warmups;}while(std::chrono::duration<double>(Clock::now()-begin).count()<0.2);
    std::printf("WARMUP seconds=0.2 launches=%d\n",warmups);
    CUevent start,stop;CK(cuEventCreate(&start,0));CK(cuEventCreate(&stop,0));
    for(int round=0;round<(sweep ? 4 : 9);++round)for(size_t i=0;i<cfg.size();++i) {
        const auto &c=cfg[(i+round)%cfg.size()];CK(cuEventRecord(start,nullptr));
        for(int rep=0;rep<3;++rep)d.launch(c);
        CK(cuEventRecord(stop,nullptr));CK(cuEventSynchronize(stop));float ms;CK(cuEventElapsedTime(&ms,start,stop));
        std::printf("TIME M=%d K=%d N=%d method=%d R=%d S=%d round=%d pipeline_us=%.5f\n",m,k,n,c.method,c.reuse,c.split,round,ms*1000/3);
    }
    CK(cuEventDestroy(start));CK(cuEventDestroy(stop));std::puts("BENCH_PASS");
}
int main(int argc,char **argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        require(argc>=5,"usage: worker CUBIN ROUND2_CUBIN ROUND1_CUBIN validate|smoke|sweep|bench [--m M --k K --n N --configs FILE]");
        std::string mode=argv[4],path;int m=5120,k=5120,n=1;
        require(mode=="validate" || mode=="smoke" || mode=="sweep" || mode=="bench","mode");
        for(int i=5;i<argc;i+=2){require(i+1<argc,"argument value");std::string key=argv[i];if(key=="--configs")path=argv[i+1];else if(key=="--m")m=std::stoi(argv[i+1]);else if(key=="--k")k=std::stoi(argv[i+1]);else if(key=="--n")n=std::stoi(argv[i+1]);else require(false,"argument");}
        require(m>0 && m<=17408 && k>0 && k<=17408 && k%32==0 && n>0 && n<=4,"dimensions");
        CK(cuInit(0));int count;CK(cuDeviceGetCount(&count));require(count==1,"one visible GPU");
        CUdevice dev;CK(cuDeviceGet(&dev,0));int major,minor;CK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev));CK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev));require(major==6 && minor==0,"SM60 only");
        CUcontext ctx;CK(cuCtxCreate(&ctx,0,dev));CK(cuModuleLoad(&module,argv[1]));
        CK(cuModuleLoad(&round2_module,argv[2]));CK(cuModuleLoad(&original_module,argv[3]));
        if(mode=="validate" || mode=="smoke")correctness(mode=="smoke");else benchmark(m,k,n,mode=="sweep",path);
        CK(cuCtxSynchronize());CK(cuModuleUnload(module));CK(cuModuleUnload(round2_module));CK(cuModuleUnload(original_module));CK(cuCtxDestroy(ctx));
    } catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 5;}
}
