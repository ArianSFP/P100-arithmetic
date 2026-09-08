// Driver-API worker for the previously hashed cubin. No binary modification.
#include <cuda.h>
#include "common.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

void ck(CUresult e,int line) {
    if(e!=CUDA_SUCCESS) { const char *s="?"; cuGetErrorString(e,&s); std::fprintf(stderr,"CUDA %d %s line=%d\n",int(e),s,line); std::exit(3); }
}
#define CK(x) ck((x),__LINE__)
void require(bool ok,const char *s) { if(!ok) { std::fprintf(stderr,"FAIL %s\n",s); std::exit(5); } }
uint32_t random32(uint32_t &s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
using Clock=std::chrono::steady_clock;
CUmodule module,baseline_module;
std::map<std::string,CUfunction> functions;
CUfunction function(const std::string &name) {
    auto it=functions.find(name); if(it!=functions.end()) return it->second;
    CUfunction f;
    bool baseline=name.rfind("baseline:",0)==0;
    CK(cuModuleGetFunction(&f,baseline ? baseline_module : module,baseline ? name.c_str()+9 : name.c_str())); functions[name]=f; return f;
}
struct Buffer {
    CUdeviceptr p=0; size_t bytes;
    explicit Buffer(size_t n):bytes(n) { CK(cuMemAlloc(&p,n)); }
    ~Buffer() { if(p) cuMemFree(p); }
    template<class T> void upload(const std::vector<T> &v) { require(v.size()*sizeof(T)<=bytes,"upload bounds"); CK(cuMemcpyHtoD(p,v.data(),v.size()*sizeof(T))); }
};
struct Config {
    int bits,format,method,reuse,split,old;
    std::string name() const {
        if(method>=10) {
            char s[128];std::snprintf(s,sizeof(s),"baseline:_Z6matvecILi%dELi%dEEvPKjS1_PKfS3_PKsS5_Pfii",bits,method-10);return s;
        }
        char s[128]; std::snprintf(s,sizeof(s),"_Z4rowsILi%dELi%dELi%dELi%dELb%dEEvPKjS1_PKsPKfS5_Pfii",bits,format,method,reuse,old); return s;
    }
};
std::vector<Config> all_configs(int bits,int split) {
    std::vector<Config> c;
    auto add=[&](int f,int m,int r=1,int old=0){c.push_back({bits,f,m,r,split,old});};
    add(0,0); add(0,1); add(2,0); add(2,1); add(1,2);
    for(int r:{1,2,4}) {add(1,0,r);add(1,1,r);for(int m:{0,2,3,4,5})add(3,m,r);}
    add(1,0,1,1);add(1,1,1,1);add(3,0,1,1);add(3,2,1,1);
    require(c.size()==30,"configuration inventory"); return c;
}
struct Packed { Buffer w,scale; Packed(size_t nw,size_t ns):w(nw*4),scale(ns*4){} };
struct Dataset {
    int bits,m,k,n,groups,padded;
    std::vector<int8_t> weights,activations;
    std::vector<uint32_t> awords;
    std::vector<int16_t> asums;
    std::vector<float> scales,ascales;
    std::vector<int> dots;
    std::map<int,std::unique_ptr<Packed>> packed;
    std::map<int,std::vector<float>> references;
    Buffer a,as,asc,partial,output,dummy;
    Dataset(int wb,int rows,int cols,int batch):bits(wb),m(rows),k(cols),n(batch),groups(k/32),padded((m+31)/32*32),
        weights(size_t(m)*k),activations(n*k),awords(n*k/4,0),asums(n*groups,0),scales(m*groups),ascales(n*groups),dots(size_t(n)*m*groups),
        a(n*k),as(n*groups*2),asc(n*groups*4),partial(size_t(n)*m*64*4),output(size_t(n)*m*4),dummy(2) {
        require(k%32==0 && m>0 && n>0,"dimensions");
        uint32_t rng=0x60be0000u^uint32_t(bits)^uint32_t(m*31+k*17+n);
        for(auto &v:weights) v=int8_t(signed_code(random32(rng)&((1<<bits)-1),bits));
        for(int row=0;row<std::min(m,1<<bits);++row) for(int j=0;j<k;++j) weights[row*k+j]=int8_t(signed_code(row,bits));
        for(int b=0;b<n;++b) for(int j=0;j<k;++j) {
            int av=j<32 ? (j&1 ? -128 : 127) : int(random32(rng)&255)-128;
            activations[b*k+j]=int8_t(av); awords[(b*k+j)/4]|=(unsigned(av)&255)<<(8*(j%4)); asums[b*groups+j/32]+=av;
        }
        for(auto &v:scales) v=float((random32(rng)%127)+1)/257.0f;
        for(auto &v:ascales) v=float((random32(rng)%127)+1)/251.0f;
        for(int b=0;b<n;++b) for(int row=0;row<m;++row) for(int g=0;g<groups;++g) {
            int dot=0; for(int j=0;j<32;++j) dot+=int(weights[size_t(row)*k+32*g+j])*int(activations[b*k+32*g+j]);
            dots[(size_t(b)*m+row)*groups+g]=dot;
        }
        a.upload(awords);as.upload(asums);asc.upload(ascales);
    }
    Packed &get(int format,int old) {
        int key=format*2+old; auto it=packed.find(key);if(it!=packed.end()) return *it->second;
        auto start=Clock::now();
        std::vector<uint32_t> wp(size_t(padded)*groups*bits,0);
        std::vector<float> sp(size_t(padded)*groups,0);
        for(int row=0;row<m;++row) for(int g=0;g<groups;++g) {
            sp[scale_address(row,g,groups,old)]=scales[row*groups+g];
            for(int j=0;j<32;++j) {
                unsigned q=unsigned(weights[size_t(row)*k+32*g+j]);
                for(int p=0;p<bits;++p) wp[weight_address(row,g,p,groups,bits,old)]|=(((q>>p)&1)^(format==2))<<bit_position(format,j);
            }
        }
        double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
        auto pack=std::make_unique<Packed>(wp.size(),sp.size());pack->w.upload(wp);pack->scale.upload(sp);
        std::printf("PACK W=%d M=%d K=%d F=%d old=%d cpu_us=%.1f payload_bytes=%zu scale_bytes=%zu\n",bits,m,k,format,old,us,wp.size()*4,sp.size()*4);
        auto &r=*pack;packed[key]=std::move(pack);return r;
    }
    const std::vector<float> &reference(int split) {
        auto it=references.find(split);if(it!=references.end())return it->second;
        std::vector<float> ref(size_t(n)*m,0);
        for(int b=0;b<n;++b) for(int row=0;row<m;++row) {
            if(split==-1) {
                float lanes[32]={};
                for(int lane=0;lane<32;++lane)for(int g=lane;g<groups;g+=32) {
                    float scale=scales[row*groups+g]*ascales[b*groups+g];
                    lanes[lane]=std::fma(float(dots[(size_t(b)*m+row)*groups+g]),scale,lanes[lane]);
                }
                for(int offset=16;offset;offset/=2)for(int lane=0;lane<32-offset;++lane)lanes[lane]+=lanes[lane+offset];
                ref[b*m+row]=lanes[0];continue;
            }
            float total=0;
            for(int s=0;s<split;++s) {
                float sum=0;
                for(int g=s;g<groups;g+=split) {
                    float scale=scales[row*groups+g]*ascales[b*groups+g];
                    sum=std::fma(float(dots[(size_t(b)*m+row)*groups+g]),scale,sum);
                }
                total+=sum;
            }
            ref[b*m+row]=total;
        }
        return references.emplace(split,std::move(ref)).first->second;
    }
    void prepare() {
        int count=n*groups;void *args[]={&a.p,&as.p,&count};
        CK(cuLaunchKernel(function("prepare_asums"),(count+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));
    }
    void launch(const Config &c,bool prep=true) {
        auto &p=get(c.format,c.old);
        if(c.method>=10) {
            require(c.format==0 && c.old==1 && c.reuse==1 && c.split==1,"baseline configuration");
            if(prep && c.method==17)prepare();
            void *args[]={&p.w.p,&a.p,&p.scale.p,&asc.p,&as.p,&dummy.p,&output.p,&m,&groups};
            CK(cuLaunchKernel(function(c.name()),(m+7)/8,n,1,256,1,1,0,nullptr,args,nullptr));return;
        }
        if(prep && (c.method==1 || c.format==2))prepare();
        CUdeviceptr dst=c.split==1 ? output.p : partial.p;
        void *args[]={&p.w.p,&a.p,&as.p,&p.scale.p,&asc.p,&dst,&m,&groups};
        CK(cuLaunchKernel(function(c.name()),(m+128*c.reuse-1)/(128*c.reuse),n,c.split,128,1,1,0,nullptr,args,nullptr));
        if(c.split>1) {
            int splits=c.split;void *rargs[]={&partial.p,&output.p,&m,&splits};
            CK(cuLaunchKernel(function("reduce_splits"),(m+127)/128,n,1,128,1,1,0,nullptr,rargs,nullptr));
        }
    }
    void validate(const Config &c,bool verbose) {
        launch(c);
        std::vector<float> got(size_t(n)*m);CK(cuMemcpyDtoH(got.data(),output.p,got.size()*4));
        const auto &ref=reference(c.method>=10 ? -1 : c.split);
        for(size_t i=0;i<got.size();++i) if(std::memcmp(&got[i],&ref[i],4)) {
            std::fprintf(stderr,"MISMATCH W%d F%d method%d R%d S%d old%d M%d K%d N%d index%zu actual=%a ref=%a\n",bits,c.format,c.method,c.reuse,c.split,c.old,m,k,n,i,double(got[i]),double(ref[i]));std::exit(5);
        }
        if(verbose)std::printf("PASS W=%d M=%d K=%d N=%d F=%d method=%d R=%d S=%d old=%d bit_identical=1\n",bits,m,k,n,c.format,c.method,c.reuse,c.split,c.old);
    }
};

void validate_groups(int bits) {
    constexpr int count=65536;uint32_t rng=0x60ab8000+bits;
    std::vector<int8_t> w(size_t(count)*32);std::vector<uint32_t> a(count*8,0);std::vector<int16_t> as(count,0);std::vector<int> ref(count,0),got(count);
    for(int i=0;i<count;++i) for(int j=0;j<32;++j) {
        int av=i<32 ? (i&1 ? -128 : 127) : int(random32(rng)&255)-128;
        int weight=signed_code((i<32 ? unsigned(i/2) : random32(rng))&((1<<bits)-1),bits);
        w[i*32+j]=int8_t(weight);a[i*8+j/4]|=(unsigned(av)&255)<<(8*(j%4));as[i]+=av;ref[i]+=av*weight;
    }
    Buffer da(a.size()*4),ds(as.size()*2),dw(size_t(count)*bits*4),dout(count*4);da.upload(a);ds.upload(as);
    int n=count;void *prep[]={&da.p,&ds.p,&n};CK(cuLaunchKernel(function("prepare_asums"),count/128,1,1,128,1,1,0,nullptr,prep,nullptr));
    std::vector<int16_t> actual_as(count);CK(cuMemcpyDtoH(actual_as.data(),ds.p,count*2));require(actual_as==as,"GPU activation sums");
    for(int f=0;f<3;++f) {
        std::vector<uint32_t> planes(size_t(count)*bits,0);
        for(int i=0;i<count;++i) for(int j=0;j<32;++j) for(int p=0;p<bits;++p)
            planes[i*bits+p]|=(((unsigned(w[i*32+j])>>p)&1)^(f==2))<<bit_position(f,j);
        dw.upload(planes);char name[100];std::snprintf(name,sizeof(name),"_Z9sad_groupILi%dELi%dEEvPKjS1_PKsPi",bits,f);
        void *args[]={&dw.p,&da.p,&ds.p,&dout.p};CK(cuLaunchKernel(function(name),count/128,1,1,128,1,1,0,nullptr,args,nullptr));
        CK(cuMemcpyDtoH(got.data(),dout.p,count*4));require(got==ref,"SAD group raw outputs");
        std::printf("GROUP_PASS W=%d F=%d groups=%d G=32\n",bits,f,count);
    }
}

void correctness(bool smoke) {
    for(int bits:{2,4}) {
        if(!smoke)validate_groups(bits);
        for(int m:smoke ? std::vector<int>{33} : std::vector<int>{33,129,513}) {
            int k=m==33 ? 96 : m==129 ? 32 : 544;
            Dataset d(bits,m,k,2);
            for(int split:smoke ? std::vector<int>{1,4} : std::vector<int>{1,2,4,8}) {
                auto cfg=all_configs(bits,split);
                for(const auto &c:cfg) {
                    if(smoke && (c.old || c.format!=3 || c.method<3 || c.reuse==2))continue;
                    d.validate(c,!smoke);
                }
            }
        }
    }
    CK(cuCtxSynchronize());std::puts(smoke ? "SMOKE_PASS" : "VALIDATE_PASS all_60_A8_kernels all_split_and_tail_cases");
}

void bench(int bits,int m,int k,int n,bool sweep,const std::string &config_path) {
    std::vector<Config> cfg;
    if(sweep) {
        for(int r:{1,2,4})for(int s:{1,2,4,8,16,32})for(auto fm:std::vector<std::pair<int,int>>{{1,0},{1,1},{3,5},{3,3}})cfg.push_back({bits,fm.first,fm.second,r,s,0});
        cfg.push_back({bits,3,4,1,8,0});cfg.push_back({bits,2,1,1,8,0});cfg.push_back({bits,3,0,1,8,0});cfg.push_back({bits,1,2,1,8,0});
    } else {
        std::ifstream in(config_path);require(bool(in),"config file");Config c;
        while(in>>c.bits>>c.format>>c.method>>c.reuse>>c.split>>c.old)if(c.bits==bits)cfg.push_back(c);
    }
    require(!cfg.empty() && cfg.size()<=100,"configuration count");
    Dataset d(bits,m,k,n);
    for(const auto &c:cfg) {require(c.split>0 && c.split<=64 && (c.reuse==1 || c.reuse==2 || c.reuse==4),"config limits");d.validate(c,true);}
    CK(cuCtxSynchronize());
    auto warm=Clock::now();int launches=0;
    do {d.launch(cfg[launches%cfg.size()]);CK(cuCtxSynchronize());++launches;} while(std::chrono::duration<double>(Clock::now()-warm).count()<0.20);
    std::printf("WARMUP launches=%d seconds=0.20\n",launches);
    CUevent start,stop;CK(cuEventCreate(&start,0));CK(cuEventCreate(&stop,0));
    int rounds=sweep ? 4 : 9;
    for(int round=0;round<rounds;++round)for(size_t order=0;order<cfg.size();++order) {
        const Config &c=cfg[(order+round)%cfg.size()];
        CK(cuEventRecord(start,nullptr));for(int rep=0;rep<3;++rep)d.launch(c);
        CK(cuEventRecord(stop,nullptr));CK(cuEventSynchronize(stop));float ms;CK(cuEventElapsedTime(&ms,start,stop));
        std::printf("TIME W=%d M=%d K=%d N=%d F=%d method=%d R=%d S=%d old=%d round=%d pipeline_us=%.5f\n",bits,m,k,n,c.format,c.method,c.reuse,c.split,c.old,round,ms*1000/3);
    }
    CK(cuEventDestroy(start));CK(cuEventDestroy(stop));std::puts("BENCH_PASS");
}

int main(int argc,char **argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc<3) {std::fprintf(stderr,"usage: gpu-worker CUBIN validate|smoke|sweep|bench [--bits B --m M --k K --n N --configs FILE]\n");return 2;}
    std::string mode=argv[2],configs,baseline;int bits=2,m=5120,k=5120,n=1;
    require(mode=="validate" || mode=="smoke" || mode=="sweep" || mode=="bench","mode");
    for(int i=3;i<argc;i+=2) {
        require(i+1<argc,"argument value");std::string key=argv[i];
        if(key=="--baseline")baseline=argv[i+1];else if(key=="--configs")configs=argv[i+1];else if(key=="--bits")bits=std::stoi(argv[i+1]);else if(key=="--m")m=std::stoi(argv[i+1]);else if(key=="--k")k=std::stoi(argv[i+1]);else if(key=="--n")n=std::stoi(argv[i+1]);else require(false,"unknown argument");
    }
    require((bits==2 || bits==4) && m>0 && m<=17408 && k>0 && k<=17408 && k%32==0 && n>0 && n<=4,"argument range");
    CK(cuInit(0));int count;CK(cuDeviceGetCount(&count));require(count==1,"Exactly one visible GPU required");
    CUdevice dev;CK(cuDeviceGet(&dev,0));int major,minor,sms;CK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev));CK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev));CK(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev));require(major==6 && minor==0,"SM60 required");
    char name[100];CK(cuDeviceGetName(name,sizeof(name),dev));std::printf("DEVICE %s SM=%d.%d sms=%d\n",name,major,minor,sms);
    CUcontext ctx;CK(cuCtxCreate(&ctx,0,dev));CK(cuModuleLoad(&module,argv[1]));
    if(!baseline.empty())CK(cuModuleLoad(&baseline_module,baseline.c_str()));
    if(mode=="validate" || mode=="smoke")correctness(mode=="smoke");else bench(bits,m,k,n,mode=="sweep",configs);
    CK(cuCtxSynchronize());if(!baseline.empty())CK(cuModuleUnload(baseline_module));CK(cuModuleUnload(module));CK(cuCtxDestroy(ctx));return 0;
}
