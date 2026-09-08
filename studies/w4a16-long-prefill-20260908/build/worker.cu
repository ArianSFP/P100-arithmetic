#include "/home/arian/P100-arithmetic/studies/w4a16-tile256-20260908/one_rail512.cuh"
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
#include "/home/arian/P100-arithmetic/studies/w4a16-t64-f32-20260908/kernels.cuh"
#include "/home/arian/P100-arithmetic/studies/w4a16-delivery-leads-20260908/leads.cuh"
static half *lead16; static float *lead32; static unsigned char *leadw; static const float *leadraw;
constexpr int AW_KSTAGE=32, AW_Q8_BLOCK_BYTES=34, AW_T64_ROWS=64, AW_T64_STAGE_BYTES=2176;
#include "aw_control.inc"
#include "aw_q4_control.inc"
#include "current_control.inc"
#include "/home/arian/P100-arithmetic/studies/w4a16-long-prefill-20260908/wide.cuh"
#include "/home/arian/P100-arithmetic/studies/w4a16-long-prefill-20260908/compact.cuh"
#include "/home/arian/P100-arithmetic/studies/w4a16-tile256-20260908/tile256.cuh"
#include "/home/arian/P100-arithmetic/studies/w4a16-tile256-20260908/liveness.cuh"
#include "/home/arian/P100-arithmetic/studies/w4a16-tile256-20260908/direct_stage.cuh"
#include "/home/arian/P100-arithmetic/studies/w4a16-long-prefill-20260908/staging.cuh"
static half *stagedw;
static aw_work_desc *current_desc;

#define CU(x) do {cudaError_t e_=(x);if(e_!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e_));}while(0)
template<class T> struct Device {
    T *p; size_t count;
    explicit Device(size_t n):count(n){CU(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    void put(const std::vector<T>&v){if(v.size()!=count)throw std::runtime_error("upload size");CU(cudaMemcpy(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T>v(count);CU(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
void verify_dequant(){
    Device<unsigned> result(65536u*16);
    decode_check_kernel<<<4096,256>>>(result.p);CU(cudaGetLastError());CU(cudaDeviceSynchronize());
    const auto got=result.get();size_t checked=0,bad=0;
    for(unsigned code=0;code<16;++code)for(unsigned scale=0;scale<65536;++scale){
        if((scale&0x7c00)==0x7c00)continue;
        unsigned short raw=(unsigned short)scale;half d;memcpy(&d,&raw,2);
        const float f=__half2float(d);
        half lo=__float2half_rn(float(int(code)-8)*f),hi=__float2half_rn(float(7-int(code))*f);
        unsigned short lb,hb;memcpy(&lb,&lo,2);memcpy(&hb,&hi,2);
        const unsigned expected=unsigned(lb)|(unsigned(hb)<<16);
        if(got[code*65536+scale]!=expected){
            if(bad<3)printf("DEQUANT_MISMATCH code=%u scale=%04x got=%08x expected=%08x\n",code,scale,got[code*65536+scale],expected);
            ++bad;
        }
        ++checked;
    }
    printf("DEQUANT_CHECK finite_scale_code_pairs=%zu bad=%zu\n",checked,bad);
    if(bad)throw std::runtime_error("packed dequantization failed");
    puts("PASS");
}
uint32_t rng(uint32_t &s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
uint32_t bits(float x){uint32_t b;memcpy(&b,&x,4);return b;}
struct Config {int mt,nt,bt,sk,single,chunk,kind;std::string name;int column=0;};

#define CANDIDATES(X) X(64,64,128,1,1,8)

void launch(const Config&c,const half*a,const unsigned char*w,float*out,int t,int n,int k,int e,int grid,
            aw_work_desc*d32,aw_work_desc*d16,aw_work_desc*d4,aw_tile_desc*tiles,int ntiles) {
    if(c.kind==-4){aw_q8_service_m64_n128_halfpipe_sync<false><<<grid,256>>>(current_desc,tiles,ntiles,n,k);return;}
    if(c.kind==-5){aw_special_q4<false><<<grid,256>>>(current_desc,tiles,ntiles,n,k);return;}
    if(c.kind==1&&c.column>=11000){
        if(c.column==11032)compact_q4<32,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==12032)compact_q4<32,false><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==13032)wide_q4<32,true,true,32><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==14032)tile256_q4<32,3><<<grid,256>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==18032)compact_q4<32,true,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==19032){if(n!=2048||k!=512)throw std::runtime_error("down specialization shape");compact_q4<32,true,true,2048,512><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);}
        else if(c.column==20032){if(n!=512||k!=2048)throw std::runtime_error("gate/up specialization shape");compact_q4<32,true,true,512,2048><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);}
        else if(c.column==21032){if(n!=2048||k!=512)throw std::runtime_error("down chunk shape");compact_q4<32,true,true,2048,512,1><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);}
        else if(c.column==22032){if(n!=512||k!=2048)throw std::runtime_error("gate/up chunk shape");compact_q4<32,true,true,512,2048,1><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);}
        else if(c.column==23032)tile256_direct_stage_q4<<<grid,256>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==25032)tile512_one_rail_q4<<<grid,512>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==24032){if(n!=2048||k!=512)throw std::runtime_error("down cluster shape");compact_q4<32,true,true,2048,512,2><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);}
        else throw std::runtime_error("unknown compact shape");
        return;
    }
    if(c.kind==1&&c.column>=7000){
        if(c.column==7008)wide_q4<8,false><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==7016)wide_q4<16,false><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==7032)wide_q4<32,false><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==8008)wide_q4<8,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==8016)wide_q4<16,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==8032)wide_q4<32,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==9032)wide_q4<32,true,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==10032)wide_q4<32,false,true><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else throw std::runtime_error("unknown wide shape");
        return;
    }
    if(c.kind==1&&c.column>=6000){
        if(c.column==6016)staged_f32<16><<<grid,128>>>(leadraw,stagedw,out,t,n,k,e);
        else if(c.column==6032)staged_f32<32><<<grid,128>>>(leadraw,stagedw,out,t,n,k,e);
        else throw std::runtime_error("unknown staged shape");
        return;
    }
    if(c.kind==1&&c.column>=5000){
        if(c.column==5016)delivery_lead<4,1,16,0><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else if(c.column==5032)delivery_lead<4,1,32,0><<<grid,128>>>(leadraw,leadw,out,t,n,k,e);
        else throw std::runtime_error("unknown long shape");
        return;
    }
    if(c.kind==1){launch_delivery(c.column,a,lead16,lead32,w,leadw,out,t,n,k,e,grid,leadraw);return;}
    if(c.kind==-3){aw_q8_service_m64<false,true><<<grid,256>>>(d32,tiles,ntiles,n,k);return;}
    if(c.kind==-2){aw_q8_service_m64<true,true><<<grid,256>>>(d16,tiles,ntiles,n,k);return;}
    if(c.kind==-1){aw_q4_service_m64<true,true><<<grid,256>>>(d4,tiles,ntiles,n,k);return;}
    #define RUN(M,N,B,S,D,U) if(c.mt==M&&c.nt==N&&c.bt==B&&c.sk==S&&c.single==D&&c.chunk==U){if(c.column)q4_f32_tile<M,N,B,S,D,U,1><<<grid,B>>>(a,w,out,t,n,k,e);else q4_f32_tile<M,N,B,S,D,U,0><<<grid,B>>>(a,w,out,t,n,k,e);return;}
    CANDIDATES(RUN)
    #undef RUN
    throw std::runtime_error("unknown config");
}

int main(int argc,char**argv) try {
    int t=64,n=512,k=2048,experts=16,reps=9,iters=8,seed=3911,pattern=0,gridmul=2,warmup=1000;
    int approved=0,vectorprep=0,prepgrid=0,rawwitness=0;
    int quantcheck=0,scalestress=0;
    std::string only;
    for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing argument");std::string key=argv[i];
        if(key=="--only"){only=argv[i+1];continue;}
        int v=std::stoi(argv[i+1]);
        if(key=="--gpu-approved")approved=v;else if(key=="--quant-check")quantcheck=v;else if(key=="--scale-stress")scalestress=v;else if(key=="--raw-witness")rawwitness=v;else if(key=="--prep-grid")prepgrid=v;else if(key=="--vector-prep")vectorprep=v;else if(key=="--tokens")t=v;else if(key=="--n")n=v;else if(key=="--k")k=v;
        else if(key=="--experts")experts=v;else if(key=="--reps")reps=v;else if(key=="--iters")iters=v;
        else if(key=="--seed")seed=v;else if(key=="--pattern")pattern=v;else if(key=="--grid")gridmul=v;
        else if(key=="--warmup-ms")warmup=v;else throw std::runtime_error("unknown argument");
    }
    if(t<1||t>512||n<128||n>2048||n%128||k<32||k>2048||k%32||experts<1||experts>64||reps<1||reps>30||iters<1||iters>30||pattern<0||pattern>3||gridmul<1||gridmul>8||warmup<0||warmup>5000)throw std::runtime_error("outside reviewed shape limits");
    if(approved!=1)throw std::runtime_error("GPU execution disabled: requires released GPUs and coordinated --gpu-approved 1");
    if(vectorprep<0||vectorprep>1||prepgrid<0||prepgrid>4096)throw std::runtime_error("invalid vector preparation flag/grid");
    printf("VECTOR_PREP %d GRID %d\n",vectorprep,prepgrid);
    if(quantcheck&&scalestress)throw std::runtime_error("quant-check and scale-stress are separate tests");
    if(scalestress&&(scalestress!=1||rawwitness||(only!="tile512_one_rail"&&
        (only!="wide_packed_split_u32"&&only!="wide_f32_split_u32"&&only!="compact_packed_u32"&&only!="compact_scalar_u32"&&only!="mid_packed_u32"&&only!="tile256_cap3"&&only!="compact_vstore"&&only!="compact_down_spec"&&only!="compact_gu_spec"&&only!="compact_down_chunk"&&only!="compact_gu_chunk"&&only!="tile256_direct_stage"&&only!="compact_down_cluster")))
        ) throw std::runtime_error("scale-stress requires an explicit supported split-wide candidate");
    cudaDeviceProp prop{};int count=0;CU(cudaGetDeviceCount(&count));if(count!=1)throw std::runtime_error("exactly one visible GPU required");
    CU(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor!=0)throw std::runtime_error("SM60 required");
    if(quantcheck){if(quantcheck!=1)throw std::runtime_error("invalid quant check");verify_dequant();return 0;}
    int groups=k/32,grid=2*prop.multiProcessorCount;
    size_t ac=size_t(experts)*t*k,oc=size_t(experts)*t*n,wc=size_t(experts)*n*k;
    std::vector<float>a(ac);std::vector<half>ha(ac);std::vector<int8_t>codes(wc);std::vector<half>scales(wc/32);
    std::vector<unsigned char>w4(wc/32*18,0),w8(wc/32*34,0);
    uint32_t rs=seed;
    for(size_t i=0;i<ac;++i){float x=(int(rng(rs)%4097)-2048)/1024.f;
        if(pattern==1)x=std::ldexp(x,int(rng(rs)%17)-8);
        if(pattern==2)x=(i%2?-1.f:1.f)*(1.f+(rng(rs)%8)/1024.f);
        if(pattern==3)x=2048.f;
        ha[i]=__float2half_rn(x);a[i]=__half2float(ha[i]);}
    for(int e=0;e<experts;++e)for(int row=0;row<n;++row)for(int g=0;g<groups;++g){
        size_t bg=(size_t(e)*n+row)*groups+g;
        half scale=__float2half_rn(pattern==3?1/1024.f:((rng(rs)%127)+1)/1024.f);
        if(scalestress){
            unsigned short raw=(unsigned short)(0x1800u+rng(rs)%0x2000u);
            if(rng(rs)&1)raw|=0x8000u;
            memcpy(&scale,&raw,2);
        }
        scales[bg]=scale;
        size_t b4=(size_t(e)*(n/64)*groups+size_t(row/64)*groups+g)*1152;
        size_t b8=(size_t(e)*(n/64)*groups+size_t(row/64)*groups+g)*2176;
        memcpy(w4.data()+b4+2*(row%64),&scale,2);memcpy(w8.data()+b8+2*(row%64),&scale,2);
        for(int j=0;j<32;++j){int q=pattern==3?-8:int(rng(rs)%16)-8;codes[bg*32+j]=int8_t(q);
            w4[b4+128+(row%64)*16+j%16]|=uint8_t(q+8)<<(j<16?0:4);
            w8[b8+128+(row%64)*32+j]=uint8_t(int8_t(q));}
    }
    Device<float> da(ac),out(oc);Device<half> dh(ac);Device<unsigned char>dw4(w4.size()),dw8(w8.size());
    if(rawwitness<0||rawwitness>1)throw std::runtime_error("invalid raw witness");
    std::vector<float> raw(a);
    if(rawwitness){
        const float witnesses[]={1.f+0x1p-11f,1.f+3*0x1p-11f,-1.f-0x1p-11f,-1.f-3*0x1p-11f,
            0x1p-25f,3*0x1p-25f,0x1p-14f+0x1p-25f,-0x1p-14f-0x1p-25f};
        size_t changed=0;
        for(size_t i=0;i<ac;++i){
            raw[i]=i%16<8?witnesses[i%8]:a[i]+std::ldexp(std::max(std::abs(a[i]),0x1p-14f),-12);
            ha[i]=__float2half_rn(raw[i]);a[i]=__half2float(ha[i]);
            changed+=bits(raw[i])!=bits(a[i]);
        }
        if(!changed)throw std::runtime_error("empty conversion witness");
        printf("RAW_WITNESS changed=%zu count=%zu accuracy_only=1\n",changed,ac);
    }
    Device<float> draw(rawwitness?ac:1);
    if(rawwitness)draw.put(raw);
    da.put(a);dh.put(ha);dw4.put(w4);dw8.put(w8);
    std::vector<unsigned char> wtile(w4.size());
    for(size_t b=0;b<w4.size();b+=1152){
        memcpy(wtile.data()+b,w4.data()+b,128);
        for(int row=0;row<64;++row)for(int chunk=0;chunk<4;++chunk)
            memcpy(wtile.data()+b+128+(chunk*64+row)*4,w4.data()+b+128+row*16+chunk*4,4);
    }
    const size_t padded=size_t(experts)*((t+63)/64)*64*k;
    Device<half> dl16(padded);Device<float> dl32(padded);Device<unsigned char>dwt(wtile.size());
    dwt.put(wtile);lead16=dl16.p;lead32=dl32.p;leadw=dwt.p;leadraw=rawwitness?draw.p:da.p;
    printf("SCRATCH tiled_half_bytes=%zu tiled_float_bytes=%zu weight_layout_bytes=%zu\n",padded*2,padded*4,wtile.size());
    std::vector<aw_work_desc>d32,d16,d4;std::vector<aw_tile_desc>tiles;
    for(int e=0;e<experts;++e){
        aw_work_desc d={da.p+size_t(e)*t*k,reinterpret_cast<const char*>(dw8.p)+size_t(e)*n*groups*34,out.p+size_t(e)*t*n,nullptr,0,t,0,e};
        d32.push_back(d);
        d.input=reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(dh.p+size_t(e)*t*k)|uintptr_t(1));d16.push_back(d);
        d.weight=reinterpret_cast<const char*>(dw4.p)+size_t(e)*n*groups*18;d4.push_back(d);
        for(int row=0;row<t;row+=64)tiles.push_back({e,row,std::min(64,t-row)});
    }
    Device<aw_work_desc>dd32(d32.size()),dd16(d16.size()),dd4(d4.size());Device<aw_tile_desc>dt(tiles.size());
    dd32.put(d32);dd16.put(d16);dd4.put(d4);dt.put(tiles);
    Device<half> dsw(wc);stagedw=dsw.p;
    printf("WEIGHT_SCRATCH bytes=%zu prep1_charged_every_run=1\n",wc*2);
    auto current=d32;
    for(int e=0;e<experts;++e)current[e].weight=(const char*)((uintptr_t)(dw4.p+size_t(e)*n*groups*18)|5u);
    Device<aw_work_desc> dc(current.size());dc.put(current);current_desc=dc.p;
    std::vector<Config>cfg={{64,64,256,2,0,16,-3,"aw_t64_f32"},{64,64,256,2,0,16,-2,"aw_t64_a16"},{64,64,256,2,0,16,-1,"aw_q4_a16"}};
    cfg.push_back({64,128,256,2,0,16,-4,"aw_current_q4"});
    cfg.push_back({64,128,256,2,0,16,-5,"aw_special_q4"});
    #define ADD(M,N,B,S,D,U) for(int order=0;order<2;++order)cfg.push_back({M,N,B,S,D,U,0,std::string("q4_m"#M"n"#N"b"#B"s"#S"d"#D"u"#U)+(order?"c":"r"),order});
    CANDIDATES(ADD)
    #undef ADD
    for(int layout=0;layout<4;++layout)for(int bw=0;bw<2;++bw)for(int u:{8,16,32})
        cfg.push_back({64,64,128,1,1,u,1,"delivery_a"+std::to_string(layout)+"b"+std::to_string(bw)+"u"+std::to_string(u),layout*100+bw*10+u});
    for(int layout=0;layout<4;++layout){
        cfg.push_back({64,64,128,1,1,4,1,"delivery_a"+std::to_string(layout)+"b1u4",layout*100+14});
        for(int u:{4,8})cfg.push_back({64,64,128,1,1,u,1,"delivery_a"+std::to_string(layout)+"b1u"+std::to_string(u)+"_4x8",1000+layout*100+10+u});
    }
    for(int u:{4,8,16,32})cfg.push_back({32,64,128,1,1,u,1,"delivery_m32u"+std::to_string(u),2010+u});
    for(int u:{8,16,32})cfg.push_back({16,64,128,1,1,u,1,"delivery_m16u"+std::to_string(u),3010+u});
    for(int u:{16,32})cfg.push_back({32,64,128,1,1,u,1,"delivery_fused_u"+std::to_string(u),4000+u});
    for(int u:{16,32})cfg.push_back({64,64,128,1,1,u,1,"delivery_fused_m64u"+std::to_string(u),5000+u});
    for(int u:{16,32})cfg.push_back({32,64,128,1,1,u,1,"staged_u"+std::to_string(u),6000+u});
    for(int packed:{0,1})for(int u:{8,16,32})cfg.push_back({64,128,128,1,1,u,1,"wide_"+std::string(packed?"packed":"f32")+"_u"+std::to_string(u),7000+packed*1000+u});
    cfg.push_back({64,128,128,2,1,32,1,"wide_packed_split_u32",9032});
    cfg.push_back({64,128,128,2,1,32,1,"wide_f32_split_u32",10032});
    cfg.push_back({32,64,128,2,1,32,1,"compact_packed_u32",11032});
    cfg.push_back({32,64,128,2,1,32,1,"compact_scalar_u32",12032});
    cfg.push_back({32,128,128,2,1,32,1,"mid_packed_u32",13032});
    cfg.push_back({64,64,256,2,1,32,1,"tile256_cap3",14032});
    cfg.push_back({32,64,128,2,1,32,1,"compact_vstore",18032});
    if(n==2048&&k==512)cfg.push_back({32,64,128,2,1,32,1,"compact_down_spec",19032});
    if(n==512&&k==2048)cfg.push_back({32,64,128,2,1,32,1,"compact_gu_spec",20032});
    if(n==2048&&k==512)cfg.push_back({32,64,128,2,1,32,1,"compact_down_chunk",21032});
    if(n==512&&k==2048)cfg.push_back({32,64,128,2,1,32,1,"compact_gu_chunk",22032});
    cfg.push_back({64,64,256,2,1,32,1,"tile256_direct_stage",23032});
    cfg.push_back({64,64,512,2,1,32,1,"tile512_one_rail",25032});
    if(n==2048&&k==512)cfg.push_back({32,64,128,2,1,32,1,"compact_down_cluster",24032});
    if(!only.empty())cfg.erase(std::remove_if(cfg.begin(),cfg.end(),[&](const Config&c){return c.kind>=0&&c.name!=only&&c.name!="q4_m64n64b128s1d1u8r"&&c.name!="delivery_fused_u32";}),cfg.end());
    if(cfg.size()<4)throw std::runtime_error("no candidate");
    auto run=[&](const Config&c,bool prep){
        if(prep&&c.kind==1&&c.column>=6000&&c.column<7000)stage_q4_weights<<<std::min<size_t>((wc+255)/256,4096),256>>>(dw4.p,stagedw,wc);
        if(prep&&c.kind!=-3&&c.kind!=-4&&c.kind!=-5&&!(c.kind==1&&c.column>=4000)) {
            if(c.kind==1&&((c.column%1000)/100==1||(c.column%1000)/100==3))prepare_tiled<false><<<std::min<size_t>(padded/1024,4096),256>>>(da.p,lead16,t,k,experts);
            else if(c.kind==1&&(c.column%1000)/100==2)prepare_tiled<true><<<std::min<size_t>(padded/1024,4096),256>>>(da.p,lead32,t,k,experts);
            else if(vectorprep)prepare_a16_vector<<<std::min<size_t>((ac/8+255)/256,prepgrid?prepgrid:4096),256>>>(da.p,dh.p,ac);
            else prepare_a16<<<std::min<size_t>((ac+255)/256,4096),256>>>(da.p,dh.p,ac);
        }
        launch(c,dh.p,dw4.p,out.p,t,n,k,experts,c.kind<0?grid:prop.multiProcessorCount*gridmul,dd32.p,dd16.p,dd4.p,dt.p,int(tiles.size()));CU(cudaGetLastError());};
    printf("META tokens=%d n=%d k=%d experts=%d seed=%d pattern=%d sm=%d candidate_grid=%d warmup_ms=%d f32_only=1\n",t,n,k,experts,seed,pattern,prop.multiProcessorCount,gridmul,warmup);fflush(stdout);
    if(scalestress){
        if(scalestress!=1||rawwitness)throw std::runtime_error("invalid scale stress");
        cfg.erase(std::remove_if(cfg.begin(),cfg.end(),[](const Config&c){
            return c.column!=25032&&c.kind!=-4&&c.kind!=-5&&c.column!=9032&&c.column!=10032&&c.column!=11032&&c.column!=12032&&c.column!=13032&&c.column!=14032&&c.column!=18032&&c.column!=19032&&c.column!=20032&&c.column!=21032&&c.column!=22032&&c.column!=23032&&c.column!=24032;
        }),cfg.end());
        if(std::none_of(cfg.begin(),cfg.end(),[&](const Config&c){return c.name==only;}))
            throw std::runtime_error("requested scale-stress candidate missing");
        puts("SCALE_STRESS correctness_only=1 half_rounded_weights=1");
    }
    int compact_blocks=0,lead_blocks=0;
    CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&compact_blocks,compact_q4<32,true>,128,0));
    CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&lead_blocks,delivery_lead<4,1,32,2>,128,0));
    printf("OCCUPANCY compact_packed_blocks=%d old_fused_blocks=%d\n",compact_blocks,lead_blocks);
    std::vector<float>reference,sequential;
    const size_t samples=oc<=16384?oc:256;
    for(const auto&c:cfg){CU(cudaMemset(out.p,0xff,oc*4));run(c,true);CU(cudaDeviceSynchronize());auto got=out.get();
        if(c.kind==-3||(scalestress&&c.kind==-4))reference=got;
        if(c.kind==0)sequential=got;
        if(c.kind==1&&c.sk==1){
            if(sequential.empty())throw std::runtime_error("missing frozen winner");
            for(size_t i=0;i<oc;++i)if(bits(got[i])!=bits(sequential[i]))throw std::runtime_error("delivery differs from frozen sequential winner");
        }
        double err2=0,ref2=0,maxabs=0;size_t nonfinite=0,bad=0,changed=0;
        for(size_t i=0;i<oc;++i){if(!std::isfinite(got[i])){nonfinite++;continue;}double d=double(got[i])-reference[i];err2+=d*d;ref2+=double(reference[i])*reference[i];maxabs=std::max(maxabs,std::abs(d));changed+=bits(got[i])!=bits(reference[i]);}
        uint32_t sample=uint32_t(seed)^9921;
        for(size_t ix=0;ix<samples;++ix){size_t index=samples==oc?ix:rng(sample)%oc;
            int col=index%n,row=(index/n)%t,e=index/(size_t(n)*t);float split[2]={};
            for(int g=0;g<groups;++g){float scale=__half2float(scales[(size_t(e)*n+col)*groups+g]);
                for(int j=0;j<32;++j){float q=codes[((size_t(e)*n+col)*groups+g)*32+j];float x=a[(size_t(e)*t+row)*k+g*32+j];
                    const int rail=c.sk==2?j/16:0;split[rail]=std::fma(x,scalestress?__half2float(__float2half_rn(q*scale)):q*scale,split[rail]);}}
            float expected=c.sk==2?split[0]+split[1]:split[0];
            if(bits(got[index])!=bits(expected)){if(bad<3)printf("MISMATCH %s %zu %.9g %.9g\n",c.name.c_str(),index,got[index],expected);bad++;}
        }
        const double rel=std::sqrt(err2/std::max(ref2,1e-300));
        printf("CHECK %s outputs=%zu samples=%zu bad=%zu changed=%zu nonfinite=%zu rel_l2=%.10g maxabs=%.10g\n",c.name.c_str(),oc,samples,bad,changed,nonfinite,rel,maxabs);fflush(stdout);
        if(bad||nonfinite||(c.sk==2&&changed)||rel>0.00005)throw std::runtime_error("arithmetic gate failed");
    }
    if(scalestress){puts("PASS");return 0;}
    cudaEvent_t start,end;CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));
    const auto warmend=std::chrono::steady_clock::now()+std::chrono::milliseconds(warmup);
    do {for(const auto&c:cfg)run(c,true);CU(cudaDeviceSynchronize());} while(std::chrono::steady_clock::now()<warmend);
    puts("WARMUP_COMPLETE");fflush(stdout);
    for(int rep=0;rep<reps;++rep)for(size_t j=0;j<cfg.size();++j){const auto&c=cfg[(j+rep)%cfg.size()];
        for(int pi=0;pi<2;++pi){const int prep=(pi+rep)%2;CU(cudaEventRecord(start));for(int it=0;it<iters;++it)run(c,prep);CU(cudaEventRecord(end));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,start,end));
            printf("TIME %s prep=%d rep=%d us=%.9g\n",c.name.c_str(),prep,rep,ms*1000/iters);fflush(stdout);}}
    CU(cudaEventDestroy(start));CU(cudaEventDestroy(end));puts("PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"STOP: %s\n",e.what());return 2;}
