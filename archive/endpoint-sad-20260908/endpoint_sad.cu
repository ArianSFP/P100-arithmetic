// Isolated SM60 formulation comparison. Synthetic signed W2/W4, G=32.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"CUDA error %d %s at line %d\n",int(e),cudaGetErrorString(e),__LINE__); \
    std::exit(3); } } while(0)
constexpr int G=32, METHODS=9;
const char *names[METHODS]={"cpp_integer","vmad_integer","masked_popc","endpoint_popc",
                           "masked_metadata","endpoint_metadata","endpoint_local_asum",
                           "endpoint_direct_popc","endpoint_direct_metadata"};
uint32_t random32(uint32_t &s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
int sign_extend(unsigned q,int bits) { return (q&(1u<<(bits-1))) ? int(q)-(1<<bits) : int(q); }

__host__ __device__ __forceinline__ uint32_t sad4(uint32_t a,uint32_t b,uint32_t c) {
#ifdef __CUDA_ARCH__
    uint32_t r;
    asm volatile("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a),"r"(b),"r"(c));
    return r;
#else
    for(int j=0;j<4;++j) c+=std::abs(int((a>>(8*j))&255)-int((b>>(8*j))&255));
    return c;
#endif
}
__host__ __device__ __forceinline__ uint32_t byte_mask(uint32_t plane,int quad) {
    uint32_t shifted=plane<<(7-quad), r;
#ifdef __CUDA_ARCH__
    asm volatile("prmt.b32 %0, %1, %2, 0xba98;" : "=r"(r) : "r"(shifted),"r"(0u));
#else
    r=0;
    for(int j=0;j<4;++j) r|=(((shifted>>(8*j+7))&1)*255u)<<(8*j);
#endif
    return r;
}
__host__ __device__ __forceinline__ int population(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}
__host__ __device__ __forceinline__ int vmad4(uint32_t w,uint32_t a,int c) {
#ifdef __CUDA_ARCH__
    asm volatile("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
                 "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
                 "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
                 "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
                 : "+r"(c) : "r"(w),"r"(a));
#else
    for(int j=0;j<4;++j) c+=int(int8_t(w>>(8*j)))*int(int8_t(a>>(8*j)));
#endif
    return c;
}

// Plane bit 8*byte+quad describes activation 4*quad+byte.
// This offline bit permutation preserves the W-bit payload size.
template<int W,int METHOD>
__host__ __device__ __forceinline__ int group_dot(const uint32_t (&planes)[W],const uint32_t *a,
                                        int activation_sum,int stored_weight_sum) {
    constexpr bool direct=METHOD==7 || METHOD==8;
    constexpr bool endpoint=METHOD==3 || METHOD==5 || METHOD==6 || direct;
    int weight_sum=0;
    if (METHOD>=2) {
        if (METHOD==4 || METHOD==5 || METHOD==8) weight_sum=stored_weight_sum;
        else {
            #pragma unroll
            for(int p=0;p<W;++p) weight_sum+=(p==W-1 ? -(1<<p) : 1<<p)*population(planes[p]);
        }
    }
    uint32_t references[W], sums[W];
    #pragma unroll
    for(int p=0;p<W;++p) { references[p]=(endpoint && !direct) ? ~planes[p] : planes[p]; sums[p]=0; }
    int integer_sum=0;
    uint32_t local_asum=0;
    #pragma unroll
    for(int q=0;q<8;++q) {
        uint32_t av=a[q];
        if (METHOD<2) {
            uint32_t wb=0;
            #pragma unroll
            for(int p=0;p<W-1;++p) wb|=((planes[p]>>q)&0x01010101u)<<p;
            constexpr uint32_t high_mask=(256u-(1u<<(W-1)))*0x01010101u;
            wb|=byte_mask(planes[W-1],q)&high_mask;
            if(METHOD==1) integer_sum=vmad4(wb,av,integer_sum);
            else {
                #pragma unroll
                for(int j=0;j<4;++j) integer_sum+=int(int8_t(wb>>(8*j)))*int(int8_t(av>>(8*j)));
            }
        } else {
            uint32_t u=av^0x80808080u;
            if(METHOD==6) local_asum=sad4(u,0,local_asum);
            #pragma unroll
            for(int p=0;p<W;++p) {
                uint32_t mask=byte_mask(references[p],q);
                if(endpoint) sums[p]=sad4(u,mask,sums[p]);
                else sums[p]=sad4(u&mask,0,sums[p]);
            }
        }
    }
    if(METHOD<2) return integer_sum;
    int combined=0;
    #pragma unroll
    for(int p=0;p<W;++p) combined+=(p==W-1 ? -(1<<p) : 1<<p)*int(sums[p]);
    if(endpoint) {
        if(METHOD==6) activation_sum=int(local_asum)-128*G;
        int numerator=direct ? -combined-activation_sum-128*G-weight_sum
                             : combined-activation_sum+127*G-weight_sum;
        int result;
        // The proven numerator is even, including for negative dots.
#ifdef __CUDA_ARCH__
        asm("shr.s32 %0, %1, 1;" : "=r"(result) : "r"(numerator));
#else
        if(numerator%2) std::abort();
        result=numerator/2;
#endif
        return result;
    }
    return combined-128*weight_sum;
}

__global__ void prepare_asums(const uint32_t *a,int16_t *s,int groups) {
    int g=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(g>=groups) return;
    uint32_t sum=0;
    #pragma unroll
    for(int q=0;q<8;++q) sum=sad4(a[8*g+q]^0x80808080u,0,sum);
    s[g]=int16_t(int(sum)-128*G);
}

template<int W,int METHOD>
__global__ void dot_probe(const uint32_t *w,const uint32_t *a,const int16_t *as,
                          const int16_t *ws,int *out,int n) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=n) return;
    uint32_t planes[W];
    #pragma unroll
    for(int p=0;p<W;++p) planes[p]=w[p*n+i];
    int asum=0,wsum=0;
    if(METHOD==3 || METHOD==5 || METHOD==7 || METHOD==8) asum=as[i];
    if(METHOD==4 || METHOD==5 || METHOD==8) wsum=ws[i];
    out[i]=group_dot<W,METHOD>(planes,a+8*i,asum,wsum);
}

template<int W,int METHOD>
__global__ void matvec(const uint32_t *w,const uint32_t *a,const float *wscale,
                       const float *ascale,const int16_t *as,const int16_t *ws,
                       float *out,int rows,int groups) {
    int lane=int(threadIdx.x)&31;
    int row=int(blockIdx.x)*8+(int(threadIdx.x)>>5);
    if(row>=rows) return;
    int batch=int(blockIdx.y);
    float acc=0;
    for(int g=lane;g<groups;g+=32) {
        uint32_t planes[W];
        #pragma unroll
        for(int p=0;p<W;++p) planes[p]=w[(row*W+p)*groups+g];
        int asum=0,wsum=0;
        if(METHOD==3 || METHOD==5 || METHOD==7 || METHOD==8) asum=as[batch*groups+g];
        if(METHOD==4 || METHOD==5 || METHOD==8) wsum=ws[row*groups+g];
        int dot=group_dot<W,METHOD>(planes,a+8*(batch*groups+g),asum,wsum);
        float scale=wscale[row*groups+g]*ascale[batch*groups+g];
        acc=fmaf(float(dot),scale,acc);
    }
    #pragma unroll
    for(int offset=16;offset;offset>>=1) acc+=__shfl_down_sync(0xffffffff,acc,offset);
    if(lane==0) out[batch*rows+row]=acc;
}

template<int W> void launch_probe(int method,const uint32_t *w,const uint32_t *a,
    const int16_t *as,const int16_t *ws,int *out,int n) {
    #define CASE(M) case M: dot_probe<W,M><<<(n+127)/128,128>>>(w,a,as,ws,out,n); break;
    switch(method) { CASE(0) CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) }
    #undef CASE
    CK(cudaGetLastError());
}
template<int W> void launch_matvec(int method,const uint32_t *w,const uint32_t *a,
    const float *ds,const float *da,const int16_t *as,const int16_t *ws,float *out,
    int m,int groups,int n,bool prepare) {
    if(prepare && (method==3 || method==5 || method==7 || method==8)) {
        prepare_asums<<<(n*groups+127)/128,128>>>(a,const_cast<int16_t *>(as),n*groups);
        CK(cudaGetLastError());
    }
    dim3 grid((m+7)/8,n);
    #define CASE(M) case M: matvec<W,M><<<grid,256>>>(w,a,ds,da,as,ws,out,m,groups); break;
    switch(method) { CASE(0) CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) }
    #undef CASE
    CK(cudaGetLastError());
}

int endpoint_reference(const std::vector<int> &a,const std::vector<int> &w,int bits,bool is_signed) {
    int weighted_d=0,weighted_e=0,asum=0,wsum=0;
    for(size_t j=0;j<a.size();++j) { asum+=a[j]; wsum+=w[j]; }
    for(int p=0;p<bits;++p) {
        int dp=0,ep=0;
        for(size_t j=0;j<a.size();++j) {
            int bit=int((unsigned(w[j])>>p)&1);
            dp+=std::abs(a[j]+128-255*(1-bit));
            ep+=std::abs(a[j]+128-255*bit);
        }
        int coefficient=is_signed && p==bits-1 ? -(1<<p) : 1<<p;
        weighted_d+=coefficient*dp;
        weighted_e+=coefficient*ep;
    }
    int coefficient_sum=is_signed ? -1 : (1<<bits)-1;
    int numerator=weighted_d+coefficient_sum*asum-127*coefficient_sum*int(a.size())-wsum;
    int direct_numerator=coefficient_sum*asum+128*coefficient_sum*int(a.size())-wsum-weighted_e;
    if(numerator!=direct_numerator) { std::fprintf(stderr,"endpoint identities differ\n"); std::exit(5); }
    if(numerator%2) { std::fprintf(stderr,"non-even numerator\n"); std::exit(5); }
    return numerator/2;
}
template<int W> int packed_cpu_tests() {
    uint32_t rng=0x60e0b000+W;
    constexpr int cases=65536;
    for(int i=0;i<cases;++i) {
        uint32_t planes[W]={},a[8]={};
        int reference=0,asum=0,wsum=0;
        for(int j=0;j<32;++j) {
            int av=i<32 ? (i&1 ? -128 : 127) : int(random32(rng)&255)-128;
            unsigned qw=i<32 ? (unsigned(i)>>1)&((1<<W)-1) : random32(rng)&((1<<W)-1);
            int wv=sign_extend(qw,W);
            for(int p=0;p<W;++p) planes[p]|=((qw>>p)&1)<<(8*(j%4)+j/4);
            a[j/4]|=(unsigned(av)&255)<<(8*(j%4));
            reference+=av*wv; asum+=av; wsum+=wv;
        }
        int psum=0;
        for(int p=0;p<W;++p) psum+=(p==W-1 ? -(1<<p) : 1<<p)*population(planes[p]);
        uint32_t usum=0;
        for(int q=0;q<8;++q) usum=sad4(a[q]^0x80808080u,0,usum);
        if(psum!=wsum || int(usum)-128*G!=asum) return 5;
        #define HOST(M) group_dot<W,M>(planes,a,asum,wsum)
        const int actual[]={HOST(0),HOST(1),HOST(2),HOST(3),HOST(4),HOST(5),HOST(6),HOST(7),HOST(8)};
        #undef HOST
        for(int method=0;method<METHODS;++method) if(actual[method]!=reference) {
            std::fprintf(stderr,"CPU packed mismatch W=%d method=%s case=%d\n",W,names[method],i);
            return 5;
        }
    }
    std::printf("CPU PASS packed W=%d groups=%d methods=%d group_size=32\n",W,cases,METHODS);
    return 0;
}

int affine_cpu_tests() {
    // Q2_K-style algebra, not GGUF byte-layout validation. Integer numerators
    // stand for scales with a common denominator. G16 weights intersect G32 A.
    uint32_t rng=0x60e0aff2;
    for(int trial=0;trial<4096;++trial) {
        int dw=int(random32(rng)&255),dm=int(random32(rng)&255);
        int64_t reference=0,endpoint=0;
        for(int ag=0;ag<8;++ag) {
            int da=int(random32(rng)&255);
            for(int sub=0;sub<2;++sub) {
                int scale=int(random32(rng)&15),minimum=int(random32(rng)&15),asum=0;
                std::vector<int> a(16),q(16);
                for(int j=0;j<16;++j) {
                    a[j]=int(random32(rng)&255)-128; q[j]=int(random32(rng)&3);
                    asum+=a[j];
                    reference+=int64_t(da)*a[j]*(dw*scale*q[j]-dm*minimum);
                }
                int dot=endpoint_reference(a,q,2,false);
                endpoint+=int64_t(da)*(dw*scale*dot-dm*minimum*asum);
            }
        }
        if(reference!=endpoint) return 5;
    }
    std::puts("CPU PASS Q2_K-style affine algebra superblocks=4096 G_weight=16 G_activation=32 integer_scale_numerators=1");
    return 0;
}

int cpu_tests() {
    uint32_t rng=0x60e0d002;
    for(int bits : {2,4}) {
        uint64_t count=0;
        for(bool is_signed : {true,false}) {
            for(int a=-128;a<128;++a) for(int q=0;q<(1<<bits);++q) {
                int w=is_signed ? sign_extend(q,bits) : q;
                if(endpoint_reference({a},{w},bits,is_signed)!=a*w) return 5;
                ++count;
            }
            for(int len : {4,16,32,64,128}) {
                int runs=len==4 ? 262144 : 4096;
                for(int i=0;i<runs;++i) {
                    std::vector<int> a(len),w(len);
                    int direct=0;
                    for(int j=0;j<len;++j) {
                        a[j]=int(random32(rng)&255)-128;
                        unsigned q=random32(rng)&((1<<bits)-1);
                        w[j]=is_signed ? sign_extend(q,bits) : int(q);
                        direct+=a[j]*w[j];
                    }
                    if(endpoint_reference(a,w,bits,is_signed)!=direct) return 5;
                    ++count;
                }
            }
        }
        std::printf("CPU PASS bits=%d signed_and_unsigned=1 cases=%llu seed=0x60e0d002\n",bits,(unsigned long long)count);
    }
    // A negative control: differing scales cannot be moved outside a sum.
    std::vector<int> a0(16,-128),w0(16,-2),a1(16,127),w1(16,1);
    int d0=endpoint_reference(a0,w0,2,true),d1=endpoint_reference(a1,w1,2,true);
    if(2*d0+3*d1==2*(d0+d1)) return 5;
    std::puts("CPU PASS common-scale-group boundary negative control");
    if(packed_cpu_tests<2>() || packed_cpu_tests<4>() || affine_cpu_tests()) return 5;
    return 0;
}

template<class T> T *upload(const std::vector<T> &v) {
    T *p; CK(cudaMalloc(&p,v.size()*sizeof(T)));
    CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
template<int W> void correctness() {
    constexpr int n=65536;
    uint32_t rng=0x60e0d000+W;
    std::vector<uint32_t> w(n*W),a(n*8);
    std::vector<int16_t> as(n),ws(n);
    std::vector<int> ref(n),out(n);
    for(int i=0;i<n;++i) {
        for(int p=0;p<W;++p) w[p*n+i]=random32(rng);
        for(int q=0;q<8;++q) a[i*8+q]=random32(rng);
        // Include extreme signed A8 and all-zero/all-one weight planes.
        if(i<32) {
            for(int p=0;p<W;++p) w[p*n+i]=(i>>(p+1))&1 ? 0xffffffffu : 0;
            for(int q=0;q<8;++q) a[i*8+q]=i&1 ? 0x80808080u : 0x7f7f7f7fu;
        }
        int sa=0,sw=0,dot=0;
        for(int j=0;j<32;++j) {
            int av=sign_extend((a[i*8+j/4]>>(8*(j%4)))&255,8);
            unsigned qw=0;
            for(int p=0;p<W;++p) qw|=((w[p*n+i]>>(8*(j%4)+j/4))&1)<<p;
            int wv=sign_extend(qw,W);
            sa+=av; sw+=wv; dot+=av*wv;
        }
        as[i]=int16_t(sa); ws[i]=int16_t(sw); ref[i]=dot;
    }
    auto dw=upload(w),da=upload(a); auto ds=upload(as),dws=upload(ws);
    int *dr; CK(cudaMalloc(&dr,n*sizeof(int)));
    // Verify the actually timed activation preparation, not just host metadata.
    prepare_asums<<<(n+127)/128,128>>>(da,ds,n); CK(cudaGetLastError());
    std::vector<int16_t> actual_as(n);
    CK(cudaMemcpy(actual_as.data(),ds,n*sizeof(int16_t),cudaMemcpyDeviceToHost));
    if(actual_as!=as) { std::fprintf(stderr,"activation prep mismatch\n"); std::exit(5); }
    for(int method=0;method<METHODS;++method) {
        launch_probe<W>(method,dw,da,ds,dws,dr,n);
        CK(cudaMemcpy(out.data(),dr,n*sizeof(int),cudaMemcpyDeviceToHost));
        if(out!=ref) { std::fprintf(stderr,"group mismatch W=%d method=%s\n",W,names[method]); std::exit(5); }
        std::printf("GPU PASS W=%d method=%s groups=%d group_size=32\n",W,names[method],n);
    }
    CK(cudaFree(dw)); CK(cudaFree(da)); CK(cudaFree(ds)); CK(cudaFree(dws)); CK(cudaFree(dr));
}

template<int W> void benchmark(int m,int k,int n) {
    int groups=k/G;
    uint32_t rng=0x60e00000+W+m+k+n;
    std::vector<uint32_t> w(size_t(m)*W*groups),a(n*groups*8);
    std::vector<float> ds(m*groups),da(n*groups);
    std::vector<int16_t> as(n*groups),ws(m*groups);
    for(auto &v:w) v=random32(rng);
    for(auto &v:a) v=random32(rng);
    for(auto &v:ds) v=float(1u<<(random32(rng)%3))/256;
    for(auto &v:da) v=float(1u<<(random32(rng)%3))/128;
    for(int row=0;row<m;++row) for(int g=0;g<groups;++g) {
        int sum=0;
        for(int p=0;p<W;++p) sum+=(p==W-1 ? -(1<<p) : 1<<p)*__builtin_popcount(w[(row*W+p)*groups+g]);
        ws[row*groups+g]=int16_t(sum);
    }
    auto dw=upload(w),dact=upload(a); auto dwscale=upload(ds),dascale=upload(da);
    auto das=upload(as),dws=upload(ws);
    float *dr; CK(cudaMalloc(&dr,m*n*sizeof(float)));
    std::vector<float> reference(m*n),actual(m*n);
    // Both integer baselines and all SAD arms use the same FP32 scale/reduction order.
    launch_matvec<W>(0,dw,dact,dwscale,dascale,das,dws,dr,m,groups,n,true);
    CK(cudaMemcpy(reference.data(),dr,m*n*sizeof(float),cudaMemcpyDeviceToHost));
    for(int method=0;method<METHODS;++method) {
        launch_matvec<W>(method,dw,dact,dwscale,dascale,das,dws,dr,m,groups,n,true);
        CK(cudaMemcpy(actual.data(),dr,m*n*sizeof(float),cudaMemcpyDeviceToHost));
        if(std::memcmp(reference.data(),actual.data(),m*n*sizeof(float))) {
            std::fprintf(stderr,"scaled matvec mismatch W=%d method=%s M=%d K=%d N=%d\n",W,names[method],m,k,n); std::exit(5);
        }
    }
    std::printf("MATVEC PASS W=%d M=%d K=%d N=%d all_methods_bit_identical=1 payload_bpw=%d metadata_extra_bpw=0.5\n",W,m,k,n,W);
    cudaEvent_t start,stop; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
    for(int rep=0;rep<2;++rep) for(int method=0;method<METHODS;++method)
        launch_matvec<W>(method,dw,dact,dwscale,dascale,das,dws,dr,m,groups,n,true);
    CK(cudaDeviceSynchronize());
    // Rotate the order across rounds. Three pipeline launches per timed sample.
    for(int round=0;round<7;++round) for(int order=0;order<METHODS;++order) {
        int method=(order+round)%METHODS;
        CK(cudaEventRecord(start));
        for(int rep=0;rep<3;++rep) launch_matvec<W>(method,dw,dact,dwscale,dascale,das,dws,dr,m,groups,n,true);
        CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
        float ms; CK(cudaEventElapsedTime(&ms,start,stop));
        std::printf("TIME W=%d M=%d K=%d N=%d method=%s round=%d pipeline_us=%.4f\n",W,m,k,n,names[method],round,ms*1000/3);
    }
    // Preparation is measured separately but is already charged above where used.
    CK(cudaEventRecord(start));
    for(int rep=0;rep<20;++rep) prepare_asums<<<(n*groups+127)/128,128>>>(dact,das,n*groups);
    CK(cudaGetLastError()); CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
    float ms; CK(cudaEventElapsedTime(&ms,start,stop));
    std::printf("PREP W=%d M=%d K=%d N=%d activation_sum_us=%.4f\n",W,m,k,n,ms*1000/20);
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(stop));
    CK(cudaFree(dw)); CK(cudaFree(dact)); CK(cudaFree(dwscale)); CK(cudaFree(dascale));
    CK(cudaFree(das)); CK(cudaFree(dws)); CK(cudaFree(dr));
}

int main(int argc,char **argv) {
    if(argc==2 && !std::strcmp(argv[1],"--cpu")) return cpu_tests();
    if(argc!=2 || std::strcmp(argv[1],"--gpu")) { std::fprintf(stderr,"usage: endpoint-sad --cpu|--gpu\n"); return 2; }
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    CK(cudaSetDevice(0)); cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=6 || prop.minor!=0) return 4;
    std::printf("DEVICE name=%s sm=%d.%d sms=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);
    correctness<2>(); correctness<4>();
    const int shapes[][3]={{5120,5120,1},{17408,5120,1},{5120,17408,1},{5120,5120,4}};
    for(const auto &shape : shapes) {
        benchmark<2>(shape[0],shape[1],shape[2]);
        benchmark<4>(shape[0],shape[1],shape[2]);
    }
    CK(cudaDeviceSynchronize());
    return 0;
}
