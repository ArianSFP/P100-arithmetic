#pragma once
#include <cstdint>
#include <cstdlib>
#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

// F0: byte-striped LSB first; F1: byte-striped MSB first;
// F2: F1 with offline-complemented bits; F3: consecutive quartet nibbles.
HD int coefficient(int p,int bits) { return p==bits-1 ? -(1<<p) : 1<<p; }
HD int signed_code(unsigned q,int bits) { return int(q)-int((q>>(bits-1))<<bits); }
HD int bit_position(int format,int j) {
    return format==3 ? j : 8*(j%4)+(format==0 ? j/4 : 7-j/4);
}
HD uint32_t sign_bytes(uint32_t x) {
#ifdef __CUDA_ARCH__
    uint32_t r;
    asm volatile("prmt.b32 %0, %1, %2, 0xba98;" : "=r"(r) : "r"(x),"r"(0u));
    return r;
#else
    uint32_t r=0;
    for(int j=0;j<4;++j) r|=255u*((x>>(8*j+7))&1)<<(8*j);
    return r;
#endif
}
HD int sad4(uint32_t a,uint32_t b,int c) {
#ifdef __CUDA_ARCH__
    int r;
    asm volatile("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a),"r"(b),"r"(c));
    return r;
#else
    for(int j=0;j<4;++j) c+=std::abs(int((a>>(8*j))&255)-int((b>>(8*j))&255));
    return c;
#endif
}
HD int population(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}
HD int vmad4(uint32_t w,uint32_t a,int c) {
#ifdef __CUDA_ARCH__
    asm volatile("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
                 "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
                 "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
                 "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
                 : "+r"(c) : "r"(w),"r"(a));
#else
    for(int j=0;j<4;++j) c+=signed_code((w>>(8*j))&255,8)*signed_code((a>>(8*j))&255,8);
#endif
    return c;
}
template<int F> HD unsigned index4(uint32_t plane,int q) {
    if constexpr(F==3) return (plane>>(4*q))&15;
    uint32_t x=plane>>(F==0 ? q : 7-q);
    return (x&1)|((x>>7)&2)|((x>>14)&4)|((x>>21)&8);
}
template<int W,int F> HD uint32_t decode4(const uint32_t (&planes)[W],int q) {
    uint32_t wb=0;
    if constexpr(F==3) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            unsigned code=0;
            #pragma unroll
            for(int p=0;p<W;++p) code|=((planes[p]>>(4*q+j))&1)<<p;
            wb|=(unsigned(signed_code(code,W))&255)<<(8*j);
        }
    } else {
        #pragma unroll
        for(int p=0;p<W-1;++p) wb|=((planes[p]>>(F==0 ? q : 7-q))&0x01010101u)<<p;
        constexpr uint32_t high=(256u-(1u<<(W-1)))*0x01010101u;
        wb|=sign_bytes(planes[W-1]<<(F==0 ? 7-q : q))&high;
    }
    return wb;
}
HD int table_entry(uint32_t a,unsigned index) {
    int sum=0;
    #pragma unroll
    for(int j=0;j<4;++j) if(index&(1<<j)) sum+=signed_code((a>>(8*j))&255,8);
    return sum;
}
HD int table_entry_cast(uint32_t a,unsigned index) {
    int sum=0;
    #pragma unroll
    for(int j=0;j<4;++j) if(index&(1<<j)) sum+=int(int8_t(a>>(8*j)));
    return sum;
}
template<int W> HD int combine(const int (&sums)[W]) {
    int total=0;
    #pragma unroll
    for(int p=0;p<W;++p) total+=coefficient(p,W)*sums[p];
    return total;
}
template<int W,int F> HD int finish_sad(const uint32_t (&planes)[W],const int (&sums)[W],int asum) {
    int stored_sum=0;
    #pragma unroll
    for(int p=0;p<W;++p) stored_sum+=coefficient(p,W)*population(planes[p]);
    int numerator;
    if constexpr(F==2) {
        // Complemented two's-complement codes satisfy w=-1-stored_w.
        numerator=combine(sums)-asum+128*32+stored_sum;
    } else numerator=-combine(sums)-asum-128*32-stored_sum;
#ifdef __CUDA_ARCH__
    int r;
    asm("shr.s32 %0, %1, 1;" : "=r"(r) : "r"(numerator));
    return r;
#else
    if(numerator%2) std::abort();
    return numerator/2;
#endif
}
// Word indices; rows must be padded to a multiple of 32 for tiled storage.
HD uint64_t weight_address(int row,int g,int p,int groups,int bits,bool old) {
    return old ? (uint64_t(row)*bits+p)*groups+g
               : ((uint64_t(row/32)*groups+g)*bits+p)*32+row%32;
}
HD uint64_t scale_address(int row,int g,int groups,bool old) {
    return old ? uint64_t(row)*groups+g : (uint64_t(row/32)*groups+g)*32+row%32;
}
