#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

struct Q4Block { uint16_t d; uint8_t qs[16]; };
static_assert(sizeof(Q4Block)==18,"native Q4_0 block size");
HD inline int native_code(const Q4Block &b,int j) { return int((b.qs[j&15]>>(4*(j>>4)))&15)-8; }
HD inline size_t word_address(int row,int g,int p,int groups,bool old=false) {
    return old ? (size_t(row)*4+p)*groups+g : (((size_t(row/32)*groups+g)*4+p)*32+row%32);
}
HD inline size_t scale_address(int row,int g,int groups,bool old=false) {
    return old ? size_t(row)*groups+g : (size_t(row/32)*groups+g)*32+row%32;
}
HD inline unsigned signed_nibble(int value) { return unsigned(value)&15; }
HD inline int unpack_nibble(uint32_t word,int j) { return int(((word>>(4*(j&7)))&15)^8)-8; }

inline int64_t half_fixed(uint16_t h) {
    unsigned e=(h>>10)&31,m=h&1023;
    if(e==31) throw std::runtime_error("nonfinite in rational oracle");
    int64_t value=e ? int64_t(1024+m)<<(e-1) : m;
    return h&0x8000 ? -value : value;
}
inline float half_float(uint16_t h) {
    unsigned e=(h>>10)&31,m=h&1023;
    if(e==31) { uint32_t b=(uint32_t(h&0x8000)<<16)|0x7f800000u|(m<<13);float f;std::memcpy(&f,&b,4);return f; }
    float value=std::ldexp(float(half_fixed(h)),-24);
    return h==0x8000 ? -0.0f : value;
}
inline uint32_t float_bits(float f) {uint32_t x;std::memcpy(&x,&f,4);return x;}
inline uint32_t random32(uint32_t &s) {s^=s<<13;s^=s>>17;s^=s<<5;return s;}
inline void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
inline void pack_block(const Q4Block &b,int format,uint32_t (&words)[4]) {
    for(auto &w:words)w=0;
    for(int j=0;j<32;++j) {
        unsigned code=signed_nibble(native_code(b,j));
        if(format==0)words[j/8]|=code<<(4*(j%8));
        else for(int p=0;p<4;++p)words[p]|=((code>>p)&1)<<j;
    }
}
inline int unpack_block(const uint32_t (&words)[4],int format,int j) {
    if(format==0)return unpack_nibble(words[j/8],j);
    int code=0;for(int p=0;p<4;++p)code|=((words[p]>>j)&1)<<p;
    return (code^8)-8;
}
inline float tree_sum(float *values,int splits) {
    for(int offset=splits/2;offset;offset/=2)for(int i=0;i<offset;++i)values[i]+=values[i+offset];
    return values[0];
}
