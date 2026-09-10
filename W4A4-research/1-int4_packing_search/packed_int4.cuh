#pragma once
// Numerical building blocks, not an optimized GEMM and NOT GPU-validated.
// Host integer versions are tested by host_check.cpp. CUDA branches have not
// been compiled here. Use CUDA 12.x with SM60 support, inspect the actual SASS.
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_fp16.h>
#define PACK_HD __host__ __device__ __forceinline__
#else
#define PACK_HD inline
#endif

namespace packed_int4 {
struct Dot2 { int first; int second; };

PACK_HD uint32_t mad_wide_s16(int16_t a, int16_t b, uint32_t c) {
#if defined(__CUDA_ARCH__)
    // PTX operation, NOT a promise that ptxas emits one native instruction.
    uint32_t result;
    asm("mad.wide.s16 %0, %1, %2, %3;"
        : "=r"(result) : "h"(a), "h"(b), "r"(c));
    return result;
#else
    // Each product fits int32; addition intentionally wraps as uint32.
    return c + static_cast<uint32_t>(static_cast<int32_t>(a)*static_cast<int32_t>(b));
#endif
}

// ALL source weights and activations must be integer codes in [-8,7].
// Store original nibbles in HBM; these pack operations are for on-chip staging.
PACK_HD int16_t pack_shared32(int w0, int w1) {
    // Range [-30728,30727]. +2048 centres the high digit in [-7.5,7.5].
    return static_cast<int16_t>(w0 + 4096*w1 + 2048);
}
PACK_HD Dot2 unpack_shared32(uint32_t raw, int activation_sum) {
    // Remove the affine packing offset once per group. All operations that can
    // wrap are unsigned; there is no C++ signed-overflow dependence.
    uint32_t t=raw-static_cast<uint32_t>(2048*activation_sum)+1792u;
    int lo=static_cast<int>(t & 4095u)-1792;
    // Explicit sign extension of the remaining 20-bit field.
    int hi=(static_cast<int>(t>>12)^0x80000)-0x80000;
    return {lo,hi};
}
PACK_HD Dot2 shared_dot32(const int16_t* packed_weights, const int8_t* a) {
    uint32_t c=0;
    int sum_a=0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for(int k=0;k<32;++k) {
        c=mad_wide_s16(packed_weights[k],static_cast<int16_t>(a[k]),c);
        sum_a+=a[k]; // Production: compute once and share across output rows.
    }
    return unpack_shared32(c,sum_a);
}

PACK_HD int16_t pack_independent8(int x0, int x1) {
    // Range [-16392,14343], so signed 16-bit storage is exact.
    return static_cast<int16_t>(x0+2048*x1);
}
PACK_HD Dot2 unpack_independent8(uint32_t raw) {
    constexpr uint32_t B=2048;
    constexpr uint32_t bias=448u + B*896u + B*B*448u;
    uint32_t t=raw+bias;
    return {static_cast<int>(t&2047u)-448,
            static_cast<int>((t>>22)&1023u)-448};
}
PACK_HD Dot2 independent_dots8(const int16_t* packed_w,
                               const int16_t* packed_a) {
    uint32_t c=0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for(int k=0;k<8;++k) c=mad_wide_s16(packed_w[k],packed_a[k],c);
    return unpack_independent8(c);
}

// Eight packed pairs -> ONE sixteen-term dot (reverse activation packing).
PACK_HD int unpack_kpair16(uint32_t raw) {
    uint32_t t=raw+448u+2048u*896u;
    return static_cast<int>((t>>11)&2047u)-896;
}

#ifdef __CUDACC__
// Mandatory baseline: ordinary half2 lanes, NO radix packing. Exactly computes
// two G32 integer dots. Do not multiply floating scales inside this loop.
__device__ __forceinline__ float2 half2_integer_dot32(const __half2* w,
                                                     const __half* a) {
    __half2 c=__float2half2_rn(0.0f);
#pragma unroll
    for(int k=0;k<32;++k) c=__hfma2(w[k],__half2half2(a[k]),c);
    return __half22float2(c);
}
#endif
}
#undef PACK_HD
