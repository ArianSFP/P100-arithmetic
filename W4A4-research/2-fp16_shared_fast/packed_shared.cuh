#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// UNCOMPILED / UNTIMED CUDA CANDIDATES.
// CPU numerical identities are verified by verify.py and host_check.cpp.
// Target: SM60, CUDA 12.x toolchain. Inspect SASS before benchmarking.
// All weights and activations below are RAW signed INT4 codes in [-8,7].
// Do not apply quantization scales until after extracting separate outputs.

struct Split4 {
    __half2 low_scaled; // two low-row products, each divided by 128
    __half2 high;       // two high-row products
};

// Each half of p is w_high + w_low/128, represented exactly in FP16.
// Each half of a is the shared integer activation for that output-row pair.
// Set both halves of a equal to share one activation across all four outputs.
// Three half2 arithmetic operations; sign/source handling depends on compiler.
__device__ __forceinline__ Split4 split4_exact(__half2 p, __half2 a) {
    const __half2 magic = __float2half2_rn(1536.0f);
    __half2 t  = __hfma2(p, a, magic);
    __half2 hi = __hsub2_rn(t, magic);
    __half2 lo = __hfma2(p, a, __hneg2(hi));
    return {lo, hi};
}

// Five half2 arithmetic operations total. Not a claim of a speed advantage.
// Reset these accumulators every <=32 terms for a full-domain exact guarantee.
// low_acc remains scaled by 1/128; remove this factor in the FP32 epilogue.
__device__ __forceinline__ void split4_mac(
    __half2 p, __half2 a, __half2 &low_acc, __half2 &high_acc) {
    const Split4 v = split4_exact(p, a);
    low_acc  = __hadd2_rn(low_acc, v.low_scaled);
    high_acc = __hadd2_rn(high_acc, v.high);
}

// Matched exact-INT4 baseline: two half2 FMAs for the same four useful MACs.
__device__ __forceinline__ void plain4_mac(
    __half2 low_weights, __half2 high_weights, __half2 a,
    __half2 &low_acc, __half2 &high_acc) {
    low_acc  = __hfma2(low_weights, a, low_acc);
    high_acc = __hfma2(high_weights, a, high_acc);
}

// Alternative: retain ONE FP16 multiply per two products, with integer repair.
// Caller computes r=RN16((w_low+128*w_high)*a), then converts r exactly to int.
// rho=(w_low*a) mod8. This conversion and output accumulation are NOT free.
// On the targeted CUDA device, signed right shift is arithmetic.
__device__ __forceinline__ int2 repair_one_from_int(int r, unsigned int rho) {
    const int hi = (r + 63) >> 7;
    const int delta = ((static_cast<int>(rho) - r + 4) & 7) - 4;
    return make_int2(r - 128*hi + delta, hi);
}

// Supply two registers containing the eight bytes [(a*j) mod8, j=0..7].
// low_nibbles contains FOUR two's-complement low-row weight codes, packed
// in bits 0..3,4..7,8..11,12..15. Their low 3 bits are enough modulo8.
// The four output bytes are rho0,rho1,rho2,rho3, each in 0..7.
// This is ONE PRMT, plus any selector preparation. It is not four full repairs.
__device__ __forceinline__ uint32_t residues4(
    uint32_t table_lo, uint32_t table_hi, uint32_t low_nibbles) {
    uint32_t out;
    const uint32_t selector = low_nibbles & 0x7777u;
    asm("prmt.b32 %0, %1, %2, %3;"
        : "=r"(out) : "r"(table_lo), "r"(table_hi), "r"(selector));
    return out;
}
