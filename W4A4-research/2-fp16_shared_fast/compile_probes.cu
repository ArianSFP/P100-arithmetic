#include "packed_shared.cuh"

// Compile-only probes. These do not benchmark throughput or certify GPU results.
extern "C" __global__ void probe_split(const __half2 *p, const __half2 *a,
                                      __half2 *low, __half2 *high) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const Split4 r = split4_exact(p[i], a[i]);
    low[i] = r.low_scaled;
    high[i] = r.high;
}

extern "C" __global__ void probe_split_mac(const __half2 *p, const __half2 *a,
    const __half2 *lin, const __half2 *hin, __half2 *low, __half2 *high) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    __half2 l=lin[i], h=hin[i];
    split4_mac(p[i], a[i], l, h);
    low[i]=l; high[i]=h;
}

extern "C" __global__ void probe_plain_mac(const __half2 *wl, const __half2 *wh,
    const __half2 *a, const __half2 *lin, const __half2 *hin,
    __half2 *low, __half2 *high) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    __half2 l=lin[i], h=hin[i];
    plain4_mac(wl[i], wh[i], a[i], l, h);
    low[i]=l; high[i]=h;
}

extern "C" __global__ void probe_prmt(const uint32_t *tl, const uint32_t *th,
    const uint32_t *selectors, uint32_t *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = residues4(tl[i], th[i], selectors[i]);
}

extern "C" __global__ void probe_one_multiply(const __half2 *p, const __half2 *a,
    const uint32_t *rho_bytes, int2 *out0, int2 *out1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // This variant uses UNNORMALISED p = w_low + 128*w_high.
    const __half2 r = __hmul2_rn(p[i], a[i]);
    const int r0 = __half2int_rn(__low2half(r));
    const int r1 = __half2int_rn(__high2half(r));
    out0[i] = repair_one_from_int(r0, rho_bytes[i]&7u);
    out1[i] = repair_one_from_int(r1, (rho_bytes[i]>>8)&7u);
}
