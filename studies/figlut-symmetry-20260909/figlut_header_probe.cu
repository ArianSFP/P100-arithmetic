#include "/home/arian/P100-arithmetic/P100-FIGLU-experiment/p100_figlut_experiment/half_lut4_sm60.cuh"

// Minimal compile-only wrappers around the supplied header. These expose one
// R1 G32 contribution so ptxas resources and SM60 instruction counts can be
// audited. They are not complete GEMMs and are never launched by this study.

extern "C" __global__ __launch_bounds__(256, 2)
void figlut_header_shuffle(const half *x, const uint32_t *w, float *out) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    const float xv = __half2float(x[id]);
    const unsigned base = 4u * id;
    out[id] = p100_figlut::q4_g32_shuffle(
        xv, w[base], w[base + 1], w[base + 2], w[base + 3]);
}

extern "C" __global__ __launch_bounds__(256, 2)
void figlut_header_shared(const half *x, const uint32_t *w, float *out) {
    __shared__ float scratch[8][32];
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned warp = threadIdx.x >> 5;
    const float xv = __half2float(x[id]);
    const unsigned base = 4u * id;
    out[id] = p100_figlut::q4_g32_shared(
        xv, w[base], w[base + 1], w[base + 2], w[base + 3], scratch[warp]);
}
