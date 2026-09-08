#include <cstdint>

// Keep this kernel deliberately boring. The output is the raw 32-bit result
// of one documented packed-half FMA; all instruction selection is inspected
// in the generated cubin rather than inferred from the CUDA source.
extern "C" __global__ void probe(const std::uint32_t * a,
                                  const std::uint32_t * b,
                                  const std::uint32_t * c,
                                  std::uint32_t * out,
                                  unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const std::uint32_t av = a[i];
    const std::uint32_t bv = b[i];
    const std::uint32_t cv = c[i];
    std::uint32_t result;

    asm volatile(
        "fma.rn.f16x2 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(av), "r"(bv), "r"(cv));

    out[i] = result;
}
