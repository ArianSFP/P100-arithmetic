#include <cstdint>

// The destination is a read-write operand so the SASS destination register is
// initialized independently of the third FMA operand. This is required before
// interpreting MRG_H0/MRG_H1 behavior.
extern "C" __global__ void merge_probe(const std::uint32_t * a,
                                        const std::uint32_t * b,
                                        const std::uint32_t * c,
                                        const std::uint32_t * seed,
                                        std::uint32_t * out,
                                        std::uint32_t * side,
                                        unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const std::uint32_t av = a[i];
    const std::uint32_t bv = b[i];
    const std::uint32_t cv = c[i];
    volatile const std::uint32_t initial = seed[i];
    std::uint32_t result = initial;
    side[i] = result;
    asm volatile("membar.gl;" ::: "memory");
    asm volatile(
        "fma.rn.f16x2 %0, %1, %2, %3;"
        : "+r"(result)
        : "r"(av), "r"(bv), "r"(cv));
    out[i] = result;
    side[i] = initial;
}
