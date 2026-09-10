#include <cuda_runtime.h>
#include <stdint.h>

// Compile-only register-LUT probes.  One lane owns four output columns and one
// uint32 byte-pack represents four A4 tokens.  Each kernel performs one exact
// G32 raw-code contribution (512 useful scalar W4A4 MACs/thread).  pos/neg are
// already prepared packed-byte activation magnitudes; their preparation and
// the G32 scale/correction finish are intentionally outside this core probe.

constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ uint32_t add3(uint32_t a, uint32_t b, uint32_t c) {
    // Ptxas for SM60 normally contracts this source expression to IADD3.
    return a + b + c;
}

template<bool HALF8>
__global__ __launch_bounds__(256, 2)
void activation_register_lut_core(const uint32_t *__restrict__ pos_input,
                                  const uint32_t *__restrict__ neg_input,
                                  const uint32_t *__restrict__ key_input,
                                  uint64_t *__restrict__ output) {
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned id = blockIdx.x * blockDim.x + tid;
    const unsigned pos_self = pos_input[id];
    const unsigned neg_self = neg_input[id];

    // Four output columns x four offset-code weight bit planes.  Each word has
    // eight mu4 keys.  Full16 stores raw masks; half8 stores {flip,index}.
    uint32_t key[4][4];
    #pragma unroll
    for (int o = 0; o < 4; ++o) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const unsigned word = unsigned(o * 4 + p);
            key[o][p] = key_input[word * gridDim.x * blockDim.x + id];
        }
    }

    uint32_t acc[4][4] = {};
    if constexpr (!HALF8) {
        // Two 16-entry tables occupy the two warp halves. Four waves cover G32.
        #pragma unroll
        for (unsigned wave = 0; wave < 4; ++wave) {
            const unsigned table = lane >> 4;
            const unsigned mask = lane & 15u;
            const unsigned base = wave * 8u + table * 4u;
            uint32_t value = 0;
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const uint32_t pos = __shfl_sync(FULL_MASK, pos_self, base + j);
                const uint32_t neg = __shfl_sync(FULL_MASK, neg_self, base + j);
                const uint32_t selected = (mask & (1u << j)) ? pos : neg;
                value += selected;
            }
            #pragma unroll
            for (int o = 0; o < 4; ++o) {
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const unsigned m0 = (key[o][p] >> (8u * wave)) & 15u;
                    const unsigned m1 = (key[o][p] >> (8u * wave + 4u)) & 15u;
                    const uint32_t v0 = __shfl_sync(FULL_MASK, value, m0);
                    const uint32_t v1 = __shfl_sync(FULL_MASK, value, 16u + m1);
                    acc[o][p] = add3(acc[o][p], v0, v1);
                }
            }
        }
    } else {
        // Four canonical 8-entry tables occupy four 8-lane subgroups. Two
        // waves cover G32. Bit 3 is fixed clear in the stored table.
        #pragma unroll
        for (unsigned wave = 0; wave < 2; ++wave) {
            const unsigned table = lane >> 3;
            const unsigned mask = lane & 7u;
            const unsigned base = wave * 16u + table * 4u;
            uint32_t pos[4], neg[4];
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                pos[j] = __shfl_sync(FULL_MASK, pos_self, base + j);
                neg[j] = __shfl_sync(FULL_MASK, neg_self, base + j);
            }
            uint32_t value = neg[3];
            #pragma unroll
            for (unsigned j = 0; j < 3; ++j) value += (mask & (1u << j)) ? pos[j] : neg[j];
            uint32_t limit = 0;
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) limit += pos[j] + neg[j];
            uint32_t limits[4];
            #pragma unroll
            for (unsigned t = 0; t < 4; ++t) limits[t] = __shfl_sync(FULL_MASK, limit, 8u * t);

            #pragma unroll
            for (int o = 0; o < 4; ++o) {
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    uint32_t decoded[4];
                    #pragma unroll
                    for (unsigned t = 0; t < 4; ++t) {
                        const unsigned packed_key = (key[o][p] >> (4u * (wave * 4u + t))) & 15u;
                        const unsigned source = 8u * t + (packed_key & 7u);
                        const uint32_t stored = __shfl_sync(FULL_MASK, value, source);
                        decoded[t] = (packed_key & 8u) ? limits[t] - stored : stored;
                    }
                    acc[o][p] = add3(acc[o][p], decoded[0], decoded[1]);
                    acc[o][p] = add3(acc[o][p], decoded[2], decoded[3]);
                }
            }
        }
    }

    uint32_t lo = 0, hi = 0;
    #pragma unroll
    for (int o = 0; o < 4; ++o) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            if (o < 2) lo ^= acc[o][p]; else hi ^= acc[o][p];
        }
    }
    output[id] = uint64_t(lo) | (uint64_t(hi) << 32);
}

template __global__ void activation_register_lut_core<false>(
    const uint32_t *, const uint32_t *, const uint32_t *, uint64_t *);
template __global__ void activation_register_lut_core<true>(
    const uint32_t *, const uint32_t *, const uint32_t *, uint64_t *);
