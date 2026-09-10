#include <cuda_runtime.h>
#include <stdint.h>

// Compile-only probes. Inputs escape every result; no kernel in this file is
// intended to be timed or treated as a complete GEMM.

template<int ROWS, int MIN_BLOCKS, bool HALF>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void prebuilt_consumer(const uint32_t *seed, uint64_t *out, int groups) {
    __shared__ uint32_t table[HALF ? 128 : 256][32];
    __shared__ uint32_t cap[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int id = blockIdx.x * blockDim.x + tid;
    for (int i = tid; i < (HALF ? 128 : 256) * 32; i += blockDim.x) {
        table[i / 32][i & 31] = seed[(i + id) & 8191];
    }
    if (tid < 32) cap[tid] = seed[(id + 4096) & 8191] & 0x3f3f3f3fu;
    __syncthreads();
    volatile uint32_t (*lookup)[32] = table;
    const uint32_t lane_cap = cap[lane];
    uint32_t acc[ROWS][4];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) acc[r][p] = uint32_t(r * 17 + p * 13 + lane);
    }
    uint32_t state = seed[(blockIdx.x * 8 + (tid >> 5)) & 8191];
    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        state = state * 1664525u + 1013904223u;
        #pragma unroll
        for (int slice = 0; slice < 4; ++slice) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                #pragma unroll
                for (int plane = 0; plane < 4; ++plane) {
                    const unsigned mask = (state + r * 29 + slice * 53 + plane * 71) & 255;
                    if constexpr (HALF) {
                        const unsigned flip = mask >> 7;
                        const unsigned index = (mask & 127u) ^ (flip * 127u);
                        const uint32_t v = lookup[index][lane];
                        const uint32_t decoded = flip ? lane_cap - v : v;
                        acc[r][plane] += decoded;
                    } else {
                        acc[r][plane] += lookup[mask][lane];
                    }
                }
            }
        }
    }
    uint32_t lo = state, hi = 0;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            if (r < ROWS / 2) lo ^= acc[r][p]; else hi ^= acc[r][p];
        }
    }
    out[id] = uint64_t(lo) | (uint64_t(hi) << 32);
}

// Optimistic upper bound when activation preparation has already converted
// every raw mask to {flip,index}. This moves complement-XOR work out of the
// consumer but still charges extraction, C-value reconstruction, and select.
template<int ROWS, int MIN_BLOCKS>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void precanonical_half_consumer(const uint32_t *seed, uint64_t *out, int groups) {
    __shared__ uint32_t table[128][32];
    __shared__ uint32_t cap[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int id = blockIdx.x * blockDim.x + tid;
    for (int i = tid; i < 128 * 32; i += blockDim.x) table[i / 32][i & 31] = seed[(i + id) & 8191];
    if (tid < 32) cap[tid] = seed[(id + 4096) & 8191] & 0x3f3f3f3fu;
    __syncthreads();
    volatile uint32_t (*lookup)[32] = table;
    const uint32_t lane_cap = cap[lane];
    uint32_t acc[ROWS][4];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) acc[r][p] = uint32_t(r * 17 + p * 13 + lane);
    }
    uint32_t state = seed[(blockIdx.x * 8 + (tid >> 5)) & 8191];
    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        state = state * 1664525u + 1013904223u;
        #pragma unroll
        for (int slice = 0; slice < 4; ++slice) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                #pragma unroll
                for (int plane = 0; plane < 4; ++plane) {
                    // Treat this byte as if the producer already canonicalized
                    // it: bits 0..6 index half128; bit 7 selects C-value.
                    const unsigned key = (state + r * 29 + slice * 53 + plane * 71) & 255;
                    const unsigned index = key & 127u;
                    const unsigned flip = key >> 7;
                    const uint32_t v = lookup[index][lane];
                    const uint32_t decoded = flip ? lane_cap - v : v;
                    acc[r][plane] += decoded;
                }
            }
        }
    }
    uint32_t lo = state, hi = 0;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            if (r < ROWS / 2) lo ^= acc[r][p]; else hi ^= acc[r][p];
        }
    }
    out[id] = uint64_t(lo) | (uint64_t(hi) << 32);
}

// One mu8 table-construction slice. Each lane is one four-output packed table;
// the eight warps partition masks. pos[j]/neg[j] are already-prepared packed
// nonnegative byte contributions, so this omits compressed-nibble decoding.
// Gray updates use two scalar IADD operations safely: current-old cannot borrow
// across bytes and adding new cannot carry because each final byte is <=64.
constexpr unsigned seed_placeholder = 73u;

template<bool STORE_COMPLEMENT>
__global__ __launch_bounds__(256, 2)
void build_gray_slice(const uint32_t *pos_in, const uint32_t *neg_in,
                      uint32_t *checksum) {
    __shared__ volatile uint32_t table[STORE_COMPLEMENT ? 256 : 128][32];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    uint32_t pos[8], neg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned at = (blockIdx.x * 32u + lane) * 8u + unsigned(j);
        pos[j] = pos_in[at];
        neg[j] = neg_in[at];
    }

    // Bit 7 is fixed at zero for canonical entries. Warps own bits 4..6 and
    // Gray-iterate bits 0..3. The full variant writes each complementary entry
    // at the same time instead of rebuilding it independently.
    constexpr unsigned LOW_BITS = 4u;
    const unsigned base = warp << LOW_BITS;
    uint32_t value = (base & 1u ? pos[0] : neg[0]);
    #pragma unroll
    for (int j = 1; j < 8; ++j) value += (base & (1u << j)) ? pos[j] : neg[j];

    unsigned prior = 0;
    #pragma unroll
    for (unsigned step = 0; step < 16; ++step) {
        const unsigned gray = step ^ (step >> 1);
        if (step != 0) {
            const unsigned changed = gray ^ prior;
            const int bit = __ffs(changed) - 1;
            const bool set = (gray & changed) != 0;
            const uint32_t old_value = set ? neg[bit] : pos[bit];
            const uint32_t new_value = set ? pos[bit] : neg[bit];
            value = value - old_value;
            value = value + new_value;
        }
        const unsigned mask = base | gray;
        table[mask][lane] = value;
        if constexpr (STORE_COMPLEMENT) {
            // cap is sum(abs(w_j)) == sum(pos_j+neg_j). The complement entry
            // is lane-wise cap-value, with no byte borrow.
            uint32_t cap = pos[0] + neg[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) cap += pos[j] + neg[j];
            table[255u ^ mask][lane] = cap - value;
        }
        prior = gray;
    }
    __syncthreads();
    if (tid == 0) checksum[blockIdx.x] = table[seed_placeholder][0];
}

template __global__ void prebuilt_consumer<8, 2, false>(const uint32_t*, uint64_t*, int);
template __global__ void prebuilt_consumer<8, 2, true>(const uint32_t*, uint64_t*, int);
template __global__ void prebuilt_consumer<16, 1, false>(const uint32_t*, uint64_t*, int);
template __global__ void prebuilt_consumer<16, 1, true>(const uint32_t*, uint64_t*, int);
template __global__ void precanonical_half_consumer<8, 2>(const uint32_t*, uint64_t*, int);
template __global__ void precanonical_half_consumer<16, 1>(const uint32_t*, uint64_t*, int);
template __global__ void build_gray_slice<false>(const uint32_t*, const uint32_t*, uint32_t*);
template __global__ void build_gray_slice<true>(const uint32_t*, const uint32_t*, uint32_t*);
