#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(x) do { cudaError_t e = (x); if (e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e)); } while (0)

__device__ __forceinline__ uint32_t half2_bits(half2 x) {
    __half2_raw raw = x;
    return uint32_t(raw.x) | (uint32_t(raw.y) << 16);
}

// Same optimistic issued-work control used by the paused W4A4 study.  The
// shared footprint deliberately remains 32 KiB, as in the full mu8 design and
// the fast M128 W16/W4 control geometry.  It omits delivery and group finish.
template<int ROWS>
__global__ __launch_bounds__(256, 2)
void resident_half2(const uint32_t *seed, uint64_t *out, int groups) {
    __shared__ uint32_t scratch[256][32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int id = blockIdx.x * blockDim.x + tid;
    for (int i = tid; i < 256 * 32; i += blockDim.x) scratch[i / 32][i & 31] = seed[(i + id) & 8191];
    __syncthreads();
    const uint32_t v = scratch[(seed[id & 8191] >> 24) & 255][lane];
    const float x = float(int(v & 15) - 8);
    const float y = float(int((v >> 4) & 15) - 8);
    const half2 a = __floats2half2_rn(x, y);
    const half2 b = __floats2half2_rn(y, x);
    half2 acc[ROWS][4];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) for (int c = 0; c < 4; ++c) acc[r][c] = __float2half2_rn(0);
    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] = __hfma2((k ^ c) & 1 ? a : b, (k ^ r) & 1 ? b : a, acc[r][c]);
            }
        }
    }
    uint32_t lo = 0, hi = 0;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) for (int c = 0; c < 4; ++c) {
        const uint32_t z = half2_bits(acc[r][c]);
        if (r < ROWS / 2) lo ^= z; else hi ^= z;
    }
    out[id] = uint64_t(lo) | (uint64_t(hi) << 32);
}

// Full 256-entry, already-built consumer.  table_input is constructed so that
// its upper half is the exact complement of its lower half under limit[lane].
template<int ROWS, int MIN_BLOCKS>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void prebuilt_full256(const uint32_t *table_input, const uint32_t *mask_seed,
                      uint64_t *out, int groups) {
    __shared__ uint32_t table[256][32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int id = blockIdx.x * blockDim.x + tid;
    for (int i = tid; i < 256 * 32; i += blockDim.x) table[i / 32][i & 31] = table_input[i];
    __syncthreads();
    volatile uint32_t (*lookup)[32] = table;
    uint32_t acc[ROWS][4];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) for (int p = 0; p < 4; ++p) acc[r][p] = uint32_t(r * 17 + p * 13 + lane);
    uint32_t state = mask_seed[(blockIdx.x * 8 + warp) & 8191];
    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        state = state * 1664525u + 1013904223u;
        #pragma unroll
        for (int slice = 0; slice < 4; ++slice) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                #pragma unroll
                for (int plane = 0; plane < 4; ++plane) {
                    const unsigned mask = (state + r * 29u + slice * 53u + plane * 71u) & 255u;
                    acc[r][plane] += lookup[mask][lane];
                }
            }
        }
    }
    uint32_t lo = state, hi = 0;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) for (int p = 0; p < 4; ++p) {
        if (r < ROWS / 2) lo ^= acc[r][p]; else hi ^= acc[r][p];
    }
    out[id] = uint64_t(lo) | (uint64_t(hi) << 32);
}

// FIGLUT-derived complement symmetry.  The high mask bit and canonical index
// are warp-uniform.  Each limit byte is >= the corresponding U byte, so the
// ordinary packed subtraction cannot borrow across byte fields.
template<int ROWS, int MIN_BLOCKS>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void prebuilt_half128(const uint32_t *table_input, const uint32_t *limits,
                      const uint32_t *mask_seed, uint64_t *out, int groups) {
    __shared__ uint32_t table[128][32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int id = blockIdx.x * blockDim.x + tid;
    for (int i = tid; i < 128 * 32; i += blockDim.x) table[i / 32][i & 31] = table_input[i];
    __syncthreads();
    volatile uint32_t (*lookup)[32] = table;
    const uint32_t limit = limits[lane];
    uint32_t acc[ROWS][4];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) for (int p = 0; p < 4; ++p) acc[r][p] = uint32_t(r * 17 + p * 13 + lane);
    uint32_t state = mask_seed[(blockIdx.x * 8 + warp) & 8191];
    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        state = state * 1664525u + 1013904223u;
        #pragma unroll
        for (int slice = 0; slice < 4; ++slice) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                #pragma unroll
                for (int plane = 0; plane < 4; ++plane) {
                    const unsigned mask = (state + r * 29u + slice * 53u + plane * 71u) & 255u;
                    const unsigned flip = mask >> 7;
                    const unsigned index = (mask & 127u) ^ (flip * 127u);
                    uint32_t value = lookup[index][lane];
                    if (flip) value = limit - value;
                    acc[r][plane] += value;
                }
            }
        }
    }
    uint32_t lo = state, hi = 0;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) for (int p = 0; p < 4; ++p) {
        if (r < ROWS / 2) lo ^= acc[r][p]; else hi ^= acc[r][p];
    }
    out[id] = uint64_t(lo) | (uint64_t(hi) << 32);
}

struct Mode {
    const char *name;
    int rows;
    int kind; // 0 half2, 1 full256, 2 half128
};

int main(int argc, char **argv) try {
    if (argc != 4 || std::string(argv[1]) != "--gpu-approved" || std::string(argv[2]) != "1")
        throw std::runtime_error("usage: --gpu-approved 1 groups");
    const int groups = std::atoi(argv[3]);
    if (groups < 8 || groups > 4096) throw std::runtime_error("group bounds");
    int devices = 0;
    CU(cudaGetDeviceCount(&devices));
    if (devices != 1) throw std::runtime_error("exactly one visible GPU required");
    cudaDeviceProp prop{};
    CU(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 6 || prop.minor != 0) throw std::runtime_error("SM60 required");
    const int grid = prop.multiProcessorCount * 2;
    const int block = 256;
    const int count = grid * block;

    std::vector<uint32_t> low(128 * 32), full(256 * 32), limits(32), seeds(8192);
    uint32_t rng = 3911;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
    for (int lane = 0; lane < 32; ++lane) {
        // Per-byte limit 64.  Every low-half table byte is in [0,63].
        limits[lane] = 0x40404040u;
    }
    for (int mask = 0; mask < 128; ++mask) for (int lane = 0; lane < 32; ++lane) {
        const uint32_t value = next() & 0x3f3f3f3fu;
        low[mask * 32 + lane] = value;
        full[mask * 32 + lane] = value;
        full[((~mask) & 255) * 32 + lane] = limits[lane] - value;
    }
    for (auto &v : seeds) v = next();

    uint32_t *d_full = nullptr, *d_low = nullptr, *d_limits = nullptr, *d_seeds = nullptr;
    uint64_t *d_out = nullptr;
    CU(cudaMalloc(&d_full, full.size() * sizeof(uint32_t)));
    CU(cudaMalloc(&d_low, low.size() * sizeof(uint32_t)));
    CU(cudaMalloc(&d_limits, limits.size() * sizeof(uint32_t)));
    CU(cudaMalloc(&d_seeds, seeds.size() * sizeof(uint32_t)));
    CU(cudaMalloc(&d_out, count * sizeof(uint64_t)));
    CU(cudaMemcpy(d_full, full.data(), full.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_low, low.data(), low.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_limits, limits.data(), limits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_seeds, seeds.data(), seeds.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    const Mode modes[] = {
        {"resident-half2-r16", 16, 0}, {"prebuilt-full256-r16", 16, 1}, {"prebuilt-half128-r16", 16, 2},
        {"resident-half2-r8", 8, 0}, {"prebuilt-full256-r8", 8, 1}, {"prebuilt-half128-r8", 8, 2},
        {"resident-half2-r4", 4, 0}, {"prebuilt-full256-r4", 4, 1}, {"prebuilt-half128-r4", 4, 2},
    };
    auto launch = [&](int mode) {
        const Mode &m = modes[mode];
        if (m.rows == 16 && m.kind == 0) resident_half2<16><<<grid, block>>>(d_seeds, d_out, groups);
        else if (m.rows == 16 && m.kind == 1) prebuilt_full256<16, 1><<<grid, block>>>(d_full, d_seeds, d_out, groups);
        else if (m.rows == 16 && m.kind == 2) prebuilt_half128<16, 1><<<grid, block>>>(d_low, d_limits, d_seeds, d_out, groups);
        else if (m.rows == 8 && m.kind == 0) resident_half2<8><<<grid, block>>>(d_seeds, d_out, groups);
        else if (m.rows == 8 && m.kind == 1) prebuilt_full256<8, 2><<<grid, block>>>(d_full, d_seeds, d_out, groups);
        else if (m.rows == 8 && m.kind == 2) prebuilt_half128<8, 2><<<grid, block>>>(d_low, d_limits, d_seeds, d_out, groups);
        else if (m.rows == 4 && m.kind == 0) resident_half2<4><<<grid, block>>>(d_seeds, d_out, groups);
        else if (m.rows == 4 && m.kind == 1) prebuilt_full256<4, 2><<<grid, block>>>(d_full, d_seeds, d_out, groups);
        else prebuilt_half128<4, 2><<<grid, block>>>(d_low, d_limits, d_seeds, d_out, groups);
        CU(cudaGetLastError());
    };

    // Full and half consumers must be bit-identical for each ownership width.
    std::vector<uint64_t> got_full(count), got_half(count);
    for (int base : {0, 3, 6}) {
        launch(base + 1); CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(got_full.data(), d_out, count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        launch(base + 2); CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(got_half.data(), d_out, count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (got_full != got_half) throw std::runtime_error(std::string("full/half checksum mismatch at ") + modes[base].name);
    }

    auto warm_end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        for (int mode = 0; mode < 9; ++mode) launch(mode);
        CU(cudaDeviceSynchronize());
    } while (std::chrono::steady_clock::now() < warm_end);

    cudaEvent_t begin{}, end{};
    CU(cudaEventCreate(&begin)); CU(cudaEventCreate(&end));
    constexpr int rounds = 9, repeats = 3;
    for (int rep = 0; rep < rounds; ++rep) for (int j = 0; j < 9; ++j) {
        const int mode = (j + 2 * rep) % 9;
        CU(cudaEventRecord(begin));
        for (int q = 0; q < repeats; ++q) launch(mode);
        CU(cudaEventRecord(end)); CU(cudaEventSynchronize(end));
        float ms = 0; CU(cudaEventElapsedTime(&ms, begin, end));
        const double us = ms * 1000.0 / repeats;
        const double useful = double(count) * groups * modes[mode].rows * 128.0;
        std::printf("TIME mode=%s rep=%d us=%.9g useful_tmac_s=%.9g\n",
                    modes[mode].name, rep, us, useful / (us * 1e6));
    }

    uint64_t checksum = 0;
    CU(cudaMemcpy(got_half.data(), d_out, count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (uint64_t v : got_half) checksum ^= v;
    std::printf("META grid=%d block=%d threads=%d groups=%d seed=3911 checksum=%llu prebuilt_table=1 omitted_build=1 omitted_finish=1 full_half_bitidentical=1\n",
                grid, block, count, groups, (unsigned long long)checksum);
    CU(cudaFree(d_out)); CU(cudaFree(d_seeds)); CU(cudaFree(d_limits)); CU(cudaFree(d_low)); CU(cudaFree(d_full));
    std::puts("PASS");
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "STOP: %s\n", e.what());
    return 1;
}
