#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(x) do { cudaError_t e = (x); if (e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e)); } while (0)

constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t packed_sum_warp(uint32_t value) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1) value += __shfl_down_sync(FULL, value, delta);
    return __shfl_sync(FULL, value, 0);
}

// Lower-bound exact-G32 half2 control.  Each half lane is one K16 rail, so one
// accumulator represents one 32-term output.  Four token rows x four output
// columns give the same 512 useful scalar MACs/thread/G32 as packed4_full16.
__global__ __launch_bounds__(512, 1)
void half2_complete(const uint32_t *seed, float *out, int groups) {
    const int tid = threadIdx.x;
    const int id = blockIdx.x * blockDim.x + tid;
    float total[4][4] = {};
    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        // Runtime-varying operands prevent the compiler from hoisting the
        // complete G32 chain out of the group loop.
        const uint32_t source = seed[(id + g * 521) & 8191];
        const float x = float(int(source & 15) - 7);
        const float y = float(int((source >> 4) & 15) - 7);
        const half2 a = __floats2half2_rn(x, y);
        const half2 b = __floats2half2_rn(y, x);
        half2 acc[4][4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) for (int c = 0; c < 4; ++c) {
            const float initial = float(t * 4 + c) + float(source & 1u);
            acc[t][c] = __floats2half2_rn(initial, -initial);
        }
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) acc[t][c] = __hfma2((k ^ t) & 1 ? a : b, (k ^ c) & 1 ? b : a, acc[t][c]);
            }
        }
        const float scale = 1.0f + float((source + unsigned(g)) & 1u) * 0.0009765625f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float2 pair = __half22float2(acc[t][c]);
                total[t][c] = __fmaf_rn(__fadd_rn(pair.x, pair.y), scale, total[t][c]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) for (int c = 0; c < 4; ++c) out[size_t(id) * 16 + t * 4 + c] = total[t][c];
}

// Full16 activation-derived register LUT.  Four A4 tokens occupy the four byte
// fields of every table entry.  Each lane owns four output columns.  Two
// 16-entry tables fit a warp, so four waves cover the eight K4 quartets in G32.
// Runtime plane words are lossless W4 keys and are staged once per CTA/G32.
__global__ __launch_bounds__(512, 1)
void packed4_full16(const uint32_t *positive, const uint32_t *negative,
                    const uint32_t *key_words, const float *activation_scales,
                    const float *weight_scales, float *out, int groups) {
    __shared__ uint32_t shared_keys[16][32];
    __shared__ float shared_wscale[4][32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int id = blockIdx.x * blockDim.x + tid;
    float total[4][4] = {};

    #pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        // 512 words encode four output columns x four bit planes x eight
        // quartet masks for all 32 lanes.  A 512-thread CTA stages one each.
        shared_keys[tid >> 5][tid & 31] = key_words[size_t(g) * 512 + tid];
        if (tid < 128) shared_wscale[tid >> 5][tid & 31] = weight_scales[size_t(g) * 128 + tid];
        __syncthreads();

        uint32_t words[4][4];
        float ws[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            ws[c] = shared_wscale[c][lane];
            #pragma unroll
            for (int p = 0; p < 4; ++p) words[c][p] = shared_keys[c * 4 + p][lane];
        }

        const size_t activation_index = (size_t(warp) * groups + g) * 32 + lane;
        const uint32_t pos_lane = positive[activation_index];
        const uint32_t neg_lane = negative[activation_index];
        const uint32_t pos_total = packed_sum_warp(pos_lane);
        const uint32_t neg_total = packed_sum_warp(neg_lane);
        int correction[4];
        float as[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int ps = int((pos_total >> (8 * t)) & 255u);
            const int ns = int((neg_total >> (8 * t)) & 255u);
            correction[t] = 8 * ps + 7 * ns; // 15*B + 8*A = 8*positive + 7*negative.
            as[t] = activation_scales[(size_t(warp) * groups + g) * 4 + t];
        }

        uint32_t planes[4][4] = {};
        #pragma unroll
        for (int wave = 0; wave < 4; ++wave) {
            const int table = lane >> 4;
            const int index = lane & 15;
            const int base = wave * 8 + table * 4;
            uint32_t entry = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t pos = __shfl_sync(FULL, pos_lane, base + j);
                const uint32_t neg = __shfl_sync(FULL, neg_lane, base + j);
                entry += (index & (1 << j)) ? pos : neg;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint32_t word = words[c][p];
                    const unsigned mask0 = (word >> (4 * (2 * wave))) & 15u;
                    const unsigned mask1 = (word >> (4 * (2 * wave + 1))) & 15u;
                    const uint32_t value0 = __shfl_sync(FULL, entry, mask0);
                    const uint32_t value1 = __shfl_sync(FULL, entry, 16u + mask1);
                    planes[c][p] += value0 + value1;
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int p0 = int((planes[c][0] >> (8 * t)) & 255u);
                const int p1 = int((planes[c][1] >> (8 * t)) & 255u);
                const int p2 = int((planes[c][2] >> (8 * t)) & 255u);
                const int p3 = int((planes[c][3] >> (8 * t)) & 255u);
                const int dot = p0 + 2 * p1 + 4 * p2 + 8 * p3 - correction[t];
                total[t][c] = __fmaf_rn(float(dot), __fmul_rn(as[t], ws[c]), total[t][c]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) for (int c = 0; c < 4; ++c) out[size_t(id) * 16 + t * 4 + c] = total[t][c];
}

// Slow independent integer definition used only for the small correctness
// launch.  It reconstructs all 32 signed W4 values from the plane masks.
__global__ void direct_integer_oracle(const uint32_t *positive, const uint32_t *negative,
                                      const uint32_t *key_words, const float *activation_scales,
                                      const float *weight_scales, float *out, int groups) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int id = blockIdx.x * blockDim.x + tid;
    float total[4][4] = {};
    for (int g = 0; g < groups; ++g) {
        const size_t ai = (size_t(warp) * groups + g) * 32;
        for (int t = 0; t < 4; ++t) for (int c = 0; c < 4; ++c) {
            int dot = 0;
            for (int j = 0; j < 32; ++j) {
                const uint32_t pp = positive[ai + j];
                const uint32_t nn = negative[ai + j];
                const int a = int((pp >> (8 * t)) & 255u) - int((nn >> (8 * t)) & 255u);
                const int quartet = j / 4, within = j & 3;
                int code = 0;
                for (int p = 0; p < 4; ++p) {
                    const uint32_t word = key_words[size_t(g) * 512 + (c * 4 + p) * 32 + lane];
                    const int mask = int((word >> (4 * quartet)) & 15u);
                    code |= ((mask >> within) & 1) << p;
                }
                dot += a * (code - 8);
            }
            const float scale = activation_scales[(size_t(warp) * groups + g) * 4 + t] * weight_scales[size_t(g) * 128 + c * 32 + lane];
            total[t][c] = __fmaf_rn(float(dot), scale, total[t][c]);
        }
    }
    for (int t = 0; t < 4; ++t) for (int c = 0; c < 4; ++c) out[size_t(id) * 16 + t * 4 + c] = total[t][c];
}

int main(int argc, char **argv) try {
    if (argc != 4 || std::string(argv[1]) != "--gpu-approved" || std::string(argv[2]) != "1")
        throw std::runtime_error("usage: --gpu-approved 1 groups");
    const int groups = std::atoi(argv[3]);
    if (groups < 8 || groups > 512) throw std::runtime_error("group bounds");
    int devices = 0; CU(cudaGetDeviceCount(&devices));
    if (devices != 1) throw std::runtime_error("exactly one visible GPU required");
    cudaDeviceProp prop{}; CU(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 6 || prop.minor != 0) throw std::runtime_error("SM60 required");
    const int block = 512;
    const int grid = prop.multiProcessorCount * 2;
    const int count = grid * block;
    const int warps = block / 32;

    uint32_t rng = 3911;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
    std::vector<uint32_t> seed(8192), pos(size_t(warps) * groups * 32), neg(pos.size());
    std::vector<uint32_t> keys(size_t(groups) * 512);
    std::vector<float> as(size_t(warps) * groups * 4), ws(size_t(groups) * 128);
    for (auto &v : seed) v = next();
    for (size_t i = 0; i < pos.size(); ++i) {
        uint32_t p = 0, n = 0;
        for (int t = 0; t < 4; ++t) {
            const int a = int(next() % 15) - 7;
            p |= uint32_t(std::max(a, 0)) << (8 * t);
            n |= uint32_t(std::max(-a, 0)) << (8 * t);
        }
        pos[i] = p; neg[i] = n;
    }
    for (auto &v : keys) v = next();
    for (auto &v : as) v = 0.0078125f * float(1 + next() % 8);
    for (auto &v : ws) v = 0.0078125f * float(1 + next() % 8);

    uint32_t *dseed = nullptr, *dpos = nullptr, *dneg = nullptr, *dkeys = nullptr;
    float *das = nullptr, *dws = nullptr, *dout = nullptr, *doracle = nullptr;
    CU(cudaMalloc(&dseed, seed.size() * 4)); CU(cudaMalloc(&dpos, pos.size() * 4)); CU(cudaMalloc(&dneg, neg.size() * 4));
    CU(cudaMalloc(&dkeys, keys.size() * 4)); CU(cudaMalloc(&das, as.size() * 4)); CU(cudaMalloc(&dws, ws.size() * 4));
    CU(cudaMalloc(&dout, size_t(count) * 16 * 4)); CU(cudaMalloc(&doracle, size_t(block) * 16 * 4));
    CU(cudaMemcpy(dseed, seed.data(), seed.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice)); CU(cudaMemcpy(dneg, neg.data(), neg.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dkeys, keys.data(), keys.size() * 4, cudaMemcpyHostToDevice)); CU(cudaMemcpy(das, as.data(), as.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dws, ws.data(), ws.size() * 4, cudaMemcpyHostToDevice));

    // Full exact small-shape comparison against the independent definition.
    constexpr int check_groups = 8;
    packed4_full16<<<1, block>>>(dpos, dneg, dkeys, das, dws, dout, check_groups);
    direct_integer_oracle<<<1, block>>>(dpos, dneg, dkeys, das, dws, doracle, check_groups);
    CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
    std::vector<float> got(size_t(block) * 16), expected(got.size());
    CU(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(expected.data(), doracle, expected.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) if (got[i] != expected[i]) ++bad;
    std::printf("CHECK groups=%d outputs=%zu bit_mismatches=%zu\n", check_groups, got.size(), bad);
    if (bad) throw std::runtime_error("packed4 LUT differs from direct integer definition");

    auto launch = [&](int mode) {
        if (mode == 0) half2_complete<<<grid, block>>>(dseed, dout, groups);
        else packed4_full16<<<grid, block>>>(dpos, dneg, dkeys, das, dws, dout, groups);
        CU(cudaGetLastError());
    };
    auto warm_end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do { launch(0); launch(1); CU(cudaDeviceSynchronize()); } while (std::chrono::steady_clock::now() < warm_end);
    cudaEvent_t begin{}, end{}; CU(cudaEventCreate(&begin)); CU(cudaEventCreate(&end));
    constexpr int rounds = 9, repeats = 3;
    const char *names[] = {"half2-complete-lower-bound", "packed4-full16-complete-g32"};
    for (int rep = 0; rep < rounds; ++rep) for (int j = 0; j < 2; ++j) {
        const int mode = (j + rep) & 1;
        CU(cudaEventRecord(begin));
        for (int q = 0; q < repeats; ++q) launch(mode);
        CU(cudaEventRecord(end)); CU(cudaEventSynchronize(end));
        float ms = 0; CU(cudaEventElapsedTime(&ms, begin, end));
        const double us = ms * 1000.0 / repeats;
        const double useful = double(count) * groups * 512.0;
        std::printf("TIME mode=%s rep=%d us=%.9g useful_tmac_s=%.9g\n", names[mode], rep, us, useful / (us * 1e6));
    }
    CU(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
    uint32_t checksum = 0; for (float v : got) { uint32_t bits; std::memcpy(&bits, &v, 4); checksum ^= bits; }
    std::printf("META grid=%d block=%d threads=%d groups=%d useful_macs_per_thread_group=512 checksum=%u prepared_posneg=1 staged_weight_keys=1 exact_g32_finish=1 persistent_fp32_totals=1\n",
                grid, block, count, groups, checksum);
    CU(cudaFree(doracle)); CU(cudaFree(dout)); CU(cudaFree(dws)); CU(cudaFree(das)); CU(cudaFree(dkeys)); CU(cudaFree(dneg)); CU(cudaFree(dpos)); CU(cudaFree(dseed));
    std::puts("PASS"); return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "STOP: %s\n", e.what()); return 1;
}
