#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(call) do {                                                        \
    const cudaError_t error_ = (call);                                       \
    if (error_ != cudaSuccess)                                               \
        throw std::runtime_error(std::string(#call) + ": " +                \
                                 cudaGetErrorString(error_));                 \
} while (0)

// This is an arithmetic-throughput probe, not a GEMM.  The fixed launch is two
// 256-thread CTAs per each of the P100's 56 SMs.  Every loop iteration executes
// a positive and negative raw-INT4 activation step.  The paired steps cancel
// exactly, so arbitrarily long timing loops do not exceed the G32 FP16 bound.
constexpr int kBlockThreads = 256;
constexpr int kGridBlocks = 112;
constexpr int kChains = 8;
constexpr int kUsefulMacsPerIteration = kChains * 2 * 4; // 64 scalar INT4 MACs

// T_a[j] = (a*j) mod 8, j=0..7.  A dynamic row is loaded once per activation
// step and reused by four PRMTs, each covering two half2 packed products.
__device__ __constant__ uint32_t kMod8TableLo[8] = {
    0x00000000u, 0x03020100u, 0x06040200u, 0x01060300u,
    0x04000400u, 0x07020500u, 0x02040600u, 0x05060700u,
};
__device__ __constant__ uint32_t kMod8TableHi[8] = {
    0x00000000u, 0x07060504u, 0x06040200u, 0x05020704u,
    0x04000400u, 0x03060104u, 0x02040600u, 0x01020304u,
};

__device__ __forceinline__ uint32_t half2_bits(const half2 value) {
    const __half2_raw raw = value;
    return uint32_t(raw.x) | (uint32_t(raw.y) << 16);
}

__device__ __forceinline__ half2 half2_from_bits(const uint32_t bits) {
    return __halves2half2(__ushort_as_half(static_cast<unsigned short>(bits)),
                          __ushort_as_half(static_cast<unsigned short>(bits >> 16)));
}

__device__ __forceinline__ int signed_nibble(const uint32_t word,
                                             const unsigned shift) {
    const int nibble = int((word >> shift) & 15u);
    return (nibble ^ 8) - 8;
}

__device__ __forceinline__ uint32_t residues4(const uint32_t table_lo,
                                              const uint32_t table_hi,
                                              const uint32_t selector) {
    uint32_t result;
    asm volatile("prmt.b32 %0, %1, %2, %3;"
                 : "=r"(result)
                 : "r"(table_lo), "r"(table_hi), "r"(selector & 0x7777u));
    return result;
}

struct Dot2i {
    int low;
    int high;
};

__device__ __forceinline__ Dot2i repair_one(const int rounded,
                                            const unsigned residue) {
    // CUDA's signed SHR is arithmetic.  rounded is an integer-valued FP16.
    const int high = (rounded + 63) >> 7;
    const int delta = ((int(residue) - rounded + 4) & 7) - 4;
    return {rounded - (high << 7) + delta, high};
}

__device__ __forceinline__ uint64_t mix32(const uint32_t *values, int count) {
    uint32_t lo = 0, hi = 0;
    for (int i = 0; i < count; ++i) {
        if (i & 1) hi ^= values[i];
        else       lo ^= values[i];
    }
    return uint64_t(lo) | (uint64_t(hi) << 32);
}

extern "C" __global__ __launch_bounds__(kBlockThreads, 2)
void level2_plain_hfma2(const uint4 *input, uint64_t *output, const int iters) {
    const int id = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const uint4 words0 = input[2 * id + 0];
    const uint4 words1 = input[2 * id + 1];
    const uint32_t words[kChains] = {
        words0.x, words0.y, words0.z, words0.w,
        words1.x, words1.y, words1.z, words1.w,
    };

    half2 low_weight[kChains], high_weight[kChains];
    half2 low_acc[kChains], high_acc[kChains];
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
        const float wl0 = float(signed_nibble(words[chain], 0));
        const float wh0 = float(signed_nibble(words[chain], 4));
        const float wl1 = float(signed_nibble(words[chain], 8));
        const float wh1 = float(signed_nibble(words[chain], 12));
        low_weight[chain] = __floats2half2_rn(wl0, wl1);
        high_weight[chain] = __floats2half2_rn(wh0, wh1);
        low_acc[chain] = __floats2half2_rn(float(chain - 4), float(4 - chain));
        high_acc[chain] = __floats2half2_rn(float(chain - 3), float(chain - 5));
    }

    int activation_row = int((words0.x >> 16) & 7u);

    #pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
        // The real state change keeps all arithmetic inside the timed loop.
        // It is identical in all three modes and amortized over 64 useful MACs.
        activation_row = (activation_row + 1) & 7;
        const half2 positive = __half2half2(__int2half_rn(activation_row));
        const half2 negative = __hneg2(positive);
        #pragma unroll
        for (int chain = 0; chain < kChains; ++chain) {
            low_acc[chain]  = __hfma2(low_weight[chain],  positive, low_acc[chain]);
            high_acc[chain] = __hfma2(high_weight[chain], positive, high_acc[chain]);
            low_acc[chain]  = __hfma2(low_weight[chain],  negative, low_acc[chain]);
            high_acc[chain] = __hfma2(high_weight[chain], negative, high_acc[chain]);
        }
    }

    uint32_t result[2 * kChains];
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
        const half2 initial_low =
            __floats2half2_rn(float(chain - 4), float(4 - chain));
        const half2 initial_high =
            __floats2half2_rn(float(chain - 3), float(chain - 5));
        result[2 * chain + 0] = half2_bits(low_acc[chain]) ^ half2_bits(initial_low);
        result[2 * chain + 1] = half2_bits(high_acc[chain]) ^ half2_bits(initial_high);
    }
    output[id] = mix32(result, 2 * kChains); // zero means exact cancellation
}

__device__ __forceinline__ void normalized_split_mac(const half2 packed,
                                                      const half2 activation,
                                                      half2 &low_acc,
                                                      half2 &high_acc) {
    const half2 magic = __float2half2_rn(1536.0f);
    const half2 t = __hfma2(packed, activation, magic);
    const half2 high = __hsub2_rn(t, magic);
    const half2 low_scaled = __hfma2(packed, activation, __hneg2(high));
    low_acc = __hadd2_rn(low_acc, low_scaled);
    high_acc = __hadd2_rn(high_acc, high);
}

extern "C" __global__ __launch_bounds__(kBlockThreads, 2)
void level2_normalized_split(const uint4 *input, uint64_t *output,
                             const int iters) {
    const int id = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const uint4 words0 = input[2 * id + 0];
    const uint4 words1 = input[2 * id + 1];
    const uint32_t words[kChains] = {
        words0.x, words0.y, words0.z, words0.w,
        words1.x, words1.y, words1.z, words1.w,
    };

    half2 packed[kChains], low_acc[kChains], high_acc[kChains];
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
        const float wl0 = float(signed_nibble(words[chain], 0));
        const float wh0 = float(signed_nibble(words[chain], 4));
        const float wl1 = float(signed_nibble(words[chain], 8));
        const float wh1 = float(signed_nibble(words[chain], 12));
        packed[chain] = __floats2half2_rn(wh0 + wl0 * (1.0f / 128.0f),
                                          wh1 + wl1 * (1.0f / 128.0f));
        low_acc[chain] = __floats2half2_rn(float(chain + 1) / 128.0f,
                                           -float(chain + 1) / 128.0f);
        high_acc[chain] = __floats2half2_rn(float(chain - 3), float(chain - 5));
    }

    int activation_row = int((words0.x >> 16) & 7u);

    #pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
        activation_row = (activation_row + 1) & 7;
        const half2 positive = __half2half2(__int2half_rn(activation_row));
        const half2 negative = __hneg2(positive);
        #pragma unroll
        for (int chain = 0; chain < kChains; ++chain) {
            normalized_split_mac(packed[chain], positive,
                                 low_acc[chain], high_acc[chain]);
            normalized_split_mac(packed[chain], negative,
                                 low_acc[chain], high_acc[chain]);
        }
    }

    uint32_t result[2 * kChains];
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
        const half2 initial_low =
            __floats2half2_rn(float(chain + 1) / 128.0f,
                              -float(chain + 1) / 128.0f);
        const half2 initial_high =
            __floats2half2_rn(float(chain - 3), float(chain - 5));
        result[2 * chain + 0] = half2_bits(low_acc[chain]) ^ half2_bits(initial_low);
        result[2 * chain + 1] = half2_bits(high_acc[chain]) ^ half2_bits(initial_high);
    }
    output[id] = mix32(result, 2 * kChains);
}

__device__ __forceinline__ void repair_half2(const half2 rounded,
                                             const uint32_t residue_bytes,
                                             const unsigned byte_offset,
                                             int (&low_acc)[2],
                                             int (&high_acc)[2]) {
    const int r0 = __half2int_rn(__low2half(rounded));
    const int r1 = __half2int_rn(__high2half(rounded));
    const Dot2i d0 = repair_one(r0, (residue_bytes >> byte_offset) & 7u);
    const Dot2i d1 = repair_one(r1, (residue_bytes >> (byte_offset + 8)) & 7u);
    low_acc[0] += d0.low;
    high_acc[0] += d0.high;
    low_acc[1] += d1.low;
    high_acc[1] += d1.high;
}

extern "C" __global__ __launch_bounds__(kBlockThreads, 2)
void level2_hmul_prmt_repair(const uint4 *input, uint64_t *output,
                             const int iters) {
    const int id = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const uint4 words0 = input[2 * id + 0];
    const uint4 words1 = input[2 * id + 1];
    const uint32_t words[kChains] = {
        words0.x, words0.y, words0.z, words0.w,
        words1.x, words1.y, words1.z, words1.w,
    };

    half2 packed[kChains];
    uint32_t selectors[kChains / 2];
    int low_acc[kChains][2], high_acc[kChains][2];
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
        const int wl0 = signed_nibble(words[chain], 0);
        const int wh0 = signed_nibble(words[chain], 4);
        const int wl1 = signed_nibble(words[chain], 8);
        const int wh1 = signed_nibble(words[chain], 12);
        packed[chain] = __floats2half2_rn(float(wl0 + 128 * wh0),
                                          float(wl1 + 128 * wh1));
        low_acc[chain][0] = 3 * chain + 1;
        low_acc[chain][1] = -3 * chain - 2;
        high_acc[chain][0] = 5 * chain + 3;
        high_acc[chain][1] = -5 * chain - 4;
    }
    #pragma unroll
    for (int pair = 0; pair < kChains / 2; ++pair) {
        const uint32_t a = words[2 * pair + 0];
        const uint32_t b = words[2 * pair + 1];
        selectors[pair] = ((a >> 0) & 7u) | (((a >> 8) & 7u) << 4) |
                          (((b >> 0) & 7u) << 8) | (((b >> 8) & 7u) << 12);
    }

    int activation_row = int((words0.x >> 16) & 7u);

    #pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
        activation_row = (activation_row + 1) & 7;
        const half2 positive = __half2half2(__int2half_rn(activation_row));
        const half2 negative = __hneg2(positive);
        const int positive_row = activation_row;
        const int negative_row = (-activation_row) & 7;

        // Charge two dynamic 64-byte-family row selections per (+a,-a) pair.
        // Each selected pair of table registers is then reused by four PRMTs.
        const uint32_t positive_lo = kMod8TableLo[positive_row & 7];
        const uint32_t positive_hi = kMod8TableHi[positive_row & 7];
        const uint32_t negative_lo = kMod8TableLo[negative_row & 7];
        const uint32_t negative_hi = kMod8TableHi[negative_row & 7];

        #pragma unroll
        for (int pair = 0; pair < kChains / 2; ++pair) {
            const int c0 = 2 * pair + 0;
            const int c1 = 2 * pair + 1;
            const uint32_t rho = residues4(positive_lo, positive_hi, selectors[pair]);
            repair_half2(__hmul2_rn(packed[c0], positive), rho, 0,
                         low_acc[c0], high_acc[c0]);
            repair_half2(__hmul2_rn(packed[c1], positive), rho, 16,
                         low_acc[c1], high_acc[c1]);
        }
        #pragma unroll
        for (int pair = 0; pair < kChains / 2; ++pair) {
            const int c0 = 2 * pair + 0;
            const int c1 = 2 * pair + 1;
            const uint32_t rho = residues4(negative_lo, negative_hi, selectors[pair]);
            repair_half2(__hmul2_rn(packed[c0], negative), rho, 0,
                         low_acc[c0], high_acc[c0]);
            repair_half2(__hmul2_rn(packed[c1], negative), rho, 16,
                         low_acc[c1], high_acc[c1]);
        }
    }

    uint32_t result[4 * kChains];
    #pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
        result[4 * chain + 0] = uint32_t(low_acc[chain][0] - (3 * chain + 1));
        result[4 * chain + 1] = uint32_t(low_acc[chain][1] - (-3 * chain - 2));
        result[4 * chain + 2] = uint32_t(high_acc[chain][0] - (5 * chain + 3));
        result[4 * chain + 3] = uint32_t(high_acc[chain][1] - (-5 * chain - 4));
    }
    output[id] = mix32(result, 4 * kChains);
}

int main(int argc, char **argv) try {
    if (argc != 4 || std::string(argv[1]) != "--gpu-approved" ||
        std::string(argv[2]) != "1")
        throw std::runtime_error("usage: level2-throughput --gpu-approved 1 ITERATIONS");
    const int iters = std::atoi(argv[3]);
    if (iters < 32 || iters > 1048576)
        throw std::runtime_error("ITERATIONS must be in [32,1048576]");

    int devices = 0;
    CU(cudaGetDeviceCount(&devices));
    if (devices != 1) throw std::runtime_error("exactly one visible GPU required");
    cudaDeviceProp property{};
    CU(cudaGetDeviceProperties(&property, 0));
    if (property.major != 6 || property.minor != 0 || property.multiProcessorCount != 56)
        throw std::runtime_error("56-SM GP100 required");

    constexpr int count = kGridBlocks * kBlockThreads;
    std::vector<uint4> host_input(2 * count);
    uint32_t state = 3911;
    for (uint4 &value : host_input) {
        auto next = [&]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
        };
        value = make_uint4(next(), next(), next(), next());
    }

    uint4 *device_input = nullptr;
    uint64_t *device_output = nullptr;
    CU(cudaMalloc(&device_input, host_input.size() * sizeof(uint4)));
    CU(cudaMalloc(&device_output, count * sizeof(uint64_t)));
    CU(cudaMemcpy(device_input, host_input.data(), host_input.size() * sizeof(uint4),
                  cudaMemcpyHostToDevice));

    struct Mode { const char *name; };
    const Mode modes[] = {{"plain-hfma2"}, {"normalized-split"},
                          {"hmul-prmt-repair"}};
    auto launch = [&](const int mode) {
        if (mode == 0)
            level2_plain_hfma2<<<kGridBlocks, kBlockThreads>>>(device_input,
                                                               device_output, iters);
        else if (mode == 1)
            level2_normalized_split<<<kGridBlocks, kBlockThreads>>>(device_input,
                                                                    device_output, iters);
        else
            level2_hmul_prmt_repair<<<kGridBlocks, kBlockThreads>>>(device_input,
                                                                    device_output, iters);
        CU(cudaGetLastError());
    };

    std::vector<uint64_t> host_output(count);
    for (int mode = 0; mode < 3; ++mode) {
        launch(mode);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(host_output.data(), device_output, count * sizeof(uint64_t),
                      cudaMemcpyDeviceToHost));
        for (const uint64_t value : host_output)
            if (value != 0) throw std::runtime_error(std::string("exact cancellation failed: ") + modes[mode].name);
    }

    const auto warm_end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        for (int mode = 0; mode < 3; ++mode) launch(mode);
        CU(cudaDeviceSynchronize());
    } while (std::chrono::steady_clock::now() < warm_end);

    cudaEvent_t begin{}, end{};
    CU(cudaEventCreate(&begin));
    CU(cudaEventCreate(&end));
    constexpr int rounds = 9, repeats = 3;
    for (int round = 0; round < rounds; ++round) {
        for (int position = 0; position < 3; ++position) {
            const int mode = (position + round) % 3;
            CU(cudaEventRecord(begin));
            for (int repeat = 0; repeat < repeats; ++repeat) launch(mode);
            CU(cudaEventRecord(end));
            CU(cudaEventSynchronize(end));
            float milliseconds = 0;
            CU(cudaEventElapsedTime(&milliseconds, begin, end));
            const double microseconds = milliseconds * 1000.0 / repeats;
            const double useful_macs = double(count) * double(iters) *
                                       double(kUsefulMacsPerIteration);
            std::printf("TIME mode=%s rep=%d us=%.9g useful_tmac_s=%.9g\n",
                        modes[mode].name, round, microseconds,
                        useful_macs / (microseconds * 1.0e6));
        }
    }
    std::printf("META grid=%d block=%d threads=%d chains=%d iterations=%d "
                "useful_macs_per_thread_iteration=%d seed=3911\n",
                kGridBlocks, kBlockThreads, count, kChains, iters,
                kUsefulMacsPerIteration);
    CU(cudaEventDestroy(end));
    CU(cudaEventDestroy(begin));
    CU(cudaFree(device_output));
    CU(cudaFree(device_input));
    std::puts("PASS");
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "STOP: %s\n", error.what());
    return 1;
}
