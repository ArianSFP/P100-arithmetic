#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %d %s at %s:%d\n", int(e), \
                 cudaGetErrorString(e), __FILE__, __LINE__); std::exit(3); } } while (0)

struct Input { uint32_t a, b, c; };
struct Output { uint32_t sad, selected_sum; int32_t w2a8, w4a8; uint32_t converted; };

__device__ __forceinline__ uint32_t sad4(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm volatile("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__global__ void correctness(const Input *in, Output *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Input v = in[i];
    uint32_t m0 = 0, m1 = 0;
    int sum_weights = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned w = (v.b >> (2*j)) & 3;
        m0 |= (0u - (w & 1)) & (255u << (8*j));
        m1 |= (0u - (w >> 1)) & (255u << (8*j));
        sum_weights += int(w & 1) - 2*int(w >> 1);
    }
    uint32_t biased = v.a ^ 0x80808080u;
    int d0 = int(sad4(biased & m0, 0, 0));
    int d1 = int(sad4(biased & m1, 0, 0));
    int dot4 = 0, sum_w4 = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned w = (v.b >> (4*j)) & 15;
        sum_w4 += int(w & 7) - int(w & 8);
    }
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t mask = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            mask |= (0u - ((v.b >> (4*j+p)) & 1)) & (255u << (8*j));
        int coefficient = p == 3 ? -8 : (1 << p);
        dot4 += coefficient * int(sad4(biased & mask, 0, 0));
    }
    float converted;
    asm volatile("{ .reg .s32 t; bfe.s32 t, %1, 16, 8; "
                 "cvt.rn.f32.s32 %0, t; }" : "=f"(converted) : "r"(v.a));
    out[i] = {sad4(v.a, v.b, v.c), sad4(v.a & m0, 0, v.c),
              d0 - 2*d1 - 128*sum_weights, dot4 - 128*sum_w4,
              __float_as_uint(converted)};
}

__global__ void latency(uint64_t *cycles, uint32_t *out, uint32_t a,
                        uint32_t b, int groups) {
    uint32_t r = threadIdx.x;
    uint64_t start = clock64();
    for (int j = 0; j < groups; ++j) {
        #pragma unroll
        for (int k = 0; k < 32; ++k) r = sad4(a, b, r);
    }
    uint64_t stop = clock64();
    cycles[threadIdx.x] = stop - start;
    out[threadIdx.x] = r;
}

template<int CHAINS>
__global__ void throughput(uint32_t *out, uint32_t a, uint32_t b, int groups) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t r[CHAINS];
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) r[c] = i + c;
    for (int j = 0; j < groups; ++j) {
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            #pragma unroll
            for (int c = 0; c < CHAINS; ++c) r[c] = sad4(a, b, r[c]);
        }
    }
    uint32_t sum = 0;
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) sum += r[c];
    out[i] = sum;
}

uint32_t next_u32(uint32_t &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 6 || prop.minor != 0) return 4;
    std::printf("device=%s sm=%d.%d sms=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    constexpr unsigned n = 262144;
    uint32_t rng = 0x60a8c002;
    std::vector<Input> inputs(n);
    std::vector<Output> actual(n);
    for (unsigned i = 0; i < n; ++i) {
        // Exhaustive scalar byte pairs, then reproducible mixed backgrounds.
        uint32_t a = i < 65536 ? (i & 255)*0x01010101u : next_u32(rng);
        uint32_t b = i < 65536 ? (i >> 8)*0x01010101u : next_u32(rng);
        inputs[i] = {a, b, next_u32(rng)};
    }
    Input *di;
    Output *dout;
    CHECK(cudaMalloc(&di, n*sizeof(Input)));
    CHECK(cudaMalloc(&dout, n*sizeof(Output)));
    CHECK(cudaMemcpy(di, inputs.data(), n*sizeof(Input), cudaMemcpyHostToDevice));
    correctness<<<n/256, 256>>>(di, dout, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(actual.data(), dout, n*sizeof(Output), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < n; ++i) {
        uint32_t sad = inputs[i].c, selected = inputs[i].c;
        int32_t dot = 0, dot4 = 0;
        for (int j = 0; j < 4; ++j) {
            int a = (inputs[i].a >> (8*j)) & 255;
            int b = (inputs[i].b >> (8*j)) & 255;
            unsigned w = (inputs[i].b >> (2*j)) & 3;
            sad += std::abs(a-b);
            selected += (w & 1)*a;
            dot += (a < 128 ? a : a-256)*(int(w & 1)-2*int(w >> 1));
            int w4 = (inputs[i].b >> (4*j)) & 15;
            dot4 += (a < 128 ? a : a-256)*(w4 < 8 ? w4 : w4-16);
        }
        int byte = (inputs[i].a >> 16) & 255;
        float expected_float = float(byte < 128 ? byte : byte-256);
        uint32_t expected_bits;
        static_assert(sizeof(expected_bits) == sizeof(expected_float), "float must be 32 bits");
        __builtin_memcpy(&expected_bits, &expected_float, sizeof(expected_bits));
        if (actual[i].sad != sad || actual[i].selected_sum != selected ||
            actual[i].w2a8 != dot || actual[i].w4a8 != dot4 || actual[i].converted != expected_bits) {
            std::fprintf(stderr, "mismatch i=%u a=%08x b=%08x c=%08x\n",
                         i, inputs[i].a, inputs[i].b, inputs[i].c);
            return 5;
        }
    }
    std::printf("correctness=PASS cases=%u seed=0x60a8c002 operations=sad4,masked_sum,w2a8,w4a8,i8_to_f32_bfe\n", n);
    for (unsigned i : {0u, 255u, 65535u, 65536u, n-1})
        std::printf("witness i=%u a=%08x b=%08x c=%08x sad=%08x selected=%08x w2a8=%d w4a8=%d converted=%08x\n",
                    i, inputs[i].a, inputs[i].b, inputs[i].c, actual[i].sad,
                    actual[i].selected_sum, actual[i].w2a8, actual[i].w4a8, actual[i].converted);
    CHECK(cudaFree(di));
    CHECK(cudaFree(dout));

    uint64_t *dc;
    uint32_t *dr;
    const int blocks = prop.multiProcessorCount * 16;
    const int threads = blocks * 256;
    CHECK(cudaMalloc(&dc, 32*sizeof(uint64_t)));
    CHECK(cudaMalloc(&dr, threads*sizeof(uint32_t)));
    const uint32_t av = 0x0305070b, bv = 0x0d111317;
    for (int warm = 0; warm < 10; ++warm) throughput<4><<<blocks, 256>>>(dr, av, bv, 32);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    for (int groups : {8, 16, 32}) {
        for (int sample = 0; sample < 5; ++sample) {
            latency<<<1, 32>>>(dc, dr, av, bv, groups);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            uint64_t cycles[32];
            CHECK(cudaMemcpy(cycles, dc, sizeof(cycles), cudaMemcpyDeviceToHost));
            std::printf("latency chain_length=%d sample=%d lane0_cycles=%llu\n", groups*32,
                        sample, (unsigned long long)cycles[0]);
        }
    }
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    for (int chains : {1, 4}) {
        for (int sample = 0; sample < 5; ++sample) {
            CHECK(cudaEventRecord(start));
            if (chains == 1) throughput<1><<<blocks, 256>>>(dr, av, bv, 32);
            else throughput<4><<<blocks, 256>>>(dr, av, bv, 32);
            CHECK(cudaGetLastError());
            CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            uint32_t last;
            CHECK(cudaMemcpy(&last, dr+threads-1, sizeof(last), cudaMemcpyDeviceToHost));
            uint32_t delta = 0;
            for (int j = 0; j < 4; ++j) delta += std::abs(int((av>>(8*j))&255)-int((bv>>(8*j))&255));
            uint32_t expected = chains*uint32_t(threads-1) + chains*(chains-1)/2 + chains*1024u*delta;
            if (last != expected) return 6;
            double ginst = double(threads)*1024*chains/(ms*1e6);
            std::printf("throughput chains=%d sample=%d ms=%.6f thread_Ginst_s=%.3f\n", chains, sample, ms, ginst);
        }
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dc)); CHECK(cudaFree(dr));
    return 0;
}
