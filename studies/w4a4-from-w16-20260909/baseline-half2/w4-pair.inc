// Exact signed-INT4 G32 arithmetic grafted onto the W16 M128 pair geometry.
// A half2 lane pair represents two activation rows.  Each lane accumulates one
// raw-code dot of at most 32 * 8 * 8 = 2048, so every prefix is exactly
// representable in binary16.  Weight and activation scales are then applied
// at the G32 boundary with FP32 multiplication/FMA.
constexpr int W4_STAGE_BYTES = 1152;

struct w4_work_desc {
    const unsigned char * input_codes; // row-major [row][K/32][16]
    const float * input_scales;        // row-major [row][K/32]
    const char * weight;               // tiled [N/64][K/32][1152]
    float * output;
    const int32_t * input_rows;
    int32_t rows;
    int32_t expert;
};

__device__ __forceinline__ half2 w4_decode_pair(unsigned byte) {
    // Two offset-binary Q4 nibbles -> signed integer half lanes, no I2F.
    __half2_raw raw;
    raw.x = (unsigned short) (0x6400u | (((byte      ) & 15u) ^ 8u));
    raw.y = (unsigned short) (0x6400u | (((byte >> 4) & 15u) ^ 8u));
    return __hsub2(half2(raw), __float2half2_rn(1032.0f));
}

__device__ __forceinline__ void w4_load_exact_stage(
        const w4_work_desc & desc, const aw_tile_desc & tile,
        int stage, int n_kblocks, int ngroup,
        int a_r, int a_k, int producer_half, int b_r,
        half (*stage_a)[128], float * stage_as,
        half2 (*stage_b)[128], float * stage_ws) {
    const int local0 = tile.row + min(a_r, tile.rows - 1);
    const int local1 = tile.row + min(a_r + 64, tile.rows - 1);
    const int input0 = desc.input_rows != nullptr ? desc.input_rows[local0] : local0;
    const int input1 = desc.input_rows != nullptr ? desc.input_rows[local1] : local1;
    const size_t group0 = (size_t) input0*n_kblocks + stage;
    const size_t group1 = (size_t) input1*n_kblocks + stage;
    const unsigned word0 = __ldg((const unsigned *)
            (desc.input_codes + group0*16 + a_k/2));
    const unsigned word1 = __ldg((const unsigned *)
            (desc.input_codes + group1*16 + a_k/2));
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const half2 q0 = w4_decode_pair((word0 >> (pair*8)) & 0xffu);
        const half2 q1 = w4_decode_pair((word1 >> (pair*8)) & 0xffu);
        stage_a[a_k + pair*2    ][a_r     ] = __low2half(q0);
        stage_a[a_k + pair*2 + 1][a_r     ] = __high2half(q0);
        stage_a[a_k + pair*2    ][a_r + 64] = __low2half(q1);
        stage_a[a_k + pair*2 + 1][a_r + 64] = __high2half(q1);
    }
    if (a_k == 0) {
        stage_as[a_r] = __ldg(desc.input_scales + group0);
        stage_as[a_r + 64] = __ldg(desc.input_scales + group1);
    }

    const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
    const int b_local = b_r % AW_T64_ROWS;
    const char * weight_stage = desc.weight +
            ((size_t) ntile*n_kblocks + stage)*W4_STAGE_BYTES;
    if (producer_half == 0) {
        stage_ws[b_r] = __half2float(*(const half *)
                (weight_stage + b_local*sizeof(half)));
    }
    const unsigned * code_words = (const unsigned *)
            (weight_stage + AW_T64_ROWS*sizeof(half));
    const unsigned packed0 = __ldg(code_words + producer_half*128 + b_local);
    const unsigned packed1 = __ldg(code_words + producer_half*128 + 64 + b_local);
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        stage_b[producer_half*8 + pair][b_r] =
                w4_decode_pair((packed0 >> (pair*8)) & 0xffu);
        stage_b[producer_half*8 + pair + 4][b_r] =
                w4_decode_pair((packed1 >> (pair*8)) & 0xffu);
    }
}

struct w4_pair_smem {
    half A[2][32][128];
    half2 B[2][16][128];
    float AS[2][128];
    float WS[2][128];
};

template<int N=0, int K=0>
__global__ __launch_bounds__(256, 1)
static void w4_pair_exact_g32(
        const w4_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    if constexpr(N > 0) n = N;
    if constexpr(K > 0) k = K;
    __shared__ w4_pair_smem sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int producer_half = warp >> 2;
    const int b_r = (warp & 3)*32 + lane;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int ngroup_shift = __ffs(n_ngroups) - 1;
    const int ngroup_mask = n_ngroups - 1;
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id >> ngroup_shift;
        const int ngroup = work_id & ngroup_mask;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const w4_work_desc desc = descs[tile.work];
        const int n_kblocks = k/AW_KSTAGE;
        float total[16][4] = {};
        int buffer = 0;

        w4_load_exact_stage(desc, tile, 0, n_kblocks, ngroup,
                a_r, a_k, producer_half, b_r,
                sm.A[buffer], sm.AS[buffer], sm.B[buffer], sm.WS[buffer]);
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            if (!last) {
                w4_load_exact_stage(desc, tile, stage + 1, n_kblocks, ngroup,
                        a_r, a_k, producer_half, b_r,
                        sm.A[buffer ^ 1], sm.AS[buffer ^ 1],
                        sm.B[buffer ^ 1], sm.WS[buffer ^ 1]);
            }

            // One exact G32 raw-code half2 dot per output.  Unlike the W16
            // control, the half accumulator is deliberately reset every stage.
            half2 raw[8][4];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    raw[i][j] = __float2half2_rn(0.0f);

            const int row0 = warp*16;
            #pragma unroll
            for (int ks = 0; ks < 32; ++ks) {
                const half2 * av = (const half2 *) &sm.A[buffer][ks][row0];
                const half2 packed_b[4] = {
                    sm.B[buffer][ks/2][lane],
                    sm.B[buffer][ks/2][lane + 32],
                    sm.B[buffer][ks/2][lane + 64],
                    sm.B[buffer][ks/2][lane + 96]
                };
                const half bv[4] = {
                    (ks & 1) ? __high2half(packed_b[0]) : __low2half(packed_b[0]),
                    (ks & 1) ? __high2half(packed_b[1]) : __low2half(packed_b[1]),
                    (ks & 1) ? __high2half(packed_b[2]) : __low2half(packed_b[2]),
                    (ks & 1) ? __high2half(packed_b[3]) : __low2half(packed_b[3])
                };
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        raw[i][j] = __hfma2(av[i], __half2half2(bv[j]), raw[i][j]);
                }
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float as0 = sm.AS[buffer][row0 + 2*i];
                const float as1 = sm.AS[buffer][row0 + 2*i + 1];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float ws = sm.WS[buffer][lane + j*32];
                    const float2 dot = __half22float2(raw[i][j]);
                    total[2*i][j] = __fmaf_rn(dot.x, __fmul_rn(as0, ws), total[2*i][j]);
                    total[2*i + 1][j] = __fmaf_rn(dot.y, __fmul_rn(as1, ws), total[2*i + 1][j]);
                }
            }

            if (!last) __syncthreads();
            buffer ^= 1;
        }

        const int row0 = warp*16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (row0 + i < tile.rows) {
                float * output = desc.output +
                        (size_t) (tile.row + row0 + i)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    output[lane + j*32] = total[i][j];
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(256)
static void w4_quantize_a4(
        const half * __restrict__ input,
        unsigned char * __restrict__ codes,
        float * __restrict__ scales,
        size_t groups) {
    const int lane = threadIdx.x & 31;
    const size_t warp = (size_t) blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
    const size_t stride = (size_t) gridDim.x*(blockDim.x/32);
    for (size_t group = warp; group < groups; group += stride) {
        const float value = __half2float(input[group*32 + lane]);
        float maximum = fabsf(value);
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            maximum = fmaxf(maximum,
                    __shfl_xor_sync(0xffffffffu, maximum, offset));
        const float scale = __fdiv_rn(maximum, 7.0f);
        const int q = scale == 0.0f ? 0 :
                max(-7, min(7, __float2int_rn(__fdiv_rn(value, scale))));
        const int q_next = __shfl_down_sync(0xffffffffu, q, 1);
        if ((lane & 1) == 0)
            codes[group*16 + lane/2] =
                    (unsigned char) ((q & 15) | ((q_next & 15) << 4));
        if (lane == 0) scales[group] = scale;
    }
}
