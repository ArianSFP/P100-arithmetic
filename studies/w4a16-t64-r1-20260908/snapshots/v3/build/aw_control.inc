struct aw_work_desc {
    const void    * input;
    const char    * weight;
    float         * output;
    const int32_t * input_rows;
    int32_t         row_offset;
    int32_t         rows;
    int32_t         layer;
    int32_t         expert;
};

struct aw_tile_desc {
    int32_t work;
    int32_t row;
    int32_t rows;
};


__device__ __forceinline__ float aw_bf16_to_float(uint16_t value) {
    return __uint_as_float((uint32_t) value << 16);
}

__device__ __forceinline__ uint16_t aw_float_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t) (bits >> 16);
}

template<bool BF16_INPUT>
__device__ __forceinline__ float aw_load_input(const void * input, size_t index) {
    if constexpr (BF16_INPUT) {
        const uintptr_t tagged = (uintptr_t) input;
        const uint16_t * input16 = (const uint16_t *) (tagged & ~(uintptr_t) 1);
        const uint16_t value = input16[index];
        return (tagged & 1) != 0 ?
                __half2float(__ushort_as_half(value)) : aw_bf16_to_float(value);
    } else {
        return ((const float *) input)[index];
    }
}

__device__ __forceinline__ void aw_load_bf16x8(
        const void * input, size_t index, float4 & lo, float4 & hi) {
    const uintptr_t tagged = (uintptr_t) input;
    const uint16_t * input16 = (const uint16_t *) (tagged & ~(uintptr_t) 1);
    const bool fp16 = (tagged & 1) != 0;
    const uint4 packed = *(const uint4 *) (input16 + index);
    const uint32_t words[4] = { packed.x, packed.y, packed.z, packed.w };
    float * lo_values = (float *) &lo;
    float * hi_values = (float *) &hi;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint16_t lo0 = (uint16_t) words[i];
        const uint16_t lo1 = (uint16_t) (words[i] >> 16);
        const uint16_t hi0 = (uint16_t) words[i + 2];
        const uint16_t hi1 = (uint16_t) (words[i + 2] >> 16);
        lo_values[2*i] = fp16 ? __half2float(__ushort_as_half(lo0)) : aw_bf16_to_float(lo0);
        lo_values[2*i + 1] = fp16 ? __half2float(__ushort_as_half(lo1)) : aw_bf16_to_float(lo1);
        hi_values[2*i] = fp16 ? __half2float(__ushort_as_half(hi0)) : aw_bf16_to_float(hi0);
        hi_values[2*i + 1] = fp16 ? __half2float(__ushort_as_half(hi1)) : aw_bf16_to_float(hi1);
    }
}
union aw_smem_m64 {
    struct {
        float A[2][AW_KSTAGE][AW_T64_ROWS];
        float B[2][AW_KSTAGE][AW_T64_ROWS];
    } stage;
    float red[AW_T64_ROWS][AW_T64_ROWS];
};

template<bool BF16_INPUT, bool T64_LAYOUT, bool F32_WEIGHT = false, int SPLIT_K = 2,
        bool INTERLEAVE = false>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kgrp = warp >> 2;
    const int wm = warp & 3;
    const int rm = lane >> 3;
    const int nt = lane & 7;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 2;
    const int b_k = (tid & 3)*8;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ntiles;
        const int col0 = (work_id % n_ntiles)*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;

        float4 pa0, pa1;
        float4 pb0, pb1;
        unsigned short pq[4];
        float pd;
        auto fetcha = [&](int stage) {
            const size_t base = (size_t) input_row*k + stage*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, base, pa0, pa1);
            } else {
                const float4 * input4 = (const float4 *) ((const float *) desc.input + base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
        };
        auto fetchb = [&](int stage) {
            if constexpr (F32_WEIGHT) {
                const float4 * weight4 = (const float4 *) ((const float *) desc.weight +
                        (size_t) (col0 + b_r)*k + stage*AW_KSTAGE + b_k);
                pb0 = weight4[0];
                pb1 = weight4[1];
                return;
            }
            const char * p;
            if constexpr (T64_LAYOUT) {
                const int n_kblocks = k/AW_KSTAGE;
                const char * tile_stage = desc.weight +
                        ((size_t) (col0/AW_T64_ROWS)*n_kblocks + stage)*AW_T64_STAGE_BYTES;
                pd = __half2float(*(const half *) (tile_stage + b_r*2));
                p = tile_stage + AW_T64_ROWS*2 + b_r*AW_KSTAGE;
            } else {
                p = desc.weight + ((size_t) (col0 + b_r)*(k/AW_KSTAGE) + stage)*AW_Q8_BLOCK_BYTES;
                pd = __half2float(*(const half *) p);
                p += 2;
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                pq[i] = *(const unsigned short *) (p + b_k + 2*i);
            }
        };
        auto stage_a_part = [&](int buffer, int i) {
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            sm.stage.A[buffer][a_k + i][a_r] = a0[i];
            sm.stage.A[buffer][a_k + i + 4][a_r] = a1[i];
        };
        auto stage_b_part = [&](int buffer, int i) {
            const float * b0 = (const float *) &pb0;
            const float * b1 = (const float *) &pb1;
            if constexpr (F32_WEIGHT) {
                sm.stage.B[buffer][b_k + i][b_r] = b0[i];
                sm.stage.B[buffer][b_k + i + 4][b_r] = b1[i];
            } else {
                sm.stage.B[buffer][b_k + 2*i][b_r] = (float) (signed char) (pq[i] & 0xff)*pd;
                sm.stage.B[buffer][b_k + 2*i + 1][b_r] = (float) (signed char) (pq[i] >> 8)*pd;
            }
        };
        auto stage_data = [&](int buffer) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                stage_a_part(buffer, i);
                stage_b_part(buffer, i);
            }
        };

        float acc[4][8] = {};
        fetcha(0);
        fetchb(0);
        stage_data(0);
        __syncthreads();
        int buffer = 0;
        for (int kb = 0; kb < k; kb += AW_KSTAGE) {
            const bool last = kb + AW_KSTAGE >= k;
            if (!last) {
                const int next = (kb + AW_KSTAGE)/AW_KSTAGE;
                fetcha(next);
                fetchb(next);
            }
            if constexpr (SPLIT_K == 1) {
                #pragma unroll
                for (int kk = 0; kk < AW_KSTAGE; ++kk) {
                    const int row0 = warp*8 + rm*2;
                    const float2 av = *(const float2 *) &sm.stage.A[buffer][kk][row0];
                    const float4 bv0 = *(const float4 *) &sm.stage.B[buffer][kk][nt*8];
                    const float4 bv1 = *(const float4 *) &sm.stage.B[buffer][kk][nt*8 + 4];
                    const float a[2] = { av.x, av.y };
                    const float b[8] = { bv0.x, bv0.y, bv0.z, bv0.w, bv1.x, bv1.y, bv1.z, bv1.w };
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[i][j] += a[i]*b[j];
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int kk = 0; kk < AW_KSTAGE/2; ++kk) {
                    const int ks = kgrp*(AW_KSTAGE/2) + kk;
                    const int row0 = wm*16 + rm*4;
                    const float4 av = *(const float4 *) &sm.stage.A[buffer][ks][row0];
                    const float4 bv0 = *(const float4 *) &sm.stage.B[buffer][ks][nt*8];
                    const float4 bv1 = *(const float4 *) &sm.stage.B[buffer][ks][nt*8 + 4];
                    const float a[4] = { av.x, av.y, av.z, av.w };
                    const float b[8] = { bv0.x, bv0.y, bv0.z, bv0.w, bv1.x, bv1.y, bv1.z, bv1.w };
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[i][j] += a[i]*b[j];
                        }
                    }
                    if constexpr (INTERLEAVE) {
                        static_assert(SPLIT_K == 2);
                        if ((kk & 3) == 3) {
                            stage_b_part(buffer ^ 1, kk >> 2);
                        }
                    }
                }
            }
            if (!last) {
                if constexpr (INTERLEAVE) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        stage_a_part(buffer ^ 1, i);
                    }
                } else {
                    stage_data(buffer ^ 1);
                }
            }
            __syncthreads();
            buffer ^= 1;
        }

        if constexpr (SPLIT_K == 1) {
            const int output_row0 = warp*8 + rm*2;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int row = output_row0 + i;
                if (row < tile.rows) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        desc.output[(size_t) (tile.row + row)*n + col0 + nt*8 + j] = acc[i][j];
                    }
                }
            }
        } else {
            const int output_row0 = wm*16 + rm*4;
            if (kgrp == 1) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sm.red[output_row0 + i][nt*8 + j] = acc[i][j];
                    }
                }
            }
            __syncthreads();
            if (kgrp == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = output_row0 + i;
                    if (row < tile.rows) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            desc.output[(size_t) (tile.row + row)*n + col0 + nt*8 + j] =
                                    acc[i][j] + sm.red[row][nt*8 + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

