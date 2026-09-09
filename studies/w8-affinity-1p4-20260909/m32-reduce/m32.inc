union aw_smem {struct {float4 A[2][32][8];float4 B[2][32][16];}stage;float red[8][256];};
constexpr int AW_Q8_BLOCK_BYTES=34;
template<bool BF16_INPUT, bool T64_LAYOUT,
        bool INTERLEAVE = false,
        bool WARP_QUAD = false,
        int FIXED_N = 0,
        int FIXED_K = 0>
__global__ __launch_bounds__(128, 2)
static void aw_q8_service_m32(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem sm;
    static_assert(FIXED_N == 0 ||
            FIXED_N % AW_T64_ROWS == 0);
    static_assert(FIXED_K == 0 ||
            FIXED_K % AW_KSTAGE == 0);
    static_assert(FIXED_K == 0 ||
            (FIXED_K/AW_KSTAGE) % 2 == 0);
    const int n_stride =
            FIXED_N != 0 ? FIXED_N : n;
    const int k_stride =
            FIXED_K != 0 ? FIXED_K : k;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kgrp = WARP_QUAD ?
            lane & 3 : warp;
    const int mt = WARP_QUAD ?
            warp : lane >> 3;
    const int nt = WARP_QUAD ?
            lane >> 2 : lane & 7;
    const int a_r  = tid >> 2;
    const int a_k  = (tid & 3)*8;
    const int b_r0 = tid >> 2;
    const int b_r1 = 32 + (tid >> 2);
    const int xk   = tid & 3;
    const int b_k  = xk*8;

    const int aslot  = (((a_r >> 2) & 1)*4 + (a_r >> 3)) ^ xk;
    const int aoff   = aslot*4 + (a_r & 3);
    const int bslot0 = ((((b_r0 >> 2) & 1)*8 + (b_r0 >> 3)) ^ xk);
    const int boff0  = bslot0*4 + (b_r0 & 3);
    const int bslot1 = ((((b_r1 >> 2) & 1)*8 + (b_r1 >> 3)) ^ xk);
    const int boff1  = bslot1*4 + (b_r1 & 3);
    const int sa0 = (0 + mt) ^ kgrp;
    const int sa1 = (4 + mt) ^ kgrp;
    const int sb0 = (0 + nt) ^ kgrp;
    const int sb1 = (8 + nt) ^ kgrp;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles =
            n_stride/AW_T64_ROWS;
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ntiles;
        const int col0 = (work_id % n_ntiles)*64;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const char * gq0 = nullptr;
        const char * gq1 = nullptr;
        if constexpr (!T64_LAYOUT) {
            gq0 = desc.weight +
                    (size_t) (col0 + b_r0)*
                    (k_stride/AW_KSTAGE)*
                    AW_Q8_BLOCK_BYTES;
            gq1 = desc.weight +
                    (size_t) (col0 + b_r1)*
                    (k_stride/AW_KSTAGE)*
                    AW_Q8_BLOCK_BYTES;
        }

        float4 pa0, pa1;
        unsigned short pq0[4], pq1[4];
        float pd0, pd1;
        auto fetcha = [&](int stage) {
            const size_t base =
                    (size_t) input_row*k_stride +
                    stage*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, base, pa0, pa1);
            } else {
                // Match the proven [TAG_MOE_PLAN] path: two aligned LDG.128s
                // instead of eight scalar loads through the generic descriptor.
                const float4 * input4 = (const float4 *) ((const float *) desc.input + base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
        };
        auto fetchb = [&](int stage) {
            const char * p0;
            const char * p1;
            if constexpr (T64_LAYOUT) {
                const int n_kblocks =
                        k_stride/AW_KSTAGE;
                const char * tile_stage = desc.weight +
                        ((size_t) (col0/AW_T64_ROWS)*n_kblocks + stage)*AW_T64_STAGE_BYTES;
                p0 = tile_stage + AW_T64_ROWS*2 + b_r0*AW_KSTAGE - 2;
                p1 = tile_stage + AW_T64_ROWS*2 + b_r1*AW_KSTAGE - 2;
                pd0 = __half2float(*(const half *) (tile_stage + b_r0*2));
                pd1 = __half2float(*(const half *) (tile_stage + b_r1*2));
            } else {
                p0 = gq0 + (size_t) stage*AW_Q8_BLOCK_BYTES;
                p1 = gq1 + (size_t) stage*AW_Q8_BLOCK_BYTES;
                pd0 = __half2float(*(const half *) p0);
                pd1 = __half2float(*(const half *) p1);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                pq0[i] = *(const unsigned short *) (p0 + 2 + b_k + 2*i);
                pq1[i] = *(const unsigned short *) (p1 + 2 + b_k + 2*i);
            }
        };
        auto stage_a_part = [&](int buffer, int i) {
            float * fA = (float *) sm.stage.A[buffer];
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            fA[(a_k + i    )*32 + aoff] = a0[i];
            fA[(a_k + i + 4)*32 + aoff] = a1[i];
        };
        auto stage_b_part = [&](int buffer, int i) {
            float * fB = (float *) sm.stage.B[buffer];
            fB[(b_k + 2*i    )*64 + boff0] = (float) (signed char) (pq0[i] & 0xff)*pd0;
            fB[(b_k + 2*i + 1)*64 + boff0] = (float) (signed char) (pq0[i] >> 8  )*pd0;
            fB[(b_k + 2*i    )*64 + boff1] = (float) (signed char) (pq1[i] & 0xff)*pd1;
            fB[(b_k + 2*i + 1)*64 + boff1] = (float) (signed char) (pq1[i] >> 8  )*pd1;
        };
        auto stage_data = [&](int buffer) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                stage_a_part(buffer, i);
                stage_b_part(buffer, i);
            }
        };

        float acc[8][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[i][j] = 0.0f;
            }
        }

        fetcha(0);
        fetchb(0);
        stage_data(0);
        __syncthreads();
        int buffer = 0;
        #pragma unroll 1
        for (int kb = 0;
                kb < k_stride;
                kb += AW_KSTAGE) {
            const bool last =
                    kb + AW_KSTAGE >= k_stride;
            if (!last) {
                const int next = (kb + AW_KSTAGE)/AW_KSTAGE;
                fetcha(next);
                fetchb(next);
            }
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int ks = kgrp*8 + kk;
                const float4 va0 = sm.stage.A[buffer][ks][sa0];
                const float4 va1 = sm.stage.A[buffer][ks][sa1];
                const float4 vb0 = sm.stage.B[buffer][ks][sb0];
                const float4 vb1 = sm.stage.B[buffer][ks][sb1];
                const float av[8] = { va0.x, va0.y, va0.z, va0.w, va1.x, va1.y, va1.z, va1.w };
                const float bv[8] = { vb0.x, vb0.y, vb0.z, vb0.w, vb1.x, vb1.y, vb1.z, vb1.w };
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] += av[i]*bv[j];
                    }
                }
                if constexpr (INTERLEAVE) {
                    if ((kk & 1) != 0) {
                        stage_b_part(buffer ^ 1, kk >> 1);
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

        if constexpr (WARP_QUAD) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rail_lane =
                            (lane >> 2)*4;
                    const float rail1 =
                            __shfl_sync(
                                0xffffffffu,
                                acc[i][j],
                                rail_lane + 1);
                    const float rail2 =
                            __shfl_sync(
                                0xffffffffu,
                                acc[i][j],
                                rail_lane + 2);
                    const float rail3 =
                            __shfl_sync(
                                0xffffffffu,
                                acc[i][j],
                                rail_lane + 3);
                    if ((lane & 3) == 0) {
                        const int row = mt*8 + i;
                        if (row < tile.rows) {
                            const float rail01 =
                                    __fadd_rn(
                                        acc[i][j],
                                        rail1);
                            const float rail012 =
                                    __fadd_rn(
                                        rail01,
                                        rail2);
                            desc.output[
                                    (size_t)
                                        (tile.row + row)*
                                        n_stride +
                                    col0 + nt*8 + j] =
                                    __fadd_rn(
                                        rail012,
                                        rail3);
                        }
                    }
                }
            }
            __syncthreads();
        } else {
            for (int round = 0; round < 8;
                    ++round) {
                if (kgrp != 0) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sm.red[j][
                                (kgrp - 1)*32 +
                                lane] =
                                acc[round][j];
                    }
                }
                __syncthreads();
                if (kgrp == 0) {
                    const int row = mt*8 + round;
                    if (row < tile.rows) {
                        #pragma unroll
                        for (int j = 0; j < 8;
                                ++j) {
                            const float value =
                                    acc[round][j] +
                                    sm.red[j][lane] +
                                    sm.red[j][32 + lane] +
                                    sm.red[j][64 + lane];
                            desc.output[
                                    (size_t)
                                        (tile.row + row)*
                                        n_stride +
                                    col0 + nt*8 + j] =
                                    value;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

