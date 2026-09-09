struct w8_half_smem {half A[2][32][256];half2 B[32][128];};
template<bool BF16_INPUT,int FLUSH>
__global__ __launch_bounds__(256, 1)
static void w8_packed_half(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ w8_half_smem sm;

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
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;
        size_t input_stage_base =
                (size_t) input_row*k + a_k;
        const char * weight_stage =
                desc.weight +
                (size_t) ntile*n_kblocks*
                    AW_T64_STAGE_BYTES;

        half2 acc0[16][4];float total[32][4]={};
        for(int i=0;i<16;++i)for(int j=0;j<4;++j)acc0[i][j]=__float2half2_rn(0.f);
        int buffer = 0;
        {
            float4 pa0, pa1;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input,
                        input_stage_base,
                        pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *) desc.input +
                         input_stage_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r] = __float2half_rn(a0[i]);
                sm.A[buffer][a_k + i + 4][a_r] = __float2half_rn(a1[i]);
            }
            {
            const int extra_local=tile.row+min(a_r+64,tile.rows-1);
            const int extra_row=desc.input_rows?desc.input_rows[extra_local]:extra_local;
            const size_t extra_base=size_t(extra_row)*k+a_k;
            float4 pa0, pa1;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input,
                        extra_base,
                        pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *) desc.input +
                         extra_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r+64] = __float2half_rn(a0[i]);
                sm.A[buffer][a_k + i + 4][a_r+64] = __float2half_rn(a1[i]);
            }
            }{
            const int extra_local=tile.row+min(a_r+128,tile.rows-1);
            const int extra_row=desc.input_rows?desc.input_rows[extra_local]:extra_local;
            const size_t extra_base=size_t(extra_row)*k+a_k;
            float4 pa0, pa1;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input,
                        extra_base,
                        pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *) desc.input +
                         extra_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r+128] = __float2half_rn(a0[i]);
                sm.A[buffer][a_k + i + 4][a_r+128] = __float2half_rn(a1[i]);
            }
            }{
            const int extra_local=tile.row+min(a_r+192,tile.rows-1);
            const int extra_row=desc.input_rows?desc.input_rows[extra_local]:extra_local;
            const size_t extra_base=size_t(extra_row)*k+a_k;
            float4 pa0, pa1;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input,
                        extra_base,
                        pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *) desc.input +
                         extra_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r+192] = __float2half_rn(a0[i]);
                sm.A[buffer][a_k + i + 4][a_r+192] = __float2half_rn(a1[i]);
            }
            }

            const float scale =
                    __half2float(
                        *(const half *)
                        (weight_stage +
                         b_local*2));
            const char * values =
                    weight_stage +
                    AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + producer_half*16;
            const uint4 packed4 =
                    __ldg((const uint4 *) values);
            const uint32_t packed[4] = {
                packed4.x, packed4.y,
                packed4.z, packed4.w
            };
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[producer_half*16 + group*4 + byte][b_r] =__float2half2_rn((float) (signed char)
                            ((packed[group] >>
                              (byte*8)) & 0xffu)*scale);
                }
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            if (!last) {
                float4 next_a0, next_a1;
                const size_t next_a_base =
                        input_stage_base +
                        AW_KSTAGE;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input +
                            next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = __float2half_rn(na0[i]);
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = __float2half_rn(na1[i]);
                }
                {
                const int extra_local=tile.row+min(a_r+64,tile.rows-1);
                const int extra_row=desc.input_rows?desc.input_rows[extra_local]:extra_local;
                float4 next_a0, next_a1;
                const size_t next_a_base =
                        size_t(extra_row)*k+size_t(stage+1)*AW_KSTAGE+a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input +
                            next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r+64] = __float2half_rn(na0[i]);
                    sm.A[buffer ^ 1][a_k + i + 4][a_r+64] = __float2half_rn(na1[i]);
                }
                }{
                const int extra_local=tile.row+min(a_r+128,tile.rows-1);
                const int extra_row=desc.input_rows?desc.input_rows[extra_local]:extra_local;
                float4 next_a0, next_a1;
                const size_t next_a_base =
                        size_t(extra_row)*k+size_t(stage+1)*AW_KSTAGE+a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input +
                            next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r+128] = __float2half_rn(na0[i]);
                    sm.A[buffer ^ 1][a_k + i + 4][a_r+128] = __float2half_rn(na1[i]);
                }
                }{
                const int extra_local=tile.row+min(a_r+192,tile.rows-1);
                const int extra_row=desc.input_rows?desc.input_rows[extra_local]:extra_local;
                float4 next_a0, next_a1;
                const size_t next_a_base =
                        size_t(extra_row)*k+size_t(stage+1)*AW_KSTAGE+a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input +
                            next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r+192] = __float2half_rn(na0[i]);
                    sm.A[buffer ^ 1][a_k + i + 4][a_r+192] = __float2half_rn(na1[i]);
                }
                }
            }

            if (stage != 0 && producer_half == 1) {
                const float scale =
                        __half2float(*(const half *)
                        (weight_stage + b_local*2));
                const char * values =
                        weight_stage +
                        AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + 16;
                const uint4 packed4 =
                        __ldg((const uint4 *) values);
                const uint32_t packed[4] = {
                    packed4.x, packed4.y,
                    packed4.z, packed4.w
                };
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[16 + group*4 + byte][b_r] =__float2half2_rn((float) (signed char)
                                ((packed[group] >>
                                  (byte*8)) & 0xffu)*scale);
                    }
                }
            }

            const int row0 = warp*32;
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const half2 *av=(const half2*)&sm.A[buffer][ks][row0];
                    const half2 bv[4]={sm.B[ks][lane],sm.B[ks][lane+32],sm.B[ks][lane+64],sm.B[ks][lane+96]};
                    #pragma unroll
                    for(int i=0;i<16;++i){
                        #pragma unroll
                        for(int j=0;j<4;++j)acc0[i][j]=__hfma2(av[i],bv[j],acc0[i][j]);
                    }
                }
            }
            __syncthreads();

            if (!last && producer_half == 0) {
                const char * next_weight_stage =
                        weight_stage +
                        AW_T64_STAGE_BYTES;
                const float scale =
                        __half2float(*(const half *)
                        (next_weight_stage +
                         b_local*2));
                const char * values =
                        next_weight_stage +
                        AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE;
                const uint4 packed4 =
                        __ldg((const uint4 *) values);
                const uint32_t packed[4] = {
                    packed4.x, packed4.y,
                    packed4.z, packed4.w
                };
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                            sm.B[group*4 + byte][b_r] =__float2half2_rn((float) (signed char)
                                    ((packed[group] >>
                                      (byte*8)) & 0xffu)*scale);
                    }
                }
            }

            #pragma unroll
            for (int group = 4; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const half2 *av=(const half2*)&sm.A[buffer][ks][row0];
                    const half2 bv[4]={sm.B[ks][lane],sm.B[ks][lane+32],sm.B[ks][lane+64],sm.B[ks][lane+96]};
                    #pragma unroll
                    for(int i=0;i<16;++i){
                        #pragma unroll
                        for(int j=0;j<4;++j)acc0[i][j]=__hfma2(av[i],bv[j],acc0[i][j]);
                    }
                }
            }
            if constexpr(FLUSH>0) {
                if((stage+1)%FLUSH==0 || last){
                    #pragma unroll
                    for(int i=0;i<16;++i){
                        #pragma unroll
                        for(int j=0;j<4;++j){
                            float2 lo=__half22float2(acc0[i][j]);
                            total[2*i][j]+=lo.x;total[2*i+1][j]+=lo.y;
                            acc0[i][j]=__float2half2_rn(0.f);
                        }
                    }
                }
            }
            if (!last) {
                __syncthreads();
            }
            input_stage_base += AW_KSTAGE;
            weight_stage += AW_T64_STAGE_BYTES;
            buffer ^= 1;
        }

        const int row0 = warp*32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if constexpr(FLUSH>0)output[lane+j*32]=total[i][j];else {float2 lo=__half22float2(acc0[i/2][j]);output[lane+j*32]=(i%2?lo.y:lo.x);}
                }
            }
        }
        __syncthreads();
    }
}