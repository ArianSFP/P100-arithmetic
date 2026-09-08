__host__ __device__ static inline int aw_q4_tag(const char * p) {
    return (int) ((uintptr_t) p & 3u);
}
__host__ __device__ static inline const char * aw_expert_ptr(
        const char * p, size_t expert, size_t q8_bytes) {
    const int tag = aw_q4_tag(p);
    return p + expert*(q8_bytes/34)*(tag == 1 ? 18 : tag == 2 ? 20 : 34);
}
__device__ static inline float aw_q4_value(const char * tagged,
        int row, int kblock, int ki, int k) {
    const int tag = aw_q4_tag(tagged);
    const bool tiled = ((uintptr_t) tagged & 4u) != 0;
    const char * base = (const char *) ((uintptr_t) tagged & ~(uintptr_t)7);
    const int metadata = tag == 1 ? 2 : 4;
    const char * block = tiled ? base + ((size_t)(row/64)*(k/32)+kblock)*64*(metadata+16)
                              : base + ((size_t)row*(k/32)+kblock)*(metadata+16);
    const char * scale = tiled ? block + (row%64)*2 : block;
    const float d = __half2float(*(const half *) scale);
    const unsigned char * qs = (const unsigned char *)(block + (tiled ? metadata*64+(row%64)*16+(ki&15) : metadata+(ki&15)));
    const int q = (*qs >> (ki < 16 ? 0 : 4)) & 15;
    if (tag == 1) return __half2float(__float2half_rn(__fmul_rn(d, (float)(q - 8))));
    return __half2float(__float2half_rn(__fadd_rn(__fmul_rn(d, (float)q), __half2float(*(const half *)(tiled ? block+128+(row%64)*2 : block+2)))));
}


__device__ __forceinline__ const char * aw_q4_stage(const char * tagged,int row,int stage,int k) {
    int md=aw_q4_tag(tagged)==1?2:4;
    return (const char*)((uintptr_t)tagged & ~(uintptr_t)7)+((size_t)(row/64)*(k/32)+stage)*64*(md+16);
}
__device__ __forceinline__ uint4 aw_q4_load16(const char * tagged,int row,int stage,int k) {
    int md=aw_q4_tag(tagged)==1?2:4;
    return __ldg((const uint4*)(aw_q4_stage(tagged,row,stage,k)+md*64+(row%64)*16));
}
__device__ __forceinline__ float aw_q4_scale(const char * tagged,int row,int stage,int k) {
    return __half2float(*(const half*)(aw_q4_stage(tagged,row,stage,k)+(row%64)*2));
}
__device__ __forceinline__ float aw_q4_offset(const char * tagged,int row,int stage,int k) {
    return aw_q4_tag(tagged)==2 ? __half2float(*(const half*)(aw_q4_stage(tagged,row,stage,k)+128+(row%64)*2)):0.0f;
}
__device__ __forceinline__ float aw_q4_decode_byte(unsigned word,int byte,int upper,int tag,float d,float m) {
    const int q=(word>>(byte*8+upper*4))&15;
    float v=__fmul_rn(d,float(tag==1?q-8:q));
    if(tag==2)v=__fadd_rn(v,m);
    return __half2float(__float2half_rn(v));
}

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
struct aw_smem_m64_n128_256 {
    float A[2][AW_KSTAGE][AW_T64_ROWS];
    float B[AW_KSTAGE][2*AW_T64_ROWS];
};
__device__ __forceinline__ float aw_bf16_to_float(uint16_t value) {
    return __uint_as_float((uint32_t) value << 16);
}
__device__ __forceinline__ uint16_t aw_float_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t) (bits >> 16);
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
template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64_n128_halfpipe_sync(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_256 sm;

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

        float acc0[8][4] = {};
        float acc1[8][4] = {};
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
                sm.A[buffer][a_k + i][a_r] = (aw_q4_tag(desc.weight) ? __half2float(__float2half_rn(a0[i])) : a0[i]);
                sm.A[buffer][a_k + i + 4][a_r] = (aw_q4_tag(desc.weight) ? __half2float(__float2half_rn(a1[i])) : a1[i]);
            }

            const float scale = aw_q4_tag(desc.weight) ? 0.0f :
                    __half2float(
                        *(const half *)
                        (weight_stage +
                         b_local*2));
            const char * values =
                    weight_stage +
                    AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + producer_half*16;
            const uint4 packed4 = ((uintptr_t)desc.weight & 4u) ? aw_q4_load16(desc.weight,col0+b_r,0,k) : aw_q4_tag(desc.weight) ? uint4{} :
                    __ldg((const uint4 *) values);
            const uint32_t packed[4] = {
                packed4.x, packed4.y,
                packed4.z, packed4.w
            };
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[producer_half*16 + group*4 + byte][b_r] = ((uintptr_t)desc.weight & 4u) ? aw_q4_decode_byte(packed[group],byte,producer_half,aw_q4_tag(desc.weight),aw_q4_scale(desc.weight,col0+b_r,0,k),aw_q4_offset(desc.weight,col0+b_r,0,k)) : aw_q4_tag(desc.weight) ? aw_q4_value(desc.weight,col0+b_r,0,producer_half*16 + group*4 + byte,k) : (float) (signed char)
                            ((packed[group] >>
                              (byte*8)) & 0xffu)*scale;
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
                    sm.A[buffer ^ 1][a_k + i][a_r] = (aw_q4_tag(desc.weight) ? __half2float(__float2half_rn(na0[i])) : na0[i]);
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = (aw_q4_tag(desc.weight) ? __half2float(__float2half_rn(na1[i])) : na1[i]);
                }
            }

            if (stage != 0 && producer_half == 1) {
                const float scale = aw_q4_tag(desc.weight) ? 0.0f :
                        __half2float(*(const half *)
                        (weight_stage + b_local*2));
                const char * values =
                        weight_stage +
                        AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + 16;
                const uint4 packed4 = ((uintptr_t)desc.weight & 4u) ? aw_q4_load16(desc.weight,col0+b_r,stage,k) : aw_q4_tag(desc.weight) ? uint4{} :
                        __ldg((const uint4 *) values);
                const uint32_t packed[4] = {
                    packed4.x, packed4.y,
                    packed4.z, packed4.w
                };
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[16 + group*4 + byte][b_r] = ((uintptr_t)desc.weight & 4u) ? aw_q4_decode_byte(packed[group],byte,1,aw_q4_tag(desc.weight),aw_q4_scale(desc.weight,col0+b_r,stage,k),aw_q4_offset(desc.weight,col0+b_r,stage,k)) : aw_q4_tag(desc.weight) ? aw_q4_value(desc.weight,col0+b_r,stage,16 + group*4 + byte,k) : (float) (signed char)
                                ((packed[group] >>
                                  (byte*8)) & 0xffu)*scale;
                    }
                }
            }

            const int row0 = warp*8;
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 =
                            *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc0[i][j] += av[i]*bv[j];
                        }
                    }
                }
            }
            __syncthreads();

            if (!last && producer_half == 0) {
                const char * next_weight_stage =
                        weight_stage +
                        AW_T64_STAGE_BYTES;
                const float scale = aw_q4_tag(desc.weight) ? 0.0f :
                        __half2float(*(const half *)
                        (next_weight_stage +
                         b_local*2));
                const char * values =
                        next_weight_stage +
                        AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE;
                const uint4 packed4 = ((uintptr_t)desc.weight & 4u) ? aw_q4_load16(desc.weight,col0+b_r,stage+1,k) : aw_q4_tag(desc.weight) ? uint4{} :
                        __ldg((const uint4 *) values);
                const uint32_t packed[4] = {
                    packed4.x, packed4.y,
                    packed4.z, packed4.w
                };
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                            sm.B[group*4 + byte][b_r] = ((uintptr_t)desc.weight & 4u) ? aw_q4_decode_byte(packed[group],byte,0,aw_q4_tag(desc.weight),aw_q4_scale(desc.weight,col0+b_r,stage+1,k),aw_q4_offset(desc.weight,col0+b_r,stage+1,k)) : aw_q4_tag(desc.weight) ? aw_q4_value(desc.weight,col0+b_r,stage+1,group*4 + byte,k) : (float) (signed char)
                                    ((packed[group] >>
                                      (byte*8)) & 0xffu)*scale;
                    }
                }
            }

            #pragma unroll
            for (int group = 4; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 =
                            *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc1[i][j] += av[i]*bv[j];
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

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc0[i][j] + acc1[i][j];
                }
            }
        }
        __syncthreads();
    }
}