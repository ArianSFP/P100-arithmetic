__device__ __forceinline__ void w8_load_fp16_8(const void*input,size_t index,half*h){
 const uint4 v=__ldg((const uint4*)((const unsigned short*)((uintptr_t)input&~uintptr_t(1))+index));
 const unsigned w[4]={v.x,v.y,v.z,v.w};
 #pragma unroll
 for(int i=0;i<8;++i)h[i]=__ushort_as_half((w[i/2]>>((i%2)*16))&65535);
}
#include "decode.inc"
struct w8_half_smem {half A[1][32][128];half2 B[1][16][64];};
template<bool BF16_INPUT,int FLUSH>
__global__ __launch_bounds__(256, 3)
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
    const int producer_half = warp >> 1;
    const int b_r = (warp & 1)*32 + lane;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/AW_T64_ROWS;
    const int ngroup_shift = __ffs(n_ngroups) - 1;
    const int ngroup_mask = n_ngroups - 1;
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id >> ngroup_shift;
        const int ngroup = work_id & ngroup_mask;
        const int col0 = ngroup*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = ngroup;
        const int b_local = b_r % AW_T64_ROWS;
        size_t input_stage_base =
                (size_t) input_row*k + a_k;
        const char * weight_stage =
                desc.weight +
                (size_t) ntile*n_kblocks*
                    AW_T64_STAGE_BYTES;

        half2 acc0[8][2],acc1[8][2];float total[16][2]={};
        for(int i=0;i<8;++i)for(int j=0;j<2;++j)acc0[i][j]=acc1[i][j]=__float2half2_rn(0.f);
        const int buffer=0;
        for(int stage=0;stage<n_kblocks;++stage){
            const bool last=stage+1==n_kblocks;

            half h[8];w8_load_fp16_8(desc.input,input_stage_base,h);
            #pragma unroll
            for(int i=0;i<8;++i)sm.A[buffer][a_k+i][a_r]=h[i];
            {
            const int extra_local=tile.row+min(a_r+64,tile.rows-1);
            const int extra_row=desc.input_rows?desc.input_rows[extra_local]:extra_local;
            const size_t extra_base=size_t(extra_row)*k+a_k+stage*AW_KSTAGE;
            half h[8];w8_load_fp16_8(desc.input,extra_base,h);
            #pragma unroll
            for(int i=0;i<8;++i)sm.A[buffer][a_k+i][a_r+64]=h[i];
            }
            const float scale =
                    __half2float(
                        *(const half *)
                        (weight_stage +
                         b_local*2));
            const char * values =
                    weight_stage +
                    AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + producer_half*8;
            const uint2 packed4 =
                    __ldg((const uint2 *) values);
            const uint32_t packed[2] = {packed4.x, packed4.y};
            #pragma unroll
            for(int group=0;group<2;++group){
                unsigned q=packed[group];
                w8_decode4(q,scale,sm.B[buffer][producer_half*4+group*2][b_r],sm.B[buffer][producer_half*4+group*2+1][b_r]);
            }
        
__syncthreads();
            const int row0 = warp*16;
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const half2 *av=(const half2*)&sm.A[buffer][ks][row0];
                    const half2 bv[2]={__half2half2((ks&1)?__high2half(sm.B[buffer][ks/2][lane]):__low2half(sm.B[buffer][ks/2][lane])),__half2half2((ks&1)?__high2half(sm.B[buffer][ks/2][lane+32]):__low2half(sm.B[buffer][ks/2][lane+32]))};
                    #pragma unroll
                    for(int i=0;i<8;++i){
                        #pragma unroll
                        for(int j=0;j<2;++j)acc0[i][j]=__hfma2(av[i],bv[j],acc0[i][j]);
                    }
                {
                    const int ks = group*4 + kk + 16;
                    const half2 *av=(const half2*)&sm.A[buffer][ks][row0];
                    const half2 bv[2]={__half2half2((ks&1)?__high2half(sm.B[buffer][ks/2][lane]):__low2half(sm.B[buffer][ks/2][lane])),__half2half2((ks&1)?__high2half(sm.B[buffer][ks/2][lane+32]):__low2half(sm.B[buffer][ks/2][lane+32]))};
                    #pragma unroll
                    for(int i=0;i<8;++i){
                        #pragma unroll
                        for(int j=0;j<2;++j)acc1[i][j]=__hfma2(av[i],bv[j],acc1[i][j]);
                    }
                }
}
            }
            

            #pragma unroll
            
            if constexpr(FLUSH>0) {
                if((stage+1)%FLUSH==0 || last){
                    #pragma unroll
                    for(int i=0;i<8;++i){
                        #pragma unroll
                        for(int j=0;j<2;++j){
                            float2 lo=__half22float2(acc0[i][j]),hi=__half22float2(acc1[i][j]);
                            total[2*i][j]+=lo.x+hi.x;total[2*i+1][j]+=lo.y+hi.y;
                            acc0[i][j]=acc1[i][j]=__float2half2_rn(0.f);
                        }
                    }
                }
            }
__syncthreads();
input_stage_base+=AW_KSTAGE;weight_stage+=AW_T64_STAGE_BYTES;
}
        const int row0 = warp*16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    if constexpr(FLUSH>0)output[lane+j*32]=total[i][j];else {float2 lo=__half22float2(acc0[i/2][j]),hi=__half22float2(acc1[i/2][j]);output[lane+j*32]=(i%2?lo.y:lo.x)+(i%2?hi.y:hi.x);}
                }
            }
        }
        __syncthreads();
    }
}
