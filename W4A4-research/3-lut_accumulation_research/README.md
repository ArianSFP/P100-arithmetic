# P100: replacing packed-FP16 repair with LUT-ready accumulators

9 September 2026. **CPU arithmetic investigation and literature review only.**
No CUDA compiler or GPU was available in this runtime. No GPU kernel was compiled,
executed, or timed. No throughput, model-quality, or novelty claim is made.

## Answer to the motivating question

LUT-GEMM helps by changing the computation, not by fixing FP16 rounding for free.
Precompute partial dot products in the accumulator's representation. Then lookup
and add, without decoding two products from a scalar FP16 result on each K step.
There is still no demonstrated exact one-HFMA2-for-four-MACs implementation here.

A simple state bound clarifies the obstacle: with 32 shared activations all equal
to one, each signed INT4 weight-row sum can be any of 481 integers, [-256,224].
Two independent outputs have 481^2=231361 possibilities, requiring at least 18
bits. One 16-bit state cannot distinguish them all. This bound assumes the final
decoder has only that state and the fixed activations; additional state, rereading
weights, restricted inputs, and approximate outputs change the problem. It is not
a performance lower bound on all algorithms.

## A. Two rows sharing ONE activation, exact packed G32 accumulation

Use offset-binary q=w+8, not raw two's-complement nibble interpretation.
The four bit-plane coefficients are [1,2,4,8], followed by an activation-sum
correction. This matches offset-coded INT4 such as Q4_0's code-minus-eight model;
other formats/scales/codebooks need their own interpretation.

For each activation subvector of length mu=4, construct:

    S(mask) = sum_j bit(mask,j) * a[j]
    c = sum_j max(-a[j], 0)
    U(mask) = S(mask) + c

Every U is nonnegative, with U <= sum_j |a[j]| <= 32. Make a 256-entry paired
lookup table:

    T[m0 | (m1<<4)] = U(m0) | (U(m1)<<16)

Each entry is uint32 and contains one partial sum for each output row. Its
size is 256*4=1024 bytes per mu=4 activation subvector. A scalar 16-entry
FP32 LUT would be only 64 bytes: this pairing is NOT free storage.

For all eight subvectors of a G32 block, keep four packed accumulators:

    P[b] += T[pair_mask[b]], b=0,1,2,3

The masks come from the q-bit planes of the two rows. No per-lookup product
conversion, rounding repair, or lane extraction is needed. At the group end:

    R = P[0] + (P[1]<<1) + (P[2]<<2) + (P[3]<<3)
    correction = 15*sum_k max(-a[k],0) + 8*sum_k a[k]
    dot0 = int(R & 65535) - correction
    dot1 = int(R >> 16)   - correction

All additions are unsigned 32-bit operations. Each bit-plane field is at most
sum |a| <=256; after coefficient combination it is at most 15*256=3840. Neither
stage can carry across a 16-bit lane boundary. This is an exact proof for all
signed INT4 G32 blocks, not merely a random-test observation.

Each output row can have its own weight scale. Apply scales AFTER recovering the
integer dots. Align activation/weight group boundaries; never silently accumulate
across different scale groups. Bias/sum preparation must be charged and reused.

Cost caution: there are 32 table accesses plus 32 packed additions for 64 useful
MACs, before finishing work. An ordinary half2 control needs 32 HFMA2s. This
version solves deferred unpacking but has no automatic instruction-count gain.

## B. Multiple tokens, ONE weight-row mask

When different output columns/tokens use the same weight row, the bit-plane mask
is identical for every token. Thus a table can directly store multiple tokens'
partial sums under ONE key; it does not require the Cartesian product of masks.

For mu=8 and two token lanes, each 256-entry table holds two biased subset sums
in the low/high 16-bit halves of uint32. Total size is 1 KiB per eight activation
positions for two tokens. G32 accumulates and combines exactly as above, but the
correction is different for each token.

For four tokens, put one biased subset sum in each BYTE of uint32. With G16,
each cumulative field is at most sum |a| <=128, so ordinary IADD is a four-way
packed accumulation without carry interaction. Keep FOUR packed accumulators,
one per bit plane. After G16, widen even and odd bytes into two uint32 values
with two 16-bit fields BEFORE applying coefficients:

    E[b] = P[b] & 0x00ff00ff
    O[b] = (P[b] >> 8) & 0x00ff00ff
    E = E[0] + 2*E[1] + 4*E[2] + 8*E[3]
    O = O[0] + 2*O[1] + 4*O[2] + 8*O[3]

Extract the four fields and subtract each lane's correction. Weighted sums are
at most 15*128=1920, safely within 16-bit fields. G32 can use two exact G16
flushes and combine integer results before applying the G32 quantization scales.

Do NOT claim universal G32 accumulation in byte lanes: all -8 inputs can produce
256 in a field. The retained counterexample shows the carry corruption. An
adaptive bound could be explored separately, but this reference uses G16.

The four-token table still has 256 uint32 entries, or 1 KiB per mu=8 block. It
requires quantized INT4 activations; this narrow-table proof does not hold for
arbitrary A16 values.

### Stronger bound: exact four-byte G32 with one endpoint normalization

G16 is a sufficient unconditional bound for the unmodified representation, not
an inherent limit. For any signed INT4 row of length 32, sum(abs(w)) <=255
unless ALL 32 values equal -8. This follows because -8 is the only code with
magnitude 8: one other value reduces the maximum 256 by at least one.

For the exceptional all-minus-eight row, construct the LUT from w'=-4 instead,
and retain a factor f=2 for that row/group. All other rows use w'=w and f=1.
Then sum(abs(w')) <=255 for EVERY input. Compute the dot by byte-packed G32
accumulation and restore the factor after unpacking. The adjustment is exact,
requires no activation-dependent fallback, and can be made during weight staging.
It does NOT require changing stored model weights. Factor two can be folded into
FP32 weight-scale application after the integer dot. Never double a finite half
scale in FP16 without considering overflow; widen the scale first.

This extends the four-row common-mask scheme to a full G32 with only one finish.
The proof and tests include mixed ordinary/exceptional row quartets and the
near-boundary row containing 31 copies of -8 and one -7 (L1=255). The final
weighted 16-bit field is at most 15*255=3825. The table still has 256 entries per
mu=8 slice. This is a stronger performance candidate than forcing G16 flushes,
but table construction and finishing costs still need actual GPU measurement.

## C. Preserve the native FP16 machinery: half2-valued partial-sum LUT

For two token outputs with common weight bit masks, store ordinary two-lane
half2 partial sums directly. They are NOT two values packed into one scalar
FP16. With signed two's-complement weight bit planes [1,2,4,-8]:

    half2 P[4] = {0};
    for each mu-subvector:
        P[b] = hadd2(P[b], table[mask[b]])
    out = P[0] + 2*P[1] + 4*P[2] - 8*P[3]

A4 subset sums and G32 plane accumulators are small exact integers. Combining
planes in the displayed order is also exact in FP16 for raw signed INT4 G32:
positive lower planes correspond to coefficients in [0,7], and the final signed
weight lies in [-8,7]. Each relevant intermediate is within [-2048,2048]. Use
HFMA2 for the three weighted combinations and FP32 for external scales/groups.

This is the most direct control for existing fast FP16 tiles. For mu=8, one row
and two tokens, four lookups + four HADD2 operations implement 16 useful INT4
MACs per subvector. Ordinary half2 needs eight HFMA2s. Those counts are equal
before overheads and do NOT imply a speedup. Vectorized shared loads may reduce
issue instructions but not the bytes/transactions required.

## D. Prefill-specific transpose: build LUTs from WEIGHTS

Since both sides are signed INT4 in this experiment, interchange table-building
and bit-index roles. Construct subset-sum tables for four weight rows, and use
each token's activation bit-plane masks as keys. All four output rows share one
activation mask, so no Cartesian-product table is needed. This is not a free
W8A16 operand flip: it changes the tiling and on-chip representation.

This orientation reuses a weight-derived table across a token tile. It avoids
building a new activation-derived table for every token. It is a candidate for
prefill, not a demonstrated winner, and its bias/scale bookkeeping follows the
same algebra with weights and activations interchanged.

One possible bank-owned layout:

    uint32 table[256][32];
    value = table[activation_mask][lane_id];

Each lane's column holds a DISTINCT table for four output rows. A warp covers
128 rows. There are 32 KiB of tables for one mu=8 weight slice. Assuming 32
four-byte banks and aligned base, the bank index is

    (mask*32 + lane_id) % 32 = lane_id.

This is a bank-conflict-free address mapping for those 32-bit accesses, even
when masks differ. It is not a guarantee of instruction throughput, occupancy,
or conflict-free construction/epilogue. Test all stages. P100 has 64 KiB shared
memory per SM and a 48 KiB per-block limit; a 32 KiB allocation restricts design
freedom and allows at most two such blocks by shared-memory capacity alone.
Two such full tables cannot be double-buffered within one P100 block.

Keep original INT4 weights compressed in HBM. Build tables only for an on-chip
slice. Persisting the full expanded representation would massively inflate
weight storage. Compute offsets during staging rather than silently adding
large per-weight metadata. Do not assume table reuse crosses expert boundaries.

Arithmetic-count screen, mu=8 and four outputs:

- ordinary half2: 16 HFMA2s for 32 useful INT4 MACs;
- four-byte LUT: 4 shared loads + 4 packed adds, BEFORE table construction,
  masks/addresses, byte-lane widening, plane combination, scales and reduction.

The 2x ratio counts selected inner operations ONLY. Use the normalized G32
variant above to halve finishing frequency relative to G16. Finishing and setup
work can still consume the whole saving. No 2x hardware speed follows.
Test real T64-like token reuse, not just asymptotic large batches.

## Research evidence and negative controls

- LUT-GEMM (ICLR 2024) precomputes activation-dependent binary partial dots and
  replaces multiplication with lookup/reduction. The paper focuses on single-
  batch/generation and explicitly notes reduced benefits at larger batches.
  Its A100 Table 1 INT4 timing is 0.2688 ms versus AWQ 0.3238 ms, roughly 1.20x;
  its 2.1x headline is a different 3-bit model-level comparison against OPTQ.
  These are not P100 T64 results.
- The user's archived W4A16 round-1 study already tested a 16-entry FP32 SHFL LUT.
  Square N1 was 52.928 us versus direct 49.019 us; it did not beat the best direct
  kernel. This was FP32 accumulation, not the user's newer fast HFMA2 prefill
  kernel. A generic LUT port is not a new untested optimization category.
- T-MAC provides useful register-LUT, symmetry, layout and late-widening ideas.
  Its CPU table instructions/rounded averaging do not automatically transfer
  to GP100 or preserve exact integer results.
- LUT Tensor Core and FIGLUT address lookup/accumulation and table access through
  CUSTOM hardware. Their throughput numbers are not software modes on P100.
  Software ideas such as table-precompute fusion remain useful inspiration.
- FLUTE is principally LUT-based DEQUANTIZATION feeding floating-point GEMM;
  it is not the same as multiplication-free partial-sum lookup.
- ULPPACK explicitly addresses packed integer MACs and local accumulation to
  amortize extraction. Its integer-register budgets do not become FP16
  significand bits.

## Prioritized hardware experiment

1. Run matched native-half2 arithmetic and half2-valued LUT controls with exactly
   the same A4 codes, weight scales, shapes, output definitions and complete
   preparation cost. The existing W4A16 path is a separate precision baseline.
2. Compare paired-row G32 integer LUT, four-output G16 control, and four-output
   normalized G32 integer LUT. Include
   finishing work in every latency claim; keep arithmetic-only prebuilt-table
   timing strictly diagnostic.
3. For compute-heavy prefill, test weight-table orientation and bank ownership,
   sweeping mu=4/5/8 and token reuse 16/32/64/128. Table construction, barriers,
   occupancy, registers, instruction mix and real expert token counts must be
   measured. Do not allocate more than 48 KiB per block or assume shared state
   survives arbitrary block scheduling.
4. Only after a full-device-pipeline win, evaluate real-model A4 quality. Exact
   arithmetic on quantized codes is not preservation of the original A16 model.
   Respect production KLD/perplexity requirements and shared-rig reservation
   rules. No firmware/unknown-opcode experiments are needed for these candidates.

## Reproduction

    python verify.py --blocks 20000
    g++ -O2 -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined \
        -fno-sanitize-recover=all host_check.cpp -o host_check
    ./host_check

Python tests construct and index the actual small tables. Eight numerical
variants/controls each test 20000 blocks, including endpoint blocks. Total:
560000 output-dot comparisons, zero mismatches. All 65536 activation quartets
and their 1048576 table entries satisfy the asserted bounds. An independent
scalar C++ implementation tests 10000 two-lane G32 blocks, 10000 four-lane G16
blocks, and 10000 normalized four-lane G32 blocks, 100000 output dots, under UBSAN; zero mismatches. Random tests are not
exhaustive over complete dot products; the stated bounds give the general proof.

## Primary sources

LUT-GEMM: https://arxiv.org/abs/2206.09557
Official code: https://github.com/naver-aics/lut-gemm
T-MAC: https://arxiv.org/html/2407.00088v2
LUT Tensor Core: https://arxiv.org/html/2408.06003v2
FIGLUT: https://arxiv.org/html/2503.06862v1
FLUTE: https://aclanthology.org/2024.findings-emnlp.724.pdf
ULPPACK: https://proceedings.mlsys.org/paper_files/paper/2022/file/e09d45e14e9ece7142217550ddd3c4d0-Paper.pdf
Pascal: https://docs.nvidia.com/cuda/archive/12.8.1/pascal-tuning-guide/index.html
CUDA programming guide: https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html
Half2: https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html
PTX: https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html
User's negative LUT control: https://github.com/ArianSFP/P100-arithmetic/blob/d8a16b7f149683d22bc43238a3409cbb15dfe72b/archive/w4a16-20260908/RESULTS.md

Our exact packing variants and tests are adaptations/derivations in this
investigation, not results claimed by these sources or a claim of first invention.
