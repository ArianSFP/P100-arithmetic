# In-progress GPU0 long-prefill study

Goal remains OPEN. No1.5x workload-qualified or full-model gain yet. All runs
here use GPU0 only. GPU0/global lock have now been released at the staging
checkpoint for waiting sessions; see coordination before reacquisition.

## Current-service comparison, first screen

64equal-size experts, FP32 accumulation, all activation preparation charged.
N512/K2048 gate/up; N2048/K512 down. Seven rotated samples,8iterations/sample,
2s warmup, candidate grid280 and current service grid112 (source2*SMs).
Raw tags `current-{gu,down}-m{128,256,512}`. Not a ragged routing replay.

| Projection | M | Current Q4 T64 us | Fused M32 us | Ratio to current | Ratio to fastest legacy control |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up | 128 | 3600.656 | 2505.044 | 1.437x | 1.239x |
| down | 128 | 3426.352 | 2553.108 | 1.342x | 1.256x |
| gate/up | 256 | 7050.244 | 4953.436 | 1.423x | 1.207x |
| down | 256 | 6803.548 | 5057.852 | 1.345x | 1.243x |
| gate/up | 512 | 13664.004 | 9842.116 | 1.388x | 1.203x |
| down | 512 | 13665.676 | 10107.565 | 1.352x | 1.229x |

Do not hide the faster legacy control: the current compressed-Q4 runtime path
has format handling and operand rounding costs absent from the older primitive.
Neither denominator proves a model throughput gain. Warm allocation, HTD and
static weight repack excluded; current fixture scales make weight operands
half-representable. Real mixed Q4_0/Q4_1, arbitrary scales, ragged tails, owner
service and model accuracy remain required before acceptance.

Q4_0-only specialization of current T64 measured1.159–1.176x current, with exact
split-K output equality on fixtures;121registers versus128,32KiBshared,no spills.
The original current-control kernel body is extracted byte-for-byte unchanged.
No FP16 accumulation; both preserve the same FP32 partial sum schedule.

## Larger fused M64 tile rejected

At M256/e64, fused M64U32 takes5950.956us gate/up and6341.092us down;
the paired fused M32 takes4954.144/5060.032us. M64U16 also loses. Larger tile
reuse did not offset resource/scheduling costs (114/115registers,16KiBshared,
no spills). Do not repeat this unchanged. Tags `m64u{16,32}-{gu,down}-m256`.

Three offline checks pass: original control identity, precision/dispatch/spills,
and complete manifest hashes. `current-smoke` and `m64-smoke` pass memcheck with
zero errors. Every candidate output is checked for finiteness/differences;
new sequential kernels match every frozen sequential winner word. Independent
CPU FMA oracle checks all8448outputs in current smoke and256samples/config
in large cases. This is not PPL/KLD qualification.

## Temporary weight staging rejected

Implemented staging.cuh: Q4→temporary transposed FP16 weights, then FP32 GEMM.
Both U16/U32 pass initial memcheck and numerical checks.83/82registers,12KiB
shared,no spills. The64-expert fixture requires128MiB additional weight scratch.
All dequantization traffic is charged EVERY prep=1 run; prep=0 is diagnostic only.

At M256, U32 pipeline5683.516us gate/up /5782.712us down, versus paired fused
4953.176/5058.948us. U16 pipeline5660.488/5754.008us also loses. The staging
pass adds about720us, while the GEMM itself is approximately unchanged. Closed
unchanged; do not reinterpret prep=0 or a precomputed shadow as a pipeline win.
Raw tags staged-u{16,32}-{gu,down}-m256; snapshots/v3 freezes built source/binary.
Release record gpu-results/release-1788899791180594700.json: idle23MiB/ECC0,
normal exit, no faults/resets/kills, existing editor retained.

## Next lead (CPU-prepared, NOT compiled or GPU-tested)

wide.cuh:64x128 tile,128threads,8x8 outputs/thread, vector shared loads; retains
compressed word-major Q4 input and fused activation preparation. Test F32 weight
dequantization first, then an exact-result packed-half dequantization comparator.
Both GEMM paths use FP32 weights-times-activations and FP32 accumulation.

The packed comparator forms half2(1024+code), subtracts1032 exactly, multiplies
by the half scale with half rounding, then widens the weight toFP32. This must
equal current T64's FP32 dequantization rounded tohalf: a half scale times a
four-bit integer has an exact binary32 product. CPU model covers every finite
half scale/all16codes including overflow and signed zero, plus tile/address
coverage. Hardware proof pending via --quant-check1: compare1,015,808 paired
outputs against CPU FP32-product/half-round oracle before any GEMM measurement.

Precision gate now permits HMUL2 only inside explicitly named weight-decode
test/packed-wide kernels; HFMA2 remains forbidden and the GEMM mainloop is
FP32 FFMA. This is NOT the forbidden FP16 partial-accumulation lever. The wider
tile and exact packing have not been compiled or timed; no gain claimed.

## Historical next-lead hypothesis (resolved above)

Test whether dequantizing a temporary FP16 weight tile once for reuse across
many token tiles beats repeated nibble unpacking/scaling. Charge the entire
dequantization pass, allocation footprint and expanded-memory traffic; no free
precomputed W16 shadow claim. Keep native W4 storage and FP32 accumulation.
This is a prefill large-M experiment, not the closed small-M decode widening
study. Preserve current-service weight rounding and establish a proper oracle
for arbitrary scales before treating it as deployable. A win still needs the
ragged workload and per-prompt acceptance in LONG-PROMPT-TARGET.md (sibling).
