# Independent W4A4 long-prefill validation audit

2026-09-08. CPU-only subagent: no GPU reservations, CUDA, compilers, background
jobs, production edits, or commits. Parent owns all timing/lock coordination.
GPU0 ONLY remains binding; parallel agents do not authorize parallel GPUs.

## Baseline and honest 2x budgets

Current source independently SHA256-verified:
`studies/qwen35-q4-t64-20260908/affinity-wave.cu`,
`fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47`.
The reference is `aw_q8_service_m64_n128_halfpipe_sync` (line3154), not merely
the legacy `aw_q8_service_m64`. Q4 activations round to half at staging;
Q4_0/Q4_1 weights round to half after dequantization (helpers lines778–813).
The mainloop has two FP32 rails for each group's K0–15 and K16–31, summed
at output. Group-scaled integer arithmetic is a DIFFERENT rounding schedule.
Extract original control unchanged, preserve tails, packing and owner stages.

Measured parent prep=1 seven-sample medians give these screening budgets:

| Projection (N,K) | M, 64 equal experts | T64 us | Maximum complete W4A4 us for 2x |
| --- | ---: | ---: | ---: |
| Gate/up (512,2048) | 128 | 3600.656 | 1800.328 |
| Gate/up (512,2048) | 256 | 7050.244 | 3525.122 |
| Gate/up (512,2048) | 512 | 13664.004 | 6832.002 |
| Down (2048,512) | 128 | 3426.352 | 1713.176 |
| Down (2048,512) | 256 | 6803.548 | 3401.774 |
| Down (2048,512) | 512 | 13665.676 | 6832.838 |

`budgets.py` independently checks raw PASS records, source identity and samples,
then emits medians and hashes. These are planning budgets, NOT a denominator
for a later candidate run and NOT six measured prompt workloads. Re-run paired
current T64 with each candidate. All quantization/packing, scale reductions,
gathers, group corrections, recurring cache/repacking and output reductions
must fit inside the budget; GEMM-only under half of T64 is insufficient.

## Closed leads and permissible reopening

Prior `studies/w4a4-20260908/RESULTS.md` measured N1/N4 dense/decode workloads:
signed integer W4A4 was 0.40–8.86% slower than newer W4A16 R3. Sixteen AND/POPC
plane products made modest internal gains but no replacement win; packed-half
decode/FMA paths lost18–43%. Synthetic output relative-L2 was9.72–10.20%, NOT
model quality. Do not rerun identical geometry or claim their CPU exactness
proves activation fidelity. Large routed M with reuse is a genuinely changed
premise, so long-prefill W4A4 remains TESTABLE, not proven fast or disproven.
The separate prefill study's E2M1 experiments were CPU-only.

Parent's temporary expanded-weight staging lost once recurring preparation was
charged. A free W16 shadow is not a W4 pipeline gain. FP16 accumulation is a
separate lever and cannot count toward this comparison, even when exact on
particular small-integer groups. No DP4A availability is inferred from SM61.

## Reference contracts and CPU evidence

`oracle.py` and `test_oracle.py` are independently reusable, standard-library
only; no imports from performance implementation. Six tests PASS:

- Both families: all256 scalar code pairs in constant G32 groups (512 total)
  and2048 random G32 groups each (4096 total), direct integer vs signed-plane
  and endpoint-SAD identities.
- Signed INT4 codes decode -8..7; production-like symmetric quantizer emits
  -7..7 with G32 maxabs/7 FP32 scale. Direct tests include -8.
- E2M1 codes decode sign/magnitude {0,.5,1,1.5,2,3,4,6}; the integer oracle
  uses DOUBLED values {0,1,2,3,4,6,8,12}. A doubled dot needs division by2,
  not an INT4 reinterpretation. Five signed planes suffice for doubled values;
  naive five-plane expansion is20 AND/POPC terms, not16. Four-bit code storage
  is possible because values are nonlinear; code bits cannot be multiplied
  using INT4 coefficients. The proposed E2M1 quantizer is maxabs/6, G32 FP32
  scale with nearest code, tie-even code-index, preserving signed zero.
- RN-even FP32 rational oracle handles ties/subnormals/overflow;4096 random
  finite FP32 values round-trip bit-identically. FP32 group scale multiply
  followed by exactly rounded group FMA is specified separately from T64.
- Quantizer zero groups, extrema and ties are tested; nonfinites explicitly
  rejected. Scale underflow requires fallback. Finite FP16 activation inputs
  avoid extreme FP32-scale underflow, but arbitrary A32 inputs need a policy.
- Unsigned-offset correction and E2M1-as-INT4 traps have concrete failing
  witnesses. G32 signed INT4 dot magnitude<=2048; doubled-E2M1 dot<=3072,
  both exact in FP32. The latter does NOT imply exact FP16:31 products96 plus
  one product1 yields2977, not representable in half. Accumulate across scale
  groups in FP32 and never move corrections across unequal scales.

These are arithmetic tests, not exhaustive quantizer validation. GPU promotion
requires device-vs-oracle encoded codes, scales, raw integer groups and FP32
outputs; cover every finite half input, all codes, activation outliers, signed
zeros, half rounding boundaries, tiny/negative/large weight scales and ragged
tails. Test independent seeds plus memcheck/racecheck/synccheck. Validate scale
underflow/overflow and no NaN silently accepted. Charge fallback performance.
Q4_1 affine offsets require separate equations/reference; Q8_0 experts need
correct fallback. IQ and NVFP4 weight formats are not two's-complement W4.

## Quality and route-weighted acceptance

1. Name each family separately (INT4 A4 vs E2M1 A4). Neither G32 FP32-scale
   format here is NVFP4. Report actual payload+scale workspace (four bits plus
   32/32=1bit/activation for a standalone FP32-scale G32 representation).
2. Exactness after quantization is only a kernel correctness gate. Model
   accuracy requires byte identity OR PPL delta within +/-0.003 using paired
   KLD methodology, against matched current-Q4 T64/model inputs. No NVFP4
   equivalence without a separately matched NVFP4 model evaluation.
3. Capture actual2k/4k/8k code/prose routing and owner-local service partition
   using a non-dispatch-changing instrument; `GGML_CUDA_MOE_HIST` disables
   the current T64 dispatch and is invalid for unchanged-service capture.
   A full4-GPU capture is NOT authorized for this subagent; ask parent for an
   existing trace or separately authorized acquisition. Never launch it here.
4. Replay all ragged jobs, M16/32/64 tails and M128–512 plus hot M1024–4096,
   both projections and real Q4_0/Q4_1/Q8 fallback mix. Historical4k useful
   work shares aboveM128 are73.57%code/80.72%wiki; those are NOT time weights.
   Do not halve/double aggregated4k counts to fabricate2k/8k routing.
5. Freeze selection before confirmation. Parent serializes GPU0 workers via
   existing shared lock; fresh processes, full provenance, interleaved matched
   controls, at least three independent workers per suite, robust warmup and
   retained raw timing distributions. No concurrent compilers or other timing.
6. For each prompt/domain suite use SUM(baseline complete pipeline times) /
   SUM(candidate complete pipeline times), not average speedup or FLOP-weighted
   ratios. Require>=2.0 for each suite; report per-projection gaps and fallback
   coverage. Gate and up count separately; quantization may be shared ONLY if
   the real fused schedule demonstrably reuses that same activation group.
   Down receives post-SwiGLU activations and needs its own charged quantizer.
7. Distinguish GEMM-pipeline success from complete model prefill tokens/sec;
   no primitive timing establishes end-to-end gain. Parent's W4A16 goal stays
   independent and must not be displaced by this side investigation.

## Safe commands for parent (no build/GPU slot needed)

```sh
PYTHONDONTWRITEBYTECODE=1 python3 studies/w4a4-long-prefill-20260908/verification/test_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 studies/w4a4-long-prefill-20260908/verification/budgets.py
```

No GPU command requested by validation agent. Implementation agent should supply
its build/test command for parent review and its explicitly granted slot.
