# Pre-launch independent implementation review

Read-only CPU review, 2026-09-08. No compiler, CUDA, GPU reservation or external
mutation. User's latest GPU2/3 authorization supersedes earlier GPU0-only
assignment: parentGPU2, DaltonGPU3; parent still centrally coordinates slots.
No direct agent-messaging tool is exposed to this reviewer; handoff via parent.

Reviewed hashes before any owner fixes:

- kernels.cuh: b554a40f04ae3abb39e8e825c7bb7777165d6cecb2a8f0eef0239d9d706b8c88
- worker.cu: 8c6069972c8bb9eef71166330e064fcad9bd44bec3e1940af1039fa707382f5a
- build.py: 043ee11cb06e25a531ecc3698c506522db121e93cd8bc7a8e3d87f31cd549c76

## Immediate launch blocker (owner resolved during review)

worker.cu:35–36 hard-codes GPU0 UUID and rejects GPU3. Update the worker guard
and supervisor/README consistently for Dalton's authorized physical GPU3;
do not weaken it to accept arbitrary devices. No performance trial should be
launched under stale GPU0 instructions. Reviewer does not change owned source.

Re-read after owner edits confirms worker now uses GPU3 UUID
4868830a-c1cf-90bd-8018-2360c55293b8 and rejects other GPUs. Worker hash now
a98377225149c948dcfc04aa38833b1b0cdc4268fd119c0f7bdeb8cceed7241e.
Build now hashes supervisor dependencies too; hash
035ed0d4ce72d14de4315b6531672f8e4d02c02c00909f4b71ffc9cc02067882.
The GPU guard blocker is resolved; poisoning/fixture gaps below remain at
this re-read. Full supervisor review is outside the three-file assignment.

## Correctness-validation improvements before promotion

1. worker.cu:78 reuses one output buffer across all candidates with no poison.
   A candidate that omits writes can inherit another candidate's valid result
   (especially popc4 after popc2), passing samples/finite checks. Before EACH
   correctness launch, fill output with NaNs (e.g. cudaMemset 0xff), outside
   timed events. Require every output finite afterward. This is a harness gap,
   not a finding that current kernel omits writes; CPU mapping checks pass.
2. worker.cu:39–50 has one fixed random fixture; inputs are pre-rounded half,
   scales make every q*d half-exact. It cannot validate actual A32->A16 rounding,
   arbitrary weight-dequant rounding, zero groups, ties, range/outliers or seeds.
   Add explicit fixtures for all-zero and signed-zero groups; maxabs7 and
   activations .5/1.5/2.5 for RN-even witnesses; finite-half extrema/subnormals;
   raw A32 just above/below half midpoints; arbitrary positive/negative half
   weight scales. Include code -8 through direct preparation vectors because
   symmetric activation quantization intentionally emits only -7..7.
3. When adding non-half A32 input, CPU reference MUST separately construct
   rounded A16 before group max/quantization AND T64 dot. Current CPU references
   read a directly, correct only because the current fixture was pre-rounded.
   Preserve original A32 as the shared input to both GPU paths; never feed T64
   reconstructed A4 or silently give it different activation information.
4. Require finite inputs after half conversion or define a logged fallback for
   NaN/Inf/out-of-half-range A32. The current quantizer fmax/divide/integer cast
   does not provide a reliable rejection policy for such values. Present finite
   [-2,2] fixture is safe; production-general claims are not covered.

Initial bounded sanitizer smoke can be useful after GPU identity is fixed;
missing broader fixtures prohibit promotion/quality claims, not mathematics
exploration. Poisoning is cheap and should precede that first smoke.

## Findings that PASS source/model review

- Original current T64 function body is byte-identical to the hash-verified
  source. build.py does not substitute its A4 control for T64. Source extraction
  preserves A32->A16 staging and two FP32 rails followed by output addition.
- Signed plane arithmetic is correct: coefficients [1,2,4,-8], offset-binary
  Q4_0 unpack subtracts8, activation quantizer ballots signed codes. Max group
  magnitude<=2048, exact int32->FP32 conversion. No missing unsigned correction.
- Quantizer max reduction, RN FP32 maxabs/7, RN FP32 v/d and integer RN agree
  with oracle for finite-half input under host default RN. Zero max takes zero
  code/scale. Host nearbyint follows current host rounding mode; explicitly
  assert FE_TONEAREST if future wrapper could change it. No fast-math flag.
- GPU group scale is __fmul_rn(ad,wd), followed by __fmaf_rn(dot,scale,acc).
  Host uses scalar integer codes and std::fma, not GPU plane dots/results: the
  core output oracle is meaningfully independent. Fully checks small outputs,
  samples256/expert otherwise; full finite checks alone aren't full correctness.
- All W4A4 timed iterations call run(c,true): both activation quantizer and
  weight-plane conversion execute inside the CUDA event interval EVERY time.
  Baseline's conversion is inside unchanged T64, not unfairly charged the A4
  prep. Allocations/HTD and initial shared tiled-Q4 packing excluded on both.
- Integer dot plus FP32 scale/accumulation; no half accumulation in source.
  Build rejects HFMA2/HMUL2, with disassembly evidence pending actual build.
- Both RM2/RM4 output maps cover tails once; shared producer/consumer phases
  have CTA-wide barriers. No source-level obvious OOB/race identified. Hardware
  sanitizer remains necessary. Independent1024 packed groups and24 tail cases
  pass review_checks.py, which does not execute implementation code.

## Acceptance blockers (screening remains explicitly limited)

- worker.cu:58 dispatches every tail through M64; actual cohortrail M32/M16
  service is absent. Arbitrary counts help, but not an unchanged full dispatcher.
- No indexed gather/owner reduction/recurring original compressed-cache miss
  costs; no Q4_1/Q8 fallback, actual model scales or traced routing values.
  The README correctly disclaims these. A fast result is primitive screening,
  NOT verified2x current service at2k/4k/8k. Restore full costs before acceptance.
- Quantized activation error AND changed weight/group rounding must meet paired
  model PPL±.003 or identity before quality qualification. rel_l2 is diagnostic.
  This implementation is INT4 A4 only, not E2M1 or NVFP4.
- matched_A4_F32 reconstructs codes from planes; beating this attribution control
  does not mean beating optimal FP32 W4A4 or current T64. Use current_q4_T64_A16
  as paired denominator, including complete prep; FP16 accumulation cannot count.

CPU recheck, no build or CUDA:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 studies/w4a4-long-prefill-20260908/verification/review_checks.py
PYTHONDONTWRITEBYTECODE=1 python3 studies/w4a4-long-prefill-20260908/verification/test_oracle.py
```
