# Two changed-liveness cap4 candidates — CPU prepared only

Original tile256.cuh remains FROZEN and untouched. Parent hardware evidence:
cap4 spills(rejected without execution); cap3 passes hard-scale identity and
memcheck/racecheck/synccheck but gate/up M256~5.023ms is not better than compact
~5.013ms. This new file changes future operand liveness, not another tile sweep.

Include liveness.cuh after verified wide.cuh; two distinct instantiations:

    tile256_liveness_q4<false><<<grid,256>>>(a,w,out,t,n,k,experts)
    tile256_liveness_q4<true ><<<grid,256>>>(a,w,out,t,n,k,experts)

Both fix unroll32 and __launch_bounds__(256,4), retain16KiB shared,4x4 outputs
per thread and32 FP32 rail accumulators. Target<=64registers/no spills must be
proved by the compiler, not assumed from C++ lifetimes.

- false: no next-group A/B fetch until current group computation is complete.
  Eight raw activation registers therefore need not survive the compute loop.
  It loses global-load overlap but may admit4blocks/SM. Fetch after compute is
  private-register-only; the existing barrier still precedes shared overwrite.
- true: preserve next-group prefetch, but convert eight raw A32 values immediately
  to four half2 registers via __floats2half2_rn. Shared staging widens these halves
  toFP32. This moves the SAME RN A16 conversion from stage to fetch; it neither
  retains more A32 precision nor uses FP16 accumulation. All conversion remains
  inside timed GEMM. Transient float4 loads may still increase register pressure;
  four source-level half2 variables are not proof of four physical live registers.

Both retain one packed Q4 word/scale per thread, verified decode_q4_pair, identical
K0..15/K16..31 persistent FP32 rails and finalFP32 sum, the same tails and output
mapping. No weight scale movement, additional global scratch, or external prep.
Half operations outside verified weight decode must be conversions/packing only,
never HADD2/HMUL2/HFMA2 used on activations or accumulators. Parent SASS precision
gate remains necessary; HFMA2 forbidden everywhere.

No compiler/CUDA/GPU slot requested or used by this source-preparation task.
CPU model checks every finite half bit pattern in packed pairs, raw-A32 midpoint
witnesses/lane placement, delayed vs early fetch stage ordering, and inherited
geometry. They do not prove device conversion flags, register reuse or hardware
correctness. Parent should first compile/reject spills, then run the same hard
scale/raw-A32 oracle and sanitizer gates before paired GU/down timing. Keep the
frozen compact winner and unchanged current T64 controls; charge all work.

    python3 studies/w4a16-tile256-20260908/test_liveness.py
