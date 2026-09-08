# W4A16 round 3: paired sharing and exact-order local reduction

2026-09-08. User explicitly authorizes GPU2; GPU1 belongs to the separate
prefill session. No archive edits, production changes, model run, commit/push,
activation quantization, FP16 arithmetic or unknown encodings.

## New evidence and bounded hypotheses

The round-2 winner broadcasts 32 individually widened FP32 activations per
group. It did not cross the warp-sharing variant with CTA warp-count tuning;
the earlier CTA sweep used scalar/pair loads. Three bounded axes are new:

1. Retain the winner's warp sharing; test 1/2/8 warps per CTA and R1/R2/R4.
2. Shuffle 16 packed FP16 pairs instead of 32 widened floats. This halves
   shuffles but increases per-lane widening; instruction savings are NOT
   assumed. Test paired sharing at 2/4 warps per CTA.
3. Keep all 32 accumulation stripes and their exact reduction tree, but
   compute them within one CTA using 8/16/32 warps. This removes the global
   partial-output write/read and separate reduction launch, trading them for
   shared memory, barriers and fewer independent CTAs. This is not the closed
   production launch-only fusion proposal or the rejected changed-stripe order.

39 total configurations: two live archived controls, 25 sharing/CTA variants,
12 local-reduction variants. No open-ended tuning campaign. Both N1 and N4
control cubins are the sealed round-2 binaries, loaded in every fresh worker.
The sealed round-1 module supplies the original reduction/conversion kernels.

## Gates and measurements

- Unchanged Q4_0 codes/FP16 scales, FP16 input bits, sequential 32-element
  FP32 group FMAs, group-scale FMAs, 32-stripe accumulation and reduction tree.
- CPU checks normally/UBSAN, plus mapping/reduction invariants for fused CTAs.
- Compile SM60; inspect resource use and SASS, no half multiply/accumulate or
  spill/local-memory traffic. Check launch dimensions before any GPU call.
- Full numerical suite on GPU2: all configurations, row/group/batch tails,
  four numerical families and all finite half activation patterns. Exact
  operation-order reference plus independent rational reference retained.
- Memcheck, synccheck and racecheck smoke are required because local reduction
  introduces shared memory. Stop on any CUDA/correctness/health error or hang;
  investigate without automatic retry/reset.
- Sweep four existing (M,K,N) shapes, freeze a small candidate/control set,
  then 3 fresh workers per shape, 9 rotated rounds, discard round 0, 3 complete
  pipelines/event, 200 ms warmup. Keep outliers and report worker ranges.
- A tiny or inconsistent difference is not a demonstrated gain. Compare live
  controls on the same GPU; do not divide by historical GPU1 times. Timing
  includes all device stages/scaling/reduction, excludes CPU repack/alloc/copies.

GPU2 stays reserved through builds, gaps and final health/artifact review.
Other GPU workloads are not killed or used. Shared-host interference is logged
and limits timing certainty. Production comparison on real-layer inputs and
ppl +/-0.003 remains the next integration gate even if synthetic timing wins.

References: NVIDIA's documented [shuffle semantics](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)
and [Pascal resource limits](https://docs.nvidia.com/cuda/archive/12.8.1/pascal-tuning-guide/index.html).
