# W4A16 round 2: improve the confirmed R4/S32 kernel

- Preserve the original study directory and sealed cubin. Use its exact R4/S32
  and R2/S32 kernels as live controls on each tested GPU, not historical times.
- Keep Q4_0 codes/scales and FP16 activation bits unchanged. Require the same
  sequential group FMAs, 32 group stripes and final FP32 tree, byte-identically.
- Bounded axes: signed BFE nibble extraction; 32/128-bit activation loads;
  warp activation broadcast; streamed weight words; vector-friendly lossless
  weight layout; rows/thread 2/4/6/8; CTA warps 1/2/4/8; batch-column reuse.
- Build, SASS/local-memory audit and CPU tests first. Fresh GPU workers,
  exhaustive finite conversion/layout checks, tails/edges and sanitizer gates.
- Tune on GPU1, freeze candidates before repeated confirmation on GPUs1/2/3.
  Use one GPU worker at a time. Do not pool absolute times from different GPUs.
- Hold GPUs1/2/3 across builds, worker gaps and final checks under the shared
  coordination claim. GPU0 desktop untouched. No four-GPU/model job, resets,
  driver changes, unknown instructions, half accumulation or production edits.
- Final report must distinguish kernel-only gains from model/stock speedups;
  include preprocessing/layout costs and unsuccessful candidates.
- SASS precheck: the sealed baseline already emits signed BFE; do not claim
  a new extraction mode. Replace costly general 64-bit address expressions
  with explicit tile/lane algebra and bounded unsigned 32-bit element indices.
  Extra GPUs are used for per-device confirmation, not pooled timing samples.
