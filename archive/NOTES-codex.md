# Codex notes — isolated HFMA2 GP100 probe

This is a scratch experiment under `/home/arian/hfma2-gp100-probe`. It does
not modify the llama.cpp worktrees or protected builds.

## Initial state

- CUDA toolkit: `/usr/local/cuda-12.8`, `nvcc` 12.8.61.
- `nvcc --list-gpu-arch` includes `compute_60`.
- The sandbox cannot access the NVIDIA driver; GPU execution is pending an
  approved outside-sandbox run and an idle-GPU check.
- The shared project notes record a prior gpuocelot/cuasm SM60 round-trip gate,
  but the checkout is not present in this new directory yet.

## Safety status

- At the initial setup point, no candidate had been executed.
- The 256-entry selector set is not generated until the baseline instruction
  word and exact section-relative offset are recorded from the compiled cubin.
- No automatic reset logic is included.

## Measured results

- Target `.text.probe + 0xf0`, baseline word `0x5d0003800047020a`.
- Baseline, B=`10`, B=`11`, B=`01`, A=`01/10/11`, and C=`01/10/11`
  candidates all loaded and completed on a P100 in fresh workers.
- B=`01` is labelled `.F32` by `cuobjdump` and behaves as a scalar FP32 source
  with packed-half outputs. It is not evidence for smaller packed lanes.
- The initialized merge template's first `.MRG_H0` candidate returned
  `CUDA_ERROR_ILLEGAL_INSTRUCTION` (715). The sweep stopped; `.MRG_H1` was not
  attempted and no reset was issued.
- The initial no-CUDA reservation check saw idle GPUs, but a separate Qwen3.8
  server started later and was found resident during the merge health check.
  No further GPU work is permitted until that server releases the rig.

## Extended investigation - 2026-09-08

- Current findings: [research report](research-20260908/REPORT.md). Treat its
  audit as superseding broad claims in the earlier HFMA2 results.
- User lifted the sub-Q8 prohibition; global Codex AGENTS.md now permits
  Q4/Q2 use and evaluation. Claude memory remains historical and read-only.
- Corrected an important prior research conclusion: unsigned
  VABSDIFF4.U8.U8.ACC is native on SM60. CUDA 12.8 emits it and the P100
  executed it correctly. This is a documented PTX operation, not hidden DP4A.
- Signed W2xA8/W4xA8 reconstruction using unsigned SAD and masks, and fused
  I2F signed-byte conversion, passed 262144 cases each on GPU 0.
- SAD-only four-chain throughput was about 1.95 T thread-instructions/s;
  no full-kernel or inference speedup has been demonstrated.
- Compiled 22 diagnostic kernels. Explicit BFE + integer-to-float folds to
  I2F.F32.S8 with .B2; normal high-level half broadcasts already fuse.
- VMAD's earlier local Q8 neutral result and generic half2/FP64 packing
  failures remain closed for those tested workloads. LUT bank layout and
  unsigned SAD provide specific new evidence, not a reason to repeat them.
- GPU reservations/checks are recorded in the active Qwen3.8 coordination
  file. The user requested holding GPU 0 across the remaining test series.
- The remaining series and final artifact/health checks completed; the
  reservation was held through them and then explicitly released. Final
  check: no compute processes, 5 MiB/GPU, 0% utilization, zero volatile
  uncorrected ECC errors. No further tests are running.

## Endpoint-SAD formulation — 2026-09-08

- New isolated work: [endpoint-SAD findings](endpoint-sad-20260908/RESULTS.md).
- The user's endpoint identity is correct. CUDA 12.8 distributes the plane
  complement into per-quad operations in this layout, replacing the AND cost.
- Reversing endpoints to 255*b removes that cost with signed correction
  dot=-(Eweighted+sum_a+128G+sum_w)/2. CPU math, packed-layout and Q2_K-style
  affine algebra tests pass; native SM60 SASS and no-spill gates pass.
- Corrections/preparation still matter: full W2 matvec static instruction
  count is nearly unchanged versus masked SAD. No speedup has been measured.
- All nine comparison arms and a fresh-worker reservation-gated supervisor
  are built. No endpoint-SAD GPU execution occurred: LMCache still explicitly
  holds all four GPUs between its tests. This investigation is queued and
  has NOT acquired a conflicting reservation. Keep any later reservation
  held for the entire series, inter-run gaps and final checks.

## Compressed layouts and register-LUT - 2026-09-08

- CPU/compiler-only [follow-up](layout-lut-20260908/RESULTS.md), as requested
  while GPUs remain reserved. No CUDA/NVML initialization or GPU job.
- Storing complemented byte-sign bits OFFLINE fixes the prior runtime-XOR
  problem. Four lossless layouts and matched ordinary controls validated.
- Important correction: a warp reading ONE contiguous 16-entry 32-bit shared
  table has no intrawarp bank conflicts; duplicate indices broadcast. The old
  replicated-table proof showed sufficiency, not necessity. Compact 64-byte
  shared LUT is a valid direct competitor to register LUT.
- Compiled 74 SM60 kernels, all spill-free. Row reuse at 32/64/128 rows per
  warp increases register pressure and underfills the grid without split-K.
  Split-K prototype and final reduction compiled, not hardware-validated.
- A8 CPU group equations/layouts are exact; separate FP32-LUT construction
  changes rounding versus direct FMA on finite witnesses. Not a quality gate.
- Next: jointly benchmark byte-layout integer, endpoint-SAD, register-LUT and
  compact shared-LUT with all preparation/reduction costs. Do not infer a
  speedup from the static instruction counts or an unfavorable layout control.

## Single-P100 hardware gate — 2026-09-08

- Reserved physical GPU1 for the entire series, including inter-run/build and
  artifact-check gaps. GPU0's user editor context was left untouched. Only
  documented compiler-generated SM60 kernels executed; no production changes.
- [GPU report](layout-lut-20260908/GPU-RESULTS.md): all 60 A8 row kernels passed
  720 pipeline cases; six SAD group variants passed 65536 vectors each;
  memcheck/racecheck/synccheck smoke checks were clean.
- 23/24 frozen benchmark workers completed. Last W4 square/batch-one repetition
  was denied by execution policy due to the newly pasted AGENTS sub-Q8 ban,
  conflicting with the earlier user lift and permissive on-disk global file.
  No bypass/retry. That cell has two repetitions; all other cells have three.
- W2 register-LUT beats improved byte-layout integer by 2.7–3.9% latency on
  batch-one K5120 shapes. Endpoint-SAD wins by 8.6% on K17408 and 15.6% on
  square/batch-four. W4 LUT does not convincingly beat integer. Shared LUT
  variants lose. Original one-row-per-warp endpoint-SAD does not beat masked SAD.
- Controls use identical weights/activations/non-power-of-two scales. Sum prep,
  LUT construction, group corrections and split-K reduction are charged. Most
  old-to-new gain is mapping/layout, not proof of a better arithmetic primitive.
- CPU packing, A8 quantization and transfers are excluded. Synthetic signed
  G32 uses an extra FP32 scale (1bpw); not a native GGUF or model speedup result.
  CPU references reproduce each FP32 reduction order; production byte identity
  or perplexity/KLD gate remains pending. No inference integration.
- Resolve current sub-Q8 authorization conflict before any more low-bit GPU
  evaluation; then finish the missing repetition and test one real-format W2
  kernel with integer/LUT/SAD controls, not an extended shared-LUT campaign.
- Final artifacts and health checks completed; GPU1 reservation released at
  11:57 UTC. Final GPU1: 5 MiB, 0% utilization, no compute process, ECC0.
  No resets; no remaining test worker. Future tests need a new reservation.

## Explicit sub-Q8 authorization and GPU1 completion - 2026-09-08

- User explicitly reaffirmed that the sub-Q8 restriction no longer exists and
  requested continuing on GPU1. Global Codex AGENTS.md now states this
  supersedes the old pasted ban and historical prohibitions. Accuracy gates
  remain unchanged; Claude's memory/plans were not edited.
- New GPU1 reservation lowbit-20260908-1200-gpu1 acquired after prior release,
  coordination review and idle/ECC/process checks. Only the supervisor's fixed
  reservation token changed; all its safety gates and worker/cubin hashes
  stayed intact. No other GPU/user process was touched.
- Missing W4 repetition passed in layout-1788869038900603625. Full frozen
  series now has 24/24 successful workers, three per cell, 312 configuration
  validations and 2496 retained samples. Aggregate regeneration passed.
- W4 square/batch-one medians: integer 44.496 us, SAD 48.832 us, register-LUT
  45.696 us. Conclusion unchanged: integer remains preferable on this shape;
  W2's shape-dependent register-LUT/SAD opportunity remains the next priority.
- Updated report: layout-lut-20260908/GPU-RESULTS.md. No model-quality or
  production integration claim.
- Continuation reservation held through artifact/health verification, then
  released. Final GPU1 check 12:05:39 UTC: 5 MiB, 0%, no compute process,
  ECC0. No remaining worker or reset; other GPU processes untouched.

## W4A16 precision-preserving kernel result - 2026-09-08

- User requested W4A16 gains without compromising activations. Isolated study
  in w4a16-20260908; no production changes or sub-Q8 prohibition reinstated.
- Real Q4_0 semantics: 18-byte G32 blocks, lossless coalesced nibble/plane
  repacks, unchanged FP16 activations, exact widening and FP32 arithmetic.
  No A8 quantizer, HFMA2/HMUL2 or half accumulation. Closed half2 schemes
  were not retried. Compiler HADD2.F32 is verified exact half conversion.
- Main result: direct R4/S32 versus matched tiled R2/S32 takes 49.019 vs
  66.069 us (square N1), 154.608 vs 203.040 (tall), 148.112 vs 205.413
  (long K), 134.891 vs 189.403 (square N4): 23.9-28.8% lower latency.
  All main-result outputs are bit-identical to the study's FP32 reference.
  R4 reuses each activation across four rows; R2 already reuses two rows.
- Register-LUT does not beat the best direct path and changes summation
  order. One-time exact FP32 preparation, charged inside every timed
  pipeline and supplied to matching controls, does not improve the winner.
  Faster S8/S16 direct reductions also change output bits; not promoted.
- Final build: 30 SM60 entries, no spill/stack/local-memory accesses. CPU
  exhaustive finite-half/product and lossless-layout tests pass, including
  UBSAN. GPU: all 65536 conversion patterns, 1560 correctness pipelines,
  zero-error memcheck/synccheck smoke; exact integer/rational error oracle.
- Four bounded sweeps frozen into 17 configs before confirmation. All
  12 fresh benchmark workers pass: 204 checks, 1632 retained event samples.
  Source/worker/cubin/config hashes verified; raw logs and aggregate CSV/JSON
  retained. Timing includes group scaling, reduction and any GPU preparation.
- CPU repacking/checks and transfers excluded, with costs logged. Real block
  format but synthetic data; no real-layer, stock dispatch, KLD/ppl, logits
  or end-to-end speedup demonstrated. Group-scale application/order differs
  potentially from production; study bit identity is NOT stock bit identity.
- Next: direct R4 reuse versus actual W4A16 production path on identical
  real layer inputs; apply production byte-identity or ppl +/-0.003 gate.
  Do not extend SAD/half2 work or integrate LUT based on these measurements.
- [Report](w4a16-20260908/RESULTS.md). GPU1 reservation
  w4a16-20260908-1227-gpu1 held through all tests/gaps/final checks, then
  released after 12:51:34 UTC idle/health check: 5 MiB, 0%, no GPU1 process,
  ECC0. Other session released GPUs 0/2/3 before frozen timing. GPU0 editor
  untouched; no resets, commits, pushes or PRs.

## W4A16 round 2: confirmed on three P100s - 2026-09-08

- User made all GPUs available and requested further gains. Reserved GPUs
  1/2/3 for the whole series under w4a16-r2-20260908-1258-three; GPU0's
  desktop editor excluded/untouched. One GPU worker at a time, no model job.
- [Report](w4a16-r2-20260908/RESULTS.md). Sealed prior R4/S32 cubin is loaded
  as a separate module and rerun as the live control on each device.
- New N1 winner r2_14_r2: coalesced one-half-per-lane activation load, exact
  FP32 widening, SHFL broadcast to the warp, two rows per lane, S32. New N4
  winner r2_36_r2 additionally shares weight decoding/scales across four
  activation vectors. Same compressed layout as prior winner; no extra
  repack, activation buffer, metadata, half multiplication or half accumulation.
- GPU1 old -> new us: square N1 49.579 -> 40.949 (-17.4%); tall
  154.405 -> 116.389 (-24.6%); long K 147.173 -> 114.523 (-22.2%);
  square N4 136.411 -> 67.477 (-50.5%, 2.02x). GPUs2/3 confirm about
  17%, 24-25%, 22%, 50% respectively. All individual worker pairs improve;
  no absolute timings pooled across GPUs. These are incremental kernel gains.
- Signed BFE already existed in the old SASS: no new opcode discovery.
  Unsigned tile indexing/cache/structure alone regresses; wider loads help,
  but warp sharing wins. Alternate vector weight layout not needed by winner.
  More rows/CTA tuning did not displace it. Four-vector kernel regresses N1;
  do not use it for batch one. N2/N3 performance dispatch remains unmeasured.
- 77 new SM60 kernels, zero spills/stack/local-memory accesses. CPU finite-half,
  exact-product, layout, vector-alignment and bounded-index tests pass normally
  and UBSAN. Full GPU suites pass 1027 cases/device, 3081 across GPUs1/2/3,
  including exhaustive finite-input coverage. Memcheck/synccheck zero errors.
- 4 exploratory sweeps; 12 configs frozen before 36 confirmation workers
  (3 repeats x 4 shapes x 3 GPUs). All 432 frozen checks pass, 3456 retained
  event samples. Full artifact audit: 45 successful workers, 3987 bit-identical
  pipeline checks; sealed baseline, new binaries, source and config hashes pass.
- FP16 inputs, FP32 group FMA/scaling/S32 tree remain unchanged. Bit identity
  is to the study reference, not stock. Synthetic real-Q4_0-format data only;
  no real layer/model, stock dispatch, ppl/KLD, logits or e2e speedup claim.
  Next priority: test actual production W4A16 with identical real layer inputs
  and production byte-identity or ppl +/-0.003 gate before integration.
- Held GPUs1/2/3 through tests/builds/gaps/final checks, then released after
  13:27:11 UTC health check: all 5 MiB, 0%, no compute process, ECC0.
  No resets, user-log truncation, production edits, commits, pushes or PRs.
