# W8A16 1.9x follow-up

The user raised the target to 1.9x. **It has not been reached.** Thirty-six new variants were screened; the best complete pipelines were confirmed with three fresh workers per projection. Model accuracy and full-model throughput remain unqualified.

## Confirmed complete pipeline

Same unchanged original AffinityWave M64 control, using the faster original input branch per worker. M256 tokens/expert, 64 experts; gate/up N512 K2048, down N2048 K512. These numbers include GPU tile planning, M128 computation, and M64 leftover computation. All three launches are timed; there is no excluded host M128 descriptor conversion.

| Input storage | Gate/up | Down | Source directory |
|---|---:|---:|---|
| F32, rounded to FP16 inside the kernel | **1.786183x** | **1.842659x** | `compact-plan-f32/` |
| Already stored as FP16 | **1.835868x** | **1.885056x** | `compact-plan-dimensions/` |

F32 median times: 4625.57 -> 2589.64 us gate/up, 4529.27 -> 2458.01 us down. FP16 median times: 4625.09 -> 2519.25 us gate/up, 4530.28 -> 2401.91 us down. Speedups are medians of within-worker ratios. Raw outputs, metadata and `summary.json` are in each source directory.

The serving profile in `bench/coding-serve/run.py` specifies `WIRE=f32`. Consequently the first row is the directly relevant storage case. The second row is not a free serving improvement: changing storage or adding a conversion requires its own cost and model accuracy checks. Both benchmark controls use A16-representable values, stored either in F32 or FP16. Separate semantic tests exercise non-half-representable F32 values and gathered rows.

Compared with the previous confirmed FP16-storage pipeline (1.708x / 1.748x), the new FP16-storage pipeline adds about 7.5% / 7.8% throughput. This is not an overall model-speed claim. M32/M16 services and all other model operators remain outside these ratios.

## What changed

- Interleave the two independent FP16 reduction rails while preserving the order within each rail.
- Decode adjacent signed Q8 values together with byte permutation and half2 arithmetic. The decoder first forms exact half representations of the integer codes and then multiplies by the original half scale. An exhaustive check of all 256 codes and 63,488 finite half-scale encodings (16,252,928 cases) matches the prior float-product-to-half conversion, including signs and overflow results.
- Specialize activation loaders separately for F32 and FP16 storage, eliminating a runtime input-type choice in the inner loop. The F32 loader rounds internally and does not require an external conversion.
- Specialize the two projection dimensions without changing the generic fallback.
- Compact the existing M64 tile list on the GPU into eligible M128 pairs and leftover M64 jobs. Warp ballots and shared counters produce the lists. The compute kernels consume device counts; an empty leftover list terminates immediately.

The first implicit-pair integration lost the direct kernel gain, despite similar arithmetic instruction counts and no spills. A separately timed compact planner recovered it. This is measured code-generation sensitivity, not evidence that launch overhead explains the initial regression.

All selected kernels retain two full-length FP16 accumulation rails and FP32 final addition/output. They are byte-identical to the preceding dual-rail candidate on the checked outputs, **not to the original FP32-accumulating control**. Synthetic relative L2 on the timing fixture remains 0.00465019 gate/up and 0.00234354 down. Required paired model PPL delta remains within +/-0.003 with KLD-pair evidence.

## Evidence and limits

The timing contract is unchanged from `RESULTS.md`: CUDA 12.8/g++13/sm_60, O3 without fast-math for synthetic workers, two-second warmup, seven rotated rounds with predefined round zero excluded, six launches/event, three fresh workers per confirmed projection. Static weights and allocations are excluded equally; every candidate planner/compute launch is included. All new GPU runs were sequential on an idle, unreserved GPU2 with coordination/device locks and watchdogs.

`analyze-1p9.py` checks source/binary hashes, worker status, timing sample counts and full-output comparisons. `target-1p9-screen.json` contains all 36 screened variants; `target-1p9-confirmed.json` contains the confirmed summaries. Mode A512 identifies the full-length accumulation candidate; it is not a bit width.

Selected validation includes complete-output comparisons to the preceding candidate, full-output finiteness checks, 256 sampled complete CPU dots, mixed 129-row leftovers, and racecheck across three K stages and persistent tile reuse. Additional `compact-plan-f32-semantics/` checks use reversed gathered rows and non-half-representable F32 values. Validation status and hashes are recorded in `target-1p9-validation.json` after completion.

Two development failures are retained, not counted as speed results: the first synchronous prototype missed the advancing K offset for its second activation row group; memcheck found no memory error but the CPU oracle rejected its output, and the corrected version passed. The first non-half F32 semantic harness stopped in an inherited, unused activation-quantizer check before reaching the candidate. That unrelated check was removed from the semantic harness; timing workers were not changed. Original failed builds/logs are preserved in `dual-sync-c2/initial/` and `compact-plan-f32-semantics/initial-unused-quantizer/`.

Closed leads include wider M/N tiles, 512-thread blocks, higher occupancy through smaller shared storage, alternate activation layouts, shared-memory swizzles, vector B layout, shorter loop bodies, loop interchange, late/split activation prefetch, different rail switching intervals, hoisted row addressing, full-row specialization, and alternate cache policies. Several increased register pressure or spilled. The best isolated dimension/cache screen reached about 1.845x gate/up and 1.898x down; this does not satisfy the 1.9x threshold and is not a confirmed complete-pipeline result. `dual-512-balanced/` was compiled but not GPU screened after it increased spills.

## Isolated model adapter

`model/` now has mode 6 for the faster F32-storage pipeline. Four dynamic-count launch sites pass their existing buffer capacity to size scratch storage safely. Scratch is reused per host thread, GPU and CUDA stream. The BF16/FP16 template retains the previous generic mode-5 implementation. First allocations are charged to the first call and must be warmed before steady-state comparisons.

`GGML_CUDA_AW_W8_MODE=0..6` or `run-warm-job.py --w8-mode` selects an isolated prefill mode. The quality validator selects and interleaves its own modes, checks original-control stability, requires mode-6 fast-dispatch coverage and enforces the PPL gate. Model mode 6 is built but **not model-tested**. Its production compiler flags include fast-math, so synthetic validation is not a substitute for that gate. The isolated build does not replace the production library.

The original 37,801,097,504-byte Q8 GGUF is still missing, with only about 11 GB free on the persistent volume. The pinned restoration record is `/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0-REDOWNLOAD.md`. Another copy or storage space is needed for the paired PPL/KLD and full-model benchmarks. No Q4 model was silently substituted, no unrelated files deleted, and no production promotion, commit, push or PR performed.

The 1.9x goal remains open. This round establishes an improved, reproducible candidate and records the unsuccessful directions; it does not establish that 1.9x is impossible.
