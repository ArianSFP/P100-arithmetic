# W4A16 independent acceptance audit and GPU2/3 replay plan

2026-09-08. W4A4 is paused. This directory owns new independent verification;
historical W4A4 verification files remain untouched. No GPU/reservation/compiler
activity by reviewer. Parent coordinates held GPU2; Dalton's new work is W4A16.
GPU0/1 excluded. No production edits, model launch, commit or push proposed.

## Evidence and unmet gates

Reference source `studies/qwen35-q4-t64-20260908/affinity-wave.cu`, SHA256
fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47:

| Gate | Evidence today | Still needed |
| --- | --- | --- |
| Current dispatch | aw_live_launch_projection line10192, cohortrail branch10211 | Replay M64, M32 and M16 actual tile/cohort queues; not every tail through M64 |
| M16 packing | aw_launch_cohortrail_m16 line10103: P4/P3/P2 cohorts and rail-pair singletons | Capture max_cohort, queue-ready/singletons flags, launch settings and counts; charge aw_build_m16_cohorts when current service does |
| Activation semantics | Both BF16_INPUT true/false branches at10233/10264 | Capture actual source dtype, gather indices/rows, activation bytes, conversion placement, post-SwiGLU down inputs |
| Weight semantics | Q4 helpers778–813 round Q4_0/Q4_1 dequant to half | Actual model scales/codes, Q4_1 offsets, Q8 fallback coverage; packed decoder test alone is not model coverage |
| Recurring preparation | T64 RESULTS.md43:12 compressed cache slots,448MiB/device, misses repack in timing | Native-to-layout repack/cache access sequence; word-major preparation cannot remain a free shadow |
| Routes | Historical shape-inventory has4k Q8/fallback counts only | Current Q4 token-level2k/4k/8k code/prose owner partitions and dispatch; no histogram scaling |
| Speed | COMPACT-RESULTS.md:1.406x gate/up,1.331x down at equalM256/e64 | >=1.5x complete matched routed workload per prompt/domain suite with fresh independent workers |
| Correctness | Parent reports exhaustive packed decode and fixture split identity/sanitizers; compact report has fixture identity | Each new tile own sanitizer/rounding/real-scale checks; whole replay bit identity OR paired model PPL within±.003 |

`ggml_cuda_affinity_wave_t64_enabled` line15166 explicitly requires both
GGML_CUDA_MOE_HIST and GGML_CUDA_MOE_DEBUG_STATS absent. Neither is a valid
unchanged-T64 capture switch. Source source-dtype/owner-reduction semantics
must not be flattened to an all-A32/all-Q4_0 fixture for acceptance.

Prior whole-model reference RESULTS.md39–43 reports131072 mixed-Q4/tail kernel
comparisons and512 full-vocabulary rows after8128 tokens byte-identical to the
accepted native Q4 port, whose paired PPL delta+.001801171 passed. New kernels
do not inherit that gate automatically. Preserving reference weights, FP32 rail
ordering and owner epilogues offers a path to identity rather than a new
accuracy compromise; no NVFP4 or full-model throughput claim follows.

## In-scope replay ladder

1. CPU inventory existing artifacts, not a new model run. Find actual token-level
   Q4 service traces/activation captures if available and verify source/model/
   input hashes. Historical4k histograms may prioritize synthetic tests only.
   If real traces are unavailable, record this gate as unmet; request separate
   authorization for an isolated capture build/run, never silently use GPU1 or
   assume a two-GPU run reproduces the four-owner service. No production patch.
2. Build an isolated replay harness when parent grants a compiler slot. Extract
   unchanged current M64/M32/M16/cohort helpers and dispatch policy. Load every
   captured owner shard sequentially onto GPU2, preserving original logical
   owner ID and expert set rather than repartitioning experts to two owners.
   This is local-compute replay, NOT measured4-GPU transport/arrival timing.
3. Inputs must include projection, layer/microbatch, logical owner, expert IDs,
   input-row indirection, M and tile/cohort lists, source activation dtype and
   bytes, Q4_0/Q4_1/Q8 tensor IDs/bytes/scales, cache slot hit/miss/eviction order,
   plus output-reduction metadata and hashes. Retain M0 and every tail, hot
   M1024–4096 jobs, realK512/2048,N512/2048. Gate/up are separate unless the
   captured schedule truly shares preparation; down needs its own inputs.
4. For each candidate, use unchanged controls on the SAME physical GPU and
   inputs. Before any timing, poison output, check every result versus control,
   and independent CPU oracle samples including adversarial scales/half ties.
   New256-thread tile gets its own memcheck/racecheck/synccheck. No inherited
   correctness from compact/wide despite shared decoding.
5. Event interval covers queue/gather/conversion, required native/cache/layout
   repacks, all kernels/tails/fallbacks and required output reduction. Report
   memory footprint and cold/steady state separately. Exclude only truly common
   initialization outside both paths; recurring repack is never amortized away.
   Charge unmodified fallback in both arms, so unsupported formats dilute gain.
6. Freeze candidate selection after screen. Parent serializes timing and
   compiler work under shared coordination. Run>=3 fresh workers per suite with
   rotated controls/order; keep all samples and health/provenance. Then repeat
   held-out confirmation on GPU3 when Dalton/parent grants its slot. Do not
   compare GPU2 candidate times with GPU3 controls or benchmark simultaneously
   merely because devices differ.
7. Each prompt/domain score is SUM matched control pipeline time divided by SUM
   candidate pipeline time over the same captured jobs, not an arithmetic mean
   of speedups or FLOP-weighted ratio. Report every logical-owner/projection
   subtotal, coverage/fallback regressions, and warm/cold policy. Require>=1.5
   for each2k/4k/8k code/prose suite; one fast M256 projection is not success.
8. This proves only the stated local GEMM-pipeline scope. Actual full-model
   prefill timing/quality on the matched deployment remains separate work,
   requiring its own authorized hardware/capture/eval. Replaying four owner
   shards serially on one P100 cannot certify distributed critical-path gain.

## Current independent tile-review status

Dalton's64x64/256-thread/4x4-output-per-thread split-FP32 source appeared during
audit. Actual source reviewed independently: see TILE256-REVIEW.md and three
passing tests in test_tile256.py. Kernel geometry is not dispatch/build proof;
build wiring was not yet present at review cutoff.
