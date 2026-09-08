# Target of record: 1.5x T64 for 2k+ prompt-relevant GEMMs

User clarification supersedes selection by the64-token equal-expert benchmark.
Target2048/4096/8192-token prompt workloads, including activation preparation,
keeping FP32 accumulation. Status: NOT achieved. GPU0/GPU1 are released after
the preceding bounded round; this shape-attribution phase is CPU-only.

## Exact projection dimensions

Use M=routed tokens in a service job, N=output channels, K=reduction channels:

| Projection | N | K | Relative useful work per routed token |
| --- | ---: | ---: | ---: |
| Gate | 512 | 2048 | 1 |
| Up | 512 | 2048 | 1 |
| Down | 2048 | 512 | 1 |

These are source-derived from AW_EMBD=2048, AW_EXPERT_FF=512 in the accepted
Q4 service. Separate dense/shared/attention operators must retain their real
dispatch and be attributed separately; do not substitute the other dense model's
N4352/K5120 shapes. Gate/up fusion is two projections, not free doubling of work.

Top8 across256experts implies unpartitioned MEAN M=64/128/256 at2k/4k/8k.
That is conservation of routes, NOT measured per-expert M, a service partition,
or a justification for equal16-expert batches. `panel2048` denotes output N
panel width; it does not cap the prompt/token dimension at2048.

## Historical routing makes the priority clear

shape_inventory.py reads and hashes the historical4096-token code/wiki routing
logs, checks256experts and8routes/token, and counts gate records only (down
duplicates routing). The source logger omits trailing zero experts; these are
restored, not treated as missing data. Output: shape-inventory.json.

For code, experts with M>128 account for73.57% of useful expert GEMM work;
for wiki,80.72%. Experts M<=64 account for only10.81%/8.41%, respectively.
These are useful FLOP shares, NOT GPU-time shares. Historical Q8 fallback
routing is not the current Q4 service and cannot be silently promoted to it.
The4096 aggregate counts lack token order: do NOT halve/double them and claim
real2048/8192 prompt distributions or owner-local partitions.

Priority: high-work M128–512 plus the hot tail M1024–4096, both projections;
retain M16/M32/M64 tails in a mixed grouped workload rather than optimizing
them as the entire workload. M32 compute tiles may still be useful inside a
larger-M GEMM; problem M and tile M are different quantities.

## Required next benchmark changes

1. Extract/replay the CURRENT compressed-Q4 T64 service controls. The recorded
   profile is cohortrail/interleave, panel2048, M64 split2. Source
   aw_live_launch_projection dispatches M64 to
   aw_q8_service_m64_n128_halfpipe_sync, plus M32 and M16 cohort/tail paths.
   The old primitive uses aw_q8_service_m64 and is not the sole acceptance bar.
   Preserve actual Q4_0/Q4_1 operand rounding and reduction semantics.
2. Obtain current Q4 routing/dispatch records on code and prose at2k/4k/8k,
   matching actual batch/ubatch, owner/expert placement and warm/cold cache
   policy. Use non-dispatch-changing capture in an isolated experimental build.
   GGML_CUDA_MOE_HIST makes ggml_cuda_affinity_wave_t64_enabled return false;
   it is NOT a valid unchanged-service profiling switch. Historical logs are
   only provisional lead prioritization. Capture overhead must be excluded from
   subsequent replay/timing or measured against an uninstrumented control.
3. Replay complete ragged expert groups and actual M64/M32/M16 tails, with
   all64 primary experts/device where applicable. Charge gather/conversion,
   layout preparation, recurring compressed-cache misses, and required output
   reductions consistently. Current worker maxM512/equalM needs redesign for
   hot experts and ragged batches; merely loosening the CLI guard is insufficient.
4. Test larger reuse tiles/persistent work assignment for the large-M regime,
   and a separate down-projection schedule. Do not repeat the already-losing
   identical vector-prep, M16,4x8 or grid-only sweeps. No half-accumulation gains
   count. Evaluate fusion only with the real activation and quantization paths.

## Acceptance and reporting

For each prompt class, use the ratio of total matched baseline pipeline time
to total candidate pipeline time across the captured workload; never average
per-shape speedup ratios. Require>=1.5 for each2k/4k/8k suite and report code/prose
separately. Report dispatch coverage and any fallback regressions. Test fresh
workers onGPU0/GPU1 with fixed source/binary/input provenance, interleaved
controls, correctness, sanitizers and repeated samples.

Then check actual prompt processing on the same Q4 model/binary profile and
PPL within±0.003 using the project's KLD-pair gate (or byte identity). A1.5x
GEMM pipeline does not imply1.5x whole-model prefill; both must be reported.
No Q4/Q8 or NVFP4 quality equivalence follows from a same-Q4 gate.

Current limitations: no new full-model trace or2k/4k/8k model timing in this
round. The required four-GPU service cannot be run while another session's
server occupiesGPU2/3; authority to killGPU0 clients does not authorize touching
that server. Two-GPU isolated replay work can proceed after renewed coordination.
After our release, the coordination log records GPU1/global lock held by
`prefill-fp16-ratio-1788898174313271440-gpu1`; do not reacquire over that claim.
