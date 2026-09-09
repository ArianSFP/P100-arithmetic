# W8A16 AffinityWave optimization — 2026-09-09

Latest follow-up: see [TARGET-1P9.md](TARGET-1P9.md). The raised 1.9x target remains unmet; newer confirmed F32/FP16-storage pipelines and model mode 6 are recorded there.

The revised **1.7x FP16-accumulation target is reached on the M256 kernel workload**, including dispatch from the original M64 tile list. This is **not yet an accuracy-qualified or full-model result**. The unchanged original uses FP32 accumulation; selected candidates use two FP16 rails with final FP32 addition/output. Required model PPL delta remains within +/-0.003 with paired KLD evidence.

## Confirmed result

64 experts, 256 tokens per expert; gate/up N512 K2048, down N2048 K512. Three fresh workers per projection, comparing against the faster of both unchanged original input branches within each worker.

| Candidate | Gate/up | Down |
|---|---:|---:|
| M128 paired-K, double-buffered B, dual FP16 rails | 1.7293x | 1.7664x |
| Same arithmetic, existing M64 tile list plus leftover dispatch, both launches timed | **1.7080x** | **1.7482x** |

Existing-list dispatch worker ranges: gate/up 1.70735–1.70885x; down 1.74573–1.74850x. Median times: original 4627.84 us / candidate 2709.38 us gate/up; original 4529.68 us / candidate 2592.08 us down. Speedups are medians of worker ratios, so ratios of separately aggregated times can differ slightly.

Selected source: `virtual-pairs-dual/packed.inc`, `pair.inc`, `fallback.inc`. The pair kernel consumes adjacent eligible full M64 tiles of the same expert as an M128 tile; remaining tiles use the M64 fallback. Pair indexing is virtual and requires no host descriptor conversion. Both kernel launches are inside timing. Each candidate output word matches the earlier dual-rail implementation.

The optimized M128 kernel reuses weights across twice as many rows, stores adjacent K weights together in half2 shared words, double-buffers B, and uses one barrier per K stage. Two independent FP16 rails improve instruction scheduling and reduce rounding error relative to one full-length rail. The full mode compiles to 125 registers, 32 KiB shared memory, no spills. This is W8A16 with FP16 accumulation, not activation INT8/INT4 quantization.

## Scope and accuracy

Synthetic relative L2 versus the original FP32 reduction is 0.00465019 gate/up and 0.00234354 down. Outputs are not byte-identical to the original. Synthetic checks establish implementation semantics; they do not establish the model PPL requirement.

An additional single-worker shape screen of the direct M128 kernel gives:

| Tokens/expert | Gate/up | Down |
|---|---:|---:|
| 128 | 1.6887x | 1.7013x |
| 512 | 1.7393x | 1.7716x |

These are screens, not three-worker confirmations. There is no claim of 1.7x across all expert sizes, M32/M16 services, or the complete model. The ~2.9k prefill baseline cannot be multiplied by these kernel ratios.

Validation completed:

- Full-output finiteness and error accounting; 256 complete CPU dot-product checks per arm. Half FMA oracle rounds directly to half, avoiding intermediate FP32 double rounding.
- Full-output equality to the preceding dual-rail implementation, including 65/129-row mixed leftovers.
- Memcheck on direct and existing-list candidates; racecheck on the double-buffered dual candidate (three K stages, persistent reuse, ragged rows).
- Separate `virtual-dual-semantics/` memchecks for tagged FP16, BF16 and F32 input, reversed gathered rows, and non-null device tile count at 129 rows. All pass the CPU oracle.
- Hash-verified builds and workers, GPU telemetry, zero worker failures in selected candidate runs. Raw `.out`, `.err`, `.meta.json` and `summary.json` remain in each variant directory.

The isolated `model/` adapter redirects eight original M64 launch sites. Modes 0–4 retain original and earlier arithmetic controls; mode 5 is the selected existing-list paired implementation. M32/M16 stay original. The paired validator interleaves warmed original controls and checks dispatch coverage, control stability, PPL and saves logits for paired KLD analysis. The isolated CUDA library and updated paired validator both build successfully. Model runs have **not occurred**.

The original Q8 GGUF was deleted under an earlier user instruction. Restoration requires the pinned 37,801,097,504-byte file documented in `/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0-REDOWNLOAD.md`; only about 11 GB is free on the persistent volume. A user question requesting another copy or a location with at least 45 GB free is pending. No Q4 substitution, deletion of unrelated files or unqualified promotion was performed.

## Benchmark contract

CUDA 12.8, g++13, sm_60, O3, no fast-math in synthetic workers; both arms compiled together. Single idle/unreserved P100, env-i CUDA stack and CPU affinity 0–11, device locks and coordination logs, ECC/temperature/disk/watchdog guards. All current selected workers used GPU2. Two seconds warmup; seven rotated rounds, six launches/event, predefined round zero excluded. Static weight packing, upload and descriptor allocation are outside timing equally. The inherited `META quantization_timed=1` label is legacy: none of the measured W8 optimization arms performs activation quantization. A16/A32 mean actual FP16 versus F32-backed A16-valued input controls; A64/A128/A256/A512 are variant IDs, not bit widths.

`model/` intentionally uses the live build's production compiler flags, including fast-math, and therefore still requires model validation. No production library was modified, no commits/pushes/PRs made.

## Closed experiments in this study

- Exact FP32 M32 reduction batching: 0.991x gate/up, 1.006x down. Not selected.
- Initial M64 dual-half pipeline: 1.468–1.521x across eight confirmed shapes.
- M64 single-rail short partials: about 1.43–1.46x at M256; full about 1.51–1.53x.
- M128 single-rail paired-K: 1.680x / 1.727x; two rails improve both speed and synthetic error.
- M256 tiles, N256/512-thread tiles, reduced unrolling, direct input staging, grid sweeps, register caps, fused weight decoding, A shared-memory swizzle and constant-dimension specialization did not beat the selected implementation. FP32 swizzle also failed to improve the original control.
- Charged Q8-to-F16 + cuBLAS + output widening: best screened FP16 algorithm about 1.352x / 1.382x, below custom kernel. Default vendor algorithm was slower still.
- `m128-typed-input/` compiled but was not GPU screened; it is not a measured result.

Next required gate: restore the exact Q8 model, run warmed paired quality/KLD validation of mode 5, then measure full-model prefill on the same baseline conditions. If quality fails, the 1.7x candidate remains rejected regardless of these timings.
