# FP32 W4A16 delivery round — 2026-09-08

Outcome: **1.54–1.55x gate/up at64 expert tokens, 1.35–1.36x down**,
including activation conversion. The128-token gate/up result is only1.27x.
This is NOT a general1.5x result, full-model speedup, or comparison against the
newer compressed-Q4 T64 service. User subsequently clarified the target is
GEMMs relevant to2k+ prompts; that requires workload-weighted evaluation.

GPU0/GPU1 were reserved under the shared lock, sequential fresh workers,
env-i/taskset0–11, two-second warmup, rotated/interleaved measurements. GPU2/3
were untouched. Final controller release is
gpu-results/release-1788897831413629951.json: normal exit, both reserved devices
idle/ECC0, GPU0 existing editor retained. No kills, faults, resets, commits,
pushes, production changes, or Claude-memory edits.

## Matched primary measurements

Each cell is a median within its own worker; ratio uses the fastest of all
three included T64 controls, always aw_t64_f32 for these pipeline results.
Nine repetitions,12 iterations/sample. Do not pool devices or cherry-pick a
slower control. Gate/up N512 K2048; down N2048 K512;16 experts,64 tokens/expert.

| Raw result directory in gpu-results/ | T64 pipeline us | Fused pipeline us | Ratio |
| --- | ---: | ---: | ---: |
| gpu1-fused-gateup | 534.560 | 345.915 | 1.545x |
| gpu0-fused-gateup | 535.099 | 345.749 | 1.548x |
| gpu1-fused-gateup-seed2 | 534.835 | 346.405 | 1.544x |
| gpu1-fused-down | 466.283 | 343.712 | 1.357x |
| gpu0-fused-down | 465.139 | 343.691 | 1.353x |

GPU1 gate/up candidate range345.768–346.448us, GPU0 range345.200–346.808us.
Down control is noisier: GPU0 control range457.779–471.024us; all samples retained.
The frozen T64 launch remains112CTAs; candidate grid280. This does not establish
the best possible retuned T64 baseline. Every worker retains original FP32 T64,
A16 T64, Q4-storage same-schedule T64, and frozen sequential FP32 winner.

## What changed

Winner `delivery_fused_u32`: M32N64,128threads,4x4 outputs/thread, K32stage,
full32 unroll,88registers,12KiB shared, no spills. Weights remain compressed
18B/32 values with a lossless word-major byte permutation. FP32 raw activations
are loaded using float4, rounded FP16-RN, widened back toFP32 in shared staging.
This removes the separate activation-preparation launch/traffic. Its conversion
cost remains inside GEMM in BOTH prep=0 and prep=1, not omitted from timing.

All dot products and weight scaling stay FP32, sequentialK; SASS contains FFMA,
explicit F2F.F16.F32 conversion and widening, no HFMA2/HMUL2. This is independent
of the separately parked FP16-accumulation lever. No unknown opcode study here.

## Shape sensitivity

GPU1, same gate/up N/K/experts/grid, seven repetitions:

| Expert token count | T64 pipeline us | Fused pipeline us | Ratio |
| ---: | ---: | ---: | ---: |
| 33 | 517.693 | 344.699 | 1.502x |
| 65 | 831.779 | 507.488 | 1.639x |
| 128 | 858.371 | 677.904 | 1.266x |

Tags gpu1-fused-t33/t65/t128. The65-token benefit includes less tile padding;
it is not evidence of higher asymptotic throughput. Prompt length is NOT expert
token count. These equal-size16-expert batches do not reproduce routing skew,
service partitioning, owner reduction, or a complete2k+ prompt. In the current
profile `panel2048` specifies output-channel panel width, NOT prompt length.

## Leads tested and rejected

- V1 tiled-half and tiled-widened-half activation layouts helped kernel-only,
  but the transpose erased their advantage. Word-major Q4 was useful.
- V2 4x8 register geometry and smaller unroll lost.
- V3 M32 reduced resource use versus M64 and won; V4 M16 lost badly.
- V4 redundant activation bounds checks raised register/scheduling cost;
  V5 restored the compile-time M32 guard. Preserve this distinction.
- Vector preparation saved only about1–2us. V5 M32U32 achieved1.46x gate/up
  including preparation; fusion closed the remaining gate/up gap.
- Fused U16 down347.003us versus U32~343.7us. Down grid2/3/4/6/7/8 medians
  respectively422.461/378.720/350.075/423.957/386.531/361.544us; grid5 remains
  best among the tested settings. Do not repeat this blind sweep unchanged.

## Correctness and remaining quality gate

Five offline groups pass (format/address models, finite-half roundtrip,
inventory/hash/safety checks). Each new GPU output word matches the frozen
sequential FP32 winner before timing. Independent CPU FMA checks cover every
output on small fixtures and256 samples per configuration on large fixtures.
The frozen winner itself differs from T64's split-K reduction order.

- gpu1-v6-smoke: all8448 outputs, memcheck0errors.
- gpu1-fused-raw-witness:135168 non-half-representable raw activations, including
  signed rounding ties and half-subnormal boundaries; all8448 outputs match
  half-rounded CPU/frozen reference, memcheck0errors. Controls receive already
  rounded inputs here, so this run is accuracy-only, never a speed claim.
- gpu1-fused-synccheck/racecheck:65tokens,128outputs,K96,64experts, persistent
  CTA reuse and tails; zero synchronization errors or race hazards.
- gpu1-fused-pattern1/2/3: outliers, cancellation and range witness atK2048;
  all8448 outputs checked. All finite; same frozen-winner bits.

Synthetic scale values make dequantized weights half-representable. Real Q4
scales/offsets, Q4_1 handling, weight-operand rounding and production reduction
must be matched before deployment. No PPL/KLD evaluation this round; it does
not pass the model accuracy gate merely because synthetic error is small.

## Reproduction and scope

Read RUNBOOK.md; build.py generates worker.cu from hash-verified frozen sources.
Each GPU tag contains prelaunch.json (commands/hashes/health), stdout/stderr,
and result.json. Telemetry covers both reserved devices. snapshots/v1..v6
preserve earlier source/build artifacts; current build adds raw-witness tests.
analyze.py excludes sanitizer/raw-witness timings; audit.py records static
whole-function counts, not dynamic instruction counts or unavailable counters.

Events exclude allocations/HTD/offline weight repacking, routing, gate/up
combination, SwiGLU, owner/communication and model attention. Diagnostic scratch
buffers coexist even when unused by the fused path. No deployment memory or
cache-miss traffic claim. Do not multiply these gains by the separate packed-Q4
service gain: optimizations overlap.
