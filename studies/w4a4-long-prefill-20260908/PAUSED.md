# W4A4 paused — user instruction, 2026-09-08

No further W4A4 builds, GPU experiments, tuning or quality studies are scheduled.
Both subagents have been reassigned to W4A16. GPU3 was released cleanly after
the W4A4 screen; future GPU2/3 work from this session is W4A16 only.

## Findings

Both signed-INT4 bit-plane tiles lost to unchanged current compressed-Q4 T64:

| Tile | Gate/up speed relative to T64 | Down speed relative to T64 |
|---|---:|---:|
|32×64|0.676×|0.690×|
|16×64|0.671×|0.685×|

These are M256/64-expert screening results, not measured full-prompt speed.
Every timed pipeline includes activation quantization and weight-plane
preparation. Accumulation remains FP32. Two sanitizer fixtures and four paired
projection screens passed; no GPU faults or resets occurred.

Static inspection finds roughly47 core integer instructions per output/group
replacing32 FP32 FMAs, before preparation and other overhead. This is a
source-derived explanation, not a measured stage-time decomposition. Achieving
2× T64 would require roughly another3× reduction in candidate latency.

No model-quality or NVFP4 equivalence was established. This symmetric INT4
activation format is not NVFP4/E2M1. Neither the failed implementation nor the
negative screen proves that every possible W4A4 method must fail.

Detailed measurements, methodology, hashes, limitations and raw artifacts:
[implementation/RESULTS.md](implementation/RESULTS.md).
Independent numerical and source audits remain in verification/.

Resume W4A4 only on a new user instruction. Current priority: W4A16 ≥1.5×
current T64 including activation preparation, retaining FP32 accumulation,
on workloads relevant to2k+ prompt processing and with the existing accuracy gate.
