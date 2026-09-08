# W4A16 versus AffinityWave T64: first matched investigation

User requests working toward twice AffinityWave T64 performance, with quality
competitive with NVFP4. No production modification, commit or push authorized.

## Measurement contract

- Compare live controls and candidates on identical expert GEMM dimensions,
  tokens, routing and input values. Keep kernel-only and preparation-inclusive
  results separate from the full expert service and model tokens/sec.
- Retain original Q8 T64 M64 split-K2 control, and a same-Q4 FP32 control.
  Stored weight precision and arithmetic changes are separate comparisons.
- New axis: paired output-token HFMA2 with group-local short partials, then
  FP32 scale/global accumulation. R2/R4 first; R8 and layout/tile variants as
  needed. Q4_0 G32 only initially; Q4_1 needs its offset correction and the
  installed mixed-format model needs explicit dispatch/fallbacks.
- Validate CPU references, exact-input fixtures, cancellation/outlier cases,
  CUDA memory/synchronization checks, repeated interleaved timing, SASS.
- NVFP4-level model quality is not implied by synthetic errors. A specific
  comparable reference is pending user input; no production promotion without
  real-model validation. Prior strict quality results remain historical facts.

## Coordination

Work isolated here. The other session owns qwen35-q4-t64-20260908 and is
currently running serialized four-GPU tests. CPU/compiler work only until a
GPU can be reserved without overlap. Use the existing four-GPU lock even for
our eventual single-GPU worker, fresh processes, UUID isolation and a persistent
reservation across builds/workers/gaps. No resets or desktop log truncation.

## Prior evidence

Round-2 W4A16's 2.02x was against its previous N4 kernel, not AffinityWave.
The separate approximately 1.6x cuBLAS result used dense Q8-derived inputs.
July's Q8 short-half experiments incurred conversion and reduction overhead;
the new premise is Q4 compressed storage and pairing independent output tokens.
Existing FabWave Q8 partial-sum code/results are reviewed before implementation.

## First hardware progress

GPU2 reserved with the global lock after the other session's packed-server
release. The first two candidate builds pass CPU spot oracles on every tested
configuration; V2 memcheck passes. At the 16-expert, 64-token, N512/K2048
synthetic shape, V2's best R1 gives 1.39x against the live unmodified T64 M64
split-K2 projection control including A32-to-A16 conversion (not repacking or
the complete expert service). R2 gives 1.33x. No quality or model speed claim.

These are exploratory runs. V1 down screening overlapped CPU compilation and
cannot be used as frozen confirmation. V3 tests single-buffer weight staging,
8-step scheduling chunks, row-major compressed payload and launch grid size.
Other session's newer Q4 vector T64 and full model tests remain independent;
do not compare their pending throughput to these projection-only numbers.
