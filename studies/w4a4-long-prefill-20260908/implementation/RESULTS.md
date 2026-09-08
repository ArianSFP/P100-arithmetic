# W4A4 tiled planes: first long-prefill screen rejected

Status: PAUSED by explicit user instruction. No further W4A4 experimentation.

2026-09-08, physical GPU3 only, P100/SM60, CUDA12.8/GCC14. Current T64
FP32 accumulation retained; all candidate activation quantization and recurring
weight-plane preparation charged. **No2x gain: both tiles are substantially
slower than the fresh, unchanged current compressed-Q4 T64 comparator.**
Do not tune this mapping further without materially new evidence.

## Paired results

M256 for64 equal experts,16384 routed tokens total. Actual expert projection
dimensions, NOT captured2k/4k/8k routing and NOT a full-model benchmark.
Each row is a fresh worker,1s warmup,5 rotated samples,3 complete pipelines
per event. All samples retained; median of5, microseconds. No compiler overlapped
these timings. T64 grid112, candidate grid280; no post-selection retuning.

| Projection | Tile | Current T64 us | W4A4 us | T64/W4A4 |
| --- | --- | ---: | ---: | ---: |
| Gate/up N512 K2048 | M32 N64,4x4/thread | 7020.811 | 10381.493 | 0.6763x |
| Down N2048 K512 | M32 N64,4x4/thread | 6801.195 | 9862.741 | 0.6896x |
| Gate/up N512 K2048 | M16 N64,2x4/thread | 7049.856 | 10499.701 | 0.6714x |
| Down N2048 K512 | M16 N64,2x4/thread | 6824.320 | 9963.168 | 0.6850x |

Raw tags gpu3-{gu,down}-popc{2,4}. Same deterministic finite A16-origin
activation and Q4 fixture for each paired comparison. Symmetric INT4 A4 input,
G32 FP32 scales; no NVFP4 format or quality-equivalence claim. Ordinary fixture
relative-L2 difference versus current A16 T64 is about6.82%; this is synthetic
arithmetic error, NOT PPL/KLD. Exact equations/format limitations are in README.

Both candidates would need roughly another2.9–3.0x reduction in complete latency
to achieve2x this T64 comparator. This is not a near miss worth grid/unroll sweeps.

## Validation and artifacts

- Full current-service source hash frozen at
  fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47.
  Current kernel body is extracted unchanged; CPU tests verify identity.
- Build and independent CPU tests pass. M16/M32 candidates use64/90registers,
  1600/1920bytes shared, zero stack/spills. T64 uses128registers/32KiBshared.
  Matched A4 FP32 attribution control uses255registers, no spills; it is NOT
  a fast competing kernel and is not used as the speedup denominator.
- SASS contains noHFMA2/HMUL2. Integer group dots are exact, followed byFP32
  group scale multiplication andFP32 FMA accumulation. This never uses the
  separately disallowed FP16 accumulation lever.
- gpu3-smoke and gpu3-stress memcheck PASS, zero errors. Both check every
  prepared plane word/scale and all12544 outputs of each of4 configurations
  against independent CPU oracles. Output is NaN-poisoned before EACH check;
  A4 variants also cross-check full output words.
- Stress includes zero/signed-zero groups, quantizer RN-even ties,1888 raw-A32
  half-rounding witnesses, finite-half extrema and subnormals/tiny-only groups,
  arbitrary signed half weight scales, independent seed9117. CPU references
  explicitly round original A32 toA16 before both paths. Hard-scale A4 follows
  its own equation, not identity to T64's per-weight rounding/order.
- Each large configuration checks16384 independent CPU output samples and
  all outputs finite (8,388,608 gate/up;33,554,432 down). Every activation and
  weight preparation group is independently checked. All6workers PASS; no
  sanitizer/driver error, hang, reset, kill, production edit or commit/push.
- Racecheck/synccheck and exhaustive finite-half GPU quantizer validation were
  not run because this bounded screen already rejected performance. No
  numerical promotion or model-quality qualification is claimed.
- gpu-results stores prelaunch source/binary hashes, exact command/shape/seed,
  stdout/stderr, status and pre/post device health. build/manifest.json records
  build-time gpu_tested=false; the subsequent GPU evidence is in these workers.

GPU3/global lock were held across this build and all6workers, then released
normally. Final record release-1788901891411887388.json:5MiB,0%util,ECC0,
no compute clients. Both coordination logs contain final release. ParentGPU2
subsequently acquired its own slot. No GPU0/1/2 used by this worker.

## CPU cost accounting; no further GPU tuning

No separate stage timings were taken; do not invent a measured split between
preparation and GEMM. The following counts are static/source-derived, not
hardware counters or HBM transaction measurements.

Each output/G32 requires16 AND/POPC pairs. The M32 function contains256POPC,
258LOP and240ISCADD instructions: divided over its16 outputs, the core includes
16POPC,16AND and15 shifted integer combinations per output/group, before
integer-to-float conversion, scale multiply,FP32 FMA, loop/address work and
additionalXMAD. The M16 function has128POPC/130LOP/120ISCADD for8 outputs,
the same per-output pattern. Static whole-function counts are not dynamic
profile shares, but they expose why replacing32 scalar FMAs is not a2x
arithmetic instruction-count saving. Shared/register reuse reduced delivery
duplication without eliminating these47 core integer operations per group.

Minimum representation-conversion bytes per complete pipeline:

| Component | Gate/up | Down |
| --- | ---: | ---: |
| Original A32 read | 128MiB | 32MiB |
| A4 planes+FP32 scales written | 20MiB | 5MiB |
| Q4 source weight groups read | 36MiB | 36MiB |
| Weight planes+FP32 scales written | 40MiB | 40MiB |
| Total preparation logical bytes | 224MiB | 113MiB |

These exclude transaction inefficiency, cache effects and GEMM traffic. Both
weight representations remain resident in this diagnostic;40MiB temporary
planes are NOT free compressed payload. Candidate-specific scratch totals
60MiB gate/up,45MiB down. For M32, logical GEMM plane reads before cache reuse
are160MiB activations+320MiB weights per projection. M16 doubles the weight
read duplication to640MiB. Actual HBM traffic may be much lower because of
cache reuse; these are not bandwidth measurements or an attribution proof.

### Next-lead admission test, CPU-only

Do not simply replace INT4 with E2M1 in these four planes. The independent
verification oracle shows doubled E2M1 values require five signed binary
planes; naive expansion raises16 to20 AND/POPC pairs. That makes this arithmetic
count worse, while retaining scale/preparation cost.

A four-activation register-LUT alternative costs8quartets x4weight planes =
32 lookups/output/G32 plus weighted accumulation, table construction and group
scales. A straightforward SHFL+add implementation is not fewer instructions
than32FP32 FMAs, and this lookup method also works with A16 combinations;
activation quantization alone is not its source of reuse. Reject blind LUT
sweeps unless a concrete representation/fusion reduces that count or removes
a separately measured dominating delivery cost. A global sixteen-entry table
for each quartet would also expand G32 activation storage drastically versus
these20-byte groups; table residency/reuse must be explicitly budgeted.

Thus the next useful work is an algorithm with demonstrably fewer lookup/bit
operations, or a measured preparation bottleneck that can genuinely be shared
between real gate/up projections. Do not claim sharing from standalone timings,
and do not share down quantization with pre-SwiGLU activation groups. No further
kernel source, compilation or GPU trial was performed after this rejection.

Reproduce the paired analysis without CUDA:

```sh
python3 studies/w4a4-long-prefill-20260908/implementation/analyze.py gpu3-gu-popc4 gpu3-down-popc4 gpu3-gu-popc2 gpu3-down-popc2
```

Full current M32/M16 tail dispatch, indexed gather/owner reduction, real cache
behavior, Q4_1/Q8fallback and actual code/prose routing remain outside this
screen. PPL±.003/KLD and2k/4k/8k per-suite acceptance remain unmet. This rejects
these two tiled-plane implementations, not every possible W4A4 algorithm.
