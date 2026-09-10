# W4A4: small internal gains, no win over current W4A16

2026-09-08, physical GPU3, Tesla P100-PCIE-16GB / SM60, CUDA 12.8 / GCC 14.
**The tested W4A4 kernels do not provide a stronger replacement for current
W4A16.** Bit-plane/popcount gives modest gains over matched W4A4 FP32, but
activation quantization plus the complete pipeline remains slower than the
latest W4A16 candidates. Exact packed-half arithmetic works and is slower.

## Confirmed results

Median of three fresh-worker medians; controls and frozen candidates interleaved
on the same GPU and identical source data. Microseconds, including activation
quantization for W4A4, scaling and final reduction. M = output rows, K = reduction
length, N = vectors. R2/R3 = W4A16 study round, not row reuse.

| M x K | N | A16 R2 | A16 R3 | Best selected A4 | A4 slower than R3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5120 x 5120 | 1 | 39.025 | 36.767 | 39.733 | 8.07% |
| 17408 x 5120 | 1 | 114.984 | 107.423 | 111.141 | 3.46% |
| 5120 x 17408 | 1 | 111.936 | 106.731 | 107.161 | 0.40% |
| 5120 x 5120 | 4 | 65.013 | 63.616 | 69.255 | 8.86% |

Every frozen paired worker favors R3: slowdown ranges are 6.86–8.33%,
3.07–3.92%, 0.12–0.51%, and 8.75–9.31%. These are observed repetitions, not
confidence intervals. Treat the long-K result as essentially tied; no useful
W4A4 advantage is established. The R3 N4 candidate itself remains provisional
in its originating study; A4 also loses to the established R2 N4 control by 6.52%.

A4 does beat older R2 A16 on tall/long-K by 3.34%/4.27%, but the newer A16
kernels consume that advantage without activation quantization. The R3 study
completed during this work; its cubin was copied and added as a live control,
rather than comparing timing numbers recorded on a different GPU.

| Shape / N | Frozen A4 selection | Best retained FP32 A4, us | Selected A4, us | Lower A4 latency | Best retained half2 A4, us |
| --- | --- | ---: | ---: | ---: | ---: |
| Square / 1 | `w4a4_h2_r4_b1` | 40.741 | 39.733 | 2.47% | 54.593 |
| Tall / 1 | `w4a4_h2_r2_b1_w8_f` | 112.049 | 111.141 | 0.81% | 149.048 |
| Long K / 1 | `w4a4_h2_r8_b1` | 111.864 | 107.161 | 4.20% | 153.548 |
| Square / 4 | `w4a4_h0_r4_b4` | 69.255 | 69.255 | 0% | 81.553 |

The tall internal 0.81% difference is small; no broad advantage is inferred.
At N4 ordinary FP32 remains the selected A4 method. Configuration selections
were frozen before the twelve confirmation workers. All raw samples, including
outliers, are preserved in `gpu-results/`; `aggregate.csv/json` are reproducible.

## What was tested

**Exact half2 dots (H1).** For signed INT4 operands, each product has magnitude
at most 64. A G32 scalar prefix is bounded by 2048, and binary16 represents every
integer in [-2048,2048]. The kernel pairs even/odd K positions: 16 products per
half lane, magnitude at most 1024, followed by exact FP32 lane addition. There
is no flush inside the group. Two nibbles decode to half2 by constructing
1024 + offset_binary_code and subtracting 1032, avoiding I2F instructions.

This is a valid new numerical premise relative to historical Q8/arbitrary-A16
half substitutions, but it does not deliver speed. At R2/B1 the compiled dot
uses 32 HFMA2 instead of 64 FFMA and 16 instead of 32 SHFL, with zero I2F;
packed decoding adds HADD2 and bit operations. The complete H1 pipelines lose
by approximately 18–43% to the selected A4 paths. No throughput conclusion
should be drawn from the reduced FMA count alone. The independent concurrent
[prefill study](../prefill-20260908/RESULTS.md) also found near-flat full-cost
half2/integer W4A4 group-dot performance in a different mapping.

**Bit-plane integer dots (H2).** Losslessly repack each G32 weight block into
four 32-bit planes. The quantizer emits four activation planes. With signed
coefficients c = [1,2,4,-8], the exact group dot is

```text
dot = sum(p=0..3, q=0..3) c[p] * c[q] * popcount(W[p] & A[q])
```

This replaces per-element weight/activation conversion with sixteen AND/POPC
terms per output group, one final integer-to-float conversion, and the same
FP32 scale-FMA. No extra weight metadata is needed. R4 works best for square
N1; R8 for long K; shared local reduction helps tall N1. The matching FP32
controls inherit the same row/batch reuse, group striping and reduction ideas.

**Geometry and local reduction.** The bounded search totals 55 A4 dot kernels:
12 initial H0/H1, six initial H2, 21 added H2 row/CTA variants, and 16 H0/H2
local-reduction variants. Separate-reduction variants retain S32; fused variants
place all 32 stripes in one CTA and reproduce the same 16/8/4/2/1 FP32 tree in
shared memory. Rows/lane, batch reuse, CTA warp count and fusion vary. No stripe
count or floating-point arithmetic-order relaxation was used to gain speed.

All instructions are compiler-generated documented operations; there is no
binary patch, hidden arithmetic-lane claim or new hardware dot instruction.
[NVIDIA's mixed-precision description](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/)
distinguishes GP100 packed FP16 from the other Pascal chips' DP4A support.

## Formats and accuracy

Weights use Q4_0's code values and FP16 scales, losslessly repacked: 4.5 bits per
weight including scales. A4 here means signed **integer** codes, not NVFP4.
The production-path test quantizer uses symmetric [-7,7], nearest-even rounding,
G32 FP32 scale maxabs/7 and zero scale for all-zero groups. Its payload is packed
four-bit nibbles or four bit planes; with the FP32 scale this is five effective
bits per activation. Direct packed tests additionally cover activation -8.

Quantization occurs on GPU from the original finite FP16 inputs. The final
pipelines generate only the representation each method needs. Earlier staged
sweeps used a dual-output quantizer; those timings are excluded from final
confirmation. The test worker holds both weight layouts for comparisons, but
each is independently 4.5 bits/weight; this does not establish an in-place
production repacker or a production workspace requirement.

Every A4 candidate is bit-identical to independently computed integer group
dots followed by the specified FP32 scale multiplication, FMA and S32 tree.
The sealed A16 R2 kernel is checked against its own original-input CPU oracle,
and R3 is checked bit-identical to R2 on the same data.

This is exactness **after activation quantization**, not identity to original
A16 inputs. Synthetic output relative-L2 error against A16 is 9.72–10.20%; it is
not perplexity/KLD and cannot certify or quantify model-quality impact. No
real model/layer capture, stock dispatch comparison, logits, perplexity/KLD or
tokens/sec run was performed. The production accuracy rule remains byte
identity or perplexity within +/-0.003 using KLD-pair methodology.

## Validation and measurement

- CPU: all 256 INT4 products and all 256 packed decode pairs; 244,524 exact
  bounded update transitions. A scalar G33 counterexample, 2049, demonstrates
  why a general unlimited half accumulator is invalid.
- Final CUDA build: 55 dot kernels, three quantization entry points and one
  reducer. All 59 entries have zero stack frames/spills and no LDL/STL.
- Final full suite: 880 pipelines = 55 variants x 16 datasets. Four numerical
  families use M33/K96/N1, M129/K32/N3, M513/K544/N4; three additional packed
  endpoint datasets include full G32 -8/-8 groups; M16/K65536/N1 covers every
  finite raw FP16 pattern and all sixteen weight codes. Quantizers are checked
  against a CPU reference; all tested pipeline outputs are bit-identical.
- Final memcheck, synccheck and racecheck each passed the complete suite,
  including the exhaustive finite-input dataset: zero errors or hazards.
- **33 successful fresh workers, 7,797 pipeline checks and 10,119,216 explicitly
  counted A4 output comparisons** across stages, sweeps and confirmation.
  The output count excludes separate A16/quantizer checks, which also pass.
- **12 frozen workers / 600 retained event samples**, three workers per shape.
  Each worker warms the selected pipelines and both A16 controls for at least
  200 ms, rotates execution order through nine rounds, and times twelve full
  pipelines per event. Round zero is uniformly excluded from aggregation.
  Shape order rotates between repetitions; all excluded warmup samples remain.
- Times include device quantization, scaling, final reduction and device launch
  gaps. CPU repacking, allocation, CPU reference and host/device transfers are
  excluded consistently. No historical absolute timing is used as a denominator.
- Five staged build manifests and all referenced sources/cubins/workers verify
  by SHA-256. Configuration hashes match the pre-confirmation frozen selection.
  The repository's sealed archive verifier passes all 1,152 archived files.

The source of the initial hypothesis and preparation is retained in
`PREPARATION.md`. Early build artifacts remain under `initial-build/`,
`plane-build/`, `geometry-build/`, and `sweep-build/`; they are not final winners.
`analyze.py` associates each raw worker with its exact staged manifest. The
current scripts are dedicated to this reservation/host and are not a portable
GPU scheduler. Reproduce analysis without GPU access with:

```sh
python3 studies/w4a4-20260908/analyze.py
python3 scripts/verify_archive.py
```

## Decision

Keep W4A16 as the preferred path for these shapes. Bank the exact INT4-half
construction and modest bit-plane A4 improvements as research findings. Do not
trade activation precision for the measured outcomes: they provide no useful
speed advantage over the current A16 implementation. This bounded experiment
does not prove every possible W4A4 algorithm, FP4 format or prefill shape is
closed.

Only GPU3 was used, held through builds, workers, gaps, artifact checks and
final health. GPU0's existing desktop process was untouched. GPU1/GPU2 release
notices each triggered a coordination-hash stop **before launching**, were
reviewed, and their new hashes recorded before continuing. A later CPU-only
R3 publication notice similarly stopped the final health check before querying;
it was reviewed without touching that session's Git index. The initial
sandboxed nvidia-smi failure was an access restriction; the read-only host check
outside the sandbox passed before reservation or CUDA initialization. No GPU
fault, timeout, reset, driver change, production edit, commit or push occurred.
Final health and release are recorded in `final-health.json` and both
coordination logs.
