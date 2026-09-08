# Decode: FP16 accumulation does not approach 2x at one token

2026-09-08, physical GPU1, Tesla P100 / SM60, CUDA 12.8, GCC 14. Isolated research; no production changes.

## Result

After four kernel-tuning rounds, **single-token FP16 local accumulation is only 1.04–1.09x as fast as the tuned FP32 controls** across five shapes. Four-token decode ranges from 0.99–1.07x; eight-token batches reach 1.28–1.40x. **2x was not achieved.** These are synthetic Q4_0 matrix-vector/small-batch kernel results, not model tokens/s.

The half candidates are also **not accuracy-qualified**. They change almost every normal-case output, and the large finite-activation stress produces nonfinite half results while FP32 stays finite. No full-model PPL/KLD comparison was run. Keep production FP32; these kernels are experimental artifacts, not accepted replacements.

## Final paired measurements

Three fresh workers per shape, three seeds; median of worker medians. Each worker contains the frozen candidate and live FP32 controls. The denominator is the fastest confirmed FP32 control, including the new exact-order B8 extension; a slower same-mapping control is not used to inflate the primary speedup. All normal-case outputs are finite.

Primary timings include FP32 activation conversion to FP16, compressed weight loads/unpacking, scaling, accumulation, all reduction kernels and FP32 output. They exclude one-time lossless weight repacking, allocation and H2D transfers equally.

| M | K | Tokens N | FP32, us | FP16 group, us | Speedup | FP16 local, us | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4352 | 5120 | 1 | 39.099 | 36.392 | 1.074x | 35.835 | 1.091x |
| 5120 | 4352 | 1 | 34.251 | 33.019 | 1.037x | 32.504 | 1.054x |
| 5120 | 5120 | 1 | 38.819 | 37.277 | 1.041x | 36.661 | 1.059x |
| 5120 | 17408 | 1 | 109.952 | 106.712 | 1.030x | 105.525 | 1.042x |
| 17408 | 5120 | 1 | 109.221 | 104.675 | 1.043x | 103.821 | 1.052x |
| 4352 | 5120 | 4 | 63.040 | 60.061 | 1.050x | 59.304 | 1.063x |
| 5120 | 4352 | 4 | 62.315 | 60.149 | 1.036x | 58.480 | 1.066x |
| 5120 | 5120 | 4 | 66.365 | 68.504 | 0.969x | 66.915 | 0.992x |
| 4352 | 5120 | 8 | 110.603 | 90.323 | 1.225x | 86.429 | 1.280x |
| 5120 | 4352 | 8 | 107.472 | 80.371 | 1.337x | 76.613 | 1.403x |
| 5120 | 5120 | 8 | 122.541 | 92.027 | 1.332x | 87.976 | 1.393x |

FP16 group = two-lane half dot within each 32-value Q4 block, widened before FP32 group scaling and accumulation. FP16 local = half dot and half scaled accumulation over groups in each stripe. **Both retain final FP32 stripe/lane reduction and FP32 output.** Neither means an entirely FP16 output pipeline. These semantics differ from the earlier cuBLAS COMPUTE_16F prefill study.

Up projection is M4352/K5120; down is M5120/K4352. The square, tall and long-K cases are additional checks. N is the number of input vectors processed together, not context length. A single request with 2048+ context tokens still normally has N1 for each new decode token.

Final worker ranges, frozen names and per-worker paths are retained in [confirmation-summary.json](confirmation-summary.json) and [plans.json](plans.json). Ranges are repetitions, not confidence intervals. No N2/N3/N5 performance extrapolation follows from tail correctness.

## Order and fixed-overhead checks

An audit caught a timing-harness issue: `(j + round*7) % config_count` fails to rotate a seven-configuration set. The first 33 `confirm-*` runs remain archived but are excluded from primary results. Rotation was changed to one position per round, and **all 33** frozen jobs were repeated as `final-*`. Device SASS is byte-identical before/after this host-only change.

Sixteen further isolated workers measured one arm per process, reversing arm order for the second seed. These independently corroborate the paired gains:

| M/K | N | Isolated full pipeline speedup | Kernel-only diagnostic speedup |
| --- | ---: | ---: | ---: |
| 4352/5120 | 1 | 1.098x | 1.106x |
| 5120/4352 | 1 | 1.055x | 1.064x |
| 4352/5120 | 8 | 1.289x | 1.291x |
| 5120/4352 | 8 | 1.401x | 1.427x |

The eight kernel-only workers omit activation conversion from **both** arms. Even that diagnostic reaches only about 1.43x at best; eliminating the conversion launch cannot produce 2x. These are separate worker medians, not a causal per-stage decomposition. Raw values are in [isolated-summary.json](isolated-summary.json).

## Why 2x arithmetic does not become 2x decode

1. **Weight traffic is unchanged.** Both paths read the same losslessly repacked Q4_0 codes and FP16 scales: 18 bytes/32 weights. Single-token useful GEMV intensity is only `2/(18/32) = 3.56 FLOP/weight-byte`, before activation, output and other costs. Ideal reuse lifts this to 14.22 at N4 and 28.44 at N8, but requires actually reusing each weight across those vectors.
2. **Packing and scaling still consume instructions.** The cheap final conversion uses a documented LOP3 to extract codes 16 bits apart, followed by an exact half subtraction. This removes substantial integer/packing work but does not turn unpacking, SHFL, address generation, scaling, synchronization and loads into HFMA2.
3. **The theoretical 2x applies to arithmetic throughput.** NVIDIA specifies 9.3 TF/s FP32,18.7 TF/s FP16 and 732 GB/s for the 16 GB PCIe P100. Accumulation precision does not double that bandwidth. This is consistent with the limited N1 gain; it is a traffic/issue explanation, not a hardware-counter attribution. Counters are unavailable on this rig. [NVIDIA P100 PCIe datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf).
4. **A stronger comparator matters.** The same new mapping with FP32 arithmetic gives larger apparent half gains:1.20–1.35x at N1,1.50–1.57x at N4,1.59–1.62x at N8. Those FP32 mappings lose to the archived winner; their ratios are diagnostic, not the useful achieved speedup.

With a hypothetical perfect2x arithmetic substitution and unchanged serial overhead, Amdahl's law gives `speedup = 1 / (1 - f/2)`, where f is the original time attributable to that accelerated work. Reaching 1.9x would require f ≈ 94.7%. These measurements do not support that premise for single-token decode. Overlap makes real kernels more complicated; the formula is illustrative, not a fitted profile.

## What was implemented and rejected

- **V1:** packed adjacent nibbles, half group/local accumulation, row/warp/stripe/token reuse sweep. All half candidates lost to tuned FP32. SASS exposed redundant pair assembly and expensive code packing.
- **V2:** safe memcpy half2 bitcast and separated-code extraction. In a representative R1/B1/W16/S32 function, the V1 code had 32 PRMTs; V2 has zero. The half dot has 16 HFMA2 operations; the original FP32 dot has 32 FFMA plus one group-scaling FMA. These are static whole-function counts, not dynamic issue fractions. N1 became a small win; N8 reached about 1.3–1.4x.
- **V3:** global partial outputs with a separate reduction, multiple CTA geometries and B8 weight reuse. Global reduction helps some N4 cases but does not establish a general win. The exact-order FP32 B8 extension did not beat the retained B4 winner.
- **V4:** two/four lanes per output row, shorter local dots and explicit final lane reduction. All lane-partition candidates lost to the leaders; do not repeat this mapping unchanged.

139 final configurations were exercised on the tail fixture. [sass-audit.json](sass-audit.json) records 175 compiled device kernels (including unused archived control variants), zero spills/local stack in all builds, selected static instruction counts and the unchanged final device SASS. Earlier sources and binaries remain in `snapshots/`; all actual executed hashes are retained.

## Numerical evidence and adoption decision

Normal data uses all Q4 codes, signed varying non-dyadic FP16 scales approximately 0.003–0.015 and non-half-exact FP32 activation inputs rounded by the timed preparation kernel. Every output is compared to the live archived FP32 baseline; independent complete-dot CPU emulation checks all small-fixture outputs and 64 samples/configuration on large shapes. GPU outputs are rechecked for exact repeatability after timing.

FP16 local accumulation has roughly 0.00069–0.0011 relative L2 error on the normal confirmation shapes. That is **not** a PPL delta and does not establish the +/-0.003 KLD-pair project gate. Reordered FP32 controls are also experimental unless their arithmetic matches the archived reference; all `r3*` controls remain byte-identical.

The M129/K544/N5 stress uses 645 outputs. On the large finite-activation family, all 645 outputs of the selected half group and half local candidates become nonfinite while all FP32 baseline outputs remain finite. The shorter lane-partition variants also fail that range family. The group dot can overflow before a small block scale is applied. The alternating-sign and subnormal families remain finite and match their CPU operation-order oracles; the half local subnormal case reaches about 0.00279 relative L2 error. Nonfinite rows are counted explicitly; a reported finite-only residual of zero must never be read as accuracy success.

**No candidate passed the production adoption gate.** No llama.cpp dispatch/build/model/launcher was changed, and no model logits, PPL/KLD or end-to-end tokens/s were measured. Half throughput gains cannot be adopted by relaxing the user's numerical requirements.

## Prior actual-decode evidence

The shared project history already closed native-layout Q8/Q4 half2 substitution. Those experiments were read rather than repeated. The new test was justified specifically by September's compressed row-coalesced W4A16 mapping and FP16 activation semantics, which differ from MMVQ's quantized-activation integer path.

Historical July 13 DMMV full-model F16/F32 results were 15.17/14.70 tok/s for Q8 (~1.03x, noisy FP32) and 15.36/15.39 for Q4 (~1.00x). Both lost substantially to normal MMVQ (25.59/26.60 respectively). Phase4Y's integrated Q8 magic-half path was 0.35% slower than its integer anchor, despite a faster arithmetic primitive. These are historical results, not new paired measurements, and normal integer MMVQ is not an FP32-accumulation denominator. Paths and conclusions are recorded in [NOTES-codex.md](NOTES-codex.md).

## Verification and scope

- 119 successful workers; 2,763 configuration checks, 47,092,489 output comparisons and 459,913 independently emulated CPU complete dots (counts include repeated shapes/positions).
- Four clean memcheck workers across the four kernel versions; one clean synccheck and one clean racecheck worker on 15 selected final configurations. Sanitizer timings are excluded. This is tail-fixture coverage, not full-shape sanitizer coverage.
- Final performance evidence: 33 rotated paired workers,16 isolated workers and 8 explicitly kernel-only diagnostic workers. The earlier 33 confirmation workers are retained but superseded. Three numerical stress workers are not performance evidence.
- CPU checks cover both nibble-pair mappings, shared-address/reduction geometry and two/four-lane row coverage, including ragged rows. No unknown opcodes, driver changes, resets or retries after GPU failures.
- One fresh GPU1 reservation, global benchmark lock held across builds/workers; env-i, CPU0–11, UUID isolation, telemetry and pre/post health. GPU0/2/3 untouched. Final GPU1 state 5 MiB,0% utilization,405 MHz idle SM clock,ECC0, no compute client; lock released.
- No commits or pushes in this round. [audit.json](audit.json) and the per-file manifest retain the evidence; [README.md](README.md) documents reproduction.

## Consequence for the 2x target

An accumulation-only route to approximately 2x **single-token** decode is not supported by these measurements. Keep the tuned FP32 decode path. The half kernels offer only experimental, shape-dependent batched improvements and fail the broad finite-range check. Approaching2x useful decode throughput requires an additional change that reduces weight bytes per accepted token or amortizes them over more useful tokens, followed by an actual model accuracy/performance gate. This study does not claim that such a change has been delivered or that these ratios multiply with speculative decoding gains.
