# W8A8 / W8A4 quick feasibility screen — 2026-09-09

The tested direct packed-activation variants do not beat original W8A16 AffinityWave. This is evidence about this implementation, not a restriction on lower precision or a proof that all low-activation designs lose.

## Paired results

Median microseconds across three fresh workers per projection, GPU2, 64 experts, M=256. Ratios use the faster original Q8 branch within each worker; original F32-backed and actual FP16 activation storage branches have identical numerical outputs. Round 0 excluded as declared before measurement; all six subsequent rounds retained, six launches per event.

| Projection (N,K) | Original W8A16 | W8A8 including quantization | W8A4 including quantization | W8A8 speedup | W8A4 speedup |
|---|---:|---:|---:|---:|---:|
| Gate/up (512,2048) | 4625.1 | 5303.9 | 5326.5 | 0.872x | 0.868x |
| Down (2048,512) | 4529.8 | 4757.6 | 4769.5 | 0.952x | 0.950x |

Even with activation quantization excluded, W8A8/W8A4 consumers achieved only 0.983x/0.982x on gate/up and 0.984x/0.983x on down. Prequantized-input timing is a diagnostic, not the acceptance boundary. Primary timings charge the quantizer reading real FP16 inputs and writing packed codes/scales. Static uploads, descriptors and weight repacking are excluded equally.

## Implementation and interpretation

`prepare.py` copies the unchanged original Q8 M64/N128 HalfPipe kernel and changes only its two activation-load sites for the candidates. Weight delivery, FP32 accumulation rails/order, shared-memory geometry and persistent mapping remain the same. Activations use symmetric per-token G32 quantization: INT8 [-127,127] or INT4 [-7,7], with FP32 group scales. Consumers decode directly into the original shared-memory staging; there is no separate expansion kernel. Weights exercise the full signed INT8 range.

Lower activation storage does not reduce the FP32 FMA work in this design. The candidates add code decoding and scale multiplication; producer cost adds further latency. This explains the implementation's cost structure, although these timings alone do not attribute individual cycles. Compiler reports show 122 registers for the original and 127 for candidates, 32 KiB shared memory each, and zero spills. Register counts alone do not establish a residency loss.

The hardware limitation is absence of DP4A on GP100/P100, not a project rule. NVIDIA documents fast packed FP16 on P100 and DP4A on GP102/104/106: https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/ . A different packed-FP16 arithmetic design remains open to investigation under the accuracy requirement. Merely compressing activations while retaining the original FP32 inner loop has not produced a win here.

`arithmetic.py` records why an exact HFMA2 shortcut is not automatic. W8A4 individual integer products are half-exact, but sums need not be (three 889 terms produce 2668 instead of 2667 under sequential half accumulation). W8A8 already has half-inexact individual products. Splitting W8 into two nibbles permits exact short W8A4 partials, but consumes both HFMA2 lanes for one original product plus reconstruction. Approximate HFMA2 remains permitted; it requires a new kernel and paired model accuracy evaluation. The corresponding W8A16 FP16-accumulation control should also be tested to distinguish arithmetic benefits from activation compression.

## Verification and limits

All six fresh workers passed: every quantized code/scale validated on CPU, 256 output dots per arm checked against a complete two-rail FP32 CPU oracle, all output values finite. Both original Q8 input branches are byte-identical. A separate M=33, two-expert gate/up compute-sanitizer run reports zero errors. All GPU reservations were released.

Synthetic relative output L2 error is approximately 0.00471 for A8 and 0.0878 for A4 versus original Q8. These are not perplexity measurements or model accuracy qualification. Inputs include zero groups and outliers, but are not captured model activations. No full-model throughput, ragged-workload result, PPL acceptance, or 1.4x claim follows from this screen.

Compiler command and input/binary hashes are retained in `manifest.json` and each worker metadata file. Shared study flags omit `--use_fast_math`; these are controlled standalone comparisons, not production flag qualification. `analyze.py` verifies hashes, successful exits, checks and sample counts and writes `summary.json`. Raw `*.out`, `*.err`, `*.meta.json`, `build.log`, `provenance.json`, and `arithmetic.json` preserve the evidence.
