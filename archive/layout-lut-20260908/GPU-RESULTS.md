# Single-P100 endpoint-SAD and LUT results — 2026-09-08

## Outcome

The useful result is **shape-dependent W2 arithmetic on a better row/load
mapping**, not evidence of hidden FP4/FP2 hardware. Register-LUT reduces latency
by 2.7–3.9% versus the tested byte-layout integer arm on the two batch-one,
K=5120 shapes. Endpoint-SAD reduces it by 8.6% on the longer K=17408 shape and
15.6% at batch four. W4 has no convincing LUT advantage over the integer arm.
Neither shared-LUT implementation is competitive with the leading arms here.

These are prepared-device synthetic matvec/batched-matvec pipelines, **not
llama.cpp or model inference speedups**. No production code, driver, firmware,
unknown encoding, commit or push was changed/executed in this series.

Status: **all 24 planned fresh benchmark workers completed**, with three workers
for every shape/precision cell. All numerical checks passed. The last W4,
M=K=5120, N=1 repetition was initially policy-blocked by the pasted sub-Q8 ban.
The user subsequently explicitly removed that restriction and requested GPU1
continuation. The global project instructions now record that this supersedes
the old ban without relaxing numerical accuracy requirements; Claude history
was left read-only. Under a new GPU1 reservation, worker
`gpu-results/layout-1788869038900603625/` completed the missing repetition with
identical worker, cubin and configuration hashes. The initial interruption is
historical, not an outstanding authorization blocker.

GPU1's reservation was held across the entire series and final artifact/health
checks, then explicitly released at 11:57 UTC. Final check at 11:56:46 UTC:
5 MiB, 0% utilization, no target compute process, zero volatile uncorrected ECC
errors. GPU0's editor was untouched. See [final-health.json](final-health.json).
This describes the original reservation. GPU1 was freshly reserved for the
authorized completion above; its final release is recorded in the continuation
section below.

## Main timings

Microseconds per complete device pipeline; lower is better. M=output rows,
K=reduction length, N=input vectors. Median of each worker's eight retained
timing samples, then median across workers. Each sample times three pipelines.

| Weights | M × K | N | Byte-layout integer | Endpoint-SAD | Register-LUT | Compact shared LUT | Replicated shared LUT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W2 | 5120 × 5120 | 1 | 36.245 | 36.640 | **34.832** | 49.147 | 47.355 |
| W2 | 17408 × 5120 | 1 | 96.512 | 98.208 | **93.872** | 132.251 | 126.677 |
| W2 | 5120 × 17408 | 1 | 99.984 | **91.339** | 94.389 | 137.520 | 131.600 |
| W2 | 5120 × 5120 | 4 | 106.203 | **89.621** | 101.408 | 137.760 | 129.355 |
| W4 | 5120 × 5120 | 1 | **44.496** | 48.832 | 45.696 | 70.187 | 70.480 |
| W4 | 17408 × 5120 | 1 | 133.915 | 137.915 | 134.016 | 195.824 | 194.160 |
| W4 | 5120 × 17408 | 1 | 126.027 | 132.192 | 126.187 | 209.797 | 214.939 |
| W4 | 5120 × 5120 | 4 | **129.093** | 140.069 | 135.019 | 198.224 | 188.277 |

Near-tied W4 results on the larger batch-one shapes do not establish a winner.
Full configuration data, individual
samples, worker medians and ranges are in [GPU-SUMMARY.json](GPU-SUMMARY.json)
and [GPU-SUMMARY.csv](GPU-SUMMARY.csv). W2 leading-arm worker ranges are:

| M × K / N | Leading arm | Worker-median range, µs |
| --- | --- | ---: |
| 5120 × 5120 / 1 | Register-LUT | 34.544–35.269 |
| 17408 × 5120 / 1 | Register-LUT | 93.803–94.256 |
| 5120 × 17408 / 1 | Endpoint-SAD | 91.221–92.384 |
| 5120 × 5120 / 4 | Endpoint-SAD | 89.131–90.800 |

One retained sample was a 559.051 µs outlier in the third W4 batch-four integer
worker (that worker's median was 129.808 µs). It remains in the raw data; no
outlier trimming was used. These short runs on a shared host are not a
tail-latency guarantee, and tiny differences should not be promoted to wins.

## Attribution controls and interpretation

All W2 main arms use 128 rows/warp (R4) and 32 splits. W4 register/shared arms
also use R4/S32. W4 integer has frozen R2/S8 and R4/S32 alternatives; SAD has
R2/S16 and R4/S32. The table shows the faster aggregate of those alternatives
per shape; both remain in the CSV. W4's integer chooses R2/S8 on the square
shapes and R4/S32 on the other two. These are bounded tested settings, not
globally optimal tuning claims for every shape.

The old one-row-per-warp kernels were compiled from the unchanged earlier
source into a second cubin and loaded by the same driver. They use identical
canonical weights, activations and non-power-of-two scales. This avoids
comparing different random/scaling datasets or crediting all mapping gains to
the new arithmetic.

| Weights | M × K / N | Old integer VMAD | Old masked SAD | Old reversed endpoint SAD | New best arm |
| --- | --- | ---: | ---: | ---: | ---: |
| W2 | 5120 × 5120 / 1 | 40.032 | 40.133 | 44.576 | 34.832 |
| W2 | 17408 × 5120 / 1 | 119.904 | 119.765 | 128.629 | 93.872 |
| W2 | 5120 × 17408 / 1 | 125.787 | 125.952 | 134.443 | 91.339 |
| W2 | 5120 × 5120 / 4 | 140.288 | 140.112 | 150.064 | 89.621 |
| W4 | 5120 × 5120 / 1 | 49.019 | 52.325 | 54.069 | 44.496 |
| W4 | 17408 × 5120 / 1 | 149.589 | 152.203 | 158.155 | 133.915 |
| W4 | 5120 × 17408 / 1 | 159.707 | 162.363 | 166.971 | 126.027 |
| W4 | 5120 × 5120 / 4 | 174.891 | 177.301 | 184.779 | 129.093 |

Thus merely removing the per-plane AND did **not** make the old endpoint kernel
win. The new row reuse, coalesced layout and split-K mapping substantially
improve even ordinary arithmetic. The W2 arithmetic advantage is the smaller
increment against that improved integer control, not the entire old-to-new gain.

The quartet-layout integer control is also included: e.g. its W2 square/R4/S32
pipeline is much slower than the byte-layout integer arm. Beating that weaker
control alone would exaggerate LUT's benefit. Both layouts have identical
payload sizes and exact lossless repacking; their decode costs differ.

The precomplemented endpoint layout (F2) works on hardware. At matched R1/S8 it
tracks the reversed-endpoint layout (F1) closely, with no consistent meaningful
performance advantage across shapes. For example W2 square/batch-one is
44.347 vs 44.544 µs; long-K is 127.541 vs 127.616 µs. Main SAD timings use F1.
F2 currently has only an R1 compiled arm, so this is not an R4 F1/F2 comparison.

The compact table is conflict-free for this particular 16-entry/32-bit/one-table
per-warp mapping, as explained in the [offline report](RESULTS.md). Nevertheless
its lookup, synchronization and resource costs make it slower here. Replication
occasionally beats compact storage but neither beats the leading register/
integer/SAD arm. No shared-memory redesign is justified by these measurements.

## Correctness and test coverage

- All 60 compiled A8 row kernels passed 720 GPU pipeline cases across W2/W4,
  M=33/129/513, K=96/32/544, N=2 and splits 1/2/4/8. This exercises incomplete
  rows, R1/R2/R4 reuse and splits exceeding the number of groups.
- Six independent SAD group probes passed 65,536 G32 vectors each, with endpoint
  orientations F0/F1/F2 and both precisions. GPU activation sums also matched.
- The earlier nine-method endpoint executable passed its W2/W4 raw-group and
  scaled-output checks. Its initial timing dataset is not mixed into the tables.
- Memcheck, racecheck and synccheck each passed the focused shared/register-LUT
  smoke workload: zero reported errors/hazards/warnings. This covers 24 pipeline
  cases per tool, **not** every kernel/configuration under every sanitizer.
- The exploratory R/split sweeps passed all 76 configurations per precision.
  Their timing samples selected the frozen configurations and are excluded from
  the final aggregates.
- All 312 configuration/shape validations in the 24 frozen workers passed
  before timing. There are 104 aggregates and 2,496 retained timing samples.

Integer group dots are compared against independent scalar signed-integer CPU
arithmetic, including uniform weight codes and activation -128/+127. Scaled
outputs are bit-identical to CPU references reproducing **each kernel's own
FP32 reduction order**. Split-K and the old warp tree can differ from one
another in rounding. This does NOT satisfy a byte-identical production output
gate by itself. No perplexity/KLD result exists; the model accuracy gate remains
pending. The separate FP32-activation LUT prototypes were not executed here.

## Measurement scope and reproducibility

- Physical GPU1, Tesla P100-PCIE-16GB, SM60/56 SMs; UUID
  `GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b`; driver 580.173.02; CUDA 12.8;
  GCC14. No clock settings were modified. Each worker warms for 200 ms; clocks
  and temperatures are sampled before/after, not continuously during kernels.
- Fresh process per worker; Driver API loads hashed, unmodified cubins.
  Controller never initializes CUDA. UUID isolation, `env -i`, CPU affinity
  0–11, persistent prelaunch records, timeouts, occupancy/ECC checks, and no
  automatic GPU reset/recovery. Other GPUs and user processes remain untouched.
- CUDA event timing includes activation-sum preparation for SAD, LUT table
  construction, group corrections/scales, partial writes/reads and final split-K
  reduction, including their device-pipeline launch gaps. Nine rounds rotate
  method order; round zero is excluded uniformly. Three pipelines per sample.
- Input A8 quantization, host/device copies, allocation and CPU weight repacking
  are **not timed**. CPU packing time is logged separately; the unoptimized
  host prototype took roughly 0.13–0.25 s for the first W2 5120-square layouts.
  Repacking must be amortized at model load, not repeated for every inference.
- Payload is 2 or 4 bits/weight plus one FP32 scale per G32: an additional
  1 bit/weight before tail padding. No weight-sum metadata is stored in the main
  arms; SAD uses POPC. Its int16 activation sums are shared across rows and
  computed in the charged preparation kernel. N=4 is batched matvec, not an
  optimized tiled multi-vector GEMM with cross-vector weight reuse.
- Weights are signed two's-complement synthetic codes, not native Q2_K/Q4_K or
  IQ codebooks. Group scales and affine offsets need format-specific treatment.

Run [analyze_gpu.py](analyze_gpu.py) to regenerate summaries from the preserved
logs. It checks exact source/config/worker hashes, samples and expected worker
counts, requiring three workers for every cell. See
[gpu-build.json](gpu-build.json), [frozen-configs.txt](frozen-configs.txt), and
[results/inventory.json](results/inventory.json) for commands and provenance.

Key artifacts:

- Full GPU correctness: `gpu-results/layout-1788867334568923618/`.
- Memcheck/racecheck/synccheck: `layout-1788867356179028641`,
  `layout-1788867372012605577`, `layout-1788867449129405075` under `gpu-results/`.
- Frozen run IDs and complete pre/post health records: `GPU-SUMMARY.json`.
- Original endpoint worker: `gpu-results/endpoint-1788867050354663480/`.
  Worker passed but the initial supervisor rejected post-run nonzero utilization.
  Inspection showed no process, 5 MiB and ECC0; later idle check passed. The
  supervisor now permits lagged post-run utilization while still checking memory,
  processes and ECC. The original STOP record is preserved.
- A later prelaunch idle guard stopped at `layout-1788868155265157617`: 4%
  utilization, no target process, 5 MiB, ECC0. No worker launched. Manual checks
  returned 0%, and remaining permitted jobs were explicitly resumed. This
  prelaunch-only record is listed separately, not counted as a benchmark worker.
- A coordination-file change was also reviewed before resuming: the other
  session explicitly confined itself to GPUs 0/2/3 and respected GPU1.

## Next priority

The missing W4 repetition is now complete. Take the W2 register-LUT and
endpoint-SAD candidates into **one real quantization-format kernel**, with
the matched integer path retained. Measure complete activation preparation,
actual scale/offset metadata and representative dimensions, then run the
production numerical gate before any integration. W4 should retain the integer
path as the baseline; do not spend a long campaign tuning shared LUT or assume
that the W2 SAD benefit generalizes to W4. These findings warrant a bounded
real-format investigation, not a broad unknown-opcode sweep.

## GPU1 continuation close-out

The new reservation `lowbit-20260908-1200-gpu1` was held across the final W4
worker, aggregate regeneration and final artifact/health checks, then released
in the coordination log. Final check at 12:05:39 UTC: GPU1 5 MiB, 0% utilization,
no compute process and zero volatile uncorrected ECC errors; no reset. Other
GPU processes were left untouched. See
[final-health-continuation.json](final-health-continuation.json).
No test remains running. The sub-Q8 restriction is removed; any subsequent
GPU test still needs a fresh reservation and the unchanged accuracy gates.
