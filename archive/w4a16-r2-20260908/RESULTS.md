# W4A16 round 2: another 17-25% latency reduction, about 2x at batch four

2026-09-08. Confirmed on three P100s. FP16 inputs unchanged; FP32 arithmetic;
all tested outputs bit-identical to the previous study's operation order.

## Results against the previous winner

The comparison is against the **sealed, previously winning R4/S32 cubin**, loaded
as a separate CUDA module in every worker. These are incremental gains, not
comparisons against the older two-row or native-format reference. Control and
candidate run on the same GPU with identical data and interleaved timings.

GPU1, median of three fresh-worker medians; M = output rows, K = reduction
length, N = activation vectors. All times include scaling and final reduction.

| M x K, N | Previous winner (us) | Round 2 (us) | Latency reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 5120 x 5120, 1 | 49.579 | 40.949 | 17.4% | 1.21x |
| 17408 x 5120, 1 | 154.405 | 116.389 | 24.6% | 1.33x |
| 5120 x 17408, 1 | 147.173 | 114.523 | 22.2% | 1.29x |
| 5120 x 5120, 4 | 136.411 | 67.477 | 50.5% | 2.02x |

The prior reported historical times differ slightly from these live controls.
No cross-session or cross-device absolute times are used to calculate a gain.

| M x K, N | GPU1 reduction | GPU2 reduction | GPU3 reduction |
| --- | ---: | ---: | ---: |
| 5120 x 5120, 1 | 17.4% | 17.2% | 17.5% |
| 17408 x 5120, 1 | 24.6% | 24.7% | 24.3% |
| 5120 x 17408, 1 | 22.2% | 22.3% | 22.4% |
| 5120 x 5120, 4 | 50.5% | 50.3% | 50.3% |

Every paired worker improves. Across all individual pairs, the corresponding
gain ranges are 16.8-17.6%, 24.2-24.7%, 22.0-22.4% and 49.7-50.5%.
These ranges describe measured repetitions, not statistical confidence intervals.

## What makes it faster

**Batch one: warp-wide activation sharing, `r2_14_r2`.** Each lane loads one
FP16 activation from a coalesced group of 32 and widens it exactly to FP32.
`SHFL.IDX` distributes each activation to all lanes in turn. Each lane updates
two output rows, using ordinary FP32 FMAs in the same order as before.

**Batch four: also reuse weight decoding across columns, `r2_36_r2`.** Each lane
holds four activation values, one from each input vector. The warp shares these
values, and each decoded weight contributes to four separate FP32 dot products.
The result is less weight traffic and much less repeated decode/conversion work.
It is not extra arithmetic lanes or a dot-product instruction discovery.

Both use four warps/CTA, two rows/lane and the original 32 group stripes. They
keep the previous `[row tile][group][word][lane]` compressed weight layout and
the same separate half-scale plane. No new repacking, metadata, activation
quantization, FP32 activation scratch or table-construction pass is required
relative to the previous winner. Existing load-time repacking is still required
relative to native Q4_0 storage, and is not free.

The new kernel family also uses explicit tile/lane indexing with bounded
unsigned 32-bit element indices and read-only, non-aliasing input pointers.
SASS confirms cached input loads. This address/cache/code-structure bundle
alone is **not** a win; the scalar-load control regresses. Do not attribute the
measured gain solely to 32-bit indexing.

SASS counts below cover the straightforward group body and include group-scale
FMAs/conversions. The row/batch coverage differs, so counts are not throughput
measurements by themselves.

| Kernel | Rows x vectors per lane | FP32 FFMA | BFE / I2F | Half-to-float conversions | SHFL | Registers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sealed `direct_r4` | 4 x 1 | 132 | 128 / 128 | 36 | 0 | 64 |
| `r2_14_r2` | 2 x 1 | 66 | 64 / 64 | 3 | 32 | 32 |
| `r2_36_r2` | 2 x 4 | 264 | 64 / 64 | 6 | 128 | 48 |

The four-vector kernel needs one quarter as many weight extractions/conversions
per output as the prior kernel. The number of FMAs per output remains unchanged.
Warp shuffles add work; their benefit must come from the measured reduction of
other work, not from treating communication as free.

## Accuracy and scope

Weights retain Q4_0's exact codes and FP16 scales: 18 bytes per 32 weights,
4.5 bits/weight including the scale. Activations are original IEEE binary16
values, not BF16 or A8. The sequential 32-element FP32 group dot, scale-FMA,
32-way group striping and final FP32 reduction tree are unchanged. No half
multiplication or half accumulation occurs. The unchanged baseline reduction
kernel is used by every candidate.

The previous study already established that compiler `HADD2.F32` is used for
exact half-to-float widening, not FP16 accumulation. This build allows only
those conversion forms and contains no HFMA2/HMUL2, spills, local stack frames,
LDL or STL. SASS also showed that signed BFE was **already present** in the old
kernel, so the proposed cheaper-nibble-extraction axis was closed before testing
a redundant opcode idea. [PTX's documented BFE semantics](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe)
support sign extension; this experiment needs no undocumented encoding.

Byte identity is against the sealed study kernel and the operation-order CPU
reference, **not stock llama.cpp**. Stock code can place scaling or reductions
differently. No real model weights/activations, stock dispatch benchmark,
perplexity/KLD, full logits or tokens/sec test was run. These remain kernel
results on deterministic synthetic data in a real Q4_0 format, not demonstrated
end-to-end inference gains or a Q4 model-quality claim.

## Bounded search and unsuccessful variants

The sweep covered 77 new kernels plus the two sealed controls. Axes were
scalar/pair/128-bit/warp-shared activation loads; streamed weight words;
vector-friendly lossless weight layout; rows/lane 2/4/6/8; CTA warps 1/2/4/8;
and two/four-vector weight reuse with rows/lane 1/2/4. Every configuration kept
S32 and passed bit-identity checks. The GPU1 winner was fixed before testing
its timing on GPUs2/3 and before repeated confirmation.

Representative GPU1 frozen results:

| Configuration | Square N1 (us) | Square N4 (us) |
| --- | ---: | ---: |
| Previous R4/S32 | 49.579 | 136.411 |
| New indexing/cache/structure, scalar loads, R4 | 54.592 | 155.579 |
| Pair activation loads, R4 | 46.480 | 125.376 |
| 128-bit activation loads, R4 | 44.224 | 121.984 |
| Vector-friendly weight layout + pair loads, R4 | 43.547 | 116.021 |
| Warp sharing, R2, one vector | **40.949** | 117.285 |
| Pair loads + four-vector weight reuse, R4 | 89.573 | 97.216 |
| Warp sharing + four-vector weight reuse, R2 | 62.069 | **67.477** |

More rows are not automatically better. The streamed-weight R8 control loses,
and neither the larger-row nor CTA-warp sweeps displaced the selected winners.
The alternate vector-friendly weight repack works but does not beat the winner
using the existing layout. Four-vector batching must not be dispatched for
batch one: it does unnecessary work. Only N1 and N4 have confirmed performance;
N3 tail correctness does not establish optimal N2/N3 dispatch.

## Verification and measurement

- CPU tests pass normally and under UBSAN: all 63,488 finite half patterns,
  1,015,808 exact half-times-W4 products, 65,536 random lossless blocks, three
  address bijections, vector alignment and the tested 32-bit index bounds.
- Each GPU passes 1,027 pipelines: 79 configurations x (three tail/edge shapes
  x four numerical families + one exhaustive finite-input coverage dataset).
  Tail cases include M33/K96/N1, M129/K32/N3 and M513/K544/N4. The final dataset
  uses M16/K16384/N4 to cover all finite FP16 input patterns and all 16 weight
  codes. **3,081 full-suite checks / 2,358,624 outputs** pass across GPUs1/2/3.
- Each full suite also runs the sealed exhaustive 65,536-pattern conversion
  test: finite values/infinities/signed zeros bitwise, NaN classification only.
  Nonfinite matrix arithmetic and NaN payload preservation are not claimed.
- An independent 128-bit integer/rational dot oracle is retained. Maximum
  L1-normalized error in the full suites is 2.620e-7; this is an arithmetic
  diagnostic, not a substitute for the production perplexity/KLD gate.
- GPU1 memcheck and synccheck smoke both pass all 79 configurations, zero
  errors. No shared-memory kernel is introduced; racecheck was not run.
- Four GPU1 exploratory sweeps, then a frozen 12-configuration set. **36/36
  fresh benchmark workers pass**, three per shape on each of three GPUs:
  **432 configuration checks, 3,456 retained CUDA-event samples**. Every
  frozen output is baseline-identical; seeds and reference metrics are
  identical across repetitions and devices.
- Each worker warms for 200 ms, rotates configuration order over nine rounds,
  and times three complete pipelines per event. Round zero is discarded.
  Results are medians of the three worker medians, each containing eight
  retained samples. Shape and device order change between repetitions. No
  absolute times are pooled across devices; every worker-pair improves.
- GPU1 candidate worker-median ranges: 40.485-41.072 us (square N1),
  116.341-116.661 (tall), 114.352-114.752 (long K), 67.397-68.672 (square N4).
  All device/configuration ranges and raw worker IDs are retained in JSON/CSV.

CUDA event time includes loads, widening, decode, FMAs, scaling, partial-output
stores and final reduction. It excludes allocation, CPU repacking/oracles and
host/device transfers. Neither winner adds preprocessing relative to the sealed
winner, but an application still pays for its original repack and for dispatch.
Both use the same 32-stripe FP32 partial-output requirement. The diagnostic host
retains a larger 64-stripe buffer and unused oracle/control allocations inherited
from round 1; these allocations are not necessary deployment requirements.

Environment: Tesla P100-PCIE-16GB, SM60, driver 580.173.02, CUDA 12.8 and GCC14.
`env -i`, affinity 0-11, UUID-isolated fresh CUDA process; one GPU worker at a
time. No clock changes or performance-counter attempts (the driver limitation
is already recorded in shared memory). GPUs1/2/3 were held throughout the study.
GPU0's desktop application was excluded and left untouched. The supervisor
persists exact inputs/provenance, checks reservation hashes and device/process
health, and monitors disk space and `.xsession-errors` growth without modifying
user logs. No four-GPU/model job, reset, firmware/driver change, commit or push.

## Next step and reproduction

The kernel-level follow-up has a clear winner. The next useful step is to
compare it against the **actual production W4A16 implementation on identical
real-layer inputs**, then require production byte identity or ppl +/-0.003
before any integration. This study does not change model precision or deploy
a replacement. Large additional synthetic sweeps are lower priority than that
comparison; the present result does not establish the hardware's absolute limit.

- [Kernel source](kernels.cu), [host/oracles](worker.cpp), [CPU checks](cpu_tests.cpp),
  [plan](PLAN.md), [build/SASS inventory](build/inventory.json), [SASS](build/kernels.sass).
- [Frozen selection](frozen-manifest.json), [configs](frozen-configs.txt),
  [aggregates](aggregate.csv), [full JSON](aggregate.json), [analyzer](analyze.py),
  [supervisor](supervise.py), [previous study](../w4a16-20260908/RESULTS.md).
- Full-validation run IDs: GPU1 `1788872964593010039`, GPU2 `1788873243252414483`,
  GPU3 `1788873249296026188`. Memcheck `1788873011347734426`; synccheck
  `1788873016611542060`. All runs retain prelaunch arguments/hashes, health,
  raw stdout/stderr and status under `gpu-results/`.
- Verify archived results with `python3 analyze.py aggregate`. Rebuild with
  `python3 build.py` in a separate copy that retains access to the sealed
  round-1 cubin; do not overwrite this frozen build inventory. GPU repetition
  requires a new coordinated reservation, new token and reviewed coordination
  hash, then `validate`, sanitizer `smoke`, and `bench` through the supervisor.
  Do not reuse a released token. Default GPU1/N1; select GPU2/3 explicitly.

Round-2 cubin SHA256: `64fdbef07437dc6918821a6209577000cfe7f61a818fb973fd13b1c643d84d94`.
Worker SHA256: `332086a4789abd7caddf52be550fe6c9eaafc0881c79f99526b79454113c44ac`.
Sealed control SHA256: `21e86e715086128e8929111fc57bb419dbe4e6c490a940c1234bfb8558020b2e`.

Final [artifact audit](audit.json) passes: all 45 workers and 3,987 pipeline
checks are successful and bit-identical, with unchanged source/binary hashes.
GPUs1/2/3 were held through the entire series, gaps and final checks, then
released after the [13:27:11 UTC health check](gpu-results/1788874031294483130/final-health.json).
Each device had 5 MiB allocated, 0% utilization, no compute process and ECC0.
GPU0's existing desktop editor remained untouched. No reset or production change.
