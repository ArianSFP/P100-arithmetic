# W4A16 on P100: confirmed precision-preserving kernel gains

2026-09-08. Isolated GPU1 study; no production inference code changed.

## Main result

Four-row activation reuse reduces complete kernel-pipeline latency by **23.9-28.8%** against an already tiled, two-row-reuse direct-arithmetic control. Both load the same FP16 activations, perform FP32 arithmetic/accumulation, use the same lossless Q4_0 repack, and produce bit-identical outputs in all tested cases. No activation quantization or half-precision accumulation is involved.

M = output rows, K = reduction length, N = activation vectors. Times include group scaling and the final split-K reduction. All entries below use the same 32 K stripes and reduction tree.

| M x K, N | Two-row control (us) | Four-row candidate (us) | Latency reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 5120 x 5120, 1 | 66.069 | 49.019 | 25.8% | 1.35x |
| 17408 x 5120, 1 | 203.040 | 154.608 | 23.9% | 1.31x |
| 5120 x 17408, 1 | 205.413 | 148.112 | 27.9% | 1.39x |
| 5120 x 5120, 4 | 189.403 | 134.891 | 28.8% | 1.40x |

These are stronger-baseline comparisons, not gains against an intentionally simple reference. Against the same tiled kernel with only one row per thread, the reduction is 49.5-55.2%. The simple one-warp-per-row native/repacked references are much slower still; their large ratios are **not** the headline claim.

The earlier W4A8 study reported 9.2-26.2% reductions against its old integer control. These W4A16 gains are comparable or larger without A8 conversion, but the two studies use different weight-scale storage and controls. Their percentages and absolute times are not an apples-to-apples activation-precision comparison.

## What changed

- A lane handles four output rows, spaced 32 rows apart. It reuses each activation load and exact half-to-float conversion across four independent FP32 accumulators. The control handles two rows.
- Weight words are coalesced across rows: `[row tile][scale group][word][lane]`. FP16 scales use `[row tile][group][lane]`. Both comparison arms use exactly this layout.
- Each G32 dot executes the same 32 FP32 FMAs in the same order. Scaling stays inside its group. Group stripes and the final FP32 tree match the reference's 32 lanes.
- The four-row main kernel uses 64 registers versus 40 for two rows and 32 for one row. There are no spills or local-memory stack accesses. These are normal compiler-generated SM60 instructions, not undocumented opcode discoveries.

The byte-identical guarantee is against this study's explicitly defined FP32 operation order and native-format reference, **not stock llama.cpp**. In particular, applying a group scale after its dot need not match a production implementation that scales individual weights first.

## Precision and format

Weights use real Q4_0 block semantics: 32 offset-coded nibbles and one FP16 scale, 18 bytes/group, or 4.5 bits/weight including the scale. Native low nibbles represent weights 0-15; high nibbles represent weights 16-31; decoded integers are `code - 8`. The nibble and plane transformations preserve every code and scale bit. Benchmark dimensions require no row padding; general tails pad to 32 rows.

This follows the local [block definition](/home/arian/llama.cpp-qwen38-p100/ggml/src/ggml-common.h:194) and [decoder](/home/arian/llama.cpp-qwen38-p100/ggml/src/ggml-quants.c:459). It does not reinterpret Q4_K or IQ codebooks as Q4_0.

A16 here means IEEE binary16, not BF16. Inputs remain their original 16-bit words and are widened exactly to FP32 for arithmetic. The compiler emits `HADD2.F32 ...H0_H0, -RZ.H0_H0` for this conversion; that is **not** half-precision accumulation. Its output was checked exhaustively over all 65,536 input patterns: finite values, infinities and signed zeros match bitwise; NaNs match classification. See NVIDIA's [half conversion documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html).

No HFMA2/HMUL2 occurs in the cubin. Finite arithmetic tests include subnormals, cancellation, mixed signs and extreme half values/scales. Arithmetic on NaN/infinity inputs was not exhaustively tested; this report does not promise preservation of NaN payloads or exceptional arithmetic behavior.

## Competing paths and stronger controls

| M x K, N | Strict direct R4/S32 | Direct plus one-time FP32 preparation | Register-LUT R4/S32 | Fastest selected direct, reordered |
| --- | ---: | ---: | ---: | ---: |
| 5120 x 5120, 1 | 49.019 | 51.344 | 52.928 | 47.664, S16 |
| 17408 x 5120, 1 | 154.608 | 161.680 | 170.587 | 143.856, S8 |
| 5120 x 17408, 1 | 148.112 | 151.243 | 155.819 | 145.589, S16 |
| 5120 x 5120, 4 | 134.891 | 137.819 | 143.579 | 124.160, S8 |

All times are microseconds. The FP32-preparation control reads the original FP16 inputs and writes FP32 scratch once **per timed pipeline**, charging its launch, conversion and traffic. It preserves all activation values but does not help the winning mapping. Matching preparation variants were also tested for native, old-repacked, one-row, two-row, bit-plane direct and LUT kernels.

Register-LUT builds FP32 16-entry activation tables and uses SHFL lookup, with table construction and plane combination included. It beats the deliberately matched bit-plane direct decoder but not the best nibble direct path. Its summation order changes many output bits: for the square N1 case, 3,928/5,120 results differ from the baseline. The LUT is not a precision-preserving replacement in the byte-identity sense.

S8/S16 direct variants also change the group reduction order. They may merit a later perplexity/KLD comparison, but are **not** promoted under the current strict accuracy result. No FP16 value is quantized by these variants either; their difference is FP32 summation order.

## Validation and timing protocol

- CPU: all 63,488 finite half patterns; 1,015,808 exact half-times-signed-W4 products; 65,536 random lossless Q4_0 blocks; both address-layout bijections. Normal and UBSAN builds pass.
- Final GPU build: 30 kernel entries; no half multiply/accumulate, spills, stack frames, LDL or STL. SASS audit permits only the verified exact half-widening HADD2 pattern.
- Full correctness: 130 configurations x 3 shapes x 4 numerical families = **1,560 pipelines**. Tail shapes are M33/K96, M129/K32, M513/K544, all N2. Every output matches its CPU operation-order model bitwise. The strict configurations cover 86,400 baseline-identical output checks in this suite.
- An independent integer/rational oracle represents finite half values in units of 2^-24 and scaled dot products in units of 2^-48, using 128-bit integer totals. Across the full suite, maximum L1-normalized error is 3.619e-7; maximum relative L2 error is 1.539e-7. These are arithmetic diagnostics, not a model-quality gate.
- Memcheck and synccheck smoke runs both report zero errors. No shared-memory LUT is used; this study did not run racecheck.
- Four exploratory sweeps, each with 94 configurations, selected a fixed 17-configuration file before confirmation runs. Selection and source/cubin/worker hashes are preserved.
- **12/12 frozen workers pass**, three fresh processes per shape, with 204 configuration checks. Each worker warms for 200 ms, rotates configuration order across nine rounds, and times three complete pipelines per CUDA-event sample. Round zero is discarded. Reported time is the median of three worker medians, each based on eight retained samples: **1,632 retained event samples** total.
- Strict candidate worker-median ranges: 48.779-49.323, 154.512-155.291, 147.840-148.171 and 134.773-135.317 us, respectively. These ranges are descriptive, not statistical confidence intervals.

Environment: P100-PCIE-16GB / SM60, CUDA 12.8, GCC 14, driver 580.173.02, UUID `GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b`. Fresh CUDA workers run under `env -i`, CPU affinity 0-11 and UUID-based single-GPU visibility. No clock or driver changes. CUDA events include all kernel stages, but do not represent whole-application CPU scheduling/launch latency.

The earlier sweep overlapped another session using GPUs 0/2/3, not GPU1. That session released them before frozen timing. The reservation guard stopped the first frozen launch when the coordination file changed; after reviewing the other session's release and our unchanged GPU1 claim, execution resumed with the new exact coordination hash. No CUDA worker launched during that guard stop. The user's GPU0 editor process was left untouched.

## Costs and limitations

Host repacking/round-trip validation, allocation and host/device transfers are outside event timing. Weight payload stays compressed (18 bytes/group), but repacking is not free. Representative first-worker CPU nibble repack plus validation costs are 147, 374, 412 and 132 ms for the four shapes. These CPU routines are diagnostic, not optimized model loaders; transfer time is additional. Integration must amortize preparation and account for peak memory while native and repacked copies coexist.

The strict candidate needs FP32 partial-output scratch of `32*M*N*4` bytes. The optional pre-widening control additionally uses `N*K*4` bytes. Native, nibble and bit-plane weights coexist in the study for comparison; a deployment need not retain every representation.

Data are deterministic synthetic codes, scales and activations in a real block format. No real GGUF layer weights, production dispatch, perplexity/KLD, model logits, tokens/sec, end-to-end latency, Q4_K or IQ validation was performed. The strict result preserves this study's arithmetic; the project's production byte-identity or ppl +/-0.003 gate remains pending.

**Next priority:** benchmark this exact FP16-input/FP32-accumulation row-reuse kernel against the actual production W4A16 path on identical real layer inputs, then apply the production accuracy gate before any integration. Further SAD or half2 work is not justified by these results.

## Reproduction and artifacts

- [Plan](PLAN.md), [kernel source](kernels.cu), [host worker/oracles](worker.cpp), [CPU tests](cpu_tests.cpp), [shared layout helpers](common.hpp).
- [Build/SASS inventory](build/inventory.json), [SASS](build/kernels.sass), [frozen configurations](frozen-configs.txt), [selection manifest](frozen-manifest.json).
- [All aggregates](aggregate.csv), [full JSON including error metrics, worker IDs and packing costs](aggregate.json), [analyzer](analyze.py), [supervisor](supervise.py).
- Final-build full validation: `gpu-results/1788871205529957340`; memcheck: `1788871224803703859`; synccheck: `1788871237534431239`. Each run retains exact prelaunch arguments/hashes, health/process snapshots, raw stdout/stderr and final status.
- Initial reduction-stack and pre-widening builds are archived separately. They are not mixed into the frozen results. The reduction stack allocation was removed with compile-time tree indexing before timing the final build.

Verify/aggregate these archived results with `python3 analyze.py aggregate`. For an offline rebuild, run `python3 build.py` in a separate copy: rebuilding changes the build inventory, which the frozen-result audit correctly rejects. GPU reproduction requires a **new coordinated GPU1 reservation**, reviewing current occupancy/health, and updating the supervisor's reservation token and exact coordination hash. Do not reuse a released claim. Run `validate`, sanitizer `smoke`, and `bench --m M --k K --n N --configs /absolute/path/frozen-configs.txt` through the supervisor, one worker at a time. It stops on worker failure, timeout, health failure or changed coordination; it never resets a GPU.

Final cubin SHA256: `21e86e715086128e8929111fc57bb419dbe4e6c490a940c1234bfb8558020b2e`.
Final worker SHA256: `0d7610a83a73b5089a51f6b1bf12b6de8630adbfe6b621adadd651de07a62e8f`.

GPU1 was held across the full series, build/inter-run gaps and artifact checks, then released after the final 12:51:34 UTC device check: 5 MiB, 0% utilization, no compute process on GPU1, ECC0. No reset or production changes. The existing GPU0 editor was untouched. See [final checks](final-checks.json) and the explicit release in the shared coordination log.
