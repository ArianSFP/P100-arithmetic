# FP16 accumulation: approaching 2x at 2048+ tokens

**The tested optimizations reach about 1.89–1.91x on four of the six shard shapes against a tuned FP32 pipeline. A uniform 2x was not achieved.** The 2048-token up projection remains variable; the 8192-token down projection also falls short. All costs below include Q8 decoding, activation preparation, GEMM and a final FP32 output.

Target shapes are the original report's four-way dense FFN shards: up M4352/K5120 and down M5120/K4352. N is the number of tokens in this single GEMM, not total prompt length. The user narrowed this follow-up to N2048/4096/8192. These results do not cover the separate two-GPU M8704 shard study.

## Frozen paired confirmation

Three fresh workers per shape, new seeds and a broader finite weight distribution than screening. Each worker has nine rotated ABBA rounds per comparison; both same-mode legs are averaged within each round before taking worker medians. Table entries are medians across the three worker medians. Both arithmetic modes receive native aligned Q8 decoding and independently selected exact cuBLAS/layout optimizations. No cached weights or excluded transposition costs.

| Projection | Tokens | Tuned FP32 pipeline ms | Selected FP16 pipeline ms | FP16 speedup |
| --- | ---: | ---: | ---: | ---: |
| Up | 2048 | 10.792 | 6.295 | 1.714x |
| Down | 2048 | 10.994 | 5.814 | 1.891x |
| Up | 4096 | 21.321 | 11.249 | 1.895x |
| Down | 4096 | 21.704 | 11.478 | 1.891x |
| Up | 8192 | 42.294 | 22.185 | 1.906x |
| Down | 8192 | 41.309 | 22.768 | 1.814x |

**The 8192 down FP16 plan is the native TN/default-algorithm path itself.** Its apparent latency differences between identical paired arms are an order effect, not an optimization. We therefore also measured each selected arithmetic mode in its own fresh worker, after 30 complete pipeline warmups. These secondary checks contain 31 event samples of two full pipelines each.

## Isolated-mode sensitivity check

| Projection | Tokens | FP32 workers | FP16 workers | FP32 ms | FP16 ms | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Up | 2048 | 2 | 2 | 10.793 | 6.040 | 1.787x |
| Down | 2048 | 2 | 2 | 10.996 | 5.815 | 1.891x |
| Up | 4096 | 2 | 2 | 21.321 | 11.249 | 1.895x |
| Down | 4096 | 2 | 2 | 21.700 | 11.476 | 1.891x |
| Up | 8192 | 2 | 2 | 42.293 | 22.183 | 1.907x |
| Down | 8192 | 2 | 2 | 41.310 | 22.769 | 1.814x |

At 2048 up, the two isolated FP16 worker medians are 5.784 and 6.297 ms against 10.793 ms FP32: approximately 1.87x and 1.71x. Their average does not establish a stable 1.79x path. The paired confirmation gives 1.714x. The best individual timing is not promoted as the result. The exact cause of this timing variation remains unresolved; storage and output-shape padding did not eliminate it.

## Why the original table showed about 1.6x

The original table timed entire pipelines, not only arithmetic. Both cuBLAS modes already receive the same FP16 weights and FP16 activations. Switching accumulation precision therefore does not halve their input traffic, and it does not accelerate Q8 decoding or activation preparation. The FP16 path additionally widens its half output to FP32.

Fresh N256 screen medians (six workers/shape across the first two sweeps):

| Projection | FP32 GEMM us | FP16 GEMM us | GEMM ratio | Full-pipeline ratio |
| --- | ---: | ---: | ---: | ---: |
| Up | 1668.55 | 844.80 | 1.975x | 1.688x |
| Down | 1585.24 | 885.09 | 1.791x | 1.599x |

The original scalar weight decoder takes about 231 us, activation conversion16–18 us, and output widening16–18 us at N256. These separately timed stages diagnose the cost; their medians are not an exactly additive decomposition because cache state and surrounding launches differ.

An idealized accounting is `T32 = D + A + G32`, `T16 = D + A + G16 + W`. Even if `G16 = G32/2`, common preparation D/A and extra widening W keep the full-pipeline ratio below2. A different cuBLAS algorithm can also have different useful arithmetic efficiency.

[NVIDIA's Pascal guide](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html) describes the packed-half arithmetic advantage. For the measured56 SMs at 1328 MHz, the ordinary dense-FMA peaks are 9.519 TF/s FP32 and 19.038 TF/s FP16. These are calculated ceilings, not hardware-counter measurements. The best large GEMMs consume most of this arithmetic budget, leaving limited room to hide preparation/output traffic.

cuBLAS12.8 documents COMPUTE_16F with F16 A/B/C; it does not offer F32 output for that compute-type combination. Removing the widening pass while keeping this output contract would require a different kernel or a consumer fusion. [Supported type table](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasgemmex)

## Implemented and tested changes

1. Native aligned Q8 decoding as the final common control. The earlier four-value half2 decoder reduced the study's scalar decoder from about 232 to 162 us, but it is not claimed as an improvement over the backend's native decoder.
2. Per-shape cuBLAS algorithm selection for both FP32 and FP16. IDs 0–23 were screened at large shapes, retaining supported configurations only. Numerical changes are recorded and excluded from the exact candidates.
3. Eight operand-storage/orientation combinations, including computing the transposed output and transposing/widening it back. Every required conversion and transpose is charged.
4. Fused Q8 decoding plus weight transpose, and fused F32-to-F16 activation conversion plus activation transpose. These preserve every independently prepared input word; all selected final output words are checked too.
5. Activation/weight storage padding 8/32/128/256/512, followed by zero-filled output-row padding 128/256/512 on the weak2048 up shape. Row padding lost: half pipelines around6.75–6.83 ms versus the unpadded probe's6.12 ms. The storage-pad8 lead did not stabilize fresh-worker timing.

FP16 selections (`layout` bits:1 transposes W;2 transposes X;4 swaps operands and transposes output):

| M | N | K | Algorithm | Layout | Leading-dimension padding |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4352 | 2048 | 5120 | 3 | 2 | 8 |
| 5120 | 2048 | 4352 | 3 | 6 | 0 |
| 4352 | 4096 | 5120 | 3 | 6 | 0 |
| 5120 | 4096 | 4352 | 3 | 3 | 0 |
| 4352 | 8192 | 5120 | 99 | 3 | 0 |
| 5120 | 8192 | 4352 | 99 | 0 | 0 |

Algorithm 99 is DEFAULT_TENSOR_OP, implemented with ordinary half arithmetic on this P100. Layout0 is original TN. Layout2 is TT with transposed activation. Layout3 materializes both input transposes. Layout6 swaps operands with transposed activation. These plans are experimental, tied to these exact shapes and CUDA12.8/cuBLAS12.8.3; they are not a general production dispatch table.

## Accuracy, controls and limits

All selected paths match their own arithmetic-mode default output byte-for-byte, before and after timing. FP16 equality is to the existing FP16 path; it is not equality to FP32 accumulation. Input transformations are checked over all prepared weights/activations; FP32 default also receives 32 independently accumulated complete CPU dot checks per worker. Fresh confirmation includes signed Q8 values −128…127 and wider finite scales 0.003…0.015.

No real-model KLD-pair/perplexity gate or full-model prefill benchmark was run. The required model tolerance ±0.003 is therefore not claimed. Changing batch size can change cuBLAS accumulation order; per-shape identity is not an across-batch identity claim.

Cached-weight experiments at N256 can show about 2.14x/2.00x against the original uncached FP32 pipeline. Giving FP32 the same cache reduces the ratios to about 1.85x/1.75x. That experiment changes preparation costs and is not used to claim a2x accumulation gain. Retaining one decoded shard matrix costs another42.5 MiB/device, so whole-model caching also needs a memory budget.

The separate large-GEMM study's handwritten half2 and Strassen regressions were reviewed and not repeated. No SASS mutation, precision relaxation, undocumented opcode, clock change, reset, production edit, commit or push was used here.

## Reproduction and evidence

[README](README.md) describes the layouts, build commands, controller and timing contracts. [Frozen requests](confirmation-jobs.json) pin algorithms/layouts. [Selected plans](plans.json) include the independently tuned FP32 comparator. [Paired confirmation](confirmation-summary.json), [isolated checks](isolated-summary.json), [large screen](large-screen-summary.json), [storage-padding screen](padding-summary.json) retain worker medians and samples. The `large-*` directories contain raw logs, commands, binary/source hashes, health and clock/power telemetry.

The arithmetic scripts require one fresh, coordinated GPU1 worker at a time under the shared rig lock. CPU affinity is0–11, with an explicit empty CUDA environment. Desktop-log/disk/deadline guards remain active; other GPUs and their clients are untouched.

Artifact audit currently covers **104 successful workers and 10 clean memcheck workers**. Every executed source/binary hash has a retained matching file. [Audit](audit.json).
