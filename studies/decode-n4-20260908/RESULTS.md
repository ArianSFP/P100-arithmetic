# Four-token decode: 1.48–1.52× over original FP32

On P100 GPU1, compressed Q4 weights and four FP16 activation columns, the new FP16 local-accumulation kernels exceed the requested 1.4× against the original FP32 baseline. Optimization also improves FP32: against the strongest exact FP32 kernels found here, the advantage is 1.31–1.34×. This is a standalone matrix-kernel result, not end-to-end model decode throughput.

| M × K, N=4 | Original FP32 µs | Improved FP32 µs | FP16 µs | vs original | vs improved |
|---|---:|---:|---:|---:|---:|
| 4352 × 5120 | 63.00 | 55.95 | 41.85 | 1.505× | 1.337× |
| 5120 × 4352 | 62.36 | 54.47 | 40.90 | 1.525× | 1.332× |
| 5120 × 5120 | 66.20 | 58.66 | 44.81 | 1.477× | 1.309× |

Each value is the median of three fresh worker medians, with eight retained rounds per worker, rotating configuration order and 12 complete pipelines per timing sample. Activation conversion and final reduction are included. Plans were frozen after screening, before confirmation. Separate single-configuration workers, two seeds with reversed arm order, confirm 1.330×, 1.339× and 1.312× against improved FP32 respectively. See `plans.json`, `confirmation-summary.json`, `audit.json`, and raw run logs.

The previous best FP16 controls measured in the same confirmation workers take 59.27 µs (up), 58.66 µs (down), and 66.92 µs (square). Thus this round improves FP16 itself by approximately 1.42×, 1.43× and 1.49×.

## What changed

The parent N4 fused kernel has only 68 CTAs for the up shape on 56 SMs. The parent N8 winner used two groups of four columns, suggesting that additional CTAs help. The new kernel partitions K into four or eight coarse pieces, computes four output rows per lane, and reduces a small FP32 partial buffer. This improves available parallel work while reusing activation loads across rows. The winning kernels convert raw FP32 activations to half inside the kernel, removing the separate preparation launch while retaining its rounding semantics. Final reduction and output remain FP32.

The winners are `f17r4s32p8` for up and `f17r4s32p4` for down/square. Exact FP32 controls use `f21r2s32p4` and `f21r6s32p4`. FP16 group accumulation with FP32 scaling is also retained, but is slower than local FP16 accumulation. Intermediate row counts, larger row reuse, explicit warp counts, and a paired activation preparation layout were tested; none beat the selected half kernels.

Residue-class stripe partitioning preserves the parent reduction tree exactly. Every partitioned half configuration compares all output words to its matching live parent arithmetic anchor. Changing the stripe count can change arithmetic relative to a different parent configuration; byte identity is claimed for matching anchors only. All selected winners use 32 stripes. FP32 controls preserve the archived FP32 output bytes.

The remaining 1.31–1.34× advantage reflects costs that do not halve with accumulator width: packed-weight loading and decoding, scale handling, addressing, activation conversion, partial output traffic, and launch/reduction work. Better scheduling helps both precisions. Reaching 1.4× against the improved FP32 denominator would require another roughly 4.5–6.5% reduction in FP16 time. No profiler attribution or universal 2× claim is made.

## Validation and accuracy limits

42 workers passed, covering 2,919 configuration checks, 42,764,874 output comparisons, 703,626 CPU dot-product checks, and 1,740 byte-identical parent-anchor checks. Four memory sanitizer runs, one synchronization check and one race check report zero errors. All four compiled revisions have zero register spills or stack frames. Exact expression-tree and group-coverage CPU checks passed. Source and binary hashes for every executed revision are retained in the current directory and `snapshots/`; all 544 sealed parent artifacts were verified unchanged.

Accuracy remains a production blocker. The large finite-range stress fixture produces 1,155/1,155 nonfinite outputs in each selected local FP16 kernel, matching its parent half anchor. Cancellation and subnormal fixtures remain finite but differ from FP32 (relative L2 approximately 0.000628 and 0.002635). These are diagnostics, not successful model qualification. No paired model PPL/KLD evaluation was run; the required ±0.003 PPL gate remains outstanding. This study does not deploy the approximate kernels.

## Reservation and reproducibility

Per the user's instruction to use unreserved devices, this run used GPU1 only, UUID `GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b`, under `/tmp/P100-arithmetic-gpu1.lock`. GPU3 work could proceed concurrently. The 1,038-sample monitor recorded GPU3 activity in 30 samples; these are not fully isolated host measurements. Fresh rotated workers and separate arms agree. No GPU reset or error retry occurred. GPU1 was released: 5 MiB, 0% utilization, 1328 MHz, zero volatile uncorrectable ECC errors at the final health sample.

`build.py` builds SM60 with CUDA 12.8, `--ftz=false`, and no host FP contraction. `control.py` runs bounded workers with a clean environment, CPU affinity 0–11, device UUID isolation, health checks, telemetry and provenance. It must be used only after checking current reservations and recording coordination. `final-jobs.json` and `extra-jobs.json` retain exact confirmation/stress requests. `analyze.py`, `report.py`, and `audit.py` regenerate summaries and verify retained evidence.
