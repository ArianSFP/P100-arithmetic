# Mixed-precision T64 experiment: parked, not a qualifying 2x result

User correction, 2026-09-08: the target is 2x T64 **independently of FP16
accumulation**. All R1/R2/R4/R8 half-rail gains below are a separate precision
lever and are explicitly excluded from the current target. No production edits.

The paired GPU2 synthetic gate/up test used T64 M64 split-K2, 64 tokens,
512 outputs, K2048, 16 experts, identical Q4-representable codes/scales and
half-representable activations. It is neither a full model benchmark nor a
comparison against the newer compressed Q4 T64 model implementation.

| Worker | FP32 T64 control, kernel us | Half-rail candidate, kernel us | Kernel ratio | Ratio including activation preparation |
| --- | ---: | ---: | ---: | ---: |
| grid5-r1 | 586.347 | 297.819 | 1.969x | 1.808x |
| grid5-r2 | 587.365 | 316.936 | 1.853x | 1.708x |

These are within-worker median ratios, not pooled across workers. Clocks ramped
during the short runs; the warmup was insufficient for a stabilized final
performance claim. Neither measured pipeline reached 2x even with half rails.
Uniform synthetic relative-L2 errors against the FP32 control were 0.00082827
and 0.00059922 respectively; these do not establish model/NVFP4 quality.

Original and optimized smoke checks passed. The V2 all-config memcheck passed
with zero errors. Numerical CPU oracle checks sampled 256 output elements per
configuration; all GPU outputs were checked for non-finites and difference
statistics. No model accuracy, outlier-pattern, racecheck or synccheck campaign
was completed. No outlier fallback is implemented. The down-v1 timing overlapped
a CPU compile and is exploratory only. Raw commands, binary hashes, health,
outputs and immutable source/build snapshots remain in gpu-results/ and
snapshots/. Reproduce summaries with analyze.py; do not reuse the released claim.

GPU2 and the shared lock were released at 18:29 UTC, with 5 MiB idle memory,
zero utilization/ECC errors, and no GPU2 compute clients. No reset or fault.
Further work moves to ../w4a16-t64-f32-20260908/ with FP32 held fixed.
