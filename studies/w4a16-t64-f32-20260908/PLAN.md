# Independent W4A16 / T64 FP32 optimization

## Next milestone: 1.5x including activation preparation

User follow-up: target1.5x complete-pipeline performance next. This is the next
milestone toward the earlier2x ambition, not permission to change accumulation
precision or omit activation work. FP32 products/accumulation and A16 values
remain fixed. The completed first sweep's RESULTS.md remains historical.

Using the paired confirmation measurements as planning numbers:

| Shape | Current pipeline us | Control pipeline us | 1.5x target us | Further latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Gate/up | 440.805 | 534.488 | 356.325 | 19.2% |
| Down | 380.072 | 466.357 | 310.905 | 18.2% |

Acceptance uses fresh interleaved control timings, not these historical absolute
thresholds. All per-call preparation, conversion, transposition and output
reduction must be inside the timed pipeline. No overlap/double-counting with
the separate model T64 packing gain. Model-level1.5x is a separate claim.

First proposed experiment: fuse activation conversion with a lossless tiled
A16 transpose, then consume that layout using coalesced vector loads and
contiguous shared-memory staging. Compare against both the frozen row-major
winner and unchanged T64 controls; include the transpose cost and scratch.
This tests whether preparation can save repeated operand-delivery work across
output tiles. It is not yet implemented or measured. Merely deleting the
preparation pass would leave gate/up at about1.29x, so this must also improve
the GEMM itself. Preserve exact activation bits and the winner's FP32 K order.

Before any promotion, compare actual compressed T64 consumers on real expert
inputs, matching weight preparation and Q4_1 handling, and pass model accuracy.
Do not resurrect the closed generic LUT/tile sweeps without a new hypothesis.
No GPU is reserved for this planning update; new tests need a fresh claim.

## Contract

The user's 2026-09-08 correction requires 2x T64 independently of changing
FP32 accumulations to FP16. Every timed candidate in this study must use FP32
products and accumulations. Original FP16 activation bits remain unchanged.
Half-rail results are preserved separately and cannot count toward this target.
No hidden instruction encoding, model replacement or production integration.

Compare identical shapes, codes/scales, FP16-valued inputs, output requirements
and GPU conditions. Include original M64 split-K2 T64 and its existing A16
input specialization; report against the faster control. Q4 uses the same
representable codes as the Q8-layout control to isolate layout/kernel effects,
not a new Q8-to-Q4 model-quality claim. This historical primitive is not the
newer full Q4 CohortRail service: a win here must subsequently beat that control
on real layers/model before a production 2x claim.

## First bounded experiment

Existing naive FP32 R0 repeats widening and group reduction in every compute
thread. Stage FP32 activations and scaled Q4 weights once per CTA, then reuse
them with vector shared loads and FP32 outer-product FMAs. Sweep register tile,
CTA size, stage buffering and persistent grid. Keep compressed weights at
18 bytes per group of 32. No globally expanded-weight performance claim.

Also build a Q4-storage specialization of the exact T64 schedule, so compression
alone is distinguishable from mapping/scheduling gains. Charge any per-call
activation packing in separate complete-pipeline timings. Weight repacking is
offline/static in this microbenchmark and must be charged/amortized explicitly
in any model experiment.

First require CPU-oracle checks, SASS audit (no HFMA2), and sanitizer smoke.
Use same-process interleaved controls after a sustained warmup; preserve all
timing samples. Report gate/up and down, including token-tail shapes. Arithmetic
order changes still require the applicable model KLD/PPL gate (historical
PPL delta +/-0.003) before integration; no claim of NVFP4 equivalence without
a specified, matched NVFP4 comparison.

## Coordination

CPU preparation while the queued qwen35-q4-t64-20260908 session uses the GPUs.
Do not edit its source or results. Claim one GPU only after coordination/health
checks and hold the shared lock across the test suite; fresh supervised workers,
timeout, desktop/disk guards, stop on faults, no resets. No commit/push requested.

If operand-delivery gains plateau below 2x, record that result rather than
switching precision or selecting a weak control. Only revisit old LUT or other
closed paths with a materially new reuse/mapping hypothesis and its own controls.
