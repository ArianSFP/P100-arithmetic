# FIGLUT symmetry follow-up status — paused 2026-09-09

The final bounded experiment is complete.  `P100-FIGLU-experiment` does not
provide a credible route to a complete W4A4 kernel that is 1.3x faster than the
confirmed W16A16 baseline on GP100.

- W16 1.3x targets: 1894.23 us gate/up and 1810.74 us down.
- Literal package mapping: W4A16 scalar-FP32 LUT, not W4A4 and not a measured
  GPU kernel.
- Exact weight-derived half128 transfer: proved, but only 0.503--0.521x matched
  resident half2 in the prebuilt-table gate.
- Fresh full256 R16 ceiling: 1.3236x before mandatory construction and finish.
- Symmetry-built full256: 4 GiB mandatory shared stores/projection; in the
  direct sequential schedule, even an ideal bank-peak store estimate consumes
  essentially all 1.3x headroom.
- Exact activation-derived packed-four full16: 2.9958 TMAC/s versus the
  9.4878 TMAC/s down target; 0.4854x its corrected half2 component control.
- Exactness: package CPU checks pass; half128 packed proof passes; 320,144
  packed-four CPU dots pass; 8,192 GPU outputs match the direct definition bit
  for bit.
- Complete GEMM and PPL/KLD: not run because no candidate passed its component
  admission gate.
- Production/model source: unchanged.  No commit, push, PR or reset.
- GPU1: reservation released in both coordination logs after final health
  check.

Read `RESULTS.md`, `AUDIT.md` and `activation_audit.md` before resuming.  Do not
repeat the half128 consumer or packed-four full16/half8 paths without materially
new evidence.  A new W4A4 idea should first exceed 10.25 useful TMAC/s in a
streamed exact-G32 component with runtime operands and live totals.
