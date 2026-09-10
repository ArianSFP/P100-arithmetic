2026-09-09: Study opened at user request. GPU1 reserved. Suggestions are being
evaluated in explicit priority order 1 > 2 > 3; prior bit-plane/popcount and
long-prefill mappings are treated as negative controls rather than retried.

2026-09-09 priority-1 lowering: CUDA 12.8 / sm_60 lowers each explicit
`mad.wide.s16` to one `XMAD.S16.S16`; the supplied packed-FP16 baseline emits
32 `HFMA2`.  The level-2 supplied probes emit two `HFMA2` plus three `HADD2`
for normalized `split4_mac`, versus two `HFMA2` for the same four useful MACs
in the ordinary control.  The one-multiply repair emits one `HMUL2`, two F2I
conversions and a substantial integer repair sequence.  None of these compile
probes spills.

2026-09-09 priority-1 hardware ceiling, GPU1: 112 blocks x 256 threads,
65,536 iterations, two-second warmup, nine rotated rounds, three launches per
event; rep0 excluded.  Median useful throughput was 6.7355 TMAC/s for HFMA2,
7.0931 for centered signed-16 XMAD pack2 (1.0531x), 7.0825 for FP32 pack2
(1.0515x), 8.7493 for FP64 pack4 (1.2990x), and 9.1283 for the mixed
HFMA2/FP64 stream (1.3553x).  These deliberately omit decode, extraction,
scales, quantization, planning and output.  The best ideal stream is therefore
already below 2x; do not build packed/mixed full GEMMs without new evidence.

2026-09-09 priority-3 optimistic LUT ceiling, GPU1: `table[256][32]` occupies
32 KiB and SASS has exactly one `LDS.U.32` per lookup.  The original R16/two-CTA
form reaches the 128-register bound and spills 216 store + 220 load bytes per
thread; it runs at only 0.8682x its matched resident half2 control.  A refined
warp-uniform, slice-major screen gives: R16/one-CTA, 243 registers, no spills,
1.3609x; R8/two-CTA, 109 registers, no spills, 1.1090x; R4/two-CTA, 64
registers, no spills, 1.0014x.  This is a generous lower-bound workload: the
table is prebuilt and resident, packed byte fields are allowed to wrap, and
mask preparation, resets, byte widening, plane combination, correction,
per-G32 scaling and all four table builds/eight barriers are absent.  Since
even the best such consumer misses 2x, a complete weight-derived LUT cannot
meet the stated target on P100.  Narrower R8/R4 forms also multiply table-build
cost per output by two/four while the 32 KiB allocation still caps occupancy at
two CTAs/SM.

2026-09-09 priority-2 hardware, GPU1: exact cancellation passes for all output
threads.  At 64 useful scalar MACs/thread-iteration, the matched plain stream is
5.6118 TMAC/s.  Normalized split is 3.2233 TMAC/s (0.5744x) and HMUL2 plus
amortized PRMT/mod-8 repair is 0.8669 TMAC/s (0.1545x).  Resources are 64, 56
and 121 registers respectively, all spill-free.  Ptxas charges twice as many
in-loop half-word merges to the plain stream, so the candidate comparison is
already generous.

2026-09-09 complete exact-G32 half2 result, GPU1: M33 semantic, M33 memcheck
and M129 mixed-pair/leftover racecheck pass.  At M256/e64 with the selected
112-block grid, gate/up is W16 2459.97 us versus W4 4522.31 us (W4 speed
0.5440x); down is W16 2348.57 us versus W4 3629.11 us (0.6471x).  Every large
activation code/scale, planner coverage, output finiteness and 256 CPU dots per
arm pass.  Static W4 is 144 registers/34,816 B shared/zero spills, retains all
1,024 HFMA2 and adds 64 FFMA scale updates plus the timed A4 quantizer.

2026-09-09 PAUSED by explicit user request.  No new experiment may start from
this session.  Full checkpoint is in RESULTS.md and handoff state in STATUS.md.
GPU1 reservation is being released; no model/production edit, commit, push or
reset occurred.
