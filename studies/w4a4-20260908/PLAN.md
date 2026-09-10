# W4A4 on SM60 — bounded arithmetic study

Scope: signed integer W4/A4 (not FP4), Q4_0 codes and half weight scales,
32-element common scale groups. Preserve the sealed archive and other studies.
User approved GPU3 execution on 2026-09-08; reservation token
w4a4-20260908-1514-gpu3 is held through final checks.
GPU1 and GPU2 are reserved by independent sessions. No production integration.

Hypothesis: after activation quantization, a G32 integer dot has every prefix
bounded by 32*64=2048. All integers in [-2048,2048] are exactly representable
in binary16. Thus documented HFMA2 can accumulate two independent K stripes
exactly, followed by exact FP32 lane addition. Unlike W4A16/Q8 attempts, this
requires no intermediate flush inside the group. It is new numerical evidence,
not a reopening of the refuted Q8 or arbitrary-FP16 substitution experiments.

Compare paired-K HFMA2 with a matched warp-sharing FP32 kernel, inheriting
round-2 weight layout, row/batch reuse, scale FMA and S32 final tree. Include
FP16-to-A4 quantization, packed activation storage, scales, and final reduction
in timing. Repacking weights is offline and excluded consistently. Include a
live sealed W4A16 round-2 control on the original inputs, explicitly recognizing
that activation quantization changes its numerical problem.

Correctness: exact integer CPU oracle for each group; identical scaled FP32
results for W4A4 candidates. Quantizer independently checked. Endpoint,
cancellation, zero, tail and random cases precede timings. Quantization error
against A16 is reported separately; no synthetic error metric establishes the
production +/-0.003 perplexity requirement using KLD-pair methodology.

Only compiler-generated documented instructions, no binary patches or resets.
If GPU access is approved, reserve by UUID, hold across gaps, fresh worker per
run, fail closed on health/process/CUDA/correctness errors; freeze candidates
before fresh-worker repetitions. Record all observations, including losses.

## Adaptive follow-up after first two sweeps

Initial half2 candidate passed all hardware arithmetic/quantizer checks and
memcheck/synccheck, but square N1 lost (~61 us vs ~43 us FP32 and ~41 us A16).
Preserve initial-build artifacts and raw sweeps. Test signed bit-plane POPC
(H=2): lossless four-plane weight repack and quantizer-generated A4 planes,
16 AND/POPC terms per G32 dot with coefficients [1,2,4,-8] on each operand.
Include ballot/plane preparation in all matched W4A4 pipeline timings. This
changes the quantizer relative to the initial sweep; gains use live controls.
No numerical rounding inside the integer dot; same scale FMA and S32 tree.

## Final bounded sweep

The initial bit-plane R4/N1 kernel nearly ties R2 A16 (~41 us) and improves
matched A4 FP32 only modestly. Geometry changes alone did not improve square N1.
Use per-format quantization (only needed nibbles or planes are written), then
try the independently completed W4A16 round-3 shared reduction idea for both
FP32 A4 and bit-plane A4. Keep S32 arithmetic order. Add a copied sealed R3
A16 cubin as an additional live control, checked byte-identical to R2. Include
its 4.2-7.2% N1 improvement when interpreting current-baseline speed claims.
CPU/compilation and 55-candidate hardware suite are green before this sweep.
