# Decode FP16 investigation, 2026-09-08

Question: can FP16 accumulation approach twice the speed of tuned FP32 at decode shapes?

Existing Q8 MMVQ/half2 and historical DMMV failures are closed. Read the seven-refutation memory and phase4y/DMMV raw reports before starting. Do not rebuild those paths. They establish no existing reliable 2x decode gain.

New evidence permitting this bounded test: September W4A16 round-3 has a different compressed, row-coalesced layout, FP16 activations (no Q8 activation quantization), and exact-order CTA reductions. Test half2 accumulation in THAT current mapping, alongside its archived current winners and a matched new FP32 path. Never attribute layout/geometry gains to accumulation alone.

Inputs: Q4_0 codes/scales losslessly repacked into the exact existing 18B/32 weight format; unchanged FP16 activation bits. M/K square 5120/5120, up 4352/5120, down 5120/4352, tall 17408/5120 and long K 5120/17408 as useful; N1 single-token and N4/N8 batched decode, not 2048-token GEMMs. Context length does not turn single-request decode into a 2048-column GEMM.

Candidates: packed pairs with exact nibble-to-half conversion; FP32 sequential group dot, FP16 two-lane group dot widened before FP32 group scaling/accumulation, FP16 group dot plus FP16 local scaled accumulation. All variants produce FP32 output and use a FP32 CTA reduction. Sweep row/warp/split geometry. FP16 local accumulation is explicitly distinguished from full-FP16 global reduction.

Correctness: independently emulate every arithmetic operation on CPU for sampled complete output dots, compare every GPU output against current FP32 baseline and record changed outputs, nonfinite outputs, absolute/relative L2 error. Exact same-order FP32 configurations must be byte-identical; changed-order/FP16 candidates are experimental and cannot pass project adoption without +/-0.003 paired PPL/KLD. Include rounding/range/cancellation fixtures. No inference-quality claim from small synthetic error.

Performance: warmup, rotated repeated event measurements, multiple fresh workers and seeds for selected controls. Include all per-step kernels and FP32 output/reduction. One-time lossless weight repack/allocation/H2D excluded equally; activations already FP16 equally. No comparison of legacy timings from a different worker as denominator. Inspect SASS/registers and use memory/synchronization checks on bounded tail fixtures. Stop on CUDA errors/timeouts; no reset/retry.

GPU1 by UUID only, shared global lock held across builds and bounded workers; fresh controller reservation, health/ECC/desktop-log/disk checks. No edits outside this study and mixed coordination record. No production/model or git publication action implied.
