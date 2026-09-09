# W16A16 dual-rail AffinityWave, F32 wire — 2026-09-09

Test the FP16-weight kernel against the fastest confirmed W8A16 serving input
path, which receives F32 activations and rounds them to FP16 inside the kernel.
Preserve the W8 winner's M128 pairing, dual HFMA2 rails, dimension
specialization, GPU compaction, leftovers and launch geometry. Replace only
Q8 weight loading/decoding with vectorized FP16 weight loads.

Use identical F32 activation storage for W8 and W16 and identical FP16 weight
operands. Require full-output bit identity and sampled CPU half-FMA checks.
Confirm M256 gate/up and down with three fresh workers per projection, including
all planner and compute launches. This is an isolated synthetic comparison;
native model accuracy and full-model throughput remain separate gates.
