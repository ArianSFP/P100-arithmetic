# W16A16 dual-rail AffinityWave plan — 2026-09-09

Test whether FP16-stored weights can match the fastest confirmed W8A16
FP16-storage pipeline without changing its arithmetic or dispatch structure.

The control is the frozen `compact-plan-dimensions` W8 winner. Preserve its
M128 pair kernel, M64 leftover kernel, two full-length HFMA2 accumulation rails,
FP32 final rail addition, dimension specialization, GPU tile compaction and
launch geometry. Replace only packed Q8 scale/code loads and exact paired-Q8
decoding with vectorized loads from a row-major FP16 weight stage.

Use the same signed-Q8-times-FP16-scale fixture, rounded once to FP16 when the
W16 storage is created. This makes the W8 decoded operands and W16 stored
operands identical. Require every W16 output word to match W8 and sampled CPU
half-FMA oracles before timing. Confirm M256 gate/up and down in three fresh
workers each. Include planner, pair and leftover launches in both arms. Run
partial/ragged memcheck and mixed pair/leftover racecheck. Static packing,
allocation and upload remain outside timing for both persistent formats.

This is a synthetic kernel comparison. It does not qualify native FP16 model
weights, model PPL/KLD, full-model throughput, decode, or a production change.
