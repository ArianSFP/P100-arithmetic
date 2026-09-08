# Direct staging and contiguous mapping, 2026-09-08

The 1.5x goal remains unmet. All GEMM products and accumulation are FP32.
These are equal-M primitive screens, including recurring activation conversion;
they do not establish whole-prompt or current-Q4 ragged performance.

Weight layout contract: candidate word-major Q4 repacking and upload occur
before events. They are excluded from these primitive times. A complete
pipeline result must either verify persistent immutable repacking or charge
repacking/copies on cache misses. Descriptor construction and routing are also
outside this timing boundary. Independent audit: ../w4a16-verification-20260908/
DIRECT-STAGE-REVIEW.md.

Seven within-worker medians, eight iterations/repetition, M256, 64 experts:

| Candidate | Projection N,K | Current T64 us | Candidate us | Ratio |
| --- | --- | ---: | ---: | ---: |
| compact_down_spec | 2048,512 | 6816.668 | 4966.460 | 1.3725 |
| compact_vstore | 2048,512 | 6797.632 | 4982.160 | 1.3644 |
| compact_gu_spec | 512,2048 | 7043.812 | 4953.120 | 1.4221 |
| compact_down_chunk | 2048,512 | 6801.088 | 4956.668 | 1.3721 |
| compact_gu_chunk | 512,2048 | 7050.084 | 4949.500 | 1.4244 |
| tile256_direct_stage | 2048,512 | 6801.300 | 5825.664 | 1.1675 |
| tile512_one_rail | 2048,512 | 6821.980 | 6400.412 | 1.0659 |
| compact_down_cluster | 2048,512 | 6811.884 | 5015.188 | 1.3583 |

Tags: gpu3-down-spec-m256, gpu3-vstore-down-m256, gpu3-gu-spec-m256,
gpu3-down-chunk-m256, gpu3-gu-chunk-m256-r7, gpu3-direct-down-m256.
Reproduce exact summaries using analyze.py with those tags.
Additional tags: gpu3-rail512-down-m256, gpu3-cluster-down-m256.

Rail512 splits low/high rail ownership between thread groups and merges in
16KiB shared storage after the final group. 63 registers, zero stack/spills;
grid2. Hard-scale down M65 has266240 identical output words and256 CPU samples.
Raw-A32, racecheck and synccheck M65/N128/K64/e2 also pass. Tags
gpu3-rail512-down-scale, gpu3-rail512-raw, gpu3-rail512-race, gpu3-rail512-sync.
Close this tested implementation: it loses both compact and the fastest legacy
control. Resource efficiency alone did not produce a throughput improvement.

The separate 2x2 cluster ordering (MAP2, not contiguous MAP1) compiles91regs,
zero spills,12KiBshared,grid5. CPU coverage includes odd tile-row counts;
gpu3-cluster-down-scale checks M65/N2048/K512/e2 with full266240 output identity
and clean memcheck. Down screen loses~1% against specialized/contiguous compact;
close this tested mapping rather than expanding an unpromising grid sweep.

Contiguous mapping changes block-strided jobs to balanced contiguous ranges.
The latency changes are under 0.3% on both projections. Close this specific
mapping lead; it does not close untested 2D cluster mappings.

Direct staging uses rolled immediate load/convert/store producers, eliminates
future-group payloads, and reduces tile256 to 64 registers, zero stack/spills,
16KiB shared. This meets the static four-block residency resource target.
No runtime occupancy value was logged for this candidate. It loses clearly
despite passing the resource gate; do not sweep unchanged grids or report a win.
Loss of prefetch and extra producer-loop work are plausible causes, not measured
hardware-counter conclusions. Direct grid=4; compact grid=5; current T64 retains
its original 112-CTA grid. The older fused control in the direct run shares
grid=4, so it is not a fresh same-grid comparison with compact grid=5.

Correctness: direct down M65/N2048/K512/e2 hard scales matches all 266240
outputs, 256 CPU oracle samples, memcheck zero. GU raw-A32 rounding witnesses
M65/N512/K2048/e2 match all 66560 outputs, memcheck zero. M65/N128/K64/e2
racecheck and synccheck pass. GPU result tags gpu3-direct-down-scale-m65,
gpu3-direct-gu-raw-m65, gpu3-direct-race-m65, gpu3-direct-sync-m65.
The raw-A32 test compares the candidate against the equivalent rounded-A16
reference; it does not feed the raw witness buffer into current T64 itself.
Sanitizer/raw-witness times are excluded by analyze.py.

Operational record: r5 stopped on an invalid CLI before CUDA initialization;
r6 exited normally on input EOF without running a worker. Both released with
GPU3 idle5MiB/ECC0. r7 reacquired with persistent input; no GPU fault/reset.
Another session announced GPU1-specific concurrent decode work during r7.
Treat performance as screening; replicate any promoted candidate with fresh
paired controls and recorded contention. No 1.5x or model-quality claim.

r7 completed and released normally: release-1788906978930344770.json,
GPU3 idle5MiB,ECC0,no compute clients,stopped=false. Both coordination logs
record final release. No GPU worker remains in this study.
