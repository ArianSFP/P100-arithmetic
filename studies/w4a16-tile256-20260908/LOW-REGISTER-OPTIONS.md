# Conditional next options after cap4 liveness results

CPU-only review. Existing tile256.cuh and liveness.cuh are unchanged and
hash-checked by test_lower_register_models.py. Parent GPU3 timing/compilation
queue remains exclusively parent-controlled. No compiler, CUDA, reservation,
parent-file edit or W4A4 work performed. **These are two design proposals with
CPU models, not new GPU kernel artifacts. Wait for the current two liveness
variants before implementing either.**

## What the existing source can and cannot remove

The cap4 problem is measured: original U32 needs72bytes stack and76bytes spill
loads/stores. Cap3 is80registers/no spills but roughly5.023ms gate/up, no better
than compact~5.013ms. The current no-prefetch variant removes future operands
across compute but still constructs both raw float4s before staging. Half2
prefetch retains four packed registers but temporarily constructs both raw
float4s before conversions. Both still hold32 persistent FP32 accumulators and
eight shared-memory operand values at a time; fully unrolled scheduling can
increase compiler liveness beyond these source counts. Neither lexical scope
nor launch_bounds proves physical-register reuse. Await actual compiler data.

## Proposal1: direct streamed shared producer,256threads/cap4

Changed premise versus current no-prefetch: remove pa0/pa1/pha/fetch captures
entirely, not merely move fetch after compute. At each group:

    barrier (prior shared readers finished)
    load one A float4 -> RN A16 -> widen -> immediately store its4 shared cells
    load second A float4 -> same immediate conversion/store
    load one B word/scale -> decode/store each paired byte immediately
    barrier (complete stage)
    compute unchanged two FP32 rails

Tail inputs remain zero-filled, current64x64/256thread map unchanged. No global
prefetch overlap is claimed. Peak source-level raw-A temporaries fall from8 to4,
and no future operand arrays exist across mainloop. Conversion is still in the
GEMM; there is no extra global scratch or launch. Compare directly to existing
no-prefetch, so an apparent win is not incorrectly attributed to the tile.

This is a modest, conditional change: ptxas can still hoist/reorder loads and
decodes, and fully unrolled compute may dominate register pressure. Do not add
volatile traffic or device-function calls just to force a lexical lifetime.
Gate is actual<=64registers with zero spills plus retained vector loads and
unchanged precision. If the existing no-prefetch variant already meets that
gate and performs well, skip this near-neighbor unless its SASS exposes a
specific remaining producer lifetime problem.

## Proposal2:512thread64x64 tile, one FP32 rail/thread,cap2

Material change: distribute rail ownership, rather than trying to squeeze both
rails into every thread. Keep64x64 outputs, but use512threads (16warps):

    rail=warp/8; local_warp=warp%8
    ar=(local_warp/2)*16+(lane/8)*4
    bc=(local_warp%2)*32+(lane%8)*4

Each thread owns4x4 outputs for ONE persistent FP32 rail (16 accumulator
registers). Rail0 consumes K%32=0..15 across every G32; rail1 consumes16..31,
in original order. All512threads load one A float4 each: row=tid/8,
Kstart=(tid%8)*4. First256threads load/decode the256 B words into the full
32x64 shared B stage; the other256 do not duplicate weight decode. Producer
participation is warp-uniform; all threads reach each barrier.

At completion, after all mainloop consumers finish, reuse a shared-memory
union: original A+B stage is16KiB and4096 FP32 low partials are also16KiB.
Low-rail threads uniquely store their4096 partials, block barrier, then high-rail
threads read low and perform EXACTLY add_rn(low,high) before valid output stores.
Final barrier precedes the next persistent tile's staging. No atomics,
cross-CTA reduction, half accumulation or changed group scaling. Low/high
thread separation changes ownership, not either FMA sequence or final add order.

Target two blocks/SM at<=64registers (32 resident warps). That register budget
is no larger per thread than the256/cap4 goal, but16 rather than32 accumulator
registers leave substantially more room for address/stage temporaries. Total
CTA accumulator registers remain8192, not halved; additional threads duplicate
address state. Only two64x64 tiles/SM versus the intended four256-thread tiles,
so this is not a guaranteed occupancy/performance win. It adds4096 shared stores
and4096 shared loads per tile plus reduction barriers. That penalty is
particularly relevant to down K512. Test it only if actual low-register
headroom justifies the added reduction/producer imbalance costs.

This differs from previously failed larger128thread output tiles (their
accumulators/thread increased) and from the old reduced-token M16 schedule.
It is not permission to rerun those closed variants. It preserves exact current
split32 numerics; independent hardware identity remains mandatory.

## Separate A16 prepass: not admitted as a third candidate

Evidence against repeating it unchanged: delivery-leads RESULTS records tiled
A16/widened-A16 kernel-only wins erased by transpose; vector preparation saved
only1–2us and fused conversion closed the earlier gate/up gap. Long-prefill
temporary weight staging likewise lost after~720us recurring preparation.
These are different scales/representations and cannot be used as measured
latency for a new activation prepass, but they rule out assuming prep is free.

At M256/e64, the minimum A32-read+A16-write traffic is192MiB gate/up
(128+64MiB) and48MiB down(32+16MiB), plus a launch and full scratch residency.
For N64 output tiles, logical activation GEMM reads would fall from1024MiB
raw A32 to512MiB A16 per projection, before caches. After charging preparation,
the ideal logical read/write reduction is320MiB gate/up and464MiB down.
These are SOURCE BYTE COUNTS, not measured HBM traffic or a throughput forecast;
existing cross-tile cache reuse can erase much of the apparent saving.

Necessary gate for any later prepass proposal is

    timed_prepass + GEMM_from_A16 <= paired_current_T64 / 1.5

and it must beat the best fused compact pipeline, not the spilling cap4 control.
With illustrative measured gate/up values~7.05ms T64/~5.013ms compact,
required total is<=4.70ms. Thus the A16 GEMM must save at least~0.313ms PLUS
the ENTIRE measured prepass cost relative to compact. At an optimistic assumed
500GB/s effective conversion traffic rate,192MiB alone costs~0.403ms, implying
~0.716ms GEMM savings before launch/other overhead. That bandwidth is a scenario,
not a measurement of this kernel. Prepacking cannot be justified by register
pressure alone while same-kernel immediate conversion remains untested.

No full ping-pong32x64 A+B staging is proposed: double buffering requires32KiB
shared and caps residency at two blocks by shared memory, directly contradicting
the256/cap4 hypothesis. A half-K ping-pong design adds barrier/stage complexity
and resembles the historically losing smaller-Kstage axis; no new evidence
currently warrants reopening it.

CPU check (independent ownership/address/order model, not imported GPU code):

    python3 studies/w4a16-tile256-20260908/test_lower_register_models.py

Any implemented winner still needs no-spill/SASS gates, NaN-poisoned exact
current-control oracles (raw A32/hard scales/tails), sanitizers and all-cost
paired GU/down timing. No primitive result establishes2k/4k/8k workload or
full-model acceptance. W4A4 remains paused.
