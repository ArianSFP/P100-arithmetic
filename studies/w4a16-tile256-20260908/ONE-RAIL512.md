# 512-thread single-rail candidate — tested and closed

## Parent GPU3 result: correctness PASS, performance rejected

Parent integrated and compiled the frozen header in the long-prefill harness.
Observed compile resources:63registers,16KiB shared, zero stack/spill loads/stores.
This met the register target but did not improve the best compact pipeline.

Paired [gpu3-rail512-down-m256](../w4a16-long-prefill-20260908/gpu-results/gpu3-rail512-down-m256/stdout.txt):
M256/e64,N2048/K512,grid multiplier2 (112CTAs),7 rotated samples,8iterations,
2s warmup. Median prep=1 current Q4 T64=6821.979980us; rail512=6400.412110us,
**1.06587x current T64**. All activation conversion remains charged inside GEMM.
It clearly loses the parent's best compact down result~4960us, recorded in a
different worker; that cross-worker comparison is not a paired speedup estimate.
No1.5x gain, broader prompt-processing or full-model gain is established.

Companion parent artifacts, all result.json statuses PASS:

- [gpu3-rail512-down-scale](../w4a16-long-prefill-20260908/gpu-results/gpu3-rail512-down-scale/stdout.txt):
  hard-scale M65/e2,N2048/K512;266240 outputs bit-identical to current split
  reference,256 independent CPU samples, memcheck0errors.
- [gpu3-rail512-raw](../w4a16-long-prefill-20260908/gpu-results/gpu3-rail512-raw/stdout.txt):
  8320 raw-A32 rounding witnesses,M65/e2,N128/K64;16640 output words identical,
  256 CPU samples, memcheck0errors.
- [gpu3-rail512-race](../w4a16-long-prefill-20260908/gpu-results/gpu3-rail512-race/stdout.txt)
  and [gpu3-rail512-sync](../w4a16-long-prefill-20260908/gpu-results/gpu3-rail512-sync/stdout.txt):
  M65/e2,N128/K64 hard-scale cases;16640 outputs identical,256 CPU samples,
  racecheck/synccheck PASS with no hazards/errors.
- Timed down case also checks all33554432 output words unchanged/finite against
  current split reference and256 independent CPU samples, bad=0.

**Tested variant CLOSED.** Do not spend another grid/register sweep on it
unchanged. Header remains frozen. This update only records read-only inspection
of parent artifacts; no new code, compilation or GPU work by this subagent.

## Original implementation contract and CPU preparation

New files: one_rail512.cuh, test_one_rail512.py, ONE-RAIL512.md. No existing
header, parent study, coordination log or W4A4 file modified. In particular
direct_stage.cuh remains frozen at SHA256
a968022fcfd4ebec465d4f7aaca1755ff579ece8267b577c9267fb3e70248c9b.
No compiler, GPU, reservation or background process started.

Self-contained header, one bounded512thread/cap2 candidate:

    tile512_one_rail_q4<<<grid,512>>>(raw_a32,word_major_q4,out,M,N,K,experts)

Same uniform-expert interface,64x64 output tile, word-major Q4_0 input and
shape/alignment/nonaliasing contract as frozen direct staging. No extra global
scratch, persistent shadow, prepass or excluded activation-conversion work.

## Exact FP32 schedule

Sixteen warps: rail=warp/8, local_warp=warp%8. Each rail's8warps independently
cover64x64 outputs with4x4/thread. Each thread retains16 FP32 accumulators,
not32. Low owners consume K%32=0..15 across groups; high owners consume16..31,
in original ascending FMA order. Each thread still does FP32 multiply/FMA;
half arithmetic is restricted to the verbatim-body copy of verified weight
decode. A32 rounds RN toA16, then widens before any GEMM operation.

After ALL mainloop shared reads complete, low owners uniquely write4096 FP32
partials into a shared union. A block barrier precedes high-owner reads, then
high owners perform add_rn(low,high), in that argument order, and write valid
outputs. No shuffle tree, atomic reduction, group-scale movement, swapped
rounding sequence or half partial sum. Float stores/loads preserve partial bits;
GPU equivalence must still be checked against current T64, not merely against
another experimental kernel.

## Producers and synchronization

All512threads load one A float4: token row=tid/8, Koffset=(tid%8)*4. Tail A rows
zero-fill. Only first256threads load the256 weight words/64 scales into full
32x64 shared B; remaining256 do not duplicate decode. Weight decode loop stays
rolled, immediately storing each decoded pair as in direct staging. The B
producer predicate is warp-uniform and contains no block barrier.

A32x64+B32x64 float staging occupies16KiB. The union aliases it with4096 float
low partials (also16KiB), explicitly16byte-aligned. Barrier immediately before
union reuse prevents low writes clobbering any high/low mainloop readers.
Barrier after low writes protects high reads. Final barrier after output protects
the next persistent tile's producers. Two uniform barriers/G32 plus3 tile-end
barriers =2G+3, two more than direct-stage's2G+1. Shared merge adds16KiB stores
and16KiB reads/tile. Only high owners write global outputs; low owners also
initialize shared words for padded rows, avoiding partial initialization.

## Resource hypothesis and admission gates

Target<=64registers, no stack/spills, two blocks/SM (32warps).32rail accumulators
per direct256thread become16/thread here, but total CTA accumulator storage is
unchanged:512x16=256x32=8192registers. More threads duplicate pointer/address
state; only two output tiles can reside at the proposed cap versus four direct
tiles. B producers are imbalanced, and reduction/barrier costs matter especially
for down K512. No register/occupancy/speed advantage is claimed before compilation.

CPU test verifies copied decoder body, frozen direct header, all A/B staging
owners, both output-rail maps, one low writer/one high reader per shared word,
tail/vector/global bounds, rail ordering and union barriers. Parent must gate
compile resources/SASS (HFMA2 forbidden; HMUL2/HADD2 only verified weight decode),
hard-scale/raw-A32 exact oracle with NaN-poisoned outputs, and memcheck/racecheck/
synccheck before paired performance promotion. Keep fresh current T64 plus
compact/direct-stage controls; all pipeline costs included. Equal expert screens
do not establish real2k/4k/8k prompt or full-model gains.

    python3 studies/w4a16-tile256-20260908/test_one_rail512.py
