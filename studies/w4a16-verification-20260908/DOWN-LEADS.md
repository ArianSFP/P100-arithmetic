# Ranked remaining W4A16 down/M128–256 leads

CPU/read-only analysis2026-09-08. No compiler, GPU, reservation or edits outside
verification. W4A4 excluded. Rankings are hypotheses, NOT measured speedups.
Use current source/results, not stale historical NEXT paragraphs in notes.

## Exact remaining gap

Paired medians re-extracted from PASS raw stdout via inspect_down.py:

| Shape,64experts | Current T64 us | Capped compact us | Required us for1.5x | Required latency reduction |
| --- | ---: | ---: | ---: | ---: |
| GU M128,N512,K2048 |3605.843990|2533.808110|2403.895993|5.13%|
| Down M128,N2048,K512 |3448.323970|2585.743900|2298.882647|11.09%|
| GU M256,N512,K2048 |7046.812010|5012.711910|4697.874673|6.28%|
| Down M256,N2048,K512 |6814.372070|5119.140140|4542.914713|11.26%|

Tile256 cap3 is NOT a new win: paired M256 GU5024.436us (1.404x),
down5164.656us (1.324x), comparable/slower than capped compact. Do not rerun
unchanged cap3/compact, wide64x128 or mid32x128 expecting an unexplained win.
GU M64 crosses1.5x but is not the unresolved large-work/complete-pipeline gate.

## What the evidence says about the bottleneck

The strongest measured intervention was occupancy/compiler shaping: compact
103->95regs, fixed12KiBshared, five128-thread blocks/SM, no spills. GU dropped
6346->5013us; down reached5119us. That intervention already banked a large gain.
No hardware counter result proves the current remaining stall type; the rig's
counter limitation remains in shared memory. Source/static SASS is hypothesis
evidence, not proof of instruction issue throughput or DRAM transactions.

For compact32x64, M256/e64, equal usefulFLOPs:

- GU:4096 jobs x64 K32 stages; down:16384 jobs x16 stages. Both262144 job-stages.
- Both execute the same mainloop body per stage. Down pays job setup/epilogue
  FOUR times as often, and emits128MiB of FP32 output versus32MiB for GU.
- Source-level A loads are1GiB in each case before cache reuse; logical compressed
  B tiles sum288MiB each. Scale loads are replicated four times per64-row tile,
  so source-requested B bytes are384MiB (not a measured DRAM count).
- Down's activation working set32MiB vsGU128MiB offers stronger possible reuse
  across N, but current col-fast job order already intentionally favors A reuse.
  An M-fast change may favor weights while hurting A; it is not automatically good.

Existing SASS SHA2566292bb32062ccc649d9124099117349c2df1b78ff6c9fc9d71e8cdba8aad42b4:

| Kernel | Static K-loop backward interval | FFMA | LDS.128 | STS | HMUL2 | BAR | Whole function |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact packedU32 |750|512|64|24|8|2|1350|
| tile256 cap3 |702|512|64|16|4|2|1248|

Both have16 scalar STG.E and16 final FADD instructions per thread in the
epilogue. Compact's per-job backward interval is1278 static instructions,
giving approximately528 instructions outside its750-instruction K interval.
At down16groups that is only~4.2% of the simple issue-count model, vs~1.1%
atGU64groups. This is NOT dynamic count measurement: predication, taken paths,
load stalls and setup-loop code affect actual cost. Still, it argues AGAINST
claiming runtime division removal alone will buy15%.

Current T64 static function4440 instructions includes Q4_0/Q4_1/Q8 alternatives;
its K interval2964/1024FFMA is NOT an apples-to-apples executed Q4 instruction
count. Do not use those totals to promise3x. Paired timings are the denominator.

## Ranked leads and bounded tests

### 1. Finish the already-prepared tile256 cap4 liveness falsification

Highest evidence-backed chance of>=10–15%: source redesign intended to move
tile256 from3 to4 resident256-thread blocks (24->32warps) without spills, while
retaining its smaller staging/decoding count per output. This is the pending
no-prefetch/half-register-prefetch work, not a request to repeat closed cap3.
The cap3 tile's same~5ms time despite reduced static staging supports a
latency/residency limitation as a plausible explanation, not established fact.

Exact plan: after owner build, require actual occupancy4 and spill-free SASS,
not launch_bounds alone. Each variant independently passes scale-stress/raw-A32
identity and sanitizers. Screen paired M256 down thenGU, e64, grid4 (224CTAs),
7reps x8iters,2s warmup. Retain compact grid5 and current T64 grid112 as controls;
do not force all kernels to the new grid. If neither saves>=5% vscompact down,
close this mechanism instead of sweeping every grid/unroll. If promising,
confirm M128 down/GU and three fresh workers. Required down target~4543us using
historical M256 denominator, recalculated per fresh control.

### 2. Down-specific epilogue and shape/address specialization, SAME winning tile

Most useful distinct next code lead after liveness. Combine two separately
attributed variants before considering a composed result:

A. Vectorize output4 contiguous columns/thread into float4 stores. Current16
STG.E ->4 STG.E.128 per thread, same __fadd_rn per result. Source scalar stores
have stride4 across adjacent participating lanes, hence sparse sector use per
instruction; vector stores may reduce requests. Cache/store merging may already
hide much of it. Output128MiB atdown gives a plausible few-percent opportunity;
10–15% is possible only if store/request stalls are large, not from12 fewer
instructions alone. No out-of-range tail writes; N multiples64 preserve alignment.

B. Template K512/N2048 while retaining runtime ragged M, and use invariant stage
base +1152-byte pointer recurrence instead of recomputing wide multiply/divide
expressions per group. Optional M128/256 specialization is DIAGNOSTIC for setup
headroom only, not the final ragged dispatch. Preserve64-bit safe addressing or
prove explicit32-bit offset bounds, never truncate arbitrary production pointers.
Keep rails/group count16 and current operand rounding. Static job-setup evidence
suggests standalone gain likely<5%; combined address/register/epilogue effects
could approach needed margin, but no>=10% guarantee exists.

Exact plan: baseline compactcap5 unchanged; scalar-store specialized-shape and
vector-store generic-shape as separate names. Poison/all-output identity,
M33/65 tails,N128 control if generic,N2048K512 stress; source-specific static
check must show4 STG.128 and no added spills. Screen M128/M256 down7x8 with
paired current/compact. Keep only a>=3% improvement replicated in fresh worker;
compose only if both independently help. ConfirmGU unchanged fallback. Avoid
whole-K512 unrolling: code size/register pressure likely repeats prior failures.

### 3. Locality-aware persistent job mapping, not another wider tile

Plausible>=10% ONLY if cache reuse/working-set ordering is currently limiting.
Keep compact or successful cap4 geometry unchanged; compare bounded2x2 tile
clusters within each expert (two M tiles x two N tiles), rather than globally
swapping all axes. This can retain short-term A and B reuse together without
extra accumulators/shared buffers. Source current order is expert->M->N with
N fastest; grid stride280 on down's32 N tiles shifts columns by24 modulo32
between jobs, while M progression crosses experts. That is a concrete changed
mapping hypothesis, not a proof of cache misses.

Exact plan: new bijective index helper only; CPU prove all jobs once for ragged
M1/33/65/128/256/511 and N512/2048, preserving expert identity/tails. Test clustered
versus original at SAME grids5 forcompact or4 forcap4, M128/M256 down thenGU,
7x8. Use original fresh control in each worker. A direct M-fast order may be
included as one diagnostic but stop if clustering and M-fast both change<3%.
No broad mappings/grid cross-product. Ragged owner replay still required later.

### 4. Grid tuning: low-cost diagnostic, LOW expected standalone margin

Compact grid5 already equals measured five-block residency; grid>5 adds queued
CTAs but no simultaneous blocks. Equal workloads have16384 jobs atdownM256 and
8192 atM128, so basic global underfill is not credible. Remainder imbalance is
at most one job perCTA (~1.7%/3.4% of perCTA job count), below10–15% barring
cache/order interactions. Grid3 risks reducing residency. Therefore don't repeat
old grid-only sweeps. After a NEW mapping/residency change, at most grid4/5/6
forcompact or3/4/5 forcap4 on ONE downM256 screen; only extend a replicated win.

### 5. Preparation placement/N reuse: low priority; avoid closed repeats

Activation rounding already occurs inside stage and contributes F2F plus
widening instructions. Half-register-prefetch is the active low-cost relocation;
separate A16 preparation adds launch/global traffic and should not be revived
without evidence it removes>=10% measured runtime after charging the full pass.
Original row/tiled/pre-widen controls and temporaryW16 staging already lost.
Wide64x128/mid32x128 and unchangedtile256cap3 did not turn reuse into speed;
do not rerun them. Reuse through job-order clustering is the new low-register
alternative. A larger N microtile would need fresh register/occupancy evidence
and is NOT the recommended next bet.

## One diagnostic to discriminate setup/store from K-loop costs

If liveness also loses and attribution remains ambiguous, use same candidate,
M128/N2048/e64 at K128/256/512 (the first two explicitly diagnostic, not model
workloads), paired controls, retained completeprep timing. Fit T≈alpha*K+beta
only as a sensitivity check; changing cache footprint means it is not an exact
decomposition. A large stable intercept motivates epilogue/setup work; a small
intercept reinforces mainloop residency/staging focus. Do not benchmark no-store
kernels and present that as a deployable result.

All tests use parent-approved GPU2/3 slots and serialized timing/compiler work;
GPU1 untouched. No new model run or capture needed for bounded screening.
Full route/cache/dtype/quality acceptance remains in ACCEPTANCE.md and
CAPTURE-INVENTORY.md. None of these predicted margins certify1.5x full pipeline.
