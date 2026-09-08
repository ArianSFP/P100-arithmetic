# Independent tile256 liveness review

2026-09-08. CPU/read-only; no parent or Dalton edits, compiler, GPU access or
reservation. W4A4 remains paused. Reviewed hashes:

- liveness.cuh:36dd1bb71f94306d1ff70d4852b433afec0e7278f0150d5bee15c3f09f826b58
- test_liveness.py:01204977522c8fb4f5333465419f7a8d6408c1a733a00833958c7e6a2878afaf
- parent build.py:5ce289d33b68f31f08b285ad578471c414baf3b3a8b8511a8e2ef986148fa3c3

Verdict: no source-level arithmetic, load-address or barrier blocker found.
Hardware correctness, actual register liveness, spills and speed remain unproven.

## Half-register activation semantics

HALF_PREFETCH=true fetches8 A32 values in two float4 loads and independently
rounds adjacent pairs with __floats2half2_rn. Stage applies __half22float2,
which widens each stored half exactly, then writes K offsets2*i and2*i+1.
That is the same single RN-half rounding and exact widening as the scalar
__half2float(__float2half_rn(x)) path, only moved earlier. There is no half
activation multiply/add and no second narrowing. Signed zero/subnormal lane
mapping is retained in the source. Finite A32 overflow tohalf Inf has the same
mathematical behavior as scalar conversion but is outside accepted finite-output
GEMM fixtures. NaN payload preservation is not established or required here.

All4 half2 registers are assigned before stage, including final/tail tiles.
Invalid rows use zero float4 values and do not read beyond A. HALF_PREFETCH=false
assigns both pa0/pa1, and constexpr selects only that representation. No source
use of the uninitialized inactive representation. Register count4 versus8 is
source-level intent only: compiler allocation must be measured independently.

## Late fetch and ordering

Both versions initially fetch0/stage0/barrier before computation. Early version
fetches g+1 into private registers before computing current shared g. Late
version fetches g+1 after current compute but BEFORE the closing CTA barrier.
This is safe: it does not overwrite shared memory; the following barrier waits
for all8 warps' current shared reads before stage writes begin. A second barrier
waits for the entire new shared tile before next compute. Source arrays a,w,out
are separate allocations in the harness; in-place aliasing is not supported.

Fetch executes exactly once per group, no g+1 access for final group. All256
threads take the same persistent-job and group bounds; row guards only control
loads/stores, not barriers. The final barrier also protects next persistent job.
K0–15 andK16–31 FP32 FMA rails persist across all groups and add once with
__fadd_rn at output. Shared geometry/dequant addresses remain reviewed tile256.

## Tests independently inspected and executed

`PYTHONDONTWRITEBYTECODE=1 python3 studies/w4a16-tile256-20260908/test_liveness.py`
PASS:63,488 finite paired-half patterns,256 raw-A32 lane mapping cases,8 fetch/
stage schedules. Its imported test_cpu.py is read-only and also passes4096
output owners,2048 A/B cells,15 tails,12,288 addresses,8 rail schedules and
resource-target arithmetic. Frozen original tile256 hash remains
218e323c96e1421aae73f343f7cedc67b3b390a2c4b3cd55ea846f923f456beb.

Limits: Python half packing is a CPU conversion model, not proof of device
conversion lowering. The schedule test is a phase-level model, not a GPU race
detector or asynchronous warp simulation. Resource-target assertions are NOT
compiler register/occupancy measurements. These limitations are honestly stated
by the tests; retain device identity/sanitizer gates for both new entry points.

## Parent wiring / validation review

Generated worker and build.py agree:

-16032 tile256_no_prefetch -> tile256_liveness_q4<false>,256 threads.
-17032 tile256_half_prefetch -> tile256_liveness_q4<true>,256 threads.
-Both receive leadraw/word-major leadw, with Config.sk=2. No accidental128-thread
 launch, no old wide dispatch shadowing. Activation conversion remains inside
 the timed kernel; neither variant incurs an omitted required row-prep step.
-Both names are in pre-CUDA scale-stress whitelist and post-filter column list;
 post-filter checks requested name survival. quant-check+stress and raw+stress
 are rejected. Stress-only reference uses current T64 half-rounded weights.
-Every validation launch cudaMemset-poisons outputs; all nonfinites fail. Every
 sk2 output must match current reference bits, plus independent CPU rail samples.
-SASS gate forbids HFMA2 globally; HMUL2 permitted in tile256_liveness_q4 only
 alongside FFMA because BOTH variants call shared packed weight decoder. This
 does not mean half arithmetic was introduced by activation prefetch; activation
 prefetch is conversion-only in reviewed source. Regex alone is not semantic
 proof, hence source review and final manifest/SASS checks remain necessary.

## Nonblocking reporting caveat and recommended GPU gates

Inherited --raw-witness prints accuracy_only=1 but still reaches warmup/events.
It supplies raw A32 to candidate and prepared A16-widened A32 to control. That
is a valid conversion-correctness witness, NOT a matched timing input stream.
Analysis must exclude its TIME rows (existing long-prefill analyzer does so).
Prefer an explicit early return after correctness in a future owner build;
do not interpret an accuracy_only label as proof that timing was suppressed.
Do not combine raw-witness with quant-check either; the latter returns early.

Before performance: separate scale-stress and raw-witness memcheck for EACH
variant, tails and multiple groups, followed by racecheck/synccheck when viable.
Use regular identical-input full-prep runs for matched performance; parent
coordinates the device/lock slot. No proposedGPU1 use or new model capture.
