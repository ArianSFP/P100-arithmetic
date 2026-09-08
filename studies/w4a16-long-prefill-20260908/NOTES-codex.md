# W4A16 autonomous continuation

CURRENT 2026-09-08 continuation (supersedes historical state below):
GPU3 controller29187 RELEASED token w4a16-long-prefill-20260908-gpu3-r7.
Release1788906978930344770, GPU3idle5MiB/ECC0,stopped=false, bothlogs updated.
Direct-stage tile256 compiled64regs/no spills; mem/race/sync and output
identity PASS, but downM2565825.664us loses to compact~4960us. Closed.
Contiguous compact mapping flat on GU/down; see DIRECT-AND-MAPPING-RESULTS.md.
Best M256 current-T64 screen ratios remain~1.424GU/~1.373down, FP32 sums.
Rail51263regs/no spills passes all checks but down6400.412us loses; closed.
2x2cluster91regs/no spills passes memcheck/identity, down5015.188us loses;
closed. W4A4 paused. Sartre completed direct integration/route correction audit.
GPU1 has
another session's explicit per-GPU decode reservation; leave it untouched.
No active timing or reservation now; r7 batch finished. Build/hash tests PASS.
Goal1.5x complete prefill pipeline remains unmet. Historical route proxy
generator route_proxy_profiles.py has an owner-mapping problem; use audited
ragged_proxy.py/RAGGED-PROXY.md instead. No current-Q4 activation trace exists.

Historical notes follow; their GPU/controller and NEXT entries are superseded.

LATEST USER ADDITION: simultaneously delegate W4A4 testing for2xT64. Two agents
spawned, no GPU/compiler authority until parent assigns a slot:
- Dalton 01a082c8-8221-7e92-94c2-b3854a9ac00b: implementation, owns only
  studies/w4a4-long-prefill-20260908/implementation/.
- Sartre 01a082c8-8317-7f82-b67e-67704f4b31b7: independent CPU reference/feasibility,
  owns only studies/w4a4-long-prefill-20260908/verification/.
Keep main W4A16 goal intact; do not duplicate their tasks. Both instructed to
review closed W4A4 results, charge activation quantization and use current T64,
retain FP32 accumulation for main comparison, no model/NVFP4 quality claim.
GPU0 ONLY shared by both studies, serialized by parent. No production edits.

Current parent controller85210 is LIVE WAITING for global lock with fresh token
w4a16-long-prefill-20260908-gpu0-r2 (old89584 exited). Do not start another on
an observation timeout. Queue expires600s; check its actual handle/output.
After acquisition rebuild BEFORE any worker (working generated source currently
newer than frozen executable/manifest). No compilation during competing timing.
wide.cuh now also has split-FP32 variants wide_{packed,f32}_split_u32 retaining
current T64 reduction order. Candidate sk2 is checked against the original
split CPU oracle/full-output baseline, not forced to match sequential winner.
No GPU test or compilation for wide variants yet. Pending current-service GPU1
holder1274397 was verified live earlier; inspect current state, don't assume.

AUTHORITATIVE NEWEST STATE: GPU0/global lock RELEASED, controller89584 exited0.
Release JSON1788899791180594700, GPU0 idle23MiB/ECC0. Bothcoordlogs final-released
token w4a16-long-prefill-20260908-gpu0. Other GPU1/GPU2/3 sessions had requested
handoff; respect their held/pending claims. Use onlyGPU0 when reacquiring with a
FRESH token (old supervisor token now invalid due final-release guard).

Staged weights were built and measured this goal turn: all4M256workersPASS,
but~5.66–5.78ms versusfused4.95–5.06ms; dequantpass~720us. CLOSED unchanged.
v3snapshot preserves built source/binary. Current working build/worker.cu has
subsequently been regenerated with NEW wide.cuh; executable/manifest are OLD.
MUST rebuild under reservation before running; mismatch should fail safety check.

NEW CPU-prepared source: wide.cuh 64x128/128threads/8x8microtile, U8/16/32;
FP32-dequant and packed-half-dequant variants. ALL GEMM products/accumulations
remain FP32. Packed HALF is only a bit-exact current-weight preparation method,
not short-half partial sums. test_wide_model.py covers all finitehalf scales and
16codes plus stage/output/globalweight-address coverage. --quant-check1 hardware
test is generated to verify1,015,808 finite-scale/code paired outputs BEFORE
packed GEMM. Kernel tests/timings still pending; no wide compile yet.

Build/test gates permit HMUL2 solely in wide_q4<...,true> and decode_check_kernel;
HFMA2 still forbidden. Need inspect emitted SASS, spills and real precision before
GPUexecution. Independent mainloop source is fmaf(float,float,float).
First new tests: --gpu-approved1 --quant-check1 (memcheck), then wide_f32_u16 /
wide_packed_u16 smoke (t33,n128,k96,e2), then M256,e64 gate/up/down ifPASS.
Current wide paths half-round weights; fixture stilldyadic. Arbitrary-scale GPU
dequantcheck proves that operand step, not whole-model/order accuracy. Ragged
workload, Q4_1 fallback and final numerical/model gate remain outstanding.

LATEST: controller acquired GPU0/global lock and is STILL HOLDING it. All first
screen workers finished PASS; no current worker running. Active PTY89584 takes
JSON requests or `release`. Keep it across goal turns. Started~20:23UTC with1h
guard lifetime; check before assuming live. Build v2 complete,3offline checks
PASS; current_control/m64-smoke memcheck0errors. See RESULTS.md for6shape
ratios: fused M32~1.39–1.44x gate/up and1.34–1.35x down vs current Q4 T64,
only1.20–1.26x faster than legacy controls. Format specialization~1.16–1.18x.
M64fusedU16/U32 both LOST onM256/e64 and are closed unchanged. V1source/build
snapshot preserved in snapshots/v1. Current analyze.py excludes sanitizer and
raw-witness timings and reports BOTH current/legacy denominators.

NEXT implementation lead: temporary FP16 dequantized weight staging, charged
per pipeline run, for reuse at largeM; keep nativeW4 storage, no half accumulation.
No implementation yet. Generate a staged variant from the frozen delivery_lead
template, replacing B nibble decode with uint4 half loads + shared widening,
and add a timed nativeQ4→tiledhalf preparation pass. More bandwidth/scratch may
lose; measure rather than assume. Need real-scale rounding and ragged oracle
work after the first screen. Never silently count prep0/precomputed weights.

Goal: >=1.5x current T64 pipeline on2k/4k/8k-relevant W4A16 expert workloads,
activation preparation included, FP32 accumulation unchanged. Not achieved.
GPU0 ONLY; do not useGPU1/2/3. No production edits, commits or pushes.

Current controller queues at most600s for the global benchmark lock; an earlier
acquisition attempt lost a race to the GPU1 study. No GPU0 worker launched yet.
After acquiring, rebuild (supervisor hash changed) BEFORE running any worker.
Reservation token w4a16-long-prefill-20260908-gpu0 in both coordination logs.
No automatic reset/retry after any worker fault. Keep lock across work gaps.
At final release send `release`, inspect health, append release to BOTH logs.

build.py derives from hash-verified frozen delivery worker and extracts the
current service's aw_q8_service_m64_n128_halfpipe_sync WITHOUT editing its body.
New aw_current_q4 control uses compressed Q4_0 pointer tag5 and raw A32 inputs;
conversion is inside GEMM, so no separate activation pass is charged/needed.
Original three controls and frozen sequential winner remain for attribution.
Current control has128registers/32KiBshared/no spills; specialized Q4_0 clone
has121registers/same shared/no spills. This is only an offline compiler result.

aw_special_q4 fixes format/tag/layout-dependent branches and helper offsets
to Q4_0. Acc0/acc1 half-stage split and final FP32 sum are unchanged. It should
match the current control bitwise; existing CPU split-FMA oracle checks it too.
Do NOT dispatch this specialization for Q4_1 or Q8. Generic control is retained.
All FP32 product/accumulation; no HFMA2/HMUL2 in emitted SASS.

Immediate tests: fresh memcheck smoke t33,n128,k96,e2 then gate/up/down
t128/256/512,e64,grid5,only delivery_fused_u32 (retains both new controls).
Screen equal-size groups as a first control/lead check, NOT an actual ragged
workload or2k+ achieved claim. Real Q4_1, non-half-representable scales and
activation stress, ragged64-expert workload and larger hot M still need coverage.
Current CLI stays maxM512; don't merely raise it and call it trace replay.
Source-derived current service launch is2*SMs (lines11593/14128/16275 of source).

Source provenance: current service SHA256
fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47.
Historical trace prioritization/limitations: sibling delivery-leads study's
LONG-PROMPT-TARGET.md and shape-inventory.json. Full-model qualification cannot
be claimed from this single-GPU primitive; accuracy gate remains byteidentity
or paired PPL within±.003. Do not weaken it to reach a performance number.
