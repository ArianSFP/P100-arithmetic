# W4A4 long-prefill implementation slot

Status: PAUSED by explicit user instruction; no further W4A4 work.
First GPU3 slot COMPLETE, both tiles rejected at0.67–0.69x current
T64 including preparation. See RESULTS.md. GPU3/shared lock explicitly released.
UPDATE: parent relayed new user authorization for GPU2/3. This delegated worker
now owns GPU3 ONLY, with guarded shared-lock acquisition. Parent owns GPU2;
GPU0/1 are untouched. All worker/supervisor UUID checks require physical GPU3.
Do not run the worker directly. No claim of2x or quality qualification.

## Bounded lead and why it is distinct

The prior `w4a4-20260908` bit-plane experiment lost to W4A16 at one/four vectors
and used stripe reductions. This candidate is a grouped long-prefill GEMM with
16x64 or32x64 tiles, two/four by four output microtiles, plane reuse in shared
memory/registers, and no cross-CTA reduction. It targets the actual N512/K2048
and N2048/K512 projections, not dense 5120-channel decode. The integer identity
is prior art within this repository; only its prefill mapping is new. This is
a bounded falsification test, not an expectation that a renamed old kernel wins.

There are sixteen AND/POPC pairs plus coefficient combination per32 scalar
products/output group. Ordinary FP32 needs32 FMAs for the same group. Even
before instruction throughput, sixteen pairs are not a compelling arithmetic
instruction-count reduction; a2x claim would need demonstrated savings in
delivery/dequantization. Do not launch a large sweep if the two tiles miss badly.

## Explicit format and output equation

Weights: Q4_0 offset-binary nibble payload interpreted q_w in[-8,7], FP16 scale
d_w per32 weights. Input is the current service's tiled1152-byte Q4 layout.
The candidate converts each group into four two's-complement planes and an
FP32 widened scale. This20-byte scratch is5bits/weight, not4.5; original18-byte
groups remain resident too. Conversion is charged on EVERY timed pipeline,
including zero-token experts (conservative; no free persistent cache assumption).

Activations: incoming FP32 rounded RN toFP16 as in current Q4 T64, then each
G32 token group uses d_a=maxabs/7 inFP32. Codes q_a=clamp(RN(a16/d_a),-7,7);
zero groups use zero scale/codes. Finite inputs only. Four planes plusFP32
scale are20bytes/G32 (5bits/activation), not NVFP4 or E2M1.

For coefficients c=[1,2,4,-8], candidate group dot is the exact signed integer
sum I_g=sum(p,q)c_p*c_q*popcount(A_p & W_q). Integer magnitude<=2048; promotion
toFP32 is exact. Output is sequential group-order

    C = fma_f32(float(I_g), mul_f32(d_a, d_w), C)

starting at+0. The matched A4 FP32 control reconstructs integer codes from the
same planes and computes32 FP32 FMAs/group, which are exact for these bounded
integers, then uses the SAME scale-FMA. NoHFMA2/HMUL2 accepted by build audit.

This is NOT equivalent to current T64 on original A16: activation quantization
changes values, scaling is regrouped, and arbitrary Q4 scales differ from the
current service's per-weight FP16 rounding. Fixture scales are selected to make
q_w*d_w exactly half-representable; this deliberately isolates activation cost,
but does not validate arbitrary model scales, Q4_1, KLD or NVFP4 equivalence.
The model accuracy gate remains PPL±.003 using KLD-pair (or byte identity).

## Frozen comparator, costs and replay scope

build.py verifies full current-service source SHA256
fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47, then extracts
aw_q8_service_m64_n128_halfpipe_sync unchanged. Baseline uses112CTAs and its
original FP32 split16+16 accumulation, original A32->A16 conversion and weight
decode. Its CPU oracle matches that schedule. The A4 oracle uses independent
host integer dots and FP32 group FMA; all prepared plane words/scales are checked.
All outputs get finite checks;<=16384 outputs all get CPU checks, otherwise256
per expert/config. Synthetic relative-L2 is reported, never called model quality.

`--counts` accepts up to64 ragged expert counts, each0..8192, total<=65536.
Actual captured owner-local counts must be supplied by parent with provenance.
Default33,65 is a smoke case, not a2k+ benchmark. This replays all counts with
M64 T64 tail masking; actual current M32/M16 tail dispatch is NOT extracted yet.
No indexed gather, real routing values, owner-service reduction or compressed
cache misses are modeled. Thus this worker is screening, not workload acceptance.
Both paths exclude allocation/HTD and preexisting common Q4 tiled-layout creation;
candidate-specific activation and weight plane generation are inside every
timed pipeline. Current baseline does not pay candidate preparation.

Pre-launch review fixes: every correctness run now poisons output with NaNs,
outside timed events, then checks all outputs finite. `--pattern 1` exercises
signed zeros, INT4 RN-even ties, raw A32 half midpoints, finite-half extrema,
outliers and tiny/subnormal-only groups, plus arbitrary positive/negative half
weight scales. Both CPU oracles explicitly construct A16 from shared raw A32;
nonfinite converted inputs are rejected, host RN-even is asserted. `--seed`
supports independent numerical fixtures. Hard-scale A4 outputs use their own
group-dot equation, NOT identity to differently rounded T64. Pattern0 retains
half-exact weights to isolate quantization effects; neither is model validation.

## Parent handoff commands (NOT run here)

After acquiring GPU3/shared lock and ensuring no other benchmark is timing:

```sh
taskset --cpu-list 12-23 python3 studies/w4a4-long-prefill-20260908/implementation/build.py --compile
```

Source preparation/light CPU checks (no compiler or CUDA):

```sh
python3 studies/w4a4-long-prefill-20260908/implementation/build.py
python3 studies/w4a4-long-prefill-20260908/implementation/test_cpu.py
```

The guarded fresh-worker supervisor sets only physicalGPU3 UUID visible,
env-i production stack/taskset0–11, persist manifest/input/command, monitor
health/disk and enforces180s timeout/stop-on-error. Worker itself verifies GPU3
UUID after CUDA initialization but DOES NOT acquire locks or supervise recovery.

Initial worker args:

    --gpu-approved 1 --counts 33,65 --n 128 --k 96 --warmup-ms 0 --reps 1 --iters 1

Run memcheck before timing, then synccheck/racecheck if viable. Screen with
captured ragged counts at N512/K2048 and N2048/K512, `--only popc4` and `popc2`.
`all` includes the deliberately expensive matched-A4 scalar FP32 attribution
control; do not mistake beating that control for beating current T64.
Timing is rotated and warm; report ratios of paired medians, then3fresh workers
only if close to2x. For true acceptance parent must add current tail dispatch,
all actual recurrent preparation, source/input provenance and model accuracy.
