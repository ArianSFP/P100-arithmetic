# Compact exact-Q4_0 split-FP32 study

GPU2, M256/e64 gate/up N512 K2048,7reps/8iters/2s warmup, full activation
conversion included. Current T64 comparator unchanged. Initial snapshot v5:

| Decoder | Candidate µs | Current T64 µs | Ratio |
|---|---:|---:|---:|
| paired half weights |6346.288|7044.872|1.110×|
| scalar half-rounded weights |6612.156|7038.404|1.064×|

Both variants use103 registers,12KiB shared,no spills. Both pass8448-output
full CPU oracle and bit identity to current T64 under signed/non-dyadic scale
stress, with zero memcheck errors. Independent CPU address/rail review passes.
These results are below the existing smaller lead (~4954µs on easy fixtures).
That older lead does not implement T64's half-rounded weights or split order.

Artifacts gpu2-compact-{packed,scalar}-{smoke,gu-m256}. Scalar timing has more
variation (6531–6895µs); this is screening, not a fine-grained decoder win claim.
No1.5× claim, no down/model claim from gate/up.

Next isolated hypothesis:103 registers may reduce block residency. Tighten
only compact launch bounds to five128-thread blocks/SM, check no spills,
query occupancy explicitly, rerun correctness and fresh controls. Never change
the comparator's compilation/register allocation to improve the denominator.

## Five-block launch bound: measured improvement

Changing only compact launch bounds to128 threads / five minimum blocks reduced
registers103→95 without spills. Runtime occupancy reports five blocks/SM for
both capped compact and old fused. gpu2-compact-cap-smoke:8448 full CPU/identity
checks PASS under non-dyadic signed scales and activation-range stress;
memcheck0. Source/manifest/SASS checks PASS after completed build.

| Projection | Current T64 µs | Capped compact µs | Ratio |
|---|---:|---:|---:|
| gate/up |7046.812|5012.712|1.406×|
| down |6814.372|5119.140|1.331×|

Tags gpu2-compact-cap-{gu,down}-m256, same7×8 timing/fullprep configuration.
Against fastest legacy control the ratios are1.191× and1.229×; do not conceal
that denominator. Exact Q4_0 operand/reduction semantics now approach the older
easy-fixture lead's speed, but still do not meet1.5× or prove prompt-level gains.

Next hypothesis is intermediate32x128 tile with controlled residency: more
weight/output reuse than compact32x64 without the64x128 variant's excessive
accumulator footprint. This is unimplemented/unmeasured at this checkpoint.
GPU2 controller39500 tokenw4a16-long-prefill-20260908-gpu2-r4 remains held for
bounded next iteration; no other GPU use, no worker currently running.

## Intermediate32x128 screen

Implemented via wide_q4 MT32 specialization,167 registers,20KiB shared,no
spills,three-block launch bound. CPU tile coverage forMT32/64 PASS; memcheck
gpu2-mid-smoke checks8448 outputs bit-identical under hard-scale fixture.
M256/e64,grid3,7×8 paired full-preparation timings:

- gate/up5359.724µs vs current7049.624µs =1.315×.
- down5616.832µs vs current6804.672µs =1.211×.

No gain over capped compact atgrid5, so this tested intermediate configuration
does not advance the target. Raw tags gpu2-mid-{gu,down}-m256-g3. No new1.5×
claim. User pauses W4A4; both agents now assigned W4A16, including a genuinely
different64x64/256-thread layout and independent full-pipeline acceptance review.

## Equal-M coverage extension (same capped compact)

GPU2,64experts,grid5,7×8 full-activation-cost screens, tags
gpu2-compact-cap-{gu,down}-m{64,128}:

|M|Projection|Current T64 µs|Compact µs|Ratio|
|---:|---|---:|---:|---:|
|64|gate/up|2013.356|1302.740|1.545×|
|64|down|1791.124|1309.928|1.367×|
|128|gate/up|3605.844|2533.808|1.423×|
|128|down|3448.324|2585.744|1.334×|

One projection/shape crosses1.5×; the complete2k/4k/8k goal is NOT achieved.
These equal-M fixtures are not captured Q4 owner routes. New independent
acceptance audit lists full dispatch/cache/dtype/capture requirements in
sibling w4a16-verification-20260908/ACCEPTANCE.md.
