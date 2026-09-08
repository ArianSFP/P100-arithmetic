# Direct staging candidate — implemented, CPU-only

Changed/new files only: direct_stage.cuh, test_direct_stage.py, DIRECT-STAGE.md.
No parent files, frozen tile256.cuh/liveness.cuh, coordination logs or W4A4
files changed. No compilation, GPU launch, reservation or background process.

Header is self-contained with CUDA runtime/half headers and a private namespace
copy of the verified parent weight decoder. CPU test compares decoder bodies
exactly. Parent can include it without wide.cuh and launch one bounded candidate:

    tile256_direct_stage_q4<<<grid,256>>>(raw_a32,word_major_q4,out,M,N,K,experts)

Same64x64 output tile,256threads,4x4 outputs/thread,16KiB shared,cap4,
full32 compute unroll. Required shape/allocation contract matches tile256:
Mpositive,Npositive multiple64,Kpositive multiple32,positive experts,
aligned/nonaliasing device arrays and reviewed integer task-count bounds.

## Live-value changes

There are no fetched-operand arrays, producer lambdas or next-group register
payloads. A tiny rolled2-iteration producer loop loads one float4, immediately
rounds each value toA16, widens toFP32 and stores its4 shared cells before the
next vector. A separate rolled4-iteration weight loop holds one compressed word
and scale, decodes one byte pair, widens and immediately stores2 values.
No eight-element decoded-weight array exists. `#pragma unroll 1` on both small
producer loops limits compiler lookahead opportunities; it adds loop/control
and index instructions, so this is not an assumed free speed improvement.

Source-level intended staging peak:4 raw A floats, or one compressed B word,
one half scale and one decoded float2 plus decoder temporaries. Producer and
compute scopes do not overlap by design.32 FP32 accumulator values persist
throughout; mainloop reads4 A+4 B values at a time. Address state, compiler
scheduling and register-allocation granularity remain additional costs.
Target is<=64registers with zero spills for4blocks/SM, NOT a measured register
count. Reject if any spill load/store or local stack traffic appears. Do not
infer physical liveness solely from the lexical C++ scopes or unroll pragmas.

## Arithmetic and synchronization

Raw A32 is rounded RN toA16 then widened exactly as current T64. The copied
packed decoder only prepares half-rounded Q4_0 weights; HADD2/HMUL2 may occur
there, never in GEMM products or accumulation. Both persistent FP32 rails keep
the original schedule: every group's K0..15 feeds lo, K16..31 feeds hi,
ascending order across groups; final output is add_rn(lo,hi). No group scale
movement, extra quantization, or FP16 partial sum. Every conversion remains
inside the GEMM's timing interval; no prepass or persistent shadow is assumed.

Two uniform CTA barriers per G32: before shared producers overwrite the prior
stage, and after complete A/B staging before consumers. One final CTA barrier
after output protects the next persistent tile. Total2G+1 barriers/tile; the
earlier prefetch template had2G+2 including initial and redundant last-stage
barriers. This deliberate barrier-count difference should be recorded in A/B
attribution. No barrier occurs inside a tail predicate. Tail A values zero-fill
and tail outputs do not store. No global prefetch overlap survives.

CPU tests independently enumerate A/B shared ownership,4,096 output owners,
global/tail/vector-alignment addresses, exact rail operation ordering and all
barrier phases; verify unchanged frozen headers and copied decoder identity.
These models are not numerical GPU tests or proof that the compiler uses the
intended register lifetimes. Parent must run existing hard-scale/raw-A32
byte-identity oracles with NaN output poisoning, SASS precision/spill inspection,
memcheck/racecheck/synccheck before any performance promotion. Keep fresh current
T64 and compact controls and charge all work on both real projection shapes.

    python3 studies/w4a16-tile256-20260908/test_direct_stage.py
