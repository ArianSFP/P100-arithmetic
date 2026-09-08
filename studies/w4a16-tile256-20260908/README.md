# W4A16 256-thread output tile (CPU prepared)

Owns only this study directory. Parent files are read-only. No GPU reservation,
compiler or CUDA launched. Future GPU3 testing requires a separately assigned
slot and the shared benchmark lock; parentGPU2 controller39500 is active.
W4A4 is PAUSED; this is a separate W4A16 FP32-accumulation investigation.

## Hypothesis

Parent compact32x64/128thread kernel retains95registers and5resident blocks;
recorded fresh comparisons are1.406x gate/up and1.331x down. Larger128thread
tiles lost to resource pressure. Instead distribute64x64 outputs over256threads,
with4x4 outputs/thread and32 persistent FP32 accumulator registers (two rails).
This doubles weight reuse across tokens while assigning one packed weight word
per thread instead of two. Each thread still loads8 A32 values, explicitly
rounded A16 then widenedFP32 in shared memory. No extra activation preparation
pass or free persistent weight shadow is introduced.

Shared staging is A32x64+B32x64 FP32 =16KiB/block. Eight warps/block map:

    ar=(warp/2)*16+(lane/8)*4
    bc=(warp%2)*32+(lane%8)*4

Each thread contributes all8 A values to its tid/4 row and all8 weight values
from one word to row tid%64, K chunk(tid/64)*4 and corresponding upper nibble.
Original Q4_0 word-major layout is unchanged:1152bytes per64rows/G32,
128scale bytes followed by256 words ordered[chunk4][row64].

## Interface and bounded candidates

Include parent's verified wide.cuh first, then tile256.cuh. Reuse the existing
decode_q4_pair exactly (CPU test pins its body). Its half operations only
construct/scale half-rounded weights; all GEMM products and accumulations use
__fmaf_rn. No HFMA2 is permitted. HMUL2 is allowed only in decoded weight flow,
subject to the parent's established SASS whitelist/provenance checks.

    tile256_q4<U,MIN_BLOCKS><<<grid,256>>>(raw_a32,word_major_q4,out,M,N,K,experts)

Required shape contract: M>0,Kpositive multiple32,Npositive multiple64,
experts>0; allocation sizes cover all experts,4/16byte alignment as inherited
from cudaMalloc and element/word offsets. Uniform expert counts only here;
parent must not mistake this tile header for a complete ragged service.

Start U32 with MIN_BLOCKS3 and4. U16 is available only as the second bounded
unroll choice, not authorization for a sweep. MIN_BLOCKS4 targets<=64registers
(32warps resident); MIN_BLOCKS3 targets<=80registers after allocation granularity
(24warps resident). These are register/shared-memory feasibility calculations,
NOT measured register counts or occupancy. Reject cap4 if any local spills.
Preserve source, SASS and both control times if cap3 is faster or cap4 fails.

## Numerical schedule and parent wiring gate

For every G32, lo consumes K0..15 and hi consumes K16..31 in ascending order;
both persist across all groups and only final output adds lo+hi inFP32.
No group scale movement or half partial sums. Weight decode matches the verified
packed helper. Prefetch writes registers, never the current shared stage;
barrier before stage overwrite and barrier before next consumption are retained.
Tail A loads zero-fill, and tail outputs never store. CPU tests prove geometric
coverage and schedule membership, not hardware numerical equivalence.

Parent can wire these launches into its existing isolated current-control
harness without modifying this header. Keep original current T64 and compact
winner in each paired worker, all raw activation cost charged. Required gates:

- inspect compile resources/spills and SASS precision provenance;
- NaN-poisoned output before every correctness run;
- existing exhaustive weight-decode gate and hard Q4-scale/raw-A32 stress,
  byte-identical to original split-rail T64;
- memcheck including M33/65 tails before GPU timings;
- clean M256/e64 gate/up and down screens, repeated only if promising;
- eventually actual2k/4k/8k ragged workloads and model accuracy, not inferred
  from equal-expert kernel tests.

No build script is supplied because parent owns harness/control extraction and
the next slot. CPU-only command (safe while no compiler/GPU work is authorized):

```sh
python3 studies/w4a16-tile256-20260908/test_cpu.py
```
