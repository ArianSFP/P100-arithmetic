# Lower-liveness continuation

W4A4 paused. Parent owns main W4A16 harness; Dalton owns tile256/liveness
headers; Sartre owns independent verification and capture inventory.

GPU2-r4 released normally: controller39500 terminal, release artifact
release-1788903588521365139.json, idle5MiB/ECC0,no stopped/fault state.
GPU1decode study acquired shared lock after handoff. New parent GPU3 queue
session21563 uses tokenw4a16-long-prefill-20260908-gpu3-r5. It has NOT acquired
at this checkpoint. Poll that same handle; never restart on observation timeout.

Both new variants wired from sibling w4a16-tile256-20260908/liveness.cuh:

- tile256_no_prefetch,column16032: fetch next group only after arithmetic.
- tile256_half_prefetch,column17032: prefetch next group's eight activations
  as four half2 registers, preserving A16 rounding and FP32 GEMM arithmetic.

Both cap4,256threads,64x64tile,sk2. Originaltile256header frozen. Initial
CPU tests PASS including63488 finite paired-half patterns,256 raw-A32 lane
mappings, and fetch/stage schedules. Independent review requested.

Build is NOT current: --prepare-only generated newworker/source, manifest
and binary remain older tile256cap3 build. Rebuild only after actual shared
lock acquisition; reject spills beforeGPU and inspect HMUL2 operand dataflow.
Whitelist allows half weight decode only; HFMA2 forbidden. Then scale-stress,
raw-A32 witness, memcheck/racecheck/synccheck, then paired GU/down timings.
Use fresh controls on GPU3; never compare its candidate against GPU2 times.

Current-Q4 activation/route capture is missing from scoped existing artifacts;
see sibling verification/CAPTURE-INVENTORY.md for genuine historicalQ8 traces
and why they are not current-Q4 replay. This is an acceptance-data gap, not an
excuse to claim success or stop useful bounded kernel work.
