# Tile256 parent integration results

GPU2, user-paused W4A4; all current work W4A16. Dalton's source in sibling
w4a16-tile256-20260908, independent audit in w4a16-verification-20260908.
Current T64 extraction unchanged; all GEMM accumulation remains FP32.

## Compilation gate

- U32/cap4:64registers,72-byte stack,76-byte spill loads/stores. REJECTED
  before GPU execution. Snapshot v8-rejected-cap4 preserves evidence.
- U32/cap3:80registers,16KiB shared,zero stack/spills. Only this variant remains
  in parent active dispatch/config/stress whitelist. Source/SASS/hash tests PASS.

## Hardware correctness

- gpu2-tile256-cap3-smoke: M33,N128,K96,e2 hard signed-half scale/activation-range
  fixture;8448 full CPU oracle and full T64 bit-identity checks;memcheck0.
- gpu2-tile256-cap3-race and -sync: M65,16640 output words match T64,256 CPU
  oracle samples. Racecheck0 hazards/0 warnings; synccheck0 errors.

## Performance screen

M256/e64,grid168candidate/112currentT64,2s warmup,7rotated repetitions of8
iterations. Activation conversion inside timed GEMM; no half accumulation.
Common initial word-major layout preparation still excluded: not full-service
acceptance or a free claim about recurring cache repacks.

| Projection | Current T64 µs | Tile256 µs | Ratio |
|---|---:|---:|---:|
|gate/up|7055.260|5024.436|1.404×|
|down|6838.108|5164.656|1.324×|

Against fastest legacy controls:1.188×/1.216×. Not faster than compactcap5
~5013/5119µs from separate paired screens. No1.5× goal or model gain claimed.

Next bounded hypothesis delegated: reduce prefetch register liveness, either
fetch after computation or retain rounded A16 pairs in fewer registers, to
make cap4 possible without spills. Separate header required so this source and
its manifest remain frozen. No extra GPU/compiler activity by subagents yet.

Controller39500 GPU2-r4 remains held; no current worker. W4A4 remains paused.
