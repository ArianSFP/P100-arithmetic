# FP32-only W4A16 vs T64: first bounded sweep

2026-09-08, GPU2 Tesla P100-PCIE-16GB, CUDA12.8, GCC14, SM60.

**The 2x target is not achieved.** Holding FP32 products/accumulation fixed,
the best confirmed synthetic primitive improves gate/up by about1.28x and
down by about1.25x. Including activation preparation gives about1.21-1.23x.
The winner changes FP32 summation order; no model quality/integration claim.
FP16 partial-sum gains from the adjacent R1 study are explicitly excluded.

## What was compared

The original `aw_q8_service_m64<false,true,false,2,false>` body is mechanically
extracted unchanged from the SHA-256-verified source in build.py. Also include
its existing A16-input specialization, and an explicitly generated Q4-storage
specialization with the same schedule and accumulation order. All ratios use
the **fastest of these three controls in the same worker and timing scope**.
This conservative rule avoids counting a slowed Q4 decoder as the baseline.

Controls retain their original persistent grid of112 CTAs. Candidate grids
112/168/224 were tested. A full retuning of the original control grid was not
performed; these are comparisons to the included original launch configuration,
not a proof against every possible T64 tuning. All kernels run on one GPU.

Inputs are deterministic half-representable values. The A32 control receives
their exact widening; A16 kernels load the original half bits. Codes[-8,7] and
half scales are identical in Q8-layout and Q4-layout representations. Q4 payload
is18 bytes/G32, row-major packed bytes in each T64 stage, not expanded global
weights. Q8-layout control uses34 bytes/G32. The fixture isolates storage and
kernel work; it is **not Q8-to-Q4 model quantization or a model-quality test**.
It exercises Q4_0 only, not the actual model's Q4_1 affine expert tensors.

Weight dequantization in this worker is `float(q)*float(scale)`. This original
primitive differs from the accepted newer Q4 model port, which prepares its
dequantized operands through FP16. Current fixture scales are small multiples
of1/1024, for which these products happen to be half-representable. Real-weight
tests must preserve the chosen control's operand preparation explicitly.

Timing excludes allocations, host transfers and offline weight repacking
equally. Kernel-only starts from already prepared inputs. Complete-pipeline
timing includes A32-to-A16 preparation for every A16 kernel; the original A32
control needs no such pass. Neither scope includes MoE routing, gate/up fusion,
SwiGLU, owner reductions, compressed-cache misses or complete model execution.

## Confirmed timings

All rows use16 independent synthetic experts and the selected
`q4_m64n64b128s1d1u8r`: tile64x64,128 threads, sequential FP32 K accumulation,
one shared stage,8-step unroll, row-first outer product, grid168.

| Shape: tokens / outputs / K | Fastest control kernel us | Candidate kernel us | Kernel ratio | Ratio including A preparation |
| --- | ---: | ---: | ---: | ---: |
| Gate/up:64 /512 /2048 | 532.067 | 414.168 | 1.285x | 1.213x |
| Down:64 /2048 /512 | 463.725 | 371.976 | 1.247x | 1.227x |
| Token tail:33 /512 /2048 | 518.028 | 411.812 | 1.258x | 1.212x |

Evidence: gateup-confirm2, down-confirm2, tail33-grid3 under gpu-results/.
Their complete-pipeline control/candidate medians are respectively
534.488/440.805,466.357/380.072,517.540/426.912 microseconds.

Gate/up screening at grid168 gave1.284x; a separate grid224 confirmation gave
1.281x. Down screening gave1.248x for the selected row schedule. Confirmations
retain11 repetitions of12 launches per scope; screening and tail retain7 of8.
Each fresh worker warms all included kernels for at least2 seconds, rotates
configuration order and alternates timing-scope order. Raw samples and200ms GPU
clock/power/utilization telemetry are retained. No CPU compilation overlapped
these GPU timings. No clock locking or cross-worker absolute-time pooling.

The byte-identical split-K2 candidate
`q4_m64n64b256s2d0u16c` reached1.192x gate/up at grid168,1.128x with preparation.
This was screening, not a separate final repeated model-qualified result.

## Findings from the sweep

28 candidates =14 tile/thread/split/staging/unroll choices, each with row-first
and column-first FMA schedules. Larger64x128 and128x64 tiles did not improve
these64-token shapes. A128-token tile also wastes work at64 tokens; its result
is not evidence against larger tiles on longer token batches. Column order did
not materially beat the selected row schedule. Registers ranged79-181, no
spills. The winner uses106 registers and16 KiB shared memory; its larger grid
helps fill the SMs. All arithmetic is FFMA/FP32; build rejects any HFMA2 and
the offline test additionally rejects HMUL2.

Weights are decoded/scaled and activations widened once per shared stage,
then reused by vector shared loads and FP32 outer products. No half accumulator
or activation quantization is involved. Single staging saves shared capacity
but adds a synchronization; this trade won for the selected128-thread shape.

The storage-only Q4 control was slower than the fastest original control;
compressed bytes by themselves do not guarantee a gain. The improved mapping
is necessary in this experiment. Group scales remain inside their K32 group.

For gate/up the useful operation count is2,147,483,648 conventional FLOPs.
A2x result against the532us primitive would require roughly266us, or8.1 useful
TFLOP/s including this kernel's operand delivery. The winner delivers about
5.2 useful TFLOP/s. This arithmetic accounting is not a hardware performance
ceiling or evidence that2x is impossible, but it shows how much work remains.

## Correctness and safety

- CPU-only tests verify exact thread/output ownership, every staged input,
  vector alignment, shared-memory bounds,1,024 randomized nibble expansions,
  source hashes and the FP32-only SASS gate.
- All31 configurations passed memcheck on a33-token tail:8,448 outputs each,
  every output independently checked against its FP32 host FMA order.
- All31 passed synccheck with64 experts, exercising persistent CTA reuse and
  partial token tiles. The winner plus all three controls passed racecheck:
  zero hazards/errors/warnings.
- Large benchmarks check256 independently selected full CPU dots/configuration,
  all output words for non-finites and differences, and require split-K2
  variants to match the original GPU control bit-for-bit.
- Winner relative-L2 vs the split-K2 control:4.963e-7 gate/up,1.339e-7 down.
  Outlier and cancellation fixtures checked all8,448 outputs against the
  sequential FP32 oracle; relative-L2 vs control8.085e-7 and4.504e-7.
  All outputs were finite. A=2048,q=-8,d=1/1024 range witness was byte-identical
  and finite; no half-rail overflow problem exists in this FP32 path.
- These fixtures do not cover all FP16 patterns/NaNs or all model layers.
  Different summation order still requires the applicable model KLD/PPL gate
  before promotion. Neither +/-0.003 model PPL nor NVFP4 equivalence was tested.

GPU2 reservation and the shared lock were held across the entire suite. Fresh
worker per command, persisted prelaunch source/binary hashes,180s timeout,
desktop-log/disk guards, ECC/compute-client health checks, no resets. Released
18:50:44 UTC:5 MiB idle, utilization0, ECC0, no GPU2 compute client. No production
edits, commits or pushes. Sanitizer/stress timings are not performance evidence.

## Next priority toward2x

First compare this candidate against the **final compressed Q4 T64 consumers**
on captured real expert inputs, matching operand preparation and Q4_1 semantics,
before further broad tile tuning or any production integration. The separate
[Q4 T64 model study](../qwen35-q4-t64-20260908/RESULTS.md) reports2267.87tok/s,
1.1558x its native-Q4 control with FP32 mainloops retained. That improvement and
this microkernel ratio overlap in layout/operand work and **must not be
multiplied**. Neither establishes2x full-model prefill.

Then isolate whether the remaining cost is FFMA scheduling or operand staging
using matched resident/shared-load ablations of the winning tile. The old
N1/N4 register-LUT and ordinary4x4/8x8 scheduling studies are not new candidates;
do not repeat them unchanged. A genuinely new LUT study would need substantially
larger table reuse across output rows and full construction/lookup costs, not
a claim based on arithmetic counts alone. FP16 rails stay a separate lever.

## Reproduction

`python3 test_offline.py` checks the frozen current build without a GPU.
`python3 analyze.py gateup-confirm2 down-confirm2 tail33-grid3` reports paired
ratios. build.py rebuilds and rewrites the build inventory; copy the study first
if preserving these exact binaries. Workers' result.json files retain commands
and hashes. GPU reproduction requires a fresh coordinated claim/token and the
supervisor; do not reuse this released reservation or run worker directly on
shared GPUs. The supervisor reuses the adjacent R1 controller's audited safety
logic only, not its kernels; that dependency is hashed in the manifest.
