# W16A16 dual-rail AffinityWave, F32 wire — 2026-09-09

Yes: the W16A16 specialization slightly exceeds the fastest confirmed W8A16
serving-input pipeline while receiving the same F32 activations and performing
the same FP16 rounding inside each kernel.

## Confirmed paired result

GPU2, 64 experts, 256 tokens per expert. Gate/up is N512 K2048; down is N2048
K512. Each projection has three fresh workers. Each worker uses a two-second
warmup followed by seven rotated rounds and six complete pipelines per event.
Values are the median within each worker and then the median across workers.

| Projection | Original Q8, us | W8 dual-rail, us | W16 dual-rail, us | W16 vs W8 | W16 vs original |
|---|---:|---:|---:|---:|---:|
| Gate/up | 4623.62 | 2589.49 | **2528.34** | **1.0243x** | **1.8289x** |
| Down | 4528.38 | 2455.98 | **2411.37** | **1.0190x** | **1.8784x** |

Every fresh worker favors W16. The W16/W8 ranges are 1.02347–1.02509x
gate/up and 1.01829–1.01981x down. The in-worker W8 timings also reproduce the
archived `compact-plan-f32` medians closely: 2589.49 versus 2589.64 us gate/up
and 2455.98 versus 2458.01 us down.

All three candidate launches are timed: GPU tile compaction, M64 leftovers and
M128 pairs. Static layout creation, allocation and upload are excluded equally
for W8 and W16.

## Kernel and validation

The kernel preserves the W8 F32 loader, which performs two 128-bit F32 loads
and rounds eight activations to FP16, together with M128 pairing, two
full-length interleaved HFMA2 rails, FP32 final addition, dimension
specialization, GPU compaction and the M64 fallback. Only the packed-Q8
scale/code loads and decoder are replaced by two 128-bit FP16 weight loads per
producer.

The fixture stores each W16 weight as the exact FP16 value produced by the W8
decoder. Across all workers, all 127,076,352 W16 output words match W8 bit for
bit, 4,608 sampled CPU half-FMA dots pass, and all outputs are finite. M33
partial-tile memcheck reports zero errors; M129 mixed-pair/leftover racecheck
reports zero hazards.

The specialized W16 pair kernels use 125 registers, 32 KiB shared memory and
zero spills/local memory; W8 uses 123 registers. Both contain 1,024 static
HFMA2 instructions. W16 has 1,890 static instructions and twelve 128-bit
global loads, versus 1,974 instructions and ten loads for W8.

CUDA 12.8, g++13, sm_60, `-O3`, no fast-math. Source/binary hashes and compiler
command are frozen in `manifest.json`; parsed results are in `summary.json`,
and SASS counts are in `sass-audit.json`. This is a synthetic kernel result.
Native FP16 model weights still require the PPL/KLD gate, and full-model
throughput remains unmeasured. No production source/build, commit, push or GPU
reset occurred.
