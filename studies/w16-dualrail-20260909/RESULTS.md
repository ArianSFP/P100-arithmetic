# W16A16 dual-rail AffinityWave — 2026-09-09

Yes: the matched W16A16 kernel slightly exceeds the fastest confirmed W8A16
FP16-storage pipeline at both M256 expert-prefill projections. It retains the
W8 winner's computation and dispatch and replaces only packed-Q8 loading and
decoding with vectorized FP16 weight loads.

The separately built serving F32-wire specialization also wins: 1.0243x over
W8 gate/up and 1.0190x down. Its report is
`../w16-dualrail-f32-20260909/RESULTS.md`.

## Confirmed paired result

GPU2, 64 experts, 256 tokens per expert. Gate/up is N512 K2048; down is N2048
K512. Each projection has three fresh workers. Each worker uses a two-second
warmup followed by seven rotated rounds and six complete pipelines per event.
Values are the median within each worker and then the median across workers.

| Projection | Original Q8, us | W8 dual-rail, us | W16 dual-rail, us | W16 vs W8 | W16 vs original |
|---|---:|---:|---:|---:|---:|
| Gate/up | 4624.40 | 2514.45 | **2462.50** | **1.0214x** | **1.8777x** |
| Down | 4531.67 | 2393.43 | **2353.96** | **1.0175x** | **1.9256x** |

Every fresh worker favors W16. The paired W16/W8 ratio ranges are
1.02035–1.02192x gate/up and 1.01677–1.01765x down. All three candidate
launches are timed: GPU tile compaction, M64 leftovers and M128 pairs. Static
weight layout construction, allocation and upload are excluded for both
persistent formats.

## Shape screens

One-worker screens at the neighboring token counts show that the result holds
whenever M128 pairs are available. These are directional screens rather than
the three-worker confirmation above.

| Tokens/expert | Projection | W8 dual-rail, us | W16 dual-rail, us | W16 vs W8 |
|---:|---|---:|---:|---:|
| 64 | Gate/up | **874.62** | 934.20 | 0.9362x |
| 64 | Down | 989.40 | **966.99** | 1.0232x |
| 128 | Gate/up | 1344.23 | **1314.81** | 1.0224x |
| 128 | Down | 1256.95 | **1234.16** | 1.0185x |
| 512 | Gate/up | 4826.26 | **4707.21** | 1.0253x |
| 512 | Down | 4738.40 | **4640.22** | 1.0212x |

M64 gate/up is the exception: it uses only the M64 fallback, and W16 is 6.81%
slower than W8. With only one activation tile reusing each weight tile, its
doubled weight traffic costs more than the removed decoder saves. M64 down has
four times as many output groups and still favors W16.

## Kernel and arithmetic

`w16-pair.inc` and `w16-fallback.inc` preserve the selected W8 kernels' M128
pair/M64 fallback tiling, two full-length FP16 accumulation rails, interleaved
HFMA2 schedule, FP32 final addition/output, dimension specialization, shared
layout, barriers, planner and launch counts. Each producer replaces one 16-byte
Q8 scale/code load plus paired decode with two 16-byte FP16 loads. W16 stages
4096 bytes per M64/K32 tile; W8 stages 2176 bytes including scales.

The fixture stores each W16 weight as exactly the FP16 result produced by the
W8 decoder from the same signed Q8 code and FP16 scale. Consequently this test
isolates storage/delivery rather than changing operands. Across timing and
sanitizer workers, all 242,419,712 W16 output words match W8 bit for bit. Both
candidate arms also pass 7,680 sampled complete CPU half-FMA dots.

The specialized W16 M128 kernels compile with 126 registers, 32 KiB shared
memory and zero spills/local memory, versus 122 registers for W8. SASS has the
same 1,024 static HFMA2 instructions. W16 has 1,872 static instructions and
eight 128-bit global loads; W8 has 1,950–1,956 instructions and six 128-bit
loads. The removed decoder work slightly outweighs doubled weight traffic at
these shapes.

## Validation and limits

- M33 partial-tile memcheck: zero errors.
- M129 mixed M128-pair plus M64-leftover racecheck: zero hazards.
- Full outputs finite and W16/W8 byte-identical in all semantic and timing runs.
- Directional M64/M128/M512 screens pass the same full-output and sampled CPU
  checks; only the M256 result has three-worker confirmation.
- Source/binary hashes and compiler command are frozen in `manifest.json`;
  parsed timing results are in `summary.json`, and SASS counts are in
  `sass-audit.json`.

CUDA 12.8, g++13, sm_60, `-O3`, no fast-math. The synthetic workers use actual
FP16 activation storage. The original Q8 control retains FP32 accumulation;
W8 and W16 candidates use FP16 accumulation and therefore do not match that
control's output bits. Model PPL/KLD remains required before applying this
arithmetic to native FP16 model weights. No model or production integration was
run, and no production source/build, commit, push or GPU reset occurred.
