# W4A4: an exact packed-half candidate, awaiting GPU validation

2026-09-08. CPU/compiler preparation complete; **no GPU executed and no measured
speedup**. This study is isolated from the sealed archive and ongoing W4A16
round-3/prefill work. GPU3 permission is pending; no device is reserved here.

## Finding

Signed INT4 x INT4 makes packed FP16 accumulation exact within 32-element scale
groups. With w,a in [-8,7], |w*a| <= 64, and every G32 sequential prefix has
magnitude <= 2048. Binary16 represents every integer in [-2048,2048] exactly.
The implemented kernel splits even and odd K positions into two half lanes:
16 products per lane, each bounded by 1024, then adds the lanes in FP32 exactly.
Weight and activation scales and accumulation across groups remain FP32.

This is a different numerical premise from the archived failed Q8/arbitrary-A16
half substitutions. No intermediate half-to-float flush is needed inside G32.
The kernel uses NVIDIA's documented half2 arithmetic, not hidden INT4 hardware.
[NVIDIA's CUDA mixed-precision description](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/)
distinguishes GP100 packed FP16 from the other Pascal chips' DP4A instructions.

Two packed nibbles are decoded into half2 without I2F: construct half values
1024 + offset_binary_code and subtract 1032. Exhaustive 256-pair CPU decoding
checks cover signed endpoints. The weight layout and row/batch reuse follow
W4A16 round 2. Both A4 controls use the same packed inputs and scaled FP32
reduction order. No changes to weight payload or scales are required.

## What is verified

- All 256 signed INT4 products and 256 packed decode pairs pass CPU checks.
- All 244,524 transitions in a superset of bounded accumulator states and
  distinct INT4 products are exactly representable in binary16.
- A G33 scalar counterexample (32*64+1=2049) shows why an unbounded half dot is
  unsafe. This is a scalar-prefix boundary, not the paired kernel's lane bound.
- CUDA 12.8 with GCC 14 compiles 12 dot kernels plus quantization and reduction
  for SM60. The final build has zero stack frames/spills and no LDL/STL.
- The worker builds; supervisor/build scripts pass Python syntax checks.
- Source, binaries, sealed control and the common CPU reference header are
  SHA-256 inventoried in `build-manifest.json`.

Full kernel static SASS counts, R2 (two rows/lane):

| Batch reuse | Arithmetic | Dot FMA | Scale FFMA | SHFL | I2F | Registers |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Matched FP32 | 64 FFMA | 2 | 32 | 65 | 32 |
| 1 | Paired half2 | 32 HFMA2 | 2 | 16 | 0 | 32 |
| 4 | Matched FP32 | 256 FFMA | 8 | 128 | 68 | 48 |
| 4 | Paired half2 | 128 HFMA2 | 8 | 64 | 0 | 48 |

These are static instruction counts, not throughput predictions. The half2
path adds decoding HADD2, bit manipulation and lane extraction. For R2/B1 it
has 39 HADD2 versus 2 in FP32 (including conversions); R2/B4 has 54 versus 2.
The implementation halves dot FMA and activation-shuffle counts, not the total
instruction count. Timing may still reject it.

## Prepared hardware comparison

`worker.cpp` checks the GPU quantizer against an independent CPU implementation,
then compares all candidates bit-for-bit to integer group dots with the same
FP32 scale-FMA/tree order. The correctness suite includes row/batch tails,
short/long reductions, random values, cancellation, zero groups, finite-half
extremes and direct packed A4=-8 endpoints. GPU correctness is still pending.

The quantizer is symmetric [-7,7], nearest-even, G32 with FP32 scale maxabs/7;
all-zero groups use scale zero. This is **integer A4, not NVFP4**. It retains
packed 4-bit activation payload plus four scale bytes per 32 activations
(5 effective bits/activation including this metadata). Weight storage remains
4.5 bits/weight including original FP16 group scales. No tensor-core operation
or reduced-precision group-scale accumulation is used.

All W4A4 timed pipelines include quantization from original FP16 inputs,
packed activation writes/reads, dot/scaling and S32 reduction. Offline weight
repacking, allocation and host/device transfers are excluded. The live sealed
W4A16 round-2 cubin runs on the original FP16 inputs and has its own CPU oracle
check. Its reference tree is reproduced by the checked candidate reducer.
The harness records individual rotated-order event samples and synthetic
relative-L2 quantization error; it does not infer model quality from that error.

Proposed shapes: M5120/K5120/N1, M17408/K5120/N1, M5120/K17408/N1 and
M5120/K5120/N4. Establish correctness and sanitizer gates first, sweep the
bounded configurations, freeze winners, then repeat in three fresh workers.
The current worker performs the sweep; frozen-winner analysis is not yet run.
The controller requires current matching GPU3 reservation markers in both local
and shared coordination logs and their hashes, checks GPU3 health/processes,
uses env -i/taskset 0-11, preserves logs and stops on errors without reset/retry.

## Limits and decision

There is a credible arithmetic opportunity beyond the A16 warp-sharing work,
particularly when batch reuse amortizes packed weight decoding. But A4 does not
reduce the already-W4 weight stream. At N1 the activation footprint is small
relative to weights; four times fewer activation payload bits does not imply
four times faster inference.

Exact arithmetic **after quantization** does not preserve the original A16
activations. Production adoption still requires byte identity or perplexity
within +/-0.003 using KLD-pair methodology. No model dataset, perplexity/KLD,
stock dispatch or tokens/sec run has been performed. No model-quality or
performance improvement is established by this preparation.

CPU/compiler reproduction (no GPU access):

```sh
python3 studies/w4a4-20260908/build.py
```

Read `PLAN.md` and `supervise.py` before any execution. Do not invoke the worker
until a new GPU3 reservation is authorized and recorded. Other sessions' held
GPU1/GPU2 reservations and all sealed artifacts remain intact.
