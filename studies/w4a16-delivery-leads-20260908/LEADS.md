# CPU-prepared leads toward 1.5x with activation preparation included

Status update, 2026-09-08: user subsequently authorized GPU0/GPU1. GPU testing
found a fused-conversion M32 kernel at about1.55x gate/up and1.35–1.36x down
including activation conversion, with FP32 accumulation. See RESULTS.md for
matched measurements, limitations, and reservation disposition. The following
lead list is the historical CPU-only plan, not the final measured ranking.
All work is isolated here; previous measured sources/binaries remain unchanged.

## Target and controls

FP32 products/accumulation and existing A16 values stay fixed. All activation
conversion/transpose work remains in `prep=1` pipeline timing. Prior paired
planning budgets are gate/up356.325us vs current440.805us, down310.905us vs
current380.072us. That means another19.2%/18.2% latency reduction. Actual
acceptance must use fresh interleaved controls rather than these old times.
Removing activation preparation alone would not reach1.5x.

This worker retains all three original T64 controls and the frozen sequential
FP32 winner. New leads must match **every output word** of that winner before
timing, in addition to the independent CPU FMA checks inherited from the old
worker. The winner itself changes the original T64 split-K reduction order;
real model accuracy remains unqualified. No claim against the newer complete
compressed T64 service is made by this primitive.

## Ranked leads

| Priority | Lead | Why test it | Cost / risk |
| --- | --- | --- | --- |
| 1 | A16 conversion plus tiled transpose, four-token staging (`a3`) | Makes each thread's shared writes contiguous; new layout reused across output tiles. | Transpose and padded tails cost time; more64-bit global loads than the eight-token variant. |
| 1, matched alternative | Tiled A16, eight-token staging (`a1`) | Fewer/wider global loads and fewer shared-store instructions than row-major. | Store transaction behavior may differ from four-token staging. Must measure both. |
| 2 | Lossless Q4 word-major layout (`b1`) | Both global code loads and shared staging become contiguous across adjacent rows. | Conversion/layout costs and model cache policy must be accounted for; no extra arithmetic lanes. |
| 3, attribution control | Tiled FP32 scratch containing widened A16 (`a2`) | Separates layout benefits from repeated half-to-float conversion costs. | Doubles activation scratch/write volume versus tiled A16. Generic pre-widening already lost previously; this is a new layout control, not a reopened claim that widening is free. |
| Small matched control | Unroll8/16/32 | Tests schedule/register overhead with the new delivery layouts. | Code size and register scheduling can regress; no broad tile-size search. |

The combination `a3b1` or `a1b1` is the main performance hypothesis. `a0b1`
isolates the Q4 layout; `a1b0`/`a3b0` isolate activation layout; `a0b0` is a
new-source control against the frozen winner. There are24 lead configurations:
4 activation modes x2 weight layouts x3 unroll sizes. This is one bounded
factorial test, not24 unrelated hypotheses.

## Concrete layouts

Tiled activations:

```text
[expert][token_tile64][K32_group][k_within_group][token_within_tile]
```

A single32x32 shared transpose reads row-major A32, rounds once to A16, and
writes this layout. `a2` then widens that A16 value exactly before writing F32;
it does not silently restore the original unrounded A32 activation. Rows beyond
the valid token count are zero-filled, and invalid output rows are not stored.

The two tiled-A16 consumers differ only in delivery mapping: `a1` fetches eight
contiguous tokens/thread; `a3` fetches two groups of four tokens at different K
positions and stages each as one float4. Both preserve the sequential K order.

Q4_0 stage:

```text
Existing: half scales[64], packed_bytes[row64][byte16]
New:      half scales[64], packed_words[chunk4][row64]  (each word has4 bytes)
```

Each remains1152 bytes per64 rows x32 weights:18 bytes/G32,4.5 bits/weight
including scale. This is a byte permutation, not global dequantization. Native
low/high nibble K placement is unchanged. The worker creates the alternative
weight representation offline, just like the existing static primitive fixture.
It is **not** evidence that repacking or deployment cache misses are free.

## Actual offline evidence

- CUDA12.8/GCC14 compile to SM60 succeeds. No worker was run, including its
  disabled mode; compilation and cuobjdump are CPU-only here.
- All24 lead kernels use16KiB shared memory,100-112 registers, zero spill loads
  or stores. Tiled transpose uses38 registers and4224 bytes shared memory.
- At unroll16, tiled A16 uses100 registers with old weights and102 with new
  weights, versus103 for the new-source row-major control and106 for the frozen
  unroll8 winner. These are compiler resource reports, not measured occupancy
  or speedups.
- Static SASS at unroll16 contains512 FFMA instructions in every lead. The
  row-major bodies have64 scalar STS; tiled bodies have32 scalar STS plus8
  STS.128. These are **whole-function static counts**, not dynamic counts or
  reduced shared-memory byte traffic. `a2` reduces static HADD2.F32 conversions
  from36 to4 while increasing global activation bytes. No HFMA2/HMUL2 appears.
- Five CPU test groups pass: transpose/tails at ten token sizes1..129; complete
  activation-stage ownership/address coverage; weight permutation, nibble
  decoding and stage coverage; every finite binary16 CPU widen/roundtrip;
  source hashes, SASS, spill checks and the pre-CUDA execution guard.
- Source-derived address tests are not proof of GPU memory ordering or device
  conversion semantics. GPU memcheck/synccheck and numerical gates remain first.

Details: `build/resources.json`, `build/instruction-counts.json`,
`build/compile.log`, `build/worker.sass`, `build/manifest.json`.

## Memory and applicability caveats

Scratch count is experts * ceil(tokens/64) *64*K elements. At the prior gate/up
shape, tiled half scratch is4MiB and tiled float scratch8MiB; down is1MiB/2MiB.
Both buffers coexist in this comparison harness to avoid timed allocation;
a deployment should retain only the chosen representation. Near a token-tile
boundary padding can approach2x for33 tokens. This cost is included in the
preparation kernel and must be reported with the timing.

No extra FP16 rounding is introduced within the FP32 dot product. The inherited
synthetic fixture uses identical Q4-representable codes/scales on all controls,
not actual model Q8-to-Q4 quantization. Only Q4_0 is implemented here. Any real
Qwen comparison still needs Q4_1 affine handling, exact control operand
preparation, routing/owner costs and the applicable KLD/PPL quality gate.

## Ready-to-run handoff

Read [RUNBOOK.md](RUNBOOK.md) after the user's explicit release. The supervisor
requires a new claim and a release flag; the worker additionally rejects GPU
initialization without `--gpu-approved 1`. These are safeguards, not permission
to bypass user release. All future launches must use the shared-lock supervisor.

First memcheck all leads and synccheck persistent/tail cases, then the two
matched performance shapes. Freeze winners before repeated workers, stress
patterns and racecheck. Rank total `prep=1` time against the fastest T64 control.
No1.5x claim, integration, commit or push is warranted by these offline results.
