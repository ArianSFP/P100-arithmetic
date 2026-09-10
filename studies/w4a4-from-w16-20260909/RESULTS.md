# W4A4 from the fast W16A16 kernel — paused checkpoint, 2026-09-09

Status: **paused at the user's request. The requested 2x speedup was not
reached.** All three suggestion levels in `W4A4-research` received their
planned admission test. None of the tested realizations has enough measured
headroom to justify a full production kernel. GPU1 was used exclusively for this study and released at
the checkpoint; no model or production source was changed, and nothing was
committed or pushed.

## Reference and target

The strongest comparison is the confirmed pre-stored-FP16 W16A16 result at
M256, 64 equal experts:

| Projection | Confirmed W16A16 | Required W4A4 for 2x |
|---|---:|---:|
| Gate/up, N512 K2048 | 2462.50 us | 1231.25 us |
| Down, N2048 K512 | 2353.96 us | 1176.98 us |

The matched W16 arm in this study measured 2459.97 us and 2348.57 us on GPU1,
consistent with that reference. The serving-relevant F32-wire W16 result is
slightly slower (2528.34/2411.37 us), but an A4 implementation would also need
to quantize that wire. Using the fastest FP16-storage W16 baseline is the clean
arithmetic and storage comparison.

Static W4 weight preparation remains offline, exactly as static W16 weight
preparation does. Complete W4 timings include GPU FP16-to-A4 quantization, GPU
tile-list compaction, leftover and paired compute launches, per-G32 scales and
output. Arithmetic correctness is exact for the defined quantized-code
equation. This does not establish that A4 activations meet model quality.

## Priority 1: exact half2 and packed arithmetic

### Idealized arithmetic ceilings

The first screen uses 112 blocks, 256 threads, eight independent chains, a
two-second warmup, nine rotated rounds and three launches per event. Round zero
is predefined as warm. Useful throughput counts the independent scalar INT4
MACs represented by each packed instruction. Decode, extraction, group scales,
activation quantization, planning, memory staging and output are absent, making
these optimistic ceilings rather than GEMM predictions.

| Arithmetic stream | Median useful TMAC/s | Relative to half2 |
|---|---:|---:|
| Ordinary HFMA2 | 6.7355 | 1.0000x |
| Centered signed-16 XMAD pack2 | 7.0931 | 1.0531x |
| FP32 pack2 | 7.0825 | 1.0515x |
| FP64 pack4 | 8.7493 | 1.2990x |
| Mixed HFMA2 + FP64 | 9.1283 | **1.3553x** |

CUDA 12.8 lowers each explicit `mad.wide.s16` arithmetic operation to one
`XMAD.S16.S16` on SM60. The mixed stream is the best priority-1 result, but its
1.355x ideal rate is already below 2x before paying any of the omitted work.
This closes centered INT16, FP32/FP64 packing and mixed FP16/FP64 issue as
routes to the requested target on this P100.

Raw data and frozen code are in `priority1-throughput.out`,
`priority1-throughput.meta.json`, `priority1-throughput-summary.json`,
`microbench.cu`, `microbench-manifest.json` and
`microbench-sass-counts.json`.

### Complete exact-G32 half2 control

`baseline-half2/` ports exact signed-code W4A4 arithmetic into the committed
W16 M128 pair geometry. It keeps 256 threads, 16 token rows per warp, four
output columns per lane, a double-buffered K32 stage and the existing GPU tile
compactor. Each raw dot resets at G32, where every prefix is an exactly
representable integer in FP16; the kernel widens the result and applies the A4
and per-row weight scales with FP32 FMA.

The W4 layout uses 1,152 bytes per M64/K32 weight stage (128 bytes of FP16
scales plus 1,024 bytes of packed nibbles), versus 4,096 bytes for W16. Packed
activations use 16 code bytes plus a four-byte FP32 scale per row/G32, versus
64 bytes of FP16 activations. That delivery reduction does not reduce the
1,024 HFMA2 main-loop instructions.

| Specialized M128 kernel | Registers | Shared | Static instructions | HFMA2 | FFMA | Spills |
|---|---:|---:|---:|---:|---:|---:|
| W16A16 | 126 | 32,768 B | 1,872 | 1,024 | 0 | 0 |
| Exact-G32 W4A4, down | 144 | 34,816 B | 2,130 | 1,024 | 64 | 0 |
| Exact-G32 W4A4, gate/up | 144 | 34,816 B | 2,136 | 1,024 | 64 | 0 |

The extra scale storage crosses 32 KiB and the W4 kernel is compiled for one
CTA per SM. The selected 112-block persistent grid remains slightly faster for
W4 than 56 blocks and preserves the fastest W16 comparison.

| Projection | W16A16, us | Exact W4A4, us | W4 speed vs W16 | 2x target, us | W4 / target |
|---|---:|---:|---:|---:|---:|
| Gate/up | 2459.97 | **4522.31** | 0.5440x | 1229.98 | 3.6767x |
| Down | 2348.57 | **3629.11** | 0.6471x | 1174.28 | 3.0905x |

These are single-worker directional screens because the candidate is a large
loss, not a near winner. Each number is the median of six retained samples
after the predefined round zero; variation is very small in the raw outputs.
The 56-block W4 medians are 4535.94 us gate/up and 3630.50 us down, confirming
that persistent-grid choice does not explain the rejection.

Validation completed before timing:

- Independent CPU checks pass all 256 packed-byte decodes, 3.2 million bounded
  G32 prefix transitions including the signed `-8` domain, all 64 rows of the
  word-major tiled-weight layout, and 10,000 activation groups.
- M33/N128/K96 partial-tile semantics: every device activation code and scale
  matches the CPU result; planner coverage is exact; all 8,448 outputs are
  finite; 256 sampled complete CPU dots pass for each arm.
- M33 memcheck reports zero errors.
- M129 mixed M128-pair plus leftover racecheck reports zero errors and zero
  warnings.
- Each large projection checks every activation code/scale, planner coverage,
  all outputs for finiteness and 256 independent complete CPU dots per arm.
  Gate/up checks 1,048,576 activation groups and 8,388,608 outputs; down checks
  262,144 groups and 33,554,432 outputs.

The exact control is therefore a valid performance result for the specified
post-quantization equation. It is not a model-accuracy result.

## Priority 2: normalized split and PRMT residue repair

The level-2 screen compares 64 useful scalar INT4 MACs per thread-iteration.
Each mode uses eight independent packed chains and a dynamic positive/negative
activation pair, whose exact cancellation is checked for every output before
timing. Launch geometry is 112x256, with two CTAs per SM. All modes have zero
spills and zero local-memory instructions.

| Mode | SM60 inner work / 64 useful MACs | Registers | Useful TMAC/s | Speed vs plain |
|---|---|---:|---:|---:|
| Plain exact half2 | 32 HFMA2 | 64 | 5.6118 | 1.0000x |
| Normalized split | 32 HFMA2 + 48 HADD2 | 56 | 3.2233 | **0.5744x** |
| HMUL2 + PRMT repair | 16 HMUL2 + 32 F2I + 8 PRMT + integer repair | 121 | 0.8669 | **0.1545x** |

The repair loop also contains four dynamic constant-table loads. Its complete
backward loop has 344 static instructions, versus 95 for normalized split and
55 for plain half2. Ptxas places 16 half-word merge instructions in the plain
loop and only eight in each packed loop, so this staged-operand comparison is
already favorable to the candidates. Neither can approach 2x.

Artifacts are `level2-throughput.cu`, `level2-throughput-README.md`,
`level2-throughput-manifest.json`, `level2-throughput-sass-counts.json`,
`priority2-throughput.out`, `priority2-throughput.meta.json` and
`priority2-throughput-summary.json`.

## Priority 3: four-output weight-derived LUT

The proposed layout is `table[256][32]`: one 1 KiB four-output table per lane,
32 KiB per CTA. Address `table[activation_mask][lane]` fixes bank ownership to
the lane. One G32 consumes four mu8 slices and four activation planes. The
diagnostic compares the resulting LDS/packed-IADD stream with the same amount
of useful half2 work.

The first R16/two-CTA build hits the 128-register launch bound and spills 216
bytes of stores plus 220 bytes of loads per thread; it reaches only 0.8682x its
matched resident half2 stream. A second build uses slice-major traversal and
warp-uniform activation masks, verifies one `LDS.U.32` per lookup, and evaluates
the viable ownership widths:

| LUT ownership | Registers | Spills | Useful TMAC/s | Speed vs matched half2 |
|---|---:|---:|---:|---:|
| R16, one CTA/SM | 243 | 0 | 12.6748 | **1.3609x** |
| R8, two CTAs/SM | 109 | 0 | 10.3543 | **1.1090x** |
| R8, lane-divergent control | 117 | 0 | 10.2487 | 1.0977x |
| R4, two CTAs/SM | 64 | 0 | 9.1778 | **1.0014x** |

This is deliberately more optimistic than a valid kernel. The table is already
built and resident. Packed byte fields may wrap across the long timing loop.
The screen omits activation-mask preparation, accumulator reset, byte widening,
plane combination, signed correction, exceptional all-`-8` handling, per-G32
scaling and output. Most significantly, it omits construction of four 32 KiB
tables and at least eight CTA barriers for every G32.

Even the 243-register, one-CTA R16 issued-work ceiling reaches only 1.361x.
R8 is the plausible full-kernel register shape, but its 1.109x ceiling leaves
only 19 registers below the two-CTA bound before adding 32 persistent FP32
totals and all real kernel state. R8 and R4 also require two and four times as
many table builds per output as R16, while the table still caps residency at
two CTAs. Since the prebuilt consumer misses 2x, full table construction was
correctly not implemented.

Artifacts are `lut_consumer.cu`, `lut-consumer-manifest.json`,
`lut-consumer-sass-counts.json`, the two
`priority3-prebuilt-lut-consumer*.out/meta.json` pairs and their summary JSON.

## Decision boundary at pause

The tested admission paths are closed for the 2x objective:

1. Centered integer, FP32, FP64 and mixed packed arithmetic top out at 1.355x
   in idealized issue tests.
2. Exact normalized splitting and the smaller PRMT/mod-8 repair are slower than
   ordinary half2.
3. The tested prebuilt mu8, warp-uniform LUT consumer tops out at 1.361x before
   mandatory construction and finishing work. Alternate mu4/mu5 layouts were
   not measured in this checkpoint.
4. The complete ordinary exact-G32 W4A4 pipeline is 1.84x slower than W16 on
   gate/up and 1.55x slower on down.

This checkpoint does not claim that every possible W4A4 algorithm is
impossible. It does show that none of the tested realizations has the minimum
arithmetic headroom required to double the committed W16 kernel on GP100.
Earlier materialized bit-plane/popcount kernels remain closed negative
controls: their long-prefill result was 10.381/9.863 ms and they do not provide
new evidence against the much faster W16 baseline.

If work resumes, the useful next objective is narrower: improve the ordinary
exact W4A4 control, without expecting 2x. Candidate engineering work is an M64
token tile or a compressed shared-memory operand stage that restores two-CTA
occupancy, followed by component timing of quantization and G32 finishing. A
new route toward 2x should first demonstrate more than 2x on a realistic,
matched resident issue test; otherwise full GEMM and model integration are not
worth the GPU cost.

No PPL/KLD run was started because no performance candidate passed its admission
gate. Any future production candidate still requires the project threshold of
PPL delta within +/-0.003 using the KLD-pair methodology.
