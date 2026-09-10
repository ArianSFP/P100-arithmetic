# P100 FIGLUT follow-up for the ~1.3x W4A4 opportunity

Date: 2026-09-09  
Status: **paused and closed for the 1.3x objective after the final bounded
experiment.**

## Answer

`P100-FIGLU-experiment` contains correct and useful LUT algebra, but it does
not provide a credible way to turn the current P100 W4A4 work into a complete
kernel that is 1.3x faster than the confirmed W16A16 baseline.

The package itself proposes W4A16: it keeps FP16 activations, builds scalar
FP32 activation tables, and performs 32 table lookups plus 32 FP32 additions
per scalar output and G32.  It is an experiment specification and CUDA helper,
not a measured GEMM.  That literal mapping is less dense than the existing
W4A4 weight-derived table and does not match the 16-token ownership of the fast
W16 kernel.

Two stronger W4A4 transfers were therefore tested:

1. Apply the package's complement symmetry to the existing packed-byte mu8
   weight table, reducing it from 256 to 128 entries.  The representation is
   exact, but runtime complement recovery reduces the prebuilt consumer to
   0.503--0.521x the matched half2 stream and gives no occupancy increase at
   the useful R8/R16 ownerships.
2. Reverse the table orientation and pack four A4 tokens into each uint32
   value.  This full16 register-LUT mapping is exact, uses live runtime weight
   keys, performs the exact G32 finish and scales, and retains persistent FP32
   totals.  It reaches only 2.996 useful TMAC/s, versus 9.070 TMAC/s required
   for gate/up and 9.488 TMAC/s for down.  It is 0.4854x its non-hoistable
   half2 component control before charging raw-A4 unpack or quantization.

The remaining symmetry use is to generate 128 canonical weight-table entries
and materialize their 128 complements, retaining the fast full256 consumer.
For the target M256 workload this still requires 4 GiB of shared stores per
projection.  Even at the theoretical GP100 shared-bank peak, those stores plus
the historical best prebuilt consumer leave about 88 us for all other gate/up
work and only 4 us for down in the straightforward sequential schedule.  With
the fresh consumer measurement, the same optimistic additive estimate already
misses the down target.

This closes the tested FIGLUT mappings as the route to the requested gain.  It
does not prove that every unknown W4A4 algorithm is incapable of 1.3x.  A new
approach should pass a streamed exact-G32 admission gate above 10.25 useful
TMAC/s, with runtime operands and live persistent totals, before another full
GEMM is attempted.

## Baseline, target and admission rate

The comparison uses the fastest confirmed pre-stored-FP16 W16A16 kernel at
M256 routed tokens per expert and 64 equal experts:

| Projection | Shape per expert | Confirmed W16A16 | 1.3x latency target | Required useful rate |
|---|---|---:|---:|---:|
| Gate/up | M256, N512, K2048 | 2462.50 us | **1894.23 us** | **9.0696 TMAC/s** |
| Down | M256, N2048, K512 | 2353.96 us | **1810.74 us** | **9.4878 TMAC/s** |

Both projections contain the same useful arithmetic count:

```
64 experts * 256 tokens * 512 * 2048 = 17,179,869,184 MACs
64 experts * 256 tokens * 2048 * 512 = 17,179,869,184 MACs
```

The 10.25 TMAC/s component admission threshold is deliberately higher than the
mathematical whole-projection rate.  It leaves some room for quantization,
planning, tails, synchronization, scaling and output that a component probe
does not include.  It is a screening threshold, not a performance prediction.

The confirmed W16 numbers come from
`../w16-dualrail-20260909/RESULTS.md`.  The earlier W4A4 checkpoint and the
original prebuilt LUT ceiling are in `../w4a4-from-w16-20260909/RESULTS.md`.

## What the supplied package establishes

Every file in `P100-FIGLU-experiment` was read and hashed.  Its CPU reference
was rerun with the supplied seed.  All assertions and all discrete claims pass;
two stored relative-L2 values differ only in the last host reduction bit, by at
most `3.3087224502121107e-23`.  The package checks exhaustive quartet identities,
lossless W4 key packing, random output-row addressing, FP32 numerical
diagnostics and an address-level shared-bank model.

The supplied `half_lut4_sm60.cuh` was compiled with CUDA 12.8.61 for SM60:

| Supplied W4A16 helper | Registers | Shared | Static instructions | SHFL | LDS | FADD | Spills |
|---|---:|---:|---:|---:|---:|---:|---:|
| Register/shuffle | 32 | 0 | 264 | 46 | 0 | 44 | 0 |
| Warp-private shared | 46 | 1024 B | 312 | 14 | 32 | 44 | 0 |

For native Q4_0 offset codes `c` and scale `d`, the package uses

```
w_j = d * (c_j - 8)
s_pj = 2*bit_p(c_j) - 1
P_p = sum_j s_pj*x_j
A   = sum_j x_j
dot = d/2 * (P0 + 2*P1 + 4*P2 + 8*P3 - A).
```

For a quartet it stores eight of sixteen signed FP32 sums.  The opposite sign
pattern is recovered by negation, so the key map is lossless.  This works well
with a dedicated sign decoder and read-accumulate hardware.  On GP100, every
lookup is an ordinary SHFL or LDS plus ordinary arithmetic.

The supplied helper performs, for one scalar output and G32:

| Work | SHFL | FP32 adds | Other FP32 |
|---|---:|---:|---:|
| Two construction waves | 8 | 6 | 0 |
| 32 lookup/accumulates | 32 | 32 | 0 |
| Activation sum | 6 | 5 | 0 |
| Plane/correction finish | 0 | 1 | 3 FMA + 1 multiply |
| Total | **46** | **44** | **3 FMA + 1 multiply** |

The shared helper replaces 32 lookup shuffles with 32 shared loads; it does not
remove those lookups or their FP32 additions.  The earlier W4A16 register LUT
already measured 52.928 us against 49.019 us for its matched direct kernel.
Round 3 later recorded 38.085 us for a direct kernel on a different GPU and in
a different experiment, so it is not a matched ratio and reinforces the need
to rerun controls.  The package correctly makes no GPU speed claim.

The package's published hardware motivation also depends on structures GP100
does not have: flip-flop LUTs with many read ports, a sign decoder, fused
read-accumulate units and a parallel tree LUT generator.  The CUDA helper
compiles to legal SM60 SASS; this study did not execute the literal helper.
The ASIC throughput mechanism cannot be imported with it.

Detailed package reproduction, hashes, arithmetic boundaries and static
instruction evidence are in `AUDIT.md` and `AUDIT-SHA256SUMS`.

## Transfer 1: exact half128 weight-derived table

The strongest prior W4A4 LUT uses one `table[256][32]` in shared memory.  Each
uint32 entry packs four output rows in byte fields, and an activation bit-plane
mask selects an entry.  One lookup therefore represents eight K positions for
four output rows.  This is eight times denser per lookup than the supplied
W4A16 scalar-FP32 orientation.

For one signed-INT4 weight row over a mu8 slice, define

```
B       = sum_j max(-w_j, 0)
U(mask) = B + sum_j bit(mask,j)*w_j
L       = sum_j abs(w_j).
```

Then

```
U(~mask) = L - U(mask).
```

Only masks whose high bit is clear need to be stored:

```
flip  = mask >> 7
index = (mask & 127) ^ (127*flip)
value = table[index]
value = flip ? Lpacked - value : value.
```

The four byte-field subtractions in each packed uint32 operation are exact
because every `U` byte is between zero and its matching `L` byte.
`verify_packed_half.py` checked:

- 20,935 four-row mu8 weight vectors;
- all 256 masks per vector;
- 21,437,440 individual row values;
- 5,359,360 packed uint32 complement subtractions;
- 80,000 normalized G32 lanes, including 13 all-minus-eight vectors.

All mismatch counts are zero.  The normalized G32 L1 bound reaches 255 without
cross-byte carry.

### Compile and residency result

The timed prebuilt variants compile without spills.  The useful ownerships do
not gain residency from halving shared storage:

| Timed consumer | Registers/thread | Shared/CTA | Maximum residency |
|---|---:|---:|---:|
| full256 R16 | 241 | 32 KiB | 1 CTA/SM |
| half128 R16 | 200 | 16 KiB | 1 CTA/SM |
| full256 R8 | 111 | 32 KiB | 2 CTAs/SM |
| half128 R8 | 96 | 16 KiB | 2 CTAs/SM |

The full R16/R8 counts above are from the timed worker.  The independent static
consumer probe uses 243/109 registers; the occupancy conclusion is unchanged.

The half table keeps the same number of LDS operations and adds canonical
address/control work to every lookup plus an `L-value` reconstruction on
flipped lookups.  The flip is warp-uniform in this weight-derived mapping, so
the conditional path does not diverge; its frequency is data-dependent and is
about half for the timed random masks.  The unrolled SASS contains a
reconstruction path for every lookup.  At R16 it has 256 lookup LDS, 128 IADD3,
265 additional IADD, 246 BFE and 524 LOP/LOP3.  The matched full table has 256
LDS, 128 IADD3, 7 additional IADD, no BFE and 32 LOP/LOP3.

Precanonicalizing `{flip,index}` in the A4 producer does not fix the consumer.
The optimistic R8 core still has 1,038 static instructions versus 654 for
full256 and reaches exactly 128 registers.  R16 has 1,692 instructions versus
960, reaches 255 registers, and spills 24 bytes of stores plus 28 bytes of
loads per thread.

### Matched GPU1 prebuilt-table gate

The worker used 112 blocks, 256 threads, 512 group iterations, a two-second
warmup, nine rotated rounds and three launches per event.  Round zero was
predefined as warm and excluded, leaving eight samples per arm.  Full and half
per-thread 64-bit checksum outputs matched bit-for-bit at R4, R8 and R16 before
timing.  The independent CPU proof establishes the internal packed identity.

| Ownership | Resident half2 | full256 | full speed | half128 | half speed |
|---|---:|---:|---:|---:|---:|
| R16, 1 CTA/SM | 9.3161 TMAC/s | 12.3309 TMAC/s | **1.3236x** | 4.6841 TMAC/s | **0.5028x** |
| R8, 2 CTAs/SM | 9.3391 TMAC/s | 10.3136 TMAC/s | 1.1043x | 4.7864 TMAC/s | **0.5125x** |
| R4, 2 CTAs/SM | 9.1711 TMAC/s | 9.2062 TMAC/s | 1.0038x | 4.7778 TMAC/s | **0.5210x** |

These are optimistic consumption ceilings: tables are already built, the long
packed accumulations may wrap, and activation-mask preparation, exact finish,
scales and output are omitted.  Half128 fails even this favorable gate.

Raw evidence is in `figlut-half128-prebuilt-r0.out` and its metadata.  Frozen
medians are in `prebuilt_summary.json`; code and final SASS are
`prebuilt_worker.cu` and `prebuilt_worker.sass`.

## Transfer 1b: materialize full256 through symmetry

Symmetry can still reduce table arithmetic: generate 128 canonical entries and
write their 128 complements.  The hot consumer then keeps its simple full256
addressing.  This is the only useful direct transfer from the package to the
weight-derived W4A4 design.

The straightforward one-table schedule does not have credible 1.3x headroom
for the target workload.  One M128/N128 CTA needs four 32 KiB table fills per
G32.  The complete traffic is:

```
gate/up CTAs = 64 experts * 2 M tiles * 4 N tiles = 512
gate/up G32s = 2048 / 32 = 64
stores       = 512 * 64 * 4 * 32 KiB = 4 GiB

down CTAs    = 64 experts * 2 M tiles * 16 N tiles = 2048
down G32s    = 512 / 32 = 16
stores       = 2048 * 16 * 4 * 32 KiB = 4 GiB
```

At the observed 1.328 GHz clock, an idealized bank-peak bound is

```
56 SM * 32 banks * 4 bytes * 1.328 GHz = 9.519104 TB/s
4 GiB / 9.519104 TB/s = 451.19 us.
```

This assumes every bank accepts a store every cycle and charges no table
arithmetic, weight decode, address work or barrier.  It is therefore far more
optimistic than a real builder.

| Consumer basis | Consumer projected to 17.18 G MAC | + ideal 4 GiB sequential stores | Gate/up margin | Down margin |
|---|---:|---:|---:|---:|
| Historical best, 12.6748 TMAC/s | 1355.44 us | **1806.63 us** | 87.60 us | **4.11 us** |
| Fresh result, 12.3309 TMAC/s | 1393.23 us | **1844.43 us** | 49.80 us | **-33.69 us** |

The table must be ready before that table's reads.  A second full table would
require 64 KiB before all other state, above GP100's 48 KiB per-block shared
limit, so ordinary full-table double buffering is unavailable.  GP100 also has
no asynchronous LUT builder.  The additive values above model the direct
build, barrier, consume schedule at an impossible peak store rate.  They are
not a universal lower bound: warp specialization or partial buffering might
overlap some builder ALU or store work with other consumption.  Such a design
would still contend for ordinary issue and shared-load/store resources and is
not implemented here.  For the direct schedule, even the historical best
leaves effectively no down-projection budget for compressed-weight decode,
table arithmetic, activation masks, G32 finish, scales, planner, tails or
output; the fresh additive estimate already exceeds the down target.

Persisting expanded tables in HBM is not a deployment solution.  A 32 KiB
table represents 1,024 four-bit weights, which occupy 512 compressed bytes:
full256 is a 64x expansion and half128 remains a 32x expansion.

## Transfer 2: activation-derived packed-four W4A4 table

The package's table orientation becomes much denser if four A4 tokens occupy
the four byte fields of a uint32.  One lane owns four output columns.  For each
token `t` and K4 quartet `q`, define

```
B_tq    = sum_j max(-a_tqj, 0)
U_tq(m) = B_tq + sum_j bit(m,j)*a_tqj
        = sum_j (bit(m,j) ? max(a_tqj,0) : max(-a_tqj,0)).
B_t     = sum_q B_tq
A_t     = sum_j a_tj over the full G32.
```

Four `U_tq` values pack into one uint32 table entry.  Two 16-entry tables occupy
the two warp halves, and four waves cover the eight quartets in G32.  For
native W4 offset codes `c=w+8`, the exact dot is

```
D_t = P0_t + 2*P1_t + 4*P2_t + 8*P3_t - (15*B_t + 8*A_t).
```

Two refinements were included in the audit:

- two returned table values can feed one IADD3, so the minimum packed
  accumulator updates are 64 rather than 128 per thread/G32;
- offline `c -> c^8` repacking gives coefficients `[1,2,4,-8]` and the shorter
  exact finish `D=P0+2P1+4P2-8P3+B`.  A packed-16 `0x8000` bias/XOR finish is
  carry- and borrow-free.

### Exactness and endpoint boundary

The project's symmetric A4 quantizer emits `[-7,7]`.  Each K4 table byte is at
most 28, and a complete G32 plane byte is at most 224.  Ordinary uint32 packed
addition therefore cannot carry between tokens.  Complement recovery is also
borrow-free.

`activation_verify.py` checks 320,144 complete output dots, full16/half8
identity, packed subtraction, the two's-complement finish and endpoint cases.
Every mismatch and bound-failure count is zero; the observed maximum plane byte
is 224.

This proof does not extend unchanged to a generic `[-8,7]` activation wire.
An all-minus-eight G32 has L1 norm 256 and can carry into the next byte.  Reuse
outside the current quantizer must enforce `[-7,7]`, normalize that exceptional
case with a scale adjustment, or dispatch a wider fallback.  W4 weights may
still contain `-8`; the bound concerns the activation-derived table values.

The full algebra, packed finish and endpoint proof are documented independently
in `activation_audit.md`, `activation_verify.py` and
`activation_verify.json`.

### Full16 versus half8 static core

The independent compile-only probe includes prepared positive/negative A4
operands, runtime weight keys, table construction, 128 lookups, paired packed
updates and a live output.  Both variants compile for SM60 without spills:

| Register table | Registers | Static instructions | SHFL | BFE | IADD3 | IADD |
|---|---:|---:|---:|---:|---:|---:|
| full16 | 112 | 576 | 160 | 97 | 52 | 42 |
| half8, precanonical keys | 55 | 882 | 152 | 113 | 56 | 170 |

Half8 saves two construction waves and eight net shuffles, but it still makes
128 lookups.  Its flip is lane-varying weight data, so it adds 128 hot
complement reconstructions and is 1.53x larger in static instructions.  The
full16 variant is therefore the simpler first performance gate.  Half8's lower
register count could improve residency, so full16 is not a formal upper bound.
Even granting half8 an ideal 2x residency benefit and discounting it by only
the observed 1.53x static-work ratio projects roughly 3.9 TMAC/s from the
full16 result, still far below 9.488 TMAC/s.  A half8 GPU run is not warranted
for this target after full16 fails by more than 3x.

### Final GPU1 gate

The timed `packed4_full16` component includes:

- precomputed positive/negative A4 magnitudes loaded at runtime;
- runtime W4 plane keys loaded and staged through shared memory;
- full16 register-table construction;
- 128 actual lookup shuffles per thread/G32;
- exact byte-plane widening and offset-code correction;
- activation and per-row weight scales;
- 16 persistent FP32 output totals per thread;
- live output stores.

It omits conversion from the raw packed A4 wire and is therefore favorable to
the candidate's instruction path.  The prepared representation uses eight
bytes/K for four tokens (positive and negative uint32 words), versus two
bytes/K for four raw A4 nibbles, so it is a 4x activation-code payload and is
not a favorable deployment format.  The worker also reuses approximately
1 MiB of prepared activation words and 0.5 MiB of weight-key words across all
112 blocks, making the inputs largely cache-resident rather than modeling a
production-size HBM stream.  The timed code uses paired lookup IADD3 but keeps
the native offset-code `15*B+8*A` correction.  The shorter
`c^8`/packed-16 finish above was CPU-proved afterward and was not timed.  A
one-block, eight-G32 device check compared all 8,192 FP32 outputs against an
independent direct integer definition and found zero bit mismatches.

The timed worker uses 112 blocks, 512 threads, 256 G32 iterations, a two-second
warmup, nine alternating rounds and three launches per event.  Real target
CTAs traverse 64 G32s for gate/up and 16 for down, so this long component loop
amortizes prologue and final output stores more heavily.  Round zero is
excluded, leaving eight samples per arm:

| Component | Registers | Shared | Median | Useful rate | Relative |
|---|---:|---:|---:|---:|---:|
| Exact-G32 half2 lower bound | 56 | 0 | 1217.93 us | 6.1713 TMAC/s | 1.0000x |
| Packed-four full16 | 79 | 2560 B | 2509.04 us | **2.9958 TMAC/s** | **0.4854x** |

The candidate's SASS has 816 static instructions, including 172 SHFL, 107 BFE,
82 IADD3, 51 IADD and 16 each of I2F, FMUL and FFMA.  There are no spills.
Pairing two lookup returns into IADD3 is already present.

The 512-thread launch gives the 79-register candidate one CTA/SM and 16 active
warps, while the control can hold two CTAs.  This is a disclosed limitation of
the component comparison.  At the same register count a 256-thread retile can
hold at most three CTAs, or 24 warps, only 1.5x the candidate's active-warps
count.  Occupancy response is not necessarily linear and no 256-thread timing
was run; a linear 1.5x recovery would reach only about 4.49 TMAC/s.  The
measured candidate is 3.17x below the stricter 9.488 TMAC/s target and would
also acquire raw-A4 preparation cost or retain a 4x larger activation payload
in a real kernel.  The available occupancy headroom does not make this mapping
a credible 1.3x route.

The first `r0` run is retained only as provenance.  Its candidate result was
the same approximately 3.0 TMAC/s, but ptxas hoisted group-invariant half2 work
and produced a false 71--73 TMAC/s control.  The source was corrected so every
group consumes runtime-varying operands.  Only `r1` and
`packed4_summary.json` are used for the relative result.

Raw data are `figlut-packed4-full16-r1.out` and its metadata.  The frozen source,
build log, SASS and counts are `packed4_worker.cu`, `packed4_build.log`,
`packed4_worker.sass` and `packed4_sass_counts.json`.

## Validation matrix and limits

| Question | Evidence | Result |
|---|---|---|
| Supplied W4A16 identities and packing | package CPU reference rerun | PASS |
| Supplied header lowers on SM60 | CUDA 12.8.61 compile and SASS | PASS, zero spills |
| Weight-derived half128 identity | exhaustive/random packed CPU proof | PASS |
| full256/half128 device equivalence | R4/R8/R16 pre-timing comparison | bit-identical |
| Activation-derived full16/half8 identity | 320,144-dot CPU proof | PASS |
| Packed-four device definition | 8,192 FP32 outputs, eight G32s | bit-identical |
| Corrected component timing | eight retained interleaved samples | PASS, candidate rejected |
| Complete GEMM | admission gate failed | not built |
| Compute Sanitizer on complete kernel | no promoted complete kernel | not run |
| PPL/KLD model quality | performance gate failed | not run |

The GPU checks establish exactness for the defined synthetic quantized-code
equation and operation order.  They do not establish model quality.  Any future
candidate still needs byte-identical output or the project KLD-pair PPL delta
within +/-0.003.

## Decision at pause

The following paths are closed for the requested 1.3x result unless materially
new evidence changes the cost:

1. literal package W4A16 scalar-FP32 half-LUT in the fast M128 W4A4/W16
   ownership;
2. weight-derived half128 shared tables with runtime complement recovery;
3. moving half128 canonicalization into the activation producer;
4. activation-derived packed-four half8 register tables;
5. activation-derived packed-four full16 register tables;
6. the direct one-table, sequential symmetry-built full256 schedule for the
   M256 target shapes;
7. persisting 32x/64x expanded tables in HBM.

The complement identity and the packed-four exactness proof remain useful
building blocks.  They are not sufficient performance mechanisms on GP100.
A mu6 variant can reduce the package-style lookup count from 32 to 24 per G32,
at most 25%, far short of the measured 3.17x deficit before considering its
larger construction cost.

No production/model source was edited.  No complete GEMM, model run, commit,
push, PR or GPU reset was performed.  GPU1 was the only device used and is
released in both coordination logs after a final health check.  The exact
release snapshot is retained in `gpu-release.json`.

## Reproduction map

CPU-only checks and frozen analysis can be rerun from the repository root:

```sh
python3 P100-FIGLU-experiment/p100_figlut_experiment/reference_test.py \
  --json /tmp/figlut-reference.json
python3 studies/figlut-symmetry-20260909/verify_packed_half.py
python3 studies/figlut-symmetry-20260909/activation_verify.py
python3 studies/figlut-symmetry-20260909/audit_prebuilt_sass.py
python3 studies/figlut-symmetry-20260909/audit_packed4_sass.py
python3 studies/figlut-symmetry-20260909/audit_activation_sass.py
python3 studies/figlut-symmetry-20260909/analyze_prebuilt.py
python3 studies/figlut-symmetry-20260909/analyze_packed4.py
python3 studies/figlut-symmetry-20260909/derive_bounds.py
python3 studies/figlut-symmetry-20260909/verify_checkpoint.py
python3 studies/figlut-symmetry-20260909/checkpoint_manifest.py
```

The GPU runners intentionally require an active coordination reservation,
exclusive GPU1 lock, the recorded UUID and `--gpu-approved 1`.  The study is
paused, so they should not be rerun without a new coordinated reason.

`CHECKPOINT-MANIFEST.json` records the size and SHA-256 digest of every study
file.  Regenerate it only after an intentional documentation or artifact
change with `checkpoint_manifest.py --write`; the command shown above verifies
the frozen inventory without modifying it.
