# P100 FIGLUT package audit and W4A4 transfer analysis

Date: 2026-09-09  
Scope: read every artifact in `P100-FIGLU-experiment`, reproduce its CPU
checks, compile its supplied CUDA header for SM60, and compare the mechanism
with the priority-3 W4A4 table. No GPU was used for this audit. The separately
generated `prebuilt_*` files in this study were produced by the parent GPU
worker and are cited only as external measured evidence.

## Finding

The package is a sound starter for a **W4A16 activation-derived FP32 LUT**
experiment. It is not a W4A4 kernel and it does not establish any GPU speedup.
Its code-preserving algebra and canonical key mapping are correct. The supplied
header compiles cleanly for SM60 without spills.

It does not provide a credible direct route to retaining the earlier ~1.3x
prebuilt W4A4 ceiling. The CUDA mapping performs 32 scalar lookups and 32 FP32
accumulations for one output's G32 group, before construction, correction,
scale, and persistent output accumulation. The priority-3 W4A4 orientation is
far denser: each mu8 shared lookup returns four output rows in packed bytes.

The useful transferable idea is vertical complement symmetry. It can halve a
priority-3 table from 32 KiB to 16 KiB exactly. On GP100, however, the half
table needs runtime complement reconstruction. Static SM60 compilation shows
that reconstruction adds substantial integer/key-selection work while leaving
the number of shared loads unchanged. It also does not increase CTA residency
for the viable R8 and R16 ownerships, because registers are already limiting.
The parent worker's matched, prebuilt hardware result confirms the consequence:
half128 reaches only 0.503x, 0.513x, and 0.521x of resident half2 at R16, R8,
and R4, while full256 reaches 1.324x, 1.104x, and 1.004x respectively.

The best use of symmetry on this GPU is therefore different from the ASIC:
construct the canonical 128 entries, immediately write their 128 complements,
and consume a full 256-entry table without per-lookup decoding. This reduces
construction relative to an independently generated full table. It cannot
preserve a 1.3x complete-kernel speedup: the only prebuilt arm above 1.3x is
R16 at 1.324x, leaving about 1.8% latency margin for all table construction,
barriers, weight decoding, activation masks, finishing, scales, and output.

## Package inventory and reproduction

Every package artifact was read:

| Artifact | SHA-256 | Role |
|---|---|---|
| `Suggestion-1-FIGLUT.txt` | `b8d27f67039788e024f777fa5faa733bea0cb17d06530446417227b758a4c875` | proposal and staged experiment brief |
| `p100_figlut_experiment/README.md` | `cf57a43c327630b29fde548be3a8968f402667a2d662f2fa357e1d9f4d81ce45` | full specification and caveats |
| `p100_figlut_experiment/reference_test.py` | `2e51507b764c8b4abe84e1857f8c298989148e7a599d5fb12fca4c30e06e9de4` | CPU algebra, packing, addressing, numerical, and bank-model checks |
| `p100_figlut_experiment/cpu_results.json` | `fbfbf8d4fec6ce3502a0da692fbf69e74bf596f34ed91ab514c0a5a9f4e74144` | recorded CPU result |
| `p100_figlut_experiment/half_lut4_sm60.cuh` | `5c22964e1db71b5e116f3348a8b8dfd284458e231d59fc871c26d900c1eab3af` | shuffle/shared arithmetic components, not a GEMM |

`reference_test.py` was rerun with its default seed. All assertions and all
discrete recorded claims reproduce. Two reported relative-L2 values differ in
the final host floating-point reduction bit by at most
`3.3087224502121107e-23`; see `figlut_reference_comparison.json`. Reproduced
output is retained in `figlut_reference_reproduction.json`.

The reproduced checks cover:

- every 2-, 3-, and 4-bit quartet pattern under six fixed activation probes;
- bijective pack/unpack for 65,536 random W4 G32 code blocks;
- 4,096 random output-row instances of the four-table/two-wave addressing;
- FP32 error diagnostics for normal, wide finite FP16, and cancellation cases;
- an address-level model of the contiguous 32-word shared layout.

They do not cover a GPU operation-order oracle, tails in a complete GEMM,
device memory safety, runtime, a model-quality comparison, or W4A4 activation
quantization. The package states these limits accurately.

## Proposed package algorithm

For native Q4_0 offset codes `c` and scale `d`,

```
w_j = d * (c_j - 8)
s_pj = 2 * bit_p(c_j) - 1
P_p = sum_j s_pj * x_j
A = sum_j x_j
group = d/2 * (P_0 + 2 P_1 + 4 P_2 + 8 P_3 - A)
```

The `-A` term is necessary because
`sum_p 2^p s_pj = 2*c_j - 15`, not `2*(c_j-8)`.

For each quartet of FP16 activations widened to FP32, the package stores eight
of the sixteen signed sums:

```
H[i] = sign(i0)*x0 + sign(i1)*x1 + sign(i2)*x2 - x3
```

For the original four-bit sign pattern `p`, it stores this lossless key in the
repacked weights:

```
flip  = p >> 3
index = (p & 7) ^ (flip * 7)
key   = index | (flip << 3)
```

At runtime, `H[index]` is negated when `flip` is set. Four canonical plane
words plus the unchanged FP16 scale use 18 bytes/G32, the same 4.5 bits/weight
as the native Q4_0 code-plus-scale representation.

One warp distributes four eight-entry FP32 tables across its 32 lanes. Two
waves cover eight quartets, hence G32. Each lane is also an output owner. For
each output and G32 it performs four bit-plane lookups for each of eight
quartets: 32 lookups and 32 adds. Table construction is shared across output
reuse `R`, but lookup count is not reduced by half-table symmetry.

The shared variant stores the same four tables as 32 contiguous FP32 words per
warp. For a fixed table `t`, address `8*t+index` is in a unique bank for every
distinct index; equal indices are broadcasts. The package's bank-model result
is correct as an address claim. It is not a latency result.

## Exactness boundary

The canonical map is a bijection and the real-arithmetic identity is exact for
the stated Q4_0 code-minus-eight format. The package does not change the W4
codes or the FP16 scale bits.

The CUDA arithmetic is not operation-order identical to the current kernels:

- each table entry uses a fixed sequential FP32 addition order;
- plane partials accumulate separately;
- activation sum uses a warp reduction tree;
- plane combination uses three explicit FP32 FMAs and a final correction;
- the supplied helper returns an unscaled group result and does not update a
  persistent output accumulator.

Consequently, code identity does not imply output byte identity. Its CPU FP32
diagnostic is not an oracle for the header's GPU operation order, and its small
relative errors are not a model PPL/KLD result. The project's PPL +/-0.003 gate
would still be required after a performance win.

For the current research objective there is a larger semantic mismatch: the
package deliberately retains A16. A W4A4 adaptation must add A4 quantization,
activation scales, exact endpoint handling, and their pipeline cost.

## Supplied-header SM60 audit

`figlut_header_probe.cu` wraps the supplied R1 helpers without changing their
arithmetic. CUDA 12.8.61 compiled both for `sm_60` with zero spills:

| Supplied helper | Registers | Shared | Static instructions | SHFL | LDS | STS | FADD | FFMA | FMUL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shuffle | 32 | 0 | 264 | 46 | 0 | 0 | 44 | 3 | 1 |
| shared | 46 | 1,024 B | 312 | 14 | 32 | 2 | 44 | 3 | 1 |

The counts include wrapper load/store/address setup. The algorithmic breakdown
for one G32/output is more informative:

| Work | Shuffle instructions | FP32 adds | Other FP32 |
|---|---:|---:|---:|
| two table-build waves | 8 | 6 | 0 |
| 32 lookup/accumulates | 32 | 32 | 0 |
| activation sum | 6 | 5 | 0 |
| plane/correction combine | 0 | 1 | 3 FMA + 1 multiply |
| total in shuffle helper | 46 | 44 | 3 FMA + 1 multiply |

The shared helper substitutes 32 LDS for the 32 lookup SHFL instructions but
still needs table-construction and activation-sum shuffles. The source contains
four `__syncwarp` calls; CUDA emits no `BAR` instruction for these convergent
full-warp calls on SM60. The helper also has 32 sign-bit logic operations and
substantial key extraction/address logic.

For comparison, direct half2 arithmetic needs 16 HFMA2 instructions for 32
scalar MACs. Output-row reuse amortizes the 14 shuffle/11-add construction and
sum overhead, but it does not amortize the 32 lookup shuffles/LDS and 32 FP32
adds per output. Weight decode savings would have to overcome that issue gap.
The archived W4A16 register-LUT already lost its matched test, so this package
properly asks for a new measurement rather than claiming a win.

## Difference from the priority-3 W4A4 LUT

| Property | Package half-LUT | Priority-3 W4A4 LUT |
|---|---|---|
| Intended format | W4A16 | W4A4 |
| Table is derived from | activations | weights |
| `mu` | 4 | 8 |
| Value | one FP32 partial | four output rows in four unsigned bytes |
| Table placement | one register entry/lane or 32 FP32 shared words/warp | `uint32 table[256][32]`, 32 KiB/CTA |
| Runtime key | offline weight quartet key | activation bit-plane mask |
| G32 lookups | 32 per scalar output | 16 per four packed output rows/token |
| Effective W4 MACs per lookup | 1 | 8 |
| Mandatory finish | FP32 plane combine, `-A`, W scale | byte widening, plane combine, signed correction, endpoint factor, A/W scales |
| Measured P100 evidence | none in package | prebuilt ceiling measured; complete construction omitted |

The priority-3 orientation is eight times denser per lookup because one access
covers eight K positions for four output rows. The package's small register LUT
has cheaper storage, but its scalar FP32 values cannot pack four independent
outputs in one 32-bit word.

## Exact half128 transfer to the W4A4 table

For a signed-INT4 weight row over one mu8 slice, define

```
B       = sum_j max(-w_j, 0)
U(mask) = B + sum_j bit(mask,j) * w_j
C       = sum_j abs(w_j)
```

Then

```
U(~mask) = C - U(mask).
```

The proof follows from `sum_j w_j + 2B = C`. Store only masks with bit 7
clear. For any runtime mask:

```
flip  = mask >> 7
index = (mask & 127) ^ (flip * 127)
value = table[index]
value = flip ? C - value : value
```

Four independent rows can remain packed in bytes. For every lane,
`0 <= value <= C <= 64`, so `Cpacked-value` has no borrow between bytes and an
ordinary uint32 subtraction is exact.

`verify_packed_half.py` checks 20,935 four-row mu8 weight vectors, all 256 masks
for each vector, 21,437,440 individual row values, and 5,359,360 packed uint32
subtractions with zero mismatches. It also verifies 80,000 G32 lanes through
four packed slice accumulations after exact all-minus-eight normalization;
the normalized G32 L1 bound reaches 255 and no packed carry occurs.

This proves the representation. It does not prove that reconstruction is fast.

## Construction and consumption cost

`static_probe.cu` models an optimistic builder whose positive/negative packed
byte contributions have already been decoded from compressed W4 weights. Each
of eight warps owns 16 canonical masks for all 32 table columns and follows a
four-bit Gray sequence. With `C` already available:

| Per thread, one mu8 slice | half128 | full256 via symmetry |
|---|---:|---:|
| canonical entries generated | 16 | 16 |
| initial packed additions | 7 | 7 |
| Gray transitions | 15 | 15 |
| source-level subtract/add operations for transitions | 30 | 30 |
| complement subtractions | 0 | 16 |
| shared stores | 16 | 32 |
| prepared contribution loads | 16 | 16 |

Ptxas fuses each safe `value-old+new` transition where possible into IADD3.
The actual one-slice SM60 resources/counts are:

| Builder | Registers | Shared | Static total | LDG | STS | IADD + IADD3 | BAR | Spills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| half128 | 23 | 16,384 B | 96 | 16 | 16 | 25 | 1 | 0 |
| full256 via symmetry | 28 | 32,768 B | 168 | 16 | 32 | 65 | 1 | 0 |

These are optimistic: compressed-nibble decode, contribution preparation,
normalization metadata, table/consumer handoff, and three additional mu8 slices
are absent. A G32 needs four table builds. At the simple operation-count level,
half128 saves 64 complement subtractions and 64 shared stores per thread/G32
against symmetry-built full256.

That saving is smaller than its consumer penalty. A G32 consumer performs
`16*R` lookups per thread. Half128 adds one complement subtraction and selection
for every lookup, plus canonical index logic. At R8 it saves 64 builder
subtractions but adds 128 decode subtractions; at R16 it adds 256. Ptxas confirms
the extra work:

| Prebuilt consumer | Registers | Shared | Static total | LDS | IADD | IADD3 | BFE | LOP + LOP3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full256 R8 | 109 | 32,768 B | 654 | 128 | 8 | 64 | 0 | 16 |
| half128 R8 | 96 | 16,512 B | 1,236 | 129* | 138 | 64 | 128 | 272 |
| full256 R16 | 243 | 32,768 B | 960 | 256 | 8 | 128 | 0 | 32 |
| half128 R16 | 200 | 16,512 B | 2,304 | 257* | 266 | 128 | 246 | 524 |

`*` includes one invariant load of the packed per-lane `C`; lookup LDS count is
unchanged. Static instruction totals are not cycle predictions, but they expose
why half storage does not imply half execution time.

The separately measured prebuilt result agrees: half128 is about half as fast
as the matched resident half2 stream for all tested ownership widths. Full256
R16 is the only arm around 1.3x.

### Can grouping or upstream canonicalization remove the decoder cost?

No tested/software-realizable grouping removes most of it:

1. **Canonicalize activation masks upstream.** An optimistic static variant
   treats every input byte as an already formed `{flip,index}` key. This removes
   the complement-XOR calculation from lookup, but each access still needs an
   index/address extraction, `C-value`, selection, and accumulation. R8 falls
   only from 1,236 to 1,038 static instructions, still far above full256's 654;
   it reaches the 128-register two-CTA limit. R16 falls from 2,304 to 1,692
   versus full256's 960 and compiles at 255 registers with 24-byte stores and
   28-byte loads spilled per thread. Moving key work into the A4 quantizer also
   moves cost rather than removing it from the complete pipeline.

2. **Branch on `flip`.** The mask is warp-uniform in the priority-3 ownership,
   so a branch need not diverge. At best it skips `C-value` for roughly half of
   masks. It still performs the canonical address work and control transfer,
   while weight patterns can make the branch ratio arbitrary. Reaching 1.30x
   from a 1.324x prebuilt ceiling requires removing essentially all decoder
   overhead, not half of one instruction.

3. **Defer complement correction.** The identity can be accumulated as
   positive canonical sums, negative canonical sums, and flipped `C` terms.
   This needs additional packed accumulators and at least one correction update
   for flipped values, worsening the current register limit and merely moving
   the subtraction to the finish.

4. **Materialize only requested complements.** Computing compact decoded
   values for the observed token/plane masks adds stores, reloads, and a
   synchronization. Each table column has a different `C` and value, so no
   single lane can generate a shared scalar correction for the warp.

5. **Encode negation as XOR.** This is cheap for the package's one FP32 value,
   because IEEE sign negation is a sign-bit XOR. The W4A4 table packs four
   unsigned byte fields with four different `C` values. Their transform is
   lane-wise subtraction, not a common XOR. Signed/sign-magnitude bytes would
   then cease to support exact ordinary packed-IADD accumulation. GP100's PTX
   `vadd4` is compiler-expanded rather than a one-instruction four-byte fix.

6. **Store both forms.** Keeping `U` and `C-U`, whether as two 32-bit words or
   one 64-bit entry, is the full 32 KiB information again and still needs a
   selection unless laid out as the original 256-entry table.

The only clean way to eliminate the hot decoder is to write all complements
during construction and use the existing full table. FIGLUT's ASIC eliminates
this cost with a dedicated decoder/multiplexer; P100 has no equivalent unit.

## GP100 residency

GP100 provides 65,536 32-bit registers and 64 KiB shared memory per SM, with a
48 KiB per-block shared-memory limit. For 256-thread CTAs:

| Consumer | Registers/block | Register cap | Shared cap | Resulting maximum |
|---|---:|---:|---:|---:|
| full256 R16, 243 regs | 62,208 | 1 CTA | 2 CTAs at exactly 32 KiB | 1 CTA |
| half128 R16, 200 regs | 51,200 | 1 CTA | 3 CTAs at 16,512 B | 1 CTA |
| full256 R8, 109 regs | 27,904 | 2 CTAs | 2 CTAs at exactly 32 KiB | 2 CTAs |
| half128 R8, 96 regs | 24,576 | 2 CTAs | 3 CTAs at 16,512 B | 2 CTAs |

Thus half128 does not raise occupancy at R8 or R16. It creates shared-memory
headroom for other state. A nominal pair of 16 KiB buffers is exactly 32 KiB,
but an additional 128-byte shared `C` array makes two such CTAs exceed 64 KiB;
`C` must remain in registers or be placed within existing storage to preserve
two-CTA residency.

The synthetic R8 half consumer has 32 registers of headroom up to the 128-reg
two-CTA boundary, exactly the number of persistent FP32 totals required for
eight token rows and four output rows. That leaves no margin for a complete
kernel. R16 has 55 registers up to the architectural 255-register/thread limit,
less than its 64 persistent FP32 totals. Scheduling or ownership changes could
alter those counts, but the current mapping is not directly completable without
more register work.

## Why the complete path cannot retain 1.3x

The fresh full256 R16 prebuilt ceiling is 1.3236x versus matched resident
half2. To finish at 1.30x, every omitted stage combined may add at most

```
1.3236 / 1.30 - 1 = 1.82%
```

to the prebuilt candidate time.

Even an ideal half128 builder must write 64 shared words per thread/G32 across
four mu8 slices; the R16 consumer performs 256 shared lookup loads. Those stores
alone are 25% of the consumer's shared-operation count, before table arithmetic,
weight decoding, synchronization, activation-mask work, byte widening, plane
combination, scaling, and output accumulation. Full256 requires 128 stores,
50% of that lookup count. GP100 has no asynchronous shared-memory table-builder
unit; overlap would consume ordinary warp issue and LSU resources.

The half table cannot solve this by reducing consumer cost: its measured
prebuilt path is already a loss. The full table retains the prebuilt 1.324x but
has far more than a 1.82% mandatory construction/finish bill. Persisting expanded
tables in HBM is also unsuitable: one 32 KiB full table describes only
`128*8 = 1,024` four-bit weights, a 64x expansion over their 512 compressed
bytes; half128 is still a 32x expansion.

The result does not prove every LUT organization impossible. It does close the
specific FIGLUT half-table transfer as a way to realize the measured ~1.3x
ceiling. Symmetry-assisted **full-table construction** is the remaining useful
engineering ingredient, but it does not provide enough missing headroom on its
own to justify a complete GEMM under the current admission target.

## Hardware behind the paper's result

The CUDA package uses only documented SM60 operations and needs no custom
opcode. That makes it executable on P100, but the FIGLUT paper's efficiency
results come from hardware P100 does not contain:

- flip-flop LUTs with dedicated multiplexers and many concurrent read ports;
- a dedicated sign decoder for the half-LUT;
- read-accumulate (RAC) units that fuse table access with accumulation;
- a tree LUT generator that produces entries in parallel;
- 32 RACs sharing each LUT;
- a weight-stationary systolic array, on-chip buffers, double buffering, and a
  vector processing unit;
- a synthesized 100 MHz, 28 nm ASIC evaluation, not a GPU kernel result.

The paper selects `mu=4`; it explicitly excludes `mu=8` from its chosen hardware
because of LUT size/power. Its headline results are TOPS/W/area comparisons for
the custom architecture, not evidence of lower P100 latency. Primary source:
<https://arxiv.org/html/2503.06862v1>.

## Artifacts

- `verify_packed_half.py`, `verify_packed_half.json`: exact half-table transfer
  proof checks.
- `figlut_header_probe.cu`, cubin, SASS, build log, resource and count JSON:
  compilation audit of the package header.
- `static_probe.cu`, cubin, SASS, build log, resource and count JSON: optimistic
  half/full builders and prebuilt consumers.
- `figlut_reference_reproduction.*`, `figlut_reference_comparison.json`: package
  CPU reproduction.
