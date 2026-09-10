# Independent audit: four-token activation-derived W4A4 register LUT

Date: 2026-09-09.  CPU proof and SM60 compile audit only; this audit did not
run a GPU.  The parent integration worker's already-recorded GPU result is
used below as external measured evidence.

## Conclusion

The packed-four-token algebra is exact under the production symmetric A4
contract (`-7..7`).  Full16 is the correct software variant to gate.  Half8
is exact but adds a hot, lane-varying complement decode to every lookup and is
substantially larger in compiled SM60 instruction count.  The completed full16
resident gate already measured 2.996 useful TMAC/s, below both mathematical
1.3x targets (9.070 gate/up and 9.488 down); the route is closed for this goal.

## Exact algebra

For token `t`, quartet `q`, and activation codes `a[t,j]`, define

```
B[t,q]    = sum_j max(-a[t,j], 0)
U[t,q](m) = B[t,q] + sum_j bit(m,j)*a[t,j]
           = sum_j (bit(m,j) ? max(a[t,j],0) : max(-a[t,j],0))
T[q](m)   = sum_t U[t,q](m) << (8*t)
```

Thus every source operand to the packed word addition is nonnegative.  The
production quantizer clamps every activation code to `[-7,7]`, so

```
0 <= U[t,q](m) <= 28
sum_q U[t,q](m_q) <= sum_j abs(a[t,j]) <= 224.
```

Ordinary uint32 additions therefore cannot carry between byte fields.  For
native offset weight code `c=w+8`, let `m[p,q]` be the quartet mask of bit plane
`p` and accumulate `P[p] = sum_q T[q](m[p,q])`.  With
`B[t]=sum_q B[t,q]` and `A[t]=sum_j a[t,j]`, each scalar result is

```
D[t] = P0[t] + 2*P1[t] + 4*P2[t] + 8*P3[t]
       - (15*B[t] + 8*A[t]).
```

This is exactly `sum_j a[t,j]*(c[j]-8)`.

Half8 stores masks with bit 3 clear.  Let `L[t,q]=sum_j abs(a[t,j])`.  Then

```
U[t,q](~m) = L[t,q] - U[t,q](m)
flip  = m >> 3
index = (m & 7) ^ (7*flip).
```

Every packed `L-T` subtraction is borrow-free because each `L` byte is at
least its corresponding `U` byte.  Unlike the weight-derived table, `flip`
comes from the lane's weights and varies across a warp, so a uniform branch
cannot remove this cost.

`activation_verify.py` checks 320,144 output dots, full16/half8 identity,
packed subtraction, and the maximum plane-byte bound.  All mismatch and bound
failure counts are zero; the observed maximum is 224.  Results are in
`activation_verify.json`.

## Endpoint caveat

The proof depends on the actual `[-7,7]` A4 quantizer.  A generic signed INT4
input admitting `-8` can have an all-`-8` G32 row with L1 norm 256.  A byte
plane then wraps and can carry into the next token.  Any reuse outside the
current quantizer must either enforce `[-7,7]`, normalize an all-`-8` row to
`-4` and fold a factor of two into its activation scale, or use a wider/fallback
path.  W4 weights may still contain `-8`; this bound concerns table values,
which are derived from activations.

## Two corrections to the first implementation estimate

First, two quartet returns can be accumulated together:

```
acc = acc + value0 + value1
```

This permits one SM60 `IADD3` for two lookups.  The arithmetic lower bound is
64 packed accumulator updates, rather than 128, for four outputs x four planes
x eight quartets.  Ptxas does not contract every source occurrence, but the
optimization is valid because all terms and prefixes are at most 224.

Second, offline repacking `c -> c^8` converts the native offset nibble to a
two's-complement key with plane coefficients `[1,2,4,-8]`.  The finish becomes

```
D = P0 + 2*P1 + 4*P2 - 8*P3 + B.
```

After widening bytes into two 16-bit-field words, add `0x80008000`, add packed
`B`, and subtract `8*P3`.  Every field remains in `[30976,34560]`, so there is
no inter-field borrow or carry.  XOR with `0x80008000` converts both fields to
signed-16 encoding before `I2F.S16`.  The CPU harness proves this packed finish
for the same 320,144 dots.  This is shorter than separately forming
`15*B+8*A`; it does not change the negative admission result.

## SM60 budget

For one four-token pack, four lane-owned outputs and one G32 (512 useful MACs):

| Mandatory core work | full16 | half8, precanonical keys |
|---|---:|---:|
| lookup `SHFL` | 128 | 128 |
| minimum paired packed updates | 64 `IADD3` | 64 `IADD3` |
| key-field extraction | 128 | 128 |
| distributed table waves | 4 | 2 |
| hot complement reconstruction | 0 | 128 predicated subtract/selects |

Full16 holds two 16-entry tables per wave; half8 holds four eight-entry tables.
Half8 saves two construction waves but does not reduce lookup count.  Its 128
lane-varying decodes dominate the saved construction.

`activation_probe.cu` includes prepared positive/negative activation operands,
runtime packed weight keys, table construction, 128 lookups, paired source
adds, and a live output.  CUDA 12.8.61 compiled it for SM60 with zero spills:

| Variant | Registers | Static instructions | SHFL | BFE | IADD3 | IADD |
|---|---:|---:|---:|---:|---:|---:|
| full16 | 112 | 576 | 160 | 97 | 52 | 42 |
| half8 | 55 | 882 | 152 | 113 | 56 | 170 |

The SHFL counts include distributing prepared positive and negative operands:
32 full16 and 16 half8, plus eight half8 `L` broadcasts.  Half8 is 1.53x the
compiled static instruction count of full16 before the exact scale finish.
The lower register count can improve occupancy, but cannot compensate for the
large hot decode plus the measured full16 deficit.

For tile context, four token packs are needed per warp to match the committed
W16 warp's 16-token x four-output ownership.  Even an optimistic full16 source
lower bound with paired `IADD3` and the shorter two's-complement finish is about
the committed W16 kernel's entire 1,872-instruction body before charging raw-A4
to sign/magnitude preparation.  W16's body contains 1,024 HFMA2 instructions,
uses 126 registers and has no spills.

## Fair minimum gate and thresholds

A fair resident probe should use 256-thread CTAs and compare native occupancy,
while keeping the same useful work and input residency across arms:

1. exact G32 half2 control;
2. full16 with runtime raw weight keys, paired `IADD3`, table construction,
   byte widening, group scales and persistent FP32 totals;
3. half8 with offline canonical keys and otherwise identical work.

Use one four-token pack and four outputs per lane, interleave arms, make every
result escape, and report ptxas registers/spills and achieved resident warps.
Prepared positive/negative operands are an optimistic component bound; a
promoted kernel must time conversion from the actual packed A4 wire.  Block-512
results must disclose that 79 registers restrict the candidate to one CTA/SM;
a block-256 confirmation is the clean occupancy check.

Hard kill thresholds for the completed resident G32 component are 9.070 useful
TMAC/s for gate/up and 9.488 for down.  Require at least 10.25 TMAC/s before a
streamed GEMM.  A lookup-only diagnostic should reach at least 12.0 TMAC/s with
no spills to leave credible room for construction and finishing.

The integration worker's existing complete full16 gate is already stronger
than a lookup-only diagnostic: it includes construction, runtime keys, exact
offset-code finish, scales, and persistent FP32 totals.  It measured 2.996
TMAC/s versus 6.171 for its half2 lower-bound control, and matched an independent
integer oracle.  Although its 512-thread geometry depresses candidate occupancy,
it is 3.17x below the stricter absolute target; a 256-thread occupancy correction
cannot plausibly reverse the decision.  Half8 has a strictly heavier compiled
core, so a GPU half8 run is unnecessary for the 1.3x objective.

## Artifacts

- `activation_verify.py`, `activation_verify.json`: independent algebra,
  endpoint-bound, complement, and two's-complement-finish proof.
- `activation_probe.cu`, `activation_probe.cubin`, `activation_probe.sass`:
  independent SM60 compile-only full16/half8 core audit.
- `packed4_worker.cu`, `packed4_sass_counts.json`, `packed4_summary.json`:
  integration worker's complete resident GPU gate and SASS evidence.
