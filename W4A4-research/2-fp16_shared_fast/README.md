# Packed FP16 / shared INT4 activation: fast exact repair candidates

9 September 2026. CPU numerical validation and host C++ execution only.
The CUDA candidates were NOT compiled, executed, or timed in this session.
No P100 or model-throughput speedup is claimed.

## Findings

The previous 2 KiB `(activation, rounded_low_product)` table is unnecessary
when original weight codes remain available. Two smaller candidates follow.

### A. Three packed-half arithmetic operations, no repair table

Use NORMALISED packing `p = w_high + w_low/128`, with all raw codes in [-8,7].
All 256 p values are represented exactly in binary16. Compute in each lane:

```
t    = fma16(p, a, 1536)
high = half(t - 1536)
low_scaled = fma16(p, a, -high)
```

This yields `high = w_high*a` and `low_scaled = w_low*a/128` exactly.
Each half2 call performs this for two output-row pairs: FOUR useful products.
The output must retain its low-row scale factor of 1/128 until the epilogue.

Proof: `p*a = H + L/128`, where H and L are signed INT4 products in [-56,64].
The exact argument of the first FMA is between 1479.5625 and 1600.5, safely
inside the FP16 binade [1024,2048), which has integer spacing. Thus rounding
selects `1536+H`. The only possible half-integer tie is L=64. This requires
w_low=a=-8, so H=-8*w_high is a multiple of 8. `1536+H` is therefore even,
and ties-to-even selects the desired value. Subtraction recovers H exactly.
The second FMA cancels H BEFORE rounding and its result L/128 is representable.

This is not a one-multiply implementation: it evaluates two FMAs. It is a
compact, lookup-free exact control, not a demonstrated throughput win.

For dot products, separately add low_scaled and high into two half2 accumulators.
At G<=32, those accumulations are exact for the full signed-INT4 domain.
The maximum unscaled partial magnitude is 32*64=2048; the low accumulator is
simply the same integer lattice scaled by 1/128. Apply each output row's own
weight scale and the activation scale in FP32 after each group. Different
output rows need not have equal weight scales.

Arithmetic budget for FOUR useful MACs, excluding loads/packing/epilogues:

- Ordinary half2 baseline: TWO HFMA2 operations.
- Three-operation split plus two accumulations: FIVE half2 arithmetic operations.

Negation/moves, register pressure, compiler lowering, and scheduling also count.
Three intrinsic arithmetic operations are not a certified three-instruction
SM60 SASS listing. The negation may fold into a source modifier; inspect it.

DO NOT replace the low residual plus addition with
`low_acc = fma16(p, a, low_acc-high)`: constructing that addend can round away
accumulator bits. For w_low=w_high=a=-8 and low_acc=1/128, it returns 1/2
instead of 65/128. An independent witness is retained in results.json.

### B. ONE packed FP16 multiply, with 3-bit residue repair

This variant uses the ORIGINAL UNNORMALISED packing `P=w_low+128*w_high`.
Let `r=int(RN16(P*a))`. The int conversion is exact because r is an integer
within a small range. The first product is

```
high = floor((r+63)/128)
```

and the exact rounding error is recoverable from only

```
rho = (w_low*a) mod8
delta = ((rho-r+4) mod8)-4
low = r-128*high+delta
```

Here delta is always in [-2,2]. Since P*a is congruent to w_low*a modulo8,
and the rounding error is strictly less than half the modulus, its centred
modulo8 representative uniquely recovers the exact error. All 4096 signed
triples passed. This needs the original low-row weight code, unlike the old
2 KiB table indexed only by a and a rounded residual.

No full integer multiply is necessary. For each activation residue a mod8,
precompute eight bytes `T[j]=(a*j)&7`, j=0..7. These fit in TWO registers.
There are eight possible table rows: 64 bytes total across all activations.
With four low-weight nibbles in one selector word:

```
rho_bytes = PRMT(T_low, T_high, low_nibbles & 0x7777)
```

One PRMT selects FOUR product residues concurrently. This is only the residue
lookup, NOT four complete product repairs. Conversion, digit extraction,
correction arithmetic, packing, and accumulation must still be measured.
`mod8_tables.json` contains the constants. Table selection/setup must be
amortised across output-row pairs sharing the activation, rather than stored
as eight bytes of new global-memory metadata for every activation.

All 8 * 8^4 = 32768 activation-residue / four-way-selector combinations passed
an independent byte-permutation semantic simulation.

### C. A hard limit on delaying repair without side information

For the same two activations a=[7,6], these two different weight sequences
produce the SAME packed sum, even with exact arithmetic:

- w_low=[-8,-8], w_high=[0,0]: low dot=-104, high dot=0, packed sum=-104.
- w_low=[6,-3], w_high=[-1,1]: low dot=24, high dot=-1, packed sum=-104.

All products and partial sums in this witness are represented exactly in FP16.
Thus a decoder seeing only that packed accumulator and those activations cannot
recover both dots uniquely. Merely postponing lookup/unpacking until the end
cannot work universally for this radix-128 scheme. Additional state, larger
radices/precision, restricted domains, or approximation change the problem.

Smaller radix packing does not trivially fix extraction: B=32 gives exact
single FP16 products for every signed input, but direct high/low fields overlap.
Results for B=16,32,64,128 are retained.

## Validation and reproduction

```
python verify.py
g++ -O2 -std=c++17 -Wall -Wextra -Werror -fno-fast-math \
    -fsanitize=undefined host_check.cpp -o host_check
./host_check
```

Python requires NumPy. The FMA oracle computes the exact dyadic product and sum
in float64 then rounds ONCE to float16. All magnitudes/fractions in these checks
make the float64 intermediate exact. Random G32 tests use a fixed seed and
include constant endpoint blocks. They are not exhaustive over all blocks;
the analytical bounds above establish the full-domain G32 guarantee.
Signed-zero bit identity is not asserted; comparisons are numerical integer
product identities. Correctness here is relative to raw A4 quantized codes,
not the original A16 network.

Recorded results:

- All 4096 triples: exact normalised split, zero mismatches.
- All 4096 triples: mod8 repair, zero mismatches.
- All 32768 PRMT selector tests: zero mismatches.
- 100000 Python G32 blocks: zero mismatches for split and plain-half controls.
- Independent C++: all 4096 triples and 100000 random G32 blocks, zero mismatches.

## Targeted P100 experiment

Compile with a CUDA toolchain supporting sm_60 (e.g. the project's CUDA 12.8):

```
nvcc -O3 -std=c++17 -arch=sm_60 -cubin compile_probes.cu -o probes.cubin
cuobjdump --dump-sass probes.cubin > probes.sass
```

These commands were not run here. The probes only expose the candidate inner
operations for assembly inspection; they are NOT a complete GEMM or timing
harness. A benchmark must supply bounds-safe launches and independently checked
inputs/outputs. Do not time these short global-load/store probes as a GEMM.

Implement matched microbenchmarks, then one real kernel shape:

1. Existing fast FP16 kernel with raw A4 codes, separate scales, G32 accumulators.
2. Identical tile/reuse pattern with the normalised three-operation split.
3. Unnormalised packed multiply + integer conversion + PRMT/mod8 repair.

Use independent accumulator chains for throughput and separate dependent
chains for latency. Count useful dot outputs, not packed multiply instructions.
In SASS, inspect HFMA2/HMUL2/HADD2, sign handling, conversions, PRMT, shifts,
integer arithmetic, registers, spills, and output accumulation. Do not assume
fewer floating multiplies imply fewer issue cycles.

Pair output rows sharing a_k. For the PRMT route, use four row pairs/eight
outputs per shared activation where register and tile geometry permit. Keep
HBM weights in their original 4-bit format. Expand/prepack arithmetic operands
at shared/register tile loading and reuse them across tokens. Charge dynamic
activation quantisation and all table/selector setup to the complete pipeline.

No packed long-K accumulator without extra information. No FP16-scale mixing
inside the integer identity. No promotion without a complete device-pipeline
win and separate real-model quality validation against the project's controls.

## Primary instruction references

- NVIDIA CUDA 12.8 half2 arithmetic semantics:
  https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html
- NVIDIA PTX 8.7: fma.f16x2, conversions, prmt.b32:
  https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html
- NVIDIA Pascal tuning guide: paired FP16 arithmetic and GP100 throughput:
  https://docs.nvidia.com/cuda/archive/12.8.1/pascal-tuning-guide/index.html

The packing identities, bounds, collision, and test results in this folder are
our derivations/simulations, not claims made by those documentation pages.
