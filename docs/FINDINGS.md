# Findings and evidence guide

## Reading order and authority

The archive is chronological. Statements such as "pending", "not yet measured"
or "no push" describe the phase when a report was written, not the state of this
publication. Old sub-Q8 restrictions were explicitly lifted; accuracy and GPU
coordination requirements remain. Later reports supersede earlier conclusions
without deleting their evidence.

1. [Instruction investigation and HFMA2 audit](../archive/research-20260908/REPORT.md)
   is the authority for interpreting the earlier root HFMA2 report.
2. [Endpoint-SAD offline report](../archive/endpoint-sad-20260908/RESULTS.md)
   establishes the algebra and compiled variants.
3. [Layout/SAD/LUT GPU report](../archive/layout-lut-20260908/GPU-RESULTS.md)
   completes the hardware follow-up, including all 24 frozen workers.
4. [W4A16 round 1](../archive/w4a16-20260908/RESULTS.md) establishes the sealed
   R4/S32 baseline for the final study.
5. [W4A16 round 2](../archive/w4a16-r2-20260908/RESULTS.md) is the latest kernel
   result and the recommended starting point for further work.

## What "hidden instruction" means here

A disassembler name, missing assembler-grammar entry, compiler missed
optimization and previously undocumented silicon capability are different
things. The useful SAD, PRMT, byte-conversion and warp-sharing techniques here
use documented operations. No evidence of extra packed low-bit arithmetic lanes
was found. This does not prove that every possible GP100 encoding is understood.

The initial HFMA2 work enumerated only four two-bit selector fields: 256
combinations, holding other instruction bits fixed. Most were never executed.
Limited source-mode observations do not establish a complete mixed-precision
equation. One merge candidate failed with CUDA error 715; the template and
scheduling limitations preclude generalizing this to all GP100 merge modes.
The later audit also identifies a broken general half oracle, incomplete
round-trip execution validation and weak patcher/supervisor gates. Preserve the
raw observations, but do not use that harness for a new unknown-opcode sweep.

## Endpoint-SAD and lossless layout

For two's-complement weights, write `w = sum_p c_p*b_p`, with coefficients
`[1,-2]` for W2 and `[1,2,4,-8]` for W4. For a common scale group of G elements,
let `A = sum a`, `W = sum w`, and `u = a+128` for signed A8 activations.

The proposed complemented endpoints give:

```text
D_p = sum |u - 255*(1-b_p)|
dot = (sum c_p*D_p - A + 127*G - W) / 2
```

The reversed endpoints give an equivalent exact identity:

```text
E_p = sum |u - 255*b_p|
dot = -(sum c_p*E_p + A + 128*G + W) / 2
```

Both include activation -128. Correctness requires applying corrections inside
each common scale group; affine quantizers need additional terms. Storing a
16-bit weight sum per G32 adds 0.5 bits/weight. Computing it with POPC has its
own cost. Neither should be treated as free.

The early implementation's compiler distributed complements into many XORs.
Reversing the endpoints removed that per-plane logical work. The later
precomplemented byte-sign layout also worked on hardware, at unchanged payload
size, using generic PRMT sign replication. However, the old endpoint kernel
still lost to the old integer kernel. Row reuse, coalesced loads, split-K and
complete correction costs matter more than isolated instruction counts.

The A8 GPU study retains integer controls with the same lossless layouts and
data. W2 LUT/SAD benefits are the incremental wins against these improved
controls, not the much larger old-layout-to-new-layout change. W4 did not show
a convincing arithmetic-family win. Neither compact nor replicated shared LUT
was competitive. The compact table was already conflict-free for the tested
mapping; replication is not automatically a necessary or useful fix.

## W4A16: what changed and what did not

Round 1 found a better row-reuse configuration with an unchanged 32-stripe
reduction. Round 2 keeps the same compressed layout and scale plane, uses
coalesced FP16 loads plus warp shuffles to share widened activations, and reuses
decoded weights across four activation vectors when N=4.

- N1 winner: `r2_14_r2`, two rows/lane, four warps/CTA, S32.
- N4 winner: `r2_36_r2`, additionally reusing weights across four vectors.
- Unchanged: FP16 activation bits, Q4_0 codes/scales, sequential 32-element FP32
  group FMA order, scale FMA, striping and final FP32 reduction tree.
- No activation quantization, half multiply/accumulate, newly added scratch
  preparation pass, extra metadata or new repack versus the prior winner.
- Compiler `HADD2.F32` instructions are exact half-to-float widening, not half
  accumulation. Signed BFE was already present in the prior kernel.

The live prior cubin is loaded into every worker as a separate module. Winners
were selected on GPU1 before frozen repetitions and confirmation on GPUs2/3.
All 79 configurations passed numerical checks; only N1/N4 have confirmed
performance. N3 tail correctness is not a tuned N2/N3 dispatch result.

## Unsuccessful paths and limits

- LUT at A16 changes addition order and was slower than the leading direct
  arithmetic kernel. It was not promoted on the strength of low-bit theory.
- Separately pre-widening activations to FP32 did not help when its device pass
  was included. The final winner needs no such pass.
- Reducing stripe count in round 1 changed FP32 outputs. Slightly better timings
  did not meet the chosen exact-order gate.
- More rows per lane, streamed R8 weights, scalar indexing/cache changes and
  larger CTA sweeps did not displace the round-2 winners.
- Four-vector reuse regresses N1; dispatch must be shape-specific.
- Fewer static instructions do not establish lower runtime. Neither SAD's
  isolated throughput nor a conversion microprobe establishes model speedup.
- Reported bit identity is to study references, not stock llama.cpp. No actual
  model/layer dataset, stock comparison, perplexity/KLD or tokens/sec result is
  included. Quantization quality relative to a higher-precision model remains
  a separate question from equivalence of two kernels on the same Q4 data.

The next gate is real-layer, real-dispatch comparison and production numerical
validation, not another unrestricted search of machine-code bits. Sources and
hardware distinctions are linked at their claims in the archived reports.
