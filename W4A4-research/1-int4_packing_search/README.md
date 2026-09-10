# INT4 packing architecture investigation

9 September 2026. CPU simulation and host arithmetic validation only. No P100
kernel execution, GPU timing, SM60 compilation or model-quality measurement was
performed. The CUDA branches and compile probes are supplied for subsequent
review/compilation, not represented as validated GPU code.

## Reproduce

```
python explore.py
# Optional independent host validation, including undefined-behaviour sanitizer:
g++ -O2 -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined host_check.cpp -o host_check
./host_check
```

Python requires NumPy. The script uses a fixed random seed, explicit rounding,
and integer references. The floating bounds make its FP64 intermediate exact
for the FP16/FP32 FMA simulations. The FP64 packed experiments also stay within
exact-integer bounds, so the separate host multiplication/addition is exact.

## Principal results

All quantized signed codes below are in [-8,7]. Correctness is relative to the
integer-quantized dot product, not the unquantized or W4A16 model.

| Test | Cases / blocks | Mismatches |
|---|---:|---:|
| Original unsigned two-term identity, base 512 | 65,536 exhaustive | 0 |
| Signed two-term identity, base 2048 with bias | 65,536 exhaustive | 0 |
| Eight packed pairs -> one 16-term signed dot | 100,000 blocks | 0 |
| Two independent 8-term dots, int16 inputs / uint32 accumulation | 100,000 blocks | 0 |
| Two shared-activation 32-term dots, centred int16 packing | 100,000 blocks | 0 |
| Same shared packing using FP32 arithmetic | 100,000 blocks | 0 |
| Ordinary half2-style integer G32 accumulation | 100,000 pairs of dots | 0 |
| Three shared G2 dots packed in FP32 | 100,000 blocks | 0 |
| Four shared G32 dots packed in FP64 | 100,000 blocks | 0 |
| Seven shared single products packed in FP64 | 100,000 sets | 0 |

Block tests include constant endpoint cases and seeded random cases. They are
NOT exhaustive enumeration of the block spaces. Analytical bounds below supply
the guarantees. The C++ host checker separately passed 100,000 tests each for
the three integer constructions, with the undefined-behaviour sanitizer enabled.

### Two independent dot products

Choose B=2048 and G=8. Pack

P[k] = w0[k] + B*w1[k], Q[k] = a0[k] + B*a1[k].

Sum their products modulo 2^32. In exact arithmetic the polynomial is

C = D0 + B*X + B^2*D1,

where D0=sum(w0*a0), D1=sum(w1*a1), and X=sum(w0*a1+w1*a0).

D0,D1 lie in [-448,512]; X lies in [-896,1024]. Add the fixed bias

bias = 448 + B*896 + B^2*448 = 1,880,883,648.

Let T=(C+bias) mod 2^32. The biased digits lie in [0,960], [0,1920],
[0,960]. There is no carry between the radix-2048 digits, and the top digit
fits in the ten remaining bits of a 32-bit word. Recover

D0 = (T & 2047) - 448,
D1 = ((T >> 22) & 1023) - 448.

Both packed inputs lie in [-16392,14343], fitting signed int16. The raw
accumulator can cross the signed-int32 limit. Use deliberate uint32 wraparound,
not undefined signed C++ overflow. Eight wide MACs compute 16 useful INT4 MACs.
Block an arbitrary longer dot into G8 groups and add the extracted outputs.

Reverse activation packing, Q=a1+B*a0, to put ONE 16-term dot in the middle
field instead. See unpack_kpair16 in the header.

### Two shared-activation G32 dots

Choose B=4096. Plain signed packing w0+B*w1 slightly exceeds int16 at an
endpoint. Centre the high digit instead:

P[k] = w0[k] + 4096*w1[k] + 2048.

The range [-30728,30727] fits int16 exactly. With A=sum(a[k]), calculate

C = sum(P[k]*a[k]) - 2048*A = D0 + 4096*D1.

For G32, each dot lies in [-1792,2048]. Add 1792 to make the low digit
nonnegative. Recover D0=((C+1792) mod 4096)-1792 and
D1=floor((C+1792)/4096). The header performs explicit signed reconstruction
from uint32 fields without relying on right-shifting negative C++ integers.

The same operation is exact in FP32: the packed inputs and every intermediate
integer sum stay below 2^24 in magnitude. They are NOT generally exact FP16
inputs. For example, 4097 rounds to 4096 in FP16.

Precompute the activation sum once per token and group, sharing it across all
output pairs. Apply each output's own weight scale and the common activation
scale only after extracting the two group dots. Different output scales do not
need to be equal. Common reduction-group boundaries are required.

### Ordinary half2 is a strong exact baseline

Each signed INT4 product lies in [-56,64]. At most 32 terms have intermediate
sum magnitude <=2048. Every such integer is exactly representable in FP16.
Thus ordinary half2 FMA can compute two independent quantized G32 dots exactly,
with scales applied after the inner loop and FP32 accumulation across groups.
At 33 terms this universal guarantee disappears: 32*64 + 1 = 2049 rounds to
2048. The random G33 test happened to pass; the explicit adversarial case fails.

### Actual FP16 packing: promising arithmetic, non-free repair

For ONE shared activation, use P=w0+128*w1, which fits FP16 exactly. Form
R=round_fp16(P*a), high=floor((R+63)/128), low=R-128*high.

Across all 4096 signed triples:
- high always equals w1*a;
- low is wrong in 560 cases (13.671875%);
- the largest low error is 2 integer counts;
- low mean-squared error is 0.25390625 for this uniform exhaustive set.

The pair (a, low) uniquely determines the exact low product over this domain.
A 16-by-128 lookup table therefore repairs all cases with zero conflicts. The
provided npy table uses int16 with 999 as an unreachable-entry sentinel; a
production table can use int8 entries (2 KiB), filling unreachable entries
arbitrarily after keeping a separate validation oracle.

This is not a free four-product HFMA2: conversion, field extraction, lookup and
separate accumulation remain. Longer dots do not inherit the G1 guarantee.
A direct lookup of the low product is an important competing implementation;
the repair scheme must beat that as well as ordinary half2.

## Architecture priorities

1. Establish the exact raw-code half2 G32 baseline against the actual W4A16
   production kernel and an equal-quantization scalar-FP32 W4A4 control.
2. Compile the centred INT16 shared-G32 primitive using mad.wide.s16. Inspect
   SM60 SASS; PTX syntax alone does not establish one XMAD or its throughput.
   Also compare the equivalent FP32 packed implementation.
3. Try paired output rows (or a fused gate/up row pair) with lossless INT4
   nibble layout, expanding the packed arithmetic operand only in registers or
   shared memory. Persisting expanded INT16/FP32/FP64 weights changes bandwidth
   and is not a fair comparison. Include activation quantization and sum costs.
4. Test the G8 independent and K-pair variants where layouts already favour
   their operand pairing. Charge extraction and accumulation costs.
5. Ambitious conditional experiment: four-output FP64 G32 alongside ordinary
   HFMA2. FP64 alone has only parity in arithmetic peak with ordinary half2 on
   GP100 after multiplying its throughput by four useful packed products.
   A 2x combined ceiling would require simultaneously exploiting both paths;
   shared scheduling, issue, register and memory resources may prevent it.
   FIRST measure mixed-instruction issue throughput with independent chains,
   before building a GEMM. Prior failed FP64-packing results remain negative
   evidence; standalone FP64 packing is not automatically a new speed win.
6. Three-output FP32 G2 has 1.5x arithmetic headroom over half2 before overhead.
   Its two-term flush makes extraction expensive. Treat as a bounded microprobe.

All timing comparisons must include the full device pipeline, scale handling,
packing amortization, final reduction and dispatch. Evaluate both compute-hot
prefill shapes and bandwidth-limited decode. A4 model-quality evaluation is
separate from integer-arithmetic equivalence. Do not claim a token-rate gain
from operation counts.

## Sources consulted

- NVIDIA Pascal Tuning Guide, including half2 throughput, FP32/FP64 units and
  issue/resource caveats:
  https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html
- NVIDIA Mixed-Precision Programming with CUDA 8, half2 versus FP32 and FP64:
  https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/
- NVIDIA PTX ISA 8.7, integer mul/mad wide forms:
  https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html
- Sommer et al., DSP-Packing: Squeezing Low-precision Arithmetic into FPGA DSP
  Blocks (2022), prior art for arithmetic packing and correction:
  https://arxiv.org/html/2203.11028v1
- User's P100 arithmetic findings and instruction audit, accessed through GitHub:
  https://github.com/ArianSFP/P100-arithmetic/blob/main/docs/FINDINGS.md
  https://github.com/ArianSFP/P100-arithmetic/blob/main/archive/research-20260908/REPORT.md

These are established radix/DSP-packing ideas adapted and checked for these
specific numerical budgets, not a claim of novel hardware or first publication.
