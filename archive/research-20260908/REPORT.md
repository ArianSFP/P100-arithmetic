# GP100 instruction and low-bit inference investigation

Date: 2026-09-08. Target: Tesla P100 PCIe, SM60. This report supersedes broad conclusions in the earlier HFMA2 notes, not their raw observations.

## Answer

There are useful, easily overlooked GP100 operations and operand modes. This investigation found a native four-byte reduction that the project's earlier research incorrectly dismissed, and reproduced a compiler missed optimization for byte-to-float conversion. It did **not** discover hidden INT4/INT2/FP4 multipliers, DP4A, or tensor cores.

The best newly substantiated arithmetic candidate is **unsigned `VABSDIFF4.U8.U8.ACC` used with masks and bit planes**. Its native execution and exact W2xA8/W4xA8 reconstruction passed hardware tests. Whether it improves a complete quantized kernel remains unmeasured. It is exposed through PTX: this is a missed opportunity in the prior investigation, not an undisclosed instruction discovery.

The distinction matters: “not in one assembler grammar,” “not selected by one compiler expression,” and “never exposed by NVIDIA” are different claims.

Sub-Q8 use/evaluation is now authorized. `/home/arian/.codex/AGENTS.md` was updated at the user's request. The separate accuracy and shared-GPU rules remain unchanged; Claude's historical memory was not edited.

## Scope and evidence

- Reviewed NVIDIA architecture/PTX documentation, historical author investigations, MaxAs, envytools, CuAssembler's SM60 examples, public P100 patches, and LUT/binary-GEMM implementations. Sources are linked at the relevant claims below.
- Audited the existing 256-combination HFMA2 selector enumeration and its worker/tooling evidence.
- Compiled **22 isolated SM60 diagnostic kernels** with CUDA 12.8.61 and GCC 14; retained PTX, SASS, hashes and instruction inventories.
- Confirmed the documented DP4A target gate: identical PTX is rejected for SM60 and compiles for SM61. The SM61 binary was never executed on a P100.
- Executed only reviewed, compiler-generated documented PTX on GPU 0. The extended worker passed **262,144 cases for each of five operations**: unsigned SAD, masked unsigned byte sum, signed W2xA8, signed W4xA8, and signed byte-to-FP32 conversion.
- Ran separate instruction dependency-chain and independent-chain timings. No model benchmark, production kernel integration, unknown opcode execution, driver change or GPU reset occurred in this series.

This is broad source research and bounded experimentation, **not exhaustive coverage of GP100 silicon**. Only the earlier fixed HFMA2 selector subspace was exhaustively enumerated offline. A disassembler accepting an encoding cannot establish its legality, semantics or performance. No finite sample here proves there are no other modes.

The remaining GPU reservation was held through the full extended test series and artifact checks, then released after the final health check: no compute processes, 5 MiB/GPU, 0% utilization and zero volatile uncorrected ECC errors. No tests remain running.

## Candidate map

“Hardware tested” below means this investigation's recorded P100 runs, not merely a disassembler label.

| Family | Possible inference use | Evidence/status |
|---|---|---|
| Unsigned `VABSDIFF4` and `.ACC` | Four-byte sums; masked and bit-plane W2/W4 x A8 | **Hardware tested; live candidate.** Signed counterpart expands substantially. |
| Scalar `VMAD` byte/half selection | Extract + sign/zero extension + scalar integer MAC | Four native VMADs replace 16 arithmetic instructions in our dot4 diagnostic. Already measured neutral in the user's Q8 workload; not retested for a speedup. |
| `XMAD` selectors, `.MRG`, `.PSL`, carry modes | Integer products, packing and corrections | Real compiler-generated modes; not extra independent multipliers. Carry-chain restrictions can negate instruction-count savings. |
| `I2F` / `I2I` source selection and conversion | Fuse byte extraction/conversion; potentially clamp/pack | **Concrete I2F compiler miss reproduced and hardware-validated.** Other combinations not swept. |
| `PRMT`, including sign replication | Byte rearrangement, mask expansion, tiny register codebooks | Native compiler evidence. Must distinguish raw PTX semantics from CUDA intrinsic selector semantics. Already used in current IQ unpacking. |
| `LOP3`, `BFE`, `BFI`, `SHF`, shifts | Nibble/2-bit unpacking, masks, Boolean bit planes | Documented building blocks; no new arithmetic lanes. LOP3 select compiled to one instruction. |
| `HFMA2` source broadcasts / FP32 input interpretation | Absorb broadcasts, scaling or conversion | Prior limited hardware witnesses; normal high-level broadcast already compiles optimally. FP32-input edge semantics remain incomplete. |
| `HADD2.F32`, `HMUL2`, half result modes | Half conversion and arithmetic preparation | Compiler already uses HADD2.F32 for half-to-float. Merge experiments are not a reliable complete capability map. |
| `HSET2` / `HSETP2`, predicates and select | Packed comparisons, clipping and sign decisions | SM60 assembler corpus evidence; auxiliary operations, not packed INT2/INT4 multiply. No new semantic sweep. |
| `IADD3`, scaled adds and carry handling | Reduce correction terms / address and accumulator overhead | Existing integer datapath. Relevance is surrounding work saved, not a dot-product mode. |
| `POPC`, `LOP`, `VOTE`, `SHFL` | Binary/bit-serial dot products; register LUT lookup and reductions | Known operations. Algorithmically useful at low bits; need activation-format and packing costs. |
| `FSWZADD` and graphics-oriented modes | Possible lane-dependent add/subtract preparation | SM60 corpus contains it; no new GP100 test. It is not evidence of a fused arbitrary shuffle/reduction or low-bit multiply. |
| FP32 exact integer FMA; FP64 packing | Alternative arithmetic or accumulation | Exactness can be useful; local half/FP64 substitution experiments already failed. No new retest justified by “more FLOPS” alone. |
| Texture, load/cache and shared-memory modes | Conversion on load, lookup, reuse and data movement | Separate memory/packing research; no credible hidden multiplier evidence found. No firmware or device-register experiments. |
| DP4A/DP2A, integer/tensor MMA, FP4/FP8 arithmetic | Desired packed products and wide accumulators | No evidence found for hidden GP100 support. NVIDIA distinguishes GP100's half path from other Pascal chips' integer dot products. |

NVIDIA's [CUDA 8 architecture discussion](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/) describes the GP100/GP10x distinction. The combined [Maxwell/Pascal instruction reference](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html) is not a per-SM60 legality table. [MaxAs](https://github.com/openai/openai-gemm/blob/master/maxas/MaxAs/MaxAsGrammar.pm), [envytools](https://github.com/envytools/envytools/blob/master/envydis/gm107.c), and [CuAssembler](https://github.com/cloudcores/CuAssembler) are encoding evidence, not universal silicon guarantees. The local cuasm checkout is `a86dbdf171254ac9e33e3877e35deb2413133b84`.

## 1. Native unsigned SAD: the strongest new local finding

The old memory `p100-dp4a-arithmetic-research-20260722.md` says the vabsdiff4 family is emulated on 6.x. That blanket statement is contradicted by today's SM60 compiler output **and actual P100 execution**:

```text
PTX:  vabsdiff4.u32.u32.u32.add d, a, b, c;
SASS: VABSDIFF4.U8.U8.ACC Rd, Ra, Rb, Rc;
```

For four unsigned bytes, it computes:

`d = c + sum_j abs(a_j - b_j)` (32-bit accumulator arithmetic).

In contrast, our signed PTX SAD expanded to a long PRMT/extraction/absolute-value/add sequence. Packed `vadd4` also expanded. This is a specific unsigned-SAD exception, not a claim that all video SIMD works natively.

The historical [Pascal/Volta SAD discussion](https://forums.developer.nvidia.com/t/the-instructions-of-vabsdiff4-increases-in-cuda9-2-volta/61779) includes a NVIDIA response identifying a likely throughput-documentation error. That thread tested SM61, so it is supporting context; today's SM60 run is the direct GP100 evidence. The [scalar-video discussion](https://forums.developer.nvidia.com/t/is-the-instruction-mov-b32-r-g-b-a-r1-even-supported/43862) also explains why scalar VMAD must not be conflated with emulated packed video addition.

### Exact low-bit construction

Let `S(U,M) = SAD4(U & M, 0)`, where each mask byte is either `0x00` or `0xff`. One AND plus one unsigned SAD sums selected activation bytes. Mask construction is **not free**.

For signed A8 activations, XOR each raw byte with `0x80`; the resulting unsigned byte is `u_j = a_j + 128`. For signed two's-complement Wb weights, use bit planes `w_j = sum_p coefficient_p * bit_p(w_j)`, with the top coefficient negative. Then:

```text
dot(Wb, A8) = sum_p coefficient_p * S(U, M_p) - 128 * sum_j w_j

W2 coefficients: [1, -2]
W4 coefficients: [1, 2, 4, -8]
W8 coefficients: [1, 2, 4, 8, 16, 32, 64, -128]
```

This is an exact integer identity, not approximate low-precision accumulation. A precomputed weight sum could amortize the bias correction; our correctness kernel constructs it explicitly. The W2/W4 reconstructions were both checked on hardware against independent host integer dots. CPU tests also checked the W8 identity across 266,240 vectors, but no W8 SAD speedup is claimed.

The cost grows with weight bits: two/four/eight bit-plane reductions before expansion, combination, scaling and memory costs. This makes **W2A8 the first complete-kernel candidate**, W4A8 a secondary candidate, and W8A8 unattractive without a separate reuse argument.

This is not a drop-in implementation of every GGUF “2-bit” or “4-bit” type. Unsigned/affine formats need their own zero-point terms; IQ codebooks are not necessarily two's-complement integers. Scale groups and rounding order must be preserved. It says nothing about native FP4.

### Hardware results

Extended run: [raw result](compiler-output/sad-1788860735463197370-result.json), [pre-launch record](compiler-output/sad-1788860735463197370-prelaunch.json), [exact SASS](compiler-output/sad-1788860735463197370.sass).

- Device: P100 PCIe 16 GB, 56 SMs; driver 580.173.02; GPU UUID ends `8b4ed9f2e706`.
- Five operation checks each passed 262,144 vectors. Inputs include every scalar unsigned-byte pair in replicated lanes, followed by deterministic mixed backgrounds. This is not every possible four-lane input combination.
- Four-independent-chain throughput was approximately **1.95 trillion thread-level SAD instructions/s** in this short probe. It used 12 registers, no spills, 256 threads/block and 16 blocks/SM in the grid. This is not a peak specification, MAC rate, or end-to-end inference result.
- Warm dependency-chain medians: 256 instructions -> 1909 cycles; 512 -> 3485; 1024 -> 6637. The incremental slope is about **6.16 cycles/instruction including loop overhead**. The first cold 256-instruction sample was 2789 cycles and is retained in the raw log.
- No comparison against a full production Q2/Q4 matvec was performed. The entire masked-dot correctness kernel was not throughput-benchmarked; the timings isolate SAD itself.

## 2. An actual compiler miss: signed-byte conversion

Our C++ expression converts byte 2 of a raw word to FP32. Its emitted arithmetic was:

```text
SHR.U32 R0, R6, 0x10;
I2F.F32.S8 R10, R0;
```

The equivalent explicit PTX extraction/conversion:

```text
{ .reg .s32 t;
  bfe.s32 t, source, 16, 8;
  cvt.rn.f32.s32 result, t;
}
```

folded to `I2F.F32.S8 R10, R6.B2`. The 262,144-vector hardware check verified the FP32 output bits. This saves one instruction in this diagnostic; it does not establish that a current hot production loop contains the missed pattern. It is achievable through documented PTX, without cubin editing.

The alternate 16-bit PTX `mov` construction did **not** improve this expression. Keeping unsuccessful controls in [the inventory](compiler-output/inventory.json) avoids attributing a win to all hand-written PTX.

## 3. Compiler successes and previously closed paths

The tests also show where editing SASS would solve nothing:

- High-level `__hfma2(A, __half2half2(__low2half(B)), C)` already produces `HFMA2 ... B.H0_H0`. A PTX half-register packing expression does too. An explicit volatile PRMT followed by FMA leaves PRMT separate; that self-imposed optimization boundary is not evidence the compiler generally misses broadcasts.
- Half-to-FP32 already compiles to `HADD2.F32 ... H0_H0, -RZ.H0_H0`.
- Our four-byte C++ dot uses 16 arithmetic instructions; four PTX VMADs use four native instructions. Nevertheless, [the local Qwen3.8 decision ledger](/home/arian/llama.cpp-qwen38-p100/NOTES-codex.md:65) records no reproducible end-to-end VMAD benefit: 29.49 off versus 29.68 on in the reversed pinned repeat. Do not present the public patch's improvement as this rig's result.
- Prior [FP64-SWAR and Q4 geometry results](/home/arian/llama.cpp-qwen36/results/p100-alu-tricks-20260713/RESULTS.md) and the seven half2 substitution failures remain relevant. No generic half2 or FP64 packing retest was performed.

For XMAD specifically, the [P100 carry-chain investigation](https://forums.developer.nvidia.com/t/generating-xmad-x-cc-by-ptx/70339) reports compiler misses but also explains hardware chaining restrictions and cases where fewer instructions ran slower. Source-selection or packing fusion is plausible; instruction count alone is not a performance model.

## 4. HFMA2 audit: keep observations, narrow conclusions

The historical [Scott Gray investigation](https://forums.developer.nvidia.com/t/nvidia-pascal-titan-xp-titan-x-geforce-gtx-1080-ti-gtx-1080-gtx-1070-gtx-1060-gtx-1050-gt-1030/42660/113) used GTX 1080, not P100, and did not find the packed-half dot product with FP32 accumulation it sought.

The previous local study observed ten baseline/single-source-selector configurations completing on P100. B=`01` is decoded as **source** `R4.F32`; it is not the destination mnemonic `HFMA2.F32`. Small finite witnesses fit scalar-FP32 input interpretation, but large-input behavior is insufficiently characterized to assert a full IEEE fused mixed-precision equation. No smaller-lane behavior was demonstrated.

Important limitations discovered in the local harness:

1. All 256 selector combinations disassembled, but only a subset executed. The offline enumeration is not a complete legality/semantics table.
2. CuAssembler's round trip added padding (2312 -> 2344 bytes). Decoded instructions were preserved, but the roundtripped cubin was not itself execution-validated in this study. The original planned gate was not fully satisfied.
3. The merge experiment changed the destination from R14 to seeded R11 while retaining control words and downstream stores, including a store from R14. Dependencies must be re-audited for the new destination read/write behavior. Its error 715 establishes that **that candidate/template failed**, not that every GP100 merge form is unsupported. MRG_H1 was not executed.
4. The “exact” half oracle returns a host float for normal values below one, then can fail with `AttributeError`. `cpu_checks.py` reproduces this with half bits `0x3800`. Existing selected witnesses are not thereby disproven, but the oracle is unsuitable for broad exact testing as written.
5. The patcher's hash argument is optional and it does not reject Pascal control-word offsets. The earlier supervisor also lacks a robust reservation/health gate and has a bytes/JSON timeout-path bug. Do not reuse it for a broad unknown-encoding sweep without repair.

No unknown HFMA2 modes were executed during today's extended research. A future merge or additional-mode test needs a properly assembled, independently initialized destination template, verified scheduling, exact raw-bit references, and explicit review. Nothing here licenses broad opcode fuzzing on a shared rig.

## 5. Low-bit algorithms reopened by the user's change

### LUT kernels: especially W2/W4 with higher-precision activations

[LUT-GEMM](https://arxiv.org/abs/2206.09557) provides a multiplication-saving weight-only quantization approach; its published speedups are not measurements on this rig. The [reference kernel](https://raw.githubusercontent.com/naver-aics/lut-gemm/main/lutGEMM/src/cuda/kernels/mv_fp16.hpp) builds activation-combination tables in shared memory and loops over weight bit planes. Its use of half accumulation/atomic output needs numerical review before adoption. The [dispatcher](https://raw.githubusercontent.com/naver-aics/lut-gemm/main/lutGEMM/src/kernels.cu) routes non-vector shapes to dequantization plus cuBLAS, so its matvec implementation is not automatically a batched/prefill solution.

The local memory parked LUT work pending a conflict-free lookup layout. Two concrete alternatives address that specific objection and justify further evaluation:

- **Register lookup via SHFL:** distribute a 16-entry table for four activations across warp lanes and gather using each lane's four-bit index. Table lookup itself uses no shared-memory bank. Construction, synchronization, shuffle issue cost and reduced reuse can still lose.
- **Replicated 32-bit shared table:** word address `32*entry + lane` has bank `lane`, independent of the entry index. A 16-entry table needs 2048 bytes versus 64 bytes unreplicated, plus replication work. The address model passed 4096 CPU cases; GPU throughput is untested. This statement applies to 32-bit table entries, not arbitrary halfword layouts.

These are explicit candidate layouts, not claims of novelty or proven improvements. Preserve compressed weight storage; expanding every low-bit weight to byte masks in global memory can erase the bandwidth benefit.

### POPC / bit-serial alternatives

For two's-complement integers, a bit-serial dot is a weighted sum of `popc(weight_plane & activation_plane)`. Naively it needs `B_w * B_a` plane-pair counts per group: 16 for W2A8, 32 for W4A8, 64 for W8A8, before packing/reduction. With both operands genuinely low-bit, the cost is smaller. A binary XNOR identity cannot be applied directly to arbitrary A8/A16 activations.

The authors' [SBNN implementation](https://github.com/uuudown/SBNN) reports testing binary networks on P100/V100. It establishes feasibility of bit-based GPU algorithms, not a turnkey Qwen low-bit inference result.

### External Q4-specific kernels

The [public P100 patch stack](https://github.com/shinbunbun/llama-cpp-p100-patches) includes Q4-oriented half2 and lookup optimizations. The [Qwen GP100 model pack](https://huggingface.co/fallentree/Qwen3.8-27B-Uncensored-GP100-GGUF) also reports Q4-specific layouts/runtime results. These deserve source/format comparison now that sub-Q8 is allowed, not blind import: the latter explicitly attributes its very high warm throughput to near-perfect ngram reuse, and local generic half2 substitutions already failed. A different storage layout or eliminated conversion is the necessary new evidence. Neither source demonstrates hidden FP4 silicon.

## Next experiments, in priority order

1. **Complete W2A8 masked-SAD matvec**, with compressed inputs, actual Q2 block metadata, output corrections and the current integer kernel as matched baseline. Measure mask generation and register pressure, not just SAD. Compare several realistic K/M/batch shapes.
2. **W2/W4 register-LUT or conflict-free shared-LUT matvec**, explicitly amortizing construction across output rows. Compare A8 exact-integer and A16/FP32-accumulation variants separately.
3. **Audit hot conversion sites for I2F byte-selector fusion**, then make the smallest PTX-source change only where the miss actually occurs.
4. **HFMA2 result-handling research only after repairing its harness gates.** Destination/merge/source combinations remain semantic research, not established performance candidates.

A candidate passes only if its complete kernel improves, scales/corrections are correct, and numerical validation passes. Compare kernel changes against the same quantized model; separately report quantization quality relative to the higher-precision model. Lifting the model-format restriction did not silently waive accuracy requirements. Integrate nothing based only on microinstruction throughput.

## Artifacts and reproduction

- [Compiler probes](compiler_probes.cu), [build/inventory script](compile_inventory.py), [PTX and SASS inventory](compiler-output/inventory.json).
- [Documented instruction worker](sad_probe.cu), [fixed supervisor](run_sad_probe.py), [CPU identity/oracle checks](cpu_checks.py), [CPU results](compiler-output/cpu-checks.json).
- Current compiler-probe cubin SHA-256: `ac27bcb5a97f3d2af47835a3157aa3bdf6b61e9b5996d5b45baa4e5edab11914`.
- Extended GPU worker SHA-256: `72865333c35c8219d6a0bd5d013da652a428b7a7033165c0b800e7712ceee0fd`; source hash is recorded in its pre-launch/result JSON.

Offline reproduction:

```sh
python3 compile_inventory.py
python3 cpu_checks.py
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-14 -O3 -arch=sm_60 -lineinfo sad_probe.cu -o compiler-output/sad-probe
```

Only after arranging and holding a GPU reservation, the reviewed hardware run is `python3 run_sad_probe.py` outside the device-restricted sandbox. The controller never initializes CUDA, records hashes before launch, starts a fresh worker with an empty environment and CPU affinity 0-11, checks occupancy, and stops on failure/timeout without retry or reset. It is fixed to the documented probe, **not an authorization gate for arbitrary binary mutation**. Coordination remains a separate required step.
