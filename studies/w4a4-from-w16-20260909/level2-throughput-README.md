# Level-2 packed-FP16 throughput probe

Status: compiled and audited for SM60 with CUDA 12.8; not executed on a GPU by
the author of this probe.

This bounded executable compares the level-2 suggestions at one normalization:
each thread and loop iteration performs 64 useful scalar signed-INT4 MACs.  It
uses eight independent packed chains, two activation steps (`+a` and `-a`) per
iteration, and four useful scalar products per chain and activation step.  The
opposite activation steps cancel exactly, keeping long timing loops within the
exact FP16 integer range.

The fixed launch is 112 blocks of 256 threads (`__launch_bounds__(256, 2)`), or
two blocks per SM on the 56-SM P100.  The executable refuses any target other
than one visible SM60 GPU with 56 SMs and requires the explicit
`--gpu-approved 1` argument.

## Matched work

| Mode | SM60 loop arithmetic | Useful scalar MACs / iteration |
|---|---|---:|
| `plain-hfma2` | 32 HFMA2 | 64 |
| `normalized-split` | 32 HFMA2 + 48 HADD2 | 64 |
| `hmul-prmt-repair` | 16 HMUL2 + 32 F2I + 8 PRMT + integer repair/accumulation | 64 |

The repair mode processes two `half2` packed products per PRMT.  One PRMT
therefore supplies four low-product residues for eight useful scalar products.
For each positive or negative activation step, two dynamic constant-memory
loads select the 8-byte mod-8 table row; the resulting two registers are reused
by four PRMTs.  The loop consequently contains four LDC instructions, rather
than unrealistically preloading one fixed activation table outside the timing
loop.

All modes cycle a raw activation code through 0..7 and also process its negative,
which remains in the signed-INT4 domain.  The shared integer-to-half activation
materialization appears once per loop in every mode.  Weight representation is
prepared before the source loop, although ptxas schedules packed half-word merge
instructions inside it: 16 XMAD.PSL.CLO instructions for the two-operand plain
mode and eight for each packed mode.  Thus this executable charges the plain
mode for forming twice as many FP16 weight operands.  A result should be read as
the compiled staged-operand implementation, not as a pure prepacked-register
instruction ceiling.

## Compile and SASS audit

Run:

```text
python3 build_level2_throughput.py
```

The build uses g++ 13, `-O3`, `-arch=sm_60`, line information, and verbose
ptxas output.  It writes the executable, build log, complete disassembly,
opcode counts, and a hash manifest.  The final resource report is:

| Kernel | Registers | Spill loads/stores | Local-memory SASS |
|---|---:|---:|---:|
| `level2_plain_hfma2` | 64 | 0 / 0 | 0 |
| `level2_normalized_split` | 56 | 0 / 0 | 0 |
| `level2_hmul_prmt_repair` | 121 | 0 / 0 | 0 |

The final backward-loop audit found 55 static instructions for plain, 95 for
normalized split, and 344 for residue repair.  The repair loop contains its 16
HMUL2, 32 lane conversions, eight PRMTs, four dynamic table loads, and all
integer extraction, correction and accumulation operations.  Earlier compiler
output hoisted candidate work outside the loop; that version was discarded.
The frozen source uses a real activation-state update and the final SASS shows
all compared work inside the backward branches.

The operation-count ratios are evidence about the compiled instruction burden,
not P100 throughput.  No timing result is recorded here.  Full GEMM claims must
also charge activation quantization, G32 scale application, weight staging,
dispatch and output reduction.
