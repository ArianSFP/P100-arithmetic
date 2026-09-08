# GP100 HFMA2 bounded study — results

Audit update (2026-09-08): see [the extended investigation](research-20260908/REPORT.md), especially its HFMA2 audit. Offline decoding is not hardware validation; the merge error is template-specific and the original harness has unresolved gating/oracle limitations.

Date: 2026-09-07/08 UTC  
GPU: Tesla P100 PCIe, compute capability 6.0, driver 580.173.02  
Toolkit: CUDA 12.8.61, `nvcc -ccbin /usr/bin/g++-14`

## Baseline and patch target

The isolated kernel compiled successfully with one register-form `HFMA2`:

```text
section: .text.probe
section-relative offset: 0xf0
baseline word: 0x5d0003800047020a
baseline cubin SHA-256: f7c3facfb840256df55752e0ac2a339165cfa2932867990b1dada862eede7e8e
```

The ELF-aware patcher changes only the target instruction bytes and rejects a
hash or baseline-word mismatch. All 255 non-baseline combinations of the four
2-bit selector fields were generated offline and all disassembled with exactly
one HFMA2 at offset `0xf0`.

## Executed selector controls

| Candidate | Offline decode | P100 observation |
|---|---|---|
| baseline | `HFMA2` | clean; `(10,21)` and fused witness `0x8c008c00` |
| B=`10` | `R4.H0_H0` | clean; matches B-low broadcast `(10,17)` |
| B=`11` | `R4.H1_H1` | clean; matches B-high broadcast `(12,21)` |
| B=`01` | source `R4.F32` | clean; limited scalar-F32 source witnesses, not extra lanes |
| A=`01` | `R2.F32` | clean; scalar-F32 A behavior |
| A=`10/11` | `R2.H0_H0` / `R2.H1_H1` | clean; expected A lane broadcasts |
| C=`01` | `R7.F32` | clean; scalar-F32 C behavior |
| C=`10/11` | `R7.H0_H0` / `R7.H1_H1` | clean; expected C lane broadcasts |

For B=`01`, the lane witness `A=(1,2)`, `C=(7,11)`, and B raw bits
`0x45004200` (FP32 `2052.125`) produced low/high half bits `0x6806/0x6c05`,
which are `(2060,4116)`. The scalar-F32 witness B=`1.5` produced
`(8.5,14)`. This is a valid mixed-precision mode, not a hidden FP4/INT4
operation. The large-value fused witness also showed finite near-max half
outputs (`0x7bf97bf9`); exact overflow/saturation edge semantics remain a
follow-up if needed.

## Merge probe

The merge template independently initialized a seed register and verified that
the unmodified destination mode still computes the expected packed result after
the destination register was aliased to the initialized seed register.

The first reviewed `.MRG_H0` candidate, with that same initialized destination
register, terminated its fresh worker with:

```text
CUDA_ERROR_ILLEGAL_INSTRUCTION (715)
```

The sweep was stopped immediately. `.MRG_H1` was not executed and no GPU reset
was attempted. This is an unsupported/illegal encoding result on this GP100
path, not arithmetic evidence.

## Tooling gate

The isolated gpuocelot/cuasm checkout was fetched and its dependencies were
installed in `tooling/venv`. Its round trip preserved the decoded instruction
stream but was not byte-identical for this probe: the reassembled cubin added
trailing NOP padding (`2312` -> `2344` bytes). The shared project’s earlier
on-rig A0 gate had a byte-identical SM60 round trip on a different trivial
cubin. This experiment therefore uses the independently verified direct ELF
patcher for candidate generation and does not claim this particular cuasm
round trip passed the byte-identity gate.

## Safety / reservation note

The initial host check showed all four GPUs idle. A separate Qwen3.8 server
started later and was found owning about 11 GiB on every GPU during the merge
health check. No further GPU work is authorized while that server is active.
The merge H0 worker exited cleanly after its error, and `nvidia-smi` still
reported all four P100s at 0% utilization; no reset was performed.

## Conclusion

The bounded question found documented/reverse-engineered selector behavior and
one illegal merge encoding. It found no evidence of undocumented smaller
arithmetic lanes. The most actionable result is that B=`01` is a valid scalar
FP32-selection mode, while known broadcast modes work on GP100 as expected.
