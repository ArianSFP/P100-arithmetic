# Isolated GP100 HFMA2 selector probe

This directory is intentionally outside every llama.cpp source/build tree. It
contains a bounded SM60 experiment for the ordinary register-form `HFMA2`
instruction. It does not modify firmware, the CUDA installation, production
builds, or model files.

## Safety boundary

1. Build and inspect cubins offline first.
2. Execute only the unmodified cubin and manually reviewed selector patches.
3. Launch each candidate in a fresh worker process.
4. Use an external timeout and stop the sweep after a hang, illegal instruction,
   or device-health error. Do not automatically reset a GPU.

The first unknown candidate is the historical unlabelled B-source selector
value `01` (bits 28–29). The four-field 256-entry enumeration is generated
offline only; it is not an instruction-fuzzing permission.

## Quick start

```bash
make
build/elf_inspect.py build/probe.cubin --section .text.probe
/usr/local/cuda-12.8/bin/cuobjdump -sass build/probe.cubin
/usr/local/cuda-12.8/bin/nvdisasm build/probe.cubin
```

After recording the exact HFMA2 section-relative offset and word from the
inspection output, patch a candidate without executing it:

```bash
build/patch_cubin.py \
  --input build/probe.cubin \
  --output results/b-selector-01.cubin \
  --section .text.probe \
  --offset 0xOFFSET \
  --baseline-word 0xWORD \
  --b-selector 1
```

Run the unchanged or reviewed candidate only when the GPU reservation and
external supervision are in place:

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=2s 10s \
  build/hfma2-worker build/probe.cubin | tee results/baseline-worker.txt
```

The worker uses the CUDA Driver API and exits after one diagnostic batch, so a
candidate never shares a CUDA context with another candidate.

## Diagnostic vectors

The primary vector is `A=pack(1,2)`, `B=pack(3,5)`, `C=pack(7,11)`, whose
ordinary packed result is `(10,21)`. B-low broadcast predicts `(10,17)` and
B-high broadcast predicts `(12,21)`. The worker also includes signed and fused
rounding witnesses; `scripts/oracle.py` computes exact binary16 reference
bits without relying on host floating-point FMA ordering.

## Files

- `src/probe.cu`: one raw-bit `fma.rn.f16x2` kernel.
- `src/worker.cpp`: fresh-process Driver API loader/launcher.
- `scripts/elf_inspect.py`: pure-Python ELF64 section/word inspection.
- `scripts/patch_cubin.py`: hash-checked, one-word ELF-aware mutation.
- `scripts/enumerate_selectors.py`: offline 4-field selector generation.
- `scripts/oracle.py`: exact binary16 decode/FMA model and test vectors.
- `NOTES-codex.md`: measured findings for this isolated experiment.
