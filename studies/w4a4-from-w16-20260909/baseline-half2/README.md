# Exact-G32 W4A4 baseline from the W16 M128 kernel

Status: CPU and compiler preparation only. **No GPU result or speedup claim.**

This preserves the committed W16 dual-rail pair/fallback control and its GPU
M64-to-M128 compactor. The W4 candidate uses the same 256-thread M128 output
geometry and double-buffered K32 stages. A timed W4 pipeline performs GPU
FP16-to-symmetric-INT4 quantization, planner compaction, an M128 kernel for
leftovers, and the same M128 kernel for paired tiles.

The W4 kernel stores Q4 weights in a 1,152-byte M64/K32 stage: 128 bytes of
FP16 scales and 1,024 bytes of word-major packed nibbles. Activations use
16 packed code bytes plus one FP32 scale per G32. Each half2 lane pair computes
two activation rows. Raw signed-code products accumulate for exactly one G32;
the per-lane prefix magnitude is at most 2,048 and is therefore exact in FP16.
The raw dots are widened after every group, weight and activation scales are
multiplied in FP32, and totals accumulate with FP32 FMA. This is exact for the
defined post-quantization equation and deliberately retains the per-G32 cost
that a full-K half rail would avoid only by changing arithmetic semantics.

`worker.cu` checks every device-produced activation code and scale, planner
coverage, all outputs for finiteness, and 256 independent complete CPU dots per
arm. The same M128 kernel safely handles M64/partial leftovers by padding its
shared loads with the final valid row and guarding stores.

Build without using a GPU:

```sh
python3 build.py
```

The build runs CPU layout/arithmetic checks, compiles for SM60, rejects local
loads/stores in the selected pair kernels, and writes `manifest.json`,
`resources.txt`, `worker.sass`, and `sass-audit.json`. The resulting worker has
not been executed. A future GPU controller must reserve the user-designated
GPU1, use a fresh process and standard coordination/watchdogs, then begin with
partial-tile semantic and sanitizer runs before any timing.
