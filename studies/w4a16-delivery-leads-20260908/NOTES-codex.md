# Handoff: GPU round, 2026-09-08

LATEST USER TARGET:1.5x on GEMMs relevant to2k+ prompts. See
LONG-PROMPT-TARGET.md and shape-inventory.json. Historical4k code/wiki routes
put~74%/~81% of useful work at M>128; a64-token-only win is insufficient.
Actual current cohortrail service dispatches a different M64 T64 kernel than
the frozen primitive. Replay that control and real ragged groups next. All
GPUs released; no new GPU launches after this clarification. GPU2/3 belong to
another session's server, left untouched. No full-model capture yet.
Subsequent coordination: GPU1/global lock now held by
prefill-fp16-ratio-1788898174313271440-gpu1. Do not overlap or reclaim it.

The initial CPU-only phase is superseded by the user's GPU0/GPU1 authorization.
See RESULTS.md for the measured round. Compiler pinned12–23; GPU workers0–11,
env-i, one fresh worker at a time, no compile/timing overlap, shared lock held
through build/analysis gaps. GPU2/GPU3 untouched. No process killed or GPU reset.

45 lead configurations now compile; five offline test groups pass. Winner:
delivery_fused_u32 (M32N64,128threads,88regs,12KiB shared,no spills), grid5*56.
It fuses FP32→FP16-RN→FP32 activation staging into the GEMM; ALL dot-product
arithmetic stays FP32. Gate/up ~1.55x including conversion reproduced onGPU0/1;
down ~1.35–1.36x, so general1.5x remains open. Not a model/service result.
All new outputs match frozen sequential FP32 winner, NOT necessarily split-K T64.
No model PPL/KLD validation this round; production accuracy remains unqualified.

Closed in this round without new evidence: tiled activation prep wins kernel-only
but loses total to fused row input; 4x8/M16 geometry lost; vector prep only ~1–2us;
down fused grids2/3/4/6/7/8 lost to5; U16 lost toU32. V4 introduced a redundant
staging bounds check and regressed; V5 restored the compile-time M32 guard.
Do not reintroduce that guard. Frozen snapshots v1..v6 preserve source/build
provenance; current build adds an accuracy-only raw-conversion witness to v6.
Raw artifacts contain prelaunch hashes, commands, health, timing and correctness.
Next useful step is real expert traces/service comparison plus accuracy gates,
and a separate down-shape scheduling investigation, not another blind grid sweep.

No production files, shared Claude memory, commits or pushes were changed.
