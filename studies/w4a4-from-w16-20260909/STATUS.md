# W4A4-from-W16 status — paused 2026-09-09

The user explicitly paused this study after the level-1, level-2 and level-3
admission tests and the complete exact-G32 half2 control had finished.

- 2x target: not reached.
- Best idealized packed arithmetic: 1.3553x versus HFMA2.
- Best prebuilt LUT consumer: 1.3609x versus matched resident HFMA2.
- Complete exact W4A4: 4522.31 us gate/up and 3629.11 us down, versus matched
  W16 at 2459.97 and 2348.57 us.
- Semantics, memcheck and racecheck: pass.
- Model quality: not run; no candidate passed the speed gate.
- GPU1: reservation released at this checkpoint; no worker remains active.
- Git: no commit, push or PR for this study.

Read `RESULTS.md` before resuming. Do not repeat the centered packing,
normalized split, PRMT repair, prebuilt LUT, or prior bit-plane tests without
materially new evidence. The remaining ordinary-kernel tuning idea is to test
an M64 token tile or compressed shared operand stage to restore two-CTA
occupancy; that is an optimization for a better W4 baseline, not an identified
path to 2x.

