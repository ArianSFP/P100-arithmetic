# W4A4 from the dual-rail W16A16 kernel — 2026-09-09

Status: paused by explicit user request after the planned admission tests.
Measured checkpoint and continuation boundary: `RESULTS.md` and `STATUS.md`.

Goal: determine whether exact-on-quantized-codes W4A4 arithmetic can make the
confirmed M256 P100 kernels approximately twice as fast as the matched W16A16
pipeline.

Follow `W4A4-research` in priority order 1 > 2 > 3. First establish SM60
lowering and measured arithmetic ceilings for ordinary G32 half2, centred
signed-16 packing, packed FP32/FP64, and mixed HFMA2/DFMA. Then build the
minimum complete W4A4 pipeline around the winning fast-W16 M128 tile. Admit the
level-2 normalized split or PRMT-residue route only if its complete operation
budget can beat the level-1 controls. Admit the level-3 weight-derived LUT only
after the higher-priority paths have measured ceilings or real-kernel results.

All complete timings include GPU activation quantization from the matching
input wire, planner, M64 leftovers, M128 pairs, group scaling and output. Weight
packing remains offline as in the W16 comparison. Require independent integer
dot oracles, full-output finiteness, memcheck and racecheck before confirming a
winner. Exactness on A4 codes is separate from model quality; any production
candidate still requires PPL within +/-0.003 using the KLD-pair methodology.

GPU1 is reserved for this study. All GPU work is sequential and guarded by the
device lock, coordination records, health/ECC/disk/temperature watchdogs and a
hard deadline. No production edit, commit, push or GPU reset.
