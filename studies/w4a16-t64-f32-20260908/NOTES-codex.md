# Handoff,2026-09-08

Latest user follow-up: target1.5x INCLUDING activation preparation next, with
FP32 held fixed. PLAN.md records the new milestone, paired latency budgets
and proposed activation-layout experiment. No new GPU measurements or claim
were made for this planning update. The earlier2x ambition is not achieved.

Prior user constraint: achieve2x T64 independently of FP32-to-FP16 accumulation.
Do not count the neighboring half-rail study's1.97x kernel/1.81x pipeline result.

This FP32-only bounded sweep is completed; the2x project target is NOT achieved.
Read RESULTS.md for1.28x/1.25x synthetic primitive results and their limitations.
Keep the faster sequential-FP32 candidate separate from bit-identical SK2.
Next must compare actual compressed T64 dispatch/operands and validate model
accuracy; no new half arithmetic or broad LUT re-sweep is justified by this run.

GPU2 and lock released18:50:44 UTC, both coordination logs updated. No process
left running, no driver/toolkit/production changes, no commits/pushes. Other
sessions' root README/NOTES and Q4 T64 study changes remain untouched.
