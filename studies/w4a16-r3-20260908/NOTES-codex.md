# W4A16 round-3 notes

The user explicitly authorized GPU2 after another session held GPU1. The GPU2
claim is recorded in both project coordination logs and held through the series.
Other sessions' untracked studies and shared root NOTES edits were preserved.

Findings: another 5.57%, 7.19% and 4.18% lower N1 latency for square, tall and
long-K shapes, respectively, versus live sealed round-2 controls. Exact-order
local reduction wins shape-specifically. N4 CTA tuning gives only 1.78%, marked
provisional. Packed-pair sharing loses its shuffle saving to extra widening.
All finite FP16 input bits and FP32 accumulation semantics remain unchanged.

20 successful GPU2 workers / 900 bit-identical pipeline checks; all three
sanitizer smoke tools report zero errors. Full evidence in RESULTS.md, raw
gpu-results/, aggregate.json and audit.json. No stock/model accuracy or e2e
result exists. No production edit, archive-content edit, commit or push.

The whole-archive strict verifier observed two extra ignored Python caches
under archive/scripts/__pycache__, generated during other study activity.
All manifest-listed hashes still match; preserve other sessions' files rather
than deleting them. Root NOTES-codex.md has unrelated concurrent edits and was
not modified by this study.
The cache files were absent at the final audit; this study did not remove them.
The interim observation is preserved in archive-check-observation.json.

GPU2 released after the 15:19:12 UTC final health check (5 MiB, idle, ECC0,
no target compute process) and artifact audit. Both coordination logs updated;
no remaining worker, reset, commit or push. Other GPU reservations preserved.
