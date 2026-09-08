# W4A16 prefill investigation publication

Latest report: [DIRECT-AND-MAPPING-RESULTS.md](DIRECT-AND-MAPPING-RESULTS.md).
At M256/e64 the leading equal-M screens reach approximately 1.42x current T64
gate/up and 1.37x down, including activation conversion and retaining FP32
accumulation. The requested 1.5x complete-pipeline target remains unmet.
Candidate weight repacking/upload is outside these primitive timing intervals;
whole-prompt routing, cache costs and current-Q4 ragged replay remain unverified.

This publication includes the long-prefill, delivery-leads, t64-f32, t64-r1,
tile256 and verification studies, plus paused w4a4-long-prefill. The two frozen
control sources in qwen35-q4-affinitywave and qwen35-q4-t64 are included as build
dependencies; their surrounding model studies are not part of this publication.
Other sessions' decode/prefill work and root README/notes changes remain separate.

Sources, generated SASS, binaries, historical manifests and raw results retain
their original bytes. Historical absolute paths and GPU UUIDs describe this rig;
reproduction elsewhere requires a reviewed path/device configuration. Do not
silently disable manifest or reservation checks. Captured routing artifacts
outside this repository are referenced by path/hash, not bundled here.

CPU checks: test_offline.py, test_wide_model.py, ../w4a16-tile256-20260908/
test_direct_stage.py and test_one_rail512.py, and ../w4a16-verification-20260908/
test_ragged_proxy.py. These complement the archived GPU numerical and sanitizer
results; they do not establish whole-model speed or quality. W4A4 is paused.
