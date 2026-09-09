# Active objective

Current user target: 1.9x against original W8A16 AffinityWave, including dispatch. NOT achieved. Accuracy remains byte-identical OR paired model PPL within +/-0.003 with KLD-pair evidence.

After 36 new variant screens, confirmed M256 complete pipelines (3 fresh workers/projection): F32 storage 1.786183x GU / 1.842659x down; FP16 storage 1.835868x / 1.885056x. The serving profile uses F32 wire. See TARGET-1P9.md and target-1p9-confirmed.json. Model PPL and full-model throughput are NOT qualified.

Selected code: compact-plan-f32/ and compact-plan-dimensions/. Isolated model mode 6 implements the F32 pipeline with reusable per-device/per-stream/per-host-thread scratch; mode 6 is built only, not model-tested. Original production library unchanged.

The pinned 37.8 GB Q8 GGUF remains deleted; only about 11 GB is free. Storage-location question pending. Do not substitute Q4 silently or claim full-model speed/accuracy.

No commits, pushes or PRs. Platform goal tracker still holds the earlier paused W4 goal; do not mark that goal complete to bypass its replacement restriction.
