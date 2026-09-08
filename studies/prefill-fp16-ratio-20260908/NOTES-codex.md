Initial investigation completed, then user explicitly narrowed objective to 2048+
tokens and requested approaching 2x as closely as possible. No production change.

N256 screen: up GEMM alone1.975x, full pipeline1.688x; down1.791x/1.599x.
Original scalar weight decoder231us, activation16-18us, finish16-18us.
4-value half2 decoder162us, bit-identical complete decoded weights and outputs.
Frozen N512 tuned FP32/FP16 + decoder ratios1.897x up/1.975x down; cache gives
1.984x/2.080x. These are synthetic matrix pipelines, not model quality/speed claims.
AtN2048 tuning FP32 too exposes residual: up~1.69x, down~1.87x with vector decoder.
Changing batch can change half accumulation ordering; per-shape identity only.

GPU1-only guarded runs; GPU2/3 reservation respected. Shared lock held per suite.
24 screen workers,2 memcheck workers,18 frozen workers complete; all exited,
memcheck0 errors, ECC0, no reset. Fresh seed sets for frozen confirmation.
Sources/binaries/manifests/raw events/telemetry retained here. All reservations
explicitly released before next phase. Next:2048/4096/8192 operand-layout sweep,
including preparation/output costs and a tuned FP32 comparator.

GPU1 shared-lock handoff completed; persistent controller large-1788899908819010531 owns the bounded final suite. Native aligned decoder and fused input preparation validated, including memcheck. Full0..23 algorithm/eight-layout screen atN2048/N4096 completed. Most exact full-cost ratios~1.89-1.90; N2048 up TTalgo3 shows quantized timing variability and is not promoted from fastest samples. Padding investigation follows, then8192 and fresh-seed confirmation. No production edits.

FINAL:104 successful workers,10 clean memcheck workers, all controllers exited
and all GPU1 reservations explicitly released, ECC0 and no GPU1 compute client.
No reset/clock/production change or commit/push. Final confirmation consists of
18 paired fresh workers and24 isolated-mode workers. Every tested selected output
matched its arithmetic-mode baseline before/after timing. Broader final Q8
fixtures include−128…127 values and scales0.003…0.015.

Four of six2048+ shards reach~1.89–1.91x against independently tuned FP32, all
preparation/output costs charged.2048 up varies~1.71–1.87x (paired1.714x);8192
down1.814x. No uniform2x and no full-model prefill/PPL claim. N8192 down FP16
recipe is unchanged TN/default; apparent gains between identical paired arms
were ordering effects, resolved by isolated workers. Zero output-row padding
128/256/512 lost and is rejected; storage padding did not remove variation.
RESULTS.md,plans.json and preserved source/binaries/raw data are the handoff.
