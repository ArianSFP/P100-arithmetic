# FIGLU symmetry gate on GP100

Status: completed and paused after the final bounded experiment.  GPU1 is
released in both coordination logs.  See `RESULTS.md` and `STATUS.md`.

The supplied `P100-FIGLU-experiment` is primarily a W4A16 activation-derived
half-LUT proposal.  Its directly transferable idea for the paused W4A4 work is
the complement symmetry of a weight-derived mu8 table.  For each four-output
packed-byte table, `U(mask) + U(~mask) = L`, where each byte of `L` is the
corresponding row's L1 weight sum.  Storing only masks with bit 7 clear reduces
`table[256][32]` (32 KiB/CTA) to `table[128][32]` (16 KiB/CTA).

Admission sequence:

1. Prove the packed-byte identity exhaustively, including absence of cross-byte
   borrow, and compile SM60 probes to inspect actual lowering and resources.
2. Measure a same-worker *prebuilt-table* ceiling for full256, half128 and the
   resident HFMA2 control at R=16,8,4.  Half128 must reconstruct complement
   entries from runtime masks; results must match full256 exactly.
3. Kill the path if the optimistic half128 consumer is below 1.45x HFMA2.  The
   margin is required because table construction, four G32 slice barriers,
   activation-mask preparation, packed-field widening, plane combination,
   scaling and output are all still absent.
4. Only if that gate passes, implement a complete resident G32 component.  It
   must retain at least 1.30x after all mandatory work before a full GEMM is
   justified.

The measured half128 prebuilt consumer failed stage 3 by a wide margin.  A
separate final lead discovered during the audit is therefore admitted as an
addendum: build activation-derived full16 register tables whose uint32 values
pack four A4 tokens.  This avoids the shared-table construction traffic and is
four times denser than the supplied W4A16 helper.  Its first gate includes table
construction, runtime weight keys, exact G32 packed accumulation, correction,
scales, and persistent FP32 totals.  It must exceed 10.25 useful TMAC/s to earn
a streamed GEMM; below the production down threshold of 9.488 TMAC/s closes it
for the 1.3x objective.

The GPU work is single-device, UUID-isolated, lock-held, sequential and
watchdog-bounded.  No production/model source, commit, push or device reset.
