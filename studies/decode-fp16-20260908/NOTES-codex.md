# Decode FP16 investigation notes

## Prior evidence reviewed

Read-only shared memory: p100-fp16-half2-refuted.md; qwen36-p100-rig-benchmarking-gotchas.md; memory index, round3 execution/post-MTP plans and round3-phase0 results. Old sub-Q8 ban superseded by current user authorization. No shared memory edits.

- `/home/arian/llama.cpp-qwen36/results/decode-opt-20260712/phase4y/RESULTS.md`: integrated Q8 magic-half HFMA2 20.338ms vs integer 20.268ms (0.35% slower), despite primitive improvement. No model quality gate because performance failed.
- `/home/arian/llama.cpp-qwen36/results/dmmv-20260713/RESULTS.md`: Q8 DMMV F16/F32 15.17/14.70 tok/s (~1.032x, noisy F32); Q4 15.36/15.39 (~0.998x). Normal MMVQ 25.59/26.60 respectively. These are historical full-model results, not new paired measurements or FP32-accumulation comparisons against integer MMVQ.
- Seven-refutation memory: batched n5 half2 splice +411–462% vs integer anchor, not retested. Known compensated half schemes need >2x low precision rate and remain closed.

The new experiment is justified by September row-coalesced compressed W4A16 and FP16 activation semantics, rather than the older native-layout Q4/Q8 -> Q8_1 activation MMVQ path. It must not be sold as a reopened stock-Q8 optimization.

## Hardware reasoning

Official PCIe P100 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf
9.3 TF/s FP32,18.7 TF/s FP16,16GB model732GB/s bandwidth. Changing accumulation does not change this bandwidth or compressed weight size. Real rig measurements in previous memory are lower (~543GB/s effective ceiling for those access paths); do not confuse that measurement with the datasheet peak.

For Q4_0 N1 useful GEMV arithmetic intensity is 2/(18/32)=3.56FLOP/weight-byte before extra input/output/launch costs. N4 ideal weight reuse gives14.22FLOP/B, N8 gives28.44FLOP/B; extra conversion/issue cost is not included. Thus batching may expose compute while a single stream does not. An existing2048+ token KV context remains N1 for each new token, unlike prefill N2048. Full decode additionally includes attention, routing/communication and host work.

## Coordination

First sandbox controller was cancelled before acquisition; it executed no GPU work. Host-visible inspection found existing long-prefill controller PID1294021 holding the shared lock. Appended handoff request, starting a new guarded GPU1 waiter without bypass. CPU source and offline checks only until actual acquisition.

## Screens and new evidence
Actual GPU1 lock acquired run-1788903589380228722, no overlap. V1 adjacent-pair packing lost (0.71–0.92x fastest same-worker FP32). SASS showed redundant SHR/PRMT in half2 raw construction plus expensive adjacent nibble packing. V2 safe memcpy bitcast removed redundant pair assembly; separated nibble pairing maps codes already16 bits apart using one documented lop3 plus half subtraction. CPU pair oracle and all-output tail GPU checks passed, no spills. V2 square primary screen1.06x N1,~parity N4,1.38x N8. These are exploratory synthetic results, not acceptance or model accuracy. V3 explores global partial reduction and adds an exact-order FP32 B8 control; current archived B4 alone is insufficient to establish an accumulation-only gain at N8. All preparation/reduction is charged equally.

## Timing rotation audit
Initial33 confirmation workers all passed numerics, but timing rotation `(j+round*7)%config_count` does not rotate a7-configuration frozen set (N8 and downN4). This is a harness design issue, not a CUDA failure. Preserve all original confirm-* runs as non-primary evidence; change rotation to `(j+round)%config_count` and repeat ALL33 frozen jobs as final-* with identical plans/seeds. No kernel/source arithmetic changes, verify device SASS unchanged. Add isolated best-FP32/FP16 workers to check ordering sensitivity. Do not select the more favorable set.

## Final result and release
All119 workers completed successfully:2763 configuration checks,47092489 full-output comparisons,459913 independent CPU complete-dot emulations. Four memchecks, one synccheck and one racecheck clean. Final33 correctly rotated workers confirm1.042–1.091x N1 across five shapes; N4 ranges0.992–1.066x; N8 ranges1.280–1.403x. Sixteen isolated workers corroborate the up/down results;8 kernel-only workers show conversion removal still does not approach2x. All90 FP16 configurations fail the finite-range stress while FP32 remains finite; production accuracy gate NOT passed. No full-model run or integration.

GPU1 released normally, final UUID GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b,5MiB,0%utilization,405MHz idle SM,ECC0; no compute client. Final raw health is run-1788903589380228722/final-health.txt. No production changes, resets, commits or pushes. Root .gitattributes receives only this study's byte-preservation entry; unrelated working-tree changes untouched.

A CPU-only analyzer display bug for isolated half-only workers (no FP32 denominator) was corrected to print no paired arm; underlying raw/summary data were unchanged. This was not a GPU run failure or a reason to rerun benchmarks.
