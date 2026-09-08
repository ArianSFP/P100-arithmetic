# P100 arithmetic investigations

GP100 / Tesla P100 (SM60) research into low-bit inference arithmetic, compressed
layouts, and W4A16 kernels. Measurements were recorded on 2026-09-08 using CUDA
12.8, GCC 14, and P100-PCIE-16GB GPUs. This is a research archive, not a llama.cpp
patch or a ready-to-deploy inference backend.

## Latest W4A16 follow-up: round 3

[Round 3](studies/w4a16-r3-20260908/RESULTS.md) reduces single-vector latency
another **4.2-7.2%** versus the live round-2 winners by keeping the exact-order
reduction inside a CTA. FP16 inputs and FP32 arithmetic order are unchanged;
900 pipeline checks passed. The four-vector improvement is only 1.8% and remains
provisional. These are synthetic kernel results, not model-inference gains.
The round-2 baseline and broader arithmetic archive are described below.

## Main finding: faster W4A16 without reducing activation precision

The latest kernels preserve the original FP16 activation bits, use FP32
accumulation, and reproduce the previous study kernel's outputs bit-for-bit in
all recorded checks. Warp-wide activation sharing improves single-vector
performance; sharing weight decoding across four vectors provides a larger gain.

GPU1 results against the **previous winning kernel, rerun in each worker**:

| Output rows M | Reduction K | Vectors N | Previous, us | New, us | Lower latency | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5120 | 5120 | 1 | 49.579 | 40.949 | 17.4% | 1.21x |
| 17408 | 5120 | 1 | 154.405 | 116.389 | 24.6% | 1.33x |
| 5120 | 17408 | 1 | 147.173 | 114.523 | 22.2% | 1.29x |
| 5120 | 5120 | 4 | 136.411 | 67.477 | 50.5% | 2.02x |

These incremental gains were confirmed on three P100s. The final study contains
45 successful workers and 3,987 pipeline checks; its frozen timing set contains
36 fresh workers and 3,456 retained event samples. The weights use Q4_0's exact
codes and FP16 scales, losslessly repacked: **4.5 bits/weight including scales**.
See the [full W4A16 round-2 report](archive/w4a16-r2-20260908/RESULTS.md),
[kernel source](archive/w4a16-r2-20260908/kernels.cu),
[timings](archive/w4a16-r2-20260908/aggregate.csv), and
[final audit](archive/w4a16-r2-20260908/audit.json).

**These are synthetic-data kernel results, not measured model-inference gains.**
Timing includes the complete device pipeline, scaling and final reduction, but
excludes CPU repacking, allocation and host/device transfers. Byte identity is
against the sealed study baseline, not stock llama.cpp. There is no stock-kernel
comparison, real-model perplexity/KLD validation, logits comparison or tokens/sec
measurement here. Activation precision was not traded for the reported gains.

## What the arithmetic investigation found

| Investigation | Supported conclusion | Evidence |
| --- | --- | --- |
| Hidden GP100 instructions | No hidden INT4/INT2/FP4 multipliers, DP4A or tensor cores were discovered. The search was bounded, not exhaustive silicon coverage. | [Audited instruction report](archive/research-20260908/REPORT.md) |
| HFMA2 selectors | Known broadcasts and limited FP32-source witnesses were observed. All 256 selector combinations were enumerated **offline**, not executed. Original harness limitations prevent stronger claims. | [HFMA2 audit](archive/research-20260908/REPORT.md#4-hfma2-audit-keep-observations-narrow-conclusions) |
| Unsigned byte SAD | Documented PTX emits native `VABSDIFF4.U8.U8.ACC` on SM60; exact bit-plane W2/W4 x A8 constructions passed hardware checks. | [Instruction probes](archive/research-20260908/REPORT.md#1-native-unsigned-sad-the-strongest-new-local-finding) |
| Byte conversion | Explicit documented PTX fused byte extraction into `I2F.F32.S8 ... .B2`, saving one instruction in a diagnostic. No production speedup established. | [Compiler miss](archive/research-20260908/REPORT.md#2-an-actual-compiler-miss-signed-byte-conversion) |
| Endpoint-SAD and compressed layouts | Endpoint identities are exact; byte-sign-friendly plane packing works. Removing the AND alone did not make the old kernel faster. | [Algebra/offline study](archive/endpoint-sad-20260908/RESULTS.md), [hardware follow-up](archive/layout-lut-20260908/GPU-RESULTS.md) |
| W2A8 | Register-LUT improved two tested N1 shapes by 2.7-3.9%; endpoint-SAD improved long-K by 8.6% and N4 by 15.6%, against the improved integer control. | [Matched SAD/LUT results](archive/layout-lut-20260908/GPU-RESULTS.md) |
| W4A8 | Better ordinary-integer mapping reduced latency by about 9-26% against the older integer kernel. No convincing SAD/LUT win over the improved integer control. | [Attribution controls](archive/layout-lut-20260908/GPU-RESULTS.md#attribution-controls-and-interpretation) |
| W4A16 round 1 | Four-row activation reuse reduced latency by about 24-29% against the study's tiled two-row control. LUT and FP32 pre-widening did not beat it. | [Round-1 report](archive/w4a16-20260908/RESULTS.md) |
| W4A16 round 2 | Warp sharing and four-vector weight reuse produced the incremental gains above with unchanged study arithmetic order. | [Round-2 report](archive/w4a16-r2-20260908/RESULTS.md) |

A8 and A16 studies use different scale storage and arithmetic; their absolute
times are **not** a controlled comparison of activation precisions. The A8 study
does not include input activation quantization time. W2/W4 A8 synthetic formats
must not be reinterpreted as Q2_K, Q4_K or IQ codebooks without format-specific
scales and offsets. The [findings guide](docs/FINDINGS.md) records these limits
and the unsuccessful paths worth avoiding.

## Read or verify the evidence

- [Findings and chronology](docs/FINDINGS.md)
- [Reproduction, dependencies and portability](docs/REPRODUCING.md)
- [GPU safety and historical harness limitations](docs/SAFETY.md)
- [Snapshot provenance and exclusions](docs/PROVENANCE.md)
- [Per-file SHA-256 manifest](EXPORT-MANIFEST.json)
- [Original chronological research notes](archive/NOTES-codex.md)

From the repository root, Python 3.10+ is sufficient for a read-only check:

```sh
python3 scripts/verify_archive.py
python3 -m unittest discover -s tests -v
```

Verification hashes all archived files and independently checks the latest
W4A16 build provenance and frozen timing aggregates against raw logs. It does
not load CUDA, query a GPU, execute an archived binary, or establish new hardware
correctness. CPU source tests and analysis-only reproduction are documented
separately.

`archive/` preserves the original research tree byte-for-byte, except for the
excluded third-party tooling/venv and caches. It includes source, host workers,
cubins, PTX/SASS, vectors/seeds, raw logs, summaries and failed experiments.
Historical absolute paths and reservation tokens are provenance, not portable
configuration or active reservations. Do not run the archived GPU supervisors
without reviewing and adapting their guards in a separate working copy.

## Next useful experiment

Compare the winning W4A16 kernels against the actual production implementation
on identical real-layer inputs. Include repacking/dispatch costs and require
production byte identity or perplexity within +/-0.003 using KLD-pair methodology
before integration. Q4 and Q2 use is permitted; this does not waive accuracy
requirements. Nothing in this repository modifies a production inference build,
driver or GPU firmware.
