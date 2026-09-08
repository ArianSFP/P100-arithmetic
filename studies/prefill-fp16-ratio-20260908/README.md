# FP16 / FP32 prefill ratio investigation

Follow-up to ../prefill-20260908/RESULTS.md. The user subsequently narrowed the
optimization target to **2048+ tokens**, as close as possible to2x relative to
FP32 accumulation. Physical GPU1 only. No production integration or model-speed
claim. Q8_0 inputs and exact FP16 operand preparation are retained.

All tools use CUDA12.8, GCC14, sm_60 and cuBLAS12.8.3. Numeric comparisons are
against explicit COMPUTE_32F/F32 output and COMPUTE_16F/F16 output widened to
F32. An exact candidate matches its own arithmetic-mode baseline over the
entire output, including signed-zero bits. This does not claim FP16 equals
FP32 or establish the model's paired-perplexity tolerance.

## Artifacts

- `worker-v1.cu` / `worker-v1`: original stage/algorithm sweep.
- `worker.cu` / `worker`: vectorized decoding and explicitly labelled cached
  weight variants; original default algorithms retained for this comparison.
- `frozen.cu` / `frozen`: fresh-seed confirmation of per-shape exact selectors,
  faster decoding and cache variants; includes tuned FP32 controls.
- `layout.cu`: eight combinations of input storage and output orientation,
  with separate input transposes fully charged.
- `layout_fused-unpadded.cu`: fuses activation conversion and weight decoding
  with the required transpose. The extended screen tries all algorithm IDs0..23.
- `layout_padded.cu`: leading-dimension padding screen.
- `layout_m_padded.cu` / current `layout_fused.cu`: zero-row padding screen;
  rejected after all retained padded variants lost.
- `layout_native.cu`: uses the backend's aligned native Q8 decoder as the
  ordinary-layout control. Donor source is attributed in the file. This is a
  stronger control than the original scalar or four-value study decoder.
- `run*.py`: bounded fresh-worker supervisors. Every worker receives an empty
  environment plus explicit CUDA paths, GPU UUID, and taskset CPUs0–11.
- `confirm_large-paired.cu`: frozen paired confirmation. Current
  `confirm_large.cu` additionally accepts isolated-mode timing arguments.
- `control.py`: interactive controller holding the shared rig lock through
  the final test/build/analysis gaps. It refuses occupied GPU1, records hashes,
  monitors desktop-log/disk limits, and stops on a failed worker without reset.
- `*-summary.json`, `*-summary.txt`: reanalysis of complete retained samples.
- `run-*`, `frozen-*`, `layout-*`, `large-*`: raw events, numerical checks,
  commands, hashes, clock/power telemetry and final health. Sanitizer runs are
  separate from performance runs.

## Layout encoding

The logical product is C[M,N] = W[M,K] * X[K,N]. Original physical W stores K
contiguously per output feature; original X stores K contiguously per token.

| Bit | Effect |
| --- | --- |
| 1 | Materialize W with output features contiguous (transpose original W) |
| 2 | Materialize X with tokens contiguous (transpose original X) |
| 4 | Compute C transpose by swapping operands, then transpose/widen output |

Examples: layout0 is original TN. Layout2 is TT with transposed activation.
Layout5 swaps operands and transposes W. Every transformation preserves all
operand bits. Fused preparation kernels are compared against all independently
prepared input words before timing, not only a handful of output dots.

## Reproduction

Do not launch outside a freshly coordinated reservation. Do not reuse a
released token. The shared /tmp/affinitywave-4gpu.lock must be held; other
sessions' reservations and GPU0's desktop must be respected.

Compile a selected source while holding the agreed build/test window:

```sh
/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-14 -O3 -std=c++17 -arch=sm_60 \
  studies/prefill-fp16-ratio-20260908/layout_native.cu -lcublas \
  -o studies/prefill-fp16-ratio-20260908/layout_native
python3 studies/prefill-fp16-ratio-20260908/control.py
```

The controller accepts one JSON request per line, then `release` after all
workers/builds/analysis are complete:

```json
{"tag":"native-up-2048","binary":"layout_native","args":["4352","2048","5120","802110"]}
```

Append `--check-only` to worker args and set `"memcheck":true` for the bounded
sanitizer check. The controller records each exact command and binary/source
hash. A performance failure or numerical mismatch latches STOP.

A full-cost timing recomputes weight decoding, activation conversion,
layout transforms, GEMM and final F32 output every iteration. Allocation and
host transfers are excluded equally. Cached-weight arms in the initial
study are labelled separately and cannot be substituted for full-cost data.
Multiple calls between CUDA events amortize event/host-submit overhead; fresh
workers and rotated samples avoid retaining one chosen fast launch.

## Final confirmation invocation

`confirm_large` arguments are `M N K seed fp32_algo fp32_layout fp16_algo
fp16_layout native_fp32_algo native_fp16_algo fp32_ld_pad fp16_ld_pad`.
The default executes the frozen ABBA matrix. Append `--check-only` for
verification only; append `MODE ARM` for isolated timing (MODE0=FP32,1=FP16;
ARM0=default TN,1=tuned TN,2=selected full-layout path).

The selected recipes are in plans.json; confirmation-jobs.json records the
exact tested requests. Paired and isolated modes use different timing
contexts and are reported separately. The2048 up recipe remains variable.

The controller releases normally after30 minutes; any continuation needs a
fresh token/lock claim. No benchmark worker is automatically restarted.
