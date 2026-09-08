# Reproducing the GPU delivery study

Initial preparation was CPU-only; the user subsequently authorized GPU0/GPU1.
See RESULTS.md for measured outcomes. A new run still needs fresh user authority
and coordination: a supervisor flag is not a substitute for a reservation.

CPU-only reproducibility:

```sh
taskset --cpu-list 12-23 python3 build.py
python3 test_offline.py
```

After explicit release, read both coordination logs and current rig instructions,
check host GPU0/GPU1 health/occupancy, and record a fresh claim for token
`w4a16-delivery-leads-20260908-gpu01` in both logs. Start the supervisor with
`--user-released-gpus` through the approved host-access route. It verifies the
claim, obtains the shared lock and uses the previous study's guards/telemetry.
No direct worker launches. No automatic GPU acquisition is scheduled.
Requests accept `"gpu":0` or `"gpu":1` (default1), one worker at a time. The
existing GPU0 editor PID/path exception is specific to this session and must
be re-reviewed, not blindly reused. The lock holds both GPUs across gaps.

Send one reviewed command at a time. Each creates a fresh worker. Stop on a
failure; never reset automatically. `--gpu-approved 1` is additionally required
by the new worker before it can initialize CUDA.

1. All-candidate memcheck, then persistent-CTA synccheck:

```json
{"tag":"smoke","args":["--gpu-approved","1","--tokens","33","--n","128","--k","96","--experts","2","--reps","1","--iters","1","--warmup-ms","0"],"sanitizer":"memcheck"}
{"tag":"persistent-sync","args":["--gpu-approved","1","--tokens","33","--n","128","--k","96","--experts","64","--reps","1","--iters","1","--warmup-ms","0"],"sanitizer":"synccheck"}
```

2. Gate/up and down screening. Keep all three T64 controls and frozen row-major
winner in every worker. Each new lead must match every winner output word.

```json
{"tag":"gateup","args":["--gpu-approved","1","--tokens","64","--n","512","--k","2048","--experts","16","--grid","3","--reps","7","--iters","8","--warmup-ms","2000"]}
{"tag":"down","args":["--gpu-approved","1","--tokens","64","--n","2048","--k","512","--experts","16","--grid","3","--reps","7","--iters","8","--warmup-ms","2000"]}
```

Rank `prep=1`, not kernel-only. Use the fastest included T64 control's paired
median. Do not select a losing control, skip transpose, pool raw times across
workers or multiply gains by the separate model packing result.

3. Use `--only delivery_fused_u32 --grid 5` for the measured winner
plus the same parameters for fresh repeated workers. Also test33/65/128 tokens,
K512/K2048, outlier/cancellation/range patterns1/2/3 and the winning lead under
racecheck. For strict byte identity the existing baseline's input semantics
must be retained. Then investigate real Q4_0/Q4_1 expert traces and model quality.

`--raw-witness 1` is accuracy-only: the fused path receives raw FP32 values,
while controls/oracles receive their FP16-rounded values. This intentionally
tests the conversion, including ties/subnormals, and is NOT a timing comparison.
analyze.py excludes these and sanitizer runs. Ordinary timing includes conversion
inside the fused GEMM in both prep modes; no separate pass is necessary.

4. Release the controller, inspect final device health, append explicit final
release to both coordination logs. No production integration or commits/pushes
are authorized by this preparation task.

The build is generated from the frozen previous worker; edit build.py/leads.cuh,
not build/worker.cu. Old sources/results are left unchanged. The extra weight
layout is an offline static repack in this primitive, not free deployment cache
packing. Any model comparison must account for its actual cache/miss policy.
