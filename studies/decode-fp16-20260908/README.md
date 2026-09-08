# Decode FP16 accumulation study

Read [RESULTS.md](RESULTS.md) for measured speedups and their limits.
This is an isolated synthetic Q4_0 / FP16-activation kernel study. It does not
modify llama.cpp or qualify FP16 for production inference.

## Files

- `worker.cu`, `kernels.cuh`: current worker and candidates; `common.hpp` and
  `r3-baseline.cuh` retain the archived FP32 controls unchanged. Baseline source
  paths/hashes are in `baseline-provenance.json`.
- `snapshots/v1` through `snapshots/v4`: every earlier executed source/binary,
  build logs and SASS. Current device SASS is identical to v4; only the host
  timing rotation changed afterward. Never combine data from different builds
  without using each worker's source/binary hashes.
- `plans.json`: frozen **experimental**, shape-specific choices. Not a safe
  production dispatcher; byte identity/PPL gate has not been passed by FP16.
- `run-*/`: raw JSON lines, telemetry, commands, health and prelaunch/result
  records. `final-*` are primary paired confirmations; `confirm-*` were
  superseded after discovering a rotation defect for seven-configuration sets.
  `isolated-*` check order sensitivity; `kernelonly-*` omit activation conversion
  explicitly and are diagnostic only. `stress-*` are numerical diagnostics.
- `cpu_checks.py`, `audit.py`, `analyze.py`, `report.py`: offline checks and
  summaries. The audit proves inventory/run consistency, not model accuracy.
- `ARTIFACT-MANIFEST.json`: per-file SHA256 excluding itself and Python caches.

## Arithmetic modes and names

`m` = one-CTA reduction; `g` = global partials plus a second reduction;
`p` = two/four lanes per row, one-CTA reduction. `r` is rows/lane, `b` token
reuse, `w` warps/CTA, `s` group stripes. `p` names encode `l` lanes/row.

Mode modulo3: 0 = FP32 dot/scaling/accumulation; 1 = two-lane FP16 group dot,
then FP32 group scaling/accumulation; 2 = FP16 group dot and local scaled sums.
**All modes retain final FP32 reduction/output.** Modes0–2 used adjacent code
pairs in v1; modes3–5 pair codes16 bits apart in each packed word; modes6–8 add
lane partitioning. New FP32 modes can regroup additions; archived `r3*` controls
retain exact round-3 arithmetic. Every candidate has its own independent CPU
complete-dot oracle; per-worker records also compare every output to round3.

## Reproduction

Offline (no GPU or reservation):

```sh
python3 studies/decode-fp16-20260908/cpu_checks.py
python3 studies/decode-fp16-20260908/audit.py
python3 studies/decode-fp16-20260908/analyze.py
python3 studies/decode-fp16-20260908/report.py
```

GPU reproduction requires a new reservation, reviewed coordination and adapting
`control.py` to your rig. Never reuse a completed token or overwrite retained
run tags. It holds the shared lock across build/worker gaps, isolates physical
GPU1 by UUID, checks ECC/processes/disk/desktop log, uses `env -i` plus CPU0–11,
and stops on failures/timeouts with no reset or automatic retry. Its maximum
hold is30 minutes; each worker is bounded to240 seconds. The controller runs
no CUDA work itself. Send `release` and retain final device health after runs.
`sanitizers.py` requires that same live controller to hold the lock and must run
only between controller workers, never concurrently with them.

After actual lock acquisition, `build.py` builds with CUDA12.8, GCC14, SM60,
FTZ disabled, explicit rounding in the arithmetic and host FMA contraction off.
The historical build is retained; do not rebuild it in place merely to verify
hashes. Worker arguments:

```
M K N seed family comma-separated-config-names rounds [include-activation-prep]
```

`all` runs the bounded compiled configuration set. Families0/1/2/3 are normal,
large finite activation range, alternating-sign near-unit, and subnormal.
Default preparation is included; setting the last argument0 excludes it and
must never be used as a full-pipeline numerator against a prepared baseline.
Primary event samples are9 rounds of12 full pipelines after a0.3-second warmup;
round0 is uniformly discarded. Final worker rotates order by one position per
round. Numerical checks precede timing and full-output repeatability follows it.
There is no CPU repacking, allocation or H2D transfer in timed events. One-time
lossless weight repacking and already device-resident FP32 input are shared
assumptions. FP32 activation-to-half conversion, compressed loads/unpacking,
scaling, local sums, reduction and FP32 output are charged in primary results.
