# GPU safety and archived harnesses

No GPU access is needed to read the results, verify hashes or rerun the CPU
checks. This publication did not reserve or use GPUs. All reservations recorded
in the completed studies were released; old tokens are not permission to run.

## Before any new GPU experiment

1. Work in a separate copy; do not overwrite the sealed archive or its build
   inventories. Inspect source and rebuild host executables for your machine.
2. Review active coordination entries and obtain a new reservation. Select the
   intended device by verified UUID and hold the reservation across workers,
   gaps, log review and final health checks.
3. Adapt hard-coded paths, CPU affinity, GPU UUIDs and coordination checks to the
   actual host. Do not bypass failed guards or reuse the historical token. Keep
   the controller free of CUDA initialization and launch a fresh worker per run.
4. Persist inputs/hashes/hypothesis before launch. Check device occupancy, ECC
   and storage/log health; stop on failed queries, unexpected processes,
   correctness errors, CUDA errors, timeouts or unhealthy devices.
5. Establish correctness before timing. Include preparation, scale/offset
   corrections and reductions; use paired controls, warmup, rotated order and
   fresh-worker repetitions. Record exclusions and preserve outliers.
6. Do not automatically reset or retry after a hang. Terminating a worker does
   not prove GPU recovery. Arrange console/power-recovery access before any
   separately reviewed unknown-instruction experiment.

On the original shared four-P100 rig, only one four-GPU job may run at a time;
such runs require the `.xsession-errors` watchdog. A desktop crash loop can fill
the disk. Standard runs use the production environment stack with `env -i` and
CPU affinity 0-11; under nsys, use plain `env` so profiler injection variables
are retained. These are host-specific requirements, not universal CUDA settings.

## Do not reuse the original HFMA2 execution harness as-is

The original `archive/README.md`, patcher and supervisor are historical. The
[later audit](../archive/research-20260908/REPORT.md#4-hfma2-audit-keep-observations-narrow-conclusions)
documents incomplete round-trip execution validation, an exact-half oracle
bug, insufficient patcher/control-word checks, scheduling/template concerns and
timeout/recovery weaknesses. Disassembly of all 256 selector combinations did
not validate all 256 on hardware. One merge template returned an illegal
instruction error; it did not prove all merge modes unsupported.

Any future binary mutation needs explicit candidate review, a repaired harness,
ELF-aware single-instruction patching with mandatory exact baseline checks,
verified scheduling and independently initialized operands/destination.
Generating or disassembling a cubin does not establish safe executability.
There is no broad opcode-fuzzing command in the new publication tools.

The newer SAD/W4A16 studies used compiler-generated documented instructions.
Their supervisors still embed original-machine assumptions and are evidence of
the original workflow, not turnkey scheduling services for another host.

NVIDIA documents process-invalidating illegal-instruction errors in the
[CUDA Driver API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__TYPES.html)
and reset/recovery limitations in the
[nvidia-smi manual](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
No firmware, driver replacement, automatic reset or production integration is
part of this repository's reproduction instructions.
