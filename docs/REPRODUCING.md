# Reproduction

Run commands from the repository root unless stated otherwise. Preserve
`archive/` unchanged: its source, binaries and logs are tied together by hashes.

## Read-only archive verification

Python 3.10+ and its standard library are sufficient; no CUDA installation,
GPU driver, third-party Python package or archived executable is used:

```sh
python3 scripts/verify_archive.py
python3 -m unittest discover -s tests -v
```

The verifier checks all manifest entries and independently reconstructs the
latest W4A16 timing medians/counts from raw logs. A PASS validates the archived
evidence's internal consistency, not new hardware behavior. Tests exercise
modified/missing/extra files and unsafe paths as well as the successful case.

## Rerun the finite-FP16 and lossless-layout CPU tests

The original tests use C++17, IEEE host floating point, and GCC's `__int128`.
The recorded compiler is GCC 14. This compiles reviewed source into a newly
created temporary directory, leaving the frozen builds unchanged:

```sh
probe_scratch=$(mktemp -d)
g++-14 -O3 -std=c++17 -ffp-contract=off -Wall -Wextra \
  archive/w4a16-r2-20260908/cpu_tests.cpp -o "$probe_scratch/cpu-tests"
"$probe_scratch/cpu-tests"
g++-14 -O3 -std=c++17 -ffp-contract=off -Wall -Wextra \
  -fsanitize=undefined -fno-sanitize-recover=all \
  archive/w4a16-r2-20260908/cpu_tests.cpp -o "$probe_scratch/cpu-tests-ubsan"
"$probe_scratch/cpu-tests-ubsan"
```

Expected output includes `finite_half=63488`, `exact_W4_products=1015808`,
`random_lossless_blocks=65536`, `address_bijections=3`, and `CPU_PASS`.
These checks do not measure GPU timing or establish model accuracy.

## Re-run historical aggregate analysis without GPUs

The original analyzers write their summary files. Use a scratch copy to avoid
altering the archive; keep sibling study directories together because the final
study verifies the sealed previous cubin:

```sh
analysis_scratch=$(mktemp -d)
cp -a archive "$analysis_scratch/research"
python3 "$analysis_scratch/research/w4a16-r2-20260908/analyze.py" aggregate
python3 "$analysis_scratch/research/w4a16-20260908/analyze.py" aggregate
python3 "$analysis_scratch/research/layout-lut-20260908/analyze_gpu.py"
```

Round 2 should report 36 workers, 432 configuration checks and 3,456 retained
samples; round 1 has 12 workers, 204 checks and 1,632 samples. The A8 layout
study has 24 frozen workers and 2,496 samples. Historic failure/prelaunch-only
records remain separate from successful benchmark workers.

## Building and running new GPU experiments

This requires manual adaptation, not a copy-paste command from the archive.
Read [SAFETY.md](SAFETY.md) first. No GPU execution is part of verification.

The recorded toolchain was `/usr/local/cuda-12.8` (nvcc 12.8.61), GCC 14,
`-arch=sm_60`, CUDA Driver API workers, `cuobjdump`/`nvdisasm`, and
Compute Sanitizer. Verify that the chosen compiler actually supports SM60;
do not infer toolkit version from `nvidia-smi` or replace a working driver.
CuAssembler was used only in the historical HFMA2 work, not the winning
compiler-generated W4A16 kernels. Its checkout/version is documented in
[PROVENANCE.md](PROVENANCE.md); it is not vendored here.

In a separate copy, inspect the study's build script and inventory for the
exact flags. Adapt toolkit/compiler paths and supervisor settings, then rebuild
and audit SASS. Some workers also embed absolute sibling-cubin paths; locate
them before rebuilding:

```sh
rg -n '/home/arian|/usr/local/cuda|COORD|TOKEN|GPUS|CUDA_VISIBLE_DEVICES' \
  archive/w4a16-r2-20260908 archive/w4a16-20260908 \
  -g '*.py' -g '*.cpp' -g '*.cu'
```

Changed sources/binaries require new inventories and new result directories.
Never edit an old hash record to make a changed experiment appear to be the
measured build. Keep the sealed control cubin available and verify its hash;
new toolchain output may change code generation and therefore constitutes a
new experiment. Establish CPU checks, GPU numerical checks and sanitizer
checks before paired timing, under a fresh coordinated reservation.

For deployment, compare actual Q4_0 production dispatch on real-layer inputs,
include original repacking and integration costs, and require production byte
identity or ppl +/-0.003 using KLD-pair methodology. This archive supplies no
stock llama.cpp integration, model weights, model-quality result or turnkey
end-to-end benchmark.
