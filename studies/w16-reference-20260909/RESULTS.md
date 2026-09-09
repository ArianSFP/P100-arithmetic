# W16A16 AffinityWave reference — 2026-09-09

The matched W16A16 reference takes approximately 3.8–4.1% longer than original W8A16 Q8 AffinityWave at these two expert-prefill shapes.

| Projection | M per expert | N | K | Experts | W8A16 microseconds | W16A16 microseconds | W16 speedup vs Q8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gate/up | 256 | 512 | 2048 | 64 | 4624.04 | 4803.01 | 0.9631x |
| Down | 256 | 2048 | 512 | 64 | 4527.91 | 4714.19 | 0.9605x |

Three fresh workers per projection, all GPU2, original Q8 and W16 arms in each worker. Each worker uses two seconds of warmup followed by seven rotated timing rounds, six launches per event. Predeclared round 0 excluded, all other samples retained. Table times are medians of worker medians. Speedups are medians of within-worker ratios against the faster of original Q8's F32-backed and actual FP16 activation-input branches. Paired speedup ranges: GU 0.96260–0.96316; down 0.96047–0.96063. A8/A4 variants were also retained in the rotated benchmark; their raw timings are preserved.

## What W16A16 means here

Both weights and activations are stored as IEEE FP16, converted to FP32 when staged, and accumulated with the original two FP32 rails. `prepare.py` copies the frozen previous original Q8 kernel and replaces its three Q8 weight loading/decoding blocks with vectorized FP16 loads and conversions. M64/N128 tile geometry, 256 threads, 32 KiB shared memory, persistent scheduling, K order and output stores are retained. W16 uses a tiled 4096-byte stage versus Q8's 2176-byte stage, with no per-block weight scale load. This is a straightforward matched AffinityWave reference, not an exhaustive FP16 kernel search, cuBLAS benchmark, or FP16-accumulation/HFMA2 benchmark.

Weights are the same synthetic signed Q8 codes times scales used in the preceding investigation, rounded to FP16 for W16 storage. Activations are identical FP16 values. Original Q8's FP32 scale multiplication can retain bits that FP16 weight storage rounds away, so W16 is verified against its own CPU oracle rather than required to be byte-identical to Q8. Relative output L2 difference from Q8 is approximately 0.000199 for both projections. This is not a native FP16 model comparison or model PPL qualification.

Static weight predecoding, allocation, uploads, and descriptors are outside timing, as is static weight packing for the Q8 reference. W16 does not have a runtime quantizer. Consumer-only comparison is therefore appropriate for persistently stored FP16 weights. Storing expanded weights doubles code width; this experiment finds that removing Q8 decoding does not offset the larger weight delivery cost in this implementation. Exact cycle attribution would require further profiling.

## Validation and evidence

All six timing workers passed. Full outputs are finite; 256 sampled complete dots per arm match the corresponding CPU two-rail FP32 FMA oracle bit for bit. The original Q8 input branches are byte-identical. A separate M=33, two-expert gate/up compute-sanitizer run passed with zero errors, covering a partial M tile. GPU device reservations were recorded in both coordination logs and released; no GPU resets or production edits occurred.

Compiler: CUDA 12.8, g++13, sm_60, -O3, without --use_fast_math, shared across all arms. W16 uses 111 registers versus original Q8's 122; both have zero spills. The lower register count does not imply a throughput improvement. Production compiler flags and full-model behavior are not qualified by this standalone study.

`manifest.json` and worker metadata retain compiler command and source/binary hashes; `prepare.py` reproduces the source transformation. `analyze.py` verifies hashes, successful exits, numerical checks and sample counts before writing `summary.json`. Raw evidence is in `gu-r*.out`, `down-r*.out`, accompanying metadata/error files, `semantic.out`, and `build.log`.
