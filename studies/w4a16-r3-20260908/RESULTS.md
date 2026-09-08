# W4A16 round 3: another 4.2-7.2% lower single-vector latency

2026-09-08, physical GPU2, Tesla P100-PCIE-16GB / SM60. No activation
quantization, FP16 multiplication/accumulation, production change or model run.

## Outcome

The new single-vector winners compute the same 32 accumulation stripes within
one CTA and reduce them in shared memory in the **same FP32 tree order**.
They avoid the global partial-output buffer and the separate reduction launch.
All tested outputs are bit-identical to the sealed round-2 study baseline.

Three fresh workers per shape, median of worker medians; control and candidate
run interleaved on the same GPU with the same data:

| M x K | N | Live round-2 winner, us | Candidate, us | Lower latency | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5120 x 5120 | 1 | 40.331 | 38.085 | 5.57% | 1.059x |
| 17408 x 5120 | 1 | 115.829 | 107.504 | 7.19% | 1.077x |
| 5120 x 17408 | 1 | 113.568 | 108.816 | 4.18% | 1.044x |
| 5120 x 5120 | 4 | 67.227 | 66.032 | 1.78% | 1.018x |

The N4 candidate is a CTA-sizing change, not local reduction. Its small gain
is **provisional, not a promoted replacement**: only one GPU and three workers
were measured. The substantial N4 local-reduction variants were slower.

| Shape / N | Frozen candidate | Worker-median range, us | Paired latency reductions |
| --- | --- | ---: | --- |
| Square / 1 | `r3_200_r1` | 37.685-38.320 | 4.99%, 5.48%, 6.66% |
| Tall / 1 | `r3_202_r2` | 107.307-107.536 | 7.36%, 7.34%, 7.10% |
| Long K / 1 | `r3_200_r1` | 107.995-108.971 | 4.69%, 4.32%, 4.18% |
| Square / 4 | `r3_112_r2` | 65.328-66.635 | 1.00%, 1.78%, 2.81% |

These are additional gains versus the live previous winner, not comparisons
against round-1 or weak native-layout kernels. Historical GPU1 times are not
used as denominators. Ranges are observed repetitions, not confidence intervals.

## Mechanism and tradeoffs

- Square/long-K N1: R1 (32 output rows per CTA), eight warps, each warp computes
  four of the 32 stripes. Dynamic shared storage: 4,096 bytes/CTA.
- Tall N1: R2 (64 output rows per CTA), 32 warps, one stripe per warp. Dynamic
  shared storage: 8,192 bytes/CTA; 32 registers/thread.
- N4 provisional candidate: R2, eight warps rather than the previous four,
  with the existing four-vector weight reuse and separate reduction.

Each stripe keeps exactly the original group sequence `g=s, s+32, ...`, the
32-element sequential FP32 dot and group-scale FMA. After all stripes finish,
six CTA barriers delimit initialization and the five additions at offsets
16/8/4/2/1. No atomics or global completion flags are involved.

For N1, this removes `2*M*32*4` bytes of global partial-output write/read:
1,310,720 bytes on the square/long-K shapes, 4,456,448 bytes on the tall shape.
This is logical traffic, not a hardware-counter measurement. The fused path
also changes CTA ownership/geometry; the timing does not isolate traffic,
launch overhead and scheduling as separate causal contributions.

The compressed weights and separate FP16 scale plane are unchanged from round 2.
Q4_0 codes/scales occupy 18 bytes per 32 weights (4.5 bits/weight including
scales), with the same original lossless repack. No new metadata, repack or
activation-preparation pass is introduced. The diagnostic worker still allocates
its old partial buffer for the live control; a fused-only deployment does not
need that buffer.

## Failed or limited alternatives

Packed-half-pair broadcast halves SHFL count but duplicates widening across
lanes. At R2/N1, SHFL falls 32 -> 16 while HADD2.F32 conversions rise 3 -> 34;
at N4, SHFL falls 128 -> 64 and conversions rise 6 -> 130. FFMA, BFE and I2F
counts remain unchanged. These HADD2 forms widen exactly to FP32, not half
arithmetic. This tradeoff did not produce a leading kernel.

Selected frozen component results (all total device-pipeline times):

| Configuration | Square N1, us | Square N4, us |
| --- | ---: | ---: |
| Sealed round-2 shape-specific control | 40.331 | 67.227 |
| Same R2 warp-sharing family, one warp/CTA, N1 path | 41.195 | 114.747 |
| Same R2 warp-sharing family, eight warps/CTA, N1 path | 40.197 | 117.749 |
| Paired sharing R2/N1 path | 41.493 | 122.293 |
| Eight-warps R2/N4 path | 61.141 | 66.032 |
| Paired sharing R2/N4 path | 72.661 | 77.749 |
| Local reduction R1/eight warps/N1 | 38.085 | 128.251 |
| Local reduction R2/32 warps/N1 | 45.717 | 125.408 |
| Local reduction R2/32 warps/N4 | 79.701 | 79.467 |

Local reduction is shape-dependent: the tall winner regresses square/long-K,
and the N1 paths should not replace the N4 path. The best N4 local-reduction
sweep candidates were about 71.5 us versus 68.8 us for the sweep control and
were not selected as winners. No universal kernel or globally optimal dispatch
is claimed. No N2/N3 performance claim follows from tail correctness.

## Verification

- 37 new compiled kernels plus two archived controls: 39 configurations.
  Zero spills/local stack frames or LDL/STL. No HFMA2/HMUL2; HADD2 restricted
  to exact half-to-float widening. Largest requested dynamic shared storage is
  32 KiB; largest CTA is 1,024 threads. SASS/build logs retained.
- Fresh CPU checks passed normally and under UBSAN: all 63,488 finite half
  patterns, 1,015,808 exact W4 products and 65,536 lossless blocks. Additional
  CPU checks cover all 12 fused shared mappings/reduction orders, 65,536 packed
  half-pair extractions, and unchanged activation quartet order.
- Full GPU suite: 507 pipelines / 388,128 outputs, all baseline-bitidentical.
  Tail shapes M33/K96/N1, M129/K32/N3, M513/K544/N4, four numerical families,
  plus M16/K16384/N4 with every finite FP16 pattern and all 16 weight codes.
- All 65,536 raw half conversion patterns checked: finite/infinity/signed-zero
  bits exact; NaN classification only. Nonfinite matrix arithmetic and NaN
  payload preservation are not claimed.
- Memcheck, synccheck and racecheck each passed all 39 configurations on the
  M33/K96/N1 smoke shape: zero errors/hazards/warnings. This is not full-shape
  sanitizer coverage. Shared-index/reduction-order CPU models are retained.
- Four exploratory sweeps, then a frozen ten-configuration set and 12/12 fresh
  benchmark workers. 120 frozen numerical checks and 960 retained event samples.
  Across validation, sanitizer smoke, sweeps and repeats: **20 successful
  workers, 900 pipeline checks and 3,712,821 checked outputs**.
- The inherited operation-order CPU oracle and independent 128-bit rational
  oracle are unchanged. Source, control, worker, configuration and supervisor
  hashes are checked before launch and during final audit.

Each worker warms for 200 ms, rotates configuration order over nine rounds,
times three full pipelines per event, and discards round zero uniformly. Shape
order changes between repetitions. All outliers remain; exploratory samples
are excluded from confirmation aggregates. Loads, widening, decoding, scaling,
partial/reduction stages and device launch gaps are included. CPU repacking,
allocation, CPU oracle and host/device transfers are excluded.

## Scope, coordination and remaining gate

This is deterministic synthetic data in a real Q4_0 format, **not an end-to-end
inference or quantization-quality result**. Byte identity is against the study
baseline, not stock llama.cpp; stock scaling/reduction order can differ. FP16
activation bits and FP32 accumulation precision are unchanged. There is no real
model/layer input comparison, stock dispatch test, logits, ppl/KLD or tokens/sec
measurement. No hidden instruction capability was needed or discovered.

GPU2 alone was authorized and held continuously through builds, worker gaps,
analysis and final health checks. GPU1 belonged to a different session. Recorded
pre/post compute snapshots show only the existing GPU0 desktop editor outside
this worker; snapshots cannot rule out all transient shared-host interference.
Driver 580.173.02, CUDA 12.8, GCC14, `env -i`, CPU affinity 0-11, UUID isolation.
No clock/counter/driver changes, resets, production edits, commits or pushes.

One guard stopped before launching a sanitizer because another session appended
a CPU-only W4A4 coordination notice that explicitly respected GPU2. The change
was manually reviewed and the new hash recorded in [reservation-review.json](reservation-review.json)
before resuming. No CUDA/correctness failure or hang occurred.

An interim strict archive check flagged two unmanifested Python cache files
during concurrent research. All 1,152 manifested hashes matched throughout;
the cache files were absent at the final audit. This study neither created nor
deleted those files. The observation is retained in
[archive-check-observation.json](archive-check-observation.json); final source,
log and archive checks are recorded in [audit.json](audit.json).

Next: test the winning N1 kernels on actual production W4A16 real-layer inputs,
against the true dispatch, including integration/repacking cost, and require
production byte identity or ppl +/-0.003 with KLD-pair methodology before
integration. Independent P100 confirmation would strengthen the modest gains;
do not resume the closed half-accumulation path or sacrifice A16 precision.

## Artifacts and reproduction

- [Plan](PLAN.md), [kernels](kernels.cu), [worker/oracles](worker.cpp),
  [CPU mapping checks](cpu_layout_checks.py), [build inventory](build/inventory.json),
  [selected SASS counts](build/selected-sass-counts.json).
- [Frozen selection](frozen-manifest.json), [CSV](aggregate.csv),
  [complete aggregates](aggregate.json), [analyzer](analyze.py),
  [supervisor](supervise.py), [audit script](audit.py).
- CPU/log-only: `python3 analyze.py aggregate`, then `python3 audit.py`.
  The audit also checks this host's original coordination logs for release;
  those external logs are not portable dependencies of the kernel itself.
  Build with `python3 build.py` only in a separate working copy; this study's
  inventory and the sibling archived control cubins are sealed evidence.
- GPU repetition requires a new reservation/token and reviewed hashes of both
  coordination logs. Do not reuse this study's released token. Run `validate`,
  sanitizer `smoke`, then frozen `bench` through an adapted supervisor.

Candidate cubin SHA256: `09b125f359d982e9870205427da7d602844090f0c018e856e6397fd49bde233c`.
Worker SHA256: `290cede0d7bec5fe9e1a94fc1b24f4316d66300d595e313cbcb7be124f51b404`.
Round-2 control SHA256: `64fdbef07437dc6918821a6209577000cfe7f61a818fb973fd13b1c643d84d94`.

GPU2 was released after the [15:19:12 UTC final health check](final-health.json):
5 MiB, 0% utilization, no GPU2 compute process and zero volatile uncorrected ECC
errors. The desktop log was 22,835 bytes and disk free space 16.7 GB. All study
workers exited; no reset. Both coordination logs record the final release;
GPU1/GPU3 reservations and other sessions' files remain untouched. Final report
and build/log hashes are recorded in [audit.json](audit.json).
