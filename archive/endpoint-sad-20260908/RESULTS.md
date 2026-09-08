# Endpoint-SAD formulation investigation — 2026-09-08

## Decision and status

For the GP100 low-bit research path, first measure a complete **W2A8 endpoint-SAD
matvec against masked SAD and decoded-integer baselines**, with all preparation,
scale, correction and metadata traffic included. W4A8 is a secondary control.
Do this before a large launch-parameter sweep or more undocumented-encoding work.
This is not a demonstrated replacement for the existing Q8 production path.

**Completed here: CPU correctness, SM60 compilation/SASS audit and offline
supervisor tests. No endpoint-SAD GPU correctness or performance run has occurred.**
The LMCache session explicitly holds all four GPUs for its follow-up tests, even
between jobs. Its reservation is near the top of
`/home/arian/llama.cpp-qwen38-p100/bench/COORDINATION-20260908.md`.
Older releases farther down concern a different series and do not release it.
This investigation is queued, not reserved. It has not queried or initialized
CUDA/NVML while that reservation is held.

## Algebra: the proposal is correct, but reverse the endpoints

Let `u=a+128`, `w=sum_p c_p*b_p`, `C=sum_p c_p`, `A=sum_j a_j`,
`Wsum=sum_j w_j`. For two's-complement W2/W4, `C=-1`.

The supplied proposal uses `r=255*(1-b)`:

```
|a+128-255*(1-b)| = (2*b-1)*a +127+b
Dweighted = 2*dot -C*A +127*C*G +Wsum
dot = (Dweighted +C*A -127*C*G -Wsum)/2
```

In the compiled implementation, complementing each plane before byte expansion
does NOT give one cheap complement per group: the compiler distributes the
complement through the unrolled shifts. Per W2 group, 14 immediate XORs plus
2 register complements replace the masked formulation's 16 ANDs. For W4,
28 XORs plus 4 complements replace 32 ANDs. This observation is specific to this
source, byte-plane layout and CUDA 12.8 compilation; it is not a hardware limit.

Use the equivalent **non-complemented endpoint `r=255*b`** instead:

```
|a+128-255*b| = (1-2*b)*a +128-b
Eweighted = C*A -2*dot +128*C*G -Wsum
dot = (C*A +128*C*G -Wsum -Eweighted)/2

Signed W2: Eweighted = E0 -2*E1
Signed W4: Eweighted = E0 +2*E1 +4*E2 -8*E3
Both:      dot = -(Eweighted +A +128*G +Wsum)/2
```

The exact numerator is even, including negative dots and activation -128.
Integer division by two can therefore be an arithmetic shift. The implemented
host path checks parity. Alternatively, if the activation preparation stores
`sum_u=A+128G` directly, the signed numerator becomes
`-(Eweighted+sum_u+Wsum)`; that additional simplification is not yet implemented.

## Actual offline results

Build: `/usr/local/cuda-12.8/bin/nvcc -ccbin /usr/bin/g++-14 -std=c++17 -O3
-arch=sm_60 -lineinfo -Xptxas=-v`. No cubin/SASS mutation. Native
`VABSDIFF4.U8.U8.ACC` is emitted from documented PTX
[`vabsdiff4...add`](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#simd-video-instructions-vabsdiff4).

CPU checks passed:

- 559,104 W2 and 565,248 W4 mathematical cases, covering signed AND unsigned
  weights. Both endpoint orientations checked on every case. Every scalar A8
  pair; 262,144 random four-lane dots per signedness/precision; 4,096 cases
  per group size 16, 32, 64 and 128 per signedness/precision.
- Another 65,536 packed G32 cases per precision, checking all nine actual
  `group_dot` source variants using host emulations of the three PTX primitives.
  Independently packed scalar weights/activations supply the oracle. Checks
  include byte-plane positioning, weight POPC sums, activation preparation,
  all uniform weight codes against both activation extrema, and random groups.
- 4,096 Q2_K-style affine superblocks with G16 weight scales inside G32
  activation groups, using exact integer scale numerators. This checks algebra,
  not actual GGUF byte decoding or floating-point reduction order.
- A differing-scale negative control and five offline supervisor tests.

Single-group `dot_probe`, G32; other setup instructions omitted from this table:

| Formulation (POPC sum) | W2 SAD / PRMT | W2 per-plane logical ops | W4 SAD / PRMT | W4 per-plane logical ops |
| --- | ---: | ---: | ---: | ---: |
| Masked | 16 / 16 | 16 AND | 32 / 32 | 32 AND |
| Proposed complemented endpoints | 16 / 16 | 14 XOR + 2 NOT | 32 / 32 | 28 XOR + 4 NOT |
| Reversed endpoints | 16 / 16 | 0 | 32 / 32 | 0 |

All three also use eight activation-bias XORs per group. Computed weight sums
use two POPCs for W2 or four for W4. Metadata variants replace those with a
signed 16-bit load. All 37 compiled kernels have zero spill loads/stores.

**Do not equate the removed logical ops with a full-kernel instruction saving.**
In the current W2 matvec, the masked and reversed POPC arms both contain 192
static instructions including setup, control and NOP padding (30 vs 32 registers).
Excluding NOPs, reversed has only two fewer. Addressing, activation-sum loads,
corrections and dependency barriers absorb much of the local saving.
For W4 the corresponding counts are 258 vs 240 (31 vs 32 registers).
These are static code counts, not dynamic loop counts, cycles or throughput.
The reversed endpoint also incurs the separately charged activation-preparation
kernel. A speedup over either masked SAD or VMAD remains unproven.

## Implemented GPU comparison, pending execution

`endpoint_sad.cu` provides nine variants: C++ integer decode/dot, explicit
four-VMAD byte dot, masked SAD with POPC or metadata, original endpoints with
POPC or metadata, original endpoints with a local activation sum, and reversed
endpoints with POPC or metadata.

The packed layout stores one bit-plane word per 32 weights. Bit
`8*(j%4)+j/4` represents weight lane `j`. Shifting by `7-quad`, then
`PRMT ... 0xba98`, expands four plane bits to four endpoint bytes. This requires
an offline bit permutation but no activation transpose. It is **not the current
GGUF layout**. All comparison arms use the same layout and include decoding.

Pending hardware gates:

1. Compare every variant against independently computed integer dots on
   65,536 groups per precision; verify GPU activation preparation as well.
2. Require bit-identical scaled matvec outputs against the C++ baseline for
   every variant/shape, with a fixed FP32 scale and warp-reduction order.
3. Time `(M,K,N)` = `(5120,5120,1)`, `(17408,5120,1)`, `(5120,17408,1)` and
   `(5120,5120,4)` for W2/W4. Two warm-up rounds, seven rotating-order rounds,
   three pipelines per timed sample, three fresh worker processes. Discard
   measurement round zero; report per-worker distributions, not just a minimum.

Timing includes activation-sum preparation whenever used, all group
corrections, scale loads/multiplication, FP32 accumulation and output reduction.
Preparation is also timed separately for attribution; do not subtract it from
the main comparison. Offline weight repacking and metadata creation are excluded
from per-inference timing and must be costed for any integration.

The synthetic weight payload is W bits plus one FP32 scale per 32 weights
(another **1 bpw**). Metadata arms add another **0.5 bpw** and its load traffic.
This is not a 2-bpw or Q2_K end-to-end storage/performance claim. Signed W2/G32
weight sums fit in int8 (-64..32), so an eventual compact metadata arm could use
0.25 bpw; the implemented arm intentionally measures the proposed int16 cost.
W4/G32 sums require more than eight bits. Benchmark scales are positive powers
of two; production numerical validation must also cover actual scales.

## Applying this to Q2_K

The local format definition has unsigned q=0..3, G16 scale/minimum subgroups,
and an affine reconstruction. The signed-W2 formula must not be substituted.
For unsigned W2, `C=3`, hence:

```
dot_q = (3*A +384*G -Qsum -(E0+2*E1))/2
contribution = d_A * (d_W*s_g*dot_q -d_min*m_g*A)
```

Apply both terms within each common weight/activation-scale group, G16 here.
Q8_1's G32 activation sum does not give its two G16 sums for free. Int16 weight
sums per G16 would add **1 bpw**, not 0.5; unsigned Q2/G16 sums fit uint8
(0..48), allowing 0.5 bpw if stored compactly. Compare against POPC and account
for the actual format's existing metadata. IQ codebooks require another model.

Relevant source: `ggml/src/ggml-common.h`, `block_q2_K`; and
`ggml/src/ggml-cuda/vecdotq.cuh`, `vec_dot_q2_K_q8_1_impl_mmvq`, in the
Qwen3.8 worktree. Nothing there was changed.

If the synthetic gate wins, the next gate is an isolated Q2_K-layout kernel
against the actual existing SM60 fallback, including repacking, metadata and
scale boundaries. Then validate real output/quality and model-level performance.
Neither the synthetic integer baseline nor an instruction-count improvement
establishes an inference speedup.

## Reproduction and reservation discipline

Safe offline commands:

```bash
cd /home/arian/hfma2-gp100-probe/endpoint-sad-20260908
python3 build_and_audit.py
python3 test_supervisor.py
python3 run_gpu.py
```

`run_gpu.py` without `--execute` does not query NVIDIA or launch a worker.
To run hardware tests, first obtain the LMCache owner's explicit release and
review all active reservations. Append a unique line to this investigation's
coordination section: `- HELD: endpoint-SAD GPU0 token=<unique-token>`.
Record the resulting file SHA256, then use approved host execution:

```text
python3 run_gpu.py --execute --reservation-token <unique-token> --coordination-sha256 <reviewed-file-hash>
```

The supervisor rejects missing/stale reservation provenance, source/binary
changes, occupied devices, failed health queries and uncorrected ECC errors.
It logs hashes/hypothesis before launch, preserves raw stdout/stderr, uses
GPU UUID isolation plus `env -i`/`taskset 0-11`, and launches one fresh worker
at a time. A worker has a 45-second timeout. Any timeout, CUDA/correctness error
or health failure stops the entire series. No automatic reset or retry.
Killing a worker does not establish GPU recovery.

**Hold the same reservation across all three workers, gaps and final log/health
review.** The supervisor never releases it automatically. The owner records a
final release only when the series and final checks have completed safely.
No test was launched and no new reservation was acquired in this turn.

Later hardware follow-up: [single-P100 GPU results](../layout-lut-20260908/GPU-RESULTS.md).
The original endpoint tests and matched integer/SAD/LUT comparisons have now
run. The queued/no-GPU statements above describe this earlier offline phase.

Artifacts: `results/cpu.json`, `results/build.json`, `results/endpoint-sad.sass`,
`results/inventory.json` (hashes, complete instructions and register/spill data).
No production source changes, commits, pushes, unknown opcodes or model runs.
