# Compressed layouts and register-LUT: offline investigation

Follow-up: [single-P100 GPU results](GPU-RESULTS.md) now contain hardware
correctness and matched pipeline measurements. The CPU-only scope and pending
hardware statements below describe the earlier offline phase, not current status.

Date: 2026-09-08. Scope: signed W2/W4, A8 correctness and a separate FP32
activation prototype. No GPU execution, CUDA/NVML initialization, driver change,
production integration, commit or push. The existing four-GPU reservation was
respected throughout. `cpu-tests` is an ordinary host executable, not linked
against CUDA. CUDA tools were used only to compile/disassemble a cubin.

## Conclusions

1. The proposed byte-sign layout works. **Complementing the bits offline fixes
   the per-quartet XOR problem** of the prior runtime-complement prototype.
   It has the same payload size and comparable compiled cost to uncomplemented
   storage with the reversed endpoint equation. Neither orientation has a
   demonstrated performance advantage.
2. LUT should have its own losslessly repacked quartet-index layout, plus a
   matched ordinary-arithmetic control. Gathering four separated bits from the
   SAD-friendly layout adds avoidable instructions.
3. **A common, contiguous 16-entry 32-bit shared table is already free of
   intrawarp bank conflicts for the proposed row-parallel mapping.** Replication
   is not required to fix conflicts in this case. Register-LUT and this compact
   shared-LUT should both compete directly with SAD and the best integer arm.
4. More rows per warp amortize table construction but reduce the number of
   blocks and increase register pressure. Split-K and its final reduction must
   be part of the comparison, not an uncharged fix for insufficient parallelism.
5. All tested A8 group equations are exact on CPU. The FP32 table-reduction
   order is not byte-identical to direct FMA in general. No speedup or model
   quality result is claimed.

## Layouts implemented and verified

Let `j=4*q+k`, with quartet `q=0..7` and byte `k=0..3`.

| ID | Bit position in each 32-bit plane | Stored bit | Intended consumer |
| --- | --- | --- | --- |
| F0 | `8*k+q` | `b_j` | Previous byte-striped SAD / integer control |
| F1 | `8*k+7-q` | `b_j` | MSB-first SAD with reversed endpoints |
| F2 | `8*k+7-q` | `1-b_j` | User's precomplemented endpoint layout |
| F3 | `4*q+k` | `b_j` | Four-bit-index LUT / integer control |

Every layout uses exactly one word per plane per G32: 8 bytes W2, 16 bytes W4.
No byte masks are stored in global memory. Matrix row tiles are padded to a
multiple of 32 rows; this tail padding is separate from the per-group payload.
The synthetic kernels also use one FP32 weight scale per G32, another **1 bpw**.
They are not native GGUF/Q2_K layouts or 2-bpw end-to-end storage claims.

For F1/F2, `sign_bytes(plane << q)` expands quartet q. The same operation can
be written as successive one-bit shifts; seven shifts suffice per plane over
eight quartets. Bits crossing byte boundaries cannot reach the next byte's sign
bit during this eight-step traversal. All 256 input-basis-bit/shift pairs passed.
The CUDA implementation uses generic inline PTX `prmt.b32 ... 0xba98`, whose
selector high bits request sign replication. It does not use `__byte_perm` as
a substitute. [PTX specification](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt)

F3 makes each index `(plane >> (4*q)) & 15`. The F1-to-LUT control must instead
gather bits spaced eight positions apart. Both lossless layouts have ordinary
VMAD controls, so a storage improvement need not be misattributed to SAD/LUT.
The fastest arithmetic/layout combination must also be compared across layouts;
beating an integer control on the LUT-friendly layout alone is insufficient.

### Endpoint correction with stored complements

Let `c_p` be signed two's-complement plane coefficients, whose sum is -1;
`A=sum a_j`; `S=sum_p c_p*popc(stored_plane_p)`.

```
F1: E=sum_p c_p * SAD(a+128, 255*b_p)
    dot=-(E+A+128G+S)/2

F2: D=sum_p c_p * SAD(a+128, 255*(1-b_p))
    dot=(D-A+128G+S)/2
```

For F2, `sum_w=-G-S`. This avoids undoing the complement at runtime. Its
ordinary-arithmetic control also avoids undoing it: decode the stored signed
code v, compute `dot_v`, then use `dot_w=-A-dot_v`.

The six compiled single-group probes all contain only the eight activation-bias
XORs: no per-plane AND or complement instructions. W2 uses 16 PRMT + 16 SAD;
W4 uses 32 PRMT + 32 SAD. Static code sizes are respectively 102 and 156
instructions including setup/padding for each orientation. Equal counts are
not evidence of equal timing or identical instruction words.

## Thread mapping and global loads

For each plane/group, store 32 adjacent output rows together:

```
weight_word = (((row/32)*groups + group)*W + plane)*32 + row%32
scale_word  = ((row/32)*groups + group)*32 + row%32
```

There are four warps per 128-thread CTA. Each lane owns R rows, separated by
32, and R is 1, 2 or 4. The warp shares one logical activation-combination table
per quartet across its 32R rows and all weight planes. The register version
computes `T[lane & 15]`; lanes 16..31 duplicate the lower-half table. Each
consumer uses its four-bit index with SHFL. This is warp-distributed storage,
not a dynamically indexed 16-element local array in every thread.

An aligned full-warp weight load requests four 32-byte sectors with tiled
storage, versus 32 sectors if the old row-major layout is used with this new
row-parallel mapping. Conversely, the **old one-row-per-warp, lanes-across-K
mapping already has four sectors with the old layout**. The new layout is not
an inherent 8x bandwidth improvement; layout and lane mapping must agree.
This is an address model, not measured DRAM traffic. Pascal's access unit is
32 bytes. [Pascal tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/pascal-tuning-guide/index.html#unified-l1-texture-cache)

### Too few blocks without split-K

At M=5120, four warps/CTA give:

| Rows/warp | CTAs without split-K | CTAs with 8 splits |
| ---: | ---: | ---: |
| 32 | 40 | 320 |
| 64 | 20 | 160 |
| 128 | 10 | 80 |

The local P100 has 56 SMs. Even the first mapping cannot occupy every SM
without more parallel work; extra row reuse worsens this. Multiple input
vectors can help, but batch-one must be assessed on its own.

The prototypes implement `grid.z` split-K over whole scale groups:
`g=blockIdx.z; g<groups; g+=gridDim.z`. CPU ownership checks confirm each group
is visited exactly once. Partial output is `[batch][split][row]` and a separate
ordered reduction kernel is compiled. Do not split across or move corrections
outside their common scale group. Eight splits at M=5120 require 163,840 bytes
of FP32 partial storage per input vector. Writes, reads, reduction, launch cost
and any changed FP32 reduction order must all be accounted for. Enough blocks
for one wave is not proof of adequate occupancy or latency hiding.

## Shared-LUT correction

For a common 16-entry table of 32-bit words, `bank=(base+index)%32`.
Different indices 0..15 cannot alias a bank. Repeated indices refer to the same
word, which uses broadcast/multicast. Counting repeated bank numbers as conflicts
would be an incorrect model here. [NVIDIA shared-memory documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)

This does not say arbitrary LUTs are conflict-free: multiple different tables
within one warp, larger tables, or different element widths need a new analysis.
The earlier replicated-address proof established a sufficient layout, not a
necessity. LUT-GEMM's public reference uses a different 256-entry half table for
eight activations; it should not be conflated with this 16-entry 32-bit design.
[Reference implementation](https://raw.githubusercontent.com/naver-aics/lut-gemm/main/lutGEMM/src/cuda/kernels/mv_fp16.hpp)

Implemented compact and replicated controls both keep one quartet table per
warp live, reuse it across R rows/lane, and use `__syncwarp` before reads and
before overwrite. No lane exits before a shuffle/synchronization, including
invalid tail rows. Resource costs for four warps/CTA are:

| Design | Bytes/warp | Bytes/CTA | Entry-word writes/quartet/warp |
| --- | ---: | ---: | ---: |
| Compact | 64 | 256 | 16 |
| 32x replicated | 2048 | 8192 | 512 |

The replicated table has no conflict advantage for this mapping and remains
a control, not the default. Absence of bank conflicts does not make shared
loads, addressing, synchronization or inter-warp throughput free. GP100 has
64 KiB shared memory/SM and a 48 KiB/CTA limit; the streaming prototypes fit.
Caching eight replicated quartet tables per warp would instead require
64 KiB/CTA with four warps, exceeding that limit.
[Pascal shared-memory limits](https://docs.nvidia.com/cuda/archive/12.9.1/pascal-tuning-guide/index.html#shared-memory-capacity)

## Compiler results: reuse is real but not free

CUDA 12.8, GCC 14, `-O3 -arch=sm_60`. **74 kernels compiled; all have zero
spill stores/loads.** This includes 60 A8 mapping/method variants, six isolated
SAD probes, six separate FP32-LUT variants and two preparation/reduction kernels.

The initial generic signed-byte table expression was unnecessarily expensive.
Using direct `int8_t` casts reduced register-LUT static code size, e.g. W2/R1
from 294 to 234 and W4/R1 from 336 to 282. The original remains M2 as an
attribution control; M5 is the cast variant. Shared controls use the cast form.

Selected whole-function **static instructions / registers per thread** below.
Counts include setup, loop control, tail handling and padding, but exclude the
separate preparation/reduction kernels. They are not dynamic counts or cycles.

| W / rows per warp | Byte-layout VMAD | Endpoint SAD | Quartet register-LUT |
| --- | ---: | ---: | ---: |
| W2 / 32 | 168 / 29 | 162 / 31 | 234 / 32 |
| W2 / 64 | 282 / 40 | 264 / 40 | 324 / 48 |
| W2 / 128 | 516 / 56 | 456 / 56 | 498 / 64 |
| W4 / 32 | 204 / 32 | 222 / 32 | 282 / 32 |
| W4 / 64 | 354 / 47 | 378 / 48 | 420 / 56 |
| W4 / 128 | 660 / 64 | 690 / 64 | 690 / 80 |

Both SAD and LUT scale linearly with plane count; LUT does not escape that
factor. LUT construction is shared across planes/rows, while per-plane SHFL,
index extraction and accumulation still increase. W4/R4 closes the static-count
gap with SAD but uses more registers. This warrants a hardware comparison, not
a LUT victory claim. At R4, the compact/replicated W4 shared arms both use 112
registers despite having no spills, so resource-limited occupancy is a concern.

## Correctness evidence and limits

- 65,536 independent G32 groups per precision, tested in all four layouts:
  lossless pack/unpack, ordinary controls, LUT equations and all three SAD
  orientations agree exactly. Includes all uniform signed weight codes against
  activation -128 and +127, then random groups with recorded seeds.
- 9,437,184 random/edge quartet mask-expansion checks across the three
  byte-sign formats, plus all 256 basis-bit/shift cases.
- 12,800 warp-lane cases exercising R=1/2/4 reuse and incomplete output tiles;
  shared host/device weight and scale address functions checked separately.
- 262,144 random bank-index vectors. Both table layouts have one distinct word
  per accessed bank; an adversarial distinct-word/same-bank control gives 32.
- Split-K group ownership and aligned sector models passed.
- Host tests also passed with undefined-behavior sanitization.

These are **CPU tests and compiler evidence**, not execution validation of GPU
shuffles, shared-memory races, scheduling, scale pipelines or performance.

The separate `rows_f32` prototypes use FP32 activation combinations and FP32
accumulation, with no A8 quantization. In 32,768 finite CPU quartet tests per
precision, regrouped LUT results differed from sequential direct FMA in 117
W2 and 886 W4 cases. Some LUT results were closer to exact FP64, some not;
neither comparison is a quality gate. Exact hexadecimal witnesses are recorded
in `results/cpu.json`. FP32 full-group behavior, nonfinite inputs, split-K
rounding and real-model quality remain unvalidated. Do not import the reference
LUT-GEMM half/atomic accumulation scheme without separate analysis.

## Next hardware gate, after explicit reservation release

Make register-LUT a direct competitor immediately; do not heavily tune SAD first.
Use the same canonical weights/scales to construct each lossless layout.
Start with A8 and compare byte-layout VMAD, endpoint SAD, quartet register-LUT,
compact shared-LUT, and a small replicated control. Keep both within-layout
integer controls and the best across-layout integer arm.

Test R=1/2/4 and modest split-K values on actual matrix dimensions and batch
sizes, including batch-one. Charge activation-sum preparation for SAD and the
complemented integer arm, table construction for every LUT arm, scales, split
partial traffic and final reduction. Measure weight repacking/load-time cost
separately. First validate every result against independent integer group dots
and a reference with the same FP32 reduction order. Only then collect repeated,
interleaved full-pipeline timings in fresh supervised workers. A new reduction
order or higher-precision activation scheme needs a separate numerical gate.

`kernels.cu` has **no host GPU launcher**. It is not yet wired into the earlier
reservation-gated hardware harness. Link/integrate that harness, verify device
correctness and race handling, and obtain/hold the reservation before any run.
These synthetic G32 two's-complement kernels are not an implementation of
Q2_K's unsigned G16 affine format or an IQ codebook. Those remain later gates.

LUT-GEMM supports investigating this algorithmic family; its published results
do not establish a P100 speedup. [LUT-GEMM paper](https://arxiv.org/abs/2206.09557)

## Reproduction / artifacts

```bash
cd /home/arian/hfma2-gp100-probe/layout-lut-20260908
python3 build_and_check.py
# Optional: --verbose prints the full method/layout/resource inventory.
```

Sources: `common.hpp`, `kernels.cu`, `cpu_tests.cpp`, `build_and_check.py`.
Evidence: `results/cpu.json`, `results/address-model.json`, `results/cuda-build.json`,
`results/kernels.sass`, `results/inventory.json` with source/cubin hashes.
The existing reservation remains untouched; no GPU job was launched.
