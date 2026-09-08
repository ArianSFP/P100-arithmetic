# Independent direct-stage audit — 2026-09-08

CPU/read-only review outside this verification directory. No build, CUDA,
reservation, production edit, or new GPU measurement. No correctness blocker
found within the stated aligned, N%64/K%32, positive-size kernel contract.
Direct staging is already a closed performance loser in the parent's screen;
passing this audit is not a recommendation to repeat it unchanged.

## Reviewed snapshots

- `../w4a16-tile256-20260908/direct_stage.cuh` SHA256
  `a968022fcfd4ebec465d4f7aaca1755ff579ece8267b577c9267fb3e70248c9b`.
- `../w4a16-long-prefill-20260908/build.py` SHA256
  `f5c688d0f9ed6a2f4ed5c945c41c07bcb4c1a04d9d3554d56a1b2885f7d83cd0`.
- Generated `build/worker.cu` SHA256
  `75f0d23fe4db0dd0ed92049aadd299a3a711634311dc061882cef99837bd7a3c`.

## Preparation accounting: valid primitive, not complete pipeline

Dispatch column23032 launches 256 threads with sk=2 and leadraw/leadw.
Both prep=0 and prep=1 execute A32 -> RN-half -> FP32 shared staging and
Q4_0 -> half-rounded weight -> FP32 shared staging INSIDE the timed kernel.
The generic preparation exclusion for kind1/column>=4000 is correct here:
there is no missing separate A16 conversion. Current kind-4 T64 also converts
its A32 input inside its unchanged kernel. No FP16 accumulation gain is used.

However, the host constructs candidate-specific word-major `wtile` and uploads
`dwt` before events. This is NOT charged by prep=1. Initial native tiled weight
construction/upload and descriptor building are also outside this primitive.
The print `WEIGHT_SCRATCH ... prep1_charged_every_run=1` describes the staged-W16
family (columns6000..6999), not a blanket guarantee for direct-stage's layout.
Its scratch allocation is not evidence direct-stage requires temporary W16.

Promotion requires an explicit cache contract: amortize immutable repacking
only when it genuinely persists; otherwise charge cache-miss/recurring repack,
copies and synchronization. Charge routing/gather/job construction/fallbacks
at the complete-pipeline boundary. Equal-M tile lists are not ragged T64
M64/M32/M16 dispatch. No 1.5x full-pipeline claim follows from these timings.

## Raw witness: valid numerical oracle, invalid timing comparator

Worker lines169–208 create raw A32 half-midpoint/subnormal witnesses, set
`a = float(half_rn(raw))`, and pass raw through `leadraw` only to the candidate.
Current T64 descriptors retain `da`, the already-rounded A32 values. This is a
valid expected-value construction: rounding that prepared finite A16 value
again is idempotent. The CPU oracle uses `a`, independent signed Q4 codes and
split std::fma rails, not the candidate's bit decoder or shared addressing.

It does NOT test current T64 itself on the original raw buffer; input bytes
differ. In normal fixtures, q*scale is exactly half representable, so the
legacy Q8 split reference equals current Q4. Scale-stress instead selects
current compressed-Q4 T64 and explicitly half-rounds q*scale in the CPU oracle.
Raw witness and hard scales are separate tests, not a joint exhaustive test.

Actionable hardening for a future parent build: return after correctness for
raw-witness as already done for scale-stress. Currently raw mode prints
accuracy_only=1 but still emits TIME lines. `analyze.py` correctly excludes
RAW_WITNESS, SCALE_STRESS and DEQUANT_CHECK runs; every other consumer must do
the same. No current published timing blocker if this exclusion is preserved.
For a stronger same-input conversion check, point current descriptors at the
raw buffer in a dedicated correctness-only mode, retaining the prepared CPU
oracle. That would test both device conversion paths on identical bytes.

## Addressing, decode, ordering and guards

Each of 256 producers writes eight unique A cells and eight unique B cells:
complete 32x64 shared stages. `tid%64` selects the weight scale; `tid/64`
selects the four-byte word within that row. Low/high nibbles become K positions
0..15/16..31, consistent with host word-major repacking. Q codes are exactly
nibble-8, then multiplied and rounded in half BEFORE FP32 GEMM.

Each thread owns a disjoint 4x4 output region; all4096 cells are covered.
Invalid M-tail rows perform no A load and no output store; shared A is zeroed.
float4 accesses are aligned under the contract. Uniform barriers protect stage
reuse and persistent-job transitions. No asynchronous late fetch exists here.
Both FP32 rails span all groups in original order (j<16 versus j>=16), followed
by one FP32 add, matching current T64. The half instructions only prepare data.

Build integration includes the header, 256-thread dispatch, sk2 checking,
scale-stress whitelist and requested-name survival check. Output is poisoned
before each correctness launch; nonfinite, sampled CPU mismatch, and full
split-reference bit differences fail. HMUL2 whitelist admits the named kernel
with FFMA; this name-based screen alone is not proof of arithmetic semantics.

## Evidence checked

Executed Dalton's CPU-only `test_direct_stage.py`: PASS,2048 A/B cells,
4096 outputs,14336 address cases,4 rail/barrier schedules; copied decoder and
frozen earlier headers match. This is a structural model, not an independent
numerical GPU test. Executed owned `test_ragged_proxy.py`: all3 tests PASS.

Read existing parent GPU logs (not rerun): raw GU M65 has266240 changed input
words and66560 output identities; hard-scale down M65 has266240 output
identities; both256 CPU samples pass with memcheck0. Race/sync M65 tests have
16640 identical outputs; parent's report records sanitizer passes.
Parent DIRECT-AND-MAPPING-RESULTS.md reports64 registers/no spills and down
M2565825.664us versus T646801.300us (1.1675x), worse than compact. Resource
counts are existing build evidence, not new compilation or occupancy measurement.

Changed only this document and RAGGED-PROXY.md. No parent source was changed.
