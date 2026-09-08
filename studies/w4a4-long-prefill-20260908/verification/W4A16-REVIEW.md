# Independent W4A16 wide/decode/scale-stress review

2026-09-08, CPU-only/read-only parent review. No build, CUDA, reservations or
parent edits. Only this report written in reviewer's owned verification dir.
Ran `PYTHONDONTWRITEBYTECODE=1 python3 studies/w4a16-long-prefill-20260908/test_wide_model.py`:
all4 tests PASS in2.114s. This is CPU evidence, not GPU decode/GEMM validation.

Reviewed SHA256:

- wide.cuh: ac1f3449f4eb8d3e2011c6f2038a3f5db3c8db73e32c44d5750d3dd1a47d4ea9
- build.py: 439037c38aa834450fdcfeaea1a7c921ffba44642d58a71bc90e0e5d968fdd53
- test_wide_model.py: d536366e08eefd8cc910dc7cb61e1ba8d582503a8646696724292ba178f0227f
- generated build/worker.cu: 644ebd0d82f2a7d2b692981b0c4395bd7e93ab7dd7569208d5b817f654ffc0ae

## Actionable false-pass blocker: scale-stress selection

build.py:133–140 applies the scale-stress filter AFTER the inherited --only
filter and after its size check. The stress filter keeps current/special T64
plus column9032/10032 (split variants). It does not require a candidate remain.
Thus these commands can report PASS while exercising NO wide kernel:

    --scale-stress 1 --only wide_packed_u32
    --scale-stress 1 --only typo

CPU reproduction of the two predicates yielded only aw_current_q4 and
aw_special_q4 in both cases. Correct selection --only wide_packed_split_u32
retains that candidate; no --only retains both split candidates.

Fix before accepting a stress PASS: validate --only against registered names,
then require post-filter at least one kind1 split-wide candidate, and require
the specifically requested candidate if --only is present. Log surviving
candidate names/count. A generic cfg.size check before filtering is insufficient.
For intended two-path comparison, omit --only and assert BOTH split paths remain.

Also reject simultaneous --quant-check and --scale-stress: generated worker
line117 returns immediately after quant-check, silently skipping requested
stress testing. Same caution applies to quant-check combined with raw-witness.
Individually invoking the two correctness modes avoids this false-pass route.

These are harness-selection blockers, not an observed arithmetic defect.
No other source-level arithmetic blocker found for supported shapes.

## decode_q4_pair: exactness reasoning and scope

For each offset-binary nibble c in0..15, raw half bits0x6400|c encode1024+c.
Subtract1032 in half: result c-8 is exactly representable, including +0 for
c=8. Pair bit layout puts the low/high nibbles in the intended lanes. Current
callers mask to a byte or explicitly construct a byte; the function itself
assumes byte<=255 (worth asserting/documenting if exposed to new callers).

Multiplying finite half d by integer q in[-8,7] has an exact FP32 product:
at most15 significant binary bits and exponent comfortably inside FP32 range.
Therefore RN-half(q*d) equals half multiply RN, including half subnormal
rounding, overflow and signed zero, provided device half multiply preserves
the specified behavior. The scalar F32 path explicitly __fmul_rn then rounds
to half; it does not perform a fused add or change scales. No FP16 GEMM
accumulation: both paths widen prepared weights and use FP32 FMA.

CPU tests exhaust all16 codes x63,488 finite half encodings =1,015,808 scalar
pairs and compare raw half bits, including signed zero and overflowing products.
The hardware --quant-check covers1,015,808 packed outputs, each with code c and
complementary code15-c, hence both lanes/all individual codes. It executes
nonfinite scales too but excludes them from comparison. No NaN payload/Inf
semantics or every possible256 nibble pair per scale is claimed. CPU scalar
simulation alone cannot prove actual HMUL2 lowering or subnormal handling;
the planned hardware quant-check is a necessary gate before trusting PACKED.

## Scale-stress fixture and current T64 oracle

Stress changes magnitude scale bits to0x1800..0x37ff and randomizes sign;
q*d genuinely needs half rounding and remains finite. CPU witness verifies
>16,000 changed products. This is materially stronger than the prior all-exact
1..127/1024 fixture. It is not an exhaustive distribution of real model scales.

The stress filter intentionally removes legacy unrounded-weight and sequential
candidates. Reference is changed to aw_current_q4, encountered before special
and wide split paths. The CPU oracle rounds q*scale to half in stress mode;
activations are the same prepared A16 values used by current T64. Rail j/16
is maintained across all K32 groups, then rail0+rail1 once at output: this
matches the current T64 source and wide SPLIT=true exactly, NOT contiguous
whole-K halves or per-group rail reductions.

Every correctness launch poisons output with0xff. All outputs must be finite.
All split candidates must match EVERY reference output bit (`sk==2&&changed`
fails), and independently computed CPU FP32-FMA samples must match; up to
16,384 outputs are all sampled, larger outputs use256 global samples. That
is stronger than finite-only checking. Stress exits before warmup/events:
correctness-only mode is genuinely not a timing run.

Scale-stress forbids raw-witness; thus combined activation-rounding and
arbitrary-scale stress is not covered in one run. Separate raw-witness mode
feeds raw A32 to wide while current receives its rounded A16 widening, which
is mathematically equivalent for current's conversion but not an identical
input byte stream or a performance denominator. Preserve its correctness-only
label and consider a combined mode later with explicit corresponding oracle.

## Layout, costs and build notes

CPU tests cover every64x128 output once, A32x64 and B32x128 staging once, and
correct two-N64-stage addresses at N128/512/2048,K96/512/2048. Source masking
handles M tails; shared consumers/producers are separated by CTA barriers.
Vector offsets are16-byte aligned and Q4 words4-byte aligned for supported
dimensions. GPU memcheck/racecheck/synccheck remain required.

Activation conversion occurs inside wide staging, so skipping a separate
prepare_a16 pass is correct cost accounting. Wide IDs>=7000 are excluded from
the temporary weight staging pass, avoiding an accidental extra preparation
charge. Original Q4->word-major layout is still host-prepacked/excluded:
real recurring service repack/cache/gather/owner costs remain acceptance work.
M<=512 equal-expert CLI limit is still not full hot-expert/ragged2k–8k coverage.

build.py default INVOKES compiler; --prepare-only writes generated sources but
does not refresh the final binary manifest. Reviewer ran neither. Do not treat
an old binary/manifest after preparation as validating new stress code. Final
build hashes and parent supervisor checks must match the reviewed sources.
SASS gate forbids HFMA2, permits HMUL2 only in decode_check or packed wide
functions with FFMA; this is acceptable because the source locates half math
only in weight decoding. No precision-based accumulation gain is accepted.

## Follow-up: compact32x64 candidate (CPU-only review)

Reviewed compact.cuh SHA256
dbdaabf8284b7e98ce3f77367aec8aa161c95106bbaaa0a22162a445f8d9ed1f,
build.py a79016d1fc8ed05858d0da8c6b571ec04fd7c208b5330d95da6cc13b0056a565,
and corresponding generated worker while owner build was running. No parent
file edits, build invocation, GPU queries/launches or reservation by reviewer.

No address/arithmetic blocker found. Independent test_compact_model.py:3 tests
PASS in0.022s, validating:

- All32x64 outputs are assigned once by128 threads with4x4 microtiles.
- A staging covers every32x32 element once. B staging covers every32x64
  element once: at=tid+l*128, row=at%64, chunk=at/64, low K=chunk*4+i,
  high K=low+16. Load128+at*4 is precisely word-major chunk/row indexing,
  final word ends at1152, scale index remains in the first128 bytes.
- Expert/token/column job arithmetic, ragged M1/31/32/33/63/64/65/128/511/512,
  N128/512/2048, and representative K96 staging offsets/alignment.
- Low/high FP32 rails preserve current T64's per-group K0–15/K16–31 order
  across groups; result is their single final FP32 sum, not a per-group sum.

Source further has conditional tail A loads and stores; all supported valid
float4 offsets are aligned. Next-group fetch changes only registers while
current shared data is consumed. CTA barriers precede shared overwrite and
follow staging, including a final barrier before the next persistent tile.
No early thread return or divergent barrier path observed. Packed decode
reuses reviewed decode_q4_pair; scalar path rounds the FP32 q*d product to
half identically before widening. No FP16 accumulation introduced.

Build wiring dispatches11032/12032 BEFORE the >=7000 wide branch, passes
leadraw and word-major leadw, sets config sk=2, and uses the full-output
current-T64 identity gate plus sampled CPU split oracle. Compact IDs are
excluded from redundant row-prep and temporary weight staging; activation
rounding is inside compact stage and therefore charged to the kernel.

Earlier scale-stress false-pass guard is NOW FIXED in source/generated worker:
unsupported/missing --only, raw-witness conflict, and quant-check+stress are
rejected before CUDA access; compact_packed_u32 and compact_scalar_u32 are
allowed. Post-filter confirms the requested name survives. Stress retains
current/special control plus requested compact and exits before timing.
An empty --only intentionally fails stress now; choose each supported variant
explicitly. Generic non-stress misspelled --only is still not strictly rejected
by the inherited filter, so always verify named candidate CHECK/TIME records.

Suggested bounded validation after owner build succeeds: separate memcheck
stress requests for each compact variant at M33/65 and multi-expert K96 or
K2048, then racecheck/synccheck as appropriate. Hardware identity remains
unproven for compact until those actual launches pass. The previous wide
hardware proof does not substitute for compact addressing/synchronization.
