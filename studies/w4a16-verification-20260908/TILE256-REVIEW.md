# Independent64x64/256-thread split-FP32 source review

Reviewed `studies/w4a16-tile256-20260908/tile256.cuh`, SHA256
218e323c96e1421aae73f343f7cedc67b3b390a2c4b3cd55ea846f923f456beb.
No parent/Dalton edits, build, GPU or reservation. W4A4 remains paused.

Verdict: no source-level address/dequant/rail blocker found. Three independent
CPU tests PASS in0.019s (`test_tile256.py`), with no implementation import.

- Eight warps each cover16x32 outputs; warp/2 selects16-token bands and warp%2
  selects32-output bands. Lane/8 and lane%8 select4x4 microtiles. All4096 outputs
  have exactly one owner; maximum row/column63, no missing upper warp band.
- Each thread stages8 activation values: tid/4 spans64 tokens, tid%4 spans
  four K8 chunks. Every32x64 shared A element is written once, not duplicated.
- One B word per thread: tid/64 selects four low-K quartets, tid%64 selects
  output row. High nibble is low K+16. Scale offsets fit0..127, packed words
  fit128..1151 in the1152-byte word-major stage. Every32x64 B element has one
  writer; this is not the original row-major payload layout.
- Job decomposition across expert/token-tile/output-tile is bijective; CPU
  tests check M1/31/32/33/63/64/65/127/128/129/511/512 and N128/512/2048.
  Invalid tail rows use zero activation staging and no output store. A vector
  alignment follows K multiple32 and tid%4*8; B words are4-byte aligned and
  shared float4 reads start at multiples of4 elements.
- Both U16/U32 preserve K0..15 low-rail and K16..31 high-rail order separately
  across G32 groups. __fmaf_rn on FP32 rails and one __fadd_rn at output match
  current T64 schedule. No half accumulation, no group-end rounded partial sum.
- Parent decode_q4_pair is used with masked bytes. Independent model tests all
  256 packed bytes/both lanes across10 scale encodings (signed zeros, tiny,
  rounding boundaries, overflow) and raw-bit half equality. Parent exhaustive
  hardware decode result is useful but does not validate this new tile's loads.
- Prefetch mutates only thread registers; shared buffers are overwritten after
  all current consumers reach a CTA-wide barrier. Stage completion has another
  CTA barrier; final barrier protects next persistent job. All256 threads take
  the same group/job loop bounds, with no early thread exit. Sanitizers remain
  required; CPU/source analysis does not prove device race freedom.

Build/worker wiring did NOT yet exist at this review cutoff. Before GPU:
verify kernel launch256 threads; include parent's wide.cuh before this header;
feed word-major leadw and original/raw leadraw; set Config.sk=2; allow selected
names through stress validation/filter; reject nonexistent selections before
CUDA. Preserve full-output poison/identity plus independent split oracle.
Do not route tile256 through the128-thread compact launch or legacy column
fallback. Whitelist HMUL2 only for this packed decode-bearing function with
FFMA present, never permit HFMA2. Record launch-bound and spill/occupancy data;
MIN_BLOCKS3/4 promises alone do not establish achieved occupancy or performance.

Timing must charge activation rounding inside stage and all actual recurring
layout/cache preparation. Initial isolated screening still excludes native
cache-to-word-major conversion; do not relabel that as service-level1.5x.

Next owner-run gates: each chosen U/residency variant gets poisoned-output
scale-stress smoke at M33/65,N128,K96,expert count>1, then memcheck/racecheck/
synccheck and held-out range/rounding fixtures; only successful variants proceed
to matched gate/up and down full-prep screening. Hardware acceptance remains
pending, and build changes require a fresh provenance check.
