# Existing route/activation capture inventory

2026-09-08. CPU/read-only inventory; no model run, build, GPU access/reservation,
production patch, or edits outside this verification directory.

## Finding

No matched current-Q4 token-route + expert-activation capture was found in the
scoped project artifacts below. Genuine older Q8 token routes DO exist; absence
of all routes would be an incorrect conclusion. They cannot prove current-Q4
2k/4k/8k service replay because model, dispatch and captured fields differ.

This is an evidence-backed blocker specifically for authentic current-Q4
full-pipeline replay/acceptance from EXISTING data, not a reason to stop kernel
development or declare the whole optimization goal blocked. No new capture was
attempted. A future capture scope decision is needed only when parent reaches
this acceptance step; synthetic/historical screening can continue honestly.

## Search scope and method

Used rg file inventories/name filters and scoped text searches, including
`--hidden --no-ignore` where worktree/ignored artifacts could otherwise vanish:

- P100-arithmetic studies/qwen35-q4-{baseline,affinitywave,t64}-20260908,
  including all non-object result manifests, logs, request JSON and capture-like
  filenames. Checked run-job.py/server-probe.py output contracts.
- /home/arian/llama.cpp-qwen36/results, then focused35b-moe-pp-20260721 results.
- /home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/results
  and docs/p100-pairwave-project/evidence (runtime, frontier and replay reports).
- Expanded capture-name-only search across llama.cpp-q36-moe and llama.cpp-qwen36
  for cache-relay-routes*, *.awtr, activation binary/NumPy dumps and route captures;
  this discovered the ElasticWave worktree artifacts below.
- Relevant read-only shared memories: p100-affinitywave-parked-20260723,
  p100-pp-research-round3-20260722, qwen36-35b-moe-pp-execution.

This is not a claim to have searched every unrelated disk/mount/archive. No
unbounded binary content grep or expensive20GB model rehash was performed.

## Genuine token-level artifacts found: OLD Q8/fallback capture

Directory:
`/home/arian/llama.cpp-q36-moe/.worktrees/elasticwave/results/qwen36-35b-moe-pp-20260721/elasticwave-20260723/`

| File | Bytes | Records | SHA256 |
| --- | ---: | ---: | --- |
| routes-code-smoke.awtr |7865288|120|c286122a82123a6393a6dae21fac9a12c377e6a8c4620c0210356d2e41724623|
| routes-code-smoke-v2.awtr |2621768|40|d9233403213864cf945e8e9d82633cbc9065c95c94f780c013b2b26a1cc6ef6e|
| routes-code-16x4096.awtr |41948168|640|8f78f5338327becbac14ed9c82f93b1199eb4d7bf6455fef5e3b58cc744038c3|

Independent inventory_awtr.py streamed SHA256 and validated header framing;
every record has4096 tokens/top8, layer<40. It deliberately did NOT revalidate
every expert ID/unique-top8 tuple; the historical parser/notes report those
checks. The final file hash matches the retained NOTES-codex.md capture record.

Format AWTRV001 has8-byte magic; repeated little-endian<HIH headers contain
layer, token_count, top_k; payload is token-major uint16 expert IDs. Thus token
co-occurrence/order inside each record exists, unlike histograms. Missing:
activation bytes/dtype, routing weights, actual owner-local service descriptors,
cohort queues, cache events, output values, current Q4 model binding.

Provenance is decisive: capture-code-routes.sh line8 names
Qwen3.6-35B-A3B-Q8_0.gguf; lines76–78 use4096 context/ubatch with16 chunks.
NOTES-codex.md says capture deliberately OMITTED GGML_CUDA_MOE_PLAN and csort
reuse so IDs reached the host hook. This is NOT unchanged current T64 dispatch,
whose eligibility requires MOE_PLAN. Do not relabel this code corpus as current
Q4 code/prose or halve/double its records for genuine2k/8k execution.

Useful now: parser fixtures, authentic historical ragged-route shapes and
co-occurrence stress, always explicitly labeled Q8/fallback4096 proxy.

## Current Q4 artifacts: good run provenance, wrong tensors for replay

`studies/qwen35-q4-baseline-20260908/metadata.json` names current model
Qwen_Qwen3.6-35B-A3B-Q4_0.gguf and recorded download-verified SHA256
52312daa5b2190c1f5723d33c3315c01c55af4206f6c6e6eb63f3d8dd52bb85e.
Metadata hash:74e4589dd698216f2adc22788433f3dfda775a86213827fafad37b02a2273be1.
Model weights/scales can be read offline from existing GGUF if needed; this
does not reconstruct the activations/routes of a recorded forward pass.

`qwen35-q4-t64-20260908/vector-final-quality.meta.json` has exact command,
source environment, model path, wiki corpus path,8128 context input and library
hashes. It records cohortrail/interleave, deterministic route scatter,
AW_PARTIAL=bf16, AW_WIRE=f32, M64_SPLIT=2 and placement paths. Hash:
69642017def7b1428b338187521b9fcec7cce906c7cb62d2319159bc4f7e5ec8.
Its output is `vector-final-quality.logits`,508561428bytes,512 full-vocabulary
rows according to the quality report. These are FINAL logits, not internal
expert inputs, top8 router choices or owner queues. PRECAPTURE settings mean
CUDA graph/service capture behavior, not a saved activation payload.

`vector-final-server.requests.json` contains prompt_tokens, repetition/warmup,
HTTP prompt/decode timings, generated content and stop reason. Hash:
9fec9e635638b95f56deb60a4b30105f54127d65976469552ceadfe73e0d7d03.
It does not contain per-layer expert IDs, input tensors, dispatch/cache events.
Its64/8128 serving requests are not the desired2k/4k/8k code/prose replay suite.
Sibling native-Q4 study likewise has logits/quality/meta/service logs, not
the missing route+activation payload. No capture-file output hook was found
in the reviewed Q4 job/probe scripts; their explicit saved tensor is logits.

## Other plausible leads checked

- round3-phase0/hist-code.txt and hist-wiki.txt contain per-layer aggregate
  counts, no token co-occurrence. hist-code SHA256
  daab9279fb0aa8f790e8b316fa7c834e87137826364c0e6ebf9f8a01e65ba750.
  Shared memory/phase0 report explicitly identify Q8/fallback provenance.
- Historical affinitywave-trace-05dcabf9.patch is source for a route hook,
  NOT a capture artifact. Later TIMELINE-RESULTS.md uses GGML_CUDA_AW_TRACE=1
  for NVTX timing ranges on Q8 output-withheld service. Those nsys/sqlite files
  describe activity/timing, not activation memory or top8 arrays.
- Current deployment placement-hot16.json still has historical Q8 gguf_sha256
  c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131.
  It is a static ownership/hot-set map reused by the Q4 run, not a Q4 route trace.
- evidence/frontier/pairwave-replay-v3.json and temporal-queue-replay-v1.json
  are derived historical replay/calibration reports. They cite
  results/.../affinitywave-frontier-20260724/cache-relay-routes-pp8128-v1.bin,
  recorded41628160bytes,640records/4passes, hash
  0dd89bf538822716dbed425a6aadbe1b90f452de0a867859e9ca3ad4a1c618e4.
  That raw capture was NOT found by the scoped expanded search; cited result
  directories are absent in both main and current coding worktree. Reports
  retain historical owner/tile-derived data with Q8 tile calibration, not
  current Q4 activation bytes. Their hashes are respectively
  6e84760294ab974b4d02c5cc68a387b68c8fd4a3bbc7eb2cb3b01218d48638ea and
  ca789e9e5a68be04f7316ae4c9124c702e7ceb5566d31043331c145dd4572325.
- q4-track-20260713 results are Qwen3.6-27B dense requant benchmarks, not this
  35B MoE model. Their Q4 name is insufficient provenance.

## Fields needed beyond existing data

Current Q4 model/build/input hashes + tokenized2k/4k/8k code/prose identities;
layer/microbatch/token/lane and owner partition; per-token experts/router weights;
input-row maps and actual gate/up and post-SwiGLU down activation bytes/dtypes;
native tensor IDs/scales/offsets and fallback types; actual M64/M32/M16/cohort
queues/launch policy; recurring cache/repack sequence and owner-output reduction.
Existing sources provide rules, not the realized runtime state for these fields.

Requesting a new isolated capture later would be justified by this concrete
gap, but no authorization is inferred here. Full4-owner capture cannot silently
use GPU0/1 under currentGPU2/3 scope. Parent may first seek an existing capture
from another session or explicitly approve a scoped acquisition alternative;
either must retain honest model/topology provenance. No need to repeat closed
kernel failures while awaiting that acceptance data.
