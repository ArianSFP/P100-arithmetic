# Publication notes — 2026-09-08

The user explicitly requested a documented commit and SSH push to the separate
`ArianSFP/P100-arithmetic` repository. Its SSH remote was accessible and empty
before preparation. The original isolated research directory is not a Git
repository and remains unchanged. No llama.cpp worktree was committed or pushed.

The snapshot preserves every isolated research artifact except third-party
tooling/virtual environments, Python caches and nested Git metadata. The new
front documentation distinguishes the final W4A16 synthetic kernel gains from
model-level performance and flags the earlier HFMA2 harness's limitations.
There is no hidden low-bit arithmetic-lane discovery claim.

Publication verification is CPU-only: exact source-to-snapshot comparison,
per-file SHA-256 inventory, recorded build/control hashes, independent round-2
log/aggregate checks, verifier regression tests, fresh CPU source tests, and
historical analyzers in a temporary copy. No GPU reservation, CUDA test, driver
change, model run, reset or production integration is performed for publication.

Claude's shared memory and plans remain read-only and are not exported. Local
paths/device IDs in original reports are retained as historical provenance;
private keys, API credentials, external datasets and model files are not included.

## Verification results

- Source-to-archive checksum comparison: no differences, 1,152 archived files,
  87,115,957 bytes (excluding only the documented tooling/cache patterns).
- Read-only verifier: PASS; 45 final-round workers, 3,987 pipeline checks,
  36 frozen benchmark workers and 3,456 retained samples agree with raw logs.
- All 12 manifest/timing verifier regression tests passed.
- Fresh GCC 14 CPU tests passed normally and under UBSAN: 63,488 finite half
  patterns, 1,015,808 exact W4 products and 65,536 lossless blocks.
- All three historical aggregate analyzers passed in a separate temporary copy:
  final W4A16 (36 workers), first W4A16 (12), and A8 layout study (24).
- Pre-publication credential-pattern screening found no credentials.

The standalone commit uses the existing project-local author identity, Arian
`<arian@dell-c4130.local>`, configured only in the new repository. No global Git
configuration was changed. The destination remote uses SSH with strict existing
host-key verification; no force push is needed for the initially empty remote.
