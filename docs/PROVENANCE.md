# Snapshot provenance

Published at the user's explicit request to the standalone
`ArianSFP/P100-arithmetic` repository. Publication does not commit or push any
llama.cpp worktree and does not rerun GPU tests.

## Included

`archive/` is a byte-preserving copy of `/home/arian/hfma2-gp100-probe`, including
all isolated research reports/notes, source, CPU checks, worker/supervisor code,
original and modified cubins, host executables, PTX/SASS, build inventories,
frozen selections, raw stdout/stderr, error records, health checks and aggregates.
Historical builds and unsuccessful experiments are retained. The manifest
records the exact file count, byte count, relative path, size and SHA-256 of
each file; see [EXPORT-MANIFEST.json](../EXPORT-MANIFEST.json).

The source tree was not edited to make the snapshot. Source and binary hashes
in the historical results therefore retain their meaning. The new top-level
documentation, manifest tools and tests are publication additions, not part of
the measured kernel builds.

## Excluded

- `tooling/`: the third-party CuAssembler checkout and its Python virtual
  environment. The recorded checkout was
  `a86dbdf171254ac9e33e3877e35deb2413133b84` of
  [cloudcores/CuAssembler](https://github.com/cloudcores/CuAssembler).
- `__pycache__/`, `*.pyc` and any nested `.git/` metadata.
- Material outside the isolated research tree: model weights, datasets,
  production repositories, Claude memory/plans, shared coordination files,
  SSH keys/configuration and machine/toolkit installations.

No user research artifact was excluded merely for an unfavorable result.
Executable artifacts are retained for hash provenance, not recommended for
blind execution. Third-party tools must be obtained under their own terms.
This publication does not choose or grant a new repository license.

## Historical metadata and external references

Raw artifacts retain local paths, device UUIDs, timestamps, process names/PIDs,
compiler commands, reservation labels and coordination-file hashes. Reservation
labels are inactive local coordination identifiers, not SSH/API credentials.
Keeping them connects runs to their recorded environment. Credential-pattern
screening was performed before publication; no credentials were identified.

Absolute links to other local worktrees or shared-memory notes are historical
references and will not resolve on GitHub. Those external files are not part of
this archive; claims relying only on them are not independently reproduced by
this repository. Use the standalone studies and their raw evidence for the
published performance claims.

Some inventories include absolute original paths; relocation does not make
them new valid launch commands. The read-only verifier checks relative archive
contents and never follows a historical command or contacts the original host.
The final round-2 audit contains both a historical pre-release report digest
and the final report digest. Only the final digest identifies the archived final
report; the earlier digest is retained as history, not a failed current check.

## Verification scope

The manifest detects missing/modified/extra archive files and rejected symlinks.
It is an integrity inventory, not an independently signed attestation. A party
able to replace both files and manifest can create a different snapshot; use a
trusted Git commit ID when referring to this publication.

The verifier additionally checks the final W4A16 source/build/control hashes,
frozen configuration and report hashes, worker status/provenance, raw timing
rounds, recomputed medians, numerical-check counts and aggregate records.
Earlier studies are byte-verified but not all semantically re-audited by this
new utility. Hash agreement and successful log analysis do not establish new
hardware correctness or production inference performance.
