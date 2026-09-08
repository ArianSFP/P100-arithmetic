# Publication checkpoint

The user explicitly requested a commit and SSH push of this completed W4A16
round-3 study to `ArianSFP/P100-arithmetic`. The source, compiled artifacts,
frozen configurations, raw logs and final report are preserved as measured.
Statements such as "no commit or push" in the experimental report describe
the test phase, before this separate publication request.

Included: the entire round-3 study except ignored Python caches, plus its
top-level README summary and a Git attribute preserving exact artifact bytes.
Excluded: the other sessions' prefill/W4A4 studies, their root documentation
changes, live shared coordination logs and any production worktree changes.
No GPU access, model run or production change is part of publication.

The archived control modules remain in `../../archive/`. The aggregate analyzer
can run without CUDA. The historical `audit.py` also checks original-host
coordination logs, as documented in RESULTS.md; those logs are deliberately
not exported. The recorded audit and final-health JSON preserve its findings.
Released reservation tokens are provenance, not permission for a future run.
