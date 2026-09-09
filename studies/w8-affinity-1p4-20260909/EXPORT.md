# Source and evidence export

Start with [TARGET-1P9.md](TARGET-1P9.md) for the latest results. The 1.9x goal is not achieved. Reported gains use two FP16 accumulation rails with FP32 final addition/output; F32/FP16 table labels describe input storage. Model accuracy and throughput remain unverified.

This export includes kernel sources, harnesses, analysis scripts, raw timing and sanitizer logs, manifests, and historical failed experiments. It also includes the isolated model adapter source. The companion W8 low-activation and W16 reference studies are exported alongside it.

Generated executables, shared libraries, object files, disassembly, linker response files, and build directories remain local. Recorded binary hashes identify the measured local builds; the binaries themselves are not shipped. Analysis scripts that verify a binary require that local artifact or a matching rebuild. CUDA 12.8 and the compiler paths recorded by each build script are required. Model-adapter builds also require the referenced llama.cpp worktree and its existing build dependencies.

Historical generators and snapshots document development; they are not a single installer. Use the final source files and their build scripts. No production library or model weights are included.
