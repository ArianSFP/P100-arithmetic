#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import subprocess

p = Path(__file__).resolve().parent
seed_dir = p.parent.parent / "w16-dualrail-20260909"
seed_files = [
    "q8.inc", "decode.inc", "packed.inc", "pair.inc", "fallback.inc",
    "w16-pair.inc", "w16-fallback.inc",
]
for name in seed_files:
    assert (p / name).read_bytes() == (seed_dir / name).read_bytes(), name
sources = [
    "worker.cu", "q8.inc", "decode.inc", "packed.inc", "pair.inc",
    "fallback.inc", "w16-pair.inc", "w16-fallback.inc", "w4-pair.inc",
    "cpu_checks.py", "audit.py", "build.py", "README.md", "PREPARATION.md",
]
cpu = subprocess.run(["python3", str(p / "cpu_checks.py")],
                     capture_output=True, text=True)
(p / "cpu-checks.log").write_text(cpu.stdout + cpu.stderr)
cpu.check_returncode()
cmd = [
    "/usr/local/cuda-12.8/bin/nvcc", "-ccbin", "/usr/bin/g++-13",
    "-O3", "-std=c++17", "-arch=sm_60", "-lineinfo", "-Xptxas=-v",
    str(p / "worker.cu"), "-o", str(p / "worker"),
]
result = subprocess.run(cmd, capture_output=True, text=True)
(p / "build.log").write_text(result.stdout + result.stderr)
result.check_returncode()
audit = subprocess.run(["python3", str(p / "audit.py")],
                       capture_output=True, text=True)
(p / "audit.log").write_text(audit.stdout + audit.stderr)
audit.check_returncode()
hashes = {name: hashlib.sha256((p / name).read_bytes()).hexdigest()
          for name in sources + ["worker", "worker.sass", "resources.txt",
                                 "sass-audit.json", "cpu-checks.log"]}
(p / "manifest.json").write_text(json.dumps({
    "gpu_tested": False,
    "source_seed": "studies/w16-dualrail-20260909",
    "seed_files_byte_identical": seed_files,
    "command": cmd,
    "cpu_checks": cpu.stdout.strip(),
    "hashes": hashes,
}, indent=2) + "\n")
print("PASS CPU and SM60 compile; GPU not executed")
