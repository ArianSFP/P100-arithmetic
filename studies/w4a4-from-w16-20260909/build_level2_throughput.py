#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "level2-throughput.cu"
BINARY = ROOT / "level2-throughput"
BUILD_LOG = ROOT / "level2-throughput-build.log"
SASS_FILE = ROOT / "level2-throughput.sass"
COUNTS_FILE = ROOT / "level2-throughput-sass-counts.json"
MANIFEST = ROOT / "level2-throughput-manifest.json"

command = [
    "/usr/local/cuda-12.8/bin/nvcc",
    "-ccbin", "/usr/bin/g++-13",
    "-O3", "-std=c++17", "-arch=sm_60", "-lineinfo", "-Xptxas=-v",
    str(SOURCE), "-o", str(BINARY),
]
completed = subprocess.run(command, capture_output=True, text=True)
BUILD_LOG.write_text(completed.stdout + completed.stderr)
completed.check_returncode()

sass = subprocess.check_output(
    ["/usr/local/cuda-12.8/bin/cuobjdump", "--dump-sass", str(BINARY)],
    text=True,
)
SASS_FILE.write_text(sass)

function_blocks = re.findall(
    r"Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)", sass, re.S
)
opcodes = (
    "HFMA2", "HMUL2", "HADD2", "F2I", "PRMT", "IADD", "IADD3",
    "SHR", "SHL", "LOP", "LDC", "LDG", "STG", "LDL", "STL", "BRA",
)
counts = {}
for name, body in function_blocks:
    if not name.startswith("level2_"):
        continue
    instructions = [
        line for line in body.splitlines()
        if re.search(r"/\*[0-9a-f]+\*/", line, re.I)
    ]
    row = {"instructions": len(instructions)}
    for opcode in opcodes:
        row[opcode] = sum(
            re.search(r"\b" + opcode + r"(?:\.|\b)", line) is not None
            for line in instructions
        )
    counts[name] = row

expected = {
    "level2_plain_hfma2",
    "level2_normalized_split",
    "level2_hmul_prmt_repair",
}
if set(counts) != expected:
    raise RuntimeError(f"unexpected kernels: {sorted(counts)}")
for name, row in counts.items():
    if row["LDL"] or row["STL"]:
        raise RuntimeError(f"local-memory spill instruction in {name}: {row}")
COUNTS_FILE.write_text(json.dumps(counts, indent=2) + "\n")

hash_names = [
    SOURCE.name, Path(__file__).name, BINARY.name, BUILD_LOG.name,
    SASS_FILE.name, COUNTS_FILE.name,
]
MANIFEST.write_text(json.dumps({
    "scope": "compile/SASS audit only; no GPU execution",
    "launch_contract": {"blocks": 112, "threads_per_block": 256,
                        "independent_packed_chains": 8},
    "normalization": {
        "useful_scalar_int4_macs_per_thread_iteration": 64,
        "derivation": "8 chains * (+a,-a) * 4 products per chain",
        "plain_inner": "32 HFMA2 instructions / 64 useful scalar MACs",
        "normalized_split_inner": "32 HFMA2 + 48 HADD2 / 64 useful scalar MACs",
        "repair_inner": "16 HMUL2 + 8 PRMT + 32 F2I plus integer repair/accumulation / 64 useful scalar MACs",
    },
    "command": command,
    "hashes": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in hash_names
    },
}, indent=2) + "\n")
print(json.dumps(counts, indent=2))
