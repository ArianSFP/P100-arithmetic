#!/usr/bin/env python3
"""Offline SASS screening for every generated selector candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def disassemble(candidate: Path, nvdisasm: str):
    completed = subprocess.run(
        [nvdisasm, str(candidate)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if "HFMA2" in line]
    return {
        "path": str(candidate),
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "returncode": completed.returncode,
        "hfma2_lines": lines,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--nvdisasm", default="/usr/local/cuda-12.8/bin/nvdisasm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = sorted(args.candidate_dir.glob("*.cubin"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda p: disassemble(p, args.nvdisasm), candidates))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"candidates": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
