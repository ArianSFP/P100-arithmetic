#!/usr/bin/env python3
"""Generate all 4^4 selector words offline; never launches a candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import product
from pathlib import Path


FIELDS = ("a", "b", "c", "dest")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--section", required=True)
    parser.add_argument("--offset", required=True, type=lambda x: int(x, 0))
    parser.add_argument("--baseline-word", required=True, type=lambda x: int(x, 0))
    parser.add_argument("--patcher", type=Path, default=Path(__file__).with_name("patch_cubin.py"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    index = {
        "input": str(args.input),
        "input_sha256": source_hash,
        "section": args.section,
        "offset": args.offset,
        "baseline_word": f"0x{args.baseline_word:016x}",
        "candidates": [],
        "execution": "offline_generation_only",
    }
    for values in product(range(4), repeat=4):
        selectors = dict(zip(FIELDS, values))
        tag = "".join(str(v) for v in values)
        output = args.output_dir / f"selectors-{tag}.cubin"
        if tag == "0000":
            index["candidates"].append({
                "tag": tag,
                "selectors": selectors,
                "path": str(args.input),
                "sha256": source_hash,
                "baseline": True,
            })
            continue
        command = [
            str(args.patcher), "--input", str(args.input), "--output", str(output),
            "--section", args.section, "--offset", hex(args.offset),
            "--baseline-word", hex(args.baseline_word),
        ]
        for field, value in selectors.items():
            command.extend([f"--{field}-selector", str(value)])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        index["candidates"].append({
            "tag": tag,
            "selectors": selectors,
            "path": str(output),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        })
    (args.output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"generated": len(index["candidates"]), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
