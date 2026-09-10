#!/usr/bin/env python3
"""Regenerate the instruction and resource summaries for activation_probe."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
SASS = ROOT / "activation_probe.sass"
CUBIN = ROOT / "activation_probe.cubin"


def instruction_counts() -> dict[str, dict[str, int]]:
    functions: dict[str, Counter[str]] = {}
    current: str | None = None
    for line in SASS.read_text().splitlines():
        match = re.search(r"Function\s+:\s+(\S+)", line)
        if match:
            current = match.group(1)
            functions[current] = Counter()
            continue
        if current is None:
            continue
        match = re.search(
            r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?([A-Z][A-Z0-9_.]+)", line
        )
        if match:
            functions[current][match.group(1).split(".")[0]] += 1
    return {
        name: {"total": sum(counts.values()), **dict(sorted(counts.items()))}
        for name, counts in functions.items()
    }


def resources() -> dict[str, dict[str, int]]:
    raw = subprocess.check_output(
        ["cuobjdump", "--dump-resource-usage", str(CUBIN)], text=True
    )
    result: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in raw.splitlines():
        match = re.match(r" Function (\S+):", line)
        if match:
            current = match.group(1)
            continue
        if current is None:
            continue
        match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if match:
            result[current] = {
                "registers_per_thread": int(match.group(1)),
                "stack_bytes_per_thread": int(match.group(2)),
                "shared_bytes_per_cta": int(match.group(3)),
                "local_bytes_per_thread": int(match.group(4)),
            }
            current = None
    return result


def main() -> None:
    counts = instruction_counts()
    resource_data = resources()
    (ROOT / "activation_probe-counts.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )
    (ROOT / "activation_probe-resources.json").write_text(
        json.dumps(resource_data, indent=2) + "\n"
    )
    assert len(counts) == 2 and len(resource_data) == 2
    for name in sorted(counts):
        c = counts[name]
        r = resource_data[name]
        print(
            name,
            f"registers={r['registers_per_thread']}",
            f"static={c['total']}",
            f"SHFL={c.get('SHFL', 0)}",
            f"BFE={c.get('BFE', 0)}",
            f"IADD3={c.get('IADD3', 0)}",
            f"IADD={c.get('IADD', 0)}",
        )


if __name__ == "__main__":
    main()
