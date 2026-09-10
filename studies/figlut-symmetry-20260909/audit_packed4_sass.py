#!/usr/bin/env python3
from collections import Counter
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parent
functions = {}
current = None
for line in (root / "packed4_worker.sass").read_text().splitlines():
    match = re.search(r"Function\s+:\s+(\S+)", line)
    if match:
        current = match.group(1)
        functions[current] = Counter()
        continue
    if current is None:
        continue
    match = re.search(r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?([A-Z][A-Z0-9_.]+)", line)
    if match:
        functions[current][match.group(1).split(".")[0]] += 1

result = {name: dict(sorted(counts.items())) for name, counts in functions.items()}
(root / "packed4_sass_counts.json").write_text(json.dumps(result, indent=2) + "\n")
for name, counts in result.items():
    selected = {key: counts.get(key, 0) for key in ("BAR", "BFE", "F2I", "FADD", "FFMA", "FMUL", "HFMA2", "I2F", "IADD", "IADD3", "LDG", "LDS", "LOP", "LOP3", "PRMT", "SHFL", "STS")}
    print(name, "total", sum(counts.values()), json.dumps(selected, sort_keys=True))
