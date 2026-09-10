#!/usr/bin/env python3
from collections import Counter
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parent
functions = {}
current = None
for line in (root / "prebuilt_worker.sass").read_text().splitlines():
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

wanted = ("LDS", "STS", "LDG", "IADD", "IADD3", "BRA", "SSY", "SYNC",
          "BFE", "SHR", "SHL", "LOP", "LOP3", "ISETP", "SEL", "HFMA2")
result = {
    name: {op: counts[op] for op in wanted if counts[op]}
    for name, counts in functions.items()
}
(root / "prebuilt_sass_counts.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
