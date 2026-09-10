#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
import json
import re
import statistics

root = Path(__file__).resolve().parent
source = root / "figlut-half128-prebuilt-r0.out"
pattern = re.compile(r"TIME mode=(\S+) rep=(\d+) us=([0-9.eE+-]+) useful_tmac_s=([0-9.eE+-]+)")
rows = defaultdict(list)
for line in source.read_text().splitlines():
    match = pattern.fullmatch(line)
    if match and int(match.group(2)) != 0:
        rows[match.group(1)].append((float(match.group(3)), float(match.group(4))))

report = {}
for name, values in rows.items():
    report[name] = {
        "samples": len(values),
        "median_us": statistics.median(v[0] for v in values),
        "min_us": min(v[0] for v in values),
        "max_us": max(v[0] for v in values),
        "median_useful_tmac_s": statistics.median(v[1] for v in values),
    }
for r in (16, 8, 4):
    base = report[f"resident-half2-r{r}"]["median_us"]
    for kind in ("prebuilt-full256", "prebuilt-half128"):
        item = report[f"{kind}-r{r}"]
        item["speedup_vs_matched_half2"] = base / item["median_us"]
        item["passes_1p45_admission"] = item["speedup_vs_matched_half2"] >= 1.45
        item["above_1p30_optimistic"] = item["speedup_vs_matched_half2"] >= 1.30

target = root / "prebuilt_summary.json"
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
