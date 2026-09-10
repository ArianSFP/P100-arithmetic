#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
import json
import re
import statistics

root = Path(__file__).resolve().parent
pattern = re.compile(r"TIME mode=(\S+) rep=(\d+) us=([0-9.eE+-]+) useful_tmac_s=([0-9.eE+-]+)")
rows = defaultdict(list)
for line in (root / "figlut-packed4-full16-r1.out").read_text().splitlines():
    match = pattern.fullmatch(line)
    if match and int(match.group(2)) != 0:
        rows[match.group(1)].append((float(match.group(3)), float(match.group(4))))
report = {name: {"samples": len(values), "median_us": statistics.median(v[0] for v in values),
                 "min_us": min(v[0] for v in values), "max_us": max(v[0] for v in values),
                 "median_useful_tmac_s": statistics.median(v[1] for v in values)}
          for name, values in rows.items()}
candidate = report["packed4-full16-complete-g32"]
base = report["half2-complete-lower-bound"]
candidate["speedup_vs_half2_lower_bound"] = base["median_us"] / candidate["median_us"]
candidate["passes_10p25_tmac_streamed_admission"] = candidate["median_useful_tmac_s"] >= 10.25
candidate["passes_9p488_down_mathematical_rate"] = candidate["median_useful_tmac_s"] >= 9.488
candidate["passes_9p070_gu_mathematical_rate"] = candidate["median_useful_tmac_s"] >= 9.070
(root / "packed4_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
