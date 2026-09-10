from pathlib import Path
import json
import re
import statistics

root = Path(__file__).resolve().parent
tags = ["gu-ctas1-r0", "gu-ctas2-r0", "down-ctas1-r0", "down-ctas2-r0"]
result = {}
for tag in tags:
    samples = {}
    checks = []
    for line in (root / f"{tag}.out").read_text().splitlines():
        match = re.fullmatch(r"TIME mode=(\S+) rep=(\d+) us=([0-9.eE+-]+)", line)
        if match and int(match.group(2)) > 0:
            samples.setdefault(match.group(1), []).append(float(match.group(3)))
        if line.startswith(("QUANT_CHECK", "PLANNER_CHECK", "CHECK ")):
            checks.append(line)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    result[tag] = {
        "samples_per_mode": {name: len(values) for name, values in samples.items()},
        "median_us": medians,
        "w4_vs_w16_speed": medians["w16-dualrail"] / medians["w4a4-exact-g32"],
        "w16_vs_w4_speed": medians["w4a4-exact-g32"] / medians["w16-dualrail"],
        "w4_target_us_for_2x": medians["w16-dualrail"] / 2,
        "w4_over_2x_target": medians["w4a4-exact-g32"] / (medians["w16-dualrail"] / 2),
        "checks": checks,
    }
(root / "timing-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
