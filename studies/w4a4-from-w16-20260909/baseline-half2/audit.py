#!/usr/bin/env python3
from pathlib import Path
import json
import re
import subprocess

p = Path(__file__).resolve().parent
tool = "/usr/local/cuda-12.8/bin/cuobjdump"
sass = subprocess.check_output([tool, "--dump-sass", str(p / "worker")], text=True)
resources = subprocess.check_output([tool, "--dump-resource-usage", str(p / "worker")], text=True)
(p / "worker.sass").write_text(sass)
(p / "resources.txt").write_text(resources)
blocks = {name: body for name, body in re.findall(
    r"Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)", sass, re.S)}
selected = {}
for name, body in blocks.items():
    family = None
    if "w4_pair_exact_g32" in name and ("Li512ELi2048" in name or "Li2048ELi512" in name):
        family = "w4_exact_pair"
    elif "w16_pair_half" in name and ("Li512ELi2048" in name or "Li2048ELi512" in name):
        family = "w16_pair"
    elif "w16_fallback_half" in name:
        family = "w16_fallback"
    elif "w4_quantize_a4" in name:
        family = "w4_quantizer"
    if family is None:
        continue
    instructions = [line for line in body.splitlines()
                    if re.search(r"/\*[0-9a-f]+\*/", line, re.I)]
    selected[name] = {
        "family": family,
        "instructions": len(instructions),
        "hfma2": sum(" HFMA2" in line for line in instructions),
        "ffma": sum(" FFMA" in line for line in instructions),
        "popc": sum(" POPC" in line for line in instructions),
        "ldl": sum(" LDL" in line for line in instructions),
        "stl": sum(" STL" in line for line in instructions),
    }
assert sum(v["family"] == "w4_exact_pair" for v in selected.values()) == 2, selected
assert all(v["ldl"] == 0 and v["stl"] == 0 for v in selected.values()
           if v["family"] in {"w4_exact_pair", "w16_pair"}), selected
assert all(v["hfma2"] > 0 for v in selected.values()
           if v["family"] in {"w4_exact_pair", "w16_pair"}), selected
(p / "sass-audit.json").write_text(json.dumps(selected, indent=2) + "\n")
print(json.dumps(selected, indent=2))
