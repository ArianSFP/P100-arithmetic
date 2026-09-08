#!/usr/bin/env python3
"""Paired worker medians, always against the fastest included control."""
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent
for tag in sys.argv[1:]:
    folder = ROOT/'gpu-results'/tag
    result = json.loads((folder/'result.json').read_text())
    assert result['status'] == 'PASS', result['status']
    values = defaultdict(list)
    for line in (folder/'stdout.txt').read_text().splitlines():
        if line.startswith('META '): print(tag, line)
        m = re.fullmatch(r'TIME (\S+) prep=(\d+) rep=(\d+) us=(\S+)', line)
        if m: values[m[1],int(m[2])].append(float(m[4]))
    for prep in (0,1):
        med = {name:statistics.median(v) for (name,p),v in values.items() if p==prep}
        base = min((v,name) for name,v in med.items() if name.startswith('aw_'))
        print('prep', prep, 'fastest_control', base)
        for name,us in sorted(med.items(), key=lambda item:item[1]):
            v = values[name,prep]
            print(f'{name:28s} {us:9.3f} us {base[0]/us:6.3f}x range={min(v):.3f}..{max(v):.3f} n={len(v)}')
