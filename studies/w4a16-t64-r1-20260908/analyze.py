#!/usr/bin/env python3
"""Read-only timing summary; no cross-worker/cross-shape speedup ratios."""
from collections import defaultdict
from pathlib import Path
import re
import statistics
import sys

root=Path(__file__).resolve().parent
for tag in sys.argv[1:]:
    path=root/'gpu-results'/tag/'stdout.txt'
    values=defaultdict(list)
    checks=[]
    for line in path.read_text().splitlines():
        if line.startswith('META '):print(tag,line)
        if line.startswith('CHECK '):checks.append(line)
        m=re.fullmatch(r'TIME (\S+) prep=(\d+) rep=(\d+) us=(\S+)',line)
        if m:values[m[1],int(m[2])].append(float(m[4]))
    for prep in (0,1):
        if ('aw_t64_m64_sk2',prep) not in values:continue
        base=statistics.median(values['aw_t64_m64_sk2',prep])
        print('prep',prep,'control',round(base,3))
        for (name,p),samples in sorted(values.items(),key=lambda kv:statistics.median(kv[1])):
            if p==prep:
                med=statistics.median(samples)
                print(f'{name:24s} {med:10.3f} us {base/med:7.3f}x n={len(samples)}')
    for line in checks:print(line)
