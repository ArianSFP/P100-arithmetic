#!/usr/bin/env python3
"""Paired worker medians; never pool devices or count kernel-only as pipeline."""
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import sys

root=Path(__file__).resolve().parent
for tag in sys.argv[1:]:
    folder=root/'gpu-results'/tag
    meta=json.loads((folder/'result.json').read_text())
    assert meta['status']=='PASS',meta['status']
    request=json.loads((folder/'prelaunch.json').read_text())['request']
    output=(folder/'stdout.txt').read_text()
    if request.get('sanitizer') or 'RAW_WITNESS ' in output:
        print(tag,'PASS correctness-only; timing excluded')
        continue
    values=defaultdict(list)
    for line in output.splitlines():
        if line.startswith(('META ','SCRATCH ')):print(tag,line)
        m=re.fullmatch(r'TIME (\S+) prep=(\d+) rep=(\d+) us=(\S+)',line)
        if m:values[m[1],int(m[2])].append(float(m[4]))
    for prep in (1,0):
        med={name:statistics.median(v) for (name,p),v in values.items() if p==prep}
        control=min((v,n) for n,v in med.items() if n.startswith('aw_'))
        print('prep',prep,'fastest_control',control)
        for name,us in sorted(med.items(),key=lambda x:x[1]):
            v=values[name,prep]
            print(f'{name:28s} {us:9.3f} us {control[0]/us:6.3f}x range={min(v):.3f}..{max(v):.3f} n={len(v)}')
