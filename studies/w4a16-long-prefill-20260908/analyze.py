#!/usr/bin/env python3
"""Paired within-worker medians; never call equal-M screening trace replay."""
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import sys
ROOT=Path(__file__).resolve().parent
def summarize(tag):
    p=ROOT/'gpu-results'/tag
    result=json.loads((p/'result.json').read_text());assert result['status']=='PASS'
    request=json.loads((p/'prelaunch.json').read_text())['request']
    output=(p/'stdout.txt').read_text()
    if request.get('sanitizer') or any(marker in output for marker in ('RAW_WITNESS ', 'SCALE_STRESS ', 'DEQUANT_CHECK ')):
        return dict(tag=tag,correctness_only=True)
    values=defaultdict(list)
    for name,prep,us in re.findall(r'TIME (\S+) prep=(\d+) rep=\d+ us=(\S+)',output):
        if prep=='1':values[name].append(float(us))
    med={n:statistics.median(v) for n,v in values.items()}
    control=med['aw_current_q4']
    old=min(med[n] for n in ('aw_t64_f32','aw_t64_a16','aw_q4_a16'))
    return dict(tag=tag,scope='equal-M screening, NOT whole-prompt/ragged acceptance',
                meta=re.search(r'^META .+$',output,re.M)[0],current_q4_us=control,
                fastest_legacy_control_us=old,
                candidates={n:dict(us=v,vs_current=control/v,vs_fastest_control=min(old,control)/v,
                                   minimum=min(values[n]),maximum=max(values[n]),samples=len(values[n]))
                            for n,v in med.items() if n=='aw_special_q4' or n.startswith(('delivery_fused','staged_','wide_','compact_','mid_','tile256_','tile512_'))})
if __name__=='__main__':
    reports=[summarize(t) for t in sys.argv[1:]]
    print(json.dumps(reports,indent=2))
