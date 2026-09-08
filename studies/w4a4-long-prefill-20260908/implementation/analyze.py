#!/usr/bin/env python3
"""Print paired pipeline medians from successful, manifest-verified workers."""
import hashlib,json,re,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for tag in sys.argv[1:]:
    p=ROOT/'gpu-results'/tag
    meta=json.loads((p/'result.json').read_text())
    assert meta['status']=='PASS' and meta['returncode']==0
    for name,h in meta['build']['hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    text=(p/'stdout.txt').read_text();samples={}
    assert text.rstrip().endswith('PASS')
    for _,name,value in re.findall(r'TIME rep=(\d+) config=(\S+) pipeline_us=([\d.]+)',text):samples.setdefault(name,[]).append(float(value))
    med={name:statistics.median(values) for name,values in samples.items()}
    baseline=med['current_q4_T64_A16']
    print(json.dumps({'tag':tag,'median_us':med,'speedup_to_current_T64':{name:baseline/value for name,value in med.items()},'min_max_us':{name:[min(v),max(v)] for name,v in samples.items()},'scope':'synthetic grouped screening, not model acceptance'},indent=2))
