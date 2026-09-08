#!/usr/bin/env python3
"""Sequential fresh supervised workers, fixed configs, stop on any failure."""
import argparse
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--coordination-sha256',required=True);p.add_argument('--round',type=int,required=True)
args=p.parse_args()
assert args.round in (0,1,2)
shapes=[(5120,5120,1),(17408,5120,1),(5120,17408,1),(5120,5120,4)]
jobs=[(w,*s) for s in shapes for w in (2,4)]
jobs=jobs[args.round:]+jobs[:args.round]
for w,m,k,n in jobs:
    print(f'FRESH round={args.round} W={w} M={m} K={k} N={n}',flush=True)
    result=subprocess.run(['python3',str(ROOT/'supervise.py'),'--coordination-sha256',args.coordination_sha256,
        '--kind','layout','--timeout','90','--','bench','--bits',str(w),'--m',str(m),'--k',str(k),'--n',str(n),
        '--configs',str(ROOT/'frozen-configs.txt')])
    if result.returncode:raise SystemExit(result.returncode)
print('FROZEN ROUND COMPLETE; reservation remains held.',flush=True)
