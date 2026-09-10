"""Read-only raw sweep summary. No GPU or artifact changes."""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parent
paths=[Path(p) for p in sys.argv[1:]] or sorted((root/'gpu-results').glob('*/stdout.txt'))[-1:]
for p in paths:
 t=p.read_text();data=re.search(r'DATA M=(\d+) K=(\d+) N=(\d+)',t)
 print(p.parent.name, data.groups() if data else 'no shape')
 v=defaultdict(list)
 for rnd,name,us in re.findall(r'TIME round=(\d+) config=(\w+) us=([\d.]+)',t):
  if int(rnd)>0:v[name].append(float(us))
 for name,ts in sorted(v.items(),key=lambda kv:statistics.median(kv[1]))[:15]:
  print(name,round(statistics.median(ts),3), 'range',round(min(ts),3),round(max(ts),3))
