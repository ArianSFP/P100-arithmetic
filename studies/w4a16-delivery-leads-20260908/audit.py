#!/usr/bin/env python3
"""Static disassembly inventory, not dynamic counts or hardware counters."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent
raw=(root/'build/worker.sass').read_bytes()
sections=re.split(r'Function\s*:\s*(\S+)',raw.decode())
out={}
for name,body in zip(sections[1::2],sections[2::2]):
    counts=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s',body))
    if 'delivery_lead' in name or 'prepare_tiled' in name:
        out[name]=dict(sorted(counts.items()))
        if 'ELi16EE' in name:
            print(name, {key:value for key,value in counts.items() if key.startswith(('STS','LDG','HADD2','FFMA'))})
(root/'build/instruction-counts.json').write_text(json.dumps(dict(sass_sha256=hashlib.sha256(raw).hexdigest(),scope='whole-function static counts; NOT dynamic execution counts',functions=out),indent=2)+'\n')
