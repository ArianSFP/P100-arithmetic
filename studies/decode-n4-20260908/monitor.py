#!/usr/bin/env python3
"""Read-only shared-host activity snapshots during concurrent GPU reservations."""
import json,subprocess,time,sys
from pathlib import Path
r=Path(__file__).resolve().parent;run=r/sys.argv[1];end=time.monotonic()+1500
with (run/'concurrency.jsonl').open('x') as out:
 while time.monotonic()<end and not (run/'stop-monitor').exists():
  entry=dict(time=time.time())
  for key,cmd in [('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm','--format=csv,noheader,nounits']),('clients',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),('cpu',['ps','-eo','pid,comm,pcpu'])]:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
   text=p.stdout
   if key=='cpu':text='\n'.join(s for s in text.splitlines() if any(n in s for n in ['nvcc','ptxas','cc1plus','cicc','worker','g++']))
   entry[key]=dict(returncode=p.returncode,stdout=text,stderr=p.stderr)
  out.write(json.dumps(entry)+'\n');out.flush();time.sleep(1)
print('MONITOR STOPPED')
