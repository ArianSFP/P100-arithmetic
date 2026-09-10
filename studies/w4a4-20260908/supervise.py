"""GPU3-only fresh-worker controller. Requires a new active reservation.
No CUDA imports. No automatic reset/retry. All paths are repository relative.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
COORD=REPO/'bench/COORDINATION-20260908.md'
SHARED=Path('/home/arian/llama.cpp-qwen38-p100/bench/COORDINATION-20260908.md')
UUID='GPU-4868830a-c1cf-90bd-8018-2360c55293b8'
DESKTOP=Path('/home/arian/.xsession-errors')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def health(idle):
 def command(args):
  p=subprocess.run(args,capture_output=True,text=True,timeout=5)
  if p.returncode:raise RuntimeError('health query failed: '+p.stderr)
  return p.stdout
 device=command(['/usr/bin/nvidia-smi','-i',UUID,'--query-gpu=timestamp,uuid,name,memory.used,utilization.gpu,temperature.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'])
 processes=command(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])
 c=[v.strip() for v in device.strip().split(',')]
 if len(c)!=7 or c[1]!=UUID or 'P100' not in c[2] or int(c[3])>64 or (idle and int(c[4])) or int(c[5])>=85 or int(c[6]):raise RuntimeError('GPU3 busy/unhealthy')
 if any(line.startswith(UUID) for line in processes.splitlines()):raise RuntimeError('GPU3 compute process exists')
 return {'device':device,'processes':processes}
def storage():
 if DESKTOP.stat().st_size>100_000_000 or shutil.disk_usage(ROOT).free<2_000_000_000:raise RuntimeError('disk/desktop log guard')
def reservation(token,digests):
 for path,digest in zip([COORD,SHARED],digests):
  text=path.read_text()
  if sha(path)!=digest:raise RuntimeError('coordination changed; review before launch')
  if f'- HELD: w4a4 GPU3 token={token}' not in text.splitlines() or f'- FINAL RELEASE: w4a4 GPU3 token={token}' in text:raise RuntimeError('reservation missing/released')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--token',required=True);ap.add_argument('--coordination-sha256',required=True);ap.add_argument('--shared-sha256',required=True)
 ap.add_argument('--sanitizer',choices=['memcheck','synccheck','racecheck']);ap.add_argument('--timeout',type=int,default=120);ap.add_argument('tail',nargs=argparse.REMAINDER);a=ap.parse_args()
 if not 1<=a.timeout<=120:raise RuntimeError('timeout must be 1..120 seconds')
 tail=a.tail[1:] if a.tail[:1]==['--'] else a.tail
 if not (tail==['check'] or (len(tail)==4 and tail[0]=='bench') or (len(tail)==5 and tail[0]=='frozen')):raise RuntimeError('mode required: check | bench M K N')
 digests=[a.coordination_sha256,a.shared_sha256];reservation(a.token,digests);storage()
 inv=json.loads((ROOT/'build-manifest.json').read_text())
 for path,digest in inv['hashes'].items():
  if sha(ROOT/path)!=digest:raise RuntimeError('build artifact changed: '+path)
 run=ROOT/'gpu-results'/str(time.time_ns());run.mkdir(parents=True)
 worker=[str(ROOT/'worker'),str(ROOT/'kernels.cubin'),*tail]
 if a.sanitizer:worker=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',a.sanitizer,'--error-exitcode','9',*worker]
 argv=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES='+UUID,'/usr/bin/taskset','--cpu-list','0-11',*worker]
 record={'config_sha256':sha(Path(tail[-1])) if tail[0]=='frozen' else None,'argv':argv,'token':a.token,'manifest_sha256':sha(ROOT/'build-manifest.json'),'supervisor_sha256':sha(Path(__file__)),'coordination_sha256':digests,'hypothesis':(ROOT/'PLAN.md').read_text()}
 proc=None
 try:
  record['before']=health(True);reservation(a.token,digests)
  (run/'prelaunch.json').write_text(json.dumps(record,indent=2)+'\n')
  with (run/'stdout.txt').open('x') as out,(run/'stderr.txt').open('x') as err:
   proc=subprocess.Popen(argv,cwd=REPO,stdout=out,stderr=err,start_new_session=True);start=time.monotonic()
   while proc.poll() is None:
    if time.monotonic()-start>a.timeout:raise RuntimeError('timeout: no retry/reset')
    storage()
    try:proc.wait(timeout=1)
    except subprocess.TimeoutExpired:pass
  if proc.returncode:raise RuntimeError(f'worker failed rc={proc.returncode}: no retry/reset')
  record['after']=health(False);record['status']='PASS'
 except BaseException as exc:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGKILL)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:pass
  record['status']='STOP: '+str(exc)
 (run/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(run,record['status'])
 return 0 if record['status']=='PASS' else 3
if __name__=='__main__':raise SystemExit(main())
