#!/usr/bin/env python3
"""GPU1-only bounded workers, serialized by the rig's shared lock."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time,sys
from pathlib import Path
R=Path(__file__).resolve().parent
UUID='GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b'
TOKEN='prefill-fp16-ratio-'+str(time.time_ns())+'-gpu1'
COORD=[R.parents[1]/'bench/COORDINATION-20260908.md',Path('/home/arian/llama.cpp-qwen38-p100/bench/COORDINATION-20260908.md')]
def health():
 raw=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=uuid,memory.used,utilization.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True,timeout=5)
 c=[x.strip() for x in raw.split(',')]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=5)
 if c[0]!=UUID or int(c[1])>64 or int(c[4]) or UUID in apps:raise RuntimeError('GPU1 busy/unhealthy: '+raw+apps)
 return raw

def guard():
 if Path('/home/arian/.xsession-errors').stat().st_size>100_000_000 or shutil.disk_usage(R).free<3_000_000_000:raise RuntimeError('Storage/desktop guard')
def append(s):
 for p in COORD:
  with p.open('a') as f:f.write('\n'+s+'\n')
lock=open('/tmp/affinitywave-4gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
guard();before=health()
append('HELD: '+TOKEN+'; GPU1 only, stage timing and cuBLAS sweep. Global lock held; GPU2/3 untouched. No production change.')
run=R/('run-'+str(time.time_ns()));run.mkdir();p=None
try:
 manifest={x:hashlib.sha256((R/x).read_bytes()).hexdigest() for x in ['worker','worker.cu','run.py']}
 (run/'manifest.json').write_text(json.dumps(dict(hashes=manifest,before=before,token=TOKEN),indent=2))
 with (run/'telemetry.csv').open('w') as tf:
  sampler=subprocess.Popen(['nvidia-smi','-i',UUID,'--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader,nounits','-lms','200'],stdout=tf)
  try:
   widths=[int(x) for x in sys.argv[1:]] or [256]
   for n in widths:
    for m,k in [(4352,5120),(5120,4352)]:
     for rep in range(3):
      guard();h=health();tag=f'{m}-{n}-{k}-{rep}'
      cmd=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES='+UUID,'/usr/bin/taskset','--cpu-list','0-11',str(R/'worker'),str(m),str(n),str(k),str(600908+rep)]
      (run/(tag+'.meta.json')).write_text(json.dumps(dict(command=cmd,before=h),indent=2))
      with (run/(tag+'.jsonl')).open('w') as out,(run/(tag+'.err')).open('w') as err:
       p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True);deadline=time.monotonic()+180
       while p.poll() is None:
        guard()
        if time.monotonic()>deadline:raise RuntimeError('Worker deadline; no retry/reset')
        try:p.wait(timeout=0.5)
        except subprocess.TimeoutExpired:pass
      if p.returncode:raise RuntimeError('Worker failed '+tag)
      print('PASS',tag,health().strip(),flush=True)
  finally:
   sampler.terminate();sampler.wait(timeout=5)
finally:
 if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 final=health();(run/'final-health.txt').write_text(final)
 append('FINAL RELEASE: '+TOKEN+'; all workers exited, '+final.strip()+'. No reset or production change. Artifacts '+str(run))
 print('RELEASE',run,flush=True)
