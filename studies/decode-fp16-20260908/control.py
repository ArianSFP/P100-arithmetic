#!/usr/bin/env python3
"""Hold GPU1's rig lock across the remaining bounded screen/build/confirm steps."""
import fcntl,hashlib,json,os,select,shutil,signal,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
UUID='GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b'
TOKEN='decode-fp16-'+str(time.time_ns())+'-gpu1'
COORD=[R.parents[1]/'bench/COORDINATION-20260908.md']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def append(s):
 for p in COORD:
  with p.open('a') as f:f.write('\n'+s+'\n')
def health():
 raw=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=uuid,memory.used,utilization.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True,timeout=5)
 c=[x.strip() for x in raw.split(',')]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=5)
 if c[0]!=UUID or int(c[1])>64 or int(c[4]) or UUID in apps:raise RuntimeError('GPU1 busy/unhealthy: '+raw+apps)
 return raw

def guard():
 if Path('/home/arian/.xsession-errors').stat().st_size>100_000_000 or shutil.disk_usage(R).free<3_000_000_000:raise RuntimeError('Storage/desktop guard')
lock=open('/tmp/affinitywave-4gpu.lock','a');deadline=time.monotonic()+900
while True:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:
  if time.monotonic()>deadline:raise RuntimeError('Lock wait expired; no CUDA execution')
  time.sleep(1)
guard();h=health();append('HELD: '+TOKEN+'; GPU1-only decode FP16 study, shared lock held through bounded tests/build gaps. GPU0/2/3 untouched. No production edits.')
run=R/('run-'+str(time.time_ns()));run.mkdir();save(run/'hold.json',dict(token=TOKEN,before=h,pid=os.getpid(),controller_sha256=sha(Path(__file__))))
p=None;sampler=None;stopped=False
try:
 with (run/'telemetry.csv').open('w') as tf:
  sampler=subprocess.Popen(['nvidia-smi','-i',UUID,'--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader,nounits','-lms','200'],stdout=tf)
  print('HELD',run,flush=True);end=time.monotonic()+1800
  while time.monotonic()<end:
   guard()
   if not select.select([sys.stdin],[],[],1)[0]:continue
   line=sys.stdin.readline()
   if not line or line.strip()=='release':break
   if stopped:print('STOPPED; release only',flush=True);continue
   req=json.loads(line);tag=req['tag'];name=req['binary']
   if not tag.replace('-','').isalnum() or name not in ['worker']:raise RuntimeError('Unknown worker/tag')
   d=run/tag;d.mkdir();binary=R/name;source=R/(name+'.cu')
   cmd=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES='+UUID,'/usr/bin/taskset','--cpu-list','0-11']
   if req.get('memcheck'):cmd+=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','9']
   cmd+=[str(binary)]+req['args']
   before=health();guard();meta=dict(command=cmd,before=before,binary_sha256=sha(binary),source_sha256=sha(source),other_sources={n:sha(R/n) for n in ['kernels.cuh','r3-baseline.cuh','common.hpp']},token=TOKEN)
   save(d/'prelaunch.json',meta)
   try:
    with (d/'stdout.jsonl').open('w') as out,(d/'stderr.txt').open('w') as err:
     p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True);deadline=time.monotonic()+240
     while p.poll() is None:
      guard()
      if time.monotonic()>deadline:raise RuntimeError('Worker deadline; no retry/reset')
      try:p.wait(timeout=.5)
      except subprocess.TimeoutExpired:pass
    if p.returncode:raise RuntimeError('Worker failed: '+str(p.returncode))
    meta.update(status='PASS',returncode=p.returncode,after=health())
   except Exception as e:
    if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
    meta.update(status='STOP',error=str(e));stopped=True
   save(d/'result.json',meta);print(meta['status'],tag,flush=True)
finally:
 if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 if sampler is not None:sampler.terminate();sampler.wait(timeout=5)
 h=health();(run/'final-health.txt').write_text(h)
 append('FINAL RELEASE: '+TOKEN+'; all workers exited, '+h.strip()+'. No reset/production change. Artifacts '+str(run))
 print('RELEASE',run,flush=True)
