from pathlib import Path
import argparse,fcntl,subprocess,time,hashlib,json,os,signal
p=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('--tokens',type=int,choices=[33,64,128,256,512],required=True);ap.add_argument('--projection',choices=['gu','down'],default='down');ap.add_argument('--memcheck',action='store_true');ap.add_argument('--ctas',type=int,choices=[1,2,3,4,5,6],default=1);ap.add_argument('--racecheck',action='store_true');a=ap.parse_args()
coords=[p.parents[2]/'bench/COORDINATION-20260908.md',Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/bench/COORDINATION-20260908.md')]
f=None
for device in [2,1,3,0]:
 lock=open(f'/tmp/P100-arithmetic-gpu{device}.lock','a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();continue
 apps=subprocess.check_output(['nvidia-smi','-i',str(device),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if apps:lock.close();continue
 f=lock;break
if f is None:raise RuntimeError('No unreserved idle GPU available; no worker launched')
def health():
 raw=subprocess.check_output(['nvidia-smi','-i',str(device),'--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();v=raw.split(',')
 if int(v[-1])!=0 or int(v[3])>80:raise RuntimeError('GPU health guard '+raw)
 return raw
def note(s):
 for c in coords:
  with c.open('a') as x:x.write('\n'+s+'\n')
initial=health();tag='w8-affinity-m128-pairs-dual-'+a.tag
note(f'HELD: {tag}; GPU{device} device lock only per user instruction, W8A16 packed-half partial accumulation candidates versus original Q8; bounded watchdog worker.')
meta={'args':vars(a),'device':device,'initial_gpu':initial,'started':time.time(),'build':json.loads((p/'manifest.json').read_text()),'runner_sha256':hashlib.sha256((p/'run.py').read_bytes()).hexdigest()}
for name,h in meta['build']['hashes'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
cmd=['env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_VISIBLE_DEVICES='+str(device),'taskset','--cpu-list','0-11']
if a.memcheck or a.racecheck:cmd+=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','racecheck' if a.racecheck else 'memcheck','--error-exitcode','99']
cmd += [str(p/'worker'),'--gpu-approved','1',str(a.tokens),'128' if a.racecheck else '512' if a.projection=='gu' else '2048','96' if a.racecheck else '2048' if a.projection=='gu' else '512','2' if a.memcheck else '64',str(a.ctas)]
meta['command']=cmd;child=None
try:
 with (p/(a.tag+'.out')).open('x') as out,(p/(a.tag+'.err')).open('x') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True);deadline=time.monotonic()+180;meta['telemetry']=[]
  while child.poll() is None:
   if time.monotonic()>deadline:raise RuntimeError('Deadline')
   xs=Path('/home/arian/.xsession-errors')
   if xs.exists() and xs.stat().st_size>50*1024**2:raise RuntimeError('Desktop log guard')
   if os.statvfs(p).f_bavail*os.statvfs(p).f_frsize<5*1024**3:raise RuntimeError('Disk guard')
   meta['telemetry'].append([time.time(),health()]);time.sleep(2)
  meta['exit_code']=child.returncode
  if child.returncode:raise RuntimeError('Worker failed; no automatic retry')
finally:
 if child and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
 meta['finished']=time.time();meta['final_gpu']=health();(p/(a.tag+'.meta.json')).write_text(json.dumps(meta,indent=2)+'\n');note(f'FINAL RELEASE: {tag}; GPU{device}, exit={meta.get("exit_code")}, no reset.');f.close()
print('PASS',a.tag,'GPU',device)
