from pathlib import Path
import argparse,fcntl,subprocess,time,hashlib,json,os,signal
p=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('--tokens',type=int,required=True);ap.add_argument('--projection',choices=['gu','down'],default='down');ap.add_argument('--memcheck',action='store_true');ap.add_argument('--racecheck',action='store_true');ap.add_argument('--ctas',type=int,choices=range(1,7),default=2);a=ap.parse_args()
coords=[p.parents[1]/'bench/COORDINATION-20260908.md',Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/bench/COORDINATION-20260908.md')]
lock=None
for device in [2,1,3,0]:
 f=open(f'/tmp/P100-arithmetic-gpu{device}.lock','a')
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:f.close();continue
 apps=subprocess.check_output(['nvidia-smi','-i',str(device),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if apps:f.close();continue
 lock=f;break
if lock is None:raise RuntimeError('No unreserved idle GPU available')
def health():
 raw=subprocess.check_output(['nvidia-smi','-i',str(device),'--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();v=raw.split(',')
 if int(v[-1])!=0 or int(v[3])>80:raise RuntimeError('GPU health guard '+raw)
 return raw
def note(s):
 for c in coords:
  with c.open('a') as x:x.write('\n'+s+'\n')
initial=health();fulltag='w16-dualrail-f32-'+a.tag
note(f'HELD: {fulltag}; GPU{device} device lock, matched F32-wire W8/W16 dual-rail benchmark; bounded watchdog worker.')
manifest=json.loads((p/'manifest.json').read_text())
for name,h in manifest['hashes'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
meta={'args':vars(a),'device':device,'initial_gpu':initial,'started':time.time(),'build':manifest,'runner_sha256':hashlib.sha256((p/'run.py').read_bytes()).hexdigest()}
cmd=['env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_VISIBLE_DEVICES='+str(device),'taskset','--cpu-list','0-11']
if a.memcheck or a.racecheck:cmd+=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','racecheck' if a.racecheck else 'memcheck','--error-exitcode','99']
cmd += [str(p/'worker'),'--gpu-approved','1',str(a.tokens),'512' if a.projection=='gu' else '2048','2048' if a.projection=='gu' else '512','2' if (a.memcheck or a.racecheck) else '64',str(a.ctas)]
meta['command']=cmd;child=None
try:
 with (p/(a.tag+'.out')).open('x') as out,(p/(a.tag+'.err')).open('x') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True);deadline=time.monotonic()+240;meta['telemetry']=[]
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
 meta['finished']=time.time();meta['final_gpu']=health();(p/(a.tag+'.meta.json')).write_text(json.dumps(meta,indent=2)+'\n');note(f'FINAL RELEASE: {fulltag}; GPU{device}, exit={meta.get("exit_code")}, no reset.');lock.close()
print('PASS',a.tag,'GPU',device)
