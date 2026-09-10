from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time

p=Path(__file__).resolve().parent
tag='priority1-throughput'
coords=[p.parents[1]/'bench/COORDINATION-20260908.md',
        Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/bench/COORDINATION-20260908.md')]
for c in coords:
    raw=c.read_text()
    assert 'W4A4-from-W16 research ACTIVE — GPU1 RESERVED' in raw,c

lock=open('/tmp/P100-arithmetic-gpu1.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
apps=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid',
                              '--format=csv,noheader'],text=True).strip()
if apps:raise RuntimeError('GPU1 has a compute client: '+apps)

def health():
    raw=subprocess.check_output(['nvidia-smi','-i','1',
        '--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total',
        '--format=csv,noheader,nounits'],text=True).strip()
    values=[x.strip() for x in raw.split(',')]
    if int(values[-1])!=0 or int(values[3])>80:raise RuntimeError('GPU health guard '+raw)
    return raw

def note(message):
    for c in coords:
        with c.open('a') as stream:stream.write('\n'+message+'\n')

manifest=json.loads((p/'microbench-manifest.json').read_text())
for name,want in manifest['hashes'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,name
initial=health();note('HELD: w4a4-from-w16-priority1-throughput; GPU1 reserved device lock; bounded arithmetic-throughput worker.')
meta={'tag':tag,'device':1,'initial_gpu':initial,'started':time.time(),
      'manifest':manifest,'runner_sha256':hashlib.sha256((p/'run_microbench.py').read_bytes()).hexdigest()}
cmd=['env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin',
     'LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_VISIBLE_DEVICES=1',
     'taskset','--cpu-list','0-11',str(p/'microbench'),'--gpu-approved','1','65536']
meta['command']=cmd;child=None
try:
    with (p/(tag+'.out')).open('x') as out,(p/(tag+'.err')).open('x') as err:
        child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
        deadline=time.monotonic()+120;meta['telemetry']=[]
        while child.poll() is None:
            if time.monotonic()>deadline:raise RuntimeError('deadline')
            xs=Path('/home/arian/.xsession-errors')
            if xs.exists() and xs.stat().st_size>50*1024**2:raise RuntimeError('desktop log guard')
            if os.statvfs(p).f_bavail*os.statvfs(p).f_frsize<5*1024**3:raise RuntimeError('disk guard')
            meta['telemetry'].append([time.time(),health()]);time.sleep(1)
        meta['exit_code']=child.returncode
        if child.returncode:raise RuntimeError('worker failed; no automatic retry')
finally:
    if child and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
    meta['finished']=time.time();meta['final_gpu']=health()
    (p/(tag+'.meta.json')).write_text(json.dumps(meta,indent=2)+'\n')
    note(f'FINAL RELEASE: w4a4-from-w16-priority1-throughput; GPU1 worker lock released, exit={meta.get("exit_code")}, reservation retained, no reset.')
    lock.close()
print('PASS',tag,'GPU1')
