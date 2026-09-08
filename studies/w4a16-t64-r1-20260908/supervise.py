#!/usr/bin/env python3
"""Hold the shared GPU lock across fresh workers and compiler gaps; no CUDA."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
UUID='GPU-2aa85c85-bc04-0ac3-fc1b-4827d8303d81'
TOKEN='w4a16-t64-r1-20260908-gpu2'
COORDS=[ROOT.parents[1]/'bench/COORDINATION-20260908.md',
        Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/bench/COORDINATION-20260908.md')]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def persist(p,x):
    with p.open('x') as f:
        json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def health():
    cmd=['/usr/bin/nvidia-smi','-i',UUID,'--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits']
    raw=subprocess.check_output(cmd,text=True,timeout=5).strip()
    apps=subprocess.check_output(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=5)
    c=[s.strip() for s in raw.split(',')]
    if len(c)!=7 or c[0]!=UUID or 'P100' not in c[1] or int(c[2])>64 or int(c[6]):raise RuntimeError('GPU health failed: '+raw)
    if any(line.startswith(UUID) for line in apps.splitlines()):raise RuntimeError('GPU2 has another compute process')
    return dict(gpu=raw,apps=apps)
def storage():
    x=Path('/home/arian/.xsession-errors')
    if x.exists() and x.stat().st_size>50*1024**2:raise RuntimeError('Desktop log guard; no truncation')
    if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk guard')
def reservation():
    for p in COORDS:
        s=p.read_text()
        if 'HELD: '+TOKEN not in s or 'FINAL RELEASE: '+TOKEN in s:raise RuntimeError('Reservation not held: '+str(p))
def main():
    (ROOT/'gpu-results').mkdir(exist_ok=True)
    lock=open('/tmp/affinitywave-4gpu.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    storage();reservation();h=health()
    hold=ROOT/'gpu-results'/('hold-'+str(time.time_ns())+'.json')
    persist(hold,dict(token=TOKEN,health=h,pid=os.getpid(),started=time.time()))
    print('HELD '+TOKEN+'; send JSON {"tag":...,"args":[...]} or release',flush=True)
    deadline=time.monotonic()+3600
    stopped=False
    try:
        while time.monotonic()<deadline:
            storage()
            if not select.select([sys.stdin],[],[],1)[0]:continue
            line=sys.stdin.readline()
            if not line or line.strip()=='release':break
            req=json.loads(line)
            if stopped:print('STOPPED; only release is allowed',flush=True);continue
            tag=req['tag']
            if not tag.replace('-','').replace('_','').isalnum():raise RuntimeError('invalid tag')
            outdir=ROOT/'gpu-results'/tag;outdir.mkdir()
            manifest=json.loads((ROOT/'build/manifest.json').read_text())
            for name,expected in manifest['hashes'].items():
                if digest(ROOT/name)!=expected:raise RuntimeError('Changed build: '+name)
            args=req.get('args',[])
            if not all(isinstance(x,str) for x in args):raise RuntimeError('args must be strings')
            cmd=[str(ROOT/'build/worker')]+args
            tool=req.get('sanitizer')
            if tool:
                if tool not in ['memcheck','racecheck','synccheck']:raise RuntimeError('unknown sanitizer')
                cmd=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99']+cmd
            cmd=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin',
                 'LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_VISIBLE_DEVICES='+UUID,
                 'CUDA_DEVICE_ORDER=PCI_BUS_ID','/usr/bin/taskset','--cpu-list','0-11']+cmd
            reservation();before=health();storage()
            meta=dict(token=TOKEN,command=cmd,request=req,build=manifest,before=before,
                      supervisor_sha256=digest(Path(__file__)),coordination={str(p):digest(p) for p in COORDS},started=time.time())
            persist(outdir/'prelaunch.json',meta)
            proc=None
            try:
                with (outdir/'stdout.txt').open('x') as fo,(outdir/'stderr.txt').open('x') as fe:
                    proc=subprocess.Popen(cmd,stdout=fo,stderr=fe,start_new_session=True)
                    end=time.monotonic()+180
                    while proc.poll() is None:
                        storage()
                        if time.monotonic()>end:raise RuntimeError('Worker timeout; STOP, no reset/retry')
                        try:proc.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:pass
                meta['returncode']=proc.returncode
                meta['after']=health()
                if proc.returncode:raise RuntimeError('Worker failure; STOP and inspect')
                meta['status']='PASS'
            except Exception as exc:
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                meta['status']='STOP: '+str(exc);stopped=True
            meta['finished']=time.time();persist(outdir/'result.json',meta)
            print(tag+' '+meta['status']+' '+str(outdir),flush=True)
            print((outdir/'stdout.txt').read_text()[-1200:],flush=True)
            print((outdir/'stderr.txt').read_text()[-1200:],flush=True)
        persist(ROOT/'gpu-results'/('release-'+str(time.time_ns())+'.json'),dict(token=TOKEN,health=health(),ended=time.time(),stopped=stopped))
    finally:
        lock.close()
        print('LOCK_RELEASED; append final coordination release after reviewing health',flush=True)

if __name__=='__main__': main()
