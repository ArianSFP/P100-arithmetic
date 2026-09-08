#!/usr/bin/env python3
"""One fresh worker at a time on a reviewed, reserved P100; no CUDA here."""
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
COORD=Path('/home/arian/llama.cpp-qwen38-p100/bench/COORDINATION-20260908.md')
TOKEN='w4a16-r2-20260908-1258-three'
GPUS={1:'GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b',2:'GPU-2aa85c85-bc04-0ac3-fc1b-4827d8303d81',3:'GPU-4868830a-c1cf-90bd-8018-2360c55293b8'}
DESKTOP=Path('/home/arian/.xsession-errors')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def persist(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def command(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=5)
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def health(gpu):
    return dict(device=command(['/usr/bin/nvidia-smi','-i',GPUS[gpu],
        '--query-gpu=timestamp,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits']),
        processes=command(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']))
def check_health(h,gpu,idle=True):
    if h['device']['returncode'] or h['processes']['returncode']:raise RuntimeError('Health query failed')
    c=[x.strip() for x in h['device']['stdout'].strip().split(',')]
    if len(c)!=9 or c[1]!=GPUS[gpu] or 'P100' not in c[2] or int(c[4])>64 or (idle and int(c[5])) or int(c[8]):raise RuntimeError('Device unhealthy/busy')
    if any(line.startswith(uuid) for uuid in GPUS.values() for line in h['processes']['stdout'].splitlines()):raise RuntimeError('Reserved GPU has a compute process')
def reservation(digest):
    t=COORD.read_text()
    if sha(COORD)!=digest or '- HELD: w4a16-r2 GPU1/2/3 token='+TOKEN not in t.splitlines():raise RuntimeError('Reservation missing/changed')
    if '- FINAL RELEASE: w4a16-r2 GPU1/2/3 token='+TOKEN in t:raise RuntimeError('Reservation released')
def storage_guard():
    if DESKTOP.stat().st_size>100_000_000 or shutil.disk_usage(ROOT).free<2_000_000_000:raise RuntimeError('Desktop-log/disk guard: STOP; user logs are not truncated')
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=GPUS,default=1);p.add_argument('--coordination-sha256',required=True)
    p.add_argument('--sanitizer',choices=['memcheck','synccheck','racecheck']);p.add_argument('--timeout',type=int,default=120)
    p.add_argument('--health-only',action='store_true');p.add_argument('tail',nargs=argparse.REMAINDER);a=p.parse_args()
    if not 1<=a.timeout<=120:raise SystemExit('Timeout must be 1..120')
    reservation(a.coordination_sha256);storage_guard()
    inv=json.loads((ROOT/'build/inventory.json').read_text())
    for f,h in inv['hashes'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Build/input hash changed: '+f)
    out=ROOT/'gpu-results';out.mkdir(exist_ok=True);run=out/str(time.time_ns());run.mkdir()
    if a.health_only:
        records={}
        for gpu in GPUS:records[gpu]=health(gpu);check_health(records[gpu],gpu)
        persist(run/'final-health.json',dict(status='PASS',token=TOKEN,reservation_held=True,devices=records))
        print('FINAL_HEALTH_PASS',run);return 0
    tail=a.tail[1:] if a.tail[:1]==['--'] else a.tail
    if not tail:raise RuntimeError('Worker mode required')
    hashes=dict(inv['hashes'])
    for i,v in enumerate(tail):
        if v=='--configs':hashes['config_file']=sha(Path(tail[i+1]))
    worker=[str(ROOT/'build/worker'),str(ROOT/'build/kernels.cubin'),*tail]
    if a.sanitizer:worker=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',a.sanitizer,'--error-exitcode','9',*worker]
    argv=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64',
          'CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES='+GPUS[a.gpu],'/usr/bin/taskset','--cpu-list','0-11',*worker]
    r=dict(argv=argv,gpu=a.gpu,gpu_uuid=GPUS[a.gpu],token=TOKEN,hashes=hashes,supervisor_sha256=sha(Path(__file__)),
           coordination_sha256=a.coordination_sha256,sanitizer=a.sanitizer,status='prelaunch',hypothesis='Faster exact-order Q4_0 x FP16; FP32 accumulation; sealed live control')
    proc=None
    try:
        r['before']=health(a.gpu);check_health(r['before'],a.gpu);reservation(a.coordination_sha256)
        persist(run/'prelaunch.json',r);print('RUN',run,flush=True)
        with (run/'stdout.txt').open('x') as stdout,(run/'stderr.txt').open('x') as stderr:
            proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True);r['pid']=proc.pid;start=time.monotonic()
            while proc.poll() is None:
                if time.monotonic()-start>a.timeout:raise RuntimeError('Worker timeout: STOP, no reset/retry')
                storage_guard()
                try:proc.wait(timeout=1)
                except subprocess.TimeoutExpired:pass
            r['returncode']=proc.returncode
        if proc.returncode:raise RuntimeError('Worker failed: STOP and inspect')
        r['after']=health(a.gpu);check_health(r['after'],a.gpu,False);r['status']='PASS'
    except (RuntimeError,OSError,subprocess.TimeoutExpired,KeyboardInterrupt) as e:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        r['status']='STOP: '+str(e)
    persist(run/'result.json',r);print(r['status'],flush=True)
    for f in ['stdout.txt','stderr.txt']:
        if (run/f).exists():print((run/f).read_text()[-1200:])
    print('GPUs1/2/3 remain reserved; no reset or release performed.')
    return 0 if r['status']=='PASS' else 3
if __name__=='__main__':raise SystemExit(main())
