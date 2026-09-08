#!/usr/bin/env python3
"""Single-GPU2 fresh-process supervisor. No CUDA initialization or GPU reset."""
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
TOKEN='w4a16-r3-20260908-1459-gpu2'
UUID='GPU-2aa85c85-bc04-0ac3-fc1b-4827d8303d81'
DESKTOP=Path('/home/arian/.xsession-errors')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def persist(path,record):
    with path.open('x') as stream:
        json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def command(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=5)
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def health():
    return dict(device=command(['/usr/bin/nvidia-smi','-i',UUID,
        '--query-gpu=timestamp,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits']),
        processes=command(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']))
def check_health(record,idle=True):
    if record['device']['returncode'] or record['processes']['returncode']:raise RuntimeError('Health query failed')
    c=[x.strip() for x in record['device']['stdout'].strip().split(',')]
    if len(c)!=9 or c[1]!=UUID or 'P100' not in c[2] or int(c[4])>64 or (idle and int(c[5])) or int(c[8]):raise RuntimeError('GPU2 unhealthy/busy')
    if any(line.startswith(UUID) for line in record['processes']['stdout'].splitlines()):raise RuntimeError('GPU2 has a compute process')
def reservation(local_sha,shared_sha):
    for path,digest in ((COORD,local_sha),(SHARED,shared_sha)):
        text=path.read_text()
        if sha(path)!=digest or '- HELD: w4a16-r3 GPU2 token='+TOKEN not in text.splitlines():raise RuntimeError('Reservation missing/changed: '+str(path))
        if '- FINAL RELEASE: w4a16-r3 GPU2 token='+TOKEN in text:raise RuntimeError('Reservation released')
def storage_guard():
    if DESKTOP.stat().st_size>100_000_000 or shutil.disk_usage(ROOT).free<2_000_000_000:raise RuntimeError('Desktop-log/disk guard; no log truncation performed')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordination-sha256',required=True);parser.add_argument('--shared-sha256',required=True)
    parser.add_argument('--sanitizer',choices=['memcheck','synccheck','racecheck'])
    parser.add_argument('--timeout',type=int,default=120);parser.add_argument('--health-only',action='store_true')
    parser.add_argument('tail',nargs=argparse.REMAINDER);args=parser.parse_args()
    if not 1<=args.timeout<=120:raise RuntimeError('Timeout must be 1..120 seconds')
    reservation(args.coordination_sha256,args.shared_sha256);storage_guard()
    inventory=json.loads((ROOT/'build/inventory.json').read_text())
    for name,digest in inventory['hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Build/control hash changed: '+name)
    tail=args.tail[1:] if args.tail[:1]==['--'] else args.tail
    if not args.health_only and (not tail or tail[0] not in ('validate','smoke','sweep','bench')):raise RuntimeError('Reviewed worker mode required')
    result_dir=ROOT/'gpu-results'/str(time.time_ns());result_dir.mkdir(parents=True)
    hashes=dict(inventory['hashes'])
    for i,value in enumerate(tail):
        if value=='--configs':
            path=Path(tail[i+1]).resolve()
            if not path.is_relative_to(ROOT):raise RuntimeError('Config outside study')
            hashes['config_file']=sha(path)
    if args.health_only:
        h=health();check_health(h)
        persist(result_dir/'final-health.json',dict(status='PASS',gpu=2,token=TOKEN,reservation_held=True,health=h,
            desktop_bytes=DESKTOP.stat().st_size,free_bytes=shutil.disk_usage(ROOT).free))
        print('FINAL_HEALTH_PASS',result_dir);return 0
    worker=[str(ROOT/'build/worker'),str(ROOT/'build/kernels.cubin'),
        str(REPO/'archive/w4a16-r2-20260908/build/kernels.cubin'),str(REPO/'archive/w4a16-20260908/build/kernels.cubin'),*tail]
    if args.sanitizer:worker=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',args.sanitizer,'--error-exitcode','9',*worker]
    argv=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64',
        'CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES='+UUID,'/usr/bin/taskset','--cpu-list','0-11',*worker]
    rec=dict(status='prelaunch',gpu=2,gpu_uuid=UUID,token=TOKEN,argv=argv,hashes=hashes,
        supervisor_sha256=sha(Path(__file__)),coordination_sha256=args.coordination_sha256,shared_sha256=args.shared_sha256,
        sanitizer=args.sanitizer,hypothesis='Exact-order W4A16: paired sharing, CTA sizing, local reduction; live sealed R2 controls')
    proc=None
    try:
        rec['before']=health();check_health(rec['before']);reservation(args.coordination_sha256,args.shared_sha256)
        persist(result_dir/'prelaunch.json',rec);print('RUN',result_dir,flush=True)
        with (result_dir/'stdout.txt').open('x') as stdout,(result_dir/'stderr.txt').open('x') as stderr:
            proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True)
            rec['pid']=proc.pid;start=time.monotonic()
            while proc.poll() is None:
                if time.monotonic()-start>args.timeout:raise RuntimeError('Timeout: STOP, no reset/retry')
                storage_guard()
                try:proc.wait(timeout=1)
                except subprocess.TimeoutExpired:pass
            rec['returncode']=proc.returncode
        if proc.returncode:raise RuntimeError('Worker failed: STOP and inspect')
        rec['after']=health();check_health(rec['after'],idle=False);rec['status']='PASS'
    except (RuntimeError,OSError,subprocess.TimeoutExpired,KeyboardInterrupt) as error:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        rec['status']='STOP: '+str(error)
    persist(result_dir/'result.json',rec);print(rec['status'],flush=True)
    for name in ('stdout.txt','stderr.txt'):
        path=result_dir/name
        if path.exists():print(path.read_text()[-1000:])
    print('GPU2 remains reserved; no reset or release performed.')
    return 0 if rec['status']=='PASS' else 3
if __name__=='__main__':raise SystemExit(main())
