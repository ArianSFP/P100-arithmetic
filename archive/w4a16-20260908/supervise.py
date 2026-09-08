#!/usr/bin/env python3
"""Fresh W4A16 workers on reserved GPU1; no CUDA in this controller."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
HELPER=ROOT.parent/'layout-lut-20260908/supervise.py'
spec=importlib.util.spec_from_file_location('health_helpers',HELPER)
health_helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(health_helpers)
sha,persist,health,check_health=health_helpers.sha,health_helpers.persist,health_helpers.health,health_helpers.check_health
COORD=health_helpers.COORD
GPU=health_helpers.GPU
TOKEN='w4a16-20260908-1227-gpu1'

def reservation(digest):
    text=COORD.read_text()
    if sha(COORD)!=digest or '- HELD: w4a16 GPU1 token='+TOKEN not in text.splitlines():raise RuntimeError('Reservation missing/changed')
    if '- FINAL RELEASE: w4a16 GPU1 token='+TOKEN in text:raise RuntimeError('Reservation released')

def main():
    p=argparse.ArgumentParser();p.add_argument('--coordination-sha256',required=True)
    p.add_argument('--timeout',type=int,default=90);p.add_argument('--sanitizer',choices=['memcheck','racecheck','synccheck'])
    p.add_argument('worker_args',nargs=argparse.REMAINDER);args=p.parse_args()
    if not 1<=args.timeout<=120:raise SystemExit('Timeout range 1..120')
    reservation(args.coordination_sha256)
    inv=json.loads((ROOT/'build/inventory.json').read_text())
    for file,digest in inv['hashes'].items():
        if sha(ROOT/file)!=digest:raise RuntimeError('Hash changed: '+file)
    tail=args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
    worker=[str(ROOT/'build/worker'),str(ROOT/'build/kernels.cubin'),*tail]
    hashes=dict(inv['hashes'])
    for i,v in enumerate(tail):
        if v=='--configs':hashes['config_file']=sha(Path(tail[i+1]))
    if args.sanitizer:worker=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',args.sanitizer,'--error-exitcode','9',*worker]
    argv=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin',
          'LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_DEVICE_ORDER=PCI_BUS_ID',
          'CUDA_VISIBLE_DEVICES='+GPU,'/usr/bin/taskset','--cpu-list','0-11',*worker]
    out=ROOT/'gpu-results';out.mkdir(exist_ok=True);run=out/str(time.time_ns());run.mkdir()
    record=dict(argv=argv,token=TOKEN,gpu_uuid=GPU,hashes=hashes,coordination_sha256=args.coordination_sha256,
                supervisor_sha256=sha(Path(__file__)),health_helper_sha256=sha(HELPER),sanitizer=args.sanitizer,
                hypothesis='Q4_0 FP16 inputs; no A8 quantization; FP32 arithmetic and exact-order controls',status='prelaunch')
    proc=None
    try:
        record['before']=health();check_health(record['before']);reservation(args.coordination_sha256)
        persist(run/'prelaunch.json',record);print('RUN',run,flush=True)
        with (run/'stdout.txt').open('x') as stdout,(run/'stderr.txt').open('x') as stderr:
            proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True)
            record['pid']=proc.pid
            try:record['returncode']=proc.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                os.killpg(proc.pid,signal.SIGKILL)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                raise RuntimeError('Worker timeout/interruption; STOP, no reset/retry')
        if record['returncode']:raise RuntimeError('Worker failed; STOP and inspect before further tests')
        record['after']=health();check_health(record['after'],require_idle=False);record['status']='PASS'
    except (RuntimeError,OSError,ValueError) as e:
        record['status']='STOP: '+str(e)
        if proc is not None and 'after' not in record:record['after']=health()
    persist(run/'result.json',record);print(record['status'],flush=True)
    for name in ['stdout.txt','stderr.txt']:
        if (run/name).exists():print((run/name).read_text()[-1800:])
    print('GPU1 reservation remains held; no reset/release performed.')
    return 0 if record['status']=='PASS' else 3
if __name__=='__main__':raise SystemExit(main())
