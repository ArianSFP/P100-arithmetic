#!/usr/bin/env python3
"""Fixed single-GPU workers, persistent provenance, no CUDA in supervisor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpu-results'
COORD=Path('/home/arian/llama.cpp-qwen38-p100/bench/COORDINATION-20260908.md')
GPU='GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b'
TOKEN='lowbit-20260908-1200-gpu1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def persist(p,obj):
    with p.open('x') as f:
        json.dump(obj,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=5)
        return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired: return dict(returncode=None,status='timeout_STOP')
def health():
    return dict(device=command(['/usr/bin/nvidia-smi','-i',GPU,
        '--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total',
        '--format=csv,noheader,nounits']),
        processes=command(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']))
def check_health(h,require_idle=True):
    if h['device']['returncode']!=0 or h['processes']['returncode']!=0: raise RuntimeError('Health query failed')
    cols=list(map(str.strip,h['device']['stdout'].strip().split(',')))
    if len(cols)!=8 or cols[0]!=GPU or 'P100' not in cols[1] or int(cols[3])>64 or (require_idle and int(cols[4])!=0) or int(cols[7])!=0:
        raise RuntimeError('Target device not idle/healthy')
    if any(line.startswith(GPU) for line in h['processes']['stdout'].splitlines()): raise RuntimeError('Target has a compute process')
def reservation(digest):
    txt=COORD.read_text()
    marker='- HELD: lowbit GPU1 token='+TOKEN
    if sha(COORD)!=digest or marker not in txt.splitlines(): raise RuntimeError('Reservation snapshot changed/missing')
    if '- FINAL RELEASE: lowbit GPU1 token='+TOKEN in txt: raise RuntimeError('Reservation released')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--coordination-sha256',required=True)
    p.add_argument('--kind',choices=['endpoint','layout'],required=True)
    p.add_argument('--sanitizer',choices=['memcheck','racecheck','synccheck'])
    p.add_argument('--timeout',type=int,default=55)
    p.add_argument('worker_args',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if not 1<=args.timeout<=120: raise SystemExit('Timeout must be 1..120 seconds')
    reservation(args.coordination_sha256)
    OUT.mkdir(exist_ok=True)
    run=OUT/(args.kind+'-'+str(time.time_ns())); run.mkdir()
    if args.kind=='endpoint':
        parent=ROOT.parent/'endpoint-sad-20260908'
        inv=json.loads((parent/'results/inventory.json').read_text())
        worker=parent/'endpoint-sad'
        if sha(worker)!=inv['binary_sha256'] or sha(parent/'endpoint_sad.cu')!=inv['source_sha256']: raise RuntimeError('Endpoint hashes changed')
        worker_argv=[str(worker),'--gpu']
        hashes={'worker':sha(worker),'source':inv['source_sha256']}
    else:
        inv=json.loads((ROOT/'results/inventory.json').read_text())
        for name,digest in inv['hashes'].items():
            f=ROOT/('results/'+name if name.endswith('.cubin') else name)
            if sha(f)!=digest: raise RuntimeError('Offline source/cubin changed: '+name)
        build=json.loads((ROOT/'gpu-build.json').read_text())
        worker=ROOT/'gpu-worker'
        if sha(worker)!=build['worker_sha256'] or sha(ROOT/'gpu_worker.cpp')!=build['source_sha256']: raise RuntimeError('Worker hashes changed')
        tail=args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
        if sha(ROOT/'gpu-baseline.cubin')!=build['baseline_cubin_sha256']:raise RuntimeError('Baseline cubin changed')
        worker_argv=[str(worker),str(ROOT/'results/kernels.cubin'),*tail,'--baseline',str(ROOT/'gpu-baseline.cubin')]
        hashes={**inv['hashes'],'worker':sha(worker),'worker_source':build['source_sha256']}
        hashes['baseline_cubin']=build['baseline_cubin_sha256'];hashes['baseline_source']=build['baseline_source_sha256']
        for i,value in enumerate(tail):
            if value=='--configs' and i+1<len(tail): hashes['config_file']=sha(Path(tail[i+1]))
    if args.sanitizer:
        worker_argv=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',args.sanitizer,'--error-exitcode','9',*worker_argv]
    argv=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin',
          'LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_DEVICE_ORDER=PCI_BUS_ID',
          'CUDA_VISIBLE_DEVICES='+GPU,'/usr/bin/taskset','--cpu-list','0-11',*worker_argv]
    record=dict(argv=argv,gpu_uuid=GPU,token=TOKEN,coordination_sha256=args.coordination_sha256,
                hashes=hashes,supervisor_sha256=sha(Path(__file__)),
                hypothesis='Exact low-bit formulations; compare complete synthetic G32 pipelines',
                status='prelaunch',sanitizer=args.sanitizer)
    proc=None
    try:
        record['before']=health(); check_health(record['before']); reservation(args.coordination_sha256)
        persist(run/'prelaunch.json',record)
        print('RUN',run,flush=True)
        with (run/'stdout.txt').open('x') as out,(run/'stderr.txt').open('x') as err:
            proc=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True)
            record['pid']=proc.pid
            try: record['returncode']=proc.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                proc.kill()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: pass
                raise RuntimeError('Worker timeout/interrupted; STOP, no reset or retry')
        if record['returncode']!=0: raise RuntimeError('Worker failed; inspect logs before further tests')
        # Utilization describes the preceding sampling interval, including our
        # just-finished worker. Still require no process, low allocation, ECC 0.
        record['after']=health(); check_health(record['after'],require_idle=False)
        record['status']='PASS'
    except (RuntimeError,OSError,ValueError) as exc:
        record['status']='STOP: '+str(exc)
        if proc is not None and 'after' not in record: record['after']=health()
    persist(run/'result.json',record)
    print(record['status'],flush=True)
    if (run/'stdout.txt').exists(): print((run/'stdout.txt').read_text()[-1800:])
    if (run/'stderr.txt').exists(): print((run/'stderr.txt').read_text()[-1800:])
    print('Reservation remains held; no automatic release or reset.')
    return 0 if record['status']=='PASS' else 3
if __name__=='__main__': raise SystemExit(main())
