#!/usr/bin/env python3
"""GPU3 only; frozen safety controller, queued shared lock, fresh workers."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
UUID='GPU-4868830a-c1cf-90bd-8018-2360c55293b8'
TOKEN='w4a4-long-prefill-20260908-gpu3-r1'
def main():
    if sys.argv[1:]!=['--gpu3-approved']:raise SystemExit('Explicit GPU3 approval required')
    path=ROOT.parents[1]/'w4a16-t64-r1-20260908/supervise.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='35b6d6e4c4a1611ddcdb9e1db92a831a88ae5d133828bbf269b084a08911e02c'
    spec=importlib.util.spec_from_file_location('frozen_guard',path)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.ROOT=ROOT;guard.UUID=UUID;guard.TOKEN=TOKEN
    def health():
        raw=subprocess.check_output(['/usr/bin/nvidia-smi','-i',UUID,'--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        c=[x.strip() for x in raw.split(',')]
        if len(c)!=7 or c[0]!=UUID or 'P100' not in c[1] or int(c[2])>64 or int(c[6]):raise RuntimeError('GPU3 health failed: '+raw)
        apps=subprocess.check_output(['/usr/bin/nvidia-smi','-i',UUID,'--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=5)
        if any(line.startswith(UUID) for line in apps.splitlines()):raise RuntimeError('GPU3 has another compute process: '+apps)
        return dict(gpu=raw,apps=apps)
    guard.health=health
    original=guard.fcntl.flock
    def acquire(fd,operation):
        end=time.monotonic()+600;notice=0
        while True:
            try:return original(fd,operation)
            except BlockingIOError:
                guard.storage();guard.reservation()
                if time.monotonic()>end:raise RuntimeError('Wait expired; no GPU3 acquisition')
                if time.monotonic()>notice:
                    print('WAITING shared lock; GPU3 NOT acquired; no compiler/CUDA',flush=True);notice=time.monotonic()+30
                time.sleep(.5)
    guard.fcntl.flock=acquire
    guard.main()
if __name__=='__main__':main()
