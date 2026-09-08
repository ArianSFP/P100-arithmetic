#!/usr/bin/env python3
"""GPU0-only guarded reservation. Controller never initializes CUDA."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
def main():
    if sys.argv[1:]!=['--gpu0-approved']:raise SystemExit('Explicit GPU0 authorization required')
    path=ROOT.parent/'w4a16-t64-r1-20260908/supervise.py'
    spec=importlib.util.spec_from_file_location('guard',path)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.ROOT=ROOT;guard.TOKEN='w4a16-long-prefill-20260908-gpu0'
    guard.UUID='GPU-bb126d7c-3d48-3911-bf1f-8b4ed9f2e706'
    def health():
        raw=subprocess.check_output(['/usr/bin/nvidia-smi','-i',guard.UUID,'--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        c=list(map(str.strip,raw.split(',')))
        if len(c)!=7 or c[0]!=guard.UUID or 'P100' not in c[1] or int(c[2])>96 or int(c[6]):raise RuntimeError('GPU0 health: '+raw)
        apps=subprocess.check_output(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=5)
        for line in apps.splitlines():
            fields=list(map(str.strip,line.split(',')))
            if fields[0]==guard.UUID and fields!=[guard.UUID,'1121899','/usr/bin/gnome-text-editor']:raise RuntimeError('GPU0 unexpected client: '+line)
        return dict(gpu=raw,apps=apps,existing_editor_retained=True)
    guard.health=health
    # Wait for a real handoff, never bypass another session's global lock.
    original_flock=guard.fcntl.flock
    def queued_flock(fd,operation):
        end=time.monotonic()+600;notice=0
        while True:
            try:return original_flock(fd,operation)
            except BlockingIOError:
                guard.storage()
                if time.monotonic()>end:raise RuntimeError('Reservation wait expired; no GPU acquired')
                if time.monotonic()>notice:
                    print('WAITING for shared benchmark lock; GPU0 not acquired',flush=True)
                    notice=time.monotonic()+30
                time.sleep(0.5)
    guard.fcntl.flock=queued_flock
    guard.main()
if __name__=='__main__':main()
