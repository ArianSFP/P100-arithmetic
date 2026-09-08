#!/usr/bin/env python3
"""Dormant until user release AND a fresh coordination claim. Never auto-run."""
import importlib.util
from pathlib import Path
import sys
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parent

def main():
    if sys.argv[1:]!=['--user-released-gpus']:
        raise SystemExit('Disabled: wait for user GPU release and a fresh coordinated claim.')
    path=ROOT.parent/'w4a16-t64-r1-20260908/supervise.py'
    spec=importlib.util.spec_from_file_location('prior_supervisor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT
    module.TOKEN='w4a16-delivery-leads-20260908-gpu01'
    uuids={0:'GPU-bb126d7c-3d48-3911-bf1f-8b4ed9f2e706',1:'GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b'}
    module.UUID=uuids[1]
    original_stdin=sys.stdin
    class SelectGPU:
        def fileno(self):return original_stdin.fileno()
        def readline(self):
            line=original_stdin.readline()
            if line and line.strip()!='release':
                req=json.loads(line);gpu=req.get('gpu',1)
                if type(gpu)!=int or gpu not in uuids:raise RuntimeError('Only reserved GPU0/GPU1 allowed')
                module.UUID=uuids[gpu]
            return line
    sampler=None;stream=None
    def health():
        nonlocal sampler,stream
        raw=subprocess.check_output(['/usr/bin/nvidia-smi','-i',','.join(uuids.values()),
            '--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total',
            '--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        rows=[list(map(str.strip,line.split(','))) for line in raw.splitlines()]
        if len(rows)!=2 or {r[0] for r in rows}!=set(uuids.values()):raise RuntimeError('Reserved device identity changed')
        for r in rows:
            if len(r)!=7 or 'P100' not in r[1] or int(r[2])>96 or int(r[6]):raise RuntimeError('Reserved GPU health failed: '+str(r))
        apps=subprocess.check_output(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=5)
        for line in apps.splitlines():
            fields=list(map(str.strip,line.split(',')))
            if fields[0] in uuids.values() and fields!=[uuids[0],'1121899','/usr/bin/gnome-text-editor']:
                raise RuntimeError('Unexpected client on reserved GPU: '+line)
        if sampler is None:
            stream=(ROOT/'gpu-results'/('telemetry-'+str(time.time_ns())+'.csv')).open('x')
            sampler=subprocess.Popen(['/usr/bin/nvidia-smi','-i',','.join(uuids.values()),
                '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits','-lms','200'],stdout=stream,stderr=subprocess.DEVNULL)
        return dict(gpus=raw,apps=apps,selected=module.UUID,desktop_exception='GPU0 existing PID1121899 only')
    module.health=health
    sys.stdin=SelectGPU()
    try:module.main()
    finally:
        sys.stdin=original_stdin
        if sampler is not None:
            sampler.terminate()
            try:sampler.wait(timeout=3)
            except subprocess.TimeoutExpired:sampler.kill();sampler.wait(timeout=3)
        if stream is not None:stream.close()

if __name__=='__main__':main()
