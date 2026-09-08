#!/usr/bin/env python3
"""Reuse the existing lock/worker/health controller with a distinct claim."""
import importlib.util
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('controller', ROOT.parent/'w4a16-t64-r1-20260908/supervise.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
controller.ROOT = ROOT
controller.TOKEN = 'w4a16-t64-f32-20260908-gpu2'

def main():
    # Start sampling only after the reused controller holds the shared lock and
    # passes its first health check. This NVML process never initializes CUDA.
    original_health=controller.health
    sampler=None
    stream=None
    def health():
        nonlocal sampler,stream
        result=original_health()
        if sampler is None:
            path=ROOT/'gpu-results'/('telemetry-'+str(time.time_ns())+'.csv')
            stream=path.open('x')
            sampler=subprocess.Popen(['/usr/bin/nvidia-smi','-i',controller.UUID,
                '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits','-lms','200'],stdout=stream,stderr=subprocess.DEVNULL)
        return result
    controller.health=health
    try: controller.main()
    finally:
        if sampler is not None:
            sampler.terminate()
            try: sampler.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sampler.kill();sampler.wait(timeout=3)
        if stream is not None:stream.close()

if __name__ == '__main__': main()
