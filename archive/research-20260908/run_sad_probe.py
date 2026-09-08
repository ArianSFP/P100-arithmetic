#!/usr/bin/env python3
"""Fixed documented-PTX worker only. No CUDA in controller, no reset or retry."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'compiler-output'
GPU = 'GPU-bb126d7c-3d48-3911-bf1f-8b4ed9f2e706'

def command(args, timeout=5):
    p = subprocess.run([str(x) for x in args], capture_output=True, text=True,
                       timeout=timeout, check=False)
    return {'argv': [str(x) for x in args], 'returncode': p.returncode,
            'stdout': p.stdout, 'stderr': p.stderr}

def persist(path, obj):
    # Exclusive create keeps earlier runs immutable.
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def health():
    return command(['/usr/bin/nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm', '--format=csv,noheader'])

def main():
    worker = OUT / 'sad-probe'
    disasm = command(['/usr/local/cuda-12.8/bin/cuobjdump', '-sass', worker])
    if disasm['returncode'] or 'VABSDIFF4.U8.U8.ACC' not in disasm['stdout']:
        raise SystemExit('Expected SM60 unsigned SAD code not found')
    stamp = str(time.time_ns())
    record = {'id': stamp, 'hypothesis': 'SM60 unsigned SAD plus masks implements exact W2A8/W4A8; signed-byte conversion fuses',
              'scope': 'Documented PTX, compiler-generated unmodified binary; no model evaluation.',
              'gpu_uuid': GPU, 'cases': 262144, 'input_seed': '0x60a8c002',
              'source_sha256': hashlib.sha256((ROOT/'sad_probe.cu').read_bytes()).hexdigest(),
              'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
              'health_before': health()}
    apps = command(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'])
    record['processes_before'] = apps
    if apps['returncode'] or apps['stdout'].strip() or record['health_before']['returncode']:
        persist(OUT/f'sad-{stamp}-blocked.json', record)
        raise SystemExit('Rig not idle/healthy; no worker launched')
    for row in record['health_before']['stdout'].strip().splitlines():
        cols = [x.strip() for x in row.split(',')]
        if len(cols) != 8 or int(cols[4].split()[0]) > 64 or int(cols[5].split()[0]) != 0:
            raise SystemExit('Unexpected memory use/utilization; no worker launched')
    argv = ['/usr/bin/env', '-i', 'PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin',
            'LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES='+GPU, '/usr/bin/taskset', '--cpu-list', '0-11', str(worker)]
    record['argv'] = argv
    (OUT/f'sad-{stamp}.sass').write_text(disasm['stdout'])
    persist(OUT/f'sad-{stamp}-prelaunch.json', record)
    try:
        record['worker'] = command(argv, timeout=20)
        record['status'] = 'completed'
    except subprocess.TimeoutExpired as exc:
        def txt(x): return x.decode(errors='replace') if isinstance(x, bytes) else (x or '')
        record['worker'] = {'returncode': None, 'stdout': txt(exc.stdout), 'stderr': txt(exc.stderr)}
        record['status'] = 'timeout_STOP_no_reset'
    try:
        record['health_after'] = health()
    except subprocess.TimeoutExpired:
        record['health_after'] = {'returncode': None, 'status': 'timeout_STOP_no_reset'}
    persist(OUT/f'sad-{stamp}-result.json', record)
    print(json.dumps(record, indent=2))
    return 0 if (record['status'] == 'completed' and record['worker']['returncode'] == 0
                 and record['health_after']['returncode'] == 0) else 3

if __name__ == '__main__':
    raise SystemExit(main())
