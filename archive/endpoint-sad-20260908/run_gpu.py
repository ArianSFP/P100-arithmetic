#!/usr/bin/env python3
"""Fail-closed supervisor. Never initializes CUDA or acquires/releases a reservation.

Default is offline validation only. See RESULTS.md before --execute.
The reservation owner must keep GPU0 held through final health/log review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
COORD = Path('/home/arian/llama.cpp-qwen38-p100/bench/COORDINATION-20260908.md')
GPU = 'GPU-bb126d7c-3d48-3911-bf1f-8b4ed9f2e706'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def command(argv, timeout=5):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired:
        return dict(argv=argv, returncode=None, status='timeout_STOP_no_reset')


def check_reservation(token, digest):
    text = COORD.read_text()
    if sha(COORD) != digest:
        raise RuntimeError('Coordination snapshot changed; re-review all reservations')
    # This token is only written AFTER the other owner explicitly releases.
    marker = '- HELD: endpoint-SAD GPU0 token=' + token
    if not token or marker not in text.splitlines():
        raise RuntimeError('No explicitly reviewed endpoint-SAD reservation token')
    return dict(path=str(COORD), sha256=digest, token=token)


def health():
    return dict(
        devices=command(['/usr/bin/nvidia-smi',
            '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total',
            '--format=csv,noheader,nounits']),
        processes=command(['/usr/bin/nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader']))


def check_health(h):
    if h['devices']['returncode'] != 0 or h['processes']['returncode'] != 0:
        raise RuntimeError('GPU health query failed; stop without reset/retry')
    if h['processes']['stdout'].strip():
        raise RuntimeError('A compute process is active; no overlapping test')
    rows = [list(map(str.strip, row.split(','))) for row in h['devices']['stdout'].splitlines()]
    if len(rows) != 4 or sum(row[1] == GPU for row in rows if len(row) == 9) != 1:
        raise RuntimeError('Expected four-P100 rig and unique target GPU')
    for row in rows:
        if len(row) != 9 or 'P100' not in row[2] or int(row[4]) > 64 or int(row[5]) != 0 or int(row[8]) != 0:
            raise RuntimeError('Rig not idle/healthy; stop without reset/retry')


def verify_offline():
    inventory = json.loads((OUT/'inventory.json').read_text())
    if sha(ROOT/'endpoint_sad.cu') != inventory['source_sha256'] or sha(ROOT/'endpoint-sad') != inventory['binary_sha256']:
        raise RuntimeError('Source/binary changed since offline gates')
    if not inventory['offline_gates'].startswith('PASS:'):
        raise RuntimeError('Offline gates did not pass')
    cpu = json.loads((OUT/'cpu.json').read_text())
    if cpu['returncode'] != 0 or 'CPU PASS packed W=4' not in cpu['stdout']:
        raise RuntimeError('CPU correctness gate missing')
    return dict(source_sha256=inventory['source_sha256'], binary_sha256=inventory['binary_sha256'],
                inventory_sha256=sha(OUT/'inventory.json'), cpu_sha256=sha(OUT/'cpu.json'),
                supervisor_sha256=sha(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reservation-token', default='')
    parser.add_argument('--coordination-sha256', default='')
    args = parser.parse_args()
    provenance = verify_offline()
    if not args.execute:
        print(json.dumps(dict(status='OFFLINE PASS; no NVIDIA query or GPU launch', **provenance), indent=2))
        return 0
    reservation = check_reservation(args.reservation_token, args.coordination_sha256)
    series = OUT / ('gpu-series-' + str(time.time_ns()))
    series.mkdir()
    print('Reservation remains held across all workers and final review:', series, flush=True)
    argv = ['/usr/bin/env', '-i', 'PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin',
            'LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES='+GPU, '/usr/bin/taskset', '--cpu-list', '0-11',
            str(ROOT/'endpoint-sad'), '--gpu']
    for run in range(3):
        record = dict(run=run, gpu_uuid=GPU, argv=argv, reservation=reservation,
                      **provenance, hypothesis='Compare exact signed W2/W4 G32 formulations with scale and preparation costs',
                      input_seeds='Embedded deterministic CPU/GPU seeds in hashed source',
                      scope='Unmodified compiler-generated documented PTX; synthetic format, not a model benchmark')
        try:
            check_reservation(args.reservation_token, args.coordination_sha256)
            verify_offline()
            record['health_before'] = health()
            check_health(record['health_before'])
            persist(series/f'{run}-prelaunch.json', record)
            # Keep raw output on disk even if a worker crashes or times out.
            with (series/f'{run}.stdout').open('x') as out, (series/f'{run}.stderr').open('x') as err:
                proc = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
                record['pid'] = proc.pid
                try:
                    record['returncode'] = proc.wait(timeout=45)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass  # Uninterruptible GPU worker is NOT assumed recovered.
                    raise RuntimeError('Worker timeout/interrupted: STOP, no reset, no retry')
            if record['returncode'] != 0:
                raise RuntimeError('Worker failed; stop all remaining runs')
            record['health_after'] = health()
            check_health(record['health_after'])
            record['status'] = 'PASS'
        except (RuntimeError, ValueError, OSError) as exc:
            record['status'] = 'STOP: ' + str(exc)
            if 'pid' in record and 'health_after' not in record:
                record['health_after'] = health()
        persist(series/f'{run}-result.json', record)
        print(f'worker {run}: {record["status"]}', flush=True)
        if record['status'] != 'PASS':
            print('No reservation release: owner must inspect device state and logs.', flush=True)
            return 3
    print('Three workers complete. Keep reservation held until final log/health review; release manually.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
