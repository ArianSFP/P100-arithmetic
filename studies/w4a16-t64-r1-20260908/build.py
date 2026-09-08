#!/usr/bin/env python3
"""Build isolated kernels; mechanically extract the unmodified T64 control."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'qwen35-q4-affinitywave-20260908/affinity-wave.original.cu'
EXPECTED = '173003f61c4a4c33a37748657633f14a740911ec005e0a5706e9ab0af4bdafcb'

def main():
    data = SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED, 'Control source changed'
    source = data.decode()
    blocks = []
    for start, end in [('struct aw_work_desc {', 'struct aw_m16_cohort {'),
                       ('__device__ __forceinline__ float aw_bf16_to_float',
                        '__global__ static void aw_repack_q8_t64('),
                       ('union aw_smem_m64 {', 'union aw_smem_m64_gate_up {')]:
        a = source.index(start)
        b = source.index(end, a)
        # The helper region ends immediately after aw_load_bf16x8.
        if start.startswith('__device__'):
            helper = source.index('__device__ __forceinline__ void aw_load_bf16x8', a)
            brace = source.index('{', helper)
            depth = 1
            b = brace + 1
            while depth:
                depth += (source[b] == '{') - (source[b] == '}')
                b += 1
        blocks.append(source[a:b])
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    (build / 'aw_control.inc').write_text('\n'.join(blocks))
    cmd = ['/usr/local/cuda-12.8/bin/nvcc', '-ccbin', '/usr/bin/g++-14',
           '-O3', '-std=c++17', '-arch=sm_60', '-lineinfo', '-Xptxas=-v',
           '-I'+str(build), str(ROOT/'worker.cu'), '-o', str(build/'worker')]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (build/'compile.log').write_text(p.stdout)
    print(p.stdout[-6000:])
    p.check_returncode()
    with (build/'worker.sass').open('w') as f:
        subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump', '-sass', str(build/'worker')], stdout=f, check=True)
    paths = [ROOT/'worker.cu', ROOT/'kernels.cuh', Path(__file__),
             build/'aw_control.inc', build/'worker']
    manifest = dict(control_source=str(SOURCE), control_sha256=EXPECTED, command=cmd,
                    hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
    (build/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    main()
