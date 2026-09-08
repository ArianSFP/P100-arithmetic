#!/usr/bin/env python3
"""Build a separate FP32-only worker and hash every source/control."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'qwen35-q4-affinitywave-20260908/affinity-wave.original.cu'
EXPECTED = '173003f61c4a4c33a37748657633f14a740911ec005e0a5706e9ab0af4bdafcb'

def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    source = raw.decode()
    blocks = []
    for start, end in [('struct aw_work_desc {', 'struct aw_m16_cohort {'),
                       ('__device__ __forceinline__ float aw_bf16_to_float',
                        '__global__ static void aw_repack_q8_t64('),
                       ('union aw_smem_m64 {', 'union aw_smem_m64_gate_up {')]:
        a = source.index(start)
        b = source.index(end, a)
        if start.startswith('__device__'):
            helper = source.index('__device__ __forceinline__ void aw_load_bf16x8', a)
            b = source.index('{', helper) + 1
            depth = 1
            while depth:
                depth += (source[b] == '{') - (source[b] == '}')
                b += 1
        blocks.append(source[a:b])
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    (build/'aw_control.inc').write_text('\n'.join(blocks))
    # An explicit storage-only control retains the original schedule and
    # arithmetic. Its only semantic change is how the same signed codes load.
    clone = blocks[-1].replace('aw_q8_service_m64(', 'aw_q4_service_m64(')
    clone = clone.replace('*AW_T64_STAGE_BYTES;', '*1152;')
    clone = clone.replace('b_r*AW_KSTAGE;', 'b_r*16;')
    old = 'pq[i] = *(const unsigned short *) (p + b_k + 2*i);'
    new = '''const int ki = b_k + 2*i;
                const unsigned char x0 = (unsigned char) p[ki % 16];
                const unsigned char x1 = (unsigned char) p[(ki + 1) % 16];
                const int q0 = int((x0 >> (ki >= 16 ? 4 : 0)) & 15) - 8;
                const int q1 = int((x1 >> (ki >= 16 ? 4 : 0)) & 15) - 8;
                pq[i] = (unsigned char) q0 | ((unsigned short) (unsigned char) q1 << 8);'''
    assert clone.count(old) == 1
    clone = clone.replace(old, new)
    clone = clone[clone.index('template<bool BF16_INPUT'):]
    (build/'aw_q4_control.inc').write_text(clone)
    cmd = ['/usr/local/cuda-12.8/bin/nvcc', '-ccbin', '/usr/bin/g++-14',
           '-O3', '-std=c++17', '-arch=sm_60', '-lineinfo', '-Xptxas=-v',
           '-I'+str(build), str(ROOT/'worker.cu'), '-o', str(build/'worker')]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (build/'compile.log').write_text(p.stdout)
    print(p.stdout[-6000:])
    p.check_returncode()
    with (build/'worker.sass').open('w') as f:
        subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump', '-sass', str(build/'worker')], stdout=f, check=True)
    sass = (build/'worker.sass').read_text()
    assert 'HFMA2' not in sass, 'Half accumulation is outside this study'
    paths = [ROOT/'worker.cu', ROOT/'kernels.cuh', Path(__file__),
             build/'aw_control.inc', build/'aw_q4_control.inc', build/'worker',
             ROOT/'supervise.py', ROOT/'analyze.py', ROOT/'test_offline.py',
             ROOT.parent/'w4a16-t64-r1-20260908/supervise.py']
    hashes = {str(x.relative_to(ROOT)) if x.is_relative_to(ROOT) else str(x):
              hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    (build/'manifest.json').write_text(json.dumps(dict(control_source=str(SOURCE),
        control_sha256=EXPECTED, command=cmd, hashes=hashes), indent=2)+'\n')

if __name__ == '__main__': main()
