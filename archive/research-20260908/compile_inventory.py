#!/usr/bin/env python3
"""Compile and disassemble fixed documented PTX only; never initialize CUDA."""
import collections
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
CUDA = pathlib.Path('/usr/local/cuda-12.8/bin')

def run(args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    return {'argv': [str(x) for x in args], 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}

def main():
    output = ROOT / 'compiler-output'
    output.mkdir(exist_ok=True)
    cubin = output / 'compiler_probes.sm60.cubin'
    compile_result = run([CUDA / 'nvcc', '-ccbin', '/usr/bin/g++-14', '-O3',
                          '-arch=sm_60', '-cubin', ROOT / 'compiler_probes.cu',
                          '-o', cubin])
    (output / 'compile.json').write_text(json.dumps(compile_result, indent=2) + '\n')
    if compile_result['returncode']:
        raise SystemExit(compile_result['stderr'])
    ptx_result = run([CUDA/'nvcc', '-ccbin', '/usr/bin/g++-14', '-O3',
                      '-arch=sm_60', '-ptx', ROOT/'compiler_probes.cu', '-o',
                      output/'compiler_probes.sm60.ptx'])
    if ptx_result['returncode']:
        raise SystemExit(ptx_result['stderr'])
    disasm = run([CUDA / 'cuobjdump', '-sass', cubin])
    if disasm['returncode']:
        raise SystemExit(disasm['stderr'])
    (output / 'compiler_probes.sm60.sass').write_text(disasm['stdout'])
    kernels = {}
    for name, section in re.findall(r'Function\s*:\s*(\w+)\n(.*?)(?=Function\s*:|\Z)',
                                    disasm['stdout'], re.S):
        instructions = []
        for offset, ins in re.findall(r'/\*([0-9a-f]+)\*/\s+([^;\n]+);', section):
            ins = ins.strip().lstrip('{').strip()
            instructions.append({'offset': '0x' + offset, 'sass': ins})
        counts = collections.Counter(x['sass'].split()[0].split('.')[0]
                                     for x in instructions)
        kernels[name] = {'counts_including_setup': dict(counts),
                         'instructions': instructions}
    artifact = {
        'scope': 'Compiler-only; NOT execution, throughput, or numerical validation.',
        'source_sha256': hashlib.sha256((ROOT / 'compiler_probes.cu').read_bytes()).hexdigest(),
        'cubin_sha256': hashlib.sha256(cubin.read_bytes()).hexdigest(),
        'nvcc': run([CUDA / 'nvcc', '--version']),
        'kernels': kernels,
    }
    gates = {}
    for sm in (60, 61):
        ptx = output / f'dp4a_gate.sm{sm}.ptx'
        ptx.write_text((ROOT/'dp4a_gate.ptx').read_text().replace('sm_60', f'sm_{sm}'))
        gates[str(sm)] = run([CUDA/'ptxas', f'-arch=sm_{sm}', ptx, '-o',
                             output/f'dp4a_gate.sm{sm}.cubin'])
    artifact['dp4a_ptx_gates'] = gates
    if gates['60']['returncode'] == 0 or gates['61']['returncode'] != 0:
        raise SystemExit('Unexpected documented DP4A target gate behavior')
    (output / 'inventory.json').write_text(json.dumps(artifact, indent=2) + '\n')
    for name, kernel in sorted(kernels.items()):
        ops = [x['sass'] for x in kernel['instructions']
               if x['sass'].split()[0].split('.')[0] not in
               {'MOV', 'LDG', 'STG', 'EXIT', 'BRA', 'NOP'}]
        print(name + ': ' + ' | '.join(ops))
    print('DP4A gate: sm60 rejected, sm61 compiled. No hardware execution.')

if __name__ == '__main__':
    main()
