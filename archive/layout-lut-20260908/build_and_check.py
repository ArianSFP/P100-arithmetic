#!/usr/bin/env python3
"""CPU-only compiler/address study. This file never queries or launches a GPU."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import random
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)

def run(args):
    p=subprocess.run([str(x) for x in args],text=True,capture_output=True,timeout=60)
    record=dict(argv=list(map(str,args)),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    if p.returncode: print(p.stderr)
    return record

def save(name,record):
    (OUT/name).write_text(json.dumps(record,indent=2)+'\n')

def banks(word_addresses):
    # Repeated reads of the SAME word are broadcasts, not bank conflicts.
    distinct=defaultdict(set)
    for address in word_addresses: distinct[address%32].add(address)
    return max(map(len,distinct.values()))

def address_checks():
    rng=random.Random(0x60ba9832)
    for _ in range(262144):
        indices=[rng.randrange(16) for _ in range(32)]
        assert banks(indices)==1
        assert banks([32*idx+lane for lane,idx in enumerate(indices)])==1
    assert banks([0]*32)==1  # Broadcast positive control.
    assert banks([32*i for i in range(32)])==32  # Distinct words, same bank.
    for rows in (32,64,128):
        for groups in (1,3,160):
            for bits in (2,4):
                addresses=[((row//32*groups+g)*bits+p)*32+row%32
                           for row in range(rows) for g in range(groups) for p in range(bits)]
                assert len(set(addresses))==rows*groups*bits and min(addresses)==0 and max(addresses)==len(addresses)-1
    sectors={}
    for bits in (2,4):
        groups=160
        for row_parallel in (True,False):
            for tiled in (False,True):
                words=[]
                for lane in range(32):
                    row,g=(lane,0) if row_parallel else (0,lane)
                    words.append(((row//32*groups+g)*bits)*32+row%32 if tiled else row*bits*groups+g)
                sectors[f'W{bits}_row_parallel_{row_parallel}_tiled_{tiled}']=len({w//8 for w in words})
    grid=[]
    for rows in (5120,17408):
        for reuse in (1,2,4):
            for splits in (1,2,4,8,16):
                grid.append(dict(rows=rows,rows_per_warp=32*reuse,split_k=splits,
                    ctas=((rows+128*reuse-1)//(128*reuse))*splits,
                    partial_bytes_per_batch=rows*splits*4 if splits>1 else 0))
    for groups in (1,3,32,160,544):
        for splits in (1,2,4,8,16):
            owned=[g for s in range(splits) for g in range(s,groups,splits)]
            assert sorted(owned)==list(range(groups))
    record=dict(status='PASS_CPU_ADDRESS_MODEL_ONLY',seed='0x60ba9832',bank_cases=262144,
                compact_shared_bytes_per_warp=64,replicated_shared_bytes_per_warp=2048,
                compact_bank_distinct_words_max=1,replicated_bank_distinct_words_max=1,
                negative_control_bank_distinct_words_max=32,weight_load_32byte_sectors=sectors,
                grid_model=grid,split_k_group_ownership='PASS',
                caveat='Per-instruction aligned full-warp sector model; not measured bandwidth or latency')
    save('address-model.json',record)
    print('PASS CPU bank, coalescing, row-grid and split-K ownership models')

cpu_build=run(['/usr/bin/g++-14','-O3','-std=c++17','-Wall','-Wextra','-Wno-unknown-pragmas',ROOT/'cpu_tests.cpp','-o',OUT/'cpu-tests'])
save('cpu-build.json',cpu_build)
if cpu_build['returncode']: raise SystemExit(2)
cpu=run([OUT/'cpu-tests']); save('cpu.json',cpu); print(cpu['stdout'])
if cpu['returncode']: raise SystemExit(2)
address_checks()
build=run(['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14','-std=c++17','-O3','-arch=sm_60',
           '-lineinfo','-Xptxas=-v','-cubin',ROOT/'kernels.cu','-o',OUT/'kernels.cubin'])
save('cuda-build.json',build)
if build['returncode']: raise SystemExit(2)
disasm=run(['/usr/local/cuda-12.8/bin/cuobjdump','-sass',OUT/'kernels.cubin'])
if disasm['returncode']: raise SystemExit(2)
(OUT/'kernels.sass').write_text(disasm['stdout'])
functions={}
for name,body in re.findall(r'Function\s*:\s*(\S+)\n(.*?)(?=Function\s*:|\Z)',disasm['stdout'],re.S):
    ins=[re.sub(r'^@!?P\d+\s+','',s.strip().lstrip('{').strip())
         for s in re.findall(r'/\*[0-9a-f]+\*/\s+([^;\n]+);',body)]
    functions[name]=dict(instructions=ins,counts=dict(Counter(s.split()[0].split('.')[0] for s in ins)))
resources={}
for name,stack,stores,loads,regs,tail in re.findall(
    r"Compiling entry function '([^']+)'.*?(\d+) bytes stack frame, (\d+) bytes spill stores, "
    r"(\d+) bytes spill loads.*?Used (\d+) registers,([^\n]+)",build['stderr'],re.S):
    smem=re.search(r'(\d+) bytes smem',tail)
    resources[name]=dict(registers=int(regs),stack_bytes=int(stack),spill_stores=int(stores),spill_loads=int(loads),shared_bytes=int(smem.group(1)) if smem else 0)
assert set(resources)==set(functions), 'Missing compiler resource record'
summary=[]
for name,f in functions.items():
    row=re.search(r'rowsILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb([01])E',name)
    if row:
        bits,fmt,method,reuse,old=map(int,row.groups())
        counts=f['counts']
        if method==1: assert counts.get('VABSDIFF4',0)==8*bits*reuse
        if method in (2,5): assert counts.get('SHFL',0)==8*bits*reuse
        if method==3: assert resources[name]['shared_bytes']==256
        if method==4: assert resources[name]['shared_bytes']==8192
        summary.append(dict(name=name,W=bits,F=fmt,M=method,R=reuse,old=old,
            static_instructions=len(f['instructions']),**resources[name],
            key_ops={op:counts.get(op,0) for op in ('VABSDIFF4','PRMT','SHFL','LOP','LOP32I','BFE','SHL','SHR','VMAD','LDS','STS','BAR')}))
    if 'sad_group' in name:
        bits,fmt=map(int,re.search(r'ILi(\d+)ELi(\d+)E',name).groups())
        logic=[i for i in f['instructions'] if i.startswith('LOP')]
        assert len(logic)==8 and all(i.startswith('LOP32I.XOR ') and i.endswith('0x80808080') for i in logic), name
        assert f['counts']['PRMT']==8*bits and f['counts']['VABSDIFF4']==8*bits
assert len(summary)==60 and len(functions)==74, (len(summary),len(functions))
record=dict(scope='CPU tests and SM60 compiler output only; no GPU execution',
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'common.hpp',ROOT/'kernels.cu',ROOT/'cpu_tests.cpp',OUT/'kernels.cubin')},
    functions=functions,resources=resources,rows_summary=summary)
save('inventory.json',record)
if '--verbose' in sys.argv:
    for s in sorted(summary,key=lambda x:(x['W'],x['F'],x['M'],x['R'],x['old'])):
        print('SASS', {k:v for k,v in s.items() if k!='name'})
print(f'PASS compiler audit: {len(functions)} kernels; resource/spill counts are evidence, not hardware validation')
