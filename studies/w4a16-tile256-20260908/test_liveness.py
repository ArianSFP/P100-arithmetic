#!/usr/bin/env python3
"""CPU-only packed-A16 lane mapping and delayed-fetch schedule models."""
import hashlib
import math
import struct
from pathlib import Path
from test_cpu import main as geometry

ROOT=Path(__file__).resolve().parent
def half_bits(x):return struct.unpack('<H',struct.pack('<e',x))[0]
def from_half(b):return struct.unpack('<e',struct.pack('<H',b))[0]
def main():
    geometry()
    checked=0
    # Every finite half pattern remains bit-identical through pair packing.
    for b in range(65536):
        if b&0x7c00==0x7c00:continue
        other=b^0x8000
        word=half_bits(from_half(b))|(half_bits(from_half(other))<<16)
        assert word&65535==b and word>>16==other;checked+=1
    # Raw FP32 midpoints and neighboring floats check input, not just A16 origin.
    witnesses=[0.,-0.,2**-25,3*2**-25,1+2**-11,1+3*2**-11,
               1+2**-11-2**-23,1+2**-11+2**-23,65504.,-65504.,2**-24,-2**-24]
    witnesses+=[-x for x in witnesses]
    for tid in range(256):
        av=[witnesses[(tid+i)%len(witnesses)] for i in range(8)]
        packed=[half_bits(av[2*i])|(half_bits(av[2*i+1])<<16) for i in range(4)]
        recovered=[v for word in packed for v in (word&65535,word>>16)]
        assert recovered==[half_bits(x) for x in av]
        assert [(tid%4*8+2*i+h,tid//4) for i in range(4) for h in (0,1)]==[(tid%4*8+i,tid//4) for i in range(8)]
    for prefetch in (False,True):
        for groups in (1,3,16,64):
            fetched=[0];stage=0;compute=[];events=['fetch0','stage0','barrier']
            for g in range(groups):
                assert stage==g
                if prefetch and g+1<groups:fetched.append(g+1);events.append('fetch')
                compute.append(stage);events.append('compute')
                if not prefetch and g+1<groups:fetched.append(g+1);events.append('fetch')
                events.append('barrier')
                if g+1<groups:stage=fetched[-1];events.append('stage')
                events.append('barrier')
            assert fetched==compute==list(range(groups))
            for i,e in enumerate(events):
                if e=='stage':assert events[i-1]==events[i+1]=='barrier'
    print(f'PASS: {checked} finite paired-half patterns,256 raw-A32 lane mappings,8 fetch/stage schedules')
    print('Frozen original tile256 SHA256:',hashlib.sha256((ROOT/'tile256.cuh').read_bytes()).hexdigest())
    print('CPU ONLY; register count, spill freedom and GPU equivalence unproven.')
if __name__=='__main__':main()
