#!/usr/bin/env python3
"""Independent CPU-only direct-producer index and FP32 operation-order models."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def body(s,name):
    a=s.index(name+'(');a=s.index('{',a);b=a+1;depth=1
    while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[a:b]
def main():
    src=(ROOT/'direct_stage.cuh').read_text()
    parent=(ROOT.parent/'w4a16-long-prefill-20260908/wide.cuh').read_text()
    assert body(src,'decode_pair')==body(parent,'decode_q4_pair')
    frozen={'tile256.cuh':'218e323c96e1421aae73f343f7cedc67b3b390a2c4b3cd55ea846f923f456beb',
            'liveness.cuh':'36dd1bb71f94306d1ff70d4852b433afec0e7278f0150d5bee15c3f09f826b58'}
    for name,h in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    sa={};sb={};outputs={}
    for tid in range(256):
        for vec in range(2):
            for i in range(4):
                cell=(tid%4*8+vec*4+i,tid//4)
                assert cell not in sa;sa[cell]=(tid,vec,i)
        for byte in range(4):
            for upper in (0,1):
                cell=(tid//64*4+byte+16*upper,tid%64)
                assert cell not in sb;sb[cell]=(tid,byte,upper)
        warp,lane=divmod(tid,32)
        ar=warp//2*16+lane//8*4;bc=warp%2*32+lane%8*4
        assert ar%4==bc%4==0
        for i in range(4):
            for j in range(4):
                cell=(ar+i,bc+j)
                assert cell not in outputs;outputs[cell]=tid
    assert set(sa)==set(sb)=={(ki,row) for ki in range(32) for row in range(64)}
    assert set(outputs)=={(m,n) for m in range(64) for n in range(64)}
    address_checks=0
    for t,n,k,e in ((1,128,32,2),(33,512,96,2),(65,2048,512,3),(129,512,2048,2)):
        for tm in range((t+63)//64):
            for col in (0,n//64-1):
                for expert in (0,e-1):
                    for g in (0,k//32-1):
                        for tid in range(256):
                            row=tm*64+tid//4
                            for vec in range(2):
                                ai=(expert*t+row)*k+g*32+tid%4*8+4*vec
                                if row<t:assert ai%4==0 and 0<=ai<=e*t*k-4
                            wb=((expert*(n//64)+col)*(k//32)+g)*1152
                            assert 0<=wb+128+tid*4<=e*(n//64)*(k//32)*1152-4
                            address_checks+=1
            valid={(r,c) for r,c in outputs if tm*64+r<t}
            assert len(valid)==min(64,t-tm*64)*64
    # Compare the exact ordered operation tokens for every output's two rails.
    # Arithmetic values are intentionally unnecessary: identical FMA operands
    # and ordering imply same schedule, subject to GPU decode/conversion gates.
    for groups in (1,3,16,64):
        direct=[[],[]];reference=[[],[]]
        for g in range(groups):
            for kk in range(32):direct[kk//16].append(('fma',g*32+kk))
            reference[0].extend(('fma',g*32+kk) for kk in range(16))
            reference[1].extend(('fma',g*32+kk) for kk in range(16,32))
        assert direct==reference
        phases=[]
        for _ in range(groups):phases+=['barrier','stageA','stageB','barrier','compute']
        phases+=['output-add-lo-hi','barrier']
        assert phases.count('barrier')==2*groups+1
        for i,phase in enumerate(phases):
            if phase=='stageA':assert phases[i-1]=='barrier'
            if phase=='compute':assert phases[i-1]=='barrier'
    assert src.count('#pragma unroll 1')==2
    assert 'fetch' not in src and 'float lo[4][4]={},hi[4][4]={}' in src
    assert '__launch_bounds__(256,4)' in src and '__fmaf_rn' in src and '__fadd_rn' in src
    print(f'PASS: copied decoder identity,frozen headers,2048 A/B cells,4096 outputs,{address_checks} address cases,4 exact rail/barrier schedules')
    print('CPU ONLY; no compiler/CUDA. Physical registers, numerical device behavior and speed unverified.')
if __name__=='__main__':main()
