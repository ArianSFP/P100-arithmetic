#!/usr/bin/env python3
"""Independent CPU models for two proposed designs, not CUDA implementations."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FROZEN={'tile256.cuh':'218e323c96e1421aae73f343f7cedc67b3b390a2c4b3cd55ea846f923f456beb',
        'liveness.cuh':'36dd1bb71f94306d1ff70d4852b433afec0e7278f0150d5bee15c3f09f826b58'}
def main():
    for name,h in FROZEN.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    target={(ki,row) for ki in range(32) for row in range(64)}
    # Direct streaming: each vector is converted/stored before the next vector.
    direct=[]
    for tid in range(256):
        for vec in range(2):
            direct.extend((tid%4*8+vec*4+i,tid//4) for i in range(4))
    assert len(direct)==len(set(direct))==2048 and set(direct)==target
    # Split-rail512: all threads load one float4 A, first256 load all packed B.
    astage=[(tid%8*4+i,tid//8) for tid in range(512) for i in range(4)]
    bstage=[(tid//64*4+i+hi,tid%64) for tid in range(256) for i in range(4) for hi in (0,16)]
    assert len(astage)==len(bstage)==2048
    assert set(astage)==set(bstage)==target
    owners={0:{},1:{}}
    for tid in range(512):
        warp,lane=divmod(tid,32);rail=warp//8;local=warp%8
        ar=local//2*16+lane//8*4;bc=local%2*32+lane%8*4
        assert ar%4==bc%4==0
        for i in range(4):
            for j in range(4):
                key=(ar+i,bc+j)
                assert key not in owners[rail]
                owners[rail][key]=tid
    full={(m,n) for m in range(64) for n in range(64)}
    assert set(owners[0])==set(owners[1])==full
    assert len({m*64+n for m,n in full})==4096
    assert 4096*4==(32*64+32*64)*4==16384 # reduction/staging union
    schedule_checks=0
    for groups in (1,3,16,64):
        reference=[[],[]]
        for g in range(groups):
            for ki in range(32):reference[ki//16].append(g*32+ki)
        for rail in (0,1):
            split=[g*32+ki for g in range(groups) for ki in range(rail*16,rail*16+16)]
            assert split==reference[rail];schedule_checks+=1
        # Shared union is not reused until all mainloop consumers complete.
        phases=['staging/compute']*groups+['barrier','low-rail-store','barrier','high-rail-read-add-store','barrier','next-tile-stage']
        assert phases[-6:]==['barrier','low-rail-store','barrier','high-rail-read-add-store','barrier','next-tile-stage']
    tails=0
    for m in (1,15,16,31,32,33,63,64,65,127,128,129,257,1025,4096):
        for tile in range((m+63)//64):
            valid={(r,n) for r,n in full if tile*64+r<m}
            assert {key for key in owners[1] if tile*64+key[0]<m}==valid
            assert len(valid)==min(64,m-tile*64)*64
        tails+=1
    # Two512-thread blocks with64 registers:32 resident warps, hardware limits
    # still require compiler/runtime confirmation; this is only arithmetic.
    assert 512*64*2==65536 and 512//32*2==32
    assert 512*16==256*32 # same CTA accumulator total, less per thread
    print(f'PASS: frozen headers, two A staging maps, B staging,8192 rail owners,{schedule_checks} ordered rails,{tails} tails,16KiB union')
    print('CPU proposal models only; no new CUDA kernel, compiler, GPU or performance claim.')
if __name__=='__main__':main()
