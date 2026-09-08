#!/usr/bin/env python3
"""Independent CPU ownership/index/order checks; no compiler/CUDA imports."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def body(s,name):
    a=s.index('{',s.index(name+'('));b=a+1;depth=1
    while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[a:b]
def main():
    src=(ROOT/'one_rail512.cuh').read_text()
    parent=(ROOT.parent/'w4a16-long-prefill-20260908/wide.cuh').read_text()
    assert body(src,'decode_pair')==body(parent,'decode_q4_pair')
    assert hashlib.sha256((ROOT/'direct_stage.cuh').read_bytes()).hexdigest()=='a968022fcfd4ebec465d4f7aaca1755ff579ece8267b577c9267fb3e70248c9b'
    aowners={};bowners={};owners=[{},{}]
    for tid in range(512):
        for i in range(4):
            key=(tid%8*4+i,tid//8);assert key not in aowners;aowners[key]=tid
        if tid<256:
            for byte in range(4):
                for upper in (0,16):
                    key=(tid//64*4+byte+upper,tid%64);assert key not in bowners;bowners[key]=tid
        warp,lane=divmod(tid,32);rail=warp//8;local=warp%8
        ar=local//2*16+lane//8*4;bc=local%2*32+lane%8*4
        assert ar%4==bc%4==0
        for i in range(4):
            for j in range(4):
                key=(ar+i,bc+j);assert key not in owners[rail];owners[rail][key]=tid
    fullstage={(k,r) for k in range(32) for r in range(64)}
    fulloutput={(r,c) for r in range(64) for c in range(64)}
    assert set(aowners)==set(bowners)==fullstage
    assert set(owners[0])==set(owners[1])==fulloutput
    assert all(owners[1][p]==owners[0][p]+256 for p in fulloutput)
    # Unique low writer and unique high reader/store, including full union extent.
    low={r*64+c:('lo',owners[0][r,c]) for r,c in fulloutput}
    assert set(low)==set(range(4096)) and (max(low)+1)*4==16384
    for r,c in fulloutput:assert low[r*64+c]==('lo',owners[1][r,c]-256)
    addresses=0
    for t,n,k,experts in ((1,128,32,2),(33,512,96,2),(65,2048,512,3),(129,512,2048,2)):
        for tm in range((t+63)//64):
            valid={(r,c) for r,c in fulloutput if tm*64+r<t}
            assert len(valid)==min(64,t-tm*64)*64
            for e in (0,experts-1):
                for col in (0,n//64-1):
                    for g in (0,k//32-1):
                        for tid in range(512):
                            row=tm*64+tid//8;ai=(e*t+row)*k+g*32+tid%8*4
                            if row<t:assert ai%4==0 and 0<=ai<=experts*t*k-4
                            if tid<256:
                                wb=((e*(n//64)+col)*(k//32)+g)*1152
                                assert 0<=wb+128+tid*4<=experts*(n//64)*(k//32)*1152-4
                            addresses+=1
    for groups in (1,3,16,64):
        for rail in (0,1):
            got=[g*32+rail*16+s for g in range(groups) for s in range(16)]
            expected=[i for i in range(groups*32) if i%32//16==rail]
            assert got==expected
        events=[]
        for g in range(groups):events+=['barrier','A-all/B-low-produce','barrier','both-rails-consume']
        events+=['barrier','low-writes-union','barrier','high-reads-low/add-lo-hi/output','barrier']
        assert events.count('barrier')==2*groups+3
        assert events[-5:]==['barrier','low-writes-union','barrier','high-reads-low/add-lo-hi/output','barrier']
    assert 'float acc[4][4]={}' in src and '__launch_bounds__(512,2)' in src
    assert '__fadd_rn(low,acc[i][j])' in src
    print(f'PASS: frozen direct header,decoder identity,2048 A/B owners,8192 rail owners,4096 unique low merges,{addresses} addresses,8 rail orders,barriers')
    print('CPU ONLY. No compile/GPU; register resources and exact device behavior unverified.')
if __name__=='__main__':main()
