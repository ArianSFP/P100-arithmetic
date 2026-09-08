#!/usr/bin/env python3
"""CPU-only geometry/address/schedule checks. No CUDA imports or compiler."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DECODER='''__device__ __forceinline__ half2 decode_q4_pair(unsigned byte,half d){
    const unsigned bits=0x64006400u|(byte&15)|((byte>>4)<<16);
    const half2 q=__hsub2(*reinterpret_cast<const half2*>(&bits),__float2half2_rn(1032.f));
    return __hmul2(q,__halves2half2(d,d));
}
'''
def main():
    parent=(ROOT.parent/'w4a16-long-prefill-20260908/wide.cuh').read_text()
    assert parent.split('__global__',1)[0].removeprefix('#pragma once\n')==DECODER
    output=[];astage=[];bstage=[]
    for tid in range(256):
        warp,lane=divmod(tid,32)
        ar=warp//2*16+lane//8*4;bc=warp%2*32+lane%8*4
        assert ar%4==bc%4==0
        output.extend((ar+i,bc+j) for i in range(4) for j in range(4))
        astage.extend((tid%4*8+i,tid//4) for i in range(8))
        bstage.extend((tid//64*4+i+h,tid%64) for i in range(4) for h in (0,16))
        assert 128+tid*4+3<1152
        assert 2*(tid%64)+1<128
    assert len(output)==len(set(output))==4096
    assert set(output)=={(m,n) for m in range(64) for n in range(64)}
    fullstage={(k,r) for k in range(32) for r in range(64)}
    assert len(astage)==len(set(astage))==2048 and set(astage)==fullstage
    assert len(bstage)==len(set(bstage))==2048 and set(bstage)==fullstage
    # All true/tail token rows are covered once; absent rows never store.
    for t in (1,15,16,31,32,33,63,64,65,127,128,129,257,1025,4096):
        seen=[base+r for base in range(0,t,64) for r in range(64) if base+r<t]
        assert seen==list(range(t))
    address_checks=0
    for n,k in ((128,96),(512,2048),(2048,512)):
        t=65;experts=3;ng=n//64;groups=k//32
        for e in (0,experts-1):
            for tm in (0,1):
                for col in (0,ng-1):
                    for g in (0,groups-1):
                        for tid in range(256):
                            row=tm*64+tid//4
                            ai=(e*t+row)*k+g*32+tid%4*8
                            if row<t:
                                assert ai%4==0 and 0<=ai<=experts*t*k-8
                            wb=((e*ng+col)*groups+g)*1152
                            assert (wb+128+tid*4)%4==0
                            assert 0<=wb+128+tid*4<=experts*ng*groups*1152-4
                            address_checks+=1
    # The exact K membership/order of each persistent FP32 rail, both unrolls.
    for groups in (1,3,16,64):
        for u in (16,32):
            lo=[];hi=[];staged=0
            for g in range(groups):
                assert staged==g
                prefetched=g+1 if g+1<groups else None
                for chunk in range(32//u):
                    for s in range(u):
                        kk=chunk*u+s
                        (lo if kk<16 else hi).append(g*32+kk)
                if prefetched is not None:staged=prefetched
            assert lo==[x for x in range(groups*32) if x%32<16]
            assert hi==[x for x in range(groups*32) if x%32>=16]
    # Resource targets only: register allocation rounded to8 regs/thread.
    for regs,blocks in ((64,4),(80,3)):
        allocated=((regs+7)//8)*8*256
        assert min(65536//allocated,65536//16384)==blocks
    source=(ROOT/'tile256.cuh').read_text()
    assert '__launch_bounds__(256,MIN_BLOCKS)' in source
    assert 'float lo[4][4]={},hi[4][4]={}' in source
    assert '__fmaf_rn' in source and '__fadd_rn' in source
    assert '__hmul2' not in source and '__hfma' not in source
    print(f'PASS: decoder identity,4096 output owners,2048 A/B stage cells,15 tails,{address_checks} addresses,8 rail schedules,resource targets')
    print('CPU ONLY. No compilation, register count, numerical GPU proof or performance result.')
if __name__=='__main__':main()
