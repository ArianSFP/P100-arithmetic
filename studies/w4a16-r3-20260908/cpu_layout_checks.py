#!/usr/bin/env python3
"""Offline bijection/order checks for the additional round-3 mappings."""
import json
from pathlib import Path

cases=0
for nw in (8,16,32):
    stripes=[warp+slot*nw for slot in range(32//nw) for warp in range(nw)]
    assert sorted(stripes)==list(range(32))
    for r in (1,2):
        for batch in (1,4):
            count=batch*32*r*32
            writes=[((b*32+s)*r+row)*32+lane for s in stripes
                    for row in range(r) for b in range(batch) for lane in range(32)]
            assert sorted(writes)==list(range(count))
            values=[(i*1351+71)%65537-32768 for i in range(count)]
            reference=[]
            for b in range(batch):
                for row in range(r*32):
                    parts=[values[(b*32+s)*r*32+row] for s in range(32)]
                    for offset in (16,8,4,2,1):
                        for s in range(offset):parts[s]+=parts[s+offset]
                    reference.append(parts[0])
            for offset in (16,8,4,2,1):
                seen=set()
                for thread in range(nw*32):
                    for i in range(thread,batch*offset*r*32,nw*32):
                        row=i%(r*32);s=(i//(r*32))%offset;b=i//(offset*r*32)
                        index=(b*32+s)*r*32+row
                        assert index not in seen;seen.add(index)
                        values[index]+=values[index+offset*r*32]
                assert len(seen)==batch*offset*r*32
            result=[values[b*32*r*32+row] for b in range(batch) for row in range(r*32)]
            assert result==reference
            cases+=1
for bits in range(65536):
    high=bits^0xa571;packed=bits|(high<<16)
    assert packed&65535==bits and packed>>16==high
for chunk in range(4):
    assert [2*(chunk*4+j)+h for j in range(4) for h in range(2)]==list(range(chunk*8,chunk*8+8))
result=dict(status='PASS',fused_mapping_cases=cases,packed_half_pairs=65536,
            reduction_order_unchanged=True,max_dynamic_shared_bytes=32*2*4*32*4)
print(json.dumps(result))
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    (root/'build/cpu-layout-checks.json').write_text(json.dumps(result,indent=2)+'\n')
