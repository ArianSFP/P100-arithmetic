#!/usr/bin/env python3
"""Select one representative historical-Q8 4096-token owner-shard profile.

This is shape evidence only. It is deliberately not current-Q4 acceptance.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import struct

ROUTES=Path('/home/arian/llama.cpp-q36-moe/.worktrees/elasticwave/results/qwen36-35b-moe-pp-20260721/elasticwave-20260723/routes-code-16x4096.awtr')
ROUTES_SHA='8f78f5338327becbac14ed9c82f93b1199eb4d7bf6455fef5e3b58cc744038c3'
PLACEMENT=Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/docs/p100-pairwave-project/evidence/runtime/placement-hot16.json')
PLACEMENT_SHA='96de3b381bb197b5d843bc9e536496114b74db2cfb77d6f7bf254c75dd851030'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    assert sha(ROUTES)==ROUTES_SHA
    actual_placement_sha=sha(PLACEMENT)
    if actual_placement_sha!=PLACEMENT_SHA:
        raise RuntimeError(f'placement hash changed: {actual_placement_sha}')
    placement=json.loads(PLACEMENT.read_text())
    assert placement['experts_per_layer']==256 and placement['gpu_count']==4
    records=[]
    with ROUTES.open('rb') as f:
        assert f.read(8)==b'AWTRV001'
        index=0
        while header:=f.read(8):
            layer,tokens,topk=struct.unpack('<HIH',header)
            assert tokens==4096 and topk==8
            ids=struct.unpack('<'+str(tokens*topk)+'H',f.read(tokens*topk*2))
            counts=Counter(ids)
            assert all(0<=e<256 for e in ids)
            for token in range(tokens):
                route=ids[token*topk:(token+1)*topk]
                assert len(set(route))==topk
            owners=placement['placement'][layer]['primary_owner']
            shards=[]
            for owner in range(4):
                experts=[e for e,x in enumerate(owners) if x==owner]
                assert len(experts)==64
                values=[counts[e] for e in experts]
                assert sum(values)==sum(counts[e] for e in range(256) if owners[e]==owner)
                shards.append(dict(owner=owner,global_experts=experts,counts=values,
                    routes=sum(values),m64_tiles=sum((x+63)//64 for x in values),
                    m32_tiles=sum((x+31)//32 for x in values),maximum=max(values),minimum=min(values)))
            records.append(dict(index=index,layer=layer,shards=shards));index+=1
    assert len(records)==640
    # Medoid in the two relevant tile-work totals; deterministic index tie break.
    work=[(sum(x['m64_tiles'] for x in r['shards']),sum(x['m32_tiles'] for x in r['shards'])) for r in records]
    med64=statistics.median(x[0] for x in work);med32=statistics.median(x[1] for x in work)
    chosen=min(range(len(records)),key=lambda i:(abs(work[i][0]-med64)+abs(work[i][1]-med32),i))
    result=dict(scope='historical Q8/4096 route-shape proxy, NOT current-Q4 or 2k/8k acceptance',
        routes_path=str(ROUTES),routes_sha256=ROUTES_SHA,placement_path=str(PLACEMENT),
        placement_sha256=PLACEMENT_SHA,records=len(records),selection='medoid of total M64/M32 tile counts',
        medians=dict(m64_tiles=med64,m32_tiles=med32),selected=records[chosen])
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
