"""Historical Q8 4096-token primary-owner proxy, NEVER current-Q4 acceptance.

No CUDA/compiler imports. Validate all route tuples; emit counts and current
T64-policy tiles, not purported captured dispatch/cohort/activation values.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import struct

ROOT=Path(__file__).resolve().parent
TRACE=Path('/home/arian/llama.cpp-q36-moe/.worktrees/elasticwave/results/qwen36-35b-moe-pp-20260721/elasticwave-20260723/routes-code-16x4096.awtr')
TRACE_SHA='8f78f5338327becbac14ed9c82f93b1199eb4d7bf6455fef5e3b58cc744038c3'
SOURCE=ROOT.parent/'qwen35-q4-t64-20260908/affinity-wave.cu'
SOURCE_SHA='fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47'


def tiles(m):
    """Matches current descriptor tiling: >=48 -> M64, >16 -> M32, else M16."""
    if not isinstance(m,int) or not 0<=m<=4096:
        raise ValueError('M outside capture')
    row=0
    result=[]
    while m-row>=48:
        count=min(64,m-row)
        result.append(dict(row=row,rows=count,tile_m=64))
        row+=count
    if m-row>16:
        count=min(32,m-row)
        result.append(dict(row=row,rows=count,tile_m=32))
        row+=count
    if row<m:
        result.append(dict(row=row,rows=m-row,tile_m=16))
    return result


def profile(counts,sample,layer,owner):
    result=dict(sample=sample,layer=layer,owner=owner,counts=counts,
                routed_rows=sum(counts),max_m=max(counts),active_experts=sum(m>0 for m in counts))
    mix={str(tm):dict(tiles=0,useful_rows=0,padded_rows=0) for tm in (64,32,16)}
    for m in counts:
        for tile in tiles(m):
            v=mix[str(tile['tile_m'])]
            v['tiles']+=1;v['useful_rows']+=tile['rows'];v['padded_rows']+=tile['tile_m']
    result['mix']=mix
    result['tail_work_share']=(mix['32']['useful_rows']+mix['16']['useful_rows'])/max(sum(counts),1)
    return result


def load_profiles():
    raw=TRACE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==TRACE_SHA
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
    assert raw[:8]==b'AWTRV001'
    offset=8;record=0;profiles=[]
    while offset<len(raw):
        layer,ntop,used=struct.unpack_from('<HIH',raw,offset);offset+=8
        assert layer==record%40 and ntop==4096 and used==8
        payload=struct.unpack_from('<32768H',raw,offset);offset+=65536
        counts=[0]*256
        for start in range(0,len(payload),8):
            route=payload[start:start+8]
            assert len(set(route))==8 and max(route)<256
            for expert in route:counts[expert]+=1
        assert sum(counts)==32768 and max(counts)<=4096
        for owner in range(4):
            profiles.append(profile(counts[owner*64:(owner+1)*64],record//40,layer,owner))
        record+=1
    assert offset==len(raw) and record==640
    return profiles


def enrich(p,label):
    result=dict(p,label=label)
    jobs=[]
    for local,m in enumerate(p['counts']):
        for tile in tiles(m):
            jobs.append(dict(expert_local=local,expert_runtime=p['owner']*64+local,**tile))
    assert sum(j['rows'] for j in jobs)==p['routed_rows']
    result['jobs']=jobs
    result['counts_cli']=','.join(map(str,p['counts']))
    result['projections']=[dict(name='gate_or_up',N=512,K=2048,multiplicity=2),
                           dict(name='down',N=2048,K=512,multiplicity=1)]
    return result


def generate():
    profiles=load_profiles()
    # First8 captured samples select cases; final8 remain held-out confirmation.
    select=[p for p in profiles if p['sample']<8]
    rows_median=statistics.median(p['routed_rows'] for p in select)
    tails_median=statistics.median(p['tail_work_share'] for p in select)
    central=min(select,key=lambda p:abs(p['routed_rows']/rows_median-1)+abs(p['tail_work_share']-tails_median))
    # Actual high-tail observation near p90, not synthetic mixed count vectors.
    sorted_tails=sorted(select,key=lambda p:(p['tail_work_share'],p['sample'],p['layer'],p['owner']))
    tail_target=sorted_tails[int(.9*(len(sorted_tails)-1))]['tail_work_share']
    tail=min((p for p in select if p!=central),key=lambda p:abs(p['tail_work_share']-tail_target))
    hot=max((p for p in select if p not in (central,tail)),key=lambda p:(p['max_m'],p['routed_rows']))
    totals={str(tm):dict(tiles=0,useful_rows=0,padded_rows=0) for tm in (64,32,16)}
    m_work=Counter()
    for p in profiles:
        for tm in totals:
            for key in totals[tm]:totals[tm][key]+=p['mix'][tm][key]
        for m in p['counts']:
            label='0' if m==0 else '<=64' if m<=64 else '65..128' if m<=128 else '129..512' if m<=512 else '>512'
            m_work[label]+=m
    total_rows=sum(p['routed_rows'] for p in profiles)
    assert total_rows==16*40*4096*8
    for v in totals.values():v['useful_work_percent']=100*v['useful_rows']/total_rows
    return dict(schema='w4a16.historical_q8_ragged_proxy.v1',acceptance=False,gpu_executed=False,
                trace=str(TRACE),trace_sha256=TRACE_SHA,policy_source=str(SOURCE),policy_sha256=SOURCE_SHA,
                identity_space='Captured runtime/EPLB-permuted expert IDs. Contiguous primary-owner proxy expert//64; NOT reapplied original-ID placement.',
                warning='Historical Q8 host-sort code routing, derived current-policy tiles; not captured Q4 dispatch, activations, cohort queues or cache events.',
                records=640,owner_profiles=len(profiles),total_routed_rows=total_rows,tile_mix=totals,
                m_useful_work_percent={key:100*v/total_rows for key,v in sorted(m_work.items())},
                selection='Three authentic owner observations selected only from samples0..7; samples8..15 reserved for confirmation.',
                selected=[enrich(central,'central'),enrich(tail,'p90_tail_share'),enrich(hot,'hot_expert')])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=generate()
    if args.output:
        destination=args.output.resolve()
        if destination.parent!=ROOT:raise SystemExit('Output must be in owned verification directory')
        destination.write_text(json.dumps(report,indent=2)+'\n')
    summary={k:v for k,v in report.items() if k!='selected'}
    summary['selected']=[{k:v for k,v in p.items() if k not in ('jobs','counts','counts_cli')} for p in report['selected']]
    print(json.dumps(summary,indent=2))
