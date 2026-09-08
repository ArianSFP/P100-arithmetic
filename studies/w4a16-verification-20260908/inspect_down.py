"""Read-only existing SASS/timing attribution; no compiler/GPU/new measurements."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent.parent/'w4a16-long-prefill-20260908'


def main():
    path=ROOT/'build/worker.sass'
    text=path.read_text()
    print('SASS_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    sections=re.split(r'Function\s*:\s*(\S+)',text)
    for name,body in zip(sections[1::2],sections[2::2]):
        if not (name.startswith('_Z10compact_q4ILi32ELb1') or name.startswith('_Z10tile256_q4') or name.startswith('_Z36aw_q8_service_m64_n128_halfpipe_sync')):
            continue
        inst=[]
        for addr,statement in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',body):
            words=statement.split()
            if not words:continue
            op=words[1] if words[0].startswith('@') else words[0]
            inst.append((int(addr,16),op,statement))
        counts=Counter(op for _,op,_ in inst)
        selected={op:n for op,n in counts.items() if op.startswith(('FFMA','FADD','F2F','HMUL','HADD','LDG','STG','LDS','STS','BAR','BRA','SHR','SHL','XMAD','IDP','IADD'))}
        branches=[]
        for addr,op,statement in inst:
            if op.startswith('BRA'):
                target=re.search(r'0x([0-9a-f]+)',statement)
                if target and int(target[1],16)<addr:
                    start=int(target[1],16)
                    loop=Counter(o for a,o,_ in inst if start<=a<=addr)
                    branches.append(dict(start=hex(start),end=hex(addr),static_instructions=sum(loop.values()),
                                         counts={o:n for o,n in loop.items() if o.startswith(('FFMA','FADD','F2F','HMUL','LDG','STG','LDS','STS','BAR','BRA','XMAD','IADD'))}))
        print(json.dumps(dict(function=name,whole_static=sum(counts.values()),selected=selected,backward_intervals=branches)))
    for tag in ('gpu2-compact-cap-gu-m128','gpu2-compact-cap-down-m128','gpu2-compact-cap-gu-m256',
                'gpu2-compact-cap-down-m256','gpu2-tile256-cap3-gu-m256','gpu2-tile256-cap3-down-m256'):
        folder=ROOT/'gpu-results'/tag
        result=json.loads((folder/'result.json').read_text())
        assert result['status']=='PASS'
        output=(folder/'stdout.txt').read_text()
        timings={}
        for name,us in re.findall(r'TIME (\S+) prep=1 rep=\d+ us=(\S+)',output):
            timings.setdefault(name,[]).append(float(us))
        med={n:statistics.median(v) for n,v in timings.items()}
        chosen='tile256_cap3' if 'tile256' in tag else 'compact_packed_u32'
        print(json.dumps(dict(tag=tag,control_us=med['aw_current_q4'],candidate_us=med[chosen],
                              ratio=med['aw_current_q4']/med[chosen],goal_us=med['aw_current_q4']/1.5,
                              required_latency_cut=1-med['aw_current_q4']/1.5/med[chosen])))


if __name__=='__main__':main()
