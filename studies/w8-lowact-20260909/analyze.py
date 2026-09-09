import hashlib,json,re,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
manifest=json.loads((p/'manifest.json').read_text())
for f,h in manifest['hashes'].items():
    assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h,f
rows=[]
for proj in ['gu','down']:
    for i in range(3):
        tag=f'{proj}-r{i}'
        meta=json.loads((p/f'{tag}.meta.json').read_text())
        assert meta['exit_code']==0 and meta['build']==manifest
        raw=(p/f'{tag}.out').read_text()
        assert 'PASS' in raw
        assert len(re.findall(r'QUANT bits=\d+ values=\d+ bad=0',raw))==2
        checks=re.findall(r'CHECK A(\d+) outputs=(\d+) samples=(\d+) bad=(\d+) nonfinite=(\d+) changed=(\d+) rel_l2=([\d.e+-]+)',raw)
        assert len(checks)==4
        for bits,n,s,bad,nf,changed,l2 in checks:
            assert int(bad)==int(nf)==0
            if bits in ['32','16']: assert int(changed)==0
        samples={}
        for bits,prep,rep,us in re.findall(r'TIME A(\d+) prep=(\d+) rep=(\d+) us=([\d.]+)',raw):
            if int(rep)>0: samples.setdefault(f'A{bits}_prep{prep}',[]).append(float(us))
        assert len(samples)==6 and all(len(v)==6 for v in samples.values())
        med={k:statistics.median(v) for k,v in samples.items()}
        best=min(med['A32_prep1'],med['A16_prep1'])
        rows.append(dict(tag=tag,device=meta['device'],median_us=med,best_q8_us=best,speedup={k:best/v for k,v in med.items()},rel_l2={b:float(l) for b,_,_,_,_,_,l in checks}))
summary={}
for proj in ['gu','down']:
    rr=[r for r in rows if r['tag'].startswith(proj)]
    summary[proj]={'median_us':{k:statistics.median(r['median_us'][k] for r in rr) for k in rr[0]['median_us']},'best_q8_us':statistics.median(r['best_q8_us'] for r in rr),'paired_speedup':{k:{'median':statistics.median(r['speedup'][k] for r in rr),'min':min(r['speedup'][k] for r in rr),'max':max(r['speedup'][k] for r in rr)} for k in rr[0]['median_us']},'rel_l2':rr[0]['rel_l2']}
out={'round_zero_excluded':True,'workers':rows,'summary':summary}
(p/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
