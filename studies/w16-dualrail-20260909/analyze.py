from pathlib import Path
import hashlib,json,re,statistics
p=Path(__file__).resolve().parent
manifest=json.loads((p/'manifest.json').read_text())
for name,want in manifest['hashes'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,name
workers=[]
for projection in ('gu','down'):
    for repetition in range(3):
        tag=f'{projection}-r{repetition}';raw=(p/f'{tag}.out').read_text();meta=json.loads((p/f'{tag}.meta.json').read_text())
        assert meta['exit_code']==0 and raw.rstrip().endswith('PASS'),tag
        assert raw.count('bad=0 nonfinite=0')==3 and 'mode=w16-dualrail' in raw and 'diff_vs_w8=0' in raw,tag
        times={mode:[] for mode in ('original-q8','w8-dualrail','w16-dualrail')}
        for mode,us in re.findall(r'TIME mode=(\S+) rep=\d+ us=([0-9.]+)',raw):times[mode].append(float(us))
        assert all(len(v)==7 for v in times.values()),tag
        med={mode:statistics.median(v) for mode,v in times.items()}
        workers.append({'tag':tag,'device':meta['device'],'us':med,
            'w16_vs_w8':med['w8-dualrail']/med['w16-dualrail'],
            'w16_vs_original':med['original-q8']/med['w16-dualrail']})
summary={}
for projection in ('gu','down'):
    rows=[w for w in workers if w['tag'].startswith(projection+'-')]
    summary[projection]={
        'original_q8_us':statistics.median(w['us']['original-q8'] for w in rows),
        'w8_dualrail_us':statistics.median(w['us']['w8-dualrail'] for w in rows),
        'w16_dualrail_us':statistics.median(w['us']['w16-dualrail'] for w in rows),
        'w16_vs_w8':statistics.median(w['w16_vs_w8'] for w in rows),
        'w16_vs_original':statistics.median(w['w16_vs_original'] for w in rows),
        'w16_vs_w8_range':[min(w['w16_vs_w8'] for w in rows),max(w['w16_vs_w8'] for w in rows)]}
screens=[]
for tokens in (64,128,512):
    for projection in ('gu','down'):
        tag=f'screen-m{tokens}-{projection}';raw=(p/f'{tag}.out').read_text();meta=json.loads((p/f'{tag}.meta.json').read_text())
        assert meta['exit_code']==0 and raw.rstrip().endswith('PASS'),tag
        assert raw.count('bad=0 nonfinite=0')==3 and 'mode=w16-dualrail' in raw and 'diff_vs_w8=0' in raw,tag
        times={mode:[] for mode in ('original-q8','w8-dualrail','w16-dualrail')}
        for mode,us in re.findall(r'TIME mode=(\S+) rep=\d+ us=([0-9.]+)',raw):times[mode].append(float(us))
        assert all(len(v)==7 for v in times.values()),tag
        med={mode:statistics.median(v) for mode,v in times.items()}
        screens.append({'tag':tag,'tokens':tokens,'projection':projection,'device':meta['device'],'us':med,
            'w16_vs_w8':med['w8-dualrail']/med['w16-dualrail'],
            'w16_vs_original':med['original-q8']/med['w16-dualrail']})
out={'workers':workers,'summary':summary,'screens':screens}
(p/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'summary':summary,'screens':screens},indent=2))
