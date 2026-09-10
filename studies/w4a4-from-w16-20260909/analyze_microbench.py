from pathlib import Path
import json,re,statistics

p=Path(__file__).resolve().parent
raw=(p/'priority1-throughput.out').read_text()
rows={}
for mode,rep,us,rate in re.findall(
        r'TIME mode=(\S+) rep=(\d+) us=([0-9.]+) useful_tmac_s=([0-9.]+)',raw):
    if int(rep)==0:continue
    rows.setdefault(mode,[]).append((float(us),float(rate)))
assert set(rows)=={'hfma2','xmad-s16-pack2','fp32-pack2','dfma-pack4','mixed-hfma2-dfma'}
assert all(len(v)==8 for v in rows.values())
summary={mode:{'median_us':statistics.median(x[0] for x in values),
               'median_useful_tmac_s':statistics.median(x[1] for x in values)}
         for mode,values in rows.items()}
base=summary['hfma2']['median_useful_tmac_s']
for values in summary.values():values['throughput_vs_hfma2']=values['median_useful_tmac_s']/base
summary['mixed-hfma2-dfma']['overlap_efficiency_vs_separate']=(
    (summary['hfma2']['median_us']+summary['dfma-pack4']['median_us'])/
    summary['mixed-hfma2-dfma']['median_us'])
(p/'priority1-throughput-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
