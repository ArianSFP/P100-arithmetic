from pathlib import Path
import json,re,subprocess
p=Path(__file__).resolve().parent
sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(p/'worker')],text=True)
blocks={name:body for name,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S)}
selected={}
for family in ('w8_pair_half','w16_pair_half','w8_fallback_half','w16_fallback_half'):
    for name,body in blocks.items():
        if family not in name:continue
        if 'pair_half' in family and not ('Li512ELi2048' in name or 'Li2048ELi512' in name):continue
        instructions=[line for line in body.splitlines() if re.search(r'/\*[0-9a-f]+\*/',line,re.I)]
        selected[name]={
            'instructions':len(instructions),
            'hfma2':sum(' HFMA2' in line for line in instructions),
            'ldg_128':sum('LDG.' in line and '.128 ' in line for line in instructions),
            'ldl':sum(' LDL' in line for line in instructions),
            'stl':sum(' STL' in line for line in instructions),
        }
assert len(selected)==6,selected.keys()
assert all(v['hfma2']>0 and v['ldl']==0 and v['stl']==0 for v in selected.values())
(p/'sass-audit.json').write_text(json.dumps(selected,indent=2)+'\n')
print(json.dumps(selected,indent=2))
