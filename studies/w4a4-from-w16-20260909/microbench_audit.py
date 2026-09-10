from pathlib import Path
import json,re,subprocess

p=Path(__file__).resolve().parent
sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',
                              str(p/'microbench')],text=True)
(p/'microbench.sass').write_text(sass)
blocks=re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S)
ops=('HFMA2','FFMA','DFMA','XMAD','IADD','ISETP','BRA','LDG','STG','LDL','STL')
result={}
for name,body in blocks:
    if not name.startswith('bench_'):continue
    ins=[line for line in body.splitlines() if re.search(r'/\*[0-9a-f]+\*/',line,re.I)]
    result[name]={'instructions':len(ins)}
    result[name].update({op:sum(re.search(r'\b'+op+r'(?:\.|\b)',line) is not None
                                for line in ins) for op in ops})
assert len(result)==5,result
assert all(v['LDL']==0 and v['STL']==0 for v in result.values()),result
(p/'microbench-sass-counts.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
