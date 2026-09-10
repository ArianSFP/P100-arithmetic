from pathlib import Path
import json,re,subprocess

p=Path(__file__).resolve().parent
opcodes=(
    'HFMA2','HMUL2','HADD2','FFMA','FMUL','DFMA','DMUL','XMAD','IMAD','IADD',
    'ISCADD','LOP','SHR','SHL','BFE','PRMT','F2I','I2F','LDG','STG','LDS','STS',
    'LDL','STL')
result={}
for level in (1,2):
    cubin=p/'probes'/f'level{level}-probes.cubin'
    sass=subprocess.check_output(
        ['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(cubin)],text=True)
    (p/'probes'/f'level{level}-probes.sass').write_text(sass)
    blocks=re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S)
    for name,body in blocks:
        instructions=[line for line in body.splitlines()
                      if re.search(r'/\*[0-9a-f]+\*/',line,re.I)]
        counts={op:sum(re.search(r'\b'+op+r'(?:\.|\b)',line) is not None
                       for line in instructions) for op in opcodes}
        result[name]={'level':level,'instructions':len(instructions),
                      'register_local':counts}
        if name in ('probe_shared32','probe_independent8','probe_half2',
                    'probe_split','probe_split_mac','probe_plain_mac',
                    'probe_prmt','probe_one_multiply'):
            result[name]['interesting_lines']=[
                line.strip() for line in instructions
                if any(re.search(r'\b'+op+r'(?:\.|\b)',line) for op in opcodes)
            ]
(p/'probes'/'sass-counts.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({name:{'instructions':v['instructions'],
                        **{k:n for k,n in v['register_local'].items() if n}}
                  for name,v in result.items()},indent=2))
