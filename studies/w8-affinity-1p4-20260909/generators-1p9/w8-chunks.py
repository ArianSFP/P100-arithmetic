from pathlib import Path
import shutil
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');base=(r/'dual-raw-f16/packed.inc').read_text()
def block(s,a):
 op=s.index('{',a);d=1;b=op+1
 while d:d+=(s[b]=='{')-(s[b]=='}');b+=1
 return op,b
start=base.index('            #pragma unroll\n            for (int group = 0; group < 4; ++group)');op,end=block(base,start)
math=base[start:end];a=math.index('                    const int ks =');b=math.index('                    const int ks =',a+1);second_open=math.rfind('{',a,b)
# First scalar step ends immediately before the standalone scope for the upper rail.
first=math[a:second_open];assert 'acc0' in first and 'acc1' not in first
# Strip no braces: first includes its i-loop closure only.
for chunk in [2,4,8]:
 p=r/f'dual-raw-chunk{chunk}';p.mkdir(exist_ok=True)
 for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','decode.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'dual-raw-f16'/f,p/f)
 step=first.replace('group*4 + kk',f'group*{chunk} + kk')
 repl=f'''            #pragma unroll
            for(int group=0;group<{16//chunk};++group){{
                #pragma unroll
                for(int kk=0;kk<{chunk};++kk){{
'''+step+f'''                }}
                #pragma unroll
                for(int kk=0;kk<{chunk};++kk){{
'''+step.replace(f'group*{chunk} + kk;',f'group*{chunk} + kk + 16;').replace('acc0','acc1')+'                }\n            }'
 (p/'packed.inc').write_text(base[:start]+repl+base[end:])
