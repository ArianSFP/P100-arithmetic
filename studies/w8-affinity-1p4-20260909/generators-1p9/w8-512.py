from pathlib import Path
import shutil,re
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');src=r/'dual-magic-decode';p=r/'dual-512';p.mkdir(exist_ok=True)
for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','decode.inc','build.py','run.py','analyze.py']:shutil.copy2(src/f,p/f)
s=(src/'packed.inc').read_text()
def block(s,a):
 op=s.index('{',a);d=1;b=op+1
 while d:d+=(s[b]=='{')-(s[b]=='}');b+=1
 return op,b
# Remove second activation-row load: 512 threads cover all 128 rows directly.
for needle in ['            {\n            const int extra_local','                {\n                const int extra_local']:
 a=s.index(needle);op,b=block(s,a);s=s[:a]+s[b:]
a=s.index('            const float scale =');b=s.index('        __syncthreads();',a);end=s.rfind('        }',a,b);s=s[:a]+'if(tid<256){\n'+s[a:end]+'}\n'+s[end:]
s=s.replace('if (!last) {\n                const float scale','if (!last && tid<256) {\n                const float scale')
s=s.replace('__launch_bounds__(256, 1)','__launch_bounds__(512, 2)').replace('acc0[8][4],acc1[8][4]','acc0[4][4],acc1[4][4]').replace('total[16][4]','total[8][4]').replace('for(int i=0;i<8;++i)','for(int i=0;i<4;++i)').replace('row0 = warp*16','row0 = warp*8').replace('i < 16','i < 8')
(p/'packed.inc').write_text(s)
s=(p/'worker.cu').read_text();s=re.sub(r'(w8_packed_half<true,\d+><<<candidate_grid),256',r'\1,512',s);(p/'worker.cu').write_text(s)
