from pathlib import Path
import shutil
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');src=r/'dual-interleave'
s=(src/'packed.inc').read_text()
def block(text,start):
 a=text.index('{',start);depth=1;b=a+1
 while depth:depth+=(text[b]=='{')-(text[b]=='}');b+=1
 return a,b
start=s.index('        int buffer = 0;');init_a,init_b=block(s,start)
init=s[init_a+1:init_b-1]
a=s.index('            const int row0 = warp*16;');b=s.index('            if constexpr(FLUSH>0)',a);math=s[a:b]
a2,b2=block(s,b);flush=s[b:b2]
out=s[s.index('        const int row0 = warp*16;',b2):]
prefix=s[:start]
# All staging now occurs synchronously before math; no ping-pong buffers.
prefix=prefix.replace('half A[2][32][128];half2 B[2][16][128]','half A[1][32][128];half2 B[1][16][128]')
body=prefix+'        const int buffer=0;\n        for(int stage=0;stage<n_kblocks;++stage){\n            const bool last=stage+1==n_kblocks;\n'+init+'\n__syncthreads();\n'+math+flush+'\n__syncthreads();\ninput_stage_base+=AW_KSTAGE;weight_stage+=AW_T64_STAGE_BYTES;\n}\n'+out
for ct in [2,3]:
 p=r/f'dual-sync-c{ct}';p.mkdir(exist_ok=True)
 for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'m128-pairs-dual'/f,p/f)
 (p/'packed.inc').write_text(body.replace('__launch_bounds__(256, 1)',f'__launch_bounds__(256, {ct})'))
 (p/'run.py').write_text((p/'run.py').read_text().replace('default=2);ap.add_argument(\'--racecheck\'',f'default={ct});ap.add_argument(\'--racecheck\''))
