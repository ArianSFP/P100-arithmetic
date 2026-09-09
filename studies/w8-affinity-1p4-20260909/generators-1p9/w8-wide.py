from pathlib import Path
import shutil,re
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');p=r/'dual-n256';p.mkdir(exist_ok=True)
for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'m128-pairs-dual'/f,p/f)
s=(r/'dual-interleave/packed.inc').read_text()
s=s.replace('half2 B[2][16][128]','half2 B[2][16][256]').replace('__launch_bounds__(256, 1)','__launch_bounds__(512, 1)').replace('producer_half = warp >> 2','producer_half = warp >> 3').replace('b_r = (warp & 3)*32 + lane','b_r = (warp & 7)*32 + lane').replace('n/(2*AW_T64_ROWS)','n/(4*AW_T64_ROWS)').replace('ngroup*2*AW_T64_ROWS','ngroup*4*AW_T64_ROWS').replace('2*ngroup + b_r/AW_T64_ROWS','4*ngroup + b_r/AW_T64_ROWS').replace('row0 = warp*16','row0 = (warp&7)*16')
s=s.replace('const int lane = tid & 31;','const int lane = tid & 31;\n    const int col_lane=lane+(warp>>3)*128;')
s=s.replace('[lane]', '[col_lane]').replace('[lane+32]', '[col_lane+32]').replace('[lane+64]', '[col_lane+64]').replace('[lane+96]', '[col_lane+96]').replace('output[lane+j*32]','output[col_lane+j*32]')
a=s.index('            float4 pa0, pa1;');b=s.index('            const float scale =',a);s=s[:a]+'if(tid<256){\n'+s[a:b]+'}\n'+s[b:]
s=s.replace('            if (!last) {\n                float4 next_a0','            if (!last && tid<256) {\n                float4 next_a0')
(p/'packed.inc').write_text(s)
s=(p/'worker.cu').read_text();s=re.sub(r'(w8_packed_half<true,\d+><<<candidate_grid),256',r'\1,512',s);(p/'worker.cu').write_text(s)
(p/'run.py').write_text((p/'run.py').read_text().replace('default=2);ap.add_argument(\'--racecheck\'','default=1);ap.add_argument(\'--racecheck\''))
