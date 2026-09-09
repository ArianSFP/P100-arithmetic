from pathlib import Path
import shutil,re
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');p=r/'dual-sync-n64';p.mkdir(exist_ok=True)
for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'dual-sync-c2'/f,p/f)
s=(r/'dual-sync-c2/packed.inc').read_text().replace('half2 B[1][16][128]','half2 B[1][16][64]').replace('__launch_bounds__(256, 2)','__launch_bounds__(256, 3)').replace('producer_half = warp >> 2','producer_half = warp >> 1').replace('b_r = (warp & 3)*32 + lane','b_r = (warp & 1)*32 + lane').replace('n/(2*AW_T64_ROWS)','n/AW_T64_ROWS').replace('ngroup*2*AW_T64_ROWS','ngroup*AW_T64_ROWS').replace('2*ngroup + b_r/AW_T64_ROWS','ngroup').replace('acc0[8][4],acc1[8][4]','acc0[8][2],acc1[8][2]').replace('total[16][4]','total[16][2]').replace('j<4','j<2').replace('j < 4','j < 2')
s=s.replace('producer_half*16','producer_half*8').replace('const uint4 packed4','const uint2 packed4').replace('__ldg((const uint4 *) values)','__ldg((const uint2 *) values)').replace('const uint32_t packed[4] = {\n                packed4.x, packed4.y,\n                packed4.z, packed4.w\n            };','const uint32_t packed[2] = {packed4.x, packed4.y};').replace('for(int group=0;group<4;++group)','for(int group=0;group<2;++group)').replace('producer_half*8+group*2','producer_half*4+group*2')
# Preserve the first two column loads in each math step.
pat=r'const half2 bv\[4\]=\{([^;]+)\};'
def smaller(m):
 vals=m[1].split(',');assert len(vals)==4;return 'const half2 bv[2]={'+','.join(vals[:2])+'};'
s,n=re.subn(pat,smaller,s);assert n==2
(p/'packed.inc').write_text(s)
(p/'run.py').write_text((p/'run.py').read_text().replace('default=2);ap.add_argument(\'--racecheck\'','default=3);ap.add_argument(\'--racecheck\''))
