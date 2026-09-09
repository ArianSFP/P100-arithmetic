from pathlib import Path
import shutil,re
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');p=r/'dual-m256-512';p.mkdir(exist_ok=True)
for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','decode.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'dual-magic-decode'/f,p/f)
s=(r/'dual-magic-decode/packed.inc').read_text().replace('half A[2][32][128]','half A[2][32][256]').replace('__launch_bounds__(256, 1)','__launch_bounds__(512, 1)').replace('a_r+64','a_r+128')
a=s.index('            const float scale =');b=s.index('        __syncthreads();',a);end=s.rfind('        }',a,b);s=s[:a]+'if(tid<256){\n'+s[a:end]+'}\n'+s[end:]
s=s.replace('if (!last) {\n                const float scale','if (!last && tid<256) {\n                const float scale')
(p/'packed.inc').write_text(s)
s=(p/'worker.cu').read_text();pos=s.index(' auto run=[&]');s=s[:pos]+''' std::vector<aw_tile_desc>ts256;for(int e=0;e<ex;++e)for(int r=0;r<m;r+=256)ts256.push_back({e,r,std::min(256,m-r)});Device<aw_tile_desc>dt256(ts256.size());dt256.put(ts256);
'''+s[pos:]
s=re.sub(r'(w8_packed_half<true,\d+><<<candidate_grid),256(>>>\(dd16.p,)dt128.p,ts128.size\(\)',r'\1,512\2dt256.p,ts256.size()',s);(p/'worker.cu').write_text(s)
(p/'run.py').write_text((p/'run.py').read_text().replace('default=2);ap.add_argument(\'--racecheck\'','default=1);ap.add_argument(\'--racecheck\''))
