from pathlib import Path
import shutil,re
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');p=r/'dual-magic-decode';p.mkdir(exist_ok=True)
for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'m128-pairs-dual'/f,p/f)
s=(r/'m128-fused-decode/decode.inc').read_text();a=s.index('__device__ __forceinline__ void w8_decode4');b=s.index('__global__ void w8_check_decode',a)
s=s[:a]+'''__device__ __forceinline__ void w8_decode4(unsigned q,float scale,half2&lo,half2&hi){
 unsigned biased=q^0x80808080u;
 half2 a=w8_from_bits(__byte_perm(biased,0x64646464u,0x5140));
 half2 b=w8_from_bits(__byte_perm(biased,0x64646464u,0x5342));
 half2 offset=__float2half2_rn(1152.f),d=__float2half2_rn(scale);
 lo=__hmul2(__hsub2(a,offset),d);hi=__hmul2(__hsub2(b,offset),d);
}
'''+s[b:];(p/'decode.inc').write_text(s)
s=(r/'dual-interleave/packed.inc').read_text()
pat=r'sm.B\[buffer\^?1?\]?'
lines=s.splitlines();res=[];i=0
while i<len(lines):
 if 'sm.B[' in lines[i] and '=__floats2half2_rn' in lines[i]:
  first=lines[i].split('=')[0].strip();second=lines[i+1].split('=')[0].strip();res.append(f'                w8_decode4(q,scale,{first},{second});');i+=2
 else:res.append(lines[i]);i+=1
(p/'packed.inc').write_text('#include "decode.inc"\n'+'\n'.join(res)+'\n')
s=(p/'worker.cu').read_text();old=(r/'m128-fused-decode/worker.cu').read_text();check=next(x for x in old.splitlines() if 'unsigned *decode_bad' in x);pos=s.index(' size_t ac=');s=s[:pos]+check+'\n'+s[pos:];(p/'worker.cu').write_text(s)
s=(p/'build.py').read_text().replace("'packed.inc',","'packed.inc','decode.inc',");(p/'build.py').write_text(s)
