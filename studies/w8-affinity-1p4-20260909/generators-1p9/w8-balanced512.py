from pathlib import Path
import shutil,re
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909')
for name,src in [('dual-m256-balanced','dual-m256-512-raw'),('dual-512-balanced','dual-512')]:
 p=r/name;p.mkdir(exist_ok=True)
 for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','decode.inc','build.py','run.py','analyze.py']:shutil.copy2(r/src/f,p/f)
 s=(r/src/'packed.inc').read_text()
 # Remove guard braces around initial weight producer only.
 a=s.index('if(tid<256){');op=s.index('{',a);d=1;b=op+1
 while d:d+=(s[b]=='{')-(s[b]=='}');b+=1
 s=s[:a]+s[op+1:b-1]+s[b:]
 s=s.replace('if (!last && tid<256)','if (!last)').replace('producer_half*16','producer_half*8').replace('producer_half*8+group*2','producer_half*4+group*2').replace('const uint4 packed4','const uint2 packed4').replace('__ldg((const uint4 *) values)','__ldg((const uint2 *) values)')
 s=re.sub(r'const uint32_t packed\[4\] = \{\s*packed4.x, packed4.y,\s*packed4.z, packed4.w\s*\};','const uint32_t packed[2] = {packed4.x,packed4.y};',s)
 s=s.replace('for(int group=0;group<4;++group)','for(int group=0;group<2;++group)')
 if src=='dual-512':
  # Replace its two float activation-load blocks with direct FP16 staging.
  helper=(r/'dual-raw-f16/packed.inc').read_text().split('#include "decode.inc"')[0]
  a=s.index('            float4 pa0, pa1;');b=s.index('            const float scale =',a)
  s=s[:a]+'''            half h[8];w8_load_fp16_8(desc.input,input_stage_base,h);
            #pragma unroll
            for(int i=0;i<8;++i)sm.A[buffer][a_k+i][a_r]=h[i];
'''+s[b:]
  a=s.index('            if (!last) {');op=s.index('{',a);d=1;b=op+1
  while d:d+=(s[b]=='{')-(s[b]=='}');b+=1
  s=s[:a]+'''            if(!last){half h[8];w8_load_fp16_8(desc.input,input_stage_base+AW_KSTAGE,h);
            #pragma unroll
            for(int i=0;i<8;++i)sm.A[buffer^1][a_k+i][a_r]=h[i];}
'''+s[b:]
  s=helper+s
 (p/'packed.inc').write_text(s)
