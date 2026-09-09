from pathlib import Path
import shutil
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');base=(r/'dual-magic-decode/packed.inc').read_text();typed=(r/'m128-typed-input/packed.inc').read_text()
def block(s,a):
 op=s.index('{',a);d=1;b=op+1
 while d:d+=(s[b]=='{')-(s[b]=='}');b+=1
 return op,b
start=base.index('            float4 pa0, pa1;');end=base.index('            const float scale =',start)
a=typed.index('            half h[8]');b=typed.index('            const float scale =',a);base=base[:start]+typed[a:b]+base[end:]
a=base.index('            if (!last) {');op,b=block(base,a);c=typed.index('            if (!last) {');op,d=block(typed,c);base=base[:a]+typed[c:d]+base[b:]
helper=typed[:typed.index('struct w8_half_smem')]
for name,generic in [('dual-raw-f16',False),('dual-raw-generic',True)]:
 p=r/name;p.mkdir(exist_ok=True)
 for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','decode.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'dual-magic-decode'/f,p/f)
 s=helper+base
 if generic:
  s=s.replace('__device__ __forceinline__ void w8_load_fp16_8','template<bool BF16_INPUT> __device__ __forceinline__ void w8_load_fp16_8').replace('w8_load_fp16_8(desc.input','w8_load_fp16_8<BF16_INPUT>(desc.input')
  needle=' const uint4 v=';pos=s.index(needle);s=s[:pos]+''' if constexpr(!BF16_INPUT){
  #pragma unroll
  for(int i=0;i<8;++i)h[i]=__float2half_rn(((const float*)input)[index+i]);
  return;
 }
'''+s[pos:]
  s=s.replace('h[i]=__ushort_as_half((w[i/2]>>((i%2)*16))&65535);','{unsigned u=(w[i/2]>>((i%2)*16))&65535;h[i]=((uintptr_t)input&1)?__ushort_as_half(u):__float2half_rn(__uint_as_float(u<<16));}')
 (p/'packed.inc').write_text(s)
