from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent;src=p.parent/'q4-halfpipe-20260908/build/q8.inc';s=src.read_text();(p/'q8.inc').write_text(s)
k=s[s.index('template<bool BF16_INPUT>\n__global__'):].replace('template<bool BF16_INPUT>','template<int ABITS>').replace('aw_q8_service_m64_n128_halfpipe_sync','w8_lowact')
a=k.index('            if constexpr (BF16_INPUT) {');b=k.index('            const float * a0',a)
k=k[:a]+'            load_activation<ABITS>(desc.input,input_stage_base,pa0,pa1);\n'+k[b:]
a=k.index('                if constexpr (BF16_INPUT) {');b=k.index('                const float * na0',a)
k=k[:a]+'                load_activation<ABITS>(desc.input,next_a_base,next_a0,next_a1);\n'+k[b:]
(p/'lowact.inc').write_text(k)
(p/'provenance.json').write_text(json.dumps({'original_q8':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'change':'Only input loads replaced; weight loading, FP32 accumulators, tile geometry and K order retained.'},indent=2)+'\n')
