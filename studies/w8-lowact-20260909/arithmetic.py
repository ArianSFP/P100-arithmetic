"""Integer range and binary16 exactness audit; no GPU claims."""
from pathlib import Path
import struct,json
half=lambda x:struct.unpack('e',struct.pack('e',float(x)))[0]
rows={}
for abits,amax in [(4,7),(8,127)]:
 changed=sum(half(w*a)!=w*a for w in range(-128,128) for a in range(-amax,amax+1))
 rows[str(abits)]={'activation_integer_range':[-amax,amax],'weight_integer_range':[-128,127],'g32_abs_sum_bound':32*128*amax,'fp32_integer_dot_exact_g32':32*128*amax<=2**24,'fp16_inexact_products':changed,'product_cases':256*(2*amax+1)}
rows['w8a4_counterexample']={'terms':[127*7]*3,'integer_sum':3*127*7,'half_sequential_sum':half(half(half(127*7)+127*7)+127*7)}
rows['w8a4_exact_nibble_split']={'equation':'w=(w & 15)+16*(w >> 4)','g16_low_abs_sum_bound':16*15*7,'g16_high_abs_sum_bound':16*8*7,'note':'Both integer partials exact in half, but two nibble FMAs per original product consume one half2 FMA instruction: no automatic instruction-count reduction versus one FP32 FMA.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
