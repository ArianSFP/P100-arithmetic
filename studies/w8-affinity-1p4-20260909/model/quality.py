from pathlib import Path
import struct,json,sys
import numpy as np
p=Path(__file__).resolve().parent
n=int(sys.argv[1]);files=[(p/(tag+'.logits')).open('rb') for tag in sys.argv[2:4]]
headers=[f.read(20) for f in files];assert headers[0]==headers[1] and headers[0][:8]==b'AWQLOG01'
total,rows,vocab=struct.unpack('<III',headers[0][8:]);assert total==n
loss=[0.,0.];kld=0.;identical=0;same_top=0;maxdiff=0.
for row in range(rows):
 targets=[struct.unpack('<I',f.read(4))[0] for f in files];assert targets[0]==targets[1]
 raw=[f.read(vocab*4) for f in files];identical+=raw[0]==raw[1]
 vals=[np.frombuffer(b,dtype='<f4').astype(np.float64) for b in raw];assert all(np.isfinite(v).all() for v in vals)
 logs=[]
 for j,v in enumerate(vals):
  z=v.max()+np.log(np.exp(v-v.max()).sum());log=v-z;logs.append(log);loss[j]-=log[targets[j]]
 kld+=float(np.sum(np.exp(logs[0])*(logs[0]-logs[1])))
 same_top+=int(np.argmax(vals[0])==np.argmax(vals[1]));maxdiff=max(maxdiff,float(np.max(np.abs(vals[0]-vals[1]))))
assert all(f.read()==b'' for f in files)
a,b=[float(np.exp(x/rows)) for x in loss]
q={'tokens':n,'rows':rows,'byte_identical':identical==rows,'identical_rows':identical,'t64_ppl':a,'candidate_ppl':b,'paired_ppl_delta':b-a,'mean_kld':kld/rows,'same_top_fraction':same_top/rows,'max_abs_logit_delta':maxdiff,'ppl_gate_pass':abs(b-a)<=.003}
(p/f'quality-{sys.argv[3]}.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
if not q['ppl_gate_pass']:raise SystemExit(2)
