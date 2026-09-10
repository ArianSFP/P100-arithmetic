"""CPU exact-domain checks, not a GPU speed benchmark.
All arithmetic inputs below are dyadic and exactly representable in float64.
Thus fma16 emulates one rounding of the mathematical product-plus-add.
"""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent

def fma16(a,b,c):
    return (np.asarray(a,dtype=np.float64)*np.asarray(b,dtype=np.float64)+np.asarray(c,dtype=np.float64)).astype(np.float16)

def add16(a,b):
    return (np.asarray(a,dtype=np.float64)+np.asarray(b,dtype=np.float64)).astype(np.float16)

def prmt(a,b,s):
    # PTX prmt.b32 default-mode semantics. Selectors are masked to 0..7 here.
    src=int(a)|(int(b)<<32)
    return sum(((src>>(8*((s>>(4*j))&7)))&255)<<(8*j) for j in range(4))

w0,w1,a=np.indices((16,16,16),dtype=np.int64).reshape(3,-1)-8
p=((w0+128*w1)/128).astype(np.float16)
t=fma16(p,a,1536)
hi=add16(t,-1536)
lo_scaled=fma16(p,a,-hi.astype(np.float64))
assert np.all(hi==w1*a)
assert np.all(lo_scaled.astype(np.float64)*128==w0*a)
assert np.all(p.astype(np.float64)==(w0+128*w1)/128)
result={'normalised_three_op_split':dict(cases=len(a),high_mismatches=int(np.count_nonzero(hi!=w1*a)),low_mismatches=int(np.count_nonzero(lo_scaled.astype(np.float64)*128!=w0*a)))}

P=w0+128*w1
r=fma16(P,a,0).astype(np.int64)
h=(r+63)//128
low_round=r-128*h
rho=(w0*a)&7
delta=((rho-r+4)&7)-4
lo=low_round+delta
assert np.all(lo==w0*a)
assert np.all(h==w1*a)
result['mod8_repair']=dict(cases=len(a),mismatches=int(np.count_nonzero(lo!=w0*a)),max_correction=int(abs(delta).max()))
result['uncorrected_by_activation_magnitude']=[]
for s in range(9):
 ix=abs(a)==s
 result['uncorrected_by_activation_magnitude'].append(dict(magnitude=s,cases=int(sum(ix)),mismatches=int(np.count_nonzero(low_round[ix]!=w0[ix]*a[ix])),max_absolute_error=int(np.max(abs(low_round[ix]-w0[ix]*a[ix])))))

# Test all 8 activation residues and every four-way selector 0..7.
T=[]
for ar in range(8):
 vals=[(ar*j)&7 for j in range(8)]
 T.append((sum(vals[j]<<(8*j) for j in range(4)),sum(vals[j+4]<<(8*j) for j in range(4))))
prmt_bad=0
prmt_cases=0
for ar in range(8):
 for ss in np.ndindex((8,8,8,8)):
  selector=sum(int(v)<<(4*j) for j,v in enumerate(ss))
  got=prmt(*T[ar],selector)
  expect=sum(((int(v)*ar)&7)<<(8*j) for j,v in enumerate(ss))
  prmt_bad+=got!=expect; prmt_cases+=1
assert prmt_bad==0
result['prmt_mod8_table']=dict(cases=prmt_cases,mismatches=int(prmt_bad),bytes=64)
result['prmt_tables_hex']=[dict(a_mod8=i,lo=f'0x{x:08x}',hi=f'0x{y:08x}') for i,(x,y) in enumerate(T)]

# Independently emulated full G32 dots, including all constant endpoint triples.
rng=np.random.default_rng(20260909)
count=100_000
bad=0
base_bad=0
for start in range(0,count,5000):
 n=min(5000,count-start)
 v=rng.integers(-8,8,size=(n,32,3),dtype=np.int64)
 if start==0:
  corners=np.indices((2,2,2)).reshape(3,-1).T*15-8
  v[:8]=corners[:,None,:]
 x,y,aa=(v[:,:,i] for i in range(3))
 pp=((x+128*y)/128).astype(np.float16)
 al=np.zeros(n,dtype=np.float16); ah=al.copy()
 bl=al.copy(); bh=al.copy()
 for k in range(32):
  tt=fma16(pp[:,k],aa[:,k],1536)
  hh=add16(tt,-1536)
  ll=fma16(pp[:,k],aa[:,k],-hh.astype(np.float64))
  al=add16(al,ll); ah=add16(ah,hh)
  bl=fma16(x[:,k],aa[:,k],bl); bh=fma16(y[:,k],aa[:,k],bh)
 rl=np.sum(x*aa,axis=1); rh=np.sum(y*aa,axis=1)
 bad+=np.count_nonzero((al.astype(np.float64)*128!=rl)|(ah!=rh))
 base_bad+=np.count_nonzero((bl!=rl)|(bh!=rh))
result['g32_accumulation']=dict(blocks=count,packed_split_mismatched_blocks=int(bad),ordinary_half_mismatched_blocks=int(base_bad))
assert bad==base_bad==0

# Fully exact two-step collision: extra state is necessary even without rounding.
collision=[]
for low,high in [([-8,-8],[0,0]),([6,-3],[-1,1])]:
 c=np.float16(0)
 for l,h,ak in zip(low,high,[7,6]):
  c=fma16(l+128*h,ak,c)
 collision.append(dict(low_weights=low,high_weights=high,activations=[7,6],packed_result=float(c),dot_low=int(np.dot(low,[7,6])),dot_high=int(np.dot(high,[7,6]))))
assert collision[0]['packed_result']==collision[1]['packed_result']
result['g2_collision']=collision

# Radix changes: all products are exact for B32, but naive output fields overlap.
radices=[]
for B in [16,32,64,128]:
 p1=w0+B*w1
 rr=fma16(p1,a,0).astype(np.int64)
 hh=np.rint(rr/B).astype(np.int64)
 radices.append(dict(radix=B,product_rounding_mismatches=int(np.count_nonzero(rr!=p1*a)),max_rounding_error=int(np.max(abs(rr-p1*a))),naive_high_mismatches=int(np.count_nonzero(hh!=w1*a))))
result['radix_sweep']=radices

# Witness of invalid 'fuse low accumulation into cancellation' optimization.
# The addend must not be pre-rounded with an accumulated low fractional sum.
fail=None
for acc_units in [-1001,-127,-1,1,127,1001]:
 acc=np.float16(acc_units/128)
 cc=add16(acc,-hi.astype(np.float64))
 got=fma16(p,a,cc)
 expected=add16(acc,lo_scaled)
 idx=np.flatnonzero(got!=expected)
 if len(idx):
  i=int(idx[0]); fail=dict(w0=int(w0[i]),w1=int(w1[i]),a=int(a[i]),low_accumulator=float(acc),fused_observed=float(got[i]),correct_observed=float(expected[i]),rounded_addend=float(cc[i]));break
assert fail is not None
result['invalid_accumulator_fusion_witness']=fail
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'mod8_tables.json').write_text(json.dumps(result['prmt_tables_hex'],indent=2)+'\n')
print(json.dumps(result,indent=2))
