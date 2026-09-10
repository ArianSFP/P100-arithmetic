"""CPU numerical investigation, NOT a GPU benchmark.
All floating simulations round explicitly; integer tests use a wide reference.
Requires Python 3.10+ and NumPy. Run: python explore.py
"""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
RNG=np.random.default_rng(20260909)
RESULTS={}

def record(name, **kw):
    RESULTS[name]=kw
    print(name, json.dumps(kw),flush=True)

def fma_sim(a,b,c,dtype):
    # Inputs here are small integers or FP16/FP32 values with products and sums
    # exactly representable in FP64. Therefore this emulates one final rounding.
    return (a.astype(np.float64)*b.astype(np.float64)+c.astype(np.float64)).astype(dtype)

# A. Exhaustively test the originally highlighted identity.
w0,w1,a0,a1=np.indices((16,16,16,16),dtype=np.int64).reshape(4,-1)
z=(w0+512*w1)*(a1+512*a0)
t=w0*a0+w1*a1
record('unsigned_two_term_base512', cases=len(t), mismatches=int(np.count_nonzero(((z>>9)&511)!=t)))

# B. Signed two-term, same radix construction but explicit borrow/offset repair.
w0-=8;w1-=8;a0-=8;a1-=8
B=2048
z=(w0+B*w1)*(a1+B*a0)
t=w0*a0+w1*a1
u=(z+56+B*112)&0xffffffff
got=((u>>11)&2047)-112
record('signed_two_term_base2048',cases=len(t),mismatches=int(np.count_nonzero(got!=t)))

# C. Two distinct signed INT4 products with the SAME activation in one half.
x,y,a=np.indices((16,16,16),dtype=np.int64).reshape(3,-1)-8
p=x+128*y
z=(p.astype(np.float64)*a).astype(np.float16).astype(np.float64).astype(np.int64)
h=(z+63)//128
l=z-128*h
err=l-x*a
# Repair via a known-activation / rounded-residual table; verify consistency.
lut=np.full((16,128),999,dtype=np.int16)
collisions=0
for ai,li,target in zip(a,l,x*a):
    ii,jj=int(ai+8),int(li+63)
    if not 0<=jj<128:
        raise AssertionError('low digit outside repair table')
    old=lut[ii,jj]
    if old!=999 and old!=target:
        collisions+=1
    lut[ii,jj]=target
fixed=lut[a+8,l+63]
record('fp16_shared_single_product',cases=len(a),pack_input_errors=int(np.count_nonzero(p.astype(np.float16).astype(np.int64)!=p)),high_product_mismatches=int(np.count_nonzero(h!=y*a)),low_product_mismatches=int(np.count_nonzero(err)),low_product_max_absolute_error=int(abs(err).max()),low_product_mse=float(np.mean(err.astype(float)**2)),repair_table_conflicts=collisions,repair_mismatches=int(np.count_nonzero(fixed!=x*a)))
np.save(ROOT/'fp16_shared_repair_lut.npy',lut)

# D. Monte Carlo blocks plus exhaustive constant endpoint blocks.
N=100_000
counts=dict(kpair=0,independent=0,shared_int=0,shared_fp32=0,half_g32=0,half_g33=0)
checked=dict(kpair=0,independent=0,shared_int=0,shared_fp32=0,half_g32=0,half_g33=0)
for start in range(0,N,5000):
    n=min(5000,N-start)
    # 4 values each iteration, accumulated through 8 iterations.
    v=RNG.integers(-8,8,size=(n,8,4),dtype=np.int64)
    if start==0:
        corners=np.indices((2,2,2,2)).reshape(4,-1).T*15-8
        v[:16]=corners[:,None,:]
    w0,w1,a0,a1=[v[:,:,i] for i in range(4)]
    p=w0+2048*w1
    q=a1+2048*a0
    c=np.sum(p*q,axis=1)
    got=((((c+56*8+2048*112*8)&0xffffffff)>>11)&2047)-112*8
    ref=np.sum(w0*a0+w1*a1,axis=1)
    counts['kpair']+=int(np.count_nonzero(got!=ref)); checked['kpair']+=n
    # Non-reversed Q -> low and high contain TWO INDEPENDENT dots.
    q=a0+2048*a1
    c=np.sum(p*q,axis=1)
    bias=56*8+2048*112*8+(2048**2)*56*8
    u=(c+bias)&0xffffffff
    s0=(u&2047)-56*8
    s1=((u>>22)&1023)-56*8
    ref0=np.sum(w0*a0,axis=1);ref1=np.sum(w1*a1,axis=1)
    counts['independent']+=int(np.count_nonzero((s0!=ref0)|(s1!=ref1)));checked['independent']+=n
    # Shared activation, full 32-term quantization group.
    v=RNG.integers(-8,8,size=(n,33,3),dtype=np.int64)
    if start==0:
        corners=np.indices((2,2,2)).reshape(3,-1).T*15-8
        v[:8]=corners[:,None,:]
    x,y,a=[v[:,:32,i] for i in range(3)]
    # A constant +2048 centres the high weight nibble so ALL packed weights
    # fit a SIGNED 16-bit input despite radix 4096.
    p=x+4096*y+2048
    assert p.min()>=-32768 and p.max()<=32767
    c=np.sum(p*a,axis=1)-2048*np.sum(a,axis=1)
    u=c+56*32
    out0=(u&4095)-56*32
    out1=u>>12
    ref0=np.sum(x*a,axis=1);ref1=np.sum(y*a,axis=1)
    counts['shared_int']+=int(np.count_nonzero((out0!=ref0)|(out1!=ref1)));checked['shared_int']+=n
    cf=np.zeros(n,dtype=np.float32)
    pf=p.astype(np.float32);af=a.astype(np.float32)
    for k in range(32):
        cf=fma_sim(pf[:,k],af[:,k],cf,np.float32)
    ci=(cf.astype(np.float64)-2048*np.sum(a,axis=1)).astype(np.int64)
    counts['shared_fp32']+=int(np.count_nonzero(ci!=c)); checked['shared_fp32']+=n
    # Native un-packed half2 lanes: each lane accumulates one integer dot.
    h0=np.zeros(n,dtype=np.float16);h1=h0.copy()
    for k in range(32):
        h0=fma_sim(x[:,k],a[:,k],h0,np.float16)
        h1=fma_sim(y[:,k],a[:,k],h1,np.float16)
    counts['half_g32']+=int(np.count_nonzero((h0.astype(np.int64)!=ref0)|(h1.astype(np.int64)!=ref1)));checked['half_g32']+=n
    h0=fma_sim(v[:,32,0],v[:,32,2],h0,np.float16)
    h1=fma_sim(v[:,32,1],v[:,32,2],h1,np.float16)
    r0=ref0+v[:,32,0]*v[:,32,2];r1=ref1+v[:,32,1]*v[:,32,2]
    counts['half_g33']+=int(np.count_nonzero((h0.astype(np.int64)!=r0)|(h1.astype(np.int64)!=r1)));checked['half_g33']+=n
for name in counts:
    record(name,blocks=checked[name],mismatched_blocks=counts[name])
# Deliberate first failure beyond the universal G32 guarantee: 32*64 + 1.
h=np.float16(0)
for _ in range(32): h=np.float16(float(h)+64)
h=np.float16(float(h)+1)
record('half_g33_adversarial',expected=2049,observed=int(h))

# E. Explore radix budgets analytically for *shared* activation, full range.
# Requirement: each sum in [-56G,64G], biased digit width 120G < B.
# Bound for exact floating arithmetic is G*64*sum(B**j) <= 2**precision.
budgets=[]
for precision,label in [(11,'FP16'),(24,'FP32'),(53,'FP64')]:
    for G in [1,2,3,4,7,8,16,32,64]:
        B=1<<int(120*G).bit_length()
        best=0
        for lanes in range(1,10):
            worst=64*G*sum(B**j for j in range(lanes))
            if worst<=2**precision: best=lanes
        budgets.append(dict(type=label,group=G,radix=B,guaranteed_packed_outputs=best))
record('radix_budget_search',rows=budgets)

# F. FP32 three shared outputs, G2; FP64 four shared outputs, G32.
for label,dtype,B,G,R in [('fp32_three_outputs_g2',np.float32,256,2,3),('fp64_four_outputs_g32',np.float64,4096,32,4),('fp64_seven_outputs_g1',np.float64,128,1,7)]:
    n=100_000
    bad=0
    for start in range(0,n,5000):
        m=min(5000,n-start)
        w=RNG.integers(-8,8,size=(m,G,R),dtype=np.int64)
        a=RNG.integers(-8,8,size=(m,G),dtype=np.int64)
        if start==0:
            for r in range(min(4,m)):
                w[r,:,:]=-8 if r&1 else 7
                a[r,:]=-8 if r&2 else 7
        powers=np.array([B**j for j in range(R)],dtype=np.int64)
        p=np.sum(w*powers,axis=2)
        c=np.zeros(m,dtype=dtype)
        for k in range(G): c=fma_sim(p[:,k],a[:,k],c,dtype)
        ref=np.sum(w*a[:,:,None],axis=1)
        # Extract using exact int64 AFTER finishing the floating arithmetic.
        z=c.astype(np.int64)+56*G*sum(int(t) for t in powers)
        outputs=np.stack([((z//int(t))%B)-56*G for t in powers],axis=1)
        bad+=int(np.count_nonzero(np.any(outputs!=ref,axis=1)))
    record(label,blocks=n,mismatched_blocks=bad)

(ROOT/'results.json').write_text(json.dumps(RESULTS,indent=2))
print('Saved',ROOT/'results.json')
