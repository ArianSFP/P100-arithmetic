"""Lightweight CPU-only address and exact integer arithmetic checks."""
import random
from pathlib import Path
from build import block, SOURCE, EXPECTED, sha
def planes(v):return [sum(((x>>p)&1)<<b for b,x in enumerate(v)) for p in range(4)]
def dot(a,w):
    ap,wp=planes(a),planes(w);c=[1,2,4,-8]
    return sum(c[p]*c[q]*(ap[p]&wp[q]).bit_count() for p in range(4) for q in range(4))
def main():
    for a in range(-8,8):
        for w in range(-8,8):assert dot([a]*32,[w]*32)==a*w*32
    rng=random.Random(3911)
    for _ in range(2048):
        a=[rng.randrange(-8,8) for _ in range(32)];w=[rng.randrange(-8,8) for _ in range(32)]
        assert dot(a,w)==sum(x*y for x,y in zip(a,w))
    for rm in (2,4):
        covered=[((tid//16)*rm+i,(tid%16)*4+j) for tid in range(128) for i in range(rm) for j in range(4)]
        assert len(set(covered))==8*rm*64
        for m in (1,15,16,17,31,32,33,63,64,65,128,256,512,1025,4096):
            rows=[base+r for base in range(0,m,8*rm) for r in range(8*rm) if base+r<m]
            assert rows==list(range(m))
    for n,k in ((128,96),(512,2048),(2048,512)):
        ng=k//32
        # Every packed Q4 byte gets exactly the low/high nibble once.
        seen=set()
        for r in range(n):
            for g in range(ng):
                base=(r//64*ng+g)*1152
                for b in range(32):
                    address=base+128+r%64*16+b%16
                    pair=(address,b//16)
                    assert pair not in seen;seen.add(pair)
        assert len(seen)==n*k
    assert sha(SOURCE)==EXPECTED
    start='template<bool BF16_INPUT>\n__global__ __launch_bounds__(256, 2)\nstatic void aw_q8_service_m64_n128_halfpipe_sync('
    generated=(Path(__file__).parent/'build/current.inc').read_text()
    assert block(generated,start)==block(SOURCE.read_text(),start)
    print('PASS: 256 endpoint dots, 2048 random G32 dots, tile/tail coverage, Q4 addresses, frozen current control')
    print('CPU only; no compiler, CUDA or performance result')
if __name__=='__main__':main()
