"""Independent CPU model of tile256.cuh; no CUDA or implementation imports."""
from collections import Counter
import math
import struct
import unittest


def half(x):
    try:
        return struct.unpack('<e', struct.pack('<e', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def frombits(x):
    return struct.unpack('<e', struct.pack('<H', x))[0]


class Tile256Tests(unittest.TestCase):
    def test_coverage(self):
        a, b, output = Counter(), Counter(), Counter()
        for tid in range(256):
            warp,lane = divmod(tid,32)
            ar,bc = (warp//2)*16+(lane//8)*4, (warp%2)*32+(lane%8)*4
            for i in range(4):
                for j in range(4):
                    output[ar+i,bc+j] += 1
                a[(tid%4)*8+i,tid//4] += 1
                a[(tid%4)*8+i+4,tid//4] += 1
                b[(tid//64)*4+i,tid%64] += 1
                b[(tid//64)*4+i+16,tid%64] += 1
            self.assertLessEqual(128+tid*4+4,1152)
            self.assertLessEqual(2*(tid%64)+2,128)
            self.assertEqual(128+tid*4,128+((tid//64)*64+tid%64)*4)
            self.assertEqual(ar%4,0)
            self.assertEqual(bc%4,0)
        self.assertEqual(a,Counter({(k,m):1 for k in range(32) for m in range(64)}))
        self.assertEqual(b,Counter({(k,n):1 for k in range(32) for n in range(64)}))
        self.assertEqual(output,Counter({(m,n):1 for m in range(64) for n in range(64)}))

    def test_jobs_tails_and_rails(self):
        for t in (1,31,32,33,63,64,65,127,128,129,511,512):
            for n in (128,512,2048):
                mt,ng = (t+63)//64,n//64
                jobs = {(job//(mt*ng),(job//ng)%mt,job%ng) for job in range(2*mt*ng)}
                self.assertEqual(jobs,{(e,m,col) for e in range(2) for m in range(mt) for col in range(ng)})
                for tm in range(mt):
                    for tid in (0,3,4,127,128,255):
                        row = tm*64+tid//4
                        if row<t:
                            ai = row*96+64+(tid%4)*8
                            self.assertEqual(ai%4,0)
                            self.assertLessEqual(ai+8,t*96)
        for u in (16,32):
            actual = [[],[]]
            for g in range(4):
                for chunk in range(32//u):
                    for s in range(u):
                        kk=chunk*u+s
                        actual[kk//16].append(g*32+kk)
            self.assertEqual(actual,[[g*32+j for g in range(4) for j in range(r*16,(r+1)*16)] for r in (0,1)])

    def test_packed_decode_lanes(self):
        # Every packed byte, selected positive/negative/tiny/overflowing scales.
        for byte in range(256):
            word=0x64006400|(byte&15)|((byte>>4)<<16)
            for lane in (0,1):
                q=half(frombits((word>>(16*lane))&65535)-1032.)
                expected=((byte>>(4*lane))&15)-8
                self.assertEqual(q,expected)
                for raw in (0,0x8000,1,0x8001,0x03ff,0x0400,0x1801,0x3801,0x7bff,0xfbff):
                    d=frombits(raw)
                    self.assertEqual(struct.pack('<e',half(q*d)),struct.pack('<e',half(expected*d)))


if __name__=='__main__':
    unittest.main(verbosity=2)
