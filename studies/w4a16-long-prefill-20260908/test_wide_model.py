#!/usr/bin/env python3
"""CPU-only address coverage and exact dyadic half-dequantization model."""
from collections import Counter
import math
import struct
import unittest
def half(x):
    try:return struct.unpack('<e',struct.pack('<e',x))[0]
    except OverflowError:return math.copysign(math.inf,x)
def frombits(x):return struct.unpack('<e',x.to_bytes(2,'little'))[0]
def bits(x):return struct.pack('<e',x)
class Models(unittest.TestCase):
    def test_clustered_jobs(self):
        for mt in (1,2,3,4,7,8,16,47):
            for ng in (2,8,32):
                mapped=[]
                for job in range(3*mt*ng):
                    e=job//(mt*ng);local=job%(mt*ng)
                    slab=local//(2*ng);height=min(2,mt-2*slab)
                    rem=local%(2*ng)
                    row=2*slab+(rem//2)%height
                    col=2*(rem//(2*height))+rem%2
                    mapped.append((e,row,col))
                self.assertEqual(Counter(mapped),Counter((e,r,c) for e in range(3) for r in range(mt) for c in range(ng)))

    def test_contiguous_job_partition(self):
        for experts in (1,2,64):
            for mt in (1,2,3,8,47):
                for ng in (2,8,32):
                    total=experts*mt*ng
                    for grid in (1,7,56,168,224,280,336):
                        jobs=[]
                        lengths=[]
                        for block in range(grid):
                            begin=total*block//grid;end=total*(block+1)//grid
                            jobs.extend(range(begin,end));lengths.append(end-begin)
                        self.assertEqual(jobs,list(range(total)))
                        self.assertLessEqual(max(lengths)-min(lengths),1)

    def test_scale_stress_really_changes_operand_rounding(self):
        # Ensure the new fixture is not another all-exact dequantization test.
        changed=0
        for raw in range(0x1800,0x3800):
            scale=frombits(raw)
            for q in (-7,-3,3,7):
                product=q*scale
                changed+=half(product)!=product
                self.assertTrue(math.isfinite(half(product)))
        self.assertGreater(changed,16000)

    def test_packed_dequantization(self):
        # Products of a finite half and a four-bit integer are exactly
        # representable in binary32; double is exact here too. No FMA oracle.
        for code in range(16):
            q=code-8
            decoded=half(frombits(0x6400|code)-1032.)
            self.assertEqual(decoded,q)
            for h in range(65536):
                if h&0x7c00==0x7c00:continue
                d=frombits(h)
                self.assertEqual(bits(half(decoded*d)),bits(half(q*d)))

    def test_tile_coverage(self):
        self.check_tile(64)
        self.check_tile(32)

    def check_tile(self,mt):
        outputs=Counter();astage=Counter();bstage=Counter()
        for tid in range(128):
            warp,lane=divmod(tid,32)
            ar=(warp//2)*(mt//2)+(lane//8)*(mt//8);bc=(warp%2)*64+(lane%8)*8
            for i in range(mt//8):
                for j in range(8):outputs[ar+i,bc+j]+=1
            for l in range(4):
                at=tid+l*128
                if l<mt//32:
                    for i in range(8):astage[(at%4)*8+i,at//4]+=1
                for i in range(4):
                    bstage[(at//128)*4+i,at%128]+=1
                    bstage[(at//128)*4+i+16,at%128]+=1
        self.assertEqual(outputs,Counter({(i,j):1 for i in range(mt) for j in range(128)}))
        self.assertEqual(astage,Counter({(i,j):1 for i in range(32) for j in range(mt)}))
        self.assertEqual(bstage,Counter({(i,j):1 for i in range(32) for j in range(128)}))

    def test_two_n64_weight_tiles(self):
        for n in (128,512,2048):
            for k in (96,512,2048):
                groups=k//32
                for e in range(2):
                    for col in range(n//128):
                        for g in (0,groups-1):
                            for at in range(512):
                                row,chunk=at%128,at//128
                                global_row=col*128+row
                                stage=((e*(n//64)+col*2+row//64)*groups+g)*1152
                                oracle_stage=((e*(n//64)+global_row//64)*groups+g)*1152
                                self.assertEqual(stage,oracle_stage)
                                self.assertEqual(stage+128+(chunk*64+row%64)*4,
                                                 oracle_stage+128+(chunk*64+global_row%64)*4)

if __name__=='__main__':unittest.main()
