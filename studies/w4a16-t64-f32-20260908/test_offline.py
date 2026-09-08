#!/usr/bin/env python3
"""CPU-only layout, ownership and build-inventory tests; not GPU semantics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unittest

ROOT = Path(__file__).resolve().parent

class Offline(unittest.TestCase):
    def test_tiles(self):
        text = (ROOT/'worker.cu').read_text()
        shapes = re.findall(r'X\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\)', text)
        self.assertEqual(len(shapes), 14)
        for values in shapes:
            mt,nt,bt,sk,single,chunk = map(int, values)
            nw=bt//32; wn=2; wm=nw//(wn*sk)
            rm=mt//(wm*4); cn=nt//(wn*8)
            owned=Counter(); aload=Counter(); bload=Counter()
            for tid in range(bt):
                warp,lane=divmod(tid,32)
                kg=warp//(nw//sk)
                row=(warp%(nw//sk))//wn*(mt//wm)+(lane//8)*rm
                col=(warp%wn)*(nt//wn)+(lane%8)*cn
                self.assertEqual(row%2,0);self.assertEqual(col%4,0)
                for i in range(rm):
                    for j in range(cn):owned[kg,row+i,col+j]+=1
                for l in range(mt*4//bt):
                    at=tid+l*bt
                    for i in range(8):aload[at//4,8*(at%4)+i]+=1
                for l in range(nt*4//bt):
                    at=tid+l*bt
                    for i in range(4):
                        bload[at//4,4*(at%4)+i]+=1
                        bload[at//4,4*(at%4)+i+16]+=1
            self.assertEqual(owned,Counter({(s,r,c):1 for s in range(sk) for r in range(mt) for c in range(nt)}))
            self.assertEqual(aload,Counter({(r,k):1 for r in range(mt) for k in range(32)}))
            self.assertEqual(bload,Counter({(r,k):1 for r in range(nt) for k in range(32)}))
            smem=max((1 if single else 2)*32*(mt+nt)*4, (mt if sk==2 else 1)*nt*4)
            self.assertLessEqual(smem,49152)
            self.assertEqual((32//sk)%chunk,0)

    def test_nibble_loads(self):
        rng=random.Random(3911)
        for trial in range(1024):
            codes=[rng.randrange(-8,8) for _ in range(32)]
            packed=bytes((codes[i]+8)|((codes[i+16]+8)<<4) for i in range(16))
            expanded=[None]*32
            for chunk in range(4):
                word=int.from_bytes(packed[4*chunk:4*chunk+4],'little')
                for i in range(4):
                    expanded[4*chunk+i]=((word>>(8*i))&15)-8
                    expanded[4*chunk+i+16]=((word>>(8*i+4))&15)-8
            self.assertEqual(expanded,codes)
            for bk in (0,8,16,24):
                for i in range(4):
                    ki=bk+2*i
                    pair=[((packed[(ki+j)%16]>>(4 if ki>=16 else 0))&15)-8 for j in (0,1)]
                    self.assertEqual(pair,codes[ki:ki+2])

    def test_build(self):
        manifest=json.loads((ROOT/'build/manifest.json').read_text())
        for name,expected in manifest['hashes'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)
        sass=(ROOT/'build/worker.sass').read_text()
        self.assertNotIn('HFMA2',sass)
        self.assertNotIn('HMUL2',sass)
        self.assertIn('FFMA',sass)

if __name__=='__main__':unittest.main()
