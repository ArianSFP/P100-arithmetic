#!/usr/bin/env python3
"""Independent address/format models only; never loads CUDA or the worker."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import struct
import unittest

ROOT=Path(__file__).resolve().parent

class Offline(unittest.TestCase):
    def test_transpose_and_tails(self):
        for t in [1,31,32,33,63,64,65,127,128,129]:
            k=64;experts=2;mt=(t+63)//64;groups=k//32
            out=[None]*(experts*mt*groups*2048)
            for job in range(experts*mt*groups*2):
                part=job%2;g=(job//2)%groups;tm=(job//(2*groups))%mt;e=job//(2*groups*mt)
                tile=[[None]*32 for _ in range(32)]
                for tid in range(256):
                    x,y=tid%32,tid//32
                    for i in range(0,32,8):
                        row=tm*64+part*32+y+i
                        tile[y+i][x]=(e,row,g*32+x) if row<t else 0
                for tid in range(256):
                    x,y=tid%32,tid//32
                    for i in range(0,32,8):
                        at=((e*mt+tm)*groups+g)*2048+(y+i)*64+part*32+x
                        self.assertIsNone(out[at]);out[at]=tile[x][y+i]
            expected=[(e,tm*64+r,g*32+kk) if tm*64+r<t else 0
                      for e in range(experts) for tm in range(mt)
                      for g in range(groups) for kk in range(32) for r in range(64)]
            self.assertEqual(out,expected)

    def test_activation_stages(self):
        expected=Counter({(kk,r):1 for kk in range(32) for r in range(64)})
        for layout in range(4):
            covered=Counter()
            for at in range(256):
                for i in range(8):
                    if layout==0:kk,r=8*(at%4)+i,at//4
                    elif layout==3:kk,r=at//16+(16 if i>=4 else 0),4*(at%16)+i%4
                    else:kk,r=at//8,8*(at%8)+i
                    covered[kk,r]+=1
                    if layout:
                        loaded=at*4+(1024 if i>=4 else 0)+i%4 if layout==3 else at*8+i
                        self.assertEqual(loaded,kk*64+r)
            self.assertEqual(covered,expected)

    def test_weight_layout_and_stage(self):
        rng=random.Random(8191)
        for _ in range(32):
            src=bytes(rng.randrange(256) for _ in range(1152));dst=bytearray(1152)
            dst[:128]=src[:128]
            for r in range(64):
                for c in range(4):dst[128+(c*64+r)*4:128+(c*64+r+1)*4]=src[128+r*16+c*4:128+r*16+c*4+4]
            for b in range(2):
                covered=Counter()
                for at in range(256):
                    r,c=(at%64,at//64) if b else (at//4,at%4)
                    start=128+(c*64+r)*4 if b else 128+r*16+c*4
                    word=int.from_bytes((dst if b else src)[start:start+4],'little')
                    for i in range(4):
                        byte=src[128+r*16+c*4+i]
                        self.assertEqual((word>>(8*i))&15,byte&15)
                        self.assertEqual((word>>(8*i+4))&15,byte>>4)
                        covered[c*4+i,r]+=1;covered[c*4+i+16,r]+=1
                self.assertEqual(covered,Counter({(kk,r):1 for kk in range(32) for r in range(64)}))

    def test_finite_half_roundtrip(self):
        for word in range(65536):
            if word&0x7c00==0x7c00:continue
            raw=word.to_bytes(2,'little');x=struct.unpack('<e',raw)[0]
            f32=struct.unpack('<f',struct.pack('<f',x))[0]
            self.assertEqual(struct.pack('<e',f32),raw)

    def test_inventory_and_gate(self):
        m=json.loads((ROOT/'build/manifest.json').read_text())
        for name,h in m['hashes'].items():self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),h)
        self.assertFalse(m['gpu_executed'])
        s=(ROOT/'build/worker.cu').read_text().split('int main(',1)[1]
        self.assertLess(s.index('if(approved!=1)'),s.index('cudaGetDeviceCount'))
        sass=(ROOT/'build/worker.sass').read_text()
        self.assertNotIn('HFMA2',sass);self.assertNotIn('HMUL2',sass)
        resources=json.loads((ROOT/'build/resources.json').read_text())
        leads=[x for x in resources if 'delivery_lead' in x['symbol']]
        self.assertEqual(len(leads),43)
        self.assertTrue(all(x['shared_bytes'] in (10240,12288,16384) for x in leads))
        log=(ROOT/'build/compile.log').read_text()
        import re
        spills=re.findall(r'(\d+) bytes spill (?:stores|loads)',log)
        self.assertTrue(spills);self.assertTrue(all(int(x)==0 for x in spills))

if __name__=='__main__':unittest.main()
