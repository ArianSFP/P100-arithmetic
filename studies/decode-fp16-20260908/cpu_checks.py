#!/usr/bin/env python3
"""Independent format/address checks; never initializes CUDA."""
import struct,json
from pathlib import Path

def half(h):return struct.unpack('<e',struct.pack('<H',h))[0]

def main():
    for byte in range(256):
        u=byte^0x88
        magic=0x64006400 | (u&15) | ((u&240)<<12)
        actual=(half(magic&65535)-1032,half(magic>>16)-1032)
        expected=tuple(((byte>>(4*j))&15)^8 for j in range(2))
        expected=tuple(x-8 for x in expected)
        assert actual==expected,(byte,actual,expected)
    for lo in range(16):
      for hi in range(16):
        word=lo|(hi<<16)|0xfff0fff0
        magic=(word&0x000f000f)^0x64086408
        assert (half(magic&65535)-1032,half(magic>>16)-1032)==((lo^8)-8,(hi^8)-8)
    for lanes in (2,4):
      for m in (1,7,8,16,31,32,33,513,4352,5120):
        rows_per_warp=32//lanes
        seen=set()
        for block in range((m+rows_per_warp-1)//rows_per_warp):
          for lane in range(32):
            row=block*rows_per_warp+lane%rows_per_warp;part=lane//rows_per_warp
            if row>=m:continue
            for c in range(4//lanes):
              for j in range(4):
                for high in (0,1):
                  k=(part*(4//lanes)+c)*8+j+high*4
                  assert (row,k) not in seen
                  seen.add((row,k))
        assert len(seen)==m*32
    geometries=0
    for r in (1,2):
      for b in (1,4,8):
       for s in (8,16,32):
        for nw in (4,8,16,32):
         if nw>s or b*s*r*32*4>49152:continue
         writes=[]
         for warp in range(nw):
          for stripe in range(warp,s,nw):
           for row in range(r*32):
            for batch in range(b):writes.append((batch*s+stripe)*r*32+row)
         assert sorted(writes)==list(range(b*s*r*32))
         values=[{(i//(r*32))%s} for i in range(b*s*r*32)]
         off=s//2
         while off:
          dest=set();src=set()
          for i in range(b*off*r*32):
           row=i%(r*32);stripe=(i//(r*32))%off;batch=i//(off*r*32)
           ix=(batch*s+stripe)*r*32+row
           assert ix not in dest;dest.add(ix);src.add(ix+off*r*32)
           values[ix]=values[ix]|values[ix+off*r*32]
          assert not dest&src
          off//=2
         for batch in range(b):
          for row in range(r*32):assert values[batch*s*r*32+row]==set(range(s))
         geometries+=1
    result=dict(status='PASS',packed_nibble_pairs=256,separated_nibble_pairs=256,shared_geometries=geometries)
    print(json.dumps(result));return result
if __name__=='__main__':main()
