"""Independent binary16 arithmetic and nibble-decoding checks; no CUDA import."""
import json
import struct
from pathlib import Path

def half(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]

def from_bits(x):
    return struct.unpack('<e', struct.pack('<H', x))[0]

def run():
    products = {w*a for w in range(-8, 8) for a in range(-8, 8)}
    transitions = 0
    # Superset of reachable group-prefix states. No rounding can occur at any
    # valid transition, because all resulting integers are within [-2048,2048].
    for s in range(-2048, 2049):
        assert half(s) == s
        for p in products:
            if abs(s+p) <= 2048:
                assert half(s+p) == s+p
                transitions += 1
    for w in range(-8,8):
        for a in range(-8,8):
            assert half(w*a) == w*a
    for x in range(256):
        q0, q1 = (x&15)^8, ((x>>4)&15)^8
        h0, h1 = from_bits(0x6400|q0)-1032, from_bits(0x6400|q1)-1032
        assert h0 == (x&7)-(x&8)
        assert h1 == ((x>>4)&7)-((x>>4)&8)
        assert half(h0) == h0 and half(h1) == h1
    # 32 is the largest group with this unconditional signed-INT4 bound.
    assert half(2049) != 2049
    # Concrete 33-term dot: 32*(-8*-8) + (1*1) = 2049.
    assert half(32*64+1) != 32*64+1
    result = dict(status='PASS', scalar_products=256, decode_pairs=256,
                  checked_half_transitions=transitions, exact_group_bound=2048,
                  counterexample_G33=2049, gpu_executed=False)
    Path(__file__).with_name('cpu-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    run()
