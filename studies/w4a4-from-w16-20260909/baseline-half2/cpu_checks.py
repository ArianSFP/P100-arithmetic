#!/usr/bin/env python3
import random


def decode_pair(byte):
    def q(n):
        return ((n ^ 8) - 8)
    return q(byte & 15), q((byte >> 4) & 15)


for byte in range(256):
    assert decode_pair(byte) == (((byte & 15) ^ 8) - 8,
                                 (((byte >> 4) & 15) ^ 8) - 8)

# Every sequential G32 raw-code prefix is an integer in half's contiguous
# exact range.  Include -8 activation codes even though the production-like
# symmetric quantizer emits only -7..7.
rng = random.Random(3911)
transitions = 0
for _ in range(100_000):
    total = 0
    for _ in range(32):
        total += rng.randrange(-8, 8) * rng.randrange(-8, 8)
        assert -2048 <= total <= 2048
        assert float(total).is_integer()
        transitions += 1

# Verify the word-major tiled weight layout and row-major activation packing.
for row in range(64):
    stage = bytearray(1152)
    expected = []
    for k in range(32):
        q = ((row * 37 + k * 11) & 15) - 8
        expected.append(q)
        ph, word, within = k // 16, (k % 16) // 8, k % 8
        offset = 128 + ph * 512 + word * 256 + row * 4 + within // 2
        stage[offset] |= (q & 15) << ((within & 1) * 4)
    got = []
    for ph in range(2):
        for word in range(2):
            offset = 128 + ph * 512 + word * 256 + row * 4
            packed = int.from_bytes(stage[offset:offset + 4], "little")
            for pair in range(4):
                got.extend(decode_pair((packed >> (pair * 8)) & 255))
    assert got == expected

for _ in range(10_000):
    q = [rng.randrange(-7, 8) for _ in range(32)]
    packed = bytearray(16)
    for k, value in enumerate(q):
        packed[k // 2] |= (value & 15) << ((k & 1) * 4)
    assert [decode_pair(packed[k])[lane]
            for k in range(16) for lane in range(2)] == q

print(f"PASS decode_bytes=256 exact_prefix_transitions={transitions} "
      "weight_rows=64 activation_groups=10000")
