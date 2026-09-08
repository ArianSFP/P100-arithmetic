#!/usr/bin/env python3
"""Exact integer proofs exercised with finite vectors. No CUDA dependency."""
import importlib.util
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent

def signed(x, bits):
    return x - (1 << bits) if x & (1 << (bits-1)) else x

def sad_model(a, b):
    return sum(abs(((a >> (8*j)) & 255) - ((b >> (8*j)) & 255)) for j in range(4))

def bitplanes(aw, ww, bits):
    biased = aw ^ 0x80808080
    total = 0
    sum_w = 0
    for j in range(4):
        sum_w += signed((ww >> (bits*j)) & ((1 << bits)-1), bits)
    for p in range(bits):
        mask = sum((255 if (ww >> (bits*j+p)) & 1 else 0) << (8*j) for j in range(4))
        coeff = -(1 << p) if p == bits-1 else 1 << p
        total += coeff * sad_model(biased & mask, 0)
    return total - 128 * sum_w

def direct(aw, ww, bits):
    return sum(signed((aw >> (8*j)) & 255, 8) *
               signed((ww >> (bits*j)) & ((1 << bits)-1), bits) for j in range(4))

def main():
    rng = random.Random(0x6000248)
    checks = {}
    for bits in (2, 4, 8):
        n = 0
        # All scalar A8 x Wb pairs at each byte position, with nonzero backgrounds.
        for a in range(256):
            for w in range(1 << bits):
                for lane in range(4):
                    aw, ww = 0x197fa580, 0xa53c7b91 & ((1 << (4*bits))-1)
                    aw = (aw & ~(255 << (8*lane))) | (a << (8*lane))
                    ww = (ww & ~(((1 << bits)-1) << (bits*lane))) | (w << (bits*lane))
                    assert bitplanes(aw, ww, bits) == direct(aw, ww, bits)
                    n += 1
        for _ in range(4096):
            aw, ww = rng.getrandbits(32), rng.getrandbits(4*bits)
            assert bitplanes(aw, ww, bits) == direct(aw, ww, bits)
            n += 1
        checks[f'w{bits}a8_sad_identity'] = {'cases': n, 'result': 'PASS_CPU_ONLY'}
    # Verify the proposed replicated LUT has one distinct bank per lane,
    # independently of each lane's selected table entry.
    for _ in range(4096):
        indices = [rng.randrange(16) for _ in range(32)]
        banks = [(idx*32 + lane) % 32 for lane, idx in enumerate(indices)]
        assert len(set(banks)) == 32
    checks['replicated_16_entry_lut_bank_mapping'] = {'cases': 4096, 'result': 'PASS_ADDRESS_MODEL_ONLY'}
    spec = importlib.util.spec_from_file_location('old_oracle', ROOT.parent/'scripts/oracle.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    v = oracle.half_to_fraction(0x3800)
    checks['existing_hfma_oracle_audit'] = {
        'input': '0x3800 (0.5)', 'returned_type': type(v).__name__,
        'issue': 'float returned despite Fraction annotation; not a general exact oracle'}
    try:
        oracle.fma_half(0x3800, 0x3c00, 0)
    except AttributeError as exc:
        checks['existing_hfma_oracle_audit']['reproduced_exception'] = str(exc)
    path = ROOT/'compiler-output/cpu-checks.json'
    path.write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))

if __name__ == '__main__':
    main()
