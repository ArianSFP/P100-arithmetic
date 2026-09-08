"""CPU-only exact group arithmetic; formats are intentionally separate.

Signed INT4 uses two's-complement nibbles. E2M1 uses sign/magnitude encoded
{0,.5,1,1.5,2,3,4,6}; e2m1_twice returns an INTEGER equal to twice that value.
Neither this format nor its G32 FP32 scale is a claim of NVFP4 equivalence.
"""
from fractions import Fraction
import math
import struct

E2_TWICE = (0, 1, 2, 3, 4, 6, 8, 12)


def signed4(code):
    assert 0 <= code < 16
    return (code & 7) - (code & 8)


def e2m1_twice(code):
    assert 0 <= code < 16
    return (-1 if code & 8 else 1) * E2_TWICE[code & 7]


def bits32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def rn32(value):
    """One IEEE RN-even binary32 rounding of a finite rational, no host FMA.

    Rational zero returns +0; callers testing signed-zero transport must use
    raw bits separately. Subnormals preserved, overflow to signed infinity.
    """
    v = Fraction(value)
    if not v:
        return 0.0
    sign = -1 if v < 0 else 1
    v = abs(v)
    exponent = v.numerator.bit_length() - v.denominator.bit_length()
    power = Fraction(2) ** exponent
    if v < power:
        exponent -= 1
    step = Fraction(2) ** (max(exponent, -126) - 23)
    scaled = v / step
    q, r = divmod(scaled.numerator, scaled.denominator)
    q += 2*r > scaled.denominator or (2*r == scaled.denominator and q % 2)
    rounded = q * step
    if rounded >= Fraction(2) ** 128:
        return math.copysign(math.inf, sign)
    return sign * float(rounded)


def plane_dot(w, a, a_bits=4):
    """Signed bit-plane dot, no scale. E2M1 doubled needs a_bits=5."""
    assert len(w) == len(a) == 32
    assert all(-8 <= v <= 7 for v in w)
    assert all(-(1 << (a_bits-1)) <= v < (1 << (a_bits-1)) for v in a)
    wp = [sum(((v >> p) & 1) << j for j, v in enumerate(w)) for p in range(4)]
    ap = [sum(((v >> p) & 1) << j for j, v in enumerate(a)) for p in range(a_bits)]
    wc = (1, 2, 4, -8)
    ac = [1 << p for p in range(a_bits-1)] + [-(1 << (a_bits-1))]
    return sum(wc[p]*ac[q]*(wp[p] & ap[q]).bit_count()
               for p in range(4) for q in range(a_bits))


def endpoint_dot(w, a):
    """Exact endpoint-SAD identity; activation integers must fit signed bytes."""
    assert len(w) == len(a) == 32
    assert all(-8 <= x <= 7 for x in w)
    assert all(-128 <= x <= 127 for x in a)
    d = [sum(abs(x+128-255*(1-((v >> p) & 1))) for v, x in zip(w, a))
         for p in range(4)]
    numerator = d[0]+2*d[1]+4*d[2]-8*d[3]-sum(a)+127*32-sum(w)
    assert numerator % 2 == 0
    return numerator // 2


def quantize_g32(values, family):
    """Explicit proposed quantizer, not an inferred GPU implementation.

    Finite input floats -> maxabs/(7 or 6), RN32 scale, RN32 division,
    nearest code with RN-even code-index ties. INT4 emits [-7,7], not -8.
    E2M1 signed zero is preserved in its encoded sign bit. Scale is FP32.
    All-zero scale is zero. Reject nonfinite inputs rather than hiding them.
    """
    assert len(values) == 32 and family in ('int4', 'e2m1')
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite activation requires explicit fallback')
    maximum = max(abs(x) for x in values)
    scale = rn32(Fraction(maximum) / (7 if family == 'int4' else 6))
    if maximum and not scale:
        raise ValueError('scale underflow requires explicit fallback')
    codes = []
    for x in values:
        normalized = rn32(Fraction(abs(x)) / Fraction(scale)) if scale else 0.0
        sign = math.copysign(1.0, x) < 0
        if family == 'int4':
            q = min(7, round(normalized))
            codes.append((-q if sign else q) & 15)
        else:
            # Comparing doubled values keeps all codebook values integral.
            i = min(range(8), key=lambda i: (abs(2*normalized-E2_TWICE[i]), i % 2, i))
            codes.append(i | (8 if sign else 0))
    return codes, scale


def group_dot(w_codes, a_codes, family):
    assert len(w_codes) == len(a_codes) == 32
    decode = signed4 if family == 'int4' else e2m1_twice
    if family not in ('int4', 'e2m1'):
        raise ValueError(family)
    return sum(signed4(w)*decode(a) for w, a in zip(w_codes, a_codes))


def scaled_accumulate(groups, family):
    """Sequential FP32 group FMA reference: RN32(dw*da), then RN32(dot*s+c).

    groups: iterable (weight_codes, activation_codes, FP32 dw, FP32 da).
    E2M1 dot is doubled: multiply by exactly 1/2 in the rational FMA input.
    This is NOT T64's per-weight half rounding or its split-K reduction.
    """
    total = 0.0
    for wc, ac, dw, da in groups:
        scale = rn32(Fraction(dw)*Fraction(da))
        dot = Fraction(group_dot(wc, ac, family), 2 if family == 'e2m1' else 1)
        total = rn32(dot*Fraction(scale) + Fraction(total))
    return total
