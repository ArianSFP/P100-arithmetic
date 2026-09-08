#!/usr/bin/env python3
"""Exact binary16 helpers and diagnostic expectations for HFMA2."""

from __future__ import annotations

import argparse
from fractions import Fraction


def half_to_fraction(bits: int) -> Fraction:
    bits &= 0xffff
    sign = -1 if bits & 0x8000 else 1
    exponent = (bits >> 10) & 0x1f
    fraction = bits & 0x3ff
    if exponent == 0:
        if fraction == 0:
            return Fraction(0)
        value = Fraction(fraction, 1 << 24)
    elif exponent == 0x1f:
        raise ValueError("NaN/Inf is outside the initial exact oracle")
    else:
        value = Fraction(1024 + fraction, 1 << 10) * (2 ** (exponent - 15))
    return sign * value


def round_fraction_to_half(value: Fraction) -> int:
    if value == 0:
        return 0
    sign = 0x8000 if value < 0 else 0
    value = abs(value)

    def round_even(numerator: int, denominator: int) -> int:
        quotient, remainder = divmod(numerator, denominator)
        twice = remainder * 2
        if twice > denominator or (twice == denominator and quotient & 1):
            quotient += 1
        return quotient

    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if exponent >= 0:
        if Fraction(1 << exponent, 1) > value:
            exponent -= 1
    else:
        if Fraction(1, 1 << (-exponent)) > value:
            exponent -= 1

    if exponent > 15:
        return sign | 0x7c00
    if exponent >= -14:
        if exponent >= 0:
            scaled = value * Fraction(1 << 10, 1 << exponent)
        else:
            scaled = value * Fraction((1 << 10) << (-exponent), 1)
        significand = round_even(scaled.numerator, scaled.denominator)
        if significand == 2048:
            exponent += 1
            significand = 1024
        if exponent > 15:
            return sign | 0x7c00
        return sign | ((exponent + 15) << 10) | (significand - 1024)

    scaled = value * (1 << 24)
    fraction = round_even(scaled.numerator, scaled.denominator)
    if fraction == 0:
        return sign
    if fraction >= 1024:
        return sign | (1 << 10)
    return sign | fraction


def fma_half(a: int, b: int, c: int) -> int:
    return round_fraction_to_half(half_to_fraction(a) * half_to_fraction(b) + half_to_fraction(c))


def pack(low: int, high: int) -> int:
    return (low & 0xffff) | ((high & 0xffff) << 16)


def unpack(value: int) -> tuple[int, int]:
    return value & 0xffff, (value >> 16) & 0xffff


def packed_fma(a: int, b: int, c: int, b_mode: int = 0) -> int:
    a0, a1 = unpack(a)
    b0, b1 = unpack(b)
    c0, c1 = unpack(c)
    if b_mode == 1:
        b1 = b0
    elif b_mode == 2:
        b0 = b1
    elif b_mode != 0:
        raise ValueError("destination/other selectors require a SASS-level model")
    return pack(fma_half(a0, b0, c0), fma_half(a1, b1, c1))


VECTORS = {
    "lane_distinguish": (0x40003C00, 0x45004200, 0x49804700),
    "signed_lane_distinguish": (0x4000BC00, 0xC5004200, 0xC9804700),
    "fused_rounding": (0x3C013C01, 0x5BFE5BFE, 0xDC00DC00),
    "zeros": (0, 0, 0),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-mode", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    for name, (a, b, c) in VECTORS.items():
        result = packed_fma(a, b, c, args.b_mode)
        lo, hi = unpack(result)
        print(f"vector={name} out=0x{result:08x} low=0x{lo:04x} high=0x{hi:04x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
