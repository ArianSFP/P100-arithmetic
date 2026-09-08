"""No GPU imports, compiler, subprocess, or output files. Run with python3."""
from fractions import Fraction
import math
import random
import struct
import unittest
from oracle import (E2_TWICE, signed4, e2m1_twice, rn32, bits32, plane_dot,
                    endpoint_dot, quantize_g32, group_dot, scaled_accumulate)


class ReferenceTests(unittest.TestCase):
    def test_all_scalar_pairs_and_constant_groups(self):
        for family, decode, bound in [('int4', signed4, 2048), ('e2m1', e2m1_twice, 3072)]:
            for wc in range(16):
                for ac in range(16):
                    expected = 32*signed4(wc)*decode(ac)
                    self.assertEqual(group_dot([wc]*32, [ac]*32, family), expected)
                    self.assertLessEqual(abs(expected), bound)
                    self.assertEqual(plane_dot([signed4(wc)]*32, [decode(ac)]*32,
                                              4 if family == 'int4' else 5), expected)
                    self.assertEqual(endpoint_dot([signed4(wc)]*32, [decode(ac)]*32), expected)

    def test_random_g32_and_independent_models(self):
        rng = random.Random(0xA416)
        for family, decode, abits in [('int4', signed4, 4), ('e2m1', e2m1_twice, 5)]:
            for _ in range(2048):
                wc = [rng.randrange(16) for _ in range(32)]
                ac = [rng.randrange(16) for _ in range(32)]
                w, a = list(map(signed4, wc)), list(map(decode, ac))
                expected = sum(x*y for x, y in zip(w, a))
                self.assertEqual(group_dot(wc, ac, family), expected)
                self.assertEqual(plane_dot(w, a, abits), expected)
                self.assertEqual(endpoint_dot(w, a), expected)

    def test_format_and_unsigned_correction_traps(self):
        self.assertNotEqual(signed4(7), e2m1_twice(7))
        self.assertEqual(e2m1_twice(15), -12)
        with self.assertRaises(AssertionError):
            plane_dot([1]*32, [12]*32, 4)
        w, a = [-8]*32, [-8]*32
        unsigned = sum((x+8)*(y+8) for x, y in zip(w, a))
        corrected = unsigned-8*sum(x+8 for x in w)-8*sum(y+8 for y in a)+64*32
        self.assertEqual(corrected, 2048)
        self.assertNotEqual(unsigned, corrected)

    def test_quantizer_zeros_extrema_ties_nonfinite(self):
        for family in ('int4', 'e2m1'):
            codes, scale = quantize_g32([0.0, -0.0]*16, family)
            self.assertEqual(scale, 0)
            self.assertEqual(codes[1], 8 if family == 'e2m1' else 0)
            for bad in (math.nan, math.inf, -math.inf):
                with self.assertRaises(ValueError):
                    quantize_g32([bad]+[0.0]*31, family)
        c, s = quantize_g32([7., -7., .5, 1.5, 2.5]+[0.]*27, 'int4')
        self.assertEqual(s, 1.)
        self.assertEqual(list(map(signed4, c[:5])), [7, -7, 0, 2, 2])
        c, s = quantize_g32([6., -6., .25, .75, 1.25, 1.75, 2.5, 3.5, 5.]+[0.]*23, 'e2m1')
        self.assertEqual(s, 1.)
        self.assertEqual(c[:9], [7, 15, 0, 2, 2, 4, 4, 6, 6])

    def test_rational_fp32_rounding(self):
        self.assertEqual(rn32(Fraction(1)+Fraction(1, 2**24)), 1.)
        self.assertEqual(bits32(rn32(Fraction(1)+Fraction(3, 2**24))), 0x3f800002)
        self.assertEqual(bits32(rn32(Fraction(1, 2**149))), 1)
        self.assertEqual(bits32(rn32(Fraction(1, 2**150))), 0)
        self.assertEqual(bits32(rn32(-Fraction(1, 2**150))), 0x80000000)
        self.assertTrue(math.isinf(rn32(Fraction(2)**128)))
        rng = random.Random(32)
        for _ in range(4096):
            raw = rng.randrange(0x7f800000)
            x = struct.unpack('<f', struct.pack('<I', raw))[0]
            self.assertEqual(bits32(rn32(Fraction(x))), raw)

    def test_group_scale_boundaries_and_half_counterexample(self):
        groups = [([1]*32, [1]*32, 0.5, 0.25), ([15]*32, [1]*32, 0.25, 0.25)]
        self.assertEqual(scaled_accumulate(groups, 'int4'), 2.)
        self.assertEqual(scaled_accumulate(groups, 'e2m1'), 1.)
        # E2M1 doubled G32 can exceed half's consecutive integer range.
        values = [96]*31+[1]
        exact = sum(values)
        half = struct.unpack('<e', struct.pack('<e', exact))[0]
        self.assertEqual(exact, 2977)
        self.assertNotEqual(half, exact)
        self.assertEqual(rn32(exact), exact)


if __name__ == '__main__':
    unittest.main(verbosity=2)
