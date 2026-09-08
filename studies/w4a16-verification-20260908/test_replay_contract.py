"""CPU-only replay accounting contracts, not measured acceptance results."""
from collections import defaultdict
import math
import unittest


def validate_routes(routes, tokens, experts=256, topk=8):
    """Global pre-partition (token,expert) routes; token-level input required."""
    per_token = defaultdict(set)
    for token, expert in routes:
        if not 0 <= token < tokens or not 0 <= expert < experts:
            raise ValueError('out-of-range route')
        if expert in per_token[token]:
            raise ValueError('duplicate route')
        per_token[token].add(expert)
    if len(per_token) != tokens or any(len(v) != topk for v in per_token.values()):
        raise ValueError('missing routes')


def pipeline_ratio(paired_jobs):
    """One complete replay repetition; each job appears once, all costs charged.

    Caller supplies measured full times, NOT per-stage overlaps or raw repeated
    samples pooled across different repetition counts. This mathematical helper
    cannot verify provenance or whether the caller included real preparation.
    """
    seen = set()
    baseline = candidate = 0.0
    for job, b, c in paired_jobs:
        if job in seen or not all(math.isfinite(x) and x > 0 for x in (b,c)):
            raise ValueError('duplicate job or invalid timing')
        seen.add(job)
        baseline += b
        candidate += c
    if not seen:
        raise ValueError('empty replay')
    return baseline/candidate


class ReplayTests(unittest.TestCase):
    def test_conservation_does_not_allow_duplicates_or_missing_tokens(self):
        routes = [(t,e) for t in range(16) for e in range(8)]
        validate_routes(routes, 16)
        with self.assertRaises(ValueError):
            validate_routes(routes[:-1], 16)
        with self.assertRaises(ValueError):
            validate_routes(routes[:-1]+[routes[0]], 16)

    def test_aggregation_not_mean_speedup(self):
        # Average per-job speedup is1.75, but complete workload misses1.5.
        self.assertAlmostEqual(pipeline_ratio([('small',10,5),('large',100,80)]),110/85)
        self.assertLess(pipeline_ratio([('small',10,5),('large',100,80)]),1.5)
        with self.assertRaises(ValueError):
            pipeline_ratio([('same',1,1),('same',1,1)])
        with self.assertRaises(ValueError):
            pipeline_ratio([('bad',1,float('nan'))])

    def test_fallback_must_be_counted(self):
        self.assertEqual(pipeline_ratio([('fast',100,50)]),2)
        self.assertLess(pipeline_ratio([('fast',100,50),('fallback',100,100)]),1.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
