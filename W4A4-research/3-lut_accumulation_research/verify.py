#!/usr/bin/env python3
"""Exact INT4 LUT accumulation experiments. CPU only; not a performance model.

Requires Python 3.10+ and NumPy. Reproduce: python verify.py --blocks 20000
Weights/activations are signed INT4 [-8,7]. Tables use offset-binary weight
codes q=w+8, whose four bit-plane coefficients are [1,2,4,8].
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

U32 = (1 << 32) - 1

def masks(mu: int) -> np.ndarray:
    return ((np.arange(1 << mu)[:, None] >> np.arange(mu)) & 1).astype(np.int64)

def table(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v: [blocks, output_lanes, mu]. Returns [blocks, lanes, 2**mu]."""
    bias = np.maximum(-v, 0).sum(axis=-1)
    tab = np.einsum('brk,ik->bri', v, masks(v.shape[-1])) + bias[..., None]
    assert np.all(tab >= 0)
    assert np.all(tab <= np.abs(v).sum(axis=-1)[..., None])
    return tab, bias

def finish(acc: np.ndarray, bias: np.ndarray, sums: np.ndarray, lane_bits: int) -> np.ndarray:
    """acc [blocks, 4 bitplanes], unsigned packed fields with NO carries."""
    if lane_bits == 16:
        combined = sum(acc[:, b] << b for b in range(4)) & U32
        fields = np.stack([combined & 65535, (combined >> 16) & 65535], axis=1)
    elif lane_bits == 8:
        # Widen even/odd bytes before weighting; the final dots do NOT fit bytes.
        even = sum((acc[:, b] & 0x00ff00ff) << b for b in range(4)) & U32
        odd  = sum(((acc[:, b] >> 8) & 0x00ff00ff) << b for b in range(4)) & U32
        fields = np.stack([even & 65535, odd & 65535, even >> 16, odd >> 16], axis=1)
    else:
        raise ValueError('lane_bits must be 8 or 16')
    return fields - (15 * bias + 8 * sums)

def same_mask_lut(v: np.ndarray, selector: np.ndarray, mu: int, lane_bits: int) -> np.ndarray:
    """Compute dot(v_lane, selector), shared selector across all lanes.

    Activation-table orientation: v contains multiple token activation vectors,
    selector is one weight row. Transposed weight-table orientation: v contains
    multiple weight rows, selector is one token activation vector.
    """
    blocks, lanes, G = v.shape
    if G % mu or lanes * lane_bits != 32:
        raise ValueError('unsupported dimensions')
    if np.any(v < -8) or np.any(v > 7) or np.any(selector < -8) or np.any(selector > 7):
        raise ValueError('not signed INT4')
    if lane_bits == 8 and np.any(np.abs(v).sum(axis=-1) > 255):
        raise ValueError('byte fields require source-row L1 <= 255')
    acc = np.zeros((blocks, 4), dtype=np.int64)
    bias = np.zeros((blocks, lanes), dtype=np.int64)
    q = selector + 8
    idx = np.arange(blocks)
    for k in range(0, G, mu):
        tab, b = table(v[:, :, k:k+mu])
        bias += b
        packed = sum(tab[:, j, :] << (j * lane_bits) for j in range(lanes))
        for p in range(4):
            key = (((q[:, k:k+mu] >> p) & 1) << np.arange(mu)).sum(axis=-1)
            acc[:, p] = (acc[:, p] + packed[idx, key]) & U32
    assert np.all(np.abs(v).sum(axis=-1) < (1 << lane_bits))
    return finish(acc, bias, v.sum(axis=-1), lane_bits)

def normalized_four_rows_g32(values: np.ndarray, selector: np.ndarray) -> np.ndarray:
    """Exact G32 byte packing: only all-minus-eight rows require normalization.
    Replace such a row by -4 for table construction, then restore factor two.
    This is an on-chip representation change, not lossy requantization.
    """
    assert values.shape[1:] == (4, 32)
    special = np.all(values == -8, axis=-1)
    factor = np.where(special, 2, 1)
    norm = values // factor[..., None]
    assert np.all(norm * factor[..., None] == values)
    assert np.all(np.abs(norm).sum(axis=-1) <= 255)
    return same_mask_lut(norm, selector, 8, 8) * factor

def paired_rows_lut(a: np.ndarray, weights: np.ndarray, mu: int = 4) -> np.ndarray:
    """Two weight rows, ONE activation, 2**(2*mu)-entry paired-result tables."""
    blocks, lanes, G = weights.shape
    assert lanes == 2 and G % mu == 0
    acc = np.zeros((blocks, 4), dtype=np.int64)
    q = weights + 8
    bias = np.zeros(blocks, dtype=np.int64)
    idx = np.arange(blocks)
    size = 1 << mu
    keys = np.arange(size*size)
    for k in range(0, G, mu):
        base, b = table(a[:, None, k:k+mu])
        base = base[:, 0, :]
        bias += b[:, 0]
        paired = base[:, keys % size] | (base[:, keys // size] << 16)
        for p in range(4):
            row_keys = ((((q[:, :, k:k+mu] >> p) & 1) << np.arange(mu)).sum(axis=-1))
            key = row_keys[:, 0] | (row_keys[:, 1] << mu)
            acc[:, p] = (acc[:, p] + paired[idx, key]) & U32
    return finish(acc, np.repeat(bias[:, None], 2, axis=1),
                  np.repeat(a.sum(axis=-1)[:, None], 2, axis=1), 16)

def half_lut_g32(v: np.ndarray, selector: np.ndarray, mu: int = 8) -> np.ndarray:
    """LUT values are signed FP16 partial sums, not radix-packed products.
    Two adjacent output lanes can be one native half2. Sums/combination exact
    for raw INT4 G<=32. This retains the native FP16 add/FMA path.
    """
    blocks, lanes, G = v.shape
    assert G <= 32 and G % mu == 0
    acc = np.zeros((blocks, 4, lanes), dtype=np.float16)
    q = selector & 15  # two's complement, coefficients [1, 2, 4, -8]
    for k in range(0, G, mu):
        tab = np.einsum('brk,ik->bri', v[:, :, k:k+mu], masks(mu)).astype(np.float16)
        for b in range(4):
            key = ((((q[:, k:k+mu] >> b) & 1) << np.arange(mu)).sum(axis=-1))
            look = np.take_along_axis(tab, key[:, None, None], axis=2)[..., 0]
            acc[:, b, :] = (acc[:, b, :].astype(np.float64) + look.astype(np.float64)).astype(np.float16)
    out = acc[:, 0, :]
    for b, coeff in [(1, 2), (2, 4), (3, -8)]:
        out = (out.astype(np.float64) + coeff * acc[:, b, :].astype(np.float64)).astype(np.float16)
    return out.astype(np.int64)

def half_control(v: np.ndarray, selector: np.ndarray) -> np.ndarray:
    # Float64 makes the small integer FMA intermediate exact; one cast rounds.
    h = np.zeros(v.shape[:2], dtype=np.float16)
    for k in range(v.shape[-1]):
        h = (h.astype(np.float64) + v[:, :, k] * selector[:, k, None]).astype(np.float16)
    return h.astype(np.int64)

def run(blocks: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    names = ['paired_rows_G32_mu4', 'two_tokens_G32_mu8',
             'four_tokens_G16_mu8', 'four_weight_rows_G16_mu8',
             'four_tokens_G32_two_G16', 'ordinary_half_G32', 'native_half_LUT_G32_mu8',
             'four_weight_rows_G32_normalized_mu8']
    records = {name: {'blocks': 0, 'output_dots': 0, 'mismatches': 0} for name in names}
    for offset in range(0, blocks, 250):
        n = min(250, blocks-offset)
        v = rng.integers(-8, 8, (n, 4, 32), dtype=np.int64)
        w = rng.integers(-8, 8, (n, 32), dtype=np.int64)
        if offset == 0:
            for j, (aa, ww) in enumerate([(-8,-8),(-8,7),(7,-8),(7,7),(0,-8),(-8,0)]):
                if j < n: v[j].fill(aa); w[j].fill(ww)
        if offset == 0 and n >= 8:
            v[6].fill(-8); v[6, 0, 0] = -7; v[6, 1].fill(0); v[6, 3].fill(7)
            w[6].fill(-8)
            v[7].fill(-8); v[7, 0, 0] = -7; v[7, 1].fill(0); v[7, 3].fill(7)
            w[7].fill(7)
        expected = (v * w[:, None, :]).sum(axis=-1)
        checks = {
            'paired_rows_G32_mu4': (paired_rows_lut(w, v[:, :2, :]), expected[:, :2]),
            'two_tokens_G32_mu8': (same_mask_lut(v[:, :2, :], w, 8, 16), expected[:, :2]),
            'four_tokens_G16_mu8': (same_mask_lut(v[:, :, :16], w[:, :16], 8, 8),
                                   (v[:, :, :16]*w[:, None, :16]).sum(axis=-1)),
            'four_weight_rows_G16_mu8': (same_mask_lut(v[:, :, 16:], w[:, 16:], 8, 8),
                                       (v[:, :, 16:]*w[:, None, 16:]).sum(axis=-1)),
            'four_tokens_G32_two_G16': (same_mask_lut(v[:, :, :16], w[:, :16], 8, 8)
                                      + same_mask_lut(v[:, :, 16:], w[:, 16:], 8, 8), expected),
            'ordinary_half_G32': (half_control(v, w), expected),
            'native_half_LUT_G32_mu8': (half_lut_g32(v, w), expected),
            'four_weight_rows_G32_normalized_mu8': (normalized_four_rows_g32(v, w), expected),
        }
        for name, (got, ref) in checks.items():
            records[name]['blocks'] += n
            records[name]['output_dots'] += int(ref.size)
            records[name]['mismatches'] += int(np.count_nonzero(got != ref))
    # Exhaustive all 16^4 activation quartets, all 16 subset masks.
    codes = np.arange(16**4)
    quartets = (((codes[:, None] >> (4*np.arange(4))) & 15) - 8).astype(np.int64)
    tabs, _ = table(quartets[:, None, :])
    records['exhaustive_mu4_table_bounds'] = {
        'activation_vectors': 16**4, 'table_entries': int(tabs.size),
        'minimum': int(tabs.min()), 'maximum': int(tabs.max()), 'mismatches': 0}
    # Exact collision of two radix-128 sums under identical activations.
    activation = np.array([7,6])
    cases = [([-8,-8],[0,0]), ([6,-3],[-1,1])]
    collision = []
    for low, high in cases:
        lo = int(np.dot(low, activation)); hi = int(np.dot(high, activation))
        acc = np.float16(0)
        for p,a in zip(np.array(low)+128*np.array(high), activation):
            acc = np.float16(float(p)*float(a)+float(acc))
        collision.append({'low_dot': lo, 'high_dot': hi,
                          'exact_packed_sum': lo+128*hi, 'fp16_packed_sum': float(acc)})
    records['radix128_two_term_collision'] = collision
    records['one_fp16_state_bound_G32_all_ones'] = {
        'values_per_dot': 481, 'pairs': 481**2, 'all_16bit_patterns': 2**16,
        'minimum_bits': int((481**2 - 1).bit_length())}
    # Naive byte-lane G32 is NOT universally safe.
    word = sum(64 << (8*j) for j in range(4))
    acc = (4 * word) & U32
    records['naive_G32_byte_carry_counterexample'] = {
        'true_subset_sums_plus_bias': [256]*4,
        'observed_fields': [(acc >> (8*j)) & 255 for j in range(4)],
        'packed_hex': hex(acc), 'interpretation': 'unnormalized G32 byte lanes overflow; G16 or exact endpoint normalization fixes it'}
    # Proposed bank-owned shared layout table[mask][lane], 32-bit entries.
    idx = rng.integers(0, 256, (10000, 32))
    bank = (idx * 32 + np.arange(32)) % 32
    records['bank_owned_layout_address_check'] = {
        'warp_index_patterns': len(idx),
        'bank_assignment_mismatches': int(np.count_nonzero(bank != np.arange(32))),
        'qualification': 'address arithmetic only, not GPU timing; each lane owns a distinct table column'}
    assert all(x.get('mismatches', 0) == 0 for x in records.values() if isinstance(x, dict))
    return {'seed': seed, 'evidence': 'CPU exact-code validation; no GPU compiler, GPU execution, or timings',
            'results': records}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20260909)
    args = ap.parse_args()
    if args.blocks < 6: ap.error('--blocks must be at least 6')
    results = run(args.blocks, args.seed)
    text = json.dumps(results, indent=2)
    Path(__file__).with_name('results.json').write_text(text+'\n')
    print(text)
