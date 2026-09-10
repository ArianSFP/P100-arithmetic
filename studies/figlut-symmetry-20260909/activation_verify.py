#!/usr/bin/env python3
"""CPU proof harness for the four-token activation-derived mu4 LUT.

This models the production symmetric A4 contract (-7..7), native offset-coded
W4 weights (-8..7), four activation tokens packed into uint32 byte fields, and
both the full16 and complement-canonical half8 lookup forms.  It deliberately
models ordinary uint32 additions/subtractions so any cross-byte carry or borrow
is observable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

U32 = (1 << 32) - 1


def pack4(x: np.ndarray) -> np.ndarray:
    """Pack final dimension [4] as little-endian uint32 byte fields."""
    assert x.shape[-1] == 4
    assert np.all((0 <= x) & (x <= 255))
    out = np.zeros(x.shape[:-1], dtype=np.uint64)
    for t in range(4):
        out |= x[..., t].astype(np.uint64) << (8 * t)
    return out


def unpack4(x: np.ndarray) -> np.ndarray:
    return np.stack([(x >> (8 * t)) & 255 for t in range(4)], axis=-1).astype(np.int64)


def make_table(a4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a4 is [block, token=4, mu=4]; return packed full16 and packed L."""
    pos = np.maximum(a4, 0)
    neg = np.maximum(-a4, 0)
    masks = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.int64)
    # U_t(m) = sum_j (m_j ? pos_tj : neg_tj)
    u = np.where(masks[None, None, :, :].astype(bool),
                 pos[:, :, None, :], neg[:, :, None, :]).sum(axis=-1)
    assert np.all((0 <= u) & (u <= 28))
    full = pack4(np.transpose(u, (0, 2, 1)))
    limit = pack4(np.abs(a4).sum(axis=-1))
    return full, limit


def packed_finish(p: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """p [block, output, plane] -> four signed integer dots."""
    mask = np.uint64(0x00FF00FF)
    even = np.zeros(p.shape[:2], dtype=np.uint64)
    odd = np.zeros(p.shape[:2], dtype=np.uint64)
    for plane in range(4):
        even += (p[..., plane] & mask) << plane
        odd += ((p[..., plane] >> 8) & mask) << plane
    fields = np.stack((even & 0xFFFF, odd & 0xFFFF,
                       even >> 16, odd >> 16), axis=-1).astype(np.int64)
    return fields - correction[:, None, :]


def packed_finish_twos(p: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, int]:
    """Finish two's-complement weight planes with exact packed-16 arithmetic.

    Repacking native offset codes c as c^8 gives coefficients [1,2,4,-8].
    Since every P plane contains one copy of B, the raw dot is
    P0 + 2*P1 + 4*P2 - 8*P3 + B.  A 0x8000 field bias makes the packed
    subtraction borrow-free; XOR then converts offset fields to signed int16.
    """
    mask = np.uint64(0x00FF00FF)
    b_even = bias[:, 0].astype(np.uint64) | (bias[:, 2].astype(np.uint64) << 16)
    b_odd = bias[:, 1].astype(np.uint64) | (bias[:, 3].astype(np.uint64) << 16)
    outs = []
    bound_failures = 0
    for o in range(p.shape[1]):
        e = [(p[:, o, plane] & mask) for plane in range(4)]
        d = [((p[:, o, plane] >> 8) & mask) for plane in range(4)]
        words = []
        for fields, packed_b in ((e, b_even), (d, b_odd)):
            positive = fields[0] + (fields[1] << 1) + (fields[2] << 2)
            minuend = positive + packed_b + np.uint64(0x80008000)
            negative = fields[3] << 3
            # Both 16-bit minuend fields exceed their negative field, and the
            # final fields remain below 65536, so uint32 subtraction is lane-safe.
            mf = np.stack((minuend & 0xffff, minuend >> 16), axis=-1)
            nf = np.stack((negative & 0xffff, negative >> 16), axis=-1)
            bound_failures += int(np.count_nonzero(mf < nf))
            bound_failures += int(np.count_nonzero(mf - nf >= 65536))
            encoded = ((minuend - negative) & U32) ^ np.uint64(0x80008000)
            lo = (encoded & 0xffff).astype(np.int64)
            hi = ((encoded >> 16) & 0xffff).astype(np.int64)
            lo = np.where(lo >= 32768, lo - 65536, lo)
            hi = np.where(hi >= 32768, hi - 65536, hi)
            words.extend((lo, hi))
        # even low/high are tokens 0/2; odd low/high are tokens 1/3.
        outs.append(np.stack((words[0], words[2], words[1], words[3]), axis=-1))
    return np.stack(outs, axis=1), bound_failures


def run_case(a: np.ndarray, w: np.ndarray) -> dict[str, int]:
    """a [B,4,32] in -7..7; w [B,O,32] in -8..7."""
    blocks, tokens, width = a.shape
    assert tokens == 4 and width == 32 and w.shape[0] == blocks and w.shape[2] == 32
    assert np.all((-7 <= a) & (a <= 7))
    assert np.all((-8 <= w) & (w <= 7))
    outputs = w.shape[1]
    q = w + 8
    q_twos = q ^ 8
    bias = np.maximum(-a, 0).sum(axis=-1)
    asum = a.sum(axis=-1)
    correction = 15 * bias + 8 * asum
    full_acc = np.zeros((blocks, outputs, 4), dtype=np.uint64)
    half_acc = np.zeros_like(full_acc)
    twos_acc = np.zeros_like(full_acc)
    max_plane_field = 0
    borrow_mismatches = 0
    identity_mismatches = 0
    for k0 in range(0, 32, 4):
        table, limit = make_table(a[:, :, k0:k0 + 4])
        idx_b = np.arange(blocks)[:, None]
        for plane in range(4):
            bits = (q[:, :, k0:k0 + 4] >> plane) & 1
            raw = (bits << np.arange(4)).sum(axis=-1)
            direct = table[idx_b, raw]
            flip = raw >> 3
            canonical = (raw & 7) ^ (flip * 7)
            stored = table[idx_b, canonical]
            # This is the actual scalar uint32 subtraction proposed on GPU.
            complemented_word = (limit[:, None] - stored) & U32
            selected = np.where(flip.astype(bool), complemented_word, stored)
            identity_mismatches += int(np.count_nonzero(selected != direct))
            # Explicit field subtraction independently detects a hidden borrow.
            scalar_fields = unpack4(limit[:, None]) - unpack4(stored)
            borrow_mismatches += int(np.count_nonzero(unpack4(complemented_word) != scalar_fields))
            full_acc[:, :, plane] = (full_acc[:, :, plane] + direct) & U32
            half_acc[:, :, plane] = (half_acc[:, :, plane] + selected) & U32
            twos_bits = (q_twos[:, :, k0:k0 + 4] >> plane) & 1
            twos_mask = (twos_bits << np.arange(4)).sum(axis=-1)
            twos_acc[:, :, plane] = (twos_acc[:, :, plane] + table[idx_b, twos_mask]) & U32
            max_plane_field = max(max_plane_field, int(unpack4(full_acc[:, :, plane]).max()))
    full_dot = packed_finish(full_acc, correction)
    half_dot = packed_finish(half_acc, correction)
    twos_dot, twos_bound_failures = packed_finish_twos(twos_acc, bias)
    expected = np.einsum("btk,bok->bot", a, w)
    return {
        "blocks": blocks,
        "outputs": outputs,
        "dots": int(expected.size),
        "full_mismatches": int(np.count_nonzero(full_dot != expected)),
        "half_mismatches": int(np.count_nonzero(half_dot != expected)),
        "full_half_mismatches": int(np.count_nonzero(full_dot != half_dot)),
        "twos_finish_mismatches": int(np.count_nonzero(twos_dot != expected)),
        "twos_packed16_bound_failures": twos_bound_failures,
        "half_identity_word_mismatches": identity_mismatches,
        "packed_subtraction_borrow_mismatches": borrow_mismatches,
        "maximum_accumulated_plane_byte": max_plane_field,
    }


def main() -> None:
    rng = np.random.default_rng(20260909)
    cases: list[tuple[np.ndarray, np.ndarray]] = []
    # Endpoint and sign/cancellation cases under the actual symmetric A4 domain.
    for av in (-7, 0, 7):
        for wv in (-8, 0, 7):
            cases.append((np.full((1, 4, 32), av, dtype=np.int64),
                          np.full((1, 4, 32), wv, dtype=np.int64)))
    a = rng.integers(-7, 8, size=(20000, 4, 32), dtype=np.int64)
    w = rng.integers(-8, 8, size=(20000, 4, 32), dtype=np.int64)
    # Explicitly hit L1=224 independently in every byte field.
    a[:4] = np.array([-7, -7, 7, 7], dtype=np.int64)[:, None, None]
    w[0].fill(-8); w[1].fill(7)
    cases.append((a, w))
    records = [run_case(aa, ww) for aa, ww in cases]
    totals = {
        "cases": len(records),
        "dots": sum(r["dots"] for r in records),
        "full_mismatches": sum(r["full_mismatches"] for r in records),
        "half_mismatches": sum(r["half_mismatches"] for r in records),
        "full_half_mismatches": sum(r["full_half_mismatches"] for r in records),
        "twos_finish_mismatches": sum(r["twos_finish_mismatches"] for r in records),
        "twos_packed16_bound_failures": sum(r["twos_packed16_bound_failures"] for r in records),
        "half_identity_word_mismatches": sum(r["half_identity_word_mismatches"] for r in records),
        "packed_subtraction_borrow_mismatches": sum(r["packed_subtraction_borrow_mismatches"] for r in records),
        "maximum_accumulated_plane_byte": max(r["maximum_accumulated_plane_byte"] for r in records),
        "production_a4_domain": [-7, 7],
        "native_w4_domain": [-8, 7],
    }
    assert all(totals[k] == 0 for k in (
        "full_mismatches", "half_mismatches", "full_half_mismatches",
        "twos_finish_mismatches", "twos_packed16_bound_failures",
        "half_identity_word_mismatches", "packed_subtraction_borrow_mismatches"))
    assert totals["maximum_accumulated_plane_byte"] == 224
    out = {"seed": 20260909, "totals": totals, "records": records}
    target = Path(__file__).with_name("activation_verify.json")
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
