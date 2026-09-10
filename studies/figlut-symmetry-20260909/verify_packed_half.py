#!/usr/bin/env python3
"""CPU proof checks for applying FIGLUT symmetry to the W4A4 mu8 table.

No GPU timing is performed. Four independent signed-INT4 rows are packed in
the four bytes of one uint32 table value. For one mu8 slice:

  B       = sum(max(-w_j, 0))
  U(mask) = B + sum(mask_j * w_j)
  C       = sum(abs(w_j))

Then U(~mask) = C - U(mask). Keeping only masks with bit 7 clear therefore
halves the table. The packed uint32 subtraction is byte-lane exact because
0 <= U(mask) <= C <= 64 independently in every byte, so no borrow can cross a
byte boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MASKS = np.arange(256, dtype=np.uint16)
BITS = ((MASKS[:, None] >> np.arange(8, dtype=np.uint16)) & 1).astype(np.int16)


def pack_u8(x: np.ndarray) -> np.ndarray:
    """Pack final axis of four values in [0,255] into uint32."""
    x = np.asarray(x, dtype=np.uint32)
    assert x.shape[-1] == 4
    assert np.all(x <= 255)
    return sum(x[..., j] << np.uint32(8 * j) for j in range(4)).astype(np.uint32)


def corpus(rng: np.random.Generator, random_cases: int) -> np.ndarray:
    cases: list[np.ndarray] = []
    constants = [-8, -7, -1, 0, 1, 6, 7]
    for value in constants:
        cases.append(np.full((4, 8), value, dtype=np.int16))
    # One-coordinate endpoint and sign changes in every row and position.
    for base in (-8, 0, 7):
        for replacement in (-8, -7, -1, 0, 1, 6, 7):
            for row in range(4):
                for pos in range(8):
                    x = np.full((4, 8), base, dtype=np.int16)
                    x[row, pos] = replacement
                    cases.append(x)
    # Exhaust all 16^2 ordered pairs, repeated over four positions. This covers
    # every signed-INT4 value transition and many mixed-sign borrow boundaries.
    for a in range(-8, 8):
        for b in range(-8, 8):
            x = np.empty((4, 8), dtype=np.int16)
            for row in range(4):
                x[row, 0::2] = a if row & 1 else b
                x[row, 1::2] = b if row & 1 else a
            cases.append(x)
    edge = np.stack(cases)
    random = rng.integers(-8, 8, size=(random_cases, 4, 8), dtype=np.int16)
    return np.concatenate((edge, random), axis=0)


def verify(seed: int, random_cases: int) -> dict:
    rng = np.random.default_rng(seed)
    weights = corpus(rng, random_cases)
    bias = np.maximum(-weights, 0).sum(axis=-1, dtype=np.int16)
    cap = np.abs(weights).sum(axis=-1, dtype=np.int16)
    assert np.all(cap <= 64)

    # full[c,r,m] is the ordinary biased 256-entry table for row r.
    full = bias[:, :, None] + np.einsum("cri,mi->crm", weights, BITS)
    assert np.all(full >= 0)
    assert np.all(full <= cap[:, :, None])
    half = full[:, :, :128]

    recovered = np.empty_like(full)
    packed_checks = 0
    for mask in range(256):
        flip = mask >> 7
        index = (mask & 127) ^ (127 if flip else 0)
        canonical = half[:, :, index]
        got = cap - canonical if flip else canonical
        assert np.array_equal(got, full[:, :, mask])

        packed_canonical = pack_u8(canonical)
        packed_got = (pack_u8(cap) - packed_canonical).astype(np.uint32) if flip else packed_canonical
        expected = pack_u8(got)
        assert np.array_equal(packed_got, expected)
        packed_checks += len(weights)

    # Verify four mu8 slices can accumulate in byte lanes after the exact
    # all-minus-eight endpoint normalization used by the priority-3 design.
    g32_cases = random_cases
    v = rng.integers(-8, 8, size=(g32_cases, 4, 32), dtype=np.int16)
    if g32_cases >= 4:
        v[:4] = -8
        v[1, 0, 0] = -7
        v[2, 1, 17] = 7
        v[3, 2, :] = 7
    exceptional = np.all(v == -8, axis=-1)
    vn = np.where(exceptional[..., None], -4, v)
    factor = np.where(exceptional, 2, 1)
    l1 = np.abs(vn).sum(axis=-1)
    assert np.all(l1 <= 255)
    assert np.array_equal(vn * factor[..., None], v)

    selectors = rng.integers(0, 256, size=(g32_cases, 4), dtype=np.uint16)
    lane_acc = np.zeros((g32_cases, 4), dtype=np.int16)
    packed_acc = np.zeros(g32_cases, dtype=np.uint32)
    for sl in range(4):
        part = vn[:, :, 8 * sl:8 * sl + 8]
        b = np.maximum(-part, 0).sum(axis=-1, dtype=np.int16)
        c = np.abs(part).sum(axis=-1, dtype=np.int16)
        m = selectors[:, sl]
        flip = m >> 7
        index = (m & 127) ^ (flip * 127)
        selected = ((index[:, None, None] >> np.arange(8)) & 1).astype(np.int16)
        canonical = b + (part * selected).sum(axis=-1)
        got = np.where(flip[:, None], c - canonical, canonical)
        direct_bits = ((m[:, None, None] >> np.arange(8)) & 1).astype(np.int16)
        expected = b + (part * direct_bits).sum(axis=-1)
        assert np.array_equal(got, expected)
        lane_acc += got
        packed_acc = (packed_acc + pack_u8(got)).astype(np.uint32)

    assert np.all(lane_acc <= 255)
    assert np.array_equal(packed_acc, pack_u8(lane_acc))

    return {
        "seed": seed,
        "gpu_timing_performed": False,
        "mu": 8,
        "packed_rows": 4,
        "table_entries_full": 256,
        "table_entries_half": 128,
        "shared_bytes_full_table_32_columns": 256 * 32 * 4,
        "shared_bytes_half_table_32_columns": 128 * 32 * 4,
        "weight_vectors_checked": int(len(weights)),
        "masks_per_vector": 256,
        "row_table_values_checked": int(len(weights) * 256 * 4),
        "packed_subtractions_checked": int(packed_checks),
        "identity_mismatches": 0,
        "maximum_mu8_lane_cap": int(cap.max()),
        "g32_endpoint_normalization": {
            "vectors": int(g32_cases * 4),
            "all_minus_eight_vectors": int(exceptional.sum()),
            "maximum_normalized_l1": int(l1.max()),
            "packed_four_slice_accumulation_mismatches": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--random-cases", type=int, default=20000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = verify(args.seed, args.random_cases)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
