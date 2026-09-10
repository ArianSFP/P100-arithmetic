#!/usr/bin/env python3
"""Read-only consistency checks for the paused FIGLUT study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
GPU_UUID = "GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b"


def load(name: str):
    return json.loads((ROOT / name).read_text())


def close(a: float, b: float, tolerance: float = 1e-8) -> None:
    assert abs(a - b) <= tolerance * max(1.0, abs(a), abs(b)), (a, b)


def verify_meta(name: str) -> None:
    meta = load(name)
    assert meta["exit_code"] == 0
    assert meta["device_uuid"] == GPU_UUID
    assert meta["command"][meta["command"].index("CUDA_VISIBLE_DEVICES=" + GPU_UUID)]
    for artifact, expected in meta["hashes"].items():
        actual = hashlib.sha256((ROOT / artifact).read_bytes()).hexdigest()
        assert actual == expected, (artifact, actual, expected)


def main() -> None:
    packed_half = load("verify_packed_half.json")
    assert packed_half["identity_mismatches"] == 0
    assert packed_half["row_table_values_checked"] == 21_437_440
    assert packed_half["packed_subtractions_checked"] == 5_359_360
    assert packed_half["g32_endpoint_normalization"][
        "packed_four_slice_accumulation_mismatches"
    ] == 0

    activation = load("activation_verify.json")["totals"]
    assert activation["dots"] == 320_144
    assert activation["production_a4_domain"] == [-7, 7]
    assert activation["maximum_accumulated_plane_byte"] == 224
    for key in (
        "full_mismatches",
        "half_mismatches",
        "full_half_mismatches",
        "twos_finish_mismatches",
        "twos_packed16_bound_failures",
        "half_identity_word_mismatches",
        "packed_subtraction_borrow_mismatches",
    ):
        assert activation[key] == 0, key

    prebuilt = load("prebuilt_summary.json")
    close(prebuilt["prebuilt-full256-r16"]["median_useful_tmac_s"], 12.33092615)
    close(prebuilt["prebuilt-full256-r16"]["speedup_vs_matched_half2"], 1.3236079250450508)
    for rows in (16, 8, 4):
        assert 0.50 < prebuilt[f"prebuilt-half128-r{rows}"]["speedup_vs_matched_half2"] < 0.53

    packed = load("packed4_summary.json")
    candidate = packed["packed4-full16-complete-g32"]
    control = packed["half2-complete-lower-bound"]
    close(candidate["median_useful_tmac_s"], 2.99577131)
    close(control["median_useful_tmac_s"], 6.17128128)
    close(candidate["speedup_vs_half2_lower_bound"], 0.4854169953489802)
    assert not candidate["passes_10p25_tmac_streamed_admission"]
    assert not candidate["passes_9p488_down_mathematical_rate"]
    assert not candidate["passes_9p070_gu_mathematical_rate"]

    out = (ROOT / "figlut-packed4-full16-r1.out").read_text()
    assert "CHECK groups=8 outputs=8192 bit_mismatches=0" in out
    assert out.rstrip().endswith("PASS")
    invalid = (ROOT / "figlut-packed4-full16-r0.out").read_text()
    rates = [
        float(value)
        for value in re.findall(
            r"mode=half2-complete-lower-bound .*?useful_tmac_s=([0-9.]+)", invalid
        )
    ]
    assert min(rates) > 50.0  # compiler-hoisted control; retained but excluded

    counts = load("packed4_sass_counts.json")
    full_symbol = next(name for name in counts if "packed4_full16" in name)
    full = counts[full_symbol]
    for op, expected in {
        "SHFL": 172,
        "BFE": 107,
        "IADD3": 82,
        "IADD": 51,
        "I2F": 16,
        "FMUL": 16,
        "FFMA": 16,
    }.items():
        assert full[op] == expected, (op, full[op], expected)

    bounds = load("bounds.json")
    assert bounds["useful_macs_per_projection"] == 17_179_869_184
    close(bounds["targets"]["gate_up"]["target_us"], 1894.230769230769)
    close(bounds["targets"]["down"]["target_us"], 1810.7384615384615)
    close(bounds["ideal_shared_store_us"], 451.1944922546718)
    assert bounds["consumers"]["fresh_full256_r16"]["down_margin_us"] < 0

    for name in (
        "figlut-half128-prebuilt-r0.meta.json",
        "figlut-packed4-full16-r1.meta.json",
    ):
        verify_meta(name)
    assert (ROOT / "figlut-half128-prebuilt-r0.err").stat().st_size == 0
    assert (ROOT / "figlut-packed4-full16-r1.err").stat().st_size == 0

    release = load("gpu-release.json")
    assert release["device_uuid"] == GPU_UUID
    assert release["compute_clients"] == []
    assert release["device_lock_available"] is True
    assert release["volatile_uncorrected_ecc_errors"] == 0
    assert release["reset_performed"] is False

    print("PASS: FIGLUT paused checkpoint is internally consistent")


if __name__ == "__main__":
    main()
