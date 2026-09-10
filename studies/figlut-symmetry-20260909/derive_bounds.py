#!/usr/bin/env python3
"""Regenerate the target-rate and ideal shared-store lower-bound arithmetic."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MACS = 64 * 256 * 512 * 2048
BASELINES_US = {"gate_up": 2462.50, "down": 2353.96}
CONSUMER_RATES = {
    "historical_best_full256_r16": 12.6748,
    "fresh_full256_r16": 12.33092615,
}
SM_COUNT = 56
BANKS_PER_SM = 32
BYTES_PER_BANK_CYCLE = 4
OBSERVED_CLOCK_HZ = 1.328e9
TABLE_STORE_BYTES = 4 * 1024**3


def main() -> None:
    peak_bytes_s = (
        SM_COUNT * BANKS_PER_SM * BYTES_PER_BANK_CYCLE * OBSERVED_CLOCK_HZ
    )
    store_us = TABLE_STORE_BYTES / peak_bytes_s * 1e6
    targets = {}
    for name, baseline_us in BASELINES_US.items():
        target_us = baseline_us / 1.3
        targets[name] = {
            "baseline_us": baseline_us,
            "target_us": target_us,
            "required_tmac_s": MACS / (target_us * 1e6),
        }
    consumers = {}
    for name, rate in CONSUMER_RATES.items():
        consumer_us = MACS / (rate * 1e6)
        total_us = consumer_us + store_us
        consumers[name] = {
            "rate_tmac_s": rate,
            "projected_consumer_us": consumer_us,
            "consumer_plus_ideal_store_us": total_us,
            "gate_up_margin_us": targets["gate_up"]["target_us"] - total_us,
            "down_margin_us": targets["down"]["target_us"] - total_us,
        }
    report = {
        "useful_macs_per_projection": MACS,
        "targets": targets,
        "full_table_store_bytes_per_projection": TABLE_STORE_BYTES,
        "ideal_shared_store_peak_bytes_s": peak_bytes_s,
        "ideal_shared_store_us": store_us,
        "consumers": consumers,
        "assumption": (
            "Ideal bank peak: every bank accepts one 32-bit store every cycle; "
            "no builder arithmetic, decode, addressing, barriers, masks, finish, "
            "scales, tails, planning or output is charged."
        ),
    }
    (ROOT / "bounds.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
