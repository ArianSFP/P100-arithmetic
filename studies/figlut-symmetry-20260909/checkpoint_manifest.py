#!/usr/bin/env python3
"""Write or verify the SHA-256 inventory for this paused study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "CHECKPOINT-MANIFEST.json"


def inventory() -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(ROOT.iterdir()):
        if not path.is_file() or path == TARGET or path.name.startswith(".nfs"):
            continue
        result[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    current = {
        "study": "figlut-symmetry-20260909",
        "status": "paused-after-final-experiment",
        "files": inventory(),
    }
    if args.write:
        TARGET.write_text(json.dumps(current, indent=2) + "\n")
        print(f"WROTE {TARGET.name}: {len(current['files'])} files")
        return
    recorded = json.loads(TARGET.read_text())
    assert recorded == current
    print(f"PASS {TARGET.name}: {len(current['files'])} files")


if __name__ == "__main__":
    main()
