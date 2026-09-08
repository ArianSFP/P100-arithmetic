#!/usr/bin/env python3
"""One-candidate supervisor; it never initializes CUDA itself."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    candidate_bytes = args.candidate.read_bytes()
    record = {
        "candidate": str(args.candidate),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "worker": str(args.worker),
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        "cuda_visible_devices": args.gpu,
        "timeout_seconds": args.timeout,
        "launch_unix_ns": time.time_ns(),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    prelaunch = args.log.with_suffix(args.log.suffix + ".prelaunch.json")
    prelaunch.write_text(json.dumps(record, indent=2) + "\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(args.worker), str(args.candidate)],
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        record.update({
            "status": "completed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        })
        code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        record.update({
            "status": "timeout",
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        })
        code = 124
    record["elapsed_seconds"] = time.monotonic() - started
    args.log.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "returncode", "candidate_sha256", "elapsed_seconds")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
