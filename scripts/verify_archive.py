#!/usr/bin/env python3
"""Read-only integrity and round-2 log checks; never import CUDA or run workers."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import sys


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archive_files(archive):
    require(archive.is_dir() and not archive.is_symlink(), "Invalid archive root")
    files = {}
    for directory, dirs, names in os.walk(archive, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            require(not path.is_symlink(), f"Symlink rejected: {path}")
        for name in names:
            path = Path(directory) / name
            require(path.is_file(), f"Non-regular file: {path}")
            files[path.relative_to(archive).as_posix()] = path
    return dict(sorted(files.items()))


def safe_manifest_path(name):
    require(isinstance(name, str) and name and "\\" not in name,
            "Invalid manifest path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts
            and path.parts and path.as_posix() == name,
            f"Unsafe/noncanonical manifest path: {name}")
    return name


def verify_manifest(archive, manifest):
    require(manifest["schema_version"] == 1, "Unsupported manifest schema")
    files = archive_files(archive)
    recorded = {}
    for entry in manifest["files"]:
        name = safe_manifest_path(entry["path"])
        require(name not in recorded, f"Duplicate manifest entry: {name}")
        require(type(entry["size"]) is int and entry["size"] >= 0,
                f"Invalid file size: {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None,
                f"Invalid SHA-256: {name}")
        recorded[name] = entry
    require(files.keys() == recorded.keys(),
            f"File set differs: missing={sorted(recorded.keys() - files.keys())[:5]}, "
            f"extra={sorted(files.keys() - recorded.keys())[:5]}")
    for name, path in files.items():
        entry = recorded[name]
        require(path.stat().st_size == entry["size"], f"Size changed: {name}")
        require(sha256(path) == entry["sha256"], f"SHA-256 changed: {name}")
    size = sum(entry["size"] for entry in recorded.values())
    require(len(files) == manifest["file_count"], "Manifest file count mismatch")
    require(size == manifest["total_bytes"], "Manifest byte count mismatch")
    return {"files": len(files), "bytes": size}


def fields(line):
    return dict(re.findall(r"(\w+)=([^ ]+)", line))


def config(data):
    return tuple(int(data[key]) for key in ("method", "R", "S"))


def shape(data):
    return tuple(int(data[key]) for key in ("M", "K", "N"))


def retained_median(samples, rounds=9):
    require(sorted(r for r, _ in samples) == list(range(rounds)),
            "Missing, duplicate or unexpected timing round")
    require(all(math.isfinite(value) and value > 0 for _, value in samples),
            "Invalid timing value")
    return statistics.median(value for r, value in samples if r > 0)


def verify_round2(archive):
    study = archive / "w4a16-r2-20260908"
    inventory = read_json(study / "build/inventory.json")
    audit = read_json(study / "audit.json")
    frozen = read_json(study / "frozen-manifest.json")
    aggregate = read_json(study / "aggregate.json")
    require(audit["status"] == "PASS", "Final audit did not pass")
    require(inventory["hashes"] == audit["build_hashes"], "Audit/build mismatch")
    for name, digest in inventory["hashes"].items():
        path = (study / name).resolve()
        require(path.is_relative_to(archive.resolve()), "Build hash escapes archive")
        require(sha256(path) == digest, f"Round-2 build hash changed: {name}")
    require(sha256(study / "build/inventory.json") == frozen["inventory_sha256"],
            "Frozen inventory changed")
    config_sha = sha256(study / "frozen-configs.txt")
    require(config_sha == frozen["configuration_sha256"] == audit["configuration_sha256"],
            "Frozen configs changed")
    require(sha256(study / "RESULTS.md") == audit["report_sha256"], "Final report changed")
    require(sha256(study / "supervise.py") == audit["supervisor_sha256"],
            "Supervisor changed")
    configs = {tuple(c) for c in frozen["configs"]}
    require(len(configs) == 12, "Unexpected frozen configuration count")
    config_lines = {tuple(map(int, line.split())) for line in
                    (study / "frozen-configs.txt").read_text().splitlines()}
    require(config_lines == configs, "Frozen config file/manifest disagree")
    categories = Counter()
    checks_total = full_outputs = 0
    references = {}
    benchmark_workers = defaultdict(list)
    cells = defaultdict(list)
    for result_path in sorted((study / "gpu-results").glob("*/result.json")):
        run = result_path.parent
        rec = read_json(result_path)
        require(rec["status"] == "PASS" and rec["returncode"] == 0,
                f"Worker failed: {run.name}")
        require(all(rec["hashes"].get(k) == v for k, v in inventory["hashes"].items()),
                f"Worker build provenance differs: {run.name}")
        require(rec["supervisor_sha256"] == audit["supervisor_sha256"],
                f"Worker supervisor differs: {run.name}")
        pre = read_json(run / "prelaunch.json")
        require(pre["status"] == "prelaunch", f"Missing prelaunch state: {run.name}")
        for key in ("argv", "hashes", "gpu", "gpu_uuid", "token",
                    "supervisor_sha256", "coordination_sha256", "sanitizer"):
            require(rec[key] == pre[key], f"Prelaunch provenance changed: {run.name}/{key}")
        require(not (run / "stderr.txt").read_text().strip(), f"Worker stderr: {run.name}")
        log = (run / "stdout.txt").read_text()
        checks = [fields(line) for line in log.splitlines() if line.startswith("CHECK ")]
        for check in checks:
            require(check["operation_order_bitexact"] == "1"
                    and check["baseline_changed"] == "0" and check["S"] == "32",
                    f"Nonidentical pipeline: {run.name}")
            ref_key = config(check) + shape(check) + (int(check["family"]),)
            require(ref_key not in references or references[ref_key] == check,
                    f"Reference metrics differ: {run.name}")
            references[ref_key] = check
        checks_total += len(checks)
        if rec["sanitizer"]:
            category = rec["sanitizer"]
            require(category in ("memcheck", "synccheck"), "Unexpected sanitizer")
            require("SMOKE_PASS" in log and "ERROR SUMMARY: 0 errors" in log
                    and len(checks) == 79, f"Sanitizer evidence incomplete: {run.name}")
        elif "validate" in rec["argv"]:
            category = "full_validation"
            require("VALIDATE_PASS" in log and "EXHAUSTIVE_FINITE_ACTIVATIONS_PASS" in log
                    and len(checks) == 1027, f"Validation incomplete: {run.name}")
            full_outputs += sum(int(check["outputs"]) for check in checks)
        else:
            category = "bench" if "bench" in rec["argv"] else "sweep"
            require(category in rec["argv"] and "BENCH_PASS" in log,
                    f"Unknown/incomplete worker: {run.name}")
            require(len(checks) == (12 if category == "bench" else 79),
                    f"Unexpected check count: {run.name}")
        categories[category] += 1
        if category != "bench":
            continue
        require(rec["hashes"]["config_file"] == config_sha, "Worker configs changed")
        shapes = {shape(check) for check in checks}
        require(len(shapes) == 1 and {config(check) for check in checks} == configs,
                f"Benchmark coverage differs: {run.name}")
        worker_shape = shapes.pop()
        benchmark_workers[rec["gpu"], worker_shape].append(run.name)
        times = defaultdict(list)
        for line in log.splitlines():
            if line.startswith("TIME "):
                data = fields(line)
                require(shape(data) == worker_shape, "Timing shape differs")
                times[config(data)].append((int(data["round"]), float(data["pipeline_us"])))
        require(times.keys() == configs, f"Timing config coverage differs: {run.name}")
        for check in checks:
            key = config(check)
            cells[(rec["gpu"],) + worker_shape + key].append(
                (run.name, retained_median(times[key]), check))
    expected_categories = dict(sweep=4, bench=36, full_validation=3, memcheck=1, synccheck=1)
    require(dict(categories) == expected_categories == audit["worker_categories"],
            "Worker categories/counts differ")
    require(checks_total == 3987 == audit["all_pipeline_checks"], "Pipeline count differs")
    require(full_outputs == 2358624, "Full-suite output count differs")
    shapes = {(5120, 5120, 1), (17408, 5120, 1), (5120, 17408, 1), (5120, 5120, 4)}
    require(set(benchmark_workers) == {(g, s) for g in (1, 2, 3) for s in shapes}
            and all(len(ids) == 3 for ids in benchmark_workers.values()),
            "Need three fresh workers per device/shape")
    require(aggregate["manifest"] == frozen and aggregate["worker_count"] == 36
            and aggregate["configuration_checks"] == 432
            and aggregate["retained_event_samples"] == 3456,
            "Aggregate provenance/counts differ")
    require(audit["frozen_workers"] == 36 and audit["frozen_checks"] == 432
            and audit["retained_samples"] == 3456, "Audit frozen counts differ")
    seen = set()
    for row in aggregate["rows"]:
        key = (row["gpu"],) + shape(row) + config(row)
        require(key in cells and key not in seen, "Missing/duplicate aggregate key")
        seen.add(key)
        values = cells[key]
        worker_us = [value[1] for value in values]
        require(row["runs"] == [value[0] for value in values]
                and row["worker_us"] == worker_us
                and row["median_us"] == statistics.median(worker_us)
                and row["min_worker_us"] == min(worker_us)
                and row["max_worker_us"] == max(worker_us), f"Aggregate timings differ: {key}")
        for name in ("outputs", "baseline_changed", "max_abs", "max_L1_relative", "relative_L2"):
            require(all(float(value[2][name]) == row[name] for value in values),
                    f"Aggregate numerical metric differs: {key}/{name}")
    require(seen == cells.keys() and len(seen) == 144, "Aggregate coverage differs")
    with (study / "aggregate.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(csv_rows == [{k: str(v) for k, v in row.items()} for row in aggregate["rows"]],
            "CSV/JSON aggregates disagree")
    return {"workers": sum(categories.values()), "pipeline_checks": checks_total,
            "frozen_workers": 36, "retained_samples": 3456}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        archive = args.root / "archive"
        manifest = verify_manifest(archive, read_json(args.root / "EXPORT-MANIFEST.json"))
        latest = verify_round2(archive)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"VERIFY_FAIL: {error}", file=sys.stderr)
        return 1
    print("ARCHIVE_PASS " + " ".join(f"{k}={v}" for k, v in manifest.items()))
    print("ROUND2_PASS " + " ".join(f"{k}={v}" for k, v in latest.items()))
    print("Read-only evidence verification; no CUDA, GPU query or binary execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
