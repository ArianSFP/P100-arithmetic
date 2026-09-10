from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

p = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--m", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--experts", type=int, required=True)
ap.add_argument("--ctas", type=int, choices=range(1, 7), default=2)
ap.add_argument("--memcheck", action="store_true")
ap.add_argument("--racecheck", action="store_true")
a = ap.parse_args()
if a.memcheck and a.racecheck:
    raise RuntimeError("choose at most one sanitizer")

coords = [
    p.parents[2] / "bench/COORDINATION-20260908.md",
    Path("/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/bench/COORDINATION-20260908.md"),
]
for coord in coords:
    assert "W4A4-from-W16 research ACTIVE — GPU1 RESERVED" in coord.read_text(), coord

lock = open("/tmp/P100-arithmetic-gpu1.lock", "a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
apps = subprocess.check_output(
    ["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader"],
    text=True,
).strip()
if apps:
    raise RuntimeError("GPU1 has a compute client: " + apps)

def health():
    raw = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,ecc.errors.uncorrected.volatile.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    fields = [x.strip() for x in raw.split(",")]
    if int(fields[-1]) != 0 or int(fields[3]) > 80:
        raise RuntimeError("GPU health guard " + raw)
    return raw

def note(message):
    for coord in coords:
        with coord.open("a") as stream:
            stream.write("\n" + message + "\n")

manifest = json.loads((p / "manifest.json").read_text())
for name, expected in manifest["hashes"].items():
    actual = hashlib.sha256((p / name).read_bytes()).hexdigest()
    assert actual == expected, name

fulltag = "w4a4-from-w16-baseline-" + a.tag
initial = health()
note(f"HELD: {fulltag}; GPU1 reserved device lock; exact-G32 half2 W4A4 pipeline worker.")
meta = {
    "tag": fulltag,
    "device": 1,
    "args": vars(a),
    "initial_gpu": initial,
    "started": time.time(),
    "manifest": manifest,
    "runner_sha256": hashlib.sha256((p / "run_gpu1.py").read_bytes()).hexdigest(),
}
cmd = [
    "env", "-i",
    "PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin",
    "LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64",
    "CUDA_VISIBLE_DEVICES=1",
    "taskset", "--cpu-list", "0-11",
]
if a.memcheck or a.racecheck:
    cmd += [
        "/usr/local/cuda-12.8/bin/compute-sanitizer",
        "--tool", "racecheck" if a.racecheck else "memcheck",
        "--error-exitcode", "99",
    ]
cmd += [
    str(p / "worker"), "--gpu-approved", "1",
    str(a.m), str(a.n), str(a.k), str(a.experts), str(a.ctas),
]
meta["command"] = cmd
child = None
try:
    with (p / (a.tag + ".out")).open("x") as out, (p / (a.tag + ".err")).open("x") as err:
        child = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True)
        deadline = time.monotonic() + 300
        meta["telemetry"] = []
        while child.poll() is None:
            if time.monotonic() > deadline:
                raise RuntimeError("deadline")
            desktop_log = Path("/home/arian/.xsession-errors")
            if desktop_log.exists() and desktop_log.stat().st_size > 50 * 1024**2:
                raise RuntimeError("desktop log guard")
            stats = os.statvfs(p)
            if stats.f_bavail * stats.f_frsize < 5 * 1024**3:
                raise RuntimeError("disk guard")
            meta["telemetry"].append([time.time(), health()])
            time.sleep(1)
        meta["exit_code"] = child.returncode
        if child.returncode:
            raise RuntimeError("worker failed; no automatic retry")
finally:
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    meta["finished"] = time.time()
    meta["final_gpu"] = health()
    (p / (a.tag + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
    note(f"FINAL RELEASE: {fulltag}; GPU1 worker lock released, exit={meta.get('exit_code')}, reservation retained, no reset.")
    lock.close()

print("PASS", a.tag, "GPU1")
