#!/usr/bin/env python3
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

root = Path(__file__).resolve().parent
tag = "figlut-half128-prebuilt-r0"
coords = [
    root.parents[1] / "bench/COORDINATION-20260908.md",
    Path("/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/bench/COORDINATION-20260908.md"),
]
marker = "FIGLU symmetry follow-up ACTIVE — GPU1 RESERVED"
for coord in coords:
    assert marker in coord.read_text(), coord

lock = open("/tmp/P100-arithmetic-gpu1.lock", "a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def query(args):
    return subprocess.check_output(["nvidia-smi", "-i", "1", *args], text=True).strip()

apps = query(["--query-compute-apps=pid", "--format=csv,noheader"])
if apps:
    raise RuntimeError("GPU1 has a compute client: " + apps)

def health():
    raw = query(["--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,pstate,ecc.errors.uncorrected.volatile.total", "--format=csv,noheader,nounits"])
    values = [x.strip() for x in raw.split(",")]
    if values[0] != "GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b":
        raise RuntimeError("unexpected GPU UUID: " + raw)
    if int(values[-1]) != 0 or int(values[3]) > 80:
        raise RuntimeError("GPU health guard: " + raw)
    return raw

def note(message):
    for coord in coords:
        with coord.open("a") as stream:
            stream.write("\n" + message + "\n")

inputs = ["prebuilt_worker.cu", "prebuilt_worker", "prebuilt_worker.sass", "prebuilt_build.log"]
hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in inputs}
initial = health()
note("HELD: figlut-half128-prebuilt-r0; GPU1 device lock; bounded exact full256/half128/HFMA2 ceiling gate.")
meta = {
    "tag": tag,
    "device_index": 1,
    "device_uuid": "GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b",
    "initial_gpu": initial,
    "started": time.time(),
    "hashes": hashes,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
command = [
    "env", "-i",
    "PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin",
    "LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu",
    "CUDA_VISIBLE_DEVICES=GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b",
    "taskset", "--cpu-list", "0-11",
    str(root / "prebuilt_worker"), "--gpu-approved", "1", "512",
]
meta["command"] = command
child = None
try:
    with (root / f"{tag}.out").open("x") as stdout, (root / f"{tag}.err").open("x") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        deadline = time.monotonic() + 180
        meta["telemetry"] = []
        while child.poll() is None:
            if time.monotonic() > deadline:
                raise RuntimeError("worker deadline")
            desktop = Path("/home/arian/.xsession-errors")
            if desktop.exists() and desktop.stat().st_size > 50 * 1024**2:
                raise RuntimeError("desktop log guard")
            stat = os.statvfs(root)
            if stat.f_bavail * stat.f_frsize < 5 * 1024**3:
                raise RuntimeError("disk guard")
            meta["telemetry"].append([time.time(), health()])
            time.sleep(1)
        meta["exit_code"] = child.returncode
        if child.returncode:
            raise RuntimeError("worker failed; no automatic retry")
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    meta["finished"] = time.time()
    meta["final_gpu"] = health()
    (root / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    note(f"FINAL RELEASE: figlut-half128-prebuilt-r0; GPU1 worker lock released, exit={meta.get('exit_code')}, reservation retained, no reset.")
    lock.close()

print("PASS", tag, "GPU1")
