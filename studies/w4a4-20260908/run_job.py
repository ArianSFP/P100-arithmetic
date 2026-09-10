"""Invoke one supervised worker, using the explicitly reviewed reservation."""
import hashlib
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# These digests correspond to the reservation reviewed at acquisition. If a
# different session appends a notice, stop and review it before updating them.
DIGESTS=['6778e47b1f86e3254edfe33c7ae8446ec8e8fad5632bdbb00910243fab1ee796', 'aca6d45a4bc5cf8675a59d05c99fa3db8f73435e23433a92e76b715839de8146']
args=['python3',str(ROOT/'supervise.py'),'--token','w4a4-20260908-1514-gpu3','--coordination-sha256',DIGESTS[0],'--shared-sha256',DIGESTS[1],*sys.argv[1:]]
raise SystemExit(subprocess.call(args))
