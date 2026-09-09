from pathlib import Path
import subprocess,json,hashlib
p=Path(__file__).resolve().parent
cmd=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-13','-O3','-std=c++17','-arch=sm_60','-lineinfo','-Xptxas=-v',str(p/'worker.cu'),'-o',str(p/'worker')]
r=subprocess.run(cmd,capture_output=True,text=True);(p/'build.log').write_text(r.stdout+r.stderr);r.check_returncode()
(p/'manifest.json').write_text(json.dumps({'command':cmd,'hashes':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['worker','worker.cu','q8.inc','lowact.inc','packed.inc','build.py']}},indent=2)+'\n')
