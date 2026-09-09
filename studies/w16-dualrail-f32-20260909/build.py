from pathlib import Path
import hashlib,json,subprocess
p=Path(__file__).resolve().parent
sources=['worker.cu','q8.inc','decode.inc','packed.inc','pair.inc','fallback.inc','w16-pair.inc','w16-fallback.inc','build.py']
cmd=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-13','-O3','-std=c++17','-arch=sm_60','-lineinfo','-Xptxas=-v',str(p/'worker.cu'),'-o',str(p/'worker')]
r=subprocess.run(cmd,capture_output=True,text=True);(p/'build.log').write_text(r.stdout+r.stderr);r.check_returncode()
hashes={f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in sources+['worker']}
(p/'manifest.json').write_text(json.dumps({'command':cmd,'hashes':hashes},indent=2)+'\n')
