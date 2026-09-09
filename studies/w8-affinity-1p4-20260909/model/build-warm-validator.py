from pathlib import Path
import subprocess,hashlib,json
p=Path(__file__).resolve().parent;r=Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906')
cmd=['/usr/bin/g++-14','-std=c++17','-O3','-DNDEBUG','-I'+str(r/'common'),'-I'+str(r/'include'),'-I'+str(r/'ggml/include'),str(p/'validate-warm.cpp'),'-L'+str(p/'build/bin'),'-Wl,-rpath,'+str(p/'build/bin'),'-lllama-common',str(r/'build-coding-serve/common/libllama-common-base.a'),'-lllama','-lggml','-lggml-base','-pthread','-ldl','-o',str(p/'test-aw-warm')]
subprocess.run(cmd,check=True)
(p/'warm-validator-manifest.json').write_text(json.dumps({'command':cmd,'hashes':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['validate-warm.cpp','test-aw-warm','build-warm-validator.py']}},indent=2)+'\n')
