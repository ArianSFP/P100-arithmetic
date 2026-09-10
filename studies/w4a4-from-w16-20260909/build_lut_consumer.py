from pathlib import Path
import hashlib,json,subprocess

p=Path(__file__).resolve().parent
cmd=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-13','-O3',
     '-std=c++17','-arch=sm_60','-lineinfo','-Xptxas=-v',
     str(p/'lut_consumer.cu'),'-o',str(p/'lut_consumer')]
r=subprocess.run(cmd,capture_output=True,text=True)
(p/'lut-consumer-build.log').write_text(r.stdout+r.stderr);r.check_returncode()
files=['lut_consumer.cu','build_lut_consumer.py','lut_consumer']
(p/'lut-consumer-manifest.json').write_text(json.dumps({
    'command':cmd,'hashes':{name:hashlib.sha256((p/name).read_bytes()).hexdigest()
                            for name in files}},indent=2)+'\n')
