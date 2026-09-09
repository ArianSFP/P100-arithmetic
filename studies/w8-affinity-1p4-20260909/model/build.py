from pathlib import Path
import json,shlex,subprocess,hashlib,os
p=Path(__file__).resolve().parent
root=Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906')
b=root/'build-coding-serve'; cu=b/'ggml/src/ggml-cuda'; cm=cu/'CMakeFiles/ggml-cuda.dir'
(p/'build/bin').mkdir(parents=True,exist_ok=True);(p/'objects').mkdir(exist_ok=True)
manifest={}
objects=[]
for i,v in enumerate(shlex.split((cm/'objects1.rsp').read_text())):
 src=(cu/v).resolve();dst=p/'objects'/f'{i:04d}-{src.name}'
 if src.name=='affinity-wave.cu.o':dst=p/'affinity-wave.cu.o'
 else:
  subprocess.run(['cp','--reflink=auto',str(src),str(dst)],check=True)
  with src.open('rb') as f:manifest[str(src)]=hashlib.file_digest(f,'sha256').hexdigest()
 objects.append(str(dst))
(p/'objects.rsp').write_text(' '.join(shlex.quote(x) for x in objects))
for src in (b/'bin').iterdir():
 if src.is_file() and ('.so' in src.name or src.name in ['llama-bench','llama-perplexity','llama-server','test-aw-serving']):
  subprocess.run(['cp','-L','--reflink=auto',str(src),str(p/'build/bin'/src.name)],check=True)
flags={}
for line in (cm/'flags.make').read_text().splitlines():
 if ' = ' in line:
  k,v=line.split(' = ',1);flags[k]=shlex.split(v)
cmd=['/usr/local/cuda-12.8/bin/nvcc','-ccbin','/usr/bin/g++-14']+flags['CUDA_DEFINES']+flags['CUDA_FLAGS']+['--options-file',str(cm/'includes_CUDA.rsp'),'-I',str(root/'ggml/src/ggml-cuda'),'-c',str(p/'affinity-wave.cu'),'-o',str(p/'affinity-wave.cu.o')]
manifest['compile_command']=cmd
(p/'build-inputs.json').write_text(json.dumps(manifest,indent=2))
print('Compiling isolated CUDA TU',flush=True)
subprocess.run(cmd,cwd=cu,check=True)
libs=shlex.split((cm/'linkLibs.rsp').read_text())
link=['/usr/bin/g++-14','-fPIC','-shared','-Wl,-soname,libggml-cuda.so.0','-o',str(p/'build/bin/libggml-cuda.so.0.15.3'),'@'+str(p/'objects.rsp')]+libs+['-L/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs','-L/usr/local/cuda-12.8/targets/x86_64-linux/lib']
subprocess.run(link,cwd=cu,check=True)
for name in ['libggml-cuda.so','libggml-cuda.so.0']:
 dst=p/'build/bin'/name;dst.unlink();dst.symlink_to('libggml-cuda.so.0.15.3')
print('Build complete',flush=True)
