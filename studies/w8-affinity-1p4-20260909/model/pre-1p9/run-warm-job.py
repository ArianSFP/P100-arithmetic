import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,resource,subprocess,time
p=pathlib.Path(__file__).resolve().parent
root=pathlib.Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906')
spec=importlib.util.spec_from_file_location('runner',root/'bench/coding-serve/run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
parser=argparse.ArgumentParser();parser.add_argument('tag');parser.add_argument('--kind',choices=['semantic','quality','prefill','decode','server'],required=True);parser.add_argument('--control',action='store_true');parser.add_argument('--native',action='store_true');parser.add_argument('--partial',choices=['f32','bf16']);parser.add_argument('--diagonal',choices=['legacy','panel2048']);parser.add_argument('--plan-cache',choices=['0','1']);parser.add_argument('--memcheck',action='store_true');parser.add_argument('--tokens',type=int,choices=[2048,8128,8192],default=8192);parser.add_argument('--f32weights',action='store_true');args=parser.parse_args()
build=root/'build-coding-serve' if args.control else p.parent/'qwen35-q4-t64-20260908/build' if args.native else p/'build'
env=r.environment(build,args.tokens,'normal' if args.control else 'serve' if args.kind in ['decode','server'] else 'qualified')
if args.kind in ['decode','server'] and not args.control:
 for key,value in dict(PARTIAL='serve-auto',SERVE_KV_PREFIX='1',SERVE_KV_SUFFIX='1',SERVE_PLAN_CACHE='1',SERVE_PLAN_CACHE_SIZE='4',SERVE_PUBLISH_OVERLAP='1',PRECAPTURE='0',SERVE_RESERVE_FULL='1',SERVE_TINY_MMVQ='1').items():env['GGML_CUDA_AW_'+key]=value
# Decode uses the serving profile, which switches from prefill to native Q4 decode.
env['GGML_CUDA_DISABLE_GRAPHS']='1'
env['GGML_CUDA_AW_PRECAPTURE']='0'
if args.native or args.control or args.f32weights:raise RuntimeError('paired validator selects its own control and candidate modes')
if args.partial:env['GGML_CUDA_AW_PARTIAL']=args.partial
if args.diagonal:env['GGML_CUDA_AW_DIAGONAL_SERVICE']=args.diagonal
if args.plan_cache:env['GGML_CUDA_AW_SERVE_PLAN_CACHE']=args.plan_cache
model=os.environ.get('W8_MODEL_PATH','/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf')
if not pathlib.Path(model).is_file():raise RuntimeError('Original Q8 model missing; restore pinned model or set W8_MODEL_PATH')
base=['-m',model,'-ngl','99','-sm','tensor','-fa','1','-b',str(args.tokens),'-ub',str(args.tokens),'-t','12']
if args.kind=='semantic':
 env['CUDA_VISIBLE_DEVICES']='1'
 cmd=['/usr/bin/python3','-c','import ctypes; lib=ctypes.CDLL("'+str(build/'bin/libggml-cuda.so')+'"); raise SystemExit(lib.aw_q4_semantic_test())']
elif args.kind=='quality':
 env['GGML_CUDA_AW_VALIDATION_OUTPUT']=str(p/(args.tag+'.logits'))
 env['GGML_CUDA_AW_VALIDATION_TOKENS']=str(args.tokens)
 cmd=[str(p/'test-aw-warm')]+base+['-c','16384','-np','1','-tb','12','--no-warmup','--no-mmap','-f','/home/arian/llama.cpp-q36-decodeopt/wikitext-2-raw/wiki.test.raw']
elif args.kind=='server':
 cmd=[str(build/'bin/llama-server')]+base+['-c','16384','-np','1','-tb','12','--no-warmup','--no-mmap','--host','127.0.0.1','--port','18109','--no-webui','--jinja']
else:
 cmd=[str(build/'bin/llama-bench')]+base+['-mmp','0','-p',str(args.tokens) if args.kind=='prefill' else '0','-n','128' if args.kind=='decode' else '0','-r','4','-o','json']
 if args.kind=='decode':cmd+=['-d','0,8128']
if args.memcheck:cmd=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99']+cmd
cmd=['env','-i']+[k+'='+v for k,v in env.items()]+['taskset','--cpu-list','0-11']+cmd
lock=open('/tmp/affinitywave-4gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
device_locks=[]
for device in range(4):
 device_lock=open('/tmp/P100-arithmetic-gpu'+str(device)+'.lock','a');fcntl.flock(device_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device_locks.append(device_lock)
clients=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
if any('/usr/bin/gnome-text-editor' not in l for l in clients.splitlines() if l.strip()):raise RuntimeError(clients)
def gpu():return subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw','--format=csv'],text=True)
coords=[root/'bench/COORDINATION-20260908.md',p.parents[2]/'bench/COORDINATION-20260908.md']
for c in coords:
 with c.open('a') as f:f.write('\nW8A16 optimization HELD '+args.tag+' '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+': all four GPUs reserved under shared lock; fresh worker and desktop/disk/deadline guards.\n')
meta={'tag':args.tag,'args':vars(args),'command':cmd,'environment':env,'started':time.time(),'initial_gpu':gpu(),'binary_sha256':{str(x):r.fingerprint(x) for x in (build/'bin').iterdir() if x.is_file() and ('.so' in x.name or x.name in ['llama-bench','test-aw-serving'])}}
meta['validator_sha256']=hashlib.sha256((p/'test-aw-warm').read_bytes()).hexdigest()
(p/(args.tag+'.meta.json')).write_text(json.dumps(meta,indent=2));resource.setrlimit(resource.RLIMIT_CORE,(0,0));child=None
try:
 with (p/(args.tag+'.out')).open('x') as out,(p/(args.tag+'.err')).open('x') as err,(p/(args.tag+'.gpu.csv')).open('x') as telemetry:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
  future=None;pool=None
  if args.kind=='server':
   import concurrent.futures,importlib.util
   spec=importlib.util.spec_from_file_location('probe',p/'server-probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
   pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);future=pool.submit(probe.probe,args.tag,lambda:child.poll() is None)
  print('pid',child.pid,flush=True);deadline=time.monotonic()+1200;tick=0
  while child.poll() is None:
   if future is not None and future.done():
    future.result();meta['server_probe_pass']=True;r.stop(child);break
   xs=pathlib.Path('/home/arian/.xsession-errors')
   if xs.exists() and xs.stat().st_size>50*1024**2:
    raise RuntimeError('desktop log watchdog tripped')
   if os.statvfs(p).f_bavail*os.statvfs(p).f_frsize<5*1024**3:raise RuntimeError('disk watchdog')
   if time.monotonic()>deadline:raise RuntimeError('deadline watchdog')
   if tick%10==0:telemetry.write(str(time.time())+'\n'+gpu());telemetry.flush()
   tick+=1;time.sleep(1)
  meta['exit_code']=child.returncode;print('exit',child.returncode,flush=True)
  if child.returncode and not meta.get('server_probe_pass'):raise RuntimeError('worker failed; no automatic retry')
finally:
 if child is not None:r.stop(child)
 meta['finished']=time.time();meta['final_gpu']=gpu();(p/(args.tag+'.meta.json')).write_text(json.dumps(meta,indent=2))
 for c in coords:
  with c.open('a') as f:f.write('\nW8A16 optimization RELEASE '+args.tag+' '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+': worker ended, GPUs released, no reset. Results in '+str(p)+'\n')
 lock.close()
