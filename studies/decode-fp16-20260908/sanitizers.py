#!/usr/bin/env python3
"""Bounded extra sanitizer workers under our already-held controller reservation."""
import hashlib,json,os,shutil,signal,subprocess,time,sys
from pathlib import Path
r=Path(__file__).resolve().parent
run=r/sys.argv[1];hold=json.loads((run/'hold.json').read_text());uuid='GPU-ecc6a1f9-42fe-2932-7ea0-1dd285491b7b'
locks=subprocess.check_output(['lslocks','--json','-o','PID,PATH'],text=True)
assert any(x['pid']==hold['pid'] and x.get('path')=='/tmp/affinitywave-4gpu.lock' for x in json.loads(locks)['locks'])
def health():
 raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True,timeout=5)
 c=raw.strip().split(',');assert c[0]==uuid and int(c[1])<64 and int(c[3])==0
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=5);assert uuid not in apps
 return raw
names='r3r1,r3b4,r3b8,m4r1b1w8s8,m5r1b1w8s8,m4r2b4w8s16,m5r2b4w8s16,g4r1b4w4s32,g5r1b4w4s32,p6l2b8s8,p7l2b8s8,p8l2b8s8,p6l4b8s16,p7l4b8s16,p8l4b8s16'
for tool in ['synccheck','racecheck']:
 d=run/('v4-'+tool);d.mkdir()
 cmd=['/usr/bin/env','-i','PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin','LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64','CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES='+uuid,'/usr/bin/taskset','--cpu-list','0-11','/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','9',str(r/'worker'),'33','96','5','600908','0',names,'1']
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 meta=dict(command=cmd,before=health(),token=hold['token'],binary_sha256=h(r/'worker'),source_sha256=h(r/'worker.cu'),other_sources={n:h(r/n) for n in ['kernels.cuh','r3-baseline.cuh','common.hpp']})
 (d/'prelaunch.json').write_text(json.dumps(meta,indent=2)+'\n')
 with (d/'stdout.jsonl').open('w') as out,(d/'stderr.txt').open('w') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
  deadline=time.monotonic()+240
  try:
   while p.poll() is None:
    assert time.monotonic()<deadline,'Timeout: no retry/reset'
    assert shutil.disk_usage(r).free>3_000_000_000
    assert Path('/home/arian/.xsession-errors').stat().st_size<100_000_000
    try:p.wait(timeout=.5)
    except subprocess.TimeoutExpired:pass
   assert p.returncode==0,'Worker failure'
   meta.update(status='PASS',returncode=0,after=health())
  except Exception as exc:
   if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
   meta.update(status='STOP',error=str(exc))
  finally:(d/'result.json').write_text(json.dumps(meta,indent=2)+'\n')
  print(tool,meta['status'],flush=True)
  if meta['status']!='PASS':raise SystemExit(1)
