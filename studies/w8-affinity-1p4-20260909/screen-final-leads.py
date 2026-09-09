from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
subprocess.run(['python3',str(p/'m128-pairs-dual/run.py'),'semantic','--tokens','128','--projection','gu','--memcheck'],check=True)
for name in ['m128-pairs-bound3','m128-pairs-dual']:
 for proj in ['gu','down']:
  subprocess.run(['python3',str(p/name/'run.py'),f'{proj}-r0','--tokens','256','--projection',proj],check=True)
for m in [65,129]:
 subprocess.run(['python3',str(p/'virtual-pairs/run.py'),f'mixed{m}','--tokens',str(m),'--projection','gu','--memcheck'],check=True)
subprocess.run(['python3',str(p/'virtual-pairs/screen.py')],check=True)
