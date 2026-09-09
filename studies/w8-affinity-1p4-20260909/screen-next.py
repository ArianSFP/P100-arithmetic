from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
for name in ['m128-tile-loop','m256-tile','m256-tile-loop']:
 for proj in ['gu','down']:
  subprocess.run(['python3',str(p/name/'run.py'),f'{proj}-r0','--tokens','256','--projection',proj],check=True)
subprocess.run(['python3',str(p/'vendor/run.py'),'semantic','--tokens','33','--projection','gu','--memcheck'],check=True)
for proj in ['gu','down']:
 subprocess.run(['python3',str(p/'vendor/run.py'),f'{proj}-r0','--tokens','256','--projection',proj],check=True)
