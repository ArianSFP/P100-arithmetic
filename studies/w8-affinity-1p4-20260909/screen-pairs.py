from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
subprocess.run(['python3',str(p/'m128-double-b-pairs/run.py'),'race','--tokens','33','--projection','gu','--racecheck','--ctas','1'],check=True)
subprocess.run(['python3',str(p/'m128-double-b-pairs/screen.py')],check=True)
for m in [128,512]:
 for proj in ['gu','down']:
  subprocess.run(['python3',str(p/'m128-tile/run.py'),f'm{m}-{proj}-r0','--tokens',str(m),'--projection',proj],check=True)
