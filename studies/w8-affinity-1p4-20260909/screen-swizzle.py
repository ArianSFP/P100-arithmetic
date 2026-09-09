from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
subprocess.run(['python3',str(p/'m128-swizzle-a/run.py'),'race','--tokens','33','--projection','gu','--racecheck','--ctas','1'],check=True)
subprocess.run(['python3',str(p/'m128-swizzle-a/screen.py')],check=True)
subprocess.run(['python3',str(p/'virtual-pairs/run.py'),'mixed-semantic','--tokens','129','--projection','gu','--memcheck'],check=True)
