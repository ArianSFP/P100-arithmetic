from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
for c in [1,3,4,6]:
 for proj in ['gu','down']:
  subprocess.run(['python3',str(p/'run.py'),f'grid{c}-{proj}','--tokens','256','--projection',proj,'--ctas',str(c)],check=True)
