from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
for r in range(3):
 for projection in ['gu','down']:
  subprocess.run(['python3',str(p/'run.py'),f'{projection}-r{r}','--tokens','256','--projection',projection],check=True)
