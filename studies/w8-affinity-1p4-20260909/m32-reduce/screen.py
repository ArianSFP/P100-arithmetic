from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
for r in range(3):
 for proj in ['gu','down']:
  subprocess.run(['python3',str(p/'run.py'),f'm32-{proj}-r{r}','--tokens','32','--projection',proj],check=True)
