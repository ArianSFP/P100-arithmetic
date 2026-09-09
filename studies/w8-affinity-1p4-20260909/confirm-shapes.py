from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
for m in [64,128,512]:
 for r in range(3):
  for proj in ['gu','down']:
   subprocess.run(['python3',str(p/'run.py'),f'm{m}-{proj}-r{r}','--tokens',str(m),'--projection',proj],check=True)
