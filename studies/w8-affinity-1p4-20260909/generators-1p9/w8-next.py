from pathlib import Path
import shutil
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909')
for name,interleave in [('dual-balanced',False),('dual-interleave',True)]:
 p=r/name;p.mkdir(exist_ok=True)
 for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'m128-pairs-dual'/f,p/f)
 s=(r/'m128-pairs-dual/packed.inc').read_text()
 def block(text,needle):
  start=text.index(needle);a=text.index('{',start);depth=1;b=a+1
  while depth:depth+=(text[b]=='{')-(text[b]=='}');b+=1
  return start,b
 a,b=block(s,'if (!last && producer_half == 0)');s=s[:a]+s[b:]
 a,b=block(s,'if (!last && producer_half == 1)');v=s[a:b].replace('!last && producer_half == 1','!last').replace('b_local*AW_KSTAGE + 16','b_local*AW_KSTAGE + producer_half*16').replace('[8+group*2','[producer_half*8+group*2');s=s[:a]+v+s[b:]
 if interleave:
  a,b=block(s,'for (int group = 0; group < 4; ++group)');first=s[a:b]
  c,d=block(s,'for (int group = 4; group < 8; ++group)');second=s[c:d]
  # Append corresponding upper-half step immediately after each lower-half step.
  x,y=block(first,'for (int kk = 0; kk < 4; ++kk)');step=first[first.index('{',x)+1:y-1]
  upper=step.replace('const int ks = group*4 + kk;','const int ks = group*4 + kk + 16;').replace('acc0','acc1')
  first=first[:y-1]+'{'+upper+'}\n'+first[y-1:]
  s=s[:c]+s[d:];s=s[:a]+first+s[b:]
 (p/'packed.inc').write_text(s)
 (p/'run.py').write_text((p/'run.py').read_text().replace('w8-affinity-m128-pairs-dual','w8-affinity-'+name))
s=(r/'STATUS.md').read_text();s='Current user target: 1.9x including dispatch (raised after the 1.7x result). Accuracy gate unchanged. New dual-balanced and dual-interleave experiments preserve the selected dual-rail arithmetic.\n\n'+s;(r/'STATUS.md').write_text(s)
