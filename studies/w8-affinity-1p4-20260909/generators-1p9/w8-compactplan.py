from pathlib import Path
import shutil,re
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');p=r/'compact-plan-raw';p.mkdir(exist_ok=True)
for f in ['worker.cu','q8.inc','lowact.inc','old-packed.inc','decode.inc','build.py','run.py','analyze.py']:shutil.copy2(r/'planned-raw-f16'/f,p/f)
s=(r/'dual-raw-f16/packed.inc').read_text().replace('w8_half_smem','w8_pair_smem').replace('w8_packed_half','w8_pair_half');(p/'pair.inc').write_text(s)
s=(r/'r1/packed.inc').read_text().replace('w8_half_smem','w8_fallback_smem').replace('w8_packed_half','w8_fallback_half');(p/'fallback.inc').write_text(s)
head=(r/'virtual-pairs-dual/packed.inc').read_text().split('#include')[0]
(p/'packed.inc').write_text(head+'''__global__ void w8_make_pairs(const aw_tile_desc*t,int count,aw_tile_desc*pairs,aw_tile_desc*left,int*counts){
 __shared__ int np,nl;if(threadIdx.x==0){np=0;nl=0;}__syncthreads();
 const unsigned lane=threadIdx.x&31,lower=(1u<<lane)-1u;
 for(int base=0;base<count;base+=blockDim.x){
  int i=base+threadIdx.x;bool valid=i<count;
  bool pair=valid&&w8_pair_head(t,i,count);
  bool rem=valid&&!pair&&!w8_pair_head(t,i-1,count);
  unsigned mask=__ballot_sync(0xffffffff,pair);int pos=0;
  if(lane==0)pos=atomicAdd(&np,__popc(mask));pos=__shfl_sync(0xffffffff,pos,0);
  if(pair){aw_tile_desc v=t[i];v.rows=128;pairs[pos+__popc(mask&lower)]=v;}
  mask=__ballot_sync(0xffffffff,rem);pos=0;
  if(lane==0)pos=atomicAdd(&nl,__popc(mask));pos=__shfl_sync(0xffffffff,pos,0);
  if(rem)left[pos+__popc(mask&lower)]=t[i];
 }
 __syncthreads();if(threadIdx.x==0){counts[0]=np;counts[1]=nl;}
}
#include "pair.inc"
#include "fallback.inc"
''')
s=(p/'worker.cu').read_text().replace('Device<aw_tile_desc>dtplan((ts.size()+1)/2);','Device<aw_tile_desc>dtplan((ts.size()+1)/2),dtleft(ts.size());Device<int>dcounts(2);')
s=s.replace('w8_make_pairs<<<(ts.size()+511)/512,256>>>(dt.p,ts.size(),dtplan.p);','w8_make_pairs<<<1,256>>>(dt.p,ts.size(),dtplan.p,dtleft.p,dcounts.p);')
s=re.sub(r'(w8_pair_half<true,\d+><<<candidate_grid,256>>>\(dd16.p,dtplan.p,\(ts.size\(\)\+1\)/2,n,k)\);',r'\1,dcounts.p);',s)
s=re.sub(r'(w8_fallback_half<true,\d+><<<prop.multiProcessorCount\*3,256>>>\(dd16.p,)dt.p,ts.size\(\),n,k\);',r'\1dtleft.p,ts.size(),n,k,dcounts.p+1);',s)
(p/'worker.cu').write_text(s)
