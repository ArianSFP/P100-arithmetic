__device__ __forceinline__ bool w8_pair_head(const aw_tile_desc*t,int pos,int count){
 if(pos<0||pos+1>=count)return false;
 const aw_tile_desc a=t[pos],b=t[pos+1];
 return a.rows==64&&b.rows==64&&a.work==b.work&&b.row==a.row+64&&(a.row&127)==0;
}

__global__ void w8_make_pairs(const aw_tile_desc*t,int count,aw_tile_desc*pairs,aw_tile_desc*left,int*counts){
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
