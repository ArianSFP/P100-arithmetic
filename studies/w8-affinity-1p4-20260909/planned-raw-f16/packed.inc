__device__ __forceinline__ bool w8_pair_head(const aw_tile_desc*t,int pos,int count){
 if(pos<0||pos+1>=count)return false;
 const aw_tile_desc a=t[pos],b=t[pos+1];
 return a.rows==64&&b.rows==64&&a.work==b.work&&b.row==a.row+64&&(a.row&127)==0;
}

__global__ void w8_make_pairs(const aw_tile_desc*t,int count,aw_tile_desc*out){
 for(int g=blockIdx.x*blockDim.x+threadIdx.x;g<(count+1)/2;g+=gridDim.x*blockDim.x){
  int pos=2*g;if(!w8_pair_head(t,pos,count))++pos;
  aw_tile_desc v={0,0,0};if(w8_pair_head(t,pos,count)){v=t[pos];v.rows=128;}out[g]=v;
 }
}
#include "pair.inc"
#include "fallback.inc"
