__device__ __forceinline__ bool w8_pair_head(const aw_tile_desc*t,int pos,int count){
 if(pos<0||pos+1>=count)return false;
 const aw_tile_desc a=t[pos],b=t[pos+1];
 return a.rows==64&&b.rows==64&&a.work==b.work&&b.row==a.row+64&&(a.row&127)==0;
}

#include "pair.inc"
#include "fallback.inc"
