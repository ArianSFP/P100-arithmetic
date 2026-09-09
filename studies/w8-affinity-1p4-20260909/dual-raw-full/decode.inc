__device__ __forceinline__ half2 w8_from_bits(unsigned u){return __half2(__half2_raw{(unsigned short)u,(unsigned short)(u>>16)});}
__device__ __forceinline__ unsigned w8_to_bits(half2 h){__half2_raw r=h;return unsigned(r.x)|(unsigned(r.y)<<16);}
__device__ __forceinline__ void w8_decode4(unsigned q,float scale,half2&lo,half2&hi){
 unsigned biased=q^0x80808080u;
 half2 a=w8_from_bits(__byte_perm(biased,0x64646464u,0x5140));
 half2 b=w8_from_bits(__byte_perm(biased,0x64646464u,0x5342));
 half2 offset=__float2half2_rn(1152.f),d=__float2half2_rn(scale);
 lo=__hmul2(__hsub2(a,offset),d);hi=__hmul2(__hsub2(b,offset),d);
}
__global__ void w8_check_decode(unsigned*bad){
 constexpr unsigned count=63488u*256u;
 for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x){
  unsigned si=i>>8,c=i&255;unsigned short bits=si<0x7c00u?si:si+0x400u;
  float d=__half2float(__ushort_as_half(bits));half2 lo,hi;w8_decode4(c*0x01010101u,d,lo,hi);
  unsigned h=__half_as_ushort(__float2half_rn(float((signed char)c)*d));unsigned expected=h|(h<<16);
  if(w8_to_bits(lo)!=expected||w8_to_bits(hi)!=expected)atomicAdd(bad,1);
 }
}
