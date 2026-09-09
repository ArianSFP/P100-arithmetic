__device__ __forceinline__ half2 w8_from_bits(unsigned u){return __half2(__half2_raw{(unsigned short)u,(unsigned short)(u>>16)});}
__device__ __forceinline__ unsigned w8_to_bits(half2 h){__half2_raw r=h;return unsigned(r.x)|(unsigned(r.y)<<16);}
__device__ __forceinline__ void w8_decode4(unsigned q,float scale,half2&lo,half2&hi){
 if(scale<0.f || scale>63.96875f || __float_as_uint(scale)==0x80000000u){
  lo=__floats2half2_rn(float((signed char)q)*scale,float((signed char)(q>>8))*scale);
  hi=__floats2half2_rn(float((signed char)(q>>16))*scale,float((signed char)(q>>24))*scale);return;
 }
 unsigned signs=(q&0x80808080u)>>7;
 unsigned mag=(q^(signs*255u))+signs;
 half2 d=__float2half2_rn(scale),bias=__float2half2_rn(-scale*1024.f);
 unsigned a=w8_to_bits(__hfma2(w8_from_bits(__byte_perm(mag,0x64646464u,0x5140)),d,bias));
 unsigned b=w8_to_bits(__hfma2(w8_from_bits(__byte_perm(mag,0x64646464u,0x5342)),d,bias));
 a^=((q&0x80u)<<8)|((q&0x8000u)<<16);
 b^=((q&0x800000u)>>8)|(q&0x80000000u);
 lo=w8_from_bits(a);hi=w8_from_bits(b);
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
