from pathlib import Path
import shutil,re
p=Path(__file__).resolve().parent;src=p.parent/'w8-lowact-20260909'
for f in ['q8.inc','lowact.inc','worker.cu','build.py','run.py','screen.py']:shutil.copyfile(src/f,p/f)
s=(p/'q8.inc').read_text();k=s[s.index('template<bool BF16_INPUT>\n__global__'):]
k=k.replace('template<bool BF16_INPUT>','template<bool BF16_INPUT,int FLUSH>').replace('aw_q8_service_m64_n128_halfpipe_sync','w8_packed_half').replace('aw_smem_m64_n128_256','w8_half_smem')
k='struct w8_half_smem {half A[2][32][64];half2 B[32][128];};\n'+k
k=k.replace('float acc0[8][4] = {};\n        float acc1[8][4] = {};','half2 acc0[4][4],acc1[4][4];float total[8][4]={};\n        for(int i=0;i<4;++i)for(int j=0;j<4;++j)acc0[i][j]=acc1[i][j]=__float2half2_rn(0.f);')
k=re.sub(r'(sm.A\[[^;]+?= )([^;]+);',r'\1__float2half_rn(\2);',k)
k=re.sub(r'(sm.B\[[^;]+?=)\s*((?:\(float\))[\s\S]*?\*scale);',r'\1__float2half2_rn(\2);',k)
for rail in [0,1]:
 a=k.index('                    const float4 av0 =');b=k.index(f'                            acc{rail}[i][j] += av[i]*bv[j];',a)
 end=k.index('\n                }',b)+len('\n                }')
 k=k[:a]+f'''                    const half2 *av=(const half2*)&sm.A[buffer][ks][row0];
                    const half2 bv[4]={{sm.B[ks][lane],sm.B[ks][lane+32],sm.B[ks][lane+64],sm.B[ks][lane+96]}};
                    #pragma unroll
                    for(int i=0;i<4;++i){{
                        #pragma unroll
                        for(int j=0;j<4;++j)acc{rail}[i][j]=__hfma2(av[i],bv[j],acc{rail}[i][j]);
                    }}
                }}'''+k[end:]
needle='            if (!last) {\n                __syncthreads();'
flush='''            if constexpr(FLUSH>0) {
                if((stage+1)%FLUSH==0 || last){
                    #pragma unroll
                    for(int i=0;i<4;++i){
                        #pragma unroll
                        for(int j=0;j<4;++j){
                            float2 lo=__half22float2(acc0[i][j]),hi=__half22float2(acc1[i][j]);
                            total[2*i][j]+=lo.x+hi.x;total[2*i+1][j]+=lo.y+hi.y;
                            acc0[i][j]=acc1[i][j]=__float2half2_rn(0.f);
                        }
                    }
                }
            }
'''
k=k.replace(needle,flush+needle)
k=k.replace('output[lane + j*32] = acc0[i][j] + acc1[i][j];','if constexpr(FLUSH>0)output[lane+j*32]=total[i][j];else {float2 lo=__half22float2(acc0[i/2][j]),hi=__half22float2(acc1[i/2][j]);output[lane+j*32]=(i%2?lo.y:lo.x)+(i%2?hi.y:hi.x);}')
(p/'packed.inc').write_text(k)
s=(p/'worker.cu').read_text().replace('#include "lowact.inc"','#include "lowact.inc"\n#include "packed.inc"')
s=s.replace('if(b==32)aw_q8','if(b==64)w8_packed_half<true,1><<<grid,256>>>(dd16.p,dt.p,ts.size(),n,k);else if(b==128)w8_packed_half<true,4><<<grid,256>>>(dd16.p,dt.p,ts.size(),n,k);else if(b==256)w8_packed_half<true,0><<<grid,256>>>(dd16.p,dt.p,ts.size(),n,k);else if(b==32)aw_q8')
s=s.replace('{32,16,8,4}','{32,16,64,128,256}').replace('i<4;++i){int b=bs[(i+r)%4]','i<5;++i){int b=bs[(i+r)%5]')
a=s.index('float sum[2]={};');b=s.index('bad+=bits(got[ix])!=bits(sum[0]+sum[1]);',a)+len('bad+=bits(got[ix])!=bits(sum[0]+sum[1]);')
s=s[:a]+'''float sum[2]={},total=0;int flush=b==64?1:b==128?4:0;for(int g=0;g<kg;++g){float d=__half2float(sc[(size_t(e)*n+col)*kg+g]);for(int j=0;j<32;++j){float weight=float(w[((size_t(e)*n+col)*kg+g)*32+j])*d;float activation=aa[(size_t(e)*m+row)*k+g*32+j];if(b>=64){weight=__half2float(__float2half_rn(weight));sum[j/16]=__half2float(__float2half_rn(std::fma(activation,weight,sum[j/16])));}else sum[j/16]=std::fma(activation,weight,sum[j/16]);}if(flush&&((g+1)%flush==0||g+1==kg)){total+=sum[0]+sum[1];sum[0]=sum[1]=0;}}float expected=flush?total:sum[0]+sum[1];bad+=bits(got[ix])!=bits(expected);'''+s[b:]
s=s.replace('(b>=16&&changed)','((b==32||b==16)&&changed)')
(p/'worker.cu').write_text(s)
s=(p/'build.py').read_text().replace("'lowact.inc',","'lowact.inc','packed.inc',")
(p/'build.py').write_text(s)
s=(p/'run.py').read_text().replace('w8-lowact-','w8-affinity-1p4-').replace('W8A8/W8A4 direct packed activation loads plus quantization against original Q8','W8A16 packed-half partial accumulation candidates versus original Q8')
(p/'run.py').write_text(s)
