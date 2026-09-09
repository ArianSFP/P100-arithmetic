from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;old=p.parent/'r1';root=Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906/ggml/src/ggml-cuda/affinity-wave.cu');s=root.read_text()
a=s.index('template<bool BF16_INPUT, bool T64_LAYOUT,\n        bool INTERLEAVE');b=s.index('struct aw_smem_m32_raillocal',a);k=s[a:b]
sm='union aw_smem {struct {float4 A[2][32][8];float4 B[2][32][16];}stage;float red[8][256];};\nconstexpr int AW_Q8_BLOCK_BYTES=34;\n'
(p/'m32.inc').write_text(sm+k)
k=k.replace('aw_q8_service_m32(','w8_m32_reduce(');a=k.index('            for (int round = 0; round < 8;');b=k.index('\n        }\n    }\n}',a)
k=k[:a]+'''            float *red=(float*)&sm;
            if(kgrp!=0){
                #pragma unroll
                for(int i=0;i<8;++i){
                    #pragma unroll
                    for(int j=0;j<8;++j)red[(i*8+j)*96+(kgrp-1)*32+lane]=acc[i][j];
                }
            }
            __syncthreads();
            if(kgrp==0){
                #pragma unroll
                for(int i=0;i<8;++i){
                    const int row=mt*8+i;
                    if(row<tile.rows){
                        #pragma unroll
                        for(int j=0;j<8;++j){
                            float value=acc[i][j]+red[(i*8+j)*96+lane]+red[(i*8+j)*96+32+lane]+red[(i*8+j)*96+64+lane];
                            desc.output[size_t(tile.row+row)*n_stride+col0+nt*8+j]=value;
                        }
                    }
                }
            }
            __syncthreads();'''+k[b:]
(p/'reduce.inc').write_text(k)
for f in ['q8.inc','lowact.inc','worker.cu','build.py','run.py']:shutil.copyfile(old/f,p/f)
s=(p/'worker.cu').read_text().replace('#include "packed.inc"','#include "m32.inc"\n#include "reduce.inc"')
a=s.index(' auto run=');b=s.index('\n std::vector<float>ref;',a)
s=s[:a]+''' auto run=[&](int b,bool prep){if(b==32)aw_q8_service_m32<false,true,true><<<grid,128>>>(dd.p,dt.p,ts.size(),n,k);else if(b==16)aw_q8_service_m32<true,true,true><<<grid,128>>>(dd16.p,dt.p,ts.size(),n,k);else w8_m32_reduce<true,true,true><<<grid,128>>>(dd16.p,dt.p,ts.size(),n,k);CU(cudaGetLastError());};'''+s[b:]
s=s.replace('r+=64)ts.push_back({e,r,std::min(64,m-r)})','r+=32)ts.push_back({e,r,std::min(32,m-r)})').replace('{32,16,64,128,256}','{32,16,64}').replace('i<5;++i){int b=bs[(i+r)%5]','i<3;++i){int b=bs[(i+r)%3]')
a=s.index('float sum[2]={},total=0;');b=s.index('bad+=bits(got[ix])!=bits(expected);',a)+len('bad+=bits(got[ix])!=bits(expected);')
s=s[:a]+'''float sum[4]={};for(int g=0;g<kg;++g){float d=__half2float(sc[(size_t(e)*n+col)*kg+g]);for(int j=0;j<32;++j)sum[j/8]=std::fma(aa[(size_t(e)*m+row)*k+g*32+j],float(w[((size_t(e)*n+col)*kg+g)*32+j])*d,sum[j/8]);}bad+=bits(got[ix])!=bits(((sum[0]+sum[1])+sum[2])+sum[3]);'''+s[b:]
s=s.replace('((b==32||b==16)&&changed)','changed');(p/'worker.cu').write_text(s)
s=(p/'build.py').read_text().replace("'packed.inc'","'m32.inc','reduce.inc'");(p/'build.py').write_text(s)
s=(p/'run.py').read_text().replace("p.parents[1]/'bench/COORDINATION-20260908.md'","p.parents[2]/'bench/COORDINATION-20260908.md'").replace('w8-affinity-1p4-','w8-m32-reduce-').replace('choices=[33,64,128,256,512]','choices=[16,31,32,33,64,128,256,512]').replace('W8A16 packed-half partial accumulation candidates versus original Q8','W8A16 exact M32 reduction batching against unchanged M32')
(p/'run.py').write_text(s)
(p/'provenance.json').write_text(json.dumps({'source':str(root),'sha256':hashlib.sha256(root.read_bytes()).hexdigest(),'change':'Only final cross-warp reduction stages are batched into existing shared-memory capacity; same sum order.'},indent=2)+'\n')
