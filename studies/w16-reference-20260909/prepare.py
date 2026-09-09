from pathlib import Path
import shutil
p=Path(__file__).resolve().parent
src=p.parent/'w8-lowact-20260909'
for f in ['q8.inc','lowact.inc','worker.cu','build.py','run.py','screen.py']:
    shutil.copyfile(src/f,p/f)
s=(p/'q8.inc').read_text();k=s[s.index('template<bool BF16_INPUT>\n__global__'):].replace('aw_q8_service_m64_n128_halfpipe_sync','w16_reference').replace('AW_T64_STAGE_BYTES','4096')
for stage,offset in [('weight_stage','producer_half*16'),('weight_stage','16'),('next_weight_stage','0')]:
    a=k.index('const float scale =')
    z=k.index('for (int group = 0; group < 4; ++group)',a)
    op=k.index('{',z);depth=1;end=op+1
    while depth:
        depth+=(k[end]=='{')-(k[end]=='}');end+=1
    replacement=f'''const half * hp = (const half *){stage} + b_local*32 + {offset};
            const uint4 h0=__ldg((const uint4*)hp),h1=__ldg((const uint4*)(hp+8));
            const unsigned hw[8]={{h0.x,h0.y,h0.z,h0.w,h1.x,h1.y,h1.z,h1.w}};
            #pragma unroll
            for(int j=0;j<16;++j) sm.B[{offset}+j][b_r]=__half2float(__ushort_as_half((hw[j/2]>>((j%2)*16))&65535));'''
    k=k[:a]+replacement+k[end:]
(p/'w16.inc').write_text(k)
s=(p/'worker.cu').read_text().replace('#include "lowact.inc"','#include "lowact.inc"\n#include "w16.inc"')
s=s.replace('std::vector<half>ha(ac);','std::vector<half>wh(wc);for(int e=0;e<ex;++e)for(int r=0;r<n;++r)for(int g=0;g<kg;++g)for(int j=0;j<32;++j){size_t si=(size_t(e)*n+r)*kg+g;size_t dst=((size_t(e)*(n/64)+r/64)*kg+g)*2048+(r%64)*32+j;wh[dst]=__float2half_rn(float(w[si*32+j])*__half2float(sc[si]));}Device<half>dwh(wc);dwh.put(wh);\n std::vector<half>ha(ac);')
s=s.replace('ds,ds16,ds8,ds4','ds,ds16,ds8,ds4,dsw')
s=s.replace('ds16.push_back(d);','ds16.push_back(d);auto wd=d;wd.weight=(const char*)(dwh.p+size_t(e)*n*k);dsw.push_back(wd);')
s=s.replace('dd(ds.size()),','ddw(dsw.size()),dd(ds.size()),').replace('dd.put(ds);','ddw.put(dsw);dd.put(ds);')
s=s.replace('if(b==32)aw_q8','if(b==64)w16_reference<true><<<grid,256>>>(ddw.p,dt.p,ts.size(),n,k);else if(b==32)aw_q8')
s=s.replace('{32,16,8,4}','{32,16,8,4,64}')
s=s.replace('float(w[((size_t(e)*n+col)*kg+g)*32+j])*d','(b==64?__half2float(__float2half_rn(float(w[((size_t(e)*n+col)*kg+g)*32+j])*d)):float(w[((size_t(e)*n+col)*kg+g)*32+j])*d)')
s=s.replace('(b>=16&&changed)','((b==32||b==16)&&changed)')
s=s.replace('i<4;++i){int b=bs[(i+r)%4]','i<5;++i){int b=bs[(i+r)%5]')
(p/'worker.cu').write_text(s)
s=(p/'build.py').read_text().replace("'lowact.inc',","'lowact.inc','w16.inc',")
(p/'build.py').write_text(s)
s=(p/'run.py').read_text().replace('w8-lowact-','w16-reference-').replace('W8A8/W8A4 direct packed activation loads plus quantization against original Q8','W16A16 predecoded weights and original Q8 paired reference')
(p/'run.py').write_text(s)
