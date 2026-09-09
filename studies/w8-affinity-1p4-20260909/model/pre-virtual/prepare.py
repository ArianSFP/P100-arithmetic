from pathlib import Path
import shutil,re,json,hashlib
p=Path(__file__).resolve().parent;old=p.parents[1]/'q4-halfpipe-paired-model-20260909';root=Path('/home/arian/llama.cpp-q36-moe/.worktrees/coding-serve-p100-20260906')
src=root/'ggml/src/ggml-cuda/affinity-wave.cu';s=src.read_text()
pattern=r'aw_q8_service_m64_n128_halfpipe_sync<\s*(true|false)\s*>\s*<<<([^>]+)>>>\s*\(([\s\S]*?)\);'
count=0
def replace(m):
 global count
 count+=1;launch=[x.strip() for x in m[2].split(',')];assert len(launch)==4 and launch[1]=='256' and launch[2]=='0'
 return f'aw_dispatch_w8<{m[1]}>({launch[0]},{launch[3]},{m[3]});'
s=re.sub(pattern,replace,s);assert count==8,count
start=s.index('static void aw_q8_service_m64_n128_halfpipe_sync(');a=s.index('{',start);depth=1;b=a+1
while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
adapter='''
#include "packed.inc"
static std::atomic<int> w8_validation_mode{0};
static std::atomic<unsigned long long> w8_validation_calls[4];
template<bool BF16_INPUT>
static void aw_dispatch_w8(int blocks,cudaStream_t stream,const aw_work_desc *d,const aw_tile_desc *t,int count,int n,int k,const int *count_dev=nullptr){
 const int mode=w8_validation_mode.load(std::memory_order_relaxed);
 w8_validation_calls[mode].fetch_add(1,std::memory_order_relaxed);
 if(mode==0)aw_q8_service_m64_n128_halfpipe_sync<BF16_INPUT><<<blocks,256,0,stream>>>(d,t,count,n,k,count_dev);
 else if(mode==1)w8_packed_half<BF16_INPUT,1><<<blocks,256,0,stream>>>(d,t,count,n,k,count_dev);
 else if(mode==2)w8_packed_half<BF16_INPUT,4><<<blocks,256,0,stream>>>(d,t,count,n,k,count_dev);
 else w8_packed_half<BF16_INPUT,0><<<blocks,256,0,stream>>>(d,t,count,n,k,count_dev);
}
'''
s=s[:b]+adapter+s[b:]
s+='''
extern "C" __attribute__((visibility("default"))) void aw_w8_set_mode(int mode){GGML_ASSERT(mode>=0&&mode<=3);w8_validation_mode.store(mode);}
extern "C" __attribute__((visibility("default"))) unsigned long long aw_w8_calls(int mode){GGML_ASSERT(mode>=0&&mode<=3);return w8_validation_calls[mode].load();}
'''
(p/'affinity-wave.cu').write_text(s);shutil.copyfile(p.parent/'packed.inc',p/'packed.inc')
for f in ['build.py','quality.py']:shutil.copyfile(old/f,p/f)
s=(old/'validate-warm.cpp').read_text().replace('aw_q4_halfpipe_set_mode','aw_w8_set_mode').replace('const int modes[]={0,0,0,1,0,2,0};','const int modes[]={0,0,0,1,0,2,0,3,0};').replace('iteration<7','iteration<9')
s=s.replace('const int mode=modes[iteration];set_mode(mode);','''const int mode=modes[iteration];set_mode(mode);
            auto calls=(unsigned long long(*)(int))dlsym(library,"aw_w8_calls");
            if(!calls)throw std::runtime_error("missing dispatch coverage counter");
            const auto before=calls(mode);''')
s=s.replace('const double ppl=std::exp(loss/rows);','''const double ppl=std::exp(loss/rows);
            const auto launched=calls(mode)-before;
            fprintf(stdout,"COVERAGE mode=%d launches=%llu\\n",mode,launched);
            if(!launched)throw std::runtime_error("candidate dispatch not exercised");''')
(p/'validate-warm.cpp').write_text(s)
s=(old/'build-warm-validator.py').read_text().replace("'build-validator.py'","'build-warm-validator.py'");(p/'build-warm-validator.py').write_text(s)
s=(old/'run-warm-job.py').read_text().replace("model='/home/arian/models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf'","model=os.environ.get('W8_MODEL_PATH','/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf')\nif not pathlib.Path(model).is_file():raise RuntimeError('Original Q8 model missing; restore pinned model or set W8_MODEL_PATH')")
s=s.replace("p.parents[1]/'bench/COORDINATION-20260908.md'","p.parents[2]/'bench/COORDINATION-20260908.md'")
s=s.replace('Q4 AffinityWave','W8A16 optimization').replace("    with xs.open('r+') as f:f.truncate(0)\n",'')
(p/'run-warm-job.py').write_text(s)
(p/'provenance.json').write_text(json.dumps({'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'redirected_m64_launch_sites':count,'scope':'Only M64 service; M32/M16 remain original. Modes 0=original,1=G32,2=G128,3=full half accumulation.'},indent=2)+'\n')
