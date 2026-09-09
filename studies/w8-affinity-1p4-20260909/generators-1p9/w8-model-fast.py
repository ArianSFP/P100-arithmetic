from pathlib import Path
import shutil,re,json
r=Path('/home/arian/P100-arithmetic/studies/w8-affinity-1p4-20260909');p=r/'model';snap=p/'pre-1p9';snap.mkdir(exist_ok=True)
for f in ['affinity-wave.cu','validate-warm.cpp','run-warm-job.py','provenance.json','selected-manifest.json']:
 if not (snap/f).exists():shutil.copy2(p/f,snap/f)
s=(r/'compact-plan-f32/pair.inc').read_text().replace('w8_pair_smem','w8_fast_smem').replace('w8_pair_half','w8_fast_pair').replace('"decode.inc"','"fast-decode.inc"');(p/'fast-pair.inc').write_text(s)
shutil.copy2(r/'compact-plan-f32/decode.inc',p/'fast-decode.inc')
s=(r/'compact-plan-f32/packed.inc').read_text();a=s.index('__global__ void w8_make_pairs');b=s.index('#include "pair.inc"',a);s=s[a:b].replace('w8_make_pairs','w8_make_fast_pairs').replace('int*counts){','int*counts,const int*count_dev=nullptr){\n count=count_dev?*count_dev:count;');(p/'fast-packed.inc').write_text(s+'#include "fast-pair.inc"\n')
s=(p/'affinity-wave.cu').read_text().replace('#include <mutex>','#include <mutex>\n#include <memory>')
s=s.replace('#include "virtual-packed.inc"','#include "virtual-packed.inc"\n#include "fast-packed.inc"\ntemplate<bool BF16_INPUT>static void aw_dispatch_w8_planned(int,cudaStream_t,const aw_work_desc*,const aw_tile_desc*,int,int,int,const int*);')
s=s.replace('static std::atomic<int> w8_validation_mode{0};','static std::atomic<int> w8_validation_mode{[](){const char*v=getenv("GGML_CUDA_AW_W8_MODE");int mode=v?atoi(v):0;GGML_ASSERT(mode>=0&&mode<=6);return mode;}()};').replace('w8_validation_calls[6]','w8_validation_calls[7]').replace('mode<=5','mode<=6')
s=s.replace(' else {w8_fallback_half<BF16_INPUT,0>',' else if(mode==6)aw_dispatch_w8_planned<BF16_INPUT>(blocks,stream,d,t,count,n,k,count_dev);\n else {w8_fallback_half<BF16_INPUT,0>')
# Supply the allocated tile capacity where the logical count comes from device memory.
pattern=r'(aw_dispatch_w8<(?:true|false)>\([^;]*?state.tiles_m64.get\(\),\s*)0(, n, k, tile_count \+ 0\);)'
s,n=re.subn(pattern,r'\1int(state.tiles_m64.size()/sizeof(aw_tile_desc))\2',s);assert n==4,n
start=s.index('class aw_device_buffer {');a=s.index('{',start);depth=1;b=a+1
while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
assert s[b]==';';b+=1
impl='''
struct aw_w8_workspace {
 int device;cudaStream_t stream;aw_device_buffer pairs,left,counts;
 aw_w8_workspace(int d,cudaStream_t s):device(d),stream(s){}
};
static std::atomic<unsigned long long> w8_fast_calls{0};
template<bool BF16_INPUT>
static void aw_dispatch_w8_planned(int blocks,cudaStream_t stream,const aw_work_desc*d,const aw_tile_desc*t,int capacity,int n,int k,const int*count_dev){
 if constexpr(BF16_INPUT){
  w8_fallback_half<true,0><<<blocks*3/2,256,0,stream>>>(d,t,capacity,n,k,count_dev);
  w8_pair_half<true,0><<<blocks,256,0,stream>>>(d,t,capacity,n,k,count_dev);
 }else{
  GGML_ASSERT(capacity>=0);if(capacity==0)return;
  int device;CUDA_CHECK(cudaGetDevice(&device));
  static thread_local std::vector<std::unique_ptr<aw_w8_workspace>>workspaces;
  aw_w8_workspace*w=nullptr;
  for(auto&v:workspaces)if(v->device==device&&v->stream==stream){w=v.get();break;}
  if(!w){workspaces.emplace_back(new aw_w8_workspace(device,stream));w=workspaces.back().get();}
  w->pairs.ensure(size_t((capacity+1)/2)*sizeof(aw_tile_desc));
  w->left.ensure(size_t(capacity)*sizeof(aw_tile_desc));w->counts.ensure(2*sizeof(int));
  auto*pairs=(aw_tile_desc*)w->pairs.get();auto*left=(aw_tile_desc*)w->left.get();auto*counts=(int*)w->counts.get();
  w8_make_fast_pairs<<<1,256,0,stream>>>(t,capacity,pairs,left,counts,count_dev);
  w8_packed_half<false,0><<<blocks*3/2,256,0,stream>>>(d,left,capacity,n,k,counts+1);
  if(n==512&&k==2048)w8_fast_pair<false,0,512,2048><<<blocks,256,0,stream>>>(d,pairs,(capacity+1)/2,n,k,counts);
  else if(n==2048&&k==512)w8_fast_pair<false,0,2048,512><<<blocks,256,0,stream>>>(d,pairs,(capacity+1)/2,n,k,counts);
  else w8_fast_pair<false,0><<<blocks,256,0,stream>>>(d,pairs,(capacity+1)/2,n,k,counts);
  CUDA_CHECK(cudaGetLastError());w8_fast_calls.fetch_add(1,std::memory_order_relaxed);
 }
}
'''
s=s[:b]+impl+s[b:];s+='\nextern "C" __attribute__((visibility("default"))) unsigned long long aw_w8_fast_calls(){return w8_fast_calls.load();}\n';(p/'affinity-wave.cu').write_text(s)
s=(p/'validate-warm.cpp').read_text().replace('0,0,0,1,0,2,0,3,0,4,0,5,0','0,0,0,1,0,2,0,3,0,4,0,5,0,6,0').replace('iteration<13','iteration<15')
s=s.replace('const auto before=calls(mode);','const auto before=calls(mode);\n            auto fast_calls=(unsigned long long(*)())dlsym(library,"aw_w8_fast_calls");\n            if(!fast_calls)throw std::runtime_error("missing optimized dispatch counter");\n            const auto fast_before=fast_calls();')
s=s.replace('if(!launched)throw std::runtime_error("candidate dispatch not exercised");','if(!launched)throw std::runtime_error("candidate dispatch not exercised");\n            if(mode==6&&fast_calls()==fast_before)throw std::runtime_error("optimized F32 dispatch not exercised");');(p/'validate-warm.cpp').write_text(s)
s=(p/'run-warm-job.py').read_text().replace('args=parser.parse_args()','parser.add_argument("--w8-mode",type=int,choices=range(7),default=0);args=parser.parse_args()').replace("env['GGML_CUDA_DISABLE_GRAPHS']='1'","env['GGML_CUDA_AW_W8_MODE']=str(args.w8_mode)\nenv['GGML_CUDA_DISABLE_GRAPHS']='1'");(p/'run-warm-job.py').write_text(s)
v=json.loads((p/'provenance.json').read_text());v['mode6']='F32-storage compact GPU tile planner + dimension-specialized M128 dual-half kernel + original dual-half M64 leftovers. BF16/FP16 input template retains mode5 generic path. Per-device/per-stream/per-host-thread workspace grows to caller tile capacity; first allocations are included in first call. Built only, not model-tested.';v['dynamic_capacity_sites']=n;v['prefill_mode']='GGML_CUDA_AW_W8_MODE=0..6, or run-warm-job.py --w8-mode 0..6; paired quality validator selects modes itself.';(p/'provenance.json').write_text(json.dumps(v,indent=2)+'\n')
