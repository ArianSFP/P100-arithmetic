#!/usr/bin/env python3
import json,re
from pathlib import Path
R=Path(__file__).resolve().parent
paired=json.loads((R/'confirmation-summary.json').read_text())['summary']
iso=json.loads((R/'isolated-summary.json').read_text())['summary']
audit=json.loads((R/'audit.json').read_text())
jobs=json.loads((R/'confirmation-jobs.json').read_text())
lines=['# FP16 accumulation: approaching 2x at 2048+ tokens','',
'**The tested optimizations reach about1.89–1.91x on four of the six shard shapes against a tuned FP32 pipeline. A uniform2x was not achieved.** The2048-token up projection remains variable; the8192-token down projection also falls short. All costs below include Q8 decoding, activation preparation, GEMM and a final FP32 output.',
'', 'Target shapes are the original report\'s four-way dense FFN shards: up M4352/K5120 and down M5120/K4352. N is the number of tokens in this single GEMM, not total prompt length. The user narrowed this follow-up to N2048/4096/8192. These results do not cover the separate two-GPU M8704 shard study.',
'', '## Frozen paired confirmation', '',
'Three fresh workers per shape, new seeds and a broader finite weight distribution than screening. Each worker has nine rotated ABBA rounds per comparison; both same-mode legs are averaged within each round before taking worker medians. Table entries are medians across the three worker medians. Both arithmetic modes receive native aligned Q8 decoding and independently selected exact cuBLAS/layout optimizations. No cached weights or excluded transposition costs.', '',
'| Projection | Tokens | Tuned FP32 pipeline ms | Selected FP16 pipeline ms | FP16 speedup |',
'| --- | ---: | ---: | ---: | ---: |']
plans=[]
for r in sorted(paired,key=lambda r:(r['N'],r['M'])):
 cs={(c['mode'],c['arm']):c for c in r['configs']};a=cs[0,2]['pipeline_us'];b=cs[1,2]['pipeline_us']
 lines.append(f"| {'Up' if r['M']==4352 else 'Down'} | {r['N']} | {a/1000:.3f} | {b/1000:.3f} | {a/b:.3f}x |")
 job=next(j for j in jobs if list(map(int,j['args'][:3]))==[r['M'],r['N'],r['K']]);args=list(map(int,job['args']))
 plans.append(dict(M=r['M'],N=r['N'],K=r['K'],fp32=dict(algo=args[4],layout=args[5],pad=args[10]),fp16=dict(algo=args[6],layout=args[7],pad=args[11]),paired_ratio=a/b,status='experimental synthetic; per-shape full-output identity passed'))
lines += ['', '**The8192 down FP16 plan is the native TN/default-algorithm path itself.** Its apparent latency differences between identical paired arms are an order effect, not an optimization. We therefore also measured each selected arithmetic mode in its own fresh worker, after30 complete pipeline warmups. These secondary checks contain31 event samples of two full pipelines each.', '', '## Isolated-mode sensitivity check', '',
'| Projection | Tokens | FP32 workers | FP16 workers | FP32 ms | FP16 ms | Ratio |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in sorted(iso,key=lambda r:(r['N'],r['M'])):
 a=r['modes'].get('0');b=r['modes'].get('1')
 if a and b:lines.append(f"| {'Up' if r['M']==4352 else 'Down'} | {r['N']} | {a['workers']} | {b['workers']} | {a['median_us']/1000:.3f} | {b['median_us']/1000:.3f} | {r['ratio']:.3f}x |")
lines += ['', 'At2048 up, the two isolated FP16 worker medians are5.784 and6.297ms against10.793ms FP32: approximately1.87x and1.71x. Their average does not establish a stable1.79x path. The paired confirmation gives1.714x. The best individual timing is not promoted as the result. The exact cause of this timing variation remains unresolved; storage and output-shape padding did not eliminate it.',
'', '## Why the original table showed about1.6x', '',
'The original table timed entire pipelines, not only arithmetic. Both cuBLAS modes already receive the same FP16 weights and FP16 activations. Switching accumulation precision therefore does not halve their input traffic, and it does not accelerate Q8 decoding or activation preparation. The FP16 path additionally widens its half output to FP32.', '',
'Fresh N256 screen medians (six workers/shape across the first two sweeps):', '',
'| Projection | FP32 GEMM us | FP16 GEMM us | GEMM ratio | Full-pipeline ratio |',
'| --- | ---: | ---: | ---: | ---: |',
'| Up | 1668.55 | 844.80 | 1.975x | 1.688x |',
'| Down | 1585.24 | 885.09 | 1.791x | 1.599x |', '',
'The original scalar weight decoder takes about231us, activation conversion16–18us, and output widening16–18us at N256. These separately timed stages diagnose the cost; their medians are not an exactly additive decomposition because cache state and surrounding launches differ.', '',
'An idealized accounting is `T32 = D + A + G32`, `T16 = D + A + G16 + W`. Even if `G16 = G32/2`, common preparation D/A and extra widening W keep the full-pipeline ratio below2. A different cuBLAS algorithm can also have different useful arithmetic efficiency.', '',
'[NVIDIA\'s Pascal guide](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html) describes the packed-half arithmetic advantage. For the measured56 SMs at1328MHz, the ordinary dense-FMA peaks are9.519TF/s FP32 and19.038TF/s FP16. These are calculated ceilings, not hardware-counter measurements. The best large GEMMs consume most of this arithmetic budget, leaving limited room to hide preparation/output traffic.', '',
'cuBLAS12.8 documents COMPUTE_16F with F16 A/B/C; it does not offer F32 output for that compute-type combination. Removing the widening pass while keeping this output contract would require a different kernel or a consumer fusion. [Supported type table](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasgemmex)', '',
'## Implemented and tested changes', '',
'1. Native aligned Q8 decoding as the final common control. The earlier four-value half2 decoder reduced the study\'s scalar decoder from about232 to162us, but it is not claimed as an improvement over the backend\'s native decoder.',
'2. Per-shape cuBLAS algorithm selection for both FP32 and FP16. IDs0–23 were screened at large shapes, retaining supported configurations only. Numerical changes are recorded and excluded from the exact candidates.',
'3. Eight operand-storage/orientation combinations, including computing the transposed output and transposing/widening it back. Every required conversion and transpose is charged.',
'4. Fused Q8 decoding plus weight transpose, and fused F32-to-F16 activation conversion plus activation transpose. These preserve every independently prepared input word; all selected final output words are checked too.',
'5. Activation/weight storage padding8/32/128/256/512, followed by zero-filled output-row padding128/256/512 on the weak2048 up shape. Row padding lost: half pipelines around6.75–6.83ms versus the unpadded probe\'s6.12ms. The storage-pad8 lead did not stabilize fresh-worker timing.', '',
'FP16 selections (`layout` bits:1 transposes W;2 transposes X;4 swaps operands and transposes output):', '',
'| M | N | K | Algorithm | Layout | Leading-dimension padding |', '| ---: | ---: | ---: | ---: | ---: | ---: |']
for p in plans:
 f=p['fp16'];lines.append(f"| {p['M']} | {p['N']} | {p['K']} | {f['algo']} | {f['layout']} | {f['pad']} |")
lines += ['', 'Algorithm99 is DEFAULT_TENSOR_OP, implemented with ordinary half arithmetic on this P100. Layout0 is original TN. Layout2 is TT with transposed activation. Layout3 materializes both input transposes. Layout6 swaps operands with transposed activation. These plans are experimental, tied to these exact shapes and CUDA12.8/cuBLAS12.8.3; they are not a general production dispatch table.',
'', '## Accuracy, controls and limits', '',
'All selected paths match their own arithmetic-mode default output byte-for-byte, before and after timing. FP16 equality is to the existing FP16 path; it is not equality to FP32 accumulation. Input transformations are checked over all prepared weights/activations; FP32 default also receives32 independently accumulated complete CPU dot checks per worker. Fresh confirmation includes signed Q8 values−128…127 and wider finite scales0.003…0.015.', '',
'No real-model KLD-pair/perplexity gate or full-model prefill benchmark was run. The required model tolerance±0.003 is therefore not claimed. Changing batch size can change cuBLAS accumulation order; per-shape identity is not an across-batch identity claim.', '',
'Cached-weight experiments at N256 can show about2.14x/2.00x against the original uncached FP32 pipeline. Giving FP32 the same cache reduces the ratios to about1.85x/1.75x. That experiment changes preparation costs and is not used to claim a2x accumulation gain. Retaining one decoded shard matrix costs another42.5MiB/device, so whole-model caching also needs a memory budget.', '',
'The separate large-GEMM study\'s handwritten half2 and Strassen regressions were reviewed and not repeated. No SASS mutation, precision relaxation, undocumented opcode, clock change, reset, production edit, commit or push was used here.', '',
'## Reproduction and evidence', '',
'[README](README.md) describes the layouts, build commands, controller and timing contracts. [Frozen requests](confirmation-jobs.json) pin algorithms/layouts. [Selected plans](plans.json) include the independently tuned FP32 comparator. [Paired confirmation](confirmation-summary.json), [isolated checks](isolated-summary.json), [large screen](large-screen-summary.json), [storage-padding screen](padding-summary.json) retain worker medians and samples. The `large-*` directories contain raw logs, commands, binary/source hashes, health and clock/power telemetry.', '',
'The arithmetic scripts require one fresh, coordinated GPU1 worker at a time under the shared rig lock. CPU affinity is0–11, with an explicit empty CUDA environment. Desktop-log/disk/deadline guards remain active; other GPUs and their clients are untouched.', '',
f"Artifact audit currently covers **{sum(len(r['workers']) for r in audit)} successful workers and {sum(w['memcheck_clean'] for r in audit for w in r['workers'])} clean memcheck workers**. Every executed source/binary hash has a retained matching file. [Audit](audit.json).", '']
(R/'plans.json').write_text(json.dumps(dict(cuda=12080,cublas=120803,architecture='sm_60',plans=plans),indent=2)+'\n')
body='\n'.join(lines)
body=re.sub(r'\b(about|uniform|The|the|after|before|takes|contains|contain|are|and|approximately|stable|receives|At|Algorithm|algorithm|give|gives|requires|at|to|than|for|from|against|covers)(?=\d)',r'\1 ',body)
body=re.sub(r'(?<=\d)(?=(?:ms|us|MiB|TF/s|MHz)\b)',' ',body)
body=body.replace('shapes0','shapes 0').replace('IDs0','IDs 0').replace('padding8','padding 8').replace('padding128','padding 128').replace('padding512','padding 512').replace('values−','values −').replace('scales0','scales 0').replace('tolerance±','tolerance ±')
(R/'RESULTS.md').write_text(body)
