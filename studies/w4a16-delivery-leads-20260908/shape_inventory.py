#!/usr/bin/env python3
"""CPU-only workload inventory; historical routing is NOT a new Q4 trace."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HIST = Path('/home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/round3-phase0')
BOUNDS = [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

def inventory(path):
    counts = Counter()
    observations = 0
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            fields = line.split()
            if not fields or not fields[0].endswith('ffn_gate_exps.weight'):
                continue  # Down repeats the same route counts; don't double count.
            prompt = int(fields[1])
            values = list(map(int, fields[2:]))
            # TAG_MOE_HIST prints indices 0..maxe, omitting trailing zero experts.
            assert 0 < len(values) <= 256 and sum(values) == prompt * 8, (str(path), lineno, prompt, len(values), sum(values))
            values += [0] * (256-len(values))
            assert prompt == 4096 and min(values) >= 0 and max(values) <= prompt
            counts.update(values)
            observations += 1
    assert observations
    total = sum(counts.values())
    rows = sum(m * c for m, c in counts.items())
    bins = []
    lower = -1
    for upper in BOUNDS:
        entries = sum(c for m, c in counts.items() if lower < m <= upper)
        work = sum(m*c for m, c in counts.items() if lower < m <= upper)
        bins.append(dict(lower_exclusive=lower, upper_inclusive=upper,
                         expert_observations=entries, observation_percent=100*entries/total,
                         routed_rows=work, useful_gemm_work_percent=100*work/rows))
        lower = upper
    assert sum(b['expert_observations'] for b in bins) == total
    assert sum(b['routed_rows'] for b in bins) == rows
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                provenance='Historical Q8/fallback 4096-token routing; not current Q4/AW dispatch',
                layer_microbatch_observations=observations, expert_observations=total,
                bins=bins)

def main():
    report = dict(
        target='>=1.5x workload-weighted T64 GEMM pipeline at 2k+ prompts; not achieved',
        dimensions={'M': 'routed tokens for this expert/service job',
                    'N': 'output channels', 'K': 'reduction channels'},
        projections=[dict(name='gate', N=512, K=2048), dict(name='up', N=512, K=2048),
                     dict(name='down', N=2048, K=512)],
        prompt_suite=[2048, 4096, 8192],
        illustrative_unpartitioned_means={str(p): p*8/256 for p in (2048,4096,8192)},
        warnings=[
            'Means are route conservation only, not measured M distributions or dispatch shapes.',
            'Do not scale the 4096 histogram to synthesize real 2048/8192 routing.',
            'panel2048 names output N panel width, not 2048 prompt tokens.',
            'Historical counts lack token order, owner partition and current Q4 routes.',
            'Useful FLOP shares are prioritization proxies, not measured GPU-time shares.',
            'GGML_CUDA_MOE_HIST disables T64 eligibility; do not profile it as unchanged AW.',
        ],
        sources=[inventory(HIST / ('hist-'+domain+'.txt')) for domain in ('code','wiki')])
    (ROOT/'shape-inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    for source in report['sources']:
        print(Path(source['source']).name, source['layer_microbatch_observations'])
        for b in source['bins']:
            print(f"  M({b['lower_exclusive']},{b['upper_inclusive']}] "
                  f"observations={b['observation_percent']:.2f}% work={b['useful_gemm_work_percent']:.2f}%")

if __name__ == '__main__':
    main()
