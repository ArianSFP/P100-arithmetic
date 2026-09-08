"""Read-only AWTR header inventory; does NOT certify every route's validity."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path('/home/arian/llama.cpp-q36-moe/.worktrees/elasticwave/results/qwen36-35b-moe-pp-20260721/elasticwave-20260723')


def inspect(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            digest.update(chunk)
    counts=Counter()
    records=0
    with path.open('rb') as f:
        assert f.read(8)==b'AWTRV001'
        while header:=f.read(8):
            assert len(header)==8
            layer,tokens,used=struct.unpack('<HIH',header)
            assert 0<=layer<40 and tokens>0 and used==8
            count=tokens*used*2
            assert f.tell()+count<=path.stat().st_size
            f.seek(count,1)
            records+=1
            counts[tokens]+=1
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest(),
                records=records,tokens_per_record=dict(counts),
                available_fields=['layer_index','token_count','top_k','per_token_expert_ids'],
                missing_fields=['activation_values','activation_dtype','router_weights','service_descriptors',
                                'cache_events','current_Q4_model_provenance'],
                check_scope='header framing only; not all route values',gpu_executed=False)


if __name__=='__main__':
    print(json.dumps([inspect(ROOT/name) for name in ('routes-code-smoke.awtr','routes-code-smoke-v2.awtr','routes-code-16x4096.awtr')],indent=2))
