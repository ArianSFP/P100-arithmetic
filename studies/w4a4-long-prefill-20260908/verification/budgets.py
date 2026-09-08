"""Read-only historical screening budgets; never a fresh paired denominator."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'studies/qwen35-q4-t64-20260908/affinity-wave.cu'
EXPECTED = 'fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47'


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    rows = []
    for projection in ('gu', 'down'):
        for m in (128, 256, 512):
            tag = f'current-{projection}-m{m}'
            folder = ROOT/'studies/w4a16-long-prefill-20260908/gpu-results'/tag
            result_bytes = (folder/'result.json').read_bytes()
            result = json.loads(result_bytes)
            assert result['status'] == 'PASS'
            assert result['build']['control_source_sha256'] == EXPECTED
            stdout_bytes = (folder/'stdout.txt').read_bytes()
            samples = [float(x) for x in re.findall(
                r'^TIME aw_current_q4 prep=1 rep=\d+ us=(\S+)$', stdout_bytes.decode(), re.M)]
            assert len(samples) == 7
            median = statistics.median(samples)
            rows.append(dict(tag=tag, projection=projection, m=m, experts=64,
                             current_t64_us=median, w4a4_2x_budget_us=median/2,
                             samples=len(samples), stdout_sha256=hashlib.sha256(stdout_bytes).hexdigest(),
                             result_sha256=hashlib.sha256(result_bytes).hexdigest()))
    print(json.dumps(dict(scope='CPU reading historical equal-M screening; NOT 2k/4k/8k acceptance',
                          gpu_executed=False, source_sha256=EXPECTED, rows=rows), indent=2))


if __name__ == '__main__':
    main()
