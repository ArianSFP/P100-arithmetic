"""Read-only implementation review: no execution/import of implementation code."""
import hashlib
from pathlib import Path
import random
from oracle import plane_dot

ROOT = Path(__file__).resolve().parents[3]
IMPL = Path(__file__).resolve().parent.parent/'implementation'


def body(source):
    start = source.index('static void aw_q8_service_m64_n128_halfpipe_sync(')
    end = source.index('{', start)
    depth = 1
    end += 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    source = ROOT/'studies/qwen35-q4-t64-20260908/affinity-wave.cu'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'fee528c1270b053d37d56a88018c185a71f9e418a1fa572a1b086a669c574a47'
    assert body(source.read_text()) == body((IMPL/'build/current.inc').read_text())
    # Independent code-value reference versus implemented offset-nibble lane mapping.
    rng = random.Random(517)
    for _ in range(1024):
        w = [rng.randrange(-8, 8) for _ in range(32)]
        a = [rng.randrange(-7, 8) for _ in range(32)]
        packed = [(w[j]+8) | ((w[j+16]+8) << 4) for j in range(16)]
        decoded = [((packed[lane % 16] >> (0 if lane < 16 else 4)) & 15)-8 for lane in range(32)]
        assert decoded == w
        assert plane_dot(decoded, a) == sum(x*y for x, y in zip(w, a))
    # The two supported RM geometries must write every valid output exactly once.
    for rm in (2, 4):
        for m in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129):
            covered = set()
            for tile in range(0, m, 8*rm):
                for tid in range(128):
                    m0, n0 = (tid//16)*rm, (tid % 16)*4
                    for i in range(rm):
                        for l in range(4):
                            row, col = tile+m0+i, n0+l
                            if row < m:
                                assert (row, col) not in covered
                                covered.add((row, col))
            assert len(covered) == m*64
    print('PASS: unchanged current T64 body, 1024 offset-nibble/plane groups, 24 tail geometry cases')
    for name in ('kernels.cuh', 'worker.cu', 'build.py'):
        print(hashlib.sha256((IMPL/name).read_bytes()).hexdigest(), name)


if __name__ == '__main__':
    main()
