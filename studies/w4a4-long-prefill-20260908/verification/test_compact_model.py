"""Independent compact32x64 address model; no parent import/build or GPU."""
from collections import Counter
import unittest


class CompactTests(unittest.TestCase):
    def test_stage_and_output_bijections(self):
        a, b, out = Counter(), Counter(), Counter()
        for tid in range(128):
            warp, lane = divmod(tid, 32)
            ar = (warp//2)*16+(lane//8)*4
            bc = (warp % 2)*32+(lane % 8)*4
            for i in range(4):
                for j in range(4):
                    out[ar+i, bc+j] += 1
            for i in range(8):
                a[(tid % 4)*8+i, tid//4] += 1
            for l in range(2):
                at = tid+l*128
                for i in range(4):
                    b[(at//64)*4+i, at % 64] += 1
                    b[(at//64)*4+i+16, at % 64] += 1
        self.assertEqual(a, Counter({(k,m):1 for k in range(32) for m in range(32)}))
        self.assertEqual(b, Counter({(k,n):1 for k in range(32) for n in range(64)}))
        self.assertEqual(out, Counter({(m,n):1 for m in range(32) for n in range(64)}))

    def test_word_major_load_and_nibble_positions(self):
        for tid in range(128):
            for l in range(2):
                at = tid+l*128
                row, chunk = at % 64, at//64
                self.assertEqual(128+at*4, 128+(chunk*64+row)*4)
                self.assertLessEqual(128+at*4+4, 1152)
                self.assertLessEqual(2*row+2, 128)
                for i in range(4):
                    self.assertEqual(chunk*4+i, at//64*4+i)
                    self.assertLess(chunk*4+i+16, 32)

    def test_expert_jobs_tails_alignment_and_rails(self):
        for t in (1,31,32,33,63,64,65,128,511,512):
            for n in (128,512,2048):
                mt, ng = (t+31)//32, n//64
                for job in range(2*mt*ng):
                    e, tm, col = job//(mt*ng), (job//ng) % mt, job % ng
                    self.assertEqual(job, (e*mt+tm)*ng+col)
                    for g in (0,2):
                        wb = ((e*ng+col)*3+g)*1152
                        self.assertEqual(wb % 16, 0)
                    for tid in (0,3,4,127):
                        row = tm*32+tid//4
                        if row < t:
                            ai = (e*t+row)*96+64+(tid % 4)*8
                            self.assertEqual(ai % 4, 0)
                            self.assertLessEqual(ai+8, (e+1)*t*96)
        # Each rail preserves increasing-group and intra-group K order.
        for u in (8,16,32):
            rails = [[], []]
            for g in range(3):
                for chunk in range(32//u):
                    for s in range(u):
                        kk = chunk*u+s
                        rails[kk//16].append(g*32+kk)
            self.assertEqual(rails, [[g*32+j for g in range(3) for j in range(r*16,(r+1)*16)] for r in (0,1)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
