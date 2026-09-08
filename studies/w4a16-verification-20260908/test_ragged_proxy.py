import unittest
from ragged_proxy import tiles,profile


class TilingTests(unittest.TestCase):
    def test_boundaries(self):
        cases={0:[],16:[(16,16)],17:[(32,17)],32:[(32,32)],33:[(32,32),(16,1)],
               47:[(32,32),(16,15)],48:[(64,48)],63:[(64,63)],64:[(64,64)],
               80:[(64,64),(16,16)],96:[(64,64),(32,32)],112:[(64,64),(64,48)]}
        for m,expected in cases.items():
            self.assertEqual([(t['tile_m'],t['rows']) for t in tiles(m)],expected)

    def test_every_m_conserves_rows(self):
        for m in range(4097):
            cursor=0
            for t in tiles(m):
                self.assertEqual(t['row'],cursor)
                self.assertTrue(0<t['rows']<=t['tile_m'])
                cursor+=t['rows']
            self.assertEqual(cursor,m)

    def test_useful_work_not_tile_count(self):
        p=profile([64,16]+[0]*62,0,0,0)
        self.assertEqual(p['tail_work_share'],.2)
        self.assertEqual(p['mix']['64']['tiles'],p['mix']['16']['tiles'])


if __name__=='__main__':unittest.main(verbosity=2)
