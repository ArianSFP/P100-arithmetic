#!/usr/bin/env python3
"""Source provenance and precision checks, no CUDA initialization."""
import json
import re
import unittest
from pathlib import Path
import build

ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
    def test_original_current_control_unchanged(self):
        source=build.SOURCE.read_text()
        marker='template<bool BF16_INPUT>\n__global__ __launch_bounds__(256, 2)\nstatic void aw_q8_service_m64_n128_halfpipe_sync('
        expected=build.block(source,marker)
        emitted=build.block((ROOT/'build/current_control.inc').read_text(),marker)
        self.assertEqual(expected,emitted)
        self.assertEqual(build.digest(build.SOURCE),build.EXPECTED)

    def test_precision_and_dispatch(self):
        sass=(ROOT/'build/worker.sass').read_text()
        self.assertNotIn('HFMA2',sass)
        sections=re.split(r'Function\s*:\s*(\S+)',sass)
        for name,body in zip(sections[1::2],sections[2::2]):
            if 'HMUL2' in body:
                if 'decode_check_kernel' not in name:
                    self.assertRegex(name,r'(?:wide|compact)_q4ILi\d+ELb1E|tile256_(?:liveness_)?q4|tile256_direct_stage_q4|tile512_one_rail_q4')
                    self.assertIn('FFMA',body)
        self.assertIn('FFMA',sass)
        text=(ROOT/'build/worker.cu').read_text()
        self.assertIn('c.kind!=-4&&c.kind!=-5',text)
        self.assertIn('aw_current_q4',text);self.assertIn('aw_special_q4',text)
        log=(ROOT/'build/compile.log').read_text()
        spills=re.findall(r'(\d+) bytes spill (?:loads|stores)',log)
        self.assertTrue(spills);self.assertTrue(all(int(x)==0 for x in spills))

    def test_manifest(self):
        manifest=json.loads((ROOT/'build/manifest.json').read_text())
        for p,h in manifest['hashes'].items():self.assertEqual(build.digest(Path(p)),h,p)

if __name__=='__main__':unittest.main()
