#!/usr/bin/env python3
"""Offline-only tests: no subprocess/GPU calls are allowed in these cases."""
import copy
import unittest
from unittest.mock import patch

import run_gpu


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.no_commands = patch.object(run_gpu.subprocess, 'run', side_effect=AssertionError('External call in offline test'))
        self.no_commands.start()
        self.addCleanup(self.no_commands.stop)

    def healthy(self):
        rows = [f'{i}, {run_gpu.GPU if i==0 else "other-"+str(i)}, Tesla P100, 580.173.02, 5, 0, 35, 1328, 0'
                for i in range(4)]
        return {'devices': {'returncode': 0, 'stdout': '\n'.join(rows)},
                'processes': {'returncode': 0, 'stdout': ''}}

    def test_healthy(self):
        run_gpu.check_health(self.healthy())

    def test_health_rejections(self):
        for bad in ('memory', 'busy', 'ecc', 'missing', 'query', 'process'):
            with self.subTest(bad=bad):
                h = copy.deepcopy(self.healthy())
                if bad == 'memory': h['devices']['stdout'] = h['devices']['stdout'].replace(', 5, 0,', ', 100, 0,')
                if bad == 'busy': h['devices']['stdout'] = h['devices']['stdout'].replace(', 5, 0,', ', 5, 50,')
                if bad == 'ecc': h['devices']['stdout'] = h['devices']['stdout'].replace('1328, 0', '1328, 1')
                if bad == 'missing': h['devices']['stdout'] = ''
                if bad == 'query': h['devices']['returncode'] = None
                if bad == 'process': h['processes']['stdout'] = 'a compute process'
                with self.assertRaises(RuntimeError): run_gpu.check_health(h)

    def test_current_no_reservation(self):
        with self.assertRaises(RuntimeError):
            run_gpu.check_reservation('', run_gpu.sha(run_gpu.COORD))

    def test_wrong_snapshot(self):
        with self.assertRaises(RuntimeError): run_gpu.check_reservation('not-held', '0'*64)

    def test_offline_provenance(self):
        run_gpu.verify_offline()


if __name__ == '__main__':
    unittest.main()
