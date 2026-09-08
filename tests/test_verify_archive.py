import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "scripts/verify_archive.py"
SPEC = importlib.util.spec_from_file_location("verify_archive", MODULE)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p100-manifest-test-")
        self.addCleanup(self.temp.cleanup)
        self.archive = Path(self.temp.name) / "archive"
        self.archive.mkdir()
        self.file = self.archive / "evidence.txt"
        self.file.write_bytes(b"evidence\n")
        self.manifest = dict(schema_version=1, file_count=1, total_bytes=9,
                             files=[dict(path="evidence.txt", size=9,
                                         sha256=verify.sha256(self.file))])

    def test_valid_archive(self):
        self.assertEqual(verify.verify_manifest(self.archive, self.manifest),
                         {"files": 1, "bytes": 9})

    def test_modified_same_length(self):
        self.file.write_bytes(b"modified\n")
        with self.assertRaisesRegex(verify.VerificationError, "SHA-256 changed"):
            verify.verify_manifest(self.archive, self.manifest)

    def test_truncated(self):
        self.file.write_bytes(b"short")
        with self.assertRaisesRegex(verify.VerificationError, "Size changed"):
            verify.verify_manifest(self.archive, self.manifest)

    def test_missing(self):
        self.file.unlink()
        with self.assertRaisesRegex(verify.VerificationError, "File set differs"):
            verify.verify_manifest(self.archive, self.manifest)

    def test_extra(self):
        (self.archive / "extra.txt").write_bytes(b"extra")
        with self.assertRaisesRegex(verify.VerificationError, "File set differs"):
            verify.verify_manifest(self.archive, self.manifest)

    def test_duplicate(self):
        self.manifest["files"] *= 2
        with self.assertRaisesRegex(verify.VerificationError, "Duplicate"):
            verify.verify_manifest(self.archive, self.manifest)

    def test_bad_paths(self):
        for name in ("../escape", "/etc/passwd", "sub/../../escape", "a\\b", "./x", "a//b", ""):
            with self.subTest(name=name):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0]["path"] = name
                with self.assertRaises(verify.VerificationError):
                    verify.verify_manifest(self.archive, manifest)

    def test_symlink_file_and_directory(self):
        for target in (self.file, Path(self.temp.name)):
            with self.subTest(target=target):
                link = self.archive / "link"
                link.symlink_to(target)
                try:
                    with self.assertRaisesRegex(verify.VerificationError, "Symlink rejected"):
                        verify.verify_manifest(self.archive, self.manifest)
                finally:
                    link.unlink()

    def test_manifest_totals(self):
        for key in ("file_count", "total_bytes"):
            manifest = copy.deepcopy(self.manifest)
            manifest[key] += 1
            with self.subTest(key=key), self.assertRaises(verify.VerificationError):
                verify.verify_manifest(self.archive, manifest)


class TimingTests(unittest.TestCase):
    def test_discard_round_zero_and_keep_outlier(self):
        samples = [(0, 1e9)] + [(r, float(r)) for r in range(1, 8)] + [(8, 1e6)]
        self.assertEqual(verify.retained_median(samples), 4.5)

    def test_missing_and_duplicate_rounds(self):
        samples = [(r, 1.0) for r in range(9)]
        for bad in (samples[:-1], samples + [(1, 1.0)], samples[:-1] + [(1, 1.0)]):
            with self.assertRaisesRegex(verify.VerificationError, "timing round"):
                verify.retained_median(bad)

    def test_invalid_times(self):
        for value in (math.nan, math.inf, -1.0, 0.0):
            with self.subTest(value=value), self.assertRaises(verify.VerificationError):
                verify.retained_median([(0, value)] + [(r, 1.0) for r in range(1, 9)])


if __name__ == "__main__":
    unittest.main()
