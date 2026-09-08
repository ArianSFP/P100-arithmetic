#!/usr/bin/env python3
"""Maintainer utility: create a NEW manifest; never overwrite an existing one."""

import json
from pathlib import Path

from verify_archive import archive_files, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    files = [{"path": name, "size": path.stat().st_size, "sha256": sha256(path)}
             for name, path in archive_files(root / "archive").items()]
    manifest = {
        "schema_version": 1,
        "publication_date_utc": "2026-09-08",
        "original_root": "/home/arian/hfma2-gp100-probe",
        "archive_directory": "archive",
        "excluded_patterns": ["/tooling/", "__pycache__/", "*.pyc", ".git/"],
        "file_count": len(files),
        "total_bytes": sum(entry["size"] for entry in files),
        "files": files,
    }
    with (root / "EXPORT-MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Created manifest: {len(files)} files, {manifest['total_bytes']} bytes")


if __name__ == "__main__":
    main()
