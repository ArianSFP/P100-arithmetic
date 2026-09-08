#!/usr/bin/env python3
"""Minimal ELF64 little-endian section and word inspector for cubins."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path


def parse_sections(blob: bytes):
    if blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        raise SystemExit("expected ELF64 little-endian input")
    e_shoff = struct.unpack_from("<Q", blob, 40)[0]
    e_shentsize = struct.unpack_from("<H", blob, 58)[0]
    e_shnum = struct.unpack_from("<H", blob, 60)[0]
    e_shstrndx = struct.unpack_from("<H", blob, 62)[0]
    raw = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        raw.append(struct.unpack_from("<IIQQQQIIQQ", blob, off))
    shstr = raw[e_shstrndx]
    names = blob[shstr[4] : shstr[4] + shstr[5]]

    def name_at(index: int) -> str:
        end = names.find(b"\0", index)
        return names[index:end].decode("utf-8", errors="replace")

    return [
        {
            "index": i,
            "name": name_at(fields[0]),
            "type": fields[1],
            "offset": fields[4],
            "size": fields[5],
            "addr": fields[3],
            "align": fields[8],
        }
        for i, fields in enumerate(raw)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cubin", type=Path)
    parser.add_argument("--section", default=None)
    parser.add_argument("--all-sections", action="store_true")
    args = parser.parse_args()

    blob = args.cubin.read_bytes()
    print(f"sha256={hashlib.sha256(blob).hexdigest()}")
    sections = parse_sections(blob)
    wanted = [
        s for s in sections
        if args.all_sections or args.section is None or s["name"] == args.section
    ]
    if not wanted:
        raise SystemExit(f"section not found: {args.section}")
    for section in wanted:
        start = section["offset"]
        end = start + section["size"]
        data = blob[start:end]
        print(
            f"section={section['name']} index={section['index']}"
            f" file_offset=0x{start:x} size=0x{section['size']:x}"
            f" addr=0x{section['addr']:x} align={section['align']}"
        )
        if section["size"] == 0:
            continue
        if section["size"] % 8:
            print("warning=section_size_not_multiple_of_8")
        for off in range(0, len(data) - 7, 8):
            word = struct.unpack_from("<Q", data, off)[0]
            print(f"word section_offset=0x{off:x} file_offset=0x{start + off:x} value=0x{word:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
