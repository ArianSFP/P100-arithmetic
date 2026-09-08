#!/usr/bin/env python3
"""Hash-checked, one-instruction ELF patcher for the HFMA2 selector study."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


SELECTOR_FIELDS = {
    "a": 47,
    "b": 28,
    "c": 35,
    "dest": 49,
}


def sections(blob: bytes):
    if blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        raise SystemExit("expected ELF64 little-endian input")
    e_shoff = struct.unpack_from("<Q", blob, 40)[0]
    e_shentsize = struct.unpack_from("<H", blob, 58)[0]
    e_shnum = struct.unpack_from("<H", blob, 60)[0]
    e_shstrndx = struct.unpack_from("<H", blob, 62)[0]
    raw = [
        struct.unpack_from("<IIQQQQIIQQ", blob, e_shoff + i * e_shentsize)
        for i in range(e_shnum)
    ]
    shstr = raw[e_shstrndx]
    names = blob[shstr[4] : shstr[4] + shstr[5]]
    out = []
    for i, fields in enumerate(raw):
        end = names.find(b"\0", fields[0])
        out.append((i, names[fields[0] : end].decode(errors="replace"), fields[4], fields[5]))
    return out


def parse_int(value: str) -> int:
    return int(value, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--section", required=True)
    parser.add_argument("--offset", required=True, type=parse_int, help="section-relative byte offset")
    parser.add_argument("--baseline-word", required=True, type=parse_int)
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--raw-mask", type=parse_int)
    parser.add_argument("--raw-value", type=parse_int)
    for field in SELECTOR_FIELDS:
        parser.add_argument(f"--{field}-selector", type=int, choices=range(4))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    source = args.input.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    if args.baseline_sha256 and args.baseline_sha256 != source_hash:
        raise SystemExit(f"baseline hash mismatch: expected {args.baseline_sha256}, got {source_hash}")
    matches = [s for s in sections(source) if s[1] == args.section]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one section named {args.section!r}, found {len(matches)}")
    _, _, file_offset, size = matches[0]
    if args.offset < 0 or args.offset % 8 or args.offset + 8 > size:
        raise SystemExit("target offset must be an aligned word inside the section")
    absolute = file_offset + args.offset
    original_bytes = source[absolute : absolute + 8]
    original_word = struct.unpack("<Q", original_bytes)[0]
    if original_word != args.baseline_word:
        raise SystemExit(
            f"baseline word mismatch at {args.section}+0x{args.offset:x}: "
            f"expected 0x{args.baseline_word:016x}, got 0x{original_word:016x}"
        )

    candidate_word = original_word
    chosen = {}
    for field, bit in SELECTOR_FIELDS.items():
        value = getattr(args, f"{field}_selector")
        if value is not None:
            candidate_word = (candidate_word & ~(0x3 << bit)) | (value << bit)
            chosen[field] = value
    raw_mask = args.raw_mask
    raw_value = args.raw_value
    if (raw_mask is None) != (raw_value is None):
        raise SystemExit("--raw-mask and --raw-value must be supplied together")
    if raw_mask is not None:
        if raw_mask < 0 or raw_mask > 0xffffffffffffffff:
            raise SystemExit("raw mask must fit in 64 bits")
        if raw_value < 0 or raw_value > 0xffffffffffffffff or raw_value & ~raw_mask:
            raise SystemExit("raw value must fit inside raw mask")
        candidate_word = (candidate_word & ~raw_mask) | raw_value
        chosen["raw_mask"] = f"0x{raw_mask:016x}"
        chosen["raw_value"] = f"0x{raw_value:016x}"
    if not chosen:
        raise SystemExit("choose at least one selector field or raw mutation")
    if candidate_word == original_word:
        raise SystemExit("candidate is identical to baseline")

    patched = bytearray(source)
    patched[absolute : absolute + 8] = struct.pack("<Q", candidate_word)
    changed = [i for i, (a, b) in enumerate(zip(source, patched)) if a != b]
    expected_changed = [
        absolute + i
        for i in range(8)
        if original_bytes[i] != patched[absolute + i]
    ]
    if changed != expected_changed:
        raise SystemExit("unexpected patch shape")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.write_bytes(patched)
    candidate_hash = hashlib.sha256(patched).hexdigest()
    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "input_sha256": source_hash,
        "output_sha256": candidate_hash,
        "section": args.section,
        "section_relative_offset": args.offset,
        "file_offset": absolute,
        "baseline_word": f"0x{original_word:016x}",
        "candidate_word": f"0x{candidate_word:016x}",
        "changed_byte_offsets": changed,
        "selectors": chosen,
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
