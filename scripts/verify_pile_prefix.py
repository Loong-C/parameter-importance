#!/usr/bin/env python3
"""Validate that a Megatron MMapIndexedDataset prefix fits in one .bin shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

MAGIC = b"MMIDIDX\x00\x00"
DTYPE_BYTES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 4, 7: 8, 8: 2}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=Path, required=True)
    p.add_argument("--bin", type=Path, required=True)
    p.add_argument("--samples", type=int, default=512 * 1024)
    p.add_argument("--tokens-per-sample", type=int, default=2049)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    with args.idx.open("rb") as f:
        magic = f.read(9)
        if magic != MAGIC:
            raise SystemExit(f"unexpected idx magic: {magic!r}")
        version = read_u64(f)
        if version != 1:
            raise SystemExit(f"unsupported idx version: {version}")
        dtype_code = f.read(1)[0]
        dtype_bytes = DTYPE_BYTES.get(dtype_code)
        if dtype_bytes is None:
            raise SystemExit(f"unsupported dtype code: {dtype_code}")
        sequence_count = read_u64(f)
        document_count = read_u64(f)
        if args.samples > sequence_count:
            raise SystemExit(f"need {args.samples} samples, idx has {sequence_count}")
        sizes_raw = f.read(args.samples * 4)
        sizes = struct.unpack(f"<{args.samples}i", sizes_raw)
        if any(size != args.tokens_per_sample for size in sizes):
            bad = next((i, size) for i, size in enumerate(sizes) if size != args.tokens_per_sample)
            raise SystemExit(f"sample size mismatch at {bad[0]}: {bad[1]}")
        f.seek((sequence_count - args.samples) * 4, 1)
        pointers_raw = f.read(args.samples * 8)
        pointers = struct.unpack(f"<{args.samples}q", pointers_raw)

    max_end = max(ptr + size * dtype_bytes for ptr, size in zip(pointers, sizes))
    bin_size = args.bin.stat().st_size
    report = {
        "idx": str(args.idx.resolve()),
        "bin": str(args.bin.resolve()),
        "idx_sha256": sha256(args.idx),
        "bin_sha256": sha256(args.bin),
        "idx_version": version,
        "dtype_code": dtype_code,
        "dtype_bytes": dtype_bytes,
        "sequence_count": sequence_count,
        "document_count": document_count,
        "required_samples": args.samples,
        "tokens_per_sample": args.tokens_per_sample,
        "max_required_end_offset": max_end,
        "bin_size": bin_size,
        "covered": max_end <= bin_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["covered"]:
        raise SystemExit(f"prefix ends at {max_end}, beyond shard size {bin_size}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
