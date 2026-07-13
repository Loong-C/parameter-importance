#!/usr/bin/env python3
"""Compare an independent mmap reader with Pythia's official batch viewer reader."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

MAGIC = b"MMIDIDX\x00\x00"
DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def load_official(source: Path):
    module_path = source / "utils" / "mmap_dataset.py"
    spec = importlib.util.spec_from_file_location("official_pythia_mmap_dataset", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MMapIndexedDataset


def read_index(idx: Path):
    with idx.open("rb") as f:
        if f.read(9) != MAGIC:
            raise RuntimeError("bad idx magic")
        version = struct.unpack("<Q", f.read(8))[0]
        if version != 1:
            raise RuntimeError(f"unsupported idx version {version}")
        dtype = DTYPES[f.read(1)[0]]
        count, documents = struct.unpack("<QQ", f.read(16))
        sizes = np.fromfile(f, dtype="<i4", count=count)
        pointers = np.fromfile(f, dtype="<i8", count=count)
    return dtype, sizes, pointers, documents


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=Path, required=True)
    p.add_argument("--bin", type=Path, required=True)
    p.add_argument("--pythia-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[0, 1, 511])
    args = p.parse_args()
    dtype, sizes, pointers, documents = read_index(args.idx)
    mmap = np.memmap(args.bin, mode="r", dtype=dtype)
    Official = load_official(args.pythia_source)
    records = []
    with tempfile.TemporaryDirectory(prefix="pythia-batch-viewer-") as tmp:
        prefix = Path(tmp) / "document"
        os.symlink(args.idx.resolve(), prefix.with_suffix(".idx"))
        os.symlink(args.bin.resolve(), prefix.with_suffix(".bin"))
        official = Official(str(prefix), skip_warmup=True)
        for step in args.steps:
            first = step * 1024
            last = (step + 1) * 1024
            direct = np.stack([
                np.asarray(mmap[pointers[i] // dtype().nbytes : pointers[i] // dtype().nbytes + sizes[i]])
                for i in range(first, last)
            ])
            reference = np.asarray(official[first:last])
            equal = np.array_equal(direct, reference)
            records.append({
                "step": step,
                "shape": list(direct.shape),
                "independent_sha256": digest(direct),
                "official_batch_viewer_sha256": digest(reference),
                "equal": equal,
            })
            if not equal:
                raise RuntimeError(f"batch mismatch at step {step}")
    report = {"official_source": str(args.pythia_source.resolve()), "documents": documents, "batches": records, "status": "ok"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
