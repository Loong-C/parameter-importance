#!/usr/bin/env python3
"""Re-verify installed model, dataset, and source assets against their manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    verified: list[dict[str, object]] = []
    for line_number, raw in enumerate(args.manifest.read_text(encoding="ascii").splitlines(), 1):
        expected_hash, expected_size_text, relative = raw.split("\t", 2)
        expected_size = int(expected_size_text)
        path = args.root / relative
        if not path.is_file():
            raise SystemExit(f"missing asset at manifest line {line_number}: {relative}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise SystemExit(
                f"asset size mismatch at line {line_number}: {relative}: "
                f"{actual_size} != {expected_size}"
            )
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise SystemExit(f"asset SHA-256 mismatch at line {line_number}: {relative}")
        verified.append({"path": relative, "size": actual_size, "sha256": actual_hash})

    report = {
        "manifest": str(args.manifest.resolve()),
        "files": len(verified),
        "bytes": sum(int(item["size"]) for item in verified),
        "status": "ok",
        "verified": verified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("files", "bytes", "status")}, indent=2))


if __name__ == "__main__":
    main()
