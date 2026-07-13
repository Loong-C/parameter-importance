#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

CORE = {
    "torch": "2.12.1+cu126",
    "transformers": "4.57.6",
    "datasets": "4.8.5",
}


def canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    pins = {canonical(item["metadata"]["name"]): item["metadata"]["version"] for item in report["install"]}
    for name, expected in CORE.items():
        if pins.get(name) != expected:
            raise SystemExit(f"core pin mismatch for {name}: {pins.get(name)!r} != {expected!r}")
    lines = ["--extra-index-url https://download.pytorch.org/whl/cu126"]
    lines.extend(f"{name}=={version}" for name, version in sorted(pins.items()))
    args.output.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {len(pins)} exact pins to {args.output}")


if __name__ == "__main__":
    main()
