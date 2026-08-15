"""Run one S1.8 route under ``torchrun`` from a hash-bound worker plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    source = repository / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from param_importance_nlp.stage1_ddp import execute_worker

    report = execute_worker(args.plan)
    if report.get("status") == "PASS":
        print(report["artifact_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
