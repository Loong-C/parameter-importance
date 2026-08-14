"""Execute one hash-bound S1.7 Pythia-14M worker plan in a fresh process."""

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
    from param_importance_nlp.stage1_single_gpu import execute_worker

    report = execute_worker(args.plan)
    print(report["artifact_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
