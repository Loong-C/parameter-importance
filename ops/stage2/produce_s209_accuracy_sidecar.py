"""CLI for the hash-bound G2.6 to S2.9 accuracy sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.experiments.stage2_s209_accuracy import (
    S209AccuracyBlocked,
    produce_s209_accuracy_sidecar,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce a strict G2.6-derived S2.9 accuracy sidecar")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--g26-root-ref", required=True, help="logical sealed G2.6 output directory under data-root")
    parser.add_argument("--output-ref", required=True, help="logical sidecar path under data-root")
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        sidecar = produce_s209_accuracy_sidecar(
            data_root=args.data_root,
            g26_root_ref=args.g26_root_ref,
            output_ref=args.output_ref,
            run_id=args.run_id,
        )
    except (S209AccuracyBlocked, OSError, ValueError) as error:
        print(f"S2.9 accuracy sidecar BLOCKED: {type(error).__name__}:{error}", file=sys.stderr)
        return 3
    print(
        f"S2.9 accuracy sidecar PASS; rows={len(sidecar['rows'])}; "
        f"artifact_hash={sidecar['artifact_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
