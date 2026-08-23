"""CLI for the independent Stage 2 G2.0 evaluator.

The command accepts only workspace-relative formal TaskArtifact commit refs.
It intentionally has no ``--status``, ``--metric`` or threshold override: all
Gate fields are derived by :mod:`stage2_g20_evaluator`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.experiments.stage2_g20_evaluator import (  # noqa: E402
    ARTIFACT_KINDS,
    evaluate_formal_g20,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate formal Stage 2 G2.0 preregistration evidence")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-ref",
        action="append",
        default=[],
        help="workspace-relative task-output commit; repeat exactly three times",
    )
    parser.add_argument(
        "--input-index",
        type=Path,
        help="JSON object mapping preregistration/hypothesis_contract/gate_record to commit refs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.input_index is not None:
            value = json.loads(args.input_index.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("--input-index must contain a JSON object")
            refs = {kind: value[kind] for kind in ARTIFACT_KINDS}
        else:
            if len(args.artifact_ref) != len(ARTIFACT_KINDS):
                raise ValueError("--artifact-ref must be supplied exactly three times")
            refs = tuple(args.artifact_ref)
        result = evaluate_formal_g20(
            args.workspace_root,
            refs,
            output_root=args.output_root,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result.get("status") == "PASS" and result.get("formal_eligible") is True else 3
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        # Invalid CLI input is always fail-closed; never print a caller supplied
        # value as if it were a GateRecord.
        print(f"G2.0 evaluation blocked: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
