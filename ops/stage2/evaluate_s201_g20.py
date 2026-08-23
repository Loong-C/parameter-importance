"""CLI for the independent Stage 2 G2.0 evaluator.

The data and repository roots are explicit.  The command has no status,
metric, threshold, or formal-eligibility override: all Gate fields are derived
by :mod:`stage2_g20_evaluator`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.experiments.stage2_g20_evaluator import (  # noqa: E402
    ARTIFACT_KINDS,
    evaluate_formal_g20,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate formal Stage 2 G2.0 preregistration evidence")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--resolved-config-ref", required=True)
    parser.add_argument("--output-dir", default="runs/stage2-g20-evaluation")
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


def _data_index_path(data_root: Path, value: Path) -> Path:
    """Read an optional input index only from DATA_ROOT, without symlinks."""

    root = data_root.absolute()
    candidate = value if value.is_absolute() else root / value
    candidate = candidate.absolute()
    current = Path(candidate.anchor)
    for part in candidate.parts[1:] if candidate.anchor else candidate.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("--input-index may not traverse a symlink")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise ValueError("--input-index must be below --data-root") from error
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.input_index is not None:
            index_path = _data_index_path(args.data_root, args.input_index)
            value = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("--input-index must contain a JSON object")
            refs = {kind: value[kind] for kind in ARTIFACT_KINDS}
        else:
            if len(args.artifact_ref) != len(ARTIFACT_KINDS):
                raise ValueError("--artifact-ref must be supplied exactly three times")
            refs = tuple(args.artifact_ref)
        result = evaluate_formal_g20(
            args.data_root,
            refs,
            repository_root=args.repository_root,
            resolved_config_ref=args.resolved_config_ref,
            output_dir=args.output_dir,
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
