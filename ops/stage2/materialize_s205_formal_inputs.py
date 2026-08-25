"""Materialize strict, append-only formal inputs for S2.5/G2.4a."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.experiments.stage2_s25_inputs import (
    S205InputBlocked,
    S205_INPUT_INDEX_SCHEMA,
    build_s205_formal_inputs,
)


def _logical(root: Path, value: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S205InputBlocked("OUTPUT_ROOT_INVALID")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise S205InputBlocked("OUTPUT_ROOT_ESCAPE")
    path = (root / Path(*logical.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise S205InputBlocked("OUTPUT_ROOT_ESCAPE") from error
    return logical.as_posix(), path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze exhaustive S2.5 development inputs")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preregistration-ref", required=True)
    parser.add_argument("--sampling-plan-task-ref", required=True)
    parser.add_argument("--formal-execution-ref", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _write_once(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        if path.is_symlink() or load_canonical_json(path) != value:
            raise S205InputBlocked(f"OUTPUT_CONFLICT:{path.name}")
        return
    write_canonical_json(path, value)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.data_root.resolve()
        output_ref, output = _logical(root, args.output_root)
        if output.exists() and (not output.is_dir() or output.is_symlink()):
            raise S205InputBlocked("OUTPUT_ROOT_INVALID_EXISTING_OBJECT")
        sampling, sweep = build_s205_formal_inputs(
            root,
            preregistration_ref=args.preregistration_ref,
            sampling_plan_ref=args.sampling_plan_task_ref,
            formal_execution_ref=args.formal_execution_ref,
        )
        sampling_ref = f"{output_ref}/sampling-plan.json"
        sweep_ref = f"{output_ref}/development-sweep-plan.json"
        index_ref = f"{output_ref}/index.json"
        index: dict[str, object] = {
            "schema_version": S205_INPUT_INDEX_SCHEMA,
            "status": "FROZEN",
            "formal_eligible": True,
            "sampling_plan_ref": sampling_ref,
            "sampling_plan_hash": sampling["plan_hash"],
            "development_sweep_plan_ref": sweep_ref,
            "development_sweep_plan_hash": sweep["artifact_hash"],
            "preregistration_ref": args.preregistration_ref,
            "sampling_plan_task_ref": args.sampling_plan_task_ref,
            "formal_execution_ref": args.formal_execution_ref,
            "primary_parameters_selected": False,
            "confirmatory_draws_generated": False,
            "reference_draws_generated": False,
        }
        index["artifact_hash"] = canonical_json_hash(index)
        output.mkdir(parents=True, exist_ok=True)
        _write_once(output / "sampling-plan.json", sampling)
        _write_once(output / "development-sweep-plan.json", sweep)
        # The index is the completion boundary and is always published last.
        _write_once(output / "index.json", index)
        print(json.dumps({**index, "index_ref": index_ref}, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, RuntimeError, S205InputBlocked) as error:
        print(json.dumps({
            "schema_version": S205_INPUT_INDEX_SCHEMA,
            "status": "BLOCKED",
            "formal_eligible": False,
            "reason": f"{type(error).__name__}:{error}",
        }, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
