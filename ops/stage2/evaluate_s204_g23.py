"""CLI for the independent Stage 2 G2.3 reference evaluator.

Only paths to formal TaskRuntime results are accepted.  The command never
accepts a metrics file or a pass/fail override; metrics are derived by
``stage2_g23_evaluator`` from committed artifacts and resume bundles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Keep the repository CLI directly runnable from a clean checkout as well as
# from the editable package installed on the formal image.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.experiments.stage2_g23_evaluator import (
    CellInput,
    _reject_symlink_chain,
    evaluate_formal_g23,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate formal Stage 2 G2.3 reference evidence")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, help="trusted repository checkout for producer/source verification")
    parser.add_argument(
        "--task-result",
        action="append",
        default=[],
        help="workspace-relative task-run-result.json; repeat once per cell",
    )
    parser.add_argument(
        "--cell-id",
        action="append",
        default=[],
        help="cell id corresponding by order to --task-result (repeat six times)",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="optional resolved formal config path corresponding by order",
    )
    parser.add_argument(
        "--input-index",
        type=Path,
        help="JSON array of {cell_id, task_result_ref, optional config_ref}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        workspace_root = args.workspace_root.resolve()
        if args.input_index is not None:
            index_path = args.input_index.resolve()
            if args.input_index.is_symlink():
                raise ValueError("--input-index symlink is forbidden")
            try:
                index_path.relative_to(workspace_root)
            except ValueError as error:
                raise ValueError("--input-index must be under --workspace-root") from error
            _reject_symlink_chain(
                workspace_root,
                index_path.relative_to(workspace_root).as_posix(),
                "--input-index",
            )

            def _no_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for key, item in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key: {key}")
                    result[key] = item
                return result

            value = json.loads(
                index_path.read_text(encoding="utf-8"),
                object_pairs_hook=_no_duplicate_pairs,
            )
            if not isinstance(value, list):
                raise ValueError("--input-index must contain a JSON array")
            cells = [CellInput.from_mapping(item) for item in value]
        else:
            if args.cell_id and len(args.cell_id) != len(args.task_result):
                raise ValueError("--cell-id count must equal --task-result count")
            if args.config and len(args.config) != len(args.task_result):
                raise ValueError("--config count must equal --task-result count")
            ids = args.cell_id or [f"cell-{index}" for index in range(len(args.task_result))]
            configs = args.config or [None] * len(args.task_result)
            cells = [CellInput(ids[index], str(ref), configs[index]) for index, ref in enumerate(args.task_result)]
        result = evaluate_formal_g23(
            workspace_root,
            cells,
            expected_cell_ids=tuple(item.cell_id for item in cells),
            output_root=args.output_root,
            repo_root=args.repo_root,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result.get("status") == "PASS" and result.get("formal_eligible") is True else 3
    except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"G2.3 evaluation blocked: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
