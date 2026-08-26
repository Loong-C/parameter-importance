"""S2.5 fresh-run rebind preflight.

The default action is read-only.  It emits a strict handoff plan only when a
fresh S2.4 run has one COMPLETE formal result for each of the six cells and a
content-addressed G2.3 PASS.  Formal execution is intentionally delegated to
the reviewed TaskRuntime producer after the plan has been independently
reviewed; this command never starts a runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import write_canonical_json
from param_importance_nlp.experiments.stage2_s25_rebind import (
    S25RebindBlocked,
    S25RebindSpec,
    prepare_s25_rebind,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare strict S2.5 rebind after fresh G2.3 PASS")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s204-run-root", required=True)
    parser.add_argument("--s204-prepared-root", required=True)
    parser.add_argument(
        "--g23-evaluation-ref",
        required=True,
        help="Exact content-addressed G2.3 evaluation.json reference approved for this handoff",
    )
    parser.add_argument(
        "--g23-evaluation-hash",
        required=True,
        help="Exact SHA-256 artifact hash of --g23-evaluation-ref",
    )
    parser.add_argument("--s205-output-root", required=True)
    parser.add_argument("--operations-root", required=True)
    parser.add_argument("--g3-ref", required=True)
    parser.add_argument("--g3-artifact-hash", required=True)
    parser.add_argument(
        "--execution-commit",
        required=True,
        help="40-hex fresh S2.4 execution commit bound to every COMPLETE final-status",
    )
    parser.add_argument("--plan-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = S25RebindSpec(
            data_root=args.data_root.resolve(),
            s204_run_root=args.s204_run_root,
            s204_prepared_root=args.s204_prepared_root,
            g23_evaluation_ref=args.g23_evaluation_ref,
            g23_evaluation_hash=args.g23_evaluation_hash,
            s205_output_root=args.s205_output_root,
            operations_root=args.operations_root,
            g3_ref=args.g3_ref,
            g3_artifact_hash=args.g3_artifact_hash,
            execution_commit=args.execution_commit,
        )
        plan = prepare_s25_rebind(spec)
        if args.plan_output is not None:
            output = args.plan_output.resolve()
            output.relative_to(spec.data_root.resolve())
            plan["plan_ref"] = output.relative_to(spec.data_root.resolve()).as_posix()
            write_canonical_json(output, plan)
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, S25RebindBlocked) as error:
        payload = {
            "schema_version": "stage2-s205-rebind-plan-v1",
            "status": "BLOCKED",
            "formal_eligible": False,
            "reason": f"{type(error).__name__}:{error}",
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
