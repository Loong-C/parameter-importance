"""Materialize a strict, preflight-ready S2.7 formal plan.

The command is read-only with respect to S2.4--S2.6 inputs.  It publishes
only the direct per-cell G2.3 GateRecords and the requested S2.7 plan, each
through the immutable canonical publisher.  No provider, draw API, GPU
worker, or server process is started.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.experiments.stage2_s207_materializer import (  # noqa: E402
    S27MaterializationBlocked,
    materialize_s27_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize strict formal S2.7 plan and G2.3 GateRecords")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--s206-freeze-ref", required=True)
    parser.add_argument("--s205-rebind-ref", required=True)
    parser.add_argument("--g23-evaluation-ref", required=True)
    parser.add_argument("--materialization-index-ref", required=True)
    parser.add_argument("--execution-evidence-ref", required=True)
    parser.add_argument("--gpu-inventory-json", type=Path, required=True)
    parser.add_argument("--failure-rule-ref", required=True)
    parser.add_argument("--g23-gate-output-root", required=True)
    parser.add_argument(
        "--checked-at",
        required=True,
        help="Explicit timezone-bearing ISO-8601 time for the derived direct G2.3 GateRecords",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = materialize_s27_plan(
            args.data_root,
            plan_id=args.plan_id,
            plan_output=args.plan_output,
            s206_freeze_ref=args.s206_freeze_ref,
            s205_rebind_ref=args.s205_rebind_ref,
            g23_evaluation_ref=args.g23_evaluation_ref,
            materialization_index_ref=args.materialization_index_ref,
            execution_evidence_ref=args.execution_evidence_ref,
            gpu_inventory_json=args.gpu_inventory_json,
            failure_rule_ref=args.failure_rule_ref,
            g23_gate_output_root=args.g23_gate_output_root,
            checked_at=args.checked_at,
        )
        print(json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, RuntimeError, S27MaterializationBlocked) as error:
        print(f"S2.7 formal plan materialization blocked: {type(error).__name__}:{error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
