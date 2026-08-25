"""Server entry point for the strict S2.11/G2.8 delivery control plane.

This launcher only consumes sealed references and publishes delivery/replay
control artifacts.  It does not execute a replay.  The 31M replay audit must
be produced by a separate, independently reviewed server command and passed
with ``--replay-audit-31m`` on a later invocation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.experiments.stage2_s211_delivery import (
    S211DeliveryBlocked,
    run_s211_g28,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict Stage 2 S2.11/G2.8 delivery and replay control plane")
    parser.add_argument("--g27b-gate", type=Path, required=True)
    for option in ("g20", "g21", "g22", "g23", "g24a", "g24b", "g25", "g26", "g27a"):
        parser.add_argument(f"--{option}-gate", type=Path, required=True)
    parser.add_argument("--g27b-decision", type=Path)
    parser.add_argument("--g27b-lineage", type=Path)
    parser.add_argument("--stage2-lineage", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--formal-14m", type=Path)
    parser.add_argument("--formal-31m", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--replay-audit-31m", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    for role in ("plan", "task-catalog", "replay-report", "gate-summary", "sync-report", "estimator-decision", "large-artifact-index", "worklog", "dirty-head-evidence", "failure-retry-amendment-history"):
        parser.add_argument(f"--{role}", type=Path, required=True)
    parser.add_argument("--run-id", default="s211-g28-delivery")
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--consumer-commit")
    parser.add_argument("--checked-at")
    parser.add_argument("--replay-command", nargs="+", help="Recorded instruction only; never executed by this launcher")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    boundaries: dict[str, Any] = {
        name: value
        for name, value in {
            "environment": args.environment,
            "assets": args.assets,
            "reference": args.reference,
            "pilot": args.pilot,
            "formal_14m": args.formal_14m,
            "formal_31m": args.formal_31m,
            "analysis": args.analysis,
            "decision": args.decision,
        }.items()
        if value is not None
    }
    try:
        predecessor_gates = {
            gate_id: getattr(args, option)
            for gate_id, option in {
                "stage2.G2.0": "g20_gate",
                "stage2.G2.1": "g21_gate",
                "stage2.G2.2": "g22_gate",
                "stage2.G2.3": "g23_gate",
                "stage2.G2.4a": "g24a_gate",
                "stage2.G2.4b": "g24b_gate",
                "stage2.G2.5": "g25_gate",
                "stage2.G2.6": "g26_gate",
                "stage2.G2.7a": "g27a_gate",
            }.items()
        }
        result = run_s211_g28(
            g27b_gate=args.g27b_gate,
            g27b_decision=args.g27b_decision,
            g27b_lineage=args.g27b_lineage,
            stage2_lineage=args.stage2_lineage,
            boundary_refs=boundaries,
            replay_audit_31m=args.replay_audit_31m,
            output_root=args.output_root,
            data_root=args.data_root,
            predecessor_gates=predecessor_gates,
            plan=args.plan,
            task_catalog=args.task_catalog,
            replay_report=args.replay_report,
            gate_summary=args.gate_summary,
            sync_report=args.sync_report,
            estimator_decision=args.estimator_decision,
            large_artifact_index=args.large_artifact_index,
            worklog=args.worklog,
            dirty_head_evidence=args.dirty_head_evidence,
            failure_retry_amendment_history=args.failure_retry_amendment_history,
            run_id=args.run_id,
            producer_commit=args.producer_commit,
            consumer_commit=args.consumer_commit,
            checked_at=args.checked_at,
            replay_command=args.replay_command,
        )
    except (S211DeliveryBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"S2.11/G2.8 BLOCKED: {type(error).__name__}:{error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "formal_eligible": result["formal_eligible"], "delivery_hash": result["delivery_hash"], "gate_hash": result["gate_hash"], "output_files": result["output_files"]}, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
