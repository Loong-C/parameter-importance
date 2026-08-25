"""CLI for the detached S2.10/G2.7b report and decision consumer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from param_importance_nlp.experiments.stage2_s210_g27b import S210G27BBlocked, run_s210_g27b


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed Stage 2 S2.10/G2.7b report and decision")
    parser.add_argument("--g26-gate", type=Path, required=True)
    parser.add_argument("--g26-quality-gates", type=Path, required=True)
    parser.add_argument("--g26-hypothesis-decisions", type=Path, required=True)
    parser.add_argument("--g26-statistics-long-table", type=Path, required=True)
    parser.add_argument("--g26-statistics-summary", type=Path)
    parser.add_argument("--g26-raw-calibration", type=Path)
    parser.add_argument("--g26-family-decisions", type=Path, required=True)
    parser.add_argument("--g27a-report", type=Path, required=True)
    parser.add_argument("--g27a-gate", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output-root", type=Path, required=True, help="new empty append-only report directory")
    parser.add_argument("--run-id", default="s210-g27b")
    parser.add_argument("--checked-at")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_s210_g27b(
            g26_gate=args.g26_gate,
            g26_quality_gates=args.g26_quality_gates,
            g26_hypothesis_decisions=args.g26_hypothesis_decisions,
            g26_statistics_long_table=args.g26_statistics_long_table,
            g26_statistics_summary=args.g26_statistics_summary,
            g26_raw_calibration=args.g26_raw_calibration,
            g26_family_decisions=args.g26_family_decisions,
            g27a_report=args.g27a_report,
            g27a_gate=args.g27a_gate,
            matrix=args.matrix,
            output_root=args.output_root,
            run_id=args.run_id,
            checked_at=args.checked_at,
        )
    except (S210G27BBlocked, OSError, ValueError) as error:
        print(f"S2.10/G2.7b BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"S2.10/G2.7b {result['status']}; outputs={len(result['output_files'])}; analysis_hash={result['analysis_hash']}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
