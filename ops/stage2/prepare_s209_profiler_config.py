"""Produce the strict, hash-bound config for the S2.9 production worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.experiments.stage2_s209_production import (  # noqa: E402
    S209ProductionBlocked,
    prepare_s209_profiler_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a strict S2.9 production profiler config")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--matrix-ref", required=True)
    parser.add_argument("--g24b-gate-ref", required=True)
    parser.add_argument("--raw-manifest-ref", required=True)
    parser.add_argument("--g25-gate-ref", required=True)
    parser.add_argument("--s27-plan-ref", required=True)
    parser.add_argument("--materialization-index-ref", required=True)
    parser.add_argument("--execution-evidence-ref", required=True)
    parser.add_argument("--measurement-plan-ref", required=True)
    parser.add_argument("--output-ref", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = prepare_s209_profiler_config(
            data_root=args.data_root,
            matrix_ref=args.matrix_ref,
            g24b_gate_ref=args.g24b_gate_ref,
            raw_manifest_ref=args.raw_manifest_ref,
            g25_gate_ref=args.g25_gate_ref,
            s27_plan_ref=args.s27_plan_ref,
            materialization_index_ref=args.materialization_index_ref,
            execution_evidence_ref=args.execution_evidence_ref,
            measurement_plan_ref=args.measurement_plan_ref,
            output_ref=args.output_ref,
            run_id=args.run_id,
        )
        # Diagnostics are JSON as well, but on stderr: stdout is reserved for
        # the exact machine-readable config summary used by detached callers.
        print(json.dumps({"status": "PASS", "artifact_hash": config["artifact_hash"]}, separators=(",", ":")))
        return 0
    except S209ProductionBlocked as error:
        print(f"S2.9 profiler config BLOCKED: {error}", file=sys.stderr)
        return 3
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        print(f"S2.9 profiler config BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
