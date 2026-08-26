"""CLI for explicitly producing S2.6 formal input contracts.

The command has no default action and never contacts a server on import.  GPU
collection is an explicit ``gpu-inventory`` subcommand; cost/retry/evidence
commands only read the supplied local DATA_ROOT references and publish new
immutable canonical artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import write_canonical_json
from param_importance_nlp.experiments.stage2_s206_inputs import (
    S206FormalInputError,
    build_cost_semantics_contract,
    build_formal_execution_evidence,
    build_retry_policy_contract,
    collect_gpu_inventory,
)


def _publish_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        from param_importance_nlp.contracts.jsonio import load_canonical_json

        existing = load_canonical_json(path)
        if existing != payload:
            raise S206FormalInputError(f"IMMUTABLE_TARGET_CONFLICT:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce immutable S2.6 formal inputs")
    sub = parser.add_subparsers(dest="kind", required=True)

    gpu = sub.add_parser("gpu-inventory", help="explicitly collect a live nvidia-smi inventory")
    gpu.add_argument("--data-root", type=Path, required=True)
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument("--source-output", type=Path, required=True)
    gpu.add_argument("--nvidia-smi", default="nvidia-smi")
    gpu.add_argument("--checked-at")

    cost = sub.add_parser("cost-semantics", help="publish the frozen cost meanings")
    cost.add_argument("--output", type=Path, required=True)
    cost.add_argument("--scientific-measurement-ref")
    cost.add_argument("--cost-io-quiescent", action="store_true")

    retry = sub.add_parser("retry-policy", help="publish the explicit retry policy")
    retry.add_argument("--output", type=Path, required=True)
    retry.add_argument("--max-cell-attempts", type=int, required=True)

    evidence = sub.add_parser("execution-evidence", help="publish an exact G23/G24a/S205 evidence amendment")
    evidence.add_argument("--data-root", type=Path, required=True)
    evidence.add_argument("--g23-evaluation", required=True)
    evidence.add_argument("--g24a-evaluation", required=True)
    evidence.add_argument("--s205-rebind", required=True)
    evidence.add_argument("--parent-evidence")
    evidence.add_argument("--matrix-prerequisite")
    evidence.add_argument("--contract-freeze-hash")
    evidence.add_argument("--asset-manifest-hash", action="append", default=[])
    evidence.add_argument("--output", type=Path, required=True)
    evidence.add_argument("--checked-at")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.kind == "gpu-inventory":
            payload = collect_gpu_inventory(
                output=args.output,
                source_output=args.source_output,
                data_root=args.data_root,
                executable=args.nvidia_smi,
                checked_at=args.checked_at,
            )
        elif args.kind == "cost-semantics":
            payload = build_cost_semantics_contract(
                scientific_measurement_ref=args.scientific_measurement_ref,
                cost_io_quiescent=args.cost_io_quiescent,
            )
            _publish_once(args.output.resolve(), payload)
        elif args.kind == "retry-policy":
            payload = build_retry_policy_contract(max_cell_attempts=args.max_cell_attempts)
            _publish_once(args.output.resolve(), payload)
        else:
            payload = build_formal_execution_evidence(
                data_root=args.data_root,
                g23_evaluation_ref=args.g23_evaluation,
                g24a_evaluation_ref=args.g24a_evaluation,
                s205_rebind_ref=args.s205_rebind,
                parent_evidence_ref=args.parent_evidence,
                matrix_prerequisite_ref=args.matrix_prerequisite,
                contract_freeze_hash=args.contract_freeze_hash,
                asset_manifest_hashes=tuple(args.asset_manifest_hash),
                checked_at=args.checked_at,
            )
            _publish_once(args.output.resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, S206FormalInputError) as error:
        print(f"S2.6 formal input blocked: {type(error).__name__}:{error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
