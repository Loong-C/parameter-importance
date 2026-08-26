"""Server entry point for the strict S2.8/G2.6 production consumer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from param_importance_nlp.contracts.jsonio import load_canonical_json
from param_importance_nlp.experiments.stage2_s208_runner import run_s208_g26_production
from param_importance_nlp.experiments.stage2_s208_production import S208ProductionBlocked


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict Stage 2 S2.8/G2.6 production analysis")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True, help="S2.7 six-cell plan/bundle containing S2.4 candidate refs")
    parser.add_argument("--g23-gate", type=Path, required=True, help="independent S2.4/G2.3 PASS evaluation")
    parser.add_argument("--materialization-index", type=Path, required=True, help="S2.4 six-cell materialization index")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--memmap-root", type=Path, default=None)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--hypothesis-contract", type=Path, required=True)
    parser.add_argument("--upstream-gates", type=Path, required=True, help="JSON mapping G2.4a/G2.4b/G2.5 records")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        upstream = load_canonical_json(args.upstream_gates)
        if not isinstance(upstream, dict):
            raise ValueError("upstream_gates:OBJECT_REQUIRED")
        result = run_s208_g26_production(
            data_root=args.data_root,
            raw_manifest=args.raw_manifest,
            raw_root=args.raw_root,
            reference_bundle=args.reference_bundle,
            g23_gate=args.g23_gate,
            materialization_index=args.materialization_index,
            reference_root=args.reference_root,
            memmap_root=args.memmap_root,
            matrix=args.matrix,
            preregistration=args.preregistration,
            hypothesis_contract=args.hypothesis_contract,
            upstream_gates=upstream,
            output_root=args.output_root,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (S208ProductionBlocked, OSError, TypeError, ValueError) as error:
        print(f"S2.8/G2.6 BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"S2.8/G2.6 {result['status']}; outputs={len(result['output_files'])}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
