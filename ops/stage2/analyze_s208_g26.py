"""CLI for detached Stage 2 S2.8/G2.6 statistics.

This entry point only consumes already sealed JSON evidence.  It never starts
model work and requires an explicit JSON object mapping all four upstream Gate
IDs to their canonical Gate records.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from param_importance_nlp.contracts.jsonio import load_canonical_json
from param_importance_nlp.experiments.stage2_s208_g26 import S28G26Blocked, analyze_s208_g26


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed Stage 2 S2.8/G2.6 statistics")
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True, help="directory containing raw_artifact_ref paths")
    parser.add_argument("--memmap-root", type=Path, default=None, help="explicit scratch root required for tensor-bundle inputs")
    parser.add_argument("--references", type=Path, required=True, help="canonical six-cell reference payload")
    parser.add_argument("--matrix", type=Path, required=True, help="canonical formal S2.6 matrix freeze")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--hypothesis-contract", type=Path, required=True)
    parser.add_argument("--upstream-gates", type=Path, required=True, help="canonical JSON mapping Gate ID to Gate record")
    parser.add_argument("--output-root", type=Path, required=True, help="new analysis directory; existing non-empty directories are rejected")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        gates = load_canonical_json(args.upstream_gates)
        if not isinstance(gates, dict):
            raise S28G26Blocked("upstream_gates:OBJECT_REQUIRED")
        result = analyze_s208_g26(
            raw_manifest=args.raw_manifest,
            raw_root=args.raw_root,
            references=args.references,
            matrix=args.matrix,
            preregistration=args.preregistration,
            hypothesis_contract=args.hypothesis_contract,
            upstream_gates=gates,
            output_root=args.output_root,
            memmap_root=args.memmap_root,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (S28G26Blocked, OSError, ValueError) as error:
        print(f"S2.8/G2.6 BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"S2.8/G2.6 {result['status']}; outputs={len(result['output_files'])}; analysis_hash={result['analysis_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
