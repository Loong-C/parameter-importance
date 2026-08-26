"""Run one strict, UUID-bound S2.9 profiler backend.

The detached S2.9 runner supplies the frozen task through environment
variables.  By default this entrypoint loads the repository-contained real
S2.7/Torch adapter, including the synchronized approved-four-UUID system
anchor; an explicit backend remains available for a separately reviewed
adapter.  Diagnostics go to stderr and exactly one JSON object goes to
stdout.  There is no synthetic fallback and no fabricated measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.experiments.stage2_s209_worker import (  # noqa: E402
    S209WorkerBlocked,
    execute_cli,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict UUID-bound S2.9 profiler worker (one JSON object on stdout)"
    )
    parser.add_argument(
        "--backend",
        default="param_importance_nlp.experiments.stage2_s209_production:run_s209_production_backend",
        help=(
            "real backend import path in module:function form; defaults to the "
            "repository-contained approved S2.7/Torch adapter (including the "
            "four-process approved-UUID system anchor)"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="canonical, hash-bound formal worker config JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute_cli(backend_spec=args.backend, config_path=args.config)
        # Keep stdout machine-only: the parent runner's JSON decoder must see
        # exactly one object, while backend diagnostics were redirected to
        # stderr by run_s209_profiler_worker().
        sys.stdout.write(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        return 0
    except S209WorkerBlocked as error:
        print(f"S2.9 profiler worker BLOCKED: {error}", file=sys.stderr)
        return 3
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        print(f"S2.9 profiler worker BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
