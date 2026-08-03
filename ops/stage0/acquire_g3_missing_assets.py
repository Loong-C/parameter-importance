"""Server entry point for the URL-free Stage 0 G3 acquisition plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from param_importance_nlp.asset_download_plan import (
    execute_g3_download_plan,
    load_g3_download_plan,
)
from param_importance_nlp.asset_layout import load_stage0_asset_layout
from param_importance_nlp.asset_requirements import load_stage0_asset_requirements


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    requirements = load_stage0_asset_requirements(arguments.requirements)
    layout = load_stage0_asset_layout(arguments.layout, requirements=requirements)
    plan = load_g3_download_plan(
        arguments.plan,
        requirements=requirements,
        layout=layout,
    )
    report = execute_g3_download_plan(
        plan=plan,
        source_root=arguments.source_root,
        data_root=arguments.data_root,
        report_path=arguments.report,
        started_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    print(
        f"status={report['status']} objects={len(report['objects'])} "
        f"plan_sha256={report['plan_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
