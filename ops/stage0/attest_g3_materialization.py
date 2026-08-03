"""Attest Stage 0 G3 materialization and publish DOWNLOADED candidates only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from param_importance_nlp.g3_lifecycle_evidence import (
    attest_stage0_g3_acquisition,
)


def _source_input(root: Path, value: Path, *, field: str) -> tuple[Path, str]:
    candidate = value if value.is_absolute() else root / value
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} must remain below --source-root") from error
    if not candidate.exists() or not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{field} must be an existing non-link file")
    return candidate, relative.as_posix()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consume the canonical 13-object report, build declared derived assets, "
            "and publish an immutable URL-free acquisition attestation"
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--download-plan", type=Path, required=True)
    parser.add_argument("--download-report-ref", required=True)
    parser.add_argument("--actor-instance-id", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--completed-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    source_root = Path(os.path.abspath(arguments.source_root)).resolve(strict=True)
    requirements_path, requirements_ref = _source_input(
        source_root, arguments.requirements, field="requirements"
    )
    layout_path, layout_ref = _source_input(
        source_root, arguments.layout, field="layout"
    )
    plan_path, plan_ref = _source_input(
        source_root, arguments.download_plan, field="download-plan"
    )
    result = attest_stage0_g3_acquisition(
        source_root=source_root,
        data_root=arguments.data_root,
        requirements=requirements_path,
        layout=layout_path,
        download_plan=plan_path,
        requirements_ref=requirements_ref,
        layout_ref=layout_ref,
        download_plan_ref=plan_ref,
        download_report_ref=arguments.download_report_ref,
        actor_instance_id=arguments.actor_instance_id,
        source_git_commit=arguments.source_git_commit,
        started_at=arguments.started_at,
        completed_at=arguments.completed_at,
    )
    print(
        f"status={result.status} assets={len(result.candidate_ids)} "
        f"acquisition_ref={result.acquisition_ref} "
        f"acquisition_sha256={result.acquisition_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
