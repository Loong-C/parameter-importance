"""Publish the completed Stage 3.10 delivery inventory for independent G3-8.

This entry point is intentionally a publisher, not an artifact generator.  It
loads an already completed manifest, adds the exact four Stage 3.10 commits to
the immutable source set, and delegates all schema/file checks to the strict
G3-8 delivery authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import load_canonical_json
from param_importance_nlp.experiments.stage3_g38_publisher import (
    publish_stage3_delivery_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a complete, hash-bound Stage 3.10 delivery manifest "
            "for independent G3-8 consumption"
        )
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis-report-ref", required=True)
    parser.add_argument("--chart-artifacts-ref", required=True)
    parser.add_argument("--handoff-manifest-ref", required=True)
    parser.add_argument("--gate-summary-ref", required=True)
    parser.add_argument(
        "--source-ref",
        action="append",
        default=[],
        help="Additional immutable source ref; may be repeated",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = load_canonical_json(arguments.manifest)
    if not isinstance(manifest, dict):
        raise TypeError("STAGE3_G38_DELIVERY_MANIFEST_MUST_BE_OBJECT")
    stage3_10_refs = {
        "analysis_report": arguments.analysis_report_ref,
        "chart_artifacts": arguments.chart_artifacts_ref,
        "handoff_manifest": arguments.handoff_manifest_ref,
        "gate_summary": arguments.gate_summary_ref,
    }
    source_refs = tuple(
        dict.fromkeys((*arguments.source_ref, *stage3_10_refs.values()))
    )
    published = publish_stage3_delivery_manifest(
        workspace_root=arguments.workspace_root,
        output_dir=arguments.output_dir,
        config_hash=arguments.config_hash,
        manifest=manifest,
        stage3_10_refs=stage3_10_refs,
        source_refs=source_refs,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "commit_ref": published.commit_ref,
                "artifact_hash": published.identity.artifact_hash,
                "source_refs": list(source_refs),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
