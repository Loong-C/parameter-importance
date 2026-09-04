#!/usr/bin/env python3
"""Publish the independent Stage 3 G3-8 Gate and canonical receipt.

This entry point accepts only immutable formal commit refs.  The underlying
publisher reloads every prerequisite Gate, all four S3.10 commits, the final
execution chain, G3-7 authorities, and the delivery manifest before it writes
the G3-8 Gate and receipt.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys
from typing import Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from param_importance_nlp.contracts.jsonio import canonical_json_bytes  # noqa: E402
from param_importance_nlp.experiments.stage3_g38_publisher import (  # noqa: E402
    REQUIRED_STAGE3_G38_GATE_IDS,
    REQUIRED_STAGE3_G38_STAGE310_KINDS,
    STAGE3_G38_RECEIPT_ARTIFACT_KIND,
    publish_stage3_g38,
)
from param_importance_nlp.runtime import load_committed_task_artifact  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    for index in range(8):
        parser.add_argument(f"--g3-{index}-ref", required=True)
    for role in REQUIRED_STAGE3_G38_STAGE310_KINDS:
        parser.add_argument(f"--{role.replace('_', '-')}-ref", required=True)
    parser.add_argument("--execution-evidence-ref", required=True)
    parser.add_argument("--g37-publication-ref", required=True)
    parser.add_argument("--recommendation-ref", required=True)
    parser.add_argument("--finalization-ref", required=True)
    parser.add_argument("--delivery-manifest-ref", required=True)
    parser.add_argument("--publication-id")
    parser.add_argument("--expected-publication-config-hash")
    parser.add_argument("--checked-at")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.workspace_root.resolve(strict=True)
    gate_refs = {
        gate_id: getattr(arguments, f"g3_{index}_ref")
        for index, gate_id in enumerate(REQUIRED_STAGE3_G38_GATE_IDS)
    }
    stage3_10_refs = {
        role: getattr(arguments, f"{role}_ref")
        for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
    }
    publication = publish_stage3_g38(
        workspace_root=root,
        output_dir=arguments.output_dir,
        gate_refs=gate_refs,
        stage3_10_refs=stage3_10_refs,
        execution_evidence_ref=arguments.execution_evidence_ref,
        g3_7_publication_ref=arguments.g37_publication_ref,
        recommendation_ref=arguments.recommendation_ref,
        finalization_ref=arguments.finalization_ref,
        delivery_manifest_ref=arguments.delivery_manifest_ref,
        publication_id=arguments.publication_id,
        publication_config_hash=arguments.expected_publication_config_hash,
        checked_at=arguments.checked_at,
    )
    output_dir = PurePosixPath(arguments.output_dir).as_posix().rstrip("/")
    publication_ref = (
        f"{output_dir}/commits/{STAGE3_G38_RECEIPT_ARTIFACT_KIND}.json"
    )
    gate_commit = load_committed_task_artifact(
        root, publication.g3_8_ref, require_formal=True
    )
    publication_commit = load_committed_task_artifact(
        root, publication_ref, require_formal=True
    )
    report = {
        "status": "PASS",
        "formal_eligible": True,
        "publication_id": publication.publication_id,
        "publication_config_hash": publication.publication_config_hash,
        "g3_8_ref": publication.g3_8_ref,
        "g3_8_hash": publication.g3_8_hash,
        "g3_8_commit_hash": gate_commit.identity.artifact_hash,
        "publication_ref": publication_ref,
        "publication_hash": publication.artifact_hash,
        "publication_commit_hash": publication_commit.identity.artifact_hash,
    }
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
