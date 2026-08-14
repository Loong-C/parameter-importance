#!/usr/bin/env python3
"""Create a reviewed, content-bound G0--G9 reuse attestation for G10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_importance_nlp.contracts import load_canonical_json, write_canonical_json
from param_importance_nlp.evidence_reuse import build_evidence_reuse_attestation
from param_importance_nlp.stage0_g10 import _REQUIRED_GATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--producer-git-commit", required=True)
    parser.add_argument("--consumer-git-commit", required=True)
    parser.add_argument("--consumer-git-branch", required=True)
    parser.add_argument("--g9-index-ref", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    review = load_canonical_json(arguments.review.resolve(strict=True))
    if (
        not isinstance(review, dict)
        or set(review) != {"schema_version", "reviews"}
        or review.get("schema_version") != "evidence-reuse-review-v1"
        or not isinstance(review.get("reviews"), list)
    ):
        raise ValueError("review must be a canonical evidence-reuse-review-v1 object")
    value = build_evidence_reuse_attestation(
        repository=arguments.repository.resolve(strict=True),
        data_root=arguments.data_root.resolve(strict=True),
        producer_commit=arguments.producer_git_commit,
        consumer_commit=arguments.consumer_git_commit,
        consumer_branch=arguments.consumer_git_branch,
        scope_id="stage0.G0-G9",
        source_evidence_ref=arguments.g9_index_ref,
        preserved_gate_ids=sorted(_REQUIRED_GATES),
        reviews=review["reviews"],
    )
    output = arguments.output.resolve()
    data_root = arguments.data_root.resolve(strict=True)
    try:
        output.relative_to(data_root)
    except ValueError as error:
        raise ValueError("output must be inside data-root") from error
    if output.exists():
        if load_canonical_json(output) != value:
            raise ValueError("immutable output already exists with different content")
    else:
        write_canonical_json(output, value)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "artifact_hash": value["artifact_hash"],
                "changed_path_count": len(value["changed_paths"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
