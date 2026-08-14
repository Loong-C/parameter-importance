#!/usr/bin/env python3
"""Execute the formal Stage 0 G10 delivery and synchronization gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_importance_nlp.stage0_bootstrap import Stage0SourceBinding
from param_importance_nlp.stage0_g10 import execute_stage0_g10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--g9-index-ref", required=True)
    parser.add_argument("--sync-observation-ref", required=True)
    parser.add_argument("--reuse-attestation-ref")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-branch", required=True)
    arguments = parser.parse_args(argv)
    result = execute_stage0_g10(
        binding=Stage0SourceBinding(
            repository=arguments.repository.resolve(strict=True),
            git_commit=arguments.git_commit,
            git_branch=arguments.git_branch,
            worktree_clean=True,
        ),
        data_root=arguments.data_root.resolve(strict=True),
        g9_index_ref=arguments.g9_index_ref,
        sync_observation_ref=arguments.sync_observation_ref,
        reuse_attestation_ref=arguments.reuse_attestation_ref,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "index_ref": result.index_ref,
                "config_ref": result.config_ref,
                "environment_ref": result.environment_ref,
                "readiness_ref": result.readiness_ref,
                "task_output_refs": dict(result.task_output_refs),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
