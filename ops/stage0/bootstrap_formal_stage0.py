#!/usr/bin/env python3
"""Publish current G0--G2 formal evidence and canonical S0.1--S0.3 outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from param_importance_nlp.stage0_bootstrap import (
    bootstrap_formal_stage0,
    inspect_stage0_runtime,
    inspect_stage0_source,
)
from param_importance_nlp.storage import DATA_ROOT_ENV


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--generator-git-commit", required=True)
    parser.add_argument("--checked-at", required=True)
    arguments = parser.parse_args(argv)

    data_root = arguments.data_root.resolve(strict=True)
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured is None or Path(configured).resolve(strict=True) != data_root:
        raise RuntimeError("STAGE0_BOOTSTRAP_DATA_ROOT_ENV_MISMATCH")
    binding = inspect_stage0_source(
        arguments.repository,
        expected_commit=arguments.generator_git_commit,
    )
    snapshot = inspect_stage0_runtime(
        binding=binding,
        data_root=data_root,
        checked_at=arguments.checked_at,
    )
    result = bootstrap_formal_stage0(
        binding=binding,
        data_root=data_root,
        snapshot=snapshot,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "index_ref": result.index_ref,
                "environment_hash": result.environment.environment_hash,
                "last_task_output_refs": result.task_output_refs[
                    "stage0.03_runtime_and_dependencies"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
