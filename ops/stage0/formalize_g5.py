#!/usr/bin/env python3
"""Execute the canonical fresh-process Stage 0 G5 single-GPU suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from param_importance_nlp.stage0_bootstrap import inspect_stage0_source
from param_importance_nlp.stage0_g5 import execute_stage0_g5
from param_importance_nlp.storage import DATA_ROOT_ENV


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--generator-git-commit", required=True)
    parser.add_argument("--g4-index-ref", required=True)
    arguments = parser.parse_args(argv)

    data_root = arguments.data_root.resolve(strict=True)
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured is None or Path(configured).resolve(strict=True) != data_root:
        raise RuntimeError("STAGE0_G5_DATA_ROOT_ENV_MISMATCH")
    binding = inspect_stage0_source(
        arguments.repository,
        expected_commit=arguments.generator_git_commit,
    )
    result = execute_stage0_g5(
        binding=binding,
        data_root=data_root,
        g4_index_ref=arguments.g4_index_ref,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "index_ref": result.index_ref,
                "environment_ref": result.environment_ref,
                "environment_hash": result.environment.environment_hash,
                "task_output_refs": dict(result.task_output_refs),
                "gate_id": "stage0.G5",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
