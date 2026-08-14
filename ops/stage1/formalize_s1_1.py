#!/usr/bin/env python3
"""Execute and publish the formal Stage 1 S1.1 entry boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_importance_nlp.stage1_s1_1 import execute_stage1_s1_1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--g10-index-ref", required=True)
    parser.add_argument("--reuse-attestation-ref")
    parser.add_argument("--attempt-id", required=True)
    arguments = parser.parse_args(argv)

    result = execute_stage1_s1_1(
        repository=arguments.repository.resolve(strict=True),
        data_root=arguments.data_root.resolve(strict=True),
        g10_index_ref=arguments.g10_index_ref,
        reuse_attestation_ref=arguments.reuse_attestation_ref,
        attempt_id=arguments.attempt_id,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "index_ref": result.index_ref,
                "config_ref": result.config_ref,
                "environment_ref": result.environment_ref,
                "result_ref": result.result_ref,
                "task_output_refs": dict(result.task_output_refs),
                "gate_artifact_hashes": dict(result.gate_artifact_hashes),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
