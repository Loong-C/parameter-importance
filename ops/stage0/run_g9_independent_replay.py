#!/usr/bin/env python3
"""Run the hash-bound Stage 0 G9 independent replay plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from param_importance_nlp.stage0_g9_replay import run_stage0_g9_independent_replay
from param_importance_nlp.storage import DATA_ROOT_ENV


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan-ref", required=True)
    arguments = parser.parse_args(argv)
    root = arguments.data_root.resolve(strict=True)
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured is None or Path(configured).resolve(strict=True) != root:
        raise RuntimeError("STAGE0_G9_REPLAY_DATA_ROOT_ENV_MISMATCH")
    report = run_stage0_g9_independent_replay(
        data_root=root,
        plan_ref=arguments.plan_ref,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "replay_id": report["replay_id"],
                "artifact_hash": report["artifact_hash"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
