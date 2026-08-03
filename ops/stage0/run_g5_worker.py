#!/usr/bin/env python3
"""Run one hash-bound fresh-process child of the Stage 0 G5 suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from param_importance_nlp.stage0_g5_worker import run_stage0_g5_worker
from param_importance_nlp.storage import DATA_ROOT_ENV


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan-ref", required=True)
    parser.add_argument("--report-ref", required=True)
    arguments = parser.parse_args(argv)

    data_root = arguments.data_root.resolve(strict=True)
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured is None or Path(configured).resolve(strict=True) != data_root:
        raise RuntimeError("STAGE0_G5_WORKER_DATA_ROOT_ENV_MISMATCH")
    report = run_stage0_g5_worker(
        data_root=data_root,
        plan_ref=arguments.plan_ref,
        report_ref=arguments.report_ref,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": report["run_id"],
                "run_kind": report["run_kind"],
                "report_ref": arguments.report_ref,
                "artifact_hash": report["artifact_hash"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

