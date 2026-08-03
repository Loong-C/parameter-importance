#!/usr/bin/env python3
"""Run one torchrun rank of one hash-bound Stage 0 G6 launch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from param_importance_nlp.stage0_g6_worker import run_stage0_g6_worker
from param_importance_nlp.storage import DATA_ROOT_ENV


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan-ref", required=True)
    arguments = parser.parse_args(argv)

    data_root = arguments.data_root.resolve(strict=True)
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured is None or Path(configured).resolve(strict=True) != data_root:
        raise RuntimeError("STAGE0_G6_WORKER_DATA_ROOT_ENV_MISMATCH")
    run_stage0_g6_worker(data_root=data_root, plan_ref=arguments.plan_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
