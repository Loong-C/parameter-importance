#!/usr/bin/env python3
"""Run one immutable Stage 0 G8 capacity-worker plan."""

from __future__ import annotations

import argparse

from param_importance_nlp.stage0_g8_worker import run_stage0_g8_worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--plan-ref", required=True)
    arguments = parser.parse_args()
    run_stage0_g8_worker(data_root=arguments.data_root, plan_ref=arguments.plan_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
