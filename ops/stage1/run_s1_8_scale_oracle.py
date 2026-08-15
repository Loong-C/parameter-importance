"""CLI entry point for S1.8's independent pre-route gradient-scale oracle."""

from __future__ import annotations

import argparse

from param_importance_nlp.stage1_ddp_scale_oracle import execute_scale_oracle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    arguments = parser.parse_args()
    execute_scale_oracle(arguments.plan)
    return 0


if __name__ == "__main__":  # pragma: no cover - command wrapper
    raise SystemExit(main())
