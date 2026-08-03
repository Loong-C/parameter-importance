#!/usr/bin/env python3
"""Collect the immutable post-sync observation consumed by Stage 0 G10."""

from __future__ import annotations

import argparse
from pathlib import Path

from param_importance_nlp.stage0_g10_sync import (
    collect_stage0_g10_sync_observation,
    write_sync_observation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--previous-github-head", required=True)
    parser.add_argument("--previous-server-head", required=True)
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    repository = arguments.repository.resolve(strict=True)
    output = arguments.output.resolve()
    try:
        output.relative_to(repository)
    except ValueError:
        pass
    else:
        parser.error("--output must be outside the Git repository so the post-sync worktree stays clean")
    observation = collect_stage0_g10_sync_observation(
        repository=repository,
        branch=arguments.branch,
        previous_github_head=arguments.previous_github_head,
        previous_server_head=arguments.previous_server_head,
        authorization_ref=arguments.authorization_ref,
    )
    write_sync_observation(output, observation)
    print(observation["artifact_hash"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
