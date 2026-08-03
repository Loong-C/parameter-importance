"""Independent zero-network verifier for Stage 0 G3 DOWNLOADED candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from param_importance_nlp.g3_lifecycle_evidence import (
    G3VerificationFailed,
    verify_stage0_g3_acquisition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consume an immutable acquisition chain, hash every local file, and "
            "publish VERIFIED candidates without acquisition or network fallback"
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--download-plan", type=Path, required=True)
    parser.add_argument("--acquisition-ref", required=True)
    parser.add_argument("--actor-instance-id", required=True)
    parser.add_argument("--generator-git-commit", required=True)
    parser.add_argument("--checked-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = verify_stage0_g3_acquisition(
            source_root=arguments.source_root,
            data_root=arguments.data_root,
            requirements=arguments.requirements,
            layout=arguments.layout,
            download_plan=arguments.download_plan,
            acquisition_ref=arguments.acquisition_ref,
            actor_instance_id=arguments.actor_instance_id,
            generator_git_commit=arguments.generator_git_commit,
            checked_at=arguments.checked_at,
        )
    except G3VerificationFailed as error:
        result = error.result
        print(
            f"status=FAILED assets={len(result.candidate_ids)} "
            f"verification_ref={result.verification_ref} "
            f"verification_sha256={result.verification_sha256}"
        )
        return 2
    print(
        f"status={result.status} assets={len(result.candidate_ids)} "
        f"verification_ref={result.verification_ref} "
        f"verification_sha256={result.verification_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
