#!/usr/bin/env python3
"""Verify and immutably record the canonical G3-8 handoff for Stage 4."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from param_importance_nlp.contracts.jsonio import (  # noqa: E402
    JSONValue,
    canonical_json_bytes,
)
from param_importance_nlp.experiments.stage3_g38_publisher import (  # noqa: E402
    validate_stage3_g38_handoff_authority,
)
from param_importance_nlp.runtime import publish_canonical_immutable  # noqa: E402


def _output_path(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    absolute = Path(os.path.abspath(target))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError("STAGE3_G38_HANDOFF_OUTPUT_OUTSIDE_WORKSPACE") from error
    if absolute.suffix.casefold() != ".json":
        raise ValueError("STAGE3_G38_HANDOFF_OUTPUT_MUST_BE_JSON")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("STAGE3_G38_HANDOFF_OUTPUT_SYMLINK")
    return absolute


def verify_and_publish_handoff(
    *,
    workspace_root: str | Path,
    gate_ref: str,
    publication_ref: str,
    output: str | Path,
) -> Mapping[str, JSONValue]:
    root = Path(workspace_root).resolve(strict=True)
    audit = validate_stage3_g38_handoff_authority(
        root,
        gate_ref=gate_ref,
        publication_ref=publication_ref,
    )
    publish_canonical_immutable(_output_path(root, Path(output)), audit)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--g3-8-gate-ref", required=True)
    parser.add_argument("--g3-8-publication-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    audit = verify_and_publish_handoff(
        workspace_root=arguments.workspace_root,
        gate_ref=arguments.g3_8_gate_ref,
        publication_ref=arguments.g3_8_publication_ref,
        output=arguments.output,
    )
    print(canonical_json_bytes(audit).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
