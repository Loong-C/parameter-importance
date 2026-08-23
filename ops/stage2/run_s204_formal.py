#!/usr/bin/env python3
"""S2.4 formal reference preflight/launcher.

This entry point is intentionally small and fail-closed.  ``--plan-only`` is
the only mode enabled until a fixed-state provider command is supplied by the
server execution environment.  It validates the immutable G2.1/S2.3 inputs,
computes the legal sizing workload, and emits per-cell progress locations.  It
never creates a synthetic provider or silently chooses a final ``B_ref``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


G21_ARTIFACT = "259831e2a1b16afbbef34c9cea602e636756b0f6173d1a8f4c32ec554c653f79"
ASSET_DIGEST = "f57decd5cf00e69e45ab2f02c994abb202f5c614e1441acb8aebcb1807ff76ee"
DATA_DIGEST = "df8eeac5178305d409cf6128ac5d5648567aae895592c79fa21542e84a28e0f1"
EXCLUDED_PCI = "0000:50:00.0"
DEFAULT_CANDIDATES = (512, 1024, 2048, 4096)
DEFAULT_BLOCK_SIZE = 32


def _canonical_hash(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label}: expected lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(g21: Mapping[str, Any], assets: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    if g21.get("status") != "PASS":
        raise ValueError("G2.1 evidence is not PASS")
    if g21.get("artifact_hash") != G21_ARTIFACT:
        raise ValueError("G2.1 artifact identity mismatch")
    smoke = g21.get("current_gpu_smoke", {})
    if not isinstance(smoke, Mapping) or smoke.get("status") != "PASS":
        raise ValueError("G2.1 current GPU smoke is not PASS")
    excluded = smoke.get("excluded_pci_bus_ids", [])
    if EXCLUDED_PCI not in excluded or smoke.get("excluded_scheduled") is not False:
        raise ValueError("required failed GPU exclusion is not bound")
    if assets.get("asset_resolution_hash") != ASSET_DIGEST or assets.get("status") != "READY":
        raise ValueError("S2.3 asset digest/status mismatch")
    checkpoints = assets.get("checkpoints")
    if assets.get("checkpoint_matrix_complete") is not True or not isinstance(checkpoints, list) or len(checkpoints) != 6:
        raise ValueError("S2.3 does not contain the six ready checkpoints")
    if any(not isinstance(item, Mapping) or item.get("state") != "ready" for item in checkpoints):
        raise ValueError("S2.3 checkpoint matrix contains a non-ready checkpoint")
    if data.get("data_range_hash") != DATA_DIGEST:
        raise ValueError("S2.3 data digest mismatch")
    if data.get("sample_id_min") != 0 or data.get("sample_id_max_exclusive") != 524288:
        raise ValueError("S2.3 sample range drift")
    if data.get("input_sequence_length") != 2048:
        raise ValueError("S2.3 sequence length drift")


def build_plan(
    g21: Mapping[str, Any],
    assets: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    output_root: Path,
    candidates: tuple[int, ...] = DEFAULT_CANDIDATES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    per_sequence_seconds: float = 0.25,
) -> dict[str, Any]:
    validate_inputs(g21, assets, data)
    if tuple(sorted(set(candidates))) != candidates or len(candidates) < 2 or any(item <= 0 for item in candidates):
        raise ValueError("candidate sizing counts must be strictly increasing positive integers")
    if any(item % block_size for item in candidates):
        raise ValueError("candidate sizing counts must be block aligned")
    if not math.isfinite(per_sequence_seconds) or per_sequence_seconds <= 0:
        raise ValueError("per_sequence_seconds must be finite and positive")
    cells: list[dict[str, Any]] = []
    rows = assets["checkpoints"]
    for checkpoint in rows:
        cell_id = str(checkpoint["checkpoint_id"])
        # Sizing is one independent stream; final A and B are each full-length.
        sizing_draws = candidates[-1]
        final_draws_per_stream = "UNFROZEN"
        fixed_work_units = 3 * sizing_draws
        cells.append(
            {
                "cell_id": cell_id,
                "model_id": checkpoint["model_id"],
                "training_stage": checkpoint["training_stage"],
                "checkpoint_revision": checkpoint["revision"],
                "parameter_registry_hash": checkpoint["parameter_registry_hash"],
                "candidate_sample_counts": list(candidates),
                "block_size": block_size,
                "b_ref_status": "UNFROZEN_UNTIL_SIZING_PASS",
                "minimum_legal_candidate_max_per_stream": sizing_draws,
                "final_sample_count_per_stream": final_draws_per_stream,
                "fixed_work_units_at_candidate_max": fixed_work_units,
                "estimated_seconds_at_candidate_max": fixed_work_units * per_sequence_seconds,
                "progress_path": (output_root / cell_id / "progress.jsonl").as_posix(),
            }
        )
    total_units = sum(int(item["fixed_work_units_at_candidate_max"]) for item in cells)
    return {
        "schema_version": "stage2-s204-formal-plan-v1",
        "stage": "stage2.04_reference_target",
        "scope": "formal_preflight",
        "formal_eligible": False,
        "g2_1_artifact_hash": G21_ARTIFACT,
        "asset_resolution_digest": ASSET_DIGEST,
        "data_range_digest": DATA_DIGEST,
        "excluded_pci": EXCLUDED_PCI,
        "reference_protocol": "independent_reference_sizing_then_one_shot_A_B",
        "optional_stopping": False,
        "replacement_or_resampling": False,
        "candidate_sample_counts": list(candidates),
        "block_size": block_size,
        "b_ref_status": "UNFROZEN_UNTIL_INDEPENDENT_SIZING_PASS",
        "cell_count": len(cells),
        "cells": cells,
        "total_fixed_work_units_at_candidate_max": total_units,
        "total_estimated_seconds_at_candidate_max": total_units * per_sequence_seconds,
        "estimated_duration_note": "Conservative planning estimate only; measure a legal GPU smoke before launch.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S2.4 formal reference plan/preflight")
    parser.add_argument("--g21-evidence", type=Path, required=True)
    parser.add_argument("--asset-resolution", type=Path, required=True)
    parser.add_argument("--data-range", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="validate and emit plan; never run gradients")
    parser.add_argument("--candidate-sizes", type=int, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--per-sequence-seconds", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.plan_only:
        print("S2.4 execute mode is disabled until a server fixed-state provider command is explicitly bound", file=sys.stderr)
        return 2
    try:
        plan = build_plan(
            _load(args.g21_evidence),
            _load(args.asset_resolution),
            _load(args.data_range),
            output_root=args.output_root,
            candidates=tuple(args.candidate_sizes),
            block_size=args.block_size,
            per_sequence_seconds=args.per_sequence_seconds,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        output = args.output_root / "s204-formal-plan.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
        print(f"plan_path={output}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"S2.4 preflight blocked: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
