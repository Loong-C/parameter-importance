"""Production S2.8/G2.6 runner.

This is a control-plane adapter only.  It never creates draws or loads a
model.  The only scientific inputs it opens are the sealed S2.7 raw manifest,
the frozen matrix/contracts, the S2.4 candidate reference bundle and the
independent G2.3 PASS supplied to :func:`load_s208_reference_bundle`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Mapping

from ..contracts.jsonio import canonical_json_hash, write_canonical_json
from .stage2_s208_g26 import S28G26Blocked, analyze_s208_g26
from .stage2_s208_production import (
    S208ProductionBlocked,
    _safe_path,
    load_s208_reference_bundle,
    materialize_s208_matrix,
)


S208_RUNNER_SCHEMA = "stage2-s208-g26-production-runner-v1"


def _new_descendant(path: str | Path, parent: Path, *, field: str) -> Path:
    candidate = Path(path).resolve()
    boundary = parent.resolve()
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as error:
        raise S208ProductionBlocked(f"{field}:OUTSIDE_REQUIRED_BOUNDARY") from error
    if not relative.parts:
        raise S208ProductionBlocked(f"{field}:UNIQUE_NAMESPACE_REQUIRED")
    if candidate.exists():
        raise S208ProductionBlocked(f"{field}:NAMESPACE_ALREADY_EXISTS")
    return candidate


def _validate_production_paths(
    data_root: str | Path,
    memmap_root: str | Path | None,
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise S208ProductionBlocked("data_root:DIRECTORY_REQUIRED")
    if memmap_root is None:
        raise S208ProductionBlocked("S208_EXPLICIT_MEMMAP_ROOT_REQUIRED")
    scratch = _new_descendant(memmap_root, root / "tmp", field="memmap_root")
    destination = _new_descendant(
        output_root,
        root / "results" / "stage2" / "derived",
        field="output_root",
    )
    return root, scratch, destination


def _resolve_analysis_gate_refs(root: Path, gates: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve DATA_ROOT-relative gate refs before the detached analyzer reads them."""

    resolved: dict[str, Any] = {}
    for gate_id, value in gates.items():
        if isinstance(value, (str, Path)):
            resolved[str(gate_id)] = _safe_path(root, str(value), f"{gate_id}.ref")
        else:
            resolved[str(gate_id)] = value
    return resolved


def _atomic_publish(destination: Path, files: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    if destination.exists():
        raise S28G26Blocked("OUTPUT_ANALYSIS_DIRECTORY_MUST_BE_NEW")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.stage-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name, value in files.items():
            write_canonical_json(staging / name, value)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return tuple(files)


def _blocked_gate(*, reason: str, source_lineage: Mapping[str, Any], upstream_gates: Mapping[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "stage2-s208-g26-gate-v1",
        "gate_id": "stage2.G2.6",
        "stage": 2,
        "status": "BLOCKED",
        "quality_gate_dependency": False,
        "measured": {"production_runner": S208_RUNNER_SCHEMA},
        "threshold": {"frozen_thresholds": True},
        "reasons": [reason],
        "upstream_gate_hashes": {str(key): value.get("artifact_hash") for key, value in upstream_gates.items() if isinstance(value, Mapping)},
        "reference_lineage": dict(source_lineage),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def run_s208_g26_production(
    *,
    data_root: str | Path,
    raw_manifest: str | Path,
    raw_root: str | Path,
    reference_bundle: str | Path,
    g23_gate: str | Path,
    materialization_index: str | Path,
    matrix: str | Path,
    preregistration: str | Path,
    hypothesis_contract: str | Path,
    upstream_gates: Mapping[str, Mapping[str, Any] | str | Path],
    output_root: str | Path,
    reference_root: str | Path | None = None,
    memmap_root: str | Path | None = None,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260825,
) -> dict[str, Any]:
    """Run S2.8 from real sealed refs and publish PASS/BLOCKED atomically."""

    root, scratch, destination = _validate_production_paths(data_root, memmap_root, output_root)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=False, exist_ok=False)
    loaded: dict[str, Any] | None = None
    try:
        loaded = load_s208_reference_bundle(
            root,
            reference_bundle,
            g23_gate,
            reference_root=reference_root,
            memmap_root=scratch,
        )
        gates = dict(upstream_gates)
        # The G2.3 object consumed by the strict reference loader is the only
        # accepted G2.3 identity; a caller cannot substitute a second mapping.
        gates["stage2.G2.3"] = loaded["g23_gate"]
        g23_gate_ref = loaded.get("lineage", {}).get("g23_gate_ref") if isinstance(loaded.get("lineage"), Mapping) else None
        if not isinstance(g23_gate_ref, str) or not g23_gate_ref:
            raise S208ProductionBlocked("stage2.G2.3:REFERENCE_REQUIRED")
        g24a_input = gates.get("stage2.G2.4a")
        g24b_input = gates.get("stage2.G2.4b")
        if g24a_input is None:
            raise S208ProductionBlocked("stage2.G2.4a:INPUT_REQUIRED")
        if g24b_input is None:
            raise S208ProductionBlocked("stage2.G2.4b:INPUT_REQUIRED")
        if not isinstance(g24a_input, (str, Path)):
            raise S208ProductionBlocked("stage2.G2.4a:REFERENCE_REQUIRED")
        if not isinstance(g24b_input, (str, Path)):
            raise S208ProductionBlocked("stage2.G2.4b:REFERENCE_REQUIRED")
        analysis_gates = _resolve_analysis_gate_refs(root, gates)
        matrix_materialization = materialize_s208_matrix(
            root,
            materialization_index,
            matrix=matrix,
            preregistration=preregistration,
            g23_gate=g23_gate_ref,
            g24a_gate=g24a_input,
            g24b_gate=g24b_input,
            references=loaded,
        )
        result = analyze_s208_g26(
            raw_manifest=raw_manifest,
            raw_root=raw_root,
            references=loaded,
            matrix=matrix,
            matrix_materialization=matrix_materialization,
            preregistration=preregistration,
            hypothesis_contract=hypothesis_contract,
            upstream_gates=analysis_gates,
            output_root=None,
            memmap_root=scratch,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        source_lineage = dict(loaded["lineage"])
        source_lineage["matrix_materialization_hash"] = matrix_materialization["artifact_hash"]
        source_lineage["matrix_materialization_ref"] = "matrix_materialization.json"
        source_lineage["artifact_hash"] = canonical_json_hash({key: value for key, value in source_lineage.items() if key != "artifact_hash"})
        derived_artifacts = result["lineage_manifest"].get("derived_artifacts", [])
        if not isinstance(derived_artifacts, list):
            derived_artifacts = []
        result["lineage_manifest"]["derived_artifacts"] = [
            *derived_artifacts,
            "matrix_materialization.json",
        ]
        result["lineage_manifest"]["derived_artifact_hashes"] = {
            "matrix_materialization.json": matrix_materialization["artifact_hash"],
        }
        result["lineage_manifest"]["reference_source_lineage"] = source_lineage
        result["lineage_manifest"]["artifact_hash"] = canonical_json_hash({key: value for key, value in result["lineage_manifest"].items() if key != "artifact_hash"})
        result["input_audit"]["reference_source_lineage"] = source_lineage
        result["input_audit"]["matrix_materialization"] = {
            "ref": "matrix_materialization.json",
            "artifact_hash": matrix_materialization["artifact_hash"],
        }
        result["input_audit"]["artifact_hash"] = canonical_json_hash({key: value for key, value in result["input_audit"].items() if key != "artifact_hash"})
        files: dict[str, Mapping[str, Any]] = {
            "analysis_input_audit.json": result["input_audit"],
            "statistics_long_table.json": {"schema_version": result["schema_version"], "rows": result["statistics_long_table"], "artifact_hash": canonical_json_hash({"schema_version": result["schema_version"], "rows": result["statistics_long_table"]})},
            "statistics_summary.json": {"schema_version": result["schema_version"], "rows": result["statistics_summary"], "artifact_hash": canonical_json_hash({"schema_version": result["schema_version"], "rows": result["statistics_summary"]})},
            "raw_calibration.json": {"schema_version": result["schema_version"], "rows": result["raw_calibration"], "artifact_hash": canonical_json_hash({"schema_version": result["schema_version"], "rows": result["raw_calibration"]})},
            "confirmatory_family_decisions.json": result["confirmatory_family_decisions"],
            "quality_gates.json": result["quality_gates"],
            "hypothesis_decisions.json": result["hypothesis_decisions"],
            "lineage_manifest.json": result["lineage_manifest"],
            "matrix_materialization.json": matrix_materialization,
            "g2.6-gate.json": result["g2_6_gate"],
        }
        result["output_files"] = list(_atomic_publish(destination, files))
        result["runner"] = {"schema_version": S208_RUNNER_SCHEMA, "published_at": datetime.now(timezone.utc).isoformat(), "reference_lineage_hash": source_lineage["artifact_hash"]}
        result["analysis_hash"] = canonical_json_hash(result)
        return result
    except (S208ProductionBlocked, S28G26Blocked, OSError, TypeError, ValueError) as error:
        lineage = loaded.get("lineage", {}) if loaded is not None else {"reference_bundle_ref": str(reference_bundle), "g23_gate_ref": str(g23_gate)}
        gate = _blocked_gate(reason=f"{type(error).__name__}:{error}", source_lineage=lineage, upstream_gates=upstream_gates)
        _atomic_publish(destination, {"g2.6-gate.json": gate})
        return {"schema_version": S208_RUNNER_SCHEMA, "status": "BLOCKED", "g2_6_gate": gate, "output_files": ["g2.6-gate.json"], "lineage": lineage}


__all__ = ["S208_RUNNER_SCHEMA", "run_s208_g26_production"]
