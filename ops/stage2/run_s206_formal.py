"""Detached S2.6/G2.4b pilot launcher.

The launcher has a strict read-only preflight.  ``--execute`` calls that same
preflight immediately before starting any real formal cell child, binds the
S2.4/S2.5 provider/reference boundary, and dynamically schedules the 24
anchor/B slices over four approved GPUs.  It never creates confirmatory draws
until the blinded reducer and G2.4b matrix qualification have passed.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.experiments.stage2_s206_formal import (
    ANCHOR_IDS,
    APPROVED_GPU_UUIDS,
    CELL_GPU_BINDINGS,
    EXCLUDED_PCI,
    GPU_INVENTORY_SCHEMA,
    PILOT_B_VALUES,
    PILOT_REPETITIONS,
    S206PreflightSpec,
    S206PreparationBlocked,
    GlobalPilotMappingManifest,
    BlindPilotMeasurement,
    build_g24b_gate,
    build_formal_cell_specs,
    build_formal_confirmatory_mapping,
    build_global_pilot_mapping,
    qualify_formal_matrix,
    reduce_blinded_pilot,
    run_formal_pilot_cell,
    strict_preflight,
    normalize_gpu_inventory,
)
from param_importance_nlp.experiments.stage2_s206_delta_consumer import (
    CorrectedDeltaBinding,
    CorrectedDeltaRejected,
    load_bound_corrected_delta,
)
from param_importance_nlp.experiments.stage2_pilot import CostSemantics
from param_importance_nlp.experiments.sampling import SamplingPlan
from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact
from param_importance_nlp.runtime.task_runtime import (
    TaskExecutionRequest,
    TaskRuntimeEnvironment,
)
from param_importance_nlp.runtime.tensor_bundle import load_tensor_bundle


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _production_lpt_jobs() -> tuple[tuple[str, int], ...]:
    """Return the deterministic LPT order for the frozen pilot work units.

    No 31M provider-smoke multiplier is committed yet, so the only approved
    work estimate is the preregistered draw count ``50 * B``.  If a measured
    multiplier is later bound into the formal handoff, it must be added as a
    separately hashed input; this function intentionally does not infer one
    from scientific results or uncommitted timing.
    """

    jobs = tuple(
        (anchor_id, batch_size)
        for anchor_id in ANCHOR_IDS
        for batch_size in PILOT_B_VALUES
    )
    anchor_order = {anchor_id: index for index, anchor_id in enumerate(ANCHOR_IDS)}
    batch_order = {batch_size: index for index, batch_size in enumerate(PILOT_B_VALUES)}
    return tuple(
        sorted(
            jobs,
            key=lambda job: (
                -(PILOT_REPETITIONS * job[1]),
                anchor_order[job[0]],
                batch_order[job[1]],
            ),
        )
    )


def _logical(root: Path, value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S206PreparationBlocked(f"{field}:INVALID_LOGICAL_PATH")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise S206PreparationBlocked(f"{field}:PATH_ESCAPE")
    result = (root / path).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise S206PreparationBlocked(f"{field}:PATH_ESCAPE") from error
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inventory_snapshot(
    path: Path | None,
    *,
    data_root: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load the hash-bound inventory artifact used by every formal path."""

    if path is None:
        raise S206PreparationBlocked("GPU_INVENTORY_JSON_REQUIRED")
    resolved = path.resolve()
    expected_source_ref: str | None = None
    if data_root is not None:
        root = data_root.resolve()
        try:
            expected_source_ref = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise S206PreparationBlocked("GPU_INVENTORY_PATH_OUTSIDE_DATA_ROOT") from error
    try:
        value = load_canonical_json(resolved)
    except (OSError, TypeError, ValueError) as error:
        raise S206PreparationBlocked("GPU_INVENTORY_JSON_INVALID") from error
    if not isinstance(value, Mapping):
        raise S206PreparationBlocked("GPU_INVENTORY_JSON_ENVELOPE_REQUIRED")
    if value.get("schema_version") != GPU_INVENTORY_SCHEMA:
        raise S206PreparationBlocked("GPU_INVENTORY_SCHEMA_INVALID")
    rows = value.get("rows")
    if rows is None:
        # ``gpus`` was the older supported wire key.  It remains accepted only
        # inside the same hash-bound envelope; raw lists are not formal input.
        rows = value.get("gpus")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise S206PreparationBlocked("GPU_INVENTORY_JSON_ROWS_REQUIRED")
    source_ref = value.get("source_ref")
    artifact_hash = value.get("artifact_hash")
    if not isinstance(source_ref, str) or not source_ref:
        raise S206PreparationBlocked("GPU_INVENTORY_SOURCE_REF_REQUIRED")
    if expected_source_ref is not None and source_ref != expected_source_ref:
        raise S206PreparationBlocked("GPU_INVENTORY_SOURCE_REF_PATH_MISMATCH")
    if not isinstance(artifact_hash, str) or len(artifact_hash) != 64 or any(
        char not in "0123456789abcdef" for char in artifact_hash
    ):
        raise S206PreparationBlocked("GPU_INVENTORY_ARTIFACT_HASH_REQUIRED")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(body) != artifact_hash:
        raise S206PreparationBlocked("GPU_INVENTORY_ARTIFACT_HASH_MISMATCH")
    declared_source_sha = value.get("source_sha256")
    source_sha = _file_sha256(resolved)
    if declared_source_sha is not None:
        if not isinstance(declared_source_sha, str) or declared_source_sha != source_sha:
            raise S206PreparationBlocked("GPU_INVENTORY_SOURCE_SHA256_MISMATCH")
    if "compute_apps" not in value:
        raise S206PreparationBlocked("GPU_INVENTORY_COMPUTE_APPS_REQUIRED")
    apps = value["compute_apps"]
    if not isinstance(apps, list) or not all(isinstance(item, Mapping) for item in apps):
        raise S206PreparationBlocked("GPU_INVENTORY_COMPUTE_APPS_INVALID")
    try:
        normalized = normalize_gpu_inventory(rows)
    except (TypeError, ValueError) as error:
        raise S206PreparationBlocked("GPU_INVENTORY_JSON_INVALID") from error
    return normalized, {
        "source_ref": source_ref,
        "artifact_hash": artifact_hash,
        "source_sha256": source_sha,
        "path": str(resolved),
        "schema_version": str(value.get("schema_version", "")),
        "compute_apps": [dict(item) for item in apps],  # type: ignore[list-item]
    }


def _load_inventory(path: Path | None) -> list[dict[str, object]]:
    """Compatibility wrapper returning rows while retaining strict loading."""

    rows, _identity = _load_inventory_snapshot(path)
    return rows


def _write_status(path: Path, payload: Mapping[str, object]) -> None:
    value = {"schema_version": "stage2-s206-formal-detached-status-v1", "updated_at": _now(), **payload}
    write_canonical_json(path, value)


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    inventory, inventory_identity = _load_inventory_snapshot(
        args.gpu_inventory_json,
        data_root=root,
    )
    result = strict_preflight(
        S206PreflightSpec(
            data_root=root,
            s204_root=args.s204_root,
            g23_ref=args.g23_evaluation,
            g24a_ref=args.g24a_evaluation,
        ),
        gpu_inventory=inventory,
        gpu_compute_apps=inventory_identity["compute_apps"] or (),  # type: ignore[arg-type]
    )
    if tuple(result["gpu"]["approved_gpu_uuids"]) != APPROVED_GPU_UUIDS:  # type: ignore[index]
        raise S206PreparationBlocked("APPROVED_GPU_BINDING_DRIFT")
    result["excluded_pci"] = EXCLUDED_PCI
    result["launcher"] = {
        "repository": str(args.repository.resolve()),
        "data_root": str(root),
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "cell_gpu_bindings": dict(CELL_GPU_BINDINGS),
    }
    gpu = result["gpu"]
    if not isinstance(gpu, dict):
        raise S206PreparationBlocked("GPU_INVENTORY_VALIDATION_RESULT_INVALID")
    gpu.update(
        {
            "inventory_source_ref": inventory_identity["source_ref"],
            "inventory_artifact_hash": inventory_identity["artifact_hash"],
            "inventory_source_sha256": inventory_identity["source_sha256"],
            "inventory_path": inventory_identity["path"],
            "inventory_schema_version": inventory_identity["schema_version"] or GPU_INVENTORY_SCHEMA,
        }
    )
    result["gpu_inventory_identity"] = {
        "source_ref": inventory_identity["source_ref"],
        "artifact_hash": inventory_identity["artifact_hash"],
        "source_sha256": inventory_identity["source_sha256"],
        "path": inventory_identity["path"],
    }
    result["gpu_inventory_ref"] = inventory_identity["source_ref"]
    result["gpu_inventory_artifact_hash"] = inventory_identity["artifact_hash"]
    result["gpu_inventory_source_sha256"] = inventory_identity["source_sha256"]
    result["preflight_artifact_hash"] = canonical_json_hash(result)
    return result


def _config_path(root: Path, config_root: str, anchor_id: str, batch_size: int) -> Path:
    component = f"{anchor_id.replace('.', '__')}__b{batch_size:03d}"
    base = _logical(root, config_root, field="config_root")
    candidates = (base / f"{component}.json", base / component / "resolved-config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise S206PreparationBlocked(f"CONFIG_MISSING:{component}")


def _environment_path(root: Path, environment_root: str, anchor_id: str, batch_size: int) -> Path:
    component = f"{anchor_id.replace('.', '__')}__b{batch_size:03d}"
    base = _logical(root, environment_root, field="environment_root")
    candidates = (base / f"{component}.json", base / component / "environment.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise S206PreparationBlocked(f"ENVIRONMENT_MISSING:{component}")


def _result_path(root: Path, result_root: str, anchor_id: str, batch_size: int) -> Path:
    component = f"{anchor_id.replace('.', '__')}__b{batch_size:03d}"
    return _logical(root, result_root, field="result_root") / f"{component}.json"


def _run_anchor(
    *,
    args: argparse.Namespace,
    anchor_id: str,
    gpu_uuid: str,
    status_root: Path,
) -> list[dict[str, object]]:
    root = args.data_root.resolve()
    python = str(args.python)
    records: list[dict[str, object]] = []
    for batch_size in PILOT_B_VALUES:
        config = _config_path(root, args.config_root, anchor_id, batch_size)
        environment = _environment_path(root, args.environment_root, anchor_id, batch_size)
        result = _result_path(root, args.result_root, anchor_id, batch_size)
        result.parent.mkdir(parents=True, exist_ok=True)
        if result.exists():
            try:
                existing = load_canonical_json(result)
            except (OSError, TypeError, ValueError) as error:
                raise S206PreparationBlocked(f"RESULT_NOT_CANONICAL:{result}") from error
            if not isinstance(existing, Mapping) or existing.get("status") not in {"PASS", "COMPLETE"}:
                raise S206PreparationBlocked(f"RESULT_EXISTS_REQUIRES_EXPLICIT_RESUME:{result}")
            records.append({"anchor_id": anchor_id, "batch_size": batch_size, "gpu_uuid": gpu_uuid, "status": "RESUMED_EXISTING", "result_ref": str(result)})
            continue
        env = os.environ.copy()
        env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
                "PYTHONPATH": str(args.repository.resolve() / "src"),
            }
        )
        command = [
            python,
            "-m",
            "param_importance_nlp",
            "task",
            "run",
            "--config",
            str(config),
            "--environment",
            str(environment),
            "--result",
            str(result),
        ]
        log = status_root / f"{anchor_id.replace('.', '__')}__b{batch_size:03d}.log"
        with log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=args.repository.resolve(), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        record = {
            "anchor_id": anchor_id,
            "batch_size": batch_size,
            "gpu_uuid": gpu_uuid,
            "status": "PASS" if completed.returncode == 0 else "FAILED",
            "returncode": completed.returncode,
            "config_ref": str(config),
            "environment_ref": str(environment),
            "result_ref": str(result),
            "log_ref": str(log),
        }
        records.append(record)
        if completed.returncode != 0:
            raise S206PreparationBlocked(f"S206_CELL_FAILED:{anchor_id}:B{batch_size}:{completed.returncode}")
    return records


def _load_formal_execution(args: argparse.Namespace) -> FormalExecutionEvidence:
    """Load the explicit formal authorization consumed by every child cell."""

    if not args.execution_evidence_ref:
        raise S206PreparationBlocked("EXECUTE_REQUIRES_FORMAL_EXECUTION_EVIDENCE")
    path = _logical(args.data_root.resolve(), args.execution_evidence_ref, field="execution_evidence_ref")
    value = load_canonical_json(path)
    if not isinstance(value, Mapping):
        raise S206PreparationBlocked("EXECUTION_EVIDENCE_NOT_OBJECT")
    try:
        execution = FormalExecutionEvidence.from_mapping(dict(value))
        execution.require_for_stage(2)
    except (TypeError, ValueError, S206PreparationBlocked) as error:
        raise S206PreparationBlocked(f"EXECUTION_EVIDENCE_INVALID:{error}") from error
    return execution


def _legacy_execute(args: argparse.Namespace) -> dict[str, object]:
    preflight = _preflight(args)
    execution = _load_formal_execution(args)
    root = args.data_root.resolve()
    operations = _logical(root, args.operations_root, field="operations_root")
    operations.mkdir(parents=True, exist_ok=True)
    output = operations / "status.json"
    if not args.sampling_plan_ref or not args.pilot_mapping_output:
        raise S206PreparationBlocked("EXECUTE_REQUIRES_SAMPLING_PLAN_AND_PILOT_MAPPING_OUTPUT")
    sampling_ref = _logical(root, args.sampling_plan_ref, field="sampling_plan_ref")
    mapping_output = _logical(root, args.pilot_mapping_output, field="pilot_mapping_output")
    sampling_value = load_canonical_json(sampling_ref)
    if not isinstance(sampling_value, Mapping):
        raise S206PreparationBlocked("SAMPLING_PLAN_NOT_OBJECT")
    try:
        sampling = SamplingPlan.from_mapping(sampling_value)
    except (TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"SAMPLING_PLAN_INVALID:{error}") from error
    mapping = build_global_pilot_mapping(sampling)
    if mapping_output.exists():
        existing = load_canonical_json(mapping_output)
        if not isinstance(existing, Mapping) or existing.get("artifact_hash") != mapping.artifact_hash:
            raise S206PreparationBlocked("PILOT_MAPPING_OUTPUT_CONFLICT")
    else:
        mapping_output.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(mapping_output, mapping.to_dict())
    cell_plan_dir = operations / "cell-plans"
    cell_specs = build_formal_cell_specs(
        mapping,
        execution,
        config_root=args.config_root,
        environment_root=args.environment_root,
        result_root=args.result_root,
    )
    cell_plan_refs: list[str] = []
    for spec in cell_specs:
        plan_path = cell_plan_dir / f"{spec.cell_id}.json"
        plan_payload = spec.to_dict()
        if plan_path.exists():
            if load_canonical_json(plan_path) != plan_payload:
                raise S206PreparationBlocked(f"CELL_PLAN_CONFLICT:{spec.cell_id}")
        else:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            write_canonical_json(plan_path, plan_payload)
        cell_plan_refs.append(str(plan_path))
    _write_status(output, {"run_id": args.run_id, "stage": "PREFLIGHT_PASS", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": []})
    records: list[dict[str, object]] = []
    try:
        # Wave A occupies the four approved cards.  Wave B starts only after
        # all Wave A cells have committed, preventing same-card concurrency.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_run_anchor, args=args, anchor_id=anchor_id, gpu_uuid=CELL_GPU_BINDINGS[anchor_id], status_root=operations): anchor_id
                for anchor_id in ANCHOR_IDS[:4]
            }
            for future in as_completed(futures):
                records.extend(future.result())
                _write_status(output, {"run_id": args.run_id, "stage": "WAVE_A_RUNNING", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records})
        _write_status(output, {"run_id": args.run_id, "stage": "WAVE_A_COMPLETE", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_run_anchor, args=args, anchor_id=anchor_id, gpu_uuid=CELL_GPU_BINDINGS[anchor_id], status_root=operations): anchor_id
                for anchor_id in ANCHOR_IDS[4:]
            }
            for future in as_completed(futures):
                records.extend(future.result())
                _write_status(output, {"run_id": args.run_id, "stage": "WAVE_B_RUNNING", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records})
    except Exception as error:
        _write_status(output, {"run_id": args.run_id, "stage": "BLOCKED_EXECUTION", "formal_eligible": False, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records, "reason": f"{type(error).__name__}:{error}"})
        raise
    _write_status(output, {"run_id": args.run_id, "stage": "PILOT_COMPLETE", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records, "expected_cell_count": 24, "confirmatory_draws_generated": False})
    return load_canonical_json(output)  # type: ignore[return-value]


def _absolute_or_logical(root: Path, value: object, *, field: str) -> Path:
    """Resolve a DATA_ROOT reference while accepting S2.4 absolute status refs."""

    if not isinstance(value, str) or not value:
        raise S206PreparationBlocked(f"{field}:PATH_REQUIRED")
    path = Path(value)
    if path.is_absolute():
        try:
            root_anchor = root.resolve()
            _reject_symlink_components(root, path, field=field)
            relative = path.absolute().relative_to(root_anchor).as_posix()
        except (OSError, ValueError) as error:
            raise S206PreparationBlocked(f"{field}:ABSOLUTE_PATH_OUTSIDE_DATA_ROOT") from error
        return _logical(root_anchor, relative, field=field)
    _reject_symlink_components(root, path, field=field)
    return _logical(root, value, field=field)


def _reject_symlink_components(root: Path, value: Path, *, field: str) -> None:
    """Reject symlink traversal for a DATA_ROOT-relative reference.

    Tensor bundles are directories rather than regular files.  Resolving a
    candidate before checking it would silently make a symlinked bundle look
    like an in-root directory, so inspect the lexical path first and reject
    every symlink component.  This applies to all production references, not
    just the bundle itself, so a handoff cannot redirect the provider through
    an unbound alias.
    """

    if root.is_symlink():
        raise S206PreparationBlocked(f"{field}:DATA_ROOT_SYMLINK_FORBIDDEN")
    root_anchor = root.resolve()
    candidate = value if value.is_absolute() else root_anchor / value
    candidate_absolute = candidate.absolute()
    try:
        relative = candidate_absolute.relative_to(root_anchor)
    except ValueError as error:
        raise S206PreparationBlocked(f"{field}:PATH_ESCAPE") from error
    if ".." in relative.parts:
        raise S206PreparationBlocked(f"{field}:PATH_ESCAPE")
    current = root_anchor
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise S206PreparationBlocked(f"{field}:SYMLINK_COMPONENT_FORBIDDEN")


def _derive_s206_config(source_config: ResolvedConfigV2) -> ResolvedConfigV2:
    """Retarget the S2.4 v2 wire object without dropping formal bindings.

    ``ResolvedConfigV2.resolve(source_config.base_config, ...)`` regenerates
    defaults and therefore discards the S2.4 runner's resolved providers,
    orchestration inputs, and execution overrides.  The real provider boundary
    needs those exact values.  Start from the complete S2.4 wire object and
    change only the task identity, runner kind, and output artifact contract;
    all provider/asset/evidence-relevant sections remain byte-for-byte equal
    after strict normalization.
    """

    source_wire = source_config.to_dict()
    payload = {
        key: value
        for key, value in source_wire.items()
        if key not in {"config_hash", "full_hash"}
    }
    payload["task_id"] = "stage2.06_pilot_and_matrix_freeze"
    execution = payload.get("execution")
    artifacts = payload.get("artifacts")
    if not isinstance(execution, Mapping) or not isinstance(artifacts, Mapping):
        raise S206PreparationBlocked("S204_CONFIG_SECTIONS_MISSING")
    execution_wire = dict(execution)
    execution_wire["runner_kind"] = "pilot"
    payload["execution"] = execution_wire
    artifacts_wire = dict(artifacts)
    artifacts_wire["required_kinds"] = list(
        DEFAULT_TASK_CATALOG.get("stage2.06_pilot_and_matrix_freeze").artifact_kinds
    )
    payload["artifacts"] = artifacts_wire
    try:
        derived = ResolvedConfigV2(payload)
    except (TypeError, ValueError) as error:
        raise S206PreparationBlocked("S206_CONFIG_DERIVATION_FAILED") from error

    derived_wire = derived.to_dict()
    preserved_sections = set(source_wire) - {
        "task_id",
        "execution",
        "artifacts",
        "config_hash",
        "full_hash",
    }
    for section in preserved_sections:
        if derived_wire.get(section) != source_wire.get(section):
            raise S206PreparationBlocked(
                f"S206_CONFIG_BINDING_DROPPED:{section}"
            )
    derived_execution = derived_wire.get("execution")
    source_execution = source_wire.get("execution")
    if not isinstance(derived_execution, Mapping) or not isinstance(source_execution, Mapping):
        raise S206PreparationBlocked("S206_CONFIG_EXECUTION_INVALID")
    for field in set(source_execution) - {"runner_kind"}:
        if derived_execution.get(field) != source_execution.get(field):
            raise S206PreparationBlocked(f"S206_CONFIG_EXECUTION_OVERRIDE_DROPPED:{field}")
    derived_artifacts = derived_wire.get("artifacts")
    source_artifacts = source_wire.get("artifacts")
    if not isinstance(derived_artifacts, Mapping) or not isinstance(source_artifacts, Mapping):
        raise S206PreparationBlocked("S206_CONFIG_ARTIFACTS_INVALID")
    for field in set(source_artifacts) - {"required_kinds"}:
        if derived_artifacts.get(field) != source_artifacts.get(field):
            raise S206PreparationBlocked(f"S206_CONFIG_ARTIFACT_OVERRIDE_DROPPED:{field}")
    return derived


def _load_object_ref(root: Path, value: object, *, field: str) -> tuple[Path, dict[str, object]]:
    path = _absolute_or_logical(root, value, field=field)
    try:
        payload = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"{field}:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(payload, Mapping):
        raise S206PreparationBlocked(f"{field}:OBJECT_REQUIRED")
    return path, dict(payload)


def _load_s205_rebind(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the already-produced S2.5 six-cell handoff, without editing it."""

    if not args.s205_rebind_ref:
        raise S206PreparationBlocked("PRODUCTION_REQUIRES_S205_REBIND_PLAN")
    root = args.data_root.resolve()
    _path, plan = _load_object_ref(root, args.s205_rebind_ref, field="s205_rebind_ref")
    if plan.get("schema_version") != "stage2-s205-rebind-plan-v1" or plan.get("status") != "READY" or plan.get("formal_eligible") is not True:
        raise S206PreparationBlocked("S205_REBIND_READY_FORMAL_REQUIRED")
    rows = plan.get("cells")
    if not isinstance(rows, list) or len(rows) != len(ANCHOR_IDS):
        raise S206PreparationBlocked("S205_REBIND_SIX_CELL_ROWS_REQUIRED")
    by_anchor: dict[str, dict[str, object]] = {}
    for expected, raw in zip(ANCHOR_IDS, rows):
        g23_cell_id = expected.replace(".", ":", 1)
        if not isinstance(raw, Mapping) or raw.get("cell_id") != g23_cell_id:
            raise S206PreparationBlocked("S205_REBIND_ANCHOR_ORDER_INVALID")
        row = dict(raw)
        for name in (
            "config_ref",
            "environment_ref",
            "formal_execution_ref",
            "reference_artifact_refs",
            "task_result_status_path",
            "config_hash",
            "result_hash",
        ):
            if name not in row:
                raise S206PreparationBlocked(f"S205_REBIND_{name.upper()}_MISSING:{expected}")
        for name in ("config_hash", "result_hash"):
            if not isinstance(row[name], str) or len(row[name]) != 64 or any(char not in "0123456789abcdef" for char in row[name]):
                raise S206PreparationBlocked(f"S205_REBIND_{name.upper()}_INVALID:{expected}")
        if not isinstance(row["reference_artifact_refs"], Mapping):
            raise S206PreparationBlocked(f"S205_REBIND_REFERENCE_REFS_INVALID:{expected}")
        by_anchor[expected] = row
    return plan, by_anchor


def _load_g24a_metrics(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """Return per-anchor G2.4a metrics used only for reference precision inputs."""

    root = args.data_root.resolve()
    _path, payload = _load_object_ref(root, args.g24a_evaluation, field="g24a_evaluation")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(ANCHOR_IDS):
        raise S206PreparationBlocked("G24A_RESULTS_SIX_CELL_REQUIRED")
    by_anchor: dict[str, dict[str, object]] = {}
    for raw in results:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("cell_id"), str):
            raise S206PreparationBlocked("G24A_RESULT_CELL_INVALID")
        g23_cell_id = str(raw["cell_id"])
        expected_g23_ids = {anchor.replace(".", ":", 1): anchor for anchor in ANCHOR_IDS}
        if g23_cell_id not in expected_g23_ids:
            raise S206PreparationBlocked("G24A_RESULT_CELL_SET_INVALID")
        cell_id = expected_g23_ids[g23_cell_id]
        if cell_id in by_anchor:
            raise S206PreparationBlocked("G24A_RESULT_CELL_SET_INVALID")
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            # A PASS status without output-derived precision metrics is enough
            # for the upstream gate, but not enough to size a new S2.6 pilot.
            raise S206PreparationBlocked(f"G24A_OUTPUT_DERIVED_METRICS_MISSING:{cell_id}")
        by_anchor[cell_id] = dict(raw)
    if tuple(by_anchor) != ANCHOR_IDS:
        raise S206PreparationBlocked("G24A_RESULT_ANCHOR_ORDER_INVALID")
    return by_anchor


def _validate_s204_reference_candidate(
    reference_payload: Mapping[str, object],
    convergence_payload: Mapping[str, object],
    *,
    anchor_id: str,
) -> None:
    """Accept the qualified task commit's still-unqualified S2.4 candidate.

    ``load_committed_task_artifact(require_formal=True)`` checks the immutable
    task envelope, while this function checks the payload semantics.  The
    payload must stay formal-scope and candidate-only; G2.3/G2.4a and S2.5
    provide qualification without this adapter rewriting the source artifact.
    """

    if (
        reference_payload.get("schema_version") != "reference-result-v1"
        or convergence_payload.get("schema_version") != "stage2-reference-convergence-report-v1"
        or reference_payload.get("scope") != "formal"
        or reference_payload.get("formal_eligible") is not False
    ):
        raise S206PreparationBlocked(f"S204_REFERENCE_SCHEMA_INVALID:{anchor_id}")
    metadata = reference_payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("qualification_gate_hash") is not None
    ):
        raise S206PreparationBlocked(f"S204_REFERENCE_CANDIDATE_SEMANTICS_INVALID:{anchor_id}")


def _load_production_cell_inputs(
    args: argparse.Namespace,
    *,
    anchor_id: str,
    batch_size: int,
    s205_rows: Mapping[str, Mapping[str, object]],
    g24a_rows: Mapping[str, Mapping[str, object]],
    execution: FormalExecutionEvidence,
) -> tuple[object, Mapping[str, object], Mapping[str, Mapping[str, object]], str, dict[str, float], dict[str, float], CorrectedDeltaBinding]:
    """Bind one S2.4 config/environment/reference to the real formal provider."""

    root = args.data_root.resolve()
    row = s205_rows.get(anchor_id)
    if row is None:
        raise S206PreparationBlocked(f"S205_ROW_MISSING:{anchor_id}")
    if not isinstance(row.get("config_hash"), str) or not isinstance(row.get("result_hash"), str):
        raise S206PreparationBlocked(f"S205_REBIND_CELL_IDENTITIES_MISSING:{anchor_id}")
    _status_path, final_status = _load_object_ref(
        root,
        row["task_result_status_path"],
        field=f"{anchor_id}.task_result_status_path",
    )
    if (
        final_status.get("schema_version") != "stage2-s204-cell-final-status-v3"
        or final_status.get("status") != "COMPLETE"
        or final_status.get("formal_eligible") is not True
        or final_status.get("cell_id") != anchor_id.replace(".", ":", 1)
        or final_status.get("task_result_hash") != row["result_hash"]
        or final_status.get("config_hash") != row["config_hash"]
    ):
        raise S206PreparationBlocked(f"S204_FINAL_STATUS_RESULT_BINDING_INVALID:{anchor_id}")
    if canonical_json_hash({key: item for key, item in final_status.items() if key != "artifact_hash"}) != final_status.get("artifact_hash"):
        raise S206PreparationBlocked(f"S204_FINAL_STATUS_HASH_INVALID:{anchor_id}")
    config_path, config_wire = _load_object_ref(root, row["config_ref"], field=f"{anchor_id}.config_ref")
    try:
        source_config = ResolvedConfigV2.from_mapping(config_wire)
    except (TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"S204_CONFIG_INVALID:{anchor_id}") from error
    if source_config.task_id != "stage2.04_reference_target":
        raise S206PreparationBlocked(f"S204_CONFIG_TASK_INVALID:{anchor_id}")
    if row.get("config_hash") is not None and row.get("config_hash") != source_config.config_hash:
        raise S206PreparationBlocked(f"S204_CONFIG_HASH_MISMATCH:{anchor_id}")
    try:
        config = _derive_s206_config(source_config)
    except S206PreparationBlocked as error:
        raise S206PreparationBlocked(f"{error}:{anchor_id}") from error
    if config.run_intent != "formal":
        raise S206PreparationBlocked(f"S206_CONFIG_FORMAL_REQUIRED:{anchor_id}")

    _env_path, env_wire = _load_object_ref(root, row["environment_ref"], field=f"{anchor_id}.environment_ref")
    try:
        source_environment = TaskRuntimeEnvironment.from_mapping(env_wire)
    except (TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"S204_ENVIRONMENT_INVALID:{anchor_id}") from error
    evidence_refs = dict(source_environment.evidence_refs)
    evidence_refs["formal_execution"] = str(args.execution_evidence_ref)
    environment = TaskRuntimeEnvironment(
        capabilities=source_environment.capabilities | frozenset({"server", "cuda", "model_assets", "data_assets"}),
        frozen_contract_stages=source_environment.frozen_contract_stages | frozenset({2}),
        passed_gate_ids=source_environment.passed_gate_ids | frozenset({"stage2.G2.2", "stage2.G2.3", "stage2.G2.4a"}),
        estimator_decision_ref=source_environment.estimator_decision_ref,
        evidence_refs=evidence_refs,
    )
    task = DEFAULT_TASK_CATALOG.get("stage2.06_pilot_and_matrix_freeze")
    request = TaskExecutionRequest(config=config, task=task, environment=environment)
    try:
        # This private adapter is the existing production boundary used by
        # S2.4.  Calling it here preserves FormalG3RuntimeAssets and
        # TorchFixedStateGradientProvider construction without changing the
        # shared provider or TaskRuntime implementation.
        from param_importance_nlp.experiments.stage23_task_runners import _formal_provider

        context = _formal_provider(request, root)
    except Exception as error:
        raise S206PreparationBlocked(f"FORMAL_PROVIDER_BIND_FAILED:{anchor_id}:{type(error).__name__}:{error}") from error
    if context.evidence.artifact_hash != execution.artifact_hash:
        raise S206PreparationBlocked(f"FORMAL_EXECUTION_CONTEXT_HASH_DRIFT:{anchor_id}")

    artifact_refs = row["reference_artifact_refs"]
    assert isinstance(artifact_refs, Mapping)
    reference_ref = artifact_refs.get("reference_result")
    convergence_ref = artifact_refs.get("reference_convergence_report")
    if not isinstance(reference_ref, str) or not isinstance(convergence_ref, str):
        raise S206PreparationBlocked(f"S204_REFERENCE_REFS_MISSING:{anchor_id}")
    try:
        reference_loaded = load_committed_task_artifact(root, reference_ref, require_formal=True)
        convergence_loaded = load_committed_task_artifact(root, convergence_ref, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"S204_REFERENCE_ARTIFACT_LOAD_FAILED:{anchor_id}") from error
    if (
        reference_loaded.identity.task_id != "stage2.04_reference_target"
        or reference_loaded.identity.artifact_kind != "reference_result"
        or convergence_loaded.identity.task_id != "stage2.04_reference_target"
        or convergence_loaded.identity.artifact_kind != "reference_convergence_report"
        or reference_loaded.identity.config_hash != source_config.config_hash
        or convergence_loaded.identity.config_hash != source_config.config_hash
        or reference_loaded.identity.formal_eligible is not True
        or convergence_loaded.identity.formal_eligible is not True
    ):
        raise S206PreparationBlocked(f"S204_REFERENCE_ARTIFACT_IDENTITY_INVALID:{anchor_id}")
    reference_payload = dict(reference_loaded.payload)
    convergence_payload = dict(convergence_loaded.payload)
    _validate_s204_reference_candidate(
        reference_payload,
        convergence_payload,
        anchor_id=anchor_id,
    )
    try:
        from param_importance_nlp.contracts.artifacts import validate_reference_result_artifact
        validate_reference_result_artifact(reference_payload)
    except (TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"S204_REFERENCE_MANIFEST_INVALID:{anchor_id}") from error
    artifacts = config.section("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("output_dir"), str):
        raise S206PreparationBlocked(f"S204_ARTIFACT_OUTPUT_DIR_MISSING:{anchor_id}")
    bundle_ref = reference_payload.get("tensor_bundle_ref")
    if not isinstance(bundle_ref, str) or not bundle_ref:
        raise S206PreparationBlocked(f"S204_REFERENCE_BUNDLE_REF_MISSING:{anchor_id}")
    task_output_root = _absolute_or_logical(root, artifacts["output_dir"], field=f"{anchor_id}.artifacts.output_dir")
    bundle_candidates = [
        _absolute_or_logical(root, bundle_ref, field=f"{anchor_id}.tensor_bundle_ref"),
        _absolute_or_logical(task_output_root, bundle_ref, field=f"{anchor_id}.tensor_bundle_ref_relative"),
    ]
    # ``runtime.tensor-bundle.v1`` is a directory bundle.  A regular-file
    # check would reject every real S2.4 reference and previously left the
    # production bridge effectively fixture-only.
    bundle_path = next((candidate for candidate in bundle_candidates if candidate.is_dir()), None)
    if bundle_path is None:
        raise S206PreparationBlocked(f"S204_REFERENCE_BUNDLE_MISSING:{anchor_id}")
    if bundle_path.is_symlink():
        raise S206PreparationBlocked(f"S204_REFERENCE_BUNDLE_SYMLINK_FORBIDDEN:{anchor_id}")
    try:
        bundle_state, bundle = load_tensor_bundle(bundle_path)
    except (OSError, TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"S204_REFERENCE_BUNDLE_INVALID:{anchor_id}") from error
    if not isinstance(bundle_state, Mapping) or bundle.manifest_sha256 != reference_payload.get("tensor_bundle_manifest_hash"):
        raise S206PreparationBlocked(f"S204_REFERENCE_BUNDLE_HASH_MISMATCH:{anchor_id}")
    vectors: dict[str, Mapping[str, object]] = {}
    for name in ("bias_reference", "cross_reference", "ranking_reference"):
        vector = bundle_state.get(name)
        declared = reference_payload.get(f"{name}_hash")
        if not isinstance(vector, Mapping) or not isinstance(declared, str):
            raise S206PreparationBlocked(f"S204_REFERENCE_VECTOR_MISSING:{anchor_id}:{name}")
        from param_importance_nlp.experiments.stage2_formal import _vector_digest
        if _vector_digest(vector) != declared:
            raise S206PreparationBlocked(f"S204_REFERENCE_VECTOR_HASH_MISMATCH:{anchor_id}:{name}")
        vectors[name] = vector
    metadata = reference_payload.get("metadata")
    sequence_variance = bundle_state.get("sequence_variance")
    if not isinstance(metadata, Mapping) or not isinstance(sequence_variance, Mapping):
        raise S206PreparationBlocked(f"S204_REFERENCE_SEQUENCE_VARIANCE_MISSING:{anchor_id}")
    sequence_hash = metadata.get("sequence_variance_hash")
    if not isinstance(sequence_hash, str) or _vector_digest(sequence_variance) != sequence_hash:
        raise S206PreparationBlocked(f"S204_REFERENCE_SEQUENCE_VARIANCE_HASH_MISMATCH:{anchor_id}")

    # G2.4a is the authoritative output-derived precision check.  Its three
    # endpoint half-widths are bound to the same cell; delta_sci is reloaded
    # from S2.4's independent reference_sizing-derived convergence artifact.
    g24a_row = g24a_rows.get(anchor_id)
    if g24a_row is None or not isinstance(g24a_row.get("metrics"), Mapping):
        raise S206PreparationBlocked(f"G24A_METRICS_MISSING:{anchor_id}")
    metrics = g24a_row["metrics"]
    assert isinstance(metrics, Mapping)
    half_width = {
        "bias": metrics.get("h_ref_model_total"),
        "nmse": metrics.get("h_ref_layer"),
        "rank": metrics.get("h_ref_module"),
    }
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0 for value in half_width.values()):
        raise S206PreparationBlocked(f"G24A_REFERENCE_HALF_WIDTH_INVALID:{anchor_id}")
    sizing_plan = convergence_payload.get("sizing_plan")
    expected_sizing_plan_hash = convergence_payload.get("sizing_plan_artifact_hash")
    if expected_sizing_plan_hash is None and isinstance(sizing_plan, Mapping):
        expected_sizing_plan_hash = sizing_plan.get("artifact_hash")
    expected_sizing_result_hash = convergence_payload.get("sizing_result_hash")
    expected_reference_id = sizing_plan.get("reference_id") if isinstance(sizing_plan, Mapping) else None
    registry_artifact = convergence_payload.get("parameter_registry_artifact")
    expected_registry_hash = convergence_payload.get("registry_hash")
    if expected_registry_hash is None and isinstance(registry_artifact, Mapping):
        expected_registry_hash = registry_artifact.get("registry_hash")
    try:
        corrected = load_bound_corrected_delta(
            root,
            g23_evaluation_ref=args.g23_evaluation,
            cell_id=anchor_id,
            expected_config_hash=str(row["config_hash"]),
            expected_result_hash=str(row["result_hash"]),
            expected_sizing_plan_hash=(str(expected_sizing_plan_hash) if expected_sizing_plan_hash is not None else None),
            expected_sizing_result_hash=(str(expected_sizing_result_hash) if expected_sizing_result_hash is not None else None),
            expected_reference_id=(str(expected_reference_id) if expected_reference_id is not None else None),
            expected_registry_hash=(str(expected_registry_hash) if expected_registry_hash is not None else None),
        )
        delta = corrected.delta_for(batch_size)
    except CorrectedDeltaRejected as error:
        raise S206PreparationBlocked(f"G23_CORRECTED_DELTA_INVALID:{anchor_id}:{error}") from error
    half_width = {name: float(value) for name, value in half_width.items()}
    if any(half_width[name] > delta[name] / 4.0 for name in delta):
        raise S206PreparationBlocked(f"G24A_REFERENCE_PRECISION_NOT_BOUND:{anchor_id}:B{batch_size}")
    from param_importance_nlp.experiments.stage2_formal import _vector_digest
    return (
        context.provider,
        vectors["bias_reference"],
        {
            "bias": vectors["bias_reference"],
            "cross": vectors["cross_reference"],
            "ranking": vectors["ranking_reference"],
        },
        _vector_digest(vectors["bias_reference"]),
        delta,
        half_width,
        corrected,
    )


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    """Publish a canonical S2.6 object idempotently; never overwrite drift."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_canonical_json(path)
        if existing != dict(payload):
            raise S206PreparationBlocked(f"S206_OUTPUT_CONFLICT:{path}")
        return
    write_canonical_json(path, dict(payload))


def _production_cell(args: argparse.Namespace) -> dict[str, object]:
    """Execute exactly one real formal pilot cell through the paired runner."""

    if args.cell_anchor not in ANCHOR_IDS or args.cell_batch_size not in PILOT_B_VALUES:
        raise S206PreparationBlocked("PRODUCTION_CELL_ANCHOR_OR_BATCH_INVALID")
    if args.cell_gpu_uuid not in APPROVED_GPU_UUIDS:
        raise S206PreparationBlocked("PRODUCTION_CELL_GPU_NOT_APPROVED")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_gpu != args.cell_gpu_uuid:
        raise S206PreparationBlocked("PRODUCTION_CELL_GPU_BINDING_MISMATCH")
    required = (
        args.pilot_mapping_ref,
        args.s205_rebind_ref,
        args.cell_artifact_root,
        args.cell_output,
        args.execution_evidence_ref,
    )
    if not all(required):
        raise S206PreparationBlocked("PRODUCTION_CELL_REQUIRES_MAPPING_REBIND_ARTIFACT_OUTPUT_AND_EXECUTION")
    preflight = _preflight(args)
    root = args.data_root.resolve()
    mapping_path = _absolute_or_logical(root, args.pilot_mapping_ref, field="pilot_mapping_ref")
    mapping_value = load_canonical_json(mapping_path)
    if not isinstance(mapping_value, Mapping):
        raise S206PreparationBlocked("PRODUCTION_CELL_MAPPING_NOT_OBJECT")
    mapping = GlobalPilotMappingManifest.from_mapping(dict(mapping_value))
    execution = _load_formal_execution(args)
    s205_plan, s205_rows = _load_s205_rebind(args)
    g24a_rows = _load_g24a_metrics(args)
    cell = next(
        (
            item
            for item in mapping.cells
            if item.anchor_id == args.cell_anchor
            and item.batch_size == args.cell_batch_size
        ),
        None,
    )
    if cell is None:
        raise S206PreparationBlocked("PRODUCTION_CELL_NOT_IN_MAPPING")
    output = _absolute_or_logical(root, args.cell_output, field="cell_output")
    from param_importance_nlp.experiments.stage2_s206_formal import FormalPilotCellRun

    # Recovery first validates the immutable blinded envelope and mapping
    # hash; it need not reload a model/provider or consume any draw for a
    # previously committed cell.
    if output.exists():
        existing = load_canonical_json(output)
        if not isinstance(existing, Mapping):
            raise S206PreparationBlocked("PRODUCTION_CELL_OUTPUT_NOT_OBJECT")
        resumed = FormalPilotCellRun.from_mapping(dict(existing))
        if resumed.mapping_artifact_hash != mapping.artifact_hash:
            raise S206PreparationBlocked("PRODUCTION_CELL_OUTPUT_MAPPING_HASH_CONFLICT")
        try:
            corrected_resume = load_bound_corrected_delta(
                root,
                g23_evaluation_ref=args.g23_evaluation,
                cell_id=args.cell_anchor,
                expected_config_hash=str(s205_rows[args.cell_anchor]["config_hash"]),
                expected_result_hash=str(s205_rows[args.cell_anchor]["result_hash"]),
            )
        except CorrectedDeltaRejected as error:
            raise S206PreparationBlocked(f"G23_CORRECTED_DELTA_RESUME_INVALID:{args.cell_anchor}:{error}") from error
        if (
            resumed.corrected_delta_sci_hash != corrected_resume.artifact_hash
            or resumed.corrected_delta_sci_ref != corrected_resume.ref
            or resumed.corrected_delta_sci_cell_id != corrected_resume.cell_id
            or resumed.corrected_delta_sci_config_hash != corrected_resume.config_hash
            or resumed.corrected_delta_sci_result_hash != corrected_resume.result_hash
        ):
            raise S206PreparationBlocked("PRODUCTION_CELL_OUTPUT_CORRECTED_DELTA_BINDING_CONFLICT")
        return {
            "status": "RESUMED_EXISTING",
            "cell": resumed.to_dict(),
            "preflight": preflight,
            "s205_rebind_hash": canonical_json_hash(s205_plan),
        }
    provider, reference, references, reference_hash, delta, half_width, corrected_delta = _load_production_cell_inputs(
        args,
        anchor_id=args.cell_anchor,
        batch_size=args.cell_batch_size,
        s205_rows=s205_rows,
        g24a_rows=g24a_rows,
        execution=execution,
    )
    artifact_root = _absolute_or_logical(root, args.cell_artifact_root, field="cell_artifact_root")
    try:
        result = run_formal_pilot_cell(
            cell,
            mapping=mapping,
            provider=provider,  # type: ignore[arg-type]
            execution=execution,
            reference=reference,
            references=references,
            reference_hash=reference_hash,
            artifact_root=artifact_root,
            delta_sci_by_endpoint=delta,
            corrected_delta_sci_hash=corrected_delta.artifact_hash,
            corrected_delta_sci_ref=corrected_delta.ref,
            corrected_delta_sci_batch_sizes=corrected_delta.batch_sizes,
            delta_sci_source=corrected_delta.source,
            corrected_delta_sci_cell_id=corrected_delta.cell_id,
            corrected_delta_sci_config_hash=corrected_delta.config_hash,
            corrected_delta_sci_result_hash=corrected_delta.result_hash,
            reference_half_width_by_endpoint=half_width,
            resource_within_budget=bool(args.resource_within_budget),
            cost_io_quiescent=bool(args.cost_io_quiescent),
        )
    except Exception as error:
        raise S206PreparationBlocked(
            f"PRODUCTION_CELL_RUN_FAILED:{args.cell_anchor}:B{args.cell_batch_size}:"
            f"{type(error).__name__}:{error}"
        ) from error
    payload = result.to_dict()
    _write_once(output, payload)
    return {
        "status": "COMPLETE",
        "cell": payload,
        "preflight": preflight,
        "anchor_id": args.cell_anchor,
        "batch_size": args.cell_batch_size,
        "mapping_hash": mapping.artifact_hash,
        "s205_rebind_hash": canonical_json_hash(s205_plan),
    }


def _production_smoke_cell(args: argparse.Namespace) -> dict[str, object]:
    """Run one real 14M pilot repetition as a bounded provider smoke test.

    This path deliberately stops at the recoverable paired-wave boundary.  It
    does not emit a ``FormalPilotCellRun`` (which requires all 50 repetitions),
    does not call the blinded reducer, and cannot create a confirmatory draw.
    The output is an operational smoke receipt containing only hashes and
    costs, so it is not admissible as pilot evidence or a matrix qualification.
    """

    if args.cell_anchor != ANCHOR_IDS[0]:
        raise S206PreparationBlocked("PRODUCTION_SMOKE_REQUIRES_14M_ANCHOR")
    if args.cell_batch_size not in PILOT_B_VALUES:
        raise S206PreparationBlocked("PRODUCTION_SMOKE_BATCH_INVALID")
    if args.cell_gpu_uuid not in APPROVED_GPU_UUIDS:
        raise S206PreparationBlocked("PRODUCTION_SMOKE_GPU_NOT_APPROVED")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_gpu != args.cell_gpu_uuid:
        raise S206PreparationBlocked("PRODUCTION_SMOKE_GPU_BINDING_MISMATCH")
    required = (
        args.pilot_mapping_ref,
        args.s205_rebind_ref,
        args.cell_artifact_root,
        args.cell_output,
        args.execution_evidence_ref,
    )
    if not all(required):
        raise S206PreparationBlocked("PRODUCTION_SMOKE_REQUIRES_MAPPING_REBIND_ARTIFACT_OUTPUT_AND_EXECUTION")

    preflight = _preflight(args)
    root = args.data_root.resolve()
    mapping_path = _absolute_or_logical(root, args.pilot_mapping_ref, field="pilot_mapping_ref")
    mapping_value = load_canonical_json(mapping_path)
    if not isinstance(mapping_value, Mapping):
        raise S206PreparationBlocked("PRODUCTION_SMOKE_MAPPING_NOT_OBJECT")
    mapping = GlobalPilotMappingManifest.from_mapping(dict(mapping_value))
    execution = _load_formal_execution(args)
    s205_plan, s205_rows = _load_s205_rebind(args)
    g24a_rows = _load_g24a_metrics(args)
    cell = next(
        (
            item
            for item in mapping.cells
            if item.anchor_id == args.cell_anchor
            and item.batch_size == args.cell_batch_size
        ),
        None,
    )
    if cell is None:
        raise S206PreparationBlocked("PRODUCTION_SMOKE_CELL_NOT_IN_MAPPING")
    if not cell.mappings:
        raise S206PreparationBlocked("PRODUCTION_SMOKE_CELL_HAS_NO_MAPPING")
    output = _absolute_or_logical(root, args.cell_output, field="cell_output")
    smoke_mapping = cell.mappings[0]
    smoke_mapping_hash = canonical_json_hash(smoke_mapping.to_dict())

    # A committed receipt is sufficient recovery for a completed smoke.  Do
    # not reload the model/provider or consume another repetition on resume.
    try:
        corrected_smoke = load_bound_corrected_delta(
            root,
            g23_evaluation_ref=args.g23_evaluation,
            cell_id=args.cell_anchor,
            expected_config_hash=str(s205_rows[args.cell_anchor]["config_hash"]),
            expected_result_hash=str(s205_rows[args.cell_anchor]["result_hash"]),
        )
    except CorrectedDeltaRejected as error:
        raise S206PreparationBlocked(f"G23_CORRECTED_DELTA_SMOKE_INVALID:{args.cell_anchor}:{error}") from error
    if output.exists():
        existing = load_canonical_json(output)
        if not isinstance(existing, Mapping):
            raise S206PreparationBlocked("PRODUCTION_SMOKE_OUTPUT_NOT_OBJECT")
        required_existing = {
            "schema_version",
            "scope",
            "formal_eligible",
            "qualification_gate_hash",
            "smoke_only",
            "scientific_values_masked",
            "confirmatory_draws_generated",
            "mapping_id",
            "mapping_artifact_hash",
            "mapping_cell_hash",
            "anchor_id",
            "batch_size",
            "repetitions_requested",
            "completed_repetitions",
            "unit_ids",
            "summary_hash",
            "registry_hash",
            "reference_hash",
            "reference_hashes",
            "cost_statistics",
            "cost_io_quiescent",
            "pilot_mapping_ref",
            "execution_evidence_hash",
            "s205_rebind_hash",
            "corrected_delta_sci_hash",
            "corrected_delta_sci_ref",
            "corrected_delta_sci_batch_sizes",
            "delta_sci_source",
            "artifact_hash",
        }
        if set(existing) != required_existing:
            raise S206PreparationBlocked("PRODUCTION_SMOKE_OUTPUT_FIELDS_INVALID")
        if (
            existing.get("schema_version") != "stage2-s206-formal-smoke-v1"
            or existing.get("scope") != "formal"
            or existing.get("formal_eligible") is not False
            or existing.get("qualification_gate_hash") is not None
            or existing.get("smoke_only") is not True
            or existing.get("scientific_values_masked") is not True
            or existing.get("confirmatory_draws_generated") is not False
            or existing.get("mapping_id") != mapping.mapping_id
            or existing.get("mapping_artifact_hash") != mapping.artifact_hash
            or existing.get("mapping_cell_hash") != smoke_mapping_hash
            or existing.get("anchor_id") != args.cell_anchor
            or existing.get("batch_size") != args.cell_batch_size
            or existing.get("repetitions_requested") != 1
            or existing.get("completed_repetitions") != 1
            or existing.get("unit_ids") != [smoke_mapping.repetition_id]
            or existing.get("corrected_delta_sci_hash") != corrected_smoke.artifact_hash
            or existing.get("corrected_delta_sci_ref") != corrected_smoke.ref
            or existing.get("corrected_delta_sci_batch_sizes") != list(corrected_smoke.batch_sizes)
            or existing.get("delta_sci_source") != corrected_smoke.source
        ):
            raise S206PreparationBlocked("PRODUCTION_SMOKE_OUTPUT_BINDING_DRIFT")
        if existing.get("artifact_hash") != canonical_json_hash(
            {key: value for key, value in existing.items() if key != "artifact_hash"}
        ):
            raise S206PreparationBlocked("PRODUCTION_SMOKE_OUTPUT_HASH_MISMATCH")
        return {
            "status": "RESUMED_EXISTING",
            "smoke": dict(existing),
            "preflight": preflight,
            "s205_rebind_hash": canonical_json_hash(s205_plan),
        }

    provider, reference, references, reference_hash, _delta, _half_width, corrected_delta = _load_production_cell_inputs(
        args,
        anchor_id=args.cell_anchor,
        batch_size=args.cell_batch_size,
        s205_rows=s205_rows,
        g24a_rows=g24a_rows,
        execution=execution,
    )
    artifact_root = _absolute_or_logical(root, args.cell_artifact_root, field="cell_artifact_root")
    try:
        # Reuse the same production RecoverablePairedWaveRunner bridge as the
        # 50-repetition cell, but cap this smoke to exactly one frozen mapping.
        from param_importance_nlp.experiments.stage2_formal import RecoverablePairedWaveRunner

        summary = RecoverablePairedWaveRunner(
            provider,  # type: ignore[arg-type]
            execution=execution,
        ).run(
            wave_id=f"s206-smoke-{args.cell_anchor.replace('.', '-')}-b{args.cell_batch_size:03d}",
            mappings=(smoke_mapping,),
            reference=reference,
            references=references,
            reference_hash=reference_hash,
            artifact_root=artifact_root,
            max_new_units=1,
        )
    except Exception as error:
        raise S206PreparationBlocked(
            f"PRODUCTION_SMOKE_RUN_FAILED:{type(error).__name__}:{error}"
        ) from error
    if not summary.complete or tuple(summary.completed_unit_ids) != (smoke_mapping.repetition_id,):
        raise S206PreparationBlocked("PRODUCTION_SMOKE_INCOMPLETE")
    cost_statistics = {
        name: dict(values) for name, values in summary.cost_statistics.items()
    }
    payload: dict[str, object] = {
        "schema_version": "stage2-s206-formal-smoke-v1",
        "scope": "formal",
        "formal_eligible": False,
        "qualification_gate_hash": None,
        "smoke_only": True,
        "scientific_values_masked": True,
        "confirmatory_draws_generated": False,
        "mapping_id": mapping.mapping_id,
        "mapping_artifact_hash": mapping.artifact_hash,
        "mapping_cell_hash": smoke_mapping_hash,
        "anchor_id": args.cell_anchor,
        "batch_size": args.cell_batch_size,
        "repetitions_requested": 1,
        "completed_repetitions": len(summary.completed_unit_ids),
        "unit_ids": list(summary.completed_unit_ids),
        "summary_hash": summary.artifact_hash,
        "registry_hash": summary.registry_hash,
        "reference_hash": summary.reference_hash,
        "reference_hashes": dict(summary.reference_hashes),
        "cost_statistics": cost_statistics,
        "cost_io_quiescent": bool(args.cost_io_quiescent),
        "pilot_mapping_ref": str(mapping_path),
        "execution_evidence_hash": execution.artifact_hash,
        "s205_rebind_hash": canonical_json_hash(s205_plan),
        "corrected_delta_sci_hash": corrected_delta.artifact_hash,
        "corrected_delta_sci_ref": corrected_delta.ref,
        "corrected_delta_sci_batch_sizes": list(corrected_delta.batch_sizes),
        "delta_sci_source": corrected_delta.source,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    _write_once(output, payload)
    return {
        "status": "SMOKE_COMPLETE",
        "smoke": payload,
        "preflight": preflight,
        "s205_rebind_hash": canonical_json_hash(s205_plan),
        "confirmatory_draws_generated": False,
    }


def _load_cost_source(args: argparse.Namespace) -> dict[str, object]:
    """Load the frozen cost-definition boundary, not post-S2.7 measurements.

    S2.6 may consume the shared scientific cost observed by its own pilot, but
    isolated-estimator and online-incremental measurements belong to the later
    S2.9 capacity task.  The input therefore freezes their meanings and
    measurement boundary; it must not contain fabricated seconds.
    """

    if not args.cost_semantics_ref:
        raise S206PreparationBlocked("PRODUCTION_REQUIRES_COST_SEMANTICS_REF")
    root = args.data_root.resolve()
    _path, raw = _load_object_ref(root, args.cost_semantics_ref, field="cost_semantics_ref")
    value = raw
    required = {
        "schema_version",
        "scope",
        "status",
        "measurement_boundary",
        "scientific_equal_sample_cost",
        "isolated_estimator_cost",
        "online_training_incremental_cost",
        "cost_io_quiescent",
        "artifact_hash",
    }
    if set(value) != required or value.get("schema_version") != "stage2-s206-cost-semantics-contract-v1":
        raise S206PreparationBlocked("COST_SEMANTICS_FIELDS_INVALID")
    if value.get("scope") != "formal" or value.get("status") != "FROZEN":
        raise S206PreparationBlocked("COST_SEMANTICS_FROZEN_FORMAL_REQUIRED")
    boundary = value.get("measurement_boundary")
    if not isinstance(boundary, Mapping) or set(boundary) != {
        "isolated_estimator_cost",
        "online_training_incremental_cost",
    } or any(
        not isinstance(boundary[name], str) or not boundary[name]
        for name in boundary
    ):
        raise S206PreparationBlocked("COST_SEMANTICS_MEASUREMENT_BOUNDARY_INVALID")
    if value.get("artifact_hash") != canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    ):
        raise S206PreparationBlocked("COST_SEMANTICS_HASH_MISMATCH")
    result = dict(value)
    for name in (
        "scientific_equal_sample_cost",
        "isolated_estimator_cost",
        "online_training_incremental_cost",
    ):
        observation = result[name]
        if not isinstance(observation, Mapping):
            raise S206PreparationBlocked(f"COST_SEMANTICS_{name.upper()}_DEFINITION_REQUIRED")
        if observation.get("defined") is not True:
            raise S206PreparationBlocked(f"COST_SEMANTICS_{name.upper()}_DEFINITION_NOT_FROZEN")
        if not isinstance(observation.get("definition"), str) or not observation["definition"]:
            raise S206PreparationBlocked(f"COST_SEMANTICS_{name.upper()}_MEANING_MISSING")
        measurement_status = observation.get("measurement_status")
        if measurement_status not in {"PENDING_S2.9", "OBSERVED"}:
            raise S206PreparationBlocked(f"COST_SEMANTICS_{name.upper()}_MEASUREMENT_STATUS_INVALID")
        if measurement_status == "PENDING_S2.9" and any(
            key in observation for key in ("seconds", "wall_seconds", "gpu_hours", "value")
        ):
            raise S206PreparationBlocked(f"COST_SEMANTICS_{name.upper()}_PENDING_VALUE_FORBIDDEN")
        if measurement_status == "OBSERVED" and not isinstance(observation.get("measurement_ref"), str):
            raise S206PreparationBlocked(f"COST_SEMANTICS_{name.upper()}_OBSERVATION_REF_REQUIRED")
    if type(result["cost_io_quiescent"]) is not bool:
        raise S206PreparationBlocked("COST_SEMANTICS_IO_QUIESCENT_BOOLEAN_REQUIRED")
    return result


def _load_retry_policy(args: argparse.Namespace, max_attempts: int) -> tuple[dict[str, object], str]:
    """Require a frozen retry contract whenever a cell may be retried."""

    if max_attempts < 1 or max_attempts > 3:
        raise S206PreparationBlocked("MAX_CELL_ATTEMPTS_MUST_BE_1_TO_3")
    if max_attempts == 1:
        payload = {
            "schema_version": "stage2-s206-retry-policy-v1",
            "scope": "formal",
            "status": "FROZEN",
            "max_cell_attempts": 1,
            "reuse_mapping_on_retry": True,
            "new_pilot_draws_on_retry": False,
            "preserve_failure_records": True,
        }
        return payload, canonical_json_hash(payload)
    if not args.retry_policy_ref:
        raise S206PreparationBlocked("RETRY_POLICY_REF_REQUIRED_WHEN_RETRIES_ENABLED")
    root = args.data_root.resolve()
    _path, raw = _load_object_ref(root, args.retry_policy_ref, field="retry_policy_ref")
    required = {
        "schema_version",
        "scope",
        "status",
        "max_cell_attempts",
        "reuse_mapping_on_retry",
        "new_pilot_draws_on_retry",
        "preserve_failure_records",
        "artifact_hash",
    }
    if set(raw) != required or raw.get("schema_version") != "stage2-s206-retry-policy-v1":
        raise S206PreparationBlocked("RETRY_POLICY_SCHEMA_INVALID")
    if raw.get("scope") != "formal" or raw.get("status") != "FROZEN":
        raise S206PreparationBlocked("RETRY_POLICY_FROZEN_FORMAL_REQUIRED")
    if raw.get("max_cell_attempts") != max_attempts:
        raise S206PreparationBlocked("RETRY_POLICY_ATTEMPT_COUNT_MISMATCH")
    if raw.get("reuse_mapping_on_retry") is not True or raw.get("new_pilot_draws_on_retry") is not False:
        raise S206PreparationBlocked("RETRY_POLICY_DRAW_REUSE_REQUIRED")
    if raw.get("preserve_failure_records") is not True:
        raise S206PreparationBlocked("RETRY_POLICY_FAILURE_RECORDS_REQUIRED")
    if raw.get("artifact_hash") != canonical_json_hash(
        {key: value for key, value in raw.items() if key != "artifact_hash"}
    ):
        raise S206PreparationBlocked("RETRY_POLICY_HASH_MISMATCH")
    return raw, str(raw["artifact_hash"])


def _aggregate_cell_costs(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate the measured shared-runner cost without replacing aux costs."""

    fields = (
        "sample_budget",
        "gradient_evaluations",
        "gradient_seconds",
        "formula_seconds",
        "wall_seconds",
    )
    peak_memory: list[int] = []
    totals: dict[str, object] = {field: 0 for field in fields}
    totals["defined"] = True
    totals["reason"] = None
    for run in runs:
        costs = run.get("costs")
        if not isinstance(costs, Mapping):
            raise S206PreparationBlocked("PRODUCTION_CELL_COSTS_MISSING")
        scientific = costs.get("scientific_equal_sample_cost")
        if not isinstance(scientific, Mapping) or scientific.get("defined") is not True:
            raise S206PreparationBlocked("PRODUCTION_SCIENTIFIC_COST_UNDEFINED")
        for field in fields:
            value = scientific.get(field)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise S206PreparationBlocked(f"PRODUCTION_SCIENTIFIC_COST_NONFINITE:{field}")
            totals[field] = float(totals[field]) + float(value)
        memory = scientific.get("peak_memory_bytes")
        if memory is not None:
            if not isinstance(memory, int) or isinstance(memory, bool) or memory < 0:
                raise S206PreparationBlocked("PRODUCTION_SCIENTIFIC_PEAK_MEMORY_INVALID")
            peak_memory.append(memory)
    totals["peak_memory_bytes"] = max(peak_memory) if peak_memory else None
    totals["aggregation"] = "sum_committed_formal_cell_costs"
    return totals


def _production_execute(args: argparse.Namespace) -> dict[str, object]:
    """Run the real 24-cell queue and close the S2.6/G2.4b boundary."""

    preflight = _preflight(args)
    execution = _load_formal_execution(args)
    if not args.sampling_plan_ref or not args.pilot_mapping_output:
        raise S206PreparationBlocked("PRODUCTION_EXECUTE_REQUIRES_SAMPLING_AND_MAPPING_OUTPUT")
    if not args.s205_rebind_ref or not args.cost_semantics_ref:
        raise S206PreparationBlocked("PRODUCTION_EXECUTE_REQUIRES_S205_REBIND_AND_COST_SEMANTICS")
    root = args.data_root.resolve()
    sampling_path = _absolute_or_logical(root, args.sampling_plan_ref, field="sampling_plan_ref")
    sampling_value = load_canonical_json(sampling_path)
    if not isinstance(sampling_value, Mapping):
        raise S206PreparationBlocked("SAMPLING_PLAN_NOT_OBJECT")
    sampling = SamplingPlan.from_mapping(sampling_value)
    mapping_path = _absolute_or_logical(root, args.pilot_mapping_output, field="pilot_mapping_output")
    mapping = build_global_pilot_mapping(sampling)
    if mapping_path.exists():
        existing = load_canonical_json(mapping_path)
        if not isinstance(existing, Mapping) or existing.get("artifact_hash") != mapping.artifact_hash:
            raise S206PreparationBlocked("PILOT_MAPPING_OUTPUT_CONFLICT")
    else:
        _write_once(mapping_path, mapping.to_dict())
    s205_plan, _s205_rows = _load_s205_rebind(args)
    _load_g24a_metrics(args)
    operations = _logical(root, args.operations_root, field="operations_root")
    operations.mkdir(parents=True, exist_ok=True)
    status_path = operations / "status.json"
    ordered_jobs = _production_lpt_jobs()
    jobs = deque(ordered_jobs)
    expected_work = {
        f"{anchor_id}:B{batch_size}": PILOT_REPETITIONS * batch_size
        for anchor_id, batch_size in ordered_jobs
    }
    attempts: dict[tuple[str, int], int] = {}
    active: dict[int, tuple[subprocess.Popen[bytes], tuple[str, int], str, Path]] = {}
    free_gpus = list(APPROVED_GPU_UUIDS)
    records: list[dict[str, object]] = []
    max_attempts = int(args.max_cell_attempts)
    retry_policy, retry_policy_hash = _load_retry_policy(args, max_attempts)
    _load_cost_source(args)  # fail before starting any cell

    def launch(anchor_id: str, batch_size: int, gpu_uuid: str) -> None:
        component = f"{anchor_id.replace('.', '__')}__b{batch_size:03d}"
        cell_artifact = f"{args.operations_root}/cells/{component}/artifacts"
        cell_output = f"{args.operations_root}/cells/{component}/run.json"
        log_path = operations / "cells" / component / "launcher.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.python), str(Path(__file__).resolve()), "--production-cell",
            "--data-root", str(root), "--repository", str(args.repository.resolve()),
            "--s204-root", args.s204_root, "--g23-evaluation", args.g23_evaluation,
            "--g24a-evaluation", args.g24a_evaluation, "--operations-root", args.operations_root,
            "--sampling-plan-ref", args.sampling_plan_ref, "--pilot-mapping-ref", args.pilot_mapping_output,
            "--s205-rebind-ref", args.s205_rebind_ref, "--execution-evidence-ref", args.execution_evidence_ref,
            "--cell-anchor", anchor_id, "--cell-batch-size", str(batch_size),
            "--cell-artifact-root", cell_artifact, "--cell-output", cell_output,
            "--cell-gpu-uuid", gpu_uuid,
        ]
        command.extend(["--gpu-inventory-json", str(args.gpu_inventory_json.resolve())])
        if args.resource_within_budget:
            command.append("--resource-within-budget")
        if args.cost_io_quiescent:
            command.append("--cost-io-quiescent")
        env = os.environ.copy()
        env.update({
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
            "PYTHONPATH": str(args.repository.resolve() / "src"),
        })
        handle = log_path.open("ab")
        process = subprocess.Popen(
            command,
            cwd=args.repository.resolve(),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        active[process.pid] = (process, (anchor_id, batch_size), gpu_uuid, log_path)
        attempts[(anchor_id, batch_size)] = attempts.get((anchor_id, batch_size), 0) + 1

    def status(stage: str, **extra: object) -> None:
        _write_status(
            status_path,
            {
                "run_id": args.run_id,
                "stage": stage,
                "formal_eligible": stage not in {"BLOCKED_EXECUTION", "PILOT_BLOCKED"},
                "preflight": preflight,
                "execution_evidence_hash": execution.artifact_hash,
                "pilot_mapping_ref": str(mapping_path),
                "pilot_mapping_hash": mapping.artifact_hash,
                "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
                "excluded_pci": EXCLUDED_PCI,
                "gpu_inventory_ref": preflight["gpu_inventory_ref"],
                "gpu_inventory_artifact_hash": preflight["gpu_inventory_artifact_hash"],
                "gpu_inventory_source_sha256": preflight["gpu_inventory_source_sha256"],
                "gpu_inventory_path": preflight["gpu"]["inventory_path"],  # type: ignore[index]
                "preflight_artifact_hash": preflight["preflight_artifact_hash"],
                "retry_policy_ref": args.retry_policy_ref,
                "retry_policy_hash": retry_policy_hash,
                "completed_cells": list(records),
                "expected_cell_count": 24,
                "confirmatory_draws_generated": False,
                **extra,
            },
        )

    status(
        "PREFLIGHT_PASS",
        queue_mode="dynamic_four_gpu_lpt_draw_work",
        queue_order=[
            {"anchor_id": anchor_id, "batch_size": batch_size}
            for anchor_id, batch_size in ordered_jobs
        ],
        queue_work_units=expected_work,
        queue_work_multiplier_source="none_available_uses_frozen_50_times_B",
        s205_rebind_hash=canonical_json_hash(s205_plan),
        retry_policy_ref=args.retry_policy_ref,
        retry_policy_hash=retry_policy_hash,
    )

    try:
        while jobs or active:
            while jobs and free_gpus:
                # LPT order is dynamically assigned to whichever approved UUID
                # has just become free; no card is ever shared.  The ordered
                # queue starts B=256 work first, preventing a large-cell tail.
                job = jobs.popleft()
                gpu_uuid = free_gpus.pop(0)
                launch(job[0], job[1], gpu_uuid)
            completed_any = False
            for pid, (process, job, gpu_uuid, log_path) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                completed_any = True
                del active[pid]
                free_gpus.append(gpu_uuid)
                record: dict[str, object] = {
                    "anchor_id": job[0],
                    "batch_size": job[1],
                    "gpu_uuid": gpu_uuid,
                    "attempt": attempts[job],
                    "returncode": int(code),
                    "log_ref": str(log_path),
                }
                if code != 0:
                    record["status"] = "FAILED"
                    if attempts[job] < max_attempts:
                        record["requeued"] = True
                        jobs.appendleft(job)
                    records.append(record)
                    if attempts[job] >= max_attempts:
                        status("BLOCKED_EXECUTION", failure=record)
                        raise S206PreparationBlocked(
                            f"PRODUCTION_CELL_FAILED:{job[0]}:B{job[1]}:attempts={attempts[job]}"
                        )
                else:
                    output = _logical(
                        root,
                        f"{args.operations_root}/cells/"
                        f"{job[0].replace('.', '__')}__b{job[1]:03d}/run.json",
                        field="cell_output",
                    )
                    payload = load_canonical_json(output)
                    if not isinstance(payload, Mapping):
                        raise S206PreparationBlocked(f"PRODUCTION_CELL_OUTPUT_INVALID:{job}")
                    record.update({
                        "status": "COMPLETE",
                        "result_ref": str(output),
                        "result_hash": canonical_json_hash(payload),
                    })
                    records.append(record)
                status(
                    "PILOT_RUNNING",
                    queue_depth=len(jobs),
                    active_gpu_count=len(active),
                )
            if active and not completed_any:
                time.sleep(2.0)
        if len([item for item in records if item.get("status") == "COMPLETE"]) != 24:
            raise S206PreparationBlocked("PRODUCTION_CELL_COUNT_INCOMPLETE")
    except Exception as error:
        status(
            "BLOCKED_EXECUTION",
            reason=f"{type(error).__name__}:{error}",
        )
        raise
    finally:
        for process, _job, _gpu, _log in active.values():
            if process.poll() is None:
                process.terminate()

    result_refs = [
        _logical(
            root,
            f"{args.operations_root}/cells/"
            f"{anchor_id.replace('.', '__')}__b{batch_size:03d}/run.json",
            field="cell_output",
        )
        for anchor_id in ANCHOR_IDS
        for batch_size in PILOT_B_VALUES
    ]
    from param_importance_nlp.experiments.stage2_s206_formal import FormalPilotCellRun

    runs: list[FormalPilotCellRun] = []
    measurements: list[dict[str, object]] = []
    for path, (anchor_id, batch_size) in zip(
        result_refs,
        ((anchor_id, batch_size) for anchor_id in ANCHOR_IDS for batch_size in PILOT_B_VALUES),
    ):
        value = load_canonical_json(path)
        if not isinstance(value, Mapping):
            raise S206PreparationBlocked(f"PRODUCTION_CELL_RESULT_NOT_OBJECT:{path}")
        run = FormalPilotCellRun.from_mapping(dict(value))
        if run.mapping_artifact_hash != mapping.artifact_hash:
            raise S206PreparationBlocked(f"PRODUCTION_CELL_MAPPING_HASH_DRIFT:{path}")
        row = _s205_rows.get(anchor_id)
        if row is None:
            raise S206PreparationBlocked(f"S205_ROW_MISSING:{anchor_id}")
        try:
            corrected = load_bound_corrected_delta(
                root,
                g23_evaluation_ref=args.g23_evaluation,
                cell_id=anchor_id,
                expected_config_hash=str(row["config_hash"]),
                expected_result_hash=str(row["result_hash"]),
            )
        except CorrectedDeltaRejected as error:
            raise S206PreparationBlocked(f"G23_CORRECTED_DELTA_AGGREGATE_INVALID:{anchor_id}:{error}") from error
        expected_delta = corrected.delta_for(batch_size)
        if (
            run.corrected_delta_sci_hash != corrected.artifact_hash
            or run.corrected_delta_sci_ref != corrected.ref
            or run.corrected_delta_sci_cell_id != corrected.cell_id
            or run.corrected_delta_sci_config_hash != corrected.config_hash
            or run.corrected_delta_sci_result_hash != corrected.result_hash
            or any(dict(item.delta_sci_by_endpoint) != expected_delta for item in run.measurements)
        ):
            raise S206PreparationBlocked(f"PRODUCTION_CELL_CORRECTED_DELTA_BINDING_DRIFT:{anchor_id}:B{batch_size}")
        runs.append(run)
        measurements.extend(item.to_dict() for item in run.measurements)
    measurements_payload: dict[str, object] = {
        "schema_version": "stage2-s206-blinded-measurements-v1",
        "mapping_ref": str(mapping_path),
        "mapping_hash": mapping.artifact_hash,
        "cell_count": len(runs),
        "measurement_count": len(measurements),
        "measurements": measurements,
        "scientific_values_masked": True,
        "formal_eligible": False,
    }
    measurements_payload["artifact_hash"] = canonical_json_hash(measurements_payload)
    measurements_path = _logical(
        root,
        args.measurements_output or f"{args.operations_root}/blinded-measurements.json",
        field="measurements_output",
    )
    _write_once(measurements_path, measurements_payload)
    cost_source = _load_cost_source(args)
    aggregate_scientific = _aggregate_cell_costs([run.to_dict() for run in runs])
    costs_payload: dict[str, object] = {
        "schema_version": "stage2-s206-cost-semantics-v1",
        "scientific_equal_sample_cost": aggregate_scientific,
        "isolated_estimator_cost": dict(cost_source["isolated_estimator_cost"]),  # type: ignore[arg-type]
        "online_training_incremental_cost": dict(cost_source["online_training_incremental_cost"]),  # type: ignore[arg-type]
        "cost_io_quiescent": bool(args.cost_io_quiescent and cost_source["cost_io_quiescent"]),
        "source_ref": args.cost_semantics_ref,
        "source_hash": str(cost_source["artifact_hash"]),
    }
    costs_payload["artifact_hash"] = canonical_json_hash(costs_payload)
    costs_path = _logical(
        root,
        args.costs_output or f"{args.operations_root}/cost-semantics.json",
        field="costs_output",
    )
    _write_once(costs_path, costs_payload)
    cost_semantics = CostSemantics(
        scientific_equal_sample_cost=dict(aggregate_scientific),
        isolated_estimator_cost=dict(cost_source["isolated_estimator_cost"]),  # type: ignore[arg-type]
        online_training_incremental_cost=dict(cost_source["online_training_incremental_cost"]),  # type: ignore[arg-type]
        cost_io_quiescent=bool(costs_payload["cost_io_quiescent"]),
    )
    report = reduce_blinded_pilot(
        mapping,
        tuple(BlindPilotMeasurement.from_mapping(item) for item in measurements),
        cost_semantics=cost_semantics,
    )
    report_path = _logical(
        root,
        args.report_output or f"{args.operations_root}/blinded-pilot-report.json",
        field="report_output",
    )
    _write_once(report_path, report.to_dict())
    evidence_refs = (
        str(report_path),
        str(measurements_path),
        str(costs_path),
        str(args.execution_evidence_ref),
        (
            f"{preflight['gpu_inventory_ref']}::artifact_sha256="
            f"{preflight['gpu_inventory_artifact_hash']}::source_sha256="
            f"{preflight['gpu_inventory_source_sha256']}"
        ),
    )
    if report.status != "READY_FOR_QUALIFICATION":
        gate = build_g24b_gate(
            report,
            execution,
            evidence_refs=evidence_refs,
            gpu_inventory_identity=preflight["gpu_inventory_identity"],  # type: ignore[arg-type]
        )
        gate_path = _logical(
            root,
            args.gate_output or f"{args.operations_root}/g2.4b-gate.json",
            field="gate_output",
        )
        _write_once(gate_path, gate.to_dict())
        status(
            "PILOT_BLOCKED",
            gate_ref=str(gate_path),
            report_ref=str(report_path),
            costs_ref=str(costs_path),
        )
        return {
            "status": "PILOT_BLOCKED",
            "report_ref": str(report_path),
            "gate_ref": str(gate_path),
            "report": report.to_dict(),
            "gate": gate.to_dict(),
        }
    gate = build_g24b_gate(
        report,
        execution,
        evidence_refs=evidence_refs,
        gpu_inventory_identity=preflight["gpu_inventory_identity"],  # type: ignore[arg-type]
    )
    matrix = qualify_formal_matrix(report, execution, gate, freeze_id=args.freeze_id)
    confirmatory = build_formal_confirmatory_mapping(
        matrix,
        sampling,
        pilot_draw_ids=mapping.pilot_draw_ids,
        gate=gate,
    )
    gate_path = _logical(
        root,
        args.gate_output or f"{args.operations_root}/g2.4b-gate.json",
        field="gate_output",
    )
    matrix_path = _logical(
        root,
        args.matrix_output or f"{args.operations_root}/formal-matrix.json",
        field="matrix_output",
    )
    confirmatory_path = _logical(
        root,
        args.confirmatory_output or f"{args.operations_root}/confirmatory-mapping.json",
        field="confirmatory_output",
    )
    _write_once(gate_path, gate.to_dict())
    _write_once(matrix_path, matrix.to_dict())
    _write_once(confirmatory_path, confirmatory.to_dict())
    freeze_payload: dict[str, object] = {
        "schema_version": "stage2-s206-formal-freeze-commit-v1",
        "status": "PASS",
        "formal_eligible": True,
        "gate_ref": str(gate_path),
        "gate_hash": gate.artifact_hash,
        "matrix_ref": str(matrix_path),
        "matrix_hash": matrix.artifact_hash,
        "confirmatory_mapping_ref": str(confirmatory_path),
        "confirmatory_mapping_hash": confirmatory.artifact_hash,
        "pilot_mapping_hash": mapping.artifact_hash,
        "execution_evidence_hash": execution.artifact_hash,
        "gpu_inventory_ref": preflight["gpu_inventory_ref"],
        "gpu_inventory_artifact_hash": preflight["gpu_inventory_artifact_hash"],
        "gpu_inventory_source_sha256": preflight["gpu_inventory_source_sha256"],
        "gpu_inventory_path": preflight["gpu"]["inventory_path"],  # type: ignore[index]
        "preflight_artifact_hash": preflight["preflight_artifact_hash"],
        "cost_semantics_ref": args.cost_semantics_ref,
        "cost_semantics_hash": str(cost_source["artifact_hash"]),
        "retry_policy_ref": args.retry_policy_ref,
        "retry_policy_hash": retry_policy_hash,
        "retry_policy": retry_policy,
        "confirmatory_gradients_started": False,
    }
    freeze_payload["artifact_hash"] = canonical_json_hash(freeze_payload)
    freeze_path = _logical(
        root,
        args.freeze_commit_output or f"{args.operations_root}/formal-freeze-commit.json",
        field="freeze_commit_output",
    )
    _write_once(freeze_path, freeze_payload)
    total_draws = mapping.total_draw_count
    throughput = args.throughput_sequences_per_second
    wall_model = {
        "pilot_draw_events": total_draws,
        "approved_gpu_count": len(APPROVED_GPU_UUIDS),
        "formula": "pilot_draw_events/(approved_gpu_count*throughput_sequences_per_second)",
        "throughput_sequences_per_second": throughput,
        "lower_bound_seconds": (
            None
            if throughput is None or throughput <= 0
            else total_draws / (len(APPROVED_GPU_UUIDS) * throughput)
        ),
    }
    status(
        "G2.4B_PASS_MATRIX_FROZEN",
        gate_ref=str(gate_path),
        matrix_ref=str(matrix_path),
        confirmatory_mapping_ref=str(confirmatory_path),
        freeze_commit_ref=str(freeze_path),
        wall_time_model=wall_model,
    )
    return {
        "status": "G2.4B_PASS_MATRIX_FROZEN",
        "gate_ref": str(gate_path),
        "matrix_ref": str(matrix_path),
        "confirmatory_mapping_ref": str(confirmatory_path),
        "freeze_commit_ref": str(freeze_path),
        "retry_policy_hash": retry_policy_hash,
        "wall_time_model": wall_model,
    }


def _execute(args: argparse.Namespace) -> dict[str, object]:
    return _production_execute(args)


def _reduce(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    if not args.pilot_mapping_ref or not args.measurements_ref or not args.costs_ref or not args.report_output:
        raise S206PreparationBlocked("REDUCE_REQUIRES_MAPPING_MEASUREMENTS_COSTS_AND_OUTPUT")
    mapping_path = _logical(root, args.pilot_mapping_ref, field="pilot_mapping_ref")
    measurements_path = _logical(root, args.measurements_ref, field="measurements_ref")
    costs_path = _logical(root, args.costs_ref, field="costs_ref")
    report_path = _logical(root, args.report_output, field="report_output")
    raw_mapping = load_canonical_json(mapping_path)
    raw_measurements = load_canonical_json(measurements_path)
    raw_costs = load_canonical_json(costs_path)
    if not isinstance(raw_mapping, Mapping) or not isinstance(raw_costs, Mapping):
        raise S206PreparationBlocked("REDUCE_INPUT_OBJECT_REQUIRED")
    measurement_values = raw_measurements.get("measurements") if isinstance(raw_measurements, Mapping) else raw_measurements
    if not isinstance(measurement_values, list) or not all(isinstance(item, Mapping) for item in measurement_values):
        raise S206PreparationBlocked("REDUCE_MEASUREMENTS_ARRAY_REQUIRED")
    costs = CostSemantics(
        scientific_equal_sample_cost=dict(raw_costs["scientific_equal_sample_cost"]),  # type: ignore[arg-type]
        isolated_estimator_cost=dict(raw_costs["isolated_estimator_cost"]),  # type: ignore[arg-type]
        online_training_incremental_cost=dict(raw_costs["online_training_incremental_cost"]),  # type: ignore[arg-type]
        cost_io_quiescent=raw_costs["cost_io_quiescent"],  # type: ignore[arg-type]
    )
    report = reduce_blinded_pilot(
        GlobalPilotMappingManifest.from_mapping(dict(raw_mapping)),
        tuple(BlindPilotMeasurement.from_mapping(dict(item)) for item in measurement_values),
        cost_semantics=costs,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(report_path, report.to_dict())
    return report.to_dict()


def _synthetic_cell(args: argparse.Namespace) -> dict[str, object]:
    """Reject the legacy direct bridge until it has a G2.3 sidecar binding.

    The old implementation accepted a caller-authored ``delta_sci`` table,
    which made it possible to silently consume the r23 sizing-node keys.  A
    synthetic provider is not an exception to the S2.6 consumer contract.
    """

    root = args.data_root.resolve()
    # Keep the direct bridge under the same G2.3/G2.4a and approved-GPU
    # preflight as the long-running launcher; synthetic execution is not a
    # bypass around those gates.
    _preflight(args)
    raise S206PreparationBlocked("SYNTHETIC_CELL_REQUIRES_BOUND_G23_CORRECTED_DELTA")
    required = (
        args.pilot_mapping_ref,
        args.execution_evidence_ref,
        args.cell_artifact_root,
        args.cell_output,
        args.reference_json,
        args.cell_sizing_ref,
    )
    if not all(required) or args.cell_anchor is None or args.cell_batch_size is None:
        raise S206PreparationBlocked("SYNTHETIC_CELL_REQUIRES_MAPPING_EXECUTION_CELL_AND_SIZING_INPUTS")
    mapping_path = _logical(root, args.pilot_mapping_ref, field="pilot_mapping_ref")
    mapping_value = load_canonical_json(mapping_path)
    if not isinstance(mapping_value, Mapping):
        raise S206PreparationBlocked("SYNTHETIC_CELL_MAPPING_NOT_OBJECT")
    mapping = GlobalPilotMappingManifest.from_mapping(dict(mapping_value))
    execution = _load_formal_execution(args)
    try:
        cell = next(
            item for item in mapping.cells
            if item.anchor_id == args.cell_anchor and item.batch_size == args.cell_batch_size
        )
    except StopIteration as error:
        raise S206PreparationBlocked("SYNTHETIC_CELL_NOT_IN_MAPPING") from error
    reference_value = load_canonical_json(_logical(root, args.reference_json, field="reference_json"))
    sizing_value = load_canonical_json(_logical(root, args.cell_sizing_ref, field="cell_sizing_ref"))
    if not isinstance(reference_value, Mapping) or not isinstance(sizing_value, Mapping):
        raise S206PreparationBlocked("SYNTHETIC_CELL_REFERENCE_OR_SIZING_NOT_OBJECT")
    delta = sizing_value.get("delta_sci_by_endpoint")
    half_width = sizing_value.get("reference_half_width_by_endpoint")
    if not isinstance(delta, Mapping) or not isinstance(half_width, Mapping):
        raise S206PreparationBlocked("SYNTHETIC_CELL_SIZING_FIELDS_REQUIRED")
    try:
        from param_importance_nlp.experiments.stage2_formal import _vector_digest
        from param_importance_nlp.providers.synthetic import SyntheticGradientProvider

        shapes = {
            str(name): tuple(len(value) if isinstance(value, list) else 1 for _ in [0])
            for name, value in reference_value.items()
        }
        # The synthetic bridge is intentionally limited to flat vectors.  A
        # production provider must be supplied through run_formal_pilot_cell.
        if any(not isinstance(value, list) or not value for value in reference_value.values()):
            raise ValueError("synthetic reference vectors must be non-empty lists")
        provider = SyntheticGradientProvider.from_location_scale(
            parameter_shapes=shapes,
            sample_count=int(args.synthetic_sample_count),
            seed=int(args.synthetic_seed),
        )
        reference_hash = str(args.reference_hash) if args.reference_hash else _vector_digest(reference_value)
        result = run_formal_pilot_cell(
            cell,
            mapping=mapping,
            provider=provider,
            execution=execution,
            reference=reference_value,
            reference_hash=reference_hash,
            artifact_root=_logical(root, args.cell_artifact_root, field="cell_artifact_root"),
            delta_sci_by_endpoint=dict(delta),
            reference_half_width_by_endpoint=dict(half_width),
            resource_within_budget=True,
            cost_io_quiescent=True,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise S206PreparationBlocked(f"SYNTHETIC_CELL_BRIDGE_INVALID:{error}") from error
    output = _logical(root, args.cell_output, field="cell_output")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output, result.to_dict())
    return result.to_dict()


def _qualify(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    if not args.pilot_mapping_ref or not args.measurements_ref or not args.costs_ref or not args.execution_evidence_ref or not args.matrix_output or not args.gate_output:
        raise S206PreparationBlocked("QUALIFY_REQUIRES_REDUCER_INPUTS_EXECUTION_AND_OUTPUTS")
    mapping_path = _logical(root, args.pilot_mapping_ref, field="pilot_mapping_ref")
    measurements_path = _logical(root, args.measurements_ref, field="measurements_ref")
    costs_path = _logical(root, args.costs_ref, field="costs_ref")
    execution_path = _logical(root, args.execution_evidence_ref, field="execution_evidence_ref")
    matrix_path = _logical(root, args.matrix_output, field="matrix_output")
    gate_path = _logical(root, args.gate_output, field="gate_output")
    raw_mapping = load_canonical_json(mapping_path)
    raw_measurements = load_canonical_json(measurements_path)
    raw_costs = load_canonical_json(costs_path)
    raw_execution = load_canonical_json(execution_path)
    if not all(isinstance(value, Mapping) for value in (raw_mapping, raw_costs, raw_execution)):
        raise S206PreparationBlocked("QUALIFY_INPUT_OBJECT_REQUIRED")
    measurement_values = raw_measurements.get("measurements") if isinstance(raw_measurements, Mapping) else raw_measurements
    if not isinstance(measurement_values, list) or not all(isinstance(item, Mapping) for item in measurement_values):
        raise S206PreparationBlocked("QUALIFY_MEASUREMENTS_ARRAY_REQUIRED")
    costs = CostSemantics(
        scientific_equal_sample_cost=dict(raw_costs["scientific_equal_sample_cost"]),  # type: ignore[index]
        isolated_estimator_cost=dict(raw_costs["isolated_estimator_cost"]),  # type: ignore[index]
        online_training_incremental_cost=dict(raw_costs["online_training_incremental_cost"]),  # type: ignore[index]
        cost_io_quiescent=raw_costs["cost_io_quiescent"],  # type: ignore[index]
    )
    mapping = GlobalPilotMappingManifest.from_mapping(dict(raw_mapping))
    report = reduce_blinded_pilot(
        mapping,
        tuple(BlindPilotMeasurement.from_mapping(dict(item)) for item in measurement_values),
        cost_semantics=costs,
    )
    execution = FormalExecutionEvidence.from_mapping(dict(raw_execution))
    refs = tuple(args.evidence_ref or (args.measurements_ref, args.pilot_mapping_ref, args.execution_evidence_ref))
    gate = build_g24b_gate(report, execution, evidence_refs=refs)
    matrix = qualify_formal_matrix(report, execution, gate, freeze_id=args.freeze_id)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(gate_path, gate.to_dict())
    write_canonical_json(matrix_path, matrix.to_dict())
    return {"status": "FORMAL_FROZEN", "formal_eligible": True, "gate_ref": str(gate_path), "matrix_ref": str(matrix_path), "gate": gate.to_dict(), "matrix": matrix.to_dict()}


def _verify_status_inventory_identity(
    value: Mapping[str, object],
    *,
    data_root: Path,
) -> None:
    required = (
        "gpu_inventory_path",
        "gpu_inventory_ref",
        "gpu_inventory_artifact_hash",
        "gpu_inventory_source_sha256",
        "preflight_artifact_hash",
    )
    if any(key not in value for key in required):
        raise S206PreparationBlocked("STATUS_GPU_INVENTORY_IDENTITY_MISSING")
    _rows, identity = _load_inventory_snapshot(
        Path(str(value["gpu_inventory_path"])),
        data_root=data_root,
    )
    if (
        identity["source_ref"] != value["gpu_inventory_ref"]
        or identity["artifact_hash"] != value["gpu_inventory_artifact_hash"]
        or identity["source_sha256"] != value["gpu_inventory_source_sha256"]
    ):
        raise S206PreparationBlocked("STATUS_GPU_INVENTORY_IDENTITY_DRIFT")
    preflight = value.get("preflight")
    if not isinstance(preflight, Mapping):
        raise S206PreparationBlocked("STATUS_PREFLIGHT_IDENTITY_MISSING")
    declared = preflight.get("preflight_artifact_hash")
    body = {key: item for key, item in preflight.items() if key != "preflight_artifact_hash"}
    if (
        not isinstance(declared, str)
        or value["preflight_artifact_hash"] != declared
        or canonical_json_hash(body) != declared
    ):
        raise S206PreparationBlocked("STATUS_PREFLIGHT_IDENTITY_DRIFT")


def _wait(args: argparse.Namespace) -> int:
    status_path = _logical(args.data_root.resolve(), f"{args.operations_root}/status.json", field="status")
    deadline = None if args.timeout_seconds is None else time.monotonic() + args.timeout_seconds
    while True:
        if status_path.exists():
            value = load_canonical_json(status_path)
            if isinstance(value, Mapping):
                print(json.dumps(value, ensure_ascii=False, sort_keys=True))
                if value.get("stage") in {
                    "PILOT_COMPLETE",
                    "G2.4B_PASS_MATRIX_FROZEN",
                    "PILOT_BLOCKED",
                    "BLOCKED_EXECUTION",
                }:
                    _verify_status_inventory_identity(
                        value,
                        data_root=args.data_root.resolve(),
                    )
                    return 0 if value.get("stage") == "G2.4B_PASS_MATRIX_FROZEN" else 3
        if deadline is not None and time.monotonic() >= deadline:
            return 4
        time.sleep(args.poll_seconds)


def _detach(args: argparse.Namespace) -> int:
    """Re-exec the exact execute command in a new session and return its PID."""

    if not args.execute:
        raise S206PreparationBlocked("DETACH_REQUIRES_EXECUTE")
    # Validate all immutable gates and the inventory identity before spawning
    # anything.  The child repeats this check immediately before scheduling.
    preflight = _preflight(args)
    operations = _logical(args.data_root.resolve(), args.operations_root, field="operations_root")
    operations.mkdir(parents=True, exist_ok=True)
    log_path = operations / "launcher.log"
    pid_path = operations / "launcher.pid.json"
    if pid_path.exists():
        try:
            existing = load_canonical_json(pid_path)
        except (OSError, TypeError, ValueError):
            existing = None
        if isinstance(existing, Mapping) and isinstance(existing.get("pid"), int):
            try:
                os.kill(int(existing["pid"]), 0)
            except OSError:
                pass
            else:
                raise S206PreparationBlocked(f"DETACHED_LAUNCH_ALREADY_RUNNING:{existing['pid']}")
    child_argv = [item for item in sys.argv[1:] if item != "--detach"]
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            [str(args.python), str(Path(__file__).resolve()), *child_argv],
            cwd=args.repository.resolve(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "schema_version": "stage2-s206-detached-launch-v1",
        "pid": int(process.pid),
        "run_id": args.run_id,
        "operations_root": str(operations),
        "log_ref": str(log_path),
        "status_ref": str(operations / "status.json"),
        "gpu_inventory_ref": preflight["gpu_inventory_ref"],
        "gpu_inventory_artifact_hash": preflight["gpu_inventory_artifact_hash"],
        "gpu_inventory_source_sha256": preflight["gpu_inventory_source_sha256"],
        "gpu_inventory_path": preflight["gpu"]["inventory_path"],  # type: ignore[index]
        "preflight_artifact_hash": preflight["preflight_artifact_hash"],
        "recovery_command": "--wait",
        "confirmatory_draws_generated": False,
    }
    write_canonical_json(pid_path, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict detached S2.6/G2.4b pilot launcher")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--reduce", action="store_true")
    action.add_argument("--qualify", action="store_true")
    action.add_argument("--wait", action="store_true")
    action.add_argument("--synthetic-cell", action="store_true")
    action.add_argument("--production-cell", action="store_true")
    action.add_argument("--production-smoke-cell", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--s204-root", required=True)
    parser.add_argument("--g23-evaluation", required=True)
    parser.add_argument("--g24a-evaluation", required=True)
    parser.add_argument("--gpu-inventory-json", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--run-id", default="s206-formal-g24b")
    parser.add_argument("--operations-root", required=True)
    parser.add_argument("--config-root")
    parser.add_argument("--environment-root")
    parser.add_argument("--result-root")
    parser.add_argument("--sampling-plan-ref")
    parser.add_argument("--pilot-mapping-output")
    parser.add_argument("--pilot-mapping-ref")
    parser.add_argument("--measurements-ref")
    parser.add_argument("--costs-ref")
    parser.add_argument("--measurements-output")
    parser.add_argument("--costs-output")
    parser.add_argument("--report-output")
    parser.add_argument("--execution-evidence-ref")
    parser.add_argument("--matrix-output")
    parser.add_argument("--gate-output")
    parser.add_argument("--confirmatory-output")
    parser.add_argument("--freeze-commit-output")
    parser.add_argument("--s205-rebind-ref")
    parser.add_argument("--cost-semantics-ref")
    parser.add_argument("--retry-policy-ref")
    parser.add_argument("--freeze-id", default="s206-formal-matrix-freeze")
    parser.add_argument("--evidence-ref", action="append", default=[])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--detach", action="store_true", help="detach --execute and return a PID")
    parser.add_argument("--cell-anchor")
    parser.add_argument("--cell-batch-size", type=int)
    parser.add_argument("--cell-artifact-root")
    parser.add_argument("--cell-output")
    parser.add_argument("--reference-json")
    parser.add_argument("--reference-hash")
    parser.add_argument("--cell-sizing-ref")
    parser.add_argument("--cell-gpu-uuid")
    parser.add_argument("--resource-within-budget", action="store_true")
    parser.add_argument("--cost-io-quiescent", action="store_true")
    parser.add_argument("--max-cell-attempts", type=int, default=2)
    parser.add_argument("--throughput-sequences-per-second", type=float)
    parser.add_argument("--synthetic-sample-count", type=int, default=4096)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (args.preflight or args.execute or args.detach) and args.gpu_inventory_json is None:
            raise S206PreparationBlocked("GPU_INVENTORY_JSON_REQUIRED")
        if args.detach:
            return _detach(args)
        if args.synthetic_cell:
            result = _synthetic_cell(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.production_cell:
            result = _production_cell(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.production_smoke_cell:
            result = _production_smoke_cell(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.wait:
            return _wait(args)
        if args.reduce:
            result = _reduce(args)
            if args.plan_output is not None:
                output = args.plan_output.resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                write_canonical_json(output, result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.qualify:
            result = _qualify(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        result = _execute(args) if args.execute else _preflight(args)
        if args.plan_output is not None:
            output = args.plan_output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            write_canonical_json(output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, S206PreparationBlocked) as error:
        payload = {
            "schema_version": "stage2-s206-formal-preflight-v1",
            "status": "BLOCKED",
            "formal_eligible": False,
            "reason": f"{type(error).__name__}:{error}",
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
