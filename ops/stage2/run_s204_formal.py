#!/usr/bin/env python3
"""S2.4 formal reference preflight/launcher.

The launcher owns only the six-cell sizing plan and process supervision.  Real
formal execution is delegated to the repository's ``TaskRuntime`` and its
Stage 2 specialized runner, which performs the complete Stage 0/G3, Stage 1,
G2.1/G2.2 predecessor, contract-freeze, and fixed-state provider checks.  This
module never constructs a provider or manufactures formal evidence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import threading
from typing import Any, Mapping


G21_ARTIFACT = "259831e2a1b16afbbef34c9cea602e636756b0f6173d1a8f4c32ec554c653f79"
ASSET_DIGEST = "f57decd5cf00e69e45ab2f02c994abb202f5c614e1441acb8aebcb1807ff76ee"
DATA_DIGEST = "df8eeac5178305d409cf6128ac5d5648567aae895592c79fa21542e84a28e0f1"
EXCLUDED_PCI = "0000:50:00.0"
EXCLUDED_UUID = "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
APPROVED_GPU_BINDINGS = {
    "0000:53:00.0": "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
    "0000:9C:00.0": "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "0000:9D:00.0": "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "0000:A0:00.0": "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
}
DEFAULT_CANDIDATES = (512, 1024, 2048, 4096)
DEFAULT_BLOCK_SIZE = 32
G23_GATE_ID = "stage2.G2.3"
G23_REQUIRED_METRICS = (
    "normalized_l1",
    "pearson",
    "signal_eligible_spearman",
    "layer_module_spearman",
    "topk_overlap_0_001",
    "topk_overlap_0_01",
    "topk_overlap_0_05",
    "layer_module_delta",
    "h_ref",
    "min_delta_sci",
    "epsilon_num",
)


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


def _gpu_inventory() -> list[dict[str, str]]:
    """Read the live index/PCI/UUID mapping; never infer identity from index."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"nvidia-smi inventory failed: {completed.stderr.strip()}")
    rows: list[dict[str, str]] = []
    for fields in csv.reader(line for line in completed.stdout.splitlines() if line.strip()):
        if len(fields) != 3:
            raise ValueError(f"invalid nvidia-smi inventory row: {fields!r}")
        index, pci, uuid = (item.strip() for item in fields)
        if not index.isdigit() or not pci or not uuid:
            raise ValueError(f"invalid nvidia-smi inventory identity: {fields!r}")
        pci_match = re.fullmatch(
            r"(?:[0-9A-F]{4}|[0-9A-F]{8}):([0-9A-F]{2}):([0-9A-F]{2})\.([0-9])",
            pci.upper(),
        )
        if pci_match is None:
            raise ValueError(f"invalid nvidia-smi PCI identity: {pci!r}")
        pci = f"0000:{pci_match.group(1)}:{pci_match.group(2)}.{pci_match.group(3)}"
        if not uuid.upper().startswith("GPU-"):
            uuid = "GPU-" + uuid
        rows.append({"index": index, "pci_bus_id": pci, "uuid": uuid})
    if not rows or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("nvidia-smi inventory is empty or has duplicate indices")
    return rows


def _bind_gpu(token: str) -> tuple[str, list[dict[str, str]], str]:
    inventory = _gpu_inventory()
    selected = next(
        (
            row
            for row in inventory
            if row["index"] == token or row["uuid"].casefold() == token.casefold()
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"requested GPU is absent from live inventory: {token}")
    if selected["pci_bus_id"].casefold() == EXCLUDED_PCI.casefold() or selected["uuid"].casefold() == EXCLUDED_UUID.casefold():
        raise ValueError(f"excluded GPU selected: {selected}")
    expected_uuid = APPROVED_GPU_BINDINGS.get(selected["pci_bus_id"])
    if expected_uuid is None or expected_uuid.casefold() != selected["uuid"].casefold():
        raise ValueError(f"GPU PCI/UUID mapping is not in the approved smoke set: {selected}")
    inventory_hash = _canonical_hash({"schema_version": "stage2-s204-gpu-inventory-v1", "rows": inventory})
    return selected["uuid"], inventory, inventory_hash


def _validate_gpu_smoke_artifact(plan: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    """Revalidate the original G2.1 smoke report immediately before execution."""

    ref = plan.get("gpu_smoke_ref")
    declared = plan.get("gpu_smoke_sha256")
    if not isinstance(ref, str) or not ref or not isinstance(declared, str):
        raise ValueError("G2.1 current GPU smoke ref/hash is required for formal execute")
    normalized_ref = ref.replace("\\", "/")
    if normalized_ref.startswith("$DATA_ROOT/"):
        normalized_ref = normalized_ref[len("$DATA_ROOT/") :]
    candidate = Path(normalized_ref)
    path = candidate if candidate.is_absolute() else data_root.joinpath(*PurePosixPath(normalized_ref).parts)
    path = path.resolve()
    try:
        path.relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError("G2.1 smoke ref escapes DATA_ROOT") from error
    if _file_sha256(path) != declared:
        raise ValueError("G2.1 current GPU smoke sha256 drift")
    report = _load(path)
    if report.get("status") != "PASS":
        raise ValueError("G2.1 current GPU smoke report is not PASS")
    if EXCLUDED_PCI not in report.get("excluded_pci_bus_ids", []):
        raise ValueError("G2.1 smoke report does not exclude required PCI")
    excluded = report.get("excluded_device")
    if (
        not isinstance(excluded, Mapping)
        or str(excluded.get("pci_bus_id", "")).casefold() != EXCLUDED_PCI.casefold()
        or str(excluded.get("uuid", "")).casefold() != EXCLUDED_UUID.casefold()
        or excluded.get("scheduled") is not False
    ):
        raise ValueError("G2.1 smoke report excluded device identity drift")
    allowed = report.get("allowed_devices")
    observed = {
        str(item.get("pci_bus_id", "")).upper(): str(item.get("uuid", ""))
        for item in allowed
        if isinstance(item, Mapping)
    } if isinstance(allowed, list) else {}
    expected = {key.upper(): value for key, value in APPROVED_GPU_BINDINGS.items()}
    if {key: value.casefold() for key, value in observed.items()} != {
        key: value.casefold() for key, value in expected.items()
    }:
        raise ValueError("G2.1 smoke report approved PCI/UUID set drift")
    return {
        "ref": normalized_ref,
        "sha256": declared,
        "excluded": dict(excluded),
        "allowed_devices": [dict(item) for item in allowed],
    }


def validate_inputs(g21: Mapping[str, Any], assets: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    if g21.get("status") != "PASS":
        raise ValueError("G2.1 evidence is not PASS")
    if g21.get("artifact_hash") != G21_ARTIFACT:
        raise ValueError("G2.1 artifact identity mismatch")
    smoke = g21.get("current_gpu_smoke", {})
    if not isinstance(smoke, Mapping) or smoke.get("status") != "PASS":
        raise ValueError("G2.1 current GPU smoke is not PASS")
    excluded = smoke.get("excluded_pci_bus_ids", [])
    excluded_device = smoke.get("excluded_device", {})
    if (
        EXCLUDED_PCI not in excluded
        or smoke.get("excluded_scheduled") is not False
        or (
            isinstance(excluded_device, Mapping)
            and str(excluded_device.get("uuid", "")).casefold() != EXCLUDED_UUID.casefold()
        )
    ):
        raise ValueError("required failed GPU exclusion is not bound")
    allowed_devices = smoke.get("allowed_devices")
    if not isinstance(allowed_devices, list):
        raise ValueError("G2.1 approved GPU inventory is missing")
    observed_gpu_bindings = {
        str(item.get("pci_bus_id", "")).upper(): str(item.get("uuid", ""))
        for item in allowed_devices
        if isinstance(item, Mapping)
    }
    expected_gpu_bindings = {key.upper(): value for key, value in APPROVED_GPU_BINDINGS.items()}
    if {
        key: value.casefold() for key, value in observed_gpu_bindings.items()
    } != {key: value.casefold() for key, value in expected_gpu_bindings.items()}:
        raise ValueError("G2.1 approved GPU PCI/UUID inventory drift")
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
    smoke = g21["current_gpu_smoke"]
    if not isinstance(smoke, Mapping):  # validate_inputs already checks this
        raise ValueError("G2.1 current GPU smoke is malformed")
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
        checkpoint_asset_id = checkpoint.get("asset_id", checkpoint.get("logical_asset_id"))
        checkpoint_initialization_id = checkpoint.get("initialization_id")
        checkpoint_architecture = checkpoint.get("architecture", checkpoint.get("model_id"))
        data_asset_id = data.get("logical_asset_id", data.get("asset_id", data.get("dataset_id")))
        if not all(
            isinstance(value, str) and value
            for value in (
                checkpoint_asset_id,
                checkpoint_initialization_id,
                checkpoint_architecture,
                data_asset_id,
                data.get("revision"),
            )
        ):
            raise ValueError(
                "S2.3 checkpoint/data rows must expose asset_id, revision, "
                "initialization_id, architecture, and data identity"
            )
        # Sizing is one independent stream; final A and B are each full-length.
        sizing_draws = candidates[-1]
        final_draws_per_stream = "UNFROZEN"
        fixed_work_units = 3 * sizing_draws
        cells.append(
            {
                "cell_id": cell_id,
                "model_id": checkpoint["model_id"],
                "training_stage": checkpoint["training_stage"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_asset_id": checkpoint_asset_id,
                "checkpoint_initialization_id": checkpoint_initialization_id,
                "checkpoint_architecture": checkpoint_architecture,
                "checkpoint_revision": checkpoint["revision"],
                "parameter_registry_hash": checkpoint["parameter_registry_hash"],
                "data_asset_id": data_asset_id,
                "data_revision": data.get("revision"),
                "data_sequence_length": data["input_sequence_length"],
                "candidate_sample_counts": list(candidates),
                "block_size": block_size,
                "b_ref_status": "UNFROZEN_UNTIL_SIZING_PASS",
                "b_ref": None,
                "minimum_legal_candidate_max_per_stream": sizing_draws,
                "final_sample_count_per_stream": final_draws_per_stream,
                "fixed_work_units_at_candidate_max": fixed_work_units,
                "estimated_seconds_at_candidate_max": fixed_work_units * per_sequence_seconds,
                "progress_path": (output_root / cell_id / "progress.jsonl").as_posix(),
                "progress_path_pattern": (
                    output_root / cell_id / "attempts" / "{fresh-or-resume}-{config.full_hash}" / "progress.jsonl"
                ).as_posix(),
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
        "excluded_uuid": EXCLUDED_UUID,
        "approved_gpu_bindings": dict(APPROVED_GPU_BINDINGS),
        "gpu_smoke_ref": smoke.get("ref"),
        "gpu_smoke_sha256": smoke.get("sha256"),
        "reference_protocol": "independent_reference_sizing_then_one_shot_A_B",
        "optional_stopping": False,
        "replacement_or_resampling": False,
        "candidate_sample_counts": list(candidates),
        "block_size": block_size,
        "b_ref_status": "UNFROZEN_UNTIL_INDEPENDENT_SIZING_PASS",
        "b_ref": None,
        "cell_count": len(cells),
        "cells": cells,
        "total_fixed_work_units_at_candidate_max": total_units,
        "total_estimated_seconds_at_candidate_max": total_units * per_sequence_seconds,
        "estimated_duration_note": "Conservative planning estimate only; measure a legal GPU smoke before launch.",
    }


class _Heartbeat:
    """Append-only liveness stream for a cell; never used as a result commit."""

    def __init__(self, path: Path, cell_id: str, interval: float) -> None:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("heartbeat interval must be finite and positive")
        self.path, self.cell_id, self.interval = path, cell_id, interval
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def _write(self, phase: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": "stage2-s204-heartbeat-v1",
            "cell_id": self.cell_id,
            "phase": phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def __enter__(self) -> "_Heartbeat":
        self._write("STARTED")

        def loop() -> None:
            while not self.stop.wait(self.interval):
                try:
                    self._write("RUNNING")
                except OSError:
                    # The computation must fail on its own; heartbeat failure is
                    # visible to the operator but cannot forge a result.
                    return

        self.thread = threading.Thread(target=loop, name=f"s204-heartbeat-{self.cell_id}", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval))
        self._write("STOPPED")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _publish_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a hash-bound JSON manifest without overwriting an old identity."""

    if path.exists():
        existing = _load(path)
        if existing != dict(value):
            raise RuntimeError(f"IMMUTABLE_JSON_IDENTITY_CONFLICT:{path}")
        return
    _atomic_json(path, value)


def _publish_preflight(output_root: Path, payload: Mapping[str, Any]) -> Path:
    """Publish a content-addressed preflight note without locking retries."""

    digest = _canonical_hash(dict(payload))
    path = output_root / "g2.3-preflight" / digest / "preflight.json"
    _publish_json_once(path, {**dict(payload), "artifact_hash": digest})
    return path


def _g23_gate(
    *,
    plan: Mapping[str, Any],
    metrics_path: Path | None,
    task_result_refs: tuple[str, ...],
    task_result_hashes: Mapping[str, str],
    bundle_hashes: Mapping[str, str],
    output_root: Path,
) -> tuple[str, str]:
    """Validate the complete registered G2.3 precision criteria, fail-closed.

    The specialized reference runner publishes candidate vectors and uncertainty;
    it intentionally does not invent the G2.3 decision.  A separately materialized
    metrics artifact must therefore contain every pre-registered endpoint below.
    Missing or malformed metrics produce a BLOCKED Gate, never a PASS or a relaxed
    threshold.  The function writes both the machine Gate and qualification record
    atomically/immutably so a later retry cannot replace a failed decision.
    """

    from param_importance_nlp.contracts.status import GateRecord, GateStatus

    checked = datetime.now(timezone.utc).isoformat()
    reasons: list[str] = []
    rows: list[Mapping[str, Any]] = []
    evidence_refs = list(task_result_refs)
    preflight_only = metrics_path is None
    metrics_digest = "missing"
    if metrics_path is None:
        reasons.append("G2.3_METRICS_ARTIFACT_REQUIRED")
    else:
        evidence_refs.append(metrics_path.as_posix())
        try:
            metrics = _load(metrics_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            reasons.append(f"G2.3_METRICS_UNREADABLE:{type(error).__name__}")
            metrics = {}
            preflight_only = True
        if metrics:
            if metrics.get("schema_version") != "stage2-g23-reference-evaluation-v1":
                reasons.append("G2.3_METRICS_SCHEMA_MISMATCH")
                preflight_only = True
            declared_hash = metrics.get("artifact_hash")
            if not isinstance(declared_hash, str) or _canonical_hash(
                {key: value for key, value in metrics.items() if key != "artifact_hash"}
            ) != declared_hash:
                reasons.append("G2.3_METRICS_HASH_MISMATCH")
                preflight_only = True
            else:
                metrics_digest = declared_hash
            raw_rows = metrics.get("cells")
            if not isinstance(raw_rows, list):
                reasons.append("G2.3_METRICS_CELLS_REQUIRED")
                preflight_only = True
            else:
                rows = [item for item in raw_rows if isinstance(item, Mapping)]
                expected = tuple(str(item["cell_id"]) for item in plan["cells"])
                observed = tuple(str(item.get("cell_id")) for item in rows)
                if observed != expected:
                    reasons.append("G2.3_METRICS_CELL_ORDER_OR_SET_MISMATCH")
                    preflight_only = True
            calculator = metrics.get("calculator")
            if (
                not isinstance(calculator, Mapping)
                or not isinstance(calculator.get("producer_commit"), str)
                or len(str(calculator.get("producer_commit"))) != 40
                or any(c not in "0123456789abcdef" for c in str(calculator.get("producer_commit")))
                or not isinstance(calculator.get("source_sha256"), str)
                or len(str(calculator.get("source_sha256"))) != 64
                or any(c not in "0123456789abcdef" for c in str(calculator.get("source_sha256")))
            ):
                reasons.append("G2.3_CALCULATOR_PRODUCER_IDENTITY_REQUIRED")
                preflight_only = True
    by_cell = {str(row.get("cell_id")): row for row in rows}
    expected_cells = tuple(str(item["cell_id"]) for item in plan["cells"])
    if len(by_cell) != len(expected_cells):
        reasons.append("G2.3_REQUIRES_ALL_SIX_CELLS")
        preflight_only = True
    for cell_id in expected_cells:
        row = by_cell.get(cell_id)
        if row is None:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            reasons.append(f"{cell_id}:G2.3_METRICS_MAPPING_REQUIRED")
            continue
        if metrics.get("task_result_hash") != task_result_hashes.get(cell_id):
            reasons.append(f"{cell_id}:TASK_RESULT_HASH_BINDING_FAILED")
        if metrics.get("bundle_manifest_sha256") != bundle_hashes.get(cell_id):
            reasons.append(f"{cell_id}:BUNDLE_MANIFEST_HASH_BINDING_FAILED")
        missing = [name for name in G23_REQUIRED_METRICS if name not in metrics]
        if missing:
            reasons.append(f"{cell_id}:missing_metrics={','.join(missing)}")
            continue
        try:
            values = {name: float(metrics[name]) for name in G23_REQUIRED_METRICS}
            checks = {
                "normalized_l1": values["normalized_l1"] <= 0.02,
                "pearson": values["pearson"] >= 0.995,
                "signal_eligible_spearman": values["signal_eligible_spearman"] >= 0.995,
                "layer_module_spearman": values["layer_module_spearman"] >= 0.995,
                "topk_overlap_0_001": values["topk_overlap_0_001"] >= 0.98,
                "topk_overlap_0_01": values["topk_overlap_0_01"] >= 0.98,
                "topk_overlap_0_05": values["topk_overlap_0_05"] >= 0.98,
                "layer_module_delta": values["layer_module_delta"] <= 0.01,
                "h_ref": values["h_ref"] <= values["min_delta_sci"] / 4.0,
                "epsilon_num": values["epsilon_num"] <= values["min_delta_sci"] / 10.0,
            }
            for name, passed in checks.items():
                if not passed or not math.isfinite(values[name]):
                    reasons.append(f"{cell_id}:{name}:THRESHOLD_FAILED")
            for boolean_name in (
                "a_b_interval_covered",
                "bias_cross_interval_covered",
                "ranking_bias_direction",
                "variance_scaling_verified",
                "state_replay_verified",
                "one_shot_complete",
            ):
                if metrics.get(boolean_name) is not True:
                    reasons.append(f"{cell_id}:{boolean_name}:REQUIRED_TRUE")
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"{cell_id}:G2.3_METRICS_NON_NUMERIC")
    status = GateStatus.PASS if not reasons else GateStatus.BLOCKED
    gate = GateRecord(
        G23_GATE_ID,
        2,
        status,
        checked,
        measured={"cell_count": len(by_cell), "criteria_count": len(G23_REQUIRED_METRICS)},
        threshold={"normalized_l1": 0.02, "rank_and_pearson": 0.995, "topk": 0.98, "layer_module_delta": 0.01, "h_ref_divisor": 4.0, "epsilon_num_divisor": 10.0},
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        reasons=tuple(reasons),
    )
    if preflight_only:
        _publish_preflight(
            output_root,
            {
                "schema_version": "stage2-g23-preflight-v1",
                "status": "BLOCKED",
                "formal_eligible": False,
                "reasons": reasons,
                "task_result_hashes": dict(task_result_hashes),
                "bundle_manifest_hashes": dict(bundle_hashes),
            },
        )
        return "BLOCKED", ""
    gate_path = output_root / "g2.3-attempts" / metrics_digest / "g2.3-gate.json"
    _publish_json_once(gate_path, gate.to_dict())
    qualification = {
        "schema_version": "stage2-g23-reference-qualification-v1",
        "status": "QUALIFIED" if status is GateStatus.PASS else "BLOCKED",
        "formal_eligible": status is GateStatus.PASS,
        "gate_ref": gate_path.as_posix(),
        "gate_artifact_hash": gate.artifact_hash,
        "required_cells": list(expected_cells),
        "criteria": list(G23_REQUIRED_METRICS),
        "reasons": list(reasons),
    }
    qualification["artifact_hash"] = _canonical_hash(
        {key: value for key, value in qualification.items() if key != "artifact_hash"}
    )
    _publish_json_once(gate_path.parent / "g2.3-qualification.json", qualification)
    return status.value, gate.artifact_hash


def _runtime_cell_paths(
    plan: Mapping[str, Any], config_paths: tuple[Path, ...], cell_id: str | None
) -> list[tuple[int, Path, str]]:
    cells = list(plan["cells"])
    if cell_id is None:
        if len(config_paths) != len(cells):
            raise ValueError(
                f"--runtime-config requires exactly {len(cells)} paths for six-cell execute"
            )
        selected = list(enumerate(config_paths))
    else:
        indexes = [index for index, item in enumerate(cells) if item["cell_id"] == cell_id]
        if not indexes:
            raise ValueError(f"unknown cell_id: {cell_id}")
        if len(config_paths) != 1:
            raise ValueError("single-cell recovery requires exactly one --runtime-config")
        selected = [(indexes[0], config_paths[0])]
    return [(index, path, str(cells[index]["cell_id"])) for index, path in selected]


def _runtime_cell_configs(
    plan: Mapping[str, Any], config_paths: tuple[Path, ...], cell_id: str | None
) -> list[tuple[Mapping[str, Any], Path, str]]:
    from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2

    result: list[tuple[Mapping[str, Any], Path, str]] = []
    for index, path, selected_cell_id in _runtime_cell_paths(plan, config_paths, cell_id):
        config = _load(path)
        if config.get("task_id") != "stage2.04_reference_target":
            raise ValueError(f"{path}: task_id must be stage2.04_reference_target")
        resolved = ResolvedConfigV2.from_mapping(config)
        base = resolved.base_config
        model = base.section("model")
        data = base.section("data")
        expected = plan["cells"][index]
        if not isinstance(model, Mapping) or not isinstance(data, Mapping):
            raise ValueError(f"{path}: model/data binding is not an object")
        mismatches: list[str] = []
        if str(model.get("asset_id")) != str(expected["checkpoint_asset_id"]):
            mismatches.append("model.asset_id")
        if str(model.get("revision")) != str(expected["checkpoint_revision"]):
            mismatches.append("model.revision")
        if str(model.get("initialization_id")) != str(expected["checkpoint_initialization_id"]):
            mismatches.append("model.initialization_id")
        if str(model.get("architecture")) != str(expected["checkpoint_architecture"]):
            mismatches.append("model.architecture")
        expected_data_asset = expected.get("data_asset_id")
        if expected_data_asset is None or str(data.get("asset_id")) != str(expected_data_asset):
            mismatches.append("data.asset_id")
        expected_data_revision = expected.get("data_revision")
        if expected_data_revision is None or str(data.get("revision")) != str(expected_data_revision):
            mismatches.append("data.revision")
        if int(data.get("sequence_length", -1)) != int(expected["data_sequence_length"]):
            mismatches.append("data.sequence_length")
        output_section = None
        # Each worker owns a distinct task-output subtree.  Sharing a logical
        # output directory would make two cells race the same shard commits.
        try:
            output_section = resolved.section("artifacts")
            output_dir = str(output_section["output_dir"])
            output_parts = set(PurePosixPath(output_dir).parts)
            if str(expected["cell_id"]) not in output_parts:
                mismatches.append("artifacts.output_dir.cell_id")
        except (KeyError, TypeError, ValueError):
            mismatches.append("artifacts.output_dir")
        if mismatches:
            raise ValueError(f"{path}: checkpoint/data identity mismatch: {mismatches}")
        result.append((config, path, selected_cell_id))
    return result


def _validate_task_result_bindings(
    result: Any,
    *,
    config: Any,
    data_root: Path,
    expected: Mapping[str, Any],
) -> str:
    """Re-open every formal task commit and bind its provider registry/bundle to cell."""

    from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact

    if result.task_id != "stage2.04_reference_target" or result.run_intent != "formal":
        raise ValueError("S2.4 TaskRuntime result task/run identity mismatch")
    convergence_payload: Mapping[str, Any] | None = None
    bundle_hash: str | None = None
    for kind, reference in result.artifact_refs.items():
        loaded = load_committed_task_artifact(data_root, str(reference), require_formal=True)
        if loaded.identity.config_hash != config.config_hash or loaded.identity.task_id != result.task_id:
            raise ValueError(f"S2.4 result artifact identity mismatch: {kind}")
        if kind == "reference_convergence_report":
            convergence_payload = loaded.payload
        if kind == "reference_result":
            declared = loaded.payload.get("tensor_bundle_manifest_hash")
            if not isinstance(declared, str) or len(declared) != 64:
                raise ValueError("S2.4 reference_result missing bundle manifest hash")
            bundle_hash = declared
    if convergence_payload is None or bundle_hash is None:
        raise ValueError("S2.4 result is missing convergence/reference bundle commits")
    provider = convergence_payload.get("provider")
    expected_registry = str(expected["parameter_registry_hash"])
    if not isinstance(provider, Mapping):
        raise ValueError("S2.4 convergence report missing provider binding")
    if len(expected_registry) == 64 and provider.get("registry_hash") != expected_registry:
        raise ValueError("S2.4 provider parameter registry drift")
    provenance = provider.get("asset_provenance")
    if not isinstance(provenance, list):
        raise ValueError("S2.4 provider asset provenance missing")
    model_rows = [item for item in provenance if isinstance(item, Mapping)]
    expected_model = str(expected["checkpoint_asset_id"])
    if not any(
        item.get("asset_id") == expected_model or item.get("logical_asset_id") == expected_model
        for item in model_rows
    ):
        raise ValueError("S2.4 result checkpoint asset binding failed")
    expected_data = str(expected["data_asset_id"])
    if not any(
        item.get("asset_id") == expected_data or item.get("logical_asset_id") == expected_data
        for item in model_rows
    ):
        raise ValueError("S2.4 result data asset binding failed")
    if len(expected_registry) == 64 and provider.get("registry_hash") is None:
        raise ValueError("S2.4 convergence report missing provider registry binding")
    return bundle_hash


def _cell_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable model/checkpoint/data identity carried by one cell."""

    return {
        "cell_id": str(expected["cell_id"]),
        "model_id": str(expected["model_id"]),
        "training_stage": str(expected["training_stage"]),
        "checkpoint_id": str(expected["checkpoint_id"]),
        "checkpoint_asset_id": str(expected["checkpoint_asset_id"]),
        "checkpoint_initialization_id": str(expected["checkpoint_initialization_id"]),
        "checkpoint_architecture": str(expected["checkpoint_architecture"]),
        "checkpoint_revision": str(expected["checkpoint_revision"]),
        "parameter_registry_hash": str(expected["parameter_registry_hash"]),
        "data_asset_id": str(expected["data_asset_id"]),
        "data_revision": str(expected["data_revision"]),
        "data_sequence_length": int(expected["data_sequence_length"]),
    }


def _cell_attempt_id(config: Any) -> tuple[str, str, str | None]:
    recovery = config.section("recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError("resolved config recovery section is malformed")
    resume_ref = recovery.get("resume_ref")
    if resume_ref is not None and not isinstance(resume_ref, str):
        raise ValueError("recovery.resume_ref must be a logical string or null")
    mode = "resume" if resume_ref is not None else "fresh"
    # full_hash includes recovery.resume_ref and artifacts output, so a fresh run
    # can never silently reuse a resume attempt (and vice versa).
    return f"{mode}-{config.full_hash}", mode, resume_ref


def execute_with_task_runtime(
    plan: Mapping[str, Any],
    *,
    runtime_config_paths: tuple[Path, ...],
    runtime_environment_path: Path,
    data_root: Path,
    output_root: Path,
    cuda_visible_devices: str,
    cell_id: str | None,
    heartbeat_seconds: float,
) -> list[dict[str, Any]]:
    """Delegate formal work to the existing strict TaskRuntime chain.

    The supplied resolved configs and environment are authoritative.  In
    particular, this function does not synthesize ``FormalExecutionEvidence``,
    Stage 0/G3 records, predecessor commits, draw manifests, or providers.
    """

    allowed = {"0", "2", "3", "4"}
    visible_tokens = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    visible = set(visible_tokens)
    if len(visible_tokens) != 1 or not visible.issubset(allowed) or "1" in visible:
        raise ValueError(
            "each formal process must bind exactly one smoke-approved physical GPU "
            "(0,2,3,4); four-card parallelism is four independent processes; "
            f"PCI {EXCLUDED_PCI} is excluded"
        )
    if cell_id is None:
        raise ValueError("formal execute requires --cell-id; aggregate six independent cell processes")
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be finite and positive")
    gpu_smoke = _validate_gpu_smoke_artifact(plan, data_root)
    selected_gpu_uuid, gpu_inventory, gpu_inventory_hash = _bind_gpu(visible_tokens[0])
    # UUID is the process boundary; physical indices are only an input selector.
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu_uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
    from param_importance_nlp.experiments import build_default_task_runtime
    from param_importance_nlp.runtime import TaskRunStatus, TaskRuntimeEnvironment

    environment = TaskRuntimeEnvironment.from_mapping(_load(runtime_environment_path))
    runtime = build_default_task_runtime(data_root)
    selected = _runtime_cell_configs(plan, runtime_config_paths, cell_id)
    results: list[dict[str, Any]] = []
    for config_wire, config_path, current_cell_id in selected:
        config = ResolvedConfigV2.from_mapping(config_wire)
        expected = next(item for item in plan["cells"] if item["cell_id"] == current_cell_id)
        artifacts = config.section("artifacts")
        recovery = config.section("recovery")
        if not isinstance(artifacts, Mapping) or not isinstance(recovery, Mapping):
            raise ValueError(f"{config_path}: malformed artifacts/recovery section")
        attempt_id, run_kind, resume_ref_text = _cell_attempt_id(config)
        cell_root = output_root / current_cell_id / "attempts" / attempt_id
        status_path = cell_root / "cell-status.json"
        recovery_refs = {
            "resume_ref": resume_ref_text,
            "task_output_dir": str(artifacts["output_dir"]),
            "heartbeat": (cell_root / "progress.jsonl").as_posix(),
        }
        _atomic_json(
            status_path,
            {
                "schema_version": "stage2-s204-cell-status-v2",
                "cell_id": current_cell_id,
                "attempt_id": attempt_id,
                "run_kind": run_kind,
                "config_path": config_path.as_posix(),
                "config_hash": config.config_hash,
                "config_full_hash": config.full_hash,
                "cell_identity": _cell_identity(expected),
                "cell_identity_hash": _canonical_hash(_cell_identity(expected)),
                "gpu": {
                    "requested_token": visible_tokens[0],
                    "selected_uuid": selected_gpu_uuid,
                    "inventory": gpu_inventory,
                    "inventory_sha256": gpu_inventory_hash,
                    "g21_smoke": gpu_smoke,
                },
                "checkpoint_revision": expected["checkpoint_revision"],
                "parameter_registry_hash": expected["parameter_registry_hash"],
                "status": "IN_PROGRESS",
                "formal_provider": "TaskRuntime.stage2.04_reference_target",
                "formal_eligible": False,
                "g2_3_gate": "NOT_RUN",
                "recovery": recovery_refs,
            },
        )
        with _Heartbeat(cell_root / "progress.jsonl", current_cell_id, heartbeat_seconds):
            result = runtime.execute(config, environment=environment)
        bundle_hash = None
        if result.status is TaskRunStatus.PASS:
            bundle_hash = _validate_task_result_bindings(
                result, config=config, data_root=data_root, expected=expected
            )
        wire = result.to_dict()
        task_result_path = cell_root / "task-results" / f"{result.result_hash}.json"
        _publish_json_once(task_result_path, wire)
        status = "COMPLETE" if result.status is TaskRunStatus.PASS else result.status.value
        _atomic_json(
            status_path,
            {
                "schema_version": "stage2-s204-cell-status-v2",
                "cell_id": current_cell_id,
                "attempt_id": attempt_id,
                "run_kind": run_kind,
                "config_path": config_path.as_posix(),
                "config_hash": config.config_hash,
                "config_full_hash": config.full_hash,
                "cell_identity": _cell_identity(expected),
                "cell_identity_hash": _canonical_hash(_cell_identity(expected)),
                "gpu": {
                    "requested_token": visible_tokens[0],
                    "selected_uuid": selected_gpu_uuid,
                    "inventory": gpu_inventory,
                    "inventory_sha256": gpu_inventory_hash,
                    "g21_smoke": gpu_smoke,
                },
                "checkpoint_revision": expected["checkpoint_revision"],
                "parameter_registry_hash": expected["parameter_registry_hash"],
                "status": status,
                "formal_provider": "TaskRuntime.stage2.04_reference_target",
                "formal_eligible": bool(result.formal_eligible),
                "g2_3_gate": "NOT_RUN",
                "task_result_hash": result.result_hash,
                "task_result_ref": task_result_path.as_posix(),
                "bundle_manifest_sha256": bundle_hash,
                "artifact_refs": dict(result.artifact_refs),
                "blockers": [item.to_dict() for item in result.blockers],
                "recovery": recovery_refs,
            },
        )
        results.append(
            {
                "cell_id": current_cell_id,
                "attempt_id": attempt_id,
                "run_kind": run_kind,
                "gpu_inventory_sha256": gpu_inventory_hash,
                "selected_gpu_uuid": selected_gpu_uuid,
                "config_path": config_path.as_posix(),
                "task_result_ref": task_result_path.as_posix(),
                "task_result_hash": result.result_hash,
                "bundle_manifest_sha256": bundle_hash,
                "status": status,
                "formal_eligible": bool(result.formal_eligible),
                "g2_3_gate": "NOT_RUN",
            }
        )
    return results


def aggregate_g23(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    data_root: Path,
    metrics_path: Path | None,
) -> tuple[str, str, Path]:
    """Collect six immutable cell results and invoke the downstream evaluator hook.

    Cell processes never call this function.  It is deliberately a single-writer
    operation: incomplete/ambiguous cells only receive a content-addressed
    preflight note, while a complete set gets one content-addressed G2.3 attempt.
    """

    from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
    from param_importance_nlp.runtime import TaskRunResult, TaskRunStatus

    reasons: list[str] = []
    selected: dict[str, dict[str, Any]] = {}
    task_refs: dict[str, str] = {}
    task_hashes: dict[str, str] = {}
    bundle_hashes: dict[str, str] = {}
    expected_cells = tuple(str(item["cell_id"]) for item in plan["cells"])
    for expected in plan["cells"]:
        current_cell_id = str(expected["cell_id"])
        status_paths = sorted(
            (output_root / current_cell_id).rglob("cell-status.json")
            if (output_root / current_cell_id).exists()
            else []
        )
        complete_rows: list[dict[str, Any]] = []
        for path in status_paths:
            try:
                row = _load(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if row.get("status") != "COMPLETE":
                continue
            if row.get("cell_id") != current_cell_id:
                reasons.append(f"{current_cell_id}:CELL_STATUS_ID_MISMATCH")
                continue
            identity = _cell_identity(expected)
            if row.get("cell_identity") != identity or row.get("cell_identity_hash") != _canonical_hash(identity):
                reasons.append(f"{current_cell_id}:CELL_IDENTITY_BINDING_FAILED")
                continue
            row["_status_path"] = path.as_posix()
            complete_rows.append(row)
        distinct_hashes = {str(row.get("task_result_hash")) for row in complete_rows if row.get("task_result_hash")}
        if not complete_rows:
            reasons.append(f"{current_cell_id}:COMPLETE_TASK_RESULT_REQUIRED")
            continue
        if len(distinct_hashes) != 1:
            reasons.append(f"{current_cell_id}:AMBIGUOUS_COMPLETE_ATTEMPTS")
            continue
        row = sorted(complete_rows, key=lambda item: str(item.get("attempt_id", "")))[0]
        result_ref = row.get("task_result_ref")
        result_hash = row.get("task_result_hash")
        if not isinstance(result_ref, str) or not isinstance(result_hash, str):
            reasons.append(f"{current_cell_id}:TASK_RESULT_REFERENCE_REQUIRED")
            continue
        try:
            result_wire = _load(Path(result_ref))
            result = TaskRunResult.from_mapping(result_wire)
            if result.result_hash != result_hash or result.status is not TaskRunStatus.PASS:
                raise ValueError("result status/hash mismatch")
            config_path = Path(str(row["config_path"]))
            config_wire, _, selected_cell = _runtime_cell_configs(
                plan, (config_path,), current_cell_id
            )[0]
            if selected_cell != current_cell_id:
                raise ValueError("config cell ordering mismatch")
            config = ResolvedConfigV2.from_mapping(config_wire)
            bundle_hash = _validate_task_result_bindings(
                result, config=config, data_root=data_root, expected=expected
            )
            if row.get("bundle_manifest_sha256") != bundle_hash:
                raise ValueError("cell bundle hash mismatch")
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            reasons.append(f"{current_cell_id}:TASK_RESULT_BINDING_FAILED:{type(error).__name__}")
            continue
        task_refs[current_cell_id] = result_ref
        task_hashes[current_cell_id] = result_hash
        bundle_hashes[current_cell_id] = bundle_hash
        selected[current_cell_id] = {
            "status_ref": next(
                str(item["_status_path"])
                for item in complete_rows
                if item.get("attempt_id") == row.get("attempt_id")
            ),
            "task_result_ref": result_ref,
            "task_result_hash": result_hash,
            "bundle_manifest_sha256": bundle_hash,
        }

    if reasons or tuple(selected) != expected_cells:
        if not reasons:
            reasons.append("G2.3_REQUIRES_ALL_SIX_CELLS")
        preflight = _publish_preflight(
            output_root,
            {
                "schema_version": "stage2-g23-aggregate-preflight-v1",
                "status": "BLOCKED",
                "formal_eligible": False,
                "reasons": reasons,
                "required_cells": list(expected_cells),
                "completed_cells": sorted(selected),
                "task_result_hashes": task_hashes,
                "bundle_manifest_hashes": bundle_hashes,
            },
        )
        return "BLOCKED", "", preflight

    gate_status, gate_hash = _g23_gate(
        plan=plan,
        metrics_path=metrics_path,
        task_result_refs=tuple(task_refs[cell] for cell in expected_cells),
        task_result_hashes=task_hashes,
        bundle_hashes=bundle_hashes,
        output_root=output_root,
    )
    summary = {
        "schema_version": "stage2-s204-formal-run-summary-v3",
        "status": "COMPLETE" if gate_status == "PASS" else "BLOCKED_OR_NOT_QUALIFIED",
        "g2_3_gate": gate_status,
        "g2_3_gate_artifact_hash": gate_hash,
        "cells": [selected[cell] for cell in expected_cells],
        "task_result_hashes": task_hashes,
        "bundle_manifest_hashes": bundle_hashes,
        "provider_entry": "TaskRuntime.stage2.04_reference_target",
        "excluded_pci": EXCLUDED_PCI,
    }
    summary_hash = _canonical_hash(summary)
    summary_path = output_root / "aggregate-attempts" / summary_hash / "formal-run-summary.json"
    _publish_json_once(summary_path, {**summary, "artifact_hash": summary_hash})
    return gate_status, gate_hash, summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S2.4 formal reference plan/preflight")
    parser.add_argument("--g21-evidence", type=Path, required=True)
    parser.add_argument("--asset-resolution", type=Path, required=True)
    parser.add_argument("--data-range", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true", help="validate and emit plan; never run gradients")
    mode.add_argument("--execute", action="store_true", help="run one real offline HF fixed-state reference cell")
    mode.add_argument("--aggregate", action="store_true", help="aggregate six immutable cells and evaluate G2.3")
    parser.add_argument("--data-root", type=Path, help="server DATA_ROOT; required by --execute/--aggregate")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        action="append",
        default=[],
        help="resolved formal Stage2.04 config; execute requires exactly one for --cell-id",
    )
    parser.add_argument(
        "--runtime-environment",
        type=Path,
        help="hash-bound TaskRuntimeEnvironment with formal/G3/predecessor evidence",
    )
    parser.add_argument(
        "--g23-metrics",
        type=Path,
        help="hash-bound metrics artifact from the independent G2.3 evaluator",
    )
    parser.add_argument("--cell-id", help="execute only one checkpoint cell (recovery/debug)")
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="exactly one smoke-approved physical index per process: 0,2,3,4; 0000:50:00.0 excluded",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--candidate-sizes", type=int, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--per-sequence-seconds", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        g21, assets, data = _load(args.g21_evidence), _load(args.asset_resolution), _load(args.data_range)
        plan = build_plan(
            g21,
            assets,
            data,
            output_root=args.output_root,
            candidates=tuple(args.candidate_sizes),
            block_size=args.block_size,
            per_sequence_seconds=args.per_sequence_seconds,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        plan_hash = _canonical_hash(plan)
        output = args.output_root / "plans" / plan_hash / "s204-formal-plan.json"
        _publish_json_once(output, plan)
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
        print(f"plan_path={output}")
        if args.execute:
            if args.data_root is None or args.runtime_environment is None:
                raise ValueError("--execute requires --data-root and --runtime-environment")
            if args.cell_id is None or len(args.runtime_config) != 1:
                raise ValueError("--execute requires --cell-id and exactly one --runtime-config")
            results = execute_with_task_runtime(
                plan,
                runtime_config_paths=tuple(args.runtime_config),
                runtime_environment_path=args.runtime_environment,
                data_root=args.data_root,
                output_root=args.output_root,
                cuda_visible_devices=args.cuda_visible_devices,
                cell_id=args.cell_id,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            print(json.dumps({"mode": "execute", "cells": results}, ensure_ascii=False, sort_keys=True))
            if not all(item["status"] == "COMPLETE" for item in results):
                return 3
        elif args.aggregate:
            if args.data_root is None:
                raise ValueError("--aggregate requires --data-root")
            if args.cell_id is not None or args.runtime_config or args.runtime_environment is not None:
                raise ValueError("--aggregate accepts no cell/config/environment arguments")
            gate_status, gate_hash, summary_path = aggregate_g23(
                plan,
                output_root=args.output_root,
                data_root=args.data_root,
                metrics_path=args.g23_metrics,
            )
            print(
                json.dumps(
                    {
                        "mode": "aggregate",
                        "g2_3_gate": gate_status,
                        "g2_3_gate_artifact_hash": gate_hash,
                        "summary_path": summary_path.as_posix(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            if gate_status != "PASS":
                return 3
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"S2.4 preflight blocked: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
