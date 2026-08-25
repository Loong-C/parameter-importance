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

from param_importance_nlp.contracts.g21_formal_handoff import (
    ALLOWED_DEVICES as G21_ALLOWED_DEVICES,
    EXCLUDED_PCI as G21_EXCLUDED_PCI,
    EXCLUDED_UUID as G21_EXCLUDED_UUID,
    G21FormalHandoffError,
    load_g21_formal_handoff,
)
from param_importance_nlp.experiments import (
    AssetResolutionManifest,
    DataRangeManifest,
    validate_formal_asset_identity,
)
from param_importance_nlp.experiments.stage2_s204_ids import (
    canonical_cell_id,
    cell_path_component,
)


G21_ARTIFACT = "259831e2a1b16afbbef34c9cea602e636756b0f6173d1a8f4c32ec554c653f79"
ASSET_DIGEST = "f57decd5cf00e69e45ab2f02c994abb202f5c614e1441acb8aebcb1807ff76ee"
DATA_DIGEST = "df8eeac5178305d409cf6128ac5d5648567aae895592c79fa21542e84a28e0f1"
# G2.1 is the sole hardware identity authority.  Numeric nvidia-smi indices
# are runtime selectors only and deliberately do not appear in this contract.
EXCLUDED_PCI = G21_EXCLUDED_PCI
EXCLUDED_UUID = G21_EXCLUDED_UUID
APPROVED_GPU_BINDINGS = dict(G21_ALLOWED_DEVICES)
DEFAULT_CANDIDATES = (512, 1024, 2048, 4096, 8192, 16384)
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

# S2.3 deliberately does not duplicate the G3 logical asset/config fields in
# each checkpoint row.  This is the one formal selection/config mapping used by
# this launcher.  It is derived from the frozen model/stage/step selection, not
# from optional ad-hoc fields in a manifest (and never from a fixture fallback).
_FORMAL_BASE_REVISIONS = {
    "pythia-14m": "56079904bb80b7f36d3b794089f146e7a4d6efae",
    "pythia-31m-deduped": "73628c85dd9d12d43c07be77ebcf10cef5fd9660",
}
_EXECUTION_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _repository_root() -> Path:
    """Return the repository containing this launcher, never DATA_ROOT."""

    return Path(__file__).resolve().parents[2]


def _validate_execution_lineage(execution_commit: object) -> dict[str, Any]:
    """Fail closed unless this launcher runs from the exact clean detached HEAD.

    This is the S2.4 launcher execution identity.  It deliberately does not
    inspect or overwrite ``FormalExecutionEvidence.metadata.execution_commit``
    from the upstream authorization producer.
    """

    if not isinstance(execution_commit, str) or _EXECUTION_COMMIT_RE.fullmatch(execution_commit) is None:
        raise ValueError("S2.4 execution_commit must be exactly 40 lowercase hex characters")
    repository = _repository_root()
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0 or head.stdout.strip() != execution_commit:
        raise ValueError(
            f"S2.4 EXECUTION_COMMIT_HEAD_MISMATCH:{head.stdout.strip() or '<unavailable>'}"
        )
    branch = subprocess.run(
        ["git", "-C", str(repository), "symbolic-ref", "--quiet", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if branch.returncode == 0 and branch.stdout.strip():
        raise ValueError(f"S2.4 EXECUTION_COMMIT_REQUIRES_DETACHED_HEAD:{branch.stdout.strip()}")
    if branch.returncode not in {0, 1}:
        raise ValueError("S2.4 EXECUTION_COMMIT_DETACHED_STATE_UNAVAILABLE")
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise ValueError("S2.4 EXECUTION_COMMIT_WORKTREE_STATE_UNAVAILABLE")
    if status.stdout.strip():
        raise ValueError("S2.4 EXECUTION_COMMIT_WORKTREE_NOT_CLEAN")
    return {
        "role": "stage2.04_s204_launcher_execution",
        "execution_commit": execution_commit,
        "repository": repository.as_posix(),
        "detached": True,
        "worktree_clean": True,
    }


def _formal_logical_identity(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    model_id = checkpoint.get("model_id")
    stage = checkpoint.get("training_stage")
    step = checkpoint.get("training_step")
    revision = checkpoint.get("revision")
    checkpoint_id = checkpoint.get("checkpoint_id")
    if not all(isinstance(value, str) and value for value in (model_id, stage, revision, checkpoint_id)):
        raise ValueError("formal checkpoint row identity is incomplete")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("formal checkpoint row training_step is invalid")
    base_revision = _FORMAL_BASE_REVISIONS.get(model_id)
    if base_revision is None:
        raise ValueError(f"formal checkpoint model is not selected: {model_id}")
    # Bind the launcher to the immutable S2.3 checkpoint root and identity.
    # Trained checkpoints carry a revision-qualified root basename; deriving
    # ``model-stepN`` here silently loses that identity.
    root_ref = checkpoint.get("root_ref")
    if not isinstance(root_ref, str) or not root_ref:
        raise ValueError("formal checkpoint root_ref is incomplete")
    expected_asset_id = PurePosixPath(root_ref).name
    expected_architecture = model_id
    initialization_id = checkpoint_id
    return {
        "asset_id": expected_asset_id,
        "initialization_id": initialization_id,
        "architecture": expected_architecture,
        "input_checkpoint_id": checkpoint_id,
    }


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


_GPU_QUERY_FIELDS = (
    "index,pci.bus_id,uuid,memory.used,memory.total,utilization.gpu,"
    "ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,"
    "gpu_recovery_action"
)


def _gpu_inventory() -> list[dict[str, Any]]:
    """Read identity plus health/idle fields from the live nvidia-smi inventory."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={_GPU_QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"nvidia-smi inventory failed: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for fields in csv.reader(line for line in completed.stdout.splitlines() if line.strip()):
        if len(fields) != 9:
            raise ValueError(f"invalid nvidia-smi inventory row: {fields!r}")
        index, pci, uuid, memory_used, memory_total, utilization, ecc_volatile, ecc_aggregate, recovery = (
            item.strip() for item in fields
        )
        if not index.isdigit() or not pci or not uuid:
            raise ValueError(f"invalid nvidia-smi inventory identity: {fields!r}")
        pci_match = re.fullmatch(
            r"(?:[0-9A-F]{4}|[0-9A-F]{8}):([0-9A-F]{2}):([0-9A-F]{2})\.([0-9])",
            pci.upper(),
        )
        if pci_match is None:
            raise ValueError(f"invalid nvidia-smi PCI identity: {pci!r}")
        # nvidia-smi may print a four- or eight-digit PCI domain.  The system
        # contract uses the canonical four-digit domain.
        pci = f"0000:{pci_match.group(1)[-4:]}:{pci_match.group(2)}.{pci_match.group(3)}"
        if not uuid.upper().startswith("GPU-"):
            uuid = "GPU-" + uuid
        for value, label in (
            (memory_used, "memory.used"),
            (memory_total, "memory.total"),
            (utilization, "utilization.gpu"),
            (ecc_volatile, "ecc.errors.uncorrected.volatile.total"),
            (ecc_aggregate, "ecc.errors.uncorrected.aggregate.total"),
        ):
            try:
                float(value)
            except ValueError as error:
                raise ValueError(f"invalid nvidia-smi {label}: {value!r}") from error
        rows.append(
            {
                "index": index,
                "pci_bus_id": pci,
                "uuid": uuid,
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
                "utilization_gpu_percent": utilization,
                "ecc_uncorrected_volatile": ecc_volatile,
                "ecc_uncorrected_aggregate": ecc_aggregate,
                "gpu_recovery_action": recovery,
            }
        )
    if not rows or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("nvidia-smi inventory is empty or has duplicate indices")
    return rows


def _gpu_compute_apps() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"nvidia-smi compute-app inventory failed: {completed.stderr.strip()}")
    apps: list[dict[str, str]] = []
    for fields in csv.reader(line for line in completed.stdout.splitlines() if line.strip()):
        if len(fields) != 3:
            raise ValueError(f"invalid nvidia-smi compute-app row: {fields!r}")
        pid, name, uuid = (item.strip() for item in fields)
        if not pid.isdigit() or not uuid:
            raise ValueError(f"invalid nvidia-smi compute-app identity: {fields!r}")
        apps.append({"pid": pid, "process_name": name, "gpu_uuid": uuid})
    return apps


def _validate_live_gpu_health(
    inventory: list[dict[str, Any]],
    apps: list[dict[str, str]],
    *,
    selected: Mapping[str, Any],
) -> None:
    """Validate the complete identity set and health of one bound GPU.

    The excluded device is intentionally part of the identity audit, but it is
    not a candidate device and therefore must not be subjected to the healthy
    / idle checks below.  Likewise, processes on another GPU must not make a
    concurrently launched cell fail its selected-device preflight.
    """

    expected = {pci.casefold(): uuid.casefold() for pci, uuid in APPROVED_GPU_BINDINGS.items()}
    expected[EXCLUDED_PCI.casefold()] = EXCLUDED_UUID.casefold()
    observed: dict[str, str] = {}
    for row in inventory:
        pci = str(row.get("pci_bus_id", "")).casefold()
        uuid = str(row.get("uuid", "")).casefold()
        if pci in expected:
            if pci in observed and observed[pci] != uuid:
                raise ValueError(f"duplicate live PCI identity: {row}")
            observed[pci] = uuid
    if observed != expected:
        raise ValueError(
            "live GPU inventory must contain the complete approved+excluded PCI/UUID set"
        )
    pci_ids = [str(row.get("pci_bus_id", "")).casefold() for row in inventory]
    if len(set(pci_ids)) != len(pci_ids):
        raise ValueError("live GPU inventory contains duplicate PCI identities")
    if len({str(row.get("uuid", "")).casefold() for row in inventory}) != len(inventory):
        raise ValueError("live GPU inventory contains duplicate UUIDs")
    selected_pci = str(selected.get("pci_bus_id", "")).casefold()
    selected_uuid = str(selected.get("uuid", "")).casefold()
    if (
        not selected_pci
        or not selected_uuid
        or selected_pci == EXCLUDED_PCI.casefold()
        or selected_uuid == EXCLUDED_UUID.casefold()
    ):
        raise ValueError(f"excluded GPU selected: {selected}")
    if selected_pci not in expected or expected[selected_pci] != selected_uuid:
        raise ValueError(f"selected GPU PCI/UUID mapping is not smoke-approved: {selected}")
    selected_rows = [
        row
        for row in inventory
        if str(row.get("pci_bus_id", "")).casefold() == selected_pci
        and str(row.get("uuid", "")).casefold() == selected_uuid
    ]
    if len(selected_rows) != 1:
        raise ValueError(f"selected GPU identity is not unique in live inventory: {selected}")
    selected_apps = [
        app
        for app in apps
        if str(app.get("gpu_uuid", "")).strip().casefold() == selected_uuid
    ]
    if selected_apps:
        raise ValueError(
            f"selected GPU inventory is not idle; compute apps present: {selected_apps}"
        )
    row = selected_rows[0]
    try:
        memory_used = float(row["memory_used_mib"])
        memory_total = float(row["memory_total_mib"])
        utilization = float(row["utilization_gpu_percent"])
        ecc_volatile = float(row["ecc_uncorrected_volatile"])
        ecc_aggregate = float(row["ecc_uncorrected_aggregate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"live GPU health fields missing: {row}") from error
    if memory_total <= 0.0 or memory_used != 0.0 or utilization != 0.0:
        raise ValueError(f"GPU is not idle: {row}")
    if ecc_volatile != 0.0 or ecc_aggregate != 0.0:
        raise ValueError(f"GPU ECC health is not clean: {row}")
    recovery = str(row.get("gpu_recovery_action", "")).strip().casefold()
    if recovery not in {"none", "0", "n/a", "na"}:
        raise ValueError(f"GPU recovery/health state is not clean: {row}")


def _bind_gpu(token: str) -> tuple[str, list[dict[str, Any]], str]:
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
    selected_pci = selected["pci_bus_id"].casefold()
    expected_uuid = next(
        (
            uuid
            for pci, uuid in APPROVED_GPU_BINDINGS.items()
            if pci.casefold() == selected_pci
        ),
        None,
    )
    if expected_uuid is None or expected_uuid.casefold() != selected["uuid"].casefold():
        raise ValueError(f"GPU PCI/UUID mapping is not in the approved smoke set: {selected}")
    apps = _gpu_compute_apps()
    _validate_live_gpu_health(inventory, apps, selected=selected)
    for row in inventory:
        row["compute_apps"] = [dict(item) for item in apps]
    inventory_hash = _canonical_hash(
        {
            "schema_version": "stage2-s204-gpu-inventory-v2",
            "rows": inventory,
            "compute_apps": apps,
        }
    )
    return selected["uuid"], inventory, inventory_hash


def _validate_gpu_smoke_artifact(plan: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    """Revalidate the original G2.1 smoke report immediately before execution."""

    ref = plan.get("gpu_smoke_ref")
    declared = plan.get("gpu_smoke_sha256")
    if not isinstance(ref, str) or not ref or not isinstance(declared, str):
        raise ValueError("G2.1 current GPU smoke ref/hash is required for formal execute")
    declared = _require_sha256(declared, "G2.1 current GPU smoke sha256")
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
    if (
        report.get("schema_version") != "stage2-s202-current-gpu-smoke-v1"
        or report.get("status") != "PASS"
        or report.get("atomic_publication") is not True
    ):
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


def _formal_asset_manifests(
    assets: Mapping[str, Any], data: Mapping[str, Any]
) -> tuple[AssetResolutionManifest, DataRangeManifest]:
    """Load both independent S2.3 manifest shapes before projecting a plan."""

    try:
        asset_manifest = AssetResolutionManifest.from_mapping(assets)
        data_manifest = DataRangeManifest.from_mapping(data)
        validate_formal_asset_identity(asset_manifest)
    except (TypeError, ValueError) as error:
        raise ValueError(f"S2.3 formal manifest loader rejected input: {error}") from error
    if asset_manifest.digest != ASSET_DIGEST:
        raise ValueError("S2.3 asset digest mismatch")
    if data_manifest.digest != DATA_DIGEST:
        raise ValueError("S2.3 data digest mismatch")
    if data_manifest.to_dict() != asset_manifest.data_range.to_dict():
        raise ValueError("S2.3 data range differs between independent manifests")
    if asset_manifest.status != "READY" or not asset_manifest.checkpoint_matrix_complete:
        raise ValueError("S2.3 does not contain the six ready checkpoints")
    if any(not checkpoint.ready for checkpoint in asset_manifest.checkpoints):
        raise ValueError("S2.3 checkpoint matrix contains a non-ready checkpoint")
    return asset_manifest, data_manifest


def validate_inputs(
    g21: Mapping[str, Any],
    assets: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    g21_path: Path | None = None,
    data_root: Path | None = None,
) -> None:
    # The CLI calls load_g21_formal_handoff before this projection.  Keeping an
    # optional path here also makes direct callers use the canonical loader
    # instead of treating a hand-written mapping as a formal handoff.
    if g21_path is not None:
        try:
            loaded = load_g21_formal_handoff(g21_path, data_root=data_root)
        except G21FormalHandoffError as error:
            raise ValueError(f"G2.1 formal handoff rejected: {error}") from error
        if dict(loaded) != dict(g21):
            raise ValueError("G2.1 loaded handoff differs from caller mapping")
    if g21.get("status") != "PASS":
        raise ValueError("G2.1 evidence is not PASS")
    if g21.get("artifact_hash") != G21_ARTIFACT:
        raise ValueError("G2.1 artifact identity mismatch")
    smoke = g21.get("current_gpu_smoke", {})
    if not isinstance(smoke, Mapping) or smoke.get("status") != "PASS":
        raise ValueError("G2.1 current GPU smoke is not PASS")
    excluded = smoke.get("excluded_pci_bus_ids", [])
    # The canonical handoff intentionally omits the raw excluded-device object;
    # only the bound raw smoke report is required to carry it.  If a mapping
    # does include the field, validate it, but absence is not a failure here.
    excluded_device = smoke.get("excluded_device")
    excluded_device_valid = True
    if "excluded_device" in smoke:
        excluded_device_valid = (
            isinstance(excluded_device, Mapping)
            and str(excluded_device.get("pci_bus_id", EXCLUDED_PCI)).casefold()
            == EXCLUDED_PCI.casefold()
            and str(excluded_device.get("uuid", "")).casefold()
            == EXCLUDED_UUID.casefold()
        )
    if (
        EXCLUDED_PCI not in excluded
        or smoke.get("excluded_scheduled") is not False
        or not excluded_device_valid
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
    _formal_asset_manifests(assets, data)


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
    asset_manifest, data_manifest = _formal_asset_manifests(assets, data)
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
    rows = asset_manifest.checkpoints
    for checkpoint_record in rows:
        checkpoint = checkpoint_record.to_dict()
        identity = _formal_logical_identity(checkpoint)
        cell_id = canonical_cell_id(checkpoint_record.model_id, checkpoint_record.training_stage)
        if (
            not cell_id
            or "\\" in cell_id
            or PurePosixPath(cell_id).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(cell_id).parts)
            or len(PurePosixPath(cell_id).parts) != 1
        ):
            raise ValueError("formal checkpoint_id must be one safe cell path component")
        checkpoint_asset_id = identity["asset_id"]
        checkpoint_initialization_id = identity["initialization_id"]
        checkpoint_architecture = identity["architecture"]
        input_checkpoint_id = identity["input_checkpoint_id"]
        data_asset_id = "pile-selected-prefix"
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
                "input_checkpoint_id": input_checkpoint_id,
                "checkpoint_asset_id": checkpoint_asset_id,
                "checkpoint_initialization_id": checkpoint_initialization_id,
                "checkpoint_architecture": checkpoint_architecture,
                "checkpoint_revision": checkpoint["revision"],
                "checkpoint_root_ref": checkpoint["root_ref"],
                "checkpoint_manifest_ref": checkpoint["manifest_ref"],
                "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
                "parameter_registry_hash": checkpoint["parameter_registry_hash"],
                "data_asset_id": data_asset_id,
                "data_revision": data_manifest.revision,
                "data_sequence_length": data_manifest.input_sequence_length,
                "candidate_sample_counts": list(candidates),
                "block_size": block_size,
                "b_ref_status": "UNFROZEN_UNTIL_SIZING_PASS",
                "b_ref": None,
                "minimum_legal_candidate_max_per_stream": sizing_draws,
                "final_sample_count_per_stream": final_draws_per_stream,
                "fixed_work_units_at_candidate_max": fixed_work_units,
                "estimated_seconds_at_candidate_max": fixed_work_units * per_sequence_seconds,
                "progress_path": (
                    output_root / cell_path_component(cell_id) / "progress.jsonl"
                ).as_posix(),
                "progress_path_pattern": (
                    output_root
                    / cell_path_component(cell_id)
                    / "attempts"
                    / "{fresh-or-resume}-{config.full_hash}"
                    / "progress.jsonl"
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
        "logical_identity_mapping": "formal-stage2-checkpoint-selection-v1",
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
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _publish_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a hash-bound JSON manifest without overwriting an old identity."""

    if path.exists():
        existing = _load(path)
        if existing != dict(value):
            raise RuntimeError(f"IMMUTABLE_JSON_IDENTITY_CONFLICT:{path}")
        return
    _atomic_json(path, value)


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one fsync'd attempt event; existing events are never rewritten."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": "stage2-s204-attempt-event-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **dict(payload),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class _AttemptLease:
    """Exclusive lease for one resolved TaskRuntime artifacts.output_dir."""

    def __init__(self, output_dir: Path, *, cell_id: str, attempt_id: str) -> None:
        self.output_dir = output_dir.resolve()
        self.cell_id = cell_id
        self.attempt_id = attempt_id
        self.path = self.output_dir / ".stage2-s204-lease.json"
        self._acquired = False

    def __enter__(self) -> "_AttemptLease":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "stage2-s204-lease-v1",
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with self.path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise RuntimeError(
                f"S2.4_TASK_OUTPUT_LEASE_HELD:{self.output_dir.as_posix()}"
            ) from error
        self._acquired = True
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if not self._acquired:
            return
        if exc_type is not None:
            # Preserve the lease after an interrupted/failed attempt.  An
            # operator must inspect the append-only events before removing it
            # and starting a distinct resume attempt.
            self._acquired = False
            return
        try:
            owner = _load(self.path)
            if owner.get("cell_id") != self.cell_id or owner.get("attempt_id") != self.attempt_id:
                raise RuntimeError("S2.4_TASK_OUTPUT_LEASE_OWNER_DRIFT")
            # Releasing the lease is safe only after the caller has published its
            # immutable final-status/result.  A crash leaves the lease for an
            # operator-reviewed resume instead of allowing a competing writer.
            self.path.unlink()
        finally:
            self._acquired = False


def _resolve_data_root_ref(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise ValueError(f"{field}: invalid logical reference")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"{field}: reference escapes DATA_ROOT")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field}: reference escapes DATA_ROOT") from error
    return candidate


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
    # This launcher is only the execution/lineage boundary.  It must not turn
    # caller-supplied numeric JSON into a scientific Gate; the dedicated
    # output-derived evaluator owns that decision and will replace this
    # fail-closed adapter during integration.
    if not preflight_only:
        reasons.append("G2.3_OUTPUT_DERIVED_EVALUATOR_NOT_BOUND")
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
        identity = base.section("identity")
        model = base.section("model")
        data = base.section("data")
        expected = plan["cells"][index]
        if not isinstance(identity, Mapping) or not isinstance(model, Mapping) or not isinstance(data, Mapping):
            raise ValueError(f"{path}: identity/model/data binding is not an object")
        mismatches: list[str] = []
        if str(identity.get("input_checkpoint_id")) != str(expected["input_checkpoint_id"]):
            mismatches.append("identity.input_checkpoint_id")
        if str(identity.get("stage")) != "2":
            mismatches.append("identity.stage")
        if identity.get("task") != "stage2.04_reference_target":
            mismatches.append("identity.task")
        if identity.get("run_intent") != "formal":
            mismatches.append("identity.run_intent")
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
            output_parts = PurePosixPath(output_dir).parts
            planned_cell_components = {
                cell_path_component(str(item["cell_id"])) for item in plan["cells"]
            }
            expected_component = cell_path_component(str(expected["cell_id"]))
            if (
                expected_component not in output_parts
                or any(
                    item != expected_component and item in output_parts
                    for item in planned_cell_components
                )
            ):
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
    from param_importance_nlp.runtime.tensor_bundle import load_tensor_bundle
    from param_importance_nlp.experiments.stage23_task_runners import _vector_digest

    task_definition = config.task_definition
    execution = config.section("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("S2.4 config execution section is malformed")
    expected_formal = config.run_intent == "formal" and not bool(execution.get("dry_run"))
    if (
        result.task_id != "stage2.04_reference_target"
        or result.task_id != config.task_id
        or result.stage != task_definition.stage
        or result.runner_kind is not task_definition.runner_kind
        or result.run_intent != config.run_intent
        or result.run_intent != "formal"
        or result.config_hash != config.config_hash
        or result.formal_eligible is not expected_formal
        or result.recovery_mode is not task_definition.recovery_mode
    ):
        raise ValueError("S2.4 TaskRuntime result task/run identity mismatch")
    convergence_payload: Mapping[str, Any] | None = None
    reference_payload: Mapping[str, Any] | None = None
    bundle_hash: str | None = None
    if set(result.artifact_refs) != {
        "reference_result",
        "reference_convergence_report",
        "gate_record",
    }:
        raise ValueError("S2.4 TaskRuntime result artifact kinds are incomplete or drifted")
    for kind, reference in result.artifact_refs.items():
        loaded = load_committed_task_artifact(data_root, str(reference), require_formal=True)
        if (
            loaded.identity.config_hash != config.config_hash
            or loaded.identity.task_id != result.task_id
            or loaded.run_intent != result.run_intent
            or loaded.identity.formal_eligible is not result.formal_eligible
        ):
            raise ValueError(f"S2.4 result artifact identity mismatch: {kind}")
        if kind == "reference_convergence_report":
            convergence_payload = loaded.payload
        if kind == "reference_result":
            reference_payload = loaded.payload
            declared = loaded.payload.get("tensor_bundle_manifest_hash")
            if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
                raise ValueError("S2.4 reference_result missing bundle manifest hash")
            bundle_hash = declared
    if convergence_payload is None or reference_payload is None or bundle_hash is None:
        raise ValueError("S2.4 result is missing convergence/reference bundle commits")
    bundle_ref = reference_payload.get("tensor_bundle_ref")
    if not isinstance(bundle_ref, str) or not bundle_ref:
        raise ValueError("S2.4 reference_result missing tensor bundle ref")
    artifacts = config.section("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("output_dir"), str):
        raise ValueError("S2.4 config artifacts.output_dir is malformed")
    task_output_root = _resolve_data_root_ref(
        data_root, str(artifacts["output_dir"]), field="artifacts.output_dir"
    )
    bundle_path = _resolve_data_root_ref(
        task_output_root, bundle_ref, field="reference_result.tensor_bundle_ref"
    )
    try:
        bundle_state, bundle = load_tensor_bundle(bundle_path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"S2.4 tensor bundle cannot be reloaded: {error}") from error
    if bundle.manifest_sha256 != bundle_hash or not isinstance(bundle_state, Mapping):
        raise ValueError("S2.4 tensor bundle manifest/root binding failed")
    for vector_name in ("bias_reference", "cross_reference", "ranking_reference"):
        vector = bundle_state.get(vector_name)
        declared_vector_hash = reference_payload.get(f"{vector_name}_hash")
        if not isinstance(vector, Mapping) or not isinstance(declared_vector_hash, str):
            raise ValueError(f"S2.4 tensor bundle missing {vector_name} or hash")
        if _vector_digest(vector) != declared_vector_hash:
            raise ValueError(f"S2.4 {vector_name} vector hash mismatch")
    metadata = reference_payload.get("metadata")
    sequence_variance = bundle_state.get("sequence_variance")
    if not isinstance(metadata, Mapping) or not isinstance(sequence_variance, Mapping):
        raise ValueError("S2.4 tensor bundle missing sequence variance metadata")
    sequence_hash = metadata.get("sequence_variance_hash")
    if not isinstance(sequence_hash, str) or _vector_digest(sequence_variance) != sequence_hash:
        raise ValueError("S2.4 sequence variance hash mismatch")
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
    # G3 qualifies one immutable step-0 model per architecture.  The selected
    # trained checkpoint remains independently bound by the S2.3 six-cell
    # manifest and the result's checkpoint_identity; it is not a fabricated G3
    # logical asset entry.
    expected_base_model = f"{expected['model_id']}-step0"
    if not any(item.get("logical_asset_id") == expected_base_model for item in model_rows):
        raise ValueError("S2.4 result G3 base-model asset binding failed")
    checkpoint_identity = reference_payload.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError("S2.4 reference_result checkpoint identity missing")
    checkpoint_fields = {
        "checkpoint_id": expected["checkpoint_id"],
        "checkpoint_asset_id": expected["checkpoint_asset_id"],
        "checkpoint_revision": expected["checkpoint_revision"],
        "checkpoint_hash": expected["checkpoint_manifest_sha256"],
    }
    if any(checkpoint_identity.get(key) != value for key, value in checkpoint_fields.items()):
        raise ValueError("S2.4 result checkpoint identity binding failed")
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
        "input_checkpoint_id": str(expected["input_checkpoint_id"]),
        "checkpoint_asset_id": str(expected["checkpoint_asset_id"]),
        "checkpoint_initialization_id": str(expected["checkpoint_initialization_id"]),
        "checkpoint_architecture": str(expected["checkpoint_architecture"]),
        "checkpoint_revision": str(expected["checkpoint_revision"]),
        "checkpoint_manifest_sha256": str(expected["checkpoint_manifest_sha256"]),
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
    execution_commit: str | None = None,
) -> list[dict[str, Any]]:
    """Delegate formal work to the existing strict TaskRuntime chain.

    The supplied resolved configs and environment are authoritative.  In
    particular, this function does not synthesize ``FormalExecutionEvidence``,
    Stage 0/G3 records, predecessor commits, draw manifests, or providers.
    """

    data_root = data_root.resolve()
    output_root = output_root.resolve()

    if not isinstance(cuda_visible_devices, str):
        raise ValueError("exactly one current nvidia-smi GPU index or approved GPU UUID is required")
    visible_tokens = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    if len(visible_tokens) != 1:
        raise ValueError(
            "each formal process must bind exactly one current nvidia-smi GPU index "
            "or approved GPU UUID; four-card parallelism is four independent processes; "
            f"PCI {EXCLUDED_PCI} is excluded"
        )
    if cell_id is None:
        raise ValueError("formal execute requires --cell-id; aggregate six independent cell processes")
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be finite and positive")
    execution_lineage = _validate_execution_lineage(execution_commit)
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
        config_path = config_path.resolve()
        config = ResolvedConfigV2.from_mapping(config_wire)
        expected = next(item for item in plan["cells"] if item["cell_id"] == current_cell_id)
        artifacts = config.section("artifacts")
        recovery = config.section("recovery")
        if not isinstance(artifacts, Mapping) or not isinstance(recovery, Mapping):
            raise ValueError(f"{config_path}: malformed artifacts/recovery section")
        attempt_id, run_kind, resume_ref_text = _cell_attempt_id(config)
        cell_root = output_root / cell_path_component(current_cell_id) / "attempts" / attempt_id
        event_path = cell_root / "attempt-events.jsonl"
        final_status_path = cell_root / "final-status.json"
        task_output_dir = _resolve_data_root_ref(
            data_root, str(artifacts["output_dir"]), field="artifacts.output_dir"
        )
        recovery_refs = {
            "resume_ref": resume_ref_text,
            "task_output_dir": str(artifacts["output_dir"]),
            "heartbeat": (cell_root / "progress.jsonl").as_posix(),
        }
        with _AttemptLease(task_output_dir, cell_id=current_cell_id, attempt_id=attempt_id):
            _append_event(
                event_path,
                {
                    "event": "STARTED",
                    "execution_commit": execution_lineage["execution_commit"],
                    "execution_lineage": dict(execution_lineage),
                    "status": "IN_PROGRESS",
                    "cell_id": current_cell_id,
                    "attempt_id": attempt_id,
                    "run_kind": run_kind,
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
                    "recovery": recovery_refs,
                },
            )
            try:
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
                final_status_payload = {
                    "schema_version": "stage2-s204-cell-final-status-v3",
                    "execution_commit": execution_lineage["execution_commit"],
                    "execution_lineage": dict(execution_lineage),
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
                    "input_checkpoint_id": expected["input_checkpoint_id"],
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
                    "events_ref": event_path.as_posix(),
                }
                final_status = {
                    **final_status_payload,
                    "artifact_hash": _canonical_hash(final_status_payload),
                }
                _publish_json_once(final_status_path, final_status)
                _append_event(
                    event_path,
                    {
                        "event": "FINAL_PUBLISHED",
                        "status": status,
                        "task_result_hash": result.result_hash,
                        "bundle_manifest_sha256": bundle_hash,
                        "final_status_ref": final_status_path.as_posix(),
                    },
                )
            except Exception as error:
                _append_event(
                    event_path,
                    {
                        "event": "FAILED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise
        results.append(
            {
                "cell_id": current_cell_id,
                "execution_commit": execution_lineage["execution_commit"],
                "execution_lineage": dict(execution_lineage),
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
                "final_status_ref": final_status_path.as_posix(),
            }
        )
    return results


def aggregate_g23(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    data_root: Path,
    metrics_path: Path | None,
    execution_commit: str | None = None,
) -> tuple[str, str, Path]:
    """Collect six immutable cell results and invoke the downstream evaluator hook.

    Cell processes never call this function.  It is deliberately a single-writer
    operation: incomplete/ambiguous cells only receive a content-addressed
    preflight note, while a complete set gets one content-addressed G2.3 attempt.
    """

    data_root = data_root.resolve()
    output_root = output_root.resolve()
    execution_lineage = _validate_execution_lineage(execution_commit)

    from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
    from param_importance_nlp.runtime import TaskRunResult, TaskRunStatus

    gpu_smoke = _validate_gpu_smoke_artifact(plan, data_root)
    reasons: list[str] = []
    selected: dict[str, dict[str, Any]] = {}
    task_refs: dict[str, str] = {}
    task_hashes: dict[str, str] = {}
    bundle_hashes: dict[str, str] = {}
    expected_cells = tuple(str(item["cell_id"]) for item in plan["cells"])
    for expected in plan["cells"]:
        current_cell_id = str(expected["cell_id"])
        status_paths = sorted(
            (output_root / cell_path_component(current_cell_id)).rglob("final-status.json")
            if (output_root / cell_path_component(current_cell_id)).exists()
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
            if (
                not isinstance(row.get("artifact_hash"), str)
                or _canonical_hash(
                    {key: value for key, value in row.items() if key != "artifact_hash"}
                )
                != row.get("artifact_hash")
            ):
                reasons.append(f"{current_cell_id}:FINAL_STATUS_HASH_INVALID")
                continue
            if row.get("input_checkpoint_id") != expected.get("input_checkpoint_id"):
                reasons.append(f"{current_cell_id}:INPUT_CHECKPOINT_ID_BINDING_FAILED")
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
            execution = config.section("execution")
            recovery = config.section("recovery")
            row_recovery = row.get("recovery")
            if (
                not isinstance(execution, Mapping)
                or not isinstance(recovery, Mapping)
                or not isinstance(row_recovery, Mapping)
            ):
                raise ValueError("config execution/recovery sections malformed")
            expected_formal = config.run_intent == "formal" and not bool(execution.get("dry_run"))
            if (
                result.task_id != config.task_id
                or result.stage != config.task_definition.stage
                or result.run_intent != config.run_intent
                or result.config_hash != config.config_hash
                or result.formal_eligible is not expected_formal
                or result.recovery_mode is not config.task_definition.recovery_mode
                or row.get("formal_eligible") is not result.formal_eligible
                or row_recovery.get("resume_ref") != recovery.get("resume_ref")
            ):
                raise ValueError("TaskRunResult/config/recovery identity mismatch")
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
                "execution_commit": execution_lineage["execution_commit"],
                "execution_lineage": dict(execution_lineage),
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
        "execution_commit": execution_lineage["execution_commit"],
        "execution_lineage": dict(execution_lineage),
        "status": "COMPLETE" if gate_status == "PASS" else "BLOCKED_OR_NOT_QUALIFIED",
        "g2_3_gate": gate_status,
        "g2_3_gate_artifact_hash": gate_hash,
        "cells": [selected[cell] for cell in expected_cells],
        "task_result_hashes": task_hashes,
        "bundle_manifest_hashes": bundle_hashes,
        "provider_entry": "TaskRuntime.stage2.04_reference_target",
        "excluded_pci": EXCLUDED_PCI,
        "g21_smoke": gpu_smoke,
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
        "--execution-commit",
        help="required for execute/aggregate: exact clean detached launcher repository HEAD",
    )
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
        default=None,
        help="exactly one current nvidia-smi index or approved GPU UUID per process; "
        f"PCI {EXCLUDED_PCI} is excluded",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--candidate-sizes",
        type=int,
        nargs="+",
        default=None,
        help="explicit sizing nodes; required for --execute/--aggregate",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--per-sequence-seconds", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (args.execute or args.aggregate) and args.candidate_sizes is None:
            raise ValueError("--execute/--aggregate require explicit --candidate-sizes")
        candidate_sizes = tuple(DEFAULT_CANDIDATES if args.candidate_sizes is None else args.candidate_sizes)
        if args.execute or args.aggregate:
            if args.execution_commit is None:
                raise ValueError(f"--{'execute' if args.execute else 'aggregate'} requires --execution-commit")
            # Validate before loading/publishing a plan or entering TaskRuntime;
            # the cell/aggregate records must never be produced by a dirty or
            # attached checkout.
            _validate_execution_lineage(args.execution_commit)
        g21_root = args.data_root if (args.execute or args.aggregate) else None
        # Always consume the canonical formal handoff loader.  In execute and
        # aggregate mode it additionally re-hashes the bound raw smoke report.
        g21 = load_g21_formal_handoff(args.g21_evidence, data_root=g21_root)
        assets, data = _load(args.asset_resolution), _load(args.data_range)
        plan = build_plan(
            g21,
            assets,
            data,
            output_root=args.output_root,
            candidates=candidate_sizes,
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
            if args.execution_commit is None:
                raise ValueError("--execute requires --execution-commit")
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
                execution_commit=args.execution_commit,
            )
            print(json.dumps({"mode": "execute", "cells": results}, ensure_ascii=False, sort_keys=True))
            if not all(item["status"] == "COMPLETE" for item in results):
                return 3
        elif args.aggregate:
            if args.execution_commit is None:
                raise ValueError("--aggregate requires --execution-commit")
            if args.data_root is None:
                raise ValueError("--aggregate requires --data-root")
            if args.cell_id is not None or args.runtime_config or args.runtime_environment is not None:
                raise ValueError("--aggregate accepts no cell/config/environment arguments")
            gate_status, gate_hash, summary_path = aggregate_g23(
                plan,
                output_root=args.output_root,
                data_root=args.data_root,
                metrics_path=args.g23_metrics,
                execution_commit=args.execution_commit,
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
