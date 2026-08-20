"""Fail-closed S1.11 exit-gate evidence assembly and offline replay.

This module is deliberately an *evidence consumer*.  It never runs a test,
launches a worker, regenerates a chart, or upgrades a local fixture into a
formal result.  A formalizer may call it only after the immutable S1.1--S1.10
publications and their chart files exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts.jsonio import canonical_json_hash, load_canonical_json


TASK_ID = "stage1.11_reporting_and_exit_gate"
GATE_ID = "G1-EXIT"
DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("G1-ENTRY", "stage1.01_entry_and_contract", "G1-ENTRY"),
    ("G1-CONTRACT", "stage1.01_entry_and_contract", "G1-CONTRACT"),
    ("G1-REGISTRY", "stage1.02_architecture_and_parameter_registry", "G1-REGISTRY"),
    ("G1-ORACLE", "stage1.03_fixtures_and_oracles", "G1-ORACLE"),
    ("G1-GRAD", "stage1.04_loss_and_gradient_scale", "G1-GRAD"),
    ("G1-EST", "stage1.05_estimators", "G1-EST"),
    ("G1-STEP", "stage1.06_training_integration_and_accumulators", "G1-STEP"),
    ("G1-SINGLE", "stage1.07_single_gpu_pythia14m", "G1-SINGLE"),
    ("G1-DDP", "stage1.08_ddp_and_gradient_accumulation", "G1-DDP"),
    ("G1-NUMERIC", "stage1.09_precision_clipping_and_optimizer_boundaries", "G1-NUMERIC"),
    ("G1-RESUME", "stage1.10_checkpoint_resume_and_artifacts", "G1-RESUME"),
)

# These are producer wire contracts, not a consumer-selected convention.  A
# future index version must be added here deliberately, after its immutable
# producer schema has been audited.
_STANDARD_INDEX_WIRES: dict[str, tuple[frozenset[str], dict[str, str]]] = {
    "G1-EST": (frozenset({"stage1-s1-5-formalization-index-v1"}), {"estimator_report": "estimator-report.json", "oracle_report": "oracle-report.json", "tensor_bundle": "tensor-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-est-record.json"}),
    "G1-STEP": (frozenset({"stage1-s1-6-formalization-index-v1"}), {"step_report": "step-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-step-record.json"}),
    "G1-SINGLE": (frozenset({"stage1-s1-7-formalization-index-v1"}), {"fixture_manifest": "fixture-manifest.json", "single_gpu_report": "worker-report.json", "gradient_bundle": "arrays-manifest.json", "comparison_table": "comparison-table.json", "gate_record": "g1-single-record.json"}),
    "G1-DDP": (frozenset({"stage1-s1-8-formalization-index-v8"}), {"fixture_manifest": "fixture-manifest.json", "ddp_report": "ddp-report.json", "array_bundle": "array-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-ddp-record.json"}),
    "G1-NUMERIC": (frozenset({"stage1-s1-9-formalization-index-v8"}), {"numeric_report": "numeric-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-numeric-record.json"}),
    "G1-RESUME": (frozenset({"stage1-s1-10-formalization-index-v2"}), {"resume_report": "resume-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "artifact_manifest": "artifact-manifest.json", "gate_record": "g1-resume-record.json"}),
}

# S1.9 v8 is the released G1-NUMERIC producer wire.  Its primary evidence
# roles intentionally keep their v1 filenames, while the immutable
# reproduction closure adds the final GPU-quiescence and compatibility roles.
# Do not accept a predecessor merely because those five primary roles match.
_S1_9_V8_REPRODUCTION_REFS: dict[str, str] = {
    "attempt_start": "attempt-start.json",
    "upstream_compatibility": "upstream-compatibility.json",
    "preflight": "preflight.json",
    "prelease_gpu": "prelease-gpu.json",
    "post_worker_quiescence": "post-worker-quiescence.json",
    "lease_history": "lease-history.json",
    "single_worker": "single-bf16.json",
    "single_stdout": "single.stdout.txt",
    "single_stderr": "single.stderr.txt",
    "single_child_fingerprint": "single-child-fingerprint.json",
    "bf16_resume_checkpoint_store": "bf16-resume-store-index.json",
    "ddp_worker": "ddp-skip.json",
    "ddp_stdout": "ddp.stdout.txt",
    "ddp_stderr": "ddp.stderr.txt",
    "ddp_child_fingerprint": "ddp-child-fingerprint.json",
    "chart_csv_0": "bf16-fp32-heatmap.csv",
    "chart_csv_1": "clip-norm-factor.csv",
    "chart_csv_2": "skip-zero-difference.csv",
    "chart_csv_3": "t-amp-scale.csv",
    "chart_csv_4": "u-single-factor-identity.csv",
    "chart_csv_5": "u-single-factor-ratio-diagnostic.csv",
    "chart_svg_0": "bf16-fp32-heatmap.svg",
    "chart_svg_1": "clip-norm-factor.svg",
    "chart_svg_2": "skip-zero-difference.svg",
    "chart_svg_3": "t-amp-scale.svg",
    "chart_svg_4": "u-single-factor-identity.svg",
    "chart_svg_5": "u-single-factor-ratio-diagnostic.svg",
}
_S1_9_V8_HASHED_REPRODUCTION_SCHEMAS: dict[str, tuple[str, str | None]] = {
    "attempt_start": ("stage1-s1-9-attempt-start-v1", "STARTED"),
    "preflight": ("stage1-s1-9-preflight-v1", "PASS"),
    "lease_history": ("runtime.project-gpu-lease-history.v1", None),
    "single_worker": ("stage1-s1-9-single-bf16-worker-v1", "PASS"),
    "single_child_fingerprint": ("stage1-s1-9-child-fingerprint-v1", "EXITED"),
    "bf16_resume_checkpoint_store": ("stage1-s1-9-bf16-checkpoint-store-reproduction-v2", None),
    "ddp_worker": ("stage1-s1-9-ddp-skip-worker-v1", "PASS"),
    "ddp_child_fingerprint": ("stage1-s1-9-child-fingerprint-v1", "EXITED"),
}
_S1_9_V8_INDEX_FIELDS = frozenset({
    "schema_version", "status", "gate_id", "task_id", "fixture_id",
    "generator_git_commit", "consumer_git_commit", "git_branch", "checked_at",
    "s1_7_handoff", "s1_8_handoff", "role_refs", "role_sha256",
    "reproduction_role_refs", "reproduction_role_sha256", "gate_artifact_hash",
    "csv_sha256", "svg_sha256", "validation_ref", "validation_sha256",
    "replay_ref", "replay_sha256", "replay_hash", "next_task_ids", "artifact_hash",
})
_S1_9_SOURCE_PATHS = frozenset({
    "fixtures/stage1/stage1-s19-precision-fixture-v1.json",
    "src/param_importance_nlp/stage1_precision.py",
    "src/param_importance_nlp/stage1_precision_oracle.py",
    "src/param_importance_nlp/contracts/jsonio.py",
    "src/param_importance_nlp/contracts/runtime_evidence.py",
    "src/param_importance_nlp/core/accumulator.py",
    "src/param_importance_nlp/core/estimators.py",
    "src/param_importance_nlp/core/registry.py",
    "src/param_importance_nlp/core/sufficient_statistics.py",
    "src/param_importance_nlp/core/tensors.py",
    "src/param_importance_nlp/providers/training.py",
    "src/param_importance_nlp/runtime/gradients.py",
    "src/param_importance_nlp/runtime/operations.py",
    "src/param_importance_nlp/runtime/optimizer.py",
    "src/param_importance_nlp/runtime/reducers.py",
    "src/param_importance_nlp/runtime/task_artifacts.py",
    "src/param_importance_nlp/runtime/training.py",
    "ops/stage1/formalize_s1_6.py",
    "ops/stage1/formalize_s1_9.py",
    "ops/stage1/run_s1_9_single_bf16_worker.py",
    "ops/stage1/run_s1_9_ddp_skip_worker.py",
    "tests/test_stage1_s19_precision.py",
    "schemas/stage1/s1-9-precision-fixture-v1.json",
    "schemas/stage1/s1-9-numeric-report-v1.json",
    "schemas/stage1/s1-9-oracle-bundle-v1.json",
    "schemas/stage1/s1-9-trace-bundle-v1.json",
    "schemas/stage1/s1-9-comparison-table-v1.json",
    "schemas/stage1/s1-9-gate-record-v1.json",
    "schemas/stage1/s1-9-replay-validation-v1.json",
    "schemas/stage1/s1-9-validation-v1.json",
    "schemas/stage1/s1-9-formalization-index-v1.json",
    "schemas/stage1/s1-9-single-bf16-worker-v2.json",
    "schemas/stage1/s1-9-ddp-skip-worker-v1.json",
    "schemas/stage1/s1-9-bf16-checkpoint-store-reproduction-v1.json",
})
_S1_10_SOURCE_PATHS = frozenset({
    "fixtures/stage1/stage1-s110-checkpoint-fixture-v1.json",
    "src/param_importance_nlp/stage1_checkpoint_resume.py",
    "src/param_importance_nlp/stage1_checkpoint_oracle.py",
    "src/param_importance_nlp/runtime/checkpoint.py",
    "src/param_importance_nlp/runtime/checkpoint_group.py",
    "src/param_importance_nlp/runtime/training.py",
    "ops/stage1/formalize_s1_10.py",
    "ops/stage1/run_s1_10_resume_worker.py",
    "tests/test_stage1_s110_checkpoint_resume.py",
    "schemas/stage1/s1-10-checkpoint-fixture-v1.json",
    "schemas/stage1/s1-10-resume-report-v1.json",
    "schemas/stage1/s1-10-resume-report-v2.json",
    "schemas/stage1/s1-10-oracle-bundle-v1.json",
    "schemas/stage1/s1-10-trace-bundle-v1.json",
    "schemas/stage1/s1-10-comparison-table-v1.json",
    "schemas/stage1/s1-10-artifact-manifest-v1.json",
    "schemas/stage1/s1-10-gate-record-v1.json",
    "schemas/stage1/s1-10-replay-validation-v1.json",
    "schemas/stage1/s1-10-validation-v1.json",
    "schemas/stage1/s1-10-validation-v2.json",
    "schemas/stage1/s1-10-formalization-index-v1.json",
    "schemas/stage1/s1-10-formalization-index-v2.json",
    "schemas/stage1/s1-10-formal-observation-v1.json",
})
_S1_10_V2_INDEX_FIELDS = frozenset({
    "schema_version", "status", "gate_id", "task_id", "fixture_id", "generator_git_commit",
    "consumer_git_commit", "git_branch", "checked_at", "upstream", "role_refs", "role_sha256",
    "chart_csv_sha256", "chart_svg_sha256", "formal_observation_ref", "formal_observation_sha256",
    "formal_observation_artifact_hash", "formal_run_token_sha256", "formal_single_report_ref",
    "formal_single_report_sha256", "formal_four_rank_report_ref", "formal_four_rank_report_sha256",
    "gate_artifact_hash", "validation_ref", "validation_sha256", "replay_ref", "replay_sha256",
    "replay_hash", "next_task_ids", "artifact_hash",
})
REQUIRED_CHARTS: tuple[str, ...] = (
    "gradient-identity", "u-identity", "weighted-reduction", "ddp-identity", 
    "module-metric-heatmap", "clip-factor", "resume-errors", "noise-smoke", "accumulator-residual",
)
REQUIREMENT_IDS: tuple[str, ...] = tuple(f"S1.11-R{number:02d}" for number in range(1, 29))
# Plan §3's 28 rows are frozen in this order.  The consumer must not let a
# caller relabel a correct measurement as evidence for a different gate.
REQUIREMENT_GATE_IDS: tuple[str, ...] = (
    "G1-REGISTRY", "G1-EST", "G1-EST", "G1-EST", "G1-EST", "G1-EST", "G1-EST", "G1-EST",
    "G1-ORACLE", "G1-ORACLE", "G1-EST", "G1-EST", "G1-EST", "G1-EST", "G1-STEP", "G1-STEP",
    "G1-STEP", "G1-STEP", "G1-STEP", "G1-SINGLE", "G1-SINGLE", "G1-DDP", "G1-DDP", "G1-DDP",
    "G1-RESUME", "G1-NUMERIC", "G1-ENTRY", "G1-CONTRACT",
)


class Stage1ExitGateError(RuntimeError):
    """Raised for a malformed or incomplete S1.11 evidence closure."""


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_HASH_INVALID")
    return value


def _commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_COMMIT_INVALID")
    return value


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_OBJECT_REQUIRED")
    return dict(value)


def _safe_relative(root: Path, ref: object, *, field: str) -> Path:
    if not isinstance(ref, str) or not ref or Path(ref).is_absolute():
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_REF_INVALID")
    candidate = (root / ref).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_REF_ESCAPES_ROOT") from error
    return candidate


def _self_hash(value: Mapping[str, object], *, field: str) -> None:
    body = dict(value)
    declared = body.pop("artifact_hash", None)
    if not isinstance(declared, str) or declared != canonical_json_hash(body):
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_SELF_HASH_INVALID")


def _named_self_hash(value: Mapping[str, object], *, field: str, hash_field: str) -> None:
    body = dict(value)
    declared = body.pop(hash_field, None)
    if not isinstance(declared, str) or declared != canonical_json_hash(body):
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_SELF_HASH_INVALID")


def _source_map(value: object, *, field: str, expected_paths: frozenset[str]) -> dict[str, str]:
    raw = _mapping(value, field=field)
    if set(raw) != expected_paths:
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_KEYSET_INVALID")
    result: dict[str, str] = {}
    for reference, digest in raw.items():
        logical = PurePosixPath(reference)
        if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
            raise Stage1ExitGateError(f"S1_11_{field.upper()}_REF_INVALID")
        result[reference] = _hash(digest, field=f"{field}_{reference}")
    return result


def _s1_9_v2_determinism(value: object, *, field: str) -> None:
    determinism = _mapping(value, field=field)
    if determinism != {
        "algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allowed_nondeterministic_kernel_classes": [],
        "kernel_policy": "empty_pre_registered_allowlist",
    }:
        raise Stage1ExitGateError(f"S1_11_{field.upper()}_INVALID")


def _validate_s1_9_current_v2_roles(
    index_dir: Path,
    reproduction_paths: Mapping[str, Path],
    validation: Mapping[str, object],
) -> None:
    """Validate the final V2 validators behind S1.9's otherwise frozen V1 roles."""

    loaded: dict[str, dict[str, Any]] = {}
    for role, (schema_version, status) in _S1_9_V8_HASHED_REPRODUCTION_SCHEMAS.items():
        value = _mapping(load_canonical_json(reproduction_paths[role]), field=f"s1_9.{role}")
        _self_hash(value, field=f"s1_9.{role}")
        if value.get("schema_version") != schema_version or (status is not None and value.get("status") != status):
            raise Stage1ExitGateError("S1_11_S1_9_V8_REPRODUCTION_SCHEMA_INVALID")
        loaded[role] = value

    single = loaded["single_worker"]
    if set(single) != {
        "schema_version", "status", "execution_commit", "run_token",
        "approved_gpu_uuid", "cuda_visible_devices", "environment_summary",
        "observation", "artifact_hash",
    }:
        raise Stage1ExitGateError("S1_11_S1_9_V8_SINGLE_WORKER_V2_INVALID")
    observation = _mapping(single.get("observation"), field="s1_9.single_worker_observation")
    _s1_9_v2_determinism(observation.get("determinism"), field="s1_9_single_worker_v2_determinism")

    if set(validation) != {
        "schema_version", "status", "gate_id", "task_id", "execution_scope",
        "fixture_id", "producer_commit", "consumer_commit", "upstream",
        "regression", "direct_checks", "role_sha256", "csv_sha256",
        "svg_sha256", "replay_sha256", "replay_hash", "artifact_hash",
    } or validation.get("schema_version") != "stage1-s1-9-validation-v2":
        raise Stage1ExitGateError("S1_11_S1_9_V8_VALIDATION_V2_INVALID")
    regression = _mapping(validation.get("regression"), field="s1_9.validation_regression")
    if regression.get("kernel_allowlist") != []:
        raise Stage1ExitGateError("S1_11_S1_9_V8_VALIDATION_V2_INVALID")
    _s1_9_v2_determinism(regression.get("bf16"), field="s1_9_validation_v2_determinism")

    checkpoint = loaded["bf16_resume_checkpoint_store"]
    required_checkpoint_fields = {
        "schema_version", "checkpoint_id", "commit_ref", "commit_sha256",
        "bundle_manifest_ref", "bundle_manifest_sha256", "bundle_file_hashes",
        "artifact_hash",
    }
    checkpoint_id = checkpoint.get("checkpoint_id")
    if (
        set(checkpoint) != required_checkpoint_fields
        or not isinstance(checkpoint_id, str)
        or not checkpoint_id
        or len(checkpoint_id) > 128
        or checkpoint_id[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in checkpoint_id)
        or checkpoint.get("commit_ref") != f"commits/{checkpoint_id}.json"
        or checkpoint.get("bundle_manifest_ref") != f"objects/{checkpoint_id}/manifest.json"
    ):
        raise Stage1ExitGateError("S1_11_S1_9_V8_CHECKPOINT_V2_INVALID")
    store_root = index_dir / "bf16-resume-store"
    commit_path = _safe_relative(store_root, checkpoint["commit_ref"], field="s1_9_checkpoint_commit")
    manifest_path = _safe_relative(store_root, checkpoint["bundle_manifest_ref"], field="s1_9_checkpoint_manifest")
    if (
        not commit_path.is_file()
        or _hash_file(commit_path) != _hash(checkpoint.get("commit_sha256"), field="s1_9_checkpoint_commit")
        or not manifest_path.is_file()
        or _hash_file(manifest_path) != _hash(checkpoint.get("bundle_manifest_sha256"), field="s1_9_checkpoint_manifest")
    ):
        raise Stage1ExitGateError("S1_11_S1_9_V8_CHECKPOINT_FILE_HASH_INVALID")
    file_rows = checkpoint.get("bundle_file_hashes")
    if not isinstance(file_rows, list) or not file_rows:
        raise Stage1ExitGateError("S1_11_S1_9_V8_CHECKPOINT_V2_INVALID")
    listed: dict[str, str] = {}
    expected_prefix = f"objects/{checkpoint_id}/"
    for position, raw in enumerate(file_rows):
        row = _mapping(raw, field=f"s1_9.checkpoint_file_{position}")
        ref = row.get("ref")
        if set(row) != {"ref", "sha256"} or not isinstance(ref, str) or not ref.startswith(expected_prefix) or ref in listed:
            raise Stage1ExitGateError("S1_11_S1_9_V8_CHECKPOINT_V2_INVALID")
        path = _safe_relative(store_root, ref, field=f"s1_9_checkpoint_file_{position}")
        digest = _hash(row.get("sha256"), field=f"s1_9_checkpoint_file_{position}")
        if not path.is_file() or _hash_file(path) != digest:
            raise Stage1ExitGateError("S1_11_S1_9_V8_CHECKPOINT_FILE_HASH_INVALID")
        listed[ref] = digest
    actual = {
        path.relative_to(store_root).as_posix(): _hash_file(path)
        for path in sorted((store_root / "objects" / checkpoint_id).rglob("*"))
        if path.is_file()
    }
    if listed != actual or checkpoint["bundle_manifest_ref"] not in listed:
        raise Stage1ExitGateError("S1_11_S1_9_V8_CHECKPOINT_FILE_CLOSURE_INVALID")


@dataclass(frozen=True, slots=True)
class DependencyBinding:
    """Immutable index and gate-role binding supplied by the final formalizer."""

    gate_id: str
    task_id: str
    index_ref: str
    index_schema_version: str
    index_sha256: str
    index_artifact_hash: str
    producer_commit: str
    gate_role: str
    gate_ref: str
    gate_schema_version: str
    gate_sha256: str
    gate_artifact_hash: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DependencyBinding":
        raw = _mapping(value, field="dependency")
        expected = {
            "gate_id", "task_id", "index_ref", "index_schema_version", "index_sha256", "index_artifact_hash",
            "producer_commit", "gate_role", "gate_ref", "gate_schema_version", "gate_sha256", "gate_artifact_hash",
        }
        if set(raw) != expected:
            raise Stage1ExitGateError("S1_11_DEPENDENCY_FIELDS_INVALID")
        gate_id, task_id = raw["gate_id"], raw["task_id"]
        if not isinstance(gate_id, str) or not isinstance(task_id, str):
            raise Stage1ExitGateError("S1_11_DEPENDENCY_IDENTITY_INVALID")
        refs = (raw["index_ref"], raw["gate_ref"], raw["gate_role"], raw["index_schema_version"], raw["gate_schema_version"])
        if not all(isinstance(item, str) and item for item in refs):
            raise Stage1ExitGateError("S1_11_DEPENDENCY_REF_INVALID")
        return cls(
            gate_id, task_id, raw["index_ref"], raw["index_schema_version"], _hash(raw["index_sha256"], field="index"),
            _hash(raw["index_artifact_hash"], field="index_artifact"), _commit(raw["producer_commit"], field="producer"),
            raw["gate_role"], raw["gate_ref"], raw["gate_schema_version"], _hash(raw["gate_sha256"], field="gate"),
            _hash(raw["gate_artifact_hash"], field="gate_artifact"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "gate_id": self.gate_id, "task_id": self.task_id, "index_ref": self.index_ref, "index_schema_version": self.index_schema_version,
            "index_sha256": self.index_sha256, "index_artifact_hash": self.index_artifact_hash,
            "producer_commit": self.producer_commit, "gate_role": self.gate_role, "gate_ref": self.gate_ref,
            "gate_schema_version": self.gate_schema_version, "gate_sha256": self.gate_sha256,
            "gate_artifact_hash": self.gate_artifact_hash,
        }


def _validate_binding(root: Path, binding: DependencyBinding) -> dict[str, object]:
    index_path = _safe_relative(root, binding.index_ref, field="index")
    if not index_path.is_file() or _hash_file(index_path) != binding.index_sha256:
        raise Stage1ExitGateError("S1_11_INDEX_FILE_HASH_MISMATCH")
    index = _mapping(load_canonical_json(index_path), field="index")
    _self_hash(index, field="index")
    if index.get("schema_version") != binding.index_schema_version:
        raise Stage1ExitGateError("S1_11_INDEX_SCHEMA_VERSION_MISMATCH")
    if index.get("artifact_hash") != binding.index_artifact_hash:
        raise Stage1ExitGateError("S1_11_INDEX_ARTIFACT_HASH_MISMATCH")
    if index.get("status") != "PASS":
        raise Stage1ExitGateError("S1_11_INDEX_IDENTITY_OR_STATUS_INVALID")
    if index.get("generator_git_commit") != binding.producer_commit:
        raise Stage1ExitGateError("S1_11_INDEX_PRODUCER_COMMIT_MISMATCH")
    legacy_s11 = binding.gate_id in {"G1-ENTRY", "G1-CONTRACT"}
    if legacy_s11:
        # S1.1's formal index is an early producer wire: every embedded ref is
        # DATA_ROOT-relative (normally ``evidence/...``), not index-directory
        # relative.  Reuse its own immutable readers rather than treating the
        # semantic config/environment/result hashes as file digests.
        from .contracts.status import GateRecord, GateStatus
        from .runtime import TaskRunResult

        if index.get("schema_version") != "stage1-s1-1-formalization-index-v1" or binding.task_id != "stage1.01_entry_and_contract":
            raise Stage1ExitGateError("S1_11_S1_1_ADAPTER_IDENTITY_INVALID")
        hashes = _mapping(index.get("gate_artifact_hashes"), field="s1_1.gate_artifact_hashes")
        expected_key = f"stage1.{binding.gate_id}"
        if hashes.get(expected_key) != binding.gate_artifact_hash:
            raise Stage1ExitGateError("S1_11_S1_1_GATE_ARTIFACT_HASH_MISMATCH")
        role_refs = _mapping(index.get("task_output_refs"), field="s1_1.task_output_refs")
        try:
            config_path = _safe_relative(root, index.get("config_ref"), field="s1_1_config_ref")
            environment_path = _safe_relative(root, index.get("environment_ref"), field="s1_1_environment_ref")
            result_path = _safe_relative(root, index.get("result_ref"), field="s1_1_result_ref")
            config = _mapping(load_canonical_json(config_path), field="s1_1_config")
            environment = _mapping(load_canonical_json(environment_path), field="s1_1_environment")
            result = TaskRunResult.from_mapping(_mapping(load_canonical_json(result_path), field="s1_1_result"))
        except Exception as error:
            raise Stage1ExitGateError("S1_11_S1_1_SEMANTIC_SOURCE_CLOSURE_INVALID") from error
        formal_records = result.metadata.get("formal_gate_records")
        expected_gate_ids = ("stage1.G1-ENTRY", "stage1.G1-CONTRACT")
        try:
            if not isinstance(formal_records, list) or len(formal_records) != len(expected_gate_ids):
                raise ValueError("formal record count")
            formal_gates = [GateRecord.from_mapping(_mapping(record, field="s1_1_formal_gate")) for record in formal_records]
            actual_gate_hashes = {record.gate_id: record.artifact_hash for record in formal_gates}
        except Exception as error:
            raise Stage1ExitGateError("S1_11_S1_1_FORMAL_GATE_RECORDS_INVALID") from error
        if any(record.gate_id != expected or record.status is not GateStatus.PASS for expected, record in zip(expected_gate_ids, formal_gates, strict=True)) or actual_gate_hashes != hashes:
            raise Stage1ExitGateError("S1_11_S1_1_FORMAL_GATE_RECORDS_INVALID")
        if not config or not environment or result.result_hash != index.get("result_hash") or result.config_hash != index.get("config_hash") or result.task_id != binding.task_id or result.status.value != "PASS" or result.run_intent != "formal" or result.formal_eligible is not True or dict(result.artifact_refs) != role_refs:
            raise Stage1ExitGateError("S1_11_S1_1_SEMANTIC_SOURCE_CLOSURE_INVALID")
        if binding.gate_role != "gate_record" or not isinstance(role_refs.get("gate_record"), str):
            raise Stage1ExitGateError("S1_11_S1_1_GATE_ROLE_INVALID")
        gate_path = _safe_relative(root, role_refs["gate_record"], field="s1_1_gate_record")
        gate = _mapping(load_canonical_json(gate_path), field="s1_1_gate_record")
        if gate.get("artifact_hash") and isinstance(gate.get("artifact_hash"), str):
            _self_hash(gate, field="s1_1_gate_record")
        if binding.gate_ref != role_refs["gate_record"] or _hash_file(gate_path) != binding.gate_sha256:
            raise Stage1ExitGateError("S1_11_S1_1_GATE_ROLE_BINDING_MISMATCH")
        if binding.gate_schema_version != gate.get("schema_version"):
            raise Stage1ExitGateError("S1_11_GATE_SCHEMA_VERSION_MISMATCH")
        # The shared record is a structured aggregate; the immutable index's
        # exact per-gate artifact hash is the authoritative cross-binding.
        return {"binding": binding.to_dict(), "index_artifact_hash": index["artifact_hash"], "gate_requirements": [binding.gate_id]}
    if index.get("gate_id") != binding.gate_id or index.get("task_id") != binding.task_id:
        raise Stage1ExitGateError("S1_11_INDEX_IDENTITY_OR_STATUS_INVALID")
    if binding.gate_id == "G1-NUMERIC" and (
        set(index) != _S1_9_V8_INDEX_FIELDS
        or index.get("fixture_id") != "stage1-s19-precision-fixture-v1"
        or index.get("validation_ref") != "validation.json"
        or index.get("replay_ref") != "replay-validation.json"
        or index.get("next_task_ids") != ["stage1.10_checkpoint_resume_and_artifacts"]
    ):
        raise Stage1ExitGateError("S1_11_S1_9_V8_INDEX_SCHEMA_CLOSURE_INVALID")
    if binding.gate_id == "G1-RESUME" and (
        set(index) != _S1_10_V2_INDEX_FIELDS
        or index.get("fixture_id") != "stage1-s110-checkpoint-fixture-v1"
        or index.get("validation_ref") != "validation.json"
        or index.get("replay_ref") != "replay-validation.json"
        or index.get("formal_observation_ref") != "formal-observation.json"
        or index.get("formal_single_report_ref") != "formal-single-report.json"
        or index.get("formal_four_rank_report_ref") != "formal-four-rank-report.json"
        or index.get("next_task_ids") != ["stage1.11_reporting_and_exit_gate"]
    ):
        raise Stage1ExitGateError("S1_11_S1_10_V2_INDEX_SCHEMA_CLOSURE_INVALID")
    if "consumer_git_commit" in index and _commit(index.get("consumer_git_commit"), field="consumer") != index.get("consumer_git_commit"):
        raise Stage1ExitGateError("S1_11_INDEX_CONSUMER_COMMIT_INVALID")
    # S1.2--S1.4 predate the uniform role_refs wire.  Their named report
    # fields remain immutable, schema-specific roles, not fallback strings.
    legacy_named = {
        "G1-REGISTRY": ("report", "report_ref", "report_sha256", "validation_ref", "validation_sha256"),
        "G1-ORACLE": ("oracle_validation_report", "oracle_validation_report_ref", "oracle_validation_report_sha256", "replay_ref", "replay_sha256"),
        "G1-GRAD": ("gate_record", "gate_record_ref", "gate_record_sha256", "replay_ref", "replay_sha256"),
    }
    named_schema_versions = {
        "G1-REGISTRY": {"stage1-s1-2-formalization-index-v1", "stage1-s1-2-formalization-index-v2"},
        "G1-ORACLE": {"stage1-s1-3-formalization-index-v1", "stage1-s1-3-formalization-index-v2"},
        "G1-GRAD": {"stage1-s1-4-formalization-index-v1"},
    }
    if binding.gate_id in legacy_named and index.get("schema_version") in named_schema_versions[binding.gate_id]:
        expected_role, role_ref_key, role_sha_key, replay_ref_key, replay_sha_key = legacy_named[binding.gate_id]
        if binding.gate_role != expected_role or binding.gate_ref != index.get(role_ref_key) or binding.gate_sha256 != index.get(role_sha_key):
            raise Stage1ExitGateError("S1_11_NAMED_ROLE_BINDING_MISMATCH")
        gate_path = _safe_relative(index_path.parent, binding.gate_ref, field="named_gate")
        if not gate_path.is_file() or _hash_file(gate_path) != binding.gate_sha256:
            raise Stage1ExitGateError("S1_11_NAMED_ROLE_FILE_HASH_MISMATCH")
        gate = _mapping(load_canonical_json(gate_path), field="named_gate")
        if gate.get("schema_version") != binding.gate_schema_version:
            raise Stage1ExitGateError("S1_11_GATE_SCHEMA_VERSION_MISMATCH")
        if "artifact_hash" in gate:
            _self_hash(gate, field="named_gate")
            if gate["artifact_hash"] != binding.gate_artifact_hash:
                raise Stage1ExitGateError("S1_11_GATE_ARTIFACT_HASH_MISMATCH")
        replay_path = _safe_relative(index_path.parent, index.get(replay_ref_key), field="named_replay")
        if not replay_path.is_file() or _hash_file(replay_path) != _hash(index.get(replay_sha_key), field="named_replay"):
            raise Stage1ExitGateError("S1_11_REPLAY_FILE_HASH_MISMATCH")
        return {"binding": binding.to_dict(), "index_artifact_hash": index["artifact_hash"], "gate_requirements": [binding.gate_id]}
    wire = _STANDARD_INDEX_WIRES.get(binding.gate_id)
    if wire is None or index.get("schema_version") not in wire[0]:
        raise Stage1ExitGateError("S1_11_UNSUPPORTED_INDEX_WIRE_VERSION")
    refs, hashes = _mapping(index.get("role_refs"), field="index.role_refs"), _mapping(index.get("role_sha256"), field="index.role_sha256")
    if refs != wire[1]:
        raise Stage1ExitGateError("S1_11_INDEX_ROLE_REF_WIRE_INVALID")
    if set(hashes) != set(wire[1]):
        raise Stage1ExitGateError("S1_11_INDEX_ROLE_HASH_WIRE_INVALID")
    if refs.get(binding.gate_role) != binding.gate_ref or hashes.get(binding.gate_role) != binding.gate_sha256:
        raise Stage1ExitGateError("S1_11_GATE_ROLE_BINDING_MISMATCH")
    if set(refs) != set(hashes):
        raise Stage1ExitGateError("S1_11_INDEX_ROLE_SET_MISMATCH")
    index_dir = index_path.parent
    for role, ref in refs.items():
        role_path = _safe_relative(index_dir, ref, field=f"role_{role}")
        if not role_path.is_file() or _hash_file(role_path) != hashes[role]:
            raise Stage1ExitGateError("S1_11_INDEX_ROLE_FILE_HASH_MISMATCH")
    role_self_hash_contracts = {
        "G1-NUMERIC": {
            "numeric_report": ("report_hash", "stage1-s1-9-numeric-report-v1"),
            "oracle_bundle": ("oracle_hash", "stage1-s1-9-oracle-bundle-v1"),
            "trace_bundle": ("trace_hash", "stage1-s1-9-trace-bundle-v1"),
            "comparison_table": ("table_hash", "stage1-s1-9-comparison-table-v1"),
            "gate_record": ("artifact_hash", "stage1-s1-9-gate-record-v1"),
        },
        "G1-RESUME": {
            "resume_report": ("report_hash", "stage1-s1-10-resume-report-v2"),
            "oracle_bundle": ("oracle_hash", "stage1-s1-10-oracle-bundle-v1"),
            "trace_bundle": ("trace_hash", "stage1-s1-10-trace-bundle-v1"),
            "comparison_table": ("table_hash", "stage1-s1-10-comparison-table-v1"),
            "artifact_manifest": ("manifest_hash", "stage1-s1-10-artifact-manifest-v1"),
            "gate_record": ("artifact_hash", "stage1-s1-10-gate-record-v1"),
        },
    }
    for role, (hash_field, schema_version) in role_self_hash_contracts.get(binding.gate_id, {}).items():
        role_value = _mapping(load_canonical_json(_safe_relative(index_dir, refs[role], field=f"role_{role}")), field=f"role_{role}")
        _named_self_hash(role_value, field=f"role_{role}", hash_field=hash_field)
        if role_value.get("schema_version") != schema_version:
            raise Stage1ExitGateError("S1_11_INDEX_ROLE_SCHEMA_INVALID")
    validation_ref, validation_sha = index.get("validation_ref"), index.get("validation_sha256")
    validation_path = _safe_relative(index_dir, validation_ref, field="validation")
    if not validation_path.is_file() or _hash_file(validation_path) != _hash(validation_sha, field="validation"):
        raise Stage1ExitGateError("S1_11_VALIDATION_FILE_HASH_MISMATCH")
    validation = _mapping(load_canonical_json(validation_path), field="validation")
    if validation.get("status") != "PASS":
        raise Stage1ExitGateError("S1_11_VALIDATION_NOT_PASS")
    if "artifact_hash" in validation:
        _self_hash(validation, field="validation")
    gate_path = _safe_relative(index_dir, binding.gate_ref, field="gate")
    gate = _mapping(load_canonical_json(gate_path), field="gate")
    _self_hash(gate, field="gate")
    if gate.get("schema_version") != binding.gate_schema_version:
        raise Stage1ExitGateError("S1_11_GATE_SCHEMA_VERSION_MISMATCH")
    if gate.get("artifact_hash") != binding.gate_artifact_hash:
        raise Stage1ExitGateError("S1_11_GATE_ARTIFACT_HASH_MISMATCH")
    if index.get("gate_artifact_hash") != binding.gate_artifact_hash:
        raise Stage1ExitGateError("S1_11_INDEX_GATE_ARTIFACT_HASH_MISMATCH")
    if gate.get("status") != "PASS" or gate.get("gate_id") != binding.gate_id or gate.get("task_id") != binding.task_id:
        raise Stage1ExitGateError("S1_11_GATE_IDENTITY_OR_STATUS_INVALID")
    requirements = _mapping(gate.get("requirements"), field="gate.requirements")
    if not requirements or any(type(value) is not bool or not value for value in requirements.values()):
        raise Stage1ExitGateError("S1_11_GATE_REQUIREMENTS_NOT_ALL_PASS")
    replay_ref, replay_sha = index.get("replay_ref"), index.get("replay_sha256")
    replay_path = _safe_relative(index_dir, replay_ref, field="replay")
    if not replay_path.is_file() or _hash_file(replay_path) != _hash(replay_sha, field="replay"):
        raise Stage1ExitGateError("S1_11_REPLAY_FILE_HASH_MISMATCH")
    replay = _mapping(load_canonical_json(replay_path), field="replay")
    if replay.get("status") != "PASS":
        raise Stage1ExitGateError("S1_11_REPLAY_NOT_PASS")
    if "artifact_hash" in replay:
        _self_hash(replay, field="replay")
    if binding.gate_id in {"G1-NUMERIC", "G1-RESUME"}:
        _named_self_hash(replay, field="replay", hash_field="replay_hash")
        expected_replay_schema = {
            "G1-NUMERIC": "stage1-s1-9-replay-validation-v1",
            "G1-RESUME": "stage1-s1-10-replay-validation-v1",
        }[binding.gate_id]
        if (
            replay.get("schema_version") != expected_replay_schema
            or replay.get("replay_hash") != index.get("replay_hash")
            or replay.get("source_gate_artifact_hash") != binding.gate_artifact_hash
        ):
            raise Stage1ExitGateError("S1_11_REPLAY_SEMANTIC_BINDING_INVALID")
    if binding.gate_id == "G1-NUMERIC":
        reproduction_refs = _mapping(index.get("reproduction_role_refs"), field="s1_9.reproduction_role_refs")
        reproduction_hashes = _mapping(index.get("reproduction_role_sha256"), field="s1_9.reproduction_role_sha256")
        if reproduction_refs != _S1_9_V8_REPRODUCTION_REFS:
            raise Stage1ExitGateError("S1_11_S1_9_V8_REPRODUCTION_REF_WIRE_INVALID")
        if set(reproduction_hashes) != set(_S1_9_V8_REPRODUCTION_REFS):
            raise Stage1ExitGateError("S1_11_S1_9_V8_REPRODUCTION_HASH_WIRE_INVALID")
        reproduction_paths: dict[str, Path] = {}
        for role, ref in reproduction_refs.items():
            role_path = _safe_relative(index_dir, ref, field=f"s1_9_reproduction_{role}")
            digest = _hash(reproduction_hashes[role], field=f"s1_9_reproduction_{role}")
            if not role_path.is_file() or _hash_file(role_path) != digest:
                raise Stage1ExitGateError("S1_11_S1_9_V8_REPRODUCTION_FILE_HASH_MISMATCH")
            reproduction_paths[role] = role_path
        _validate_s1_9_current_v2_roles(index_dir, reproduction_paths, validation)
        numeric_report = _mapping(load_canonical_json(_safe_relative(index_dir, refs["numeric_report"], field="s1_9_numeric_report")), field="s1_9.numeric_report")
        _named_self_hash(numeric_report, field="s1_9.numeric_report", hash_field="report_hash")
        if numeric_report.get("schema_version") != "stage1-s1-9-numeric-report-v1" or numeric_report.get("status") != "PASS":
            raise Stage1ExitGateError("S1_11_S1_9_V8_NUMERIC_REPORT_INVALID")
        _source_map(numeric_report.get("implementation_source_sha256"), field="s1_9_source", expected_paths=_S1_9_SOURCE_PATHS)
        compatibility = _mapping(load_canonical_json(reproduction_paths["upstream_compatibility"]), field="s1_9.upstream_compatibility")
        _self_hash(compatibility, field="s1_9.upstream_compatibility")
        handoff = _mapping(compatibility.get("s1_8_v8_handoff"), field="s1_9.upstream_compatibility_handoff")
        s1_8_sources = _mapping(compatibility.get("s1_8_source_dependencies"), field="s1_9.s1_8_source_dependencies")
        handoff_sources = _mapping(handoff.get("implementation_source_sha256"), field="s1_9.s1_8_handoff_sources")
        handoff_reproduction_refs = _mapping(handoff.get("reproduction_role_refs"), field="s1_9.s1_8_reproduction_refs")
        handoff_reproduction_hashes = _mapping(handoff.get("reproduction_role_sha256"), field="s1_9.s1_8_reproduction_hashes")
        if (
            compatibility.get("schema_version") != "stage1-s1-9-upstream-compatibility-v7"
            or compatibility.get("status") != "PASS"
            or handoff.get("index_schema_version") != "stage1-s1-8-formalization-index-v8"
            or handoff.get("ddp_report_schema_version") != "stage1-s1-8-ddp-report-v8"
            or handoff.get("validation_schema_version") != "stage1-s1-8-validation-v8"
            or handoff.get("replay_schema_version") != "stage1-s1-8-replay-validation-v3"
            or handoff.get("comparison_table_schema_version") != "stage1-s1-8-comparison-table-v2"
            or handoff.get("array_bundle_schema_version") != "stage1-s1-8-array-bundle-v2"
            or s1_8_sources != handoff_sources
            or len(s1_8_sources) != 61
            or any(_hash(value, field="s1_8_source") != value for value in s1_8_sources.values())
            or set(handoff_reproduction_refs) != set(handoff_reproduction_hashes)
            or len(handoff_reproduction_refs) != 84
            or any(_hash(value, field="s1_8_reproduction") != value for value in handoff_reproduction_hashes.values())
        ):
            raise Stage1ExitGateError("S1_11_S1_9_V8_UPSTREAM_SOURCE_CLOSURE_INVALID")
        prelease = _mapping(load_canonical_json(reproduction_paths["prelease_gpu"]), field="s1_9.prelease_gpu")
        _self_hash(prelease, field="s1_9.prelease_gpu")
        prelease_quiescence = _mapping(prelease.get("quiescence"), field="s1_9.prelease_quiescence")
        _self_hash(prelease_quiescence, field="s1_9.prelease_quiescence")
        post_worker = _mapping(load_canonical_json(reproduction_paths["post_worker_quiescence"]), field="s1_9.post_worker_quiescence")
        _self_hash(post_worker, field="s1_9.post_worker_quiescence")
        if (
            prelease.get("schema_version") != "stage1-s1-9-gpu-prelease-v3"
            or prelease.get("status") != "PASS"
            or prelease_quiescence.get("schema_version") != "stage1-s1-9-gpu-quiescence-v3"
            or prelease_quiescence.get("status") != "PASS"
            or prelease_quiescence.get("phase") != "prelease"
            or post_worker.get("schema_version") != "stage1-s1-9-gpu-quiescence-v3"
            or post_worker.get("status") != "PASS"
            or post_worker.get("phase") != "post_worker"
        ):
            raise Stage1ExitGateError("S1_11_S1_9_V8_GPU_SOURCE_CLOSURE_INVALID")
    if binding.gate_id == "G1-RESUME":
        resume_report_path = _safe_relative(index_dir, refs["resume_report"], field="s1_10_resume_report")
        resume_report = _mapping(load_canonical_json(resume_report_path), field="s1_10.resume_report")
        _named_self_hash(resume_report, field="s1_10.resume_report", hash_field="report_hash")
        if resume_report.get("schema_version") != "stage1-s1-10-resume-report-v2" or resume_report.get("status") != "PASS":
            raise Stage1ExitGateError("S1_11_S1_10_V2_RESUME_REPORT_INVALID")
        _source_map(resume_report.get("implementation_source_sha256"), field="s1_10_source", expected_paths=_S1_10_SOURCE_PATHS)
        upstream = _mapping(index.get("upstream"), field="s1_10.index_upstream")
        if set(upstream) != {"s1_8", "s1_9"} or resume_report.get("upstream") != upstream or validation.get("upstream") != upstream or validation.get("schema_version") != "stage1-s1-10-validation-v2":
            raise Stage1ExitGateError("S1_11_S1_10_V2_UPSTREAM_BINDING_INVALID")
        expected_upstream = {
            "s1_8": ("stage1-s1-8-formalization-index-v8", "stage1.08_ddp_and_gradient_accumulation", "G1-DDP", 61, 84, {"prelease_gpu_quiescence", "post_worker_gpu_quiescence", "post_release_gpu_quiescence", "reacquire_preflight_gpu_quiescence"}),
            "s1_9": ("stage1-s1-9-formalization-index-v8", "stage1.09_precision_clipping_and_optimizer_boundaries", "G1-NUMERIC", 34, 27, {"upstream_compatibility", "prelease_gpu", "post_worker_quiescence"}),
        }
        upstream_fields = {"index_ref", "index_sha256", "index_artifact_hash", "producer_commit", "gate_artifact_hash", "role_sha256", "validation_sha256", "source_map_sha256", "source_map_entries", "reproduction_role_sha256", "reproduction_role_set_sha256", "reproduction_role_count", "schema_version", "task_id", "gate_id"}
        for name, (schema, task, gate_id, sources, reproduction_count, required_reproduction) in expected_upstream.items():
            row = _mapping(upstream[name], field=f"s1_10.upstream_{name}")
            role_hashes = _mapping(row.get("role_sha256"), field=f"s1_10.upstream_{name}_roles")
            reproduction_hashes = _mapping(row.get("reproduction_role_sha256"), field=f"s1_10.upstream_{name}_reproduction")
            if (
                set(row) != upstream_fields or row.get("schema_version") != schema or row.get("task_id") != task
                or row.get("gate_id") != gate_id or row.get("source_map_entries") != sources
                or row.get("reproduction_role_count") != reproduction_count or set(reproduction_hashes) != required_reproduction
                or "gate_record" not in role_hashes or any(_hash(value, field=f"s1_10_{name}_role") != value for value in role_hashes.values())
                or any(_hash(value, field=f"s1_10_{name}_reproduction") != value for value in reproduction_hashes.values())
            ):
                raise Stage1ExitGateError("S1_11_S1_10_V2_UPSTREAM_CLOSURE_INVALID")
        formal_observation_path = _safe_relative(index_dir, index.get("formal_observation_ref"), field="s1_10_formal_observation")
        if not formal_observation_path.is_file() or _hash_file(formal_observation_path) != _hash(index.get("formal_observation_sha256"), field="s1_10_formal_observation"):
            raise Stage1ExitGateError("S1_11_S1_10_V2_FORMAL_OBSERVATION_HASH_INVALID")
        formal_observation = _mapping(load_canonical_json(formal_observation_path), field="s1_10.formal_observation")
        _self_hash(formal_observation, field="s1_10.formal_observation")
        if formal_observation.get("schema_version") != "stage1-s1-10-formal-observation-v1" or formal_observation.get("status") != "PASS" or formal_observation.get("artifact_hash") != index.get("formal_observation_artifact_hash") or formal_observation.get("run_token_sha256") != index.get("formal_run_token_sha256"):
            raise Stage1ExitGateError("S1_11_S1_10_V2_FORMAL_OBSERVATION_INVALID")
        for role, reference_field, digest_field in (
            ("single", "formal_single_report_ref", "formal_single_report_sha256"),
            ("four_rank", "formal_four_rank_report_ref", "formal_four_rank_report_sha256"),
        ):
            worker_path = _safe_relative(index_dir, index.get(reference_field), field=f"s1_10_{role}_report")
            if not worker_path.is_file() or _hash_file(worker_path) != _hash(index.get(digest_field), field=f"s1_10_{role}_report"):
                raise Stage1ExitGateError("S1_11_S1_10_V2_WORKER_REPORT_HASH_INVALID")
            worker = _mapping(load_canonical_json(worker_path), field=f"s1_10.{role}_report")
            _self_hash(worker, field=f"s1_10.{role}_report")
            if worker.get("schema_version") != "stage1-s1-10-formal-worker-report-v1" or worker.get("status") != "PASS":
                raise Stage1ExitGateError("S1_11_S1_10_V2_WORKER_REPORT_INVALID")
        for field, expected_files in (
            ("chart_csv_sha256", {"resume-errors.csv", "state-timeline.csv"}),
            ("chart_svg_sha256", {"resume-errors.svg", "state-timeline.svg"}),
        ):
            chart_hashes = _mapping(index.get(field), field=f"s1_10.{field}")
            if set(chart_hashes) != expected_files:
                raise Stage1ExitGateError("S1_11_S1_10_V2_CHART_WIRE_INVALID")
            for filename, digest in chart_hashes.items():
                chart_path = _safe_relative(index_dir, filename, field="s1_10_chart")
                if not chart_path.is_file() or _hash_file(chart_path) != _hash(digest, field="s1_10_chart"):
                    raise Stage1ExitGateError("S1_11_S1_10_V2_CHART_HASH_INVALID")
    return {"binding": binding.to_dict(), "index_artifact_hash": index["artifact_hash"], "gate_requirements": sorted(requirements)}


def audit_exit_dependencies(root: Path, dependencies: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate all immutable upstream roles; missing or altered input fails closed."""

    if len(dependencies) != len(DEPENDENCIES):
        raise Stage1ExitGateError("S1_11_DEPENDENCY_COUNT_INVALID")
    bindings = [DependencyBinding.from_mapping(item) for item in dependencies]
    expected = [(gate, task) for gate, task, _ in DEPENDENCIES]
    observed = [(item.gate_id, item.task_id) for item in bindings]
    if observed != expected:
        raise Stage1ExitGateError("S1_11_DEPENDENCY_ORDER_OR_IDENTITY_INVALID")
    refs = [item.index_ref for item in bindings]
    if refs[0] != refs[1] or len(set(refs[2:])) != len(refs[2:]) or refs[0] in set(refs[2:]):
        raise Stage1ExitGateError("S1_11_DEPENDENCY_INDEX_CARDINALITY_INVALID")
    return [_validate_binding(root, binding) for binding in bindings]


def build_exit_gate_summary(
    root: Path,
    dependencies: Sequence[Mapping[str, object]],
    *,
    unresolved_failures: Sequence[Mapping[str, object]],
    charts: Mapping[str, Mapping[str, object]],
    formal_observation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a summary without manufacturing a formal verdict.

    With ``formal_observation=None`` the result is always ``NOT_RUN`` even if
    every supplied immutable input validates.  This makes local/synthetic
    rehearsal useful while making it incapable of claiming G1-EXIT PASS.
    """

    audits = audit_exit_dependencies(root, dependencies)
    if unresolved_failures:
        raise Stage1ExitGateError("S1_11_UNRESOLVED_FAILURES_PRESENT")
    if set(charts) != set(REQUIRED_CHARTS):
        raise Stage1ExitGateError("S1_11_REQUIRED_CHART_SET_INVALID")
    chart_rows: list[dict[str, str]] = []
    for chart_id in REQUIRED_CHARTS:
        chart = _mapping(charts[chart_id], field=f"chart_{chart_id}")
        if set(chart) != {"csv_ref", "csv_sha256", "svg_ref", "svg_sha256"}:
            raise Stage1ExitGateError("S1_11_CHART_BINDING_FIELDS_INVALID")
        for kind in ("csv", "svg"):
            if Path(str(chart[f"{kind}_ref"])).name != f"{chart_id}.{kind}":
                raise Stage1ExitGateError("S1_11_CHART_REF_WIRE_INVALID")
            path = _safe_relative(root, chart[f"{kind}_ref"], field=f"chart_{chart_id}_{kind}")
            digest = _hash(chart[f"{kind}_sha256"], field=f"chart_{chart_id}_{kind}")
            if not path.is_file() or _hash_file(path) != digest:
                raise Stage1ExitGateError("S1_11_CHART_FILE_HASH_MISMATCH")
        chart_rows.append({"chart_id": chart_id, **{key: chart[key] for key in sorted(chart)}})
    # This library is deliberately a read-only rehearsal consumer.  A caller
    # may not turn it into a formal PASS by supplying a self-hashed object;
    # only the formalizer derives an observation from the independently
    # audited frozen inputs and atomically emits a final summary.
    if formal_observation is not None:
        raise Stage1ExitGateError("S1_11_FORMAL_OBSERVATION_CALLER_SUPPLIED")
    formal = None
    status, verdict = "NOT_RUN", "BLOCKED_FORMAL_OBSERVATION_MISSING"
    body: dict[str, object] = {
        "schema_version": "stage1-s1-11-gate-summary-v1", "status": status,
        "gate_id": GATE_ID, "task_id": TASK_ID, "exit_verdict": verdict,
        "dependency_audits": audits, "unresolved_failure_count": 0,
        "charts": chart_rows, "formal_observation": formal,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def replay_exit_gate_summary(root: Path, summary: Mapping[str, object]) -> dict[str, object]:
    """Offline replay checks source closure instead of trusting summary text."""

    _self_hash(summary, field="summary")
    raw = _mapping(summary, field="summary")
    audits = raw.get("dependency_audits")
    if not isinstance(audits, list):
        raise Stage1ExitGateError("S1_11_SUMMARY_AUDITS_INVALID")
    dependencies = [
        _mapping(item, field="summary.audit").get("binding")
        for item in audits
    ]
    if not all(isinstance(item, Mapping) for item in dependencies):
        raise Stage1ExitGateError("S1_11_SUMMARY_BINDING_INVALID")
    rebuilt = audit_exit_dependencies(root, [dict(item) for item in dependencies if isinstance(item, Mapping)])
    if rebuilt != audits:
        raise Stage1ExitGateError("S1_11_OFFLINE_REPLAY_AUDIT_MISMATCH")
    charts = {
        _mapping(row, field="summary.chart")["chart_id"]: {
            key: value for key, value in _mapping(row, field="summary.chart").items() if key != "chart_id"
        }
        for row in raw.get("charts", []) if isinstance(row, Mapping)
    }
    if len(charts) != len(REQUIRED_CHARTS):
        raise Stage1ExitGateError("S1_11_OFFLINE_REPLAY_CHARTS_INVALID")
    rebuilt_summary = build_exit_gate_summary(
        root, [dict(item) for item in dependencies if isinstance(item, Mapping)],
        unresolved_failures=[], charts=charts,
        formal_observation=raw.get("formal_observation") if isinstance(raw.get("formal_observation"), Mapping) else None,
    )
    if rebuilt_summary != raw:
        raise Stage1ExitGateError("S1_11_OFFLINE_REPLAY_SUMMARY_MISMATCH")
    result = {"schema_version": "stage1-s1-11-replay-validation-v1", "status": "PASS", "summary_artifact_hash": raw["artifact_hash"]}
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _validate_verification_matrix(root: Path, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Require the plan's 28 measured/threshold/evidence records verbatim."""

    if len(rows) != len(REQUIREMENT_IDS):
        raise Stage1ExitGateError("S1_11_VERIFICATION_MATRIX_COUNT_INVALID")
    normalized: list[dict[str, object]] = []
    for expected_id, expected_gate_id, value in zip(REQUIREMENT_IDS, REQUIREMENT_GATE_IDS, rows, strict=True):
        row = _mapping(value, field="verification_row")
        if set(row) != {"requirement_id", "gate_id", "measured", "threshold", "status", "evidence"}:
            raise Stage1ExitGateError("S1_11_VERIFICATION_MATRIX_FIELDS_INVALID")
        if row.get("requirement_id") != expected_id or row.get("gate_id") != expected_gate_id or row.get("status") != "PASS":
            raise Stage1ExitGateError("S1_11_VERIFICATION_MATRIX_STATUS_INVALID")
        if not isinstance(row["measured"], (str, int, float)) or isinstance(row["measured"], bool):
            raise Stage1ExitGateError("S1_11_VERIFICATION_MATRIX_MEASURED_INVALID")
        if not isinstance(row["threshold"], str) or not row["threshold"]:
            raise Stage1ExitGateError("S1_11_VERIFICATION_MATRIX_THRESHOLD_INVALID")
        evidence = _mapping(row["evidence"], field="verification_evidence")
        if set(evidence) != {"ref", "sha256"}:
            raise Stage1ExitGateError("S1_11_VERIFICATION_MATRIX_EVIDENCE_FIELDS_INVALID")
        path = _safe_relative(root, evidence["ref"], field="verification_evidence")
        if not path.is_file() or _hash_file(path) != _hash(evidence["sha256"], field="verification_evidence"):
            raise Stage1ExitGateError("S1_11_VERIFICATION_MATRIX_EVIDENCE_HASH_MISMATCH")
        source = _mapping(load_canonical_json(path), field="verification_evidence")
        # S1.1 predates the per-gate formal-index wire shape: its one immutable
        # index deliberately carries both G1-ENTRY and G1-CONTRACT.  The
        # dependency adapter independently verifies the respective gate record.
        shared_s11_index = (
            source.get("schema_version") == "stage1-s1-1-formalization-index-v1"
            and row["gate_id"] in {"G1-ENTRY", "G1-CONTRACT"}
        )
        if (not shared_s11_index and source.get("gate_id") != row["gate_id"]) or source.get("status") != "PASS":
            raise Stage1ExitGateError("S1_11_VERIFICATION_MATRIX_EVIDENCE_IDENTITY_INVALID")
        normalized.append(row)
    return normalized


def build_exit_gate_evidence(
    root: Path,
    dependencies: Sequence[Mapping[str, object]],
    *,
    verification_matrix: Sequence[Mapping[str, object]],
    unresolved_failures: Sequence[Mapping[str, object]],
    charts: Mapping[str, Mapping[str, object]],
    formal_observation: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Assemble S1.11's machine-readable report roles without writing files.

    The caller must persist these canonical objects atomically.  In rehearsal
    mode the gate summary remains ``NOT_RUN``; only an independently produced
    formal observation can make the verdict a formal PASS.
    """

    matrix_rows = _validate_verification_matrix(root, verification_matrix)
    matrix = {
        "schema_version": "stage1-s1-11-requirements-matrix-v1", "task_id": TASK_ID,
        "gate_id": GATE_ID, "rows": matrix_rows,
    }
    matrix["artifact_hash"] = canonical_json_hash(matrix)
    summary = build_exit_gate_summary(
        root, dependencies, unresolved_failures=unresolved_failures, charts=charts,
        formal_observation=formal_observation,
    )
    report = {
        "schema_version": "stage1-s1-11-stage-report-v1", "status": summary["status"],
        "task_id": TASK_ID, "gate_id": GATE_ID, "summary_hash": summary["artifact_hash"],
        "requirements_matrix_hash": matrix["artifact_hash"],
        "scope_statement": "Stage 1 validates implementation correctness only; it makes no scientific conclusion.",
    }
    report["artifact_hash"] = canonical_json_hash(report)
    delivery = {
        "schema_version": "stage1-s1-11-delivery-manifest-v1", "task_id": TASK_ID,
        "gate_id": GATE_ID, "summary_hash": summary["artifact_hash"],
        "requirements_matrix_hash": matrix["artifact_hash"], "stage_report_hash": report["artifact_hash"],
        "dependency_index_hashes": {item["binding"]["gate_id"]: item["binding"]["index_sha256"] for item in summary["dependency_audits"]},
        "chart_ids": list(REQUIRED_CHARTS),
    }
    delivery["artifact_hash"] = canonical_json_hash(delivery)
    return {
        "requirements_matrix": matrix, "gate_summary": summary,
        "stage_report": report, "delivery_manifest": delivery,
    }
