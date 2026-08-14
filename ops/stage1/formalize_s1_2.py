#!/usr/bin/env python3
"""Publish the formal server-CPU G1-REGISTRY evidence for S1.2.

The registry Gate is a code-contract Gate and does not require model weights or a
GPU.  This entry point still runs only after a clean, explicitly identified Git
checkout and a previously published S1.1 handoff have been checked.  Every attempt
uses a new DATA_ROOT evidence directory and publishes canonical JSON only after the
targeted server regression and direct contract probes pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
INDEX_SCHEMA = "stage1-s1-2-formalization-index-v2"
REPORT_SCHEMA = "g1-registry-report-v2"
VALIDATION_SCHEMA = "stage1-s1-2-validation-v1"
TASK_ID = "stage1.02_architecture_and_parameter_registry"
REQUIRED_UPSTREAM_GATES = ("stage1.G1-ENTRY", "stage1.G1-CONTRACT")
CONFIG_COVERAGE_MANIFEST_REF = (
    "configs/stage1/s1-2-config-field-behavior-coverage-v2.json"
)
CONFIG_COVERAGE_SCHEMA_REF = (
    "schemas/stage1/s1-2-config-field-behavior-coverage-v2.json"
)


class Stage1S12FormalError(RuntimeError):
    """S1.2 formal evidence cannot be safely published."""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise Stage1S12FormalError(f"S1_2_GIT_COMMAND_FAILED:{arguments[0]}")
    return completed.stdout.strip()


def _logical_path(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S12FormalError(f"S1_2_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S12FormalError(f"S1_2_LOGICAL_REF_ESCAPE:{field}")
    path = (root.joinpath(*logical.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S12FormalError(f"S1_2_LOGICAL_REF_ESCAPE:{field}") from error
    return path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise Stage1S12FormalError(f"S1_2_IMMUTABLE_TARGET_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    from param_importance_nlp.contracts.jsonio import write_canonical_json

    write_canonical_json(path, value)


def _with_artifact_hash(value: dict[str, Any]) -> dict[str, Any]:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash

    payload = dict(value)
    payload["artifact_hash"] = canonical_json_hash(value)
    return payload


def _load_upstream_handoff(data_root: Path, index_ref: str) -> dict[str, Any]:
    from param_importance_nlp.contracts.jsonio import load_canonical_json

    path = _logical_path(data_root, index_ref, field="s1_1_index_ref")
    raw = load_canonical_json(path)
    if not isinstance(raw, dict):
        raise Stage1S12FormalError("S1_2_S1_1_INDEX_NOT_OBJECT")
    if raw.get("status") != "PASS" or raw.get("next_task_id") != TASK_ID:
        raise Stage1S12FormalError("S1_2_S1_1_HANDOFF_NOT_READY")
    gate_hashes = raw.get("gate_artifact_hashes")
    if not isinstance(gate_hashes, dict):
        raise Stage1S12FormalError("S1_2_S1_1_GATE_HASHES_MISSING")
    for gate_id in REQUIRED_UPSTREAM_GATES:
        digest = gate_hashes.get(gate_id)
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise Stage1S12FormalError(f"S1_2_UPSTREAM_GATE_INVALID:{gate_id}")
    raw["_index_path"] = path
    return raw


def _direct_contract_probe(
    work_root: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from param_importance_nlp.contracts import ResolvedConfig, ResolvedConfigV2
    from param_importance_nlp.core import BUFFER_POLICY, ImportanceState, ParameterRegistry
    from param_importance_nlp.stage1_config_behavior import compile_config_behavior

    class OrderModel(torch.nn.Module):
        def __init__(self, *, reverse: bool = False) -> None:
            super().__init__()
            names = ("z", "a") if reverse else ("a", "z")
            for name in names:
                setattr(self, name, torch.nn.Linear(2, 2, bias=False))

    model_a = OrderModel(reverse=False)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    registry_a = ParameterRegistry.from_model(model_a, optimizer_a)
    model_b = OrderModel(reverse=True)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    registry_b = ParameterRegistry.from_model(model_b, optimizer_b)

    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            raise Stage1S12FormalError(f"S1_2_CONTRACT_CHECK_FAILED:{check_id}")

    check(
        "canonical_order_stable",
        registry_a.eligible_names == registry_b.eligible_names
        and registry_a.coordinate_registry_hash == registry_b.coordinate_registry_hash,
        str(registry_a.eligible_names),
    )

    alias_model = torch.nn.Module()
    alias_model.attention = torch.nn.Linear(2, 2, bias=False)
    alias_model.alias = torch.nn.Module()
    alias_model.alias.weight = alias_model.attention.weight
    alias_optimizer = torch.optim.SGD([alias_model.attention.weight], lr=0.1, weight_decay=0.01)
    alias_registry = ParameterRegistry.from_model(alias_model, alias_optimizer)
    alias_record = alias_registry.record("alias.weight")
    check(
        "shared_alias_once",
        alias_record.canonical_name == "alias.weight"
        and len(alias_registry.eligible_records) == 1
        and alias_registry.canonical_name("attention.weight") == "alias.weight",
        str(alias_record.aliases),
    )

    frozen = torch.nn.Module()
    frozen.trainable = torch.nn.Parameter(torch.ones(2))
    frozen.locked = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    frozen_optimizer = torch.optim.SGD([frozen.trainable], lr=0.1)
    frozen_registry = ParameterRegistry.from_model(frozen, frozen_optimizer)
    check(
        "frozen_excluded",
        not frozen_registry.record("locked").eligible
        and frozen_registry.record("locked").eligibility_reason == "requires_grad_false",
        frozen_registry.record("locked").eligibility_reason,
    )

    buffer_model = torch.nn.Module()
    buffer_model.weight = torch.nn.Parameter(torch.ones(2))
    buffer_model.register_buffer("running_mean", torch.zeros(2))
    buffer_optimizer = torch.optim.SGD([buffer_model.weight], lr=0.1)
    buffer_registry = ParameterRegistry.from_model(buffer_model, buffer_optimizer)
    check(
        "buffer_excluded",
        BUFFER_POLICY == "excluded_from_parameter_registry-v1"
        and buffer_registry.eligible_names == ("weight",)
        and all(
            record["canonical_name"] != "running_mean"
            for record in buffer_registry.to_manifest()["records"]
        ),
        BUFFER_POLICY,
    )

    fixture_path = repository_root / "configs" / "local-fixtures" / "resolved-config-v1.json"
    if not fixture_path.is_file():
        raise Stage1S12FormalError("S1_2_CONFIG_BEHAVIOR_FIXTURE_MISSING")
    base_config = ResolvedConfig.from_mapping(
        json.loads(fixture_path.read_text(encoding="utf-8"))
    )
    execution_default = ResolvedConfigV2.resolve(
        base_config,
        task_id="stage0.05_config_run_identity_and_seeds",
    )
    execution_dry = ResolvedConfigV2.resolve(
        base_config,
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={"execution": {"dry_run": True, "fail_on_blocked": True}},
    )
    default_behavior = compile_config_behavior(execution_default)
    dry_behavior = compile_config_behavior(execution_dry)
    check(
        "config_behavior_component_branches",
        default_behavior["execution_action"] == "execute"
        and dry_behavior["execution_action"] == "plan_only"
        and default_behavior["blocked_input_action"] == "record_blocked"
        and dry_behavior["blocked_input_action"] == "raise",
        "dry_run/fail_on_blocked compile to execution decisions",
    )

    model_a.a.weight.grad = torch.ones_like(model_a.a.weight)
    model_a.z.weight.grad = None
    expected = registry_a.validate_model_gradients(expected_missing=["z.weight"])
    check(
        "expected_gradient_none_classified",
        expected["expected_missing"] == ("z.weight",),
        str(expected),
    )
    try:
        registry_a.validate_model_gradients()
    except Exception as error:  # exact error class is covered by the package tests
        abnormal_missing = "异常缺失梯度" in str(error)
    else:
        abnormal_missing = False
    check("abnormal_gradient_none_rejected", abnormal_missing, "fail-closed")

    manifest_path = work_root / "parameter-registry.json"
    registry_a.save(manifest_path)
    restored = ParameterRegistry.load(manifest_path)
    check("manifest_roundtrip", restored.to_manifest() == registry_a.to_manifest(), restored.registry_hash)
    restored.validate_against_model(model_a, optimizer_a)
    try:
        restored.validate_against_model(model_a, torch.optim.SGD(model_a.parameters(), lr=0.2))
    except Exception as error:
        lr_rejected = "学习率" in str(error)
    else:
        lr_rejected = False
    check("optimizer_lr_mapping_rejected", lr_rejected, "learning-rate drift")

    state = ImportanceState(registry_a, include_actual_update=True, device="cpu")
    state_schema = state.schema_manifest()
    state_bundle = state.save_bundle(work_root / "importance-state.bundle")
    restored_state, restored_bundle = ImportanceState.load_bundle(state_bundle.path, registry_a)
    check(
        "state_schema_and_bundle",
        state_schema["registry_hash"] == registry_a.registry_hash
        and restored_state.slot_names == state.slot_names
        and restored_bundle.manifest_sha256 == state_bundle.manifest_sha256,
        state_schema["schema_version"],
    )

    manifest = registry_a.to_manifest()
    report = {
        "schema_version": REPORT_SCHEMA,
        "gate_id": "G1-REGISTRY",
        "status": "PASS",
        "execution_scope": "formal_server_cpu",
        "registry": {
            "coordinate_registry_hash": registry_a.coordinate_registry_hash,
            "optimizer_contract_hash": registry_a.optimizer_contract_hash,
            "runtime_layout_hash": registry_a.runtime_layout_hash,
            "record_count": len(registry_a),
            "eligible_numel": sum(record.numel for record in registry_a.eligible_records),
        },
        "checks": checks,
        "producer_commit": "",
        "consumer_commit": "",
        "registry_manifest": manifest,
        "state_schema": state_schema,
        "buffer_policy": BUFFER_POLICY,
    }
    return report, {
        "coordinate_registry_hash": registry_a.coordinate_registry_hash,
        "eligible_numel": sum(record.numel for record in registry_a.eligible_records),
        "state_bundle_manifest_sha256": state_bundle.manifest_sha256,
    }


def _load_config_coverage(repository_root: Path) -> dict[str, Any]:
    """读取 S1.2 coverage manifest，并将其 schema/content 身份写入正式 evidence。"""

    from param_importance_nlp.atomic import sha256_file
    from param_importance_nlp.stage1_config_coverage import (
        coverage_summary,
        load_config_field_behavior_coverage,
    )

    manifest_path = repository_root / CONFIG_COVERAGE_MANIFEST_REF
    schema_path = repository_root / CONFIG_COVERAGE_SCHEMA_REF
    if not manifest_path.is_file() or not schema_path.is_file():
        raise Stage1S12FormalError("S1_2_CONFIG_COVERAGE_INPUT_MISSING")
    schema_sha256 = sha256_file(schema_path)
    coverage = load_config_field_behavior_coverage(
        manifest_path,
        expected_schema_sha256=schema_sha256,
    )
    summary = coverage_summary(coverage)
    return {
        "manifest_ref": CONFIG_COVERAGE_MANIFEST_REF,
        "manifest_sha256": sha256_file(manifest_path),
        "schema_ref": CONFIG_COVERAGE_SCHEMA_REF,
        "schema_sha256": schema_sha256,
        **summary,
    }


def execute(
    *,
    repository: str | Path,
    data_root: str | Path,
    s1_1_index_ref: str,
    attempt_id: str,
    timeout_seconds: int,
) -> dict[str, str]:
    repository_root = Path(repository).resolve(strict=True)
    data_root_path = Path(data_root).resolve(strict=True)
    source_root = repository_root / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    if _COMMIT_RE.fullmatch(_git(repository_root, "rev-parse", "HEAD")) is None:
        raise Stage1S12FormalError("S1_2_REPOSITORY_COMMIT_INVALID")
    if _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S12FormalError("S1_2_FORMAL_REQUIRES_CLEAN_WORKTREE")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", attempt_id) is None:
        raise Stage1S12FormalError("S1_2_ATTEMPT_ID_INVALID")
    commit = _git(repository_root, "rev-parse", "HEAD")
    branch = _git(repository_root, "branch", "--show-current")
    upstream = _load_upstream_handoff(data_root_path, s1_1_index_ref)

    evidence_dir = data_root_path / "evidence" / "stage1" / "s1-2-formal" / commit / attempt_id
    if evidence_dir.exists():
        raise Stage1S12FormalError(f"S1_2_ATTEMPT_ALREADY_EXISTS:{evidence_dir}")
    work_dir = data_root_path / "tmp" / "stage1-s1-2" / commit / attempt_id
    if work_dir.exists():
        raise Stage1S12FormalError(f"S1_2_WORK_ATTEMPT_ALREADY_EXISTS:{work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)

    test_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "tests/test_stage1_architecture_registry.py",
        "tests/test_core_registry_and_loss.py",
        "tests/test_core_estimators_and_accumulator.py",
        "tests/test_contracts_config.py",
        "tests/test_stage1_s12_config_coverage.py",
        "tests/test_artifact_schemas_and_loaders.py",
        "tests/test_assets.py",
    ]
    completed = subprocess.run(
        test_command,
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise Stage1S12FormalError(
            f"S1_2_SERVER_REGRESSION_FAILED:returncode={completed.returncode}"
        )
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "task_id": TASK_ID,
        "command": test_command,
        "returncode": completed.returncode,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }
    validation_path = work_dir / "validation.json"
    _write_new(validation_path, validation)

    config_coverage = _load_config_coverage(repository_root)
    report, probe_summary = _direct_contract_probe(work_dir, repository_root)
    report["producer_commit"] = commit
    report["consumer_commit"] = commit
    report["config_field_behavior_coverage"] = config_coverage
    report_path = work_dir / "g1-registry-report.json"
    _write_new(report_path, report)

    from param_importance_nlp.atomic import sha256_file
    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json

    index = {
        "schema_version": INDEX_SCHEMA,
        "status": "PASS",
        "gate_id": "G1-REGISTRY",
        "task_id": TASK_ID,
        "generator_git_commit": commit,
        "git_branch": branch,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "s1_1_index_ref": s1_1_index_ref,
        "s1_1_index_sha256": sha256_file(upstream["_index_path"]),
        "s1_1_gate_artifact_hashes": {
            key: upstream["gate_artifact_hashes"][key]
            for key in REQUIRED_UPSTREAM_GATES
        },
        "validation_ref": "validation.json",
        "validation_sha256": sha256_file(validation_path),
        "report_ref": "g1-registry-report.json",
        "report_sha256": sha256_file(report_path),
        "config_field_behavior_coverage_manifest_ref": CONFIG_COVERAGE_MANIFEST_REF,
        "config_field_behavior_coverage_manifest_sha256": config_coverage["manifest_sha256"],
        "config_field_behavior_coverage_artifact_hash": config_coverage["artifact_hash"],
        "config_field_behavior_coverage_schema_ref": CONFIG_COVERAGE_SCHEMA_REF,
        "config_field_behavior_coverage_schema_sha256": config_coverage["schema_sha256"],
        "config_contract_hashes": {
            family: contract["schema_hash"]
            for family, contract in config_coverage["config_contracts"].items()
        },
        "config_shared_schema_hashes": {
            family: contract["shared_schema_hashes"]
            for family, contract in config_coverage["config_contracts"].items()
        },
        "probe_summary": probe_summary,
        "next_task_id": "stage1.03_fixtures_and_oracles",
    }
    index["artifact_hash"] = canonical_json_hash(index)

    evidence_dir.mkdir(parents=True, exist_ok=False)
    for source in (validation_path, report_path):
        target = evidence_dir / source.name
        target.write_bytes(source.read_bytes())
    index_path = evidence_dir / "index.json"
    write_canonical_json(index_path, index)
    loaded = load_canonical_json(index_path)
    if not isinstance(loaded, dict) or loaded.get("artifact_hash") != index["artifact_hash"]:
        raise Stage1S12FormalError("S1_2_INDEX_RELOAD_FAILED")
    return {
        "index_ref": index_path.relative_to(data_root_path).as_posix(),
        "report_ref": (evidence_dir / "g1-registry-report.json").relative_to(data_root_path).as_posix(),
        "validation_ref": (evidence_dir / "validation.json").relative_to(data_root_path).as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-1-index-ref", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    arguments = parser.parse_args(argv)
    result = execute(
        repository=arguments.repository,
        data_root=arguments.data_root,
        s1_1_index_ref=arguments.s1_1_index_ref,
        attempt_id=arguments.attempt_id,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
