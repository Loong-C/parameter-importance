"""CPU-only contracts for the Stage 1 S1.7 real-GPU worker.

These tests intentionally never instantiate transformers or CUDA.  They pin
the array-only oracle and the source-side formal configuration so a Windows
developer host can still reject a malformed formal implementation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
import yaml

from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.runtime.training import TrainingStepRecord
from param_importance_nlp.stage1_single_gpu import T32_ATOL, T32_NORMALIZED_L2_LIMIT, T32_RTOL, _fixed_array_bundle
from param_importance_nlp.stage1_single_gpu_oracle import Stage1S17OracleError, max_error, raw_double_and_u


def _samples(offset: float = 0.0) -> list[dict[str, torch.Tensor]]:
    return [{"weight": torch.tensor([offset + index, offset + index + 0.25], dtype=torch.float32)} for index in range(4)]


def test_s17_oracle_requires_complete_b_and_rejects_shape_drift() -> None:
    left, right = _samples(), _samples(10.0)
    right[3] = {"weight": torch.ones((3,), dtype=torch.float32)}
    # B receives the same complete validator as A, so a late B tensor cannot
    # escape validation merely because B[0] happened to be well formed.
    with pytest.raises(Stage1S17OracleError, match="S17_ORACLE_TENSOR_INVALID"):
        raw_double_and_u(left, right)


def test_s17_oracle_rejects_broadcast_comparison_and_reports_each_tensor() -> None:
    left = {"weight": torch.ones((2,), dtype=torch.float32)}
    with pytest.raises(Stage1S17OracleError, match="S17_ORACLE_COMPARE_TENSOR_INVALID"):
        max_error(left, {"weight": torch.ones((1,), dtype=torch.float32)})
    checked = max_error(left, {"weight": left["weight"].clone()})
    assert checked["within_t32"] is True
    assert set(checked["per_tensor"]) == {"weight"}


def test_s17_fixed_array_bundle_freezes_zero_based_b_keys_without_model_load() -> None:
    """Exercise only tensor-map serialization wiring, never a CUDA/Pythia load."""

    value = {"weight": torch.ones((1,), dtype=torch.float32)}
    bundle = _fixed_array_bundle(
        full=value, online_mean=value, raw=value, double=value,
        explicit_u=value, streaming_u=value,
        locals_a=[value] * 4, locals_b=[value] * 4,
    )
    assert set(bundle) == {
        "full_gradient", "online_mean_gradient", "raw", "double", "explicit_u", "streaming_u",
        "local_a_0", "local_a_1", "local_a_2", "local_a_3",
        "local_b_0", "local_b_1", "local_b_2", "local_b_3",
    }
    assert not any(name.startswith("local_b_-") for name in bundle)


def test_s17_frozen_t32_and_task_catalog_decision_exemption() -> None:
    assert (T32_ATOL, T32_RTOL, T32_NORMALIZED_L2_LIMIT) == (1.0e-7, 1.0e-5, 1.0e-5)
    task = DEFAULT_TASK_CATALOG.get("stage1.07_single_gpu_pythia14m")
    assert task.formal_eligibility.requires_estimator_decision is False


def test_s17_yaml_keeps_global_default_but_freezes_local_fixture_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "run-ready" / "layers" / "formal-stage1-pythia14m.yaml").read_text(encoding="utf-8"))
    assert config["importance"]["require_decision_for_formal"] is True
    assert config["importance"]["estimator_decision_ref"] is None
    assert config["distributed"]["world_size"] == 1
    assert config["distributed"]["device_ids"] == [0]
    assert config["batching"]["global_batch_size"] == 4
    assert config["batching"]["microbatch_size"] == 1
    assert config["batching"]["accumulation_steps"] == 4
    assert config["runtime"]["temp_root"] == "tmp/stage1-s1-7"


def _formalizer_module() -> object:
    root = Path(__file__).resolve().parents[1]
    specification = importlib.util.spec_from_file_location("s17_formalizer_test", root / "ops" / "stage1" / "formalize_s1_7.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_s17_formalizer_import_is_platform_safe() -> None:
    module = _formalizer_module()
    assert module.TASK_ID == "stage1.07_single_gpu_pythia14m"


def test_s17_attempt_markers_are_mutually_exclusive(tmp_path: Path) -> None:
    module = _formalizer_module()
    module._failure_marker(tmp_path, RuntimeError("synthetic"), phase="test")
    with pytest.raises(module.Stage1S17FormalError, match="S17_ATTEMPT_MARKER_CONFLICT"):
        module._success_marker(tmp_path, gate_hash="0" * 64, validation_sha256="1" * 64)
    assert (tmp_path / "failed.json").is_file()
    assert not (tmp_path / "success.json").exists()


def test_s17_frozen_capability_rejects_any_other_formal_ref(tmp_path: Path) -> None:
    module = _formalizer_module()
    with pytest.raises(module.Stage1S17FormalError, match="S17_GPU_CAPABILITY_REF_NOT_FROZEN"):
        module._load_capability(tmp_path, "evidence/stage0/other/commits/capability_cuda.json", "GPU-test")
    assert module.EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH == "a536e191cd59318325289d238db727f8939767e384bfccd961ae7ca1c6a11ce4"


def test_s17_frozen_capability_accepts_only_verified_exact_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _formalizer_module()
    commit = tmp_path / module.EXPECTED_GPU_CAPABILITY_REF
    commit.parent.mkdir(parents=True)
    commit.write_text("synthetic", encoding="utf-8")
    identity = SimpleNamespace(
        commit_ref=module.EXPECTED_GPU_CAPABILITY_REF,
        object_ref="evidence/stage0/bootstrap/objects/capability_cuda/synthetic.json",
        task_id="stage0.01_baseline_and_safety",
        artifact_kind="capability_cuda",
        artifact_hash=module.EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH,
        config_hash="f" * 64,
    )
    loaded = SimpleNamespace(
        identity=identity,
        source_refs=("evidence/stage0/bootstrap/commits/environment.json",),
        payload={"schema_version": "runtime-capability-evidence-v1", "scope": "formal", "capability": "cuda", "status": "VERIFIED", "checked_at": "2026-08-15T00:00:00Z", "evidence_refs": ["evidence/stage0/bootstrap/commits/environment.json"], "metadata": {"allowed_gpu_uuids": ["GPU-approved"]}, "artifact_hash": ""},
    )
    from param_importance_nlp.contracts.jsonio import canonical_json_hash
    payload = dict(loaded.payload)
    payload["artifact_hash"] = canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
    loaded.payload = payload
    import param_importance_nlp.runtime.task_artifacts as artifacts
    monkeypatch.setattr(artifacts, "load_committed_task_artifact", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(module, "_sha", lambda path: module.EXPECTED_GPU_CAPABILITY_FILE_SHA256 if path == commit else "0" * 64)
    capability = module._load_capability(tmp_path, module.EXPECTED_GPU_CAPABILITY_REF, "GPU-approved")
    assert capability["allowed_gpu_uuids"] == ["GPU-approved"]


def test_s17_historical_patch_hash_drift_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _formalizer_module()
    checkout = tmp_path / "historical"
    for ref in module.HISTORICAL_G3_CRITICAL_SOURCE_REFS:
        path = checkout / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("historical", encoding="utf-8")
        current = tmp_path / "current" / ref
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("consumer", encoding="utf-8")
    monkeypatch.setattr(module, "_git", lambda _root, *args: "\n".join(module.HISTORICAL_G3_CRITICAL_SOURCE_REFS) if args[:2] == ("diff", "--name-only") else "c" * 40)
    monkeypatch.setattr(module, "_git_bytes", lambda *_args: b"one-byte-different-patch")
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(module.Stage1S17FormalError, match="S17_HISTORICAL_PRODUCER_PATCH_DRIFT"):
        module._historical_source_attestation(tmp_path / "current", checkout)


def test_s17_historical_patch_uses_full_index_and_real_repository_attests(tmp_path: Path) -> None:
    """Pin a host-independent patch form, then attest the real source history."""

    module = _formalizer_module()
    root = Path(__file__).resolve().parents[1]
    command = ["git", "-C", str(root), "diff", "--binary", "--full-index"]
    range_and_paths = [
        module.HISTORICAL_G3_PRODUCER, "HEAD", "--",
        *module.HISTORICAL_G3_CRITICAL_SOURCE_REFS,
    ]
    # --full-index must make an otherwise caller-selectable abbreviation width
    # irrelevant to the bytes that are hash-bound in the attestation.
    short = subprocess.check_output([*command, "--abbrev=7", *range_and_paths])
    long = subprocess.check_output([*command, "--abbrev=12", *range_and_paths])
    assert short == long
    assert hashlib.sha256(short).hexdigest() == module.EXPECTED_HISTORICAL_G3_PATCH_SHA256

    checkout = tmp_path / "historical-54b1"
    for ref in module.HISTORICAL_G3_CRITICAL_SOURCE_REFS:
        target = checkout / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output([
            "git", "-C", str(root), "show", f"{module.HISTORICAL_G3_PRODUCER}:{ref}",
        ]))
    attestation = module._historical_source_attestation(root, checkout)
    assert attestation["status"] == "PASS"
    assert attestation["critical_patch_sha256"] == module.EXPECTED_HISTORICAL_G3_PATCH_SHA256


def test_s17_fixture_parser_binds_identity_and_provenance_after_rehash() -> None:
    module = _formalizer_module()
    provenance = {
        role: {
            "logical_asset_id": frozen["logical_name"], "asset_id": frozen["asset_id"],
            "ready_manifest_sha256": frozen["ready_manifest_sha256"],
            "g3_resolution_ref": module.EXPECTED_G3_RESOLUTION,
            "g3_resolution_artifact_hash": module.EXPECTED_G3_PAYLOAD_HASH,
            "source_git_commit": module.HISTORICAL_G3_PRODUCER,
        }
        for role, frozen in module.EXPECTED_RUNTIME_ASSETS.items()
    }
    fixture = {
        "schema_version": "stage1-s1-7-fixture-manifest-v1", "fixture_id": module.FIXTURE_ID,
        "assets": provenance, "asset_identity": module.EXPECTED_RUNTIME_ASSETS,
        "batching": {"global_batch_size": 4, "microbatch_size": 1, "accumulation_steps": 4, "world_size": 1},
        "records": {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7], "training": [[8, 9, 10, 11], [12, 13, 14, 15]]},
        "token_sha256": {str(index): "a" * 64 for index in range(16)},
        "execution_contract": {"model_mode": "train", "dropout_probabilities": {"attention_dropout": 0.0}, "random_layer_policy": "all_pythia_dropout_probabilities_zero", "precision": {"compute": "float32", "gradient": "float32", "statistics": "float32", "reference": "float64", "amp": False}, "loss": {"task_type": "causal_lm", "reduction": "mean", "valid_tokens_per_microbatch": 2048, "ignore_index": -100}, "optimizer": {"type": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8, "foreach": False, "fused": False}, "gradient_clip_max_norm": 1.0, "scheduler": None, "statistical_contract": {"estimator_name": "u", "statistical_unit": "microbatch_mean_gradient", "weight_unit": "effective_target_tokens", "sampling_design": "ordered_disjoint_microbatches", "weights_exogenous": True, "common_mean_assumption": True}, "determinism": {"model_seed": 1707, "training_seed": 2707, "deterministic_algorithms": True, "allow_tf32": False, "cublas_workspace_config": ":4096:8"}},
    }
    fixture["fixture_hash"] = module._canonical(fixture)
    assert module._validate_fixture_manifest(fixture)["fixture_id"] == module.FIXTURE_ID
    fixture["assets"]["model"]["ready_manifest_sha256"] = "0" * 64
    fixture["fixture_hash"] = module._canonical({key: value for key, value in fixture.items() if key != "fixture_hash"})
    with pytest.raises(module.Stage1S17FormalError, match="S17_FIXTURE_PROVENANCE_IDENTITY_INVALID:model"):
        module._validate_fixture_manifest(fixture)


def test_s17_schemas_are_strict_and_bind_fixture_identity_and_next_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root / "schemas" / "stage1" / "s1-7-fixture-manifest-v1.json").read_text(encoding="utf-8"))
    report = json.loads((root / "schemas" / "stage1" / "s1-7-single-gpu-report-v1.json").read_text(encoding="utf-8"))
    index = json.loads((root / "schemas" / "stage1" / "s1-7-formalization-index-v1.json").read_text(encoding="utf-8"))
    assert fixture["definitions"]["model_identity"]["properties"]["asset_id"]["const"].startswith("11dd681a")
    assert fixture["definitions"]["provenance"]["properties"]["directory_content_sha256"]["type"] == ["string", "null"]
    assert "observer" not in report["definitions"]
    for row in ("gradient_ready_row", "parameter_post_row", "attempt_commit_row"):
        assert report["definitions"][row]["additionalProperties"] is False
    assert index["properties"]["next_task_ids"]["const"] == ["stage1.08_ddp_and_gradient_accumulation", "stage1.09_precision_clipping_and_optimizer_boundaries"]
    assert {"historical_producer_attestation", "historical_g3_replay", "historical_g3_replay_stdout", "historical_g3_replay_stderr", "resource_summary"} <= set(index["definitions"]["reproduction_refs"]["required"])
    assert "success_marker" not in index["definitions"]["reproduction_refs"]["required"]
    assert report["definitions"]["training_record"]["properties"]["status"]["const"] == "COMMITTED"
    assert report["definitions"]["training_record"]["properties"]["estimator_name"]["enum"] == ["u", None]


def _training_record(*, estimator_name: str | None) -> dict[str, object]:
    return TrainingStepRecord(
        attempt_index=1, global_step=1, status="COMMITTED",
        batch_ids=("a", "b", "c", "d"), mean_loss=1.0,
        effective_count=8192, global_gradient_norm=1.0, clip_factor=1.0,
        estimator_name=estimator_name, parameter_post_state_hash="a" * 64,
        attempt_commit_state_hash="b" * 64,
    ).to_dict()


def test_s17_training_step_record_schema_uses_actual_committed_wire() -> None:
    root = Path(__file__).resolve().parents[1]
    module = _formalizer_module()
    report_schema = json.loads((root / "schemas" / "stage1" / "s1-7-single-gpu-report-v1.json").read_text(encoding="utf-8"))
    s16_spec = importlib.util.spec_from_file_location("s17_s16_schema", root / "ops" / "stage1" / "formalize_s1_6.py")
    assert s16_spec is not None and s16_spec.loader is not None
    s16 = importlib.util.module_from_spec(s16_spec); s16_spec.loader.exec_module(s16)
    registry = {"s1-7-single-gpu-report-v1.json": report_schema, report_schema["$id"]: report_schema}
    for record in (_training_record(estimator_name=None), _training_record(estimator_name="u")):
        s16._validate_schema(record, report_schema["definitions"]["training_record"], registry, document=report_schema, path="record")
    invalid = _training_record(estimator_name=None); invalid["status"] = "PASS"
    with pytest.raises(Exception):
        s16._validate_schema(invalid, report_schema["definitions"]["training_record"], registry, document=report_schema, path="record")
    assert module.ACCUMULATOR_FIELDS == {
        "positive", "negative_mass", "raw", "raw_clipped", "data_movement", "data_displacement",
        "total_movement", "total_displacement", "weight_decay_movement", "weight_decay_displacement",
        "actual_update_raw_importance", "magnitude", "initial_parameters", "last_parameters",
    }


def test_s17_report_context_resource_and_replay_contracts_reject_rehashed_drift() -> None:
    module = _formalizer_module()
    fixture = {"fixture_id": module.FIXTURE_ID, "fixture_hash": "f" * 64}
    assets = {"model": {"asset_id": "a" * 64}}
    report = {
        "execution_commit": "c" * 40, "run_token": "r" * 64,
        "fixture": fixture, "assets": assets,
        "device": {
            "logical_device": 0, "cuda_visible_devices": "GPU-approved",
            "physical_gpu_uuid": "GPU-approved", "physical_gpu_index_at_discovery": 3,
            "cuda_device_count": 1, "device_name": "NVIDIA A100-SXM4-80GB",
        },
    }
    context = {"execution_commit": "c" * 40, "run_token": "r" * 64, "fixture": fixture, "assets": assets, "approved_gpu_uuid": "GPU-approved", "physical_gpu_index": 3}
    module._validate_report_context(report, **context)
    rehashed = dict(report); rehashed["run_token"] = "0" * 64
    with pytest.raises(module.Stage1S17FormalError, match="S17_WORKER_REPORT_CONTEXT_INVALID"):
        module._validate_report_context(rehashed, **context)
    bounds = {"allocated_before": 100, "allocated_after": 100 + 16 * 1024**2, "allocated_growth_limit": 16 * 1024**2, "reserved_before": 11, "reserved_after": 7, "reserved_growth_explained_as_allocator_cache": 0}
    assert module._validate_temporary_state_bounds(bounds)["allocated_growth_limit"] == 16 * 1024**2
    bad_bounds = dict(bounds); bad_bounds["allocated_growth_limit"] = 1
    with pytest.raises(module.Stage1S17FormalError, match="S17_RESOURCE_BOUND_RECOMPUTE_FAILED"):
        module._validate_temporary_state_bounds(bad_bounds)
    decoded = {"fixed": {"weight": torch.ones((3,), dtype=torch.float32)}}
    assert module._validate_retained_tensor_bytes({"resources": {"retained_tensor_bytes": 12}}, decoded) == 12
    with pytest.raises(module.Stage1S17FormalError, match="S17_RETAINED_TENSOR_BYTES_RECOMPUTE_FAILED"):
        module._validate_retained_tensor_bytes({"resources": {"retained_tensor_bytes": 11}}, decoded)
    error = {"max_abs_error": 0.0, "parameter": "weight", "index": [0], "max_scaled_error": 0.0, "normalized_l2_error": 0.0, "near_zero_coordinates": 1, "violation_count": 0, "within_t32": True, "per_tensor": {"weight": {"max_abs_error": 0.0, "max_scaled_error": 0.0, "normalized_l2_error": 0.0, "near_zero_coordinates": 1, "violation_count": 0, "within_t32": True}}}
    replay = {"schema_version": "stage1-s1-7-offline-replay-v1", "status": "PASS", "oracle_import_isolated": True, "checks": {check: error for check in module.REPLAY_CHECK_IDS}, "replay_hash": "a" * 64}
    module._validate_role_schemas(Path(__file__).resolve().parents[1], {"replay": replay})
    replay["checks"] = {"replacement": error, **{key: value for key, value in replay["checks"].items() if key != "full_vs_offline_mean"}}
    with pytest.raises(module.Stage1S17FormalError, match="S17_SCHEMA_VALIDATION_FAILED:replay"):
        module._validate_role_schemas(Path(__file__).resolve().parents[1], {"replay": replay})


def test_s17_observer_contract_rejects_wrong_sequence_or_statistics_snapshot() -> None:
    module = _formalizer_module()
    rows = [
        {"boundary": boundary, "importance_snapshot": None if boundary == "attempt_commit" else "absent"}
        for boundary in ["gradient_ready", "parameter_post", "attempt_commit"] * 2
    ]
    module._validate_trace_observer_contract({"observer_rows": rows, "accumulator_fields": []}, statistics=False)
    wrong = {"observer_rows": list(rows), "accumulator_fields": []}
    wrong["observer_rows"][0] = {"boundary": "parameter_post", "importance_snapshot": "absent"}
    with pytest.raises(module.Stage1S17FormalError, match="S17_TRAINING_OBSERVER_BOUNDARY_SEQUENCE_INVALID"):
        module._validate_trace_observer_contract(wrong, statistics=False)
    fields = sorted(module.ACCUMULATOR_FIELDS)
    on_rows = [
        {"boundary": boundary, "importance_snapshot": ({"successful_steps": step, "skipped_steps": 0} if boundary == "attempt_commit" else "absent")}
        for step, boundary in ((1, "gradient_ready"), (1, "parameter_post"), (1, "attempt_commit"), (2, "gradient_ready"), (2, "parameter_post"), (2, "attempt_commit"))
    ]
    module._validate_trace_observer_contract({"observer_rows": on_rows, "accumulator_fields": fields}, statistics=True)
    on_rows[-1]["importance_snapshot"] = {"successful_steps": 1, "skipped_steps": 0}
    with pytest.raises(module.Stage1S17FormalError, match="S17_TRAINING_ON_SNAPSHOT_STEP_INVALID"):
        module._validate_trace_observer_contract({"observer_rows": on_rows, "accumulator_fields": fields}, statistics=True)


def test_s17_registry_reload_heatmap_and_validation_check_contracts_are_exact() -> None:
    module = _formalizer_module()
    registry_report = {
        "registry": {"coordinate_registry_hash": "c" * 64, "optimizer_contract_hash": "o" * 64, "runtime_layout_hash": "r" * 64},
        "registry_audit": {"eligible_numel": 3, "model_trainable_numel": 3, "coordinate_registry_hash": "c" * 64, "reload_coordinate_registry_hash": "c" * 64, "fresh_reload_coordinate_registry_hash": "c" * 64, "fresh_reload_optimizer_contract_hash": "o" * 64, "fresh_reload_runtime_layout_hash": "r" * 64, "shared_weight_alias_contract": "registry_from_model_remove_duplicate_false"},
    }
    module._validate_registry_audit(registry_report)
    registry_report["registry_audit"]["fresh_reload_runtime_layout_hash"] = "x" * 64
    with pytest.raises(module.Stage1S17FormalError, match="S17_REGISTRY_AUDIT_INVALID"):
        module._validate_registry_audit(registry_report)
    table = {"rows": [
        {"scope": "global", "metric": "raw_vs_oracle", "max_abs_error": 0.0, "parameter": "__global__"},
        {"scope": "per_tensor", "module": "embed", "layer": "input", "metric": "raw_vs_oracle", "max_scaled_error": 0.2},
        {"scope": "per_tensor", "module": "block", "layer": "0", "metric": "raw_vs_oracle", "max_scaled_error": 0.7},
        {"scope": "per_tensor", "module": "block", "layer": "0", "metric": "raw_vs_oracle", "max_scaled_error": 0.5},
    ]}
    heatmap = module._heatmap_rows(table)
    assert heatmap == [
        {"scope": "layer", "group": "0", "metric": "raw_vs_oracle", "max_scaled_error": 0.7},
        {"scope": "layer", "group": "input", "metric": "raw_vs_oracle", "max_scaled_error": 0.2},
        {"scope": "module", "group": "block", "metric": "raw_vs_oracle", "max_scaled_error": 0.7},
        {"scope": "module", "group": "embed", "metric": "raw_vs_oracle", "max_scaled_error": 0.2},
    ]
    svg = module._svg_projection("module-metric-heatmap.csv", heatmap)
    assert svg.count("<rect ") == len(heatmap)
    assert 'data-scope="layer" data-group="0" data-metric="raw_vs_oracle" data-max-scaled-error="0.7"' in svg
    checks = [{"check_id": check_id, "status": "PASS", "detail": "actual check"} for check_id in module.GATE_CHECK_IDS]
    module._validate_validation_checks({"checks": checks})
    checks[0]["check_id"] = checks[1]["check_id"]
    with pytest.raises(module.Stage1S17FormalError, match="S17_VALIDATION_CHECK_IDS_INVALID"):
        module._validate_validation_checks({"checks": checks})
    root = Path(__file__).resolve().parents[1]
    validation = {
        "schema_version": "stage1-s1-7-validation-v1", "status": "PASS", "task_id": module.TASK_ID,
        "gate_id": module.GATE_ID, "producer_commit": "c" * 40,
        "checks": [{"check_id": check_id, "status": "PASS", "detail": "actual check"} for check_id in module.GATE_CHECK_IDS],
        "role_sha256": {key: "a" * 64 for key in ("fixture_manifest", "single_gpu_report", "gradient_bundle", "comparison_table", "gate_record")},
        "replay_sha256": "a" * 64,
        "csv_sha256": {key: "a" * 64 for key in ("gradient-parity.csv", "training-parity.csv", "parameter-error.csv", "resource-timeline.csv", "module-metric-heatmap.csv")},
        "svg_sha256": {key: "a" * 64 for key in ("gradient-parity.svg", "training-parity.svg", "parameter-error.svg", "resource-timeline.svg", "module-metric-heatmap.svg")},
        "artifact_hash": "a" * 64,
    }
    module._validate_role_schemas(root, {"validation": validation})
    validation["checks"][1]["check_id"] = "replacement"
    with pytest.raises(module.Stage1S17FormalError, match="S17_SCHEMA_VALIDATION_FAILED:validation"):
        module._validate_role_schemas(root, {"validation": validation})


def test_s17_resource_summary_separates_historical_checkout(tmp_path: Path) -> None:
    module = _formalizer_module()
    (tmp_path / "worker.stdout.txt").write_bytes(b"run-output")
    historical = tmp_path / "historical-g3" / "src"
    historical.mkdir(parents=True)
    (historical / "producer.py").write_bytes(b"historical-source")
    summary = module._resource_summary(tmp_path)
    assert summary["attempt_file_total_bytes"] == len(b"run-output")
    assert summary["historical_checkout_bytes"] == len(b"historical-source")
    module._validate_resource_summary(tmp_path, summary)


@pytest.mark.parametrize("failure_point", ["schema", "rename"])
def test_s17_publish_failure_never_leaves_success_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    """Exercise the outer formalizer cleanup around the final staging phase."""

    module = _formalizer_module()
    repository = Path(__file__).resolve().parents[1]
    commit = "c" * 40
    cache = tmp_path / "cache"
    cache.mkdir()
    original_replace = module.os.replace

    def git(_root: Path, *arguments: str) -> str:
        if arguments[:2] == ("rev-parse", "HEAD"):
            return commit
        if arguments[:2] == ("status", "--porcelain"):
            return ""
        if arguments[:2] == ("branch", "--show-current"):
            return "test"
        return ""

    def asset_fixture(_repository: Path, _root: Path, work: Path, _ref: str) -> tuple[dict[str, object], dict[str, object]]:
        fixture_file = work / "fixture-inputs.safetensors"
        fixture_file.write_bytes(b"fixture")
        digest = module._sha(fixture_file)
        module._write(work / "historical-producer-attestation.json", module._with_hash({"critical_patch_sha256": module.EXPECTED_HISTORICAL_G3_PATCH_SHA256}))
        historical_replay = {"status": "PASS"}
        historical_replay["replay_hash"] = module._canonical(historical_replay)
        module._write(work / "historical-g3-replay.json", historical_replay)
        (work / "historical-g3-replay.stdout.txt").write_text("", encoding="utf-8")
        (work / "historical-g3-replay.stderr.txt").write_text("", encoding="utf-8")
        fixture = {"fixture_id": module.FIXTURE_ID, "fixture_hash": "f" * 64}
        return fixture, {"fixture_file": fixture_file.name, "model_root": "/model", "tokenizer_root": "/tokenizer", "worker_assets": {"pile": {"fixture_file_sha256": digest, "full_hash_seconds": 1.0, "full_hash_bytes": 2, "qualified_resolution_hashed_bytes": 1, "dataset_rehash_bytes": 1}}, "g3_resolution_artifact_hash": "a" * 64, "historical_producer": {}, "historical_replay_sha256": "b" * 64}

    class Lease:
        def __init__(self, root: Path, _identity: object) -> None:
            self.root = root
        def acquire(self) -> None: pass
        def heartbeat(self) -> None: pass
        def close(self) -> None: pass
        def release(self, *, outcome: str) -> Path:
            path = self.root / "lease-source.json"
            path.write_text('{"outcome":"' + outcome + '"}', encoding="utf-8")
            return path

    def discover(uuid: str) -> dict[str, object]:
        return {"selected": {"uuid": uuid, "physical_index": 0}, "discovery": [], "health_sha256": "d" * 64, "external_compute_processes": []}

    def worker(_repository: Path, work: Path, _plan: object, _uuid: str, _timeout: int, _lease: object) -> dict[str, object]:
        arrays = work / "s1-7-arrays.safetensors"
        arrays.write_bytes(b"arrays")
        module._write(work / "worker-plan.json", _plan)
        network = work / "network"
        network.mkdir()
        module._write(network / "audit.json", module._with_hash({"status": "COMPLETE", "external_attempts": []}))
        (work / "worker-start.json").write_text("{}", encoding="utf-8")
        (work / "worker.stdout.txt").write_text("", encoding="utf-8")
        (work / "worker.stderr.txt").write_text("", encoding="utf-8")
        comparison = {"within_t32": True}
        fixed = {"comparisons": {"full_vs_online": comparison, "explicit_u_vs_streaming_u": comparison}, "microbatch_resources": [{"backward_seconds": 0.0}] * 8}
        trace = {"temporary_state_bounded": True, "temporary_state_bounds": {"allocated_before": 0, "allocated_after": 0, "allocated_growth_limit": 16 * 1024**2, "reserved_before": 0, "reserved_after": 0, "reserved_growth_explained_as_allocator_cache": 0}}
        report = {"offline_guard_ref": "audit.json", "resources": {"timeline": [{}, {}, {}], "gradient_dump_bytes": arrays.stat().st_size, "retained_tensor_bytes": 1, "fixed_and_training_seconds": 0.0, "fixed_gradient_seconds": 0.0, "training_seconds": 0.0, "safetensors_serialization_seconds": 0.0, "wall_seconds": 0.0}, "training": {"statistics_on": trace, "statistics_off": trace, "bitwise_final_state_equal": True, "per_step_parity": [{}, {}]}, "fixed_state": fixed, "registry": {"coordinate_registry_hash": "x", "optimizer_contract_hash": "o", "runtime_layout_hash": "l"}, "registry_audit": {"coordinate_registry_hash": "x", "reload_coordinate_registry_hash": "x", "fresh_reload_coordinate_registry_hash": "x", "fresh_reload_optimizer_contract_hash": "o", "fresh_reload_runtime_layout_hash": "l", "eligible_numel": 1, "model_trainable_numel": 1}, "arrays": {}}
        module._write(work / "worker-report.json", report)
        return report

    def probe(_repository: Path, _uuid: str, work: Path) -> dict[str, object]:
        (work / "cuda-min-probe.stdout.txt").write_text("", encoding="utf-8")
        (work / "cuda-min-probe.stderr.txt").write_text("", encoding="utf-8")
        return {"stdout_sha256": module._sha(work / "cuda-min-probe.stdout.txt"), "stderr_sha256": module._sha(work / "cuda-min-probe.stderr.txt")}

    def charts(work: Path, _report: object, _table: object) -> tuple[dict[str, str], dict[str, str]]:
        csv = {name: "" for name in ("gradient-parity.csv", "training-parity.csv", "parameter-error.csv", "resource-timeline.csv", "module-metric-heatmap.csv")}
        svg = {name.replace(".csv", ".svg"): "" for name in csv}
        for name, payload in {**csv, **svg}.items():
            (work / name).write_text(payload, encoding="utf-8")
        return ({name: module._sha(work / name) for name in csv}, {name: module._sha(work / name) for name in svg})

    def negative(work: Path, _fixture: object, _report: object, **_context: object) -> bool:
        probe_root = work / "negative-marker-probe"
        probe_root.mkdir()
        (probe_root / "failed.json").write_text("{}", encoding="utf-8")
        (work / "negative-checks.json").write_text("{}", encoding="utf-8")
        return True

    monkeypatch.setattr(module, "_git", git)
    monkeypatch.setattr(module, "load_s1_6", lambda *_args: {"s1_6_gate_artifact_hash": module.EXPECTED_S1_6_GATE_HASH})
    monkeypatch.setattr(module, "_asset_fixture", asset_fixture)
    monkeypatch.setattr(module, "_load_capability", lambda *_args: {"allowed_gpu_uuids": ["GPU-test"]})
    monkeypatch.setattr(module, "discover_gpu", discover)
    monkeypatch.setattr(module, "_min_cuda_probe", probe)
    monkeypatch.setattr(module.os, "statvfs", lambda _path: SimpleNamespace(f_bavail=200 * 1024**3, f_frsize=1), raising=False)
    monkeypatch.setattr(module, "_pid_fingerprint", lambda _pid, token: {"pid": 1, "run_token": token})
    monkeypatch.setattr(module, "_worker", worker)
    monkeypatch.setattr(module, "_validate_report", lambda _report, **_context: None)
    monkeypatch.setattr(module, "_load_arrays", lambda work, _report: ({}, {"file_sha256": module._sha(work / "s1-7-arrays.safetensors"), "file_size_bytes": (work / "s1-7-arrays.safetensors").stat().st_size}))
    monkeypatch.setattr(module, "_oracle_replay", lambda _arrays: {"oracle_import_isolated": True, "checks": {check: {"within_t32": True} for check in module.REPLAY_CHECK_IDS}})
    monkeypatch.setattr(module, "_comparison_table", lambda *_args: {})
    monkeypatch.setattr(module, "_charts", charts)
    monkeypatch.setattr(module, "_verify_charts", lambda *_args: True)
    monkeypatch.setattr(module, "_negative_checks", negative)
    monkeypatch.setattr(module, "_task_catalog_decision_exempt", lambda: True)
    import param_importance_nlp.runtime.operations as operations
    monkeypatch.setattr(operations, "ProjectGpuLease", Lease)
    if failure_point == "schema":
        monkeypatch.setattr(module, "_validate_role_schemas", lambda *_args: (_ for _ in ()).throw(module.Stage1S17FormalError("synthetic schema failure")))
    else:
        monkeypatch.setattr(module, "_validate_role_schemas", lambda *_args: None)
        monkeypatch.setattr(module.os, "replace", lambda source, target: (_ for _ in ()).throw(OSError("synthetic rename failure")) if Path(target).name == "atomic" else original_replace(source, target))
    expected_failure = "synthetic schema failure" if failure_point == "schema" else "synthetic rename failure"
    with pytest.raises(Exception, match=expected_failure):
        module.execute(repository=repository, data_root=tmp_path, s1_6_index_ref="unused", g3_resolution_ref=module.EXPECTED_G3_RESOLUTION, gpu_capability_ref=module.EXPECTED_GPU_CAPABILITY_REF, approved_gpu_uuid="GPU-test", attempt_id="atomic", lease_owner="test")
    work = tmp_path / "tmp" / "stage1-s1-7" / commit / "atomic"
    staging = tmp_path / "evidence" / "stage1" / "s1-7-formal" / commit / ".atomic.publishing"
    target = tmp_path / "evidence" / "stage1" / "s1-7-formal" / commit / "atomic"
    assert (work / "failed.json").is_file()
    assert not (work / "success.json").exists()
    assert not target.exists()
    assert not (staging / "success.json").exists()
