"""Stage 0 S0.11 deterministic fixture, offline guard, and G9 contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from param_importance_nlp.cli import _load_mapping, _validate_project_json_schema
from param_importance_nlp.contracts import ResolvedConfig
from param_importance_nlp.contracts import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.deterministic_fixture import validate_deterministic_fixture
from param_importance_nlp.offline_guard import destination_is_local
from param_importance_nlp.runtime import TaskRuntimeEnvironment
from param_importance_nlp.stage0_bootstrap import (
    Stage0SourceBinding,
    build_stage0_formal_config,
)
from param_importance_nlp.stage0_g7_recovery import (
    G7RecoverySourceBinding,
    Stage0G7RecoveryFormalState,
    _config_overrides as _g7_config_overrides,
    _FORMAL_RECOVERY_PROVIDERS,
)
from param_importance_nlp.stage0_g8 import Stage0G8FormalState
from param_importance_nlp.stage0_g9 import (
    G9SourceBinding,
    _load_hashed,
    _validate_runbook,
    build_stage0_g9_config,
)
from param_importance_nlp.stage0_g9_replay import (
    REPLAY_PLAN_SCHEMA,
    _layer_results,
    _validate_plan,
)
from param_importance_nlp.atomic import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _template() -> ResolvedConfigV2:
    base = ResolvedConfig.resolve(
        _load_mapping(ROOT / "configs/local-fixtures/resolved-config-v1.json"),
        _load_mapping(ROOT / "configs/run-ready/layers/formal-stage1-pythia14m.yaml"),
    )
    return ResolvedConfigV2.resolve(
        base,
        task_id="stage1.07_single_gpu_pythia14m",
        overrides=_load_mapping(ROOT / "configs/run-ready/v2/stage1-pythia14m-formal.yaml"),
    )


def test_checked_in_deterministic_fixture_replays_exactly() -> None:
    value = validate_deterministic_fixture(
        ROOT / "fixtures/stage0/deterministic-training-v1.json"
    )
    assert value["fixture_id"] == "stage0-tiny-token-lm-v1"
    assert value["expected"]["effective_count"] == 6
    assert value["mathematical_semantics"]["importance_mathematics_implemented"] is False


def test_offline_destination_policy_allows_only_local_endpoints() -> None:
    local = frozenset({"localhost", "fixture-host"})
    assert destination_is_local(("127.0.0.1", 1234), local_names=local)
    assert destination_is_local(("::1", 1234), local_names=local)
    assert destination_is_local(("fixture-host", 1234), local_names=local)
    assert destination_is_local("/tmp/fixture.sock", local_names=local)
    assert not destination_is_local(("10.0.0.1", 1234), local_names=local)
    assert not destination_is_local(("example.com", 443), local_names=local)


@pytest.mark.fault
def test_sitecustomize_guard_blocks_external_dns_and_writes_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    guard = ROOT / "ops/stage0/offline_guard"
    source = ROOT / "src"
    environment = os.environ.copy()
    environment.update(
        {
            "PARAM_IMPORTANCE_OFFLINE_GUARD": "1",
            "PARAM_IMPORTANCE_NETWORK_AUDIT_DIR": str(audit),
            "PARAM_IMPORTANCE_OFFLINE_ALLOWED_HOSTS": "localhost,127.0.0.1,::1",
            "PYTHONPATH": os.pathsep.join((str(source), str(guard))),
        }
    )
    script = (
        "import socket; "
        "local=socket.socket().connect_ex(('127.0.0.1', 9)); "
        "blocked=False; "
        "\ntry: socket.getaddrinfo('example.com', 443)\n"
        "except socket.gaierror: blocked=True\n"
        "assert blocked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    files = list(audit.glob("python-network-*.json"))
    assert len(files) == 1
    value = json.loads(files[0].read_text(encoding="utf-8"))
    declared = value.pop("artifact_hash")
    from param_importance_nlp.contracts import canonical_json_hash

    assert declared == canonical_json_hash(value)
    assert value["status"] == "COMPLETE"
    assert value["allowed_local_calls"] >= 1
    assert len(value["external_attempts"]) == 1


def test_g9_matrix_and_schemas_are_hash_bound_and_valid() -> None:
    matrix = _load_hashed(
        ROOT / "configs/stage0/g9-test-matrix-v1.json",
        schema="stage0-g9-test-matrix-v1",
        field="matrix",
    )
    assert [item["layer"] for item in matrix["layers"]] == [
        "local_cpu",
        "server_cpu",
        "single_gpu",
        "four_gpu",
        "fault",
        "replay",
    ]
    assert all(item["hard"] for item in matrix["layers"])
    assert matrix["skip_policy"] == "any_unapproved_skip_in_a_hard_layer_fails_g9"
    schema_paths = sorted((ROOT / "schemas").glob("stage0-g9-*.json"))
    schema_paths.append(ROOT / "schemas/stage0-deterministic-training-fixture-v1.json")
    for path in schema_paths:
        _validate_project_json_schema(json.loads(path.read_text(encoding="utf-8")))
    assert len(list((ROOT / "schemas").glob("stage0-g9-*.json"))) == 5
    validate_deterministic_fixture(ROOT / "fixtures/stage0/deterministic-training-v1.json")


def test_g9_config_binds_g8_inputs_and_g7_real_replay_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    template = _template()
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"server", "cuda", "nccl", "model_assets", "data_assets"}),
        frozen_contract_stages=frozenset({0}),
        passed_gate_ids=frozenset({"stage0.G8"}),
        evidence_refs={},
    )
    g8 = Stage0G8FormalState(
        environment=environment,
        task_output_refs={"capacity_envelope": "evidence/g8/a.json", "operations_preflight": "evidence/g8/b.json", "fault_report": "evidence/g8/c.json"},
        config=template,
        config_ref="evidence/g8/config.json",
        environment_ref="evidence/g8/environment.json",
        index_ref="evidence/g8/index.json",
        index_sha256="1" * 64,
        gate_artifact_hash="2" * 64,
        g7_recovery_index_ref="evidence/g7/index.json",
    )
    g7 = Stage0G7RecoveryFormalState(
        environment=environment,
        task_output_refs={},
        config=template,
        config_ref="evidence/g7/config.json",
        environment_ref="evidence/g7/environment.json",
        index_ref="evidence/g7/index.json",
        index_sha256="3" * 64,
        gate_artifact_hash="4" * 64,
        g7_logging_index_ref="evidence/g7/logging-index.json",
    )
    monkeypatch.setattr(
        "param_importance_nlp.stage0_g9._selected_gpu_uuids",
        lambda _environment, _root: tuple(f"GPU-fixture-{index}" for index in range(4)),
    )
    config = build_stage0_g9_config(
        binding=Stage0SourceBinding(ROOT, "a" * 40, "fixture", True),
        data_root=tmp_path,
        state=g8,
        g7_state=g7,
        control_ref="evidence/g9/control.json",
    )
    assert config.task_id == "stage0.11_test_quality_and_replay"
    assert config.section("orchestration")["matrix_ref"] == "evidence/g9/control.json"
    assert config.section("launcher")["world_size"] == 4
    assert config.section("launcher")["backend"] == "nccl"
    assert config.section("recovery")["mode"] == "restart_idempotent"
    assert set(config.section("orchestration")["input_result_refs"]) == set(g8.task_output_refs.values())


@pytest.mark.replay
def test_independent_replay_plan_binds_formal_g7_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    template = _template()
    selected = tuple(f"GPU-fixture-{index}" for index in range(4))
    base_overrides, v2_overrides = _g7_config_overrides(
        template.base_config,
        world_size=4,
        gpu_uuids=selected,
    )
    base_overrides["identity"] = {"route": "stage0-g9-plan-fixture"}
    v2_overrides.update(
        {
            "scheduler": {"kind": "none", "warmup_steps": 0, "total_steps": None},
            "providers": template.section("providers"),
            "evaluation": {
                "enabled": False,
                "split": None,
                "every_steps": None,
                "batch_size": None,
                "max_batches": None,
                "metrics": [],
                "save_predictions": False,
            },
            "profiling": template.section("profiling"),
            "optimizer_runtime": template.section("optimizer_runtime"),
        }
    )
    config = build_stage0_formal_config(
        ROOT,
        task_id="stage0.09_checkpoint_and_resume",
        input_refs=("evidence/upstream/commit.json",),
        output_dir="evidence/g7/task-output",
        base_overrides=base_overrides,
        v2_overrides=v2_overrides,
    )
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"server", "cuda", "nccl", "model_assets", "data_assets"}),
        frozen_contract_stages=frozenset({0}),
        passed_gate_ids=frozenset({"stage0.G5", "stage0.G6", "stage0.G7"}),
        evidence_refs={},
    )
    refs = {
        "g7_config_ref": "controls/g7-config.json",
        "g7_environment_ref": "controls/g7-environment.json",
        "test_matrix_ref": "controls/test-matrix.json",
        "fixture_ref": "controls/fixture.json",
    }
    write_canonical_json(tmp_path / refs["g7_config_ref"], config.to_dict())
    write_canonical_json(tmp_path / refs["g7_environment_ref"], environment.to_dict())
    write_canonical_json(
        tmp_path / refs["test_matrix_ref"],
        load_canonical_json(ROOT / "configs/stage0/g9-test-matrix-v1.json"),
    )
    write_canonical_json(
        tmp_path / refs["fixture_ref"],
        load_canonical_json(ROOT / "fixtures/stage0/deterministic-training-v1.json"),
    )
    matrix = load_canonical_json(tmp_path / refs["test_matrix_ref"])
    fixture = load_canonical_json(tmp_path / refs["fixture_ref"])
    plan = {
        "schema_version": REPLAY_PLAN_SCHEMA,
        "replay_id": "stage0-g9-fixture",
        "run_id_prefix": "g9-fixture",
        "generator_git_commit": "a" * 40,
        **refs,
        "g7_config_sha256": sha256_file(tmp_path / refs["g7_config_ref"]),
        "g7_config_hash": config.config_hash,
        "g7_environment_sha256": sha256_file(tmp_path / refs["g7_environment_ref"]),
        "g7_environment_hash": environment.environment_hash,
        "test_matrix_sha256": sha256_file(tmp_path / refs["test_matrix_ref"]),
        "test_matrix_hash": matrix["artifact_hash"],
        "fixture_sha256": sha256_file(tmp_path / refs["fixture_ref"]),
        "fixture_hash": fixture["artifact_hash"],
        "suite_root_ref": "replays/fixture",
        "report_ref": "replays/fixture/report.json",
        "selected_gpu_uuids": list(selected),
        "pytest_timeout_seconds": 60,
        "replay_timeout_seconds": 60,
    }
    plan["artifact_hash"] = canonical_json_hash(plan)
    write_canonical_json(tmp_path / "plan.json", plan)
    monkeypatch.setattr(
        "param_importance_nlp.stage0_g9_replay._source_binding",
        lambda _expected: G7RecoverySourceBinding(ROOT, "a" * 40, "fixture"),
    )
    loaded, _path, loaded_config, loaded_environment, source = _validate_plan(
        tmp_path,
        "plan.json",
    )
    assert loaded["artifact_hash"] == plan["artifact_hash"]
    assert loaded_config.config_hash == config.config_hash
    assert loaded_environment.environment_hash == environment.environment_hash
    assert source.git_commit == "a" * 40


def test_layer_projection_rejects_no_required_layer_and_reports_all_pass() -> None:
    matrix = _load_hashed(
        ROOT / "configs/stage0/g9-test-matrix-v1.json",
        schema="stage0-g9-test-matrix-v1",
        field="matrix",
    )
    pair = {
        "world_size": 1,
        "sample_sequence_exact": True,
        "learning_rate_sequence_exact": True,
        "shared_state_hashes_exact": True,
        "rank_state_hashes_exact": True,
        "shared_state_numeric_within_tolerance": True,
        "rank_state_numeric_within_tolerance": True,
    }
    ddp = {**pair, "world_size": 4}
    result = _layer_results(
        matrix,
        {
            "status": "PASS",
            "counts": {
                "collected": 10,
                "passed": 10,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "allowed_platform_skipped": 0,
                "hard_skipped": 0,
            },
        },
        {
            "pair_metrics": {"single": pair, "ddp": ddp},
            "fault_report": {"status": "PASS", "rejection_reasons": [1, 2, 3, 4, 5]},
            "cpu_report": {"status": "PASS", "state_fields_exact": {"model": True}},
        },
        {"external_attempt_count": 0},
    )
    assert len(result) == 6
    assert all(item["status"] == "PASS" and item["skipped"] == 0 for item in result)


def test_g7_recovery_single_worker_local_launcher_is_contract_valid(
    tmp_path: Path,
) -> None:
    template = _template()
    base_overrides, v2_overrides = _g7_config_overrides(
        template.base_config,
        world_size=1,
        gpu_uuids=("GPU-fixture-00000000",),
    )
    base_overrides["identity"] = {"route": "stage0-g7-recovery-single-contract"}
    config = build_stage0_formal_config(
        ROOT,
        task_id="stage0.09_checkpoint_and_resume",
        input_refs=("evidence/upstream/commit.json",),
        output_dir="evidence/g7/task-output",
        base_overrides=base_overrides,
        v2_overrides=v2_overrides,
    )
    launcher = config.section("launcher")
    assert launcher["kind"] == "local"
    assert launcher["backend"] == "local"
    assert launcher["world_size"] == 1
    assert launcher["init_method"] == "local"
    assert launcher["rendezvous_id"] is None


def test_g7_recovery_uses_formal_offline_hf_provider() -> None:
    assert _FORMAL_RECOVERY_PROVIDERS["kind"] == "offline_hf"
    assert _FORMAL_RECOVERY_PROVIDERS["task_type"] == "causal_lm"
    assert _FORMAL_RECOVERY_PROVIDERS["task_name"] == "pile"
    assert _FORMAL_RECOVERY_PROVIDERS["local_files_only"] is True


def test_replay_runbook_has_only_resolvable_repository_links() -> None:
    result = _validate_runbook(G9SourceBinding(ROOT, "a" * 40, "fixture", {}))
    assert result["status"] == "PASS"
    assert result["required_content_count"] == 10
    assert len(result["checked_links"]) == 3
