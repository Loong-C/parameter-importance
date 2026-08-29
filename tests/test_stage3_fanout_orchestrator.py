from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.stage3 import run_stage3_fanout as fanout
from ops.stage3 import run_stage3_formal as formal
from ops.stage3 import materialize_stage3_fanout as materializer
from ops.stage3 import materialize_stage3_task as task_materializer
from ops.stage3.materialize_stage3_fanout import _schedule
from param_importance_nlp.contracts import (
    RecoveryMode,
    ResolvedConfigV2,
    RunnerKind,
    load_canonical_json,
)
from param_importance_nlp.runtime import (
    BlockerCode,
    TaskBlocker,
    TaskRunResult,
    TaskRunStatus,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"


def _write(path: Path, value: dict[str, object]) -> None:
    formal._write_atomic(path, value)


def _environment(path: Path) -> None:
    value: dict[str, object] = {
        "schema_version": "task-runtime-environment-v1",
        "capabilities": ["cuda"],
        "frozen_contract_stages": [3],
        "passed_gate_ids": ["stage3.G3-4"],
        "estimator_decision_ref": "evidence/stage2-decision.json",
        "evidence_refs": {},
    }
    value["environment_hash"] = formal._canonical_hash(value)
    _write(path, value)


def _config(path: Path, *, task_id: str, unit_id: str, resume: bool) -> str:
    value = copy.deepcopy(load_canonical_json(BASE_CONFIG))
    value["identity"].update(
        {
            "stage": 3,
            "task": task_id,
            "route": "path_integral",
            "run_intent": "formal",
            "formal_eligible": True,
        }
    )
    value["runtime"].update({"device": "cuda", "allow_dirty_worktree": False})
    value["sampling"].update(
        {
            "universe_version": "stage2-pythia-grid-20260826T145530Z",
            "reference_batch_size": 32,
            "repetition_count": 1,
        }
    )
    value["path_integration"].update(
        {
            "enabled": True,
            "probe_count": 2,
            "node_budget": 16,
            "thresholds_ref": "plans/stage3-thresholds.json",
            "default_rule": "simpson",
            "fallback_rule": "gauss_legendre_8",
        }
    )
    selector_ref = f"selectors/{unit_id}.json"
    selector: dict[str, object] = {
        "schema_version": "stage3-path-unit-selector-v1",
        "scope": "pilot",
        "unit_index_hash": "d" * 64,
        "active_unit_id": unit_id,
    }
    selector["artifact_hash"] = formal._canonical_hash(selector)
    _write(path.parent / selector_ref, selector)
    config = ResolvedConfigV2.resolve(
        value,
        task_id=task_id,
        overrides={
            "providers": {
                "kind": "offline_hf",
                "model_manifest_ref": None,
                "model_root_ref": None,
                "data_manifest_ref": None,
                "data_root_ref": None,
                "tokenizer_manifest_ref": None,
                "tokenizer_root_ref": None,
            },
            "orchestration": {
                "route_spec_ref": selector_ref,
                "input_result_refs": ["inputs/predecessor.json"],
            },
            "recovery": {
                "resume_ref": "cache/stage3-ledger" if resume else None
            },
            "artifacts": {"output_dir": "runs/stage3-fanout"},
        },
    )
    _write(path, config.to_dict())
    return config.config_hash


def _pilot_manifest(
    root: Path,
    *,
    task_id: str = "stage3.05_reference_integral_and_precision",
    phase: str = "reference",
    blocker: str = "stage3.05_reference_coverage",
) -> dict[str, object]:
    first_hash = _config(
        root / "config-1.json", task_id=task_id, unit_id="u-1", resume=False
    )
    first_retry_hash = _config(
        root / "config-1-resume.json", task_id=task_id, unit_id="u-1", resume=True
    )
    second_hash = _config(
        root / "config-2.json", task_id=task_id, unit_id="u-2", resume=True
    )
    value: dict[str, object] = {
        "schema_version": fanout.SCHEMA_VERSION,
        "task_id": task_id,
        "scope": "pilot",
        "run_config_hash": "c" * 64,
        "unit_index_ref": "unit-index.json",
        "unit_index_hash": "d" * 64,
        "state_dir": "state",
        "status_ref": "status.json",
        "steps": [
            {
                "step_id": f"{phase}-u-1",
                "unit_id": "u-1",
                "completes_phases": [phase],
                "action": "run",
                "config_ref": "config-1.json",
                "config_hash": first_hash,
                "retry_config_ref": "config-1-resume.json",
                "retry_config_hash": first_retry_hash,
                "result_ref": "results/1.json",
                "command": ["{python}", "worker.py", "{action}", "{config}", "{environment}", "{result}"],
                "expected_status": "BLOCKED",
                "expected_blocker_requirements": [blocker],
            },
            {
                "step_id": f"{phase}-u-2",
                "unit_id": "u-2",
                "completes_phases": [phase],
                "action": "resume",
                "config_ref": "config-2.json",
                "config_hash": second_hash,
                "retry_config_ref": "config-2.json",
                "retry_config_hash": second_hash,
                "result_ref": "results/2.json",
                "command": ["{python}", "worker.py", "{action}", "{config}", "{environment}", "{result}"],
                "expected_status": "PASS",
                "expected_blocker_requirements": [],
            },
        ],
    }
    value["manifest_hash"] = formal._canonical_hash(value)
    return value


def _units() -> tuple[formal.UnitRecord, ...]:
    return (
        formal.UnitRecord("u-1", "pythia-14m", 1, "early", "e" * 64, "probe-1"),
        formal.UnitRecord("u-2", "pythia-14m", 1, "early", "f" * 64, "probe-2"),
    )


def test_fanout_executes_expected_block_boundary_then_formal_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "unit-index.json", {})
    _environment(tmp_path / "environment.json")
    manifest = _pilot_manifest(tmp_path)
    monkeypatch.setattr(fanout, "load_unit_index", lambda *_args, **_kwargs: ("d" * 64, _units()))
    runner = fanout.FanoutRunner(
        manifest,
        workspace_root=tmp_path,
        data_root=tmp_path,
        environment=tmp_path / "environment.json",
    )
    calls: list[list[str]] = []

    def execute(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        result_path = Path(command[-1])
        _write(result_path, {})
        assert kwargs["shell"] is False
        return SimpleNamespace(returncode=3 if len(calls) == 1 else 0)

    def verify(step: dict[str, object], _path: Path) -> dict[str, object]:
        return {
            "status": step["expected_status"],
            "result_hash": ("1" if step["expected_status"] == "BLOCKED" else "2") * 64,
        }

    monkeypatch.setattr(runner, "_verify_result", verify)
    result = runner.run(executor=execute)
    assert result["status"] == "COMPLETE"
    assert len(calls) == 2
    status = formal._load_json(tmp_path / "status.json")
    assert [item["unit_id"] for item in status["units"]] == ["u-1", "u-2"]
    assert all(item["status"] == "PASS" for item in status["units"])


def test_fanout_accepts_serialized_retryable_asset_unavailable_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "unit-index.json", {})
    _environment(tmp_path / "environment.json")
    manifest = _pilot_manifest(tmp_path)
    monkeypatch.setattr(
        fanout, "load_unit_index", lambda *_args, **_kwargs: ("d" * 64, _units())
    )
    runner = fanout.FanoutRunner(
        manifest,
        workspace_root=tmp_path,
        data_root=tmp_path,
        environment=tmp_path / "environment.json",
    )
    step = runner.steps[0]
    result = TaskRunResult(
        task_id="stage3.05_reference_integral_and_precision",
        stage=3,
        runner_kind=RunnerKind.REFERENCE,
        run_intent="formal",
        status=TaskRunStatus.BLOCKED,
        config_hash=str(step["config_hash"]),
        formal_eligible=False,
        artifact_refs={},
        checkpoint_ref=None,
        blockers=(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                "stage3.05_reference_coverage",
                "formal reference coverage 1/12",
                True,
                ("evidence/unit-1.json",),
            ),
        ),
        error_code=None,
        message="formal reference coverage 1/12",
        recovery_mode=RecoveryMode.RESUME_SHARDS,
    )
    path = tmp_path / "blocked-result.json"
    _write(path, result.to_dict())

    parsed = runner._verify_result(step, path)

    assert parsed["blockers"][0]["code"] == "asset_unavailable"


def test_fanout_recovers_completed_process_from_immutable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "unit-index.json", {})
    _environment(tmp_path / "environment.json")
    manifest = _pilot_manifest(tmp_path)
    monkeypatch.setattr(fanout, "load_unit_index", lambda *_args, **_kwargs: ("d" * 64, _units()))
    _write(tmp_path / "results" / "1.json", {"durable": True})
    runner = fanout.FanoutRunner(
        manifest,
        workspace_root=tmp_path,
        data_root=tmp_path,
        environment=tmp_path / "environment.json",
    )
    calls = 0

    def execute(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        _write(Path(command[-1]), {})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        runner,
        "_verify_result",
        lambda step, _path: {
            "status": step["expected_status"],
            "result_hash": "3" * 64,
        },
    )
    runner.run(executor=execute)
    assert calls == 1
    attempts = formal._load_json(tmp_path / "state" / "fanout-state.json")["attempts"]
    assert attempts[0]["recovered_from_result"] is True


def test_fanout_retries_failed_first_step_with_resume_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "unit-index.json", {})
    _environment(tmp_path / "environment.json")
    manifest = _pilot_manifest(tmp_path)
    monkeypatch.setattr(
        fanout, "load_unit_index", lambda *_args, **_kwargs: ("d" * 64, _units())
    )
    first = fanout.FanoutRunner(
        manifest,
        workspace_root=tmp_path,
        data_root=tmp_path,
        environment=tmp_path / "environment.json",
    )
    with pytest.raises(formal.Stage3OrchestratorError, match="FANOUT_PROCESS_FAILED"):
        first.run(executor=lambda *_args, **_kwargs: SimpleNamespace(returncode=9))

    resumed = fanout.FanoutRunner(
        manifest,
        workspace_root=tmp_path,
        data_root=tmp_path,
        environment=tmp_path / "environment.json",
    )
    calls: list[list[str]] = []

    def execute(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        _write(Path(command[-1]), {})
        return SimpleNamespace(returncode=3 if len(calls) == 1 else 0)

    monkeypatch.setattr(
        resumed,
        "_verify_result",
        lambda step, _path: {
            "status": step["expected_status"],
            "result_hash": "4" * 64,
        },
    )
    resumed.run(executor=execute)
    assert calls[0][2] == "resume"
    assert calls[0][3].endswith("config-1-resume.json")


def test_stage306_requires_exact_observation_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "unit-index.json", {})
    _environment(tmp_path / "environment.json")
    manifest = _pilot_manifest(
        tmp_path,
        task_id="stage3.06_pilot_and_threshold_freeze",
        phase="observation",
        blocker="stage3.06_pilot_coverage",
    )
    monkeypatch.setattr(fanout, "load_unit_index", lambda *_args, **_kwargs: ("d" * 64, _units()))
    runner = fanout.FanoutRunner(
        manifest,
        workspace_root=tmp_path,
        data_root=tmp_path,
        environment=tmp_path / "environment.json",
    )
    assert all(
        step["completes_phases"] == ["observation"] for step in runner.steps
    )


def test_materialized_s307_schedule_is_99_per_unit_streaming_steps() -> None:
    units = tuple(f"formal-unit-{index:03d}" for index in range(99))
    steps = _schedule("stage3.07_formal_experiment_matrix", units)
    assert len(steps) == 99
    assert [step["unit_id"] for step in steps] == list(units)
    assert all(
        step["completes_phases"] == ["reference", "observation"] for step in steps
    )
    assert all(
        step["expected_status"] == "BLOCKED"
        and step["expected_blocker_requirements"] == ["stage3.07_matrix_coverage"]
        for step in steps[:-1]
    )
    assert steps[-1]["expected_status"] == "PASS"
    assert steps[-1]["expected_blocker_requirements"] == []


@pytest.mark.parametrize(
    ("legacy_shape", "expected_error"),
    (
        ("reference-only", "FANOUT_S307_STREAMING_PHASES_INVALID"),
        ("197-step", "FANOUT_S307_STEP_COUNT_INVALID"),
    ),
)
def test_fanout_rejects_legacy_s307_reference_wave_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_shape: str,
    expected_error: str,
) -> None:
    _write(tmp_path / "unit-index.json", {})
    _environment(tmp_path / "environment.json")
    manifest = _pilot_manifest(
        tmp_path,
        task_id="stage3.07_formal_experiment_matrix",
        phase="reference",
        blocker="stage3.07_reference_coverage",
    )
    manifest["scope"] = "formal"
    if legacy_shape == "197-step":
        manifest["steps"] = list(manifest["steps"]) * 98 + [manifest["steps"][0]]
    manifest["manifest_hash"] = formal._canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    monkeypatch.setattr(
        fanout, "load_unit_index", lambda *_args, **_kwargs: ("d" * 64, _units())
    )
    with pytest.raises(formal.Stage3OrchestratorError, match=expected_error):
        fanout.FanoutRunner(
            manifest,
            workspace_root=tmp_path,
            data_root=tmp_path,
            environment=tmp_path / "environment.json",
        )


def test_materializer_emits_real_v2_selectors_and_exact_pilot_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = copy.deepcopy(load_canonical_json(BASE_CONFIG))

    def sanitize(value: object) -> object:
        if isinstance(value, str):
            return value.replace("local-fixture", "stage2-real").replace(
                "synthetic", "stage2-real"
            ).replace("fixture", "stage2-real")
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        return value

    base = sanitize(base)
    base["loss"]["task_type"] = "causal_lm"
    base["sampling"].update(
        {
            "universe_version": "stage2-pythia-grid-20260826T145530Z",
            "reference_batch_size": 32,
            "repetition_count": 1,
        }
    )
    base["path_integration"].update(
        {
            "node_budget": 16,
            "thresholds_ref": "plans/stage3-thresholds.json",
        }
    )
    _write(tmp_path / "base.json", base)
    _write(
        tmp_path / "stage2.json",
        {
            "run_id": formal.EXPECTED_STAGE2_RUN_ID,
            "default_estimator": "U-32",
            "batch_size": 32,
            "data_variant": "Raw",
        },
    )
    _write(tmp_path / "unit-index.json", {})
    _write(
        tmp_path / "overrides.json",
        {
            "providers": {
                "kind": "offline_hf",
                "model_manifest_ref": None,
                "model_root_ref": None,
                "data_manifest_ref": None,
                "data_root_ref": None,
                "tokenizer_manifest_ref": None,
                "tokenizer_root_ref": None,
            }
        },
    )
    units = tuple(
        formal.UnitRecord(
            f"pilot-unit-{index:02d}",
            "14M",
            7,
            ("early", "middle", "late")[index // 4],
            f"{index + 1:064x}",
            f"probe-{index:02d}",
        )
        for index in range(12)
    )
    monkeypatch.setattr(
        materializer,
        "load_unit_index",
        lambda *_args, **_kwargs: ("d" * 64, units),
    )
    source = {
        "schema_version": materializer.SOURCE_SCHEMA,
        "task_id": "stage3.05_reference_integral_and_precision",
        "scope": "pilot",
        "run_config_hash": "c" * 64,
        "base_config_ref": "base.json",
        "stage2_authority_ref": "stage2.json",
        "unit_index_ref": "unit-index.json",
        "unit_index_hash": "d" * 64,
        "config_overrides_ref": "overrides.json",
        "input_result_refs_by_endpoint": {
            unit.endpoint_hash: [
                f"inputs/endpoints/{unit.endpoint_hash}.json",
                f"inputs/probes/{unit.endpoint_hash}.json",
            ]
            for unit in units
        },
        "artifact_output_dir": "runs/stage3/s305",
        "cache_root": "runs/stage3/cache",
        "config_dir": "runs/stage3/configs/s305",
        "selector_dir": "runs/stage3/selectors/pilot",
        "result_dir": "runs/stage3/results/s305",
        "state_dir": "runs/stage3/state/s305",
        "status_ref": "runs/stage3/status/s305.json",
        "manifest_ref": "runs/stage3/manifests/s305.json",
    }
    receipt = materializer.materialize(
        source, workspace_root=tmp_path, data_root=tmp_path
    )
    assert receipt["unit_count"] == receipt["step_count"] == 12
    manifest = formal._load_json(tmp_path / source["manifest_ref"])
    assert manifest["steps"][0]["action"] == "run"
    assert manifest["steps"][-1]["action"] == "resume"
    assert manifest["steps"][-1]["expected_status"] == "PASS"
    first_retry = ResolvedConfigV2.from_mapping(
        formal._load_json(tmp_path / manifest["steps"][0]["retry_config_ref"])
    )
    assert first_retry.section("recovery")["resume_ref"] == source[
        "artifact_output_dir"
    ]
    final_config = ResolvedConfigV2.from_mapping(
        formal._load_json(tmp_path / receipt["final_config_ref"])
    )
    assert final_config.run_intent == "formal"
    selector_ref = final_config.section("orchestration")["route_spec_ref"]
    selector = formal._load_json(tmp_path / selector_ref)
    assert selector["active_unit_id"] == units[-1].unit_id


def test_fanout_rejects_duplicate_phase_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "unit-index.json", {})
    _environment(tmp_path / "environment.json")
    manifest = _pilot_manifest(tmp_path)
    duplicate = dict(manifest["steps"][0])
    duplicate["step_id"] = "duplicate-u-1"
    duplicate["action"] = "resume"
    duplicate["config_ref"] = "config-2.json"
    duplicate["config_hash"] = manifest["steps"][1]["config_hash"]
    manifest["steps"].insert(1, duplicate)
    manifest["manifest_hash"] = formal._canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    monkeypatch.setattr(fanout, "load_unit_index", lambda *_args, **_kwargs: ("d" * 64, _units()))
    with pytest.raises(formal.Stage3OrchestratorError, match="FANOUT_PHASE_COVERAGE_DUPLICATE"):
        fanout.FanoutRunner(
            manifest,
            workspace_root=tmp_path,
            data_root=tmp_path,
            environment=tmp_path / "environment.json",
        )


def test_direct_task_materializer_emits_strict_formal_v2_config(tmp_path: Path) -> None:
    base = copy.deepcopy(load_canonical_json(BASE_CONFIG))

    def sanitize(value: object) -> object:
        if isinstance(value, str):
            return (
                value.replace("local-fixture", "stage2-real")
                .replace("synthetic", "stage2-real")
                .replace("fixture", "stage2-real")
            )
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        return value

    base = sanitize(base)
    base["loss"]["task_type"] = "causal_lm"
    base["sampling"].update(
        {
            "universe_version": "stage2-pythia-grid-20260826T145530Z",
            "reference_batch_size": 32,
            "repetition_count": 1,
        }
    )
    base["path_integration"].update(
        {"node_budget": 16, "thresholds_ref": "plans/stage3-thresholds.json"}
    )
    _write(tmp_path / "base.json", base)
    _write(
        tmp_path / "stage2.json",
        {
            "run_id": formal.EXPECTED_STAGE2_RUN_ID,
            "default_estimator": "U-32",
            "batch_size": 32,
            "data_variant": "Raw",
        },
    )
    _write(
        tmp_path / "overrides.json",
        {
            "providers": {
                "kind": "offline_hf",
                "model_manifest_ref": None,
                "model_root_ref": None,
                "data_manifest_ref": None,
                "data_root_ref": None,
                "tokenizer_manifest_ref": None,
                "tokenizer_root_ref": None,
            }
        },
    )
    source = {
        "schema_version": task_materializer.SOURCE_SCHEMA,
        "task_id": "stage3.01_prerequisites_and_scope",
        "base_config_ref": "base.json",
        "stage2_authority_ref": "stage2.json",
        "config_overrides_ref": "overrides.json",
        "input_result_refs": [],
        "artifact_output_dir": "runs/stage3/s301",
        "authority_output_dir": "runs/stage3/authority/s301",
        "cache_root": "runs/stage3/cache",
        "config_ref": "runs/stage3/configs/s301.json",
        "result_ref": "runs/stage3/results/s301.json",
        "evidence_refs": {},
        "external_gate_ref": None,
        "route_spec_ref": None,
    }
    receipt = task_materializer.materialize(
        source, workspace_root=tmp_path, data_root=tmp_path
    )
    resolved = ResolvedConfigV2.from_mapping(
        formal._load_json(tmp_path / receipt["config_ref"])
    )
    assert resolved.task_id == "stage3.01_prerequisites_and_scope"
    assert resolved.run_intent == "formal"
    payload = resolved.to_dict()["base_config"]
    assert payload["sampling"]["reference_batch_size"] == 32
    assert payload["importance"]["estimator_name"] == "u"
    assert receipt["output_refs"] == {
        kind: f"runs/stage3/s301/commits/{kind}.json"
        for kind in formal.EXPECTED_OUTPUTS["stage3.01_prerequisites_and_scope"]
    }
    selector = {
        "schema_version": "stage3-probe-selector-v1",
        "scope": "formal",
        "endpoint_digest": "a" * 64,
        "probe_plan_hash": "b" * 64,
        "active_probe_id": "formal-probe-01",
    }
    selector["artifact_hash"] = formal._canonical_hash(selector)
    _write(tmp_path / "selectors" / "formal-probe-01.json", selector)
    s303_source = dict(source)
    s303_source.update(
        {
            "task_id": "stage3.03_endpoint_and_probe_pipeline",
            "input_result_refs": ["inputs/endpoint.json", "inputs/probe-plan.json"],
            "artifact_output_dir": "runs/stage3/s303",
            "authority_output_dir": "runs/stage3/authority/s303",
            "config_ref": "runs/stage3/configs/s303.json",
            "result_ref": "runs/stage3/results/s303.json",
            "route_spec_ref": "selectors/formal-probe-01.json",
        }
    )
    s303 = task_materializer.materialize(
        s303_source, workspace_root=tmp_path, data_root=tmp_path
    )
    s303_config = ResolvedConfigV2.from_mapping(
        formal._load_json(tmp_path / s303["config_ref"])
    )
    assert s303_config.section("orchestration")["route_spec_ref"] == (
        "selectors/formal-probe-01.json"
    )
