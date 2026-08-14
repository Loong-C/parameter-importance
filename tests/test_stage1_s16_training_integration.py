from __future__ import annotations

from copy import deepcopy
import ast
import builtins
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from param_importance_nlp.core import ImportanceAccumulator, TensorMap
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.runtime.training import TrainingEngine, TrainingRunSpec
from param_importance_nlp.stage1_training_integration import (
    Stage1TrainingIntegrationError,
    build_stage1_s16_evidence,
    build_s16_tiny_transformer_parity_trace,
    replay_stage1_s16_evidence,
)
from param_importance_nlp.stage1_training_integration import _s16_transformer_fixture
from param_importance_nlp.stage1_training_oracle import build_stage1_s16_oracle


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_s16_fixture_oracle_production_trace_and_replay_are_independent() -> None:
    root = _root()
    oracle_source = (root / "src/param_importance_nlp/stage1_training_oracle.py").read_text(encoding="utf-8")
    assert "runtime.training" not in oracle_source
    assert "runtime.optimizer" not in oracle_source
    assert "core.accumulator" not in oracle_source
    imports = []
    for node in ast.walk(ast.parse(oracle_source)):
        if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): imports.append("." * node.level + (node.module or ""))
    assert not any(name.startswith(("param_importance_nlp", ".contracts", ".core", ".runtime")) for name in imports)
    oracle = build_stage1_s16_oracle(root)
    assert oracle["independent_implementation"] is True
    evidence = build_stage1_s16_evidence(
        root, producer_commit="d21b084faa7b9c13cdd22aa09253ea0cca75c3de"
    )
    assert len(evidence["comparison_table"]["rows"]) == 118
    assert all(row["passed"] for row in evidence["comparison_table"]["rows"])
    for step in evidence["trace_bundle"]["production_sgd_trace"]:
        views = step["accumulator_after"]
        for name in ("left", "right"):
            assert views["signed"][name][0] == views["positive"][name][0] - views["negative_mass"][name][0]
            assert views["absolute"][name][0] == views["positive"][name][0] + views["negative_mass"][name][0]
            assert views["positive"][name][0] >= 0
            assert views["negative_mass"][name][0] >= 0
    assert evidence["trace_bundle"]["production_sgd_summary"]["v3_roundtrip"] == evidence["trace_bundle"]["production_sgd_trace"][-1]["accumulator_after"]
    assert replay_stage1_s16_evidence(evidence, source_root=root)["status"] == "PASS"
    tampered = deepcopy(evidence)
    tampered["trace_bundle"]["production_sgd_trace"][0]["raw_score"]["left"][0] += 0.1
    with pytest.raises(Stage1TrainingIntegrationError, match="ROLE_HASH|OFFLINE_REPLAY"):
        replay_stage1_s16_evidence(tampered, source_root=root)


def test_s16_oracle_import_isolation_survives_production_import_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """The independent oracle can be loaded while all production imports fail."""

    root = _root(); path = root / "src/param_importance_nlp/stage1_training_oracle.py"
    original = builtins.__import__

    def guarded(name: str, globals=None, locals=None, fromlist=(), level: int = 0):  # type: ignore[no-untyped-def]
        if name.startswith("param_importance_nlp") or level > 0:
            raise AssertionError(f"production import attempted: {name!r}, level={level}")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("s16_oracle_isolation_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.FIXTURE_ID == "stage1-s16-training-fixture-v1"


def test_s16_tiny_transformer_statistics_on_off_are_bitwise_identical() -> None:
    trace = build_s16_tiny_transformer_parity_trace()
    assert trace["profile"] == "T32_SINGLE"
    assert trace["cpu_bitwise_tolerance"] == 0.0
    assert trace["observed_parameters"] == trace["control_parameters"]
    assert trace["observed_optimizer_state"] == trace["control_optimizer_state"]
    assert trace["observed_scheduler_state"] == trace["control_scheduler_state"]
    assert trace["observed_scaler_state"] == trace["control_scaler_state"]
    assert trace["observed_scale_before"] == trace["control_scale_before"] == [8.0, 16.0]
    assert trace["observed_scale_after"] == trace["control_scale_after"] == [16.0, 32.0]
    assert trace["observed_events"] == trace["control_events"]
    assert len(trace["observed_events"]) == 6
    for observed, control in zip(trace["observed_records"], trace["control_records"], strict=True):
        assert observed["mean_loss"] == control["mean_loss"]
        assert observed["parameter_post_state_hash"] == control["parameter_post_state_hash"]
        assert observed["attempt_commit_state_hash"] == control["attempt_commit_state_hash"]


def test_s16_real_cpu_gradscaler_finite_skip_finite_contract() -> None:
    from param_importance_nlp.stage1_training_integration import build_s16_nonfinite_skip_trace

    trace = build_s16_nonfinite_skip_trace()
    assert trace["scale_before"] == [8.0, 16.0, 8.0]
    assert trace["scale_after"] == [16.0, 8.0, 16.0]
    assert trace["skip_observation"]["scaler_present"] is True
    assert [record["status"] for record in trace["records"]] == ["COMMITTED", "SKIPPED", "COMMITTED"]
    assert all(trace["reference_comparison"].values())
    gradients = [event for event in trace["events"] if event["boundary"] == "gradient_ready"]
    skipped = next(event for event in trace["events"] if event["boundary"] == "skip")
    assert gradients[-1]["microbatch_ids"] == trace["reference"]["next_batch_ids"]
    assert gradients[-1]["sample_ids"] == trace["reference"]["next_sample_ids"]
    assert len(skipped["sample_ids"]) == 4


def test_s16_boundary_capture_is_opt_in_and_default_path_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    from param_importance_nlp.runtime import training as runtime_training

    calls = 0
    original = runtime_training._state_tree_hash

    def counted(value: object) -> str:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(runtime_training, "_state_tree_hash", counted)
    model, dataset = _s16_transformer_fixture(seed=6106, steps=1)
    optimizer = torch.optim.SGD(model.module.parameters(), lr=0.01, foreach=False)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "s16-boundary-default", "local_fixture", max_steps=1, max_attempts=1,
            importance_enabled=False, estimator_name="u", weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=model, optimizer=optimizer, cursor=dataset.cursor(seed=1),
    )
    assert engine.run().state.global_step == 1
    assert engine.boundary_trace == ()
    default_calls = calls

    traced_model, traced_dataset = _s16_transformer_fixture(seed=6106, steps=1)
    traced = TrainingEngine(
        spec=TrainingRunSpec(
            "s16-boundary-enabled", "local_fixture", max_steps=1, max_attempts=1,
            importance_enabled=False, estimator_name="u", weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=traced_model,
        optimizer=torch.optim.SGD(traced_model.module.parameters(), lr=0.01, foreach=False),
        cursor=traced_dataset.cursor(seed=1), capture_boundary_trace=True,
    )
    traced.run()
    assert len(traced.boundary_trace) == 17
    # Core commit hashes are always necessary; the strictly larger count is
    # evidence that only the opted-in lifecycle observations evaluated lazily.
    assert calls > default_calls


def test_s16_accumulator_v1_v2_migration_marks_actual_update_unavailable() -> None:
    template = TensorMap({"p": torch.zeros(1, dtype=torch.float64)})
    current = ImportanceAccumulator(template, accumulation_dtype=torch.float64)
    current.add_step(TensorMap({"p": torch.ones(1, dtype=torch.float64)}))
    state_v3 = current.state_dict()
    v2 = dict(state_v3)
    v2["version"] = 2
    v2.pop("actual_update_raw_importance")
    v2.pop("actual_update_raw_importance_available")
    restored = ImportanceAccumulator(template, accumulation_dtype=torch.float64)
    restored.load_state_dict(v2)
    assert restored.state_dict()["version"] == 3
    assert restored.actual_update_raw_importance_available is False
    assert torch.equal(restored.actual_update_raw_importance["p"], torch.zeros(1, dtype=torch.float64))
    restored.add_step(
        TensorMap({"p": torch.ones(1, dtype=torch.float64)}),
        actual_update_raw_importance=TensorMap({"p": torch.ones(1, dtype=torch.float64)}),
    )
    # A later observed step cannot reconstruct the legacy prefix.
    assert restored.actual_update_raw_importance_available is False
    bad = dict(restored.state_dict())
    bad["unknown"] = True
    with pytest.raises(Exception, match="字段集合"):
        restored.load_state_dict(bad)

    fresh = ImportanceAccumulator(template, accumulation_dtype=torch.float64)
    fresh.add_step(TensorMap({"p": torch.ones(1, dtype=torch.float64)}))
    assert fresh.actual_update_raw_importance_available is False


def test_s16_fixture_self_hash_rejects_tamper(tmp_path: Path) -> None:
    source = _root() / "fixtures/stage1/stage1-s16-training-fixture-v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["sgd"]["steps"][0]["learning_rates"]["left"] = 0.9
    path = tmp_path / "fixtures/stage1"
    path.mkdir(parents=True)
    (path / source.name).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="FIXTURE"):
        build_stage1_s16_oracle(tmp_path)


def test_s16_frozen_fixture_and_joint_role_rehash_tampering_fail_replay(tmp_path: Path) -> None:
    source = _root() / "fixtures/stage1/stage1-s16-training-fixture-v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["sgd"]["steps"][0]["learning_rates"]["group_0000"] = 0.125
    body = dict(payload); body.pop("fixture_hash")
    payload["fixture_hash"] = canonical_json_hash(body)
    path = tmp_path / "fixtures/stage1"; path.mkdir(parents=True)
    (path / source.name).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="FIXTURE"):
        build_stage1_s16_oracle(tmp_path)

    evidence = build_stage1_s16_evidence(_root(), producer_commit="d21b084faa7b9c13cdd22aa09253ea0cca75c3de")

    def rehash_roles(value: dict[str, object]) -> None:
        trace = value["trace_bundle"]
        assert isinstance(trace, dict)
        trace_body = dict(trace); trace_body.pop("trace_hash")
        trace["trace_hash"] = canonical_json_hash(trace_body)
        report = value["step_report"]
        assert isinstance(report, dict)
        report["trace_hash"] = trace["trace_hash"]
        report_body = dict(report); report_body.pop("report_hash")
        report["report_hash"] = canonical_json_hash(report_body)

    trace_tamper = deepcopy(evidence)
    trace_tamper["trace_bundle"]["production_sgd_trace"][0]["accumulator_after"]["raw"]["left"][0] += 0.25
    rehash_roles(trace_tamper)
    with pytest.raises(Stage1TrainingIntegrationError, match="OFFLINE_REPLAY"):
        replay_stage1_s16_evidence(trace_tamper, source_root=_root())

    parity_tamper = deepcopy(evidence)
    parity_tamper["trace_bundle"]["tiny_transformer_trace"]["observed_records"][0]["mean_loss"] += 0.125
    rehash_roles(parity_tamper)
    with pytest.raises(Stage1TrainingIntegrationError, match="OFFLINE_REPLAY"):
        replay_stage1_s16_evidence(parity_tamper, source_root=_root())

    lifecycle_tamper = deepcopy(evidence)
    lifecycle_tamper["trace_bundle"]["tiny_transformer_trace"]["observed_lifecycle"].pop()
    rehash_roles(lifecycle_tamper)
    with pytest.raises(Stage1TrainingIntegrationError, match="OFFLINE_REPLAY"):
        replay_stage1_s16_evidence(lifecycle_tamper, source_root=_root())

    adamw_tamper = deepcopy(evidence)
    adamw_tamper["trace_bundle"]["production_adamw_trace"][3]["accumulator_after"]["total_movement"]["weight"][0] += 0.5
    rehash_roles(adamw_tamper)
    with pytest.raises(Stage1TrainingIntegrationError, match="OFFLINE_REPLAY"):
        replay_stage1_s16_evidence(adamw_tamper, source_root=_root())

    clip_tamper = deepcopy(evidence)
    clip_tamper["trace_bundle"]["clip_training_engine_trace"]["gradient_event"]["optimizer_gradient"]["left"][0] += 0.25
    rehash_roles(clip_tamper)
    with pytest.raises(Stage1TrainingIntegrationError, match="OFFLINE_REPLAY"):
        replay_stage1_s16_evidence(clip_tamper, source_root=_root())
