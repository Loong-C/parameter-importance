from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch

from param_importance_nlp.contracts import GateRecord, GateStatus, canonical_json_hash
from param_importance_nlp.core.quadrature import PathSpec, simpson_rule, trapezoid_rule
from param_importance_nlp.core.tensors import TensorMap
from param_importance_nlp.experiments import (
    EndpointRecord,
    EndpointState,
    EstimatorDecision,
    FormalDecisionBlocked,
    NodeCacheKey,
    NodeGradientCache,
    PathAnalysisRunner,
    PathStateUnit,
    ProbeSpec,
    ReadOnlyPathContext,
    ReferenceLevel,
    StateMutationError,
    TrainingPhaseSpec,
    TrainingRouteSpec,
    assess_reference_convergence,
    build_fixture_quadrature_decision,
    validate_comparable_routes,
)


def _hash(character: str) -> str:
    return character * 64


def _state(
    artifact_id: str,
    artifact: str,
    parameter: str,
    *,
    optimizer: str,
    rng: str,
    cursor: str,
) -> EndpointState:
    return EndpointState(
        artifact_id,
        _hash(artifact),
        _hash(parameter),
        _hash("d"),
        _hash(optimizer),
        _hash("4"),
        _hash("5"),
        _hash(rng),
        _hash(cursor),
        _hash("a"),
    )


def _endpoint() -> EndpointRecord:
    return EndpointRecord(
        path_state_id="path-state-1",
        source_run_id="train-run-1",
        optimizer_step=7,
        parameter_registry_hash=_hash("a"),
        pre_state=_state("pre", "b", "c", optimizer="0", rng="6", cursor="8"),
        parameter_post_state=_state(
            "parameter-post", "e", "f", optimizer="3", rng="6", cursor="8"
        ),
        attempt_commit_state=_state(
            "attempt-commit", "1", "f", optimizer="3", rng="7", cursor="9"
        ),
        attempt_commit_parent_hash=_hash("e"),
        probe_buffer_snapshot_hash=_hash("d"),
        full_update_delta_hash=_hash("2"),
        update_sample_ids=(10, 11),
        replay_verified=True,
    )


def test_endpoint_probe_and_path_unit_keep_post_and_commit_distinct() -> None:
    endpoint = _endpoint()
    probe = ProbeSpec("probe-1", (20, 21), _hash("3"), _hash("4"))
    unit = PathStateUnit(endpoint, probe)
    assert endpoint.parameter_post_state.artifact_id != endpoint.attempt_commit_state.artifact_id
    assert unit.unit_id.startswith("path-unit-")

    overlapping = ProbeSpec("bad", (11, 12), _hash("5"), _hash("6"))
    with pytest.raises(ValueError, match="重叠"):
        PathStateUnit(endpoint, overlapping)

    # 外部 artifact_hash 不变时，任一恢复组件漂移也必须改变路径/cache 身份。
    changed_pre = replace(endpoint.pre_state, scheduler_hash=_hash("b"))
    changed_endpoint = replace(endpoint, pre_state=changed_pre)
    changed_unit = PathStateUnit(changed_endpoint, probe)
    assert changed_endpoint.digest != endpoint.digest
    assert changed_unit.unit_id != unit.unit_id


def test_endpoint_commit_must_bind_post_and_advance_control_state() -> None:
    valid = _endpoint()
    with pytest.raises(ValueError, match="未绑定权威"):
        replace(valid, attempt_commit_parent_hash=_hash("0"))
    unchanged_control = replace(
        valid.attempt_commit_state,
        rng_hash=valid.parameter_post_state.rng_hash,
        data_cursor_hash=valid.parameter_post_state.data_cursor_hash,
    )
    with pytest.raises(ValueError, match="未证明"):
        replace(valid, attempt_commit_state=unchanged_control)


class _Controller:
    def __init__(self) -> None:
        self.value = "clean"
        self.original = "clean"

    def digest(self) -> str:
        return self.value

    def restore(self) -> None:
        self.value = self.original


def test_read_only_path_context_restores_but_still_fails_mutated_unit() -> None:
    controller = _Controller()
    with pytest.raises(StateMutationError, match="曾污染"):
        with ReadOnlyPathContext(controller):
            controller.value = "mutated"
    assert controller.value == "clean"


def test_path_analysis_runner_delegates_math_to_core_and_checks_endpoint_losses() -> None:
    endpoint = _endpoint()
    unit = PathStateUnit(
        endpoint,
        ProbeSpec("probe", (20,), _hash("7"), _hash("8")),
    )
    pre = TensorMap({"x": torch.tensor([1.0], dtype=torch.float64)})
    post = TensorMap({"x": torch.tensor([2.0], dtype=torch.float64)})
    path = PathSpec(pre, post, probe_id="probe", loss_id="quadratic")

    def gradient(_alpha: float, state: TensorMap) -> TensorMap:
        return TensorMap({"x": 2.0 * state["x"]})

    def loss(state: TensorMap) -> torch.Tensor:
        return torch.square(state["x"]).sum()

    evaluation = PathAnalysisRunner().run(
        unit=unit,
        path_spec=path,
        rule=trapezoid_rule(),
        gradient_callback=gradient,
        loss_callback=loss,
        state_controller=_Controller(),
        expected_loss_pre=1.0,
        expected_loss_post=4.0,
    )
    assert evaluation.rule_name == "trapezoid"
    assert evaluation.result.signed["x"].item() == pytest.approx(-3.0)
    assert evaluation.result.completeness_absolute_residual == pytest.approx(0.0)


def test_path_runner_reuses_common_gradient_nodes_across_rules() -> None:
    """trapezoid 与 Simpson 共用的两个端点只能触发一次真实梯度求值。"""

    unit = PathStateUnit(
        _endpoint(),
        ProbeSpec("probe-cache", (20,), _hash("7"), _hash("8")),
    )
    path = PathSpec(
        TensorMap({"x": torch.tensor([1.0], dtype=torch.float64)}),
        TensorMap({"x": torch.tensor([2.0], dtype=torch.float64)}),
        probe_id="probe-cache",
        loss_id="quadratic",
    )
    evaluated_alphas: list[float] = []

    def gradient(alpha: float, state: TensorMap) -> TensorMap:
        evaluated_alphas.append(alpha)
        return TensorMap({"x": 2.0 * state["x"]})

    def loss(state: TensorMap) -> torch.Tensor:
        return torch.square(state["x"]).sum()

    runner = PathAnalysisRunner()
    first = runner.run(
        unit=unit,
        path_spec=path,
        rule=trapezoid_rule(),
        gradient_callback=gradient,
        loss_callback=loss,
        state_controller=_Controller(),
    )
    second = runner.run(
        unit=unit,
        path_spec=path,
        rule=simpson_rule(),
        gradient_callback=gradient,
        loss_callback=loss,
        state_controller=_Controller(),
    )

    assert sorted(evaluated_alphas) == [0.0, 0.5, 1.0]
    assert first.cache_hits == 0
    assert first.cache_misses == 2
    assert first.cache_entries_published == 2
    assert second.cache_hits == 2
    assert second.cache_misses == 1
    assert second.cache_entries_published == 1
    assert len(runner.node_cache) == 3


def test_node_cache_does_not_cross_path_or_loss_identity_and_returns_clones() -> None:
    """完整 key 的任一身份变化都隔离缓存，读取值也不能反向修改缓存。"""

    cache = NodeGradientCache()
    key = NodeCacheKey("path-unit-a", 0.0, "float64", _hash("a"), _hash("b"))
    other = NodeCacheKey("path-unit-a", 0.0, "float64", _hash("a"), _hash("c"))
    original = TensorMap({"x": torch.tensor([3.0], dtype=torch.float64)})
    assert cache.publish_many({key: original}) == 1

    # 写入后的源对象与读取副本均可被调用方修改，但缓存中的权威副本保持 3.0。
    original["x"].fill_(11.0)
    first_read = cache.get(key)
    assert isinstance(first_read, TensorMap)
    first_read["x"].fill_(17.0)
    second_read = cache.get(key)
    assert isinstance(second_read, TensorMap)
    assert second_read["x"].item() == pytest.approx(3.0)
    with pytest.raises(KeyError):
        cache.get(other)


def test_failed_state_guard_does_not_publish_pending_gradient_nodes() -> None:
    """副作用回调即使产出梯度也必须恢复状态，并丢弃本次暂存缓存。"""

    unit = PathStateUnit(
        _endpoint(),
        ProbeSpec("probe-mutating", (20,), _hash("7"), _hash("9")),
    )
    path = PathSpec(
        TensorMap({"x": torch.tensor([1.0], dtype=torch.float64)}),
        TensorMap({"x": torch.tensor([2.0], dtype=torch.float64)}),
    )
    controller = _Controller()
    cache = NodeGradientCache()
    runner = PathAnalysisRunner(node_cache=cache)

    def mutating_gradient(_alpha: float, state: TensorMap) -> TensorMap:
        controller.value = "mutated"
        return TensorMap({"x": state["x"].clone()})

    with pytest.raises(StateMutationError, match="曾污染"):
        runner.run(
            unit=unit,
            path_spec=path,
            rule=trapezoid_rule(),
            gradient_callback=mutating_gradient,
            loss_callback=lambda state: state["x"].square().sum(),
            state_controller=controller,
        )
    assert controller.value == "clean"
    assert len(cache) == 0


def test_reference_convergence_requires_two_families_and_consecutive_refinement() -> None:
    levels = [
        ReferenceLevel("gauss_legendre", 0, 8, {"x": np.array([1.0, 2.0])}),
        ReferenceLevel("gauss_legendre", 1, 16, {"x": np.array([1.0, 2.0])}),
        ReferenceLevel("composite_simpson", 0, 17, {"x": np.array([1.0, 2.0])}),
        ReferenceLevel("composite_simpson", 1, 33, {"x": np.array([1.0, 2.0])}),
    ]
    result = assess_reference_convergence(levels, tolerance=1e-12)
    assert result.converged
    with pytest.raises(FormalDecisionBlocked):
        assess_reference_convergence(levels, tolerance=1e-12, scope="formal")

    decision = build_fixture_quadrature_decision(
        passing_rules_by_cost=("trapezoid", "simpson")
    )
    assert decision.default_rule == "trapezoid"
    assert decision.formal_eligible is False


def _decision(
    scope: str = "local_fixture",
    status: str = "FIXTURE_ONLY",
    *,
    gate_status: str = "NOT_RUN",
) -> EstimatorDecision:
    payload: dict[str, object] = {
        "schema_version": "estimator-decision-v1",
        "decision_id": f"{scope}-decision",
        "scope": scope,
        "status": status,
        "state": "UNFROZEN" if scope == "local_fixture" else "FROZEN",
        "selected_estimator": "u",
        "batch_size": 32,
        "microbatch_count": 2,
        "repetitions": 2,
        "gate_id": "stage2.G2.7b",
        "gate_status": gate_status,
        "artifact_ref": None if scope == "local_fixture" else "decisions/stage2.json",
        "metadata": {"formal_eligible": scope == "formal"},
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return EstimatorDecision.from_mapping(payload)


def _route(route_id: str, decision: object) -> TrainingRouteSpec:
    pretrain = TrainingPhaseSpec(
        "pretrain",
        "pretrain",
        "init-1",
        "model-asset",
        "pretrain-data",
        "checkpoint-pretrain",
        10,
    )
    direct = TrainingPhaseSpec(
        "direct",
        "direct_supervised",
        "init-1",
        "model-asset",
        "task-data",
        "checkpoint-direct",
        5,
        task_id="task-asset",
    )
    finetune = TrainingPhaseSpec(
        "finetune",
        "finetune",
        "init-1",
        "model-asset",
        "task-data",
        "checkpoint-finetune",
        5,
        parent_phase_id="pretrain",
        input_checkpoint_id="checkpoint-pretrain",
        task_id="task-asset",
    )
    return TrainingRouteSpec(
        route_id,
        (finetune, direct, pretrain),
        "local_fixture",
        decision,
    )


def test_training_route_validates_dag_lineage_and_shared_initialization() -> None:
    route_a = _route("route-a", _decision())
    route_b = _route("route-b", _decision())
    assert route_a.topological_order == ("direct", "pretrain", "finetune")
    assert validate_comparable_routes((route_a, route_b)) == "init-1"
    assert len(route_a.lineage_hash) == 64


def test_training_route_metadata_and_mapping_gate_are_recursively_immutable() -> None:
    phase_metadata = {"optimizer": {"schedule": ["warmup", "decay"]}}
    phase = TrainingPhaseSpec(
        "direct",
        "direct_supervised",
        "init-1",
        "model",
        "data",
        "checkpoint",
        1,
        task_id="task",
        metadata=phase_metadata,
    )
    phase_metadata["optimizer"]["schedule"].append("edited")  # type: ignore[index,union-attr]
    assert phase.to_dict()["metadata"] == {
        "optimizer": {"schedule": ["warmup", "decay"]}
    }
    with pytest.raises(TypeError):
        phase.metadata["optimizer"]["new"] = True  # type: ignore[index]

    decision = _decision("formal", "PASS", gate_status="PASS")
    gate = GateRecord(
        gate_id="stage2.G2.7b",
        stage=2,
        status=GateStatus.PASS,
        checked_at="2026-07-22T00:00:00Z",
        evidence_refs=("decisions/stage2.json",),
    )
    route = TrainingRouteSpec(
        "formal-immutable",
        (phase,),
        "formal",
        decision.to_dict(),
        metadata={"matrix": {"version": 1}},
        estimator_gate=gate.to_dict(),
    )
    assert isinstance(route.estimator_decision, EstimatorDecision)
    assert isinstance(route.estimator_gate, GateRecord)
    with pytest.raises(TypeError):
        route.metadata["matrix"]["version"] = 2  # type: ignore[index]


def test_formal_route_rejects_fixture_estimator_decision() -> None:
    phase = TrainingPhaseSpec(
        "direct",
        "direct_supervised",
        "init-1",
        "model",
        "data",
        "checkpoint",
        1,
        task_id="task",
    )
    with pytest.raises(FormalDecisionBlocked):
        TrainingRouteSpec("formal-route", (phase,), "formal", _decision())


def test_formal_route_requires_and_accepts_gate_passed_decision() -> None:
    phase = TrainingPhaseSpec(
        "direct",
        "direct_supervised",
        "init-1",
        "model-asset-id",
        "dataset-asset-id",
        "checkpoint-output-id",
        1,
        task_id="task-asset-id",
    )
    decision = _decision("formal", "PASS", gate_status="PASS")
    gate = GateRecord(
        gate_id="stage2.G2.7b",
        stage=2,
        status=GateStatus.PASS,
        checked_at="2026-07-22T00:00:00Z",
        evidence_refs=("decisions/stage2.json",),
    )
    with pytest.raises(FormalDecisionBlocked, match="独立"):
        TrainingRouteSpec("formal-route-no-gate", (phase,), "formal", decision)
    route = TrainingRouteSpec(
        "formal-route", (phase,), "formal", decision, estimator_gate=gate
    )
    assert route.run_intent == "formal"

    rejected = _decision("formal", "PASS", gate_status="NOT_RUN")
    with pytest.raises(FormalDecisionBlocked, match="Gate"):
        TrainingRouteSpec(
            "formal-route-blocked", (phase,), "formal", rejected, estimator_gate=gate
        )
