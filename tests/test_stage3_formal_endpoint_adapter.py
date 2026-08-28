"""Stage 3 formal endpoint adapter 的本机可达性测试。

测试使用 tiny Torch 资源替代未安装的 Transformers，但所有输入仍采用 formal
shape：FormalExecutionEvidence、PASS Gate、formal endpoint commit 与 hash-bound
probe plan。这样只替换资源工厂，不绕过正式 handler 的 endpoint/probe 解引用、
资格校验、路径状态安装或实际 autograd。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from ops.stage3 import run_stage3_formal as formal
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.contracts.stage23 import validate_stage23_artifact
from param_importance_nlp.experiments import TrainingEndpointObserver
from param_importance_nlp.experiments import stage23_task_runners as stage23
from param_importance_nlp.providers import (
    InMemoryFrozenSampleResolver,
    TorchFixedStateGradientProvider,
    build_tiny_training_fixture,
)
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskBlockedError,
    TrainingEngine,
    TrainingRunSpec,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass
class _BaseConfig:
    sections: dict[str, object]

    def section(self, name: str):
        return self.sections[name]


@dataclass
class _Config:
    base_config: _BaseConfig
    sections: dict[str, object]
    run_intent: str = "formal"

    def section(self, name: str):
        return self.sections[name]


def _formal_evidence(probe_ref: str) -> FormalExecutionEvidence:
    gates = tuple(
        GateRecord(
            gate_id=gate_id,
            stage=3,
            status=GateStatus.PASS,
            checked_at="2026-07-22T00:00:00+00:00",
            evidence_refs=(probe_ref,),
        )
        for gate_id in ("stage3.G3-0", "stage3.G3-1")
    )
    return FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_hash("freeze"),
        asset_manifest_hashes=(_hash("model"), _hash("data")),
        prerequisite_gates=gates,
    )


def test_formal_handler_loads_training_endpoint_and_executes_tiny_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    probe_ref = "inputs/probe-plan.json"
    evidence = _formal_evidence(probe_ref)
    fixture = build_tiny_training_fixture(
        task_type="sequence_classification", seed=131, steps=4
    )
    optimizer = torch.optim.SGD(fixture.model.module.parameters(), lr=0.05, momentum=0.9)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "formal-shaped-tiny",
            "formal",
            max_steps=1,
            max_attempts=1,
            importance_enabled=True,
            estimator_decision_hash=_hash("decision"),
            estimator_gate_status="PASS",
            weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=fixture.model,
        optimizer=optimizer,
        cursor=fixture.dataset.cursor(seed=131),
    )
    observer = TrainingEndpointObserver(
        source_run_id="formal-shaped-tiny",
        parameter_registry_hash=engine.registry.coordinate_registry_hash,
        selected_steps={1},
        output_root=tmp_path / "training" / "endpoints",
        workspace_root=tmp_path,
        scope="formal",
        formal_eligible=True,
        qualification_evidence_hash=evidence.artifact_hash,
    )
    observer.bind_engine(engine)
    engine.register_observer(observer)
    assert engine.run().status == "COMPLETE"
    endpoint = observer.bundles[0]

    samples = {
        batch.batch_id: batch
        for step in fixture.dataset.steps
        for batch in step
    }
    resolver = InMemoryFrozenSampleResolver(
        samples,
        resolver_id="formal-shaped-tiny-probes",
        loss_unit="sample",
        statistical_unit="sample",
        weight_unit="sample",
        sampling_design="fixed_panel",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    provider = TorchFixedStateGradientProvider(
        fixture.model,
        resolver,
        fixed_state_id="formal-shaped-tiny-post",
        output_dtype=torch.float64,
    )
    provider_context = stage23._ProviderContext(
        provider=provider,
        sample_ids=resolver.sample_ids,
        evidence=evidence,
        provider_kind="tiny_formal_shaped_test",
        asset_manifest_hashes=evidence.asset_manifest_hashes,
    )
    monkeypatch.setattr(stage23, "_formal_provider", lambda _request, _root: provider_context)

    loss_hash = canonical_json_hash(
        {
            "task_type": "sequence_classification",
            "reduction": "mean",
            "weighting": "sample",
        }
    )
    probe_body = {
        "schema_version": "stage3-probe-plan-v1",
        "panel_id": "formal-shaped-tiny-panel",
        "endpoint_digest": endpoint.endpoint_digest,
        "entries": [
            {
                "role": "formal",
                "probe_id": f"formal-probe-{index}",
                "sample_ids": [resolver.sample_ids[-3 + index]],
                "content_hash": _hash(f"formal-probe-content-{index}"),
                "loss_contract_hash": loss_hash,
                "effective_weight_unit": "sample",
                "metadata": {"source": "tiny-formal-shaped"},
            }
            for index in range(3)
        ],
        "minimum_formal_probes": 3,
        "execution_evidence_hash": evidence.artifact_hash,
        "scope": "formal",
        "formal_eligible": True,
    }
    probe_body["artifact_hash"] = canonical_json_hash(probe_body)
    write_canonical_json(tmp_path / probe_ref, probe_body)

    endpoint_ref = Path(endpoint.commit_ref).as_posix()
    base_optimizer = {
        "type": "sgd",
        "learning_rate": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "parameter_groups": [],
    }
    config = _Config(
        _BaseConfig(
            {
                "optimizer": base_optimizer,
                "precision": {"path_accumulation_dtype": "float64"},
            }
        ),
        {
            "orchestration": {
                "input_result_refs": [endpoint_ref, probe_ref],
                "active_probe_id": "formal-probe-0",
            },
            "optimizer_runtime": {
                "betas": None,
                "eps": None,
                "amsgrad": False,
                "dampening": 0.0,
                "nesterov": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
            },
        },
    )
    request = SimpleNamespace(
        config=config,
        task=DEFAULT_TASK_CATALOG.get("stage3.03_endpoint_and_probe_pipeline"),
    )
    predecessor = SimpleNamespace(
        references=(endpoint_ref, probe_ref),
        payload=lambda kind: (
            {"undefined_policy": "defined_false_with_reason_no_epsilon"}
            if kind == "metric_contract"
            else None
        ),
    )
    monkeypatch.setattr(stage23, "_predecessor_context", lambda *_args: predecessor)
    store = TaskArtifactStore(tmp_path, "runs/stage3-formal-shaped")

    payloads, refs = stage23._run_stage3_endpoint(request, tmp_path, store)

    assert refs == (endpoint_ref, probe_ref)
    assert payloads["probe_manifest"]["formal_eligible"] is True
    assert payloads["state_restoration_report"]["formal_eligible"] is True
    assert payloads["path_spec"]["path_identity_hash"]
    # handler 构造出的 path 执行真实 tiny autograd；不同规则先复用公共端点，随后
    # 新建 context 模拟 fresh-process resume，第三次求积不再计算任何节点梯度。
    calls: list[int] = []
    original_gradient_at = provider.gradient_at_parameter_state

    def counting_gradient_at(*args, **kwargs):
        calls.append(1)
        return original_gradient_at(*args, **kwargs)

    monkeypatch.setattr(provider, "gradient_at_parameter_state", counting_gradient_at)
    trapezoid = stage23.trapezoid_rule()
    simpson = stage23.simpson_rule()
    context = stage23._formal_path_context(request, tmp_path, store)
    first = context.integrate(trapezoid)
    after_trapezoid = len(calls)
    second = context.integrate(simpson)
    after_simpson = len(calls)
    resumed = stage23._formal_path_context(request, tmp_path, store)
    third = resumed.integrate(simpson)

    # 每条规则还会计算各节点 loss 与两个端点 loss。梯形首次为 2 梯度+4 loss；
    # Simpson 只新增 alpha=.5 的 1 个梯度；fresh context 则三个梯度全由 commit 恢复。
    assert after_trapezoid == 6
    assert after_simpson - after_trapezoid == 6
    assert len(calls) - after_simpson == 5
    assert context.node_cache is not resumed.node_cache
    assert len(context.node_cache) == len(resumed.node_cache) == 3
    assert first.unique_gradient_evaluations == 2
    assert second.unique_gradient_evaluations == third.unique_gradient_evaluations == 3
    second_cost = context.cost_for(simpson)
    third_cost = resumed.cost_for(simpson)
    assert second_cost.gradient_evaluations == 1
    assert second_cost.loss_evaluations == 5  # 3 nodes + 2 endpoint losses
    assert second_cost.forward_evaluations == second_cost.backward_evaluations == 6
    assert third_cost.gradient_evaluations == 0  # persistent node cache restores all gradients
    assert third_cost.loss_evaluations == 5
    assert third_cost.forward_evaluations == third_cost.backward_evaluations == 5
    assert second_cost.wall_seconds > 0
    for name in second.signed:
        torch.testing.assert_close(second.signed[name], third.signed[name])
    cache_evidence = resumed.cache_evidence((trapezoid, simpson))
    assert cache_evidence["cross_rule_reused_key_count"] == 2
    assert cache_evidence["commit_evidence"]["all_requested_keys_committed"] is True
    assert cache_evidence["commit_evidence"]["reconciliation"]["orphan_objects"] == []
    assert all(torch.isfinite(value).all() for value in third.signed.values())
    commit_value = load_canonical_json(tmp_path / endpoint_ref)
    assert commit_value["formal_eligible"] is True

    # A three-probe formal panel must not silently make the first probe the
    # only executable unit.  Removing the explicit binding is a structured
    # contract blocker, even though the endpoint and panel themselves are
    # otherwise valid.
    unbound_config = _Config(
        config.base_config,
        {
            "orchestration": {"input_result_refs": [endpoint_ref, probe_ref]},
            "optimizer_runtime": config.sections["optimizer_runtime"],
        },
    )
    unbound_request = SimpleNamespace(
        config=unbound_config,
        task=DEFAULT_TASK_CATALOG.get("stage3.03_endpoint_and_probe_pipeline"),
    )
    with pytest.raises(TaskBlockedError, match="active_probe_id"):
        stage23._formal_path_context(unbound_request, tmp_path, store)


def test_pilot_endpoint_and_two_probe_selector_stays_pilot_through_g32(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The real endpoint handler must keep pilot scope and satisfy G3-2's pilot contract."""

    probe_ref = "inputs/pilot-probe-plan.json"
    evidence = _formal_evidence(probe_ref)
    fixture = build_tiny_training_fixture(
        task_type="sequence_classification", seed=211, steps=4
    )
    optimizer = torch.optim.SGD(fixture.model.module.parameters(), lr=0.05, momentum=0.9)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "pilot-shaped-tiny",
            "formal",
            max_steps=1,
            max_attempts=1,
            importance_enabled=True,
            estimator_decision_hash=_hash("decision-pilot"),
            estimator_gate_status="PASS",
            weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=fixture.model,
        optimizer=optimizer,
        cursor=fixture.dataset.cursor(seed=211),
    )
    observer = TrainingEndpointObserver(
        source_run_id="pilot-shaped-tiny",
        parameter_registry_hash=engine.registry.coordinate_registry_hash,
        selected_steps={1},
        output_root=tmp_path / "training" / "pilot-endpoints",
        workspace_root=tmp_path,
        scope="pilot",
        formal_eligible=False,
    )
    observer.bind_engine(engine)
    engine.register_observer(observer)
    assert engine.run().status == "COMPLETE"
    endpoint = observer.bundles[0]

    samples = {
        batch.batch_id: batch
        for step in fixture.dataset.steps
        for batch in step
    }
    resolver = InMemoryFrozenSampleResolver(
        samples,
        resolver_id="pilot-shaped-tiny-probes",
        loss_unit="sample",
        statistical_unit="sample",
        weight_unit="sample",
        sampling_design="fixed_panel",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    provider = TorchFixedStateGradientProvider(
        fixture.model,
        resolver,
        fixed_state_id="pilot-shaped-tiny-post",
        output_dtype=torch.float64,
    )
    provider_context = stage23._ProviderContext(
        provider=provider,
        sample_ids=resolver.sample_ids,
        evidence=evidence,
        provider_kind="tiny_formal_shaped_test",
        asset_manifest_hashes=evidence.asset_manifest_hashes,
    )
    monkeypatch.setattr(stage23, "_formal_provider", lambda _request, _root: provider_context)

    loss_hash = canonical_json_hash(
        {
            "task_type": "sequence_classification",
            "reduction": "mean",
            "weighting": "sample",
        }
    )
    probe_body = {
        "schema_version": "stage3-probe-plan-v1",
        "panel_id": "pilot-shaped-tiny-panel",
        "endpoint_digest": endpoint.endpoint_digest,
        "entries": [
            {
                "role": "pilot",
                "probe_id": f"pilot-probe-{index}",
                "sample_ids": [resolver.sample_ids[-2 + index]],
                "content_hash": _hash(f"pilot-probe-content-{index}"),
                "loss_contract_hash": loss_hash,
                "effective_weight_unit": "sample",
                "metadata": {"source": "tiny-pilot-shaped"},
            }
            for index in range(2)
        ],
        "minimum_formal_probes": 2,
        "execution_evidence_hash": evidence.artifact_hash,
        "scope": "pilot",
        "formal_eligible": False,
    }
    probe_body["artifact_hash"] = canonical_json_hash(probe_body)
    write_canonical_json(tmp_path / probe_ref, probe_body)
    selector = {
        "schema_version": "stage3-probe-selector-v1",
        "scope": "pilot",
        "endpoint_digest": endpoint.endpoint_digest,
        "probe_plan_hash": probe_body["artifact_hash"],
        "active_probe_id": "pilot-probe-0",
    }
    selector["artifact_hash"] = canonical_json_hash(selector)
    selector_ref = "inputs/pilot-selector.json"
    write_canonical_json(tmp_path / selector_ref, selector)

    endpoint_ref = Path(endpoint.commit_ref).as_posix()
    config = _Config(
        _BaseConfig(
            {
                "optimizer": {
                    "type": "sgd",
                    "learning_rate": 0.05,
                    "momentum": 0.9,
                    "weight_decay": 0.0,
                    "foreach": False,
                    "fused": False,
                    "parameter_groups": [],
                },
                "precision": {"path_accumulation_dtype": "float64"},
            }
        ),
        {
            "orchestration": {
                "input_result_refs": [endpoint_ref, probe_ref],
                "route_spec_ref": selector_ref,
                "active_probe_id": "pilot-probe-0",
            },
            "optimizer_runtime": {
                "betas": None,
                "eps": None,
                "amsgrad": False,
                "dampening": 0.0,
                "nesterov": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
            },
        },
    )
    request = SimpleNamespace(
        config=config,
        task=DEFAULT_TASK_CATALOG.get("stage3.03_endpoint_and_probe_pipeline"),
    )
    predecessor = SimpleNamespace(
        references=(endpoint_ref, probe_ref),
        payload=lambda kind: (
            {"undefined_policy": "defined_false_with_reason_no_epsilon"}
            if kind == "metric_contract"
            else None
        ),
    )
    monkeypatch.setattr(stage23, "_predecessor_context", lambda *_args: predecessor)
    store = TaskArtifactStore(tmp_path, "runs/stage3-pilot-shaped")

    payloads, refs = stage23._run_stage3_endpoint(request, tmp_path, store)

    assert refs == (endpoint_ref, probe_ref)
    assert payloads["probe_manifest"]["scope"] == "pilot"
    assert payloads["probe_manifest"]["formal_eligible"] is False
    assert len(payloads["probe_manifest"]["entries"]) == 2
    assert {entry["role"] for entry in payloads["probe_manifest"]["entries"]} == {"pilot"}
    assert payloads["state_restoration_report"]["scope"] == "pilot"
    assert payloads["state_restoration_report"]["formal_eligible"] is False
    validate_stage23_artifact(payloads["probe_manifest"])
    measured, threshold = formal._validate_local_gate_outputs(
        "stage3.03_endpoint_and_probe_pipeline",
        {kind: SimpleNamespace(payload=value) for kind, value in payloads.items()},
    )
    assert measured == {"probe_count": 2, "replay_verified": True}
    assert threshold == {"minimum_independent_pilot_probes": 2, "full_update_endpoint": True}


def test_formal_path_unit_selector_uses_hash_bound_route_spec(tmp_path: Path) -> None:
    selector = {
        "schema_version": "stage3-path-unit-selector-v1",
        "scope": "pilot",
        "unit_index_hash": "a" * 64,
        "active_unit_id": "path-unit-real-001",
    }
    selector["artifact_hash"] = canonical_json_hash(selector)
    write_canonical_json(tmp_path / "selector.json", selector)
    request = SimpleNamespace(
        config=_Config(
            _BaseConfig({"path_integration": {}}),
            {"orchestration": {"route_spec_ref": "selector.json"}},
        ),
        task=SimpleNamespace(task_id="stage3.05_reference_integral_and_precision"),
    )
    assert stage23._formal_stage3_probe_selector(request, tmp_path) == (
        None,
        "path-unit-real-001",
        "a" * 64,
        None,
    )


def test_formal_path_unit_selector_rejects_hash_drift(tmp_path: Path) -> None:
    selector = {
        "schema_version": "stage3-path-unit-selector-v1",
        "scope": "formal",
        "unit_index_hash": "a" * 64,
        "active_unit_id": "path-unit-real-001",
        "artifact_hash": "b" * 64,
    }
    write_canonical_json(tmp_path / "selector-drift.json", selector)
    request = SimpleNamespace(
        config=_Config(
            _BaseConfig({"path_integration": {}}),
            {"orchestration": {"route_spec_ref": "selector-drift.json"}},
        ),
        task=SimpleNamespace(task_id="stage3.07_formal_experiment_matrix"),
    )
    with pytest.raises(TaskBlockedError, match="selector hash"):
        stage23._formal_stage3_probe_selector(request, tmp_path)


def test_formal_probe_selector_binds_pre_index_endpoint_and_plan(tmp_path: Path) -> None:
    selector = {
        "schema_version": "stage3-probe-selector-v1",
        "scope": "formal",
        "endpoint_digest": "a" * 64,
        "probe_plan_hash": "b" * 64,
        "active_probe_id": "formal-probe-01",
    }
    selector["artifact_hash"] = canonical_json_hash(selector)
    write_canonical_json(tmp_path / "probe-selector.json", selector)
    request = SimpleNamespace(
        config=_Config(
            _BaseConfig({"path_integration": {}}),
            {"orchestration": {"route_spec_ref": "probe-selector.json"}},
        ),
        task=SimpleNamespace(task_id="stage3.03_endpoint_and_probe_pipeline"),
    )
    assert stage23._formal_stage3_probe_selector(request, tmp_path) == (
        "formal-probe-01",
        None,
        None,
        ("a" * 64, "b" * 64),
    )


def test_ddp_wrapper_names_are_normalized_without_rejecting_wrapper_root() -> None:
    normalized = stage23._normalize_ddp_names(
        {"": True, "module": True, "module.block": False},
        ("", "block"),
        field="model modes",
    )
    assert normalized == {"": True, "block": False}


def test_missing_formal_endpoint_is_a_structured_blocker(tmp_path: Path) -> None:
    evidence = _formal_evidence("inputs/probe-plan.json")
    config = _Config(
        _BaseConfig({}),
        {"orchestration": {"input_result_refs": []}},
    )
    request = SimpleNamespace(config=config)

    with pytest.raises(TaskBlockedError) as caught:
        stage23._load_formal_endpoint_and_probe_plan(request, tmp_path, evidence)

    assert caught.value.blockers[0].requirement == "training_endpoint_commit"


def test_formal_reference_levels_are_explicit_and_refined() -> None:
    ladder = {
        "families": {
            "gauss_legendre": [2, 4, 8],
            "composite_simpson": [2, 4, 8],
        },
        "frozen": True,
    }
    config = _Config(
        _BaseConfig({"path_integration": {"reference_ladder": ladder}}),
        {},
    )
    request = SimpleNamespace(config=config)
    context = SimpleNamespace(execution=_formal_evidence("inputs/ladder.json"))

    levels = stage23._stage3_reference_levels(request, Path("."), context)

    assert [level.unique_nodes for level in levels] == [2, 4, 8, 3, 5, 9]
    assert [level.family for level in levels] == [
        "gauss_legendre",
        "gauss_legendre",
        "gauss_legendre",
        "composite_simpson",
        "composite_simpson",
        "composite_simpson",
    ]


def test_formal_reference_levels_without_frozen_ladder_are_blocked() -> None:
    config = _Config(_BaseConfig({"path_integration": {}}), {})
    request = SimpleNamespace(config=config)
    context = SimpleNamespace(execution=_formal_evidence("inputs/ladder.json"))

    with pytest.raises(TaskBlockedError, match="reference_ladder"):
        stage23._stage3_reference_levels(request, Path("."), context)


def test_formal_reference_ladder_identity_requires_recomputable_hash() -> None:
    body = {
        "families": {
            "gauss_legendre": [2, 4, 8],
            "composite_simpson": [2, 4, 8],
        },
        "frozen": True,
    }
    ladder = dict(body)
    ladder["artifact_hash"] = canonical_json_hash(body)
    config = _Config(
        _BaseConfig({"path_integration": {"reference_ladder": ladder}}),
        {},
    )
    request = SimpleNamespace(config=config)

    document, ladder_hash, gauss, simpson = stage23._stage3_reference_ladder_document(
        request,
        Path("."),
        require_artifact_hash=True,
    )

    assert document == ladder
    assert ladder_hash == ladder["artifact_hash"]
    assert gauss == (2, 4, 8)
    assert simpson == (2, 4, 8)


def test_formal_reference_ladder_identity_rejects_unhashed_frozen_document() -> None:
    ladder = {
        "families": {
            "gauss_legendre": [2, 4, 8],
            "composite_simpson": [2, 4, 8],
        },
        "frozen": True,
    }
    config = _Config(
        _BaseConfig({"path_integration": {"reference_ladder": ladder}}),
        {},
    )
    request = SimpleNamespace(config=config)

    with pytest.raises(TaskBlockedError, match="artifact_hash"):
        stage23._stage3_reference_ladder_document(
            request,
            Path("."),
            require_artifact_hash=True,
        )


def test_formal_reference_binding_is_immutable_and_requires_two_rounds() -> None:
    binding = stage23.FormalReferenceBinding(
        formal_plan_ref="plans/stage3.json",
        formal_plan_hash=_hash("plan"),
        thresholds_hash=_hash("thresholds"),
        reference_tolerance=1e-5,
        reference_ladder_hash=_hash("ladder"),
        reference_ladder_levels={
            "gauss_legendre": [2, 4, 8],
            "composite_simpson": [2, 4, 8],
        },
        reference_ladder_nodes={
            "gauss_legendre": [2, 4, 8],
            "composite_simpson": [3, 5, 9],
        },
    )

    restored = stage23.FormalReferenceBinding.from_mapping(binding.to_dict())
    assert restored == binding
    assert binding.artifact_fields()["reference_binding_hash"] == binding.binding_hash

    with pytest.raises(ValueError, match="required_consecutive"):
        stage23.FormalReferenceBinding(
            formal_plan_ref=binding.formal_plan_ref,
            formal_plan_hash=binding.formal_plan_hash,
            thresholds_hash=binding.thresholds_hash,
            reference_tolerance=binding.reference_tolerance,
            reference_ladder_hash=binding.reference_ladder_hash,
            reference_ladder_levels=binding.reference_ladder_levels,
            reference_ladder_nodes=binding.reference_ladder_nodes,
            required_consecutive=1,
            primary_family=binding.primary_family,
        )

    with pytest.raises(ValueError, match="primary_family"):
        stage23.FormalReferenceBinding(
            formal_plan_ref=binding.formal_plan_ref,
            formal_plan_hash=binding.formal_plan_hash,
            thresholds_hash=binding.thresholds_hash,
            reference_tolerance=binding.reference_tolerance,
            reference_ladder_hash=binding.reference_ladder_hash,
            reference_ladder_levels=binding.reference_ladder_levels,
            reference_ladder_nodes=binding.reference_ladder_nodes,
            primary_family="trapezoid",
        )
