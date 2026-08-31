from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from ops.stage3 import materialize_stage3_endpoint_trajectory as materializer
from ops.stage3 import materialize_stage3_task as direct_materializer
from ops.stage3.run_stage3_formal import _canonical_hash
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.runtime import TaskRuntimeEnvironment


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"
DECISION = ROOT / "reports" / "stage3" / "g3-0-user-scope-decision-20260828.json"
G30 = ROOT / "reports" / "stage3" / "g3-0-user-scope-gate-20260828.json"


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(root: Path, ref: str, value: object) -> None:
    path = root / Path(*ref.split("/"))
    assert not path.exists()
    write_canonical_json(path, value)


def _base(root: Path, *, model: str) -> str:
    value = copy.deepcopy(load_canonical_json(BASE))
    assert isinstance(value, dict)
    # Keep the fixture template as a schema source only; every execution-facing
    # identity below is changed to a real Pythia/offline shape.
    value["identity"].update(  # type: ignore[index]
        {
            "stage": 3,
            "task": materializer.TASK_ID,
            "route": "path_integration",
            "master_seed": 7,
            "run_intent": "formal",
            "formal_eligible": True,
            "input_run_id": materializer.EXPECTED_STAGE2_RUN_ID,
        }
    )
    value["runtime"].update(  # type: ignore[index]
        {
            "environment_id": "server-cuda-formal-v1",
            "device": "cuda",
            "offline": True,
            "allow_dirty_worktree": False,
            "cache_root": "cache/stage3/base",
            "output_root": "runs/stage3/base",
            "temp_root": "cache/stage3/base/tmp",
        }
    )
    value["model"].update(  # type: ignore[index]
        {
            "asset_id": f"pythia-{model.casefold()}-step0",
            "revision": f"pythia-{model.casefold()}-revision",
            "initialization_id": f"pythia-{model.casefold()}-initialization",
            "architecture": f"pythia-{model.casefold()}",
        }
    )
    value["data"].update(  # type: ignore[index]
        {
            "asset_id": "pile-selected-prefix",
            "revision": "pile-revision",
            "split": "train",
            "sampler": "without_replacement",
            "sampling_design": "without_replacement_frozen_epoch",
        }
    )
    value["sampling"]["universe_version"] = "stage2-pythia-grid-20260826T145530Z"  # type: ignore[index]
    value["loss"]["task_type"] = "causal_lm"  # type: ignore[index]
    value["importance"].update(  # type: ignore[index]
        {
            "estimator_name": "u",
            "clip_mode": "none",
            "require_decision_for_formal": True,
        }
    )
    value["path_integration"].update(  # type: ignore[index]
        {
            "enabled": True,
            "path_name": "full_update_linear",
            "default_rule": "simpson",
            "fallback_rule": "gauss_legendre_8",
            "node_budget": 16,
            "probe_count": 3,
            "thresholds_ref": "plans/stage3-thresholds.json",
        }
    )
    predecessor_refs = [
        "stage3/s302/commits/path_math_contract.json",
        "stage3/s302/commits/metric_contract.json",
        "stage3/s302/commits/gate_record.json",
    ]
    resolved = ResolvedConfigV2.resolve(
        value,
        task_id=materializer.TASK_ID,
        overrides={
            "training": {"max_steps": 6},
            "providers": {
                "kind": "offline_hf",
                "model_manifest_ref": None,
                "model_root_ref": None,
                "data_manifest_ref": None,
                "data_root_ref": None,
                "tokenizer_manifest_ref": None,
                "tokenizer_root_ref": None,
                "task_type": "causal_lm",
                "task_name": "pile",
                "num_labels": None,
                "local_files_only": True,
                "trust_remote_code": False,
            },
            "orchestration": {"input_result_refs": predecessor_refs},
        },
    )
    _write(root, "base.json", resolved.to_dict())
    return "base.json"


def _authority(root: Path) -> tuple[str, str, str, str]:
    decision = load_canonical_json(DECISION)
    gate = load_canonical_json(G30)
    assert isinstance(decision, dict) and isinstance(gate, dict)
    # Re-publish G3-0 beside the copied decision and recompute its immutable
    # hash; the checked-in report points at its original decision path.
    g30_original = GateRecord.from_mapping(gate)
    g30_bound = GateRecord(
        gate_id=g30_original.gate_id,
        stage=g30_original.stage,
        status=g30_original.status,
        checked_at=g30_original.checked_at,
        measured=g30_original.measured,
        threshold=g30_original.threshold,
        evidence_refs=("authority/g30-decision.json",),
        reasons=g30_original.reasons,
        conditions=g30_original.conditions,
        expires_at=g30_original.expires_at,
    )
    _write(root, "authority/g30-decision.json", decision)
    _write(root, "authority/g30-gate.json", g30_bound.to_dict())
    g31 = GateRecord(
        gate_id="stage3.G3-1",
        stage=3,
        status=GateStatus.PASS,
        checked_at="2026-08-28T16:00:00Z",
        measured={"contract": "frozen"},
        threshold={"required": True},
        evidence_refs=("authority/g30-decision.json",),
    )
    _write(root, "authority/g31-gate.json", g31.to_dict())
    g35 = GateRecord(
        gate_id="stage3.G3-5",
        stage=3,
        status=GateStatus.PASS,
        checked_at="2026-08-29T00:00:00Z",
        measured={"contract": "frozen"},
        threshold={"required": True},
        evidence_refs=("authority/g35-gate.json",),
    )
    _write(root, "authority/g35-gate.json", g35.to_dict())
    evidence = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash=_h("stage3-contract"),
        asset_manifest_hashes=(_h("asset-model"), _h("asset-data")),
        prerequisite_gates=(g30_bound, g31, g35),
        metadata={"scope": "formal", "source": "published"},
    )
    _write(root, "authority/formal-execution.json", evidence.to_dict())
    estimator = {
        "run_id": materializer.EXPECTED_STAGE2_RUN_ID,
        "default_estimator": "U-32",
        "batch_size": 32,
        "data_variant": "Raw",
    }
    _write(root, "authority/stage2-estimator.json", estimator)
    g30_env = TaskRuntimeEnvironment(
        capabilities=frozenset({"server", "cuda", "model_assets", "data_assets"}),
        frozen_contract_stages=frozenset({0, 1, 3}),
        passed_gate_ids=frozenset({"stage3.G3-0"}),
        estimator_decision_ref="authority/stage2-estimator.json",
        evidence_refs={
            "stage3_scope_decision": "authority/g30-decision.json",
            "stage3_g30_gate": "authority/g30-gate.json",
        },
    )
    g31_env = TaskRuntimeEnvironment(
        capabilities=g30_env.capabilities,
        frozen_contract_stages=frozenset({0, 1, 3}),
        passed_gate_ids=frozenset({"stage3.G3-0", "stage3.G3-1", "stage3.G3-5"}),
        estimator_decision_ref="authority/stage2-estimator.json",
        evidence_refs={
            "stage3_scope_decision": "authority/g30-decision.json",
            "stage3_g30_gate": "authority/g30-gate.json",
            "gate_stage3_g3_1": "authority/g31-gate.json",
            "gate_stage3_g3_5": "authority/g35-gate.json",
            "formal_execution": "authority/formal-execution.json",
        },
    )
    _write(root, "authority/g30-environment.json", g30_env.to_dict())
    _write(root, "authority/g31-environment.json", g31_env.to_dict())
    return (
        "authority/stage2-estimator.json",
        "authority/g30-environment.json",
        "authority/g31-environment.json",
        "authority/formal-execution.json",
    )


def _source(root: Path, *, scope: str, model: str = "14M") -> dict[str, object]:
    _, g30_ref, g31_ref, _ = _authority(root)
    base_ref = _base(root, model=model)
    stage2_output = "authority/stage2-output.json"
    _write(
        root,
        stage2_output,
        {
            "run_id": materializer.EXPECTED_STAGE2_RUN_ID,
            "default_estimator": "U-32",
            "batch_size": 32,
            "data_variant": "Raw",
            "output": "real-stage2-output",
        },
    )
    steps = [1, 3] if scope == "pilot" else [2, 5]
    source: dict[str, object] = {
        "schema_version": materializer.SOURCE_SCHEMA,
        "scope": scope,
        "plan_id": f"endpoint-{scope}-{model.casefold()}",
        "model": model,
        "seed": 7,
        "max_steps": 6,
        "selected_steps": steps,
        "include_checkpoint_steps": False,
        "endpoint_metadata": {
            str(step): {"model": model, "seed": 7, "stage": "early" if step < 4 else "middle"}
            for step in steps
        },
        "probe_plan_ref": None,
        "base_config_ref": base_ref,
        "stage2_authority_ref": "authority/stage2-estimator.json",
        "stage2_output_refs": [stage2_output],
        "g30_environment_ref": g30_ref,
        "g31_environment_ref": g31_ref,
        "artifact_output_dir": f"runs/stage3/trajectory/{scope}/{model.casefold()}",
        "cache_root": f"cache/stage3/trajectory/{scope}/{model.casefold()}",
        "capture_plan_ref": f"inputs/stage3/trajectory/{scope}/{model.casefold()}/capture-plan.json",
        "config_ref": f"configs/stage3/trajectory/{scope}/{model.casefold()}.json",
        "environment_ref": f"environments/stage3/trajectory/{scope}/{model.casefold()}.json",
        "materialization_receipt_ref": f"receipts/stage3/trajectory/{scope}/{model.casefold()}.json",
        "trajectory_receipt_ref": f"results/stage3/trajectory/{scope}/{model.casefold()}.json",
    }
    return source


def test_materializer_builds_real_14m_pilot_without_execution(tmp_path: Path) -> None:
    value = materializer.materialize(_source(tmp_path, scope="pilot"), workspace_root=tmp_path, data_root=tmp_path)
    assert value["scope"] == "pilot"
    assert value["formal_eligible"] is False
    plan = load_canonical_json(tmp_path / value["capture_plan_ref"])
    assert plan["scope"] == "pilot"
    assert plan["qualification_evidence_hash"] is None
    config = ResolvedConfigV2.from_mapping(load_canonical_json(tmp_path / value["config_ref"]))
    assert config.run_intent == "formal"  # pilot is real training, not fixture
    data = config.base_config.section("data")
    assert data["split"] == "train"
    assert data["sampling_design"] == "without_replacement_frozen_epoch"
    assert data["sampler"] == "without_replacement"
    assert config.section("training")["max_steps"] == 6
    assert config.section("orchestration")["input_result_refs"] == [
        "stage3/s302/commits/path_math_contract.json",
        "stage3/s302/commits/metric_contract.json",
        "stage3/s302/commits/gate_record.json",
    ]
    assert not (tmp_path / value["trajectory_receipt_ref"]).exists()


def test_materializer_builds_31m_formal_and_binds_g31(tmp_path: Path) -> None:
    value = materializer.materialize(_source(tmp_path, scope="formal", model="31M"), workspace_root=tmp_path, data_root=tmp_path)
    assert value["scope"] == "formal"
    assert value["formal_eligible"] is True
    plan = load_canonical_json(tmp_path / value["capture_plan_ref"])
    assert plan["scope"] == "formal"
    assert plan["formal_eligible"] is True
    assert isinstance(plan["qualification_evidence_hash"], str)
    config = ResolvedConfigV2.from_mapping(load_canonical_json(tmp_path / value["config_ref"]))
    assert config.base_config.section("model")["asset_id"] == "pythia-31m-step0"
    environment = TaskRuntimeEnvironment.from_mapping(load_canonical_json(tmp_path / value["environment_ref"]))
    assert {"stage3.G3-0", "stage3.G3-1", "stage3.G3-5"}.issubset(environment.passed_gate_ids)
    assert environment.evidence_refs["stage3_endpoint_capture_plan"] == value["capture_plan_ref"]


def test_materializer_rejects_formal_environment_before_g35(tmp_path: Path) -> None:
    source = _source(tmp_path, scope="formal")
    original = TaskRuntimeEnvironment.from_mapping(
        load_canonical_json(tmp_path / "authority/g31-environment.json")
    )
    evidence_refs = dict(original.evidence_refs)
    evidence_refs.pop("gate_stage3_g3_5")
    before_g35 = TaskRuntimeEnvironment(
        capabilities=original.capabilities,
        frozen_contract_stages=original.frozen_contract_stages,
        passed_gate_ids=original.passed_gate_ids - {"stage3.G3-5"},
        estimator_decision_ref=original.estimator_decision_ref,
        evidence_refs=evidence_refs,
    )
    ref = "authority/g31-environment-before-g35.json"
    _write(tmp_path, ref, before_g35.to_dict())
    source["g31_environment_ref"] = ref
    with pytest.raises(Exception, match="G35_ENVIRONMENT_GATE_MISSING"):
        materializer.materialize(source, workspace_root=tmp_path, data_root=tmp_path)
    assert not (tmp_path / source["capture_plan_ref"]).exists()


def test_materializer_rejects_formal_execution_without_g35(tmp_path: Path) -> None:
    source = _source(tmp_path, scope="formal")
    original_evidence = FormalExecutionEvidence.from_mapping(
        load_canonical_json(tmp_path / "authority/formal-execution.json")
    )
    without_g35 = FormalExecutionEvidence(
        run_intent=original_evidence.run_intent,
        contract_freeze_hash=original_evidence.contract_freeze_hash,
        asset_manifest_hashes=original_evidence.asset_manifest_hashes,
        prerequisite_gates=tuple(
            gate for gate in original_evidence.prerequisite_gates if gate.gate_id != "stage3.G3-5"
        ),
        metadata=original_evidence.metadata,
    )
    execution_ref = "authority/formal-execution-before-g35.json"
    _write(tmp_path, execution_ref, without_g35.to_dict())
    original_environment = TaskRuntimeEnvironment.from_mapping(
        load_canonical_json(tmp_path / "authority/g31-environment.json")
    )
    evidence_refs = dict(original_environment.evidence_refs)
    evidence_refs["formal_execution"] = execution_ref
    environment = TaskRuntimeEnvironment(
        capabilities=original_environment.capabilities,
        frozen_contract_stages=original_environment.frozen_contract_stages,
        passed_gate_ids=original_environment.passed_gate_ids,
        estimator_decision_ref=original_environment.estimator_decision_ref,
        evidence_refs=evidence_refs,
    )
    environment_ref = "authority/g31-environment-without-g35-evidence.json"
    _write(tmp_path, environment_ref, environment.to_dict())
    source["g31_environment_ref"] = environment_ref
    with pytest.raises(Exception, match="FORMAL_EXECUTION_G35_MISSING"):
        materializer.materialize(source, workspace_root=tmp_path, data_root=tmp_path)
    assert not (tmp_path / source["capture_plan_ref"]).exists()


def test_materializer_pilot_remains_compatible_without_g35_environment(tmp_path: Path) -> None:
    source = _source(tmp_path, scope="pilot")
    original = TaskRuntimeEnvironment.from_mapping(
        load_canonical_json(tmp_path / "authority/g31-environment.json")
    )
    evidence_refs = dict(original.evidence_refs)
    evidence_refs.pop("gate_stage3_g3_5")
    pilot_environment = TaskRuntimeEnvironment(
        capabilities=original.capabilities,
        frozen_contract_stages=original.frozen_contract_stages,
        passed_gate_ids=original.passed_gate_ids - {"stage3.G3-5"},
        estimator_decision_ref=original.estimator_decision_ref,
        evidence_refs=evidence_refs,
    )
    ref = "authority/g31-environment-pilot-without-g35.json"
    _write(tmp_path, ref, pilot_environment.to_dict())
    source["g31_environment_ref"] = ref
    value = materializer.materialize(source, workspace_root=tmp_path, data_root=tmp_path)
    assert value["scope"] == "pilot"


def test_scope_mixing_is_rejected_before_any_output_is_written(tmp_path: Path) -> None:
    source = _source(tmp_path, scope="pilot")
    source["endpoint_metadata"] = {
        "1": {"model": "31M", "seed": 7, "stage": "early"},
        "3": {"model": "14M", "seed": 7, "stage": "early"},
    }
    with pytest.raises(Exception, match="METADATA_IDENTITY_INVALID"):
        materializer.materialize(source, workspace_root=tmp_path, data_root=tmp_path)
    assert not (tmp_path / source["capture_plan_ref"]).exists()


def test_immutable_conflict_is_fail_closed(tmp_path: Path) -> None:
    source = _source(tmp_path, scope="pilot")
    materializer.materialize(source, workspace_root=tmp_path, data_root=tmp_path)
    changed = dict(source)
    changed["max_steps"] = 7
    with pytest.raises(Exception, match="IMMUTABLE_CONFLICT"):
        materializer.materialize(changed, workspace_root=tmp_path, data_root=tmp_path)


def test_direct_task_materializer_accepts_explicit_pilot_selector(tmp_path: Path) -> None:
    # The existing direct materializer is a separate path, but its selector
    # guard must not turn the pilot phase into an accidental formal-only route.
    for scope in ("pilot", "formal"):
        root = tmp_path / scope
        root.mkdir()
        stage2_ref, _g30_ref, _g31_ref, _execution_ref = _authority(root)
        base_ref = _base(root, model="14M")
        _write(root, "overrides.json", {})
        selector = {
            "schema_version": "stage3-probe-selector-v1",
            "scope": scope,
            "endpoint_digest": _h(f"{scope}-endpoint"),
            "probe_plan_hash": _h(f"{scope}-probe"),
            "active_probe_id": f"{scope}-probe-01",
        }
        selector["artifact_hash"] = _canonical_hash(selector)
        selector_ref = "selectors/probe.json"
        _write(root, selector_ref, selector)
        source = {
            "schema_version": direct_materializer.SOURCE_SCHEMA,
            "task_id": materializer.TASK_ID,
            "base_config_ref": base_ref,
            "stage2_authority_ref": stage2_ref,
            "config_overrides_ref": "overrides.json",
            "input_result_refs": [],
            "artifact_output_dir": f"runs/stage3/direct/{scope}",
            "authority_output_dir": f"authority/stage3/direct/{scope}",
            "cache_root": f"cache/stage3/direct/{scope}",
            "config_ref": "configs/direct.json",
            "result_ref": "results/direct.json",
            "evidence_refs": {
                "stage3_scope_decision": "authority/g30-decision.json",
                "formal_execution": "authority/formal-execution.json",
            },
            "external_gate_ref": None,
            "route_spec_ref": selector_ref,
        }
        receipt = direct_materializer.materialize(
            source, workspace_root=root, data_root=root
        )
        config = ResolvedConfigV2.from_mapping(
            load_canonical_json(root / source["config_ref"])
        )
        assert receipt["task_id"] == materializer.TASK_ID
        assert config.base_config.section("path_integration")["probe_count"] == (
            2 if scope == "pilot" else 3
        )
        assert config.base_config.section("data")["split"] == "probe"
        assert config.base_config.section("data")["sampler"] == "frozen-probe-panel"
        assert (
            config.base_config.section("data")["sampling_design"]
            == "disjoint_frozen_probe_panel"
        )
