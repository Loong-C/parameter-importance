from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.stage2.materialize_s204 import (
    EXPECTED_CELL_IDS,
    S204_S22_CANONICAL_OUTPUT_DIR,
    S204_S22_COMMIT_OUTPUT_DIR,
    S204_S22_CONFIG_REF,
    S204_S22_CONTROL_OUTPUT_DIR,
    S22_G3_FORMAL_EXECUTION_G20_REF,
    S22_G3_FORMAL_EXECUTION_G21_REF,
    S204MaterializationError,
    _formal_dag_config,
    _publish_resolved_config,
    _s22_g3_ready_manifest_hashes,
    _extend_formal_execution,
    _validate_formal_s22_task_group,
    _validate_delta_sci_artifact,
    ensure_formal_s22_task_outputs,
    publish_per_cell_delta_sci_plans,
)
from ops.stage2.prepare_s204_formal import _validate_adapter_gate
from param_importance_nlp.contracts import load_canonical_json, write_canonical_json
from param_importance_nlp.contracts import FormalExecutionEvidence
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)


def _s21_formula_refs(root: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for kind in ("preregistration", "hypothesis_contract"):
        payload = {
            "schema_version": f"stage2-{kind.replace('_', '-')}-v1",
            "scope": "formal",
            "equivalence_and_precision": {
                "scientific_margin_formula": "max(0.10*Delta_c_e(B),0.01*S_c_e)",
                "absolute_floors": {"tau_model": 1e-12, "tau_layer": 1e-12, "tau_module": 1e-12},
            },
        }
        refs[kind] = TaskArtifactStore(root, f"s21/{kind}").publish(
            task_id="stage2.01_scope_hypotheses_and_preregistration",
            artifact_kind=kind,
            config_hash="a" * 64,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
        ).commit_ref
    return refs


def test_cell_id_and_path_projection_are_reversible() -> None:
    from param_importance_nlp.experiments.stage2_s204_ids import (
        canonical_cell_id,
        cell_id_from_path_component,
        cell_path_component,
    )

    assert canonical_cell_id("pythia-14m", "early") == "pythia-14m:early"
    assert cell_path_component("pythia-14m:early") == "pythia-14m__early"
    assert cell_id_from_path_component("pythia-14m__early") == "pythia-14m:early"
    with pytest.raises(ValueError):
        cell_id_from_path_component("pythia-14m-early")


def test_pre_sizing_delta_plan_has_no_numeric_margin(tmp_path: Path) -> None:
    refs = publish_per_cell_delta_sci_plans(tmp_path, s21_refs=_s21_formula_refs(tmp_path))
    assert tuple(refs) == EXPECTED_CELL_IDS
    payload = load_canonical_json(tmp_path / refs[EXPECTED_CELL_IDS[0]])
    assert payload["schema_version"] == "stage2-reference-delta-sci-plan-v1"
    assert "delta_sci_by_B" not in payload
    with pytest.raises(S204MaterializationError, match="REFERENCE_DELTA_SCI_NUMERIC_REQUIRED"):
        _validate_delta_sci_artifact(
            tmp_path,
            refs[EXPECTED_CELL_IDS[0]],
            cell_id=EXPECTED_CELL_IDS[0],
        )


def test_missing_formal_g22_commit_blocks_prepare(tmp_path: Path) -> None:
    with pytest.raises(S204MaterializationError, match="S204_FORMAL_ADAPTER_COMMIT_REQUIRED"):
        _validate_adapter_gate(
            tmp_path,
            gate_id="stage2.G2.2",
            gate_ref="evidence/stage2/s204/formal-adapters/g2-2-r7/commits/gate_record.json",
            expected_refs=("evidence/stage2/s204/asset-resolution.json",),
        )


def test_s22_g3_resolution_must_be_a_formal_task_artifact(tmp_path: Path) -> None:
    import ops.stage2.materialize_s204 as materializer

    with pytest.raises(S204MaterializationError, match="S204_S22_G3_RESOLUTION_INVALID"):
        materializer._validate_s22_g3_resolution(
            tmp_path,
            "evidence/stage0/tasks/04-missing/commits/asset_resolution.json",
        )


def _fake_s22_g3_runtime(entries: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(resolution={"entries": entries})


def test_s22_g3_ready_manifest_hashes_bind_first_execution_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ops.stage2.materialize_s204 as materializer

    hashes = tuple(f"{index:064x}" for index in range(13))
    monkeypatch.setattr(
        materializer.FormalG3RuntimeAssets,
        "load",
        staticmethod(
            lambda _root, _ref: _fake_s22_g3_runtime(
                [{"ready_manifest_sha256": digest} for digest in hashes]
            )
        ),
    )
    observed = _s22_g3_ready_manifest_hashes(tmp_path, "evidence/stage0/g3.json")
    assert observed == hashes

    base = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash="a" * 64,
        asset_manifest_hashes=("b" * 64,),
        prerequisite_gates=(
            GateRecord(
                gate_id="stage0.G10",
                stage=0,
                status=GateStatus.PASS,
                checked_at="2026-08-24T00:00:00+00:00",
                evidence_refs=("evidence/stage0/g10.json",),
            ),
        ),
    )
    base_ref = "evidence/stage2/s204/formal-execution.json"
    output_ref = S22_G3_FORMAL_EXECUTION_G20_REF
    legacy_ref = f"{S204_S22_CONTROL_OUTPUT_DIR}/formal-execution-g20.json"
    legacy_payload = {"legacy": "preserved"}
    write_canonical_json(tmp_path / base_ref, base.to_dict())
    write_canonical_json(tmp_path / legacy_ref, legacy_payload)
    gate = GateRecord(
        gate_id="stage1.G1-EXIT",
        stage=1,
        status=GateStatus.PASS,
        checked_at="2026-08-24T00:00:00+00:00",
        evidence_refs=("evidence/stage1/g1.json",),
    )
    extended, _ = _extend_formal_execution(
        tmp_path,
        evidence_ref=base_ref,
        gate=gate,
        asset_hashes=observed,
        destination=output_ref,
    )
    assert set(hashes).issubset(extended.asset_manifest_hashes)
    assert load_canonical_json(tmp_path / legacy_ref) == legacy_payload


@pytest.mark.parametrize(
    "entries,match",
    (
        ([{"ready_manifest_sha256": f"{index:064x}"} for index in range(12)], "S22_G3_MANIFEST_HASHES_INVALID"),
        ([{"ready_manifest_sha256": "0" * 64} for _ in range(13)], "S22_G3_MANIFEST_HASHES_INVALID"),
        ([{"ready_manifest_sha256": "not-a-sha"}] + [{"ready_manifest_sha256": f"{index:064x}"} for index in range(1, 13)], "S22_G3_MANIFEST_HASHES_INVALID"),
    ),
)
def test_s22_g3_ready_manifest_hashes_reject_missing_duplicate_or_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entries: list[dict[str, object]],
    match: str,
) -> None:
    import ops.stage2.materialize_s204 as materializer

    monkeypatch.setattr(
        materializer.FormalG3RuntimeAssets,
        "load",
        staticmethod(lambda _root, _ref: _fake_s22_g3_runtime(entries)),
    )
    with pytest.raises(S204MaterializationError, match=match):
        _s22_g3_ready_manifest_hashes(tmp_path, "evidence/stage0/g3.json")


def _valid_s22_group(root: Path) -> tuple[dict[str, str], dict[str, object], str, str, tuple[str, ...]]:
    """Build only control-plane S2.2 commits for postcondition tests."""

    config = _formal_dag_config(
        root,
        base_config_ref="configs/run-ready/layers/formal-stage2-estimator.yaml",
        task_id="stage2.02_stage1_handoff_and_fixed_state_contract",
        input_refs=(),
        output_dir=S204_S22_COMMIT_OUTPUT_DIR,
    )
    config_ref = S204_S22_CONFIG_REF
    _publish_resolved_config(root, config, output_ref=config_ref)
    adapter_gate_refs: dict[str, str] = {}
    for gate_id, adapter_name in (("stage2.G2.0", "g2-0"), ("stage2.G2.1", "g2-1")):
        gate = GateRecord(
            gate_id=gate_id,
            stage=2,
            status=GateStatus.PASS,
            checked_at="2026-08-24T00:00:00+00:00",
            evidence_refs=("evidence/stage2/s201/commits/gate_record.json",),
        )
        adapter_gate_refs[gate_id] = TaskArtifactStore(
            root, f"evidence/stage2/formal-adapters/{adapter_name}"
        ).publish(
            task_id="stage2.02_stage1_handoff_and_fixed_state_contract",
            artifact_kind="gate_record",
            config_hash=config.config_hash,
            run_intent="formal",
            payload=gate.to_dict(),
            formal_eligible=True,
            source_refs=("evidence/stage2/s201/commits/gate_record.json",),
        ).commit_ref
    evidence_refs = {
        "formal_execution": S22_G3_FORMAL_EXECUTION_G21_REF,
        "stage0_handoff": "evidence/stage0/handoff.json",
        "stage1_g1_exit": "evidence/stage1/index.json",
        "g3_resolution": "evidence/stage0/g3.json",
        "gate_stage2_g2_0": adapter_gate_refs["stage2.G2.0"],
        "gate_stage2_g2_1": adapter_gate_refs["stage2.G2.1"],
    }
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset({"stage1.G1-EXIT", "stage2.G2.0", "stage2.G2.1"}),
        evidence_refs=evidence_refs,
    )
    environment_ref = f"{S204_S22_CONTROL_OUTPUT_DIR}/environments/stage2-02.json"
    write_canonical_json(root / environment_ref, environment.to_dict())
    lineage = tuple(dict.fromkeys((
        "evidence/stage2/s201/commits/preregistration.json",
        "evidence/stage2/s201/commits/hypothesis_contract.json",
        "evidence/stage2/s201/commits/gate_record.json",
        adapter_gate_refs["stage2.G2.0"],
        adapter_gate_refs["stage2.G2.1"],
        *evidence_refs.values(),
    )))
    store = TaskArtifactStore(root, S204_S22_COMMIT_OUTPUT_DIR)
    payloads = {
        "handoff_manifest": {
            "schema_version": "stage2-task-handoff-manifest-v1",
            "scope": "formal",
            "status": "FORMAL_CANDIDATE",
            "formal_eligible": False,
        },
        "fixed_state_contract": {
            "schema_version": "stage2-task-fixed-state-contract-v1",
            "scope": "formal",
            "status": "FORMAL_CANDIDATE",
            "formal_eligible": False,
        },
        "gate_record": {
            "schema_version": "stage23-task-gate-candidate-v1",
            "formal_eligible": False,
        },
    }
    refs: dict[str, str] = {}
    for kind, payload in payloads.items():
        refs[kind] = store.publish(
            task_id="stage2.02_stage1_handoff_and_fixed_state_contract",
            artifact_kind=kind,
            config_hash=config.config_hash,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
            source_refs=lineage,
        ).commit_ref
    return refs, evidence_refs, config_ref, environment_ref, lineage


def test_s22_absent_produces_and_complete_rerun_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ops.stage2.materialize_s204 as materializer

    monkeypatch.setattr(materializer, "_validate_s22_g3_resolution", lambda *_args: None)
    calls: list[str] = []
    refs, evidence_refs, config_ref, environment_ref, lineage = _valid_s22_group(tmp_path)

    def fake_producer(root: Path, **kwargs: object):
        calls.append(str(kwargs["output_dir"]))
        evidence = materializer.FormalExecutionEvidence(
            run_intent="formal",
            contract_freeze_hash="a" * 64,
            asset_manifest_hashes=("b" * 64,),
            prerequisite_gates=(GateRecord(
                gate_id="stage2.G2.1",
                stage=2,
                status=GateStatus.PASS,
                checked_at="2026-08-24T00:00:00+00:00",
                evidence_refs=("evidence/stage2/g2-1.json",),
            ),),
        )
        evidence_ref = S22_G3_FORMAL_EXECUTION_G21_REF
        write_canonical_json(root / evidence_ref, evidence.to_dict())
        return refs, evidence, evidence_ref, config_ref, environment_ref

    monkeypatch.setattr(materializer, "produce_formal_s22_task_outputs", fake_producer)
    predecessor: dict[str, dict[str, str]] = {}
    producer_kwargs = {"s21_refs": {kind: ref for kind, ref in zip(("preregistration", "hypothesis_contract", "gate_record"), lineage[:3])}, "g20_ref": lineage[3], "g21_ref": lineage[4]}
    producer_kwargs["g21_resolved_config_ref"] = S204_S22_CONFIG_REF
    first = ensure_formal_s22_task_outputs(tmp_path, predecessor_refs=predecessor, output_dir=S204_S22_CONTROL_OUTPUT_DIR, producer_kwargs=producer_kwargs)
    assert first[0] == refs and calls == [S204_S22_CONTROL_OUTPUT_DIR]
    second = ensure_formal_s22_task_outputs(tmp_path, predecessor_refs=predecessor, output_dir=S204_S22_CONTROL_OUTPUT_DIR, producer_kwargs=producer_kwargs)
    assert second[0] == refs and calls == [S204_S22_CONTROL_OUTPUT_DIR]


def test_s22_alternate_namespace_and_tampered_environment_or_lineage_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ops.stage2.materialize_s204 as materializer

    monkeypatch.setattr(materializer, "_validate_s22_g3_resolution", lambda *_args: None)
    refs, evidence_refs, config_ref, environment_ref, lineage = _valid_s22_group(tmp_path)
    assert "g3_resolution" in evidence_refs
    forbidden_markers = ("g2.2", "stage2.03", "stage2-03", "s2.3", "capability", "model_assets", "data_assets")
    for ref in refs.values():
        loaded = load_committed_task_artifact(tmp_path, ref, require_formal=True)
        assert not any(
            any(marker in source.casefold() for marker in forbidden_markers)
            for source in loaded.source_refs
        )
    with pytest.raises(S204MaterializationError, match="S204_S22_FORMAL_COMMIT_NAMESPACE_INVALID"):
        _validate_formal_s22_task_group(
            tmp_path,
            {**refs, "handoff_manifest": "alternate/commits/handoff_manifest.json"},
            config_ref=config_ref,
            environment_ref=environment_ref,
            required_lineage=lineage[:5],
        )
    tampered = load_canonical_json(tmp_path / environment_ref)
    assert isinstance(tampered, dict)
    tampered["passed_gate_ids"] = ["stage2.G2.0"]
    write_canonical_json(tmp_path / environment_ref, tampered)
    with pytest.raises(S204MaterializationError, match="S204_S22_ENVIRONMENT_INVALID"):
        _validate_formal_s22_task_group(
            tmp_path, refs, config_ref=config_ref, environment_ref=environment_ref, required_lineage=lineage[:5]
        )
    # G3 is an allowed Stage 0 provider input; runtime capabilities are not.
    forbidden_refs = {
        **evidence_refs,
        "g3_resolution": "evidence/stage0/g3.json",
    }
    forbidden_environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"model_assets"}),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset({"stage1.G1-EXIT", "stage2.G2.0", "stage2.G2.1"}),
        evidence_refs=forbidden_refs,
    )
    write_canonical_json(tmp_path / environment_ref, forbidden_environment.to_dict())
    with pytest.raises(S204MaterializationError, match="S204_S22_ENVIRONMENT_CAPABILITIES_FORBIDDEN"):
        _validate_formal_s22_task_group(
            tmp_path, refs, config_ref=config_ref, environment_ref=environment_ref, required_lineage=lineage[:5]
        )
    missing_g3_environment = TaskRuntimeEnvironment(
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset({"stage1.G1-EXIT", "stage2.G2.0", "stage2.G2.1"}),
        evidence_refs={key: value for key, value in evidence_refs.items() if key != "g3_resolution"},
    )
    write_canonical_json(tmp_path / environment_ref, missing_g3_environment.to_dict())
    with pytest.raises(S204MaterializationError, match="S204_S22_G3_RESOLUTION_REQUIRED"):
        _validate_formal_s22_task_group(
            tmp_path, refs, config_ref=config_ref, environment_ref=environment_ref, required_lineage=lineage[:5]
        )
    # Restore the immutable environment and prove that missing G2.1 lineage
    # is rejected even when all three filenames are present.
    environment = TaskRuntimeEnvironment(
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset({"stage1.G1-EXIT", "stage2.G2.0", "stage2.G2.1"}),
        evidence_refs=evidence_refs,
    )
    write_canonical_json(tmp_path / environment_ref, environment.to_dict())
    with pytest.raises(S204MaterializationError, match="S204_S22_LINEAGE_BINDING_MISMATCH"):
        _validate_formal_s22_task_group(
            tmp_path, refs, config_ref=config_ref, environment_ref=environment_ref, required_lineage=(*lineage[:4], "tampered/g21.json")
        )
    config = load_canonical_json(tmp_path / config_ref)
    assert isinstance(config, dict)
    config["task_id"] = "stage2.03_assets_checkpoints_and_sampling"
    write_canonical_json(tmp_path / config_ref, config)
    with pytest.raises(S204MaterializationError, match="S204_S22_RESOLVED_CONFIG_INVALID"):
        _validate_formal_s22_task_group(
            tmp_path, refs, config_ref=config_ref, environment_ref=environment_ref, required_lineage=lineage[:5]
        )
