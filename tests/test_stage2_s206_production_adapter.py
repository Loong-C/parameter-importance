from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ops.stage2.run_s206_formal as launcher
from ops.stage2.run_s206_formal import (
    _derive_s206_config,
    _load_cost_source,
    _load_retry_policy,
    _production_lpt_jobs,
    _validate_s204_reference_candidate,
)
from param_importance_nlp.cli import _load_mapping
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    ResolvedConfig,
    canonical_json_hash,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.jsonio import write_canonical_json
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_s206_formal import (
    ANCHOR_IDS,
    APPROVED_GPU_UUIDS,
    GlobalPilotMappingManifest,
    build_global_pilot_mapping,
)
from param_importance_nlp.providers.synthetic import SyntheticGradientProvider
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.runtime.task_runtime import TaskRuntimeEnvironment
from param_importance_nlp.runtime.tensor_bundle import publish_tensor_bundle


ROOT = Path(__file__).resolve().parents[1]


def _s204_config() -> ResolvedConfigV2:
    base = ResolvedConfig.resolve(
        _load_mapping(ROOT / "configs/local-fixtures/resolved-config-v1.json"),
        _load_mapping(ROOT / "configs/run-ready/layers/formal-stage2-estimator.yaml"),
    )
    return ResolvedConfigV2.resolve(
        base,
        task_id="stage2.04_reference_target",
        overrides=_load_mapping(ROOT / "configs/run-ready/v2/stage2-reference-formal.yaml"),
    )


def _execution() -> FormalExecutionEvidence:
    gates = tuple(
        GateRecord(
            gate_id=gate_id,
            stage=2,
            status=GateStatus.PASS,
            checked_at="2026-08-25T00:00:00+00:00",
            measured={"status": "PASS"},
            threshold={"status": "PASS"},
            evidence_refs=(f"evidence/{gate_id}.json",),
        )
        for gate_id in ("stage2.G2.2", "stage2.G2.3", "stage2.G2.4a")
    )
    return FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="a" * 64,
        asset_manifest_hashes=("b" * 64,),
        prerequisite_gates=gates,
        metadata={"test": True},
    )
def test_production_config_retargets_task_without_dropping_s204_bindings() -> None:
    source = _s204_config()
    derived = _derive_s206_config(source)
    source_wire = source.to_dict()
    derived_wire = derived.to_dict()

    assert derived.task_id == "stage2.06_pilot_and_matrix_freeze"
    assert derived.section("execution")["runner_kind"] == "pilot"
    assert derived.section("artifacts")["required_kinds"] == [
        "pilot_report",
        "frozen_experiment_matrix",
        "gate_record",
    ]
    for name in source_wire:
        if name in {"task_id", "execution", "artifacts", "config_hash", "full_hash"}:
            continue
        assert derived_wire[name] == source_wire[name]
    assert derived.section("execution")["timeout_seconds"] == source.section("execution")["timeout_seconds"]
    assert derived.section("orchestration") == source.section("orchestration")
    assert derived.section("providers") == source.section("providers")


def test_candidate_reference_remains_unqualified_until_gates() -> None:
    candidate = {
        "schema_version": "reference-result-v1",
        "scope": "formal",
        "formal_eligible": False,
        "metadata": {"qualification_gate_hash": None},
    }
    convergence = {"schema_version": "stage2-reference-convergence-report-v1"}
    _validate_s204_reference_candidate(candidate, convergence, anchor_id="pythia-14m__initialization")

    qualified_payload = dict(candidate, formal_eligible=True)
    with pytest.raises(Exception, match="S204_REFERENCE_SCHEMA_INVALID"):
        _validate_s204_reference_candidate(
            qualified_payload,
            convergence,
            anchor_id="pythia-14m__initialization",
        )

    mutated_payload = dict(candidate, metadata={"qualification_gate_hash": "a" * 64})
    with pytest.raises(Exception, match="S204_REFERENCE_CANDIDATE_SEMANTICS_INVALID"):
        _validate_s204_reference_candidate(
            mutated_payload,
            convergence,
            anchor_id="pythia-14m__initialization",
        )


def test_cost_contract_allows_pending_s209_observations_without_fabricated_seconds(
    tmp_path: Path,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "stage2-s206-cost-semantics-contract-v1",
        "scope": "formal",
        "status": "FROZEN",
        "measurement_boundary": {
            "isolated_estimator_cost": "stage2.09.capacity",
            "online_training_incremental_cost": "stage2.09.capacity",
        },
        "scientific_equal_sample_cost": {
            "defined": True,
            "definition": "shared paired pilot gradient cost per equal sample",
            "measurement_status": "OBSERVED",
            "measurement_ref": "operations/s206/scientific-cost.json",
        },
        "isolated_estimator_cost": {
            "defined": True,
            "definition": "estimator-only incremental cost under isolated pilot conditions",
            "measurement_status": "PENDING_S2.9",
        },
        "online_training_incremental_cost": {
            "defined": True,
            "definition": "incremental cost over the frozen online training baseline",
            "measurement_status": "PENDING_S2.9",
        },
        "cost_io_quiescent": True,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    ref = "contracts/cost-semantics.json"
    write_canonical_json(tmp_path / ref, payload)
    result = _load_cost_source(
        SimpleNamespace(data_root=tmp_path, cost_semantics_ref=ref),
    )
    assert result["isolated_estimator_cost"]["measurement_status"] == "PENDING_S2.9"  # type: ignore[index]
    assert "seconds" not in result["online_training_incremental_cost"]  # type: ignore[operator]

    tampered = dict(payload, artifact_hash="b" * 64)
    write_canonical_json(tmp_path / "contracts/tampered.json", tampered)
    with pytest.raises(Exception, match="COST_SEMANTICS_HASH_MISMATCH"):
        _load_cost_source(
            SimpleNamespace(data_root=tmp_path, cost_semantics_ref="contracts/tampered.json"),
        )


def test_retry_policy_is_required_and_hash_bound_when_retries_enabled(tmp_path: Path) -> None:
    policy: dict[str, object] = {
        "schema_version": "stage2-s206-retry-policy-v1",
        "scope": "formal",
        "status": "FROZEN",
        "max_cell_attempts": 2,
        "reuse_mapping_on_retry": True,
        "new_pilot_draws_on_retry": False,
        "preserve_failure_records": True,
    }
    policy["artifact_hash"] = canonical_json_hash(policy)
    ref = "contracts/retry-policy.json"
    write_canonical_json(tmp_path / ref, policy)
    parsed, digest = _load_retry_policy(
        SimpleNamespace(data_root=tmp_path, retry_policy_ref=ref),
        2,
    )
    assert parsed["max_cell_attempts"] == 2
    assert digest == policy["artifact_hash"]
    with pytest.raises(Exception, match="RETRY_POLICY_REF_REQUIRED"):
        _load_retry_policy(SimpleNamespace(data_root=tmp_path, retry_policy_ref=None), 2)


def test_detached_wait_recovers_final_g24b_freeze(tmp_path: Path) -> None:
    write_canonical_json(
        tmp_path / "operations/s206/status.json",
        {
            "schema_version": "stage2-s206-formal-detached-status-v1",
            "stage": "G2.4B_PASS_MATRIX_FROZEN",
            "formal_eligible": True,
            "confirmatory_draws_generated": False,
        },
    )
    result = launcher._wait(
        SimpleNamespace(
            data_root=tmp_path,
            operations_root="operations/s206",
            timeout_seconds=0,
            poll_seconds=0.01,
        )
    )
    assert result == 0


def test_production_queue_is_deterministic_lpt_by_frozen_draw_work() -> None:
    jobs = _production_lpt_jobs()
    assert len(jobs) == len(ANCHOR_IDS) * 4
    assert set(jobs) == {
        (anchor_id, batch_size)
        for anchor_id in ANCHOR_IDS
        for batch_size in (32, 64, 128, 256)
    }
    # No measured 31M multiplier is available yet; all six B=256 cells must
    # precede every smaller work unit, with anchor order as deterministic tie
    # breaker.  This is the LPT queue consumed by the dynamic four-GPU loop.
    assert jobs[: len(ANCHOR_IDS)] == tuple((anchor_id, 256) for anchor_id in ANCHOR_IDS)
    assert [50 * batch_size for _anchor_id, batch_size in jobs] == sorted(
        (50 * batch_size for _anchor_id, batch_size in jobs),
        reverse=True,
    )


def test_production_cell_uses_candidate_bundle_and_emits_blinded_pilot_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production-cell bridge with a fake fixed-state factory.

    The provider is synthetic only as a test double; the launcher path,
    candidate envelope, directory tensor bundle, S2.5 rows, strict upstream
    preflight and real ``run_formal_pilot_cell``/paired runner are exercised.
    """

    root = tmp_path
    component_ids = [anchor_id.replace(".", "__") for anchor_id in ANCHOR_IDS]
    s204_root = root / "evidence" / "s204"
    reference_refs: dict[str, str] = {}
    convergence_refs: dict[str, str] = {}
    for component, anchor_id in zip(component_ids, ANCHOR_IDS):
        write_canonical_json(
            s204_root / component / "final-status.json",
            {
                "status": "COMPLETE",
                "formal_eligible": True,
                "artifact_refs": {
                    "reference_result": f"artifacts/s204/{component}/reference_result.json",
                    "reference_convergence_report": f"artifacts/s204/{component}/convergence.json",
                    "gate_record": f"artifacts/s204/{component}/gate.json",
                },
            },
        )

    g23 = {
        "schema_version": "stage2-g23-reference-evaluation-v1",
        "status": "PASS",
        "formal_eligible": True,
        "required_cell_count": 6,
        "complete_cell_count": 6,
        "cells": [
            {"cell_id": anchor_id, "status": "PASS", "formal_eligible": True}
            for anchor_id in ANCHOR_IDS
        ],
    }
    g23["artifact_hash"] = canonical_json_hash(g23)
    write_canonical_json(root / "evidence/g23.json", g23)
    g24a = {
        "schema_version": "stage2-g24a-formal-evaluation-v1",
        "gate_id": "stage2.G2.4a",
        "status": "PASS",
        "formal_eligible": True,
        "cell_count": 6,
        "results": [
            {
                "cell_id": anchor_id,
                "status": "PASS",
                "formal_eligible": True,
                "metrics": {
                    "h_ref_model_total": 0.01,
                    "h_ref_layer": 0.01,
                    "h_ref_module": 0.01,
                },
            }
            for anchor_id in ANCHOR_IDS
        ],
    }
    g24a["artifact_hash"] = canonical_json_hash(g24a)
    write_canonical_json(root / "evidence/g24a.json", g24a)
    inventory = [
        {"uuid": uuid, "pci_bus_id": f"0000:{index + 53:02X}:00.0"}
        for index, uuid in enumerate(APPROVED_GPU_UUIDS)
    ]
    write_canonical_json(root / "evidence/gpu-inventory.json", inventory)

    source_config = _s204_config()
    config_ref = "prepared/config.json"
    write_canonical_json(root / config_ref, source_config.to_dict())
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"server", "cuda"}),
        frozen_contract_stages=frozenset({2}),
        passed_gate_ids=frozenset({"stage2.G2.2"}),
        evidence_refs={
            "stage0_handoff": "evidence/stage0-handoff.json",
            "stage1_g1_exit": "evidence/stage1-exit.json",
            "g3_resolution": "evidence/g3-resolution.json",
            "contract_freeze": "evidence/contract-freeze.json",
            "stage2_parameter_registry": "evidence/parameter-registry.json",
        },
    )
    environment_ref = "prepared/environment.json"
    write_canonical_json(root / environment_ref, environment.to_dict())
    execution = _execution()
    execution_ref = "evidence/formal-execution.json"
    write_canonical_json(root / execution_ref, execution.to_dict())

    sampling = SamplingPlan(
        SamplingUniverse("s206-production-cell-test", tuple(range(1024))),
        {
            "reference_sizing": 11,
            "reference_A": 22,
            "reference_B": 33,
            "pilot": 44,
            "confirmatory": 55,
        },
    )
    sampling_ref = "prepared/sampling.json"
    write_canonical_json(root / sampling_ref, sampling.to_dict())
    mapping = build_global_pilot_mapping(sampling)
    mapping_ref = "prepared/pilot-mapping.json"
    write_canonical_json(root / mapping_ref, mapping.to_dict())

    provider = SyntheticGradientProvider.from_location_scale(
        parameter_shapes={"p": (1,)},
        sample_count=1024,
        seed=19,
    )
    reference = provider.gradient(mapping.cells[0].mappings[0].draws).gradients
    from param_importance_nlp.experiments.stage2_formal import _vector_digest

    reference_arrays = {
        "bias_reference": reference,
        "cross_reference": reference,
        "ranking_reference": reference,
        "sequence_variance": {name: np.square(value) for name, value in reference.items()},
    }
    bundle = publish_tensor_bundle(
        root / "runs/formal/stage2-reference/tensor-bundles/reference-final",
        reference_arrays,
    )
    candidate_payload: dict[str, object] = {
        "schema_version": "reference-result-v1",
        "reference_id": "s206-production-cell-candidate",
        "bias_reference_hash": _vector_digest(reference_arrays["bias_reference"]),
        "cross_reference_hash": _vector_digest(reference_arrays["cross_reference"]),
        "ranking_reference_hash": _vector_digest(reference_arrays["ranking_reference"]),
        "sample_count_a": 4,
        "sample_count_b": 4,
        "block_size": 1,
        "registry_hash": provider.registry_hash,
        "scope": "formal",
        "formal_eligible": False,
        "metadata": {
            "sequence_variance_hash": _vector_digest(reference_arrays["sequence_variance"]),
        },
        "tensor_bundle_ref": "runs/formal/stage2-reference/tensor-bundles/reference-final",
        "tensor_bundle_manifest_hash": bundle.manifest_sha256,
    }
    candidate_payload["artifact_hash"] = canonical_json_hash(candidate_payload)

    sizing_body: dict[str, object] = {
        "schema_version": "stage2-reference-delta-sci-v2",
        "delta_sci_by_endpoint": {
            endpoint: {str(batch_size): 0.1 for batch_size in (32, 64, 128, 256)}
            for endpoint in ("model_total", "layer", "module")
        },
    }
    sizing_body["artifact_hash"] = canonical_json_hash(sizing_body)
    sizing_ref = "prepared/delta-sci.json"
    write_canonical_json(root / sizing_ref, sizing_body)
    sizing = dict(sizing_body)
    sizing.update(
        {
            "source_ref": sizing_ref,
            "source_hash": sizing_body["artifact_hash"],
            "source_artifact_hash": sizing_body["artifact_hash"],
        }
    )
    convergence_payload = {
        "schema_version": "stage2-reference-convergence-report-v1",
        "candidate_delta_sci": sizing,
        "candidate_delta_sci_source": sizing_ref,
        "candidate_delta_sci_source_hash": sizing_body["artifact_hash"],
    }

    for component, anchor_id in zip(component_ids, ANCHOR_IDS):
        store = TaskArtifactStore(root, f"artifacts/s204/{component}")
        reference_commit = store.publish(
            task_id="stage2.04_reference_target",
            artifact_kind="reference_result",
            config_hash=source_config.config_hash,
            run_intent="formal",
            payload=candidate_payload,
            formal_eligible=True,
            source_refs=("prepared/reference-source.json",),
        )
        convergence_commit = store.publish(
            task_id="stage2.04_reference_target",
            artifact_kind="reference_convergence_report",
            config_hash=source_config.config_hash,
            run_intent="formal",
            payload=convergence_payload,
            formal_eligible=True,
            source_refs=(sizing_ref,),
        )
        reference_refs[anchor_id] = reference_commit.commit_ref
        convergence_refs[anchor_id] = convergence_commit.commit_ref

    rows = []
    for anchor_id in ANCHOR_IDS:
        component = anchor_id.replace(".", "__")
        rows.append(
            {
                "cell_id": anchor_id,
                "config_ref": config_ref,
                "environment_ref": environment_ref,
                "formal_execution_ref": execution_ref,
                "reference_artifact_refs": {
                    "reference_result": reference_refs[anchor_id],
                    "reference_convergence_report": convergence_refs[anchor_id],
                },
                "config_hash": source_config.config_hash,
                "component": component,
            }
        )
    rebind: dict[str, object] = {
        "schema_version": "stage2-s205-rebind-plan-v1",
        "status": "READY",
        "formal_eligible": True,
        "cells": rows,
    }
    rebind["artifact_hash"] = canonical_json_hash(rebind)
    rebind_ref = "prepared/s205-rebind.json"
    write_canonical_json(root / rebind_ref, rebind)

    import param_importance_nlp.experiments.stage23_task_runners as stage23_runners

    monkeypatch.setattr(
        stage23_runners,
        "_formal_provider",
        lambda request, workspace_root: SimpleNamespace(provider=provider, evidence=execution),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", APPROVED_GPU_UUIDS[0])
    args = SimpleNamespace(
        cell_anchor=ANCHOR_IDS[0],
        cell_batch_size=32,
        cell_gpu_uuid=APPROVED_GPU_UUIDS[0],
        pilot_mapping_ref=mapping_ref,
        s205_rebind_ref=rebind_ref,
        cell_artifact_root="operations/s206/cell-artifacts",
        cell_output="operations/s206/cell.json",
        execution_evidence_ref=execution_ref,
        data_root=root,
        s204_root="evidence/s204",
        g23_evaluation="evidence/g23.json",
        g24a_evaluation="evidence/g24a.json",
        gpu_inventory_json=root / "evidence/gpu-inventory.json",
        repository=ROOT,
        resource_within_budget=True,
        cost_io_quiescent=True,
    )
    result = launcher._production_cell(args)
    assert result["status"] == "COMPLETE"
    cell_payload = result["cell"]
    assert isinstance(cell_payload, dict)
    assert cell_payload["scientific_values_masked"] is True
    assert cell_payload["formal_eligible"] is False
    assert GlobalPilotMappingManifest.from_mapping(mapping.to_dict()).to_dict()[
        "confirmatory_draws_generated"
    ] is False
    assert not (root / "operations/s206/confirmatory-mapping.json").exists()

    # The bounded server smoke reuses the same production provider/input
    # bridge, but commits only one frozen 14M/B32 repetition and cannot enter
    # the reducer or confirmatory stream.
    smoke_args = SimpleNamespace(**vars(args))
    smoke_args.cell_artifact_root = "operations/s206/smoke-artifacts"
    smoke_args.cell_output = "operations/s206/smoke.json"
    smoke = launcher._production_smoke_cell(smoke_args)
    assert smoke["status"] == "SMOKE_COMPLETE"
    smoke_payload = smoke["smoke"]
    assert isinstance(smoke_payload, dict)
    assert smoke_payload["anchor_id"] == ANCHOR_IDS[0]
    assert smoke_payload["repetitions_requested"] == 1
    assert smoke_payload["completed_repetitions"] == 1
    assert smoke_payload["scientific_values_masked"] is True
    assert smoke_payload["confirmatory_draws_generated"] is False
    assert not (root / "operations/s206/confirmatory-mapping.json").exists()
