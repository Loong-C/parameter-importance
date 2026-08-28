from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from param_importance_nlp.contracts import FormalExecutionEvidence, GateRecord, GateStatus
from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage3_trajectory import STAGE3_ENDPOINT_TASK_ID, Stage3TrajectoryReceipt
from param_importance_nlp.experiments.stage3_production_plan import build_production_unit_index
from ops.stage3.materialize_stage3_probe_plan import (
    ALLOCATION_SCHEMA,
    CONTENT_SOURCE_SCHEMA,
    MATERIALIZATION_SCHEMA,
    ProbePlanMaterializationError,
    materialize_probe_plans,
)
from tests.test_stage3_production_plan import _write_endpoint


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence(root: Path) -> tuple[str, FormalExecutionEvidence]:
    gate = GateRecord(
        "stage0.G10", 0, GateStatus.PASS, "2026-07-22T00:00:00Z",
        evidence_refs=("commits/stage0-g10.json",),
    )
    gates = [gate,
        GateRecord("stage3.G3-0", 3, GateStatus.PASS, "2026-07-22T00:00:00Z", evidence_refs=("gates/g30.json",)),
        GateRecord("stage3.G3-1", 3, GateStatus.PASS, "2026-07-22T00:00:00Z", evidence_refs=("gates/g31.json",)),
    ]
    evidence = FormalExecutionEvidence(
        "formal", contract_freeze_hash=_hash("contract"),
        asset_manifest_hashes=(_hash("asset"),), prerequisite_gates=tuple(gates),
    )
    ref = "inputs/formal-execution.json"
    write_canonical_json(root / ref, evidence.to_dict())
    return ref, evidence


def _source(
    tmp_path: Path, *, endpoint: Path, digest: str, scope: str = "pilot",
    overlap_update: bool = False,
) -> tuple[Path, Path, Path]:
    root = tmp_path
    evidence_ref, evidence = _evidence(root)
    endpoint_ref = endpoint.relative_to(root).as_posix()
    count = 3 if scope == "formal" else 2
    samples = [f"probe-sample-{index}" for index in range(count)]
    content_dir = root / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    sample_entries = []
    for index, sample_id in enumerate(samples):
        content_ref = f"content/{index}.bin"
        (root / content_ref).write_bytes(f"real-frozen-content-{index}".encode())
        sample_entries.append({"sample_id": sample_id, "content_ref": content_ref})
    state_ref = "inputs/resolver-state.json"
    loss_ref = "inputs/loss-contract.json"
    write_canonical_json(root / state_ref, {"resolver_id": "real-resolver/v1", "state": "frozen"})
    write_canonical_json(root / loss_ref, {"schema_version": "loss-contract-v1", "reduction": "target_token"})
    content_body = {
        "schema_version": CONTENT_SOURCE_SCHEMA,
        "resolver_id": "real-resolver/v1",
        "resolver_state_ref": state_ref,
        "samples": sample_entries,
        "loss_contract_ref": loss_ref,
        "effective_weight_unit": "target_token",
    }
    content_ref = "inputs/content-source.json"
    write_canonical_json(root / content_ref, content_body | {"artifact_hash": canonical_json_hash(content_body)})
    update_sample = "update:14M:0:early:1"
    allocation_body = {
        "schema_version": ALLOCATION_SCHEMA,
        "scope": scope,
        "allocations": [{
            "endpoint_commit_ref": endpoint_ref,
            "probes": [
                {"role": scope, "probe_id": f"real-probe-{index}", "sample_ids": [sample], "metadata": {"source": "real-frozen"}}
                for index, sample in enumerate(samples)
            ],
        }],
    }
    if overlap_update:
        allocation_body["allocations"][0]["probes"][0]["sample_ids"] = [update_sample]  # type: ignore[index]
    allocation_ref = "inputs/allocation.json"
    write_canonical_json(root / allocation_ref, allocation_body | {"artifact_hash": canonical_json_hash(allocation_body)})
    receipt = Stage3TrajectoryReceipt(
        receipt_id="real-trajectory-receipt", task_id=STAGE3_ENDPOINT_TASK_ID,
        config_hash=_hash("config"), purpose_scope=scope, formal_eligible=scope == "formal",
        capture_plan_ref="inputs/capture-plan.json", capture_plan_hash=_hash("capture"),
        training_run_id="real-run", selected_steps=(1,), endpoint_commit_refs=(endpoint_ref,),
        endpoint_digests=(digest,), replay_verified_steps=(1,), estimator_authority_ref="inputs/estimator.json",
        formal_execution_ref=evidence_ref,
        g30_scope_decision_ref=("inputs/g30-decision.json" if scope == "formal" else None),
        g30_gate_hash=(next(gate.artifact_hash for gate in evidence.prerequisite_gates if gate.gate_id == "stage3.G3-0") if scope == "formal" else None),
        g31_gate_hash=(next(gate.artifact_hash for gate in evidence.prerequisite_gates if gate.gate_id == "stage3.G3-1") if scope == "formal" else None),
    )
    receipt_ref = "inputs/trajectory-receipt.json"
    write_canonical_json(root / receipt_ref, receipt.to_dict())
    source_body = {
        "schema_version": MATERIALIZATION_SCHEMA,
        "scope": scope,
        "trajectory_receipt_ref": receipt_ref,
        "probe_allocation_ref": allocation_ref,
        "content_source_ref": content_ref,
        "formal_execution_ref": evidence_ref,
        "output_dir": "outputs/probes",
    }
    source_ref = "inputs/materialization-source.json"
    write_canonical_json(root / source_ref, source_body | {"artifact_hash": canonical_json_hash(source_body)})
    return root / source_ref, root / allocation_ref, root / content_ref


def _source_many(tmp_path: Path, endpoints: list[Path], digests: list[str]) -> Path:
    """Create a six-endpoint pilot source to exercise the production index."""
    evidence_ref, evidence = _evidence(tmp_path)
    samples: list[str] = []
    sample_entries: list[dict[str, object]] = []
    for index in range(len(endpoints) * 2):
        sample_id = f"pilot-probe-sample-{index}"
        samples.append(sample_id)
        content_ref = f"content/{index}.bin"
        (tmp_path / content_ref).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / content_ref).write_bytes(f"frozen-record-{index}".encode())
        sample_entries.append({"sample_id": sample_id, "content_ref": content_ref})
    state_ref = "inputs/resolver-state.json"
    loss_ref = "inputs/loss-contract.json"
    write_canonical_json(tmp_path / state_ref, {"resolver_id": "real-resolver/v1", "state": "frozen"})
    write_canonical_json(tmp_path / loss_ref, {"schema_version": "loss-contract-v1", "reduction": "target_token"})
    content_body = {
        "schema_version": CONTENT_SOURCE_SCHEMA, "resolver_id": "real-resolver/v1",
        "resolver_state_ref": state_ref, "samples": sample_entries,
        "loss_contract_ref": loss_ref, "effective_weight_unit": "target_token",
    }
    content_ref = "inputs/content-source.json"
    write_canonical_json(tmp_path / content_ref, content_body | {"artifact_hash": canonical_json_hash(content_body)})
    allocations: list[dict[str, object]] = []
    for endpoint_index, endpoint in enumerate(endpoints):
        ref = endpoint.relative_to(tmp_path).as_posix()
        allocations.append({
            "endpoint_commit_ref": ref,
            "probes": [
                {"role": "pilot", "probe_id": f"pilot-probe-{endpoint_index}-a", "sample_ids": [samples[endpoint_index * 2]], "metadata": {"source": "real-frozen"}},
                {"role": "pilot", "probe_id": f"pilot-probe-{endpoint_index}-b", "sample_ids": [samples[endpoint_index * 2 + 1]], "metadata": {"source": "real-frozen"}},
            ],
        })
    allocation_body = {"schema_version": ALLOCATION_SCHEMA, "scope": "pilot", "allocations": allocations}
    allocation_ref = "inputs/allocation.json"
    write_canonical_json(tmp_path / allocation_ref, allocation_body | {"artifact_hash": canonical_json_hash(allocation_body)})
    receipt = Stage3TrajectoryReceipt(
        receipt_id="real-pilot-trajectory-six-endpoints", task_id=STAGE3_ENDPOINT_TASK_ID,
        config_hash=_hash("config"), purpose_scope="pilot", formal_eligible=False,
        capture_plan_ref="inputs/capture-plan.json", capture_plan_hash=_hash("capture"),
        training_run_id="real-run", selected_steps=tuple(range(1, len(endpoints) + 1)),
        endpoint_commit_refs=tuple(endpoint.relative_to(tmp_path).as_posix() for endpoint in endpoints),
        endpoint_digests=tuple(digests), replay_verified_steps=tuple(range(1, len(endpoints) + 1)),
        estimator_authority_ref="inputs/estimator.json", formal_execution_ref=evidence_ref,
    )
    receipt_ref = "inputs/trajectory-receipt.json"
    write_canonical_json(tmp_path / receipt_ref, receipt.to_dict())
    source_body = {
        "schema_version": MATERIALIZATION_SCHEMA, "scope": "pilot",
        "trajectory_receipt_ref": receipt_ref, "probe_allocation_ref": allocation_ref,
        "content_source_ref": content_ref, "formal_execution_ref": evidence_ref,
        "output_dir": "outputs/probes",
    }
    source_ref = "inputs/materialization-source.json"
    write_canonical_json(tmp_path / source_ref, source_body | {"artifact_hash": canonical_json_hash(source_body)})
    return tmp_path / source_ref


def test_materializer_rehashes_real_content_and_publishes_two_probe_plans(tmp_path: Path) -> None:
    endpoint, digest = _write_endpoint(tmp_path / "run", model="14M", seed=0, stage="early", ordinal=1, formal=False)
    source, _, content_source = _source(tmp_path, endpoint=endpoint, digest=digest)
    paths = materialize_probe_plans(source, workspace_root=tmp_path, data_root=tmp_path)
    assert len(paths) == 1
    plan = load_canonical_json(paths[0])
    assert plan["schema_version"] == "stage3-probe-plan-v1"
    assert len(plan["entries"]) == 2
    assert plan["entries"][0]["content_hash"] == canonical_json_hash([
        {"sample_id": "probe-sample-0", "content_sha256": hashlib.sha256(b"real-frozen-content-0").hexdigest()}
    ])
    assert plan["entries"][0]["loss_contract_hash"] == canonical_json_hash({"schema_version": "loss-contract-v1", "reduction": "target_token"})
    content_source_value = load_canonical_json(content_source)
    assert plan["entries"][0]["metadata"]["materializer_content_source_hash"] == content_source_value["artifact_hash"]
    assert plan["entries"][0]["metadata"]["materializer_resolver_state_digest"] == canonical_json_hash(load_canonical_json(tmp_path / "inputs/resolver-state.json"))
    assert plan["artifact_hash"] == canonical_json_hash({key: value for key, value in plan.items() if key != "artifact_hash"})


def test_materializer_fails_closed_on_probe_update_overlap(tmp_path: Path) -> None:
    endpoint, digest = _write_endpoint(tmp_path / "run", model="14M", seed=0, stage="early", ordinal=1, formal=False)
    source, _, _ = _source(tmp_path, endpoint=endpoint, digest=digest, overlap_update=True)
    with pytest.raises(ProbePlanMaterializationError, match="PROBE_GLOBAL_OVERLAP"):
        materialize_probe_plans(source, workspace_root=tmp_path, data_root=tmp_path)


def test_materializer_supports_formal_three_probe_panels(tmp_path: Path) -> None:
    endpoint, digest = _write_endpoint(tmp_path / "run", model="14M", seed=0, stage="early", ordinal=1, formal=True)
    source, _, _ = _source(tmp_path, endpoint=endpoint, digest=digest, scope="formal")
    paths = materialize_probe_plans(source, workspace_root=tmp_path, data_root=tmp_path)
    plan = load_canonical_json(paths[0])
    assert plan["scope"] == "formal"
    assert plan["formal_eligible"] is True
    assert len(plan["entries"]) == 3
    assert all(entry["role"] == "formal" for entry in plan["entries"])


def test_materializer_outputs_exact_six_by_two_pilot_index(tmp_path: Path) -> None:
    specs = [("early", 1), ("early", 2), ("middle", 3), ("middle", 4), ("late", 5), ("late", 6)]
    endpoints: list[Path] = []
    digests: list[str] = []
    for stage, ordinal in specs:
        endpoint, digest = _write_endpoint(tmp_path / "run", model="14M", seed=0, stage=stage, ordinal=ordinal, formal=False)
        endpoints.append(endpoint)
        digests.append(digest)
    source = _source_many(tmp_path, endpoints, digests)
    materialize_probe_plans(source, workspace_root=tmp_path, data_root=tmp_path)
    index = build_production_unit_index(
        tmp_path / "run" / "endpoints" / "commits",
        tmp_path / "outputs" / "probes",
        scope="pilot", workspace_root=tmp_path,
    )
    assert index.endpoint_count == 6
    assert index.unit_count == 12


def test_materializer_rechecks_receipt_digest_and_rejects_allocation_hash_fields(tmp_path: Path) -> None:
    endpoint, digest = _write_endpoint(tmp_path / "run", model="14M", seed=0, stage="early", ordinal=1, formal=False)
    source, allocation, _ = _source(tmp_path, endpoint=endpoint, digest=digest)
    allocation_value = load_canonical_json(allocation)
    allocation_value["allocations"][0]["probes"][0]["content_hash"] = _hash("hand-filled")  # type: ignore[index]
    body = {key: value for key, value in allocation_value.items() if key != "artifact_hash"}
    write_canonical_json(allocation, body | {"artifact_hash": canonical_json_hash(body)})
    with pytest.raises(ProbePlanMaterializationError, match="ALLOCATION_PROBE_FIELDS_MISMATCH"):
        materialize_probe_plans(source, workspace_root=tmp_path, data_root=tmp_path)
