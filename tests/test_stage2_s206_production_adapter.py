from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys

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
from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_s206_inputs import build_formal_execution_evidence
from param_importance_nlp.experiments.stage2_executor_identity import compute_executor_identity
from param_importance_nlp.experiments.stage2_s206_formal import (
    ANCHOR_IDS,
    APPROVED_GPU_BINDINGS,
    APPROVED_GPU_UUIDS,
    EXCLUDED_PCI,
    EXCLUDED_UUID,
    GPU_INVENTORY_SCHEMA,
    GlobalPilotMappingManifest,
    S206PreparationBlocked,
    build_global_pilot_mapping,
)
from param_importance_nlp.providers.synthetic import SyntheticGradientProvider
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.runtime.task_runtime import TaskRuntimeEnvironment
from param_importance_nlp.runtime.tensor_bundle import publish_tensor_bundle


ROOT = Path(__file__).resolve().parents[1]


def _write_gpu_inventory(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for pci, uuid in APPROVED_GPU_BINDINGS:
        rows.append(
            {
                "uuid": uuid,
                "pci_bus_id": pci,
                "gpu_name": "A100-SXM4-80GB",
                "temperature_c": 42,
                "memory_used_mib": 0,
                "memory_total_mib": 81920,
                "utilization_gpu_percent": 0,
                "compute_mode": "Default",
                "ecc_uncorrected_volatile": 0,
                "ecc_uncorrected_aggregate": 0,
                "row_remap_failure": 0,
                "row_remap_pending": 0,
                "row_remap_status": "None",
                "gpu_recovery_action": "None",
                "health_state": "HEALTHY",
                "compute_apps": [],
            }
        )
    rows.append(
        {
            "uuid": EXCLUDED_UUID,
            "pci_bus_id": EXCLUDED_PCI,
            "gpu_name": "A100-SXM4-80GB",
            "temperature_c": 55,
            "memory_used_mib": 0,
            "memory_total_mib": 81920,
            "utilization_gpu_percent": 0,
            "compute_mode": "Default",
            "ecc_uncorrected_volatile": 113,
            "ecc_uncorrected_aggregate": 179,
            "row_remap_failure": 0,
            "row_remap_pending": 1,
            "row_remap_status": "pending",
            "gpu_recovery_action": "None",
            "health_state": "UNHEALTHY",
            "compute_apps": [],
        }
    )
    for index, pci in enumerate(("0000:4F:00.0", "0000:51:00.0", "0000:57:00.0")):
        rows.append(
            {
                "uuid": f"GPU-test-extra-{index}",
                "pci_bus_id": pci,
                "gpu_name": "A100-SXM4-80GB",
                "temperature_c": 40,
                "memory_used_mib": 0,
                "memory_total_mib": 81920,
                "utilization_gpu_percent": 0,
                "compute_mode": "Default",
                "ecc_uncorrected_volatile": 0,
                "ecc_uncorrected_aggregate": 0,
                "row_remap_failure": 0,
                "row_remap_pending": 0,
                "row_remap_status": "None",
                "gpu_recovery_action": "None",
                "health_state": "HEALTHY",
                "compute_apps": [],
            }
        )
    data_root = path.parent.parent
    source_path = path.parent / "gpu-inventory.capture.txt"
    source_bytes = b"test live GPU capture\n"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_bytes)
    payload: dict[str, object] = {
        "schema_version": GPU_INVENTORY_SCHEMA,
        "scope": "formal",
        "status": "OBSERVED",
        "checked_at": "2026-08-26T00:00:00+00:00",
        "artifact_ref": path.relative_to(data_root).as_posix(),
        "source_ref": source_path.relative_to(data_root).as_posix(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rows": rows,
        "compute_apps": [],
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_pci": EXCLUDED_PCI,
        "excluded_gpu_uuid": EXCLUDED_UUID,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    write_canonical_json(path, payload)


def _write_hashed(root: Path, ref: str, payload: dict[str, object]) -> str:
    value = dict(payload)
    value["artifact_hash"] = canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    )
    write_canonical_json(root / ref, value)
    return str(value["artifact_hash"])


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
    inventory_path = tmp_path / "evidence/gpu-inventory.json"
    _write_gpu_inventory(inventory_path)
    _rows, identity = launcher._load_inventory_snapshot(inventory_path, data_root=tmp_path)
    g23_ref = "evidence/g23.json"
    g23_hash = _write_hashed(
        tmp_path,
        g23_ref,
        {
            "schema_version": "stage2-g23-reference-evaluation-v1",
            "status": "PASS",
            "formal_eligible": True,
            "required_cell_count": 6,
            "complete_cell_count": 6,
            "cells": [
                {"cell_id": anchor.replace(".", ":", 1), "status": "PASS", "formal_eligible": True}
                for anchor in ANCHOR_IDS
            ],
        },
    )
    g24a_ref = "evidence/g24a.json"
    g24a_hash = _write_hashed(
        tmp_path,
        g24a_ref,
        {
            "schema_version": "stage2-g24a-formal-evaluation-v1",
            "gate_id": "stage2.G2.4a",
            "status": "PASS",
            "formal_eligible": True,
            "cell_count": 6,
            "g23_evaluation_ref": g23_ref,
            "g23_evaluation_hash": g23_hash,
            "results": [
                {
                    "cell_id": anchor.replace(".", ":", 1),
                    "status": "PASS",
                    "formal_eligible": True,
                    "metrics": {"h_ref_model_total": 0.01},
                }
                for anchor in ANCHOR_IDS
            ],
        },
    )
    s205_ref = "evidence/s205-rebind.json"
    s205_hash = _write_hashed(
        tmp_path,
        s205_ref,
        {
            "schema_version": "stage2-s205-rebind-plan-v1",
            "status": "READY",
            "formal_eligible": True,
            "g23_evaluation_ref": g23_ref,
            "g23_evaluation_hash": g23_hash,
            "cells": [
                {
                    "cell_id": anchor.replace(".", ":", 1),
                    "config_ref": f"prepared/{index}.json",
                    "environment_ref": f"prepared/{index}-env.json",
                    "formal_execution_ref": "evidence/execution.json",
                    "reference_artifact_refs": {},
                    "task_result_status_path": f"prepared/{index}-status.json",
                    "config_hash": "a" * 64,
                    "result_hash": "b" * 64,
                }
                for index, anchor in enumerate(ANCHOR_IDS)
            ],
        },
    )
    execution = build_formal_execution_evidence(
        data_root=tmp_path,
        g23_evaluation_ref=g23_ref,
        g24a_evaluation_ref=g24a_ref,
        s205_rebind_ref=s205_ref,
        contract_freeze_hash="c" * 64,
        asset_manifest_hashes=("d" * 64,),
    )
    execution_ref = "evidence/execution.json"
    write_canonical_json(tmp_path / execution_ref, execution)
    executor_identity = compute_executor_identity(
        ROOT,
        ROOT / "ops/stage2/run_s206_formal.py",
    )
    inventory_identity = {
        "source_ref": identity["source_ref"],
        "artifact_ref": identity["artifact_ref"],
        "artifact_hash": identity["artifact_hash"],
        "source_sha256": identity["source_sha256"],
        "schema_version": identity["schema_version"],
    }
    preflight = {
        "status": "READY",
        "gpu_inventory_identity": inventory_identity,
        "executor_identity": executor_identity,
    }
    preflight["preflight_artifact_hash"] = canonical_json_hash(preflight)
    status_payload: dict[str, object] = {
        "schema_version": "stage2-s206-formal-detached-status-v1",
        "stage": "G2.4B_PASS_MATRIX_FROZEN",
        "formal_eligible": True,
        "confirmatory_draws_generated": False,
        "preflight": preflight,
        "preflight_artifact_hash": preflight["preflight_artifact_hash"],
        "gpu_inventory_path": identity["artifact_ref"],
        "gpu_inventory_ref": identity["artifact_ref"],
        "gpu_inventory_source_ref": identity["source_ref"],
        "gpu_inventory_artifact_hash": identity["artifact_hash"],
        "gpu_inventory_source_sha256": identity["source_sha256"],
        "gpu_inventory_identity": inventory_identity,
        "executor_identity": executor_identity,
        "execution_evidence_ref": execution_ref,
        "execution_evidence_hash": execution["artifact_hash"],
    }
    status_payload["artifact_hash"] = canonical_json_hash(status_payload)
    write_canonical_json(
        tmp_path / "operations/s206/status.json",
        status_payload,
    )
    result = launcher._wait(
        SimpleNamespace(
            data_root=tmp_path,
            operations_root="operations/s206",
            timeout_seconds=0,
            poll_seconds=0.01,
            # An accepted logical alias must be normalized in the status
            # comparison; receipts themselves remain canonical POSIX refs.
            execution_evidence_ref="evidence//execution.json",
            g23_evaluation=g23_ref,
            g24a_evaluation=g24a_ref,
            s205_rebind_ref=s205_ref,
        )
    )
    assert result == 0

    # A formal terminal status carrying an execution hash must also carry the
    # exact amendment ref; wait/recovery cannot silently fall back to the old
    # inventory-only identity check.
    status_path = tmp_path / "operations/s206/status.json"
    status_value = load_canonical_json(status_path)
    assert isinstance(status_value, dict)
    status_value["formal_eligible"] = False
    # Keep the old hash: wait must reject the stage tamper before displaying it.
    write_canonical_json(status_path, status_value)
    with pytest.raises(S206PreparationBlocked, match="STATUS_ARTIFACT_HASH_MISMATCH"):
        launcher._wait(
            SimpleNamespace(
                data_root=tmp_path,
                operations_root="operations/s206",
                timeout_seconds=0,
                poll_seconds=0.01,
            )
        )
    status_value = dict(status_payload)
    status_value.pop("artifact_hash", None)
    status_value.pop("execution_evidence_ref", None)
    status_value["artifact_hash"] = canonical_json_hash(status_value)
    write_canonical_json(status_path, status_value)
    with pytest.raises(S206PreparationBlocked, match="STATUS_EXECUTION_EVIDENCE_IDENTITY_MISSING"):
        launcher._wait(
            SimpleNamespace(
                data_root=tmp_path,
                operations_root="operations/s206",
                timeout_seconds=0,
                poll_seconds=0.01,
                execution_evidence_ref=execution_ref,
                g23_evaluation=g23_ref,
                g24a_evaluation=g24a_ref,
                s205_rebind_ref=s205_ref,
            )
        )
    assert launcher.main(
        [
            "--wait",
            "--data-root",
            str(tmp_path),
            "--s204-root",
            "evidence/s204",
            "--g23-evaluation",
            g23_ref,
            "--g24a-evaluation",
            g24a_ref,
            "--operations-root",
            "operations/s206",
        ]
    ) == 3


def _detached_test_args(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, object], FormalExecutionEvidence]:
    inventory_ref = "evidence/gpu-inventory.json"
    preflight = {
        "gpu_inventory_ref": inventory_ref,
        "gpu_inventory_source_ref": "evidence/gpu-inventory.capture.txt",
        "gpu_inventory_artifact_hash": "a" * 64,
        "gpu_inventory_source_sha256": "b" * 64,
        "preflight_artifact_hash": "c" * 64,
        "gpu": {"inventory_path": inventory_ref},
    }
    args = SimpleNamespace(
        execute=True,
        data_root=tmp_path,
        operations_root="operations/s206",
        repository=ROOT,
        python=Path(sys.executable),
        run_id="detach-test",
        execution_evidence_ref="evidence/execution.json",
        g23_evaluation="evidence/g23.json",
        g24a_evaluation="evidence/g24a.json",
        s205_rebind_ref="evidence/s205-rebind.json",
        s204_root="evidence/s204",
    )
    return args, preflight, _execution()


def test_detach_uses_append_only_attempts_and_blocks_live_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, preflight, execution = _detached_test_args(tmp_path)
    monkeypatch.setattr(launcher, "_preflight", lambda _args: preflight)
    monkeypatch.setattr(launcher, "_load_formal_execution", lambda _args: execution)
    monkeypatch.setattr(launcher.sys, "argv", ["run_s206_formal.py", "--execute", "--detach"])
    pids = iter((50101, 50102, 50103))

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = next(pids)

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(launcher.os, "kill", lambda *_args, **_kwargs: None)
    assert launcher._detach(args) == 0
    attempts = tmp_path / "operations/s206/attempts"
    receipts = sorted(attempts.glob("*/launcher.pid.json"))
    assert len(receipts) == 1
    assert not (tmp_path / "operations/s206/launcher.pid.json").exists()
    first = load_canonical_json(receipts[0])
    assert isinstance(first, dict)
    assert first["artifact_hash"] == canonical_json_hash(
        {key: value for key, value in first.items() if key != "artifact_hash"}
    )
    for field in (
        "attempt_ref",
        "operations_root",
        "log_ref",
        "status_ref",
        "gpu_inventory_ref",
        "gpu_inventory_source_ref",
        "gpu_inventory_path",
        "execution_evidence_ref",
    ):
        ref = first[field]
        assert isinstance(ref, str) and not Path(ref).is_absolute() and "\\" not in ref

    # A valid stale receipt is retained and does not get overwritten.
    monkeypatch.setattr(launcher.os, "kill", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    assert launcher._detach(args) == 0
    receipts = sorted(attempts.glob("*/launcher.pid.json"))
    assert len(receipts) == 2
    assert receipts[0] != receipts[1]

    # A live PID in any retained attempt blocks another detached launch.
    monkeypatch.setattr(launcher.os, "kill", lambda *_args, **_kwargs: None)
    with pytest.raises(S206PreparationBlocked, match="DETACHED_LAUNCH_ALREADY_RUNNING"):
        launcher._detach(args)

    # Receipt tamper is rejected before PID liveness is consulted.
    tampered = load_canonical_json(receipts[1])
    assert isinstance(tampered, dict)
    tampered["pid"] = 50199
    write_canonical_json(receipts[1], tampered)
    monkeypatch.setattr(launcher.os, "kill", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    with pytest.raises(S206PreparationBlocked, match="DETACHED_ATTEMPT_RECEIPT_HASH_MISMATCH"):
        launcher._detach(args)


def test_detach_spawn_failure_is_durable_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, preflight, execution = _detached_test_args(tmp_path)
    monkeypatch.setattr(launcher, "_preflight", lambda _args: preflight)
    monkeypatch.setattr(launcher, "_load_formal_execution", lambda _args: execution)
    monkeypatch.setattr(launcher.sys, "argv", ["run_s206_formal.py", "--execute", "--detach"])

    def fail_spawn(*_args: object, **_kwargs: object) -> object:
        raise OSError("spawn refused")

    monkeypatch.setattr(launcher.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="spawn refused"):
        launcher._detach(args)

    attempts = tmp_path / "operations/s206/attempts"
    failures = sorted(attempts.glob("*/launcher.failure.json"))
    assert len(failures) == 1
    assert not list(attempts.glob("*/launcher.pid.json"))
    failure = load_canonical_json(failures[0])
    assert isinstance(failure, dict)
    assert failure["status"] == "SPAWN_FAILED"
    assert failure["pid"] is None
    assert failure["artifact_hash"] == canonical_json_hash(
        {key: value for key, value in failure.items() if key != "artifact_hash"}
    )

    class FakeProcess:
        pid = 50201

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    assert launcher._detach(args) == 0
    assert len(list(attempts.glob("*/launcher.failure.json"))) == 1
    assert len(list(attempts.glob("*/launcher.pid.json"))) == 1


def test_detach_receipt_write_failure_terminates_waits_and_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, preflight, execution = _detached_test_args(tmp_path)
    monkeypatch.setattr(launcher, "_preflight", lambda _args: preflight)
    monkeypatch.setattr(launcher, "_load_formal_execution", lambda _args: execution)
    monkeypatch.setattr(launcher.sys, "argv", ["run_s206_formal.py", "--execute", "--detach"])
    processes: list[object] = []

    class FakeProcess:
        pid = 50202

        def __init__(self) -> None:
            self.terminated = False
            self.waited = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, *, timeout: float) -> int:
            assert timeout == 10
            self.waited = True
            return 0

    def spawn(*_args: object, **_kwargs: object) -> FakeProcess:
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    real_write = launcher.write_canonical_json

    def fail_launch_receipt(path: Path, payload: object) -> Path:
        if Path(path).name == "launcher.pid.json":
            raise OSError("receipt storage refused")
        return real_write(path, payload)

    monkeypatch.setattr(launcher, "write_canonical_json", fail_launch_receipt)
    with pytest.raises(OSError, match="receipt storage refused"):
        launcher._detach(args)
    assert len(processes) == 1
    assert processes[0].terminated is True  # type: ignore[union-attr]
    assert processes[0].waited is True  # type: ignore[union-attr]

    attempts = tmp_path / "operations/s206/attempts"
    failures = sorted(attempts.glob("*/launcher.failure.json"))
    assert len(failures) == 1
    failure = load_canonical_json(failures[0])
    assert isinstance(failure, dict)
    assert failure["status"] == "RECEIPT_WRITE_FAILED"
    assert failure["pid"] == 50202
    assert failure["cleanup_error"] is None
    assert failure["artifact_hash"] == canonical_json_hash(
        {key: value for key, value in failure.items() if key != "artifact_hash"}
    )

    # A waited/stale failed attempt is retained and does not block the retry.
    monkeypatch.setattr(launcher, "write_canonical_json", real_write)
    monkeypatch.setattr(launcher.os, "kill", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    class RetryProcess:
        pid = 50203

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: RetryProcess())
    assert launcher._detach(args) == 0
    assert len(list(attempts.glob("*/launcher.failure.json"))) == 1
    assert len(list(attempts.glob("*/launcher.pid.json"))) == 1


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("log_ref", "operations/s206/attempts/replaced/other.log", "LOG_REF_MISMATCH"),
        ("status_ref", "operations/s206/replaced-status.json", "STATUS_REF_MISMATCH"),
        ("gpu_inventory_path", "evidence/replaced-inventory.json", "GPU_INVENTORY_PATH_MISMATCH"),
    ],
)
def test_detach_rejects_hash_valid_receipt_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    pattern: str,
) -> None:
    args, preflight, execution = _detached_test_args(tmp_path)
    monkeypatch.setattr(launcher, "_preflight", lambda _args: preflight)
    monkeypatch.setattr(launcher, "_load_formal_execution", lambda _args: execution)
    monkeypatch.setattr(launcher.sys, "argv", ["run_s206_formal.py", "--execute", "--detach"])

    class FakeProcess:
        pid = 50204

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    assert launcher._detach(args) == 0
    receipt_path = next((tmp_path / "operations/s206/attempts").glob("*/launcher.pid.json"))
    receipt = load_canonical_json(receipt_path)
    assert isinstance(receipt, dict)
    receipt[field] = value
    receipt["artifact_hash"] = canonical_json_hash(
        {key: item for key, item in receipt.items() if key != "artifact_hash"}
    )
    write_canonical_json(receipt_path, receipt)
    monkeypatch.setattr(launcher.os, "kill", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    with pytest.raises(S206PreparationBlocked, match=f"DETACHED_ATTEMPT_RECEIPT_{pattern}"):
        launcher._detach(args)


@pytest.mark.parametrize("malformation", ["broken_symlink", "nonregular", "invalid_json"])
def test_detach_rejects_malformed_attempt_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    args, preflight, execution = _detached_test_args(tmp_path)
    monkeypatch.setattr(launcher, "_preflight", lambda _args: preflight)
    monkeypatch.setattr(launcher, "_load_formal_execution", lambda _args: execution)
    attempt_dir = tmp_path / "operations/s206/attempts/bad-attempt"
    attempt_dir.mkdir(parents=True)
    receipt = attempt_dir / "launcher.pid.json"
    if malformation == "broken_symlink":
        try:
            receipt.symlink_to("missing-receipt.json")
        except OSError:
            pytest.skip("symlink creation is unavailable on this host")
    elif malformation == "nonregular":
        receipt.mkdir()
    else:
        receipt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(S206PreparationBlocked):
        launcher._detach(args)


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
    _write_gpu_inventory(root / "evidence/gpu-inventory.json")

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
    # This pre-existing fixture intentionally models the invalid legacy
    # consumer shape: G23 cells have no canonical identities/bound sidecars
    # and the producer table is not a corrected amendment.  S2.6 must stop
    # before constructing a provider or consuming any pilot draw.
    with pytest.raises(S206PreparationBlocked, match="G23_(ANCHOR_SET|CORRECTED_DELTA)"):
        launcher._production_cell(args)
    assert not (root / "operations/s206/confirmatory-mapping.json").exists()
