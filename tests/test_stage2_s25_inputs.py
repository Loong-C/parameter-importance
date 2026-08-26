"""Fail-closed tests for S2.5 input extraction and exhaustive development sweep."""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.experiments.preregistration import build_stage2_preregistration
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_s25_formal import (
    APPROVED_GPU_UUIDS,
    S25ExecutionBlocked,
    S25FormalRunner,
    _experiment_plan_entries,
    _make_mappings,
)
from param_importance_nlp.experiments.stage2_s25_inputs import (
    S205InputBlocked,
    build_s205_formal_inputs,
    validate_s205_development_sweep,
)
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from ops.stage2.materialize_s205_formal_inputs import main as materialize_main
from ops.stage2.run_s205_formal import (
    _inventory,
    _load_inventory_snapshot,
    _load_materialization_index,
    _parser,
    _repository_commit,
    _validate_inventory,
    _worker,
)
from param_importance_nlp.contracts.g21_formal_handoff import ALLOWED_DEVICES, EXCLUDED_PCI, EXCLUDED_UUID


def _sampling() -> SamplingPlan:
    return SamplingPlan(
        SamplingUniverse("s205-formal-input-test", tuple(range(23))),
        {"pilot": 11, "confirmatory": 12, "reference_sizing": 13, "reference_A": 14, "reference_B": 15},
    )


def _formal_fixture(root: Path) -> tuple[str, str, str, SamplingPlan]:
    sampling = _sampling()
    preregistration = build_stage2_preregistration(
        seed_plan_hash="a" * 64,
        producer_commit="b" * 40,
        scope="formal",
    )
    store = TaskArtifactStore(root, "evidence/task-inputs")
    prereg_ref = store.publish(
        task_id="stage2.01_scope_hypotheses_and_preregistration",
        artifact_kind="preregistration",
        config_hash="c" * 64,
        run_intent="formal",
        payload=preregistration,
        formal_eligible=True,
    ).commit_ref
    sampling_ref = store.publish(
        task_id="stage2.03_assets_checkpoints_and_sampling",
        artifact_kind="sampling_plan",
        config_hash="d" * 64,
        run_intent="formal",
        payload=sampling.to_dict(),
        formal_eligible=True,
    ).commit_ref
    execution = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash="e" * 64,
        asset_manifest_hashes=("f" * 64,),
        prerequisite_gates=(GateRecord(
            gate_id="stage1.G1-EXIT",
            stage=1,
            status=GateStatus.PASS,
            checked_at="2026-08-25T00:00:00+00:00",
            evidence_refs=("evidence/stage1/exit.json",),
        ),),
    )
    execution_ref = "evidence/formal-execution.json"
    write_canonical_json(root / execution_ref, execution.to_dict())
    return prereg_ref, sampling_ref, execution_ref, sampling


def test_materializer_extracts_sampling_payload_and_freezes_complete_grid(tmp_path: Path) -> None:
    prereg_ref, sampling_ref, execution_ref, sampling = _formal_fixture(tmp_path)
    direct_sampling, sweep = build_s205_formal_inputs(
        tmp_path,
        preregistration_ref=prereg_ref,
        sampling_plan_ref=sampling_ref,
        formal_execution_ref=execution_ref,
    )
    assert direct_sampling == sampling.to_dict()
    assert direct_sampling["schema_version"] == "sampling-plan-v1"
    assert sweep["candidate_batch_sizes"] == [32, 64, 128, 256]
    assert sweep["candidate_microbatch_counts"] == [2, 4, 8, 16, 32]
    assert sweep["pilot_draw_count"] == 480
    assert sweep["primary_parameters_selected"] is False
    assert sweep["confirmatory_draws_generated"] is False
    assert sweep["reference_draws_generated"] is False
    assert [entry[0] for entry in _experiment_plan_entries(sweep)] == [0, 32, 96, 224]


def test_sweep_mappings_use_nonoverlapping_pilot_positions(tmp_path: Path) -> None:
    prereg_ref, sampling_ref, execution_ref, sampling = _formal_fixture(tmp_path)
    _direct, sweep = build_s205_formal_inputs(
        tmp_path,
        preregistration_ref=prereg_ref,
        sampling_plan_ref=sampling_ref,
        formal_execution_ref=execution_ref,
    )
    positions: list[int] = []
    for start, plan in _experiment_plan_entries(sweep):
        mappings = _make_mappings(sampling, plan, start_position=start)
        positions.extend(draw.position for mapping in mappings for draw in mapping.draws)
    assert positions == list(range(480))
    assert len(positions) == len(set(positions))


@pytest.mark.parametrize("mutation", ["missing_batch", "overlap", "select_primary", "sampling_hash"])
def test_sweep_tamper_fails_closed(tmp_path: Path, mutation: str) -> None:
    prereg_ref, sampling_ref, execution_ref, sampling = _formal_fixture(tmp_path)
    _direct, original = build_s205_formal_inputs(
        tmp_path,
        preregistration_ref=prereg_ref,
        sampling_plan_ref=sampling_ref,
        formal_execution_ref=execution_ref,
    )
    value = deepcopy(original)
    if mutation == "missing_batch":
        value["entries"].pop()
    elif mutation == "overlap":
        value["entries"][1]["start_position"] = 0
    elif mutation == "select_primary":
        value["primary_parameters_selected"] = True
    else:
        value["sampling_plan_hash"] = "0" * 64
    value["artifact_hash"] = canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"})
    with pytest.raises(S205InputBlocked):
        validate_s205_development_sweep(value, sampling=sampling)


def test_sampling_task_artifact_identity_is_not_accepted_as_preregistration(tmp_path: Path) -> None:
    _prereg_ref, sampling_ref, execution_ref, _sampling_plan = _formal_fixture(tmp_path)
    with pytest.raises(S205InputBlocked, match="TASK_ARTIFACT_IDENTITY_INVALID"):
        build_s205_formal_inputs(
            tmp_path,
            preregistration_ref=sampling_ref,
            sampling_plan_ref=sampling_ref,
            formal_execution_ref=execution_ref,
        )


def test_materializer_is_append_only_and_idempotent(tmp_path: Path) -> None:
    prereg_ref, sampling_ref, execution_ref, _sampling_plan = _formal_fixture(tmp_path)
    argv = [
        "--data-root", str(tmp_path),
        "--preregistration-ref", prereg_ref,
        "--sampling-plan-task-ref", sampling_ref,
        "--formal-execution-ref", execution_ref,
        "--output-root", "evidence/stage2/s205/inputs/run-unique",
    ]
    assert materialize_main(argv) == 0
    assert materialize_main(argv) == 0
    index = tmp_path / "evidence/stage2/s205/inputs/run-unique/index.json"
    tampered = {"schema_version": "tampered"}
    write_canonical_json(index, tampered)
    assert materialize_main(argv) == 3


def test_launcher_revalidates_materialized_input_index_source_closure(tmp_path: Path) -> None:
    prereg_ref, sampling_ref, execution_ref, _sampling_plan = _formal_fixture(tmp_path)
    output_ref = "evidence/stage2/s205/inputs/index-closure"
    assert materialize_main([
        "--data-root", str(tmp_path),
        "--preregistration-ref", prereg_ref,
        "--sampling-plan-task-ref", sampling_ref,
        "--formal-execution-ref", execution_ref,
        "--output-root", output_ref,
    ]) == 0
    index_ref = f"{output_ref}/index.json"
    loaded = _load_materialization_index(tmp_path, index_ref)
    assert loaded["status"] == "FROZEN"
    sampling = load_canonical_json(tmp_path / str(loaded["sampling_plan_ref"]))
    assert isinstance(sampling, dict)
    sampling["unapproved_extra"] = True
    write_canonical_json(tmp_path / str(loaded["sampling_plan_ref"]), sampling)
    with pytest.raises(Exception, match="S205_INPUT_INDEX"):
        _load_materialization_index(tmp_path, index_ref)


def test_launcher_gpu_inventory_requires_complete_clean_s206_snapshot() -> None:
    rows = [
        {
            "pci_bus_id": pci,
            "uuid": uuid,
            "memory_used_mib": "0",
            "memory_total_mib": "40960",
            "utilization_gpu_percent": "0",
            "ecc_uncorrected_volatile": "0",
            "ecc_uncorrected_aggregate": "0",
            "temperature_c": "40",
            "row_remap_status": "None",
            "gpu_recovery_action": "None",
        }
        for pci, uuid in ALLOWED_DEVICES
    ]
    rows.extend(
        {
            "pci_bus_id": EXCLUDED_PCI,
            "uuid": EXCLUDED_UUID,
            "memory_used_mib": "0",
            "memory_total_mib": "40960",
            "utilization_gpu_percent": "0",
            "ecc_uncorrected_volatile": "0",
            "ecc_uncorrected_aggregate": "0",
            "temperature_c": "40",
            "row_remap_status": "None",
            "gpu_recovery_action": "None",
        }
        for _ in range(1)
    )
    rows.extend(
        {
            "pci_bus_id": f"0000:{bus}:00.0",
            "uuid": f"GPU-other-{bus}",
            "memory_used_mib": "0",
            "memory_total_mib": "40960",
            "utilization_gpu_percent": "0",
            "ecc_uncorrected_volatile": "0",
            "ecc_uncorrected_aggregate": "0",
            "temperature_c": "40",
            "row_remap_status": "None",
            "gpu_recovery_action": "None",
        }
        for bus in ("51", "52", "54")
    )
    result = _validate_inventory(rows)
    assert result["inventory_count"] == 8
    rows[0]["memory_used_mib"] = "1"
    with pytest.raises(Exception, match="S205_GPU_INVENTORY"):
        _validate_inventory(rows)


def test_launcher_accepts_only_hash_bound_gpu_inventory_envelope(tmp_path: Path) -> None:
    rows = [
        {
            "pci_bus_id": pci,
            "uuid": uuid,
            "memory_used_mib": "0",
            "memory_total_mib": "40960",
            "utilization_gpu_percent": "0",
            "ecc_uncorrected_volatile": "0",
            "ecc_uncorrected_aggregate": "0",
            "temperature_c": "40",
            "row_remap_status": "None",
            "gpu_recovery_action": "None",
        }
        for pci, uuid in ALLOWED_DEVICES
    ]
    rows += [{
        "pci_bus_id": EXCLUDED_PCI,
        "uuid": EXCLUDED_UUID,
        "memory_used_mib": "0", "memory_total_mib": "40960",
        "utilization_gpu_percent": "0", "ecc_uncorrected_volatile": "0",
        "ecc_uncorrected_aggregate": "0", "temperature_c": "40",
        "row_remap_status": "None", "gpu_recovery_action": "None",
    }]
    rows += [{
        "pci_bus_id": f"0000:{bus}:00.0", "uuid": f"GPU-other-{bus}",
        "memory_used_mib": "0", "memory_total_mib": "40960",
        "utilization_gpu_percent": "0", "ecc_uncorrected_volatile": "0",
        "ecc_uncorrected_aggregate": "0", "temperature_c": "40",
        "row_remap_status": "None", "gpu_recovery_action": "None",
    } for bus in ("51", "52", "54")]
    path = tmp_path / "gpu-inventory.json"
    raw_capture = tmp_path / "nvidia-smi.raw"
    raw_capture.write_bytes(b"nvidia-smi capture fixture\n")
    value = {
        "schema_version": "stage2-s206-gpu-inventory-v1",
        "artifact_ref": "gpu-inventory.json",
        "source_ref": "nvidia-smi.raw",
        "source_sha256": hashlib.sha256(raw_capture.read_bytes()).hexdigest(),
        "rows": rows,
        "compute_apps": [],
    }
    value["artifact_hash"] = canonical_json_hash(value)
    write_canonical_json(path, value)
    rows_loaded, identity = _load_inventory_snapshot(path, data_root=tmp_path)
    assert len(rows_loaded) == 8
    assert identity["artifact_ref"] == "gpu-inventory.json"
    assert identity["source_ref"] == "nvidia-smi.raw"
    assert identity["source_sha256"] == value["source_sha256"]
    value["rows"][0]["memory_used_mib"] = "1"
    write_canonical_json(path, value)
    with pytest.raises(Exception, match="S205_GPU_INVENTORY_ARTIFACT_HASH_MISMATCH"):
        _inventory(path, data_root=tmp_path)


def test_launcher_inventory_binds_distinct_raw_capture_and_rejects_tamper(tmp_path: Path) -> None:
    rows = [
        {
            "pci_bus_id": pci,
            "uuid": uuid,
            "memory_used_mib": "0",
            "memory_total_mib": "40960",
            "utilization_gpu_percent": "0",
            "ecc_uncorrected_volatile": "0",
            "ecc_uncorrected_aggregate": "0",
            "temperature_c": "40",
            "row_remap_status": "None",
            "gpu_recovery_action": "None",
        }
        for pci, uuid in ALLOWED_DEVICES
    ]
    rows += [{
        "pci_bus_id": EXCLUDED_PCI,
        "uuid": EXCLUDED_UUID,
        "memory_used_mib": "0", "memory_total_mib": "40960",
        "utilization_gpu_percent": "0", "ecc_uncorrected_volatile": "0",
        "ecc_uncorrected_aggregate": "0", "temperature_c": "40",
        "row_remap_status": "None", "gpu_recovery_action": "None",
    }]
    rows += [{
        "pci_bus_id": f"0000:{bus}:00.0", "uuid": f"GPU-other-{bus}",
        "memory_used_mib": "0", "memory_total_mib": "40960",
        "utilization_gpu_percent": "0", "ecc_uncorrected_volatile": "0",
        "ecc_uncorrected_aggregate": "0", "temperature_c": "40",
        "row_remap_status": "None", "gpu_recovery_action": "None",
    } for bus in ("51", "52", "54")]
    raw_capture = tmp_path / "capture.raw"
    raw_capture.write_bytes(b"raw capture v1")
    path = tmp_path / "inventory.json"
    value = {
        "schema_version": "stage2-s206-gpu-inventory-v1",
        "artifact_ref": "inventory.json",
        "source_ref": "capture.raw",
        "source_sha256": hashlib.sha256(raw_capture.read_bytes()).hexdigest(),
        "rows": rows,
        "compute_apps": [],
    }
    value["artifact_hash"] = canonical_json_hash(value)
    write_canonical_json(path, value)
    _load_inventory_snapshot(path, data_root=tmp_path)
    raw_capture.write_bytes(b"tampered capture")
    with pytest.raises(Exception, match="S205_GPU_INVENTORY_SOURCE_SHA256_MISMATCH"):
        _load_inventory_snapshot(path, data_root=tmp_path)


def test_launcher_detach_and_resume_are_execute_modifiers() -> None:
    args = _parser().parse_args([
        "--execute", "--detach", "--resume", "--data-root", "root",
        "--s205-rebind-ref", "rebind.json", "--input-index-ref", "inputs/index.json",
        "--artifact-root", "out", "--operations-root", "ops",
    ])
    assert args.execute is True and args.detach is True and args.resume is True


def test_worker_requires_exact_visible_gpu_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = type("WorkerArgs", (), {
        "cell_id": "pythia-14m:initialization",
        "gpu_uuid": APPROVED_GPU_UUIDS[0],
    })()
    with pytest.raises(S25ExecutionBlocked, match="S205_WORKER_GPU_BINDING_MISMATCH"):
        _worker(args)


def test_launcher_executor_commit_requires_clean_worktree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def dirty_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=" M launcher.py\n")

    monkeypatch.setattr("ops.stage2.run_s205_formal.subprocess.run", dirty_run)
    with pytest.raises(S25ExecutionBlocked, match="S205_REPOSITORY_DIRTY"):
        _repository_commit(tmp_path)


def test_launcher_executor_commit_binds_exact_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = "a" * 40

    def clean_run(command: object, **_kwargs: object) -> SimpleNamespace:
        if isinstance(command, list) and "status" in command:
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout=expected + "\n")

    monkeypatch.setattr("ops.stage2.run_s205_formal.subprocess.run", clean_run)
    assert _repository_commit(tmp_path) == expected


def test_runner_rejects_sweep_bound_to_other_formal_execution(tmp_path: Path) -> None:
    prereg_ref, sampling_ref, execution_ref, sampling = _formal_fixture(tmp_path)
    _direct, sweep = build_s205_formal_inputs(
        tmp_path,
        preregistration_ref=prereg_ref,
        sampling_plan_ref=sampling_ref,
        formal_execution_ref=execution_ref,
    )
    with pytest.raises(S25ExecutionBlocked, match="FORMAL_EXECUTION_MISMATCH"):
        S25FormalRunner(
            data_root=tmp_path,
            rebind_plan={
                "formal_execution_hash": "0" * 64,
                "formal_execution_ref": execution_ref,
            },
            experiment_plan=sweep,
            sampling_plan=sampling,
            artifact_root=tmp_path / "output",
        )
