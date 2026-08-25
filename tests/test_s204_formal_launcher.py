from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops.stage2.run_s204_formal import (
    ASSET_DIGEST,
    APPROVED_GPU_BINDINGS,
    DATA_DIGEST,
    EXCLUDED_PCI,
    EXCLUDED_UUID,
    _Heartbeat,
    _AttemptLease,
    _canonical_hash,
    _g23_gate,
    _bind_gpu,
    _file_sha256,
    _validate_gpu_smoke_artifact,
    _runtime_cell_paths,
    G21_ARTIFACT,
    build_plan,
    execute_with_task_runtime,
    main,
)
from param_importance_nlp.experiments import (
    AssetResolutionManifest,
    CheckpointFile,
    CheckpointRecord,
    DataFile,
    DataRangeManifest,
    FORMAL_CHECKPOINT_SELECTION,
    FORMAL_DATASET_ID,
    FORMAL_DATASET_REVISION,
    FORMAL_DATA_FILES,
    FORMAL_DATA_MANIFEST_SHA256,
    FORMAL_TOTAL_TRAINING_STEPS,
)


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    g21 = {
        "status": "PASS",
        "artifact_hash": G21_ARTIFACT,
        "current_gpu_smoke": {
            "status": "PASS",
            "excluded_pci_bus_ids": [EXCLUDED_PCI],
            "excluded_scheduled": False,
            "excluded_device": {"uuid": EXCLUDED_UUID},
            "allowed_devices": [
                {"pci_bus_id": pci, "uuid": uuid}
                for pci, uuid in APPROVED_GPU_BINDINGS.items()
            ],
        },
    }
    records = []
    for i, ((model, stage), (step, revision)) in enumerate(FORMAL_CHECKPOINT_SELECTION.items()):
        digest = f"{i + 1:064x}"
        records.append(
            CheckpointRecord(
                model_id=model,
                training_stage=stage,
                checkpoint_id=f"cell-{i}",
                training_step=step,
                total_training_steps=FORMAL_TOTAL_TRAINING_STEPS,
                target_fraction={"initialization": 0.0, "early": 0.01, "mid_late": 0.5}[stage],
                repository=f"EleutherAI/{model}",
                revision=revision,
                root_ref=f"models/{model}/step{step}",
                state="ready",
                files=(
                    CheckpointFile("model.safetensors", 1, digest, "weights"),
                    CheckpointFile("config.json", 1, digest, "config"),
                    CheckpointFile("tokenizer.json", 1, digest, "tokenizer"),
                ),
                manifest_ref=f"manifests/{model}-step{step}.json",
                manifest_sha256=digest,
                parameter_registry_hash=digest,
                config_sha256=digest,
                tokenizer_sha256=digest,
                load_status="passed",
                load_evidence_ref=f"evidence/{model}-step{step}.json",
                load_evidence_sha256=digest,
            )
        )
    data_manifest = DataRangeManifest(
        dataset_id=FORMAL_DATASET_ID,
        revision=FORMAL_DATASET_REVISION,
        manifest_ref="manifests/stage2/pile-prefix.json",
        manifest_sha256=FORMAL_DATA_MANIFEST_SHA256,
        files=tuple(
            DataFile(path, size, sha, "token_shard" if path.endswith(".bin") else "index")
            for path, (size, sha) in FORMAL_DATA_FILES.items()
        ),
    )
    asset_manifest = AssetResolutionManifest(
        scope="formal",
        checkpoints=tuple(records),
        data_range=data_manifest,
        producer_commit="1" * 40,
        execution_commit="2" * 40,
    )
    return g21, asset_manifest.to_dict(), data_manifest.to_dict()


def test_plan_only_is_six_cell_and_does_not_freeze_b_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, assets, data = _inputs()
    monkeypatch.setattr("ops.stage2.run_s204_formal.ASSET_DIGEST", assets["asset_resolution_hash"])
    monkeypatch.setattr("ops.stage2.run_s204_formal.DATA_DIGEST", data["data_range_hash"])
    plan = build_plan(*_inputs(), output_root=tmp_path, candidates=(512, 1024), block_size=32)
    assert plan["cell_count"] == 6
    assert plan["b_ref_status"] == "UNFROZEN_UNTIL_INDEPENDENT_SIZING_PASS"
    assert all(cell["final_sample_count_per_stream"] == "UNFROZEN" for cell in plan["cells"])
    assert all(tmp_path.as_posix() in cell["progress_path"] for cell in plan["cells"])


@pytest.mark.parametrize("mode", ("--execute", "--aggregate"))
def test_runtime_modes_require_explicit_candidate_sizes(
    tmp_path: Path, mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "--g21-evidence",
            str(tmp_path / "g21.json"),
            "--asset-resolution",
            str(tmp_path / "assets.json"),
            "--data-range",
            str(tmp_path / "data.json"),
            "--output-root",
            str(tmp_path / "output"),
            mode,
            "--data-root",
            str(tmp_path),
            "--execution-commit",
            "a" * 40,
        ]
    )
    assert result == 3
    assert "require explicit --candidate-sizes" in capsys.readouterr().err


def test_plan_rejects_missing_failed_gpu_exclusion(tmp_path: Path) -> None:
    g21, assets, data = _inputs()
    g21["current_gpu_smoke"] = {"status": "PASS", "excluded_pci_bus_ids": [], "excluded_scheduled": True}
    with pytest.raises(ValueError, match="GPU exclusion"):
        build_plan(g21, assets, data, output_root=tmp_path)


def test_handoff_shape_without_excluded_device_is_validated_at_raw_report_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g21, assets, data = _inputs()
    g21["current_gpu_smoke"].pop("excluded_device")  # type: ignore[union-attr]
    monkeypatch.setattr("ops.stage2.run_s204_formal.ASSET_DIGEST", assets["asset_resolution_hash"])
    monkeypatch.setattr("ops.stage2.run_s204_formal.DATA_DIGEST", data["data_range_hash"])
    plan = build_plan(g21, assets, data, output_root=tmp_path, candidates=(512, 1024))
    assert plan["formal_eligible"] is False


def test_runtime_launcher_requires_one_gpu_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        execute_with_task_runtime(
            {"cells": []},
            runtime_config_paths=(),
            runtime_environment_path=tmp_path / "environment.json",
            data_root=tmp_path,
            output_root=tmp_path / "out",
            cuda_visible_devices="2,4",
            cell_id=None,
            heartbeat_seconds=1.0,
        )


def test_gpu_binding_rejects_index_reordering(monkeypatch: pytest.MonkeyPatch) -> None:
    approved_pci, approved_uuid = next(iter(APPROVED_GPU_BINDINGS.items()))
    monkeypatch.setattr(
        "ops.stage2.run_s204_formal._gpu_inventory",
        lambda: [
            {"index": "0", "pci_bus_id": EXCLUDED_PCI, "uuid": EXCLUDED_UUID},
            {"index": "2", "pci_bus_id": approved_pci, "uuid": approved_uuid},
        ],
    )
    with pytest.raises(ValueError, match="excluded GPU"):
        _bind_gpu("0")


def _healthy_inventory() -> list[dict[str, object]]:
    rows = [
        {"index": "0", "pci_bus_id": EXCLUDED_PCI, "uuid": EXCLUDED_UUID},
        {"index": "2", "pci_bus_id": "0000:53:00.0", "uuid": APPROVED_GPU_BINDINGS["0000:53:00.0"]},
        {"index": "4", "pci_bus_id": "0000:9C:00.0", "uuid": APPROVED_GPU_BINDINGS["0000:9C:00.0"]},
        {"index": "5", "pci_bus_id": "0000:9D:00.0", "uuid": APPROVED_GPU_BINDINGS["0000:9D:00.0"]},
        {"index": "6", "pci_bus_id": "0000:A0:00.0", "uuid": APPROVED_GPU_BINDINGS["0000:A0:00.0"]},
    ]
    for row in rows:
        row.update(
            {
                "memory_used_mib": "0",
                "memory_total_mib": "40960",
                "utilization_gpu_percent": "0",
                "ecc_uncorrected_volatile": "0",
                "ecc_uncorrected_aggregate": "0",
                "gpu_recovery_action": "None",
            }
        )
    return rows


def _eight_card_inventory() -> list[dict[str, object]]:
    """Shape of the live host inventory: four approved, one excluded, three other."""

    rows = _healthy_inventory()
    rows[0].update(
        {
            "ecc_uncorrected_volatile": "113",
            "ecc_uncorrected_aggregate": "179",
            "gpu_recovery_action": "Drain and Reset",
        }
    )
    rows.extend(
        [
            {"index": "7", "pci_bus_id": "0000:51:00.0", "uuid": "GPU-other-1"},
            {"index": "8", "pci_bus_id": "0000:52:00.0", "uuid": "GPU-other-2"},
            {"index": "9", "pci_bus_id": "0000:54:00.0", "uuid": "GPU-other-3"},
        ]
    )
    for row in rows[5:]:
        row.update(
            {
                "memory_used_mib": "4096",
                "memory_total_mib": "40960",
                "utilization_gpu_percent": "100",
                "ecc_uncorrected_volatile": "7",
                "ecc_uncorrected_aggregate": "8",
                "gpu_recovery_action": "Drain and Reset",
            }
        )
    return rows


def test_gpu_live_inventory_requires_idle_clean_health_and_complete_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = _healthy_inventory()
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: healthy)
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_compute_apps", lambda: [])
    selected, inventory, digest = _bind_gpu("2")
    assert selected == APPROVED_GPU_BINDINGS["0000:53:00.0"]
    assert len(inventory) == 5 and len(digest) == 64

    busy = _healthy_inventory()
    busy[1]["memory_used_mib"] = "1"
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: busy)
    with pytest.raises(ValueError, match="not idle"):
        _bind_gpu("2")

    ecc = _healthy_inventory()
    ecc[1]["ecc_uncorrected_volatile"] = "1"
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: ecc)
    with pytest.raises(ValueError, match="ECC"):
        _bind_gpu("2")

    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", _healthy_inventory)
    monkeypatch.setattr(
        "ops.stage2.run_s204_formal._gpu_compute_apps",
        lambda: [{"pid": "123", "process_name": "other", "gpu_uuid": APPROVED_GPU_BINDINGS["0000:53:00.0"]}],
    )
    with pytest.raises(ValueError, match="compute apps"):
        _bind_gpu("2")


def test_gpu_binding_resolves_any_current_index_by_approved_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _healthy_inventory()
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: inventory)
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_compute_apps", lambda: [])

    for token, (pci, uuid) in zip(("2", "4", "5", "6"), APPROVED_GPU_BINDINGS.items()):
        selected, _, _ = _bind_gpu(token)
        assert selected == uuid
        assert next(row for row in inventory if row["index"] == token)["pci_bus_id"] == pci


def test_gpu_binding_rejects_unknown_tokens_and_excluded_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _healthy_inventory()
    # Put the excluded card at an arbitrary live index so 0 and 3 are truly
    # absent selectors; the PCI/UUID identity remains the exclusion authority.
    inventory[0]["index"] = "1"
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: inventory)
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_compute_apps", lambda: [])
    for token in ("0", "3"):
        with pytest.raises(ValueError, match="absent from live inventory"):
            _bind_gpu(token)
    with pytest.raises(ValueError, match="excluded GPU"):
        _bind_gpu(EXCLUDED_UUID)

    drifted = _healthy_inventory()
    drifted[1]["uuid"] = "GPU-drifted-approved-card"
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: drifted)
    with pytest.raises(ValueError, match="approved smoke set"):
        _bind_gpu("2")


def test_gpu_live_inventory_scopes_health_to_selected_card_on_eight_card_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _eight_card_inventory()
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: inventory)
    monkeypatch.setattr(
        "ops.stage2.run_s204_formal._gpu_compute_apps",
        lambda: [
            {
                "pid": "456",
                "process_name": "other-approved",
                "gpu_uuid": APPROVED_GPU_BINDINGS["0000:9C:00.0"],
            },
            {"pid": "789", "process_name": "other-unapproved", "gpu_uuid": "GPU-other-1"},
        ],
    )

    selected, bound_inventory, digest = _bind_gpu("2")
    assert selected == APPROVED_GPU_BINDINGS["0000:53:00.0"]
    assert len(bound_inventory) == 8 and len(digest) == 64

    with pytest.raises(ValueError, match="excluded GPU"):
        _bind_gpu("0")

    busy = _eight_card_inventory()
    busy[1]["memory_used_mib"] = "1"
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: busy)
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_compute_apps", lambda: [])
    with pytest.raises(ValueError, match="not idle"):
        _bind_gpu("2")

    ecc = _eight_card_inventory()
    ecc[1]["ecc_uncorrected_volatile"] = "1"
    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", lambda: ecc)
    with pytest.raises(ValueError, match="ECC"):
        _bind_gpu("2")

    monkeypatch.setattr("ops.stage2.run_s204_formal._gpu_inventory", _eight_card_inventory)
    monkeypatch.setattr(
        "ops.stage2.run_s204_formal._gpu_compute_apps",
        lambda: [
            {
                "pid": "123",
                "process_name": "selected",
                "gpu_uuid": APPROVED_GPU_BINDINGS["0000:53:00.0"],
            }
        ],
    )
    with pytest.raises(ValueError, match="compute apps"):
        _bind_gpu("2")


def test_duplicate_formal_writer_is_rejected_by_output_lease(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "cell-0"
    with _AttemptLease(output, cell_id="cell-0", attempt_id="fresh-a"):
        with pytest.raises(RuntimeError, match="LEASE_HELD"):
            with _AttemptLease(output, cell_id="cell-0", attempt_id="resume-b"):
                pass


def test_execute_reloads_hash_bound_g21_smoke_report(tmp_path: Path) -> None:
    report = {
        "schema_version": "stage2-s202-current-gpu-smoke-v1",
        "status": "PASS",
        "atomic_publication": True,
        "excluded_pci_bus_ids": [EXCLUDED_PCI],
        "excluded_device": {
            "pci_bus_id": EXCLUDED_PCI,
            "uuid": EXCLUDED_UUID,
            "scheduled": False,
        },
        "allowed_devices": [
            {"pci_bus_id": pci, "uuid": uuid}
            for pci, uuid in APPROVED_GPU_BINDINGS.items()
        ],
    }
    report_path = tmp_path / "smoke.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    smoke = _validate_gpu_smoke_artifact(
        {
            "gpu_smoke_ref": "smoke.json",
            "gpu_smoke_sha256": _file_sha256(report_path),
        },
        tmp_path,
    )
    assert smoke["sha256"] == _file_sha256(report_path)


def test_runtime_cell_dispatch_and_heartbeat_are_recoverable(tmp_path: Path) -> None:
    paths = []
    cells = [{"cell_id": f"cell-{index}"} for index in range(6)]
    for index in range(6):
        path = tmp_path / f"config-{index}.json"
        path.write_text('{"task_id":"stage2.04_reference_target"}\n', encoding="utf-8")
        paths.append(path)
    selected = _runtime_cell_paths({"cells": cells}, tuple(paths), None)
    assert [item[2] for item in selected] == [f"cell-{index}" for index in range(6)]
    one = _runtime_cell_paths({"cells": cells}, (paths[3],), "cell-3")
    assert [item[2] for item in one] == ["cell-3"]

    heartbeat = tmp_path / "progress.jsonl"
    with _Heartbeat(heartbeat, "cell-3", 0.01):
        pass
    phases = [row.split('"phase": "', 1)[1].split('"', 1)[0] for row in heartbeat.read_text(encoding="utf-8").splitlines()]
    assert phases[0] == "STARTED"
    assert phases[-1] == "STOPPED"


def test_g23_qualification_is_fail_closed_without_complete_metrics(tmp_path: Path) -> None:
    plan = {"cells": [{"cell_id": f"cell-{index}"} for index in range(6)]}
    task_hashes = {f"cell-{index}": f"{index + 1:064x}" for index in range(6)}
    bundle_hashes = {f"cell-{index}": f"{index + 10:064x}" for index in range(6)}
    status, gate_hash = _g23_gate(
        plan=plan,
        metrics_path=None,
        task_result_refs=tuple(f"runs/cell-{index}/task-run-result.json" for index in range(6)),
        task_result_hashes=task_hashes,
        bundle_hashes=bundle_hashes,
        output_root=tmp_path,
    )
    assert status == "BLOCKED"
    assert gate_hash == ""
    preflights = list((tmp_path / "g2.3-preflight").glob("*/preflight.json"))
    assert len(preflights) == 1
    preflight = json.loads(preflights[0].read_text(encoding="utf-8"))
    assert preflight["status"] == "BLOCKED"
    assert preflight["formal_eligible"] is False


def test_g23_numeric_candidate_never_becomes_gate_without_output_evaluator(tmp_path: Path) -> None:
    plan = {"cells": [{"cell_id": f"cell-{index}"} for index in range(6)]}
    task_hashes = {f"cell-{index}": f"{index + 1:064x}" for index in range(6)}
    bundle_hashes = {f"cell-{index}": f"{index + 10:064x}" for index in range(6)}
    metrics = {
        "schema_version": "stage2-g23-reference-evaluation-v1",
        "calculator": {"producer_commit": "a" * 40, "source_sha256": "b" * 64},
        "cells": [],
    }
    for index in range(6):
        metrics["cells"].append(  # type: ignore[union-attr]
            {
                "cell_id": f"cell-{index}",
                "metrics": {
                    "task_result_hash": task_hashes[f"cell-{index}"],
                    "bundle_manifest_sha256": bundle_hashes[f"cell-{index}"],
                    "normalized_l1": 0.001,
                    "pearson": 0.999,
                    "signal_eligible_spearman": 0.999,
                    "layer_module_spearman": 0.999,
                    "topk_overlap_0_001": 0.999,
                    "topk_overlap_0_01": 0.999,
                    "topk_overlap_0_05": 0.999,
                    "layer_module_delta": 0.001,
                    "h_ref": 0.01,
                    "min_delta_sci": 1.0,
                    "epsilon_num": 0.01,
                    "a_b_interval_covered": True,
                    "bias_cross_interval_covered": True,
                    "ranking_bias_direction": True,
                    "variance_scaling_verified": True,
                    "state_replay_verified": True,
                    "one_shot_complete": True,
                },
            }
        )
    metrics["artifact_hash"] = _canonical_hash(
        {key: value for key, value in metrics.items() if key != "artifact_hash"}
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    status, gate_hash = _g23_gate(
        plan=plan,
        metrics_path=metrics_path,
        task_result_refs=tuple(f"runs/cell-{index}/result.json" for index in range(6)),
        task_result_hashes=task_hashes,
        bundle_hashes=bundle_hashes,
        output_root=tmp_path,
    )
    assert status == "BLOCKED"
    assert len(gate_hash) == 64
