from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from param_importance_nlp.runtime import publish_tensor_bundle
from param_importance_nlp.stage0_g5 import Stage0G5Error, validate_g5_report_set


def _record(step: int, loss: float) -> dict[str, object]:
    return {
        "status": "COMMITTED",
        "global_step": step,
        "mean_loss": loss,
        "global_gradient_norm": 1.0 / step,
        "parameter_post_state_hash": f"post-{step}",
        "batch_ids": ["fixed-real-batch"],
    }


def _bundle(root: Path, name: str, *, delta: float = 0.0) -> tuple[str, str]:
    path = root / "bundles" / name
    phases = {
        "parameters_pre": {"weight": torch.tensor([1.0, 2.0])},
        "mean_gradient": {"weight": torch.tensor([0.25 + delta, -0.5])},
        "optimizer_gradient": {"weight": torch.tensor([0.25 + delta, -0.5])},
        "parameters_post": {"weight": torch.tensor([0.9, 2.1])},
        "update": {"weight": torch.tensor([-0.1, 0.1])},
    }
    identity = publish_tensor_bundle(
        path,
        {
            "schema_version": "stage0-g5-selected-step-tensors-v1",
            "capture_step": 1,
            "selected_names": ["weight"],
            "tensors": phases,
        },
    )
    return path.relative_to(root).as_posix(), identity.manifest_sha256


def _success_report(
    kind: str,
    repeat: int,
    records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "run_kind": kind,
        "repeat_index": repeat,
        "status": "PASS",
        "config_hash": f"{repeat + 1:064x}",
        "training_result": {"status": "COMPLETE", "records": records},
        "training_wall_seconds": 1.0,
        "data_boundary": None,
        "phase_memory": [],
        "step_telemetry": [],
        "selected_tensor_bundle_ref": None,
        "selected_tensor_bundle_sha256": None,
    }


def _valid_report_set(root: Path) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    first_ref, first_hash = _bundle(root, "fp32-0")
    second_ref, second_hash = _bundle(root, "fp32-1")
    for repeat, reference, digest in (
        (0, first_ref, first_hash),
        (1, second_ref, second_hash),
    ):
        report = _success_report("fp32_determinism", repeat, [_record(1, 2.0)])
        report.update(
            {
                "config_hash": "a" * 64,
                "selected_tensor_bundle_ref": reference,
                "selected_tensor_bundle_sha256": digest,
            }
        )
        reports.append(report)

    losses = [2.0 - 0.025 * step for step in range(50)]
    for repeat in range(3):
        report = _success_report(
            "overfit",
            repeat,
            [_record(step + 1, loss) for step, loss in enumerate(losses)],
        )
        report["step_telemetry"] = [
            {"step": step, "sample_ids": ["pile:fixed:0001"]}
            for step in range(1, 51)
        ]
        reports.append(report)

    bf16_ref, bf16_hash = _bundle(root, "bf16")
    bf16 = _success_report("bf16", 0, [_record(1, 2.0)])
    bf16.update(
        {
            "data_boundary": {
                "all_valid": True,
                "tokenizer_asset_id": "pythia-tokenizer",
            },
            "phase_memory": [
                {"cuda_peak_allocated_bytes": value}
                for value in (1, 1024, 2048)
            ],
            "selected_tensor_bundle_ref": bf16_ref,
            "selected_tensor_bundle_sha256": bf16_hash,
        }
    )
    reports.append(bf16)

    for repeat in range(3):
        memory = _success_report(
            "memory",
            repeat,
            [_record(step, 2.0 - step / 100) for step in range(1, 26)],
        )
        memory["step_telemetry"] = [
            {
                "step": step,
                "memory": {"cuda_allocated_bytes": 1_000_000},
            }
            for step in range(1, 26)
        ]
        reports.append(memory)

    for kind in (
        "failure_invalid_asset",
        "failure_out_of_range",
        "failure_nonfinite",
        "failure_output_collision",
        "failure_checkpoint_write",
    ):
        reports.append(
            {
                "run_kind": kind,
                "repeat_index": 0,
                "status": "EXPECTED_FAILURE_CONFIRMED",
                "failure": {
                    "exception_class": "ExpectedError",
                    "last_valid_step": 0,
                    "success_commit_absent": True,
                },
            }
        )
    return reports


def test_g5_aggregate_requires_all_fresh_process_measurements(tmp_path: Path) -> None:
    metrics = validate_g5_report_set(tmp_path, _valid_report_set(tmp_path))

    assert metrics["report_count"] == 14
    assert metrics["fp32"]["selected_tensor_max_absolute_error"] == 0.0
    assert len(metrics["overfit"]) == 3
    assert all(item["last_to_first_ratio"] < 0.95 for item in metrics["overfit"])
    assert len(metrics["memory"]) == 3
    assert len(metrics["failure_paths"]) == 5


def test_g5_aggregate_rejects_tensor_drift_and_memory_growth(tmp_path: Path) -> None:
    reports = _valid_report_set(tmp_path)
    drift_ref, drift_hash = _bundle(tmp_path, "fp32-drift", delta=0.1)
    second = next(
        item
        for item in reports
        if item["run_kind"] == "fp32_determinism" and item["repeat_index"] == 1
    )
    second["selected_tensor_bundle_ref"] = drift_ref
    second["selected_tensor_bundle_sha256"] = drift_hash
    with pytest.raises(Stage0G5Error, match="G5_FP32_TENSOR_TOLERANCE_FAILED"):
        validate_g5_report_set(tmp_path, reports)

    reports = _valid_report_set(tmp_path / "memory")
    memory = next(item for item in reports if item["run_kind"] == "memory")
    for index, row in enumerate(memory["step_telemetry"]):
        row["memory"]["cuda_allocated_bytes"] = 1_000_000 + index * 100_000
    with pytest.raises(Stage0G5Error, match="G5_MEMORY_STABILITY_FAILED"):
        validate_g5_report_set(tmp_path / "memory", reports)


def test_g5_aggregate_rejects_missing_failure_path(tmp_path: Path) -> None:
    reports = _valid_report_set(tmp_path)
    reports.pop()
    with pytest.raises(Stage0G5Error, match="G5_REPORT_SET_CARDINALITY_INVALID"):
        validate_g5_report_set(tmp_path, reports)

