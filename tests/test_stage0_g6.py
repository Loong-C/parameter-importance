from __future__ import annotations

from copy import deepcopy

import pytest

from param_importance_nlp.stage0_g6 import Stage0G6Error, validate_g6_report_set


def _comparison() -> dict[str, object]:
    tensor = {
        "max_absolute_error": 2e-7,
        "max_relative_error": 2e-6,
        "relative_l2_error": 1e-6,
        "worst_tensor": "weight",
    }
    return {
        "loss_absolute_error": 1e-7,
        "gradient": dict(tensor),
        "clipped_gradient": dict(tensor),
        "parameters": dict(tensor),
        "effective_count_match": True,
        "clip_factor_absolute_error": 1e-7,
    }


def _collective_report(repeat: int) -> dict[str, object]:
    messages = []
    for elements, median in ((65536, 0.001), (262144, 0.004), (4194304, 0.02)):
        value = median * (1.0 + 0.01 * repeat)
        messages.append(
            {
                "tensor_elements": elements,
                "message_bytes": elements * 4,
                "warmup_iterations": 20,
                "measured_iterations": 50,
                "samples_per_measurement": 3,
                "median_seconds": value,
                "p95_seconds": value * 1.5,
                "throughput_bytes_per_second": elements * 16 / value,
                "sample_seconds": [value for _ in range(50)],
                "result_value": 10.0,
            }
        )
    return {
        "run_kind": "collective",
        "repeat_index": repeat,
        "metrics": {
            "message_metrics": messages,
            "broadcast_pass": True,
            "all_gather_pass": True,
            "barrier_pass": True,
            "scalar_expected_sum": 10,
            "nccl_p2p_disable": 1,
        },
    }


def _semantic_report() -> dict[str, object]:
    bf16 = []
    for rank in range(4):
        bf16.append(
            {
                "rank": rank,
                "global_steps": 1,
                "attempts": 1,
                "record_count": 1,
                "statuses": ["COMMITTED"],
                "mean_losses": [2.0],
                "effective_counts": [8192],
                "batch_ids": [f"pile-rank-{rank}"],
                "checkpoint_count": 1,
                "finite_parameters": True,
                "peak_memory_bytes": 1_000_000,
                "wall_seconds": 1.0,
                "asset_evidence_count": 3,
                "resource_profile_count": 1,
                "coordinate_registry_hash": "a" * 64,
            }
        )
    return {
        "run_kind": "semantic",
        "repeat_index": 0,
        "metrics": {
            "data_shards": {
                "expected_main": [f"sample-{index:02d}" for index in range(8)],
                "observed_main": [f"sample-{index:02d}" for index in range(8)],
                "expected_tail": [f"sample-{index:02d}" for index in range(8, 12)],
                "observed_tail": [f"sample-{index:02d}" for index in range(8, 12)],
            },
            "comparisons": {
                "ddp_full": _comparison(),
                "accumulation_sync_each": _comparison(),
                "accumulation_no_sync": _comparison(),
                "incomplete_tail": _comparison(),
            },
            "communication": {
                "sync_each_calls": 2,
                "no_sync_calls": 1,
                "microbatches_per_rank": 2,
                "event_order": [
                    "no_sync_forward_backward",
                    "sync_backward",
                    "gradient_sync_complete",
                    "clip",
                    "optimizer",
                    "scheduler",
                ],
            },
            "clipping": {
                "serial_clip_factor": 0.1,
                "serial_global_gradient_norm": 0.5,
                "shared_parameter_identity": True,
                "zero_gradient_max_abs": 0.0,
            },
            "nonfinite": [
                {
                    "rank": rank,
                    "local_finite": False,
                    "global_finite": False,
                    "skipped": True,
                    "parameters_unchanged": True,
                }
                for rank in range(4)
            ],
            "bf16": bf16,
        },
    }


def _valid_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    reports = [_collective_report(index) for index in range(3)]
    reports.append(_semantic_report())
    reports.append(
        {
            "run_kind": "recovery",
            "repeat_index": 0,
            "metrics": {"scalar_sum": 10, "healthy_restart": True},
        }
    )
    empty = {f"GPU-{index}": [] for index in range(4)}
    audits = [
        {
            "run_id": run_id,
            "return_code": 0,
            "timed_out": False,
            "duration_seconds": 1.0,
            "failure_marker_observed": False,
            "residual_compute_pids": deepcopy(empty),
        }
        for run_id in (
            "collective-00",
            "collective-01",
            "collective-02",
            "semantic-00",
            "recovery-00",
        )
    ]
    audits.append(
        {
            "run_id": "failure-rank-00",
            "return_code": 1,
            "timed_out": False,
            "duration_seconds": 12.0,
            "failure_marker_observed": True,
            "residual_compute_pids": deepcopy(empty),
        }
    )
    cards = [
        {
            "uuid": f"GPU-{index}",
            "ecc_corrected_volatile": 0,
            "ecc_uncorrected_volatile": 0,
        }
        for index in range(4)
    ]
    device = {
        "before": {
            "compute_pids": deepcopy(empty),
            "cards": deepcopy(cards),
            "topology_sha256": "b" * 64,
        },
        "after": {
            "compute_pids": deepcopy(empty),
            "cards": deepcopy(cards),
            "topology_sha256": "b" * 64,
        },
    }
    return reports, audits, device


def test_g6_replay_accepts_complete_four_gpu_suite() -> None:
    reports, audits, device = _valid_inputs()
    metrics = validate_g6_report_set(reports, audits, device)

    assert metrics["collective_process_group_rebuilds"] == 3
    assert metrics["bf16_rank_count"] == 4
    assert metrics["no_sync_gradient_collectives"] == 1
    assert metrics["residual_process_count"] == 0


def test_g6_replay_rejects_collective_instability_and_shard_overlap() -> None:
    reports, audits, device = _valid_inputs()
    reports[0]["metrics"]["message_metrics"][0]["p95_seconds"] = 0.01
    with pytest.raises(Stage0G6Error, match="G6_COLLECTIVE_MEASUREMENT_THRESHOLD_FAILED"):
        validate_g6_report_set(reports, audits, device)

    reports, audits, device = _valid_inputs()
    semantic = next(item for item in reports if item["run_kind"] == "semantic")
    semantic["metrics"]["data_shards"]["observed_main"][1] = "sample-00"
    with pytest.raises(Stage0G6Error, match="G6_DATA_SHARD_DISJOINTNESS_FAILED"):
        validate_g6_report_set(reports, audits, device)


def test_g6_replay_rejects_transport_protocol_drift() -> None:
    reports, audits, device = _valid_inputs()
    reports[0]["metrics"]["nccl_p2p_disable"] = 0
    with pytest.raises(Stage0G6Error, match="G6_COLLECTIVE_TRANSPORT_PROTOCOL_FAILED"):
        validate_g6_report_set(reports, audits, device)


def test_g6_replay_rejects_no_sync_or_failure_cleanup_drift() -> None:
    reports, audits, device = _valid_inputs()
    semantic = next(item for item in reports if item["run_kind"] == "semantic")
    semantic["metrics"]["communication"]["no_sync_calls"] = 2
    with pytest.raises(Stage0G6Error, match="G6_NO_SYNC_COMMUNICATION_SEMANTICS_FAILED"):
        validate_g6_report_set(reports, audits, device)

    reports, audits, device = _valid_inputs()
    failure = next(item for item in audits if item["run_id"] == "failure-rank-00")
    failure["residual_compute_pids"]["GPU-1"] = [1234]
    with pytest.raises(Stage0G6Error, match="G6_CONTROLLED_RANK_FAILURE_FAILED"):
        validate_g6_report_set(reports, audits, device)
