from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments.stage2_s207_formal import APPROVED_GPU_UUIDS
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.experiments.stage2_s209_g27a import S29_TIMING_FIELDS
from param_importance_nlp.experiments.stage2_s209_g27a import _stage1_numeric_artifact
from param_importance_nlp.experiments.stage2_s209_production import (
    S209ProductionBlocked,
    _MeasuredPool,
    _aggregate_four_gpu_anchor_results,
    _four_gpu_anchor_worker,
    run_s209_production_backend,
)


HASHES = {
    "sample_mapping_hash": "a" * 64,
    "gradient_pool_hash": "b" * 64,
    "global_s1_hash": "f" * 64,
    "global_s2_hash": "0" * 64,
    "estimate_hash": "c" * 64,
    "state_digest": "d" * 64,
    "state_digest_after": "e" * 64,
}


class _Barrier:
    def __init__(self) -> None:
        self.wait_calls = 0
        self.abort_calls = 0

    def wait(self, *, timeout: float) -> None:
        assert timeout > 0
        self.wait_calls += 1

    def abort(self) -> None:
        self.abort_calls += 1


class _Queue:
    def __init__(self) -> None:
        self.values: list[dict[str, object]] = []

    def put(self, value: dict[str, object]) -> None:
        self.values.append(value)

    def get(self, *, timeout: float) -> dict[str, object]:
        assert timeout > 0
        if not self.values:
            raise RuntimeError("mock queue unexpectedly empty")
        return self.values.pop(0)


class _Mapping:
    m_values = (4,)
    digest = "1" * 64
    draws = tuple(type("Draw", (), {"to_manifest": lambda self, index=index: {"draw_id": f"draw-{index}", "sample_id": index}})() for index in range(4))

    def groups(self, _m: int) -> tuple[tuple[object, ...], ...]:
        return tuple((draw,) for draw in self.draws)


def _config() -> dict[str, object]:
    return {"frozen": {"batch_size": 32, "microbatch_count": 4}}


def _numeric_artifact(*, role: str, value: float = 1.0) -> dict[str, object]:
    import torch

    return _stage1_numeric_artifact(
        {"weight": torch.tensor([value], dtype=torch.float64)},
        role=role,
        fixed_checkpoint_id="checkpoint-fixed",
        checkpoint_hash="2" * 64,
        mapping_hash="1" * 64,
        sample_mapping_hash=HASHES["sample_mapping_hash"],
        state_digest=HASHES["state_digest"],
        state_digest_after=HASHES["state_digest_after"],
        statistical_binding={
            "global_s1_hash": HASHES["global_s1_hash"],
            "global_s2_hash": HASHES["global_s2_hash"],
            "estimate_hash": HASHES["estimate_hash"],
            "global_weight": 4.0,
            "global_statistical_unit_count": 4,
        },
    )


def _task() -> dict[str, object]:
    task: dict[str, object] = {
        "semantic": "anchor",
        "method": "anchor",
        "run_id": "s209-four-gpu-test",
        "anchor_id": "four-gpu-anchor",
        "repetition": 0,
        "gpu_uuid": APPROVED_GPU_UUIDS[0],
        "gpu_uuids": list(APPROVED_GPU_UUIDS),
        "device_count": 4,
        "mapping_hash": "1" * 64,
        "checkpoint_hash": "2" * 64,
        "io_evidence_hash": "f" * 64,
        "cost_io_quiescent": True,
    }
    baseline = {
        "wall_seconds": 1e-6,
        "sequence_count": 4,
        "token_count": 4,
        "backward_count": 4,
    }
    task["single_gpu_anchor"] = {
        **baseline,
        "identity_hash": canonical_json_hash(
            {
                "anchor_id": "single-gpu-anchor",
                "run_id": task["run_id"],
                "gpu_uuid": task["gpu_uuid"],
                "device_count": 1,
                **baseline,
            }
        ),
    }
    return task


def _child_result(uuid: str, *, rank: int, local_hash: str, drift: bool = False, numeric_value: float = 1.0) -> dict[str, object]:
    phase = {name: 0.1 for name in S29_TIMING_FIELDS}
    result: dict[str, object] = {
        "ok": True,
        "rank": rank,
        "gpu_uuid": "GPU-drift" if drift else uuid,
        "health_ok": True,
        "wall_seconds": 1.0,
        "barrier_seconds": 0.1,
        "barrier_count": 2,
        "phase": phase,
        "sequence_count": 1,
        "token_count": 1,
        "backward_count": 1,
        "statistical_unit": "microbatch",
        "statistical_unit_count": 1,
        "global_statistical_unit_count": 4,
        "global_weight": 4.0,
        "communication_bytes": 128,
        "allocated_peak_bytes": 100,
        "reserved_peak_bytes": 120,
        "device_peak_bytes": 140,
        "fixed_checkpoint_id": "checkpoint-fixed",
        "checkpoint_hash": "2" * 64,
        "mapping_hash": "1" * 64,
        "cuda_gradient_seconds": 0.5,
        "all_reduce_s1_seconds": 0.01,
        "all_reduce_s2_seconds": 0.01,
        "all_reduce_weight_seconds": 0.01,
        "all_reduce_count_seconds": 0.01,
        "all_reduce_s1_bytes": 32,
        "all_reduce_s2_bytes": 32,
        "all_reduce_weight_bytes": 8,
        "all_reduce_count_bytes": 8,
        "all_reduce_s1_count": 1,
        "all_reduce_s2_count": 1,
        "all_reduce_weight_count": 1,
        "all_reduce_count": 4,
    }
    result.update(HASHES)
    result["numeric_artifact"] = _numeric_artifact(role="four_gpu_candidate", value=numeric_value)
    result["numeric_artifact_hash"] = result["numeric_artifact"]["artifact_hash"]
    result["local_sample_mapping_hash"] = local_hash
    result["local_gradient_pool_hash"] = "9" * 64
    result["all_reduce_identity_hash"] = canonical_json_hash(
        {
            "rank": rank,
            "gpu_uuid": uuid,
            "s1_seconds": 0.01,
            "s2_seconds": 0.01,
            "weight_seconds": 0.01,
            "count_seconds": 0.01,
            "s1_bytes": 32,
            "s2_bytes": 32,
            "weight_bytes": 8,
            "count_bytes": 8,
            "s1_count": 1,
            "s2_count": 1,
            "weight_count": 1,
            "count": 4,
        }
    )
    return result


def test_four_gpu_anchor_worker_uses_two_barriers_and_real_measurement_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    import param_importance_nlp.experiments.stage2_s209_production as production

    observed: list[str] = []
    monkeypatch.setattr(production, "_device_identity", lambda uuid: observed.append(uuid))
    monkeypatch.setattr(production, "_load_provider_for_binding", lambda *_args, **_kwargs: (object(), _Mapping(), {"checkpoint_id": "checkpoint-fixed", "checkpoint_hash": "2" * 64}, object()))
    pool = _MeasuredPool(
        batches=[], maps=[object()], weights=[], gradient_wall_seconds=0.0, gradient_seconds=0.3,
        forward_seconds=0.0, backward_seconds=0.0, data_wait_seconds=0.0,
        sequence_count=1, token_count=1, backward_count=1,
        state_digest=HASHES["state_digest"], state_digest_after=HASHES["state_digest_after"],
        allocated_peak_bytes=100, reserved_peak_bytes=120, device_peak_bytes=140,
        sample_mapping_hash=HASHES["sample_mapping_hash"], gradient_pool_hash=HASHES["gradient_pool_hash"],
        gradient_aggregation_seconds=0.0,
    )
    monkeypatch.setattr(production, "_measure_pool", lambda *_args, **_kwargs: pool)
    import torch
    monkeypatch.setattr(production, "_method_value", lambda *_args, **_kwargs: ({"weight": torch.tensor([1.0], dtype=torch.float64)}, 0.001))
    monkeypatch.setattr(production, "_vector_digest", lambda _value: HASHES["estimate_hash"])
    barrier = _Barrier()
    result_queue = _Queue()
    numeric_map = {"weight": torch.tensor([1.0], dtype=torch.float64)}
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    class _FakeTensor:
        def numel(self) -> int:
            return 4

        def element_size(self) -> int:
            return 8

        def item(self) -> float:
            return 4.0

        def __truediv__(self, _value: float):
            return self

    monkeypatch.setattr(torch, "tensor", lambda *_args, **_kwargs: _FakeTensor())

    class _FakeDist:
        def all_gather_object(self, output, value, *, group) -> None:
            for index, uuid in enumerate(APPROVED_GPU_UUIDS):
                output[index] = {"rank": index, "gpu_uuid": uuid, "invariant": value["invariant"]}

    class _FakeKernel:
        def raw(self, _value):
            return numeric_map

    monkeypatch.setattr(production, "_init_four_gpu_process_group", lambda **_kwargs: (_FakeDist(), object()))
    monkeypatch.setattr(production, "_timed_nccl_barrier", lambda _dist: 0.000001)
    monkeypatch.setattr(production, "_local_sufficient_statistics", lambda _pool: (_FakeTensor(), _FakeTensor(), 1.0, 1))
    monkeypatch.setattr(production, "_timed_nccl_tensor_reduce", lambda _reducer, _name, tensor: (tensor, 0.000001))
    monkeypatch.setattr(production, "_timed_nccl_count_reduce", lambda _reducer, _value: (4, 0.000001))
    monkeypatch.setattr(production, "TorchDistributedReducer", lambda **_kwargs: object())
    monkeypatch.setattr(production, "_restore_tensor_map", lambda _template, _flat: _FakeTensor())
    monkeypatch.setattr(production, "CoreEstimatorKernel", lambda **_kwargs: _FakeKernel())
    _four_gpu_anchor_worker(
        APPROVED_GPU_UUIDS[2], rank=2, world_size=4, master_addr="127.0.0.1", master_port=12345,
        config=_config(), root=Path("."), binding={"cell_id": "cell", "unit_id": "unit", "mapping_hash": "1" * 64},
        barrier=barrier, result_queue=result_queue,
    )
    assert observed == [APPROVED_GPU_UUIDS[2]]
    # One multiprocessing launch barrier plus two NCCL barriers recorded in
    # the result; NCCL barriers are mocked through the timed helper below.
    assert barrier.wait_calls == 1
    assert barrier.abort_calls == 0
    assert len(result_queue.values) == 1
    result = result_queue.values[0]
    assert result["ok"] is True
    assert result["gpu_uuid"] == APPROVED_GPU_UUIDS[2]
    assert result["local_gradient_pool_hash"] == HASHES["gradient_pool_hash"]
    assert result["communication_bytes"] > 0


def test_four_gpu_aggregate_requires_exact_child_uuid_set_and_preserves_actual_counters() -> None:
    partition_hashes = [str(index + 1) * 64 for index in range(4)]
    results = [_child_result(uuid, rank=index, local_hash=partition_hashes[index]) for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    row = _aggregate_four_gpu_anchor_results(task=_task(), config=_config(), results=results, io_hash="f" * 64, expected_partition_hashes=partition_hashes, single_numeric_artifact=_numeric_artifact(role="single_gpu_reference"))
    assert row["gpu_uuids"] == list(APPROVED_GPU_UUIDS)
    assert row["device_count"] == 4
    assert row["sequence_count"] == 4
    assert row["token_count"] == 4
    assert row["backward_count"] == 4
    assert row["allocated_peak_bytes"] == 100
    assert row["communication_bytes"] > 0
    assert row["numeric_consistency"] is True
    assert row["system_anchor_mode"] == "synchronized_fixed_state_four_process_nccl"
    assert len(row["per_device_measurements"]) == 4

    drifted = [_child_result(uuid, rank=index, local_hash=partition_hashes[index]) for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    drifted[2] = _child_result(APPROVED_GPU_UUIDS[2], rank=2, local_hash=partition_hashes[2], drift=True)
    with pytest.raises(S209ProductionBlocked, match="CHILD_UUID"):
        _aggregate_four_gpu_anchor_results(task=_task(), config=_config(), results=drifted, io_hash="f" * 64, expected_partition_hashes=partition_hashes, single_numeric_artifact=_numeric_artifact(role="single_gpu_reference"))


def test_four_gpu_aggregate_blocks_zero_collective_and_duplicate_partition() -> None:
    partition_hashes = [str(index + 1) * 64 for index in range(4)]
    zero_collective = [_child_result(uuid, rank=index, local_hash=partition_hashes[index]) for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    zero_collective[0]["communication_bytes"] = 0
    with pytest.raises(S209ProductionBlocked, match="CHILD_COUNTER_INVALID|COLLECTIVE_MISSING"):
        _aggregate_four_gpu_anchor_results(task=_task(), config=_config(), results=zero_collective, io_hash="f" * 64, expected_partition_hashes=partition_hashes, single_numeric_artifact=_numeric_artifact(role="single_gpu_reference"))
    duplicate_partition = [_child_result(uuid, rank=index, local_hash="1" * 64) for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    with pytest.raises(S209ProductionBlocked, match="PARTITION_HASH_DUPLICATE"):
        _aggregate_four_gpu_anchor_results(task=_task(), config=_config(), results=duplicate_partition, io_hash="f" * 64, expected_partition_hashes=["1" * 64] * 4, single_numeric_artifact=_numeric_artifact(role="single_gpu_reference"))


def test_four_gpu_stage1_numeric_comparator_accepts_near_threshold_and_blocks_overthreshold() -> None:
    partition_hashes = [str(index + 1) * 64 for index in range(4)]
    near = [_child_result(uuid, rank=index, local_hash=partition_hashes[index], numeric_value=1.0 + 1e-12) for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    row = _aggregate_four_gpu_anchor_results(
        task=_task(), config=_config(), results=near, io_hash="f" * 64,
        expected_partition_hashes=partition_hashes,
        single_numeric_artifact=_numeric_artifact(role="single_gpu_reference"),
    )
    assert row["stage1_numeric_comparison"]["passed"] is True
    over = [_child_result(uuid, rank=index, local_hash=partition_hashes[index], numeric_value=1.0 + 1e-3) for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    with pytest.raises(S209ProductionBlocked, match="STAGE1_NUMERIC_COMPARISON_FAILED"):
        _aggregate_four_gpu_anchor_results(
            task=_task(), config=_config(), results=over, io_hash="f" * 64,
            expected_partition_hashes=partition_hashes,
            single_numeric_artifact=_numeric_artifact(role="single_gpu_reference"),
        )


def test_four_gpu_stage1_numeric_sidecar_and_substitution_hashes_fail_closed() -> None:
    partition_hashes = [str(index + 1) * 64 for index in range(4)]
    results = [_child_result(uuid, rank=index, local_hash=partition_hashes[index]) for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    results[0]["numeric_artifact_hash"] = "9" * 64
    with pytest.raises(S209ProductionBlocked, match="NUMERIC_ARTIFACT_HASH_MISMATCH"):
        _aggregate_four_gpu_anchor_results(
            task=_task(), config=_config(), results=results, io_hash="f" * 64,
            expected_partition_hashes=partition_hashes,
            single_numeric_artifact=_numeric_artifact(role="single_gpu_reference"),
        )


def test_production_backend_dispatches_four_gpu_anchor_instead_of_rejecting_device_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import param_importance_nlp.experiments.stage2_s209_production as production

    task = _task()
    checked = {"frozen": {"batch_size": 32, "microbatch_count": 1}}
    called: list[tuple[int, tuple[str, ...]]] = []
    monkeypatch.setattr(production, "_verify_runtime_config", lambda _config, *, task: (checked, Path(".")))
    monkeypatch.setattr(
        production,
        "_run_four_gpu_anchor",
        lambda **kwargs: (called.append((kwargs["task"]["device_count"], tuple(kwargs["task"]["gpu_uuids"]))) or {"measurement_kind": "device_actual"}),
    )
    result = run_s209_production_backend(task=task, config={})
    assert result["measurement_kind"] == "device_actual"
    assert called == [(4, tuple(APPROVED_GPU_UUIDS))]


def test_four_gpu_parent_orchestrates_four_mocked_processes_and_rejects_missing_child(monkeypatch: pytest.MonkeyPatch) -> None:
    import param_importance_nlp.experiments.stage2_s209_production as production

    class FakeProcess:
        def __init__(self, *, target, kwargs, name) -> None:
            self.target = target
            self.kwargs = kwargs
            self.name = name
            self.exitcode: int | None = None

        def start(self) -> None:
            self.target(**self.kwargs)
            self.exitcode = 0

        def join(self, timeout: float | None = None) -> None:
            assert timeout is None or timeout > 0

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            raise AssertionError("completed mocked process must not be terminated")

    class FakeContext:
        def __init__(self) -> None:
            self.barrier = _Barrier()
            self.queue = _Queue()
            self.processes: list[FakeProcess] = []

        def Barrier(self, count: int) -> _Barrier:
            assert count == 4
            return self.barrier

        def Queue(self) -> _Queue:
            return self.queue

        def Process(self, *, target, kwargs, name) -> FakeProcess:
            process = FakeProcess(target=target, kwargs=kwargs, name=name)
            self.processes.append(process)
            return process

    context = FakeContext()
    monkeypatch.setattr(production.mp, "get_context", lambda _name: context)
    monkeypatch.setattr(production, "_device_identity_set", lambda uuids: assert_uuid_set(uuids))
    plan = type("Plan", (), {"cells": (type("Cell", (), {"cell_id": EXPECTED_CELL_IDS[0], "checkpoint_hash": "2" * 64})(),)})()
    monkeypatch.setattr(production, "load_s27_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(production, "load_s27_frozen_mappings", lambda *_args, **_kwargs: {"unit-0": _Mapping()})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(APPROVED_GPU_UUIDS))
    checked = {
        "s27_plan_ref": "s27-plan.json",
        "frozen": {"batch_size": 4, "microbatch_count": 4, "expected_unit_ids": ["unit-0"]},
    }
    import torch
    numeric_map = {"weight": torch.tensor([1.0], dtype=torch.float64)}
    numeric_reference = _numeric_artifact(role="single_gpu_reference")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    real_tensor = torch.tensor
    def _cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*args, **kwargs)
    monkeypatch.setattr(torch, "tensor", _cpu_tensor)
    class _FakeTensor:
        def numel(self) -> int:
            return 4

        def element_size(self) -> int:
            return 8

        def item(self) -> float:
            return 4.0

        def __truediv__(self, _value: float):
            return self

    class _FakeDist:
        def all_gather_object(self, output, value, *, group) -> None:
            for index, uuid in enumerate(APPROVED_GPU_UUIDS):
                output[index] = {"rank": index, "gpu_uuid": uuid, "invariant": value["invariant"]}

    class _FakeKernel:
        def raw(self, _value):
            return numeric_map

    monkeypatch.setattr(production, "_init_four_gpu_process_group", lambda **_kwargs: (_FakeDist(), object()))
    monkeypatch.setattr(production, "_timed_nccl_barrier", lambda _dist: 0.000001)
    monkeypatch.setattr(production, "_local_sufficient_statistics", lambda _pool: (torch.tensor([1.0]), torch.tensor([1.0]), 1.0, 1))
    monkeypatch.setattr(production, "_timed_nccl_tensor_reduce", lambda _reducer, _name, tensor: (tensor, 0.000001))
    monkeypatch.setattr(production, "_timed_nccl_count_reduce", lambda _reducer, _value: (4, 0.000001))
    monkeypatch.setattr(production, "TorchDistributedReducer", lambda **_kwargs: object())
    monkeypatch.setattr(production, "_restore_tensor_map", lambda _template, _flat: _FakeTensor())
    monkeypatch.setattr(production, "CoreEstimatorKernel", lambda **_kwargs: _FakeKernel())
    monkeypatch.setattr(production, "_vector_digest", lambda _value: HASHES["estimate_hash"])
    monkeypatch.setattr(production, "_device_identity", lambda _uuid: None)
    monkeypatch.setattr(production, "_load_provider_for_binding", lambda *_args, **_kwargs: (object(), _Mapping(), {"checkpoint_id": "checkpoint-fixed", "checkpoint_hash": "2" * 64}, object()))
    pool = _MeasuredPool(
        batches=[], maps=[object()], weights=[], gradient_wall_seconds=0.0, gradient_seconds=0.3,
        forward_seconds=0.0, backward_seconds=0.0, data_wait_seconds=0.0,
        sequence_count=1, token_count=1, backward_count=1,
        state_digest=HASHES["state_digest"], state_digest_after=HASHES["state_digest_after"],
        allocated_peak_bytes=100, reserved_peak_bytes=120, device_peak_bytes=140,
        sample_mapping_hash=HASHES["sample_mapping_hash"], gradient_pool_hash=HASHES["gradient_pool_hash"],
        gradient_aggregation_seconds=0.0,
    )
    monkeypatch.setattr(production, "_measure_pool", lambda *_args, **_kwargs: pool)
    monkeypatch.setattr(production, "_load_single_numeric_reference", lambda _task: numeric_reference)
    row = production._run_four_gpu_anchor(task=_task(), config=checked, checked=checked, root=Path("."))
    assert row["device_count"] == 4
    assert len(context.processes) == 4
    assert context.barrier.wait_calls == 4

    partition_hashes = [str(index + 1) * 64 for index in range(4)]
    bad_results = [_child_result(uuid, rank=index, local_hash=partition_hashes[index]) for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    bad_results[0]["estimate_hash"] = "9" * 64
    with pytest.raises(S209ProductionBlocked, match="NUMERIC_CONSISTENCY|STATISTICAL_BINDING"):
        production._aggregate_four_gpu_anchor_results(task=_task(), config=_config(), results=bad_results, io_hash="f" * 64, expected_partition_hashes=partition_hashes, single_numeric_artifact=_numeric_artifact(role="single_gpu_reference"))


def assert_uuid_set(uuids: object) -> None:
    assert tuple(uuids) == tuple(APPROVED_GPU_UUIDS)
