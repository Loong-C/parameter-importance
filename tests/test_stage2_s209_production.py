from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.experiments.stage2_s207_formal import APPROVED_GPU_UUIDS
from param_importance_nlp.experiments.stage2_s209_g27a import S29_TIMING_FIELDS
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
    m_values = (1,)
    digest = "1" * 64

    def groups(self, _m: int) -> tuple[tuple[object, ...], ...]:
        return ((object(),),)


def _config() -> dict[str, object]:
    return {"frozen": {"batch_size": 32, "microbatch_count": 1}}


def _task() -> dict[str, object]:
    return {
        "semantic": "anchor",
        "method": "anchor",
        "run_id": "s209-four-gpu-test",
        "anchor_id": "four-gpu-anchor",
        "repetition": 0,
        "gpu_uuid": APPROVED_GPU_UUIDS[0],
        "gpu_uuids": list(APPROVED_GPU_UUIDS),
        "device_count": 4,
        "io_evidence_hash": "f" * 64,
        "cost_io_quiescent": True,
    }


def _child_result(uuid: str, *, drift: bool = False) -> dict[str, object]:
    phase = {name: 0.1 for name in S29_TIMING_FIELDS}
    return {
        "ok": True,
        "gpu_uuid": "GPU-drift" if drift else uuid,
        "health_ok": True,
        "wall_seconds": 1.0,
        "phase": phase,
        "sequence_count": 32,
        "token_count": 1024,
        "backward_count": 1,
        "communication_bytes": 0,
        "allocated_peak_bytes": 100,
        "reserved_peak_bytes": 120,
        "device_peak_bytes": 140,
        **HASHES,
        "fixed_checkpoint_id": "checkpoint-fixed",
        "cuda_gradient_seconds": 0.5,
    }


def test_four_gpu_anchor_worker_uses_two_barriers_and_real_measurement_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    import param_importance_nlp.experiments.stage2_s209_production as production

    observed: list[str] = []
    monkeypatch.setattr(production, "_device_identity", lambda uuid: observed.append(uuid))
    monkeypatch.setattr(production, "_load_provider_for_binding", lambda *_args, **_kwargs: (object(), _Mapping(), {"checkpoint_id": "checkpoint-fixed"}, object()))
    pool = _MeasuredPool(
        batches=[], maps=[], weights=[], gradient_wall_seconds=0.0, gradient_seconds=0.3,
        forward_seconds=0.0, backward_seconds=0.0, data_wait_seconds=0.0,
        sequence_count=32, token_count=1024, backward_count=1,
        state_digest=HASHES["state_digest"], state_digest_after=HASHES["state_digest_after"],
        allocated_peak_bytes=100, reserved_peak_bytes=120, device_peak_bytes=140,
        sample_mapping_hash=HASHES["sample_mapping_hash"], gradient_pool_hash=HASHES["gradient_pool_hash"],
        gradient_aggregation_seconds=0.0,
    )
    monkeypatch.setattr(production, "_measure_pool", lambda *_args, **_kwargs: pool)
    monkeypatch.setattr(production, "_method_value", lambda *_args, **_kwargs: (object(), 0.0))
    monkeypatch.setattr(production, "_vector_digest", lambda _value: HASHES["estimate_hash"])
    barrier = _Barrier()
    result_queue = _Queue()
    _four_gpu_anchor_worker(
        APPROVED_GPU_UUIDS[2], config=_config(), root=Path("."), binding={"cell_id": "cell", "unit_id": "unit"},
        barrier=barrier, result_queue=result_queue,
    )
    assert observed == [APPROVED_GPU_UUIDS[2]]
    assert barrier.wait_calls == 2
    assert barrier.abort_calls == 0
    assert len(result_queue.values) == 1
    result = result_queue.values[0]
    assert result["ok"] is True
    assert result["gpu_uuid"] == APPROVED_GPU_UUIDS[2]
    assert result["gradient_pool_hash"] == HASHES["gradient_pool_hash"]
    assert result["communication_bytes"] == 0


def test_four_gpu_aggregate_requires_exact_child_uuid_set_and_preserves_actual_counters() -> None:
    results = [_child_result(uuid) for uuid in APPROVED_GPU_UUIDS]
    row = _aggregate_four_gpu_anchor_results(task=_task(), config=_config(), results=results, io_hash="f" * 64)
    assert row["gpu_uuids"] == list(APPROVED_GPU_UUIDS)
    assert row["device_count"] == 4
    assert row["sequence_count"] == 128
    assert row["token_count"] == 4096
    assert row["backward_count"] == 4
    assert row["allocated_peak_bytes"] == 100
    assert row["communication_bytes"] == 0
    assert row["numeric_consistency"] is True
    assert row["system_anchor_mode"] == "synchronized_fixed_state_four_process"
    assert len(row["per_device_measurements"]) == 4

    drifted = [_child_result(uuid) for uuid in APPROVED_GPU_UUIDS]
    drifted[2] = _child_result(APPROVED_GPU_UUIDS[2], drift=True)
    with pytest.raises(S209ProductionBlocked, match="CHILD_UUID"):
        _aggregate_four_gpu_anchor_results(task=_task(), config=_config(), results=drifted, io_hash="f" * 64)


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
    monkeypatch.setattr(production, "load_s27_plan", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(production, "load_s27_frozen_mappings", lambda *_args, **_kwargs: {"unit-0": _Mapping()})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(APPROVED_GPU_UUIDS))
    checked = {
        "s27_plan_ref": "s27-plan.json",
        "frozen": {"batch_size": 32, "microbatch_count": 1, "expected_unit_ids": ["unit-0"]},
    }
    monkeypatch.setattr(production, "_device_identity", lambda _uuid: None)
    monkeypatch.setattr(production, "_load_provider_for_binding", lambda *_args, **_kwargs: (object(), _Mapping(), {"checkpoint_id": "checkpoint-fixed"}, object()))
    pool = _MeasuredPool(
        batches=[], maps=[], weights=[], gradient_wall_seconds=0.0, gradient_seconds=0.3,
        forward_seconds=0.0, backward_seconds=0.0, data_wait_seconds=0.0,
        sequence_count=32, token_count=1024, backward_count=1,
        state_digest=HASHES["state_digest"], state_digest_after=HASHES["state_digest_after"],
        allocated_peak_bytes=100, reserved_peak_bytes=120, device_peak_bytes=140,
        sample_mapping_hash=HASHES["sample_mapping_hash"], gradient_pool_hash=HASHES["gradient_pool_hash"],
        gradient_aggregation_seconds=0.0,
    )
    monkeypatch.setattr(production, "_measure_pool", lambda *_args, **_kwargs: pool)
    monkeypatch.setattr(production, "_method_value", lambda *_args, **_kwargs: (object(), 0.0))
    monkeypatch.setattr(production, "_vector_digest", lambda _value: HASHES["estimate_hash"])
    row = production._run_four_gpu_anchor(task=_task(), config=checked, checked=checked, root=Path("."))
    assert row["device_count"] == 4
    assert len(context.processes) == 4
    assert context.barrier.wait_calls == 8

    bad_results = [_child_result(uuid) for uuid in APPROVED_GPU_UUIDS]
    bad_results[0]["estimate_hash"] = "9" * 64
    with pytest.raises(S209ProductionBlocked, match="NUMERIC_CONSISTENCY"):
        production._aggregate_four_gpu_anchor_results(task=_task(), config=_config(), results=bad_results, io_hash="f" * 64)


def assert_uuid_set(uuids: object) -> None:
    assert tuple(uuids) == tuple(APPROVED_GPU_UUIDS)
