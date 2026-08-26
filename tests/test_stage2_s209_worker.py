from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage2_s207_formal import APPROVED_GPU_UUIDS
from param_importance_nlp.experiments.stage2_s209_g27a import (
    S29_COUNT_FIELDS,
    S29_METHODS,
    S29_SHARED_POOL_SCHEMA,
    S29_SHARED_RUN_SCHEMA,
    S29_TIMING_FIELDS,
)
from param_importance_nlp.experiments.stage2_s209_worker import (
    S209_WORKER_CONFIG_SCHEMA,
    S209WorkerBlocked,
    run_s209_profiler_worker,
)
from param_importance_nlp.experiments.stage2_s209_production import (
    S209ProductionBlocked,
    _device_identity,
    _row_bytes,
)


PLAN_HASH = "a" * 64
PAIR_HASH = "b" * 64
POOL_HASH = "c" * 64
SAMPLE_HASH = "d" * 64
GRADIENT_HASH = "e" * 64
UUID = APPROVED_GPU_UUIDS[0]


def _environment(*, semantic: str = "isolated_estimator_cost", method: str = "raw") -> dict[str, str]:
    value = {
        "S29_SEMANTIC": semantic,
        "S29_METHOD": method,
        "S29_RUN_ID": "s209-worker-test",
        "S29_PLAN_HASH": PLAN_HASH,
        "S29_ANCHOR_ID": "anchor-0",
        "S29_REPETITION": "0",
        "S29_GPU_UUIDS": UUID,
        "CUDA_VISIBLE_DEVICES": UUID,
    }
    if semantic == "scientific_equal_sample_cost":
        value.update(
            {
                "S29_METHOD": "shared",
                "S29_SHARED_RUN_SCHEMA": S29_SHARED_RUN_SCHEMA,
                "S29_SHARED_POOL_SCHEMA": S29_SHARED_POOL_SCHEMA,
                "S29_PAIRED_RUN_ID": "paired-test",
                "S29_PAIRED_RUN_IDENTITY_HASH": PAIR_HASH,
                "S29_SHARED_METHOD_ORDER": json.dumps(["double", "raw", "u"], separators=(",", ":")),
                "S29_SHARED_POOL_REF": "shared-pools/paired-test.json",
            }
        )
    return value


def _config(run_id: str = "s209-worker-test") -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": S209_WORKER_CONFIG_SCHEMA,
        "formal_eligible": True,
        "run_id": run_id,
        "backend_context": {
            "checkpoint_ref": "g2.6/approved-checkpoint.safetensors",
            "data_ref": "g2.6/approved-data-manifest.json",
        },
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _row(task: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "measurement_kind": "actual",
        "measured": True,
        "semantic": task["semantic"],
        "method": task["method"],
        "run_id": task["run_id"],
        "anchor_id": task["anchor_id"],
        "repetition": task["repetition"],
        "gpu_uuid": task["gpu_uuid"],
        "device_count": task["device_count"],
        "health_ok": True,
        "cost_io_quiescent": True,
        "wall_seconds": 10.0,
        "allocated_peak_bytes": 100,
        "reserved_peak_bytes": 120,
        "device_peak_bytes": 140,
        "sequence_count": 32,
        "token_count": 1024,
        "backward_count": 1,
        "communication_bytes": 0,
        "output_bytes": 4096,
    }
    value.update({name: 0.5 for name in S29_TIMING_FIELDS})
    value["forward_seconds"] = 2.0
    value["backward_seconds"] = 3.0
    value["communication_seconds"] = 0.0
    value["write_seconds"] = 1.0
    value.update({name: value[name] for name in S29_COUNT_FIELDS})
    return value


def _shared_bundle(task: dict[str, object]) -> dict[str, object]:
    order = list(task["shared_method_order"])
    pool: dict[str, object] = {
        "schema_version": S29_SHARED_POOL_SCHEMA,
        "paired_run_id": task["paired_run_id"],
        "paired_run_identity_hash": task["paired_run_identity_hash"],
        "measurement_plan_hash": task["measurement_plan_hash"],
        "matrix_hash": "1" * 64,
        "raw_manifest_hash": "2" * 64,
        "source_raw_run_id": "s207-formal-run",
        "anchor_id": task["anchor_id"],
        "repetition": task["repetition"],
        "gpu_uuid": task["gpu_uuid"],
        "device_count": 1,
        "batch_size": 32,
        "microbatch_count": 16,
        "method_order": order,
        "pool_id": POOL_HASH,
        "sample_mapping_hash": SAMPLE_HASH,
        "gradient_pool_hash": GRADIENT_HASH,
        "sequence_count": 32,
        "token_count": 1024,
        "backward_count": 1,
        "cost_io_quiescent": True,
        "shared_pool_ref": task["shared_pool_ref"],
    }
    pool["artifact_hash"] = canonical_json_hash(pool)
    rows = []
    for method in order:
        row = _row({**task, "method": method})
        row.update(
            {
                "paired_run_id": task["paired_run_id"],
                "paired_run_identity_hash": task["paired_run_identity_hash"],
                "measurement_plan_hash": task["measurement_plan_hash"],
                "shared_pool_id": pool["pool_id"],
                "shared_pool_artifact_hash": pool["artifact_hash"],
                "shared_pool_ref": pool["shared_pool_ref"],
                "shared_sample_mapping_hash": pool["sample_mapping_hash"],
                "shared_gradient_pool_hash": pool["gradient_pool_hash"],
                "shared_method_order": order,
                "shared_method_index": order.index(method),
                "shared_sample_sequence_count": pool["sequence_count"],
                "shared_sample_token_count": pool["token_count"],
            }
        )
        rows.append(row)
    return {
        "schema_version": S29_SHARED_RUN_SCHEMA,
        "semantic": task["semantic"],
        "paired_run_id": task["paired_run_id"],
        "paired_run_identity_hash": task["paired_run_identity_hash"],
        "run_id": task["run_id"],
        "measurement_plan_hash": task["measurement_plan_hash"],
        "anchor_id": task["anchor_id"],
        "repetition": task["repetition"],
        "gpu_uuid": task["gpu_uuid"],
        "device_count": 1,
        "method_order": order,
        "methods": list(S29_METHODS),
        "shared_pool": pool,
        "rows": rows,
    }


def _method_backend(*, task: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    return _row(task)


def _shared_backend(*, task: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    return _shared_bundle(task)


def test_worker_method_entrypoint_requires_actual_measured_row(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = _environment()
    result = run_s209_profiler_worker(backend=_method_backend, config=_config(), environment=environment)
    assert result["measurement_kind"] == "actual"
    assert result["gpu_uuid"] == UUID
    monkeypatch.setitem(environment, "CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(S209WorkerBlocked, match="GPU_UUID_BINDING_DRIFT"):
        run_s209_profiler_worker(backend=_method_backend, config=_config(), environment=environment)


def test_worker_shared_entrypoint_keeps_one_paired_pool_and_three_methods() -> None:
    result = run_s209_profiler_worker(
        backend=_shared_backend,
        config=_config(),
        environment=_environment(semantic="scientific_equal_sample_cost", method="shared"),
    )
    assert result["methods"] == list(S29_METHODS)
    assert len(result["rows"]) == 3
    assert result["shared_pool"]["cost_io_quiescent"] is True


def test_worker_rejects_label_only_or_nonquiet_output() -> None:
    def fake_backend(*, task: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        result = _row(task)
        result["measurement_kind"] = "label_only"
        return result

    with pytest.raises(S209WorkerBlocked, match="MEASUREMENT_MARKER_REQUIRED"):
        run_s209_profiler_worker(backend=fake_backend, config=_config(), environment=_environment())


def test_worker_cli_stdout_is_exactly_one_json_object(tmp_path: Path) -> None:
    config_path = tmp_path / "worker-config.json"
    write_canonical_json(config_path, _config())
    environment = _environment()
    command = [
        sys.executable,
        "ops/stage2/run_s209_profiler_worker.py",
        "--backend",
        "tests.test_stage2_s209_worker:_method_backend",
        "--config",
        str(config_path),
    ]
    completed = subprocess.run(command, env={**__import__("os").environ, **environment}, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    decoded = json.loads(completed.stdout)
    assert isinstance(decoded, dict)
    assert completed.stdout.count("\n") == 1


def test_repository_production_backend_has_no_cpu_or_synthetic_fallback() -> None:
    # The development host is CPU-only by contract.  A production invocation
    # must stop before constructing any result instead of returning a fixture
    # measurement when CUDA is absent.
    with pytest.raises(S209ProductionBlocked, match="CUDA_SINGLE_DEVICE_REQUIRED"):
        _device_identity(UUID)


def test_output_byte_counter_is_content_derived() -> None:
    row = {"measurement_kind": "device_actual", "measured": True, "sequence_count": 1}
    observed = _row_bytes(row)
    assert observed == len(__import__("param_importance_nlp.contracts.jsonio", fromlist=["canonical_json_bytes"]).canonical_json_bytes(row))
    assert row["output_bytes"] == observed
