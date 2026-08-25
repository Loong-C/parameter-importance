from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.runtime.tensor_bundle import publish_tensor_bundle
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.experiments.stage2_s207_formal import (
    APPROVED_GPU_UUIDS,
    S27CellPlan,
)
from param_importance_nlp.experiments.stage2_s207_runner import (
    _GradientAuditProxy,
    S27ExecutionBlocked,
    S27RetryPolicy,
    build_s27_worker_command,
    load_s27_reference_views,
)
from param_importance_nlp.experiments.stage2_formal import _vector_digest


def _cell(reference_ref: str, reference_hash: str, gate_ref: str, gate_hash: str) -> S27CellPlan:
    return S27CellPlan(
        cell_id=EXPECTED_CELL_IDS[0],
        model_id="pythia-14m",
        training_stage="initialization",
        checkpoint_ref="s203/checkpoint.json",
        checkpoint_hash="1" * 64,
        checkpoint_id="checkpoint-init",
        reference_ref=reference_ref,
        reference_hash=reference_hash,
        reference_gate_ref=gate_ref,
        reference_gate_hash=gate_hash,
        expected_unit_ids=("unit-0001",),
        assigned_gpu_uuid=APPROVED_GPU_UUIDS[0],
    )


def test_reference_loader_consumes_formal_candidate_bundle_and_external_g23(tmp_path: Path) -> None:
    vector = {"layer.weight": np.array([1.0, -2.0], dtype=np.float64)}
    bundle = publish_tensor_bundle(
        tmp_path / "reference-output" / "tensor-bundles" / "reference-final",
        {
            "bias_reference": vector,
            "cross_reference": vector,
            "ranking_reference": vector,
            "sequence_variance": vector,
            "numerical_diagnostics": {"schema_version": "stage2-reference-numerical-diagnostics-v1"},
        },
    )
    gate = GateRecord(
        gate_id="stage2.G2.3",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00+00:00",
        measured={"cell_count": 1},
        threshold={"status": "PASS"},
        evidence_refs=("reference.json",),
    ).to_dict()
    write_canonical_json(tmp_path / "g23.json", gate)
    reference_body: dict[str, object] = {
        "schema_version": "reference-result-v1",
        "reference_id": "s204-independent-reference-init",
        "bias_reference_hash": _vector_digest(vector),
        "cross_reference_hash": _vector_digest(vector),
        "ranking_reference_hash": _vector_digest(vector),
        "sample_count_a": 2,
        "sample_count_b": 2,
        "block_size": 1,
        "registry_hash": "2" * 64,
        "scope": "formal",
        "formal_eligible": False,
        "metadata": {"independent": True},
        "tensor_bundle_ref": "tensor-bundles/reference-final",
        "tensor_bundle_manifest_hash": bundle.manifest_sha256,
    }
    reference = {**reference_body, "artifact_hash": canonical_json_hash(reference_body)}
    write_canonical_json(tmp_path / "reference.json", reference)
    cell = _cell(
        "reference.json",
        str(reference["artifact_hash"]),
        "g23.json",
        str(gate["artifact_hash"]),
    )
    views = load_s27_reference_views(
        tmp_path,
        cell,
        expected_registry_hash="2" * 64,
        reference_output_root_ref="reference-output",
    )
    assert tuple(views) == ("bias", "cross", "ranking")


def test_retry_policy_and_worker_command_are_fail_closed() -> None:
    with pytest.raises(S27ExecutionBlocked, match="SINGLE_ATTEMPT"):
        S27RetryPolicy(max_attempts=2)
    command = build_s27_worker_command(
        python="python",
        launcher_script="ops/stage2/run_s207_formal.py",
        data_root="/data-root",
        plan_ref="s207/plan.json",
        run_root="s207/run",
        run_id="s207-run",
        cell_id=EXPECTED_CELL_IDS[0],
        gpu_uuid=APPROVED_GPU_UUIDS[0],
        materialization_index_ref="s204/index.json",
        execution_evidence_ref="s202/formal-execution.json",
    )
    assert "--worker" in command
    assert "--gpu-uuid" in command
    assert "--draw" not in command


def test_weighted_mean_materializes_cuda_like_values_before_numpy() -> None:
    class CudaLikeTensor:
        def __init__(self, values: list[float], *, detached: bool = False, on_cpu: bool = False) -> None:
            self._values = values
            self.detached = detached
            self.on_cpu = on_cpu

        def __array__(self, dtype: object = None) -> object:
            raise TypeError("can't convert cuda:0 device type tensor to numpy")

        def detach(self) -> "CudaLikeTensor":
            return CudaLikeTensor(self._values, detached=True, on_cpu=self.on_cpu)

        def cpu(self) -> "CudaLikeTensor":
            return CudaLikeTensor(self._values, detached=self.detached, on_cpu=True)

        def numpy(self) -> np.ndarray:
            assert self.detached
            assert self.on_cpu
            return np.asarray(self._values, dtype=np.float32)

    vectors = [
        {"weight": CudaLikeTensor([1.0, 3.0])},
        {"weight": CudaLikeTensor([5.0, 7.0])},
    ]
    result = _GradientAuditProxy._weighted_vector_mean(vectors, [1.0, 3.0])
    np.testing.assert_allclose(result["weight"], np.asarray([4.0, 6.0], dtype=np.float64))
    assert result["weight"].dtype == np.float64
