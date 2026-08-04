"""Formal G0--G2 bootstrap validates source evidence before task projection."""

from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts import load_canonical_json
from param_importance_nlp.runtime import load_committed_task_artifact
from param_importance_nlp.stage0_bootstrap import (
    Stage0BootstrapError,
    Stage0RuntimeSnapshot,
    Stage0SourceBinding,
    bootstrap_formal_stage0,
    validate_existing_g0_g2,
)


ROOT = Path(__file__).resolve().parents[1]
HEAD = "a" * 40
GPU_UUIDS = (
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
    "GPU-5a81500d-5e9c-b0d7-5607-fdfdaab65ff4",
)


def _binding() -> Stage0SourceBinding:
    return Stage0SourceBinding(ROOT, HEAD, "feat/stage0-completion", True)


def _snapshot(tmp_path: Path, **flags: bool) -> Stage0RuntimeSnapshot:
    values = {
        "git_verified": True,
        "server_verified": True,
        "wheelhouse_verified": True,
        "cuda_verified": True,
        "nccl_verified": True,
    }
    values.update(flags)
    return Stage0RuntimeSnapshot(
        checked_at="2026-08-03T16:00:00Z",
        hostname="sophgo13",
        boot_id="1dc04123-4945-4d8b-abcb-27fb84725526",
        kernel="6.8.0-136-generic",
        data_root=tmp_path.as_posix(),
        python_prefix=(
            "/home/sophgo13/cjl/storage/parameter-importance/envs/"
            "parameter-importance-stage0-1bd963c65f75"
        ),
        python_version="3.12.3",
        torch_version="2.12.1+cu126",
        torch_cuda_runtime="12.6",
        cuda_device_count=4,
        allowed_gpu_uuids=GPU_UUIDS,
        **values,
    )


def test_existing_reports_validate_against_current_snapshot(tmp_path: Path) -> None:
    validated = validate_existing_g0_g2(
        binding=_binding(), snapshot=_snapshot(tmp_path)
    )
    assert set(validated) == {"g0_c", "g0_g", "g1", "g2"}
    assert validated["g0_g"]["gpu_count"] == 4
    assert validated["g1"]["persistence_satisfaction"] == (
        "TIME_BOUNDED_RISK_ACCEPTANCE"
    )


def test_stale_boot_or_expired_persistence_fails_closed(tmp_path: Path) -> None:
    stale = _snapshot(tmp_path)
    object.__setattr__(stale, "boot_id", "different-boot")
    with pytest.raises(Stage0BootstrapError, match="G0_G_CURRENT_RUNTIME_MISMATCH"):
        validate_existing_g0_g2(binding=_binding(), snapshot=stale)

    expired = _snapshot(tmp_path)
    object.__setattr__(expired, "checked_at", "2026-08-19T00:00:00Z")
    with pytest.raises(Stage0BootstrapError, match="G1_NOT_CURRENT_PASS"):
        validate_existing_g0_g2(binding=_binding(), snapshot=expired)


def test_bootstrap_publishes_strict_environment_and_s01_s03_chain(
    tmp_path: Path,
) -> None:
    result = bootstrap_formal_stage0(
        binding=_binding(),
        data_root=tmp_path,
        snapshot=_snapshot(tmp_path),
    )

    assert result.environment.frozen_contract_stages == frozenset({0})
    assert {"git", "server", "wheelhouse", "cuda", "nccl"}.issubset(
        result.environment.capabilities
    )
    assert set(result.task_output_refs) == {
        "stage0.01_baseline_and_safety",
        "stage0.02_storage_and_layout",
        "stage0.03_runtime_and_dependencies",
    }
    for task_id, refs in result.task_output_refs.items():
        for reference in refs.values():
            loaded = load_committed_task_artifact(
                tmp_path, reference, require_formal=True
            )
            assert loaded.identity.task_id == task_id
            assert loaded.payload["core_evidence"]["local_fixture_executed"] is False

    index = load_canonical_json(tmp_path / result.index_ref)
    assert index["next_task_id"] == "stage0.04_assets_and_manifests"
    assert len(index["next_input_refs"]) == 3
    assert len(index["artifact_hash"]) == 64


def test_blocked_capability_is_published_but_not_claimed(tmp_path: Path) -> None:
    result = bootstrap_formal_stage0(
        binding=_binding(),
        data_root=tmp_path,
        snapshot=_snapshot(tmp_path, cuda_verified=False, nccl_verified=False),
    )
    assert "cuda" not in result.environment.capabilities
    assert "nccl" not in result.environment.capabilities
    cuda_ref = result.environment.evidence_refs["capability_cuda"]
    loaded = load_committed_task_artifact(tmp_path, cuda_ref, require_formal=True)
    assert loaded.payload["status"] == "BLOCKED"
