from __future__ import annotations

import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts.g21_formal_handoff import (
    ALLOWED_DEVICES,
    AUTH_HASH,
    EXCLUDED_PCI,
    G21FormalHandoffError,
    build_g21_formal_handoff,
    load_g21_formal_handoff,
)
from param_importance_nlp.contracts.jsonio import write_canonical_json


def _payload() -> dict[str, object]:
    commit = "a" * 40
    return {
        "schema_version": "stage2-s2.2-g2.1-formal-handoff-v1",
        "status": "PASS",
        "gate_id": "stage2.G2.1",
        "producer_commit": commit,
        "execution_commit": commit,
        "consumer_commit": commit,
        "authorization": {
            "ref": "reports/stage2/s2.2/formal-authorization-amendment-20260823.json",
            "artifact_hash": AUTH_HASH,
            "user_authorization_original": "允许 Stage 2 结束前继续使用单副本存储，排除故障 GPU 0000:50:00.0，继续执行",
            "issued_at": "2026-08-23T16:30:00+08:00",
            "expires_at": "Stage 2 exit",
            "scope": ["reproducible_stage0_artifacts", "reproducible_stage2_artifacts"],
            "single_copy_accepted": True,
            "excluded_pci_bus_ids": [EXCLUDED_PCI],
            "excluded_non_reproducible_human_evidence": True,
        },
        "current_gpu_smoke": {
            "ref": "evidence/stage2/s202/current-gpu-smoke/ab0a530/current-20260823-02/report.json",
            "sha256": "b" * 64,
            "schema_version": "stage2-s202-current-gpu-smoke-v1",
            "status": "PASS",
            "atomic_publication": True,
            "excluded_pci_bus_ids": [EXCLUDED_PCI],
            "excluded_scheduled": False,
            "allowed_devices": [
                {"pci_bus_id": pci, "uuid": uuid} for pci, uuid in ALLOWED_DEVICES
            ],
        },
        "historical_stage0": {
            role: {"ref": f"evidence/stage0/{role}.json", "sha256": "c" * 64, "producer_commit": commit, "status": "PASS"}
            for role in ("g5", "g6", "g10")
        },
        "stage1_g1_exit": {
            "ref": "evidence/stage1/index.json",
            "sha256": "d" * 64,
            "producer_commit": commit,
            "identity": "3f18b04df8922be9894678ae4842bd999c7e8fd5",
            "status": "PASS",
        },
        "checks": {
            name: True
            for name in (
                "producer_identity", "execution_identity", "consumer_identity",
                "authorization_scope", "stage0_g5_g6_g10_identity", "stage1_identity",
                "gpu_exclusion", "atomic_publication", "hash_verified", "replay_verified",
                "loader_verified",
            )
        },
    }


def test_builder_and_loader_bind_hash_and_exclusion(tmp_path: Path) -> None:
    payload = _payload()
    evidence = build_g21_formal_handoff(payload)
    path = tmp_path / "g2.1.json"
    write_canonical_json(path, evidence)
    loaded = load_g21_formal_handoff(path)
    assert loaded["artifact_hash"] == evidence["artifact_hash"]
    assert loaded["current_gpu_smoke"]["excluded_scheduled"] is False


def test_loader_rejects_gpu_exclusion_drift(tmp_path: Path) -> None:
    evidence = build_g21_formal_handoff(_payload())
    evidence["current_gpu_smoke"]["excluded_pci_bus_ids"] = ["0000:51:00.0"]  # type: ignore[index]
    write_canonical_json(tmp_path / "tampered.json", evidence)
    with pytest.raises(G21FormalHandoffError):
        load_g21_formal_handoff(tmp_path / "tampered.json")
