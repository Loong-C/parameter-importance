"""Formal S1.11 producer bridge tests.

The authority replay itself is covered by ``test_stage1_s111_exit_gate``'s
full S1.1--S1.10-shaped fixture.  These tests exercise the producer-only
TaskArtifactStore boundary after that replay has admitted the released r4
roles; no local fixture can enter through ``evidence_ref``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json


def _formalizer():
    path = Path("ops/stage1/formalize_s1_11.py")
    spec = importlib.util.spec_from_file_location("s111_task_artifact_formalizer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_authority(tmp_path: Path, formalizer: object) -> dict[str, object]:
    authority = tmp_path / "evidence/stage1/s1-11-formal/3f18b04df8922be9894678ae4842bd999c7e8fd5/s1-11-r4-20260821"
    authority.mkdir(parents=True)
    roles: dict[str, dict[str, object]] = {}
    for kind in formalizer.S111_TASK_ARTIFACT_KINDS:
        role = {
            "schema_version": f"stage1-s1-11-{kind.replace('_', '-')}-v1",
            "status": "PASS",
            "task_id": formalizer.TASK_ID,
            "gate_id": formalizer.GATE_ID,
            "role_shape": {
                "measured": "PASS",
                "threshold": "immutable r4 role hash",
                "evidence": "full-authority-closure",
            },
        }
        role["artifact_hash"] = canonical_json_hash(role)
        roles[kind] = role
    source = authority / "full-authority-source.json"
    write_canonical_json(source, {"schema_version": "full-authority-source-v1", "status": "PASS"})
    index_path = authority / "index.json"
    write_canonical_json(index_path, {"schema_version": "full-authority-index-v1"})
    return {
        "index": {"artifact_hash": "a" * 64},
        "index_path": index_path,
        "index_dir": authority,
        "roles": roles,
        "role_file_sha256": {},
        "validation": {},
        "replay": {},
        "source_refs": [source.relative_to(tmp_path).as_posix()],
        "artifact_hashes": {
            "index": "a" * 64,
            **{kind: value["artifact_hash"] for kind, value in roles.items()},
        },
        "producer_source_sha256": {"ops/stage1/formalize_s1_11.py": "b" * 64},
        "dependency_index_sha256": {},
    }


def test_emit_rejects_r3_and_local_fixture_refs_before_reading_json(tmp_path: Path) -> None:
    formalizer = _formalizer()
    for ref in (
        "evidence/stage1/s1-11-formal/3f18b04df8922be9894678ae4842bd999c7e8fd5/s1-11-r3-20260821/index.json",
        "reports/stage1/cpu-evidence/stage_report.json",
        "fixtures/stage1/s1-11/index.json",
    ):
        with pytest.raises(formalizer.Stage1S111FormalError, match="R4_AUTHORITY_REF_REQUIRED"):
            formalizer.emit_task_artifacts(
                repository=Path.cwd(), evidence_root=tmp_path,
                output_dir="outputs/s111", evidence_ref=ref,
            )


def test_emit_is_idempotent_and_preserves_original_role_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    authority = _fake_authority(tmp_path, formalizer)
    monkeypatch.setattr(formalizer, "_emit_load_r4", lambda **_: authority)

    first = formalizer.emit_task_artifacts(
        repository=Path.cwd(), evidence_root=tmp_path,
        output_dir="outputs/s111",
    )
    second = formalizer.emit_task_artifacts(
        repository=Path.cwd(), evidence_root=tmp_path,
        output_dir="outputs/s111",
    )
    assert first == second
    assert set(first["commit_refs"]) == set(formalizer.S111_TASK_ARTIFACT_KINDS)
    for kind, ref in first["commit_refs"].items():
        commit = load_canonical_json(tmp_path / ref)
        payload = load_canonical_json(tmp_path / commit["object_ref"])["payload"]
        assert payload == authority["roles"][kind]
        assert commit["formal_eligible"] is True


def test_emit_fails_closed_on_role_commit_config_and_symlink_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    authority = _fake_authority(tmp_path, formalizer)
    monkeypatch.setattr(formalizer, "_emit_load_r4", lambda **_: authority)
    result = formalizer.emit_task_artifacts(
        repository=Path.cwd(), evidence_root=tmp_path,
        output_dir="outputs/s111",
    )

    # A changed released role cannot be silently republished over an existing
    # commit, even when the replacement is internally self-hashed.
    replacement = dict(authority["roles"]["stage_report"])
    replacement["role_shape"] = {"tampered": True}
    replacement["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in replacement.items() if key != "artifact_hash"}
    )
    authority["roles"]["stage_report"] = replacement
    with pytest.raises(formalizer.Stage1S111FormalError, match="EXISTING_PAYLOAD_DRIFT"):
        formalizer.emit_task_artifacts(
            repository=Path.cwd(), evidence_root=tmp_path,
            output_dir="outputs/s111",
        )

    # Config drift is checked before the store can discover/reuse commits.
    config_path = tmp_path / result["config_ref"]
    config = load_canonical_json(config_path)
    config["config_kind"] = "training-config-forbidden"
    write_canonical_json(config_path, config)
    authority["roles"]["stage_report"] = replacement
    with pytest.raises(formalizer.Stage1S111FormalError, match="PRODUCER_CONFIG_IDENTITY_DRIFT"):
        formalizer.emit_task_artifacts(
            repository=Path.cwd(), evidence_root=tmp_path,
            output_dir="outputs/s111",
        )

    # Existing output roots are never followed through a symlink.
    linked = tmp_path / "outputs/linked"
    target = tmp_path / "outside"
    target.mkdir()
    try:
        linked.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this Windows runner")
    with pytest.raises(formalizer.Stage1S111FormalError, match="SYMLINK_FORBIDDEN"):
        formalizer.emit_task_artifacts(
            repository=Path.cwd(), evidence_root=tmp_path,
            output_dir="outputs/linked",
        )
