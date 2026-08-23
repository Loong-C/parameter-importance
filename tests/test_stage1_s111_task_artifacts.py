"""Real-loader and recovery tests for the S1.11 formal producer boundary."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact


def _formalizer():
    path = Path("ops/stage1/formalize_s1_11.py")
    spec = importlib.util.spec_from_file_location("s111_task_artifact_formalizer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pseudo_r4(tmp_path: Path, formalizer: object) -> Path:
    """Build a same-path, internally self-hashed pseudo-r4 negative fixture."""

    directory = tmp_path / "evidence/stage1/s1-11-formal/3f18b04df8922be9894678ae4842bd999c7e8fd5/s1-11-r4-20260821"
    directory.mkdir(parents=True)
    body = {
        "schema_version": "stage1-s1-11-formalization-index-v1",
        "status": "PASS",
        "task_id": formalizer.TASK_ID,
        "gate_id": formalizer.GATE_ID,
        "note": "self-consistent local fixture, deliberately not released r4",
    }
    body["artifact_hash"] = canonical_json_hash(body)
    path = directory / "index.json"
    write_canonical_json(path, body)
    return path


def test_real_loader_rejects_same_path_self_consistent_local_fixture(tmp_path: Path) -> None:
    formalizer = _formalizer()
    _pseudo_r4(tmp_path, formalizer)
    with pytest.raises(formalizer.Stage1S111FormalError, match="R4_INDEX_SHA256_MISMATCH"):
        formalizer._emit_load_r4(
            repository=Path.cwd(),
            evidence_root=tmp_path,
            evidence_ref=formalizer.S111_R4_INDEX_REF,
            approved_data_root=tmp_path,
        )


def test_real_loader_rejects_r3_local_and_caller_selected_refs(tmp_path: Path) -> None:
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
                approved_data_root=tmp_path,
            )


def test_emit_cli_does_not_require_execute_inputs_and_still_uses_real_loader(tmp_path: Path) -> None:
    formalizer = _formalizer()
    _pseudo_r4(tmp_path, formalizer)
    with pytest.raises(formalizer.Stage1S111FormalError, match="R4_INDEX_SHA256_MISMATCH"):
        formalizer.main([
            "--emit-task-artifacts",
            "--repository", str(Path.cwd()),
            "--workspace-root", str(tmp_path),
            "--approved-data-root", str(tmp_path),
            "--task-output-dir", "outputs/s111",
        ])


def test_real_r12_to_s111_publication_uses_authority_bytes_when_mounted() -> None:
    """Exercise the complete positive path when a real evidence copy is mounted.

    CI/review machines provide ``S111_REAL_DATA_ROOT`` as a writable local copy
    of the server DATA_ROOT.  The test refuses the live server path, so running
    the suite can never turn a read-only evidence check into a server write.
    """

    formalizer = _formalizer()
    root_value = os.environ.get("S111_REAL_DATA_ROOT")
    if not root_value:
        pytest.skip("real r4/r12 DATA_ROOT copy is not mounted")
    root = Path(root_value).resolve()
    if root.as_posix() == formalizer.S111_APPROVED_DATA_ROOT:
        pytest.skip("refusing to write the live approved DATA_ROOT")
    repository = Path(os.environ.get("S111_REAL_REPOSITORY", str(Path.cwd()))).resolve()
    result = formalizer.emit_task_artifacts(
        repository=repository,
        evidence_root=root,
        approved_data_root=root,
        output_dir="evidence/stage1/tasks/11-s1-11-r4-20260821",
        s110_output_dir="evidence/stage1/tasks/10-s1-10-r12-20260821",
    )
    assert set(result["commit_refs"]) == set(formalizer.S111_TASK_ARTIFACT_KINDS)
    compat = result["s110_compatibility"]
    assert set(compat["commit_refs"]) == set(formalizer.S110_TASK_ARTIFACT_KINDS)
    for ref in (*result["commit_refs"].values(), *compat["commit_refs"].values()):
        loaded = load_committed_task_artifact(root, ref, require_formal=True)
        assert loaded.run_intent == "formal"
        assert loaded.identity.formal_eligible is True


def _payload(kind: str) -> dict[str, object]:
    # This is a complete formal-shaped task payload, not a minimal PASS blob.
    value: dict[str, object] = {
        "schema_version": f"stage1-s1-11-{kind}-v1",
        "status": "PASS",
        "task_id": "stage1.11_reporting_and_exit_gate",
        "gate_id": "G1-EXIT",
        "scope": "immutable formal evidence",
        "requirements": {"all_required_checks": True, "unresolved_failure_count": 0},
        "dependency_index_sha256": {"G1-RESUME": "a" * 64},
        "source_artifact_hashes": {"formal_observation": "b" * 64},
        "measured": [{"requirement_id": "S1.11-01", "measured": True, "threshold": "PASS"}],
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def test_group_publish_is_idempotent_and_recovers_missing_commit(tmp_path: Path) -> None:
    formalizer = _formalizer()
    source = tmp_path / "evidence/source.json"
    source.parent.mkdir(parents=True)
    write_canonical_json(source, {"schema_version": "source-v1", "status": "PASS"})
    kinds = tuple(formalizer.S111_TASK_ARTIFACT_KINDS)
    payloads = {kind: _payload(kind) for kind in kinds}
    refs = tuple(sorted(("evidence/source.json", "outputs/s111/producer-config.json")))
    config_hash = "c" * 64
    first = formalizer._publish_emit_group(
        evidence_root=tmp_path, output_dir="outputs/s111",
        task_id="stage1.11_reporting_and_exit_gate", artifact_kinds=kinds,
        config_hash=config_hash, source_refs=refs, payloads=payloads,
    )
    second = formalizer._publish_emit_group(
        evidence_root=tmp_path, output_dir="outputs/s111",
        task_id="stage1.11_reporting_and_exit_gate", artifact_kinds=kinds,
        config_hash=config_hash, source_refs=refs, payloads=payloads,
    )
    assert first == second
    missing = tmp_path / first["commit_refs"]["gate_summary"]
    missing.unlink()
    (tmp_path / first["success_ref"]).unlink()
    resumed = formalizer._publish_emit_group(
        evidence_root=tmp_path, output_dir="outputs/s111",
        task_id="stage1.11_reporting_and_exit_gate", artifact_kinds=kinds,
        config_hash=config_hash, source_refs=refs, payloads=payloads,
    )
    assert resumed == first
    assert (tmp_path / resumed["success_ref"]).is_file()


def test_group_fails_closed_on_payload_config_commit_and_symlink_drift(tmp_path: Path) -> None:
    formalizer = _formalizer()
    kinds = tuple(formalizer.S111_TASK_ARTIFACT_KINDS)
    payloads = {kind: _payload(kind) for kind in kinds}
    refs = ("outputs/s111/producer-config.json",)
    formalizer._publish_emit_group(
        evidence_root=tmp_path, output_dir="outputs/s111",
        task_id="stage1.11_reporting_and_exit_gate", artifact_kinds=kinds,
        config_hash="d" * 64, source_refs=refs, payloads=payloads,
    )
    tampered = dict(payloads["stage_report"])
    tampered["measured"] = [{"requirement_id": "S1.11-01", "measured": False, "threshold": "PASS"}]
    tampered["artifact_hash"] = canonical_json_hash({k: v for k, v in tampered.items() if k != "artifact_hash"})
    changed = dict(payloads)
    changed["stage_report"] = tampered
    with pytest.raises(formalizer.Stage1S111FormalError, match="GROUP_MANIFEST_IDENTITY_DRIFT"):
        formalizer._publish_emit_group(
            evidence_root=tmp_path, output_dir="outputs/s111",
            task_id="stage1.11_reporting_and_exit_gate", artifact_kinds=kinds,
            config_hash="d" * 64, source_refs=refs, payloads=changed,
        )
    with pytest.raises(formalizer.Stage1S111FormalError, match="GROUP_MANIFEST_IDENTITY_DRIFT"):
        formalizer._publish_emit_group(
            evidence_root=tmp_path, output_dir="outputs/s111",
            task_id="stage1.11_reporting_and_exit_gate", artifact_kinds=kinds,
            config_hash="e" * 64, source_refs=refs, payloads=payloads,
        )
    commit = tmp_path / "outputs/s111/commits/stage_report.json"
    write_canonical_json(commit, {"tampered": True})
    with pytest.raises(formalizer.Stage1S111FormalError, match="EXISTING_COMMIT_INVALID"):
        formalizer._publish_emit_group(
            evidence_root=tmp_path, output_dir="outputs/s111",
            task_id="stage1.11_reporting_and_exit_gate", artifact_kinds=kinds,
            config_hash="d" * 64, source_refs=refs, payloads=payloads,
        )
    linked = tmp_path / "outputs/linked"
    target = tmp_path / "outside"
    target.mkdir()
    try:
        linked.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this Windows runner")
    with pytest.raises(formalizer.Stage1S111FormalError, match="SYMLINK_FORBIDDEN"):
        formalizer._publish_emit_group(
            evidence_root=tmp_path, output_dir="outputs/linked",
            task_id="stage1.11_reporting_and_exit_gate", artifact_kinds=kinds,
            config_hash="f" * 64, source_refs=refs, payloads=payloads,
        )


def test_evidence_root_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    formalizer = _formalizer()
    linked = tmp_path / "linked-root"
    try:
        linked.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this Windows runner")
    with pytest.raises(formalizer.Stage1S111FormalError, match="DATA_ROOT_SYMLINK_FORBIDDEN"):
        formalizer.emit_task_artifacts(
            repository=Path.cwd(), evidence_root=linked,
            output_dir="outputs/s111", approved_data_root=tmp_path,
        )
