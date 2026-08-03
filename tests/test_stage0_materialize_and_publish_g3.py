from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

import ops.stage0.materialize_and_publish_g3 as orchestration_module
from ops.stage0.materialize_and_publish_g3 import (
    Stage0G3MaterializationError,
    materialize_and_publish_stage0_g3,
)
from param_importance_nlp.contracts import (
    GateRecord,
    GateStatus,
    canonical_json_bytes,
    load_canonical_json,
)
import param_importance_nlp.g3_asset_publication as publication_module
from param_importance_nlp.g3_gate import (
    evaluate_stage0_g3,
    g3_resolution_artifact_hash,
    validate_stage0_g3_resolution,
)
from tests.test_g3_asset_publication import (
    _CHECKED_AT,
    _LifecycleFixture,
    _fake_probe,
    _verified_lifecycle_fixture,
)


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _LifecycleFixture:
    state = _verified_lifecycle_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        orchestration_module,
        "__file__",
        str(state.source_root / "ops/stage0/materialize_and_publish_g3.py"),
    )
    monkeypatch.setattr(publication_module, "_run_semantic_probe", _fake_probe)
    return state


def _run(
    state: _LifecycleFixture,
    *,
    generator_git_commit: str | None = None,
    checked_at: str = _CHECKED_AT,
):
    assert state.acquisition is not None
    assert state.verification is not None
    return materialize_and_publish_stage0_g3(
        source_root=state.source_root,
        data_root=state.fixture.data_root,
        requirements_path=state.requirements_path,
        layout_path=state.layout_path,
        download_plan_path=state.download_plan_path,
        acquisition_ref=state.acquisition.acquisition_ref,
        verification_ref=state.verification.verification_ref,
        gate_actor_instance_id="fixture:gate:1",
        generator_git_commit=(
            state.commit if generator_git_commit is None else generator_git_commit
        ),
        checked_at=checked_at,
    )


def _assert_no_gate_outputs(state: _LifecycleFixture) -> None:
    data_root = state.fixture.data_root
    for entry in state.fixture.layout["entries"]:
        assert not data_root.joinpath(
            *PurePosixPath(entry["manifest_ref"]).parts
        ).exists()
        assert not data_root.joinpath(
            *PurePosixPath(entry["qualification_ref"]).parts
        ).exists()
        assert not (
            data_root
            / "manifests"
            / "evidence"
            / "g3"
            / entry["logical_name"]
        ).exists()


def _assert_no_report_bundle(state: _LifecycleFixture) -> None:
    assert not (state.fixture.data_root / "reports").exists()


def test_gate_only_materialization_passes_and_report_bundle_reruns_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _setup(tmp_path, monkeypatch)
    first = _run(state)

    assert first.status == "PASS"
    assert len(first.publications) == 13
    assert {item.status for item in first.publications} == {"published"}
    assert first.acquisition_ref == state.acquisition.acquisition_ref
    assert first.acquisition_sha256 == state.acquisition.acquisition_sha256
    assert first.verification_ref == state.verification.verification_ref
    assert first.verification_sha256 == state.verification.verification_sha256
    assert first.reports.status == "published"
    report_directory = state.fixture.data_root / first.reports.directory_ref
    assert {path.name for path in report_directory.iterdir()} == {
        orchestration_module.INDEX_NAME,
        orchestration_module.AUDIT_NAME,
        orchestration_module.RESOLUTION_NAME,
    }
    index = load_canonical_json(state.fixture.data_root / first.reports.index_ref)
    audit = load_canonical_json(state.fixture.data_root / first.reports.audit_ref)
    resolution = load_canonical_json(
        state.fixture.data_root / first.reports.resolution_ref
    )
    assert index["status"] == audit["status"] == resolution["status"] == "PASS"
    assert index["audit_sha256"] == first.reports.audit_sha256
    assert index["resolution_sha256"] == first.reports.resolution_sha256
    assert index["entry_count"] == 13
    assert audit["publication_count"] == 13
    for report in (index, audit):
        assert report["acquisition_ref"] == state.acquisition.acquisition_ref
        assert report["acquisition_sha256"] == state.acquisition.acquisition_sha256
        assert report["verification_ref"] == state.verification.verification_ref
        assert (
            report["verification_sha256"]
            == state.verification.verification_sha256
        )
    validate_stage0_g3_resolution(resolution)
    for reference, expected in (
        (first.reports.index_ref, first.reports.index_sha256),
        (first.reports.audit_ref, first.reports.audit_sha256),
        (first.reports.resolution_ref, first.reports.resolution_sha256),
    ):
        payload = (state.fixture.data_root / reference).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
        assert payload == canonical_json_bytes(
            load_canonical_json(state.fixture.data_root / reference)
        )

    forbidden_roots = (str(state.source_root), str(state.fixture.data_root))
    for artifact in report_directory.iterdir():
        text = artifact.read_text(encoding="utf-8")
        assert all(root not in text for root in forbidden_roots)
    assert "source_root" not in audit["source_binding"]

    before = {path.name: path.read_bytes() for path in report_directory.iterdir()}
    second = _run(state)
    assert second.reports.status == "reused"
    assert {item.status for item in second.publications} == {"existing_ready"}
    assert before == {path.name: path.read_bytes() for path in report_directory.iterdir()}

    exit_code = orchestration_module.main(
        [
            "--source-root",
            str(state.source_root),
            "--data-root",
            str(state.fixture.data_root),
            "--requirements",
            str(state.requirements_path),
            "--layout",
            str(state.layout_path),
            "--download-plan",
            str(state.download_plan_path),
            "--acquisition-ref",
            state.acquisition.acquisition_ref,
            "--verification-ref",
            state.verification.verification_ref,
            "--gate-actor-instance-id",
            "fixture:gate:1",
            "--generator-git-commit",
            state.commit,
            "--checked-at",
            _CHECKED_AT,
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert all(root not in stdout for root in forbidden_roots)

    forged = b'{"forged":true}\n'
    index_path = state.fixture.data_root / first.reports.index_ref
    index_path.write_bytes(forged)
    with pytest.raises(
        Stage0G3MaterializationError,
        match="REPORT_BUNDLE_NO_CLOBBER_MISMATCH",
    ):
        _run(state)
    assert index_path.read_bytes() == forged


def test_valid_blocked_gate_never_publishes_report_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path, monkeypatch)

    def blocked(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolution = evaluate_stage0_g3(*args, **kwargs)
        passed = GateRecord.from_mapping(resolution["gates"][0])
        resolution["gates"][0] = GateRecord(
            gate_id=passed.gate_id,
            stage=passed.stage,
            status=GateStatus.BLOCKED,
            checked_at=passed.checked_at,
            measured=passed.measured,
            threshold=passed.threshold,
            evidence_refs=passed.evidence_refs,
            reasons=("FORCED_TEST_BLOCK",),
        ).to_dict()
        resolution["status"] = "BLOCKED"
        resolution["artifact_hash"] = g3_resolution_artifact_hash(resolution)
        validate_stage0_g3_resolution(resolution)
        return resolution

    monkeypatch.setattr(orchestration_module, "evaluate_stage0_g3", blocked)
    with pytest.raises(
        Stage0G3MaterializationError,
        match="G3_FORMAL_PASS_INCOMPLETE|G3_FORMAL_GATE_BLOCKED",
    ):
        _run(state)
    _assert_no_report_bundle(state)


def test_dirty_git_source_is_rejected_before_gate_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path, monkeypatch)
    dirty = state.source_root / "untracked.py"
    dirty.write_text("raise RuntimeError('not tracked')\n", encoding="utf-8")

    with pytest.raises(
        Stage0G3MaterializationError,
        match="GIT_SOURCE_ROOT_NOT_CLEAN",
    ):
        _run(state)
    _assert_no_gate_outputs(state)
    _assert_no_report_bundle(state)


def test_wrong_imported_module_origin_is_rejected_before_gate_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path, monkeypatch)
    outside = tmp_path / "outside" / "g3_asset_publication.py"
    outside.parent.mkdir()
    outside.write_bytes(b"# wrong module origin\n")
    monkeypatch.setattr(publication_module, "__file__", str(outside))

    with pytest.raises(
        Stage0G3MaterializationError,
        match=(
            "IMPORTED_MODULE_ORIGIN_MISMATCH:"
            "param_importance_nlp.g3_asset_publication"
        ),
    ):
        _run(state)
    _assert_no_gate_outputs(state)
    _assert_no_report_bundle(state)


def test_generator_commit_must_equal_clean_source_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path, monkeypatch)
    with pytest.raises(
        Stage0G3MaterializationError,
        match="GENERATOR_GIT_COMMIT_DOES_NOT_MATCH_SOURCE_HEAD",
    ):
        _run(state, generator_git_commit="0" * 40)
    _assert_no_gate_outputs(state)
    _assert_no_report_bundle(state)


def test_source_drift_during_gate_evaluation_blocks_report_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path, monkeypatch)

    def drift(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolution = evaluate_stage0_g3(*args, **kwargs)
        state.layout_path.write_bytes(state.layout_path.read_bytes() + b" ")
        return resolution

    monkeypatch.setattr(orchestration_module, "evaluate_stage0_g3", drift)
    with pytest.raises(
        Stage0G3MaterializationError,
        match="GIT_SOURCE_ROOT_NOT_CLEAN|GIT_SOURCE_BINDING_DRIFTED",
    ):
        _run(state)
    _assert_no_report_bundle(state)


def test_source_drift_during_probe_blocks_all_gate_evidence_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _setup(tmp_path, monkeypatch)
    drifted = False

    def drift_probe(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        nonlocal drifted
        if not drifted:
            state.layout_path.write_bytes(state.layout_path.read_bytes() + b" ")
            drifted = True
        return _fake_probe(*args, **kwargs)

    monkeypatch.setattr(publication_module, "_run_semantic_probe", drift_probe)
    with pytest.raises(
        Stage0G3MaterializationError,
        match="GIT_SOURCE_ROOT_NOT_CLEAN|GIT_SOURCE_BINDING_DRIFTED",
    ):
        _run(state)
    _assert_no_gate_outputs(state)
    _assert_no_report_bundle(state)
