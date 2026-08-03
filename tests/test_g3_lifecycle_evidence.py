from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath
import importlib
import json
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from param_importance_nlp.asset_download_plan import download_plan_artifact_hash
from param_importance_nlp.cli import _validate_project_json_schema
from param_importance_nlp.contracts import (
    canonical_json_bytes,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.g3_asset_publication import (
    G3AssetPublicationError,
    NetworkEgressAttempt,
    gate_stage0_g3_assets_from_evidence,
    publish_stage0_g3_assets,
    publish_stage0_g3_prerequisites,
)
import param_importance_nlp.g3_asset_publication as publication_module
from param_importance_nlp.g3_gate import (
    evaluate_stage0_g3,
    glue_preprocessing_config_hash,
)
from param_importance_nlp.g3_lifecycle_evidence import (
    G3_CRITICAL_MODULE_ORIGINS,
    G3_CRITICAL_SOURCE_REFS,
    G3LifecycleEvidenceError,
    G3NetworkEgressAttempt,
    G3VerificationFailed,
    attest_stage0_g3_acquisition,
    validate_g3_acquisition_report,
    validate_g3_verify_report,
    verify_stage0_g3_acquisition,
)
import param_importance_nlp.g3_lifecycle_evidence as lifecycle_module
from param_importance_nlp.glue_builder import (
    normalize_tokenizer_descriptor_inventory,
)
from tests.test_g3_asset_publication import _fake_probe, _fixture


_REQUIREMENTS_REF = "configs/stage0/g3-asset-requirements-v1.json"
_LAYOUT_REF = "configs/stage0/g3-asset-layout-v1.json"
_PLAN_REF = "configs/stage0/g3-download-plan-v1.json"
_DOWNLOAD_REPORT_REF = "operations/g3/download-report.json"
_STARTED_AT = "2026-08-03T07:00:00Z"
_COMPLETED_AT = "2026-08-03T07:00:01Z"
_VERIFIED_AT = "2026-08-03T07:00:02Z"
_GATED_AT = "2026-08-03T07:00:03Z"
_ROOT = Path(__file__).resolve().parents[1]


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    value["artifact_hash"] = canonical_json_hash(
        {name: item for name, item in value.items() if name != "artifact_hash"}
    )
    return value


def _validate_schema(name: str, value: dict[str, Any]) -> None:
    schema = json.loads((_ROOT / "schemas" / name).read_text(encoding="utf-8"))
    _validate_project_json_schema(schema)
    assert schema["$id"].endswith(name)
    assert set(schema["required"]).issubset(value)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_ref(root: Path, reference: str, value: dict[str, Any]) -> Path:
    target = root.joinpath(*PurePosixPath(reference).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(value))
    return target


def _patch_module_origins(
    monkeypatch: pytest.MonkeyPatch, source_root: Path
) -> None:
    for module_name, reference in G3_CRITICAL_MODULE_ORIGINS:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(
            module,
            "__file__",
            str(source_root.joinpath(*PurePosixPath(reference).parts)),
        )


def _requirement_for(
    requirements: dict[str, Any], kind: str, requirement_name: str
) -> dict[str, Any]:
    if kind == "model":
        return next(item for item in requirements["models"] if item["name"] == requirement_name)
    if kind == "tokenizer":
        return requirements["tokenizer"]
    if kind == "pile":
        return requirements["pile"]
    return next(item for item in requirements["glue"] if item["task"] == requirement_name)


def _plan_descriptors(
    requirements: dict[str, Any], layout: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected = {
        "pythia-410m-deduped-step0",
        "pythia-tokenizer",
        "glue-mnli-raw",
        "glue-rte-raw",
    }
    values: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in layout["entries"]:
        if entry["logical_name"] not in selected:
            continue
        requirement = _requirement_for(
            requirements, entry["kind"], entry["requirement_name"]
        )
        descriptors = (
            requirement["raw_files"]
            if entry["kind"] == "glue_raw"
            else requirement["files"]
        )
        values.extend((entry, descriptor) for descriptor in descriptors)
    assert len(values) == 13
    return values


def _fake_derived_builder(
    data_root: str | Path,
    _raw_task_root: str | Path,
    raw_asset_id: str,
    _tokenizer_root: str | Path,
    tokenizer_asset_id: str,
    requirement: dict[str, Any],
    target_dir: str | Path,
    *,
    tokenizer_requirement: dict[str, Any],
    generator_git_commit: str,
) -> SimpleNamespace:
    root = Path(data_root)
    target_ref = Path(target_dir).as_posix()
    target = root.joinpath(*PurePosixPath(target_ref).parts)
    state = target / "dataset" / "state.json"
    payload = state.read_bytes()
    tokenizer_inventory = normalize_tokenizer_descriptor_inventory(
        tokenizer_requirement
    )
    tokenizer_inventory_hash = canonical_json_hash(
        [dict(item) for item in tokenizer_inventory]
    )
    derived_splits = tuple(requirement["preprocessing"]["derived_splits"])
    preprocessing_config_hash = glue_preprocessing_config_hash(requirement)
    return SimpleNamespace(
        task=requirement["task"],
        raw_asset_id=raw_asset_id,
        tokenizer_asset_id=tokenizer_asset_id,
        tokenizer_descriptor_inventory=tokenizer_inventory,
        tokenizer_descriptor_inventory_hash=tokenizer_inventory_hash,
        target_ref=target_ref,
        generator_git_commit=generator_git_commit,
        preprocessing_version="stage0-glue-preprocessing-v1",
        preprocessing_config_hash=preprocessing_config_hash,
        requirement_hash=canonical_json_hash(requirement),
        derived_splits=derived_splits,
        split_counts={split: requirement["split_counts"][split] for split in derived_splits},
        map_fingerprints=lifecycle_module._expected_derived_map_fingerprints(
            requirement,
            raw_asset_id=raw_asset_id,
            tokenizer_asset_id=tokenizer_asset_id,
            tokenizer_descriptor_inventory_hash=tokenizer_inventory_hash,
            generator_git_commit=generator_git_commit,
            preprocessing_config_hash=preprocessing_config_hash,
        ),
        file_inventory=(
            {
                "path": "dataset/state.json",
                "size_bytes": len(payload),
            },
        ),
        network_attempts=0,
    )


def _control_plane(tmp_path: Path) -> tuple[Any, Path, Path, Path, str]:
    fixture = _fixture(tmp_path)
    requirements = dict(load_canonical_json(fixture.requirements_path))
    layout = dict(load_canonical_json(fixture.layout_path))
    source = tmp_path / "source"
    source.mkdir()
    requirements_path = _write_ref(source, _REQUIREMENTS_REF, requirements)
    layout_path = _write_ref(source, _LAYOUT_REF, layout)
    plan_entries: list[dict[str, Any]] = []
    report_objects: list[dict[str, Any]] = []
    for index, (entry, descriptor) in enumerate(
        _plan_descriptors(requirements, layout)
    ):
        spec_ref = f"configs/stage0/http-objects/fixture-{index:02d}.json"
        object_id = (
            f"huggingface/fixture/g3/{entry['logical_name']}/{descriptor['path']}"
        )
        spec = {
            "schema_version": "stage0-http-object-spec-v1",
            "source_id": object_id,
            "revision": _requirement_for(
                requirements, entry["kind"], entry["requirement_name"]
            )["revision"],
            "expected_size": descriptor["size_bytes"],
            "expected_sha256": descriptor["sha256"],
        }
        _write_ref(source, spec_ref, spec)
        plan_entry = {
            "object_id": object_id,
            "spec_ref": spec_ref,
            "asset_root_ref": entry["asset_root_ref"],
            "final_path": descriptor["path"],
        }
        plan_entries.append(plan_entry)
        report_objects.append(
            {
                "object_id": object_id,
                "asset_root_ref": entry["asset_root_ref"],
                "final_path": descriptor["path"],
                "result": {
                    "schema_version": "stage0-asset-acquisition-result-v1",
                    "status": "already_ready",
                    "source_id": object_id,
                    "revision": spec["revision"],
                    "size_bytes": descriptor["size_bytes"],
                    "sha256": descriptor["sha256"],
                    "attempts": 0,
                    "resumed": False,
                    "network_accessed": False,
                },
            }
        )
    plan: dict[str, Any] = {
        "schema_version": "stage0-g3-download-plan-v1",
        "created_at": _STARTED_AT,
        "generator_git_commit": "a" * 40,
        "requirements_ref": _REQUIREMENTS_REF,
        "requirements_sha256": requirements["artifact_hash"],
        "layout_ref": _LAYOUT_REF,
        "layout_sha256": layout["artifact_hash"],
        "entries": plan_entries,
    }
    plan["artifact_hash"] = download_plan_artifact_hash(plan)
    plan_path = _write_ref(source, _PLAN_REF, plan)
    download_payload: dict[str, Any] = {
        "schema_version": "stage0-g3-download-report-v1",
        "status": "PASS",
        "started_at": _STARTED_AT,
        "plan_sha256": plan["artifact_hash"],
        "objects": report_objects,
        "runtime_urls_persisted": False,
    }
    download_report = download_payload | {
        "artifact_hash": canonical_json_hash(download_payload)
    }
    _write_ref(fixture.data_root, _DOWNLOAD_REPORT_REF, download_report)
    for reference in G3_CRITICAL_SOURCE_REFS:
        source_path = _ROOT.joinpath(*PurePosixPath(reference).parts)
        target = source.joinpath(*PurePosixPath(reference).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "Lifecycle Test")
    _git(source, "config", "user.email", "lifecycle@example.invalid")
    _git(source, "config", "core.autocrlf", "false")
    _git(source, "add", "--", ".")
    _git(source, "commit", "--no-gpg-sign", "-q", "-m", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    return fixture, source, requirements_path, layout_path, plan_path, commit


def _attest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    fixture, source, requirements, layout, plan, commit = _control_plane(tmp_path)
    _patch_module_origins(monkeypatch, source)
    monkeypatch.setattr(
        lifecycle_module, "build_glue_derived_dataset", _fake_derived_builder
    )
    result = attest_stage0_g3_acquisition(
        source_root=source,
        data_root=fixture.data_root,
        requirements=requirements,
        layout=layout,
        download_plan=plan,
        requirements_ref=_REQUIREMENTS_REF,
        layout_ref=_LAYOUT_REF,
        download_plan_ref=_PLAN_REF,
        download_report_ref=_DOWNLOAD_REPORT_REF,
        actor_instance_id="fixture:fetcher:1",
        source_git_commit=commit,
        started_at=_STARTED_AT,
        completed_at=_COMPLETED_AT,
    )
    return fixture, source, requirements, layout, plan, commit, result


def _attest_existing_control_plane(
    fixture: Any,
    source: Path,
    requirements: Path,
    layout: Path,
    plan: Path,
    commit: str,
) -> Any:
    return attest_stage0_g3_acquisition(
        source_root=source,
        data_root=fixture.data_root,
        requirements=requirements,
        layout=layout,
        download_plan=plan,
        requirements_ref=_REQUIREMENTS_REF,
        layout_ref=_LAYOUT_REF,
        download_plan_ref=_PLAN_REF,
        download_report_ref=_DOWNLOAD_REPORT_REF,
        actor_instance_id="fixture:fetcher:source-boundary",
        source_git_commit=commit,
        started_at=_STARTED_AT,
        completed_at=_COMPLETED_AT,
    )


def test_independent_lifecycle_reaches_ready_and_reruns_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, requirements, layout, plan, commit, acquisition = _attest(
        tmp_path, monkeypatch
    )
    report = dict(
        load_canonical_json(fixture.data_root / acquisition.acquisition_ref)
    )
    _validate_schema("stage0-g3-acquisition-report-v1.json", report)
    assert [entry["mode"] for entry in report["entries"]].count("canonical-plan") == 4
    assert [entry["mode"] for entry in report["entries"]].count("existing-import") == 6
    assert [entry["mode"] for entry in report["entries"]].count("derived-build") == 3
    requirement_value = dict(load_canonical_json(requirements))
    layout_value = dict(load_canonical_json(layout))
    derived_entry_index = next(
        index
        for index, entry in enumerate(report["entries"])
        if entry["mode"] == "derived-build"
    )
    wrong_version = deepcopy(report)
    wrong_version["entries"][derived_entry_index]["source_evidence"][
        "preprocessing_version"
    ] = "stage0-glue-preprocessing-forged-v1"
    _rehash(wrong_version)
    with pytest.raises(G3LifecycleEvidenceError, match="frozen lineage"):
        validate_g3_acquisition_report(
            wrong_version,
            requirements=requirement_value,
            layout=layout_value,
        )
    wrong_fingerprint = deepcopy(report)
    split = next(
        iter(
            wrong_fingerprint["entries"][derived_entry_index]["source_evidence"][
                "map_fingerprints"
            ]
        )
    )
    wrong_fingerprint["entries"][derived_entry_index]["source_evidence"][
        "map_fingerprints"
    ][split] = "f" * 64
    _rehash(wrong_fingerprint)
    with pytest.raises(G3LifecycleEvidenceError, match="frozen lineage"):
        validate_g3_acquisition_report(
            wrong_fingerprint,
            requirements=requirement_value,
            layout=layout_value,
        )
    verification = verify_stage0_g3_acquisition(
        source_root=source,
        data_root=fixture.data_root,
        requirements=requirements,
        layout=layout,
        download_plan=plan,
        acquisition_ref=acquisition.acquisition_ref,
        actor_instance_id="fixture:verifier:1",
        generator_git_commit=commit,
        checked_at=_VERIFIED_AT,
    )
    verification_report = dict(
        load_canonical_json(fixture.data_root / verification.verification_ref)
    )
    _validate_schema("stage0-g3-verify-only-report-v1.json", verification_report)
    forged_expectation = deepcopy(verification_report)
    forged_expectation["entries"][0]["files"][0]["expected_sha256"] = "f" * 64
    forged_expectation["entries"][0]["files"][0]["observed_sha256"] = "f" * 64
    _rehash(forged_expectation)
    with pytest.raises(G3LifecycleEvidenceError, match="exactly replay"):
        validate_g3_verify_report(
            forged_expectation,
            acquisition=report,
            requirements=requirement_value,
            layout=layout_value,
        )
    monkeypatch.setattr(publication_module, "_run_semantic_probe", _fake_probe)
    publications = gate_stage0_g3_assets_from_evidence(
        requirements,
        layout,
        plan,
        source,
        fixture.data_root,
        acquisition_ref=acquisition.acquisition_ref,
        verification_ref=verification.verification_ref,
        generator_git_commit=commit,
        checked_at=_GATED_AT,
        gate_actor_instance_id="fixture:gate:1",
    )
    assert len(publications) == 13
    assert {item.state for item in publications} == {"ready"}
    audit = evaluate_stage0_g3(
        requirements, layout, fixture.data_root, checked_at=_GATED_AT
    )
    assert audit["status"] == "PASS"
    actors = [
        event["actor_instance_id"]
        for event in load_canonical_json(
            fixture.data_root / publications[0].manifest_ref
        )["state_history"]
    ]
    assert actors == [
        "fixture:fetcher:1",
        "fixture:fetcher:1",
        "fixture:verifier:1",
        "fixture:gate:1",
    ]
    assert len(set(actors)) == 3
    rerun = gate_stage0_g3_assets_from_evidence(
        requirements,
        layout,
        plan,
        source,
        fixture.data_root,
        acquisition_ref=acquisition.acquisition_ref,
        verification_ref=verification.verification_ref,
        generator_git_commit=commit,
        checked_at="2026-08-03T07:00:05Z",
        gate_actor_instance_id="fixture:gate:1",
    )
    assert {item.status for item in rerun} == {"existing_ready"}
    missing_verified = fixture.data_root / verification.candidate_refs[0]
    missing_verified.unlink()
    with pytest.raises(
        G3AssetPublicationError, match="VERIFIED candidate cannot be replayed"
    ) as captured:
        gate_stage0_g3_assets_from_evidence(
            requirements,
            layout,
            plan,
            source,
            fixture.data_root,
            acquisition_ref=acquisition.acquisition_ref,
            verification_ref=verification.verification_ref,
            generator_git_commit=commit,
            checked_at="2026-08-03T07:00:06Z",
            gate_actor_instance_id="fixture:gate:1",
        )
    assert isinstance(captured.value.__cause__, FileNotFoundError)


def test_failed_verify_blocks_repair_under_same_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, requirements, layout, plan, commit, acquisition = _attest(
        tmp_path, monkeypatch
    )
    target = fixture.data_root / "models/pythia-14m-step0/config.json"
    original = target.read_bytes()
    target.write_bytes(b"x" * len(original))
    with pytest.raises(G3VerificationFailed):
        verify_stage0_g3_acquisition(
            source_root=source,
            data_root=fixture.data_root,
            requirements=requirements,
            layout=layout,
            download_plan=plan,
            acquisition_ref=acquisition.acquisition_ref,
            actor_instance_id="fixture:verifier:failed",
            generator_git_commit=commit,
            checked_at=_VERIFIED_AT,
        )
    target.write_bytes(original)
    with pytest.raises(FileExistsError, match="no-clobber"):
        verify_stage0_g3_acquisition(
            source_root=source,
            data_root=fixture.data_root,
            requirements=requirements,
            layout=layout,
            download_plan=plan,
            acquisition_ref=acquisition.acquisition_ref,
            actor_instance_id="fixture:verifier:failed",
            generator_git_commit=commit,
            checked_at=_VERIFIED_AT,
        )


def test_attestor_rejects_dirty_wrong_origin_and_midrun_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, requirements, layout, plan, commit = _control_plane(tmp_path)
    _patch_module_origins(monkeypatch, source)
    monkeypatch.setattr(
        lifecycle_module, "build_glue_derived_dataset", _fake_derived_builder
    )
    dirty = source / "untracked-source.py"
    dirty.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(G3LifecycleEvidenceError, match="completely clean"):
        _attest_existing_control_plane(
            fixture, source, requirements, layout, plan, commit
        )
    dirty.unlink()

    monkeypatch.setattr(
        lifecycle_module,
        "__file__",
        str(_ROOT / "src/param_importance_nlp/g3_lifecycle_evidence.py"),
    )
    with pytest.raises(G3LifecycleEvidenceError, match="module origin"):
        _attest_existing_control_plane(
            fixture, source, requirements, layout, plan, commit
        )
    monkeypatch.setattr(
        lifecycle_module,
        "__file__",
        str(source / "src/param_importance_nlp/g3_lifecycle_evidence.py"),
    )

    original_builder = _fake_derived_builder
    drifted = False

    def drifting_builder(*args: Any, **kwargs: Any) -> Any:
        nonlocal drifted
        result = original_builder(*args, **kwargs)
        if not drifted:
            drifted = True
            target = source / "ops/stage0/verify_g3_assets.py"
            target.write_bytes(target.read_bytes() + b"\n# mid-run drift\n")
        return result

    monkeypatch.setattr(
        lifecycle_module, "build_glue_derived_dataset", drifting_builder
    )
    with pytest.raises(G3LifecycleEvidenceError, match="completely clean"):
        _attest_existing_control_plane(
            fixture, source, requirements, layout, plan, commit
        )
    assert not (fixture.data_root / "manifests/evidence/g3/acquisition").exists()
    assert not (fixture.data_root / "manifests/candidates/g3").exists()


def test_attestor_requires_plan_specs_to_be_tracked_even_when_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, requirements, layout, plan, _ = _control_plane(tmp_path)
    _patch_module_origins(monkeypatch, source)
    monkeypatch.setattr(
        lifecycle_module, "build_glue_derived_dataset", _fake_derived_builder
    )
    plan_value = dict(load_canonical_json(plan))
    spec_ref = plan_value["entries"][0]["spec_ref"]
    _git(source, "rm", "--cached", "--", spec_ref)
    (source / ".gitignore").write_text(f"/{spec_ref}\n", encoding="utf-8")
    _git(source, "add", "--", ".gitignore")
    _git(source, "commit", "--no-gpg-sign", "-q", "-m", "ignore spec")
    commit = _git(source, "rev-parse", "HEAD")
    assert not _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    with pytest.raises(G3LifecycleEvidenceError, match="Git check failed"):
        _attest_existing_control_plane(
            fixture, source, requirements, layout, plan, commit
        )


def test_verifier_rejects_dirty_wrong_origin_and_midrun_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, requirements, layout, plan, commit, acquisition = _attest(
        tmp_path, monkeypatch
    )
    verify_kwargs = {
        "source_root": source,
        "data_root": fixture.data_root,
        "requirements": requirements,
        "layout": layout,
        "download_plan": plan,
        "acquisition_ref": acquisition.acquisition_ref,
        "actor_instance_id": "fixture:verifier:source-boundary",
        "generator_git_commit": commit,
        "checked_at": _VERIFIED_AT,
    }
    dirty = source / "untracked-verifier.py"
    dirty.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(G3LifecycleEvidenceError, match="completely clean"):
        verify_stage0_g3_acquisition(**verify_kwargs)
    dirty.unlink()

    monkeypatch.setattr(
        lifecycle_module,
        "__file__",
        str(_ROOT / "src/param_importance_nlp/g3_lifecycle_evidence.py"),
    )
    with pytest.raises(G3LifecycleEvidenceError, match="module origin"):
        verify_stage0_g3_acquisition(**verify_kwargs)
    monkeypatch.setattr(
        lifecycle_module,
        "__file__",
        str(source / "src/param_importance_nlp/g3_lifecycle_evidence.py"),
    )

    original_observer = lifecycle_module._observe_candidate_files
    drifted = False

    def drifting_observer(*args: Any, **kwargs: Any) -> Any:
        nonlocal drifted
        result = original_observer(*args, **kwargs)
        if not drifted:
            drifted = True
            target = source / "ops/stage0/attest_g3_materialization.py"
            target.write_bytes(target.read_bytes() + b"\n# mid-run drift\n")
        return result

    monkeypatch.setattr(
        lifecycle_module, "_observe_candidate_files", drifting_observer
    )
    with pytest.raises(G3LifecycleEvidenceError, match="completely clean"):
        verify_stage0_g3_acquisition(**verify_kwargs)
    assert not (fixture.data_root / "manifests/evidence/g3/verification").exists()


def test_direct_gate_rejects_dirty_wrong_origin_and_midprobe_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, requirements, layout, plan, commit, acquisition = _attest(
        tmp_path, monkeypatch
    )
    verification = verify_stage0_g3_acquisition(
        source_root=source,
        data_root=fixture.data_root,
        requirements=requirements,
        layout=layout,
        download_plan=plan,
        acquisition_ref=acquisition.acquisition_ref,
        actor_instance_id="fixture:verifier:gate-source-boundary",
        generator_git_commit=commit,
        checked_at=_VERIFIED_AT,
    )
    gate_kwargs = {
        "acquisition_ref": acquisition.acquisition_ref,
        "verification_ref": verification.verification_ref,
        "generator_git_commit": commit,
        "checked_at": _GATED_AT,
        "gate_actor_instance_id": "fixture:gate:source-boundary",
    }

    dirty = source / "untracked-gate.py"
    dirty.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(G3AssetPublicationError, match="lifecycle evidence replay"):
        gate_stage0_g3_assets_from_evidence(
            requirements,
            layout,
            plan,
            source,
            fixture.data_root,
            **gate_kwargs,
        )
    dirty.unlink()

    monkeypatch.setattr(
        publication_module,
        "__file__",
        str(_ROOT / "src/param_importance_nlp/g3_asset_publication.py"),
    )
    with pytest.raises(G3AssetPublicationError, match="lifecycle evidence replay"):
        gate_stage0_g3_assets_from_evidence(
            requirements,
            layout,
            plan,
            source,
            fixture.data_root,
            **gate_kwargs,
        )
    monkeypatch.setattr(
        publication_module,
        "__file__",
        str(source / "src/param_importance_nlp/g3_asset_publication.py"),
    )

    drifted = False

    def drifting_probe(*args: Any, **kwargs: Any) -> Any:
        nonlocal drifted
        result = _fake_probe(*args, **kwargs)
        if not drifted:
            drifted = True
            target = source / "ops/stage0/materialize_and_publish_g3.py"
            target.write_bytes(target.read_bytes() + b"\n# mid-probe drift\n")
        return result

    monkeypatch.setattr(publication_module, "_run_semantic_probe", drifting_probe)
    with pytest.raises(G3AssetPublicationError, match="source binding drifted"):
        gate_stage0_g3_assets_from_evidence(
            requirements,
            layout,
            plan,
            source,
            fixture.data_root,
            **gate_kwargs,
        )

    layout_value = dict(load_canonical_json(layout))
    for entry in layout_value["entries"]:
        assert not (fixture.data_root / entry["manifest_ref"]).exists()
        assert not (fixture.data_root / entry["qualification_ref"]).exists()
        assert not (
            fixture.data_root
            / "manifests/evidence/g3"
            / entry["logical_name"]
        ).exists()


def test_legacy_all_in_one_publication_is_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    common = {
        "generator_git_commit": "a" * 40,
        "checked_at": _GATED_AT,
    }
    with pytest.raises(G3AssetPublicationError, match="disabled"):
        publish_stage0_g3_prerequisites(
            fixture.requirements_path,
            fixture.layout_path,
            fixture.data_root,
            **common,
        )
    with pytest.raises(G3AssetPublicationError, match="disabled"):
        publish_stage0_g3_assets(
            fixture.requirements_path,
            fixture.layout_path,
            fixture.data_root,
            **common,
        )
    with pytest.raises(G3AssetPublicationError, match="disabled"):
        publication_module._publish_stage0_g3_selected(
            fixture.requirements_path,
            fixture.layout_path,
            fixture.data_root,
            include_derived=True,
            pre_exposure_check=None,
            **common,
        )
    with pytest.raises(G3AssetPublicationError, match="disabled"):
        publication_module._publish_entry(
            root=fixture.data_root,
            requirements={},
            layout={},
            entry={},
            requirement={},
            published_asset_ids={},
            pre_exposure_check=None,
            **common,
        )
    assert not (fixture.data_root / "manifests/candidates").exists()


@pytest.mark.parametrize(
    "key",
    (
        "api_key_backup",
        "access-token-old",
        "refresh_token_copy",
        "auth.token.backup",
        "secret_key_previous",
        "runtime_url_cache",
        "x-amz-signature-shadow",
    ),
)
def test_lifecycle_evidence_rejects_sensitive_key_variants(key: str) -> None:
    with pytest.raises(G3LifecycleEvidenceError, match="forbidden"):
        lifecycle_module._reject_secrets_and_urls({key: "redacted"})
    lifecycle_module._reject_secrets_and_urls(
        {"special_tokens": ["<pad>"], "runtime_urls_persisted": False}
    )


@pytest.mark.parametrize(
    ("guard", "error_type"),
    (
        (lifecycle_module._zero_network_guard, G3NetworkEgressAttempt),
        (publication_module._socket_egress_guard, NetworkEgressAttempt),
    ),
)
def test_verify_and_gate_guards_block_existing_sockets_and_children(
    guard: Any, error_type: type[Exception]
) -> None:
    datagram = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with guard() as attempts:
            with pytest.raises(error_type):
                datagram.sendto(b"blocked", ("127.0.0.1", 9))
            with pytest.raises(error_type):
                subprocess.run(
                    [sys.executable, "-c", "pass"],
                    check=True,
                )
        assert attempts == [2]
    finally:
        datagram.close()
