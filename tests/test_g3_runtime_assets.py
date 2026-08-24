"""Formal runtime consumption of committed Stage 0.04 G3 resolution."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import param_importance_nlp.g3_runtime_assets as runtime_assets_module
from param_importance_nlp.asset_layout import layout_artifact_hash
from param_importance_nlp.asset_requirements import requirements_artifact_hash
from param_importance_nlp.contracts import canonical_json_bytes, load_canonical_json
from param_importance_nlp.g3_gate import evaluate_stage0_g3
from param_importance_nlp.g3_runtime_assets import (
    FormalG3RuntimeAssets,
    G3RuntimeAssetError,
    current_g3_source_refs,
    formal_pile_route,
    reject_legacy_provider_paths,
)
from param_importance_nlp.providers import PythiaSamplingDesign, hash_local_directory
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from tests.test_g3_gate import _materialize_fixture


_CHECKED_AT = "2026-08-03T08:00:00Z"
_FIXTURE_SOURCE_HEAD = "a" * 40
_ORIGINAL_ASSERT_SOURCE_COMPATIBLE = (
    runtime_assets_module._assert_g3_source_commit_compatible
)


@pytest.fixture(autouse=True)
def _bind_fixture_source_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_assets_module,
        "_current_g3_source_head",
        lambda source_root=None: _FIXTURE_SOURCE_HEAD,
    )
    monkeypatch.setattr(
        runtime_assets_module,
        "_assert_g3_source_commit_compatible",
        lambda source_commit, *, source_root=None, critical_source_refs=None: None,
    )


def _request(reference: str | None, **provider_paths: str) -> object:
    evidence_refs = {} if reference is None else {"g3_resolution": reference}
    return SimpleNamespace(
        config=SimpleNamespace(
            run_intent="formal",
            providers={
                "model_manifest_ref": provider_paths.get(
                    "model_manifest_ref", "wrong/config/model.json"
                ),
                "data_root_ref": provider_paths.get(
                    "data_root_ref", "wrong/config/data-root"
                ),
            },
        ),
        environment=SimpleNamespace(evidence_refs=evidence_refs),
    )


def _publish_resolution(
    fixture: Any,
    *,
    output_dir: str = "runs/stage0-g3",
    task_id: str = "stage0.04_assets_and_manifests",
) -> tuple[dict[str, Any], str]:
    resolution = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )
    source_refs = list(current_g3_source_refs())
    source_refs.append(str(resolution["requirements_ref"]))
    for entry in resolution["entries"]:
        source_refs.extend(
            str(entry[field])
            for field in (
                "manifest_ref",
                "asset_root_ref",
                "candidate_ref",
                "qualification_ref",
                "acquisition_ref",
                "verification_ref",
                "semantic_evidence_ref",
            )
        )
    published = TaskArtifactStore(fixture.data_root, output_dir).publish(
        task_id=task_id,
        artifact_kind="asset_resolution",
        config_hash="a" * 64,
        run_intent="formal",
        payload=resolution,
        formal_eligible=True,
        source_refs=tuple(dict.fromkeys(source_refs)),
    )
    return resolution, published.commit_ref


def _load_runtime(fixture: Any, reference: str) -> FormalG3RuntimeAssets:
    return FormalG3RuntimeAssets.from_request(
        _request(reference),
        fixture.data_root,
        requirements_path=fixture.requirements_path,
        layout_path=fixture.layout_path,
    )


def test_runtime_uses_only_resolution_logical_ids_and_qualifies_glue(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    resolution, reference = _publish_resolution(fixture)
    runtime = FormalG3RuntimeAssets.from_request(
        _request(
            reference,
            model_manifest_ref="intentionally/wrong/model.json",
            data_root_ref="intentionally/wrong/data",
        ),
        fixture.data_root,
        requirements_path=fixture.requirements_path,
        layout_path=fixture.layout_path,
    )

    glue = runtime.resolve("glue-sst2-pretokenized", expected_kind="glue_derived")
    model = runtime.resolve("pythia-14m-step0", expected_kind="model")
    tokenizer = runtime.resolve("pythia-tokenizer", expected_kind="tokenizer")
    provenance = glue.provenance()

    assert resolution["status"] == "PASS"
    assert glue.storage_kind == "hf_load_from_disk"
    assert glue.glue_task_name == "sst2"
    assert glue.require_glue_route(task_name="sst-2", split="validation") == "sst2"
    assert glue.directory_content_sha256 == hash_local_directory(glue.resolved.root)
    assert provenance["g3_resolution_ref"] == reference
    assert provenance["g3_resolution_artifact_hash"] == resolution["artifact_hash"]
    assert provenance["ready_manifest_sha256"] == glue.ready_manifest_sha256
    assert (
        provenance["qualification_artifact_hash"]
        == glue.qualification_artifact_hash
    )
    assert provenance["acquisition_ref"] == glue.acquisition_ref
    assert provenance["acquisition_sha256"] == glue.acquisition_sha256
    assert provenance["verification_ref"] == glue.verification_ref
    assert provenance["verification_sha256"] == glue.verification_sha256
    assert provenance["manifest_ref"] != "intentionally/wrong/model.json"
    lineage = runtime.runtime_lineage_sha256(model, glue, tokenizer)
    assert lineage == runtime.runtime_lineage_sha256(tokenizer, model, glue)
    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_LINEAGE_REQUIRES_THREE_ASSETS"
    ):
        runtime.runtime_lineage_sha256(glue)
    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_LINEAGE_REQUIRES_THREE_ASSETS"
    ):
        runtime.runtime_lineage_sha256(model, glue, tokenizer, glue)


def test_runtime_rejects_failed_source_compatibility_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    _resolution, reference = _publish_resolution(fixture)

    def _reject_source(*_args: object, **_kwargs: object) -> None:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_CRITICAL_SOURCE_DRIFT")

    monkeypatch.setattr(
        runtime_assets_module,
        "_assert_g3_source_commit_compatible",
        _reject_source,
    )

    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_RESOLUTION_CRITICAL_SOURCE_DRIFT"
    ):
        _load_runtime(fixture, reference)


def test_runtime_binds_manifest_and_qualification_to_producer_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    monkeypatch.setattr(
        runtime_assets_module,
        "_current_g3_source_head",
        lambda source_root=None: "b" * 40,
    )
    _resolution, reference = _publish_resolution(fixture)
    runtime = _load_runtime(fixture, reference)

    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_RESOLUTION_ASSET_BINDING_DRIFT"
    ):
        runtime.resolve("pythia-14m-step0", expected_kind="model")


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def test_source_compatibility_allows_unrelated_commit_but_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "source-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "g3-runtime@example.invalid")
    _git(repo, "config", "user.name", "G3 Runtime Test")
    for reference in runtime_assets_module._G3_CRITICAL_SOURCE_REFS:
        path = repo.joinpath(*reference.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"baseline:{reference}\n", encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "g3 producer")
    source_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(
        runtime_assets_module,
        "_current_g3_source_head",
        lambda source_root=None: _git(Path(source_root or repo), "rev-parse", "HEAD"),
    )

    (repo / "unrelated.txt").write_text("later stage\n", encoding="utf-8")
    _git(repo, "add", "unrelated.txt")
    _git(repo, "commit", "-m", "later unrelated stage")
    _ORIGINAL_ASSERT_SOURCE_COMPATIBLE(source_commit, source_root=repo)

    critical_ref = runtime_assets_module._G3_CRITICAL_SOURCE_REFS[0]
    critical_path = repo.joinpath(*critical_ref.split("/"))
    critical_path.write_text("dirty critical source\n", encoding="utf-8")
    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_CRITICAL_SOURCE_WORKTREE_DIRTY"
    ):
        _ORIGINAL_ASSERT_SOURCE_COMPATIBLE(source_commit, source_root=repo)
    _git(repo, "add", critical_ref)
    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_CRITICAL_SOURCE_WORKTREE_DIRTY"
    ):
        _ORIGINAL_ASSERT_SOURCE_COMPATIBLE(source_commit, source_root=repo)
    _git(repo, "commit", "-m", "critical source changed")
    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_RESOLUTION_CRITICAL_SOURCE_DRIFT"
    ):
        _ORIGINAL_ASSERT_SOURCE_COMPATIBLE(source_commit, source_root=repo)


def _dynamic_source_refs(
    *,
    requirements_ref: str = "configs/stage0/g3-asset-requirements-v3.json",
    layout_ref: str = "configs/stage0/g3-asset-layout-v5.json",
    download_plan_ref: str = "configs/stage0/g3-download-plan-v5.json",
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    commit = _FIXTURE_SOURCE_HEAD
    critical = (
        requirements_ref,
        layout_ref,
        download_plan_ref,
        *runtime_assets_module._G3_CRITICAL_SOURCE_REFS[3:],
        *extra,
    )
    return tuple(f"git-source/{commit}/{reference}" for reference in critical) + (
        requirements_ref,
    )


def test_committed_source_refs_accept_versioned_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[str, ...]] = {}

    def capture(
        source_commit: str,
        *,
        source_root: str | Path | None = None,
        critical_source_refs: tuple[str, ...] = (),
    ) -> None:
        captured["refs"] = critical_source_refs

    monkeypatch.setattr(runtime_assets_module, "_assert_g3_source_commit_compatible", capture)
    source_commit = _validate_source_refs_for_test(
        _dynamic_source_refs(),
        requirements_ref="configs/stage0/g3-asset-requirements-v3.json",
    )
    assert source_commit == _FIXTURE_SOURCE_HEAD
    assert captured["refs"][:3] == (
        "configs/stage0/g3-asset-requirements-v3.json",
        "configs/stage0/g3-asset-layout-v5.json",
        "configs/stage0/g3-download-plan-v5.json",
    )


def _validate_source_refs_for_test(
    source_refs: tuple[str, ...], *, requirements_ref: str
) -> str:
    return runtime_assets_module._validate_committed_source_refs(
        source_refs,
        {"requirements_ref": requirements_ref, "entries": []},
    )


@pytest.mark.parametrize(
    "source_refs",
    (
        _dynamic_source_refs(
            layout_ref="configs/stage0/g3-asset-layout-v1.json"
        )
        + (
            f"git-source/{_FIXTURE_SOURCE_HEAD}/configs/stage0/g3-asset-layout-v5.json",
        ),
        _dynamic_source_refs(extra=("src/extra-uncontrolled.py",)),
    ),
)
def test_committed_source_refs_reject_mixed_or_extra_git_refs(
    source_refs: tuple[str, ...],
) -> None:
    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_RESOLUTION_SOURCE_REFS_INCOMPLETE"
    ):
        _validate_source_refs_for_test(
            source_refs,
            requirements_ref="configs/stage0/g3-asset-requirements-v3.json",
        )


def test_source_compatibility_checks_dynamic_control_plane_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "source-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "g3-runtime@example.invalid")
    _git(repo, "config", "user.name", "G3 Runtime Test")
    dynamic = (
        "configs/stage0/g3-asset-requirements-v3.json",
        "configs/stage0/g3-asset-layout-v5.json",
        "configs/stage0/g3-download-plan-v5.json",
        *runtime_assets_module._G3_CRITICAL_SOURCE_REFS[3:],
    )
    for reference in dynamic:
        path = repo.joinpath(*reference.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"baseline:{reference}\n", encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "g3 producer")
    source_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(
        runtime_assets_module,
        "_current_g3_source_head",
        lambda source_root=None: _git(Path(source_root or repo), "rev-parse", "HEAD"),
    )
    _ORIGINAL_ASSERT_SOURCE_COMPATIBLE(
        source_commit,
        source_root=repo,
        critical_source_refs=dynamic,
    )
    changed = repo.joinpath(*dynamic[1].split("/"))
    changed.write_text("changed\n", encoding="utf-8")
    _git(repo, "add", dynamic[1])
    _git(repo, "commit", "-m", "critical source changed")
    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_RESOLUTION_CRITICAL_SOURCE_DRIFT"
    ):
        _ORIGINAL_ASSERT_SOURCE_COMPATIBLE(
            source_commit,
            source_root=repo,
            critical_source_refs=dynamic,
        )


def test_runtime_rejects_legacy_provider_path_fallback() -> None:
    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_LEGACY_PROVIDER_PATH_FORBIDDEN"
    ):
        reject_legacy_provider_paths(
            {
                "model_manifest_ref": "manifests/model/legacy.json",
                "model_root_ref": None,
            }
        )
    reject_legacy_provider_paths(
        {
            "model_manifest_ref": None,
            "model_root_ref": None,
            "data_manifest_ref": None,
            "data_root_ref": None,
            "tokenizer_manifest_ref": None,
            "tokenizer_root_ref": None,
        }
    )


def test_runtime_rejects_glue_task_or_split_route_mismatch(tmp_path: Path) -> None:
    fixture = _materialize_fixture(tmp_path)
    _resolution, reference = _publish_resolution(fixture)
    glue = _load_runtime(fixture, reference).resolve(
        "glue-mnli-pretokenized", expected_kind="glue_derived"
    )

    with pytest.raises(G3RuntimeAssetError, match="GLUE_TASK_ROUTE_MISMATCH"):
        glue.require_glue_route(task_name="sst2", split="validation_matched")
    with pytest.raises(G3RuntimeAssetError, match="GLUE_SPLIT_ROUTE_MISMATCH"):
        glue.require_glue_route(task_name="mnli", split="validation")


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        (None, "G3_RUNTIME_RESOLUTION_REF_REQUIRED"),
        (
            "runs/missing/commits/asset_resolution.json",
            "G3_RUNTIME_REF_MISSING_OR_ESCAPE",
        ),
    ],
)
def test_runtime_fails_closed_when_resolution_is_missing(
    tmp_path: Path,
    reference: str | None,
    message: str,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    with pytest.raises(G3RuntimeAssetError, match=message):
        FormalG3RuntimeAssets.from_request(
            _request(reference),
            fixture.data_root,
            requirements_path=fixture.requirements_path,
            layout_path=fixture.layout_path,
        )


def test_runtime_rejects_copied_commit_outside_its_authoritative_store(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    _resolution, reference = _publish_resolution(fixture)
    original = fixture.data_root.joinpath(*reference.split("/"))
    copied = fixture.data_root / "copied" / "commits" / "asset_resolution.json"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(original, copied)

    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_RESOLUTION_STORE_BINDING_INVALID"
    ):
        _load_runtime(fixture, "copied/commits/asset_resolution.json")


def test_runtime_rejects_symlinked_resolution_commit(tmp_path: Path) -> None:
    fixture = _materialize_fixture(tmp_path)
    _resolution, reference = _publish_resolution(fixture)
    original = fixture.data_root.joinpath(*reference.split("/"))
    linked = fixture.data_root / "linked" / "commits" / "asset_resolution.json"
    linked.parent.mkdir(parents=True)
    try:
        linked.symlink_to(original)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")

    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_LINK_LIKE_FORBIDDEN"
    ):
        _load_runtime(fixture, "linked/commits/asset_resolution.json")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary")
def test_safe_workspace_path_rejects_windows_junction(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "target"
    junction = workspace / "junction"
    target.mkdir(parents=True)
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0 or not junction.is_junction():
        pytest.skip("junction creation unavailable")

    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_LINK_LIKE_FORBIDDEN"
    ):
        runtime_assets_module._safe_workspace_path(
            workspace.resolve(),
            "junction",
            field="test.junction",
            directory=True,
        )


def test_runtime_rejects_wrong_resolution_commit_identity(tmp_path: Path) -> None:
    fixture = _materialize_fixture(tmp_path)
    _resolution, reference = _publish_resolution(
        fixture,
        output_dir="runs/wrong-g3",
        task_id="stage0.04_wrong_assets_and_manifests",
    )

    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_RESOLUTION_COMMIT_IDENTITY_INVALID"
    ):
        _load_runtime(fixture, reference)


def test_runtime_rejects_resolution_stale_against_current_requirements(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    _resolution, reference = _publish_resolution(fixture)
    requirements = deepcopy(load_canonical_json(fixture.requirements_path))
    layout = deepcopy(load_canonical_json(fixture.layout_path))
    assert isinstance(requirements, dict) and isinstance(layout, dict)
    requirements["created_at"] = "2026-08-03T15:29:41+08:00"
    requirements["artifact_hash"] = requirements_artifact_hash(requirements)
    layout["requirements_sha256"] = requirements["artifact_hash"]
    layout["artifact_hash"] = layout_artifact_hash(layout)
    fixture.requirements_path.write_bytes(canonical_json_bytes(requirements))
    fixture.layout_path.write_bytes(canonical_json_bytes(layout))

    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_RESOLUTION_STALE_OR_BLOCKED"
    ):
        _load_runtime(fixture, reference)


@pytest.mark.parametrize("corrupt", ("manifest", "qualification"))
def test_runtime_rejects_manifest_or_qualification_corruption(
    tmp_path: Path,
    corrupt: str,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    resolution, reference = _publish_resolution(fixture)
    entry = next(
        item for item in resolution["entries"] if item["logical_name"] == "pythia-14m-step0"
    )
    field = "manifest_ref" if corrupt == "manifest" else "qualification_ref"
    path = fixture.data_root.joinpath(*entry[field].split("/"))
    path.write_bytes(b"{}")
    runtime = _load_runtime(fixture, reference)

    with pytest.raises(G3RuntimeAssetError, match="G3_RUNTIME_"):
        runtime.resolve("pythia-14m-step0", expected_kind="model")


def test_stage0_1_4_5_pile_routes_and_workload_budgets_are_explicit(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    _resolution, reference = _publish_resolution(fixture)
    runtime = _load_runtime(fixture, reference)
    pile = runtime.resolve("pile-selected-prefix", expected_kind="pile")

    expected = {
        0: ("debug", PythiaSamplingDesign.WITHOUT_REPLACEMENT),
        1: ("train", PythiaSamplingDesign.WITHOUT_REPLACEMENT),
        4: ("train", PythiaSamplingDesign.WITHOUT_REPLACEMENT),
        5: ("train", PythiaSamplingDesign.WITHOUT_REPLACEMENT),
    }
    for stage, (split, design) in expected.items():
        route = formal_pile_route(
            stage=stage,
            evaluation=False,
            declared_sampling_design="without_replacement_frozen_epoch",
            configured_split=split,
        )
        assert (route.split, route.sampling_design) == (split, design)
        assert runtime.pile_split_interval(pile, route.split)[1] > 0
        runtime.validate_pile_budget(
            stage=stage,
            split=route.split,
            requested_records=1,
            **({"max_steps": 1, "global_batch_size": 1} if stage else {}),
        )

    with pytest.raises(
        G3RuntimeAssetError, match="G3_RUNTIME_PILE_SPLIT_BUDGET_EXCEEDED"
    ):
        runtime.validate_pile_budget(
            stage=5,
            split="train",
            requested_records=2,
            max_steps=2,
            global_batch_size=1,
        )


def test_stage2_and_stage3_routes_own_sampling_outside_the_mmap_reader() -> None:
    stage2 = formal_pile_route(
        stage=2,
        evaluation=False,
        declared_sampling_design="with_replacement_versioned_universe",
        configured_split="sampling_universe",
    )
    stage3 = formal_pile_route(
        stage=3,
        evaluation=False,
        declared_sampling_design="disjoint_frozen_probe_panel",
        configured_split="probe",
    )

    assert (stage2.split, stage2.sampling_design) == ("sampling_universe", None)
    assert (stage3.split, stage3.sampling_design) == ("probe", None)
    with pytest.raises(G3RuntimeAssetError, match="PILE_SPLIT_DECLARATION_DRIFT"):
        formal_pile_route(
            stage=2,
            evaluation=False,
            declared_sampling_design="with_replacement_versioned_universe",
            configured_split="validation",
        )
