from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import pytest

from param_importance_nlp import g3_runtime_assets
from param_importance_nlp.contracts import (
    GateRecord,
    GateStatus,
    ResolvedConfig,
    canonical_json_hash,
    load_canonical_json,
    loads_strict_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG, RunnerKind
from param_importance_nlp.experiments import stage01_task_runners as stage01
from param_importance_nlp.g3_gate import (
    GATE_IDS,
    g3_resolution_artifact_hash,
)
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)
from param_importance_nlp.storage import DATA_ROOT_ENV


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "stage0.04_assets_and_manifests"
REQUIREMENTS_PATH = ROOT / "configs/stage0/g3-asset-requirements-v1.json"
LAYOUT_PATH = ROOT / "configs/stage0/g3-asset-layout-v1.json"
TEST_HEAD = "a" * 40
LATER_HEAD = "b" * 40


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


ACQUISITION_SHA256 = _digest("g3-acquisition")
ACQUISITION_REF = (
    f"manifests/evidence/g3/acquisition/{ACQUISITION_SHA256}.json"
)


def _formal_config(output: str) -> ResolvedConfigV2:
    task = DEFAULT_TASK_CATALOG.get(TASK_ID)
    value = deepcopy(
        load_canonical_json(ROOT / "configs/local-fixtures/resolved-config-v1.json")
    )
    value["identity"].update(
        {
            "stage": task.stage,
            "task": TASK_ID,
            "run_intent": "formal",
            "formal_eligible": True,
        }
    )
    value["runtime"]["allow_dirty_worktree"] = False
    value["loss"].update(
        {"task_type": "sequence_classification", "weighting": "sample"}
    )
    value["data"].update(
        {"statistical_unit": "sample", "weight_unit": "sample"}
    )
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfigV2.resolve(
        ResolvedConfig.from_mapping(value),
        task_id=TASK_ID,
        overrides={
            "providers": {"num_labels": 3},
            "orchestration": {"input_result_refs": []},
            "artifacts": {"output_dir": output},
        },
    )


def _request(output: str) -> TaskExecutionRequest:
    config = _formal_config(output)
    return TaskExecutionRequest(
        config=config,
        task=config.task_definition,
        environment=TaskRuntimeEnvironment(),
    )


def _resolution(*, blocked: bool = False) -> dict[str, Any]:
    requirements = loads_strict_json(REQUIREMENTS_PATH.read_bytes())
    layout = loads_strict_json(LAYOUT_PATH.read_bytes())
    assert isinstance(requirements, dict) and isinstance(layout, dict)
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(layout["entries"]):
        missing = blocked and index == 0
        entries.append(
            {
                "logical_name": item["logical_name"],
                "kind": item["kind"],
                "requirement_name": item["requirement_name"],
                "gate_ids": list(item["gate_ids"]),
                "manifest_ref": item["manifest_ref"],
                "asset_root_ref": item["asset_root_ref"],
                "qualification_ref": item["qualification_ref"],
                "status": "BLOCKED" if missing else "PASS",
                "checks": {
                    "qualified_resolution": not missing,
                    "verification_report_matches": not missing,
                },
                "reasons": ["INPUT_MISSING"] if missing else [],
                "asset_id": _digest(f"asset:{index}"),
                "candidate_id": _digest(f"candidate:{index}"),
                "candidate_ref": (
                    f"manifests/candidates/g3/{item['logical_name']}/"
                    f"{_digest(f'candidate:{index}')}.json"
                ),
                "candidate_sha256": _digest(f"candidate-file:{index}"),
                "ready_manifest_sha256": _digest(f"ready:{index}"),
                "qualification_artifact_hash": _digest(
                    f"qualification:{index}"
                ),
                "acquisition_ref": ACQUISITION_REF,
                "acquisition_sha256": ACQUISITION_SHA256,
                "verification_ref": f"evidence/verification-{index}.json",
                "verification_sha256": _digest(f"verification:{index}"),
                "semantic_evidence_ref": f"evidence/semantic-{index}.json",
                "semantic_evidence_sha256": _digest(f"semantic:{index}"),
                "semantic_evidence_artifact_hash": _digest(
                    f"semantic-artifact:{index}"
                ),
                "files_checked": 1,
                "bytes_checked": index + 1,
                "expected_file_policy": "requirements_exact",
            }
        )
    gates = []
    for index, gate_id in enumerate(GATE_IDS):
        gate_blocked = blocked and index == 0
        gates.append(
            GateRecord(
                gate_id=gate_id,
                stage=0,
                status=(
                    GateStatus.BLOCKED if gate_blocked else GateStatus.PASS
                ),
                checked_at="2026-08-03T09:00:00Z",
                evidence_refs=(layout["requirements_ref"],),
                reasons=("qualification missing",) if gate_blocked else (),
            ).to_dict()
        )
    value: dict[str, Any] = {
        "schema_version": "stage0-g3-resolution-audit-v1",
        "scope": "formal",
        "status": "BLOCKED" if blocked else "PASS",
        "checked_at": "2026-08-03T09:00:00Z",
        "requirements_ref": layout["requirements_ref"],
        "requirements_artifact_hash": requirements["artifact_hash"],
        "layout_artifact_hash": layout["artifact_hash"],
        "entries": entries,
        "gates": gates,
    }
    value["artifact_hash"] = g3_resolution_artifact_hash(value)
    return value


def _prepare_formal_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> stage01.Stage01CompositeTaskRunner:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(stage01, "_formal_guard", lambda request, root: None)
    monkeypatch.setattr(
        stage01,
        "_input_evidence",
        lambda request, root: ([], ()),
    )
    monkeypatch.setattr(
        stage01,
        "_git_constrained_g3_sources",
        _source_binding,
    )
    monkeypatch.setattr(
        stage01,
        "_stable_g3_evidence_identity",
        lambda binding, data_root: stage01._G3EvidenceIdentity(
            checked_at="2026-08-03T09:00:00Z",
            producer_git_commit=TEST_HEAD,
        ),
    )
    monkeypatch.setattr(
        stage01,
        "_assert_g3_producer_commit_compatible",
        lambda binding, producer_commit: None,
    )
    return stage01.Stage01CompositeTaskRunner(RunnerKind.ASSET, tmp_path)


def _source_binding(*, head_commit: str = TEST_HEAD) -> stage01._G3SourceBinding:
    return stage01._G3SourceBinding(
        source_root=ROOT,
        head_commit=head_commit,
        requirements_path=REQUIREMENTS_PATH,
        requirements_file_sha256=hashlib.sha256(
            REQUIREMENTS_PATH.read_bytes()
        ).hexdigest(),
        layout_path=LAYOUT_PATH,
        layout_file_sha256=hashlib.sha256(LAYOUT_PATH.read_bytes()).hexdigest(),
    )


def _payload(tmp_path: Path, reference: str) -> dict[str, Any]:
    loaded = load_committed_task_artifact(
        tmp_path,
        reference,
        require_formal=True,
    )
    return dict(loaded.payload)


def test_formal_g3_runner_publishes_three_distinct_validated_outputs_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _prepare_formal_runner(monkeypatch, tmp_path)
    request = _request("runs/formal-g3-success")
    calls: list[tuple[Path, Path, Path]] = []

    def evaluate(
        requirements_path: Path,
        layout_path: Path,
        data_root: Path,
        *,
        checked_at: str,
    ) -> dict[str, Any]:
        assert checked_at == "2026-08-03T09:00:00Z"
        calls.append((requirements_path, layout_path, data_root))
        return _resolution()

    monkeypatch.setattr(stage01, "evaluate_stage0_g3", evaluate)
    first = runner.run(request)
    second = runner.run(request)

    assert first.status.value == "PASS"
    assert second.status.value == "PASS"
    assert second.metadata["restored"] is True
    assert calls == [
        (REQUIREMENTS_PATH, LAYOUT_PATH, tmp_path.resolve()),
        (REQUIREMENTS_PATH, LAYOUT_PATH, tmp_path.resolve()),
    ]
    payloads = {
        kind: _payload(tmp_path, reference)
        for kind, reference in first.artifact_refs.items()
    }
    assert payloads["asset_manifest"]["schema_version"] == (
        "stage0-g3-asset-manifest-index-v1"
    )
    assert payloads["asset_audit"]["schema_version"] == (
        "stage0-g3-asset-audit-v1"
    )
    assert payloads["asset_resolution"]["schema_version"] == (
        "stage0-g3-resolution-audit-v1"
    )
    assert len(payloads["asset_manifest"]["entries"]) == 13
    assert len(payloads["asset_audit"]["entries"]) == 13
    assert len(payloads["asset_audit"]["gates"]) == 5
    assert len(payloads["asset_resolution"]["entries"]) == 13
    assert set(payloads["asset_manifest"]["entries"][0]) == {
        "logical_name",
        "kind",
        "requirement_name",
        "manifest_ref",
        "asset_id",
        "candidate_id",
        "candidate_ref",
        "candidate_sha256",
        "ready_manifest_sha256",
    }
    assert set(payloads["asset_audit"]["entries"][0]) == {
        "logical_name",
        "kind",
        "status",
        "checks",
        "reasons",
        "qualification_ref",
        "qualification_artifact_hash",
        "acquisition_ref",
        "acquisition_sha256",
        "verification_ref",
        "verification_sha256",
        "semantic_evidence_ref",
        "semantic_evidence_sha256",
        "semantic_evidence_artifact_hash",
    }
    assert all(
        gate["status"] == "PASS" for gate in payloads["asset_audit"]["gates"]
    )
    assert len(
        {
            payloads[kind]["artifact_hash"]
            for kind in ("asset_manifest", "asset_audit", "asset_resolution")
        }
    ) == 3
    for kind in ("asset_manifest", "asset_audit"):
        without_hash = {
            key: value
            for key, value in payloads[kind].items()
            if key != "artifact_hash"
        }
        assert payloads[kind]["artifact_hash"] == canonical_json_hash(
            without_hash
        )
    for kind, reference in first.artifact_refs.items():
        loaded = load_committed_task_artifact(
            tmp_path,
            reference,
            require_formal=True,
        )
        assert loaded.source_refs[0].startswith(
            f"git-source/{TEST_HEAD}/"
        )
        assert ACQUISITION_REF in loaded.source_refs
        assert "evidence/semantic-0.json" in loaded.source_refs
        if kind != "asset_resolution":
            assert (
                loaded.payload["source_binding"]["producer_git_commit"]
                == TEST_HEAD
            )


def test_formal_g3_restore_accepts_later_unrelated_head_and_keeps_producer_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _prepare_formal_runner(monkeypatch, tmp_path)
    request = _request("runs/formal-g3-unrelated-head")
    current_head = [TEST_HEAD]
    monkeypatch.setattr(
        stage01,
        "_git_constrained_g3_sources",
        lambda: _source_binding(head_commit=current_head[0]),
    )
    monkeypatch.setattr(
        stage01,
        "evaluate_stage0_g3",
        lambda *args, **kwargs: _resolution(),
    )

    first = runner.run(request)
    current_head[0] = LATER_HEAD
    restored = runner.run(request)

    assert first.status.value == "PASS"
    assert restored.status.value == "PASS"
    assert restored.metadata["restored"] is True
    loaded = load_committed_task_artifact(
        tmp_path,
        restored.artifact_refs["asset_resolution"],
        require_formal=True,
    )
    git_commits = {
        reference.split("/", 2)[1]
        for reference in loaded.source_refs
        if reference.startswith("git-source/")
    }
    assert git_commits == {TEST_HEAD}


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    (
        ("not_ancestor", "STAGE0_G3_ASSET_GENERATOR_COMMIT_NOT_ANCESTOR"),
        ("critical_drift", "STAGE0_G3_CRITICAL_SOURCE_DRIFT"),
        ("dirty_worktree", "STAGE0_G3_CRITICAL_SOURCE_DIRTY"),
        ("dirty_index", "STAGE0_G3_CRITICAL_SOURCE_DIRTY"),
        ("head_changed", "STAGE0_G3_SOURCE_BINDING_DRIFTED"),
        (
            "producer_ref_absent",
            "STAGE0_G3_ASSET_GENERATOR_SOURCE_REF_ABSENT",
        ),
    ),
)
def test_formal_g3_producer_commit_rejects_incompatible_source_state(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_error: str,
) -> None:
    binding = _source_binding(head_commit=LATER_HEAD)

    def run_git(
        source_root: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        assert source_root == ROOT
        returncode = 0
        stdout = ""
        if arguments[:2] == ("rev-parse", "--verify"):
            stdout = TEST_HEAD
        elif arguments == ("rev-parse", "HEAD"):
            stdout = TEST_HEAD if mode == "head_changed" else LATER_HEAD
        if mode == "not_ancestor" and arguments[:2] == (
            "merge-base",
            "--is-ancestor",
        ):
            returncode = 1
        elif (
            mode == "critical_drift"
            and arguments[:2] == ("diff", "--quiet")
            and f"{TEST_HEAD}..{LATER_HEAD}" in arguments
        ):
            returncode = 1
        elif (
            mode == "dirty_worktree"
            and arguments[:3] == ("diff", "--quiet", "--")
        ):
            returncode = 1
        elif (
            mode == "dirty_index"
            and arguments[:4] == ("diff", "--cached", "--quiet", "--")
        ):
            returncode = 1
        elif mode == "producer_ref_absent" and arguments[:2] == (
            "cat-file",
            "-e",
        ):
            returncode = 1
        return subprocess.CompletedProcess(
            args=arguments,
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(stage01, "_run_g3_git", run_git)
    with pytest.raises(ValueError, match=expected_error):
        stage01._assert_g3_producer_commit_compatible(binding, TEST_HEAD)


def test_formal_g3_producer_commit_accepts_ancestor_without_critical_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _source_binding(head_commit=LATER_HEAD)
    commands: list[tuple[str, ...]] = []

    def run_git(
        source_root: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        assert source_root == ROOT
        commands.append(arguments)
        return subprocess.CompletedProcess(
            args=arguments,
            returncode=0,
            stdout=(
                TEST_HEAD
                if arguments[:2] == ("rev-parse", "--verify")
                else LATER_HEAD
                if arguments == ("rev-parse", "HEAD")
                else ""
            ),
            stderr="",
        )

    monkeypatch.setattr(stage01, "_run_g3_git", run_git)
    stage01._assert_g3_producer_commit_compatible(binding, TEST_HEAD)

    assert (
        "merge-base",
        "--is-ancestor",
        TEST_HEAD,
        LATER_HEAD,
    ) in commands
    assert any(
        command[:3] == ("diff", "--quiet", f"{TEST_HEAD}..{LATER_HEAD}")
        for command in commands
    )


def test_formal_g3_producer_commit_git_integration_allows_unrelated_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "stage0-test@example.invalid")
    git("config", "user.name", "Stage0 Test")
    critical = source / "critical.py"
    critical.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "critical.py")
    git("commit", "--quiet", "-m", "producer")
    producer = git("rev-parse", "HEAD")
    unrelated = source / "unrelated.txt"
    unrelated.write_text("later\n", encoding="utf-8")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "unrelated")
    unrelated_head = git("rev-parse", "HEAD")
    monkeypatch.setattr(stage01, "_G3_CRITICAL_SOURCE_REFS", ("critical.py",))

    binding = stage01._G3SourceBinding(
        source_root=source.resolve(),
        head_commit=unrelated_head,
        requirements_path=critical,
        requirements_file_sha256=_digest("requirements"),
        layout_path=critical,
        layout_file_sha256=_digest("layout"),
    )
    stage01._assert_g3_producer_commit_compatible(binding, producer)

    critical.write_text("VALUE = 2\n", encoding="utf-8")
    git("add", "critical.py")
    git("commit", "--quiet", "-m", "critical drift")
    drifted_binding = stage01._G3SourceBinding(
        source_root=source.resolve(),
        head_commit=git("rev-parse", "HEAD"),
        requirements_path=critical,
        requirements_file_sha256=_digest("requirements"),
        layout_path=critical,
        layout_file_sha256=_digest("layout"),
    )
    with pytest.raises(ValueError, match="STAGE0_G3_CRITICAL_SOURCE_DRIFT"):
        stage01._assert_g3_producer_commit_compatible(
            drifted_binding,
            producer,
        )


@pytest.mark.parametrize("mode", ("ref_hash_mismatch", "non_unique_report"))
def test_formal_g3_resolution_rejects_inexact_acquisition_binding(mode: str) -> None:
    resolution = _resolution()
    if mode == "ref_hash_mismatch":
        resolution["entries"][0]["acquisition_sha256"] = _digest("other")
    else:
        other_sha = _digest("other")
        resolution["entries"][0]["acquisition_sha256"] = other_sha
        resolution["entries"][0]["acquisition_ref"] = (
            f"manifests/evidence/g3/acquisition/{other_sha}.json"
        )
    resolution["artifact_hash"] = g3_resolution_artifact_hash(resolution)

    with pytest.raises(
        ValueError,
        match=(
            "STAGE0_G3_ENTRY_ACQUISITION_BINDING_INVALID|"
            "STAGE0_G3_ACQUISITION_REPORT_NOT_UNIQUE"
        ),
    ):
        stage01._require_formal_g3_pass(resolution)


def test_formal_g3_critical_source_contract_matches_runtime_consumer() -> None:
    assert (
        stage01._G3_CRITICAL_SOURCE_REFS
        == g3_runtime_assets._G3_CRITICAL_SOURCE_REFS
    )


def test_formal_g3_critical_source_rejects_link_like_parent_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = source / "critical/file.py"
    target.parent.mkdir(parents=True)
    target.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(
        stage01,
        "_G3_CRITICAL_SOURCE_REFS",
        ("critical/file.py",),
    )
    monkeypatch.setattr(
        stage01,
        "_is_link_like",
        lambda path: path == target.parent,
    )

    with pytest.raises(
        ValueError,
        match="STAGE0_G3_CRITICAL_SOURCE_LINK_FORBIDDEN",
    ):
        stage01._assert_g3_critical_source_paths(source)


def test_formal_g3_source_refs_allow_existing_exact_producer_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage01,
        "_assert_g3_producer_commit_compatible",
        lambda binding, producer_commit: None,
    )
    existing = (
        f"git-source/{TEST_HEAD}/{stage01._G3_CRITICAL_SOURCE_REFS[0]}"
    )
    refs = stage01._formal_g3_source_refs(
        _source_binding(),
        _resolution(),
        (existing,),
        producer_git_commit=TEST_HEAD,
    )
    assert refs.count(existing) == 1


def test_formal_g3_source_refs_reject_ambiguous_input_git_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage01,
        "_assert_g3_producer_commit_compatible",
        lambda binding, producer_commit: None,
    )
    with pytest.raises(ValueError, match="STAGE0_G3_SOURCE_REFS_NOT_EXACT"):
        stage01._formal_g3_source_refs(
            _source_binding(),
            _resolution(),
            (f"git-source/{LATER_HEAD}/unrelated.py",),
            producer_git_commit=TEST_HEAD,
        )


@pytest.mark.parametrize("mode", ("blocked_missing_qualification", "exception"))
def test_formal_g3_runner_fails_before_publish_on_blocked_or_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    runner = _prepare_formal_runner(monkeypatch, tmp_path)
    request = _request(f"runs/formal-g3-{mode}")

    if mode == "exception":
        def evaluate(*args: object, **kwargs: object) -> dict[str, Any]:
            raise FileNotFoundError("qualification missing")

        monkeypatch.setattr(stage01, "evaluate_stage0_g3", evaluate)
    else:
        monkeypatch.setattr(
            stage01,
            "evaluate_stage0_g3",
            lambda *args, **kwargs: _resolution(blocked=True),
        )
    with pytest.raises(RuntimeError, match="STAGE0_G3_FORMAL_EVALUATION_FAILED"):
        runner.run(request)

    store = TaskArtifactStore(tmp_path, f"runs/formal-g3-{mode}")
    assert not any(store.commits.glob("*.json"))


@pytest.mark.parametrize("fail_after_publish", (1, 2))
def test_formal_g3_runner_resumes_after_partial_commit_without_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_after_publish: int,
) -> None:
    runner = _prepare_formal_runner(monkeypatch, tmp_path)
    request = _request(f"runs/formal-g3-partial-{fail_after_publish}")
    evaluations = 0

    def evaluate(*args: object, **kwargs: object) -> dict[str, Any]:
        nonlocal evaluations
        evaluations += 1
        return _resolution()

    monkeypatch.setattr(stage01, "evaluate_stage0_g3", evaluate)
    original_publish = TaskArtifactStore.publish
    published = 0

    def publish_then_interrupt(
        self: TaskArtifactStore,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal published
        result = original_publish(self, *args, **kwargs)
        published += 1
        if published == fail_after_publish:
            raise RuntimeError("injected interruption after commit")
        return result

    monkeypatch.setattr(TaskArtifactStore, "publish", publish_then_interrupt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        runner.run(request)
    store = TaskArtifactStore(
        tmp_path,
        f"runs/formal-g3-partial-{fail_after_publish}",
    )
    assert len(tuple(store.commits.glob("*.json"))) == fail_after_publish

    monkeypatch.setattr(TaskArtifactStore, "publish", original_publish)
    recovered = runner.run(request)
    assert recovered.status.value == "PASS"
    assert len(recovered.artifact_refs) == 3
    assert len(tuple(store.commits.glob("*.json"))) == 3
    assert evaluations == 2


@pytest.mark.parametrize(
    "stale_mode",
    ("asset_bytes_corrupted", "qualification_deleted", "manifest_replaced"),
)
def test_formal_g3_restore_replays_current_assets_and_blocks_stale_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_mode: str,
) -> None:
    runner = _prepare_formal_runner(monkeypatch, tmp_path)
    request = _request(f"runs/formal-g3-stale-{stale_mode}")
    calls = 0

    def evaluate(*args: object, **kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _resolution()
        if stale_mode == "asset_bytes_corrupted":
            return _resolution(blocked=True)
        if stale_mode == "qualification_deleted":
            raise FileNotFoundError("qualification deleted")
        replaced = _resolution()
        replaced["entries"][0]["asset_id"] = _digest("replaced-manifest")
        replaced["artifact_hash"] = g3_resolution_artifact_hash(replaced)
        return replaced

    monkeypatch.setattr(stage01, "evaluate_stage0_g3", evaluate)
    first = runner.run(request)
    assert first.status.value == "PASS"

    with pytest.raises(
        (FileNotFoundError, ValueError),
        match=(
            "qualification deleted|STAGE0_G3_AGGREGATE_BLOCKED|"
            "STAGE0_G3_RESTORE_PAYLOAD_DRIFT"
        ),
    ):
        runner.run(request)
    assert calls == 2


@pytest.mark.parametrize("legacy_mode", ("legacy", "formal_projection"))
def test_formal_g3_restore_rejects_legacy_and_old_projection_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_mode: str,
) -> None:
    runner = _prepare_formal_runner(monkeypatch, tmp_path)
    output = f"runs/formal-g3-restore-{legacy_mode}"
    request = _request(output)
    store = TaskArtifactStore(tmp_path, output)
    monkeypatch.setattr(stage01, "evaluate_stage0_g3", lambda *args, **kwargs: _resolution())
    for kind in request.task.artifact_kinds:
        payload: dict[str, Any]
        if legacy_mode == "formal_projection":
            payload = {
                "schema_version": "stage01-task-evidence-v1",
                "core_evidence": {
                    "execution_mode": "formal_evidence_projection",
                },
            }
        else:
            payload = {
                "schema_version": "stage0-g3-legacy-evidence-v0",
                "artifact_role": kind,
            }
        store.publish(
            task_id=TASK_ID,
            artifact_kind=kind,
            config_hash=request.config.config_hash,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
        )

    with pytest.raises(ValueError, match="STAGE0_G3_RESTORE_SCHEMA_INVALID"):
        runner.run(request)


def test_formal_g3_workspace_may_not_be_the_source_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(ROOT))

    with pytest.raises(ValueError, match="STAGE0_G3_WORKSPACE_OVERLAPS_SOURCE_ROOT"):
        stage01._formal_g3_roots(ROOT)


def test_formal_g3_checked_at_is_unique_and_bound_to_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        {
            "manifest_ref": f"manifests/{index}.json",
            "qualification_ref": f"qualifications/{index}.json",
        }
        for index in range(13)
    ]
    for entry in entries:
        for field in ("manifest_ref", "qualification_ref"):
            path = tmp_path.joinpath(*entry[field].split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}\n")
    monkeypatch.setattr(
        stage01,
        "load_stage0_asset_requirements",
        lambda path: {"artifact_hash": _digest("requirements")},
    )
    monkeypatch.setattr(
        stage01,
        "load_stage0_asset_layout",
        lambda path, requirements: {"entries": entries},
    )
    manifest_commit = TEST_HEAD
    qualification_times = {
        str(index): "2026-08-03T09:00:00Z" for index in range(13)
    }
    monkeypatch.setattr(
        stage01,
        "load_asset_manifest",
        lambda path: {"generator_git_commit": manifest_commit},
    )
    monkeypatch.setattr(
        stage01,
        "load_canonical_json",
        lambda path: {
            "generator_git_commit": TEST_HEAD,
            "checked_at": qualification_times[path.stem],
        },
    )
    monkeypatch.setattr(stage01, "validate_g3_qualification", lambda value: None)

    assert stage01._stable_g3_checked_at(
        _source_binding(),
        tmp_path.resolve(),
    ) == "2026-08-03T09:00:00Z"
    qualification_times["12"] = "2026-08-03T09:00:01Z"
    with pytest.raises(
        ValueError,
        match="STAGE0_G3_QUALIFICATION_CHECKED_AT_NOT_UNIQUE",
    ):
        stage01._stable_g3_checked_at(_source_binding(), tmp_path.resolve())


def test_formal_g3_checked_at_rejects_manifest_qualification_commit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifests/one.json"
    qualification = tmp_path / "qualifications/one.json"
    manifest.parent.mkdir(parents=True)
    qualification.parent.mkdir(parents=True)
    manifest.write_bytes(b"{}\n")
    qualification.write_bytes(b"{}\n")
    monkeypatch.setattr(stage01, "load_stage0_asset_requirements", lambda path: {})
    monkeypatch.setattr(
        stage01,
        "load_stage0_asset_layout",
        lambda path, requirements: {
            "entries": [
                {
                    "manifest_ref": "manifests/one.json",
                    "qualification_ref": "qualifications/one.json",
                }
            ]
        },
    )
    monkeypatch.setattr(
        stage01,
        "load_asset_manifest",
        lambda path: {"generator_git_commit": "b" * 40},
    )
    monkeypatch.setattr(
        stage01,
        "load_canonical_json",
        lambda path: {
            "generator_git_commit": TEST_HEAD,
            "checked_at": "2026-08-03T09:00:00Z",
        },
    )
    monkeypatch.setattr(stage01, "validate_g3_qualification", lambda value: None)

    with pytest.raises(
        ValueError,
        match="STAGE0_G3_ASSET_GENERATOR_COMMIT_AMBIGUOUS",
    ):
        stage01._stable_g3_checked_at(_source_binding(), tmp_path.resolve())


def test_stage0_asset_catalog_requires_tokenizer_and_real_g3_outcomes() -> None:
    task = DEFAULT_TASK_CATALOG.get(TASK_ID)

    assert "tokenizer_assets" in task.formal_eligibility.required_capabilities
    assert {
        "all_thirteen_assets_exactly_resolved",
        "all_five_g3_subgates_pass",
    } <= set(task.completion_rules)
    assert {
        "g3_blocked_or_exception_fails_task",
        "legacy_formal_projection_rejected",
    } <= set(task.failure_rules)
