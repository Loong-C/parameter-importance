"""Stage 0 S0.12 delivery, synchronization, and readiness contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from param_importance_nlp.atomic import sha256_file
from param_importance_nlp.cli import _load_mapping
from param_importance_nlp.contracts import (
    GateRecord,
    GateStatus,
    ResolvedConfig,
    canonical_json_hash,
    write_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRuntimeEnvironment,
)
from param_importance_nlp.stage0_bootstrap import Stage0SourceBinding
from param_importance_nlp.stage0_g9 import Stage0G9FormalState
from param_importance_nlp.stage0_g10 import (
    G10SourceBinding,
    _REQUIRED_GATES,
    _asset_inventory,
    _find_gate,
    _repository_inventory,
    build_stage0_g10_config,
    run_formal_g10_task,
    validate_formal_g10_outputs,
    validate_sync_observation,
)
from param_importance_nlp.stage0_g10_sync import (
    AGENT_FILES,
    REMOTE_URL,
    SERVER_DATA_ROOT,
    SERVER_HOST,
    SERVER_REPOSITORY,
    SYNC_COLLECTOR_VERSION,
    SYNC_OBSERVATION_SCHEMA,
    Stage0G10SyncError,
    _parse_remote_head,
)


ROOT = Path(__file__).resolve().parents[1]


def _git_head(repository: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_branch(repository: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _observation(commit: str, branch: str) -> dict[str, object]:
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in AGENT_FILES}
    value: dict[str, object] = {
        "schema_version": SYNC_OBSERVATION_SCHEMA,
        "collector_version": SYNC_COLLECTOR_VERSION,
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authorization_ref": "fixture-explicit-authorization",
        "branch": branch,
        "expected_commit": commit,
        "previous_github_head": commit,
        "previous_server_head": commit,
        "fast_forward_ancestry_verified": True,
        "force_push_used": False,
        "local": {"repository": "D:/fixture", "head": commit, "branch": branch, "worktree_clean": True},
        "github": {
            "remote": "origin",
            "remote_url": REMOTE_URL,
            "branch_ref": f"refs/heads/{branch}",
            "head": commit,
            "push_verified": True,
        },
        "server": {
            "host_alias": SERVER_HOST,
            "repository": SERVER_REPOSITORY,
            "data_root": SERVER_DATA_ROOT,
            "head": commit,
            "branch": branch,
            "worktree_clean": True,
            "fast_forward_verified": True,
        },
        "agent_sync": {
            "file_count_each_side": 5,
            "files": list(AGENT_FILES),
            "local_sha256": hashes,
            "server_sha256": hashes,
            "all_equal": True,
        },
        "bundle_cleanup": {
            "bundle_name": f"stage0-g10-sync-{commit[:12]}.bundle",
            "local_path": f"D:/fixture/.stage0-g10-sync-{commit[:12]}.bundle",
            "server_path": f"{SERVER_DATA_ROOT}/tmp/stage0-g10-sync-{commit[:12]}.bundle",
            "local_absent": True,
            "server_absent": True,
        },
        "preserved_user_content": {
            "path": "docs/mathematics.md",
            "tracked": True,
            "sha256": sha256_file(ROOT / "docs/mathematics.md"),
        },
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _template() -> ResolvedConfigV2:
    base = ResolvedConfig.resolve(
        _load_mapping(ROOT / "configs/local-fixtures/resolved-config-v1.json"),
        _load_mapping(ROOT / "configs/run-ready/layers/formal-stage1-pythia14m.yaml"),
    )
    return ResolvedConfigV2.resolve(
        base,
        task_id="stage1.07_single_gpu_pythia14m",
        overrides=_load_mapping(ROOT / "configs/run-ready/v2/stage1-pythia14m-formal.yaml"),
    )


def test_g10_sync_observation_is_hash_bound_and_current(tmp_path: Path) -> None:
    commit = _git_head()
    branch = _git_branch()
    value = _observation(commit, branch)
    reference = "observations/g10.json"
    write_canonical_json(tmp_path / reference, value)
    source = G10SourceBinding(ROOT, commit, branch, {})
    loaded = validate_sync_observation(tmp_path, reference, source)
    assert loaded["artifact_hash"] == value["artifact_hash"]
    assert loaded["agent_sync"]["all_equal"] is True
    tampered = dict(value)
    tampered["force_push_used"] = True
    tampered["artifact_hash"] = canonical_json_hash(
        {key: item for key, item in tampered.items() if key != "artifact_hash"}
    )
    write_canonical_json(tmp_path / "observations/tampered.json", tampered)
    with pytest.raises(Exception, match="G10_SYNC_OBSERVATION"):
        validate_sync_observation(tmp_path, "observations/tampered.json", source)


def test_g10_sync_remote_head_parser_is_exact() -> None:
    commit = "a" * 40
    assert _parse_remote_head(
        f"{commit}\trefs/heads/feat/stage0-completion\n",
        branch="feat/stage0-completion",
    ) == commit
    with pytest.raises(Stage0G10SyncError, match="GITHUB_BRANCH_RESULT_INVALID"):
        _parse_remote_head(
            f"{commit}\trefs/heads/main\n",
            branch="feat/stage0-completion",
        )


def test_repository_inventory_rejects_runtime_artifacts_and_accepts_clean_fixture(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    required = {
        "environment/requirements.lock": "fixture-lock\n",
        "pyproject.toml": "[project]\nname='fixture'\nversion='0.0.0'\n",
        "docs/mathematics.md": "# Math\n",
        "docs/stage0-delivery-runbook.md": "# Delivery\n",
        "docs/stage0-replay-runbook.md": "# Replay\n",
        "docs/stage1-handoff.md": "# Handoff\n",
        "worklogs/2026-08-03-stage0-remaining-tasks.md": "# Worklog\n",
    }
    for reference, content in required.items():
        path = repository / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(repository), "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "fixture",
        ],
        check=True,
    )
    commit = _git_head(repository)
    source = G10SourceBinding(repository, commit, "master", {})
    inventory = _repository_inventory(source, {"previous_github_head": commit})
    assert inventory["status"] == "PASS"
    assert inventory["tracked_file_count"] == len(required)
    assert inventory["high_confidence_secret_hits"] == []


def test_asset_inventory_rechecks_all_thirteen_manifest_file_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_root = tmp_path / "assets/model"
    asset_root.mkdir(parents=True)
    payload_path = asset_root / "weights.fixture"
    payload_path.write_bytes(b"stage0-g10-fixture")
    file_sha = sha256_file(payload_path)
    manifest: dict[str, object] = {
        "state": "ready",
        "asset_id": "a" * 64,
        "revision": "fixture-v1",
        "files": [
            {"path": "weights.fixture", "size_bytes": payload_path.stat().st_size, "sha256": file_sha}
        ],
    }
    manifest_ref = "manifests/fixture.json"
    write_canonical_json(tmp_path / manifest_ref, manifest)
    entries = [
        {
            "logical_name": f"fixture-{index:02d}",
            "kind": "model" if index < 4 else "tokenizer" if index == 4 else "dataset",
            "status": "PASS",
            "asset_id": "a" * 64,
            "manifest_ref": manifest_ref,
            "asset_root_ref": "assets/model",
            "ready_manifest_sha256": canonical_json_hash(manifest),
        }
        for index in range(13)
    ]
    monkeypatch.setattr(
        "param_importance_nlp.stage0_g10.load_committed_task_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(payload={"entries": entries}),
    )
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(),
        frozen_contract_stages=frozenset({0}),
        passed_gate_ids=frozenset(),
        evidence_refs={"g3_resolution": "commits/g3.json"},
    )
    inventory = _asset_inventory(environment, tmp_path, "b" * 40)
    assert inventory["asset_count"] == 13
    assert inventory["declared_size_bytes"] == 13 * payload_path.stat().st_size
    assert all(row["acceptance_status"] == "PASS" for row in inventory["rows"])


def test_recursive_gate_extraction_rejects_no_embedded_gate() -> None:
    gate = GateRecord(
        gate_id="stage0.G9",
        stage=0,
        status=GateStatus.PASS,
        checked_at="2026-08-03T00:00:00Z",
        measured={"layers": 6},
        threshold={"required": "PASS"},
        evidence_refs=("evidence/g9.json",),
    )
    found = _find_gate({"canonical": {"gate_record": gate.to_dict()}}, "stage0.G9")
    assert found == gate
    assert _find_gate({"canonical": {}}, "stage0.G9") is None


def test_g10_config_binds_g9_and_sync_observation() -> None:
    template = _template()
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"git", "server", "github"}),
        frozen_contract_stages=frozenset({0}),
        passed_gate_ids=frozenset({"stage0.G9"}),
        evidence_refs={},
    )
    state = Stage0G9FormalState(
        environment=environment,
        task_output_refs={
            "test_report": "evidence/g9/test.json",
            "replay_report": "evidence/g9/replay.json",
            "gate_summary": "evidence/g9/gate.json",
        },
        config=template,
        config_ref="evidence/g9/config.json",
        environment_ref="evidence/g9/environment.json",
        index_ref="evidence/g9/index.json",
        index_sha256="1" * 64,
        gate_artifact_hash="2" * 64,
        g8_index_ref="evidence/g8/index.json",
    )
    config = build_stage0_g10_config(
        binding=Stage0SourceBinding(ROOT, "a" * 40, "feat/stage0-completion", True),
        state=state,
        sync_observation_ref="observations/g10.json",
    )
    assert config.task_id == "stage0.12_delivery_and_sync"
    assert config.section("orchestration")["matrix_ref"] == "observations/g10.json"
    assert config.section("recovery")["mode"] == "manual_external"
    assert set(config.task_definition.formal_eligibility.required_gate_ids) == {"stage0.G9"}


def test_g10_formal_task_publishes_revalidatable_ready_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = _git_head()
    branch = _git_branch()
    observation_ref = "observations/g10.json"
    write_canonical_json(tmp_path / observation_ref, _observation(commit, branch))
    upstream_ref = "evidence/upstream.json"
    write_canonical_json(tmp_path / upstream_ref, {"status": "PASS"})
    g9_output_refs = {
        "test_report": "evidence/g9-test.json",
        "replay_report": "evidence/g9-replay.json",
        "gate_summary": "evidence/g9-gate.json",
    }
    for reference in g9_output_refs.values():
        write_canonical_json(tmp_path / reference, {"status": "PASS", "ref": reference})
    gate_records = {
        gate_id: GateRecord(
            gate_id=gate_id,
            stage=0,
            status=GateStatus.PASS,
            checked_at="2026-08-03T00:00:00Z",
            measured={"fixture": True},
            threshold={"required": "PASS"},
            evidence_refs=(upstream_ref,),
        )
        for gate_id in _REQUIRED_GATES
    }
    gate_refs = {gate_id: upstream_ref for gate_id in _REQUIRED_GATES}
    repository_inventory = {
        "status": "PASS",
        "git_commit": commit,
        "tracked_file_count": 1,
        "tracked_bytes": 1,
        "max_allowed_file_bytes": 10 * 1024 * 1024,
        "category_counts": {"implementation": 1},
        "forbidden_paths": [],
        "oversized_paths": [],
        "high_confidence_secret_hits": [],
        "checked_markdown_links": [],
        "whitespace_check_range": f"{commit}..{commit}",
        "agent_directory_tracked": False,
        "rows": [],
    }
    asset_rows = [
        {
            "logical_name": f"asset-{index:02d}",
            "kind": "model" if index < 4 else "tokenizer" if index == 4 else "dataset",
            "asset_id": hashlib.sha256(f"asset-{index}".encode()).hexdigest(),
            "revision": "fixture-v1",
            "manifest_ref": f"manifests/asset-{index}.json",
        }
        for index in range(13)
    ]
    assets = {
        "status": "PASS",
        "authoritative_runtime_root": str(tmp_path),
        "authoritative_runtime_copy_count": 1,
        "authoritative_runtime_copy_is_backup": False,
        "asset_count": 13,
        "declared_size_bytes": 13,
        "observed_size_bytes": 13,
        "rows": asset_rows,
    }
    evidence_inventory = {"status": "PASS", "evidence_count": 1, "total_bytes": 1, "rows": []}
    persistence = {
        "status": "PASS",
        "satisfaction": "TIME_BOUNDED_RISK_ACCEPTANCE",
        "scope": ["fixture"],
        "excluded": ["formal"],
        "expires_at": "2026-08-18T23:59:00+08:00",
        "approval_source": "fixture",
        "independent_failure_domain_copy_count": 0,
        "authoritative_runtime_copy_is_backup": False,
        "stage4_requirement": "fixture",
        "decision_ref": "reports/stage0/g1-persistence-decision-20260719.json",
        "decision_sha256": "1" * 64,
    }
    source = G10SourceBinding(ROOT, commit, branch, {})
    monkeypatch.setattr("param_importance_nlp.stage0_g10._capture_source", lambda: source)
    monkeypatch.setattr(
        "param_importance_nlp.stage0_g10._validated_gate_records",
        lambda *_args, **_kwargs: (gate_records, gate_refs),
    )
    monkeypatch.setattr(
        "param_importance_nlp.stage0_g10._repository_inventory",
        lambda *_args, **_kwargs: repository_inventory,
    )
    monkeypatch.setattr(
        "param_importance_nlp.stage0_g10._asset_inventory",
        lambda *_args, **_kwargs: assets,
    )
    monkeypatch.setattr(
        "param_importance_nlp.stage0_g10._evidence_inventory",
        lambda *_args, **_kwargs: evidence_inventory,
    )
    monkeypatch.setattr(
        "param_importance_nlp.stage0_g10._persistence_status",
        lambda *_args, **_kwargs: persistence,
    )
    template = _template()
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"git", "github", "server"}),
        frozen_contract_stages=frozenset({0}),
        passed_gate_ids=_REQUIRED_GATES,
        evidence_refs={},
    )
    state = Stage0G9FormalState(
        environment=environment,
        task_output_refs=g9_output_refs,
        config=template,
        config_ref="evidence/g9/config.json",
        environment_ref="evidence/g9/environment.json",
        index_ref="evidence/g9/index.json",
        index_sha256="2" * 64,
        gate_artifact_hash="3" * 64,
        g8_index_ref="evidence/g8/index.json",
    )
    config = build_stage0_g10_config(
        binding=Stage0SourceBinding(ROOT, commit, branch, True),
        state=state,
        sync_observation_ref=observation_ref,
    )
    request = TaskExecutionRequest(config, config.task_definition, environment)
    result = run_formal_g10_task(
        request,
        tmp_path,
        TaskArtifactStore(tmp_path, "task-store"),
        source_refs=(upstream_ref,),
    )
    assert result.status.value == "PASS"
    gate = validate_formal_g10_outputs(request, tmp_path, result.artifact_refs)
    assert gate.gate_id == "stage0.G10"
    assert gate.status is GateStatus.PASS
    readiness_refs = list(tmp_path.glob("evidence/stage0/readiness/*/*/READY.json"))
    assert len(readiness_refs) == 1
    assert json.loads(readiness_refs[0].read_text(encoding="utf-8"))["status"] == "READY"
    generated = {
        "stage0-g10-delivery-manifest-v1.json": next(
            tmp_path.glob("evidence/stage0/g10-final/*/*/delivery-manifest.json")
        ),
        "stage0-g10-sync-report-v1.json": next(
            tmp_path.glob("evidence/stage0/g10-final/*/*/sync-report.json")
        ),
        "stage0-g10-worklog-v1.json": next(
            tmp_path.glob("evidence/stage0/g10-final/*/*/worklog.json")
        ),
        "stage0-g10-stage1-handoff-v1.json": next(
            tmp_path.glob("evidence/stage0/g10-final/*/*/stage1-handoff.json")
        ),
        "stage0-g10-readiness-v1.json": readiness_refs[0],
    }
    for schema_name, artifact_path in generated.items():
        validator = Draft202012Validator(
            json.loads((ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
        )
        validator.validate(json.loads(artifact_path.read_text(encoding="utf-8")))


def test_all_g10_schemas_are_valid_and_sync_fixture_conforms() -> None:
    schema_paths = sorted((ROOT / "schemas").glob("stage0-g10-*.json"))
    assert len(schema_paths) == 8
    for path in schema_paths:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))
    schema = json.loads(
        (ROOT / "schemas/stage0-g10-sync-observation-v1.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(_observation(_git_head(), _git_branch()))
