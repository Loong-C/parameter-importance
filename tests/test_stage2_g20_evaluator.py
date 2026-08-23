"""Formal, independent G2.0 evaluator contract tests.

These fixtures use the real TaskArtifactStore for every S2.1 and Stage1.11
input.  There is no ordinary ``upstream.json`` stand-in: the evaluator must
reload four formal Stage1.11 commits and the persisted S2.1 ResolvedConfigV2.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from param_importance_nlp.contracts import SeedPlan
from param_importance_nlp.contracts.config_v2 import load_resolved_config_compatible
from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.experiments.preregistration import (
    build_stage2_hypothesis_contract,
    build_stage2_preregistration,
)
from param_importance_nlp.experiments.stage2_g20_evaluator import (
    ARTIFACT_KINDS,
    EVALUATOR_SOURCE_PATH,
    MATHEMATICS_PATH,
    STAGE1_ARTIFACT_KINDS,
    STAGE1_REPORT_PATH,
    STAGE1_TASK_ID,
    TASK_ID,
    evaluate_formal_g20,
)
from param_importance_nlp.runtime.task_artifacts import (
    TaskArtifactStore,
    load_committed_task_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head(repository: Path = ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _stage1_binding(data_root: Path, refs: dict[str, str]) -> str:
    artifacts = {
        kind: load_committed_task_artifact(data_root, refs[kind], require_formal=True)
        for kind in STAGE1_ARTIFACT_KINDS
    }
    return canonical_json_hash(
        {
            "predecessor_task_ids": [STAGE1_TASK_ID],
            "artifacts": [
                {
                    "task_id": artifacts[kind].identity.task_id,
                    "artifact_kind": artifacts[kind].identity.artifact_kind,
                    "artifact_hash": artifacts[kind].identity.artifact_hash,
                    "config_hash": artifacts[kind].identity.config_hash,
                    "run_intent": artifacts[kind].run_intent,
                    "formal_eligible": artifacts[kind].identity.formal_eligible,
                    "commit_ref": refs[kind],
                }
                for kind in STAGE1_ARTIFACT_KINDS
            ],
            "auxiliary_refs": [],
        }
    )


def _formal_stage1_sources(data_root: Path) -> dict[str, str]:
    store = TaskArtifactStore(data_root, "runs/stage1-11-formal")
    config_hash = canonical_json_hash({"fixture": "formal-stage1-11", "version": 1})
    core = {"evidence_type": "formal_stage1_11_exit", "observation": "PASS"}
    core_hash = canonical_json_hash(core)
    task_definition_hash = canonical_json_hash({"task": STAGE1_TASK_ID, "contract": "formal"})
    refs: dict[str, str] = {}
    for kind in STAGE1_ARTIFACT_KINDS:
        payload: dict[str, object] = {
            "schema_version": {
                "stage_report": "stage1-s1-11-stage-report-v1",
                "requirements_matrix": "stage1-s1-11-requirements-matrix-v1",
                "gate_summary": "stage1-s1-11-gate-summary-v1",
                "delivery_manifest": "stage1-s1-11-delivery-manifest-v1",
            }[kind],
            "status": "PASS",
            "task_id": STAGE1_TASK_ID,
            "gate_id": "G1-EXIT",
            "scope": "formal",
            "config_hash": config_hash,
            "core_evidence": core,
            "core_evidence_hash": core_hash,
            "task_definition_hash": task_definition_hash,
            "artifact_role": kind,
        }
        if kind == "gate_summary":
            payload["unresolved_failure_count"] = 0
        if kind == "requirements_matrix":
            payload["rows"] = [
                {"requirement_id": f"S1.11-R{index:02d}", "status": "PASS"}
                for index in range(1, 29)
            ]
        refs[kind] = store.publish(
            task_id=STAGE1_TASK_ID,
            artifact_kind=kind,
            config_hash=config_hash,
            run_intent="formal",
            formal_eligible=True,
            source_refs=(),
            payload=payload,  # type: ignore[arg-type]
        ).commit_ref
    return refs


def _formal_config(
    data_root: Path,
    *,
    output_dir: str = "runs/stage2-01",
    name: str = "stage2-s21-resolved-config.json",
    master_seed: int = 1337,
):
    value = load_canonical_json(BASE_CONFIG)
    assert isinstance(value, dict)
    value = copy.deepcopy(value)
    identity = value["identity"]
    assert isinstance(identity, dict)
    identity.update(
        {
            "formal_eligible": True,
            "run_intent": "formal",
            "route": "formal",
            "task": TASK_ID,
            "stage": 2,
            "master_seed": master_seed,
        }
    )
    runtime = value["runtime"]
    assert isinstance(runtime, dict)
    runtime["allow_dirty_worktree"] = False
    config = load_resolved_config_compatible(
        value,
        task_id=TASK_ID,
        overrides={"artifacts": {"output_dir": output_dir}},
    )
    ref = f"configs/{name}"
    write_canonical_json(data_root / ref, config.to_dict())
    return ref, config


def _publish_s21(
    data_root: Path,
    *,
    stage1_refs: dict[str, str],
    config,
    candidate_status: str = "NOT_RUN",
    source_refs: tuple[str, ...] | None = None,
    producer_commit: str | None = None,
) -> dict[str, str]:
    source_refs = source_refs or tuple(stage1_refs[kind] for kind in STAGE1_ARTIFACT_KINDS)
    binding = _stage1_binding(data_root, stage1_refs)
    prereg = build_stage2_preregistration(
        seed_plan_hash=SeedPlan.from_master_seed(1337).artifact_hash,
        producer_commit=producer_commit or _head(),
        mathematics_hash=_sha(ROOT / MATHEMATICS_PATH),
        stage1_report_hash=_sha(ROOT / STAGE1_REPORT_PATH),
        upstream_binding_hash=binding,
        scope="formal",
    )
    hypothesis = build_stage2_hypothesis_contract(prereg, upstream_binding_hash=binding)
    candidate = {
        "schema_version": "stage23-task-gate-candidate-v1",
        "task_id": TASK_ID,
        "gate_ids": ["stage1.G1-EXIT"],
        "gate_status": candidate_status,
        "local_validation_status": "NOT_RUN",
        "formal_eligible": False,
        "reason": "formal_gate_requires_independent_review",
        "gate_id": "stage2.G2.0",
        "preregistration_hash": prereg["preregistration_hash"],
        "hypothesis_contract_hash": hypothesis["hypothesis_contract_hash"],
        "quality_gate_status": "NOT_RUN",
        "sample_generation_status": "FORBIDDEN_UNTIL_COMMITTED",
    }
    store = TaskArtifactStore(data_root, "runs/stage2-01")
    refs: dict[str, str] = {}
    for kind, payload in (
        ("preregistration", prereg),
        ("hypothesis_contract", hypothesis),
        ("gate_record", candidate),
    ):
        refs[kind] = store.publish(
            task_id=TASK_ID,
            artifact_kind=kind,
            config_hash=config.config_hash,
            run_intent="formal",
            formal_eligible=True,
            source_refs=source_refs,
            payload=payload,
        ).commit_ref
    return refs


def _fixture(
    tmp_path: Path,
    *,
    candidate_status: str = "NOT_RUN",
    source_refs: tuple[str, ...] | None = None,
    producer_commit: str | None = None,
):
    data_root = tmp_path / "data-root"
    data_root.mkdir(parents=True)
    stage1_refs = _formal_stage1_sources(data_root)
    config_ref, config = _formal_config(data_root)
    refs = _publish_s21(
        data_root,
        stage1_refs=stage1_refs,
        config=config,
        candidate_status=candidate_status,
        source_refs=source_refs,
        producer_commit=producer_commit,
    )
    return data_root, refs, config_ref, stage1_refs


def _evaluate(data_root: Path, refs: dict[str, str], config_ref: str, **kwargs: object):
    return evaluate_formal_g20(
        data_root,
        refs,
        repository_root=ROOT,
        resolved_config_ref=config_ref,
        output_dir="runs/evaluations",
        **kwargs,
    )


def test_real_three_commit_formal_fixture_publishes_gate_record(tmp_path: Path) -> None:
    data_root, refs, config_ref, stage1_refs = _fixture(tmp_path)
    result = _evaluate(data_root, refs, config_ref)
    assert result["status"] == "PASS"
    assert result["formal_eligible"] is True
    assert isinstance(result["commit_ref"], str)
    loaded = load_committed_task_artifact(data_root, result["commit_ref"], require_formal=True)  # type: ignore[arg-type]
    gate = GateRecord.from_mapping(dict(loaded.payload))
    assert gate.status is GateStatus.PASS
    assert loaded.source_refs == tuple(refs[kind] for kind in ARTIFACT_KINDS)
    assert [item["commit_ref"] for item in gate.measured["stage1_source_artifacts"]] == [  # type: ignore[index]
        stage1_refs[kind] for kind in STAGE1_ARTIFACT_KINDS
    ]
    evaluator = gate.measured["evaluator"]  # type: ignore[index]
    assert evaluator["producer_commit"] == _head()  # type: ignore[index]
    assert evaluator["source_hashes"][EVALUATOR_SOURCE_PATH] == _sha(ROOT / EVALUATOR_SOURCE_PATH)  # type: ignore[index]
    assert result["evaluation_config_hash"] == evaluator["evaluation_config_hash"]  # type: ignore[index]


def test_same_input_reuses_identical_gate_without_checked_at_block(tmp_path: Path) -> None:
    data_root, refs, config_ref, _ = _fixture(tmp_path)
    first = _evaluate(data_root, refs, config_ref)
    second = _evaluate(data_root, refs, config_ref)
    assert first["status"] == second["status"] == "PASS"
    assert first["commit_ref"] == second["commit_ref"]
    assert first["envelope_artifact_hash"] == second["envelope_artifact_hash"]
    assert first["gate_record"]["checked_at"] == second["gate_record"]["checked_at"]  # type: ignore[index]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda refs: (refs["preregistration"], refs["preregistration"], refs["gate_record"]),
        lambda refs: (refs["preregistration"], refs["hypothesis_contract"], "wrong/path.json"),
    ],
)
def test_missing_duplicate_and_wrong_refs_fail_closed(tmp_path: Path, mutator) -> None:
    data_root, refs, config_ref, _ = _fixture(tmp_path)
    result = _evaluate(data_root, mutator(refs), config_ref)
    assert result["status"] == "BLOCKED"
    assert result["formal_eligible"] is False
    assert result["commit_ref"] is None


def test_runner_candidate_pass_is_not_a_numeric_gate(tmp_path: Path) -> None:
    data_root, refs, config_ref, _ = _fixture(tmp_path, candidate_status="PASS")
    result = _evaluate(data_root, refs, config_ref)
    assert result["status"] == "FAIL"
    assert result["formal_eligible"] is False
    assert "SELF_SIGNED" in " ".join(result["gate_record"]["reasons"])  # type: ignore[index]
    assert isinstance(result["commit_ref"], str)


def test_source_refs_must_be_four_real_formal_task_commits(tmp_path: Path) -> None:
    data_root, refs, config_ref, _ = _fixture(
        tmp_path,
        source_refs=("ordinary/a.json", "ordinary/b.json", "ordinary/c.json", "ordinary/d.json"),
    )
    for name in ("a", "b", "c", "d"):
        write_canonical_json(data_root / f"ordinary/{name}.json", {"schema_version": "ordinary-v1"})
    result = _evaluate(data_root, refs, config_ref)
    assert result["status"] == "BLOCKED"
    assert result["commit_ref"] is None


def test_wrong_resolved_config_and_tampered_payload_are_blocked(tmp_path: Path) -> None:
    data_root, refs, config_ref, _ = _fixture(tmp_path)
    wrong_ref, wrong_config = _formal_config(
        data_root,
        output_dir="runs/other",
        name="wrong-resolved-config.json",
        master_seed=7331,
    )
    write_canonical_json(data_root / wrong_ref, wrong_config.to_dict())
    wrong = _evaluate(data_root, refs, wrong_ref)
    assert wrong["status"] == "BLOCKED"

    commit = TaskArtifactStore(data_root, "runs/stage2-01").load_commit(refs["hypothesis_contract"])
    object_path = data_root / commit.object_ref
    value = load_canonical_json(object_path)
    assert isinstance(value, dict)
    value["payload"]["hypotheses"][0]["claim"] = "tampered"  # type: ignore[index]
    object_path.write_text(__import__("json").dumps(value), encoding="utf-8")
    tampered = _evaluate(data_root, refs, config_ref)
    assert tampered["status"] == "BLOCKED"
    assert tampered["commit_ref"] is None


def test_fake_producer_commit_is_rejected(tmp_path: Path) -> None:
    data_root, refs, config_ref, _ = _fixture(tmp_path, producer_commit="f" * 40)
    result = _evaluate(data_root, refs, config_ref)
    assert result["status"] == "BLOCKED"
    assert "TRUSTED_HEAD" in " ".join(result["gate_record"]["reasons"])  # type: ignore[index]


def test_dual_root_and_output_root_symlink_fail_closed(tmp_path: Path) -> None:
    data_root, refs, config_ref, _ = _fixture(tmp_path)
    assert not (data_root / "docs").exists()
    result = _evaluate(data_root, refs, config_ref)
    assert result["status"] == "PASS"

    symlink_target = tmp_path / "outside-output"
    symlink_target.mkdir()
    symlink_path = data_root / "runs" / "symlink-output"
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(symlink_target, symlink_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    blocked = evaluate_formal_g20(
        data_root,
        refs,
        repository_root=ROOT,
        resolved_config_ref=config_ref,
        output_dir="runs/symlink-output",
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["commit_ref"] is None


def test_repository_dirty_and_source_drift_fail_closed(tmp_path: Path) -> None:
    clone = tmp_path / "repository-clone"
    subprocess.run(["git", "worktree", "add", "--detach", str(clone), _head()], check=True, capture_output=True)
    data_root, refs, config_ref, _ = _fixture(tmp_path / "fixture")
    try:
        (clone / "src/param_importance_nlp/experiments/stage23_task_runners.py").write_text("\n# drift\n", encoding="utf-8")
        result = evaluate_formal_g20(
            data_root,
            refs,
            repository_root=clone,
            resolved_config_ref=config_ref,
            output_dir="runs/evaluations",
        )
        assert result["status"] == "BLOCKED"
        assert "WORKTREE_DRIFT" in " ".join(result["gate_record"]["reasons"])  # type: ignore[index]
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(clone)], check=False, capture_output=True)


def test_input_object_symlink_is_rejected(tmp_path: Path) -> None:
    data_root, refs, config_ref, _ = _fixture(tmp_path)
    commit = TaskArtifactStore(data_root, "runs/stage2-01").load_commit(refs["preregistration"])
    object_path = data_root / commit.object_ref
    backup = tmp_path / "object-backup.json"
    shutil.copy2(object_path, backup)
    object_path.unlink()
    try:
        os.symlink(backup, object_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    result = _evaluate(data_root, refs, config_ref)
    assert result["status"] == "BLOCKED"
    assert result["commit_ref"] is None
