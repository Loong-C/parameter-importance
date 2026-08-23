from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage2_g21_adapter import (
    S110_ARTIFACT_KINDS,
    S111_ARTIFACT_KINDS,
    STAGE1_ARTIFACT_KINDS,
    _load_stage1_group,
)
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore


def _stage1_fixture(root: Path) -> tuple[dict[str, str], dict[str, object], dict[str, object]]:
    output_110 = "evidence/stage1/tasks/10-fixture"
    output_111 = "evidence/stage1/tasks/11-fixture"
    store_110 = TaskArtifactStore(root, output_110)
    store_111 = TaskArtifactStore(root, output_111)
    role_110 = {
        "artifact_manifest": {"schema_version": "s1-10-fixture-v1", "status": "PASS", "role": "artifact_manifest"},
        "resume_report": {"schema_version": "s1-10-compat-fixture-v1", "status": "PASS", "role": "resume_report"},
        "gate_record": {"schema_version": "s1-10-gate-fixture-v1", "status": "PASS", "role": "gate_record"},
    }
    role_111 = {
        kind: {"schema_version": f"s1-11-{kind}-fixture-v1", "status": "PASS", "role": kind}
        for kind in S111_ARTIFACT_KINDS
    }
    s110_refs = {
        kind: f"{output_110}/commits/{kind}.json"
        for kind in S110_ARTIFACT_KINDS
    }
    s111_refs = {
        kind: f"{output_111}/commits/{kind}.json"
        for kind in S111_ARTIFACT_KINDS
    }
    s110_sources = ["evidence/stage1/s1-10-authority/index.json", f"{output_110}/producer-config.json", *s110_refs.values()]
    s111_sources = ["evidence/stage1/s1-11-authority/index.json", *s110_sources, f"{output_111}/producer-config.json", *s111_refs.values()]
    refs: dict[str, str] = {}
    for kind in S110_ARTIFACT_KINDS:
        refs[kind] = store_110.publish(
            task_id="stage1.10_checkpoint_resume_and_artifacts",
            artifact_kind=kind,
            config_hash="1" * 64,
            run_intent="formal",
            formal_eligible=True,
            payload=role_110[
                {"training_state_manifest": "artifact_manifest", "resume_equivalence_report": "resume_report", "gate_record": "gate_record"}[kind]
            ],
            source_refs=tuple(s110_sources),
        ).commit_ref
    for kind in S111_ARTIFACT_KINDS:
        refs[kind] = store_111.publish(
            task_id="stage1.11_reporting_and_exit_gate",
            artifact_kind=kind,
            config_hash="2" * 64,
            run_intent="formal",
            formal_eligible=True,
            payload=role_111[kind],
            source_refs=tuple(s111_sources),
        ).commit_ref
    return refs, {"roles": role_110, "source_refs": ["evidence/stage1/s1-10-authority/index.json"]}, {"roles": role_111, "source_refs": ["evidence/stage1/s1-11-authority/index.json"]}


def test_real_task_artifact_group_loader_passes_and_tamper_blocks(tmp_path: Path) -> None:
    refs, s110, s111 = _stage1_fixture(tmp_path)
    loaded = _load_stage1_group(tmp_path, refs, s110_authority=s110, s111_authority=s111)
    assert set(loaded) == set(STAGE1_ARTIFACT_KINDS)

    commit_path = tmp_path / Path(*refs["stage_report"].split("/"))
    commit = load_canonical_json(commit_path)
    assert isinstance(commit, dict)
    object_path = tmp_path / Path(*str(commit["object_ref"]).split("/"))
    envelope = load_canonical_json(object_path)
    assert isinstance(envelope, dict)
    envelope["payload"]["role"] = "tampered"  # type: ignore[index]
    write_canonical_json(object_path, envelope)
    with pytest.raises(Exception):
        _load_stage1_group(tmp_path, refs, s110_authority=s110, s111_authority=s111)
