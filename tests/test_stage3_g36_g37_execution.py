"""Tests for the ordered G3-6/G3-7 execution-evidence publication."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.stage3.publish_stage3_g36_g37_execution import (
    Stage3GateExecutionPublicationError,
    publish_execution_chain,
)
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.experiments.stage3_g37_publisher import Stage3G37Publisher
from param_importance_nlp.runtime import load_committed_task_artifact


def _g37_helpers():
    path = Path(__file__).with_name("test_stage3_g37_publisher.py")
    spec = importlib.util.spec_from_file_location("stage3_g37_test_helpers", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load Stage3 G3-7 test helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arguments(tmp_path: Path, refs: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=tmp_path,
        base_execution_ref=refs["execution_evidence_ref"],
        g36_publication_ref="artifacts/two-g36/commits/g36_publication.json",
        g37_publication_ref="artifacts/g37/commits/g37_publication.json",
        g36_execution_output_dir="evidence/stage3/g36-execution",
        g37_execution_output_dir="evidence/stage3/g37-execution",
        receipt=Path("results/stage3/gate-execution-receipt.json"),
    )


def test_execution_chain_appends_exact_gates_in_order_and_is_idempotent(
    tmp_path: Path,
) -> None:
    helper = _g37_helpers()
    refs = helper._pass_inputs(tmp_path)
    publication = Stage3G37Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/g37",
        **refs,
    )
    assert publication.status == "PASS"
    arguments = _arguments(tmp_path, refs)
    first = publish_execution_chain(arguments)
    second = publish_execution_chain(arguments)
    assert first == second
    assert first["final_execution_ref"] == (
        "evidence/stage3/g37-execution/execution-evidence/commits/"
        "formal_execution_evidence.json"
    )
    loaded = load_committed_task_artifact(
        tmp_path, first["final_execution_ref"], require_formal=True
    )
    execution = FormalExecutionEvidence.from_mapping(dict(loaded.payload))
    assert [gate.gate_id for gate in execution.prerequisite_gates] == [
        "stage3.G3-0",
        "stage3.G3-1",
        "stage3.G3-2",
        "stage3.G3-3",
        "stage3.G3-4",
        "stage3.G3-5",
        "stage3.G3-6",
        "stage3.G3-7",
    ]
    intermediate = load_committed_task_artifact(
        tmp_path, first["intermediate_execution_ref"], require_formal=True
    )
    assert intermediate.identity.config_hash == helper.CONFIG_HASH
    assert loaded.identity.config_hash == helper.STAGE309_CONFIG_HASH


def test_execution_chain_rejects_publication_bound_to_another_base_before_write(
    tmp_path: Path,
) -> None:
    helper = _g37_helpers()
    refs = helper._pass_inputs(tmp_path)
    Stage3G37Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/g37",
        **refs,
    )
    arguments = _arguments(tmp_path, refs)
    arguments.base_execution_ref = "artifacts/two-execution/commits/formal_execution_evidence.json"
    # The canonical helper uses a different source ref after its plan rewrite;
    # copying the same payload to a second commit makes the mismatch explicit.
    base = load_committed_task_artifact(
        tmp_path, refs["execution_evidence_ref"], require_formal=True
    )
    from param_importance_nlp.runtime import TaskArtifactStore

    alternate = TaskArtifactStore(tmp_path, "artifacts/alternate-execution").publish(
        task_id="stage3.test",
        artifact_kind="formal_execution_evidence",
        config_hash=base.identity.config_hash,
        run_intent="formal",
        payload=dict(base.payload),
        formal_eligible=True,
        source_refs=base.source_refs,
    )
    arguments.base_execution_ref = alternate.commit_ref
    with pytest.raises(
        Stage3GateExecutionPublicationError,
        match="G36_PUBLICATION_INVALID",
    ):
        publish_execution_chain(arguments)
    assert not (tmp_path / "evidence/stage3/g36-execution").exists()


def test_execution_chain_rejects_blocked_g37_before_write(tmp_path: Path) -> None:
    helper = _g37_helpers()
    refs, _g36, _source = helper._inputs(tmp_path)
    publication = Stage3G37Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/g37",
        **refs,
    )
    assert publication.status == "BLOCKED"
    arguments = SimpleNamespace(
        data_root=tmp_path,
        base_execution_ref=refs["execution_evidence_ref"],
        g36_publication_ref="artifacts/g36/commits/g36_publication.json",
        g37_publication_ref="artifacts/g37/commits/g37_publication.json",
        g36_execution_output_dir="evidence/stage3/g36-execution",
        g37_execution_output_dir="evidence/stage3/g37-execution",
        receipt=None,
    )
    with pytest.raises(
        Stage3GateExecutionPublicationError,
        match="G37_PUBLICATION_INVALID",
    ):
        publish_execution_chain(arguments)
    assert not (tmp_path / "evidence/stage3/g36-execution").exists()
