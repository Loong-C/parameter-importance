from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ops.stage3.publish_stage3_g38 import main as publish_main
from ops.stage3.verify_stage3_g38_handoff import main as handoff_main
from param_importance_nlp.contracts.jsonio import load_canonical_json
from param_importance_nlp.experiments.stage3_g38_publisher import (
    REQUIRED_STAGE3_G38_GATE_IDS,
)
from param_importance_nlp.runtime import load_committed_task_artifact


def _publisher_test_module():
    path = Path(__file__).with_name("test_stage3_g38_publisher.py")
    spec = importlib.util.spec_from_file_location("stage3_g38_entrypoint_helpers", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load Stage3 G3-8 test helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _publication_arguments(root: Path, inputs: dict[str, object]) -> list[str]:
    arguments = [
        "--workspace-root", str(root),
        "--output-dir", "outputs/g38-cli",
        "--execution-evidence-ref", str(inputs["execution_evidence_ref"]),
        "--g37-publication-ref", str(inputs["g3_7_publication_ref"]),
        "--recommendation-ref", str(inputs["recommendation_ref"]),
        "--finalization-ref", str(inputs["finalization_ref"]),
        "--delivery-manifest-ref", str(inputs["delivery_manifest_ref"]),
        "--checked-at", "2026-08-28T01:00:00Z",
    ]
    gate_refs = inputs["gate_refs"]
    for index, gate_id in enumerate(REQUIRED_STAGE3_G38_GATE_IDS):
        arguments.extend((f"--g3-{index}-ref", str(gate_refs[gate_id])))
    for role, ref in inputs["stage3_10_refs"].items():
        arguments.extend((f"--{role.replace('_', '-')}-ref", str(ref)))
    return arguments


def test_g38_and_stage4_entrypoints_publish_canonical_immutable_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    inputs = _publisher_test_module()._inputs(tmp_path)
    arguments = _publication_arguments(tmp_path, inputs)
    assert publish_main(arguments) == 0
    publication_report = json.loads(capsys.readouterr().out)
    assert publication_report["status"] == "PASS"
    assert publication_report["g3_8_ref"] == "outputs/g38-cli/commits/gate_record.json"
    assert publication_report["publication_ref"] == "outputs/g38-cli/commits/g38_publication.json"
    gate = load_committed_task_artifact(
        tmp_path, publication_report["g3_8_ref"], require_formal=True
    )
    receipt = load_committed_task_artifact(
        tmp_path, publication_report["publication_ref"], require_formal=True
    )
    assert gate.identity.artifact_hash == publication_report["g3_8_commit_hash"]
    assert receipt.identity.artifact_hash == publication_report["publication_commit_hash"]

    audit_ref = "results/stage4/stage3-g38-handoff-audit.json"
    handoff_arguments = [
        "--workspace-root", str(tmp_path),
        "--g3-8-gate-ref", publication_report["g3_8_ref"],
        "--g3-8-publication-ref", publication_report["publication_ref"],
        "--output", audit_ref,
    ]
    assert handoff_main(handoff_arguments) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit == load_canonical_json(tmp_path / audit_ref)
    assert audit["status"] == "PASS"
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas/stage4/stage3-handoff-audit-v1.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(audit)

    # Exact retries are idempotent at both immutable publication boundaries.
    assert publish_main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == publication_report
    assert handoff_main(handoff_arguments) == 0
    assert json.loads(capsys.readouterr().out) == audit
