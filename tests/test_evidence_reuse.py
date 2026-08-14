"""Cross-commit evidence reuse is explicit, complete, and fail closed."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess

import pytest

from param_importance_nlp.contracts import write_canonical_json
from param_importance_nlp.cli import _validate_project_json_schema
from param_importance_nlp.evidence_reuse import (
    EvidenceReuseError,
    build_evidence_reuse_attestation,
    validate_evidence_reuse_attestation,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_evidence_reuse_schema_is_a_valid_project_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "schemas/shared/evidence-reuse-attestation-v1.json").read_text(
            encoding="utf-8"
        )
    )
    _validate_project_json_schema(schema)


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "Fixture")
    (repository / "producer.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "producer.py")
    _git(repository, "commit", "-m", "producer")
    producer = _git(repository, "rev-parse", "HEAD")
    (repository / "consumer.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repository, "add", "consumer.py")
    _git(repository, "commit", "-m", "consumer")
    consumer = _git(repository, "rev-parse", "HEAD")
    return repository, producer, consumer


def test_reuse_attestation_verifies_exact_git_diff_and_source_hash(tmp_path: Path) -> None:
    repository, producer, consumer = _repository(tmp_path)
    data_root = tmp_path / "data"
    source_ref = "evidence/g9/index.json"
    write_canonical_json(data_root / source_ref, {"status": "PASS"})
    validation_ref = "evidence/tests/consumer.json"
    write_canonical_json(data_root / validation_ref, {"status": "PASS", "tests": 1})
    reviews = [
        {
            "path": "consumer.py",
            "classification": "consumer_only",
            "affected_gate_ids": [],
            "rationale": "Only the downstream consumer was added.",
            "validation_refs": [validation_ref],
        }
    ]
    value = build_evidence_reuse_attestation(
        repository=repository,
        data_root=data_root,
        producer_commit=producer,
        consumer_commit=consumer,
        consumer_branch="main",
        scope_id="stage0.G0-G9",
        source_evidence_ref=source_ref,
        preserved_gate_ids=["stage0.G9"],
        reviews=reviews,
        generated_at="2026-08-14T00:00:00Z",
    )
    attestation_ref = "evidence/reuse/attestation.json"
    write_canonical_json(data_root / attestation_ref, value)

    loaded = validate_evidence_reuse_attestation(
        repository=repository,
        data_root=data_root,
        attestation_ref=attestation_ref,
        producer_commit=producer,
        consumer_commit=consumer,
        consumer_branch="main",
        scope_id="stage0.G0-G9",
        source_evidence_ref=source_ref,
        required_gate_ids=["stage0.G9"],
    )

    assert loaded["status"] == "PASS"
    assert loaded["changed_paths"] == ["consumer.py"]
    assert loaded["validation_evidence"][0]["ref"] == validation_ref
    assert loaded["rerun_gate_ids"] == []

    write_canonical_json(
        data_root / validation_ref,
        {"status": "PASS", "tests": 2},
    )
    with pytest.raises(EvidenceReuseError, match="IDENTITY_INVALID"):
        validate_evidence_reuse_attestation(
            repository=repository,
            data_root=data_root,
            attestation_ref=attestation_ref,
            producer_commit=producer,
            consumer_commit=consumer,
            consumer_branch="main",
            scope_id="stage0.G0-G9",
            source_evidence_ref=source_ref,
            required_gate_ids=["stage0.G9"],
        )


def test_reuse_attestation_rejects_unreviewed_or_producer_semantic_change(
    tmp_path: Path,
) -> None:
    repository, producer, consumer = _repository(tmp_path)
    data_root = tmp_path / "data"
    source_ref = "evidence/g9/index.json"
    write_canonical_json(data_root / source_ref, {"status": "PASS"})

    with pytest.raises(EvidenceReuseError, match="NOT_FULLY_REVIEWED"):
        build_evidence_reuse_attestation(
            repository=repository,
            data_root=data_root,
            producer_commit=producer,
            consumer_commit=consumer,
            consumer_branch="main",
            scope_id="stage0.G0-G9",
            source_evidence_ref=source_ref,
            preserved_gate_ids=["stage0.G9"],
            reviews=[],
        )
    with pytest.raises(EvidenceReuseError, match="CLASSIFICATION_INVALID"):
        build_evidence_reuse_attestation(
            repository=repository,
            data_root=data_root,
            producer_commit=producer,
            consumer_commit=consumer,
            consumer_branch="main",
            scope_id="stage0.G0-G9",
            source_evidence_ref=source_ref,
            preserved_gate_ids=["stage0.G9"],
            reviews=[
                {
                    "path": "consumer.py",
                    "classification": "producer_semantics",
                    "affected_gate_ids": ["stage0.G9"],
                    "rationale": "This must be rerun, not attested away.",
                    "validation_refs": [],
                }
            ],
        )
