"""Content- and impact-scoped evidence reuse across Git commits."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from .atomic import sha256_file
from .contracts import canonical_json_hash, load_canonical_json
from .contracts.jsonio import JSONValue


SCHEMA_VERSION = "evidence-reuse-attestation-v1"
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_NON_INVALIDATING_CLASSIFICATIONS = frozenset(
    {"metadata_only", "downstream_only", "consumer_only", "validation_only"}
)
_REVIEW_FIELDS = {
    "path",
    "classification",
    "affected_gate_ids",
    "rationale",
    "validation_refs",
}


class EvidenceReuseError(RuntimeError):
    """A cross-commit evidence reuse claim is incomplete or unsafe."""


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise EvidenceReuseError(f"EVIDENCE_REUSE_COMMIT_INVALID:{field}")
    return value


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvidenceReuseError(f"EVIDENCE_REUSE_PATH_INVALID:{field}")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise EvidenceReuseError(f"EVIDENCE_REUSE_PATH_ESCAPE:{field}") from error
    return path


def changed_paths(repository: str | Path, producer_commit: str, consumer_commit: str) -> tuple[str, ...]:
    """Return the exact tracked path delta for an ancestor-to-descendant reuse claim."""

    root = Path(repository).resolve(strict=True)
    producer = _commit(producer_commit, field="producer_commit")
    consumer = _commit(consumer_commit, field="consumer_commit")
    for commit in (producer, consumer):
        probe = _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
        if probe.returncode != 0:
            raise EvidenceReuseError("EVIDENCE_REUSE_COMMIT_NOT_FOUND")
    ancestor = _git(root, "merge-base", "--is-ancestor", producer, consumer)
    if ancestor.returncode != 0:
        raise EvidenceReuseError("EVIDENCE_REUSE_PRODUCER_NOT_ANCESTOR")
    diff = _git(root, "diff", "--name-only", "--diff-filter=ACDMRTUXB", "-z", producer, consumer)
    if diff.returncode != 0:
        raise EvidenceReuseError("EVIDENCE_REUSE_DIFF_FAILED")
    return tuple(sorted(item for item in diff.stdout.split("\0") if item))


def _normalize_reviews(
    reviews: Sequence[Mapping[str, object]],
    *,
    expected_paths: Sequence[str],
) -> list[dict[str, JSONValue]]:
    normalized: list[dict[str, JSONValue]] = []
    for index, raw in enumerate(reviews):
        if set(raw) != _REVIEW_FIELDS:
            raise EvidenceReuseError(f"EVIDENCE_REUSE_REVIEW_FIELDS_INVALID:{index}")
        path = raw.get("path")
        classification = raw.get("classification")
        affected = raw.get("affected_gate_ids")
        rationale = raw.get("rationale")
        validation_refs = raw.get("validation_refs")
        if not isinstance(path, str) or not path or "\\" in path or path.startswith("/"):
            raise EvidenceReuseError(f"EVIDENCE_REUSE_REVIEW_PATH_INVALID:{index}")
        if classification not in _NON_INVALIDATING_CLASSIFICATIONS:
            raise EvidenceReuseError(f"EVIDENCE_REUSE_CLASSIFICATION_INVALID:{path}")
        if affected != []:
            raise EvidenceReuseError(f"EVIDENCE_REUSE_AFFECTED_GATES_NOT_EMPTY:{path}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise EvidenceReuseError(f"EVIDENCE_REUSE_RATIONALE_MISSING:{path}")
        if (
            not isinstance(validation_refs, list)
            or any(not isinstance(item, str) or not item for item in validation_refs)
            or len(set(validation_refs)) != len(validation_refs)
        ):
            raise EvidenceReuseError(f"EVIDENCE_REUSE_VALIDATION_REFS_INVALID:{path}")
        if classification in {"consumer_only", "validation_only"} and not validation_refs:
            raise EvidenceReuseError(f"EVIDENCE_REUSE_VALIDATION_REQUIRED:{path}")
        normalized.append(
            {
                "path": path,
                "classification": classification,
                "affected_gate_ids": [],
                "rationale": rationale.strip(),
                "validation_refs": list(validation_refs),
            }
        )
    normalized.sort(key=lambda item: str(item["path"]))
    review_paths = [str(item["path"]) for item in normalized]
    if review_paths != list(expected_paths):
        raise EvidenceReuseError("EVIDENCE_REUSE_CHANGED_PATHS_NOT_FULLY_REVIEWED")
    return normalized


def _validation_evidence(
    root: Path,
    reviews: Sequence[Mapping[str, JSONValue]],
) -> list[dict[str, JSONValue]]:
    references: set[str] = set()
    for review in reviews:
        raw_references = review.get("validation_refs")
        if not isinstance(raw_references, list):
            raise EvidenceReuseError("EVIDENCE_REUSE_VALIDATION_REFS_INVALID")
        references.update(str(reference) for reference in raw_references)
    evidence: list[dict[str, JSONValue]] = []
    for reference in sorted(references):
        path = _logical_path(root, reference, field="validation_ref")
        if not path.is_file():
            raise EvidenceReuseError(f"EVIDENCE_REUSE_VALIDATION_EVIDENCE_MISSING:{reference}")
        evidence.append({"ref": reference, "sha256": sha256_file(path)})
    return evidence


def build_evidence_reuse_attestation(
    *,
    repository: str | Path,
    data_root: str | Path,
    producer_commit: str,
    consumer_commit: str,
    consumer_branch: str,
    scope_id: str,
    source_evidence_ref: str,
    preserved_gate_ids: Sequence[str],
    reviews: Sequence[Mapping[str, object]],
    generated_at: str | None = None,
) -> dict[str, JSONValue]:
    """Build a PASS attestation only when every changed path is reviewed as non-invalidating."""

    repository_root = Path(repository).resolve(strict=True)
    evidence_root = Path(data_root).resolve(strict=True)
    producer = _commit(producer_commit, field="producer_commit")
    consumer = _commit(consumer_commit, field="consumer_commit")
    if not isinstance(consumer_branch, str) or not consumer_branch:
        raise EvidenceReuseError("EVIDENCE_REUSE_BRANCH_INVALID")
    if not isinstance(scope_id, str) or not scope_id:
        raise EvidenceReuseError("EVIDENCE_REUSE_SCOPE_INVALID")
    gates = sorted(set(preserved_gate_ids))
    if not gates or len(gates) != len(preserved_gate_ids):
        raise EvidenceReuseError("EVIDENCE_REUSE_GATE_SET_INVALID")
    source_path = _logical_path(evidence_root, source_evidence_ref, field="source_evidence_ref")
    if not source_path.is_file():
        raise EvidenceReuseError("EVIDENCE_REUSE_SOURCE_EVIDENCE_MISSING")
    paths = changed_paths(repository_root, producer, consumer)
    normalized_reviews = _normalize_reviews(reviews, expected_paths=paths)
    validation_evidence = _validation_evidence(evidence_root, normalized_reviews)
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceReuseError("EVIDENCE_REUSE_TIMESTAMP_INVALID") from error
    if parsed.tzinfo is None:
        raise EvidenceReuseError("EVIDENCE_REUSE_TIMESTAMP_TZ_MISSING")
    value: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scope_id": scope_id,
        "producer_git_commit": producer,
        "consumer_git_commit": consumer,
        "consumer_git_branch": consumer_branch,
        "generated_at": timestamp,
        "source_evidence_ref": source_evidence_ref,
        "source_evidence_sha256": sha256_file(source_path),
        "changed_paths": list(paths),
        "reviews": normalized_reviews,
        "validation_evidence": validation_evidence,
        "preserved_gate_ids": gates,
        "rerun_gate_ids": [],
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def validate_evidence_reuse_attestation(
    *,
    repository: str | Path,
    data_root: str | Path,
    attestation_ref: str,
    producer_commit: str,
    consumer_commit: str,
    consumer_branch: str,
    scope_id: str,
    source_evidence_ref: str,
    required_gate_ids: Sequence[str],
) -> dict[str, Any]:
    """Recompute Git and file identities before accepting a reuse attestation."""

    root = Path(data_root).resolve(strict=True)
    path = _logical_path(root, attestation_ref, field="attestation_ref")
    raw = load_canonical_json(path)
    if not isinstance(raw, Mapping):
        raise EvidenceReuseError("EVIDENCE_REUSE_ATTESTATION_NOT_OBJECT")
    value = dict(raw)
    declared = value.pop("artifact_hash", None)
    expected_fields = {
        "schema_version",
        "status",
        "scope_id",
        "producer_git_commit",
        "consumer_git_commit",
        "consumer_git_branch",
        "generated_at",
        "source_evidence_ref",
        "source_evidence_sha256",
        "changed_paths",
        "reviews",
        "validation_evidence",
        "preserved_gate_ids",
        "rerun_gate_ids",
    }
    if set(value) != expected_fields or declared != canonical_json_hash(value):
        raise EvidenceReuseError("EVIDENCE_REUSE_ATTESTATION_HASH_OR_FIELDS_INVALID")
    producer = _commit(producer_commit, field="expected_producer_commit")
    consumer = _commit(consumer_commit, field="expected_consumer_commit")
    expected_paths = changed_paths(repository, producer, consumer)
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or any(not isinstance(item, Mapping) for item in reviews):
        raise EvidenceReuseError("EVIDENCE_REUSE_REVIEWS_INVALID")
    normalized_reviews = _normalize_reviews(reviews, expected_paths=expected_paths)
    validation_evidence = _validation_evidence(root, normalized_reviews)
    source_path = _logical_path(root, source_evidence_ref, field="source_evidence_ref")
    required_gates = sorted(set(required_gate_ids))
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "PASS"
        or value.get("scope_id") != scope_id
        or value.get("producer_git_commit") != producer
        or value.get("consumer_git_commit") != consumer
        or value.get("consumer_git_branch") != consumer_branch
        or value.get("source_evidence_ref") != source_evidence_ref
        or value.get("source_evidence_sha256") != sha256_file(source_path)
        or value.get("changed_paths") != list(expected_paths)
        or value.get("reviews") != normalized_reviews
        or value.get("validation_evidence") != validation_evidence
        or value.get("preserved_gate_ids") != required_gates
        or value.get("rerun_gate_ids") != []
    ):
        raise EvidenceReuseError("EVIDENCE_REUSE_ATTESTATION_IDENTITY_INVALID")
    value["artifact_hash"] = declared
    return value


__all__ = [
    "EvidenceReuseError",
    "SCHEMA_VERSION",
    "build_evidence_reuse_attestation",
    "changed_paths",
    "validate_evidence_reuse_attestation",
]
