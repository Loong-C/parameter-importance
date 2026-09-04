"""Publish the independent Stage 3 G3-6 decision boundary.

Stage 3.08 produces the frozen observation table.  This module is the small
post-Stage-3.08 handoff that *reads* the already committed inputs, runs the
independent :class:`Stage3GateEvaluator`, and publishes its evaluation before
publishing G3-6.  In particular, provenance is completed before this module
runs and is never amended with the two artifacts produced here.  That keeps
the provenance/evaluation/Gate dependency acyclic.

The function deliberately accepts commit references rather than arbitrary
objects or paths.  Every input is reloaded through ``load_committed_task_artifact``
with ``require_formal=True``; a local file, an object without a commit, or a
fixture envelope cannot enter the formal decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
import math
import re

from ..analysis.report import FrozenSourceTable
from ..contracts.errors import FormalRunRejected
from ..contracts.immutable import thaw_json_value
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.provenance import ProvenanceRecord, ProvenanceStatus
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.stage3_scope import (
    validate_stage3_scope_authority,
    validate_stage3_scope_decision,
)
from ..contracts.status import GateRecord, GateStatus
from ..runtime.task_artifacts import (
    LoadedTaskArtifact,
    TaskArtifactStore,
    load_committed_task_artifact,
)
from .stage3_formal import PersistentNodeGradientCache
from .stage3_raw_storage import (
    RAW_AGGREGATE_SCHEMA,
    VECTOR_DERIVATION_HASH,
    VECTOR_DERIVATION_SCHEMA,
    _load_shard as _load_raw_shard,
)
from .stage3_gate import (
    Stage3GateEvaluation,
    Stage3GateEvaluator,
    _streaming_coverage_metrics,
    _validate_streaming_coverage,
)


STAGE3_G36_PUBLICATION_SCHEMA = "stage3-g36-publication-v1"
STAGE3_G36_TASK_ID = "stage3.08_g3_6_publisher"
STAGE3_G36_EVALUATION_ARTIFACT_KIND = "gate_evaluation"
STAGE3_G36_GATE_ARTIFACT_KIND = "gate_record"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[^\\?]+$")
_TABLE_KINDS = frozenset({"frozen_source_table"})
_PLAN_KINDS = frozenset({"formal_plan", "stage3_formal_plan"})
_PROVENANCE_KINDS = frozenset({"provenance_record", "provenance"})
_EXECUTION_KINDS = frozenset({"formal_execution_evidence", "execution_evidence"})
_SCOPE_KINDS = frozenset({"scope_authority", "stage3_scope_authority"})
_REFERENCE_AGGREGATE_FIELDS = frozenset(
    {
        "schema_version",
        "reference_scope",
        "execution_evidence_hash",
        "reference_binding_hash",
        "required_unit_ids",
        "complete_unit_ids",
        "missing_unit_ids",
        "unit_references",
        "artifact_hash",
    }
)
_REFERENCE_SHARD_FIELDS = frozenset(
    {
        "schema_version",
        "unit_id",
        "path_identity_hash",
        "execution_evidence_hash",
        "reference_binding_hash",
        "reference_artifact_hash",
        "reference",
        "contribution_bundle_ref",
        "contribution_bundle_manifest_hash",
        "required_unit_ids",
        "scientific_identity_hash",
        "artifact_hash",
    }
)
_STREAMING_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "unit_id",
        "required_unit_ids",
        "candidate_rule_names",
        "execution_evidence_hash",
        "formal_plan_ref",
        "formal_plan_hash",
        "production_unit_index_ref",
        "production_unit_index_hash",
        "reference_binding_hash",
        "path_identity_hash",
        "reference_artifact_hash",
        "reference_shard_ref",
        "reference_shard_hash",
        "reference_aggregate_ref",
        "reference_aggregate_hash",
        "observation_ledger_ref",
        "observation_artifact_hash",
        "raw_shard_ref",
        "raw_shard_hash",
        "raw_bundle_ref",
        "raw_bundle_manifest_hash",
        "raw_aggregate_ref",
        "raw_aggregate_hash",
        "node_cache_evidence_hash",
        "node_cache_seal_ref",
        "node_cache_seal_hash",
        "local_commit_state",
        "artifact_hash",
    }
)
_STREAMING_EVICTION_FIELDS = frozenset(
    {
        "schema_version",
        "unit_id",
        "unit_receipt_ref",
        "unit_receipt_hash",
        "seal_ref",
        "seal_hash",
        "cache_root_ref",
        "eviction_ref",
        "eviction_hash",
        "execution_evidence_hash",
        "reference_binding_hash",
        "external_request_id",
        "idempotency_key",
        "state",
        "artifact_hash",
    }
)


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} 必须是小写 SHA-256")
    return value


def _ref(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not _REF_RE.fullmatch(value)
        or "://" in value
    ):
        raise ValueError(f"{field} 必须是稳定 commit ref")
    lowered = value.casefold()
    if "fixture" in lowered or "synthetic" in lowered:
        raise FormalRunRejected(f"{field} 禁止 fixture/synthetic ref")
    return value


def _safe_task_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or not value[0].isalnum()
        or any(not (char.isalnum() or char in "._-") for char in value)
    ):
        raise ValueError("task_id 不是安全标识")
    return value


def _load_formal_commit(
    root: Path,
    reference: str,
    *,
    field: str,
    kinds: frozenset[str],
) -> LoadedTaskArtifact:
    ref = _ref(reference, field=field)
    try:
        loaded = load_committed_task_artifact(root, ref, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_COMMIT_INVALID") from error
    if loaded.identity.artifact_kind not in kinds:
        raise FormalRunRejected(
            f"STAGE3_G36_{field.upper()}_ARTIFACT_KIND_INVALID:{loaded.identity.artifact_kind}"
        )
    if loaded.run_intent != "formal" or loaded.identity.formal_eligible is not True:
        raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_FORMAL_COMMIT_REQUIRED")
    return loaded


def _resolve_streaming_plan_ref(
    root: Path,
    *,
    plan_artifact: LoadedTaskArtifact,
    formal_plan_ref: str,
    streaming_formal_plan_ref: str | None,
) -> str:
    """Bind a historical matrix-plan ref to its promoted formal authority.

    Streaming S3.07 receipts created before plan promotion retain the original
    canonical document ref. A later formal commit may wrap that exact payload,
    but only when its immutable source refs name the original document. This
    bridge preserves the receipts without treating the raw document as a
    formal commit.
    """

    if streaming_formal_plan_ref is None:
        return formal_plan_ref
    source_ref = _ref(
        streaming_formal_plan_ref,
        field="streaming_formal_plan_ref",
    )
    if source_ref == formal_plan_ref:
        return source_ref
    if source_ref not in plan_artifact.source_refs:
        raise FormalRunRejected("STAGE3_G36_STREAMING_PLAN_SOURCE_UNBOUND")
    source = _load_workspace_json(root, source_ref, field="streaming_formal_plan")
    if dict(source) != dict(plan_artifact.payload):
        raise FormalRunRejected("STAGE3_G36_STREAMING_PLAN_PAYLOAD_MISMATCH")
    return source_ref


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} 必须是 object")
    return value


def _load_workspace_json(root: Path, reference: object, *, field: str) -> Mapping[str, object]:
    """Load one canonical JSON object using the same workspace safety fence."""

    ref = _ref(reference, field=field)
    logical = PurePosixPath(ref)
    if (
        Path(ref).is_absolute()
        or logical.is_absolute()
        or any(part in {"", ".", ".."} for part in logical.parts)
        or "\\" in ref
    ):
        raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_REF_ESCAPE")
    if root.is_symlink():
        raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_SYMLINK")
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_SYMLINK")
    try:
        payload = load_canonical_json(root.joinpath(*logical.parts))
    except (OSError, TypeError, ValueError) as error:
        raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_LOAD_FAILED") from error
    return _mapping(payload, field=field)


def _checked_body_hash(value: Mapping[str, object], *, field: str) -> str:
    supplied = value.get("artifact_hash")
    _hash(supplied, field=f"{field}.artifact_hash")
    if canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    ) != supplied:
        raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_HASH_MISMATCH")
    return str(supplied)


def _load_reference_aggregate(
    root: Path,
    reference: str,
    *,
    declared_hash: object,
    expected_units: tuple[str, ...],
    expected_execution_hash: str,
    expected_binding_hash: str,
) -> tuple[Mapping[str, object], Mapping[str, Mapping[str, object]]]:
    """Reload the complete reference aggregate and its 99 immutable shards."""

    aggregate = _load_workspace_json(root, reference, field="reference_aggregate")
    if set(aggregate) != _REFERENCE_AGGREGATE_FIELDS:
        raise FormalRunRejected("STAGE3_G36_REFERENCE_AGGREGATE_FIELDS_INVALID")
    aggregate_hash = _checked_body_hash(aggregate, field="reference_aggregate")
    _hash(declared_hash, field="streaming_reference_aggregate_hash")
    if aggregate_hash != declared_hash:
        raise FormalRunRejected("STAGE3_G36_REFERENCE_AGGREGATE_HASH_MISMATCH")
    if (
        aggregate.get("schema_version") != "stage3-reference-aggregate-v1"
        or aggregate.get("reference_scope") != "matrix"
        or aggregate.get("execution_evidence_hash") != expected_execution_hash
        or aggregate.get("reference_binding_hash") != expected_binding_hash
    ):
        raise FormalRunRejected("STAGE3_G36_REFERENCE_AGGREGATE_BINDING_INVALID")
    if aggregate.get("required_unit_ids") != list(expected_units):
        raise FormalRunRejected("STAGE3_G36_REFERENCE_AGGREGATE_REQUIRED_UNITS_INVALID")
    if aggregate.get("complete_unit_ids") != list(expected_units) or aggregate.get("missing_unit_ids") != []:
        raise FormalRunRejected("STAGE3_G36_REFERENCE_AGGREGATE_INCOMPLETE")
    entries = aggregate.get("unit_references")
    if not isinstance(entries, Mapping) or set(entries) != set(expected_units):
        raise FormalRunRejected("STAGE3_G36_REFERENCE_AGGREGATE_ENTRIES_INVALID")

    loaded_entries: dict[str, Mapping[str, object]] = {}
    for unit_id in expected_units:
        entry = entries[unit_id]
        if not isinstance(entry, Mapping):
            raise FormalRunRejected(f"STAGE3_G36_REFERENCE_ENTRY_INVALID:{unit_id}")
        expected_entry_fields = {
            "ledger_ref",
            "ledger_artifact_hash",
            "reference_artifact_hash",
            "path_identity_hash",
            "contribution_bundle_ref",
            "contribution_bundle_manifest_hash",
            "reference_binding_hash",
        }
        if set(entry) != expected_entry_fields:
            raise FormalRunRejected(f"STAGE3_G36_REFERENCE_ENTRY_FIELDS_INVALID:{unit_id}")
        for field in (
            "ledger_artifact_hash",
            "reference_artifact_hash",
            "path_identity_hash",
            "contribution_bundle_manifest_hash",
            "reference_binding_hash",
        ):
            _hash(entry[field], field=f"reference_entry[{unit_id}].{field}")
        if entry["reference_binding_hash"] != expected_binding_hash:
            raise FormalRunRejected(f"STAGE3_G36_REFERENCE_ENTRY_BINDING_INVALID:{unit_id}")
        shard = _load_workspace_json(root, entry["ledger_ref"], field="reference_shard")
        if set(shard) != _REFERENCE_SHARD_FIELDS:
            raise FormalRunRejected(f"STAGE3_G36_REFERENCE_SHARD_FIELDS_INVALID:{unit_id}")
        shard_hash = _checked_body_hash(shard, field="reference_shard")
        _hash(shard["scientific_identity_hash"], field=f"reference_shard[{unit_id}].scientific_identity_hash")
        reference_payload = shard.get("reference")
        if not isinstance(reference_payload, Mapping):
            raise FormalRunRejected(f"STAGE3_G36_REFERENCE_PAYLOAD_INVALID:{unit_id}")
        reference_hash = canonical_json_hash(reference_payload)
        if (
            shard_hash != entry["ledger_artifact_hash"]
            or shard.get("unit_id") != unit_id
            or shard.get("required_unit_ids") != list(expected_units)
            or shard.get("execution_evidence_hash") != expected_execution_hash
            or shard.get("reference_binding_hash") != expected_binding_hash
            or shard.get("reference_artifact_hash") != reference_hash
            or entry["reference_artifact_hash"] != reference_hash
            or reference_payload.get("execution_evidence_hash") != expected_execution_hash
            or reference_payload.get("reference_binding_hash") != expected_binding_hash
            or not isinstance(reference_payload.get("refinement"), Mapping)
            or reference_payload["refinement"].get("converged") is not True
        ):
            raise FormalRunRejected(f"STAGE3_G36_REFERENCE_SHARD_BINDING_INVALID:{unit_id}")
        if reference_payload.get("path_identity_hash") != entry["path_identity_hash"]:
            raise FormalRunRejected(f"STAGE3_G36_REFERENCE_PATH_IDENTITY_INVALID:{unit_id}")
        loaded_entries[unit_id] = {
            "entry": entry,
            "shard": shard,
        }
    return aggregate, loaded_entries


def _load_raw_aggregate_for_streaming(
    root: Path,
    reference: str,
    *,
    declared_hash: object,
    expected_units: tuple[str, ...],
    expected_rules: tuple[str, ...],
    expected_execution_hash: str,
    expected_binding_hash: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Reload a raw aggregate with bounded memory.

    The raw storage loader returns every TensorBundle state in one mapping,
    which is appropriate for small callers but would retain the complete
    formal matrix in memory.  G3-6 only needs the aggregate's metadata after
    validation, so each shard/bundle is loaded, checked, and released before
    proceeding to the next unit.
    """

    raw = _load_workspace_json(root, reference, field="raw_aggregate")
    required_fields = {
        "schema_version",
        "execution_evidence_hash",
        "reference_binding_hash",
        "required_unit_ids",
        "candidate_rule_names",
        "vector_derivation_schema",
        "vector_derivation_contract_hash",
        "complete_unit_ids",
        "missing_unit_ids",
        "unit_shards",
        "artifact_hash",
    }
    if set(raw) != required_fields or raw.get("schema_version") != RAW_AGGREGATE_SCHEMA:
        raise FormalRunRejected("STAGE3_G36_RAW_AGGREGATE_FIELDS_INVALID")
    aggregate_hash = _checked_body_hash(raw, field="raw_aggregate")
    _hash(declared_hash, field="streaming_raw_aggregate_hash")
    if aggregate_hash != declared_hash:
        raise FormalRunRejected("STAGE3_G36_RAW_AGGREGATE_HASH_MISMATCH")
    entries = raw.get("unit_shards")
    if not isinstance(entries, Mapping):
        raise FormalRunRejected("STAGE3_G36_RAW_AGGREGATE_ENTRIES_INVALID")
    if (
        raw.get("vector_derivation_schema") != VECTOR_DERIVATION_SCHEMA
        or raw.get("vector_derivation_contract_hash") != VECTOR_DERIVATION_HASH
    ):
        raise FormalRunRejected("STAGE3_G36_RAW_AGGREGATE_DERIVATION_INVALID")
    if (
        raw.get("execution_evidence_hash") != expected_execution_hash
        or raw.get("reference_binding_hash") != expected_binding_hash
        or raw.get("required_unit_ids") != list(expected_units)
        or raw.get("candidate_rule_names") != sorted(expected_rules)
        or raw.get("complete_unit_ids") != list(expected_units)
        or raw.get("missing_unit_ids") != []
        or set(entries) != set(expected_units)
    ):
        raise FormalRunRejected("STAGE3_G36_RAW_AGGREGATE_BINDING_INVALID")
    _hash(raw.get("execution_evidence_hash"), field="raw_aggregate.execution_evidence_hash")
    _hash(raw.get("reference_binding_hash"), field="raw_aggregate.reference_binding_hash")
    expected_entry_fields = {
        "shard_ref",
        "shard_hash",
        "bundle_ref",
        "bundle_manifest_hash",
        "path_identity_hash",
        "reference_artifact_hash",
        "reference_identity_hash",
    }
    for unit_id in expected_units:
        entry = entries.get(unit_id)
        if not isinstance(entry, Mapping) or set(entry) != expected_entry_fields:
            raise FormalRunRejected(f"STAGE3_G36_RAW_AGGREGATE_ENTRY_INVALID:{unit_id}")
        for field in (
            "shard_hash",
            "bundle_manifest_hash",
            "path_identity_hash",
            "reference_artifact_hash",
            "reference_identity_hash",
        ):
            _hash(entry.get(field), field=f"raw_entry[{unit_id}].{field}")
        expected = {
            "unit_id": unit_id,
            "required_unit_ids": list(expected_units),
            "candidate_rule_names": sorted(expected_rules),
            "execution_evidence_hash": expected_execution_hash,
            "reference_binding_hash": expected_binding_hash,
        }
        shard = state = bundle = None
        try:
            shard, state, bundle = _load_raw_shard(
                root=root,
                shard_ref=entry.get("shard_ref"),
                expected=expected,
            )
            for field, shard_field in (
                ("shard_hash", "artifact_hash"),
                ("bundle_ref", "bundle_ref"),
                ("bundle_manifest_hash", "bundle_manifest_hash"),
                ("path_identity_hash", "path_identity_hash"),
                ("reference_artifact_hash", "reference_artifact_hash"),
                ("reference_identity_hash", "reference_identity_hash"),
            ):
                if entry.get(field) != shard.get(shard_field):
                    raise FormalRunRejected(
                        f"STAGE3_G36_RAW_AGGREGATE_ENTRY_HASH_MISMATCH:{unit_id}:{field}"
                    )
        except (OSError, TypeError, ValueError) as error:
            raise FormalRunRejected(
                f"STAGE3_G36_RAW_AGGREGATE_SHARD_INVALID:{unit_id}"
            ) from error
        finally:
            # The loader returns the full state and TensorBundle so it can
            # perform strict candidate/vector validation.  Never retain them
            # past this unit; the publisher returns only index metadata.
            del shard, state, bundle
    return raw, entries


def _load_streaming_receipts(
    root: Path,
    aggregate: Mapping[str, object],
    *,
    reference_aggregate: Mapping[str, object],
    reference_entries: Mapping[str, Mapping[str, object]],
    raw_aggregate: Mapping[str, object],
    raw_entries: Mapping[str, object],
    expected_units: tuple[str, ...],
    expected_rules: tuple[str, ...],
    expected_execution_hash: str,
    expected_plan_ref: str,
    expected_plan_hash: str,
    expected_index_ref: str,
    expected_index_hash: str,
    expected_binding_hash: str,
) -> None:
    """Reload every unit receipt, seal, eviction receipt, and tombstone."""

    receipts = aggregate.get("unit_receipts")
    if not isinstance(receipts, Mapping):
        raise FormalRunRejected("STAGE3_G36_STREAMING_RECEIPTS_INVALID")
    for unit_id in expected_units:
        summary = receipts[unit_id]
        if not isinstance(summary, Mapping):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_RECEIPT_SUMMARY_INVALID:{unit_id}")
        receipt = _load_workspace_json(root, summary.get("receipt_ref"), field="streaming_unit_receipt")
        if set(receipt) != _STREAMING_RECEIPT_FIELDS or receipt.get("schema_version") != "stage3-formal-streaming-unit-receipt-v1":
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_RECEIPT_FIELDS_INVALID:{unit_id}")
        receipt_hash = _checked_body_hash(receipt, field="streaming_unit_receipt")
        if receipt_hash != summary.get("receipt_hash"):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_RECEIPT_HASH_MISMATCH:{unit_id}")
        expected_identity = {
            "unit_id": unit_id,
            "required_unit_ids": list(expected_units),
            "candidate_rule_names": list(expected_rules),
            "execution_evidence_hash": expected_execution_hash,
            "formal_plan_ref": expected_plan_ref,
            "formal_plan_hash": expected_plan_hash,
            "production_unit_index_ref": expected_index_ref,
            "production_unit_index_hash": expected_index_hash,
            "reference_binding_hash": expected_binding_hash,
            "local_commit_state": "SEALED",
        }
        if any(receipt.get(key) != expected for key, expected in expected_identity.items()):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_RECEIPT_IDENTITY_INVALID:{unit_id}")
        for field in (
            "execution_evidence_hash", "formal_plan_hash", "production_unit_index_hash",
            "reference_binding_hash", "path_identity_hash", "reference_artifact_hash",
            "reference_shard_hash", "reference_aggregate_hash", "observation_artifact_hash",
            "raw_shard_hash", "raw_bundle_manifest_hash", "raw_aggregate_hash",
            "node_cache_evidence_hash", "node_cache_seal_hash",
        ):
            _hash(receipt[field], field=f"streaming_receipt[{unit_id}].{field}")
        ref_entry = reference_entries[unit_id]["entry"]
        raw_entry = raw_entries.get(unit_id)
        if not isinstance(raw_entry, Mapping):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_RAW_ENTRY_INVALID:{unit_id}")
        if (
            receipt.get("reference_shard_ref") != ref_entry.get("ledger_ref")
            or receipt.get("reference_shard_hash") != ref_entry.get("ledger_artifact_hash")
            or receipt.get("reference_artifact_hash") != ref_entry.get("reference_artifact_hash")
            or receipt.get("path_identity_hash") != ref_entry.get("path_identity_hash")
            or receipt.get("reference_aggregate_ref") != aggregate.get("reference_aggregate_ref")
            or receipt.get("reference_aggregate_hash") != aggregate.get("reference_aggregate_hash")
            or receipt.get("raw_shard_ref") != raw_entry.get("shard_ref")
            or receipt.get("raw_shard_hash") != raw_entry.get("shard_hash")
            or receipt.get("raw_bundle_ref") != raw_entry.get("bundle_ref")
            or receipt.get("raw_bundle_manifest_hash") != raw_entry.get("bundle_manifest_hash")
            or receipt.get("reference_artifact_hash") != raw_entry.get("reference_artifact_hash")
            or receipt.get("raw_aggregate_ref") != aggregate.get("raw_aggregate_ref")
            or receipt.get("raw_aggregate_hash") != aggregate.get("raw_aggregate_hash")
            or receipt.get("path_identity_hash") != raw_entry.get("path_identity_hash")
        ):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_RECEIPT_SOURCE_IDENTITY_INVALID:{unit_id}")
        observation = _load_workspace_json(root, receipt.get("observation_ledger_ref"), field="observation_ledger")
        if (
            observation.get("artifact_hash") != receipt.get("observation_artifact_hash")
            or _checked_body_hash(observation, field="observation_ledger") != receipt.get("observation_artifact_hash")
            or observation.get("schema_version") != "stage3-quadrature-observation-v1"
            or observation.get("unit_id") != unit_id
            or observation.get("execution_evidence_hash") != expected_execution_hash
            or observation.get("reference_artifact_hash") != receipt.get("reference_artifact_hash")
            or observation.get("reference_binding_hash") != expected_binding_hash
            or observation.get("path_identity_hash") != receipt.get("path_identity_hash")
            or observation.get("candidate_rule_names") != list(expected_rules)
            or not isinstance(observation.get("observations"), list)
            or len(observation["observations"]) != len(expected_rules)
            or any(
                not isinstance(item, Mapping) or item.get("unit_id") != unit_id
                for item in observation.get("observations", [])
            )
            or {
                item.get("rule_name")
                for item in observation.get("observations", [])
                if isinstance(item, Mapping)
            }
            != set(expected_rules)
        ):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_OBSERVATION_INVALID:{unit_id}")

        seal_ref = receipt.get("node_cache_seal_ref")
        _load_workspace_json(root, seal_ref, field="node_cache_seal")
        seal_target = root.joinpath(*PurePosixPath(str(seal_ref)).parts)
        try:
            verified_seal = PersistentNodeGradientCache.verify_receipt(seal_target)
        except (OSError, TypeError, ValueError) as error:
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_SEAL_INVALID:{unit_id}") from error
        if (
            verified_seal.get("state") != "SEALED"
            or verified_seal.get("unit_id") != unit_id
            or verified_seal.get("plan_hash") != expected_plan_hash
            or verified_seal.get("downstream_raw_shard_hash") != receipt.get("raw_shard_hash")
            or verified_seal.get("receipt_hash") != receipt.get("node_cache_seal_hash")
        ):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_SEAL_INVALID:{unit_id}")
        _hash(verified_seal.get("receipt_hash"), field=f"streaming_seal[{unit_id}].receipt_hash")

        eviction = _load_workspace_json(root, summary.get("eviction_receipt_ref"), field="streaming_eviction_receipt")
        if set(eviction) != _STREAMING_EVICTION_FIELDS or eviction.get("schema_version") != "stage3-formal-streaming-eviction-receipt-v1":
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_EVICTION_FIELDS_INVALID:{unit_id}")
        eviction_hash = _checked_body_hash(eviction, field="streaming_eviction_receipt")
        if eviction_hash != summary.get("eviction_receipt_hash"):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_EVICTION_HASH_MISMATCH:{unit_id}")
        if (
            eviction.get("unit_id") != unit_id
            or eviction.get("unit_receipt_ref") != summary.get("receipt_ref")
            or eviction.get("unit_receipt_hash") != receipt_hash
            or eviction.get("seal_ref") != seal_ref
            or eviction.get("seal_hash") != receipt.get("node_cache_seal_hash")
            or eviction.get("execution_evidence_hash") != expected_execution_hash
            or eviction.get("reference_binding_hash") != expected_binding_hash
            or eviction.get("state") != "EVICTED"
            or eviction.get("external_request_id") != eviction.get("idempotency_key")
        ):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_EVICTION_IDENTITY_INVALID:{unit_id}")
        eviction_ref = eviction.get("eviction_ref")
        tombstone = _load_workspace_json(root, eviction_ref, field="node_cache_tombstone")
        tombstone_target = root.joinpath(*PurePosixPath(str(eviction_ref)).parts)
        try:
            verified_tombstone = PersistentNodeGradientCache.verify_receipt(tombstone_target)
        except (OSError, TypeError, ValueError) as error:
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_TOMBSTONE_INVALID:{unit_id}") from error
        if (
            verified_tombstone.get("state") != "EVICTED"
            or verified_tombstone.get("unit_id") != unit_id
            or verified_tombstone.get("schema_version") != "stage3-node-cache-evicted-v1"
            or verified_tombstone.get("sealed_receipt_ref") != verified_seal.get("receipt_ref")
            or verified_tombstone.get("sealed_receipt_hash") != verified_seal.get("receipt_hash")
            or verified_tombstone.get("downstream_raw_shard_hash") != receipt.get("raw_shard_hash")
            or verified_tombstone.get("tombstone_hash") != eviction.get("eviction_hash")
        ):
            raise FormalRunRejected(f"STAGE3_G36_STREAMING_TOMBSTONE_INVALID:{unit_id}")
        # Tombstones use ``tombstone_hash`` rather than the ordinary
        # ``artifact_hash`` envelope.  ``verify_receipt`` has already
        # validated that canonical tombstone payload and its hash; checking it
        # as an artifact would reject every genuine EVICTED receipt.


def _load_streaming_coverage(
    root: Path,
    reference: str,
    *,
    expected_units: tuple[str, ...],
    expected_rules: tuple[str, ...],
    expected_execution_hash: str,
    expected_plan_ref: str,
    expected_plan_hash: str,
    expected_index_ref: object,
    expected_index_hash: object,
    expected_binding_hash: str | None = None,
    declared_hash: str | None = None,
) -> tuple[Mapping[str, object], str]:
    """Load a canonical S3.07 aggregate under the publisher's workspace root."""

    ref = _ref(reference, field="streaming_coverage_ref")
    candidate = Path(ref)
    if candidate.is_absolute() or "\\" in ref or ".." in candidate.parts:
        raise FormalRunRejected("STAGE3_G36_STREAMING_COVERAGE_REF_ESCAPE")
    if root.is_symlink():
        raise FormalRunRejected("STAGE3_G36_STREAMING_COVERAGE_SYMLINK")
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise FormalRunRejected("STAGE3_G36_STREAMING_COVERAGE_SYMLINK")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FormalRunRejected("STAGE3_G36_STREAMING_COVERAGE_REF_ESCAPE") from error
    try:
        payload = load_canonical_json(path)
        aggregate = _mapping(payload, field="streaming_coverage")
        _validate_streaming_coverage(
            aggregate,
            required_unit_ids=expected_units,
            required_rule_names=expected_rules,
            execution_evidence_hash=expected_execution_hash,
            formal_plan_ref=expected_plan_ref,
            formal_plan_hash=expected_plan_hash,
            production_unit_index_ref=expected_index_ref,  # type: ignore[arg-type]
            production_unit_index_hash=expected_index_hash,  # type: ignore[arg-type]
            reference_binding_hash=expected_binding_hash,
        )
        binding_hash = str(aggregate["reference_binding_hash"])
        reference_aggregate, reference_entries = _load_reference_aggregate(
            root,
            str(aggregate["reference_aggregate_ref"]),
            declared_hash=aggregate["reference_aggregate_hash"],
            expected_units=expected_units,
            expected_execution_hash=expected_execution_hash,
            expected_binding_hash=binding_hash,
        )
        raw_aggregate, raw_entries = _load_raw_aggregate_for_streaming(
            root,
            str(aggregate["raw_aggregate_ref"]),
            declared_hash=aggregate["raw_aggregate_hash"],
            expected_units=expected_units,
            expected_rules=expected_rules,
            expected_execution_hash=expected_execution_hash,
            expected_binding_hash=binding_hash,
        )
        _load_streaming_receipts(
            root,
            aggregate,
            reference_aggregate=reference_aggregate,
            reference_entries=reference_entries,
            raw_aggregate=raw_aggregate,
            raw_entries=raw_entries,
            expected_units=expected_units,
            expected_rules=expected_rules,
            expected_execution_hash=expected_execution_hash,
            expected_plan_ref=expected_plan_ref,
            expected_plan_hash=expected_plan_hash,
            expected_index_ref=str(expected_index_ref),
            expected_index_hash=str(expected_index_hash),
            expected_binding_hash=binding_hash,
        )
    except (OSError, TypeError, ValueError, FormalRunRejected) as error:
        raise FormalRunRejected("STAGE3_G36_STREAMING_COVERAGE_INVALID") from error
    aggregate_hash = aggregate.get("artifact_hash")
    _hash(aggregate_hash, field="streaming_coverage_hash")
    if declared_hash is not None:
        _hash(declared_hash, field="streaming_coverage_hash")
        if declared_hash != aggregate_hash:
            raise FormalRunRejected("STAGE3_G36_STREAMING_COVERAGE_HASH_MISMATCH")
    return aggregate, str(aggregate_hash)


def _source_refs_from_rows(table: FrozenSourceTable) -> tuple[str, ...]:
    """Collect every row evidence ref that the evaluator can inspect.

    Both names are collected because older Stage 3 observation rows used
    ``source_artifact_refs`` while the formal table contract uses
    ``evidence_refs``.  The evaluator itself remains the authority for whether
    either representation is present and bound.
    """

    refs: list[str] = []
    for index, raw in enumerate(table.rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"frozen_source_table.rows[{index}] 必须是 object")
        for field in ("evidence_refs", "source_artifact_refs"):
            value = raw.get(field)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    if not isinstance(item, str) or not item:
                        raise ValueError(f"frozen_source_table.rows[{index}].{field} 无效")
                    _ref(item, field=f"frozen_source_table.rows[{index}].{field}")
                    refs.append(item)
    return tuple(dict.fromkeys(refs))


def _finite_observation_table(table: FrozenSourceTable) -> bool:
    """Mirror the finite scalar part of the Stage3.08 G3-6 measured fact."""

    fields = (
        "normalized_l1_error",
        "normalized_l2_error",
        "normalized_linf_error",
        "completeness_absolute_residual",
        "completeness_relative_residual",
        "completeness_l1_scaled_residual",
        "active_spearman",
        "cosine_similarity",
        "sign_consistency",
        "layer_quality_tv",
        "module_quality_tv",
        "reference_normalized_l1_error",
    )
    for raw in table.rows:
        if not isinstance(raw, Mapping):
            return False
        for field in fields:
            value = raw.get(field)
            if isinstance(value, Mapping):
                if value.get("defined") is not True or "value" not in value:
                    return False
                value = value["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False
    return True


def _plan_shape(plan: Mapping[str, object], execution: FormalExecutionEvidence) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require the exact evaluator plan identity before publishing anything."""

    declared = plan.get("artifact_hash")
    if not isinstance(declared, str) or canonical_json_hash(
        {key: item for key, item in plan.items() if key != "artifact_hash"}
    ) != declared:
        raise FormalRunRejected("STAGE3_G36_FORMAL_PLAN_HASH_INVALID")
    if (
        plan.get("schema_version") != "stage3-formal-pilot-plan-v1"
        or plan.get("scope") != "formal"
        or plan.get("formal_eligible") is not True
        or plan.get("execution_evidence_hash") != execution.artifact_hash
    ):
        raise FormalRunRejected("STAGE3_G36_FORMAL_PLAN_BINDING_INVALID")
    rules = plan.get("candidate_rules")
    units = plan.get("required_unit_ids")
    if (
        not isinstance(rules, list)
        or not rules
        or any(not isinstance(item, str) or not item for item in rules)
        or len(set(rules)) != len(rules)
        or not isinstance(units, list)
        or not units
        or any(not isinstance(item, str) or not item for item in units)
        or len(set(units)) != len(units)
    ):
        raise FormalRunRejected("STAGE3_G36_FORMAL_PLAN_COVERAGE_INVALID")
    return tuple(str(item) for item in units), tuple(str(item) for item in rules)


def _scope_authority(
    *,
    decision_payload: Mapping[str, object],
    gate_payload: Mapping[str, object],
    decision_ref: str,
    execution: FormalExecutionEvidence,
) -> tuple[Mapping[str, object], GateRecord]:
    validate_stage3_scope_decision(decision_payload)
    try:
        gate = GateRecord.from_mapping(dict(gate_payload))
        validate_stage3_scope_authority(decision_payload, gate, decision_ref=decision_ref)
    except (TypeError, ValueError) as error:
        raise FormalRunRejected("STAGE3_G36_SCOPE_AUTHORITY_INVALID") from error
    if gate.status is not GateStatus.PASS or gate.effective_status() is not GateStatus.PASS:
        raise FormalRunRejected("STAGE3_G36_SCOPE_GATE_NOT_LIVE_PASS")
    execution_gate = next(
        (item for item in execution.prerequisite_gates if item.gate_id == "stage3.G3-0"),
        None,
    )
    if execution_gate is None or execution_gate.artifact_hash != gate.artifact_hash:
        raise FormalRunRejected("STAGE3_G36_SCOPE_GATE_EXECUTION_BINDING_MISMATCH")
    return decision_payload, gate


def _evaluation_sources(
    *,
    frozen_ref: str,
    execution_ref: str,
    plan_ref: str,
    decision_ref: str,
    scope_gate_ref: str,
    execution: FormalExecutionEvidence,
    table: FrozenSourceTable,
    streaming_coverage_ref: str | None = None,
    streaming_formal_plan_ref: str | None = None,
) -> tuple[str, ...]:
    refs: list[str] = [
        frozen_ref,
        execution_ref,
        plan_ref,
        decision_ref,
        scope_gate_ref,
    ]
    if streaming_coverage_ref is not None:
        refs.append(streaming_coverage_ref)
    if (
        streaming_formal_plan_ref is not None
        and streaming_formal_plan_ref != plan_ref
    ):
        refs.append(streaming_formal_plan_ref)
    for gate in execution.prerequisite_gates:
        refs.extend(gate.evidence_refs)
    refs.extend(_source_refs_from_rows(table))
    normalized = tuple(dict.fromkeys(refs))
    for index, ref in enumerate(normalized):
        _ref(ref, field=f"evaluator_source_artifact_refs[{index}]")
    return normalized


def _g36_measured(
    *,
    table: FrozenSourceTable,
    evaluation: Stage3GateEvaluation,
    units: tuple[str, ...],
    rules: tuple[str, ...],
    all_rows_finite: bool,
    streaming_coverage: Mapping[str, object] | None = None,
    streaming_coverage_ref: str | None = None,
    streaming_coverage_hash: str | None = None,
) -> dict[str, JSONValue]:
    # Non-matrix/local-compatible plans predate the streaming ledger.  Their
    # coverage facts remain true by the legacy complete-table contract; matrix
    # plans always pass a strictly validated aggregate here.
    if streaming_coverage is None:
        streaming_metrics: Mapping[str, object] = {
            "complete_reference_raw_observation_coverage": True,
            "streaming_receipt_coverage": True,
            "node_cache_seal_coverage": True,
            "node_cache_eviction_coverage": True,
            "streaming_required_unit_count": len(units),
            "streaming_required_rule_count": len(rules),
        }
    else:
        streaming_metrics = _streaming_coverage_metrics(
            streaming_coverage,
            required_unit_ids=units,
            required_rule_names=rules,
        )
    return {
        "source_table_hash": table.content_hash,
        "evaluation_hash": evaluation.artifact_hash,
        "observation_count": len(table.rows),
        "expected_observation_count": len(units) * len(rules),
        "unit_count": len(units),
        "rule_count": len(rules),
        "all_rows_finite": all_rows_finite,
        "complete_reference_raw_observation_coverage": bool(
            streaming_metrics["complete_reference_raw_observation_coverage"]
        ),
        "streaming_receipt_coverage": bool(streaming_metrics["streaming_receipt_coverage"]),
        "node_cache_seal_coverage": bool(streaming_metrics["node_cache_seal_coverage"]),
        "node_cache_eviction_coverage": bool(
            streaming_metrics["node_cache_eviction_coverage"]
        ),
        "streaming_required_unit_count": int(streaming_metrics["streaming_required_unit_count"]),
        "streaming_required_rule_count": int(streaming_metrics["streaming_required_rule_count"]),
        "streaming_coverage_ref": streaming_coverage_ref,
        "streaming_coverage_hash": streaming_coverage_hash,
        "passing_rules": [
            name
            for name, raw in evaluation.rule_evaluations.items()
            if isinstance(raw, Mapping) and raw.get("passing") is True
        ],
    }


@dataclass(frozen=True, slots=True)
class Stage3G36Publication:
    """Immutable receipt for the two formal artifacts published by G3-6."""

    publication_id: str
    task_id: str
    config_hash: str
    status: str
    formal_eligible: bool
    frozen_source_table_ref: str
    frozen_source_table_hash: str
    formal_plan_ref: str
    formal_plan_hash: str
    execution_evidence_ref: str
    execution_evidence_hash: str
    provenance_ref: str
    provenance_hash: str
    stage3_scope_decision_ref: str
    stage3_scope_decision_hash: str
    stage3_scope_gate_ref: str
    stage3_scope_gate_hash: str
    evaluation_ref: str
    evaluation_hash: str
    g3_6_ref: str
    g3_6_hash: str
    gate_evaluation: Stage3GateEvaluation
    g3_6_gate: GateRecord
    source_artifact_refs: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    schema_version: str = STAGE3_G36_PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_G36_PUBLICATION_SCHEMA:
            raise ValueError("STAGE3_G36_PUBLICATION_SCHEMA_UNSUPPORTED")
        _safe_task_id(self.task_id)
        if not isinstance(self.publication_id, str) or not self.publication_id:
            raise ValueError("STAGE3_G36_PUBLICATION_ID_INVALID")
        _hash(self.config_hash, field="config_hash")
        if self.status not in {"PASS", "BLOCKED"}:
            raise ValueError("STAGE3_G36_PUBLICATION_STATUS_INVALID")
        if type(self.formal_eligible) is not bool or self.formal_eligible != (self.status == "PASS"):
            raise FormalRunRejected("STAGE3_G36_PUBLICATION_ELIGIBILITY_MISMATCH")
        for field, value in (
            ("frozen_source_table_ref", self.frozen_source_table_ref),
            ("formal_plan_ref", self.formal_plan_ref),
            ("execution_evidence_ref", self.execution_evidence_ref),
            ("provenance_ref", self.provenance_ref),
            ("stage3_scope_decision_ref", self.stage3_scope_decision_ref),
            ("stage3_scope_gate_ref", self.stage3_scope_gate_ref),
            ("evaluation_ref", self.evaluation_ref),
            ("g3_6_ref", self.g3_6_ref),
        ):
            _ref(value, field=field)
        for field, value in (
            ("frozen_source_table_hash", self.frozen_source_table_hash),
            ("formal_plan_hash", self.formal_plan_hash),
            ("execution_evidence_hash", self.execution_evidence_hash),
            ("provenance_hash", self.provenance_hash),
            ("stage3_scope_decision_hash", self.stage3_scope_decision_hash),
            ("stage3_scope_gate_hash", self.stage3_scope_gate_hash),
            ("evaluation_hash", self.evaluation_hash),
            ("g3_6_hash", self.g3_6_hash),
        ):
            _hash(value, field=field)
        if not isinstance(self.gate_evaluation, Stage3GateEvaluation) or not isinstance(self.g3_6_gate, GateRecord):
            raise TypeError("STAGE3_G36_PUBLICATION_NESTED_TYPES_INVALID")
        if self.gate_evaluation.artifact_hash != self.evaluation_hash or self.g3_6_gate.artifact_hash != self.g3_6_hash:
            raise ValueError("STAGE3_G36_PUBLICATION_NESTED_HASH_MISMATCH")
        if self.gate_evaluation.status != self.status or self.g3_6_gate.status.value != self.status:
            raise FormalRunRejected("STAGE3_G36_PUBLICATION_STATUS_BINDING_MISMATCH")
        if self.g3_6_gate.gate_id != "stage3.G3-6":
            raise ValueError("STAGE3_G36_PUBLICATION_GATE_ID_INVALID")
        refs = tuple(self.source_artifact_refs)
        if not refs or len(set(refs)) != len(refs) or any(not isinstance(item, str) or not item for item in refs):
            raise ValueError("STAGE3_G36_PUBLICATION_SOURCE_REFS_INVALID")
        object.__setattr__(self, "source_artifact_refs", refs)
        if any(not isinstance(item, str) or not item for item in self.reasons):
            raise ValueError("STAGE3_G36_PUBLICATION_REASONS_INVALID")
        if self.status == "PASS" and self.reasons:
            raise FormalRunRejected("STAGE3_G36_PUBLICATION_PASS_HAS_REASONS")
        required_refs = {
            self.frozen_source_table_ref,
            self.formal_plan_ref,
            self.execution_evidence_ref,
            self.provenance_ref,
            self.stage3_scope_decision_ref,
            self.stage3_scope_gate_ref,
            self.evaluation_ref,
            self.g3_6_ref,
        }
        if not required_refs.issubset(set(refs)):
            raise ValueError("STAGE3_G36_PUBLICATION_OUTPUT_REFS_UNBOUND")

    @property
    def gate_ref(self) -> str:
        """Alias used by consumers that call the G3-6 output simply ``gate``."""

        return self.g3_6_ref

    @property
    def gate_hash(self) -> str:
        return self.g3_6_hash

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "task_id": self.task_id,
            "config_hash": self.config_hash,
            "status": self.status,
            "scope": "formal",
            "formal_eligible": self.formal_eligible,
            "frozen_source_table_ref": self.frozen_source_table_ref,
            "frozen_source_table_hash": self.frozen_source_table_hash,
            "formal_plan_ref": self.formal_plan_ref,
            "formal_plan_hash": self.formal_plan_hash,
            "execution_evidence_ref": self.execution_evidence_ref,
            "execution_evidence_hash": self.execution_evidence_hash,
            "provenance_ref": self.provenance_ref,
            "provenance_hash": self.provenance_hash,
            "stage3_scope_decision_ref": self.stage3_scope_decision_ref,
            "stage3_scope_decision_hash": self.stage3_scope_decision_hash,
            "stage3_scope_gate_ref": self.stage3_scope_gate_ref,
            "stage3_scope_gate_hash": self.stage3_scope_gate_hash,
            "evaluation_ref": self.evaluation_ref,
            "evaluation_hash": self.evaluation_hash,
            "g3_6_ref": self.g3_6_ref,
            "g3_6_hash": self.g3_6_hash,
            "gate_evaluation": self.gate_evaluation.to_dict(),
            "g3_6_gate": self.g3_6_gate.to_dict(),
            "source_artifact_refs": list(self.source_artifact_refs),
            "reasons": list(self.reasons),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage3G36Publication":
        # The wire fields are intentionally enumerated so a future dataclass
        # implementation detail cannot silently become part of the contract.
        required = {
            "schema_version", "publication_id", "task_id", "config_hash", "status", "scope", "formal_eligible",
            "frozen_source_table_ref", "frozen_source_table_hash", "formal_plan_ref", "formal_plan_hash",
            "execution_evidence_ref", "execution_evidence_hash", "provenance_ref", "provenance_hash",
            "stage3_scope_decision_ref", "stage3_scope_decision_hash", "stage3_scope_gate_ref", "stage3_scope_gate_hash",
            "evaluation_ref", "evaluation_hash", "g3_6_ref", "g3_6_hash", "gate_evaluation", "g3_6_gate",
            "source_artifact_refs", "reasons", "artifact_hash",
        }
        if set(value) != required:
            raise ValueError("STAGE3_G36_PUBLICATION_FIELDS_MISMATCH")
        supplied = value["artifact_hash"]
        _hash(supplied, field="artifact_hash")
        if value["scope"] != "formal":
            raise ValueError("STAGE3_G36_PUBLICATION_SCOPE_INVALID")
        if canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) != supplied:
            raise ValueError("STAGE3_G36_PUBLICATION_HASH_MISMATCH")
        raw_eval = value["gate_evaluation"]
        raw_gate = value["g3_6_gate"]
        if not isinstance(raw_eval, Mapping) or not isinstance(raw_gate, Mapping):
            raise TypeError("STAGE3_G36_PUBLICATION_NESTED_OBJECTS_REQUIRED")
        arrays = ("source_artifact_refs", "reasons")
        if any(not isinstance(value[name], list) for name in arrays):
            raise TypeError("STAGE3_G36_PUBLICATION_ARRAYS_INVALID")
        return cls(
            publication_id=value["publication_id"],  # type: ignore[arg-type]
            task_id=value["task_id"],  # type: ignore[arg-type]
            config_hash=value["config_hash"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            frozen_source_table_ref=value["frozen_source_table_ref"],  # type: ignore[arg-type]
            frozen_source_table_hash=value["frozen_source_table_hash"],  # type: ignore[arg-type]
            formal_plan_ref=value["formal_plan_ref"],  # type: ignore[arg-type]
            formal_plan_hash=value["formal_plan_hash"],  # type: ignore[arg-type]
            execution_evidence_ref=value["execution_evidence_ref"],  # type: ignore[arg-type]
            execution_evidence_hash=value["execution_evidence_hash"],  # type: ignore[arg-type]
            provenance_ref=value["provenance_ref"],  # type: ignore[arg-type]
            provenance_hash=value["provenance_hash"],  # type: ignore[arg-type]
            stage3_scope_decision_ref=value["stage3_scope_decision_ref"],  # type: ignore[arg-type]
            stage3_scope_decision_hash=value["stage3_scope_decision_hash"],  # type: ignore[arg-type]
            stage3_scope_gate_ref=value["stage3_scope_gate_ref"],  # type: ignore[arg-type]
            stage3_scope_gate_hash=value["stage3_scope_gate_hash"],  # type: ignore[arg-type]
            evaluation_ref=value["evaluation_ref"],  # type: ignore[arg-type]
            evaluation_hash=value["evaluation_hash"],  # type: ignore[arg-type]
            g3_6_ref=value["g3_6_ref"],  # type: ignore[arg-type]
            g3_6_hash=value["g3_6_hash"],  # type: ignore[arg-type]
            gate_evaluation=Stage3GateEvaluation.from_mapping(dict(raw_eval)),
            g3_6_gate=GateRecord.from_mapping(dict(raw_gate)),
            source_artifact_refs=tuple(value["source_artifact_refs"]),  # type: ignore[arg-type]
            reasons=tuple(value["reasons"]),  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


class Stage3G36Publisher:
    """Load formal commits, evaluate the frozen table, then publish G3-6."""

    def publish(
        self,
        *,
        workspace_root: str | Path,
        output_dir: str,
        config_hash: str,
        frozen_source_table_ref: str,
        provenance_ref: str,
        formal_plan_ref: str,
        execution_evidence_ref: str,
        stage3_scope_decision_ref: str,
        stage3_scope_gate_ref: str,
        streaming_coverage_ref: str | None = None,
        streaming_coverage_hash: str | None = None,
        streaming_formal_plan_ref: str | None = None,
        publication_id: str | None = None,
        task_id: str = STAGE3_G36_TASK_ID,
    ) -> Stage3G36Publication:
        root = Path(workspace_root).resolve()
        _hash(config_hash, field="config_hash")
        _safe_task_id(task_id)

        # All six inputs are complete formal commits.  This is intentionally
        # done before constructing the output store, so malformed input cannot
        # leave a misleading partial publisher directory behind.
        table_artifact = _load_formal_commit(
            root, frozen_source_table_ref, field="frozen_source_table", kinds=_TABLE_KINDS
        )
        provenance_artifact = _load_formal_commit(
            root, provenance_ref, field="provenance", kinds=_PROVENANCE_KINDS
        )
        plan_artifact = _load_formal_commit(
            root, formal_plan_ref, field="formal_plan", kinds=_PLAN_KINDS
        )
        execution_artifact = _load_formal_commit(
            root, execution_evidence_ref, field="execution_evidence", kinds=_EXECUTION_KINDS
        )
        scope_decision_artifact = _load_formal_commit(
            root, stage3_scope_decision_ref, field="scope_decision", kinds=_SCOPE_KINDS
        )
        scope_gate_artifact = _load_formal_commit(
            root, stage3_scope_gate_ref, field="scope_gate", kinds=frozenset({"gate_record"})
        )
        # S3.08, its promoted plan and the completed provenance describe this
        # publication attempt and therefore share the requested config hash.
        # Execution and G3-0 are immutable upstream authorities produced by
        # different canonical tasks; their exact refs and semantic hashes are
        # validated below instead of falsely requiring one producer config.
        drifted_configs = tuple(
            item.identity.commit_ref
            for item in (
                table_artifact,
                plan_artifact,
                provenance_artifact,
            )
            if item.identity.config_hash != config_hash
        )
        if drifted_configs:
            raise FormalRunRejected(
                "STAGE3_G36_INPUT_CONFIG_HASH_MISMATCH:"
                + ",".join(drifted_configs)
            )

        try:
            table = FrozenSourceTable.from_mapping(dict(table_artifact.payload))
            execution = FormalExecutionEvidence.from_mapping(dict(execution_artifact.payload))
            execution.require_for_stage(3)
            provenance = ProvenanceRecord.from_mapping(dict(provenance_artifact.payload))
            if (
                provenance.config_hash != config_hash
                or provenance.status is not ProvenanceStatus.COMPLETED
                or provenance.scope != "formal"
                or provenance.formal_eligible is not True
                or provenance.worktree_clean is not True
            ):
                raise FormalRunRejected("STAGE3_G36_PROVENANCE_NOT_COMPLETED_CLEAN_FORMAL")
            plan = _mapping(plan_artifact.payload, field="formal_plan")
            units, rules = _plan_shape(plan, execution)
            decision = _mapping(scope_decision_artifact.payload, field="stage3_scope_decision")
            scope_gate = _mapping(scope_gate_artifact.payload, field="stage3_scope_gate")
            decision, scope_gate_record = _scope_authority(
                decision_payload=decision,
                gate_payload=scope_gate,
                decision_ref=stage3_scope_decision_ref,
                execution=execution,
            )
        except (OSError, TypeError, ValueError, FormalRunRejected) as error:
            raise FormalRunRejected(f"STAGE3_G36_INPUTS_INVALID:{error}") from error

        matrix_plan = plan.get("plan_kind") == "matrix"
        streaming_coverage: Mapping[str, object] | None = None
        resolved_streaming_hash: str | None = None
        resolved_streaming_plan_ref = _resolve_streaming_plan_ref(
            root,
            plan_artifact=plan_artifact,
            formal_plan_ref=formal_plan_ref,
            streaming_formal_plan_ref=streaming_formal_plan_ref,
        )
        if matrix_plan:
            if streaming_coverage_ref is None:
                raise FormalRunRejected("STAGE3_G36_STREAMING_COVERAGE_REQUIRED")
            streaming_coverage, resolved_streaming_hash = _load_streaming_coverage(
                root,
                streaming_coverage_ref,
                expected_units=units,
                expected_rules=rules,
                expected_execution_hash=execution.artifact_hash,
                expected_plan_ref=resolved_streaming_plan_ref,
                expected_plan_hash=str(plan["artifact_hash"]),
                expected_index_ref=plan.get("production_unit_index_ref"),
                expected_index_hash=plan.get("production_unit_index_hash"),
                declared_hash=streaming_coverage_hash,
            )
        elif streaming_coverage_ref is not None:
            # Explicit non-matrix input remains supported, but it must still
            # be a valid aggregate rather than an unbound opaque ref.
            streaming_coverage, resolved_streaming_hash = _load_streaming_coverage(
                root,
                streaming_coverage_ref,
                expected_units=units,
                expected_rules=rules,
                expected_execution_hash=execution.artifact_hash,
                expected_plan_ref=resolved_streaming_plan_ref,
                expected_plan_hash=str(plan["artifact_hash"]),
                expected_index_ref=plan.get("production_unit_index_ref"),
                expected_index_hash=plan.get("production_unit_index_hash"),
                declared_hash=streaming_coverage_hash,
            )
        elif streaming_coverage_hash is not None:
            raise FormalRunRejected("STAGE3_G36_STREAMING_COVERAGE_HASH_WITHOUT_REF")

        evaluator_sources = _evaluation_sources(
            frozen_ref=frozen_source_table_ref,
            execution_ref=execution_evidence_ref,
            plan_ref=formal_plan_ref,
            decision_ref=stage3_scope_decision_ref,
            scope_gate_ref=stage3_scope_gate_ref,
            streaming_coverage_ref=streaming_coverage_ref,
            streaming_formal_plan_ref=resolved_streaming_plan_ref,
            execution=execution,
            table=table,
        )
        # This is the cycle breaker.  The provenance commit existed first and
        # must cover every evaluator source, but it must not list itself (nor
        # the future evaluation/Gate refs) as an artifact it attested.
        if provenance_ref in evaluator_sources or provenance_ref in provenance.artifact_refs:
            raise FormalRunRejected("STAGE3_G36_PROVENANCE_SELF_BINDING")
        missing_provenance = sorted(set(evaluator_sources) - set(provenance.artifact_refs))
        if missing_provenance:
            raise FormalRunRejected(
                "STAGE3_G36_PROVENANCE_SOURCE_COVERAGE_MISSING:" + ",".join(missing_provenance)
            )
        if frozen_source_table_ref not in provenance.artifact_refs:
            raise FormalRunRejected("STAGE3_G36_PROVENANCE_TABLE_REF_MISSING")

        evaluation = Stage3GateEvaluator().evaluate(
            evaluation_id=publication_id or f"stage3-g36-{table.content_hash[:16]}",
            execution=execution,
            observations=tuple(thaw_json_value(dict(row)) for row in table.rows),
            formal_plan=plan,
            formal_plan_ref=formal_plan_ref,
            required_rule_names=rules,
            required_unit_ids=units,
            thresholds=_mapping(plan["thresholds"], field="formal_plan.thresholds"),
            provenance=provenance,
            source_artifact_refs=evaluator_sources,
            stage3_scope_decision=decision,
            stage3_scope_gate=scope_gate_record,
            stage3_scope_decision_ref=stage3_scope_decision_ref,
            stage3_scope_gate_ref=stage3_scope_gate_ref,
            streaming_coverage=streaming_coverage,
        )
        all_rows_finite = _finite_observation_table(table)
        gate_status = GateStatus.PASS if evaluation.formal_eligible and all_rows_finite else GateStatus.BLOCKED
        reasons = tuple(evaluation.reasons)
        if not all_rows_finite:
            reasons = (*reasons, "FORMAL_OBSERVATION_TABLE_NONFINITE")
        if gate_status is GateStatus.BLOCKED and not reasons:
            reasons = ("STAGE3_G36_EVALUATION_BLOCKED",)
        checked_at = provenance.ended_at or provenance.started_at
        store = TaskArtifactStore(root, output_dir)

        # Publish evaluator output first.  Its commit ref is then immutable
        # evidence in the G3-6 Gate; provenance is not changed.
        evaluation_artifact = store.publish(
            task_id=task_id,
            artifact_kind=STAGE3_G36_EVALUATION_ARTIFACT_KIND,
            config_hash=config_hash,
            run_intent="formal",
            payload=evaluation.to_dict(),  # type: ignore[arg-type]
            formal_eligible=True,
            source_refs=tuple(dict.fromkeys((*evaluator_sources, provenance_ref))),
        )
        measured = _g36_measured(
            table=table,
            evaluation=evaluation,
            units=units,
            rules=rules,
            all_rows_finite=all_rows_finite,
            streaming_coverage=streaming_coverage,
            streaming_coverage_ref=streaming_coverage_ref,
            streaming_coverage_hash=resolved_streaming_hash,
        )
        gate = GateRecord(
            gate_id="stage3.G3-6",
            stage=3,
            status=gate_status,
            checked_at=checked_at,
            measured=measured,
            threshold={
                "complete_unit_rule_coverage": True,
                "all_rows_finite": True,
                "independent_evaluation_status": "PASS",
                "complete_reference_raw_observation_coverage": True,
                "streaming_receipt_coverage": True,
                "node_cache_seal_coverage": True,
                "node_cache_eviction_coverage": True,
                "streaming_required_unit_count": len(units),
                "streaming_required_rule_count": len(rules),
            },
            evidence_refs=tuple(
                dict.fromkeys(
                    (
                        frozen_source_table_ref,
                        evaluation_artifact.commit_ref,
                        provenance_ref,
                        formal_plan_ref,
                        *((streaming_coverage_ref,) if streaming_coverage_ref is not None else ()),
                    )
                )
            ),
            reasons=reasons,
        )
        gate_artifact = store.publish(
            task_id=task_id,
            artifact_kind=STAGE3_G36_GATE_ARTIFACT_KIND,
            config_hash=config_hash,
            run_intent="formal",
            payload=gate.to_dict(),  # type: ignore[arg-type]
            formal_eligible=True,
            source_refs=tuple(
                dict.fromkeys((*evaluator_sources, provenance_ref, evaluation_artifact.commit_ref))
            ),
        )
        source_refs = tuple(
            dict.fromkeys(
                (
                    *evaluator_sources,
                    provenance_ref,
                    evaluation_artifact.commit_ref,
                    gate_artifact.commit_ref,
                )
            )
        )
        publication = Stage3G36Publication(
            publication_id=publication_id or f"stage3-g36-{table.content_hash[:16]}",
            task_id=task_id,
            config_hash=config_hash,
            status=gate_status.value,
            formal_eligible=gate_status is GateStatus.PASS,
            frozen_source_table_ref=frozen_source_table_ref,
            frozen_source_table_hash=table.content_hash,
            formal_plan_ref=formal_plan_ref,
            formal_plan_hash=str(plan["artifact_hash"]),
            execution_evidence_ref=execution_evidence_ref,
            execution_evidence_hash=execution.artifact_hash,
            provenance_ref=provenance_ref,
            provenance_hash=provenance.artifact_hash,
            stage3_scope_decision_ref=stage3_scope_decision_ref,
            stage3_scope_decision_hash=str(decision["artifact_hash"]),
            stage3_scope_gate_ref=stage3_scope_gate_ref,
            stage3_scope_gate_hash=scope_gate_record.artifact_hash,
            evaluation_ref=evaluation_artifact.commit_ref,
            evaluation_hash=evaluation.artifact_hash,
            g3_6_ref=gate_artifact.commit_ref,
            g3_6_hash=gate.artifact_hash,
            gate_evaluation=evaluation,
            g3_6_gate=gate,
            source_artifact_refs=source_refs,
            reasons=tuple(reasons),
        )
        # Persist the complete two-output receipt after both immutable commits
        # exist.  The receipt does not contain its own ref, so this final step
        # remains acyclic while giving downstream consumers one bundle to
        # reload and validate.
        store.publish(
            task_id=task_id,
            artifact_kind="g36_publication",
            config_hash=config_hash,
            run_intent="formal",
            payload=publication.to_dict(),
            formal_eligible=True,
            source_refs=source_refs,
        )
        return publication


def publish_stage3_g36(**kwargs: object) -> Stage3G36Publication:
    """Functional entry point for the Stage3.08→G3-6 handoff."""

    return Stage3G36Publisher().publish(**kwargs)  # type: ignore[arg-type]


publish_stage3_g3_6 = publish_stage3_g36


__all__ = [
    "STAGE3_G36_EVALUATION_ARTIFACT_KIND",
    "STAGE3_G36_GATE_ARTIFACT_KIND",
    "STAGE3_G36_PUBLICATION_SCHEMA",
    "STAGE3_G36_TASK_ID",
    "Stage3G36Publication",
    "Stage3G36Publisher",
    "publish_stage3_g36",
    "publish_stage3_g3_6",
]
