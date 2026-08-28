"""Hash-bound raw Stage 3.07 path-result shard storage.

Formal path vectors are intentionally kept out of task-commit JSON.  This
module stores one immutable TensorBundle per production unit and keeps only a
small, strict JSON index beside it.  The index and bundle are independently
verified when S3.10 reloads the data, so a missing unit or a changed tensor
cannot silently turn into a report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path, PurePosixPath
import math

import numpy as np

from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json, write_canonical_json
from ..runtime.tensor_bundle import TensorBundle, load_tensor_bundle, publish_tensor_bundle


RAW_SHARD_SCHEMA = "stage3-formal-raw-shard-v1"
RAW_AGGREGATE_SCHEMA = "stage3-formal-raw-aggregate-v1"
VECTOR_DERIVATION_SCHEMA = "stage3-formal-vector-view-derivation-v1"
VECTOR_DERIVATION_CONTRACT = (
    "positive=max(signed,0);negative_mass=max(-signed,0);"
    "absolute=positive+negative_mass;all_fp64"
)
VECTOR_DERIVATION_HASH = canonical_json_hash(
    {
        "schema_version": VECTOR_DERIVATION_SCHEMA,
        "contract": VECTOR_DERIVATION_CONTRACT,
    }
)
_HASH_RE = set("0123456789abcdef")


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HASH_RE for c in value):
        raise ValueError(f"STAGE3_RAW_{field.upper()}_HASH_INVALID")
    return value


def _relative_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"STAGE3_RAW_{field.upper()}_REF_INVALID")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"STAGE3_RAW_{field.upper()}_REF_ESCAPE")
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"STAGE3_RAW_{field.upper()}_SYMLINK")
    target = current.resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"STAGE3_RAW_{field.upper()}_REF_ESCAPE") from error
    return target


def _relative_ref(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _tree_digest(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            digest.update(b"map\0")
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"seq\0")
            for child in item:
                visit(child)
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy\0" + str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
            return
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"torch\0" + str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        digest.update(type(item).__name__.encode("ascii"))
        digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _vector_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"STAGE3_RAW_{field.upper()}_MISSING")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"STAGE3_RAW_{field.upper()}_KEY_INVALID")
    for name, tensor in value.items():
        try:
            import torch
        except ImportError:
            torch = None
        valid = isinstance(tensor, np.ndarray) or (torch is not None and isinstance(tensor, torch.Tensor))
        if not valid:
            raise ValueError(f"STAGE3_RAW_{field.upper()}_TENSOR_INVALID:{name}")
        if isinstance(tensor, np.ndarray) and tensor.dtype != np.float64:
            raise ValueError(f"STAGE3_RAW_{field.upper()}_FP64_REQUIRED:{name}")
        if torch is not None and isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float64:
            raise ValueError(f"STAGE3_RAW_{field.upper()}_FP64_REQUIRED:{name}")
        array = tensor.detach().cpu().numpy() if torch is not None and isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"STAGE3_RAW_{field.upper()}_TENSOR_NONFINITE:{name}")
    return value


def derive_candidate_views(signed: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Derive the three loss-decomposition views exactly from FP64 signed."""

    signed = _vector_mapping(signed, field="signed")
    try:
        import torch
    except ImportError:
        torch = None
    derived: dict[str, dict[str, object]] = {
        "positive": {},
        "negative_mass": {},
        "absolute": {},
    }
    for name, value in signed.items():
        if torch is not None and isinstance(value, torch.Tensor):
            positive = torch.clamp(value, min=0)
            negative = torch.clamp(-value, min=0)
            absolute = positive + negative
        else:
            array = np.asarray(value)
            positive = np.maximum(array, 0)
            negative = np.maximum(-array, 0)
            absolute = positive + negative
        derived["positive"][name] = positive
        derived["negative_mass"][name] = negative
        derived["absolute"][name] = absolute
    return derived


def _tensor_equal(left: object, right: object) -> bool:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _bundle_state(
    reference_signed: Mapping[str, object],
    candidate_states: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    reference = {"signed": dict(_vector_mapping(reference_signed, field="reference_signed"))}
    candidates: dict[str, object] = {}
    for rule_name, state in sorted(candidate_states.items()):
        if not isinstance(rule_name, str) or not rule_name:
            raise ValueError("STAGE3_RAW_RULE_NAME_INVALID")
        if not isinstance(state, Mapping):
            raise ValueError("STAGE3_RAW_CANDIDATE_STATE_INVALID")
        signed = _vector_mapping(state.get("signed"), field="candidate_signed")
        item: dict[str, object] = {"signed": dict(signed)}
        expected_views = derive_candidate_views(signed)
        for key in ("positive", "negative_mass", "absolute"):
            supplied = state.get(key)
            if supplied is None:
                continue
            values = _vector_mapping(supplied, field=f"candidate_{key}")
            if set(values) != set(signed):
                raise ValueError("STAGE3_RAW_CANDIDATE_COORDINATE_SET_MISMATCH")
            for name in signed:
                if np.asarray(values[name]).size != np.asarray(signed[name]).size:
                    raise ValueError("STAGE3_RAW_CANDIDATE_VECTOR_SHAPE_MISMATCH")
                if not _tensor_equal(values[name], expected_views[key][name]):
                    raise ValueError(f"STAGE3_RAW_DERIVED_VIEW_MISMATCH:{key}:{name}")
            # Derived views are validated at the boundary but deliberately
            # omitted from the persisted state to avoid 3x redundant storage.
        candidates[rule_name] = item
    if not candidates:
        raise ValueError("STAGE3_RAW_CANDIDATES_MISSING")
    return {
        "vector_derivation": {
            "schema_version": VECTOR_DERIVATION_SCHEMA,
            "contract": VECTOR_DERIVATION_CONTRACT,
            "contract_hash": VECTOR_DERIVATION_HASH,
        },
        "reference": reference,
        "candidates": candidates,
    }


def _bundle_at(
    *, root: Path, bundle_path: Path, state: Mapping[str, object]
) -> TensorBundle:
    normalized = _bundle_state(
        state["reference"]["signed"],  # type: ignore[index]
        {
            name: value  # type: ignore[misc]
            for name, value in state["candidates"].items()  # type: ignore[union-attr]
        },
    )
    expected_digest = _tree_digest(normalized)
    if bundle_path.exists():
        restored, bundle = load_tensor_bundle(bundle_path)
        if _tree_digest(restored) != expected_digest:
            raise ValueError("STAGE3_RAW_EXISTING_BUNDLE_DRIFT")
        return bundle
    return publish_tensor_bundle(bundle_path, normalized)


def persist_raw_unit_shard(
    *,
    root: Path,
    ledger_root: Path,
    unit_id: str,
    required_unit_ids: Sequence[str],
    execution_evidence_hash: str,
    reference_binding_hash: str,
    path_identity_hash: str,
    reference_artifact_hash: str,
    reference_signed: Mapping[str, object],
    candidate_states: Mapping[str, Mapping[str, Mapping[str, object]]],
    rule_summaries: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, JSONValue], str]:
    """Persist one unit's tensors and its small hash-bound JSON shard."""

    for field, value in (
        ("execution_evidence", execution_evidence_hash),
        ("reference_binding", reference_binding_hash),
        ("path_identity", path_identity_hash),
        ("reference_artifact", reference_artifact_hash),
    ):
        _hash(value, field)
    if unit_id not in required_unit_ids or len(set(required_unit_ids)) != len(required_unit_ids):
        raise ValueError("STAGE3_RAW_REQUIRED_UNIT_SET_INVALID")
    state = _bundle_state(reference_signed, candidate_states)
    names = sorted(candidate_states)
    if names != sorted(rule_summaries) or any(not isinstance(value, Mapping) for value in rule_summaries.values()):
        raise ValueError("STAGE3_RAW_RULE_SUMMARY_SET_MISMATCH")
    ledger_root = ledger_root.resolve()
    bundle_path = ledger_root / "bundles" / f"{unit_id}.bundle"
    bundle = _bundle_at(root=root, bundle_path=bundle_path, state=state)
    summaries = {name: dict(rule_summaries[name]) for name in names}
    body: dict[str, JSONValue] = {
        "schema_version": RAW_SHARD_SCHEMA,
        "unit_id": unit_id,
        "required_unit_ids": list(required_unit_ids),
        "candidate_rule_names": names,
        "execution_evidence_hash": execution_evidence_hash,
        "reference_binding_hash": reference_binding_hash,
        "path_identity_hash": path_identity_hash,
        "reference_artifact_hash": reference_artifact_hash,
        "vector_derivation_schema": VECTOR_DERIVATION_SCHEMA,
        "vector_derivation_contract_hash": VECTOR_DERIVATION_HASH,
        "reference_identity_hash": canonical_json_hash({
            "unit_id": unit_id,
            "path_identity_hash": path_identity_hash,
            "execution_evidence_hash": execution_evidence_hash,
            "reference_binding_hash": reference_binding_hash,
            "reference_artifact_hash": reference_artifact_hash,
        }),
        "bundle_ref": _relative_ref(root, bundle_path),
        "bundle_manifest_hash": bundle.manifest_sha256,
        "rule_summaries": summaries,  # type: ignore[assignment]
    }
    body["artifact_hash"] = canonical_json_hash(body)
    shard_path = ledger_root / f"{unit_id}.json"
    if shard_path.exists():
        existing = load_canonical_json(shard_path)
        if existing != body:
            raise ValueError("STAGE3_RAW_SHARD_IMMUTABLE_CONFLICT")
    else:
        write_canonical_json(shard_path, body)
    return body, _relative_ref(root, shard_path)


def _load_shard(
    *, root: Path, shard_ref: object, expected: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object], TensorBundle]:
    path = _relative_path(root, shard_ref, field="shard")
    raw = load_canonical_json(path)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != RAW_SHARD_SCHEMA:
        raise ValueError("STAGE3_RAW_SHARD_SCHEMA_INVALID")
    supplied = _hash(raw.get("artifact_hash"), "shard_artifact")
    if canonical_json_hash({key: item for key, item in raw.items() if key != "artifact_hash"}) != supplied:
        raise ValueError("STAGE3_RAW_SHARD_HASH_MISMATCH")
    required = {
        "schema_version", "unit_id", "required_unit_ids", "candidate_rule_names",
        "execution_evidence_hash", "reference_binding_hash", "path_identity_hash",
        "reference_artifact_hash", "vector_derivation_schema",
        "vector_derivation_contract_hash", "reference_identity_hash", "bundle_ref",
        "bundle_manifest_hash", "rule_summaries", "artifact_hash",
    }
    if set(raw) != required:
        raise ValueError("STAGE3_RAW_SHARD_FIELDS_MISMATCH")
    for field in ("execution_evidence_hash", "reference_binding_hash", "path_identity_hash", "reference_artifact_hash", "bundle_manifest_hash"):
        _hash(raw.get(field), f"shard_{field}")
    if (
        raw.get("vector_derivation_schema") != VECTOR_DERIVATION_SCHEMA
        or raw.get("vector_derivation_contract_hash") != VECTOR_DERIVATION_HASH
    ):
        raise ValueError("STAGE3_RAW_DERIVATION_CONTRACT_MISMATCH")
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"STAGE3_RAW_SHARD_BINDING_MISMATCH:{key}")
    unit = raw.get("unit_id")
    names = raw.get("candidate_rule_names")
    required_units = raw.get("required_unit_ids")
    if not isinstance(unit, str) or not isinstance(names, list) or sorted(names) != names or len(set(names)) != len(names):
        raise ValueError("STAGE3_RAW_SHARD_RULE_SET_INVALID")
    if not isinstance(required_units, list) or any(not isinstance(item, str) for item in required_units):
        raise ValueError("STAGE3_RAW_SHARD_UNIT_SET_INVALID")
    summaries = raw.get("rule_summaries")
    if not isinstance(summaries, Mapping) or set(summaries) != set(names):
        raise ValueError("STAGE3_RAW_SHARD_SUMMARY_SET_INVALID")
    for name in names:
        summary = summaries[name]
        if not isinstance(summary, Mapping):
            raise ValueError("STAGE3_RAW_SHARD_SUMMARY_INVALID")
        if summary.get("path_identity_hash") != raw.get("path_identity_hash"):
            raise ValueError("STAGE3_RAW_SHARD_SUMMARY_IDENTITY_MISMATCH")
        alphas = summary.get("node_alphas")
        losses = summary.get("node_losses")
        if (
            not isinstance(alphas, list)
            or not isinstance(losses, list)
            or not alphas
            or len(alphas) != len(losses)
            or any(item is None or not math.isfinite(float(item)) for item in alphas)
            or any(item is None or not math.isfinite(float(item)) for item in losses)
        ):
            raise ValueError("STAGE3_RAW_SHARD_SUMMARY_NODE_LOSSES_INVALID")
    expected_reference_identity = canonical_json_hash({
        "unit_id": raw["unit_id"],
        "path_identity_hash": raw["path_identity_hash"],
        "execution_evidence_hash": raw["execution_evidence_hash"],
        "reference_binding_hash": raw["reference_binding_hash"],
        "reference_artifact_hash": raw["reference_artifact_hash"],
    })
    if raw.get("reference_identity_hash") != expected_reference_identity:
        raise ValueError("STAGE3_RAW_SHARD_REFERENCE_IDENTITY_MISMATCH")
    bundle_path = _relative_path(root, raw.get("bundle_ref"), field="bundle")
    state, bundle = load_tensor_bundle(bundle_path)
    if bundle.manifest_sha256 != raw.get("bundle_manifest_hash"):
        raise ValueError("STAGE3_RAW_SHARD_BUNDLE_HASH_MISMATCH")
    if (
        not isinstance(state, Mapping)
        or state.get("vector_derivation") != {
            "schema_version": VECTOR_DERIVATION_SCHEMA,
            "contract": VECTOR_DERIVATION_CONTRACT,
            "contract_hash": VECTOR_DERIVATION_HASH,
        }
        or not isinstance(state.get("reference"), Mapping)
        or not isinstance(state.get("candidates"), Mapping)
    ):
        raise ValueError("STAGE3_RAW_SHARD_BUNDLE_STATE_INVALID")
    reference = state["reference"]
    candidates = state["candidates"]
    if set(reference) != {"signed"}:
        raise ValueError("STAGE3_RAW_SHARD_BUNDLE_REFERENCE_FIELDS_INVALID")
    _vector_mapping(reference.get("signed"), field="reference_signed")  # type: ignore[union-attr]
    if set(candidates) != set(names):
        raise ValueError("STAGE3_RAW_SHARD_BUNDLE_RULE_SET_MISMATCH")
    reference_parameters = set(reference["signed"])  # type: ignore[index]
    reference_sizes = {
        name: np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value).size
        for name, value in reference["signed"].items()  # type: ignore[index]
    }
    for name in names:
        candidate = candidates[name]
        if not isinstance(candidate, Mapping):
            raise ValueError("STAGE3_RAW_SHARD_BUNDLE_CANDIDATE_INVALID")
        signed = _vector_mapping(candidate.get("signed"), field="candidate_signed")
        if set(candidate) != {"signed"}:
            raise ValueError("STAGE3_RAW_SHARD_BUNDLE_REDUNDANT_VIEWS_PRESENT")
        if set(signed) != reference_parameters or any(
            np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value).size
            != reference_sizes[parameter]
            for parameter, value in signed.items()
        ):
            raise ValueError("STAGE3_RAW_SHARD_BUNDLE_COORDINATE_SET_MISMATCH")
        expected_views = derive_candidate_views(signed)
        for mass_name, mass_map in expected_views.items():
            if set(mass_map) != reference_parameters or any(
                np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value).size
                != reference_sizes[parameter]
                for parameter, value in mass_map.items()
            ):
                raise ValueError("STAGE3_RAW_SHARD_BUNDLE_MASS_SHAPE_MISMATCH")
    return raw, state, bundle


def publish_raw_aggregate(
    *, root: Path, ledger_root: Path, required_unit_ids: Sequence[str],
    execution_evidence_hash: str, reference_binding_hash: str,
    candidate_rule_names: Sequence[str],
) -> tuple[Mapping[str, JSONValue], str]:
    """Snapshot all shards visible so far; callers enforce complete coverage."""

    entries: dict[str, JSONValue] = {}
    complete: list[str] = []
    missing: list[str] = []
    for unit_id in required_unit_ids:
        shard_path = ledger_root / f"{unit_id}.json"
        if not shard_path.exists():
            missing.append(unit_id)
            continue
        shard, _state, _bundle = _load_shard(
            root=root,
            shard_ref=_relative_ref(root, shard_path),
            expected={
                "unit_id": unit_id,
                "required_unit_ids": list(required_unit_ids),
                "candidate_rule_names": sorted(candidate_rule_names),
                "execution_evidence_hash": execution_evidence_hash,
                "reference_binding_hash": reference_binding_hash,
            },
        )
        entries[unit_id] = {
            "shard_ref": _relative_ref(root, shard_path),
            "shard_hash": str(shard["artifact_hash"]),
            "bundle_ref": str(shard["bundle_ref"]),
            "bundle_manifest_hash": str(shard["bundle_manifest_hash"]),
            "path_identity_hash": str(shard["path_identity_hash"]),
            "reference_artifact_hash": str(shard["reference_artifact_hash"]),
            "reference_identity_hash": str(shard["reference_identity_hash"]),
        }
        complete.append(unit_id)
    body: dict[str, JSONValue] = {
        "schema_version": RAW_AGGREGATE_SCHEMA,
        "execution_evidence_hash": execution_evidence_hash,
        "reference_binding_hash": reference_binding_hash,
        "required_unit_ids": list(required_unit_ids),
        "candidate_rule_names": sorted(candidate_rule_names),
        "vector_derivation_schema": VECTOR_DERIVATION_SCHEMA,
        "vector_derivation_contract_hash": VECTOR_DERIVATION_HASH,
        "complete_unit_ids": complete,
        "missing_unit_ids": missing,
        "unit_shards": entries,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    path = ledger_root / f"aggregate-{body['artifact_hash']}.json"
    if path.exists():
        existing = load_canonical_json(path)
        if existing != body:
            raise ValueError("STAGE3_RAW_AGGREGATE_IMMUTABLE_CONFLICT")
    else:
        write_canonical_json(path, body)
    return body, _relative_ref(root, path)


def load_raw_aggregate(
    *, root: Path, aggregate_ref: object, aggregate_hash: object, require_complete: bool = True
) -> tuple[Mapping[str, object], dict[str, tuple[Mapping[str, object], Mapping[str, object], TensorBundle]]]:
    """Strictly reload an aggregate and every referenced shard/bundle."""

    path = _relative_path(root, aggregate_ref, field="aggregate")
    raw = load_canonical_json(path)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != RAW_AGGREGATE_SCHEMA:
        raise ValueError("STAGE3_RAW_AGGREGATE_SCHEMA_INVALID")
    supplied = _hash(raw.get("artifact_hash"), "aggregate_artifact")
    if supplied != aggregate_hash or canonical_json_hash({key: item for key, item in raw.items() if key != "artifact_hash"}) != supplied:
        raise ValueError("STAGE3_RAW_AGGREGATE_HASH_MISMATCH")
    required = {"schema_version", "execution_evidence_hash", "reference_binding_hash", "required_unit_ids", "candidate_rule_names", "vector_derivation_schema", "vector_derivation_contract_hash", "complete_unit_ids", "missing_unit_ids", "unit_shards", "artifact_hash"}
    if set(raw) != required:
        raise ValueError("STAGE3_RAW_AGGREGATE_FIELDS_MISMATCH")
    required_units = raw.get("required_unit_ids")
    names = raw.get("candidate_rule_names")
    complete = raw.get("complete_unit_ids")
    missing = raw.get("missing_unit_ids")
    entries = raw.get("unit_shards")
    if not isinstance(required_units, list) or not isinstance(names, list) or not isinstance(complete, list) or not isinstance(missing, list) or not isinstance(entries, Mapping):
        raise ValueError("STAGE3_RAW_AGGREGATE_INDEX_INVALID")
    if (
        raw.get("vector_derivation_schema") != VECTOR_DERIVATION_SCHEMA
        or raw.get("vector_derivation_contract_hash") != VECTOR_DERIVATION_HASH
    ):
        raise ValueError("STAGE3_RAW_AGGREGATE_DERIVATION_CONTRACT_MISMATCH")
    if sorted(names) != names or len(set(names)) != len(names) or complete + missing != required_units or set(entries) != set(complete):
        raise ValueError("STAGE3_RAW_AGGREGATE_COVERAGE_INVALID")
    if require_complete and (missing or complete != required_units):
        raise ValueError("STAGE3_RAW_AGGREGATE_INCOMPLETE")
    _hash(raw.get("execution_evidence_hash"), "aggregate_execution")
    _hash(raw.get("reference_binding_hash"), "aggregate_binding")
    loaded: dict[str, tuple[Mapping[str, object], Mapping[str, object], TensorBundle]] = {}
    for unit_id in complete:
        entry = entries.get(unit_id)
        if not isinstance(entry, Mapping):
            raise ValueError("STAGE3_RAW_AGGREGATE_ENTRY_INVALID")
        shard_ref = entry.get("shard_ref")
        expected = {
            "unit_id": unit_id,
            "required_unit_ids": required_units,
            "candidate_rule_names": names,
            "execution_evidence_hash": raw["execution_evidence_hash"],
            "reference_binding_hash": raw["reference_binding_hash"],
        }
        shard, state, bundle = _load_shard(root=root, shard_ref=shard_ref, expected=expected)
        for field in ("shard_hash", "bundle_ref", "bundle_manifest_hash", "path_identity_hash", "reference_artifact_hash", "reference_identity_hash"):
            if entry.get(field) != shard.get(field.replace("shard_hash", "artifact_hash")):
                raise ValueError(f"STAGE3_RAW_AGGREGATE_ENTRY_HASH_MISMATCH:{unit_id}:{field}")
        loaded[unit_id] = (shard, state, bundle)
    return raw, loaded


__all__ = [
    "RAW_AGGREGATE_SCHEMA", "RAW_SHARD_SCHEMA", "VECTOR_DERIVATION_SCHEMA",
    "VECTOR_DERIVATION_CONTRACT", "VECTOR_DERIVATION_HASH",
    "derive_candidate_views", "load_raw_aggregate", "persist_raw_unit_shard",
    "publish_raw_aggregate",
]
