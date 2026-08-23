"""Canonical contracts shared by the Stage 2.04 producer and G2.3 consumer.

The formal producer and evaluator must agree on the wire-level contract, not on
duplicated ad-hoc checks.  This module deliberately contains no scientific
estimand calculation; it validates the identities and replay boundaries around
those calculations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path, PurePosixPath
import random
from typing import Any

from ..contracts.jsonio import canonical_json_bytes, canonical_json_hash


EXTERNAL_MANIFEST_SCHEMAS: Mapping[str, str] = {
    "asset_resolution": "stage2-task-asset-resolution-v1",
    "six_cell_manifest": "stage2-s204-six-cell-manifest-v1",
    "resolved_config": "resolved-config-v2",
    "checkpoint_manifest": "checkpoint-manifest-v1",
    "model_manifest": "model-manifest-v1",
    "data_manifest": "data-manifest-v1",
    "tokenizer_manifest": "tokenizer-manifest-v1",
    "parameter_registry": "stage2-parameter-registry-artifact-v1",
    "preregistration": "stage2-preregistration-v1",
    "reference_sizing_plan": "stage2-reference-sizing-plan-v1",
}

WEIGHTING_CONTRACT_FIELDS = (
    "statistical_unit",
    "weight_unit",
    "sampling_design",
    "weights_exogenous",
    "common_mean_assumption",
)


def validate_weighting_contract(value: object, *, field: str = "weighting_assumptions") -> dict[str, object]:
    """Validate the formal weighted-U assumptions and return a plain mapping."""

    if not isinstance(value, Mapping) or set(value) != set(WEIGHTING_CONTRACT_FIELDS):
        raise ValueError(f"{field}:FORMAL_WEIGHTING_CONTRACT_REQUIRED")
    result = {name: value[name] for name in WEIGHTING_CONTRACT_FIELDS}
    for name in WEIGHTING_CONTRACT_FIELDS[:3]:
        if not isinstance(result[name], str) or not result[name].strip():
            raise ValueError(f"{field}.{name}:NON_EMPTY_STRING_REQUIRED")
    if type(result["weights_exogenous"]) is not bool or type(result["common_mean_assumption"]) is not bool:
        raise ValueError(f"{field}:BOOLEAN_ASSUMPTIONS_REQUIRED")
    if result["weights_exogenous"] is not True or result["common_mean_assumption"] is not True:
        raise ValueError(f"{field}:WEIGHTED_U_ASSUMPTIONS_NOT_DECLARED")
    return result


def validate_sizing_plan_contract(
    plan: object,
    *,
    selected_sample_count: int | None = None,
    field: str = "sizing_plan",
) -> tuple[int, ...]:
    """Validate the fixed G2.3 sizing ladder and selected node identity."""

    if not isinstance(plan, Mapping) or plan.get("schema_version") != "stage2-reference-sizing-plan-v1":
        raise ValueError(f"{field}:SCHEMA_REQUIRED")
    if plan.get("required_consecutive") != 1:
        raise ValueError(f"{field}:REQUIRED_CONSECUTIVE_MUST_BE_ONE")
    raw = plan.get("candidate_sample_counts")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"{field}:CANDIDATE_COUNTS_REQUIRED")
    counts: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{field}:CANDIDATE_COUNT_INVALID")
        counts.append(item)
    block_size = plan.get("block_size")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"{field}:BLOCK_SIZE_INVALID")
    if any(count % block_size for count in counts):
        raise ValueError(f"{field}:BLOCK_ALIGNMENT_INVALID")
    if any(right != 2 * left for left, right in zip(counts, counts[1:])):
        raise ValueError(f"{field}:ADJACENT_DOUBLING_REQUIRED")
    if selected_sample_count is not None:
        if isinstance(selected_sample_count, bool) or not isinstance(selected_sample_count, int) or selected_sample_count <= 0:
            raise ValueError(f"{field}:SELECTED_NODE_INVALID")
        if selected_sample_count not in counts:
            raise ValueError(f"{field}:SELECTED_NODE_NOT_IN_CANDIDATES")
    return tuple(counts)


def _jsonable_state(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable_state(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_state(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable_state(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("GENERATOR_STATE_NOT_CANONICAL_JSON")


def _tuple_state(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuple_state(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _tuple_state(item) for key, item in value.items()}
    return value


def generator_state_digest(algorithm_version: str, state: object) -> str:
    """Hash the canonical bytes of the actual serialized generator state."""

    return hashlib.sha256(
        canonical_json_bytes({"algorithm_version": algorithm_version, "state": _jsonable_state(state)})
    ).hexdigest()


def generator_boundary(
    sampling: object,
    stream: str,
    count: int,
) -> dict[str, object]:
    """Capture actual initial/final MT19937 states for one frozen prefix."""

    if not hasattr(sampling, "_stream_seed") or count < 0:
        raise ValueError("GENERATOR_BOUNDARY_ARGUMENT_INVALID")
    algorithm_version = str(getattr(sampling, "algorithm_version"))
    rng = random.Random(sampling._stream_seed(stream))  # type: ignore[attr-defined]
    before = _jsonable_state(rng.getstate())
    universe = getattr(sampling, "universe")
    size = len(getattr(universe, "sample_ids"))
    for _ in range(count):
        rng.randrange(size)
    after = _jsonable_state(rng.getstate())
    return {
        "algorithm_version": algorithm_version,
        "stream": stream,
        "count": count,
        "state_before": before,
        "state_after": after,
        "state_before_sha256": generator_state_digest(algorithm_version, before),
        "state_after_sha256": generator_state_digest(algorithm_version, after),
    }


def validate_generator_boundary(
    value: object,
    *,
    sampling: object,
    stream: str,
    count: int,
    field: str,
) -> dict[str, object]:
    """Verify actual serialized states, replay, and both state digests."""

    if not isinstance(value, Mapping) or set(value) != {
        "algorithm_version", "stream", "count", "state_before", "state_after",
        "state_before_sha256", "state_after_sha256",
    }:
        raise ValueError(f"{field}:ACTUAL_GENERATOR_STATE_REQUIRED")
    expected = generator_boundary(sampling, stream, count)
    candidate = dict(value)
    if candidate != expected:
        raise ValueError(f"{field}:GENERATOR_STATE_REPLAY_MISMATCH")
    if generator_state_digest(str(candidate["algorithm_version"]), candidate["state_before"]) != candidate["state_before_sha256"]:
        raise ValueError(f"{field}:STATE_BEFORE_DIGEST_MISMATCH")
    if generator_state_digest(str(candidate["algorithm_version"]), candidate["state_after"]) != candidate["state_after_sha256"]:
        raise ValueError(f"{field}:STATE_AFTER_DIGEST_MISMATCH")
    return candidate


def boundary_digest(value: object, *, field: str = "rng_state") -> str:
    """Digest one or more complete boundaries, never a draw/hash summary."""

    if isinstance(value, Mapping):
        return canonical_json_hash(dict(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return canonical_json_hash(list(value))
    raise ValueError(f"{field}:BOUNDARIES_REQUIRED")


def validate_resume_prefix(
    previous_a: Sequence[Mapping[str, object]] | None,
    previous_b: Sequence[Mapping[str, object]] | None,
    current_a: Sequence[Mapping[str, object]],
    current_b: Sequence[Mapping[str, object]],
    *,
    field: str,
) -> None:
    """Require each resume commit's shard refs to extend the prior prefix."""

    if previous_a is not None and list(current_a[: len(previous_a)]) != list(previous_a):
        raise ValueError(f"{field}:SHARD_PREFIX_DRIFT_A")
    if previous_b is not None and list(current_b[: len(previous_b)]) != list(previous_b):
        raise ValueError(f"{field}:SHARD_PREFIX_DRIFT_B")


def unique_identity_values(rows: Sequence[Mapping[str, object]], fields: Sequence[str], *, field: str) -> None:
    """Require non-empty unique identities for a fixed six-cell matrix."""

    for name in fields:
        values = [row.get(name) for row in rows]
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{field}.{name}:IDENTITY_REQUIRED")
        if len(values) != len(set(values)):
            raise ValueError(f"{field}.{name}:DUPLICATE_IDENTITY")


def source_manifest_for_refs(root: str | Path, refs: Sequence[str]) -> list[dict[str, object]]:
    """Return actual size/SHA records for every non-empty source ref."""

    if not refs or len(set(refs)) != len(refs):
        raise ValueError("SOURCE_REFS_NONEMPTY_UNIQUE_REQUIRED")
    root_path = Path(root)
    result: list[dict[str, object]] = []
    for ref in refs:
        if not isinstance(ref, str) or not ref or "\\" in ref:
            raise ValueError("SOURCE_REF_PATH_INVALID")
        logical = PurePosixPath(ref)
        if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
            raise ValueError("SOURCE_REF_PATH_INVALID")
        current = root_path
        for part in logical.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("SOURCE_REF_SYMLINK_FORBIDDEN")
        data = current.read_bytes()
        result.append({
            "path": logical.as_posix(),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    return result


def validate_external_manifest(
    loaded: object,
    root: str | Path,
    *,
    expected_kind: str,
    declared_sources: object | None = None,
) -> list[dict[str, object]]:
    """Validate formal schema and bind TaskArtifact source refs to real bytes."""

    identity = getattr(loaded, "identity", None)
    payload = getattr(loaded, "payload", None)
    source_refs = getattr(loaded, "source_refs", None)
    if identity is None or not isinstance(payload, Mapping) or not isinstance(source_refs, tuple):
        raise ValueError("EXTERNAL_MANIFEST_LOADED_ARTIFACT_REQUIRED")
    if getattr(loaded, "run_intent", None) != "formal" or getattr(identity, "formal_eligible", None) is not True:
        raise ValueError("EXTERNAL_MANIFEST_FORMAL_REQUIRED")
    if getattr(identity, "artifact_kind", None) != expected_kind:
        raise ValueError("EXTERNAL_MANIFEST_KIND_MISMATCH")
    expected_schema = EXTERNAL_MANIFEST_SCHEMAS.get(expected_kind)
    if expected_schema is None or payload.get("schema_version") != expected_schema:
        raise ValueError("EXTERNAL_MANIFEST_SCHEMA_MISMATCH")
    actual = source_manifest_for_refs(root, source_refs)
    if declared_sources is not None and declared_sources != actual:
        raise ValueError("EXTERNAL_MANIFEST_SOURCE_DRIFT")
    return actual


__all__ = [
    "EXTERNAL_MANIFEST_SCHEMAS",
    "WEIGHTING_CONTRACT_FIELDS",
    "boundary_digest",
    "generator_boundary",
    "generator_state_digest",
    "source_manifest_for_refs",
    "unique_identity_values",
    "validate_external_manifest",
    "validate_generator_boundary",
    "validate_resume_prefix",
    "validate_sizing_plan_contract",
    "validate_weighting_contract",
]
