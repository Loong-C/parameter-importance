"""Strict DATA_ROOT-relative layout for Stage 0 G3 asset evidence.

The experiment configuration never embeds server absolute paths.  This
artifact binds each logical asset to a POSIX path relative to the approved
``DATA_ROOT`` and to the exact qualification artifact that authorizes use.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Final

from .assets import validate_asset_path
from .contracts.jsonio import canonical_json_hash, ensure_json_object, loads_strict_json


SCHEMA_VERSION: Final = "stage0-g3-asset-layout-v1"
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT: Final = re.compile(r"^[0-9a-f]{40}$")
_GATES: Final = (
    "stage0.G3-S1",
    "stage0.G3-S2",
    "stage0.G3-S4",
    "stage0.G3-S5",
    "stage0.G3-S6",
)
_KINDS: Final = frozenset({"model", "tokenizer", "pile", "glue_raw", "glue_derived"})
_TOP_FIELDS: Final = frozenset(
    {
        "schema_version",
        "artifact_hash",
        "created_at",
        "generator_git_commit",
        "requirements_ref",
        "requirements_sha256",
        "workspace_semantics",
        "entries",
    }
)
_ENTRY_FIELDS: Final = frozenset(
    {
        "logical_name",
        "kind",
        "requirement_name",
        "manifest_ref",
        "asset_root_ref",
        "qualification_ref",
        "gate_ids",
    }
)


class AssetLayoutError(ValueError):
    """Raised when the formal G3 logical layout is incomplete or unsafe."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AssetLayoutError(f"{field} must be an object with string keys")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: frozenset[str], *, field: str) -> None:
    if set(value) != fields:
        raise AssetLayoutError(
            f"{field} fields mismatch: missing={sorted(fields-set(value))}, "
            f"extra={sorted(set(value)-fields)}"
        )


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AssetLayoutError(f"{field} must be normalized non-empty text")
    if any(ord(character) < 32 for character in value):
        raise AssetLayoutError(f"{field} contains a control character")
    return value


def _digest(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if _SHA256.fullmatch(text) is None:
        raise AssetLayoutError(f"{field} must be a lowercase SHA-256")
    return text


def layout_artifact_hash(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("artifact_hash", None)
    return canonical_json_hash(payload)


def _expected_gate_bindings(requirements: Mapping[str, Any]) -> dict[tuple[str, str], list[str]]:
    matrix = _mapping(requirements.get("gate_matrix"), field="requirements.gate_matrix")
    expected: dict[tuple[str, str], list[str]] = {}
    for gate_id in _GATES:
        gate = _mapping(matrix.get(gate_id), field=f"requirements.gate_matrix.{gate_id}")
        for model_name in gate.get("model_names", []):
            expected.setdefault(("model", model_name), []).append(gate_id)
        if gate.get("require_tokenizer") is True:
            expected.setdefault(("tokenizer", "pythia-tokenizer"), []).append(gate_id)
        if gate.get("pile_intervals"):
            expected.setdefault(("pile", "pile"), []).append(gate_id)
        for task in gate.get("glue_tasks", []):
            expected.setdefault(("glue_raw", task), []).append(gate_id)
            if gate.get("require_glue_derived") is True:
                expected.setdefault(("glue_derived", task), []).append(gate_id)
    return expected


def validate_stage0_asset_layout(
    value: Mapping[str, Any],
    *,
    requirements: Mapping[str, Any] | None = None,
) -> None:
    layout = _mapping(value, field="asset layout")
    _exact(layout, _TOP_FIELDS, field="asset layout")
    if layout["schema_version"] != SCHEMA_VERSION:
        raise AssetLayoutError("unsupported asset layout schema_version")
    if layout["workspace_semantics"] != "paths_are_posix_relative_to_DATA_ROOT":
        raise AssetLayoutError("asset paths must be explicitly relative to DATA_ROOT")
    if _digest(layout["artifact_hash"], field="artifact_hash") != layout_artifact_hash(layout):
        raise AssetLayoutError("asset layout artifact_hash mismatch")
    created_at = _text(layout["created_at"], field="created_at")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise AssetLayoutError("created_at must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise AssetLayoutError("created_at must include a timezone")
    commit = _text(layout["generator_git_commit"], field="generator_git_commit")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise AssetLayoutError("generator_git_commit must be a Git commit")
    validate_asset_path(_text(layout["requirements_ref"], field="requirements_ref"))
    requirements_hash = _digest(layout["requirements_sha256"], field="requirements_sha256")

    entries = layout["entries"]
    if not isinstance(entries, list) or not entries:
        raise AssetLayoutError("entries must be a non-empty array")
    identities: set[tuple[str, str]] = set()
    logical_names: set[str] = set()
    manifest_refs: set[str] = set()
    qualification_refs: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(entries):
        entry = _mapping(raw, field=f"entries[{index}]")
        _exact(entry, _ENTRY_FIELDS, field=f"entries[{index}]")
        logical_name = _text(entry["logical_name"], field=f"entries[{index}].logical_name")
        kind = _text(entry["kind"], field=f"entries[{index}].kind")
        if kind not in _KINDS:
            raise AssetLayoutError(f"entries[{index}].kind is unsupported")
        requirement_name = _text(
            entry["requirement_name"], field=f"entries[{index}].requirement_name"
        )
        for path_field in ("manifest_ref", "asset_root_ref", "qualification_ref"):
            validate_asset_path(_text(entry[path_field], field=f"entries[{index}].{path_field}"))
        gates = entry["gate_ids"]
        if (
            not isinstance(gates, list)
            or not gates
            or any(gate not in _GATES for gate in gates)
            or gates != sorted(set(gates), key=_GATES.index)
        ):
            raise AssetLayoutError(f"entries[{index}].gate_ids are invalid or unordered")
        identity = (kind, requirement_name)
        if identity in identities or logical_name in logical_names:
            raise AssetLayoutError("asset layout identities and logical names must be unique")
        if entry["manifest_ref"] in manifest_refs or entry["qualification_ref"] in qualification_refs:
            raise AssetLayoutError("manifest and qualification refs must be unique")
        identities.add(identity)
        logical_names.add(logical_name)
        manifest_refs.add(entry["manifest_ref"])
        qualification_refs.add(entry["qualification_ref"])
        normalized.append(entry)

    if requirements is not None:
        declared_requirements_hash = requirements.get("artifact_hash")
        if requirements_hash != declared_requirements_hash:
            raise AssetLayoutError("layout requirements_sha256 does not bind the requirements")
        expected = _expected_gate_bindings(requirements)
        observed = {
            (entry["kind"], entry["requirement_name"]): entry["gate_ids"]
            for entry in normalized
        }
        if observed != expected:
            raise AssetLayoutError("asset layout does not exactly implement the G3 gate matrix")


def load_stage0_asset_layout(
    path: str | Path,
    *,
    requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = ensure_json_object(loads_strict_json(Path(path).read_bytes()), field="asset layout")
    result = dict(value)
    validate_stage0_asset_layout(result, requirements=requirements)
    return result


__all__ = [
    "AssetLayoutError",
    "SCHEMA_VERSION",
    "layout_artifact_hash",
    "load_stage0_asset_layout",
    "validate_stage0_asset_layout",
]
