"""Compile real Stage 3 trajectory receipts into canonical probe plans.

This module is deliberately a small, fail-closed materializer.  It consumes
three immutable inputs:

* a ``stage3-trajectory-completion-receipt-v1``;
* a pre-registered ``stage3-probe-allocation-v1`` (probe entries contain only
  role, ID, sample IDs and metadata); and
* a frozen content-source manifest whose sample files and loss-contract are
  read and hashed by this process.

No content or hash is accepted from the allocation.  ``content_hash`` is the
canonical hash of the ordered sample IDs and the SHA-256 of each resolved
sample file.  ``loss_contract_hash`` is computed from the canonical loss
contract document.  The output is one canonical ``stage3-probe-plan-v1`` per
endpoint.  A missing, stale, fixture, synthetic, overlapping or otherwise
unverifiable input aborts the complete publication.

The source file format is intentionally explicit so a real provider can be
adapted without hidden environment state::

    {
      "schema_version": "stage3-probe-plan-materialization-source-v1",
      "scope": "pilot" | "formal",
      "trajectory_receipt_ref": "...",
      "probe_allocation_ref": "...",
      "content_source_ref": "...",
      "formal_execution_ref": "...",
      "output_dir": "...",
      "artifact_hash": "..."
    }

``content_source_ref`` points to ``stage3-frozen-probe-content-source-v1``::

    {
      "schema_version": "stage3-frozen-probe-content-source-v1",
      "resolver_id": "real-frozen-resolver/...",
      "resolver_state_ref": "...",
      "samples": [{"sample_id": "...", "content_ref": "..."}],
      "loss_contract_ref": "...",
      "effective_weight_unit": "effective_target_tokens",
      "artifact_hash": "..."
    }

The resolver state and all content/contract references are explicit, allowing
Pythia mmap or another real frozen resolver to publish a stable manifest
before this command is run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.experiments.stage3_production_plan import (
    _Source,
    _load_endpoint,
)
from param_importance_nlp.experiments.stage3_trajectory import Stage3TrajectoryReceipt


MATERIALIZATION_SCHEMA = "stage3-probe-plan-materialization-source-v1"
ALLOCATION_SCHEMA = "stage3-probe-allocation-v1"
CONTENT_SOURCE_SCHEMA = "stage3-frozen-probe-content-source-v1"
PLAN_SCHEMA = "stage3-probe-plan-v1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_FORBIDDEN_RE = re.compile(r"(?:fixture|synthetic)", re.IGNORECASE)


class ProbePlanMaterializationError(ValueError):
    """A real endpoint/probe input failed a publication precondition."""


def _fail(code: str, detail: object | None = None) -> ProbePlanMaterializationError:
    return ProbePlanMaterializationError(code if detail is None else f"{code}:{detail}")


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _fail("HASH_INVALID", field)
    return value


def _no_forbidden(value: object, field: str = "source") -> None:
    if isinstance(value, str):
        if _FORBIDDEN_RE.search(value):
            raise _fail("FIXTURE_OR_SYNTHETIC_REJECTED", field)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and _FORBIDDEN_RE.search(key):
                raise _fail("FIXTURE_OR_SYNTHETIC_REJECTED", f"{field}.{key}")
            _no_forbidden(child, field)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _no_forbidden(child, field)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_ref(value: object, *, roots: Sequence[Path], field: str, require_file: bool = True) -> Path:
    # A Path is accepted only for the top-level Python API source argument;
    # all refs carried by artifacts remain POSIX strings and are checked below.
    if isinstance(value, Path):
        raw = value
    elif isinstance(value, str) and value and (Path(value).is_absolute() or "\\" not in value):
        raw = Path(value)
    else:
        raise _fail("REFERENCE_INVALID", field)
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw.resolve()]
    else:
        logical = PurePosixPath(value)
        if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
            raise _fail("REFERENCE_PATH_ESCAPE", field)
        candidates = [root.joinpath(*logical.parts).resolve() for root in roots]
    candidates = list(dict.fromkeys(candidates))
    if any(not any(_within(candidate, root) for root in roots) for candidate in candidates):
        raise _fail("REFERENCE_OUTSIDE_ALLOWED_ROOT", field)
    existing = [candidate for candidate in candidates if candidate.exists()]
    if len(existing) > 1:
        raise _fail("REFERENCE_AMBIGUOUS", field)
    if not existing:
        if not require_file:
            return candidates[0]
        raise _fail("REFERENCE_NOT_FOUND", field)
    if require_file and not existing[0].is_file():
        raise _fail("REFERENCE_NOT_FILE", field)
    return existing[0]


def _load_mapping(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("CANONICAL_JSON_INVALID", field) from error
    if not isinstance(value, Mapping):
        raise _fail("OBJECT_REQUIRED", field)
    _no_forbidden(value, field)
    return value


def _check_artifact_hash(value: Mapping[str, Any], field: str) -> None:
    supplied = _hash(value.get("artifact_hash"), f"{field}.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if supplied != canonical_json_hash(body):
        raise _fail("ARTIFACT_HASH_MISMATCH", field)


def _sample_key(value: object, field: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or (isinstance(value, str) and not value):
        raise _fail("SAMPLE_ID_INVALID", field)
    return value


def _validate_source(source: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "scope", "trajectory_receipt_ref", "probe_allocation_ref",
        "content_source_ref", "formal_execution_ref", "output_dir", "artifact_hash",
    }
    if set(source) != expected or source.get("schema_version") != MATERIALIZATION_SCHEMA:
        raise _fail("MATERIALIZATION_SOURCE_FIELDS_MISMATCH")
    if source.get("scope") not in {"pilot", "formal"}:
        raise _fail("SCOPE_INVALID")
    _no_forbidden(source, "materialization_source")
    for field in ("trajectory_receipt_ref", "probe_allocation_ref", "content_source_ref", "formal_execution_ref", "output_dir"):
        if not isinstance(source.get(field), str) or not source[field]:
            raise _fail("REFERENCE_INVALID", field)
    _check_artifact_hash(source, "materialization_source")


def _validate_allocation(value: Mapping[str, Any], *, scope: str) -> dict[str, list[Mapping[str, Any]]]:
    expected = {"schema_version", "scope", "allocations", "artifact_hash"}
    if set(value) != expected or value.get("schema_version") != ALLOCATION_SCHEMA:
        raise _fail("ALLOCATION_FIELDS_MISMATCH")
    if value.get("scope") != scope:
        raise _fail("ALLOCATION_SCOPE_MISMATCH")
    _no_forbidden(value, "allocation")
    _check_artifact_hash(value, "allocation")
    raw = value.get("allocations")
    if not isinstance(raw, list) or not raw:
        raise _fail("ALLOCATION_EMPTY")
    result: dict[str, list[Mapping[str, Any]]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {"endpoint_commit_ref", "probes"}:
            raise _fail("ALLOCATION_ENTRY_FIELDS_MISMATCH", index)
        endpoint_ref = item.get("endpoint_commit_ref")
        if not isinstance(endpoint_ref, str) or not endpoint_ref or endpoint_ref in result:
            raise _fail("ALLOCATION_ENDPOINT_REF_INVALID", index)
        probes = item.get("probes")
        if not isinstance(probes, list):
            raise _fail("ALLOCATION_PROBES_INVALID", endpoint_ref)
        result[endpoint_ref] = []
        seen_ids: set[str] = set()
        for probe_index, probe in enumerate(probes):
            expected_probe = {"role", "probe_id", "sample_ids", "metadata"}
            if not isinstance(probe, Mapping) or set(probe) != expected_probe:
                raise _fail("ALLOCATION_PROBE_FIELDS_MISMATCH", f"{endpoint_ref}:{probe_index}")
            role = probe.get("role")
            if role != scope:
                raise _fail("ALLOCATION_ROLE_SCOPE_MISMATCH", f"{endpoint_ref}:{probe_index}")
            probe_id = probe.get("probe_id")
            if not isinstance(probe_id, str) or _ID_RE.fullmatch(probe_id) is None or probe_id in seen_ids:
                raise _fail("ALLOCATION_PROBE_ID_INVALID", f"{endpoint_ref}:{probe_index}")
            samples = probe.get("sample_ids")
            if not isinstance(samples, list) or not samples:
                raise _fail("ALLOCATION_SAMPLE_IDS_INVALID", probe_id)
            normalized = [_sample_key(sample, f"{probe_id}.sample_ids") for sample in samples]
            if len(set(normalized)) != len(normalized):
                raise _fail("ALLOCATION_SAMPLE_IDS_DUPLICATE", probe_id)
            if not isinstance(probe.get("metadata"), Mapping):
                raise _fail("ALLOCATION_METADATA_INVALID", probe_id)
            if set(probe["metadata"]).intersection({
                "materializer_resolver_id", "materializer_resolver_state_digest",
                "materializer_content_source_hash",
            }):
                raise _fail("ALLOCATION_METADATA_RESERVED", probe_id)
            _no_forbidden(probe.get("metadata"), f"{probe_id}.metadata")
            seen_ids.add(probe_id)
            result[endpoint_ref].append(probe)
    return result


def _validate_content_source(value: Mapping[str, Any], *, roots: Sequence[Path]) -> tuple[dict[str | int, Path], str, str, str, str, str]:
    expected = {
        "schema_version", "resolver_id", "resolver_state_ref", "samples",
        "loss_contract_ref", "effective_weight_unit", "artifact_hash",
    }
    if set(value) != expected or value.get("schema_version") != CONTENT_SOURCE_SCHEMA:
        raise _fail("CONTENT_SOURCE_FIELDS_MISMATCH")
    _check_artifact_hash(value, "content_source")
    resolver_id = value.get("resolver_id")
    if not isinstance(resolver_id, str) or not resolver_id or _FORBIDDEN_RE.search(resolver_id):
        raise _fail("RESOLVER_ID_INVALID")
    state_path = _resolve_ref(value.get("resolver_state_ref"), roots=roots, field="resolver_state_ref")
    state = _load_mapping(state_path, "resolver_state")
    if state.get("resolver_id") is not None and state.get("resolver_id") != resolver_id:
        raise _fail("RESOLVER_STATE_ID_MISMATCH")
    resolver_state_digest = canonical_json_hash(state)
    loss_path = _resolve_ref(value.get("loss_contract_ref"), roots=roots, field="loss_contract_ref")
    loss_contract = _load_mapping(loss_path, "loss_contract")
    loss_hash = canonical_json_hash(loss_contract)
    unit = value.get("effective_weight_unit")
    if not isinstance(unit, str) or not unit or _FORBIDDEN_RE.search(unit):
        raise _fail("EFFECTIVE_WEIGHT_UNIT_INVALID")
    raw_samples = value.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise _fail("CONTENT_SOURCE_SAMPLES_EMPTY")
    files: dict[str | int, Path] = {}
    for index, item in enumerate(raw_samples):
        if not isinstance(item, Mapping) or set(item) != {"sample_id", "content_ref"}:
            raise _fail("CONTENT_SOURCE_SAMPLE_FIELDS_MISMATCH", index)
        sample_id = _sample_key(item.get("sample_id"), f"content_source.samples[{index}]")
        if sample_id in files:
            raise _fail("CONTENT_SOURCE_SAMPLE_DUPLICATE", sample_id)
        files[sample_id] = _resolve_ref(item.get("content_ref"), roots=roots, field=f"content_ref:{sample_id}")
    return files, resolver_id, resolver_state_digest, str(value["artifact_hash"]), loss_hash, unit


def _content_hash(sample_ids: Sequence[str | int], files: Mapping[str | int, Path]) -> str:
    rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        path = files.get(sample_id)
        if path is None:
            raise _fail("PROBE_CONTENT_MISSING", sample_id)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise _fail("PROBE_CONTENT_READ_FAILED", sample_id) from error
        rows.append({"sample_id": sample_id, "content_sha256": digest})
    return canonical_json_hash(rows)


def _immutable_write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _load_mapping(path, str(path))
        if dict(existing) != dict(value):
            raise _fail("IMMUTABLE_OUTPUT_CONFLICT", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(path, value)


def _check_immutable(path: Path, value: Mapping[str, Any]) -> None:
    """Preflight existing outputs so a conflict cannot leave a prefix."""
    if not path.exists():
        return
    existing = _load_mapping(path, str(path))
    if dict(existing) != dict(value):
        raise _fail("IMMUTABLE_OUTPUT_CONFLICT", path)


def materialize_probe_plans(
    source: Mapping[str, Any] | str | Path,
    *,
    workspace_root: str | Path,
    data_root: str | Path,
) -> tuple[Path, ...]:
    """Validate all real inputs and publish one canonical plan per endpoint."""

    workspace = Path(workspace_root).resolve()
    data = Path(data_root).resolve()
    roots = (data, workspace)
    if isinstance(source, Mapping):
        source_value = source
    else:
        source_path = _resolve_ref(source, roots=roots, field="materialization_source")
        source_value = _load_mapping(source_path, "materialization_source")
    _validate_source(source_value)
    scope = str(source_value["scope"])
    receipt_path = _resolve_ref(source_value["trajectory_receipt_ref"], roots=roots, field="trajectory_receipt_ref")
    receipt_value = _load_mapping(receipt_path, "trajectory_receipt")
    try:
        receipt = Stage3TrajectoryReceipt.from_mapping(receipt_value)
    except (TypeError, ValueError) as error:
        raise _fail("TRAJECTORY_RECEIPT_INVALID") from error
    if receipt.purpose_scope != scope or receipt.formal_eligible is not (scope == "formal"):
        raise _fail("TRAJECTORY_SCOPE_MISMATCH")
    evidence_ref = str(source_value["formal_execution_ref"])
    if receipt.formal_execution_ref != evidence_ref:
        raise _fail("FORMAL_EXECUTION_REF_DRIFT")
    evidence_path = _resolve_ref(evidence_ref, roots=roots, field="formal_execution_ref")
    try:
        evidence = FormalExecutionEvidence.from_mapping(_load_mapping(evidence_path, "formal_execution"))
        evidence.require_for_stage(3)
    except (TypeError, ValueError) as error:
        raise _fail("FORMAL_EXECUTION_EVIDENCE_INVALID") from error
    if evidence.run_intent != "formal":
        raise _fail("FORMAL_EXECUTION_EVIDENCE_NOT_FORMAL")
    evidence_gates = {gate.gate_id: gate for gate in evidence.prerequisite_gates}
    for gate_id, receipt_hash in (
        ("stage3.G3-0", receipt.g30_gate_hash),
        ("stage3.G3-1", receipt.g31_gate_hash),
    ):
        if receipt_hash is not None:
            gate = evidence_gates.get(gate_id)
            if gate is None or gate.artifact_hash != receipt_hash:
                raise _fail("TRAJECTORY_GATE_HASH_MISMATCH", gate_id)
    if scope == "formal" and (
        receipt.g30_gate_hash != (evidence_gates.get("stage3.G3-0").artifact_hash if evidence_gates.get("stage3.G3-0") else None)
        or receipt.g31_gate_hash != (evidence_gates.get("stage3.G3-1").artifact_hash if evidence_gates.get("stage3.G3-1") else None)
    ):
        raise _fail("FORMAL_TRAJECTORY_GATE_COVERAGE_MISMATCH")
    allocation = _validate_allocation(
        _load_mapping(_resolve_ref(source_value["probe_allocation_ref"], roots=roots, field="probe_allocation_ref"), "allocation"),
        scope=scope,
    )
    files, resolver_id, resolver_state_digest, content_source_hash, loss_hash, effective_weight_unit = _validate_content_source(
        _load_mapping(_resolve_ref(source_value["content_source_ref"], roots=roots, field="content_source_ref"), "content_source"),
        roots=roots,
    )
    receipt_refs = tuple(receipt.endpoint_commit_refs)
    if len(set(receipt.endpoint_digests)) != len(receipt.endpoint_digests):
        raise _fail("TRAJECTORY_ENDPOINT_DIGEST_DUPLICATE")
    if set(allocation) != set(receipt_refs) or len(allocation) != len(receipt_refs):
        raise _fail("ALLOCATION_ENDPOINT_COVERAGE_MISMATCH")
    output_dir = _resolve_ref(source_value["output_dir"], roots=roots, field="output_dir", require_file=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Re-read every endpoint commit/object through the production identity
    # loader.  This checks the legacy endpoint digest and update_sample_ids,
    # rather than trusting receipt copies.
    endpoint_updates: set[str | int] = set()
    endpoint_by_ref: dict[str, Any] = {}
    for index, ref in enumerate(receipt_refs):
        commit_path = _resolve_ref(ref, roots=roots, field=f"endpoint_commit_ref[{index}]")
        raw_commit = _load_mapping(commit_path, f"endpoint_commit:{ref}")
        identity = _load_endpoint(_Source(ref, commit_path), scope=scope, workspace_root=data)
        if raw_commit.get("optimizer_step") != receipt.selected_steps[index]:
            raise _fail("TRAJECTORY_ENDPOINT_STEP_MISMATCH", ref)
        if raw_commit.get("endpoint_digest") != receipt.endpoint_digests[index]:
            raise _fail("TRAJECTORY_ENDPOINT_DIGEST_MISMATCH", ref)
        if identity.endpoint_digest != receipt.endpoint_digests[index]:
            raise _fail("ENDPOINT_DIGEST_RE_READ_MISMATCH", ref)
        if identity.ref != ref or ref not in allocation:
            raise _fail("ENDPOINT_ALLOCATION_BINDING_MISMATCH", ref)
        endpoint_by_ref[ref] = identity
        endpoint_updates.update(identity.update_sample_ids)
    probe_samples: set[str | int] = set()
    probe_ids_global: set[str] = set()
    pending: list[tuple[Path, Mapping[str, Any]]] = []
    for ref in receipt_refs:
        identity = endpoint_by_ref[ref]
        raw_probes = allocation[ref]
        required_count = 2 if scope == "pilot" else 3
        if len(raw_probes) != required_count:
            raise _fail("PROBE_COUNT_INVALID", ref)
        entries: list[dict[str, Any]] = []
        for raw_probe in raw_probes:
            sample_ids = [_sample_key(item, f"{raw_probe['probe_id']}.sample_ids") for item in raw_probe["sample_ids"]]
            probe_id = str(raw_probe["probe_id"])
            if probe_id in probe_ids_global:
                raise _fail("PROBE_ID_GLOBAL_DUPLICATE", probe_id)
            probe_ids_global.add(probe_id)
            overlap = endpoint_updates.intersection(sample_ids) | probe_samples.intersection(sample_ids)
            if overlap:
                raise _fail("PROBE_GLOBAL_OVERLAP", f"{raw_probe['probe_id']}:{sorted(map(str, overlap))}")
            probe_samples.update(sample_ids)
            entries.append(
                {
                    "role": raw_probe["role"],
                    "probe_id": raw_probe["probe_id"],
                    "sample_ids": sample_ids,
                    "content_hash": _content_hash(sample_ids, files),
                    "loss_contract_hash": loss_hash,
                    "effective_weight_unit": effective_weight_unit,
                    "metadata": {
                        **dict(raw_probe["metadata"]),
                        "materializer_resolver_id": resolver_id,
                        "materializer_resolver_state_digest": resolver_state_digest,
                        "materializer_content_source_hash": content_source_hash,
                    },
                }
            )
        panel_id = f"stage3-probe-{identity.endpoint_id}"
        if _ID_RE.fullmatch(panel_id) is None:
            raise _fail("PANEL_ID_INVALID", panel_id)
        body: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA,
            "panel_id": panel_id,
            "endpoint_digest": identity.endpoint_digest,
            "entries": entries,
            "minimum_formal_probes": 3,
            "execution_evidence_hash": evidence.artifact_hash,
            "scope": scope,
            "formal_eligible": scope == "formal",
        }
        payload = body | {"artifact_hash": canonical_json_hash(body)}
        target = output_dir / f"{panel_id}.json"
        pending.append((target, payload))
    # The output set is complete only when every allocated sample was emitted.
    if not probe_samples:
        raise _fail("PROBE_GLOBAL_EMPTY")
    # Do not publish a prefix of a matrix: all source validation above must
    # succeed before the first immutable output is created.
    for target, payload in pending:
        _check_immutable(target, payload)
    for target, payload in pending:
        if not target.exists():
            _immutable_write(target, payload)
    return tuple(target for target, _ in pending)


materialize = materialize_probe_plans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="materialization source JSON")
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = materialize_probe_plans(args.source, workspace_root=args.workspace_root, data_root=args.data_root)
    print(json.dumps({"schema_version": PLAN_SCHEMA, "plan_refs": [path.as_posix() for path in paths]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOCATION_SCHEMA",
    "CONTENT_SOURCE_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "PLAN_SCHEMA",
    "ProbePlanMaterializationError",
    "build_parser",
    "main",
    "materialize",
    "materialize_probe_plans",
]
