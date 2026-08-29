"""Export immutable real-Pythia Stage 3 probe sources.

The endpoint trajectory producer and :mod:`materialize_stage3_probe_plan` are
intentionally separate.  This module is the small bridge between them: it
replays a qualified G3 asset resolution, reads records from the verified Pythia
``MMIDIDX``/``document-*.bin`` stream, and publishes the allocation and source
artifacts consumed by the probe-plan materializer.

There is no fixture or synthetic fallback in this module.  A source must carry
an explicit scope, four explicit probe intervals (the enclosing ``probe``
interval plus disjoint ``pilot``, ``formal`` and ``replay`` sub-intervals), and
an allocation seed.  Records are ranked with SHA-256 and selected without
replacement.  Endpoint update sample IDs are re-read from their committed
endpoint objects and removed from the candidate population before selection.

The output is deliberately boring and inspectable:

* ``allocation.json`` -- endpoint -> probe -> ordered 32-record IDs;
* ``content/record-*.bin`` -- one exact uint16 little-endian Pile record per
  selected sample ID (2049 source tokens, not generated content);
* ``resolver-state.json`` -- the qualified resolver identity and partition;
* ``loss-contract.json`` -- the real pre-shifted causal-LM loss contract;
* ``content-source.json`` -- content references for the existing materializer;
* ``materialization-source.json`` -- a hash-bound invocation source; and
* ``export-report.json`` -- the complete immutable publication receipt.

All JSON is canonical.  Existing files are immutable: an equal retry is
idempotent and a different retry fails closed.  The command performs all
validation and record reads before publishing any output file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

from param_importance_nlp.contracts.jsonio import (
    canonical_json_bytes,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.data.pythia_mmap import (
    PYTHIA_TOKENS_PER_RECORD,
    PythiaIndexedDataset,
)
from param_importance_nlp.g3_runtime_assets import (
    FormalG3RuntimeAssets,
    G3RuntimeAssetError,
)
from param_importance_nlp.providers.pythia_mmap import PythiaMMapFrozenSampleResolver
from param_importance_nlp.runtime import publish_canonical_immutable
from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact
from param_importance_nlp.experiments.stage3_production_plan import FORMAL_MODEL_SEEDS

from ops.stage3.materialize_stage3_probe_plan import materialize_probe_plans
from param_importance_nlp.experiments.stage3_production_plan import (
    _Source,
    _load_endpoint,
)
from param_importance_nlp.experiments.stage3_trajectory import Stage3TrajectoryReceipt


SOURCE_SCHEMA = "stage3-pythia-probe-source-export-v1"
ALLOCATION_SCHEMA = "stage3-probe-allocation-v1"
CONTENT_SOURCE_SCHEMA = "stage3-frozen-probe-content-source-v1"
RESOLVER_STATE_SCHEMA = "stage3-pythia-mmap-resolver-state-v1"
LOSS_CONTRACT_SCHEMA = "stage3-causal-lm-loss-contract-v1"
REPORT_SCHEMA = "stage3-pythia-probe-source-export-report-v1"
MATERIALIZATION_SCHEMA = "stage3-probe-plan-materialization-source-v1"

PILOT_SCOPE = "pilot"
FORMAL_SCOPE = "formal"
REPLAY_SCOPE = "replay"
SCOPES = frozenset({PILOT_SCOPE, FORMAL_SCOPE})
PARTITION_NAMES = (PILOT_SCOPE, FORMAL_SCOPE, REPLAY_SCOPE)
RECORDS_PER_PROBE = 32
PILOT_ENDPOINTS = 6
FORMAL_ENDPOINTS = 33
PILOT_PROBES = 2
FORMAL_PROBES = 3
REPLAY_RECORDS = 32
# The current preregistration uses these absolute probe subranges.  They stay
# source-controlled inputs (rather than hidden defaults): callers must include
# all four intervals in the hash-bound source, and validation only relaxes the
# interval *capacity* so a future explicitly preregistered expansion can be
# selected without replacement.
PREREGISTERED_PYTHIA_PROBE_INTERVAL = (966144, 1031680)
PREREGISTERED_PYTHIA_PILOT_INTERVAL = (966144, 974336)
PREREGISTERED_PYTHIA_FORMAL_INTERVAL = (974336, 1023488)
PREREGISTERED_PYTHIA_REPLAY_INTERVAL = (1023488, 1031680)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SAMPLE_RE_TEMPLATE = r"^pile:(?P<asset>[0-9a-f]{64}):record:(?P<index>[0-9]{12})$"
_FORBIDDEN_RE = re.compile(r"(?:fixture|synthetic)", re.IGNORECASE)


class ProbeSourceExportError(ValueError):
    """Raised when a real Stage 3 probe source cannot be published."""


def _fail(code: str, detail: object | None = None) -> ProbeSourceExportError:
    return ProbeSourceExportError(code if detail is None else f"{code}:{detail}")


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
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _no_forbidden(child, field)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_ref(
    value: object,
    *,
    roots: Sequence[Path],
    field: str,
    require_file: bool = True,
) -> Path:
    if isinstance(value, Path):
        raw = value
    elif isinstance(value, str) and value and (Path(value).is_absolute() or "\\" not in value):
        raw = Path(value)
    else:
        raise _fail("REFERENCE_INVALID", field)
    if raw.is_absolute():
        candidates = [raw.resolve()]
    else:
        logical = PurePosixPath(str(value))
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
        if require_file:
            raise _fail("REFERENCE_NOT_FOUND", field)
        return candidates[0]
    if require_file and not existing[0].is_file():
        raise _fail("REFERENCE_NOT_FILE", field)
    return existing[0]


def _logical_ref(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise _fail("OUTPUT_REFERENCE_OUTSIDE_ROOT", path) from error


def _load_object(path: Path, field: str) -> Mapping[str, Any]:
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


def _validate_source(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "schema_version",
        "scope",
        "g3_resolution_ref",
        "allocation_seed",
        "partition",
        "output_dir",
        "artifact_hash",
    }
    scope = value.get("scope")
    if scope == PILOT_SCOPE:
        expected.add("trajectory_receipt_ref")
    elif scope == FORMAL_SCOPE:
        # Formal coverage spans three independent training trajectories.  Do
        # not accept the old singular field here: silently treating one
        # receipt as the whole matrix would make the 24+9 endpoint identity
        # boundary unverifiable.
        expected.add("trajectory_receipt_refs")
    if set(value) != expected or value.get("schema_version") != SOURCE_SCHEMA:
        raise _fail("SOURCE_FIELDS_MISMATCH")
    _check_artifact_hash(value, "source")
    _no_forbidden(value)
    if scope not in SCOPES:
        raise _fail("SCOPE_INVALID")
    if (
        isinstance(value.get("allocation_seed"), bool)
        or not isinstance(value.get("allocation_seed"), int)
        or value["allocation_seed"] < 0
    ):
        raise _fail("ALLOCATION_SEED_INVALID")
    for field in ("g3_resolution_ref", "output_dir"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise _fail("REFERENCE_INVALID", field)
    if scope == PILOT_SCOPE:
        if not isinstance(value.get("trajectory_receipt_ref"), str) or not value["trajectory_receipt_ref"]:
            raise _fail("REFERENCE_INVALID", "trajectory_receipt_ref")
    else:
        refs = value.get("trajectory_receipt_refs")
        if (
            not isinstance(refs, list)
            or len(refs) != 3
            or any(not isinstance(item, str) or not item for item in refs)
            or len(refs) != len(set(refs))
        ):
            raise _fail("TRAJECTORY_RECEIPT_REFS_INVALID")
    partition = value.get("partition")
    if not isinstance(partition, Mapping) or set(partition) != {"probe", *PARTITION_NAMES}:
        raise _fail("PARTITION_FIELDS_MISMATCH")
    for name in ("probe", *PARTITION_NAMES):
        interval = partition.get(name)
        if not isinstance(interval, Mapping) or set(interval) != {"start", "stop"}:
            raise _fail("PARTITION_INTERVAL_FIELDS_MISMATCH", name)
        start, stop = interval.get("start"), interval.get("stop")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or start < 0
            or stop <= start
        ):
            raise _fail("PARTITION_INTERVAL_INVALID", name)
    return value


def _interval(value: Mapping[str, Any], name: str) -> tuple[int, int]:
    raw = value[name]
    assert isinstance(raw, Mapping)
    return int(raw["start"]), int(raw["stop"])


def _validate_partition(value: Mapping[str, Any], *, dataset: PythiaIndexedDataset, scope: str) -> dict[str, tuple[int, int]]:
    partition = value["partition"]
    assert isinstance(partition, Mapping)
    intervals = {name: _interval(partition, name) for name in ("probe", *PARTITION_NAMES)}
    if intervals["probe"] != (dataset.record_start, dataset.record_stop):
        raise _fail("PROBE_PARTITION_NOT_DATASET_SPLIT", intervals["probe"])
    probe_start, probe_stop = intervals["probe"]
    for name in PARTITION_NAMES:
        start, stop = intervals[name]
        if start < probe_start or stop > probe_stop:
            raise _fail("PARTITION_OUTSIDE_PROBE_SPLIT", name)
    for left_index, left_name in enumerate(PARTITION_NAMES):
        left_start, left_stop = intervals[left_name]
        for right_name in PARTITION_NAMES[left_index + 1 :]:
            right_start, right_stop = intervals[right_name]
            if max(left_start, right_start) < min(left_stop, right_stop):
                raise _fail("PARTITION_OVERLAP", f"{left_name}:{right_name}")
    expected = {
        PILOT_SCOPE: PILOT_ENDPOINTS * PILOT_PROBES * RECORDS_PER_PROBE,
        FORMAL_SCOPE: FORMAL_ENDPOINTS * FORMAL_PROBES * RECORDS_PER_PROBE,
    }
    if intervals[scope][1] - intervals[scope][0] < expected[scope]:
        raise _fail("PARTITION_SCOPE_SIZE_TOO_SMALL", scope)
    if intervals[REPLAY_SCOPE][1] - intervals[REPLAY_SCOPE][0] < REPLAY_RECORDS:
        raise _fail("PARTITION_REPLAY_SIZE_INVALID")
    return intervals


def _immutable_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes with immutable link semantics and fsync."""

    _reject_link_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise _fail("IMMUTABLE_OUTPUT_CONFLICT", path)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise _fail("IMMUTABLE_OUTPUT_CONFLICT", path)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_target(path: Path, payload: bytes) -> None:
    _reject_link_components(path)
    if path.exists() and (not path.is_file() or path.read_bytes() != payload):
        raise _fail("IMMUTABLE_OUTPUT_CONFLICT", path)


def _reject_link_components(path: Path) -> None:
    """Reject reparse/symlink components before binary publication."""

    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        is_junction = bool(getattr(current, "is_junction", lambda: False)())
        if current.is_symlink() or is_junction:
            raise _fail("OUTPUT_SYMLINK_FORBIDDEN", current)


def _source_root(path: Path, *, workspace: Path, data: Path) -> Path:
    if _within(path, data):
        return data
    if _within(path, workspace):
        return workspace
    raise _fail("REFERENCE_OUTSIDE_ALLOWED_ROOT", path)


def _sample_id(asset_id: str, index: int) -> str:
    return f"pile:{asset_id}:record:{index:012d}"


def _parse_update_id(value: object, *, asset_id: str, field: str) -> int:
    if isinstance(value, bool):
        raise _fail("ENDPOINT_UPDATE_SAMPLE_ID_INVALID", field)
    if isinstance(value, int):
        if value < 0:
            raise _fail("ENDPOINT_UPDATE_SAMPLE_ID_INVALID", field)
        return value
    if not isinstance(value, str):
        raise _fail("ENDPOINT_UPDATE_SAMPLE_ID_INVALID", field)
    match = re.fullmatch(_SAMPLE_RE_TEMPLATE, value)
    if match is None or match.group("asset") != asset_id:
        raise _fail("ENDPOINT_UPDATE_SAMPLE_ID_NOT_PYTHIA", field)
    return int(match.group("index"))


@dataclass(frozen=True, slots=True)
class _PileContext:
    runtime: FormalG3RuntimeAssets
    asset: Any
    dataset: PythiaIndexedDataset
    resolver: PythiaMMapFrozenSampleResolver
    asset_id: str
    data_range_hash: str
    runtime_lineage_hash: str
    resolver_id: str
    resolver_state_digest: str
    sample_prefix: str
    sampling_design: str
    storage: Mapping[str, Any]


def _load_pile_context(
    *,
    source_value: Mapping[str, Any],
    workspace: Path,
    data: Path,
    runtime_assets: FormalG3RuntimeAssets | None,
) -> _PileContext:
    resolution_path = _resolve_ref(
        source_value["g3_resolution_ref"], roots=(data, workspace), field="g3_resolution_ref"
    )
    runtime_root = _source_root(resolution_path, workspace=workspace, data=data)
    try:
        runtime = runtime_assets or FormalG3RuntimeAssets.load(runtime_root, _logical_ref(resolution_path, runtime_root))
        asset = runtime.resolve("pile-selected-prefix", expected_kind="pile")
        dataset = runtime.pythia_dataset(asset, split="probe")
    except (FileNotFoundError, OSError, TypeError, ValueError, G3RuntimeAssetError) as error:
        if isinstance(error, ProbeSourceExportError):
            raise
        raise _fail("G3_ASSET_RESOLUTION_INVALID") from error
    asset_id = str(asset.resolved.asset_id)
    _hash(asset_id, "pile.asset_id")
    storage = asset.manifest.get("metadata")
    if not isinstance(storage, Mapping) or not isinstance(storage.get("storage"), Mapping):
        dataset.close()
        raise _fail("PILE_STORAGE_METADATA_INVALID")
    storage_map = storage["storage"]
    assert isinstance(storage_map, Mapping)
    if storage_map.get("kind") != "pythia_mmap_shards":
        dataset.close()
        raise _fail("PILE_STORAGE_KIND_INVALID")
    data_range_hash = canonical_json_hash(
        {
            "schema_version": "stage3-pythia-probe-data-range-v1",
            "asset_id": asset_id,
            "ready_manifest_sha256": asset.ready_manifest_sha256,
            "qualification_artifact_hash": asset.qualification_artifact_hash,
            "split": "probe",
            "record_interval": [dataset.record_start, dataset.record_stop],
            "tokens_per_record": dataset.tokens_per_record,
            "index_sha256": storage_map.get("idx", {}).get("sha256") if isinstance(storage_map.get("idx"), Mapping) else None,
            "shards": storage_map.get("shards"),
        }
    )
    runtime_lineage_hash = canonical_json_hash(
        {"asset_resolution_hash": runtime.resolution_artifact_hash, "data_range_hash": data_range_hash}
    )
    identity_payload = {
        "schema_version": "pythia-mmap-frozen-resolver-v1",
        "asset_id": asset_id,
        "ready_manifest_sha256": asset.ready_manifest_sha256,
        "qualification_sha256": asset.qualification_artifact_hash,
        "g3_resolution_artifact_hash": runtime.resolution_artifact_hash,
        "g3_source_commit": runtime.source_git_commit,
        "g3_runtime_lineage_sha256": runtime_lineage_hash,
        "split": [dataset.record_start, dataset.record_stop],
        "tokens_per_record": dataset.tokens_per_record,
        "sampling_design": "disjoint_frozen_probe_panel",
        "weights_exogenous": True,
        "common_mean_assumption": True,
    }
    try:
        resolver = PythiaMMapFrozenSampleResolver(
            dataset,
            asset_id=asset_id,
            ready_manifest_sha256=asset.ready_manifest_sha256,
            qualification_sha256=asset.qualification_artifact_hash,
            g3_resolution_artifact_hash=runtime.resolution_artifact_hash,
            g3_source_commit=runtime.source_git_commit,
            g3_runtime_lineage_sha256=runtime_lineage_hash,
            split_start=dataset.record_start,
            split_stop=dataset.record_stop,
            sampling_design="disjoint_frozen_probe_panel",
            weights_exogenous=True,
            common_mean_assumption=True,
        )
    except (TypeError, ValueError, OSError) as error:
        dataset.close()
        raise _fail("PYTHIA_FROZEN_RESOLVER_INVALID") from error
    resolver_id = resolver.resolver_id
    resolver_state_digest = resolver.state_digest()
    if resolver_id != "pythia-mmap-frozen-" + canonical_json_hash(identity_payload)[:32]:
        resolver.close()
        raise _fail("PYTHIA_FROZEN_RESOLVER_ID_DRIFT")
    return _PileContext(
        runtime=runtime,
        asset=asset,
        dataset=dataset,
        resolver=resolver,
        asset_id=asset_id,
        data_range_hash=data_range_hash,
        runtime_lineage_hash=runtime_lineage_hash,
        resolver_id=resolver_id,
        resolver_state_digest=resolver_state_digest,
        sample_prefix=f"pile:{asset_id}:record:",
        sampling_design="disjoint_frozen_probe_panel",
        storage=storage_map,
    )


def _trajectory_receipt_refs(
    source_value: Mapping[str, Any],
    *,
    scope: str,
) -> tuple[str, ...]:
    """Return the receipt refs for one probe-source export.

    Pilot retains its historical singular reference.  Formal exports require
    exactly one receipt for each preregistered model/seed trajectory so the
    14M/31M endpoint matrix cannot be represented by an unverified hand-made
    aggregate receipt.
    """

    if scope == PILOT_SCOPE:
        return (str(source_value["trajectory_receipt_ref"]),)
    refs = source_value.get("trajectory_receipt_refs")
    if (
        not isinstance(refs, list)
        or len(refs) != 3
        or any(not isinstance(item, str) or not item for item in refs)
        or len(refs) != len(set(refs))
    ):
        raise _fail("TRAJECTORY_RECEIPT_REFS_INVALID")
    return tuple(refs)


def _load_receipt_and_endpoints(
    *,
    source_value: Mapping[str, Any],
    workspace: Path,
    data: Path,
) -> tuple[Stage3TrajectoryReceipt | tuple[Stage3TrajectoryReceipt, ...], tuple[Any, ...], dict[str, int]]:
    scope = str(source_value["scope"])
    receipt_refs = _trajectory_receipt_refs(source_value, scope=scope)
    receipts: list[Stage3TrajectoryReceipt] = []
    receipt_hashes: set[str] = set()
    execution_refs: set[str] = set()
    execution_evidences: list[FormalExecutionEvidence] = []
    gate_hashes: dict[str, set[str]] = {"g30": set(), "g31": set()}
    for receipt_index, receipt_ref in enumerate(receipt_refs):
        receipt_path = _resolve_ref(
            receipt_ref,
            roots=(data, workspace),
            field=f"trajectory_receipt_refs[{receipt_index}]",
        )
        try:
            receipt = Stage3TrajectoryReceipt.from_mapping(
                _load_object(receipt_path, "trajectory_receipt")
            )
        except (TypeError, ValueError) as error:
            raise _fail("TRAJECTORY_RECEIPT_INVALID", receipt_ref) from error
        if receipt.purpose_scope != scope or receipt.formal_eligible is not (scope == FORMAL_SCOPE):
            raise _fail("TRAJECTORY_SCOPE_MISMATCH", receipt_ref)
        if not receipt.formal_execution_ref:
            raise _fail("TRAJECTORY_FORMAL_EXECUTION_REF_MISSING", receipt_ref)
        if receipt.artifact_hash in receipt_hashes:
            raise _fail("TRAJECTORY_RECEIPT_DUPLICATE", receipt_ref)
        receipt_hashes.add(receipt.artifact_hash)
        # Every independently materialized trajectory must pass the same
        # canonical formal-execution/G3 authority reload.  Checking only the
        # reference string would let a malformed or stale receipt participate
        # in a seemingly complete 33-endpoint matrix.
        execution_evidences.append(
            _validate_execution_evidence(receipt, workspace=workspace, data=data)
        )
        execution_refs.add(receipt.formal_execution_ref)
        if receipt.g30_gate_hash is not None:
            gate_hashes["g30"].add(receipt.g30_gate_hash)
        if receipt.g31_gate_hash is not None:
            gate_hashes["g31"].add(receipt.g31_gate_hash)
        receipts.append(receipt)

    if scope == FORMAL_SCOPE and len(receipts) != 3:
        # Defensive check for callers bypassing _validate_source().
        raise _fail("TRAJECTORY_RECEIPT_COUNT_INVALID")
    if len(execution_refs) != 1 or len({item.artifact_hash for item in execution_evidences}) != 1:
        raise _fail("TRAJECTORY_EXECUTION_MIXED")
    if scope == FORMAL_SCOPE and (
        gate_hashes["g30"] != {receipts[0].g30_gate_hash}
        or gate_hashes["g31"] != {receipts[0].g31_gate_hash}
    ):
        raise _fail("TRAJECTORY_GATE_MIXED")

    endpoint_identities: list[Any] = []
    endpoint_steps: dict[str, int] = {}
    endpoint_refs: set[str] = set()
    endpoint_digests: set[str] = set()
    endpoint_ids: set[str] = set()
    update_ids: set[object] = set()
    receipt_model_seeds: list[tuple[str, int]] = []
    for receipt in receipts:
        receipt_endpoint_pairs: set[tuple[str, int]] = set()
        for index, reference in enumerate(receipt.endpoint_commit_refs):
            if reference in endpoint_refs:
                raise _fail("TRAJECTORY_ENDPOINT_REF_DUPLICATE", reference)
            endpoint_refs.add(reference)
            commit_path = _resolve_ref(
                reference,
                roots=(data, workspace),
                field=f"endpoint_commit_ref[{len(endpoint_identities)}]",
            )
            endpoint_root = _source_root(commit_path, workspace=workspace, data=data)
            try:
                identity = _load_endpoint(
                    _Source(reference, commit_path), scope=scope, workspace_root=endpoint_root
                )
            except (TypeError, ValueError) as error:
                raise _fail("ENDPOINT_COMMIT_INVALID", reference) from error
            if identity.endpoint_digest != receipt.endpoint_digests[index]:
                raise _fail("TRAJECTORY_ENDPOINT_DIGEST_MISMATCH", reference)
            if identity.endpoint_digest in endpoint_digests:
                raise _fail("TRAJECTORY_ENDPOINT_DIGEST_DUPLICATE", reference)
            endpoint_digests.add(identity.endpoint_digest)
            if identity.ref != reference:
                raise _fail("ENDPOINT_REFERENCE_DRIFT", reference)
            if identity.endpoint_id in endpoint_ids:
                raise _fail("TRAJECTORY_ENDPOINT_ID_DUPLICATE", identity.endpoint_id)
            endpoint_ids.add(identity.endpoint_id)
            step = receipt.selected_steps[index]
            endpoint_steps[reference] = step
            receipt_endpoint_pairs.add((identity.model, identity.seed))
            if update_ids.intersection(identity.update_sample_ids):
                raise _fail("ENDPOINT_UPDATE_SAMPLE_DUPLICATE", identity.endpoint_id)
            update_ids.update(identity.update_sample_ids)
            endpoint_identities.append(identity)
        if len(receipt_endpoint_pairs) != 1:
            raise _fail("TRAJECTORY_RECEIPT_MODEL_SEED_MIXED", receipt.receipt_id)
        receipt_model_seeds.extend(receipt_endpoint_pairs)

    if scope == FORMAL_SCOPE and set(receipt_model_seeds) != {
        ("14M", 4301), ("14M", 4302), ("31M", 5301)
    }:
        raise _fail("TRAJECTORY_RECEIPT_MODEL_SEED_COVERAGE_INVALID")
    _validate_endpoint_matrix(endpoint_identities, endpoint_steps, scope=scope)
    loaded_receipts: Stage3TrajectoryReceipt | tuple[Stage3TrajectoryReceipt, ...]
    loaded_receipts = receipts[0] if scope == PILOT_SCOPE else tuple(receipts)
    return loaded_receipts, tuple(endpoint_identities), endpoint_steps


def _validate_execution_evidence(
    receipt: Stage3TrajectoryReceipt,
    *,
    workspace: Path,
    data: Path,
) -> FormalExecutionEvidence:
    """Preflight the evidence checks repeated by the downstream materializer."""

    assert receipt.formal_execution_ref is not None
    path = _resolve_ref(receipt.formal_execution_ref, roots=(data, workspace), field="formal_execution_ref")
    try:
        raw = _load_object(path, "formal_execution")
        if raw.get("schema_version") == "task-output-commit-v1":
            evidence_root = _source_root(path, workspace=workspace, data=data)
            loaded = load_committed_task_artifact(
                evidence_root,
                _logical_ref(path, evidence_root),
                require_formal=True,
            )
            if loaded.identity.artifact_kind != "formal_execution_evidence":
                raise ValueError("FORMAL_EXECUTION_COMMIT_KIND_INVALID")
            raw = loaded.payload
        evidence = FormalExecutionEvidence.from_mapping(raw)
        evidence.require_for_stage(3)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise _fail("FORMAL_EXECUTION_EVIDENCE_INVALID") from error
    if evidence.run_intent != "formal":
        raise _fail("FORMAL_EXECUTION_EVIDENCE_NOT_FORMAL")
    gates = {gate.gate_id: gate for gate in evidence.prerequisite_gates}
    for gate_id, receipt_hash in (("stage3.G3-0", receipt.g30_gate_hash), ("stage3.G3-1", receipt.g31_gate_hash)):
        if receipt_hash is not None:
            gate = gates.get(gate_id)
            if gate is None or gate.artifact_hash != receipt_hash:
                raise _fail("TRAJECTORY_GATE_HASH_MISMATCH", gate_id)
    required_formal_gates = ("stage3.G3-0", "stage3.G3-1", "stage3.G3-5")
    if receipt.formal_eligible and any(
        gates.get(gate_id) is None for gate_id in required_formal_gates
    ):
        raise _fail("FORMAL_TRAJECTORY_GATE_COVERAGE_MISMATCH")
    return evidence


def _validate_endpoint_matrix(endpoints: Sequence[Any], steps: Mapping[str, int], *, scope: str) -> None:
    if scope == PILOT_SCOPE:
        if len(endpoints) != PILOT_ENDPOINTS or {item.model for item in endpoints} != {"14M"} or len({item.seed for item in endpoints}) != 1:
            raise _fail("PILOT_ENDPOINT_COVERAGE_INVALID")
        expected_per_stage = 2
        for stage in ("early", "middle", "late"):
            if sum(item.stage == stage for item in endpoints) != expected_per_stage:
                raise _fail("PILOT_STAGE_ENDPOINT_COUNT_INVALID", stage)
        return
    if len(endpoints) != FORMAL_ENDPOINTS or {item.model for item in endpoints} != {"14M", "31M"}:
        raise _fail("FORMAL_ENDPOINT_COVERAGE_INVALID")
    for model, expected_seeds, per_stage in (
        ("14M", FORMAL_MODEL_SEEDS["14M"], 4),
        ("31M", FORMAL_MODEL_SEEDS["31M"], 3),
    ):
        selected = [item for item in endpoints if item.model == model]
        if {item.seed for item in selected} != set(expected_seeds):
            raise _fail("FORMAL_SEED_COVERAGE_INVALID", model)
        for seed in expected_seeds:
            for stage in ("early", "middle", "late"):
                if sum(item.seed == seed and item.stage == stage for item in selected) != per_stage:
                    raise _fail("FORMAL_STAGE_ENDPOINT_COUNT_INVALID", f"{model}:{seed}:{stage}")


def _rank_candidates(
    *,
    start: int,
    stop: int,
    seed: int,
    scope: str,
    excluded: set[int],
    required: int,
) -> list[int]:
    candidates = [index for index in range(start, stop) if index not in excluded]
    if len(candidates) < required:
        raise _fail("PROBE_CANDIDATE_POOL_TOO_SMALL", f"required={required},available={len(candidates)}")
    ranked = sorted(
        candidates,
        key=lambda index: hashlib.sha256(
            f"stage3-probe-allocation-v1\0{seed}\0{scope}\0{index}".encode("ascii")
        ).hexdigest(),
    )
    return ranked[:required]


def _allocate(
    *,
    endpoints: Sequence[Any],
    endpoint_steps: Mapping[str, int],
    asset_id: str,
    interval: tuple[int, int],
    scope: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    expected_probes = PILOT_PROBES if scope == PILOT_SCOPE else FORMAL_PROBES
    required = len(endpoints) * expected_probes * RECORDS_PER_PROBE
    update_indices = _endpoint_update_indices(endpoints, asset_id=asset_id)
    excluded = set(update_indices)
    ranked = _rank_candidates(start=interval[0], stop=interval[1], seed=seed, scope=scope, excluded=excluded, required=required)
    endpoint_order = sorted(
        endpoints,
        key=lambda item: (item.model, item.seed, item.stage, endpoint_steps[item.ref], item.endpoint_id, item.endpoint_digest),
    )
    allocations: list[dict[str, Any]] = []
    assigned: dict[str, list[int]] = {}
    cursor = 0
    for endpoint in endpoint_order:
        probes: list[dict[str, Any]] = []
        for probe_index in range(expected_probes):
            indices = ranked[cursor : cursor + RECORDS_PER_PROBE]
            cursor += RECORDS_PER_PROBE
            if len(indices) != RECORDS_PER_PROBE or len(set(indices)) != RECORDS_PER_PROBE:
                raise _fail("ALLOCATION_WITHOUT_REPLACEMENT_FAILED", endpoint.endpoint_id)
            probe_id = f"stage3-{scope}-probe-{len(allocations) * expected_probes + probe_index:03d}"
            sample_ids = [_sample_id(asset_id, index) for index in indices]
            metadata = {
                "source": "pythia-mmap-real",
                "allocation_algorithm": "sha256-hash-ranking-without-replacement-v1",
                "allocation_seed": seed,
                "partition_scope": scope,
                "partition_start": interval[0],
                "partition_stop": interval[1],
                "record_count": RECORDS_PER_PROBE,
                "endpoint_id": endpoint.endpoint_id,
                "endpoint_digest": endpoint.endpoint_digest,
                "optimizer_step": endpoint_steps[endpoint.ref],
            }
            probes.append({"role": scope, "probe_id": probe_id, "sample_ids": sample_ids, "metadata": metadata})
            assigned[probe_id] = indices
        allocations.append({"endpoint_commit_ref": endpoint.ref, "probes": probes})
    if cursor != required or len({index for values in assigned.values() for index in values}) != required:
        raise _fail("ALLOCATION_COVERAGE_INVALID")
    return allocations, assigned


def _endpoint_update_indices(endpoints: Sequence[Any], *, asset_id: str) -> set[int]:
    """Parse and deduplicate endpoint update IDs in the Pythia namespace."""

    update_indices: set[int] = set()
    for endpoint in endpoints:
        for index, update_id in enumerate(endpoint.update_sample_ids):
            parsed = _parse_update_id(
                update_id,
                asset_id=asset_id,
                field=f"{endpoint.endpoint_id}.update_sample_ids[{index}]",
            )
            if parsed in update_indices:
                raise _fail("ENDPOINT_UPDATE_SAMPLE_DUPLICATE", parsed)
            update_indices.add(parsed)
    return update_indices


def _select_replay_indices(
    endpoints: Sequence[Any],
    *,
    asset_id: str,
    interval: tuple[int, int],
    seed: int,
) -> list[int]:
    """Select the fixed replay audit subset from its reserved interval."""

    return _rank_candidates(
        start=interval[0],
        stop=interval[1],
        seed=seed,
        scope=REPLAY_SCOPE,
        excluded=_endpoint_update_indices(endpoints, asset_id=asset_id),
        required=REPLAY_RECORDS,
    )


def _loss_contract(*, context: _PileContext) -> dict[str, Any]:
    implementation = Path(__file__).resolve().parents[2] / "src/param_importance_nlp/core/losses.py"
    try:
        implementation_hash = hashlib.sha256(implementation.read_bytes()).hexdigest()
    except OSError as error:
        raise _fail("LOSS_IMPLEMENTATION_UNAVAILABLE") from error
    pile_metadata = context.asset.manifest.get("metadata")
    pile_storage = (
        pile_metadata.get("storage") if isinstance(pile_metadata, Mapping) else None
    )
    pile_contract = (
        pile_storage.get("causal_lm_mapping")
        if isinstance(pile_storage, Mapping)
        else None
    )
    if not isinstance(pile_contract, Mapping):
        raise _fail("CAUSAL_LM_ASSET_CONTRACT_MISSING")
    required = {
        "attention_mask_policy": "all_one_for_fixed_full_record",
        "effective_target_tokens": 2048,
        "input_sequence_length": 2048,
        "input_slice": [0, 2048],
        "label_sequence_length": 2048,
        "label_slice": [1, 2049],
        "labels_alignment": "pre_shifted_next_token",
        "loss_adapter_id": "pre-shifted-next-token-cross-entropy-v1",
        "source_tokens_per_record": 2049,
    }
    for key, expected in required.items():
        if pile_contract.get(key) != expected:
            raise _fail("CAUSAL_LM_ASSET_CONTRACT_DRIFT", key)
    body: dict[str, Any] = {
        "schema_version": LOSS_CONTRACT_SCHEMA,
        "task_type": "causal_lm",
        "implementation_ref": "src/param_importance_nlp/core/losses.py",
        "implementation_sha256": implementation_hash,
        "loss_adapter": "pre_shifted_causal_lm_loss",
        "loss_adapter_id": "pre-shifted-next-token-cross-entropy-v1",
        "reduction": "sum_then_divide_by_effective_target_tokens",
        "effective_weight_unit": "effective_target_tokens",
        "ignore_index": -100,
        "attention_mask_policy": "all_one_for_fixed_full_record",
        "input_sequence_length": 2048,
        "target_sequence_length": 2048,
        "source_tokens_per_record": 2049,
        "target_tokens_per_record": 2048,
        "input_slice": [0, 2048],
        "target_slice": [1, 2049],
        "labels_alignment": "pre_shifted_next_token",
        "zero_effective_token": "reject_fail_closed",
        "regularization": "excluded",
        "asset_contract": dict(pile_contract),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _resolver_state(
    *,
    context: _PileContext,
    intervals: Mapping[str, tuple[int, int]],
    scope: str,
    allocation_seed: int,
    content_hashes: Mapping[str, str],
    replay_selected_indices: Sequence[int],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": RESOLVER_STATE_SCHEMA,
        "resolver_id": context.resolver_id,
        "resolver_state_digest": context.resolver_state_digest,
        "asset_id": context.asset_id,
        "ready_manifest_sha256": context.asset.ready_manifest_sha256,
        "qualification_artifact_hash": context.asset.qualification_artifact_hash,
        "g3_resolution_artifact_hash": context.runtime.resolution_artifact_hash,
        "g3_source_commit": context.runtime.source_git_commit,
        "g3_runtime_lineage_sha256": context.runtime_lineage_hash,
        "sampling_design": context.sampling_design,
        "split": "probe",
        "probe_interval": list(intervals["probe"]),
        "partitions": {name: list(intervals[name]) for name in PARTITION_NAMES},
        "scope": scope,
        "allocation_seed": allocation_seed,
        "tokens_per_record": PYTHIA_TOKENS_PER_RECORD,
        "input_sequence_length": 2048,
        "effective_target_tokens_per_record": 2048,
        "sample_id_format": "pile:<asset_id>:record:<12-digit-global-index>",
        "sample_id_prefix": context.sample_prefix,
        "storage": dict(context.storage),
        "record_count": intervals[scope][1] - intervals[scope][0],
        "replay_reserved": True,
        "replay_selected_indices": list(replay_selected_indices),
        "replay_selected_record_count": len(replay_selected_indices),
        "content_inventory": [
            {"sample_id": sample_id, "content_sha256": digest}
            for sample_id, digest in sorted(
                content_hashes.items(), key=lambda item: int(item[0].rsplit(":", 1)[-1])
            )
        ],
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _read_content(
    *,
    context: _PileContext,
    sample_indices: Sequence[int],
    output_base: Path,
) -> tuple[dict[str, Path], dict[str, str], dict[Path, bytes]]:
    content_paths: dict[str, Path] = {}
    content_hashes: dict[str, str] = {}
    payloads: dict[Path, bytes] = {}
    seen: set[int] = set()
    for index in sample_indices:
        if index in seen:
            raise _fail("CONTENT_SAMPLE_DUPLICATE", index)
        seen.add(index)
        if index < context.dataset.record_start or index >= context.dataset.record_stop:
            raise _fail("CONTENT_RECORD_OUTSIDE_PROBE", index)
        try:
            record = context.dataset.raw_record(index - context.dataset.record_start)
        except (IndexError, OSError, TypeError, ValueError) as error:
            raise _fail("CONTENT_RECORD_READ_FAILED", index) from error
        if record.shape != (PYTHIA_TOKENS_PER_RECORD,) or str(record.dtype) not in {"uint16", "<u2"}:
            raise _fail("CONTENT_RECORD_SHAPE_OR_DTYPE_INVALID", index)
        payload = record.astype("<u2", copy=False).tobytes()
        if len(payload) != PYTHIA_TOKENS_PER_RECORD * 2:
            raise _fail("CONTENT_RECORD_BYTE_LENGTH_INVALID", index)
        sample_id = _sample_id(context.asset_id, index)
        target = output_base / "content" / f"record-{index:012d}.bin"
        content_paths[sample_id] = target
        content_hashes[sample_id] = hashlib.sha256(payload).hexdigest()
        # Keep bytes in a local map until all records have been read; the
        # caller publishes from this deterministic preflight result.
        payloads[target] = payload
    return content_paths, content_hashes, payloads


def export_stage3_probe_source(
    source: Mapping[str, Any] | str | Path,
    *,
    workspace_root: str | Path,
    data_root: str | Path,
    runtime_assets: FormalG3RuntimeAssets | None = None,
) -> Mapping[str, Any]:
    """Export and materialize a real pilot or formal probe source.

    ``runtime_assets`` is an explicit dependency-injection seam for tests and
    for a caller that already loaded the same qualified G3 resolution.  When it
    is omitted, the resolution is always reloaded through
    :class:`FormalG3RuntimeAssets`.
    """

    workspace = Path(workspace_root).resolve()
    data = Path(data_root).resolve()
    if isinstance(source, Mapping):
        source_value = source
    else:
        source_path = _resolve_ref(source, roots=(data, workspace), field="source")
        source_value = _load_object(source_path, "source")
    _validate_source(source_value)
    scope = str(source_value["scope"])

    context = _load_pile_context(
        source_value=source_value, workspace=workspace, data=data, runtime_assets=runtime_assets
    )
    try:
        intervals = _validate_partition(source_value, dataset=context.dataset, scope=scope)
        receipt_value, endpoints, endpoint_steps = _load_receipt_and_endpoints(
            source_value=source_value, workspace=workspace, data=data
        )
        receipts = (
            (receipt_value,)
            if isinstance(receipt_value, Stage3TrajectoryReceipt)
            else tuple(receipt_value)
        )
        receipt_refs = _trajectory_receipt_refs(source_value, scope=scope)
        # _load_receipt_and_endpoints has already reloaded every evidence
        # object.  Retain the historical single evidence value for the report
        # and perform one final reload for the primary materializer input.
        evidence = _validate_execution_evidence(
            receipts[0], workspace=workspace, data=data
        )
        expected = PILOT_ENDPOINTS if scope == PILOT_SCOPE else FORMAL_ENDPOINTS
        if len(endpoints) != expected:
            raise _fail("ENDPOINT_COUNT_INVALID", scope)
        output = _resolve_ref(source_value["output_dir"], roots=(data, workspace), field="output_dir", require_file=False)
        if not _within(output, data) and not _within(output, workspace):
            raise _fail("OUTPUT_OUTSIDE_ALLOWED_ROOT")
        if output.is_symlink():
            raise _fail("OUTPUT_SYMLINK_FORBIDDEN")
        output_root = data if _within(output, data) else workspace
        output = output.resolve()

        allocations, assigned = _allocate(
            endpoints=endpoints,
            endpoint_steps=endpoint_steps,
            asset_id=context.asset_id,
            interval=intervals[scope],
            scope=scope,
            seed=int(source_value["allocation_seed"]),
        )
        replay_start, replay_stop = intervals[REPLAY_SCOPE]
        replay_indices = _select_replay_indices(
            endpoints,
            asset_id=context.asset_id,
            interval=(replay_start, replay_stop),
            seed=int(source_value["allocation_seed"]),
        )
        selected_indices = sorted(
            {index for values in assigned.values() for index in values}
            | set(replay_indices)
        )
        content_paths, content_hashes, binary_payloads = _read_content(
            context=context, sample_indices=selected_indices, output_base=output
        )
        try:
            context.resolver.assert_unchanged(context.resolver_state_digest)
        except (RuntimeError, TypeError, ValueError) as error:
            raise _fail("PYTHIA_FROZEN_RESOLVER_CHANGED") from error

        loss_contract = _loss_contract(context=context)
        resolver_state = _resolver_state(
            context=context,
            intervals=intervals,
            scope=scope,
            allocation_seed=int(source_value["allocation_seed"]),
            content_hashes=content_hashes,
            replay_selected_indices=replay_indices,
        )
        loss_path = output / "loss-contract.json"
        resolver_path = output / "resolver-state.json"
        allocation_body: dict[str, Any] = {
            "schema_version": ALLOCATION_SCHEMA,
            "scope": scope,
            "allocations": allocations,
        }
        allocation = allocation_body | {"artifact_hash": canonical_json_hash(allocation_body)}
        allocation_path = output / "allocation.json"

        sample_entries = [
            {
                "sample_id": sample_id,
                "content_ref": _logical_ref(path, output_root),
            }
            for sample_id, path in sorted(content_paths.items(), key=lambda item: int(item[0].rsplit(":", 1)[-1]))
        ]
        content_body: dict[str, Any] = {
            "schema_version": CONTENT_SOURCE_SCHEMA,
            "resolver_id": context.resolver_id,
            "resolver_state_ref": _logical_ref(resolver_path, output_root),
            "samples": sample_entries,
            "loss_contract_ref": _logical_ref(loss_path, output_root),
            "effective_weight_unit": "effective_target_tokens",
        }
        content_source = content_body | {"artifact_hash": canonical_json_hash(content_body)}
        content_source_path = output / "content-source.json"
        # The existing plan materializer intentionally accepts one trajectory
        # receipt.  For a formal matrix, preserve that contract by creating
        # one real allocation/materialization source per receipt; the global
        # allocation remains an audit artifact and is never presented as a
        # fabricated aggregate receipt.
        materialization_items: list[tuple[Path, Mapping[str, Any]]] = []
        allocation_items: list[tuple[Path, Mapping[str, Any]]] = []
        if scope == FORMAL_SCOPE:
            allocations_by_endpoint = {
                str(item["endpoint_commit_ref"]): item for item in allocations
            }
            for receipt_index, (receipt, receipt_ref) in enumerate(zip(receipts, receipt_refs, strict=True)):
                try:
                    receipt_allocations = [
                        allocations_by_endpoint[endpoint_ref]
                        for endpoint_ref in receipt.endpoint_commit_refs
                    ]
                except KeyError as error:
                    raise _fail("ALLOCATION_ENDPOINT_COVERAGE_MISMATCH", error.args[0]) from error
                per_allocation_body: dict[str, Any] = {
                    "schema_version": ALLOCATION_SCHEMA,
                    "scope": scope,
                    "allocations": receipt_allocations,
                }
                per_allocation = per_allocation_body | {
                    "artifact_hash": canonical_json_hash(per_allocation_body)
                }
                per_allocation_path = output / f"allocation-{receipt_index:03d}.json"
                allocation_items.append((per_allocation_path, per_allocation))
                materialization_body: dict[str, Any] = {
                    "schema_version": MATERIALIZATION_SCHEMA,
                    "scope": scope,
                    "trajectory_receipt_ref": receipt_ref,
                    "probe_allocation_ref": _logical_ref(per_allocation_path, output_root),
                    "content_source_ref": _logical_ref(content_source_path, output_root),
                    "formal_execution_ref": str(receipt.formal_execution_ref),
                    "output_dir": _logical_ref(output / "plans", output_root),
                }
                materialization = materialization_body | {
                    "artifact_hash": canonical_json_hash(materialization_body)
                }
                materialization_items.append(
                    (output / f"materialization-source-{receipt_index:03d}.json", materialization)
                )
        else:
            materialization_body = {
                "schema_version": MATERIALIZATION_SCHEMA,
                "scope": scope,
                "trajectory_receipt_ref": receipt_refs[0],
                "probe_allocation_ref": _logical_ref(allocation_path, output_root),
                "content_source_ref": _logical_ref(content_source_path, output_root),
                "formal_execution_ref": str(receipts[0].formal_execution_ref),
                "output_dir": _logical_ref(output / "plans", output_root),
            }
            materialization = materialization_body | {
                "artifact_hash": canonical_json_hash(materialization_body)
            }
            materialization_path = output / "materialization-source.json"
            materialization_items.append((materialization_path, materialization))

        json_payloads: dict[Path, Mapping[str, Any]] = {
            loss_path: loss_contract,
            resolver_path: resolver_state,
            allocation_path: allocation,
            content_source_path: content_source,
        }
        json_payloads.update(dict(allocation_items))
        json_payloads.update(dict(materialization_items))
        for target, payload in binary_payloads.items():
            _preflight_target(target, payload)
        for target, value in json_payloads.items():
            _preflight_target(target, canonical_json_bytes(value))
        for target, payload in binary_payloads.items():
            _immutable_bytes(target, payload)
        for target, value in json_payloads.items():
            publish_canonical_immutable(target, value)

        plan_paths: list[Path] = []
        for materialization_path, _ in materialization_items:
            plan_paths.extend(
                materialize_probe_plans(
                    materialization_path, workspace_root=workspace, data_root=data
                )
            )
        plan_refs = [_logical_ref(path, output_root) for path in plan_paths]
        report_body: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "scope": scope,
            "allocation_seed": int(source_value["allocation_seed"]),
            "g3_resolution_ref": str(source_value["g3_resolution_ref"]),
            "g3_resolution_artifact_hash": context.runtime.resolution_artifact_hash,
            "formal_execution_ref": str(receipts[0].formal_execution_ref),
            "formal_execution_hash": evidence.artifact_hash,
            "resolver_state_ref": _logical_ref(resolver_path, output_root),
            "resolver_state_hash": resolver_state["artifact_hash"],
            "loss_contract_ref": _logical_ref(loss_path, output_root),
            "loss_contract_hash": loss_contract["artifact_hash"],
            "allocation_ref": _logical_ref(allocation_path, output_root),
            "allocation_hash": allocation["artifact_hash"],
            "content_source_ref": _logical_ref(content_source_path, output_root),
            "content_source_hash": content_source["artifact_hash"],
            "probe_plan_refs": plan_refs,
            "endpoint_count": len(endpoints),
            "probe_count": PILOT_PROBES if scope == PILOT_SCOPE else FORMAL_PROBES,
            "records_per_probe": RECORDS_PER_PROBE,
            "content_record_count": len(binary_payloads),
            "partition": {name: list(intervals[name]) for name in ("probe", *PARTITION_NAMES)},
        }
        if scope == PILOT_SCOPE:
            materialization_path, materialization = materialization_items[0]
            report_body.update(
                {
                    "trajectory_receipt_ref": receipt_refs[0],
                    "trajectory_receipt_hash": receipts[0].artifact_hash,
                    "materialization_source_ref": _logical_ref(materialization_path, output_root),
                    "materialization_source_hash": materialization["artifact_hash"],
                }
            )
        else:
            report_body.update(
                {
                    "trajectory_receipt_refs": list(receipt_refs),
                    "trajectory_receipt_hashes": [item.artifact_hash for item in receipts],
                    "materialization_source_refs": [
                        _logical_ref(path, output_root) for path, _ in materialization_items
                    ],
                    "materialization_source_hashes": [
                        item["artifact_hash"] for _, item in materialization_items
                    ],
                }
            )
        report = report_body | {"artifact_hash": canonical_json_hash(report_body)}
        report_path = output / "export-report.json"
        _preflight_target(report_path, canonical_json_bytes(report))
        publish_canonical_immutable(report_path, report)
        return report
    finally:
        context.resolver.close()


# Short aliases used by orchestration callers.
export_probe_source = export_stage3_probe_source
export_real_pythia_probe_source = export_stage3_probe_source
materialize_real_probe_source = export_stage3_probe_source
materialize_stage3_probe_source = export_stage3_probe_source
materialize = export_stage3_probe_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export_stage3_probe_source(
        args.source, workspace_root=args.workspace_root, data_root=args.data_root
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOCATION_SCHEMA",
    "CONTENT_SOURCE_SCHEMA",
    "FORMAL_ENDPOINTS",
    "FORMAL_PROBES",
    "LOSS_CONTRACT_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "PILOT_ENDPOINTS",
    "PILOT_PROBES",
    "PREREGISTERED_PYTHIA_FORMAL_INTERVAL",
    "PREREGISTERED_PYTHIA_PILOT_INTERVAL",
    "PREREGISTERED_PYTHIA_PROBE_INTERVAL",
    "PREREGISTERED_PYTHIA_REPLAY_INTERVAL",
    "ProbeSourceExportError",
    "REPORT_SCHEMA",
    "RESOLVER_STATE_SCHEMA",
    "SOURCE_SCHEMA",
    "build_parser",
    "export_real_pythia_probe_source",
    "export_probe_source",
    "export_stage3_probe_source",
    "main",
    "materialize",
    "materialize_real_probe_source",
    "materialize_stage3_probe_source",
]
