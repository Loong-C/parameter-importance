"""The S2.3/G2.2 formal asset GateRecord adapter.

This module is deliberately a consumer, rather than another S2.3 runner.  It
only accepts the four formal task-output commits produced by the S2.3 runner
and the independently published server evidence.  In particular, a candidate
GateRecord, a local fixture, or a self-reported digest is never promoted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
from types import SimpleNamespace

from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.status import GateRecord, GateStatus
from ..contracts.task_catalog import DEFAULT_TASK_CATALOG
from ..core.errors import RegistryError
from ..core.registry import ParameterRegistry
from ..runtime.task_artifacts import LoadedTaskArtifact, TaskArtifactStore, load_committed_task_artifact
from .sampling import DrawStreamManifest, SamplingPlan, STREAM_NAMES
from .stage2_registry_qualification import RegistryQualificationError, load_asset_resolution_input
from .stage23_task_runners import (
    _predecessor_context,
    validate_formal_s203_payloads,
    validate_formal_s203_task_artifacts,
)
from .stage2_assets import (
    CheckpointFile,
    FORMAL_CHECKPOINT_SELECTION,
    FORMAL_TOTAL_TRAINING_STEPS,
    AssetResolutionManifest,
    DataRangeManifest,
    validate_formal_asset_identity,
)


GATE_ID = "stage2.G2.2"
TASK_ID = "stage2.03_assets_checkpoints_and_sampling"
ARTIFACT_KINDS = ("sampling_plan", "draw_manifest", "asset_resolution", "gate_record")
ADAPTER_SCHEMA_VERSION = "stage2-g2.2-gate-adapter-v1"
FORMAL_ADAPTER_OUTPUT_DIR = "evidence/stage2/s204/formal-adapters/g2-2-r7"

AUTHORITY_EVIDENCE_REF = "evidence/stage2/s203/g2.2-assets.json"
AUTHORITY_EVIDENCE_SHA256 = "b6805b6744374e7d05f193db4d72162176b930fa1db250bc00797b3ad30528a8"
# The original authority evidence and v1 asset manifest are immutable parent
# evidence.  The r7 amendment is the only asset-resolution input consumed by
# this adapter; it materializes the coordinate-registry hashes append-only.
ASSET_REF = "manifests/stage2/s203-asset-resolution.json"
ASSET_SHA256 = "1b1609ea97974a560b4da98707eecdbbbd97010e067e9899234079ab7b6cfb20"
DATA_REF = "manifests/stage2/s203-data-range.json"
DATA_SHA256 = "a6a633d30e55351679368ded522d4b697fc676e8f4ca7f63108406524459fb14"
SELECTION_REF = "manifests/stage2/s203-selection.json"
SELECTION_SHA256 = "1035e831f63862c0d15549fe24738ffd1432e8e165628ba64b40248461dea9bd"
ASSET_DIGEST = "f57decd5cf00e69e45ab2f02c994abb202f5c614e1441acb8aebcb1807ff76ee"
AMENDMENT_REF = "manifests/stage2/s203-registry-amendment-r3.json"
AMENDMENT_SHA256 = "71c0509b092b42a461ce9a1e7397fc74279329954f11dd7c17c512020cef6d7a"
AMENDMENT_ASSET_DIGEST = "8ae41ec8ed3ee8f16eee15cce06a6b082c21bcbef6565f099fe44c2e94fcb852"
REGISTRY_INDEX_REF = "evidence/stage2/s203/formal-registry-r6/registry-index.json"
REGISTRY_INDEX_SHA256 = "d309e7b4c93b66da53eb6eb0447cb22209cd9872f0d7f0459c52f34fb7cb5c29"
REGISTRY_INDEX_HASH = "67664a376379926b5d7eb5cae52251cec30ede2c9e24b9cade1d7c21a9bcad8c"
REGISTRY_INDEX_CONFIG_REF = "evidence/stage2/s204/materialized-task-inputs-r6/configs/stage2-01/resolved-config-v2.json"
DATA_DIGEST = "df8eeac5178305d409cf6128ac5d5648567aae895592c79fa21542e84a28e0f1"
PRODUCER_COMMIT = "676ab436422b1a514bddfc1181d0645fed4de7be"

OFFLINE_SHA256 = {
    "evidence/stage2/s203/pythia-14m-step0-56079904bb80/offline-load.json": "41a42aff7ce9294a3a2cd0e074d215345e3ab760ca41a19b1bad9e0d362116c1",
    "evidence/stage2/s203/pythia-14m-step1000-5b020995bfc7/offline-load.json": "e0e41e9594c25eb85b92345a4cbc16385c8bbc52edca95a34db39bdfe3e24b92",
    "evidence/stage2/s203/pythia-14m-step71000-6a9156279d41/offline-load.json": "bb3301f2e51e550f29092d2f3514dc057216a9b7ee318604280d4a9308ee7e5d",
    "evidence/stage2/s203/pythia-31m-deduped-step0-73628c85dd9d/offline-load.json": "4eff6fea9a0984bf53cc57809a3f32db98558f4bcf1c06d6445e61a071137bb6",
    "evidence/stage2/s203/pythia-31m-deduped-step1000-dd4d3eab2b00/offline-load.json": "41a72b53b562750035d28bea01dafb426a06380d78c7a6559f24a40cb34a0ea2",
    "evidence/stage2/s203/pythia-31m-deduped-step71000-aeafbd5e62a3/offline-load.json": "c079ebab849a4d3fb94ab0b58b3384065b89d26ef39d97b0580d1a4f2ecd12b2",
}
FAILED_ATTEMPTS = (
    ("tmp/stage2/s203-hf-fixed-20260823-01/publish.log", "e9889afca02c454031dfcc842e12d37e04625c2a9cc3d4a89f385d60b43c48d4"),
    ("tmp/stage2/s203-hf-fixed-20260823-02/repair.log", "706f329e2466b99ac7ceb0ff8085b50248ce10f460a9dc963bdf9315827db30f"),
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATHS = (
    "src/param_importance_nlp/experiments/stage2_assets.py",
    "src/param_importance_nlp/experiments/sampling.py",
    "schemas/shared/stage2-asset-resolution-v1.json",
    "schemas/shared/stage2-checkpoint-manifest-v1.json",
    "schemas/shared/stage2-data-range-manifest-v1.json",
    "schemas/shared/stage2-draw-stream-manifest-v1.json",
    "schemas/shared/stage2-repetition-mapping-v1.json",
    "plan/stage2/03_assets_checkpoints_and_sampling.md",
)

_MODEL_MANIFEST_NAME = "model-manifest.json"
_SHA256SUMS_NAME = "SHA256SUMS"


class G22Blocked(RuntimeError):
    """A fail-closed formal preflight result."""


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _safe_rel(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise G22Blocked("G22_PATH_INVALID")
    path = Path(*value.split("/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise G22Blocked("G22_PATH_ESCAPE")
    return path


def _reject_symlink_chain(path: Path, root: Path) -> None:
    root = root.absolute()
    path = path.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise G22Blocked("G22_PATH_ESCAPE") from error
    current = root
    for part in (root, *relative.parts):
        current = Path(part) if part == root else current / part
        if current.is_symlink():
            raise G22Blocked(f"G22_SYMLINK_REJECTED:{current}")


def _resolve(root: Path, ref: str) -> Path:
    relative = _safe_rel(ref)
    root = root.absolute()
    path = root.joinpath(relative)
    _reject_symlink_chain(path, root)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise G22Blocked("G22_PATH_ESCAPE") from error
    return path


def _validate_dual_roots(repository_root: Path, data_root: Path) -> None:
    if repository_root == data_root or repository_root.is_relative_to(data_root) or data_root.is_relative_to(repository_root):
        raise G22Blocked("G22_DUAL_ROOTS_NESTED")
    _reject_symlink_chain(repository_root, repository_root.parent)
    _reject_symlink_chain(data_root, data_root.parent)


def _load_hashed(root: Path, ref: str, expected: str) -> dict[str, JSONValue]:
    path = _resolve(root, ref)
    if not path.is_file():
        raise G22Blocked(f"G22_MISSING:{ref}")
    size, observed = _sha256(path)
    if observed != expected:
        raise G22Blocked(f"G22_SHA256_MISMATCH:{ref}")
    value = load_canonical_json(path)
    if not isinstance(value, dict) or canonical_json_hash(value) != expected:
        raise G22Blocked(f"G22_CANONICAL_HASH_MISMATCH:{ref}")
    return value


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise G22Blocked(f"G22_GIT_FAILED:{' '.join(args)}")
    return result.stdout.strip()


def _git_identity(repository_root: Path) -> dict[str, JSONValue]:
    if not repository_root.is_dir():
        raise G22Blocked("G22_REPOSITORY_ROOT_MISSING")
    _reject_symlink_chain(repository_root, repository_root.parent)
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise G22Blocked("G22_REPOSITORY_DIRTY")
    head = _git(repository_root, "rev-parse", "HEAD")
    tree = _git(repository_root, "rev-parse", "HEAD^{tree}")
    if not _HEX40.fullmatch(head) or not _HEX40.fullmatch(tree):
        raise G22Blocked("G22_GIT_OBJECT_INVALID")
    sources: dict[str, JSONValue] = {}
    for ref in _SOURCE_PATHS + ("src/param_importance_nlp/experiments/stage2_g22_adapter.py",):
        path = _resolve(repository_root, ref)
        if not path.is_file():
            raise G22Blocked(f"G22_SOURCE_MISSING:{ref}")
        raw = path.read_bytes()
        shown = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"HEAD:{ref}"],
            check=False,
            capture_output=True,
        )
        if shown.returncode or shown.stdout != raw:
            raise G22Blocked(f"G22_SOURCE_WORKTREE_DRIFT:{ref}")
        blob = _git(repository_root, "rev-parse", f"HEAD:{ref}")
        sources[ref] = {"sha256": hashlib.sha256(raw).hexdigest(), "git_blob": blob}
    return {"head": head, "tree": tree, "sources": sources}


def _producer_identity(repository_root: Path, producer: str) -> dict[str, JSONValue]:
    if not _HEX40.fullmatch(producer):
        raise G22Blocked("G22_PRODUCER_COMMIT_INVALID")
    _git(repository_root, "cat-file", "-e", f"{producer}^{{commit}}")
    current = _git(repository_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(repository_root), "merge-base", "--is-ancestor", producer, current],
        check=False,
    )
    if ancestor.returncode:
        raise G22Blocked("G22_PRODUCER_NOT_ANCESTOR")
    source_objects: dict[str, JSONValue] = {}
    for ref in _SOURCE_PATHS:
        producer_blob = _git(repository_root, "rev-parse", f"{producer}:{ref}")
        current_blob = _git(repository_root, "rev-parse", f"HEAD:{ref}")
        if producer_blob != current_blob:
            raise G22Blocked(f"G22_PRODUCER_SOURCE_DRIFT:{ref}")
        source_objects[ref] = producer_blob
    return {"commit": producer, "mode": "same_commit" if producer == current else "ancestor", "source_git_blobs": source_objects}


def _validate_task_inputs(root: Path, refs: Mapping[str, str]) -> tuple[dict[str, LoadedTaskArtifact], str]:
    try:
        return validate_formal_s203_task_artifacts(root, refs)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise G22Blocked(f"G22_S203_FORMAL_INPUT_INVALID:{type(error).__name__}") from error


def _verify_s203_lineage(
    root: Path,
    loaded: Mapping[str, LoadedTaskArtifact],
    config: ResolvedConfigV2,
    output_dir: str,
) -> tuple[str, str]:
    """Verify producer lineage against the resolved config without rewriting it.

    ``source_refs`` are the canonical task-output commits declared by the
    resolved config.  In particular, an external asset manifest is payload
    evidence, not an implicit predecessor.  The old adapter removed a known
    manifest reference and then fed the edited list into ``_predecessor_context``;
    that made a source-list mismatch invisible to the consumer.
    """
    orchestration = config.section("orchestration")
    if not isinstance(orchestration, Mapping):
        raise G22Blocked("G22_RESOLVED_CONFIG_ORCHESTRATION_INVALID")
    configured = orchestration.get("input_result_refs")
    if not isinstance(configured, list) or not configured or any(
        not isinstance(ref, str) for ref in configured
    ):
        raise G22Blocked("G22_RESOLVED_CONFIG_INPUT_REFS_INVALID")
    configured_refs = tuple(configured)
    source_sets = {item.source_refs for item in loaded.values()}
    if len(source_sets) != 1:
        raise G22Blocked("G22_S203_SOURCE_REFS_NOT_SHARED")
    source_refs = next(iter(source_sets))
    if source_refs != configured_refs:
        raise G22Blocked("G22_S203_SOURCE_REFS_CONFIG_MISMATCH")

    # Construct the same request shape used by the producer, but keep the
    # resolved config's input_result_refs untouched so the shared verifier
    # validates those exact canonical commits.
    request = SimpleNamespace(
        config=config,
        task=DEFAULT_TASK_CATALOG.get(TASK_ID),
        environment=SimpleNamespace(),
    )
    context = _predecessor_context(request, root, TaskArtifactStore(root, output_dir))
    return context.binding_hash, canonical_json_hash(context.payload("preregistration"))


def _validate_sampling_replay(payloads: Mapping[str, Mapping[str, object]]) -> dict[str, JSONValue]:
    """Replay every formal stream from the published plan.

    Parsing a ``DrawStreamManifest`` only proves that its internal hashes are
    self-consistent.  A malicious or stale manifest can still be internally
    consistent while carrying different sample IDs or generator boundaries.
    Recreate each interval from the frozen ``SamplingPlan`` and compare the
    complete manifest, including state digests and generated draw IDs.
    """
    plan = SamplingPlan.from_mapping(payloads["sampling_plan"])
    draw = payloads["draw_manifest"]
    streams = draw.get("stream_manifests")
    if not isinstance(streams, Mapping) or set(streams) != set(STREAM_NAMES):
        raise G22Blocked("G22_SAMPLING_STREAM_SET_INVALID")
    replayed: list[dict[str, JSONValue]] = []
    for stream in STREAM_NAMES:
        raw = streams.get(stream)
        if not isinstance(raw, Mapping):
            raise G22Blocked(f"G22_SAMPLING_STREAM_NOT_OBJECT:{stream}")
        observed = DrawStreamManifest.from_manifest(raw)
        if observed.sampling_plan_hash != plan.digest:
            raise G22Blocked(f"G22_SAMPLING_PLAN_HASH_MISMATCH:{stream}")
        start = observed.stream_state.start_position
        count = observed.stream_state.end_position - start
        expected = plan.draw_manifest(stream, count, start=start)
        if observed.to_manifest() != expected.to_manifest():
            raise G22Blocked(f"G22_SAMPLING_REPLAY_MISMATCH:{stream}")
        replayed.append(
            {
                "stream": stream,
                "start_position": start,
                "end_position": observed.stream_state.end_position,
                "draw_ids": [item.draw_id for item in expected.draws],
                "state_before_sha256": expected.stream_state.state_before_sha256,
                "state_after_sha256": expected.stream_state.state_after_sha256,
            }
        )
    return {"sampling_plan_hash": plan.digest, "streams": replayed}


def _validate_amendment(
    root: Path,
    parent_asset: AssetResolutionManifest,
) -> tuple[AssetResolutionManifest, dict[str, dict[str, str]], dict[str, JSONValue]]:
    """Load the r7 amendment and expose its old->new registry bindings.

    ``load_asset_resolution_input`` is the canonical amendment validator.  The
    extra checks here intentionally retain the amendment envelope and its
    qualification refs for the Gate evidence instead of flattening the old
    provider hash into the materialized manifest.
    """
    amendment = _load_hashed(root, AMENDMENT_REF, AMENDMENT_SHA256)
    if parent_asset.digest != ASSET_DIGEST:
        raise G22Blocked("G22_ASSET_AMENDMENT_PARENT_DIGEST_INVALID")
    try:
        materialized_value = load_asset_resolution_input(
            _resolve(root, AMENDMENT_REF), root=root, data_root=root
        )
    except (RegistryQualificationError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise G22Blocked(f"G22_ASSET_AMENDMENT_INVALID:{type(error).__name__}") from error
    if dict(materialized_value) != amendment.get("materialized_asset_resolution"):
        raise G22Blocked("G22_ASSET_AMENDMENT_MATERIALIZED_DRIFT")
    parent = amendment.get("parent")
    expected_parent = {
        "asset_resolution_ref": ASSET_REF,
        "asset_resolution_sha256": ASSET_SHA256,
        "asset_resolution_size_bytes": 14248,
        "asset_resolution_hash": ASSET_DIGEST,
    }
    if parent != expected_parent:
        raise G22Blocked("G22_ASSET_AMENDMENT_PARENT_NOT_APPEND_ONLY")
    if amendment.get("qualification_index") is None or not isinstance(
        amendment["qualification_index"], Mapping
    ):
        raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_INDEX_INVALID")
    index_meta = amendment["qualification_index"]
    if set(index_meta) != {"ref", "sha256", "size_bytes", "index_hash"}:
        raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_INDEX_FIELDS_INVALID")
    index_ref = index_meta.get("ref")
    index_sha = index_meta.get("sha256")
    if not isinstance(index_ref, str) or not isinstance(index_sha, str):
        raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_INDEX_REF_INVALID")
    index_value = _load_hashed(root, index_ref, index_sha)
    if index_value.get("index_hash") != index_meta.get("index_hash"):
        raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_INDEX_HASH_INVALID")
    raw_cells = amendment.get("qualification_cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 6:
        raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_CELLS_INVALID")
    asset = AssetResolutionManifest.from_mapping(materialized_value)
    if asset.digest != AMENDMENT_ASSET_DIGEST:
        raise G22Blocked("G22_ASSET_AMENDMENT_ASSET_DIGEST_INVALID")
    by_checkpoint = {item.checkpoint_id: item for item in asset.checkpoints}
    bindings: dict[str, dict[str, str]] = {}
    qualification_refs: list[str] = []
    for row in raw_cells:
        if not isinstance(row, Mapping) or set(row) != {
            "cell_id", "checkpoint_id", "qualification_ref", "qualification_sha256",
            "qualification_size_bytes", "qualification_hash",
            "provider_derived_registry_hash", "registry_hash", "parameter_count",
            "parameter_numel",
        }:
            raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_CELL_FIELDS_INVALID")
        checkpoint_id = row.get("checkpoint_id")
        cell_id = row.get("cell_id")
        qualification_ref = row.get("qualification_ref")
        qualification_sha = row.get("qualification_sha256")
        provider_hash = row.get("provider_derived_registry_hash")
        registry_hash = row.get("registry_hash")
        if not all(isinstance(value, str) for value in (checkpoint_id, cell_id, qualification_ref, qualification_sha, provider_hash, registry_hash)):
            raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_CELL_TYPES_INVALID")
        checkpoint = by_checkpoint.get(checkpoint_id)
        if checkpoint is None or cell_id != f"{checkpoint.model_id}:{checkpoint.training_stage}":
            raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_CELL_ID_INVALID")
        if registry_hash != checkpoint.parameter_registry_hash:
            raise G22Blocked("G22_ASSET_AMENDMENT_REGISTRY_HASH_NOT_MATERIALIZED")
        if checkpoint_id in bindings:
            raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_CELL_DUPLICATE")
        _load_hashed(root, qualification_ref, qualification_sha)
        bindings[checkpoint_id] = {
            "cell_id": cell_id,
            "provider_derived_registry_hash": provider_hash,
            "registry_hash": registry_hash,
            "qualification_ref": qualification_ref,
            "qualification_sha256": qualification_sha,
        }
        qualification_refs.append(qualification_ref)
    if set(bindings) != set(by_checkpoint):
        raise G22Blocked("G22_ASSET_AMENDMENT_QUALIFICATION_CELL_SET_INVALID")
    return asset, bindings, {
        "ref": AMENDMENT_REF,
        "sha256": AMENDMENT_SHA256,
        "asset_resolution_hash": asset.digest,
        "qualification_index": dict(index_meta),
        "qualification_refs": qualification_refs,
    }


def _canonical_registry_manifest_ref(declared_ref: str) -> str:
    """Resolve an r6 index manifest ref relative to the index directory."""
    if not isinstance(declared_ref, str) or not declared_ref or "\\" in declared_ref:
        raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_REF_INVALID")
    parts = declared_ref.split("/")
    relative = PurePosixPath(declared_ref)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or (parts and parts[0].endswith(":"))
    ):
        raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_REF_ESCAPE")
    joined = (PurePosixPath(REGISTRY_INDEX_REF).parent / relative).as_posix()
    # Keep the generated ref inside DATA_ROOT and reject any future change to
    # the index constant that could make this join non-relative.
    if joined.startswith("/") or any(part in {"", ".", ".."} for part in joined.split("/")):
        raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_REF_ESCAPE")
    return joined


def _validate_formal_registry_index(
    root: Path,
    manifest: AssetResolutionManifest,
    bindings: Mapping[str, Mapping[str, str]],
) -> list[dict[str, JSONValue]]:
    """Validate the formal r6 registry index and all six source manifests."""
    index = _load_hashed(root, REGISTRY_INDEX_REF, REGISTRY_INDEX_SHA256)
    required_index = {
        "allowed_s203_artifact_kinds", "asset_resolution_artifact_kind", "asset_resolution_hash",
        "cells", "index_hash", "producer", "registry_manifests_are_source_artifacts",
        "resolved_config", "schema_version", "scope", "source_artifact_refs", "task_id",
    }
    if set(index) != required_index or index.get("schema_version") != "stage2-parameter-registry-index-v1" or index.get("task_id") != TASK_ID or index.get("scope") != "formal" or index.get("asset_resolution_artifact_kind") != "asset_resolution" or index.get("asset_resolution_hash") != manifest.digest or index.get("allowed_s203_artifact_kinds") != list(ARTIFACT_KINDS) or index.get("registry_manifests_are_source_artifacts") is not True or index.get("index_hash") != REGISTRY_INDEX_HASH:
        raise G22Blocked("G22_FORMAL_REGISTRY_INDEX_IDENTITY_INVALID")
    index_body = dict(index)
    index_body.pop("index_hash", None)
    if canonical_json_hash(index_body) != REGISTRY_INDEX_HASH:
        raise G22Blocked("G22_FORMAL_REGISTRY_INDEX_HASH_INVALID")
    resolved_config = index.get("resolved_config")
    if not isinstance(resolved_config, Mapping) or set(resolved_config) != {
        "config_hash", "file_sha256", "file_size_bytes", "full_hash", "path", "payload_sha256"
    }:
        raise G22Blocked("G22_FORMAL_REGISTRY_RESOLVED_CONFIG_FIELDS_INVALID")
    config_path = resolved_config.get("path")
    if not isinstance(config_path, str) or not config_path.endswith("/" + REGISTRY_INDEX_CONFIG_REF):
        raise G22Blocked("G22_FORMAL_REGISTRY_RESOLVED_CONFIG_PATH_INVALID")
    config_sha = resolved_config.get("file_sha256")
    if not isinstance(config_sha, str) or resolved_config.get("payload_sha256") != config_sha or not isinstance(resolved_config.get("file_size_bytes"), int):
        raise G22Blocked("G22_FORMAL_REGISTRY_RESOLVED_CONFIG_DIGEST_INVALID")
    config_file = _resolve(root, REGISTRY_INDEX_CONFIG_REF)
    config_size, config_observed_sha = _sha256(config_file)
    if config_size != resolved_config["file_size_bytes"] or config_observed_sha != config_sha:
        raise G22Blocked("G22_FORMAL_REGISTRY_RESOLVED_CONFIG_FILE_INVALID")
    try:
        indexed_config = ResolvedConfigV2.from_mapping(load_canonical_json(config_file))
    except (TypeError, ValueError, KeyError) as error:
        raise G22Blocked("G22_FORMAL_REGISTRY_RESOLVED_CONFIG_INVALID") from error
    if resolved_config.get("config_hash") != indexed_config.config_hash or resolved_config.get("full_hash") != indexed_config.full_hash:
        raise G22Blocked("G22_FORMAL_REGISTRY_RESOLVED_CONFIG_HASH_INVALID")
    rows = index.get("cells")
    if not isinstance(rows, list) or len(rows) != 6 or not isinstance(index.get("source_artifact_refs"), list):
        raise G22Blocked("G22_FORMAL_REGISTRY_INDEX_CELLS_INVALID")
    expected_checkpoints = list(manifest.checkpoints)
    expected_source_refs: list[str] = []
    result: list[dict[str, JSONValue]] = []
    for checkpoint, raw in zip(expected_checkpoints, rows, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {"cell_id", "manifest_ref", "manifest_sha256", "manifest_size_bytes", "registry_hash"}:
            raise G22Blocked("G22_FORMAL_REGISTRY_CELL_FIELDS_INVALID")
        cell_id = f"{checkpoint.model_id}:{checkpoint.training_stage}"
        if raw.get("cell_id") != cell_id or raw.get("registry_hash") != checkpoint.parameter_registry_hash:
            raise G22Blocked("G22_FORMAL_REGISTRY_CELL_BINDING_INVALID")
        binding = bindings.get(checkpoint.checkpoint_id)
        if binding is None or binding.get("cell_id") != cell_id or binding.get("registry_hash") != raw.get("registry_hash"):
            raise G22Blocked("G22_FORMAL_REGISTRY_AMENDMENT_CROSS_BIND_INVALID")
        declared_ref = raw.get("manifest_ref")
        sha = raw.get("manifest_sha256")
        if not isinstance(declared_ref, str) or not isinstance(sha, str) or not isinstance(raw.get("manifest_size_bytes"), int):
            raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_REF_INVALID")
        ref = _canonical_registry_manifest_ref(declared_ref)
        value = _load_hashed(root, ref, sha)
        manifest_path = _resolve(root, ref)
        manifest_size, _ = _sha256(manifest_path)
        if manifest_size != raw["manifest_size_bytes"]:
            raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_SIZE_INVALID")
        required_manifest = {
            "actual_files", "actual_manifest", "asset_resolution_hash", "cell", "checkpoint",
            "eligible_parameter_count", "eligible_parameter_numel", "manifest_hash", "model",
            "parameter_count", "parameter_mapping", "parameter_names", "parameter_numel",
            "parameter_order", "producer", "registry", "registry_hash", "resolved_config",
            "schema_version", "scope", "task_id",
        }
        if set(value) != required_manifest or value.get("schema_version") != "stage2-parameter-registry-manifest-v1" or value.get("task_id") != TASK_ID or value.get("scope") != "formal" or value.get("asset_resolution_hash") != manifest.digest:
            raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_SCHEMA_INVALID")
        manifest_body = dict(value)
        declared_manifest_hash = manifest_body.pop("manifest_hash")
        if not isinstance(declared_manifest_hash, str) or canonical_json_hash(manifest_body) != declared_manifest_hash:
            raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_HASH_INVALID")
        cell = value.get("cell")
        model = value.get("model")
        checkpoint_value = value.get("checkpoint")
        if cell != {"cell_id": cell_id, "model_id": checkpoint.model_id, "training_stage": checkpoint.training_stage, "training_step": checkpoint.training_step} or model != {"model_id": checkpoint.model_id, "repository": checkpoint.repository} or checkpoint_value != {"checkpoint_id": checkpoint.checkpoint_id, "revision": checkpoint.revision, "root_ref": checkpoint.root_ref, "training_step": checkpoint.training_step}:
            raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_CHECKPOINT_BINDING_INVALID")
        if value.get("registry_hash") != raw.get("registry_hash"):
            raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_REGISTRY_HASH_INVALID")
        try:
            registry = ParameterRegistry.from_manifest(value["registry"])
        except (RegistryError, TypeError, ValueError) as error:
            raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_REGISTRY_INVALID") from error
        if registry.coordinate_registry_hash != raw.get("registry_hash") or value.get("parameter_count") != len(registry) or value.get("parameter_numel") != sum(item.numel for item in registry):
            raise G22Blocked("G22_FORMAL_REGISTRY_MANIFEST_REGISTRY_CROSS_BIND_INVALID")
        expected_source_refs.append(declared_ref)
        result.append({"ref": ref, "sha256": sha, "size_bytes": raw["manifest_size_bytes"], "cell_id": cell_id, "registry_hash": str(raw["registry_hash"])})
    if index["source_artifact_refs"] != expected_source_refs:
        raise G22Blocked("G22_FORMAL_REGISTRY_SOURCE_REFS_INVALID")
    return result


def _cross_bind_offline_registry_hash(
    checkpoint_id: str,
    materialized_registry_hash: str | None,
    offline_provider_hash: object,
    bindings: Mapping[str, Mapping[str, str]],
) -> tuple[str, str]:
    """Return the explicit provider-derived -> materialized hash binding."""
    binding = bindings.get(checkpoint_id)
    if binding is None or binding.get("registry_hash") != materialized_registry_hash:
        raise G22Blocked("G22_OFFLINE_REGISTRY_MATERIALIZED_HASH_INVALID")
    provider_hash = binding.get("provider_derived_registry_hash")
    if not isinstance(provider_hash, str) or offline_provider_hash != provider_hash:
        raise G22Blocked("G22_OFFLINE_PROVIDER_REGISTRY_HASH_CROSS_BIND_INVALID")
    return provider_hash, str(binding["registry_hash"])


def _validate_offline(
    root: Path,
    manifest: AssetResolutionManifest,
    registry_bindings: Mapping[str, Mapping[str, str]],
) -> list[dict[str, JSONValue]]:
    result: list[dict[str, JSONValue]] = []
    expected_pairs = {(model, stage): pair for (model, stage), pair in FORMAL_CHECKPOINT_SELECTION.items()}
    for item in manifest.checkpoints:
        if not item.load_evidence_ref or item.load_evidence_ref not in OFFLINE_SHA256 or item.load_evidence_sha256 != OFFLINE_SHA256[item.load_evidence_ref]:
            raise G22Blocked("G22_OFFLINE_LOAD_REF_OR_HASH_INVALID")
        value = _load_hashed(root, item.load_evidence_ref, item.load_evidence_sha256)
        required = {"architecture", "backward_smoke", "config_sha256", "cuda_device_excluded", "forward_smoke", "model_global_rng_untouched", "model_id", "model_state_hash", "model_state_unchanged", "model_type", "offline", "parameter_count", "parameter_dtypes", "parameter_registry_hash", "repository", "revision", "root_ref", "schema_version", "tokenizer_length", "tokenizer_sha256", "tokenizer_vocab_size", "training_stage", "training_step"}
        if set(value) != required or value["schema_version"] != "stage2-checkpoint-offline-load-v1" or value["offline"] is not True or value["cuda_device_excluded"] is not True or value["model_global_rng_untouched"] is not True or value["model_state_unchanged"] is not True:
            raise G22Blocked("G22_OFFLINE_LOAD_SCHEMA_OR_FLAGS_INVALID")
        pair = expected_pairs.get((item.model_id, item.training_stage))
        if pair is None or value["training_step"] != pair[0] or value["revision"] != pair[1] or value["model_id"] != item.model_id or value["repository"] != item.repository or value["training_stage"] != item.training_stage or value["root_ref"] != item.root_ref:
            raise G22Blocked("G22_OFFLINE_LOAD_IDENTITY_MISMATCH")
        config_files = tuple(file for file in item.files if file.role == "config")
        tokenizer_files = tuple(file for file in item.files if file.role == "tokenizer")
        if len(config_files) != 1 or len(tokenizer_files) != 1:
            raise G22Blocked("G22_CHECKPOINT_CONFIG_TOKENIZER_ROLE_INVALID")
        config_file, tokenizer_file = config_files[0], tokenizer_files[0]
        if (
            item.config_sha256 != config_file.sha256
            or item.tokenizer_sha256 != tokenizer_file.sha256
            or value["config_sha256"] != config_file.sha256
            or value["tokenizer_sha256"] != tokenizer_file.sha256
        ):
            raise G22Blocked("G22_OFFLINE_CONFIG_TOKENIZER_CROSS_BIND_INVALID")
        provider_hash, materialized_hash = _cross_bind_offline_registry_hash(
            item.checkpoint_id,
            item.parameter_registry_hash,
            value["parameter_registry_hash"],
            registry_bindings,
        )
        for field in ("model_state_hash", "parameter_registry_hash", "config_sha256", "tokenizer_sha256"):
            if not isinstance(value[field], str) or not _HEX64.fullmatch(value[field]):
                raise G22Blocked(f"G22_OFFLINE_LOAD_DIGEST_INVALID:{field}")
        expected_count = 14067712 if item.model_id == "pythia-14m" else 30494720
        if value["architecture"] != ["GPTNeoXForCausalLM"] or value["model_type"] != "gpt_neox" or value["parameter_dtypes"] != ["torch.float32"] or value["parameter_count"] != expected_count or value["tokenizer_length"] != 50277 or value["tokenizer_vocab_size"] != 50254:
            raise G22Blocked("G22_OFFLINE_LOAD_MODEL_SCHEMA_INVALID")
        forward = value["forward_smoke"]
        backward = value["backward_smoke"]
        if not isinstance(forward, Mapping) or not isinstance(backward, Mapping) or forward.get("finite") is not True or backward.get("finite") is not True or forward.get("logits_shape") != [1, 8, 50304] or forward.get("sample_id") != 0:
            raise G22Blocked("G22_OFFLINE_LOAD_SMOKE_INVALID")
        result.append({"ref": item.load_evidence_ref, "sha256": item.load_evidence_sha256, "model": item.model_id, "stage": item.training_stage, "step": item.training_step, "provider_derived_registry_hash": provider_hash, "registry_hash": materialized_hash})
    if len(result) != 6 or len({item["ref"] for item in result}) != 6:
        raise G22Blocked("G22_OFFLINE_LOAD_COUNT_INVALID")
    by_model: dict[str, set[str]] = {}
    for item in result:
        by_model.setdefault(str(item["model"]), set()).add(str(item["registry_hash"]))
    if set(by_model) != {"pythia-14m", "pythia-31m-deduped"} or any(len(values) != 1 for values in by_model.values()):
        raise G22Blocked("G22_OFFLINE_REGISTRY_MODEL_CROSS_BIND_INVALID")
    return result


def _validate_checkpoint_manifest_files(
    files: object,
    expected_checkpoint_files: Sequence[CheckpointFile],
) -> list[str]:
    """Validate a model manifest's exact file inventory and LFS declarations."""
    if not isinstance(files, list) or len(files) != len(expected_checkpoint_files):
        raise G22Blocked("G22_MODEL_MANIFEST_FILES_INVALID")
    expected_files = {
        item.path: (item.size_bytes, item.sha256, item.role)
        for item in expected_checkpoint_files
    }
    listed_names: list[str] = []
    for file in files:
        if not isinstance(file, Mapping) or set(file) != {
            "name", "official_lfs_sha256", "sha256", "size_bytes"
        }:
            raise G22Blocked("G22_MODEL_MANIFEST_FILE_SCHEMA_INVALID")
        name = file["name"]
        if not isinstance(name, str) or name in listed_names or name not in expected_files:
            raise G22Blocked("G22_CHECKPOINT_FILE_MISMATCH")
        expected_size, expected_sha, expected_role = expected_files[name]
        if file["size_bytes"] != expected_size or file["sha256"] != expected_sha:
            raise G22Blocked("G22_CHECKPOINT_FILE_MISMATCH")
        official_lfs_sha256 = file["official_lfs_sha256"]
        if official_lfs_sha256 is None:
            # The authoritative checkpoint role is the LFS semantic.  Weight
            # blobs must carry their provider LFS object id; small config and
            # tokenizer files are Git-managed and canonically use null.
            if expected_role == "weights":
                raise G22Blocked("G22_CHECKPOINT_FILE_MISMATCH")
        elif not isinstance(official_lfs_sha256, str) or not _HEX64.fullmatch(
            official_lfs_sha256
        ):
            raise G22Blocked("G22_CHECKPOINT_FILE_MISMATCH")
        listed_names.append(name)
    if set(listed_names) != set(expected_files):
        raise G22Blocked("G22_MODEL_MANIFEST_FILE_SET_MISMATCH")
    return listed_names


def _validate_checkpoint_sidecars(
    *,
    model_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    expected_file_bytes: Mapping[str, tuple[int, str]],
) -> None:
    """Bind the two known model-root sidecars without widening the payload set."""
    try:
        manifest_rel = manifest_path.relative_to(model_root).as_posix()
    except ValueError as error:
        raise G22Blocked("G22_CHECKPOINT_MANIFEST_PATH_INVALID") from error
    if manifest_rel != _MODEL_MANIFEST_NAME:
        raise G22Blocked("G22_CHECKPOINT_MANIFEST_PATH_INVALID")

    sums_path = model_root / _SHA256SUMS_NAME
    if sums_path.is_symlink() or not sums_path.is_file():
        raise G22Blocked("G22_CHECKPOINT_SHA256SUMS_MISSING")
    try:
        lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise G22Blocked("G22_CHECKPOINT_SHA256SUMS_FORMAT_INVALID") from error
    expected = [
        (name, digest) for name, (_size, digest) in expected_file_bytes.items()
    ] + [(manifest_rel, manifest_sha256)]
    observed: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^ \t].*)", line)
        if match is None or match.group(2).startswith("*"):
            raise G22Blocked("G22_CHECKPOINT_SHA256SUMS_FORMAT_INVALID")
        observed.append((match.group(2), match.group(1).lower()))
    if observed != expected:
        raise G22Blocked("G22_CHECKPOINT_SHA256SUMS_MISMATCH")


def _validate_real_assets(
    root: Path,
    *,
    asset_payload: Mapping[str, object] | None = None,
) -> dict[str, JSONValue]:
    evidence = _load_hashed(root, AUTHORITY_EVIDENCE_REF, AUTHORITY_EVIDENCE_SHA256)
    expected_evidence = {"active_partial_objects_untouched", "asset_resolution_hash", "asset_resolution_ref", "checkpoint_count", "combination_smoke", "consumer_commit", "cuda_device_excluded", "data_range_hash", "data_range_ref", "execution_commit", "failed_attempts", "gate_id", "offline_load_count", "producer_commit", "schema_version", "status"}
    attempts = evidence["failed_attempts"]
    combination = evidence["combination_smoke"]
    if set(evidence) != expected_evidence or evidence["schema_version"] != "stage2-g2.2-asset-gate-evidence-v1" or evidence["status"] != "PASS" or evidence["gate_id"] != GATE_ID or evidence["checkpoint_count"] != 6 or evidence["offline_load_count"] != 6 or evidence["asset_resolution_hash"] != ASSET_DIGEST or evidence["data_range_hash"] != DATA_DIGEST or evidence["producer_commit"] != PRODUCER_COMMIT or evidence["execution_commit"] != PRODUCER_COMMIT or evidence["consumer_commit"] is not None or evidence["cuda_device_excluded"] is not True or evidence["active_partial_objects_untouched"] is not True or attempts != [{"ref": ref, "sha256": sha} for ref, sha in FAILED_ATTEMPTS]:
        raise G22Blocked("G22_AUTHORITY_EVIDENCE_STATUS_INVALID")
    if combination != {"all_registry_hashes_present": True, "cell_count": 6, "registry_hash_count": 2, "status": "PASS"}:
        raise G22Blocked("G22_AUTHORITY_COMBINATION_SMOKE_INVALID")
    asset_value = _load_hashed(root, ASSET_REF, ASSET_SHA256)
    data_value = _load_hashed(root, DATA_REF, DATA_SHA256)
    selection_value = _load_hashed(root, SELECTION_REF, SELECTION_SHA256)
    asset = AssetResolutionManifest.from_mapping(asset_value)
    data = DataRangeManifest.from_mapping(data_value)
    validate_formal_asset_identity(asset)
    if asset.digest != ASSET_DIGEST or data.digest != DATA_DIGEST or asset.data_range.digest != data.digest:
        raise G22Blocked("G22_MANIFEST_DIGEST_MISMATCH")
    if asset.producer_commit != PRODUCER_COMMIT or asset.execution_commit != PRODUCER_COMMIT or asset.consumer_commit is not None or asset.data_range != data:
        raise G22Blocked("G22_MANIFEST_PROVENANCE_INVALID")
    if selection_value.get("schema_version") != "stage2-checkpoint-selection-v1" or selection_value.get("total_training_steps") != FORMAL_TOTAL_TRAINING_STEPS or selection_value.get("selection_rule") != "nearest checkpoint to target fraction; ties choose earlier step":
        raise G22Blocked("G22_SELECTION_MANIFEST_INVALID")
    rows = selection_value.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise G22Blocked("G22_SELECTION_ROWS_INVALID")
    expected_rows = [(model, stage, step, rev) for (model, stage), (step, rev) in FORMAL_CHECKPOINT_SELECTION.items()]
    observed_rows = [(row.get("model_id"), row.get("training_stage"), row.get("training_step"), row.get("revision")) for row in rows if isinstance(row, Mapping)]
    if observed_rows != expected_rows:
        raise G22Blocked("G22_SELECTION_IDENTITY_DRIFT")
    materialized, registry_bindings, amendment = _validate_amendment(root, asset)
    if asset_payload is None or asset_payload.get("stage2_asset_manifest") != materialized.to_dict():
        raise G22Blocked("G22_S203_ASSET_PAYLOAD_NOT_AMENDMENT_MATERIALIZED")
    if materialized.data_range != data or materialized.data_range.digest != DATA_DIGEST:
        raise G22Blocked("G22_AMENDMENT_DATA_RANGE_CROSS_BIND_INVALID")
    registry_manifests = _validate_formal_registry_index(root, materialized, registry_bindings)
    offline = _validate_offline(root, materialized, registry_bindings)
    model_manifests: list[dict[str, JSONValue]] = []
    for checkpoint in materialized.checkpoints:
        if checkpoint.manifest_ref is None or checkpoint.manifest_sha256 is None or checkpoint.revision is None or checkpoint.state != "ready":
            raise G22Blocked("G22_CHECKPOINT_MANIFEST_INCOMPLETE")
        manifest_path = _resolve(root, checkpoint.manifest_ref)
        size, digest = _sha256(manifest_path)
        if digest != checkpoint.manifest_sha256:
            raise G22Blocked("G22_CHECKPOINT_MANIFEST_SHA_MISMATCH")
        value = load_canonical_json(manifest_path)
        requested_revision = {"initialization": "step0", "early": "step1000", "mid_late": "step71000"}[checkpoint.training_stage]
        if not isinstance(value, Mapping) or value.get("schema") != "parameter-importance-model-manifest-v1" or value.get("repo") != checkpoint.repository or value.get("revision") != checkpoint.revision or value.get("requested_revision") != requested_revision:
            raise G22Blocked("G22_MODEL_MANIFEST_SCHEMA_INVALID")
        if set(value) not in ({"schema", "requested_revision", "repo", "revision", "files", "downloaded_at", "transport_endpoint"}, {"schema", "requested_revision", "repo", "revision", "files", "repair_scope"}):
            raise G22Blocked("G22_MODEL_MANIFEST_FIELDS_INVALID")
        if "transport_endpoint" in value and value["transport_endpoint"] not in {"https://huggingface.co", "https://hf-mirror.com"}:
            raise G22Blocked("G22_MODEL_MANIFEST_ENDPOINT_INVALID")
        _validate_checkpoint_manifest_files(value.get("files"), checkpoint.files)
        expected_file_bytes = {
            item.path: (item.size_bytes, item.sha256) for item in checkpoint.files
        }
        model_root = _resolve(root, checkpoint.root_ref)
        if not model_root.is_dir():
            raise G22Blocked("G22_CHECKPOINT_ROOT_MISSING")
        _validate_checkpoint_sidecars(
            model_root=model_root,
            manifest_path=manifest_path,
            manifest_sha256=checkpoint.manifest_sha256,
            expected_file_bytes=expected_file_bytes,
        )
        actual_names: list[str] = []
        for candidate in model_root.rglob("*"):
            if candidate.is_symlink():
                raise G22Blocked(f"G22_CHECKPOINT_SYMLINK_REJECTED:{candidate}")
            if candidate.is_file():
                actual_names.append(candidate.relative_to(model_root).as_posix())
            elif not candidate.is_dir():
                raise G22Blocked(f"G22_CHECKPOINT_NONREGULAR_FILE:{candidate}")
        allowed_names = set(expected_file_bytes) | {
            _MODEL_MANIFEST_NAME,
            _SHA256SUMS_NAME,
        }
        if set(actual_names) != allowed_names or len(actual_names) != len(allowed_names):
            raise G22Blocked("G22_MODEL_DIRECTORY_FILE_SET_MISMATCH")
        for name in expected_file_bytes:
            actual = _resolve(root, checkpoint.root_ref + "/" + name)
            observed_size, observed_sha = _sha256(actual)
            if (observed_size, observed_sha) != expected_file_bytes[name]:
                raise G22Blocked("G22_CHECKPOINT_FILE_BYTES_MISMATCH")
        model_manifests.append({"ref": checkpoint.manifest_ref, "sha256": checkpoint.manifest_sha256, "size_bytes": size})
    data_root = root / "datasets" / "pile-deduped-pythia-preshuffled"
    for item in materialized.data_range.files:
        actual = _resolve(data_root, item.path)
        observed_size, observed_sha = _sha256(actual)
        if (observed_size, observed_sha) != (item.size_bytes, item.sha256):
            raise G22Blocked("G22_DATA_FILE_BYTES_MISMATCH")
    prefix = _resolve(root, materialized.data_range.manifest_ref)
    psize, psha = _sha256(prefix)
    if psha != data.manifest_sha256:
        raise G22Blocked("G22_DATA_PREFIX_MANIFEST_MISMATCH")
    return {
        "evidence_ref": AUTHORITY_EVIDENCE_REF,
        "evidence_sha256": AUTHORITY_EVIDENCE_SHA256,
        "parent_asset_ref": ASSET_REF,
        "parent_asset_sha256": ASSET_SHA256,
        "parent_asset_digest": asset.digest,
        "asset_ref": AMENDMENT_REF,
        "asset_sha256": AMENDMENT_SHA256,
        "asset_digest": materialized.digest,
        "data_ref": DATA_REF,
        "data_sha256": DATA_SHA256,
        "data_digest": data.digest,
        "selection_ref": SELECTION_REF,
        "selection_sha256": SELECTION_SHA256,
        "amendment": amendment,
        "registry_index_ref": REGISTRY_INDEX_REF,
        "registry_index_sha256": REGISTRY_INDEX_SHA256,
        "registry_manifests": registry_manifests,
        "offline_loads": offline,
        "model_manifests": model_manifests,
    }


def _config(root: Path, ref: str, expected_hash: str) -> ResolvedConfigV2:
    value = load_canonical_json(_resolve(root, ref))
    if not isinstance(value, Mapping):
        raise G22Blocked("G22_RESOLVED_CONFIG_NOT_OBJECT")
    config = ResolvedConfigV2.from_mapping(value)
    if config.task_id != TASK_ID or config.run_intent != "formal" or config.formal_eligible is not True or config.config_hash != expected_hash:
        raise G22Blocked("G22_RESOLVED_CONFIG_NOT_FORMAL_OR_MISMATCH")
    return config


def _formal_adapter_output_path(data_root: Path, s203_output_dir: str) -> Path:
    """Resolve the fixed, independent G2.2 publication root.

    The S2.3 output is an input to lineage verification only.  Keeping this
    path as a code-owned constant prevents a resolved-config candidate commit
    from being mistaken for the adapter's formal GateRecord.
    """
    if s203_output_dir == FORMAL_ADAPTER_OUTPUT_DIR:
        raise G22Blocked("G22_ADAPTER_OUTPUT_DIR_COLLIDES_WITH_S203")
    path = _resolve(data_root, FORMAL_ADAPTER_OUTPUT_DIR)
    if path == data_root or not path.is_relative_to(data_root):
        raise G22Blocked("G22_ADAPTER_OUTPUT_DIR_INVALID")
    return path


def _gate(
    *,
    status: GateStatus,
    checked_at: str,
    measured: Mapping[str, JSONValue],
    refs: Sequence[str],
    reasons: Sequence[str] = (),
) -> GateRecord:
    return GateRecord(gate_id=GATE_ID, stage=2, status=status, checked_at=checked_at, measured=dict(measured), threshold={"authority_status": "PASS", "asset_resolution_hash": AMENDMENT_ASSET_DIGEST, "data_range_hash": DATA_DIGEST, "checkpoint_count": 6, "offline_load_count": 6, "formal_task_artifacts": list(ARTIFACT_KINDS)}, evidence_refs=tuple(refs), reasons=tuple(reasons))


def evaluate_formal_g22(
    *,
    repository_root: str | Path,
    data_root: str | Path,
    resolved_config_ref: str,
    s203_artifact_refs: Mapping[str, str],
    authority_evidence_ref: str = AUTHORITY_EVIDENCE_REF,
    offline_load_refs: Mapping[str, str] | None = None,
    manifest_refs: Mapping[str, str] | None = None,
    output_dir: str | None = None,
) -> dict[str, JSONValue]:
    """Validate S2.3 and publish exactly one formal G2.2 GateRecord commit.

    Missing or invalid inputs return a BLOCKED result without writing anything.
    A commit is written only after all four formal task commits, the resolved
    config, both clean roots, and all server bytes have passed validation.
    """
    checked_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    repository = Path(repository_root).absolute()
    data = Path(data_root).absolute()
    try:
        _validate_dual_roots(repository, data)
        if authority_evidence_ref != AUTHORITY_EVIDENCE_REF:
            raise G22Blocked("G22_AUTHORITY_REF_NOT_CANONICAL")
        if offline_load_refs is not None:
            if set(offline_load_refs) != set(OFFLINE_SHA256):
                raise G22Blocked("G22_OFFLINE_REF_SET_NOT_CANONICAL")
            if any(offline_load_refs[key] != key for key in OFFLINE_SHA256):
                raise G22Blocked("G22_OFFLINE_REF_ALIAS_REJECTED")
        if manifest_refs is not None:
            expected_refs = {"asset_resolution": AMENDMENT_REF, "data_range": DATA_REF, "selection": SELECTION_REF}
            if dict(manifest_refs) != expected_refs:
                raise G22Blocked("G22_MANIFEST_REF_SET_NOT_CANONICAL")
        loaded, config_hash = _validate_task_inputs(data, s203_artifact_refs)
        config = _config(data, resolved_config_ref, config_hash)
        artifacts = config.section("artifacts")
        if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("output_dir"), str):
            raise G22Blocked("G22_RESOLVED_CONFIG_OUTPUT_CONTRACT_INVALID")
        configured_output_dir = str(artifacts["output_dir"])
        if output_dir is not None and output_dir != FORMAL_ADAPTER_OUTPUT_DIR:
            raise G22Blocked("G22_ADAPTER_OUTPUT_DIR_OVERRIDE_REJECTED")
        _formal_adapter_output_path(data, configured_output_dir)
        upstream_binding_hash, preregistration_hash = _verify_s203_lineage(
            data, loaded, config, configured_output_dir
        )
        validate_formal_s203_payloads(
            {kind: item.payload for kind, item in loaded.items()},
            expected_preregistration_hash=preregistration_hash,
            expected_upstream_binding_hash=upstream_binding_hash,
        )
        replay = _validate_sampling_replay(
            {kind: item.payload for kind, item in loaded.items()}
        )
        repo_identity = _git_identity(repository)
        # The old G2.2 authority remains bound to its historical 676ab
        # producer, but the current TaskRuntime consumer must identify the
        # clean repository HEAD that produced these task artifacts.
        producer = _producer_identity(repository, str(repo_identity["head"]))
        assets = _validate_real_assets(
            data,
            asset_payload=loaded["asset_resolution"].payload,
        )
        orchestration = config.section("orchestration")
        assert isinstance(orchestration, Mapping)
        source_refs = tuple(str(ref) for ref in orchestration["input_result_refs"])
        qualification_index = assets["amendment"]["qualification_index"]
        assert isinstance(qualification_index, Mapping)
        qualification_refs = assets["amendment"]["qualification_refs"]
        assert isinstance(qualification_refs, list)
        registry_manifests = assets["registry_manifests"]
        assert isinstance(registry_manifests, list)
        evidence_refs = tuple(dict.fromkeys(
            tuple(s203_artifact_refs[k] for k in ARTIFACT_KINDS)
            + (
                AUTHORITY_EVIDENCE_REF,
                ASSET_REF,
                AMENDMENT_REF,
                str(qualification_index["ref"]),
                *(str(ref) for ref in qualification_refs),
                REGISTRY_INDEX_REF,
                *(str(item["ref"]) for item in registry_manifests),
                DATA_REF,
                SELECTION_REF,
            )
            + tuple(str(item["ref"]) for item in assets["offline_loads"])
        ))
        measured: dict[str, JSONValue] = {"adapter_schema_version": ADAPTER_SCHEMA_VERSION, "task_id": TASK_ID, "config": {"ref": resolved_config_ref, "config_hash": config.config_hash, "full_hash": config.full_hash, "run_intent": config.run_intent, "formal_eligible": config.formal_eligible}, "roots": {"repository_root": str(repository), "data_root": str(data)}, "repository": repo_identity, "producer": producer, "authority": assets, "lineage": {"upstream_binding_hash": upstream_binding_hash, "preregistration_contract_hash": preregistration_hash, "source_refs": list(source_refs)}, "sampling_replay": replay, "runtime": {"runtime": "TaskRuntime", "formal_envelope": "load_committed_task_artifact", "store": "TaskArtifactStore", "gate_schema": "gate-record-v1", "adapter_output_dir": FORMAL_ADAPTER_OUTPUT_DIR}, "input_artifacts": {kind: {"commit_ref": loaded[kind].identity.commit_ref, "artifact_hash": loaded[kind].identity.artifact_hash} for kind in ARTIFACT_KINDS}}
        gate = _gate(status=GateStatus.PASS, checked_at=checked_at, measured=measured, refs=evidence_refs)
        # The fixed logical path has already passed the DATA_ROOT boundary;
        # discovery/reuse cannot inspect the S2.3 candidate directory.
        store = TaskArtifactStore(data, FORMAL_ADAPTER_OUTPUT_DIR)
        existing = store.discover_complete(task_id=TASK_ID, config_hash=config.config_hash, artifact_kinds=("gate_record",), formal_eligible=True)
        if existing:
            loaded_gate = load_committed_task_artifact(data, existing["gate_record"], require_formal=True)
            if loaded_gate.source_refs != source_refs:
                raise G22Blocked("G22_EXISTING_GATE_SOURCE_REFS_DRIFT")
            previous = GateRecord.from_mapping(dict(loaded_gate.payload))
            previous_semantic = previous.to_dict()
            current_semantic = gate.to_dict()
            previous_semantic.pop("checked_at", None)
            previous_semantic.pop("artifact_hash", None)
            current_semantic.pop("checked_at", None)
            current_semantic.pop("artifact_hash", None)
            if previous_semantic != current_semantic:
                raise G22Blocked("G22_EXISTING_GATE_SEMANTIC_DRIFT")
            return {"schema_version": ADAPTER_SCHEMA_VERSION, "gate_record": previous.to_dict(), "status": previous.status.value, "formal_eligible": True, "commit_ref": existing["gate_record"], "reused": True}
        published = store.publish(task_id=TASK_ID, artifact_kind="gate_record", config_hash=config.config_hash, run_intent="formal", payload=gate.to_dict(), formal_eligible=True, source_refs=source_refs)
        return {"schema_version": ADAPTER_SCHEMA_VERSION, "gate_record": gate.to_dict(), "status": "PASS", "formal_eligible": True, "commit_ref": published.commit_ref, "reused": False}
    except (G22Blocked, FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        reason = str(error) or type(error).__name__
        blocked = _gate(status=GateStatus.BLOCKED, checked_at=checked_at, measured={"adapter_schema_version": ADAPTER_SCHEMA_VERSION, "roots": {"repository_root": str(repository), "data_root": str(data)}}, refs=(), reasons=(reason,))
        return {"schema_version": ADAPTER_SCHEMA_VERSION, "gate_record": blocked.to_dict(), "status": "BLOCKED", "formal_eligible": False, "commit_ref": None, "reused": False, "reason": reason}


evaluate_g22 = evaluate_formal_g22
