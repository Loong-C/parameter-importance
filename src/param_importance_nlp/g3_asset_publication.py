"""Canonical, no-clobber publication of the thirteen Stage 0 G3 assets.

This module turns the frozen requirements/layout control plane plus already
materialized local assets into VERIFIED candidates, semantic evidence,
qualifications, and immutable READY manifests.  It performs no acquisition
and deliberately exposes no network-enabled fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import struct
import subprocess
import tempfile
from typing import Any, Final

from .asset_layout import (
    load_stage0_asset_layout,
    validate_stage0_asset_layout,
)
from .asset_requirements import (
    load_stage0_asset_requirements,
    validate_stage0_asset_requirements,
)
from .assets import (
    AssetActorRole,
    AssetFile,
    AssetState,
    admit_g3_qualification,
    build_g3_qualification,
    resolve_qualified_asset,
    transition_manifest,
    validate_asset_path,
    validate_g3_manifest,
    validate_g3_qualification,
)
from .atomic import sha256_file
from .contracts import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_hash,
    ensure_json_object,
    load_canonical_json,
    loads_strict_json,
)
from .g3_gate import (
    G3GateAggregationError,
    GLUE_PREPROCESSING_VERSION,
    g3_candidate_artifact_ref,
    glue_preprocessing_config_hash,
    load_legacy_model_manifest_diagnostic,
    load_pile_reference_reader_oracle,
    qualification_check_ids_for,
    semantic_observations_match,
)
from .g3_semantic_evidence import (
    SEMANTIC_EVIDENCE_SCHEMA_VERSION,
    semantic_evidence_artifact_hash as _semantic_evidence_artifact_hash,
    validate_semantic_evidence as _validate_semantic_evidence,
)


SCHEMA_VERSION: Final = "stage0-g3-asset-publication-v1"
GENERATOR_VERSION: Final = "stage0-g3-asset-publication-v1"
_EXPECTED_ENTRY_COUNT: Final = 13
_EXPECTED_PREREQUISITE_COUNT: Final = 10
_EXPECTED_DERIVED_COUNT: Final = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class G3AssetPublicationError(ValueError):
    """Raised when formal publication cannot proceed without weakening G3."""


class NetworkEgressAttempt(G3AssetPublicationError):
    """Raised when a semantic probe attempts any socket operation."""


@dataclass(frozen=True, slots=True)
class G3AssetPublicationResult:
    logical_name: str
    kind: str
    asset_id: str
    candidate_id: str
    state: str
    status: str
    manifest_ref: str
    candidate_ref: str
    qualification_ref: str
    verification_ref: str
    semantic_evidence_ref: str

    def to_dict(self) -> dict[str, str]:
        return {
            "logical_name": self.logical_name,
            "kind": self.kind,
            "asset_id": self.asset_id,
            "candidate_id": self.candidate_id,
            "state": self.state,
            "status": self.status,
            "manifest_ref": self.manifest_ref,
            "candidate_ref": self.candidate_ref,
            "qualification_ref": self.qualification_ref,
            "verification_ref": self.verification_ref,
            "semantic_evidence_ref": self.semantic_evidence_ref,
        }


def _require_text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise G3AssetPublicationError(f"{field} must be normalized non-empty text")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field)
    if _SHA256.fullmatch(text) is None:
        raise G3AssetPublicationError(f"{field} must be a lowercase SHA-256")
    return text


def _require_timestamp(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise G3AssetPublicationError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise G3AssetPublicationError(f"{field} must include a timezone")
    return text


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _approved_data_root(value: str | Path) -> Path:
    supplied = Path(value)
    if ".." in supplied.parts:
        raise G3AssetPublicationError("DATA_ROOT may not contain parent traversal")
    root = Path(os.path.abspath(supplied))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise G3AssetPublicationError("DATA_ROOT may not traverse a link")
    if not root.exists() or not root.is_dir():
        raise G3AssetPublicationError("DATA_ROOT must be an existing directory")
    return root.resolve(strict=True)


def _target_for_ref(root: Path, reference: str) -> Path:
    normalized = validate_asset_path(reference)
    relative = PurePosixPath(normalized)
    target = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise G3AssetPublicationError(
                f"DATA_ROOT reference traverses a link: {reference}"
            )
    try:
        target.parent.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise G3AssetPublicationError(
            f"DATA_ROOT reference escapes the approved root: {reference}"
        ) from error
    return target


def _asset_root(root: Path, reference: str) -> Path:
    target = _target_for_ref(root, reference)
    if not target.exists() or not target.is_dir() or _is_link_like(target):
        raise G3AssetPublicationError(
            f"asset_root_ref must be an existing non-link directory: {reference}"
        )
    resolved = target.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise G3AssetPublicationError("asset root escapes DATA_ROOT") from error
    return resolved


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_clobber(
    root: Path,
    reference: str,
    value: Mapping[str, Any],
) -> tuple[Path, bool, str]:
    """Atomically create one canonical artifact or accept identical bytes."""

    payload = canonical_json_bytes(dict(value))
    digest = hashlib.sha256(payload).hexdigest()
    target = _target_for_ref(root, reference)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Recheck after mkdir so a raced link cannot silently redirect publication.
    _target_for_ref(root, reference)
    if target.exists() or _is_link_like(target):
        if _is_link_like(target) or not target.is_file():
            raise G3AssetPublicationError(
                f"publication target is not a regular file: {reference}"
            )
        if target.read_bytes() != payload:
            raise FileExistsError(
                f"no-clobber publication target differs: {reference}"
            )
        return target, False, digest

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.publish-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
                temporary.unlink()
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            if _is_link_like(target) or not target.is_file():
                raise G3AssetPublicationError(
                    f"publication target raced to a non-file: {reference}"
                )
            if target.read_bytes() != payload:
                raise FileExistsError(
                    f"concurrent no-clobber publication differs: {reference}"
                )
            return target, False, digest
        _fsync_directory(target.parent)
        return target, True, digest
    finally:
        temporary.unlink(missing_ok=True)


def _load_canonical_object(root: Path, reference: str) -> dict[str, Any]:
    target = _target_for_ref(root, reference)
    if not target.exists() or not target.is_file() or _is_link_like(target):
        raise FileNotFoundError(reference)
    value = load_canonical_json(target)
    return dict(ensure_json_object(value, field=reference))


def semantic_evidence_artifact_hash(value: Mapping[str, Any]) -> str:
    return _semantic_evidence_artifact_hash(value)


def validate_semantic_evidence(
    value: Mapping[str, Any],
    *,
    expected_check_ids: tuple[str, ...] | None = None,
) -> None:
    _validate_semantic_evidence(
        value,
        expected_check_ids=expected_check_ids,
        error_type=G3AssetPublicationError,
    )


def _load_requirements(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
        validate_stage0_asset_requirements(result)
        return result
    return load_stage0_asset_requirements(value)


def _load_layout(
    value: Mapping[str, Any] | str | Path,
    *,
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
        validate_stage0_asset_layout(result, requirements=requirements)
        return result
    return load_stage0_asset_layout(value, requirements=requirements)


def _requirement_index(
    requirements: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for model in requirements["models"]:
        result[("model", model["name"])] = model
    tokenizer = requirements["tokenizer"]
    result[("tokenizer", tokenizer["name"])] = tokenizer
    result[("pile", "pile")] = requirements["pile"]
    for task in requirements["glue"]:
        result[("glue_raw", task["task"])] = task
        result[("glue_derived", task["task"])] = task
    return result


def _descriptor_files(values: list[Mapping[str, Any]]) -> list[AssetFile]:
    return [
        AssetFile(
            path=item["path"],
            size_bytes=item["size_bytes"],
            sha256=item["sha256"],
            role=item["role"],
        )
        for item in values
    ]


def _recursive_inventory(asset_root: Path) -> list[AssetFile]:
    files: list[AssetFile] = []
    for current_text, directory_names, file_names in os.walk(
        asset_root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        for name in directory_names:
            child = current / name
            if _is_link_like(child):
                raise G3AssetPublicationError(
                    f"derived asset inventory contains a linked directory: {child}"
                )
        for name in file_names:
            child = current / name
            if _is_link_like(child) or not child.is_file():
                raise G3AssetPublicationError(
                    f"derived asset inventory contains a non-regular file: {child}"
                )
            relative = child.relative_to(asset_root).as_posix()
            files.append(
                AssetFile(
                    relative,
                    child.stat().st_size,
                    sha256_file(child),
                    "dataset_artifact",
                )
            )
    if not files:
        raise G3AssetPublicationError("derived GLUE inventory may not be empty")
    return sorted(files, key=lambda item: item.path)


def _preprocessing(
    *,
    version: str,
    config_hash: str,
    generator_git_commit: str,
    tokenizer_asset_id: str | None = None,
    parent_asset_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": version,
        "config_hash": config_hash,
        "code_git_commit": generator_git_commit,
        "tokenizer_asset_id": tokenizer_asset_id,
        "parent_asset_ids": parent_asset_ids or [],
    }


def _model_metadata(
    requirement: Mapping[str, Any],
    *,
    parent_asset_id: str | None,
) -> dict[str, Any]:
    config = next(
        item for item in requirement["files"] if item["role"] == "config"
    )
    checkpoint = requirement["checkpoint"]
    if not isinstance(checkpoint, str) or not checkpoint.startswith("step"):
        raise G3AssetPublicationError("model checkpoint is not an exact step")
    try:
        training_step = int(checkpoint[4:])
    except ValueError as error:
        raise G3AssetPublicationError("model checkpoint step is invalid") from error
    initialization_kind = (
        "base_initialization" if training_step == 0 else "trained_checkpoint"
    )
    if training_step == 0 and parent_asset_id is not None:
        raise G3AssetPublicationError("step0 model may not have a parent asset")
    if training_step > 0 and parent_asset_id is None:
        raise G3AssetPublicationError("trained checkpoint requires its step0 parent")
    dtype_counts = deepcopy(requirement["dtype_counts"])
    dtype = next(iter(dtype_counts)) if len(dtype_counts) == 1 else "mixed"
    return {
        "contract_version": "stage0-model-metadata-v1",
        "architecture": requirement["architecture"],
        "parameter_count": requirement["parameter_count"],
        "tensor_count": requirement["tensor_count"],
        "dtype": dtype,
        "dtype_counts": dtype_counts,
        "max_position_embeddings": requirement["max_position_embeddings"],
        "config_path": config["path"],
        "config_sha256": config["sha256"],
        "initialization_id": (
            f"{requirement['repository']}@{requirement['revision']}:"
            f"{requirement['checkpoint']}"
        ),
        "initialization_kind": initialization_kind,
        "training_step": training_step,
        "parent_model_asset_id": parent_asset_id,
    }


def _external_json_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = loads_strict_json(path.read_bytes())
        return dict(ensure_json_object(value, field=field))
    except (OSError, CanonicalJSONError, TypeError, ValueError) as error:
        raise G3AssetPublicationError(f"invalid local JSON object: {field}") from error


def _special_token_text(value: Any, *, field: str) -> str:
    if isinstance(value, str):
        return _require_text(value, field=field)
    if isinstance(value, Mapping):
        return _require_text(value.get("content"), field=field)
    raise G3AssetPublicationError(f"{field} is not a tokenizer token identity")


def _tokenizer_metadata(
    requirement: Mapping[str, Any],
    *,
    asset_root: Path,
) -> dict[str, Any]:
    config = next(
        item
        for item in requirement["files"]
        if item["role"] == "tokenizer_config"
    )
    tokenizer_model = next(
        item
        for item in requirement["files"]
        if item["role"] == "tokenizer_model"
    )
    special_descriptor = next(
        item
        for item in requirement["files"]
        if item["role"] == "special_tokens"
    )
    special_map = _external_json_object(
        asset_root / PurePosixPath(special_descriptor["path"]),
        field="tokenizer special_tokens_map",
    )
    special_tokens: dict[str, Any] = {}
    for name, token_id in requirement["special_token_ids"].items():
        token_key = f"{name}_token"
        if token_key not in special_map:
            raise G3AssetPublicationError(
                f"tokenizer special_tokens_map is missing {token_key}"
            )
        special_tokens[token_key] = {
            "token": _special_token_text(
                special_map[token_key], field=f"tokenizer.{token_key}"
            ),
            "token_id": token_id,
        }
    padding = requirement["glue_padding_token"]
    special_tokens["glue_padding_token"] = {
        "token": padding["token"],
        "token_id": padding["token_id"],
    }
    tokenizer_json = _external_json_object(
        asset_root / PurePosixPath(tokenizer_model["path"]),
        field="tokenizer.json",
    )
    tokenizer_json_version = _require_text(
        tokenizer_json.get("version"), field="tokenizer.json.version"
    )
    normalizer = tokenizer_json.get("normalizer")
    normalization = (
        "identity"
        if normalizer is None
        else f"tokenizer-json:{canonical_json_hash(normalizer)}"
    )
    return {
        "contract_version": "stage0-tokenizer-metadata-v1",
        "tokenizer_class": requirement["tokenizer_class"],
        "implementation_version": f"tokenizer-json-{tokenizer_json_version}",
        "vocab_size": requirement["vocab_size"],
        "token_count_with_added_tokens": requirement[
            "token_count_with_added_tokens"
        ],
        "vocab_mapping_sha256": requirement["vocab_mapping_sha256"],
        "glue_padding_policy": padding["policy"],
        "special_tokens": special_tokens,
        "normalization": normalization,
        "config_path": config["path"],
        "config_sha256": config["sha256"],
    }


def _causal_lm_mapping(requirement: Mapping[str, Any]) -> dict[str, Any]:
    causal = requirement["causal_lm_contract"]
    return {
        "labels_alignment": causal["labels_alignment"],
        "source_tokens_per_record": causal["source_tokens_per_record"],
        "input_sequence_length": causal["input_sequence_length"],
        "label_sequence_length": causal["target_sequence_length"],
        "input_slice": deepcopy(causal["input_slice"]),
        "label_slice": deepcopy(causal["target_slice"]),
        "attention_mask_policy": causal["attention_mask_policy"],
        "effective_target_tokens": causal[
            "effective_target_tokens_per_record"
        ],
        "loss_adapter_id": causal["loss_adapter_id"],
    }


def _pile_metadata(
    requirement: Mapping[str, Any],
    *,
    generator_git_commit: str,
    tokenizer_asset_id: str,
) -> dict[str, Any]:
    index = requirement["index"]
    contract = requirement["index_contract"]
    shards = [
        {
            "ordinal": item["ordinal"],
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
            "byte_start": item["byte_start"],
            "byte_stop": item["byte_stop"],
        }
        for item in requirement["selected_shards"]
    ]
    splits = {
        item["name"]: {
            "sample_count": item["stop"] - item["start"],
            "fields": ["tokens"],
            "cursor": {"start": item["start"], "stop": item["stop"]},
        }
        for item in requirement["cursor_intervals"]
    }
    preprocessing_version = "stage0-pile-raw-v1"
    preprocessing_hash = canonical_json_hash(
        {
            "schema_version": "stage0-pile-raw-config-v1",
            "repository": requirement["repository"],
            "revision": requirement["revision"],
            "index_contract": requirement["index_contract"],
            "causal_lm_contract": requirement["causal_lm_contract"],
            "required_cursor_stop": requirement["required_cursor_stop"],
            "cursor_intervals": requirement["cursor_intervals"],
            "reference_reader": requirement["reference_reader"],
            "reference_batch_size": requirement["reference_batch_size"],
            "reference_batch_sha256": requirement["reference_batch_sha256"],
            "last_required_record_sha256": requirement[
                "last_required_record_sha256"
            ],
        }
    )
    return {
        "contract_version": "stage0-dataset-metadata-v1",
        "dataset_kind": "raw_indexed_mmap",
        "raw_revision": requirement["revision"],
        "splits": splits,
        "preprocessing_version": preprocessing_version,
        "preprocessing": _preprocessing(
            version=preprocessing_version,
            config_hash=preprocessing_hash,
            generator_git_commit=generator_git_commit,
            tokenizer_asset_id=tokenizer_asset_id,
        ),
        "storage": {
            "kind": "pythia_mmap_shards",
            "idx": {
                "path": index["path"],
                "sha256": index["sha256"],
                "magic": bytes.fromhex(contract["magic_hex"])
                .rstrip(b"\0")
                .decode("ascii"),
                "version": contract["version"],
                "dtype_code": contract["dtype_code"],
                "itemsize_bytes": contract["dtype_bytes"],
                "sequence_count": contract["sequence_count"],
                "document_count": contract["document_count"],
            },
            "tokens_per_record": requirement["causal_lm_contract"][
                "source_tokens_per_record"
            ],
            "global_byte_coverage": {
                "start": 0,
                "stop": shards[-1]["byte_stop"],
            },
            "required_cursor_stop": requirement["required_cursor_stop"],
            "causal_lm_mapping": _causal_lm_mapping(requirement),
            "reference_reader": deepcopy(requirement["reference_reader"]),
            "reference_batch_size": requirement["reference_batch_size"],
            "reference_batch_sha256": deepcopy(
                requirement["reference_batch_sha256"]
            ),
            "last_required_record_sha256": requirement[
                "last_required_record_sha256"
            ],
            "cross_shard_policy": requirement["cross_shard_policy"],
            "shards": shards,
        },
    }


def _glue_splits(
    requirement: Mapping[str, Any],
    *,
    derived: bool,
) -> dict[str, Any]:
    split_names = (
        requirement["preprocessing"]["derived_splits"]
        if derived
        else list(requirement["split_counts"])
    )
    fields = (
        ["input_ids", "attention_mask", "labels"]
        if derived
        else [*requirement["text_fields"], "label"]
    )
    return {
        name: {
            "sample_count": requirement["split_counts"][name],
            "fields": fields,
            "cursor": {
                "start": 0,
                "stop": requirement["split_counts"][name],
            },
        }
        for name in split_names
    }


def _glue_task_contract(requirement: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": requirement["task"],
        "text_fields": deepcopy(requirement["text_fields"]),
        "label_mapping": deepcopy(requirement["label_mapping"]),
        "unlabeled_test_policy": requirement["unlabeled_test_policy"],
    }


def _glue_metadata(
    requirement: Mapping[str, Any],
    *,
    derived: bool,
    generator_git_commit: str,
    tokenizer_asset_id: str | None,
    parent_asset_id: str | None,
) -> dict[str, Any]:
    config_hash = glue_preprocessing_config_hash(requirement)
    preprocessing = _preprocessing(
        version=GLUE_PREPROCESSING_VERSION,
        config_hash=config_hash,
        generator_git_commit=generator_git_commit,
        tokenizer_asset_id=tokenizer_asset_id if derived else None,
        parent_asset_ids=[parent_asset_id] if derived and parent_asset_id else [],
    )
    if derived:
        if tokenizer_asset_id is None or parent_asset_id is None:
            raise G3AssetPublicationError(
                "derived GLUE requires tokenizer and raw-parent asset IDs"
            )
        storage = {
            "kind": "hf_load_from_disk",
            "format_version": "huggingface-datasets-load-from-disk-v1",
        }
        dataset_kind = "derived_pretokenized"
    else:
        storage = {
            "kind": "hf_raw_parquet_files",
            "splits": {
                split: [
                    {"path": item["path"], "sha256": item["sha256"]}
                    for item in requirement["raw_files"]
                    if item["role"] == split
                ]
                for split in requirement["split_counts"]
            },
        }
        dataset_kind = "hf_raw_parquet"
    return {
        "contract_version": "stage0-dataset-metadata-v1",
        "dataset_kind": dataset_kind,
        "raw_revision": requirement["revision"],
        "splits": _glue_splits(requirement, derived=derived),
        "preprocessing_version": GLUE_PREPROCESSING_VERSION,
        "preprocessing": preprocessing,
        "task_contract": _glue_task_contract(requirement),
        "storage": storage,
    }


def _files_and_metadata(
    kind: str,
    requirement: Mapping[str, Any],
    *,
    asset_root: Path,
    generator_git_commit: str,
    parent_model_asset_id: str | None,
    tokenizer_asset_id: str | None,
    raw_glue_parent_asset_id: str | None,
) -> tuple[list[AssetFile], dict[str, Any], str]:
    if kind == "model":
        return (
            _descriptor_files(requirement["files"]),
            _model_metadata(
                requirement, parent_asset_id=parent_model_asset_id
            ),
            "model",
        )
    if kind == "tokenizer":
        return (
            _descriptor_files(requirement["files"]),
            _tokenizer_metadata(requirement, asset_root=asset_root),
            "tokenizer",
        )
    if kind == "pile":
        if tokenizer_asset_id is None:
            raise G3AssetPublicationError("Pile metadata requires tokenizer asset ID")
        files = _descriptor_files(
            [requirement["index"], *requirement["selected_shards"]]
        )
        return (
            files,
            _pile_metadata(
                requirement,
                generator_git_commit=generator_git_commit,
                tokenizer_asset_id=tokenizer_asset_id,
            ),
            "dataset",
        )
    if kind == "glue_raw":
        return (
            _descriptor_files(requirement["raw_files"]),
            _glue_metadata(
                requirement,
                derived=False,
                generator_git_commit=generator_git_commit,
                tokenizer_asset_id=None,
                parent_asset_id=None,
            ),
            "dataset",
        )
    if kind == "glue_derived":
        return (
            _recursive_inventory(asset_root),
            _glue_metadata(
                requirement,
                derived=True,
                generator_git_commit=generator_git_commit,
                tokenizer_asset_id=tokenizer_asset_id,
                parent_asset_id=raw_glue_parent_asset_id,
            ),
            "dataset",
        )
    raise G3AssetPublicationError(f"unsupported G3 layout kind: {kind}")


@contextmanager
def _socket_egress_guard() -> Iterator[list[int]]:
    attempts = [0]
    socket_methods = tuple(
        name
        for name in ("connect", "connect_ex", "send", "sendall", "sendto", "sendmsg")
        if hasattr(socket.socket, name)
    )
    socket_functions = tuple(
        name
        for name in (
            "create_connection",
            "getaddrinfo",
            "gethostbyname",
            "gethostbyname_ex",
            "gethostbyaddr",
        )
        if hasattr(socket, name)
    )
    original_methods = {
        name: getattr(socket.socket, name) for name in socket_methods
    }
    original_functions = {
        name: getattr(socket, name) for name in socket_functions
    }
    original_popen = subprocess.Popen
    environment_names = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    )
    previous_environment = {name: os.environ.get(name) for name in environment_names}

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        attempts[0] += 1
        raise NetworkEgressAttempt("semantic probe attempted socket egress")

    for name in original_methods:
        setattr(socket.socket, name, blocked)
    for name in original_functions:
        setattr(socket, name, blocked)
    subprocess.Popen = blocked  # type: ignore[assignment]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        yield attempts
    finally:
        subprocess.Popen = original_popen
        for name, method in original_methods.items():
            setattr(socket.socket, name, method)
        for name, function in original_functions.items():
            setattr(socket, name, function)
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _safetensors_statistics(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise G3AssetPublicationError("safetensors file is truncated")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length == 0 or header_length > file_size - 8:
            raise G3AssetPublicationError("safetensors header length is invalid")
        header_bytes = handle.read(header_length)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G3AssetPublicationError("safetensors header is invalid JSON") from error
    if not isinstance(header, dict):
        raise G3AssetPublicationError("safetensors header must be an object")
    tensor_count = 0
    parameter_count = 0
    dtype_counts: dict[str, int] = {}
    ranges: list[tuple[int, int]] = []
    data_size = file_size - 8 - header_length
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise G3AssetPublicationError("safetensors tensor descriptor is invalid")
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise G3AssetPublicationError("safetensors tensor fields are invalid")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if not isinstance(dtype, str) or not dtype:
            raise G3AssetPublicationError("safetensors dtype is invalid")
        if (
            not isinstance(shape, list)
            or any(type(item) is not int or item < 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(item) is not int or item < 0 for item in offsets)
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
        ):
            raise G3AssetPublicationError("safetensors shape/offset contract is invalid")
        count = 1
        for dimension in shape:
            count *= dimension
        tensor_count += 1
        parameter_count += count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
        ranges.append((offsets[0], offsets[1]))
    ordered_ranges = sorted(ranges)
    for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
        if current[0] < previous[1]:
            raise G3AssetPublicationError("safetensors tensor byte ranges overlap")
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "header_length": header_length,
        "file_size": file_size,
    }


def _probe_model(
    requirement: Mapping[str, Any],
    asset_root: Path,
    *,
    legacy_manifest_replacement: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    config_descriptor = next(
        item for item in requirement["files"] if item["role"] == "config"
    )
    weights_descriptor = next(
        item for item in requirement["files"] if item["role"] == "weights"
    )
    config = _external_json_object(
        asset_root / PurePosixPath(config_descriptor["path"]),
        field="model config",
    )
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or requirement["architecture"] not in architectures
        or config.get("max_position_embeddings")
        != requirement["max_position_embeddings"]
    ):
        raise G3AssetPublicationError("model config semantic contract mismatch")
    statistics = _safetensors_statistics(
        asset_root / PurePosixPath(weights_descriptor["path"])
    )
    expected_statistics = {
        "tensor_count": requirement["tensor_count"],
        "parameter_count": requirement["parameter_count"],
        "dtype_counts": requirement["dtype_counts"],
    }
    if any(statistics[key] != expected for key, expected in expected_statistics.items()):
        raise G3AssetPublicationError("model safetensors statistics mismatch")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise G3AssetPublicationError(
            "transformers is required for formal model verification"
        ) from error
    try:
        loaded_config = AutoConfig.from_pretrained(
            str(asset_root),
            local_files_only=True,
            trust_remote_code=False,
        )
        loaded_model = AutoModelForCausalLM.from_pretrained(
            str(asset_root),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise G3AssetPublicationError("offline model load failed") from error
    try:
        loaded_parameter_count = sum(
            parameter.numel() for parameter in loaded_model.parameters()
        )
        loaded_tensor_count = len(loaded_model.state_dict())
        loaded_max_positions = getattr(
            loaded_config, "max_position_embeddings", None
        )
        if (
            loaded_parameter_count != requirement["parameter_count"]
            or loaded_tensor_count != requirement["tensor_count"]
            or loaded_max_positions != requirement["max_position_embeddings"]
        ):
            raise G3AssetPublicationError("offline model identity mismatch")
        offline_load = {
            "config_class": type(loaded_config).__name__,
            "model_class": type(loaded_model).__name__,
            "parameter_count": loaded_parameter_count,
            "tensor_count": loaded_tensor_count,
            "max_position_embeddings": loaded_max_positions,
            "local_files_only": True,
        }
    finally:
        del loaded_model
        del loaded_config
        gc.collect()
    return {
        "model_semantic_contract": {
            "architecture": requirement["architecture"],
            "max_position_embeddings": requirement["max_position_embeddings"],
            "config_sha256": config_descriptor["sha256"],
            "legacy_manifest_replacement": (
                deepcopy(dict(legacy_manifest_replacement))
                if legacy_manifest_replacement is not None
                else None
            ),
        },
        "offline_model_load": offline_load,
    }


def _probe_tokenizer(
    requirement: Mapping[str, Any],
    asset_root: Path,
) -> dict[str, dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise G3AssetPublicationError(
            "transformers is required for formal tokenizer verification"
        ) from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(asset_root),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise G3AssetPublicationError("offline tokenizer load failed") from error
    vocab = tokenizer.get_vocab()
    if (
        not isinstance(vocab, dict)
        or any(not isinstance(key, str) or type(item) is not int for key, item in vocab.items())
    ):
        raise G3AssetPublicationError("tokenizer vocabulary mapping is invalid")
    observed_mapping_hash = canonical_json_hash(vocab)
    observed = {
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "token_count_with_added_tokens": len(tokenizer),
        "vocab_mapping_sha256": observed_mapping_hash,
    }
    expected = {
        "tokenizer_class": requirement["tokenizer_class"],
        "vocab_size": requirement["vocab_size"],
        "token_count_with_added_tokens": requirement[
            "token_count_with_added_tokens"
        ],
        "vocab_mapping_sha256": requirement["vocab_mapping_sha256"],
    }
    if observed != expected:
        raise G3AssetPublicationError("tokenizer vocabulary identity mismatch")
    metadata = _tokenizer_metadata(requirement, asset_root=asset_root)
    metadata_special = metadata["special_tokens"]
    special_tokens: dict[str, dict[str, Any]] = {}
    for name, expected_id in requirement["special_token_ids"].items():
        observed_id = getattr(tokenizer, f"{name}_token_id", None)
        observed_token = getattr(tokenizer, f"{name}_token", None)
        expected_token = metadata_special[f"{name}_token"]["token"]
        if observed_id != expected_id or observed_token != expected_token:
            raise G3AssetPublicationError(
                f"tokenizer {name} token identity mismatch"
            )
        special_tokens[f"{name}_token"] = {
            "token": observed_token,
            "token_id": observed_id,
        }
    padding = requirement["glue_padding_token"]
    padding_id = tokenizer.convert_tokens_to_ids(padding["token"])
    if padding_id != padding["token_id"]:
        raise G3AssetPublicationError("tokenizer GLUE padding identity mismatch")
    return {
        "offline_tokenizer_load": observed,
        "tokenizer_semantic_contract": {
            "implementation_version": metadata["implementation_version"],
            "normalization": metadata["normalization"],
            "special_tokens": special_tokens,
            "glue_padding_token": {
                "token": padding["token"],
                "token_id": padding_id,
                "policy": padding["policy"],
            },
        },
    }


def _read_global_shard_bytes(
    requirement: Mapping[str, Any],
    asset_root: Path,
    *,
    offset: int,
    length: int,
) -> bytes:
    end = offset + length
    chunks: list[bytes] = []
    cursor = offset
    for shard in requirement["selected_shards"]:
        start = shard["byte_start"]
        stop = shard["byte_stop"]
        if cursor >= end:
            break
        if cursor >= stop or end <= start:
            continue
        read_start = max(cursor, start)
        read_stop = min(end, stop)
        with (asset_root / PurePosixPath(shard["path"])).open("rb") as handle:
            handle.seek(read_start - start)
            chunk = handle.read(read_stop - read_start)
        if len(chunk) != read_stop - read_start:
            raise G3AssetPublicationError("short read from selected Pile shard")
        chunks.append(chunk)
        cursor = read_stop
    if cursor != end:
        raise G3AssetPublicationError("selected Pile shards do not cover record bytes")
    return b"".join(chunks)


def _probe_pile(
    requirement: Mapping[str, Any],
    asset_root: Path,
    *,
    reference_reader_oracle: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    try:
        import numpy as np
        from .data.pythia_mmap import MMapIndex
    except ImportError as error:
        raise G3AssetPublicationError(
            "NumPy/Pythia mmap support is required for formal Pile verification"
        ) from error
    index_contract = requirement["index_contract"]
    index_path = asset_root / PurePosixPath(requirement["index"]["path"])
    with MMapIndex(index_path) as index:
        observed_header = {
            "magic_hex": "4d4d49444944580000",
            "version": index.version,
            "dtype_code": index.dtype_code,
            "dtype_bytes": index.dtype.itemsize,
            "sequence_count": index.sequence_count,
            "document_count": index.document_count,
        }
        if observed_header != index_contract:
            raise G3AssetPublicationError("Pile index header contract mismatch")
        required_stop = requirement["required_cursor_stop"]
        if required_stop > index.sequence_count:
            raise G3AssetPublicationError("Pile required cursor exceeds index")
        sizes = index.sizes[:required_stop]
        pointers = index.pointers[:required_stop]
        tokens_per_record = requirement["causal_lm_contract"][
            "source_tokens_per_record"
        ]
        if np.any(sizes != tokens_per_record):
            raise G3AssetPublicationError("Pile required record size mismatch")
        if np.any(pointers < 0) or np.any(pointers % index.dtype.itemsize):
            raise G3AssetPublicationError("Pile required pointer is invalid")
        if int(pointers[0]) != 0:
            raise G3AssetPublicationError("Pile selected prefix must start at byte zero")
        if required_stop > 1:
            expected_next = pointers[:-1] + sizes[:-1] * index.dtype.itemsize
            if np.any(pointers[1:] != expected_next):
                raise G3AssetPublicationError("Pile required pointers are not contiguous")
        last_end = int(pointers[-1]) + int(sizes[-1]) * index.dtype.itemsize
        coverage_stop = requirement["selected_shards"][-1]["byte_stop"]
        if last_end > coverage_stop:
            raise G3AssetPublicationError("Pile required pointer exceeds shard coverage")
        mapping = _causal_lm_mapping(requirement)
        if (
            mapping["input_slice"] != [0, 2048]
            or mapping["label_slice"] != [1, 2049]
        ):
            raise G3AssetPublicationError("Pile 2049-to-2048 mapping mismatch")
        batch_size = requirement["reference_batch_size"]
        observed_batch_hashes: dict[str, str] = {}
        for raw_batch_index, expected_hash in requirement[
            "reference_batch_sha256"
        ].items():
            batch_index = int(raw_batch_index)
            record_start = batch_index * batch_size
            record_stop = record_start + batch_size
            if record_stop > required_stop:
                raise G3AssetPublicationError(
                    "Pile reference batch exceeds the required cursor prefix"
                )
            pointer = int(index.pointers[record_start])
            byte_length = (
                batch_size * tokens_per_record * index.dtype.itemsize
            )
            batch_bytes = _read_global_shard_bytes(
                requirement,
                asset_root,
                offset=pointer,
                length=byte_length,
            )
            digest = hashlib.sha256(batch_bytes).hexdigest()
            observed_batch_hashes[raw_batch_index] = digest
            if digest != expected_hash:
                raise G3AssetPublicationError(
                    f"Pile reference batch hash mismatch at {raw_batch_index}"
                )
        last_record_index = required_stop - 1
        last_record_pointer = int(index.pointers[last_record_index])
        last_record_length = (
            int(index.sizes[last_record_index]) * index.dtype.itemsize
        )
        last_record_bytes = _read_global_shard_bytes(
            requirement,
            asset_root,
            offset=last_record_pointer,
            length=last_record_length,
        )
        observed_last_record_hash = hashlib.sha256(last_record_bytes).hexdigest()
        if observed_last_record_hash != requirement["last_required_record_sha256"]:
            raise G3AssetPublicationError("Pile last required record hash mismatch")
    return {
        "pile_index_contract": observed_header,
        "pile_cursor_coverage": {
            "required_cursor_stop": required_stop,
            "last_required_byte_stop": last_end,
            "selected_coverage_stop": coverage_stop,
        },
        "pile_causal_lm_contract": {
            "mapping": mapping,
            "reference_reader": deepcopy(requirement["reference_reader"]),
            "reference_batch_size": batch_size,
            "reference_batch_sha256": observed_batch_hashes,
            "reference_reader_oracle": (
                deepcopy(dict(reference_reader_oracle))
                if reference_reader_oracle is not None
                else None
            ),
            "last_required_record": {
                "record_index": last_record_index,
                "sha256": observed_last_record_hash,
            },
        },
    }


def _allowed_label_ids(requirement: Mapping[str, Any]) -> set[int]:
    return set(requirement["label_mapping"].values())


def _probe_glue_raw(
    requirement: Mapping[str, Any],
    asset_root: Path,
) -> dict[str, dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise G3AssetPublicationError(
            "pyarrow is required for formal raw GLUE verification"
        ) from error
    split_details: dict[str, Any] = {}
    allowed_labels = _allowed_label_ids(requirement)
    required_fields = {*requirement["text_fields"], "label"}
    for split, expected_count in requirement["split_counts"].items():
        descriptors = [
            item for item in requirement["raw_files"] if item["role"] == split
        ]
        if not descriptors:
            raise G3AssetPublicationError(f"raw GLUE split {split} has no files")
        row_count = 0
        observed_fields: set[str] = set()
        observed_labels: set[int] = set()
        for descriptor in descriptors:
            source = asset_root / PurePosixPath(descriptor["path"])
            reader = parquet.ParquetFile(source)
            row_count += reader.metadata.num_rows
            observed_fields.update(reader.schema_arrow.names)
            if "label" not in reader.schema_arrow.names:
                raise G3AssetPublicationError("raw GLUE parquet is missing label")
            values = reader.read(columns=["label"]).column("label").to_pylist()
            if any(type(item) is not int for item in values):
                raise G3AssetPublicationError("raw GLUE labels must be integers")
            observed_labels.update(values)
        if row_count != expected_count or not required_fields <= observed_fields:
            raise G3AssetPublicationError(f"raw GLUE split {split} contract mismatch")
        expected_labels = {-1} if split.startswith("test") else allowed_labels
        if observed_labels != expected_labels:
            raise G3AssetPublicationError(f"raw GLUE split {split} label drift")
        split_details[split] = {
            "sample_count": row_count,
            "fields": sorted(required_fields),
            "labels": sorted(observed_labels),
        }
    return {
        "glue_task_contract": _glue_task_contract(requirement),
        "offline_glue_raw_load": {"splits": split_details},
    }


def _probe_glue_derived(
    requirement: Mapping[str, Any],
    asset_root: Path,
    manifest: Mapping[str, Any],
    *,
    tokenizer_asset_id: str,
    raw_parent_asset_id: str,
) -> dict[str, dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as error:
        raise G3AssetPublicationError(
            "datasets is required for formal derived GLUE verification"
        ) from error
    try:
        dataset = load_from_disk(str(asset_root))
    except Exception as error:
        raise G3AssetPublicationError("offline derived GLUE load failed") from error
    if not isinstance(dataset, (DatasetDict, dict)):
        raise G3AssetPublicationError("derived GLUE root must contain split datasets")
    expected_splits = requirement["preprocessing"]["derived_splits"]
    if set(dataset) != set(expected_splits):
        raise G3AssetPublicationError("derived GLUE split set mismatch")
    expected_fields = {"input_ids", "attention_mask", "labels"}
    max_length = requirement["preprocessing"]["max_length"]
    allowed_labels = _allowed_label_ids(requirement)
    split_details: dict[str, Any] = {}
    for split_name in expected_splits:
        split = dataset[split_name]
        if len(split) != requirement["split_counts"][split_name]:
            raise G3AssetPublicationError("derived GLUE split count mismatch")
        if set(split.column_names) != expected_fields:
            raise G3AssetPublicationError("derived GLUE fields mismatch")
        observed_labels: set[int] = set()
        row_count = 0
        for batch in split.iter(batch_size=1024):
            input_ids = batch["input_ids"]
            attention_masks = batch["attention_mask"]
            labels = batch["labels"]
            if (
                any(len(row) != max_length for row in input_ids)
                or any(len(row) != max_length for row in attention_masks)
                or any(type(label) is not int for label in labels)
            ):
                raise G3AssetPublicationError("derived GLUE row shape is invalid")
            observed_labels.update(labels)
            row_count += len(labels)
        if row_count != len(split) or observed_labels != allowed_labels:
            raise G3AssetPublicationError("derived GLUE labels/iteration mismatch")
        split_details[split_name] = {
            "sample_count": row_count,
            "fields": sorted(expected_fields),
            "labels": sorted(observed_labels),
            "sequence_length": max_length,
        }
    preprocessing = manifest["metadata"]["preprocessing"]
    expected_config_hash = glue_preprocessing_config_hash(requirement)
    if (
        preprocessing["version"] != GLUE_PREPROCESSING_VERSION
        or preprocessing["config_hash"] != expected_config_hash
        or preprocessing["tokenizer_asset_id"] != tokenizer_asset_id
        or preprocessing["parent_asset_ids"] != [raw_parent_asset_id]
    ):
        raise G3AssetPublicationError("derived GLUE lineage mismatch")
    return {
        "glue_preprocessing_lineage": {
            "config_hash": expected_config_hash,
            "tokenizer_asset_id": tokenizer_asset_id,
            "parent_asset_ids": [raw_parent_asset_id],
        },
        "glue_task_contract": _glue_task_contract(requirement),
        "offline_glue_derived_load": {"splits": split_details},
    }


def _run_semantic_probe(
    kind: str,
    requirement: Mapping[str, Any],
    asset_root: Path,
    manifest: Mapping[str, Any],
    *,
    tokenizer_asset_id: str | None,
    raw_glue_parent_asset_id: str | None,
    legacy_manifest_replacement: Mapping[str, Any] | None,
    reference_reader_oracle: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if kind == "model":
        return _probe_model(
            requirement,
            asset_root,
            legacy_manifest_replacement=legacy_manifest_replacement,
        )
    if kind == "tokenizer":
        return _probe_tokenizer(requirement, asset_root)
    if kind == "pile":
        return _probe_pile(
            requirement,
            asset_root,
            reference_reader_oracle=reference_reader_oracle,
        )
    if kind == "glue_raw":
        return _probe_glue_raw(requirement, asset_root)
    if kind == "glue_derived":
        if tokenizer_asset_id is None or raw_glue_parent_asset_id is None:
            raise G3AssetPublicationError("derived GLUE probe is missing lineage")
        return _probe_glue_derived(
            requirement,
            asset_root,
            manifest,
            tokenizer_asset_id=tokenizer_asset_id,
            raw_parent_asset_id=raw_glue_parent_asset_id,
        )
    raise G3AssetPublicationError(f"unsupported semantic probe kind: {kind}")


def _semantic_evidence(
    *,
    entry: Mapping[str, Any],
    verified: Mapping[str, Any],
    requirements_ref: str,
    requirements_sha256: str,
    generator_git_commit: str,
    checked_at: str,
    details_by_check: Mapping[str, Mapping[str, Any]],
    network_attempts: int,
) -> dict[str, Any]:
    required = tuple(
        item
        for item in qualification_check_ids_for(entry["kind"])
        if item != "full_file_integrity"
    )
    if set(details_by_check) != set(required):
        raise G3AssetPublicationError(
            "semantic probe check set does not match the required G3 checks"
        )
    checks = [
        {
            "check_id": check_id,
            "status": "PASS",
            "summary": f"{check_id} passed under the offline semantic probe",
            "details": deepcopy(dict(details_by_check[check_id])),
        }
        for check_id in required
    ]
    payload: dict[str, Any] = {
        "schema_version": SEMANTIC_EVIDENCE_SCHEMA_VERSION,
        "formal": True,
        "asset_id": verified["asset_id"],
        "candidate_id": verified["candidate_id"],
        "logical_name": entry["logical_name"],
        "kind": entry["kind"],
        "requirements_ref": requirements_ref,
        "requirements_sha256": requirements_sha256,
        "checks": checks,
        "network_attempts": network_attempts,
        "checked_at": checked_at,
        "generator_git_commit": generator_git_commit,
    }
    payload["artifact_hash"] = semantic_evidence_artifact_hash(payload)
    validate_semantic_evidence(payload, expected_check_ids=required)
    return payload


def _validate_semantic_binding(
    evidence: Mapping[str, Any],
    *,
    entry: Mapping[str, Any],
    verified: Mapping[str, Any],
    requirements_ref: str,
    requirements_sha256: str,
    generator_git_commit: str,
) -> None:
    expected_ids = tuple(
        item
        for item in qualification_check_ids_for(entry["kind"])
        if item != "full_file_integrity"
    )
    validate_semantic_evidence(evidence, expected_check_ids=expected_ids)
    expected = {
        "asset_id": verified["asset_id"],
        "candidate_id": verified["candidate_id"],
        "logical_name": entry["logical_name"],
        "kind": entry["kind"],
        "requirements_ref": requirements_ref,
        "requirements_sha256": requirements_sha256,
        "generator_git_commit": generator_git_commit,
    }
    if any(evidence[key] != item for key, item in expected.items()):
        raise G3AssetPublicationError("semantic evidence binding mismatch")


def _qualification_checks(
    kind: str,
    *,
    verification_ref: str,
    verification_sha256: str,
    semantic_ref: str,
    semantic_sha256: str,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for check_id in qualification_check_ids_for(kind):
        full_integrity = check_id == "full_file_integrity"
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS",
                "evidence_ref": (
                    verification_ref if full_integrity else semantic_ref
                ),
                "evidence_sha256": (
                    verification_sha256 if full_integrity else semantic_sha256
                ),
                "summary": f"{check_id} passed under formal offline verification",
            }
        )
    return checks


def _validate_qualification_evidence(
    qualification: Mapping[str, Any],
    *,
    kind: str,
    requirements_ref: str,
    requirements_sha256: str,
    generator_git_commit: str,
    verification_ref: str,
    verification_sha256: str,
    semantic_ref: str,
    semantic_sha256: str,
    acquisition_ref: str | None = None,
    acquisition_sha256: str | None = None,
) -> None:
    validate_g3_qualification(qualification)
    if (
        qualification["requirements_ref"] != requirements_ref
        or qualification["requirements_sha256"] != requirements_sha256
        or qualification["generator_git_commit"] != generator_git_commit
        or qualification["verification_ref"] != verification_ref
        or qualification["verification_sha256"] != verification_sha256
    ):
        raise G3AssetPublicationError("qualification control-plane binding mismatch")
    if acquisition_ref is not None and (
        qualification["acquisition_ref"] != acquisition_ref
        or qualification["acquisition_sha256"] != acquisition_sha256
    ):
        raise G3AssetPublicationError("qualification acquisition binding mismatch")
    checks = qualification["checks"]
    observed = tuple(check["check_id"] for check in checks)
    expected = qualification_check_ids_for(kind)
    if observed != expected:
        raise G3AssetPublicationError("qualification check IDs do not match G3")
    for check in checks:
        if check["check_id"] == "full_file_integrity":
            expected_ref, expected_hash = verification_ref, verification_sha256
        else:
            expected_ref, expected_hash = semantic_ref, semantic_sha256
        if (
            check["evidence_ref"] != expected_ref
            or check["evidence_sha256"] != expected_hash
        ):
            raise G3AssetPublicationError("qualification check evidence mismatch")


def _load_existing_semantic(
    root: Path,
    semantic_ref: str,
    *,
    entry: Mapping[str, Any],
    verified: Mapping[str, Any],
    requirements_ref: str,
    requirements_sha256: str,
    generator_git_commit: str,
) -> tuple[dict[str, Any], str]:
    evidence = _load_canonical_object(root, semantic_ref)
    _validate_semantic_binding(
        evidence,
        entry=entry,
        verified=verified,
        requirements_ref=requirements_ref,
        requirements_sha256=requirements_sha256,
        generator_git_commit=generator_git_commit,
    )
    return evidence, hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()


def _run_guarded_semantic_probe(
    *,
    kind: str,
    requirement: Mapping[str, Any],
    data_root: Path,
    asset_root: Path,
    verified: Mapping[str, Any],
    tokenizer_asset_id: str | None,
    raw_glue_parent_asset_id: str | None,
) -> dict[str, dict[str, Any]]:
    try:
        legacy_manifest_replacement = (
            load_legacy_model_manifest_diagnostic(data_root, requirement)
            if kind == "model" and "legacy_manifest_diagnostic" in requirement
            else None
        )
        reference_reader_oracle = (
            load_pile_reference_reader_oracle(data_root, requirement)
            if kind == "pile"
            else None
        )
    except (
        CanonicalJSONError,
        G3GateAggregationError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise G3AssetPublicationError(
            "semantic source-provenance evidence is missing or invalid"
        ) from error
    with _socket_egress_guard() as attempts:
        details = _run_semantic_probe(
            kind,
            requirement,
            asset_root,
            verified,
            tokenizer_asset_id=tokenizer_asset_id,
            raw_glue_parent_asset_id=raw_glue_parent_asset_id,
            legacy_manifest_replacement=legacy_manifest_replacement,
            reference_reader_oracle=reference_reader_oracle,
        )
    if attempts[0] != 0:
        raise G3AssetPublicationError(
            "semantic probe recorded a forbidden network attempt"
        )
    return details


def _replay_current_semantics(
    *,
    data_root: Path,
    entry: Mapping[str, Any],
    requirement: Mapping[str, Any],
    asset_root: Path,
    verified: Mapping[str, Any],
    evidence: Mapping[str, Any],
    requirements_ref: str,
    requirements_sha256: str,
    generator_git_commit: str,
    tokenizer_asset_id: str | None,
    raw_glue_parent_asset_id: str | None,
) -> None:
    if not semantic_observations_match(
        entry["kind"],
        requirement,
        verified,
        evidence,
        asset_root=asset_root,
        data_root=data_root,
    ):
        raise G3AssetPublicationError(
            "admitted semantic observations do not match the frozen contract"
        )
    details = _run_guarded_semantic_probe(
        kind=entry["kind"],
        requirement=requirement,
        data_root=data_root,
        asset_root=asset_root,
        verified=verified,
        tokenizer_asset_id=tokenizer_asset_id,
        raw_glue_parent_asset_id=raw_glue_parent_asset_id,
    )
    replayed = _semantic_evidence(
        entry=entry,
        verified=verified,
        requirements_ref=requirements_ref,
        requirements_sha256=requirements_sha256,
        generator_git_commit=generator_git_commit,
        checked_at=evidence["checked_at"],
        details_by_check=details,
        network_attempts=0,
    )
    if replayed != evidence:
        raise G3AssetPublicationError(
            "current semantic probe differs from the admitted evidence"
        )


def _parent_model_asset_id(
    requirement: Mapping[str, Any],
    requirements: Mapping[str, Any],
    published_asset_ids: Mapping[tuple[str, str], str],
) -> str | None:
    if requirement["checkpoint"] == "step0":
        return None
    parent = next(
        (
            item
            for item in requirements["models"]
            if item["repository"] == requirement["repository"]
            and item["checkpoint"] == "step0"
        ),
        None,
    )
    if parent is None:
        raise G3AssetPublicationError("trained model has no frozen step0 parent")
    try:
        return published_asset_ids[("model", parent["name"])]
    except KeyError as error:
        raise G3AssetPublicationError(
            "layout order places a trained checkpoint before its step0 parent"
        ) from error


def _publish_entry(
    *,
    root: Path,
    requirements: Mapping[str, Any],
    layout: Mapping[str, Any],
    entry: Mapping[str, Any],
    requirement: Mapping[str, Any],
    generator_git_commit: str,
    checked_at: str,
    published_asset_ids: Mapping[tuple[str, str], str],
    pre_exposure_check: Callable[[], None] | None,
) -> G3AssetPublicationResult:
    raise G3AssetPublicationError(
        "legacy internal all-in-one lifecycle is disabled; VERIFIED evidence is mandatory"
    )


def _publish_stage0_g3_selected(
    requirements: Mapping[str, Any] | str | Path,
    layout: Mapping[str, Any] | str | Path,
    data_root: str | Path,
    *,
    generator_git_commit: str,
    checked_at: str,
    include_derived: bool,
    pre_exposure_check: Callable[[], None] | None,
) -> tuple[G3AssetPublicationResult, ...]:
    raise G3AssetPublicationError(
        "legacy internal lifecycle selection is disabled; use the gate-only API"
    )


def publish_stage0_g3_prerequisites(
    requirements: Mapping[str, Any] | str | Path,
    layout: Mapping[str, Any] | str | Path,
    data_root: str | Path,
    *,
    generator_git_commit: str,
    checked_at: str,
    pre_exposure_check: Callable[[], None] | None = None,
) -> tuple[G3AssetPublicationResult, ...]:
    """Retired unsafe entry point; lifecycle evidence is now mandatory."""

    raise G3AssetPublicationError(
        "legacy prerequisite publication is disabled; run acquisition attestation "
        "and independent verify-only before the gate"
    )


def publish_stage0_g3_assets(
    requirements: Mapping[str, Any] | str | Path,
    layout: Mapping[str, Any] | str | Path,
    data_root: str | Path,
    *,
    generator_git_commit: str,
    checked_at: str,
    pre_exposure_check: Callable[[], None] | None = None,
) -> tuple[G3AssetPublicationResult, ...]:
    """Retired unsafe entry point; lifecycle evidence is now mandatory."""

    raise G3AssetPublicationError(
        "legacy all-in-one publication is disabled; use "
        "gate_stage0_g3_assets_from_evidence"
    )


def _verified_inputs_from_lifecycle(
    root: Path,
    acquisition: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from .g3_lifecycle_evidence import (
        _candidate_from_acquisition_entry,
        g3_downloaded_candidate_ref,
        g3_verified_candidate_ref,
        g3_verification_report_ref,
    )

    if verification["status"] != "PASS":
        raise G3AssetPublicationError("gate refuses a non-PASS verify-only report")
    results: list[dict[str, Any]] = []
    for acquisition_entry, verification_entry in zip(
        acquisition["entries"], verification["entries"]
    ):
        downloaded = _candidate_from_acquisition_entry(
            acquisition, acquisition_entry
        )
        downloaded_ref = g3_downloaded_candidate_ref(
            acquisition_entry["logical_name"], downloaded["candidate_id"]
        )
        downloaded_path = _target_for_ref(root, downloaded_ref)
        if (
            not downloaded_path.exists()
            or not downloaded_path.is_file()
            or _is_link_like(downloaded_path)
        ):
            raise G3AssetPublicationError("DOWNLOADED candidate is missing")
        downloaded_raw = downloaded_path.read_bytes()
        try:
            observed_downloaded = _load_canonical_object(root, downloaded_ref)
        except G3AssetPublicationError:
            raise
        except (
            CanonicalJSONError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
        ) as error:
            raise G3AssetPublicationError(
                "DOWNLOADED candidate cannot be replayed"
            ) from error
        if observed_downloaded != downloaded:
            raise G3AssetPublicationError("DOWNLOADED candidate content mismatch")
        if (
            verification_entry["logical_name"]
            != acquisition_entry["logical_name"]
            or verification_entry["asset_id"] != downloaded["asset_id"]
            or verification_entry["candidate_id"] != downloaded["candidate_id"]
            or verification_entry["downloaded_manifest_ref"] != downloaded_ref
            or verification_entry["downloaded_manifest_sha256"]
            != hashlib.sha256(downloaded_raw).hexdigest()
            or verification_entry["status"] != "PASS"
            or any(item["status"] != "PASS" for item in verification_entry["files"])
        ):
            raise G3AssetPublicationError("verify-only entry binding mismatch")
        expected_verified = transition_manifest(
            downloaded,
            AssetState.VERIFIED,
            actor=verification["actor"],
            actor_role=AssetActorRole.VERIFIER,
            actor_instance_id=verification["actor_instance_id"],
            evidence_ref=g3_verification_report_ref(
                verification["acquisition_sha256"]
            ),
            evidence_sha256=verification["artifact_hash"],
            summary="independent verify-only process matched every declared file",
            at=verification["checked_at"],
        )
        verified_ref = g3_verified_candidate_ref(
            acquisition_entry["logical_name"], downloaded["candidate_id"]
        )
        try:
            observed_verified = _load_canonical_object(root, verified_ref)
        except G3AssetPublicationError:
            raise
        except (
            CanonicalJSONError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
        ) as error:
            raise G3AssetPublicationError(
                "VERIFIED candidate cannot be replayed"
            ) from error
        if observed_verified != expected_verified:
            raise G3AssetPublicationError("VERIFIED candidate content mismatch")
        results.append(observed_verified)
    return results


def gate_stage0_g3_assets_from_evidence(
    requirements: Mapping[str, Any] | str | Path,
    layout: Mapping[str, Any] | str | Path,
    download_plan: Mapping[str, Any] | str | Path,
    source_root: str | Path,
    data_root: str | Path,
    *,
    acquisition_ref: str,
    verification_ref: str,
    generator_git_commit: str,
    checked_at: str,
    gate_actor_instance_id: str,
    pre_exposure_check: Callable[[], None] | None = None,
) -> tuple[G3AssetPublicationResult, ...]:
    """Run only semantic qualification and READY admission over VERIFIED input.

    Acquisition, derived building, DOWNLOADED publication, hashing, and
    VERIFIED publication are deliberately absent from this process boundary.
    """

    from .g3_lifecycle_evidence import (
        _capture_g3_source_snapshot,
        _load_canonical_ref,
        _load_plan,
        _revalidate_g3_source_snapshot,
        g3_acquisition_report_ref,
        g3_verification_report_ref,
        load_g3_acquisition_report,
        load_g3_verify_report,
    )

    commit = _require_text(generator_git_commit, field="generator_git_commit")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise G3AssetPublicationError("generator_git_commit is invalid")
    timestamp = _require_timestamp(checked_at, field="checked_at")
    gate_instance = _require_text(
        gate_actor_instance_id, field="gate_actor_instance_id"
    )
    requirement_value = _load_requirements(requirements)
    layout_value = _load_layout(layout, requirements=requirement_value)
    root = _approved_data_root(data_root)
    try:
        plan_value = _load_plan(
            download_plan,
            requirements=requirement_value,
            layout=layout_value,
        )
        raw_acquisition, _ = _load_canonical_ref(root, acquisition_ref)
        source_control_refs = (
            raw_acquisition["requirements_ref"],
            raw_acquisition["layout_ref"],
            raw_acquisition["download_plan_ref"],
            *(entry["spec_ref"] for entry in plan_value["entries"]),
        )
        source_snapshot = _capture_g3_source_snapshot(
            source_root,
            expected_commit=commit,
            control_refs=source_control_refs,
        )
        source = Path(source_snapshot.source_root)
        tracked_requirements, _ = _load_canonical_ref(
            source, raw_acquisition["requirements_ref"]
        )
        tracked_layout, _ = _load_canonical_ref(
            source, raw_acquisition["layout_ref"]
        )
        tracked_plan, _ = _load_canonical_ref(
            source, raw_acquisition["download_plan_ref"]
        )
        if (
            tracked_requirements != requirement_value
            or tracked_layout != layout_value
            or tracked_plan != plan_value
        ):
            raise G3AssetPublicationError(
                "gate inputs differ from their tracked source refs"
            )
        acquisition = load_g3_acquisition_report(
            root,
            acquisition_ref,
            requirements=requirement_value,
            layout=layout_value,
            download_plan=plan_value,
            source_root=source,
        )
        verification = load_g3_verify_report(
            root,
            verification_ref,
            acquisition=acquisition,
            requirements=requirement_value,
            layout=layout_value,
        )
        verified_inputs = _verified_inputs_from_lifecycle(
            root, acquisition, verification
        )
    except G3AssetPublicationError:
        raise
    except (
        CanonicalJSONError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        raise G3AssetPublicationError(
            "lifecycle evidence replay failed"
        ) from error

    def source_guard() -> None:
        if pre_exposure_check is not None:
            pre_exposure_check()
        try:
            _revalidate_g3_source_snapshot(source_snapshot)
        except G3AssetPublicationError:
            raise
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise G3AssetPublicationError(
                "gate source binding drifted during execution"
            ) from error

    source_guard()
    if (
        acquisition_ref != g3_acquisition_report_ref(acquisition["artifact_hash"])
        or verification_ref
        != g3_verification_report_ref(acquisition["artifact_hash"])
        or verification["artifact_hash"]
        != next(
            event["evidence_sha256"]
            for event in verified_inputs[0]["state_history"]
            if event["to"] == AssetState.VERIFIED.value
        )
    ):
        raise G3AssetPublicationError("lifecycle report reference/hash mismatch")
    if commit != acquisition["source_git_commit"]:
        raise G3AssetPublicationError("gate source commit differs from acquisition")
    if gate_instance in {
        acquisition["actor_instance_id"],
        verification["actor_instance_id"],
    }:
        raise G3AssetPublicationError(
            "gate actor_instance_id must differ from fetcher and verifier"
        )
    if len(verified_inputs) != _EXPECTED_ENTRY_COUNT:
        raise G3AssetPublicationError("gate requires thirteen VERIFIED inputs")
    requirement_by_entry = _requirement_index(requirement_value)
    asset_ids = {
        (entry["kind"], entry["requirement_name"]): verified["asset_id"]
        for entry, verified in zip(layout_value["entries"], verified_inputs)
    }
    results: list[G3AssetPublicationResult] = []
    for entry, verified in zip(layout_value["entries"], verified_inputs):
        identity = (entry["kind"], entry["requirement_name"])
        requirement = requirement_by_entry[identity]
        acquisition_entry = next(
            item
            for item in acquisition["entries"]
            if item["logical_name"] == entry["logical_name"]
        )
        if any(
            verified[field] != acquisition_entry[field]
            for field in (
                "asset_id",
                "asset_type",
                "name",
                "source",
                "revision",
                "files",
                "metadata",
            )
        ):
            raise G3AssetPublicationError("VERIFIED identity differs from acquisition")
        asset_root = _asset_root(root, entry["asset_root_ref"])
        tokenizer_id = asset_ids.get(("tokenizer", "pythia-tokenizer"))
        raw_parent_id = asset_ids.get(("glue_raw", entry["requirement_name"]))
        candidate_ref = g3_candidate_artifact_ref(
            entry["logical_name"], verified["candidate_id"]
        )
        semantic_root = (
            f"manifests/evidence/g3/{entry['logical_name']}/"
            f"{verified['candidate_id']}"
        )
        semantic_ref = f"{semantic_root}/semantic-verification.json"
        final_target = _target_for_ref(root, entry["manifest_ref"])
        qualification_target = _target_for_ref(root, entry["qualification_ref"])

        if final_target.exists() or _is_link_like(final_target):
            ready = _load_canonical_object(root, entry["manifest_ref"])
            qualification = _load_canonical_object(root, entry["qualification_ref"])
            validate_g3_manifest(ready)
            validate_g3_qualification(qualification)
            expected_verified = deepcopy(ready)
            expected_verified["state"] = AssetState.VERIFIED.value
            expected_verified["state_history"].pop()
            if expected_verified != verified:
                raise G3AssetPublicationError(
                    "existing READY does not descend from the supplied VERIFIED input"
                )
            final_event = ready["state_history"][-1]
            if (
                final_event["actor_instance_id"] != gate_instance
                or final_event["evidence_ref"] != entry["qualification_ref"]
                or final_event["evidence_sha256"] != qualification["artifact_hash"]
                or qualification["acquisition_ref"] != acquisition_ref
                or qualification["acquisition_sha256"]
                != acquisition["artifact_hash"]
                or qualification["verification_ref"] != verification_ref
                or qualification["verification_sha256"]
                != verification["artifact_hash"]
            ):
                raise G3AssetPublicationError(
                    "existing READY is bound to a different lifecycle or gate actor"
                )
            semantic, semantic_sha = _load_existing_semantic(
                root,
                semantic_ref,
                entry=entry,
                verified=verified,
                requirements_ref=layout_value["requirements_ref"],
                requirements_sha256=requirement_value["artifact_hash"],
                generator_git_commit=commit,
            )
            _validate_qualification_evidence(
                qualification,
                kind=entry["kind"],
                requirements_ref=layout_value["requirements_ref"],
                requirements_sha256=requirement_value["artifact_hash"],
                generator_git_commit=commit,
                verification_ref=verification_ref,
                verification_sha256=verification["artifact_hash"],
                semantic_ref=semantic_ref,
                semantic_sha256=semantic_sha,
                acquisition_ref=acquisition_ref,
                acquisition_sha256=acquisition["artifact_hash"],
            )
            source_guard()
            _replay_current_semantics(
                data_root=root,
                entry=entry,
                requirement=requirement,
                asset_root=asset_root,
                verified=verified,
                evidence=semantic,
                requirements_ref=layout_value["requirements_ref"],
                requirements_sha256=requirement_value["artifact_hash"],
                generator_git_commit=commit,
                tokenizer_asset_id=tokenizer_id,
                raw_glue_parent_asset_id=raw_parent_id,
            )
            resolve_qualified_asset(
                ready,
                asset_root,
                qualification,
                qualification_ref=entry["qualification_ref"],
                requirements_artifact_hash=requirement_value["artifact_hash"],
            )
            results.append(
                G3AssetPublicationResult(
                    logical_name=entry["logical_name"],
                    kind=entry["kind"],
                    asset_id=ready["asset_id"],
                    candidate_id=ready["candidate_id"],
                    state=ready["state"],
                    status="existing_ready",
                    manifest_ref=entry["manifest_ref"],
                    candidate_ref=candidate_ref,
                    qualification_ref=entry["qualification_ref"],
                    verification_ref=verification_ref,
                    semantic_evidence_ref=semantic_ref,
                )
            )
            continue

        if qualification_target.exists() or _is_link_like(qualification_target):
            qualification = _load_canonical_object(root, entry["qualification_ref"])
            validate_g3_qualification(qualification)
            semantic, semantic_sha = _load_existing_semantic(
                root,
                semantic_ref,
                entry=entry,
                verified=verified,
                requirements_ref=layout_value["requirements_ref"],
                requirements_sha256=requirement_value["artifact_hash"],
                generator_git_commit=commit,
            )
            _validate_qualification_evidence(
                qualification,
                kind=entry["kind"],
                requirements_ref=layout_value["requirements_ref"],
                requirements_sha256=requirement_value["artifact_hash"],
                generator_git_commit=commit,
                verification_ref=verification_ref,
                verification_sha256=verification["artifact_hash"],
                semantic_ref=semantic_ref,
                semantic_sha256=semantic_sha,
                acquisition_ref=acquisition_ref,
                acquisition_sha256=acquisition["artifact_hash"],
            )
            source_guard()
            _replay_current_semantics(
                data_root=root,
                entry=entry,
                requirement=requirement,
                asset_root=asset_root,
                verified=verified,
                evidence=semantic,
                requirements_ref=layout_value["requirements_ref"],
                requirements_sha256=requirement_value["artifact_hash"],
                generator_git_commit=commit,
                tokenizer_asset_id=tokenizer_id,
                raw_glue_parent_asset_id=raw_parent_id,
            )
            ready_at = timestamp
            if datetime.fromisoformat(ready_at.replace("Z", "+00:00")) < datetime.fromisoformat(
                qualification["checked_at"].replace("Z", "+00:00")
            ):
                raise G3AssetPublicationError(
                    "gate checked_at precedes recovered qualification"
                )
            publication_status = "recovered_ready"
        else:
            source_guard()
            details = _run_guarded_semantic_probe(
                kind=entry["kind"],
                requirement=requirement,
                data_root=root,
                asset_root=asset_root,
                verified=verified,
                tokenizer_asset_id=tokenizer_id,
                raw_glue_parent_asset_id=raw_parent_id,
            )
            semantic = _semantic_evidence(
                entry=entry,
                verified=verified,
                requirements_ref=layout_value["requirements_ref"],
                requirements_sha256=requirement_value["artifact_hash"],
                generator_git_commit=commit,
                checked_at=timestamp,
                details_by_check=details,
                network_attempts=0,
            )
            if not semantic_observations_match(
                entry["kind"],
                requirement,
                verified,
                semantic,
                asset_root=asset_root,
                data_root=root,
            ):
                raise G3AssetPublicationError(
                    "semantic probe observations do not match frozen requirements"
                )
            semantic_sha = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
            qualification = build_g3_qualification(
                verified,
                requirements_ref=layout_value["requirements_ref"],
                requirements_sha256=requirement_value["artifact_hash"],
                acquisition_ref=acquisition_ref,
                acquisition_sha256=acquisition["artifact_hash"],
                verification_ref=verification_ref,
                verification_sha256=verification["artifact_hash"],
                checks=_qualification_checks(
                    entry["kind"],
                    verification_ref=verification_ref,
                    verification_sha256=verification["artifact_hash"],
                    semantic_ref=semantic_ref,
                    semantic_sha256=semantic_sha,
                ),
                checked_at=timestamp,
                generator_git_commit=commit,
                network_attempts=0,
            )
            # The semantic probe can be long-running.  Re-check the frozen
            # source boundary before exposing either evidence artifact so a
            # mid-probe source mutation leaves no new gate evidence behind.
            source_guard()
            _publish_no_clobber(root, semantic_ref, semantic)
            _publish_no_clobber(root, entry["qualification_ref"], qualification)
            publication_status = "published"
        ready = admit_g3_qualification(
            verified,
            qualification,
            qualification_ref=entry["qualification_ref"],
            requirements_artifact_hash=requirement_value["artifact_hash"],
            actor="stage0-g3-gate",
            actor_instance_id=gate_instance,
            summary="formal offline semantic qualification admitted",
            at=timestamp,
        )
        resolve_qualified_asset(
            ready,
            asset_root,
            qualification,
            qualification_ref=entry["qualification_ref"],
            requirements_artifact_hash=requirement_value["artifact_hash"],
        )
        source_guard()
        _publish_no_clobber(root, entry["manifest_ref"], ready)
        results.append(
            G3AssetPublicationResult(
                logical_name=entry["logical_name"],
                kind=entry["kind"],
                asset_id=ready["asset_id"],
                candidate_id=ready["candidate_id"],
                state=ready["state"],
                status=publication_status,
                manifest_ref=entry["manifest_ref"],
                candidate_ref=candidate_ref,
                qualification_ref=entry["qualification_ref"],
                verification_ref=verification_ref,
                semantic_evidence_ref=semantic_ref,
            )
        )
    source_guard()
    return tuple(results)


__all__ = [
    "G3AssetPublicationError",
    "G3AssetPublicationResult",
    "GENERATOR_VERSION",
    "NetworkEgressAttempt",
    "SCHEMA_VERSION",
    "SEMANTIC_EVIDENCE_SCHEMA_VERSION",
    "gate_stage0_g3_assets_from_evidence",
    "semantic_evidence_artifact_hash",
    "validate_semantic_evidence",
]
