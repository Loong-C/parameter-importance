from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Final, Iterator

from .atomic import atomic_write_bytes, sha256_file, stable_json_bytes, stable_json_hash


SCHEMA_VERSION: Final = "stage0-asset-manifest-v1"
VERIFICATION_SCHEMA_VERSION: Final = "stage0-asset-verification-v1"
QUALIFICATION_SCHEMA_VERSION: Final = "stage0-asset-qualification-v1"
G3_CONTRACT_PROFILE: Final = "stage0-g3-v1"
G3_MODEL_METADATA_VERSION: Final = "stage0-model-metadata-v1"
G3_TOKENIZER_METADATA_VERSION: Final = "stage0-tokenizer-metadata-v1"
G3_DATASET_METADATA_VERSION: Final = "stage0-dataset-metadata-v1"
G3_SOURCE_METADATA_VERSION: Final = "stage0-source-metadata-v1"
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class AssetManifestError(ValueError):
    """Base class for an invalid or unusable asset manifest."""


class AssetEncodingError(AssetManifestError):
    """Raised when a manifest is not strict UTF-8 JSON without a BOM."""


class AssetValidationError(AssetManifestError):
    """Raised when a decoded manifest violates the Stage 0 contract."""


class AssetVerificationError(AssetManifestError):
    """Raised when files do not match a valid manifest."""


class AssetNotReadyError(AssetManifestError):
    """Raised when a non-ready manifest is passed to the runtime resolver."""


class AssetType(StrEnum):
    MODEL = "model"
    TOKENIZER = "tokenizer"
    DATASET = "dataset"
    SOURCE = "source"


class AssetState(StrEnum):
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    VERIFIED = "verified"
    READY = "ready"
    INVALID = "invalid"


class AssetActorRole(StrEnum):
    FETCHER = "fetcher"
    VERIFIER = "verifier"
    GATE = "gate"


ASSET_TRANSITIONS: Final[dict[AssetState, frozenset[AssetState]]] = {
    AssetState.DOWNLOADING: frozenset({AssetState.DOWNLOADED}),
    AssetState.DOWNLOADED: frozenset({AssetState.VERIFIED, AssetState.INVALID}),
    AssetState.VERIFIED: frozenset({AssetState.READY, AssetState.INVALID}),
    # A later integrity audit may invalidate an admitted asset, but a new
    # candidate manifest must be published rather than overwriting history.
    AssetState.READY: frozenset({AssetState.INVALID}),
    AssetState.INVALID: frozenset(),
}
ROLE_TRANSITIONS: Final[
    dict[AssetActorRole, frozenset[tuple[AssetState, AssetState]]]
] = {
    AssetActorRole.FETCHER: frozenset(
        {(AssetState.DOWNLOADING, AssetState.DOWNLOADED)}
    ),
    AssetActorRole.VERIFIER: frozenset(
        {
            (AssetState.DOWNLOADED, AssetState.VERIFIED),
            (AssetState.DOWNLOADED, AssetState.INVALID),
            (AssetState.VERIFIED, AssetState.INVALID),
        }
    ),
    AssetActorRole.GATE: frozenset(
        {
            (AssetState.VERIFIED, AssetState.READY),
            (AssetState.READY, AssetState.INVALID),
        }
    ),
}

_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "asset_id",
        "asset_type",
        "state",
        "name",
        "source",
        "revision",
        "files",
        "state_history",
        "created_at",
        "generator_version",
        "metadata",
        "contract_profile",
        "candidate_id",
        "generator_git_commit",
    }
)
_REQUIRED_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "asset_id",
        "asset_type",
        "state",
        "name",
        "source",
        "revision",
        "files",
        "state_history",
        "created_at",
        "generator_version",
        "metadata",
    }
)
_FILE_FIELDS: Final = frozenset({"path", "size_bytes", "sha256", "role"})
_LEGACY_HISTORY_FIELDS: Final = frozenset(
    {
        "from",
        "to",
        "at",
        "actor",
        "actor_role",
        "evidence_ref",
        "summary",
    }
)
_G3_HISTORY_FIELDS: Final = _LEGACY_HISTORY_FIELDS | frozenset(
    {"actor_instance_id", "evidence_sha256"}
)
_G3_TOP_LEVEL_FIELDS: Final = frozenset(
    {"contract_profile", "candidate_id", "generator_git_commit"}
)
_QUALIFICATION_FIELDS: Final = frozenset(
    {
        "schema_version",
        "formal",
        "asset_id",
        "candidate_id",
        "verified_manifest_sha256",
        "requirements_ref",
        "requirements_sha256",
        "acquisition_ref",
        "acquisition_sha256",
        "verification_ref",
        "verification_sha256",
        "checks",
        "network_attempts",
        "checked_at",
        "generator_git_commit",
        "artifact_hash",
    }
)
_QUALIFICATION_CHECK_FIELDS: Final = frozenset(
    {"check_id", "status", "evidence_ref", "evidence_sha256", "summary"}
)
_TEMPORARY_PATH_TOKENS: Final = frozenset({"part", "partial", "lock", "tmp", "temp"})
_GENERIC_REVISIONS: Final = frozenset(
    {
        "unknown",
        "latest",
        "main",
        "master",
        "head",
        "default",
        "current",
        "none",
        "null",
        "unspecified",
        "na",
        "n/a",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_text(value: Any, *, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise AssetValidationError(f"{field} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise AssetValidationError(f"{field} contains a control character")
    return value


def _require_timestamp(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise AssetValidationError(f"{field} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise AssetValidationError(f"{field} must include a timezone")
    return text


def _timestamp_value(value: Any, *, field: str) -> datetime:
    text = _require_timestamp(value, field=field)
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _require_integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise AssetValidationError(f"{field} must be a {qualifier} integer")
    return value


def _require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise AssetValidationError(f"{field} must be 64 lowercase hex characters")
    return value


def _require_git_commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not GIT_COMMIT_PATTERN.fullmatch(value):
        raise AssetValidationError(
            f"{field} must be a 40- or 64-character lowercase Git digest"
        )
    return value


def _require_exact_fields(
    value: Any,
    *,
    field: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetValidationError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AssetValidationError(f"{field} object keys must be strings")
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise AssetValidationError(
            f"Invalid {field} fields; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return deepcopy(dict(value))


def _validate_revision(value: Any) -> str:
    revision = _require_text(value, field="revision", maximum=512)
    if revision != revision.strip():
        raise AssetValidationError("revision may not contain surrounding whitespace")
    if revision.casefold() in _GENERIC_REVISIONS:
        raise AssetValidationError(
            f"revision must be immutable and specific, not {revision!r}"
        )
    return revision


def _validate_metadata(
    asset_type: AssetType,
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetValidationError("metadata must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AssetValidationError("metadata object keys must be strings")
    metadata = deepcopy(dict(value))
    try:
        stable_json_bytes(metadata)
    except (TypeError, ValueError) as error:
        raise AssetValidationError("metadata is not canonical JSON data") from error

    if asset_type is AssetType.MODEL:
        _require_text(
            metadata.get("architecture"),
            field="metadata.architecture",
            maximum=256,
        )
        _require_integer(
            metadata.get("parameter_count"),
            field="metadata.parameter_count",
            minimum=1,
        )
        _require_text(metadata.get("dtype"), field="metadata.dtype", maximum=128)
        _require_text(
            metadata.get("initialization_id"),
            field="metadata.initialization_id",
            maximum=512,
        )
    elif asset_type is AssetType.TOKENIZER:
        _require_text(
            metadata.get("tokenizer_class"),
            field="metadata.tokenizer_class",
            maximum=256,
        )
        _require_integer(
            metadata.get("vocab_size"),
            field="metadata.vocab_size",
            minimum=1,
        )
        special_tokens = metadata.get("special_tokens")
        if not isinstance(special_tokens, Mapping):
            raise AssetValidationError("metadata.special_tokens must be an object")
        if any(not isinstance(key, str) for key in special_tokens):
            raise AssetValidationError(
                "metadata.special_tokens object keys must be strings"
            )
        _require_text(
            metadata.get("normalization"),
            field="metadata.normalization",
            maximum=256,
        )
    elif asset_type is AssetType.DATASET:
        splits = metadata.get("splits")
        if not isinstance(splits, Mapping) or not splits:
            raise AssetValidationError("metadata.splits must be a non-empty object")
        for split_name, split in splits.items():
            normalized_name = _require_text(
                split_name,
                field="metadata.splits key",
                maximum=128,
            )
            if not isinstance(split, Mapping):
                raise AssetValidationError(
                    f"metadata.splits.{normalized_name} must be an object"
                )
            _require_integer(
                split.get("sample_count"),
                field=f"metadata.splits.{normalized_name}.sample_count",
                minimum=0,
            )
            fields = split.get("fields")
            if not isinstance(fields, list) or not fields:
                raise AssetValidationError(
                    f"metadata.splits.{normalized_name}.fields must be a non-empty array"
                )
            normalized_fields = [
                _require_text(
                    field_name,
                    field=f"metadata.splits.{normalized_name}.fields",
                    maximum=128,
                )
                for field_name in fields
            ]
            if len(normalized_fields) != len(set(normalized_fields)):
                raise AssetValidationError(
                    f"metadata.splits.{normalized_name}.fields must be unique"
                )
        _require_text(
            metadata.get("preprocessing_version"),
            field="metadata.preprocessing_version",
            maximum=256,
        )
    else:
        _require_text(
            metadata.get("source_kind"),
            field="metadata.source_kind",
            maximum=128,
        )
        _require_text(
            metadata.get("license"),
            field="metadata.license",
            maximum=256,
        )
    return metadata


def _g3_file_index(
    files: Iterable[AssetFile | Mapping[str, Any]],
) -> dict[str, AssetFile]:
    return {item.path: item for item in _normalize_files(files)}


def _require_file_binding(
    *,
    path: Any,
    sha256: Any,
    file_index: Mapping[str, AssetFile],
    field: str,
) -> AssetFile:
    normalized_path = validate_asset_path(path)
    digest = _require_hash(sha256, field=f"{field}.sha256")
    descriptor = file_index.get(normalized_path)
    if descriptor is None:
        raise AssetValidationError(
            f"{field}.path must reference an entry in manifest.files"
        )
    if descriptor.sha256 != digest:
        raise AssetValidationError(
            f"{field}.sha256 must match manifest.files for {normalized_path!r}"
        )
    return descriptor


def _validate_g3_model_metadata(
    metadata: Mapping[str, Any],
    file_index: Mapping[str, AssetFile],
) -> None:
    required = frozenset(
        {
            "contract_version",
            "architecture",
            "parameter_count",
            "tensor_count",
            "dtype",
            "dtype_counts",
            "max_position_embeddings",
            "config_path",
            "config_sha256",
            "initialization_id",
            "initialization_kind",
            "training_step",
            "parent_model_asset_id",
        }
    )
    value = _require_exact_fields(metadata, field="metadata", required=required)
    if value["contract_version"] != G3_MODEL_METADATA_VERSION:
        raise AssetValidationError(
            f"metadata.contract_version must be {G3_MODEL_METADATA_VERSION!r}"
        )
    _require_text(value["architecture"], field="metadata.architecture", maximum=256)
    parameter_count = _require_integer(
        value["parameter_count"], field="metadata.parameter_count", minimum=1
    )
    _require_integer(value["tensor_count"], field="metadata.tensor_count", minimum=1)
    dtype_summary = _require_text(
        value["dtype"], field="metadata.dtype", maximum=128
    )
    dtype_counts = value["dtype_counts"]
    if not isinstance(dtype_counts, Mapping) or not dtype_counts:
        raise AssetValidationError("metadata.dtype_counts must be a non-empty object")
    normalized_dtype_counts: dict[str, int] = {}
    for raw_dtype, raw_count in dtype_counts.items():
        dtype_name = _require_text(
            raw_dtype, field="metadata.dtype_counts key", maximum=128
        )
        normalized_dtype_counts[dtype_name] = _require_integer(
            raw_count,
            field=f"metadata.dtype_counts.{dtype_name}",
            minimum=1,
        )
    if sum(normalized_dtype_counts.values()) != parameter_count:
        raise AssetValidationError(
            "metadata.dtype_counts values must sum to metadata.parameter_count"
        )
    expected_dtype_summary = (
        next(iter(normalized_dtype_counts))
        if len(normalized_dtype_counts) == 1
        else "mixed"
    )
    if dtype_summary != expected_dtype_summary:
        raise AssetValidationError(
            "metadata.dtype must equal the sole dtype_counts key or be 'mixed' "
            "when multiple dtypes are present"
        )
    _require_integer(
        value["max_position_embeddings"],
        field="metadata.max_position_embeddings",
        minimum=1,
    )
    _require_file_binding(
        path=value["config_path"],
        sha256=value["config_sha256"],
        file_index=file_index,
        field="metadata.config",
    )
    _require_text(
        value["initialization_id"], field="metadata.initialization_id", maximum=512
    )
    initialization_kind = value["initialization_kind"]
    if initialization_kind not in {"base_initialization", "trained_checkpoint"}:
        raise AssetValidationError(
            "metadata.initialization_kind must be base_initialization or "
            "trained_checkpoint"
        )
    training_step = _require_integer(
        value["training_step"], field="metadata.training_step", minimum=0
    )
    parent = value["parent_model_asset_id"]
    if parent is not None:
        _require_hash(parent, field="metadata.parent_model_asset_id")
    if initialization_kind == "base_initialization":
        if training_step != 0 or parent is not None:
            raise AssetValidationError(
                "base_initialization requires training_step=0 and "
                "parent_model_asset_id=null"
            )
    elif training_step == 0 or parent is None:
        raise AssetValidationError(
            "trained_checkpoint requires training_step>0 and a parent_model_asset_id"
        )


def _validate_g3_tokenizer_metadata(
    metadata: Mapping[str, Any],
    file_index: Mapping[str, AssetFile],
) -> None:
    required = frozenset(
        {
            "contract_version",
            "tokenizer_class",
            "implementation_version",
            "vocab_size",
            "token_count_with_added_tokens",
            "vocab_mapping_sha256",
            "glue_padding_policy",
            "special_tokens",
            "normalization",
            "config_path",
            "config_sha256",
        }
    )
    value = _require_exact_fields(
        metadata,
        field="metadata",
        required=required,
        optional=frozenset({"model_max_length"}),
    )
    if value["contract_version"] != G3_TOKENIZER_METADATA_VERSION:
        raise AssetValidationError(
            f"metadata.contract_version must be {G3_TOKENIZER_METADATA_VERSION!r}"
        )
    _require_text(
        value["tokenizer_class"], field="metadata.tokenizer_class", maximum=256
    )
    _require_text(
        value["implementation_version"],
        field="metadata.implementation_version",
        maximum=256,
    )
    vocab_size = _require_integer(
        value["vocab_size"], field="metadata.vocab_size", minimum=1
    )
    token_count = _require_integer(
        value["token_count_with_added_tokens"],
        field="metadata.token_count_with_added_tokens",
        minimum=1,
    )
    if token_count < vocab_size:
        raise AssetValidationError(
            "metadata.token_count_with_added_tokens may not be smaller than "
            "metadata.vocab_size"
        )
    _require_hash(
        value["vocab_mapping_sha256"],
        field="metadata.vocab_mapping_sha256",
    )
    if value["glue_padding_policy"] != "explicit_existing_added_token":
        raise AssetValidationError(
            "metadata.glue_padding_policy must be "
            "explicit_existing_added_token"
        )
    special_tokens = value["special_tokens"]
    if not isinstance(special_tokens, Mapping) or not special_tokens:
        raise AssetValidationError(
            "metadata.special_tokens must be a non-empty object for G3"
        )
    for token_name, token_identity in special_tokens.items():
        normalized_name = _require_text(
            token_name, field="metadata.special_tokens key", maximum=128
        )
        identity = _require_exact_fields(
            token_identity,
            field=f"metadata.special_tokens.{normalized_name}",
            required=frozenset({"token", "token_id"}),
        )
        _require_text(
            identity["token"],
            field=f"metadata.special_tokens.{normalized_name}.token",
            maximum=512,
        )
        token_id = _require_integer(
            identity["token_id"],
            field=f"metadata.special_tokens.{normalized_name}.token_id",
            minimum=0,
        )
        if token_id >= token_count:
            raise AssetValidationError(
                f"metadata.special_tokens.{normalized_name}.token_id must be "
                "smaller than metadata.token_count_with_added_tokens"
            )
    if special_tokens.get("glue_padding_token") != {
        "token": "<|padding|>",
        "token_id": 1,
    }:
        raise AssetValidationError(
            "metadata.special_tokens.glue_padding_token must bind "
            "'<|padding|>' to token_id=1"
        )
    _require_text(
        value["normalization"], field="metadata.normalization", maximum=256
    )
    _require_file_binding(
        path=value["config_path"],
        sha256=value["config_sha256"],
        file_index=file_index,
        field="metadata.config",
    )
    if "model_max_length" in value:
        _require_integer(
            value["model_max_length"],
            field="metadata.model_max_length",
            minimum=1,
        )


def _validate_g3_dataset_splits(value: Any) -> None:
    if not isinstance(value, Mapping) or not value:
        raise AssetValidationError("metadata.splits must be a non-empty object")
    for split_name, raw_split in value.items():
        normalized_name = _require_text(
            split_name, field="metadata.splits key", maximum=128
        )
        split = _require_exact_fields(
            raw_split,
            field=f"metadata.splits.{normalized_name}",
            required=frozenset({"sample_count", "fields", "cursor"}),
        )
        sample_count = _require_integer(
            split["sample_count"],
            field=f"metadata.splits.{normalized_name}.sample_count",
            minimum=0,
        )
        fields = split["fields"]
        if not isinstance(fields, list) or not fields:
            raise AssetValidationError(
                f"metadata.splits.{normalized_name}.fields must be a non-empty array"
            )
        normalized_fields = [
            _require_text(
                field_name,
                field=f"metadata.splits.{normalized_name}.fields",
                maximum=128,
            )
            for field_name in fields
        ]
        if len(normalized_fields) != len(set(normalized_fields)):
            raise AssetValidationError(
                f"metadata.splits.{normalized_name}.fields must be unique"
            )
        cursor = _require_exact_fields(
            split["cursor"],
            field=f"metadata.splits.{normalized_name}.cursor",
            required=frozenset({"start", "stop"}),
        )
        start = _require_integer(
            cursor["start"],
            field=f"metadata.splits.{normalized_name}.cursor.start",
            minimum=0,
        )
        stop = _require_integer(
            cursor["stop"],
            field=f"metadata.splits.{normalized_name}.cursor.stop",
            minimum=0,
        )
        if stop < start or stop - start != sample_count:
            raise AssetValidationError(
                f"metadata.splits.{normalized_name}.cursor must span sample_count"
            )


def _validate_g3_preprocessing(value: Any, *, preprocessing_version: str) -> dict[str, Any]:
    preprocessing = _require_exact_fields(
        value,
        field="metadata.preprocessing",
        required=frozenset(
            {
                "version",
                "config_hash",
                "code_git_commit",
                "tokenizer_asset_id",
                "parent_asset_ids",
            }
        ),
    )
    version = _require_text(
        preprocessing["version"], field="metadata.preprocessing.version", maximum=256
    )
    if version != preprocessing_version:
        raise AssetValidationError(
            "metadata.preprocessing.version must equal metadata.preprocessing_version"
        )
    _require_hash(
        preprocessing["config_hash"], field="metadata.preprocessing.config_hash"
    )
    _require_git_commit(
        preprocessing["code_git_commit"],
        field="metadata.preprocessing.code_git_commit",
    )
    tokenizer_id = preprocessing["tokenizer_asset_id"]
    if tokenizer_id is not None:
        _require_hash(tokenizer_id, field="metadata.preprocessing.tokenizer_asset_id")
    parents = preprocessing["parent_asset_ids"]
    if not isinstance(parents, list):
        raise AssetValidationError(
            "metadata.preprocessing.parent_asset_ids must be an array"
        )
    normalized_parents = [
        _require_hash(item, field="metadata.preprocessing.parent_asset_ids")
        for item in parents
    ]
    if len(normalized_parents) != len(set(normalized_parents)):
        raise AssetValidationError(
            "metadata.preprocessing.parent_asset_ids must be unique"
        )
    return preprocessing


def _require_exact_integer_pair(value: Any, *, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise AssetValidationError(f"{field} must be a two-integer half-open interval")
    start = _require_integer(value[0], field=f"{field}[0]", minimum=0)
    stop = _require_integer(value[1], field=f"{field}[1]", minimum=0)
    if stop < start:
        raise AssetValidationError(f"{field} must be a valid half-open interval")
    return start, stop


def _validate_g3_causal_lm_mapping(value: Any, *, field: str) -> None:
    mapping = _require_exact_fields(
        value,
        field=field,
        required=frozenset(
            {
                "labels_alignment",
                "source_tokens_per_record",
                "input_sequence_length",
                "label_sequence_length",
                "input_slice",
                "label_slice",
                "attention_mask_policy",
                "effective_target_tokens",
                "loss_adapter_id",
            }
        ),
    )
    expected_scalars = {
        "labels_alignment": "pre_shifted_next_token",
        "source_tokens_per_record": 2049,
        "input_sequence_length": 2048,
        "label_sequence_length": 2048,
        "attention_mask_policy": "all_one_for_fixed_full_record",
        "effective_target_tokens": 2048,
        "loss_adapter_id": "pre-shifted-next-token-cross-entropy-v1",
    }
    for field_name, expected in expected_scalars.items():
        if mapping[field_name] != expected or (
            isinstance(expected, int) and isinstance(mapping[field_name], bool)
        ):
            raise AssetValidationError(
                f"{field}.{field_name} must be {expected!r}"
            )
    if _require_exact_integer_pair(
        mapping["input_slice"], field=f"{field}.input_slice"
    ) != (0, 2048):
        raise AssetValidationError(f"{field}.input_slice must be [0, 2048)")
    if _require_exact_integer_pair(
        mapping["label_slice"], field=f"{field}.label_slice"
    ) != (1, 2049):
        raise AssetValidationError(f"{field}.label_slice must be [1, 2049)")


def _validate_reference_batch_hashes(
    value: Any,
) -> None:
    field = "metadata.storage.reference_batch_sha256"
    if not isinstance(value, Mapping):
        raise AssetValidationError(f"{field} must be an object")
    required_keys = {"0", "1", "511"}
    if set(value) != required_keys:
        raise AssetValidationError(
            f"{field} must contain exactly batch keys 0, 1, and 511"
        )
    for raw_index, raw_digest in value.items():
        if not isinstance(raw_index, str) or re.fullmatch(r"(?:0|[1-9][0-9]*)", raw_index) is None:
            raise AssetValidationError(
                f"{field} keys must be canonical non-negative integer strings"
            )
        _require_hash(raw_digest, field=f"{field}.{raw_index}")


def _validate_g3_raw_pile_storage(
    value: Any,
    file_index: Mapping[str, AssetFile],
) -> None:
    storage = _require_exact_fields(
        value,
        field="metadata.storage",
        required=frozenset(
            {
                "kind",
                "idx",
                "tokens_per_record",
                "global_byte_coverage",
                "required_cursor_stop",
                "causal_lm_mapping",
                "reference_reader",
                "reference_batch_size",
                "reference_batch_sha256",
                "last_required_record_sha256",
                "cross_shard_policy",
                "shards",
            }
        ),
    )
    if storage["kind"] != "pythia_mmap_shards":
        raise AssetValidationError(
            "raw_indexed_mmap requires metadata.storage.kind=pythia_mmap_shards"
        )
    idx = _require_exact_fields(
        storage["idx"],
        field="metadata.storage.idx",
        required=frozenset(
            {
                "path",
                "sha256",
                "magic",
                "version",
                "dtype_code",
                "itemsize_bytes",
                "sequence_count",
                "document_count",
            }
        ),
    )
    idx_file = _require_file_binding(
        path=idx["path"],
        sha256=idx["sha256"],
        file_index=file_index,
        field="metadata.storage.idx",
    )
    _require_text(idx["magic"], field="metadata.storage.idx.magic", maximum=128)
    _require_integer(idx["version"], field="metadata.storage.idx.version", minimum=0)
    _require_integer(
        idx["dtype_code"], field="metadata.storage.idx.dtype_code", minimum=0
    )
    itemsize = _require_integer(
        idx["itemsize_bytes"],
        field="metadata.storage.idx.itemsize_bytes",
        minimum=1,
    )
    sequence_count = _require_integer(
        idx["sequence_count"],
        field="metadata.storage.idx.sequence_count",
        minimum=0,
    )
    _require_integer(
        idx["document_count"],
        field="metadata.storage.idx.document_count",
        minimum=0,
    )
    tokens_per_record = _require_integer(
        storage["tokens_per_record"],
        field="metadata.storage.tokens_per_record",
        minimum=1,
    )
    if tokens_per_record != 2049:
        raise AssetValidationError(
            "Pythia mmap metadata.storage.tokens_per_record must be 2049"
        )
    if storage["cross_shard_policy"] != (
        "explicit_ordered_global_byte_stream_selected_prefix_must_be_fully_covered"
    ):
        raise AssetValidationError(
            "metadata.storage.cross_shard_policy must be "
            "explicit_ordered_global_byte_stream_selected_prefix_must_be_fully_covered"
        )
    coverage = _require_exact_fields(
        storage["global_byte_coverage"],
        field="metadata.storage.global_byte_coverage",
        required=frozenset({"start", "stop"}),
    )
    coverage_start = _require_integer(
        coverage["start"],
        field="metadata.storage.global_byte_coverage.start",
        minimum=0,
    )
    coverage_stop = _require_integer(
        coverage["stop"],
        field="metadata.storage.global_byte_coverage.stop",
        minimum=0,
    )
    if coverage_start != 0 or coverage_stop <= coverage_start:
        raise AssetValidationError(
            "metadata.storage.global_byte_coverage must be a non-empty [0, stop) interval"
        )
    shards = storage["shards"]
    if not isinstance(shards, list) or not shards:
        raise AssetValidationError("metadata.storage.shards must be a non-empty array")
    cursor = coverage_start
    seen_paths: set[str] = set()
    for expected_ordinal, raw_shard in enumerate(shards):
        field = f"metadata.storage.shards[{expected_ordinal}]"
        shard = _require_exact_fields(
            raw_shard,
            field=field,
            required=frozenset(
                {
                    "ordinal",
                    "path",
                    "size_bytes",
                    "sha256",
                    "byte_start",
                    "byte_stop",
                }
            ),
        )
        ordinal = _require_integer(shard["ordinal"], field=f"{field}.ordinal", minimum=0)
        if ordinal != expected_ordinal:
            raise AssetValidationError(
                "metadata.storage.shards ordinals must be exactly 0..N-1 in order"
            )
        descriptor = _require_file_binding(
            path=shard["path"],
            sha256=shard["sha256"],
            file_index=file_index,
            field=field,
        )
        if descriptor.path == idx_file.path or descriptor.path in seen_paths:
            raise AssetValidationError("metadata.storage.shards paths must be unique bins")
        seen_paths.add(descriptor.path)
        size = _require_integer(shard["size_bytes"], field=f"{field}.size_bytes", minimum=1)
        if size != descriptor.size_bytes:
            raise AssetValidationError(
                f"{field}.size_bytes must match manifest.files for {descriptor.path!r}"
            )
        byte_start = _require_integer(
            shard["byte_start"], field=f"{field}.byte_start", minimum=0
        )
        byte_stop = _require_integer(
            shard["byte_stop"], field=f"{field}.byte_stop", minimum=0
        )
        if byte_start != cursor or byte_stop - byte_start != size:
            raise AssetValidationError(
                "metadata.storage.shards byte intervals must be contiguous and match size_bytes"
            )
        cursor = byte_stop
    if cursor != coverage_stop:
        raise AssetValidationError(
            "metadata.storage.shards must exactly cover global_byte_coverage"
        )
    required_cursor_stop = _require_integer(
        storage["required_cursor_stop"],
        field="metadata.storage.required_cursor_stop",
        minimum=1,
    )
    if required_cursor_stop > sequence_count:
        raise AssetValidationError(
            "metadata.storage.required_cursor_stop may not exceed idx.sequence_count"
        )
    _validate_g3_causal_lm_mapping(
        storage["causal_lm_mapping"],
        field="metadata.storage.causal_lm_mapping",
    )
    reference_reader = _require_exact_fields(
        storage["reference_reader"],
        field="metadata.storage.reference_reader",
        required=frozenset({"repository", "revision"}),
    )
    if reference_reader != {
        "repository": "EleutherAI/pythia",
        "revision": "a19eecb807ec2c79a39ebf18108816e6ffffc1d5",
    }:
        raise AssetValidationError(
            "metadata.storage.reference_reader must bind the frozen official reader"
        )
    reference_batch_size = _require_integer(
        storage["reference_batch_size"],
        field="metadata.storage.reference_batch_size",
        minimum=1,
    )
    if reference_batch_size != 1024:
        raise AssetValidationError(
            "metadata.storage.reference_batch_size must be 1024"
        )
    _validate_reference_batch_hashes(
        storage["reference_batch_sha256"],
    )
    _require_hash(
        storage["last_required_record_sha256"],
        field="metadata.storage.last_required_record_sha256",
    )
    required_prefix_bytes = required_cursor_stop * tokens_per_record * itemsize
    if coverage_stop - coverage_start < required_prefix_bytes:
        raise AssetValidationError(
            "metadata.storage.global_byte_coverage is too short for "
            "required_cursor_stop"
        )

def _validate_g3_derived_pile_storage(
    value: Any,
    *,
    parent_asset_ids: list[Any],
    tokenizer_asset_id: Any,
) -> None:
    storage = _require_exact_fields(
        value,
        field="metadata.storage",
        required=frozenset(
            {
                "kind",
                "format",
                "causal_lm_mapping",
                "source_row_mapping",
                "converter_version",
                "converter_git_commit",
            }
        ),
    )
    if storage["kind"] != "pile_derived_pretokenized":
        raise AssetValidationError(
            "derived Pile requires metadata.storage.kind=pile_derived_pretokenized"
        )
    if storage["format"] != "huggingface_load_from_disk":
        raise AssetValidationError(
            "derived Pile metadata.storage.format must be huggingface_load_from_disk"
        )
    _require_text(
        storage["converter_version"],
        field="metadata.storage.converter_version",
        maximum=256,
    )
    _require_git_commit(
        storage["converter_git_commit"],
        field="metadata.storage.converter_git_commit",
    )
    _validate_g3_causal_lm_mapping(
        storage["causal_lm_mapping"],
        field="metadata.storage.causal_lm_mapping",
    )
    source_mapping = _require_exact_fields(
        storage["source_row_mapping"],
        field="metadata.storage.source_row_mapping",
        required=frozenset(
            {
                "parent_asset_id",
                "source_split",
                "source_cursor_start",
                "derived_row_start",
                "row_count",
            }
        ),
    )
    parent_id = _require_hash(
        source_mapping["parent_asset_id"],
        field="metadata.storage.source_row_mapping.parent_asset_id",
    )
    if parent_id not in parent_asset_ids:
        raise AssetValidationError(
            "metadata.storage.source_row_mapping.parent_asset_id must appear in "
            "metadata.preprocessing.parent_asset_ids"
        )
    if tokenizer_asset_id is None:
        raise AssetValidationError(
            "derived Pile requires metadata.preprocessing.tokenizer_asset_id"
        )
    _require_text(
        source_mapping["source_split"],
        field="metadata.storage.source_row_mapping.source_split",
        maximum=128,
    )
    _require_integer(
        source_mapping["source_cursor_start"],
        field="metadata.storage.source_row_mapping.source_cursor_start",
        minimum=0,
    )
    _require_integer(
        source_mapping["derived_row_start"],
        field="metadata.storage.source_row_mapping.derived_row_start",
        minimum=0,
    )
    _require_integer(
        source_mapping["row_count"],
        field="metadata.storage.source_row_mapping.row_count",
        minimum=1,
    )


def _validate_g3_hf_storage(value: Any) -> None:
    storage = _require_exact_fields(
        value,
        field="metadata.storage",
        required=frozenset({"kind", "format_version"}),
    )
    if storage["kind"] != "hf_load_from_disk":
        raise AssetValidationError(
            "hf_dataset requires metadata.storage.kind=hf_load_from_disk"
        )
    _require_text(
        storage["format_version"],
        field="metadata.storage.format_version",
        maximum=256,
    )


def _validate_g3_raw_parquet_storage(
    value: Any,
    *,
    file_index: Mapping[str, AssetFile],
    split_names: frozenset[str],
) -> None:
    storage = _require_exact_fields(
        value,
        field="metadata.storage",
        required=frozenset({"kind", "splits"}),
    )
    if storage["kind"] != "hf_raw_parquet_files":
        raise AssetValidationError(
            "hf_raw_parquet requires metadata.storage.kind=hf_raw_parquet_files"
        )
    splits = storage["splits"]
    if not isinstance(splits, Mapping) or set(splits) != split_names:
        raise AssetValidationError(
            "metadata.storage.splits must exactly match metadata.splits"
        )
    referenced: set[str] = set()
    for split_name in sorted(split_names):
        entries = splits[split_name]
        if not isinstance(entries, list) or not entries:
            raise AssetValidationError(
                f"metadata.storage.splits.{split_name} must be a non-empty array"
            )
        for index, raw_entry in enumerate(entries):
            field = f"metadata.storage.splits.{split_name}[{index}]"
            entry = _require_exact_fields(
                raw_entry,
                field=field,
                required=frozenset({"path", "sha256"}),
            )
            descriptor = _require_file_binding(
                path=entry["path"],
                sha256=entry["sha256"],
                file_index=file_index,
                field=field,
            )
            if not descriptor.path.endswith(".parquet"):
                raise AssetValidationError(f"{field}.path must name a parquet file")
            if descriptor.role != split_name:
                raise AssetValidationError(
                    f"{field}.path manifest role must equal split name {split_name!r}"
                )
            if descriptor.path in referenced:
                raise AssetValidationError(
                    "metadata.storage raw parquet paths must be unique across splits"
                )
            referenced.add(descriptor.path)
    if referenced != set(file_index):
        raise AssetValidationError(
            "hf_raw_parquet manifest.files must be exactly the split parquet files"
        )


def _validate_g3_task_contract(
    value: Any,
    *,
    splits: Mapping[str, Any],
    raw_parquet: bool,
) -> None:
    contract = _require_exact_fields(
        value,
        field="metadata.task_contract",
        required=frozenset(
            {"task", "text_fields", "label_mapping", "unlabeled_test_policy"}
        ),
    )
    task = _require_text(
        contract["task"], field="metadata.task_contract.task", maximum=32
    )
    if task not in {"sst2", "mnli", "rte"}:
        raise AssetValidationError(
            "metadata.task_contract.task must be sst2, mnli, or rte"
        )
    text_fields = contract["text_fields"]
    if not isinstance(text_fields, list) or not text_fields:
        raise AssetValidationError(
            "metadata.task_contract.text_fields must be a non-empty array"
        )
    normalized_text_fields = [
        _require_text(
            field_name,
            field="metadata.task_contract.text_fields",
            maximum=128,
        )
        for field_name in text_fields
    ]
    if len(normalized_text_fields) != len(set(normalized_text_fields)):
        raise AssetValidationError(
            "metadata.task_contract.text_fields must be unique"
        )
    label_mapping = contract["label_mapping"]
    if not isinstance(label_mapping, Mapping) or not label_mapping:
        raise AssetValidationError(
            "metadata.task_contract.label_mapping must be a non-empty object"
        )
    label_ids: list[int] = []
    for raw_label, raw_label_id in label_mapping.items():
        label = _require_text(
            raw_label,
            field="metadata.task_contract.label_mapping key",
            maximum=128,
        )
        label_ids.append(
            _require_integer(
                raw_label_id,
                field=f"metadata.task_contract.label_mapping.{label}",
                minimum=0,
            )
        )
    if len(label_ids) != len(set(label_ids)) or set(label_ids) != set(
        range(len(label_ids))
    ):
        raise AssetValidationError(
            "metadata.task_contract.label_mapping IDs must be unique and contiguous "
            "from zero"
        )
    if contract["unlabeled_test_policy"] != (
        "retain_in_raw_exclude_from_derived_training_and_evaluation"
    ):
        raise AssetValidationError(
            "metadata.task_contract.unlabeled_test_policy must be "
            "retain_in_raw_exclude_from_derived_training_and_evaluation"
        )
    if raw_parquet:
        expected_fields = set(normalized_text_fields)
        for split_name, split in splits.items():
            if not expected_fields <= set(split["fields"]):
                raise AssetValidationError(
                    f"metadata.splits.{split_name}.fields must include every "
                    "task_contract text field"
                )
    else:
        if any(split_name.casefold().startswith("test") for split_name in splits):
            raise AssetValidationError(
                "derived GLUE splits may not include unlabeled test splits"
            )
        required_derived_fields = {"input_ids", "attention_mask", "labels"}
        for split_name, split in splits.items():
            if not required_derived_fields <= set(split["fields"]):
                raise AssetValidationError(
                    f"metadata.splits.{split_name}.fields must include input_ids, "
                    "attention_mask, and labels for derived GLUE"
                )


def _validate_g3_dataset_metadata(
    metadata: Mapping[str, Any],
    file_index: Mapping[str, AssetFile],
) -> None:
    value = _require_exact_fields(
        metadata,
        field="metadata",
        required=frozenset(
            {
                "contract_version",
                "dataset_kind",
                "raw_revision",
                "splits",
                "preprocessing_version",
                "preprocessing",
                "storage",
            }
        ),
        optional=frozenset({"task_contract"}),
    )
    if value["contract_version"] != G3_DATASET_METADATA_VERSION:
        raise AssetValidationError(
            f"metadata.contract_version must be {G3_DATASET_METADATA_VERSION!r}"
        )
    dataset_kind = value["dataset_kind"]
    if dataset_kind not in {
        "raw_indexed_mmap",
        "derived_pretokenized",
        "hf_dataset",
        "hf_raw_parquet",
    }:
        raise AssetValidationError(
            "metadata.dataset_kind must be raw_indexed_mmap, "
            "derived_pretokenized, hf_dataset, or hf_raw_parquet"
        )
    raw_revision = _require_text(
        value["raw_revision"], field="metadata.raw_revision", maximum=512
    )
    if raw_revision.casefold() in _GENERIC_REVISIONS or raw_revision != raw_revision.strip():
        raise AssetValidationError("metadata.raw_revision must be immutable and specific")
    _validate_g3_dataset_splits(value["splits"])
    preprocessing_version = _require_text(
        value["preprocessing_version"],
        field="metadata.preprocessing_version",
        maximum=256,
    )
    preprocessing = _validate_g3_preprocessing(
        value["preprocessing"], preprocessing_version=preprocessing_version
    )
    parents = preprocessing["parent_asset_ids"]
    tokenizer_id = preprocessing["tokenizer_asset_id"]
    storage_kind = (
        value["storage"].get("kind")
        if isinstance(value["storage"], Mapping)
        else None
    )
    is_raw_glue = dataset_kind == "hf_raw_parquet"
    is_derived_glue = (
        dataset_kind == "derived_pretokenized"
        and storage_kind == "hf_load_from_disk"
    )
    if is_raw_glue or is_derived_glue:
        if "task_contract" not in value:
            raise AssetValidationError(
                "GLUE dataset metadata requires metadata.task_contract"
            )
        _validate_g3_task_contract(
            value["task_contract"],
            splits=value["splits"],
            raw_parquet=is_raw_glue,
        )
    elif "task_contract" in value:
        raise AssetValidationError(
            "metadata.task_contract is only valid for raw or derived GLUE datasets"
        )
    if dataset_kind == "raw_indexed_mmap":
        _validate_g3_raw_pile_storage(value["storage"], file_index)
        required_cursor_stop = value["storage"]["required_cursor_stop"]
        split_stops = [
            split["cursor"]["stop"] for split in value["splits"].values()
        ]
        if max(split_stops) > required_cursor_stop:
            raise AssetValidationError(
                "metadata.storage.required_cursor_stop must cover every dataset split"
            )
    elif dataset_kind == "hf_raw_parquet":
        _validate_g3_raw_parquet_storage(
            value["storage"],
            file_index=file_index,
            split_names=frozenset(value["splits"]),
        )
    elif dataset_kind == "derived_pretokenized":
        if not parents:
            raise AssetValidationError(
                "derived_pretokenized requires preprocessing.parent_asset_ids"
            )
        if storage_kind == "pile_derived_pretokenized":
            _validate_g3_derived_pile_storage(
                value["storage"],
                parent_asset_ids=parents,
                tokenizer_asset_id=tokenizer_id,
            )
        else:
            _validate_g3_hf_storage(value["storage"])
            if tokenizer_id is None:
                raise AssetValidationError(
                    "derived_pretokenized requires preprocessing.tokenizer_asset_id"
                )
    else:
        _validate_g3_hf_storage(value["storage"])


def _validate_g3_source_metadata(metadata: Mapping[str, Any]) -> None:
    value = _require_exact_fields(
        metadata,
        field="metadata",
        required=frozenset(
            {"contract_version", "source_kind", "license", "upstream_locator"}
        ),
    )
    if value["contract_version"] != G3_SOURCE_METADATA_VERSION:
        raise AssetValidationError(
            f"metadata.contract_version must be {G3_SOURCE_METADATA_VERSION!r}"
        )
    _require_text(value["source_kind"], field="metadata.source_kind", maximum=128)
    _require_text(value["license"], field="metadata.license", maximum=256)
    locator = _require_text(
        value["upstream_locator"], field="metadata.upstream_locator", maximum=1024
    )
    if "?" in locator:
        raise AssetValidationError(
            "metadata.upstream_locator must be stable, not a query or signed URL"
        )


def _validate_g3_metadata(
    asset_type: AssetType,
    value: Any,
    files: Iterable[AssetFile | Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = _validate_metadata(asset_type, value)
    file_index = _g3_file_index(files)
    if asset_type is AssetType.MODEL:
        _validate_g3_model_metadata(metadata, file_index)
    elif asset_type is AssetType.TOKENIZER:
        _validate_g3_tokenizer_metadata(metadata, file_index)
    elif asset_type is AssetType.DATASET:
        _validate_g3_dataset_metadata(metadata, file_index)
    else:
        _validate_g3_source_metadata(metadata)
    return metadata


def _validate_evidence_ref(
    value: Any,
    *,
    field: str,
    required: bool,
) -> str | None:
    if value is None:
        if required:
            raise AssetValidationError(f"{field} requires a non-empty evidence_ref")
        return None
    text = _require_text(value, field=field, maximum=4096)
    if "?" in text:
        raise AssetValidationError(
            f"{field} must be a stable evidence reference, not a query or signed URL"
        )
    return text


def validate_asset_path(value: str) -> str:
    """Return a normalized safe manifest-relative POSIX path.

    Asset paths are deliberately platform-neutral.  Absolute paths, path
    traversal, Windows separators/drive syntax, and temporary/lock suffixes
    are rejected before any filesystem access occurs.
    """

    text = _require_text(value, field="file.path", maximum=4096)
    if "\\" in text or ":" in text:
        raise AssetValidationError(f"Unsafe asset path syntax: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) != text or not path.parts:
        raise AssetValidationError(f"Asset path must be normalized and relative: {text!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AssetValidationError(f"Asset path contains traversal: {text!r}")
    for part in path.parts:
        tokens = {token for token in re.split(r"[._-]+", part.casefold()) if token}
        if tokens & _TEMPORARY_PATH_TOKENS:
            raise AssetValidationError(
                f"Asset path references a temporary or lock object: {text!r}"
            )
    return text


@dataclass(frozen=True, slots=True)
class AssetFile:
    path: str
    size_bytes: int
    sha256: str
    role: str | None = None

    @classmethod
    def from_value(cls, value: "AssetFile | Mapping[str, Any]") -> "AssetFile":
        if isinstance(value, cls):
            candidate = value
        elif isinstance(value, Mapping):
            extra = set(value) - _FILE_FIELDS
            missing = {"path", "size_bytes", "sha256"} - set(value)
            if missing or extra:
                raise AssetValidationError(
                    f"Invalid file descriptor fields; missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            size = value["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise AssetValidationError("file.size_bytes must be a non-negative integer")
            digest = value["sha256"]
            if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
                raise AssetValidationError("file.sha256 must be 64 lowercase hex characters")
            role_value = value.get("role")
            role = (
                None
                if role_value is None
                else _require_text(role_value, field="file.role", maximum=128)
            )
            candidate = cls(
                path=validate_asset_path(value["path"]),
                size_bytes=size,
                sha256=digest,
                role=role,
            )
        else:
            raise AssetValidationError("Each files entry must be an object")

        # Revalidate dataclass instances so constructing AssetFile directly
        # cannot bypass the same contract.
        validate_asset_path(candidate.path)
        if (
            isinstance(candidate.size_bytes, bool)
            or not isinstance(candidate.size_bytes, int)
            or candidate.size_bytes < 0
        ):
            raise AssetValidationError("file.size_bytes must be a non-negative integer")
        if not SHA256_PATTERN.fullmatch(candidate.sha256):
            raise AssetValidationError("file.sha256 must be 64 lowercase hex characters")
        if candidate.role is not None:
            _require_text(candidate.role, field="file.role", maximum=128)
        return candidate

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if self.role is not None:
            value["role"] = self.role
        return value


def _normalize_files(
    files: Iterable[AssetFile | Mapping[str, Any]],
) -> list[AssetFile]:
    normalized = [AssetFile.from_value(value) for value in files]
    if not normalized:
        raise AssetValidationError("files must contain at least one final object")
    normalized.sort(key=lambda item: item.path)
    lowered = [item.path.casefold() for item in normalized]
    if len(lowered) != len(set(lowered)):
        raise AssetValidationError("files contains duplicate or case-colliding paths")
    return normalized


def compute_asset_id(
    manifest: Mapping[str, Any] | None = None,
    *,
    asset_type: AssetType | str | None = None,
    name: str | None = None,
    source: str | None = None,
    revision: str | None = None,
    files: Iterable[AssetFile | Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Compute the stable logical identity of an immutable asset.

    State, timestamps, local roots, and audit actors are intentionally absent
    from the identity.  File order is normalized, so equivalent inventories
    produce the same ID on every platform.
    """

    if manifest is not None:
        if any(
            value is not None
            for value in (asset_type, name, source, revision, files, metadata)
        ):
            raise TypeError("Pass either manifest or explicit identity fields, not both")
        asset_type = manifest.get("asset_type")
        name = manifest.get("name")
        source = manifest.get("source")
        revision = manifest.get("revision")
        files = manifest.get("files")
        metadata = manifest.get("metadata")
    if files is None:
        raise AssetValidationError("files is required to compute asset_id")
    if metadata is None:
        raise AssetValidationError("metadata is required to compute asset_id")
    try:
        normalized_type = AssetType(asset_type)
    except (TypeError, ValueError) as error:
        raise AssetValidationError(f"Unknown asset_type: {asset_type!r}") from error
    normalized_name = _require_text(name, field="name", maximum=256)
    normalized_source = _require_text(source, field="source", maximum=512)
    normalized_revision = _validate_revision(revision)
    if "?" in normalized_source:
        raise AssetValidationError(
            "source must be a stable identifier, not a query or signed URL"
        )
    identity = {
        "asset_type": normalized_type.value,
        "name": normalized_name,
        "source": normalized_source,
        "revision": normalized_revision,
        "files": [item.as_dict() for item in _normalize_files(files)],
        "metadata": _validate_metadata(normalized_type, metadata),
    }
    return stable_json_hash(identity)


def _uses_g3_profile(manifest: Mapping[str, Any]) -> bool:
    present = _G3_TOP_LEVEL_FIELDS & set(manifest)
    if not present:
        return False
    missing = _G3_TOP_LEVEL_FIELDS - set(manifest)
    if missing:
        raise AssetValidationError(
            "G3 manifest identity fields are all-or-none; "
            f"missing={sorted(missing)}"
        )
    if manifest["contract_profile"] != G3_CONTRACT_PROFILE:
        raise AssetValidationError(
            f"contract_profile must be {G3_CONTRACT_PROFILE!r}"
        )
    _require_hash(manifest["candidate_id"], field="candidate_id")
    _require_git_commit(
        manifest["generator_git_commit"], field="generator_git_commit"
    )
    return True


def compute_candidate_id(manifest: Mapping[str, Any]) -> str:
    """Compute the immutable identity of one G3 acquisition candidate.

    Unlike ``asset_id``, this identity binds the generator and initial audit
    event.  State and later history entries remain excluded so the same
    candidate keeps its ID while advancing through the lifecycle.
    """

    if not isinstance(manifest, Mapping):
        raise AssetValidationError("Manifest root must be an object")
    if not _uses_g3_profile(manifest):
        raise AssetValidationError(
            f"candidate_id requires contract_profile={G3_CONTRACT_PROFILE}"
        )
    asset_id = _require_hash(manifest.get("asset_id"), field="asset_id")
    created_at = _require_timestamp(manifest.get("created_at"), field="created_at")
    generator_version = _require_text(
        manifest.get("generator_version"), field="generator_version", maximum=256
    )
    generator_git_commit = _require_git_commit(
        manifest.get("generator_git_commit"), field="generator_git_commit"
    )
    history = manifest.get("state_history")
    if not isinstance(history, list) or not history or not isinstance(history[0], Mapping):
        raise AssetValidationError(
            "candidate_id requires a non-empty state_history creation event"
        )
    creation = history[0]
    required_creation_fields = frozenset(
        {
            "from",
            "to",
            "at",
            "actor",
            "actor_role",
            "evidence_ref",
            "summary",
            "actor_instance_id",
            "evidence_sha256",
        }
    )
    if set(creation) != required_creation_fields:
        raise AssetValidationError(
            "candidate_id requires one complete G3 creation event"
        )
    if (
        creation["from"] is not None
        or creation["to"] != AssetState.DOWNLOADING.value
        or creation["actor_role"] != AssetActorRole.FETCHER.value
    ):
        raise AssetValidationError(
            "candidate_id creation event must be fetcher null -> downloading"
        )
    creation_payload = {
        "at": _require_timestamp(creation["at"], field="state_history[0].at"),
        "actor": _require_text(
            creation["actor"], field="state_history[0].actor", maximum=256
        ),
        "actor_instance_id": _require_text(
            creation["actor_instance_id"],
            field="state_history[0].actor_instance_id",
            maximum=512,
        ),
        "evidence_ref": _validate_evidence_ref(
            creation["evidence_ref"], field="state_history[0]", required=True
        ),
        "evidence_sha256": _require_hash(
            creation["evidence_sha256"],
            field="state_history[0].evidence_sha256",
        ),
    }
    identity = {
        "schema_version": "stage0-asset-candidate-id-v1",
        "contract_profile": G3_CONTRACT_PROFILE,
        "asset_id": asset_id,
        "created_at": created_at,
        "generator_version": generator_version,
        "generator_git_commit": generator_git_commit,
        "creation_event": creation_payload,
    }
    return stable_json_hash(identity)


def validate_state_transition(
    previous: AssetState | str,
    target: AssetState | str,
    *,
    actor_role: AssetActorRole | str,
) -> None:
    try:
        current_state = AssetState(previous)
        target_state = AssetState(target)
        normalized_role = AssetActorRole(actor_role)
    except (TypeError, ValueError) as error:
        raise AssetValidationError("Unknown asset state or actor_role") from error
    if target_state not in ASSET_TRANSITIONS[current_state]:
        raise AssetValidationError(
            f"Forbidden asset state transition: {current_state.value} -> "
            f"{target_state.value}"
        )
    if (current_state, target_state) not in ROLE_TRANSITIONS[normalized_role]:
        raise AssetValidationError(
            f"actor_role={normalized_role.value} is not authorized for "
            f"{current_state.value} -> {target_state.value}"
        )


def _validate_state_history(
    value: Any,
    final_state: AssetState,
    *,
    created_at: str,
    g3_profile: bool,
) -> None:
    if not isinstance(value, list) or not value:
        raise AssetValidationError("state_history must be a non-empty array")
    previous: AssetState | None = None
    previous_at = _timestamp_value(created_at, field="created_at")
    allowed_fields = _G3_HISTORY_FIELDS if g3_profile else _LEGACY_HISTORY_FIELDS
    role_instances: dict[AssetActorRole, str] = {}
    for index, event in enumerate(value):
        if not isinstance(event, Mapping):
            raise AssetValidationError("state_history entries must be objects")
        missing = allowed_fields - set(event)
        extra = set(event) - allowed_fields
        if missing or extra:
            raise AssetValidationError(
                f"Invalid state_history[{index}] fields; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        raw_from = event["from"]
        try:
            event_from = None if raw_from is None else AssetState(raw_from)
            event_to = AssetState(event["to"])
            actor_role = AssetActorRole(event["actor_role"])
        except (TypeError, ValueError) as error:
            raise AssetValidationError(
                f"state_history[{index}] contains an unknown state or actor_role"
            ) from error
        if index == 0:
            if (
                event_from is not None
                or event_to is not AssetState.DOWNLOADING
                or actor_role is not AssetActorRole.FETCHER
            ):
                raise AssetValidationError(
                    "state_history must begin with a fetcher null -> downloading event"
                )
        else:
            if event_from is not previous:
                raise AssetValidationError(
                    f"state_history[{index}] does not continue the previous state"
                )
            validate_state_transition(
                event_from,
                event_to,
                actor_role=actor_role,
            )
        event_at = _timestamp_value(event["at"], field=f"state_history[{index}].at")
        _require_text(event["actor"], field=f"state_history[{index}].actor", maximum=256)
        _validate_evidence_ref(
            event["evidence_ref"],
            field=f"state_history[{index}]",
            required=(
                g3_profile
                or event_to
                in {AssetState.VERIFIED, AssetState.READY, AssetState.INVALID}
            ),
        )
        if g3_profile:
            actor_instance_id = _require_text(
                event["actor_instance_id"],
                field=f"state_history[{index}].actor_instance_id",
                maximum=512,
            )
            previous_instance = role_instances.get(actor_role)
            if previous_instance is not None and previous_instance != actor_instance_id:
                raise AssetValidationError(
                    f"G3 actor_role={actor_role.value} must use one actor_instance_id"
                )
            for other_role, other_instance in role_instances.items():
                if other_role is not actor_role and other_instance == actor_instance_id:
                    raise AssetValidationError(
                        "G3 fetcher, verifier, and gate actor_instance_id values "
                        "must be pairwise distinct"
                    )
            role_instances[actor_role] = actor_instance_id
            _require_hash(
                event["evidence_sha256"],
                field=f"state_history[{index}].evidence_sha256",
            )
            _require_text(
                event["summary"],
                field=f"state_history[{index}].summary",
                maximum=2048,
            )
            if event_at < previous_at:
                raise AssetValidationError(
                    "G3 state_history timestamps must be >= created_at and monotonic"
                )
        elif not isinstance(event["summary"], str):
            raise AssetValidationError(f"state_history[{index}].summary must be a string")
        previous = event_to
        previous_at = event_at
    if previous is not final_state:
        raise AssetValidationError(
            "state does not match the final state_history transition"
        )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise AssetValidationError("Manifest root must be an object")
    missing = _REQUIRED_TOP_LEVEL_FIELDS - set(manifest)
    extra = set(manifest) - _TOP_LEVEL_FIELDS
    if missing or extra:
        raise AssetValidationError(
            f"Invalid manifest fields; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise AssetValidationError(
            f"Unsupported schema_version: {manifest['schema_version']!r}"
        )
    try:
        state = AssetState(manifest["state"])
        asset_type = AssetType(manifest["asset_type"])
    except (TypeError, ValueError) as error:
        raise AssetValidationError("Unknown asset_type or state") from error
    _require_text(manifest["name"], field="name", maximum=256)
    source = _require_text(manifest["source"], field="source", maximum=512)
    if "?" in source:
        raise AssetValidationError(
            "source must be a stable identifier, not a query or signed URL"
        )
    _validate_revision(manifest["revision"])
    created_at = _require_timestamp(manifest["created_at"], field="created_at")
    _require_text(
        manifest["generator_version"], field="generator_version", maximum=256
    )
    if not isinstance(manifest["files"], list):
        raise AssetValidationError("files must be an array")
    normalized_files = _normalize_files(manifest["files"])
    g3_profile = _uses_g3_profile(manifest)
    _validate_state_history(
        manifest["state_history"],
        state,
        created_at=created_at,
        g3_profile=g3_profile,
    )
    if g3_profile:
        _validate_g3_metadata(asset_type, manifest["metadata"], normalized_files)
    else:
        _validate_metadata(asset_type, manifest["metadata"])
    asset_id = manifest["asset_id"]
    if not isinstance(asset_id, str) or not SHA256_PATTERN.fullmatch(asset_id):
        raise AssetValidationError("asset_id must be 64 lowercase hex characters")
    expected = compute_asset_id(manifest)
    if asset_id != expected:
        raise AssetValidationError(
            f"asset_id mismatch: declared={asset_id}, computed={expected}"
        )
    if g3_profile:
        candidate_id = manifest["candidate_id"]
        expected_candidate_id = compute_candidate_id(manifest)
        if candidate_id != expected_candidate_id:
            raise AssetValidationError(
                "candidate_id mismatch: "
                f"declared={candidate_id}, computed={expected_candidate_id}"
            )


def validate_g3_manifest(manifest: Mapping[str, Any]) -> None:
    """Require a manifest to satisfy the additive Stage 0 G3 profile."""

    validate_manifest(manifest)
    if not _uses_g3_profile(manifest):
        raise AssetValidationError(
            f"G3 eligibility requires contract_profile={G3_CONTRACT_PROFILE}"
        )


def inspect_manifest_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a read-only structural eligibility inspection.

    ``g3_eligible`` means only that the manifest satisfies the G3 manifest
    profile.  It does not claim asset qualification or a G3 Gate decision.
    """

    profile = manifest.get("contract_profile") if isinstance(manifest, Mapping) else None
    try:
        validate_manifest(manifest)
    except AssetManifestError as error:
        return {
            "schema_version": "stage0-asset-contract-inspection-v1",
            "manifest_valid": False,
            "contract_profile": profile,
            "g3_eligible": False,
            "violations": [str(error)],
        }
    try:
        validate_g3_manifest(manifest)
    except AssetManifestError as error:
        return {
            "schema_version": "stage0-asset-contract-inspection-v1",
            "manifest_valid": True,
            "contract_profile": profile,
            "g3_eligible": False,
            "violations": [str(error)],
        }
    return {
        "schema_version": "stage0-asset-contract-inspection-v1",
        "manifest_valid": True,
        "contract_profile": G3_CONTRACT_PROFILE,
        "g3_eligible": True,
        "violations": [],
    }


def _normalize_qualification_checks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise AssetValidationError("qualification.checks must be a non-empty array")
    checks: list[dict[str, Any]] = []
    check_ids: set[str] = set()
    for index, raw_check in enumerate(value):
        field = f"qualification.checks[{index}]"
        check = _require_exact_fields(
            raw_check,
            field=field,
            required=_QUALIFICATION_CHECK_FIELDS,
        )
        check_id = _require_text(
            check["check_id"], field=f"{field}.check_id", maximum=256
        )
        if check_id in check_ids:
            raise AssetValidationError("qualification.checks check_id values must be unique")
        check_ids.add(check_id)
        if check["status"] != "PASS":
            raise AssetValidationError(
                f"{field}.status must be PASS for qualification admission"
            )
        _validate_evidence_ref(
            check["evidence_ref"], field=field, required=True
        )
        _require_hash(check["evidence_sha256"], field=f"{field}.evidence_sha256")
        _require_text(check["summary"], field=f"{field}.summary", maximum=2048)
        checks.append(check)
    return checks


def validate_g3_qualification(qualification: Mapping[str, Any]) -> None:
    """Validate a formal, hash-bound Stage 0 asset qualification artifact."""

    value = _require_exact_fields(
        qualification,
        field="qualification",
        required=_QUALIFICATION_FIELDS,
    )
    if value["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
        raise AssetValidationError(
            f"qualification.schema_version must be {QUALIFICATION_SCHEMA_VERSION!r}"
        )
    if value["formal"] is not True:
        raise AssetValidationError("qualification.formal must be true")
    _require_hash(value["asset_id"], field="qualification.asset_id")
    _require_hash(value["candidate_id"], field="qualification.candidate_id")
    _require_hash(
        value["verified_manifest_sha256"],
        field="qualification.verified_manifest_sha256",
    )
    _validate_evidence_ref(
        value["requirements_ref"], field="qualification.requirements", required=True
    )
    _require_hash(
        value["requirements_sha256"], field="qualification.requirements_sha256"
    )
    _validate_evidence_ref(
        value["acquisition_ref"], field="qualification.acquisition", required=True
    )
    _require_hash(
        value["acquisition_sha256"], field="qualification.acquisition_sha256"
    )
    _validate_evidence_ref(
        value["verification_ref"], field="qualification.verification", required=True
    )
    _require_hash(
        value["verification_sha256"], field="qualification.verification_sha256"
    )
    _normalize_qualification_checks(value["checks"])
    network_attempts = value["network_attempts"]
    if (
        isinstance(network_attempts, bool)
        or not isinstance(network_attempts, int)
        or network_attempts != 0
    ):
        raise AssetValidationError("qualification.network_attempts must be exactly 0")
    _require_timestamp(value["checked_at"], field="qualification.checked_at")
    _require_git_commit(
        value["generator_git_commit"],
        field="qualification.generator_git_commit",
    )
    artifact_hash = _require_hash(
        value["artifact_hash"], field="qualification.artifact_hash"
    )
    payload = dict(value)
    del payload["artifact_hash"]
    try:
        expected = stable_json_hash(payload)
    except (TypeError, ValueError) as error:
        raise AssetValidationError(
            "qualification is not canonical JSON data"
        ) from error
    if artifact_hash != expected:
        raise AssetValidationError(
            "qualification.artifact_hash does not match the canonical payload"
        )


def build_g3_qualification(
    verified_manifest: Mapping[str, Any],
    *,
    requirements_ref: str,
    requirements_sha256: str,
    acquisition_ref: str | None = None,
    acquisition_sha256: str | None = None,
    verification_ref: str,
    verification_sha256: str,
    checks: Iterable[Mapping[str, Any]],
    checked_at: str,
    generator_git_commit: str,
    network_attempts: int = 0,
) -> dict[str, Any]:
    """Build a formal PASS qualification for one exact VERIFIED candidate.

    ``requirements_sha256`` is the frozen requirements artifact's embedded
    ``artifact_hash``.  The referenced artifact is replayed by the formal Gate,
    outside this local manifest-layer builder.
    """

    validate_g3_manifest(verified_manifest)
    if AssetState(verified_manifest["state"]) is not AssetState.VERIFIED:
        raise AssetValidationError(
            "G3 qualification requires a manifest in state=verified"
        )
    normalized_requirements_ref = _validate_evidence_ref(
        requirements_ref, field="qualification requirements", required=True
    )
    normalized_requirements_sha = _require_hash(
        requirements_sha256, field="qualification.requirements_sha256"
    )
    downloaded_events = verified_manifest["state_history"][:2]
    if (
        len(downloaded_events) != 2
        or downloaded_events[0]["to"] != AssetState.DOWNLOADING.value
        or downloaded_events[1]["to"] != AssetState.DOWNLOADED.value
        or downloaded_events[0]["evidence_ref"] != downloaded_events[1]["evidence_ref"]
        or downloaded_events[0]["evidence_sha256"]
        != downloaded_events[1]["evidence_sha256"]
    ):
        raise AssetValidationError(
            "qualification requires one acquisition report bound to both fetcher events"
        )
    normalized_acquisition_ref = _validate_evidence_ref(
        acquisition_ref
        if acquisition_ref is not None
        else downloaded_events[0]["evidence_ref"],
        field="qualification acquisition",
        required=True,
    )
    normalized_acquisition_sha = _require_hash(
        acquisition_sha256
        if acquisition_sha256 is not None
        else downloaded_events[0]["evidence_sha256"],
        field="qualification.acquisition_sha256",
    )
    if (
        downloaded_events[0]["evidence_ref"] != normalized_acquisition_ref
        or downloaded_events[0]["evidence_sha256"] != normalized_acquisition_sha
    ):
        raise AssetValidationError(
            "qualification acquisition ref/hash must match both fetcher history events"
        )
    normalized_verification_ref = _validate_evidence_ref(
        verification_ref, field="qualification verification", required=True
    )
    normalized_verification_sha = _require_hash(
        verification_sha256, field="qualification.verification_sha256"
    )
    verified_event = verified_manifest["state_history"][-1]
    if (
        verified_event["to"] != AssetState.VERIFIED.value
        or verified_event["evidence_ref"] != normalized_verification_ref
        or verified_event["evidence_sha256"] != normalized_verification_sha
    ):
        raise AssetValidationError(
            "qualification verification ref/hash must match the VERIFIED history event"
        )
    normalized_checked_at = _require_timestamp(
        checked_at, field="qualification.checked_at"
    )
    if _timestamp_value(
        normalized_checked_at, field="qualification.checked_at"
    ) < _timestamp_value(verified_event["at"], field="verified history timestamp"):
        raise AssetValidationError(
            "qualification.checked_at may not precede the VERIFIED event"
        )
    normalized_generator_commit = _require_git_commit(
        generator_git_commit, field="qualification.generator_git_commit"
    )
    normalized_checks = sorted(
        _normalize_qualification_checks(list(checks)),
        key=lambda item: item["check_id"],
    )
    payload: dict[str, Any] = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "formal": True,
        "asset_id": verified_manifest["asset_id"],
        "candidate_id": verified_manifest["candidate_id"],
        "verified_manifest_sha256": stable_json_hash(verified_manifest),
        "requirements_ref": normalized_requirements_ref,
        "requirements_sha256": normalized_requirements_sha,
        "acquisition_ref": normalized_acquisition_ref,
        "acquisition_sha256": normalized_acquisition_sha,
        "verification_ref": normalized_verification_ref,
        "verification_sha256": normalized_verification_sha,
        "checks": normalized_checks,
        "network_attempts": network_attempts,
        "checked_at": normalized_checked_at,
        "generator_git_commit": normalized_generator_commit,
    }
    value = payload | {"artifact_hash": stable_json_hash(payload)}
    validate_g3_qualification(value)
    return value


def _validate_qualification_for_verified_manifest(
    verified_manifest: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    requirements_artifact_hash: str,
) -> None:
    validate_g3_manifest(verified_manifest)
    if AssetState(verified_manifest["state"]) is not AssetState.VERIFIED:
        raise AssetValidationError(
            "qualification binding requires a VERIFIED G3 manifest"
        )
    validate_g3_qualification(qualification)
    expected_requirements_hash = _require_hash(
        requirements_artifact_hash,
        field="requirements_artifact_hash",
    )
    if qualification["requirements_sha256"] != expected_requirements_hash:
        raise AssetValidationError(
            "qualification requirements_sha256 does not match the frozen "
            "requirements artifact_hash"
        )
    if qualification["asset_id"] != verified_manifest["asset_id"]:
        raise AssetValidationError("qualification asset_id does not match the manifest")
    if qualification["candidate_id"] != verified_manifest["candidate_id"]:
        raise AssetValidationError(
            "qualification candidate_id does not match the manifest"
        )
    if qualification["verified_manifest_sha256"] != stable_json_hash(
        verified_manifest
    ):
        raise AssetValidationError(
            "qualification verified_manifest_sha256 does not match the manifest"
        )
    verified_event = verified_manifest["state_history"][-1]
    if (
        verified_event["to"] != AssetState.VERIFIED.value
        or qualification["verification_ref"] != verified_event["evidence_ref"]
        or qualification["verification_sha256"]
        != verified_event["evidence_sha256"]
    ):
        raise AssetValidationError(
            "qualification verification evidence does not match VERIFIED history"
        )
    acquisition_events = verified_manifest["state_history"][:2]
    if (
        len(acquisition_events) != 2
        or acquisition_events[0]["to"] != AssetState.DOWNLOADING.value
        or acquisition_events[1]["to"] != AssetState.DOWNLOADED.value
        or any(
            event["evidence_ref"] != qualification["acquisition_ref"]
            or event["evidence_sha256"] != qualification["acquisition_sha256"]
            for event in acquisition_events
        )
    ):
        raise AssetValidationError(
            "qualification acquisition evidence does not match fetcher history"
        )


def admit_g3_qualification(
    verified_manifest: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    qualification_ref: str,
    requirements_artifact_hash: str,
    actor: str,
    actor_instance_id: str,
    summary: str = "formal G3 qualification admitted",
    at: str | None = None,
) -> dict[str, Any]:
    """Admit one exactly bound formal qualification as VERIFIED -> READY."""

    _validate_qualification_for_verified_manifest(
        verified_manifest,
        qualification,
        requirements_artifact_hash=requirements_artifact_hash,
    )
    normalized_ref = _validate_evidence_ref(
        qualification_ref, field="qualification admission", required=True
    )
    _require_text(actor, field="actor", maximum=256)
    _require_text(actor_instance_id, field="actor_instance_id", maximum=512)
    _require_text(summary, field="summary", maximum=2048)
    timestamp = at or _utc_now()
    _require_timestamp(timestamp, field="at")
    if _timestamp_value(timestamp, field="at") < _timestamp_value(
        qualification["checked_at"], field="qualification.checked_at"
    ):
        raise AssetValidationError(
            "qualification admission time may not precede qualification.checked_at"
        )
    updated = deepcopy(dict(verified_manifest))
    updated["state"] = AssetState.READY.value
    updated["state_history"].append(
        {
            "from": AssetState.VERIFIED.value,
            "to": AssetState.READY.value,
            "at": timestamp,
            "actor": actor,
            "actor_role": AssetActorRole.GATE.value,
            "actor_instance_id": actor_instance_id,
            "evidence_ref": normalized_ref,
            "evidence_sha256": qualification["artifact_hash"],
            "summary": summary,
        }
    )
    validate_g3_manifest(updated)
    return updated


def build_manifest(
    *,
    asset_type: AssetType | str,
    name: str,
    source: str,
    revision: str,
    files: Iterable[AssetFile | Mapping[str, Any]],
    actor: str,
    actor_role: AssetActorRole | str,
    evidence_ref: str | None,
    generator_version: str,
    metadata: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the initial immutable-identity manifest in `downloading` state."""

    timestamp = created_at or _utc_now()
    normalized_files = _normalize_files(files)
    try:
        normalized_type = AssetType(asset_type)
    except (TypeError, ValueError) as error:
        raise AssetValidationError(f"Unknown asset_type: {asset_type!r}") from error
    normalized_metadata = _validate_metadata(normalized_type, metadata)
    try:
        normalized_role = AssetActorRole(actor_role)
    except (TypeError, ValueError) as error:
        raise AssetValidationError(f"Unknown actor_role: {actor_role!r}") from error
    if normalized_role is not AssetActorRole.FETCHER:
        raise AssetValidationError(
            "Only actor_role=fetcher may create a downloading candidate"
        )
    _validate_evidence_ref(
        evidence_ref,
        field="candidate creation",
        required=False,
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": compute_asset_id(
            asset_type=asset_type,
            name=name,
            source=source,
            revision=revision,
            files=normalized_files,
            metadata=normalized_metadata,
        ),
        "asset_type": normalized_type.value,
        "state": AssetState.DOWNLOADING.value,
        "name": name,
        "source": source,
        "revision": revision,
        "files": [item.as_dict() for item in normalized_files],
        "state_history": [
            {
                "from": None,
                "to": AssetState.DOWNLOADING.value,
                "at": timestamp,
                "actor": actor,
                "actor_role": normalized_role.value,
                "evidence_ref": evidence_ref,
                "summary": "candidate created",
            }
        ],
        "created_at": timestamp,
        "generator_version": generator_version,
        "metadata": normalized_metadata,
    }
    validate_manifest(value)
    return value


def build_g3_candidate_manifest(
    *,
    asset_type: AssetType | str,
    name: str,
    source: str,
    revision: str,
    files: Iterable[AssetFile | Mapping[str, Any]],
    actor: str,
    actor_role: AssetActorRole | str,
    actor_instance_id: str,
    evidence_ref: str,
    evidence_sha256: str,
    generator_version: str,
    generator_git_commit: str,
    metadata: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a downloading candidate under the additive Stage 0 G3 profile."""

    value = build_manifest(
        asset_type=asset_type,
        name=name,
        source=source,
        revision=revision,
        files=files,
        actor=actor,
        actor_role=actor_role,
        evidence_ref=evidence_ref,
        generator_version=generator_version,
        metadata=metadata,
        created_at=created_at,
    )
    _require_text(actor_instance_id, field="actor_instance_id", maximum=512)
    _require_hash(evidence_sha256, field="evidence_sha256")
    _require_git_commit(generator_git_commit, field="generator_git_commit")
    value["contract_profile"] = G3_CONTRACT_PROFILE
    value["generator_git_commit"] = generator_git_commit
    value["state_history"][0]["actor_instance_id"] = actor_instance_id
    value["state_history"][0]["evidence_sha256"] = evidence_sha256
    value["candidate_id"] = "0" * 64
    value["candidate_id"] = compute_candidate_id(value)
    validate_g3_manifest(value)
    return value


def transition_manifest(
    manifest: Mapping[str, Any],
    target: AssetState | str,
    *,
    actor: str,
    actor_role: AssetActorRole | str,
    evidence_ref: str | None,
    summary: str,
    at: str | None = None,
    actor_instance_id: str | None = None,
    evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a new manifest value with one audited state transition."""

    validate_manifest(manifest)
    previous = AssetState(manifest["state"])
    g3_profile = _uses_g3_profile(manifest)
    try:
        target_state = AssetState(target)
        normalized_role = AssetActorRole(actor_role)
    except (TypeError, ValueError) as error:
        raise AssetValidationError("Unknown target state or actor_role") from error
    validate_state_transition(
        previous,
        target_state,
        actor_role=normalized_role,
    )
    if (
        g3_profile
        and previous is AssetState.VERIFIED
        and target_state is AssetState.READY
    ):
        raise AssetValidationError(
            "G3 verified -> ready requires the qualification admission API"
        )
    _require_text(actor, field="actor", maximum=256)
    if g3_profile:
        _require_text(summary, field="summary", maximum=2048)
        _require_text(
            actor_instance_id, field="actor_instance_id", maximum=512
        )
        _require_hash(evidence_sha256, field="evidence_sha256")
    elif actor_instance_id is not None or evidence_sha256 is not None:
        raise AssetValidationError(
            "actor_instance_id/evidence_sha256 require contract_profile=stage0-g3-v1"
        )
    elif not isinstance(summary, str):
        raise AssetValidationError("summary must be a string")
    timestamp = at or _utc_now()
    _require_timestamp(timestamp, field="at")
    _validate_evidence_ref(
        evidence_ref,
        field=f"{previous.value} -> {target_state.value}",
        required=(
            g3_profile
            or target_state
            in {AssetState.VERIFIED, AssetState.READY, AssetState.INVALID}
        ),
    )
    updated = deepcopy(dict(manifest))
    updated["state"] = target_state.value
    event = {
        "from": previous.value,
        "to": target_state.value,
        "at": timestamp,
        "actor": actor,
        "actor_role": normalized_role.value,
        "evidence_ref": evidence_ref,
        "summary": summary,
    }
    if g3_profile:
        event["actor_instance_id"] = actor_instance_id
        event["evidence_sha256"] = evidence_sha256
    updated["state_history"].append(event)
    validate_manifest(updated)
    return updated


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AssetEncodingError(f"Duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate strict UTF-8 JSON, explicitly rejecting a BOM."""

    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise AssetEncodingError(f"UTF-8 BOM is forbidden: {manifest_path}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AssetEncodingError(f"Manifest is not UTF-8: {manifest_path}") from error

    def reject_constant(value: str) -> None:
        raise AssetEncodingError(f"Non-finite JSON number is forbidden: {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except AssetEncodingError:
        raise
    except json.JSONDecodeError as error:
        raise AssetEncodingError(f"Invalid JSON: {manifest_path}: {error}") from error
    if not isinstance(decoded, dict):
        raise AssetValidationError("Manifest root must be an object")
    validate_manifest(decoded)
    return decoded


def _coerce_manifest(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        return load_manifest(value)
    copied = deepcopy(dict(value))
    validate_manifest(copied)
    return copied


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_new_file(target: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        # A hard-link publication is atomic and fails if the immutable target
        # already exists.  The temporary name is then removed precisely.
        os.link(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(
        getattr(path, "is_junction", lambda: False)()
    )


def _reject_symlink_chain(path: Path, *, field: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() or _is_link_like(current):
            if _is_link_like(current):
                raise AssetValidationError(
                    f"{field} may not contain symlinks or junctions: {current}"
                )


def _approved_manifest_target(
    path: str | Path,
    manifest_root: str | Path,
) -> tuple[Path, Path]:
    supplied_root = Path(manifest_root)
    if ".." in supplied_root.parts:
        raise AssetValidationError("manifest_root may not contain '..'")
    root = Path(os.path.abspath(supplied_root))
    _reject_symlink_chain(root, field="manifest_root")
    if not root.exists() or not root.is_dir():
        raise AssetValidationError(
            f"manifest_root must be an existing directory: {supplied_root}"
        )

    supplied_target = Path(path)
    if ".." in supplied_target.parts:
        raise AssetValidationError("Manifest target may not contain '..'")
    target = (
        Path(os.path.abspath(supplied_target))
        if supplied_target.is_absolute()
        else Path(os.path.abspath(root / supplied_target))
    )
    try:
        target.relative_to(root)
    except ValueError as error:
        raise AssetValidationError(
            f"Manifest target escapes approved manifest_root: {target}"
        ) from error
    validate_asset_path(target.name)
    if not target.parent.exists() or not target.parent.is_dir():
        raise AssetValidationError(
            f"Manifest target parent must already exist: {target.parent}"
        )
    _reject_symlink_chain(target.parent, field="Manifest target parent")
    resolved_root = root.resolve(strict=True)
    resolved_parent = target.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as error:
        raise AssetValidationError(
            f"Manifest target parent escapes approved manifest_root: {target.parent}"
        ) from error
    if target.exists() or _is_link_like(target):
        if _is_link_like(target):
            raise AssetValidationError(f"Manifest target may not be a symlink: {target}")
    return target, resolved_root


@contextmanager
def _advisory_manifest_lock(target: Path) -> Iterator[None]:
    """Hold a persistent per-target advisory lock across a CAS replacement."""

    lock_path = target.parent / f".{target.name}.publish.lock"
    if _is_link_like(lock_path):
        raise AssetValidationError(f"Manifest lock may not be a symlink: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o644)
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise AssetValidationError(
                f"Manifest lock must be a single-link regular file: {lock_path}"
            )
        get_effective_uid = getattr(os, "geteuid", None)
        if get_effective_uid is not None and metadata.st_uid != get_effective_uid():
            raise AssetValidationError(
                f"Manifest lock has an unexpected owner: {lock_path}"
            )
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_replacement(previous: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    previous_state = AssetState(previous["state"])
    if previous_state is AssetState.READY:
        raise AssetValidationError(
            "A published ready manifest is immutable at its existing path; "
            "publish the invalidation record to a new path"
        )
    if previous_state is AssetState.INVALID:
        raise AssetValidationError(
            "A published invalid manifest is terminal and cannot be replaced"
        )
    previous_history = previous["state_history"]
    next_history = value["state_history"]
    if value["asset_id"] != previous["asset_id"]:
        raise AssetValidationError("Replacement cannot change asset_id")
    if len(next_history) <= len(previous_history) or (
        next_history[: len(previous_history)] != previous_history
    ):
        raise AssetValidationError(
            "Replacement must append to the existing state_history"
        )
    immutable_previous = {
        key: item
        for key, item in previous.items()
        if key not in {"state", "state_history"}
    }
    immutable_next = {
        key: item
        for key, item in value.items()
        if key not in {"state", "state_history"}
    }
    if immutable_next != immutable_previous:
        raise AssetValidationError(
            "Replacement cannot rewrite identity or manifest metadata"
        )


def publish_manifest_atomic(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    manifest_root: str | Path,
    allow_replace: bool = False,
    expected_previous_sha256: str | None = None,
) -> Path:
    """Validate and atomically publish canonical UTF-8 JSON.

    Historical manifests are immutable by default.  `allow_replace` exists for
    an audited state advance of the same candidate, not for changing identity,
    rewriting metadata, or flipping an invalid candidate back to success.
    """

    value = deepcopy(dict(manifest))
    validate_manifest(value)
    target, _ = _approved_manifest_target(path, manifest_root)
    payload = stable_json_bytes(value)
    if payload.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("Canonical JSON encoder unexpectedly emitted a BOM")
    if expected_previous_sha256 is not None and (
        not isinstance(expected_previous_sha256, str)
        or not SHA256_PATTERN.fullmatch(expected_previous_sha256)
    ):
        raise AssetValidationError(
            "expected_previous_sha256 must be 64 lowercase hex characters"
        )
    if expected_previous_sha256 is not None and not allow_replace:
        raise AssetValidationError(
            "expected_previous_sha256 is only valid with allow_replace=True"
        )

    if allow_replace:
        optimistic_exists = target.exists()
        optimistic_digest = sha256_file(target) if optimistic_exists else None
        with _advisory_manifest_lock(target):
            if _is_link_like(target):
                raise AssetValidationError(
                    f"Manifest target may not become a symlink: {target}"
                )
            current_exists = target.exists()
            current_digest = sha256_file(target) if current_exists else None
            if (optimistic_exists, optimistic_digest) != (
                current_exists,
                current_digest,
            ):
                raise AssetValidationError(
                    "Concurrent manifest publication detected during CAS"
                )
            if not current_exists:
                if expected_previous_sha256 is not None:
                    raise AssetValidationError(
                        "Stale expected_previous_sha256: target does not exist"
                    )
                _publish_new_file(target, payload)
            else:
                if expected_previous_sha256 is None:
                    raise AssetValidationError(
                        "allow_replace requires expected_previous_sha256 for an "
                        "existing target"
                    )
                if current_digest != expected_previous_sha256:
                    raise AssetValidationError(
                        "Stale expected_previous_sha256 for manifest replacement"
                    )
                previous = load_manifest(target)
                if sha256_file(target) != current_digest:
                    raise AssetValidationError(
                        "Concurrent manifest publication detected during CAS read"
                    )
                _validate_replacement(previous, value)
                atomic_write_bytes(target, payload)
    else:
        try:
            _publish_new_file(target, payload)
        except FileExistsError as error:
            raise FileExistsError(f"Manifest already exists: {target}") from error
    return target


@dataclass(frozen=True, slots=True)
class ResolvedAssetFile:
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str
    role: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    asset_id: str
    asset_type: AssetType
    name: str
    revision: str
    root: Path
    files: tuple[ResolvedAssetFile, ...]

    def path_for(self, relative_path: str) -> Path:
        normalized = validate_asset_path(relative_path)
        for item in self.files:
            if item.relative_path == normalized:
                return item.path
        raise KeyError(relative_path)


def _resolve_file(root: Path, descriptor: AssetFile) -> Path:
    relative = PurePosixPath(descriptor.path)
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AssetVerificationError(
                f"Symlinks are forbidden in ready assets: {descriptor.path}"
            )
    if not candidate.exists():
        raise AssetVerificationError(f"Asset file is missing: {descriptor.path}")
    if not candidate.is_file():
        raise AssetVerificationError(f"Asset path is not a regular file: {descriptor.path}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AssetVerificationError(
            f"Asset file escapes its root: {descriptor.path}"
        ) from error
    return resolved


def _verify_files(
    manifest: Mapping[str, Any],
    asset_root: str | Path,
) -> tuple[ResolvedAssetFile, ...]:
    supplied_root = Path(asset_root)
    if supplied_root.is_symlink():
        raise AssetVerificationError(f"Asset root may not be a symlink: {supplied_root}")
    try:
        root = supplied_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise AssetVerificationError(f"Asset root is missing: {supplied_root}") from error
    if not root.is_dir():
        raise AssetVerificationError(f"Asset root is not a directory: {root}")
    resolved_files: list[ResolvedAssetFile] = []
    for descriptor in _normalize_files(manifest["files"]):
        path = _resolve_file(root, descriptor)
        actual_size = path.stat().st_size
        if actual_size != descriptor.size_bytes:
            raise AssetVerificationError(
                f"Size mismatch for {descriptor.path}: expected "
                f"{descriptor.size_bytes}, got {actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != descriptor.sha256:
            raise AssetVerificationError(
                f"SHA-256 mismatch for {descriptor.path}: expected "
                f"{descriptor.sha256}, got {actual_hash}"
            )
        resolved_files.append(
            ResolvedAssetFile(
                relative_path=descriptor.path,
                path=path,
                size_bytes=descriptor.size_bytes,
                sha256=descriptor.sha256,
                role=descriptor.role,
            )
        )
    return tuple(resolved_files)


def verify_only(
    manifest: Mapping[str, Any] | str | Path,
    asset_root: str | Path,
) -> dict[str, Any]:
    """Fully verify an existing candidate without writing files or state."""

    value = _coerce_manifest(manifest)
    state = AssetState(value["state"])
    if state in {AssetState.DOWNLOADING, AssetState.INVALID}:
        raise AssetVerificationError(
            f"verify-only refuses state={state.value}; acquisition must finish first"
        )
    files = _verify_files(value, asset_root)
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "asset_id": value["asset_id"],
        "state": state.value,
        "files_checked": len(files),
        "bytes_checked": sum(item.size_bytes for item in files),
        "ok": True,
    }


def resolve_ready_asset(
    manifest: Mapping[str, Any] | str | Path,
    asset_root: str | Path,
) -> ResolvedAsset:
    """Resolve a ready logical asset to safe final paths.

    READY resolution always performs full size and SHA-256 verification.
    """

    value = _coerce_manifest(manifest)
    state = AssetState(value["state"])
    if state is not AssetState.READY:
        raise AssetNotReadyError(
            f"Runtime resolution requires state=ready, got {state.value}"
        )
    if _uses_g3_profile(value):
        raise AssetNotReadyError(
            "G3 ready assets require resolve_qualified_asset and qualification evidence"
        )
    supplied_root = Path(asset_root)
    files = _verify_files(value, supplied_root)
    resolved_root = supplied_root.resolve(strict=True)
    return ResolvedAsset(
        asset_id=value["asset_id"],
        asset_type=AssetType(value["asset_type"]),
        name=value["name"],
        revision=value["revision"],
        root=resolved_root,
        files=files,
    )


def resolve_qualified_asset(
    manifest: Mapping[str, Any] | str | Path,
    asset_root: str | Path,
    qualification: Mapping[str, Any],
    *,
    qualification_ref: str,
    requirements_artifact_hash: str,
) -> ResolvedAsset:
    """Resolve a G3 READY asset bound to its formal qualification artifact."""

    value = validate_qualified_ready_manifest(
        manifest,
        qualification,
        qualification_ref=qualification_ref,
        requirements_artifact_hash=requirements_artifact_hash,
    )
    supplied_root = Path(asset_root)
    files = _verify_files(value, supplied_root)
    resolved_root = supplied_root.resolve(strict=True)
    return ResolvedAsset(
        asset_id=value["asset_id"],
        asset_type=AssetType(value["asset_type"]),
        name=value["name"],
        revision=value["revision"],
        root=resolved_root,
        files=files,
    )


def validate_qualified_ready_manifest(
    manifest: Mapping[str, Any] | str | Path,
    qualification: Mapping[str, Any],
    *,
    qualification_ref: str,
    requirements_artifact_hash: str,
) -> dict[str, Any]:
    """Validate G3 READY admission metadata without rehashing asset payload files."""

    value = _coerce_manifest(manifest)
    if not _uses_g3_profile(value):
        raise AssetNotReadyError("Qualified resolution requires a G3 manifest")
    if AssetState(value["state"]) is not AssetState.READY:
        raise AssetNotReadyError(
            f"Qualified resolution requires state=ready, got {value['state']}"
        )
    normalized_ref = _validate_evidence_ref(
        qualification_ref, field="qualified resolution", required=True
    )
    final_event = value["state_history"][-1]
    if (
        final_event["from"] != AssetState.VERIFIED.value
        or final_event["to"] != AssetState.READY.value
        or final_event["actor_role"] != AssetActorRole.GATE.value
        or final_event["evidence_ref"] != normalized_ref
    ):
        raise AssetNotReadyError(
            "G3 READY history is not bound to the supplied qualification_ref"
        )
    validate_g3_qualification(qualification)
    if final_event["evidence_sha256"] != qualification["artifact_hash"]:
        raise AssetNotReadyError(
            "G3 READY history qualification hash does not match the report"
        )
    if _timestamp_value(
        final_event["at"], field="READY history timestamp"
    ) < _timestamp_value(
        qualification["checked_at"], field="qualification.checked_at"
    ):
        raise AssetNotReadyError(
            "G3 READY history predates the supplied qualification"
        )
    verified = deepcopy(value)
    verified["state"] = AssetState.VERIFIED.value
    verified["state_history"].pop()
    _validate_qualification_for_verified_manifest(
        verified,
        qualification,
        requirements_artifact_hash=requirements_artifact_hash,
    )
    return value


# Explicit aliases keep call sites readable without weakening the single
# implementation of parsing and resolution semantics.
load_asset_manifest = load_manifest
resolve_ready_manifest = resolve_ready_asset
