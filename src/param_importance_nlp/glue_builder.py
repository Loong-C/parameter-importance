"""Strictly offline Stage 0 GLUE derived-dataset construction.

This module deliberately has no CLI or runner integration.  It accepts only
already-qualified local asset roots, one frozen GLUE requirement, and the
verified tokenizer requirement that names the three selected descriptors.  It
builds a Hugging Face ``DatasetDict`` in a private ``DATA_ROOT/tmp`` staging
directory, validates the complete saved result, and publishes with an atomic
no-clobber directory rename.

The network guard is process-wide.  Construction is therefore serialized in
the current process while the guard is active, and every *external* socket
egress attempt is counted and rejected.  Loopback connections and local
subprocess machinery used by the Hugging Face datasets multiprocessing
manager are permitted so the offline build can complete without any external
network access.  Hugging Face dependencies are imported lazily so the base
package remains usable without the optional ``server`` extra.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import errno
import ipaddress
from numbers import Integral
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
from types import ModuleType
from typing import Any, Final
import uuid

from .atomic import atomic_write_bytes, sha256_file
from .contracts import (
    DependencyUnavailable,
    canonical_json_bytes,
    canonical_json_hash,
    ensure_json_object,
    loads_strict_json,
)
from .g3_gate import (
    GLUE_PREPROCESSING_VERSION,
    glue_preprocessing_config_hash,
)
from .providers.optional import require_optional_dependency


RESULT_SCHEMA_VERSION: Final = "stage0-glue-derived-build-result-v2"
FAILURE_SCHEMA_VERSION: Final = "stage0-glue-derived-build-failure-v1"
SIDECAR_SCHEMA_VERSION: Final = "stage0-glue-derived-dataset-v2"
MAP_FINGERPRINT_SCHEMA_VERSION: Final = "stage0-glue-map-fingerprint-v2"
SIDECAR_NAME: Final = "stage0-glue-derived-build.json"
OUTPUT_COLUMNS: Final = ("input_ids", "attention_mask", "labels")
MAX_LENGTH: Final = 512
PAD_TOKEN: Final = "<|padding|>"
PAD_TOKEN_ID: Final = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TASK_FIELDS: Final[Mapping[str, tuple[str, ...]]] = {
    "sst2": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
}
_TASK_LABELS: Final[Mapping[str, Mapping[str, int]]] = {
    "sst2": {"negative": 0, "positive": 1},
    "mnli": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "rte": {"entailment": 0, "not_entailment": 1},
}
_REQUIREMENT_FIELDS: Final = frozenset(
    {
        "task",
        "repository",
        "revision",
        "raw_files",
        "split_counts",
        "text_fields",
        "label_mapping",
        "unlabeled_test_policy",
        "preprocessing",
    }
)
_PREPROCESSING_FIELDS: Final = frozenset(
    {
        "tokenizer_name",
        "max_length",
        "truncation",
        "padding",
        "pad_token",
        "derived_splits",
    }
)
_RAW_FILE_FIELDS: Final = frozenset(
    {"path", "size_bytes", "sha256", "role"}
)
_TOKENIZER_DESCRIPTOR_ROLES: Final[Mapping[str, str]] = {
    "special_tokens_map.json": "special_tokens",
    "tokenizer.json": "tokenizer_model",
    "tokenizer_config.json": "tokenizer_config",
}
_OFFLINE_ENVIRONMENT: Final[Mapping[str, str]] = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
_PROCESS_OFFLINE_LOCK = threading.RLock()


class _BuildFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}:{message}")


class _NetworkEgressBlocked(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _Dependencies:
    datasets: ModuleType
    transformers: ModuleType


@dataclass(frozen=True, slots=True)
class GlueDerivedBuildFailureReport:
    """JSON-safe failure information, including any preserved staging path."""

    code: str
    message: str
    task: str | None
    target_path: str
    staging_path: str | None
    network_attempts: int

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "status": "failed",
            "code": self.code,
            "message": self.message,
            "task": self.task,
            "target_path": self.target_path,
            "staging_path": self.staging_path,
            "network_attempts": self.network_attempts,
        }
        canonical_json_bytes(value)
        return value


class GlueDerivedBuildError(RuntimeError):
    """Raised after preserving the private staging directory on failure."""

    def __init__(self, report: GlueDerivedBuildFailureReport) -> None:
        self.report = report
        super().__init__(f"{report.code}: {report.message}")


@dataclass(frozen=True, slots=True)
class GlueDerivedBuildResult:
    """Strict JSON-safe result of a build or read-only reuse validation."""

    status: str
    task: str
    raw_asset_id: str
    tokenizer_asset_id: str
    tokenizer_descriptor_inventory: tuple[Mapping[str, Any], ...]
    tokenizer_descriptor_inventory_hash: str
    target_path: str
    target_ref: str
    generator_git_commit: str
    preprocessing_version: str
    preprocessing_config_hash: str
    requirement_hash: str
    derived_splits: tuple[str, ...]
    split_counts: Mapping[str, int]
    map_fingerprints: Mapping[str, str]
    file_inventory: tuple[Mapping[str, Any], ...]
    network_attempts: int

    def __post_init__(self) -> None:
        if self.status not in {"built", "reused"}:
            raise ValueError("GLUE_BUILD_RESULT_STATUS_INVALID")
        try:
            normalized_inventory = normalize_tokenizer_descriptor_inventory(
                {
                    "name": "pythia-tokenizer",
                    "files": [dict(item) for item in self.tokenizer_descriptor_inventory],
                }
            )
        except ValueError as error:
            raise ValueError("GLUE_BUILD_RESULT_TOKENIZER_INVENTORY_INVALID") from error
        if tuple(dict(item) for item in self.tokenizer_descriptor_inventory) != (
            normalized_inventory
        ):
            raise ValueError("GLUE_BUILD_RESULT_TOKENIZER_INVENTORY_NOT_CANONICAL")
        if (
            _SHA256.fullmatch(self.tokenizer_descriptor_inventory_hash) is None
            or canonical_json_hash([dict(item) for item in normalized_inventory])
            != self.tokenizer_descriptor_inventory_hash
        ):
            raise ValueError("GLUE_BUILD_RESULT_TOKENIZER_INVENTORY_HASH_INVALID")
        canonical_json_bytes(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": self.status,
            "task": self.task,
            "raw_asset_id": self.raw_asset_id,
            "tokenizer_asset_id": self.tokenizer_asset_id,
            "tokenizer_descriptor_inventory": [
                dict(item) for item in self.tokenizer_descriptor_inventory
            ],
            "tokenizer_descriptor_inventory_hash": (
                self.tokenizer_descriptor_inventory_hash
            ),
            "target_path": self.target_path,
            "target_ref": self.target_ref,
            "generator_git_commit": self.generator_git_commit,
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_config_hash": self.preprocessing_config_hash,
            "requirement_hash": self.requirement_hash,
            "derived_splits": list(self.derived_splits),
            "split_counts": dict(self.split_counts),
            "map_fingerprints": dict(self.map_fingerprints),
            "file_inventory": [dict(item) for item in self.file_inventory],
            "network_attempts": self.network_attempts,
        }


@dataclass(slots=True)
class _NetworkCounter:
    attempts: int = 0


def _load_dependencies() -> _Dependencies:
    return _Dependencies(
        datasets=require_optional_dependency(
            "datasets", feature="stage0_glue_derived_builder"
        ),
        transformers=require_optional_dependency(
            "transformers", feature="stage0_glue_derived_builder"
        ),
    )


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(
        getattr(path, "is_junction", lambda: False)()
    )


def _absolute_without_resolution(path: str | Path) -> Path:
    supplied = Path(path)
    if ".." in supplied.parts:
        raise _BuildFailure("PATH_PARENT_TRAVERSAL", str(supplied))
    return Path(os.path.abspath(supplied))


def _approved_data_root(path: str | Path) -> Path:
    root = _absolute_without_resolution(path)
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise _BuildFailure("DATA_ROOT_LINK_FORBIDDEN", str(current))
    if not root.exists() or not root.is_dir():
        raise _BuildFailure("DATA_ROOT_INVALID", str(root))
    return root.resolve(strict=True)


def _candidate_under_root(root: Path, value: str | Path, *, field: str) -> Path:
    supplied = Path(value)
    if ".." in supplied.parts:
        raise _BuildFailure(f"{field}_PARENT_TRAVERSAL", str(supplied))
    candidate = (
        _absolute_without_resolution(supplied)
        if supplied.is_absolute()
        else _absolute_without_resolution(root / supplied)
    )
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise _BuildFailure(f"{field}_OUTSIDE_DATA_ROOT", str(candidate)) from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or _is_link_like(current):
            if _is_link_like(current):
                raise _BuildFailure(f"{field}_LINK_FORBIDDEN", str(current))
    return candidate


def _existing_directory_under_root(
    root: Path, value: str | Path, *, field: str
) -> Path:
    candidate = _candidate_under_root(root, value, field=field)
    if not candidate.exists() or not candidate.is_dir():
        raise _BuildFailure(f"{field}_DIRECTORY_MISSING", str(candidate))
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise _BuildFailure(f"{field}_OUTSIDE_DATA_ROOT", str(candidate)) from error
    return resolved


def _target_under_root(root: Path, value: str | Path) -> Path:
    target = _candidate_under_root(root, value, field="TARGET")
    if target == root:
        raise _BuildFailure("TARGET_IS_DATA_ROOT", str(target))
    if target.exists() and not target.is_dir():
        raise _BuildFailure("TARGET_NOT_DIRECTORY", str(target))
    return target


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _validate_identity(value: str, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _BuildFailure(f"{field}_INVALID", str(value))
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise _BuildFailure(f"{field}_INVALID", str(value))
    return value


def normalize_tokenizer_descriptor_inventory(
    tokenizer_requirement: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return the canonical frozen tokenizer file inventory used by GLUE.

    The caller must supply the verified tokenizer requirement rather than a
    directory discovered at runtime.  Only the three tokenizer descriptors
    used by ``AutoTokenizer`` participate in derived-dataset lineage; model
    files and operational lock files in the shared asset root are excluded.
    """

    if not isinstance(tokenizer_requirement, Mapping):
        raise ValueError("TOKENIZER_REQUIREMENT_NOT_OBJECT")
    if tokenizer_requirement.get("name") != "pythia-tokenizer":
        raise ValueError("TOKENIZER_REQUIREMENT_NAME_INVALID")
    raw_files = tokenizer_requirement.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != len(
        _TOKENIZER_DESCRIPTOR_ROLES
    ):
        raise ValueError("TOKENIZER_DESCRIPTOR_COUNT_INVALID")

    inventory: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping) or set(raw) != _RAW_FILE_FIELDS:
            raise ValueError(f"TOKENIZER_DESCRIPTOR_FIELDS_INVALID:{index}")
        path = raw.get("path")
        if not isinstance(path, str) or path in seen_paths:
            raise ValueError(f"TOKENIZER_DESCRIPTOR_PATH_INVALID:{index}")
        expected_role = _TOKENIZER_DESCRIPTOR_ROLES.get(path)
        if expected_role is None or raw.get("role") != expected_role:
            raise ValueError(f"TOKENIZER_DESCRIPTOR_ROLE_INVALID:{path}")
        size = raw.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"TOKENIZER_DESCRIPTOR_SIZE_INVALID:{path}")
        digest = raw.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"TOKENIZER_DESCRIPTOR_SHA256_INVALID:{path}")
        seen_paths.add(path)
        inventory.append(
            {
                "path": path,
                "size_bytes": size,
                "sha256": digest,
                "role": expected_role,
            }
        )
    if seen_paths != set(_TOKENIZER_DESCRIPTOR_ROLES):
        raise ValueError("TOKENIZER_DESCRIPTOR_PATH_SET_INVALID")
    inventory.sort(key=lambda item: item["path"])
    canonical_json_bytes(inventory)
    return tuple(inventory)


def _validate_requirement(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REQUIREMENT_FIELDS:
        raise _BuildFailure("GLUE_REQUIREMENT_FIELDS_INVALID", "single task requirement")
    requirement = deepcopy(dict(value))
    task = _require_text(requirement["task"], field="GLUE_TASK")
    if task not in _TASK_FIELDS:
        raise _BuildFailure("GLUE_TASK_UNSUPPORTED", task)
    if requirement["repository"] != "nyu-mll/glue":
        raise _BuildFailure("GLUE_REPOSITORY_INVALID", str(requirement["repository"]))
    _validate_identity(
        requirement["revision"], field="GLUE_REVISION", pattern=_GIT_COMMIT
    )
    if requirement["text_fields"] != list(_TASK_FIELDS[task]):
        raise _BuildFailure("GLUE_TEXT_FIELDS_INVALID", task)
    if requirement["label_mapping"] != dict(_TASK_LABELS[task]):
        raise _BuildFailure("GLUE_LABEL_MAPPING_INVALID", task)
    if requirement["unlabeled_test_policy"] != (
        "retain_in_raw_exclude_from_derived_training_and_evaluation"
    ):
        raise _BuildFailure("GLUE_UNLABELED_TEST_POLICY_INVALID", task)

    split_counts = requirement["split_counts"]
    if not isinstance(split_counts, Mapping) or not split_counts:
        raise _BuildFailure("GLUE_SPLIT_COUNTS_INVALID", task)
    normalized_counts: dict[str, int] = {}
    for split, count in split_counts.items():
        split_name = _require_text(split, field="GLUE_SPLIT")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise _BuildFailure("GLUE_SPLIT_COUNT_INVALID", split_name)
        normalized_counts[split_name] = count
    requirement["split_counts"] = normalized_counts

    raw_files = requirement["raw_files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise _BuildFailure("GLUE_RAW_FILES_INVALID", task)
    normalized_files: list[dict[str, Any]] = []
    roles: set[str] = set()
    paths: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping) or set(raw) != _RAW_FILE_FIELDS:
            raise _BuildFailure("GLUE_RAW_FILE_FIELDS_INVALID", str(index))
        path = _require_text(raw["path"], field="GLUE_RAW_FILE_PATH")
        path_object = Path(path)
        if (
            path_object.name != path
            or path in {".", ".."}
            or "/" in path
            or "\\" in path
            or not path.endswith(".parquet")
            or path in paths
        ):
            raise _BuildFailure("GLUE_RAW_FILE_PATH_INVALID", path)
        size = raw["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise _BuildFailure("GLUE_RAW_FILE_SIZE_INVALID", path)
        digest = _validate_identity(
            raw["sha256"], field="GLUE_RAW_FILE_SHA256", pattern=_SHA256
        )
        role = _require_text(raw["role"], field="GLUE_RAW_FILE_ROLE")
        if role not in normalized_counts:
            raise _BuildFailure("GLUE_RAW_FILE_ROLE_INVALID", role)
        paths.add(path)
        roles.add(role)
        normalized_files.append(
            {"path": path, "size_bytes": size, "sha256": digest, "role": role}
        )
    if roles != set(normalized_counts):
        raise _BuildFailure("GLUE_RAW_FILE_SPLITS_MISMATCH", task)
    requirement["raw_files"] = normalized_files

    preprocessing = requirement["preprocessing"]
    if not isinstance(preprocessing, Mapping) or set(preprocessing) != _PREPROCESSING_FIELDS:
        raise _BuildFailure("GLUE_PREPROCESSING_FIELDS_INVALID", task)
    expected = {
        "tokenizer_name": "pythia-tokenizer",
        "max_length": MAX_LENGTH,
        "truncation": True,
        "padding": "max_length",
        "pad_token": PAD_TOKEN,
    }
    if any(preprocessing.get(name) != expected_value for name, expected_value in expected.items()):
        raise _BuildFailure("GLUE_PREPROCESSING_CONTRACT_INVALID", task)
    derived = preprocessing["derived_splits"]
    if (
        not isinstance(derived, list)
        or not derived
        or any(not isinstance(split, str) or not split for split in derived)
        or len(set(derived)) != len(derived)
        or any(split not in normalized_counts for split in derived)
        or any(split.startswith("test") for split in derived)
    ):
        raise _BuildFailure("GLUE_DERIVED_SPLITS_INVALID", task)
    requirement["preprocessing"] = {
        **expected,
        "derived_splits": list(derived),
    }
    # This is also the public canonical-JSON boundary shared with G3.
    glue_preprocessing_config_hash(requirement)
    return requirement


def _regular_file(path: Path, *, code: str) -> os.stat_result:
    if _is_link_like(path):
        raise _BuildFailure(code, str(path))
    try:
        metadata = path.stat()
    except OSError as error:
        raise _BuildFailure(code, str(path)) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise _BuildFailure(code, str(path))
    return metadata


def _selected_raw_inventory(
    raw_root: Path, requirement: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], str]:
    selected = set(requirement["preprocessing"]["derived_splits"])
    inventory: list[dict[str, Any]] = []
    for item in requirement["raw_files"]:
        if item["role"] not in selected:
            continue
        path = raw_root / item["path"]
        metadata = _regular_file(path, code="GLUE_RAW_FILE_INVALID")
        if metadata.st_size != item["size_bytes"]:
            raise _BuildFailure("GLUE_RAW_FILE_SIZE_MISMATCH", str(path))
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise _BuildFailure("GLUE_RAW_FILE_HASH_MISMATCH", str(path))
        inventory.append(
            {
                "path": item["path"],
                "role": item["role"],
                "size_bytes": item["size_bytes"],
                "sha256": digest,
            }
        )
    if not inventory:
        raise _BuildFailure("GLUE_RAW_DERIVED_FILES_EMPTY", str(raw_root))
    inventory.sort(key=lambda item: (item["role"], item["path"]))
    return tuple(inventory), canonical_json_hash(inventory)


def _verified_tokenizer_inventory(
    root: Path,
    expected_inventory: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    observed: list[dict[str, Any]] = []
    for item in expected_inventory:
        path = root / str(item["path"])
        metadata = _regular_file(path, code="TOKENIZER_DESCRIPTOR_FILE_INVALID")
        if metadata.st_size != item["size_bytes"]:
            raise _BuildFailure("TOKENIZER_DESCRIPTOR_SIZE_MISMATCH", str(path))
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise _BuildFailure("TOKENIZER_DESCRIPTOR_HASH_MISMATCH", str(path))
        observed.append(
            {
                "path": item["path"],
                "size_bytes": int(metadata.st_size),
                "sha256": digest,
                "role": item["role"],
            }
        )
    observed.sort(key=lambda item: item["path"])
    return tuple(observed), canonical_json_hash(observed)


def _reverify_tokenizer_inventory(
    root: Path,
    expected_inventory: Sequence[Mapping[str, Any]],
    initial_inventory: Sequence[Mapping[str, Any]],
    initial_hash: str,
) -> None:
    try:
        observed, observed_hash = _verified_tokenizer_inventory(
            root,
            expected_inventory,
        )
    except _BuildFailure as error:
        raise _BuildFailure(
            "TOKENIZER_SOURCE_CHANGED_DURING_BUILD",
            f"{error.code}:{error.message}",
        ) from error
    if tuple(observed) != tuple(initial_inventory) or observed_hash != initial_hash:
        raise _BuildFailure("TOKENIZER_SOURCE_CHANGED_DURING_BUILD", str(root))


def _file_inventory(root: Path) -> tuple[dict[str, Any], ...]:
    if _is_link_like(root) or not root.is_dir():
        raise _BuildFailure("DATASET_ROOT_INVALID", str(root))
    files: list[dict[str, Any]] = []
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = current / name
            if _is_link_like(child):
                raise _BuildFailure("DATASET_LINK_FORBIDDEN", str(child))
        for name in file_names:
            child = current / name
            metadata = _regular_file(child, code="DATASET_FILE_INVALID")
            files.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "size_bytes": int(metadata.st_size),
                }
            )
    files.sort(key=lambda item: item["path"])
    if not files:
        raise _BuildFailure("DATASET_DIRECTORY_EMPTY", str(root))
    return tuple(files)


def _stat_snapshot(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (
            item["path"],
            item["size_bytes"],
            int((root / item["path"]).stat().st_mtime_ns),
        )
        for item in _file_inventory(root)
    )


def _as_list(value: object, *, field: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise _BuildFailure("GLUE_BATCH_FIELD_NOT_SEQUENCE", field)


def _normalize_encoded_batch(
    encoded: Mapping[str, Any],
    labels_value: object,
    *,
    label_count: int,
) -> dict[str, list[Any]]:
    if not isinstance(encoded, Mapping):
        raise _BuildFailure("TOKENIZER_OUTPUT_NOT_MAPPING", type(encoded).__name__)
    inputs = _as_list(encoded.get("input_ids"), field="input_ids")
    masks = _as_list(encoded.get("attention_mask"), field="attention_mask")
    labels = _as_list(labels_value, field="label")
    if not (len(inputs) == len(masks) == len(labels)):
        raise _BuildFailure("TOKENIZER_BATCH_LENGTH_MISMATCH", str(len(labels)))
    normalized_inputs: list[list[int]] = []
    normalized_masks: list[list[int]] = []
    normalized_labels: list[int] = []
    for row_index, (raw_inputs, raw_mask, raw_label) in enumerate(
        zip(inputs, masks, labels, strict=True)
    ):
        row_inputs = _as_list(raw_inputs, field=f"input_ids[{row_index}]")
        row_mask = _as_list(raw_mask, field=f"attention_mask[{row_index}]")
        if len(row_inputs) != MAX_LENGTH or len(row_mask) != MAX_LENGTH:
            raise _BuildFailure("TOKENIZER_SEQUENCE_LENGTH_INVALID", str(row_index))
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in row_inputs):
            raise _BuildFailure("TOKENIZER_INPUT_IDS_INVALID", str(row_index))
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in row_mask):
            raise _BuildFailure("TOKENIZER_ATTENTION_MASK_INVALID", str(row_index))
        input_row = [int(item) for item in row_inputs]
        mask_row = [int(item) for item in row_mask]
        if any(item < 0 for item in input_row) or any(item not in {0, 1} for item in mask_row):
            raise _BuildFailure("TOKENIZER_ROW_VALUES_INVALID", str(row_index))
        if not any(mask_row):
            raise _BuildFailure("TOKENIZER_ATTENTION_MASK_ALL_ZERO", str(row_index))
        if any(
            mask == 0 and token != PAD_TOKEN_ID
            for token, mask in zip(input_row, mask_row, strict=True)
        ):
            raise _BuildFailure("TOKENIZER_PADDING_ID_INVALID", str(row_index))
        if isinstance(raw_label, bool) or not isinstance(raw_label, Integral):
            raise _BuildFailure("GLUE_LABEL_NOT_INTEGER", str(row_index))
        label = int(raw_label)
        if not 0 <= label < label_count:
            raise _BuildFailure("GLUE_LABEL_OUT_OF_RANGE", str(row_index))
        normalized_inputs.append(input_row)
        normalized_masks.append(mask_row)
        normalized_labels.append(label)
    return {
        "input_ids": normalized_inputs,
        "attention_mask": normalized_masks,
        "labels": normalized_labels,
    }


def _dataset_columns(dataset: object) -> tuple[str, ...]:
    columns = getattr(dataset, "column_names", None)
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes, bytearray)):
        raise _BuildFailure("DATASET_COLUMNS_UNAVAILABLE", type(dataset).__name__)
    return tuple(str(item) for item in columns)


def _dataset_length(dataset: object) -> int:
    try:
        value = len(dataset)  # type: ignore[arg-type]
    except TypeError as error:
        raise _BuildFailure("DATASET_LENGTH_UNAVAILABLE", type(dataset).__name__) from error
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _BuildFailure("DATASET_LENGTH_INVALID", str(value))
    return value


def _iter_dataset_batches(dataset: object, *, batch_size: int = 512) -> Iterator[Mapping[str, Any]]:
    iterator = getattr(dataset, "iter", None)
    if callable(iterator):
        for batch in iterator(batch_size=batch_size):
            if not isinstance(batch, Mapping):
                raise _BuildFailure("DATASET_BATCH_NOT_MAPPING", type(batch).__name__)
            yield batch
        return
    length = _dataset_length(dataset)
    for start in range(0, length, batch_size):
        rows = [
            dataset[index]  # type: ignore[index]
            for index in range(start, min(start + batch_size, length))
        ]
        if any(not isinstance(row, Mapping) for row in rows):
            raise _BuildFailure("DATASET_ROW_NOT_MAPPING", str(start))
        yield {
            column: [row[column] for row in rows]  # type: ignore[index]
            for column in OUTPUT_COLUMNS
        }


def _validate_dataset_split(
    dataset: object,
    *,
    split: str,
    expected_rows: int,
    expected_fingerprint: str,
    label_count: int,
    check_loaded_fingerprint: bool = True,
) -> None:
    observed_columns = _dataset_columns(dataset)
    if len(observed_columns) != len(OUTPUT_COLUMNS) or set(observed_columns) != set(
        OUTPUT_COLUMNS
    ):
        raise _BuildFailure("DATASET_OUTPUT_COLUMNS_INVALID", split)
    if _dataset_length(dataset) != expected_rows:
        raise _BuildFailure("DATASET_ROW_COUNT_MISMATCH", split)
    if (
        check_loaded_fingerprint
        and str(getattr(dataset, "_fingerprint", "")) != expected_fingerprint
    ):
        raise _BuildFailure("DATASET_FINGERPRINT_MISMATCH", split)
    observed_rows = 0
    for batch in _iter_dataset_batches(dataset):
        normalized = _normalize_encoded_batch(
            {
                "input_ids": batch.get("input_ids"),
                "attention_mask": batch.get("attention_mask"),
            },
            batch.get("labels"),
            label_count=label_count,
        )
        observed_rows += len(normalized["labels"])
    if observed_rows != expected_rows:
        raise _BuildFailure("DATASET_SCANNED_ROW_COUNT_MISMATCH", split)


def _mapping_fingerprint(
    *,
    requirement: Mapping[str, Any],
    split: str,
    raw_asset_id: str,
    tokenizer_asset_id: str,
    tokenizer_descriptor_inventory_hash: str,
    generator_git_commit: str,
    preprocessing_config_hash: str,
) -> str:
    source_files = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in requirement["raw_files"]
        if item["role"] == split
    ]
    return canonical_json_hash(
        {
            "schema_version": MAP_FINGERPRINT_SCHEMA_VERSION,
            "task": requirement["task"],
            "split": split,
            "raw_asset_id": raw_asset_id,
            "tokenizer_asset_id": tokenizer_asset_id,
            "tokenizer_descriptor_inventory_hash": (
                tokenizer_descriptor_inventory_hash
            ),
            "generator_git_commit": generator_git_commit,
            "preprocessing_version": GLUE_PREPROCESSING_VERSION,
            "preprocessing_config_hash": preprocessing_config_hash,
            "source_files": source_files,
            "expected_rows": requirement["split_counts"][split],
        }
    )


def _feature_contract(datasets: ModuleType) -> object | None:
    features_type = getattr(datasets, "Features", None)
    sequence_type = getattr(datasets, "Sequence", None)
    value_type = getattr(datasets, "Value", None)
    if not all(callable(item) for item in (features_type, sequence_type, value_type)):
        return None
    return features_type(
        {
            "input_ids": sequence_type(value_type("int64"), length=MAX_LENGTH),
            "attention_mask": sequence_type(value_type("int64"), length=MAX_LENGTH),
            "labels": value_type("int64"),
        }
    )


def _load_local_tokenizer(transformers: ModuleType, root: Path) -> object:
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    loader = getattr(auto_tokenizer, "from_pretrained", None)
    if not callable(loader):
        raise _BuildFailure("AUTO_TOKENIZER_LOADER_UNAVAILABLE", "transformers")
    tokenizer = loader(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
    )
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(converter) or converter(PAD_TOKEN) != PAD_TOKEN_ID:
        raise _BuildFailure("TOKENIZER_PADDING_TOKEN_ID_MISMATCH", str(root))
    try:
        tokenizer.pad_token = PAD_TOKEN
    except Exception as error:
        raise _BuildFailure("TOKENIZER_PADDING_TOKEN_ASSIGNMENT_FAILED", str(root)) from error
    if getattr(tokenizer, "pad_token", None) != PAD_TOKEN or getattr(
        tokenizer, "pad_token_id", None
    ) != PAD_TOKEN_ID:
        raise _BuildFailure("TOKENIZER_PADDING_TOKEN_CONFIGURATION_INVALID", str(root))
    if not callable(tokenizer):
        raise _BuildFailure("TOKENIZER_NOT_CALLABLE", str(root))
    return tokenizer


def _build_splits(
    *,
    dependencies: _Dependencies,
    raw_root: Path,
    tokenizer_root: Path,
    cache_root: Path,
    requirement: Mapping[str, Any],
    raw_asset_id: str,
    tokenizer_asset_id: str,
    generator_git_commit: str,
    preprocessing_config_hash: str,
    fingerprints: Mapping[str, str],
) -> object:
    dataset_type = getattr(dependencies.datasets, "Dataset", None)
    from_parquet = getattr(dataset_type, "from_parquet", None)
    dataset_dict_type = getattr(dependencies.datasets, "DatasetDict", None)
    if not callable(from_parquet) or not callable(dataset_dict_type):
        raise _BuildFailure("DATASETS_LOCAL_PARQUET_API_UNAVAILABLE", "datasets")
    tokenizer = _load_local_tokenizer(dependencies.transformers, tokenizer_root)
    feature_contract = _feature_contract(dependencies.datasets)
    text_fields = tuple(requirement["text_fields"])
    label_count = len(requirement["label_mapping"])
    derived: dict[str, object] = {}
    for split in requirement["preprocessing"]["derived_splits"]:
        paths = [
            str(raw_root / item["path"])
            for item in requirement["raw_files"]
            if item["role"] == split
        ]
        if not paths:
            raise _BuildFailure("GLUE_SPLIT_PARQUET_FILES_EMPTY", split)
        raw_dataset = from_parquet(
            paths,
            cache_dir=str(cache_root),
            keep_in_memory=False,
            num_proc=1,
        )
        if _dataset_length(raw_dataset) != requirement["split_counts"][split]:
            raise _BuildFailure("GLUE_RAW_ROW_COUNT_MISMATCH", split)
        raw_columns = _dataset_columns(raw_dataset)
        required_columns = {*text_fields, "label"}
        if not required_columns.issubset(raw_columns):
            raise _BuildFailure("GLUE_RAW_COLUMNS_MISSING", split)

        def tokenize_batch(batch: Mapping[str, Any]) -> dict[str, list[Any]]:
            if not isinstance(batch, Mapping):
                raise _BuildFailure("GLUE_RAW_BATCH_NOT_MAPPING", split)
            tokenizer_arguments = [batch.get(text_fields[0])]
            if len(text_fields) == 2:
                tokenizer_arguments.append(batch.get(text_fields[1]))
            encoded = tokenizer(
                *tokenizer_arguments,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            return _normalize_encoded_batch(
                encoded,
                batch.get("label"),
                label_count=label_count,
            )

        map_kwargs: dict[str, Any] = {
            "batched": True,
            "batch_size": 256,
            "writer_batch_size": 256,
            "remove_columns": list(raw_columns),
            "num_proc": 1,
            "load_from_cache_file": False,
            "new_fingerprint": fingerprints[split],
            "desc": f"stage0-glue-{requirement['task']}-{split}",
        }
        if feature_contract is not None:
            map_kwargs["features"] = feature_contract
        mapper = getattr(raw_dataset, "map", None)
        if not callable(mapper):
            raise _BuildFailure("DATASET_MAP_UNAVAILABLE", split)
        mapped = mapper(tokenize_batch, **map_kwargs)
        _validate_dataset_split(
            mapped,
            split=split,
            expected_rows=requirement["split_counts"][split],
            expected_fingerprint=fingerprints[split],
            label_count=label_count,
        )
        derived[split] = mapped
    return dataset_dict_type(derived)


def _base_sidecar(
    *,
    requirement: Mapping[str, Any],
    raw_asset_id: str,
    tokenizer_asset_id: str,
    raw_files_hash: str,
    tokenizer_descriptor_inventory: Sequence[Mapping[str, Any]],
    tokenizer_descriptor_inventory_hash: str,
    generator_git_commit: str,
    preprocessing_config_hash: str,
    requirement_hash: str,
    fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "task": requirement["task"],
        "raw_asset_id": raw_asset_id,
        "tokenizer_asset_id": tokenizer_asset_id,
        "raw_selected_files_hash": raw_files_hash,
        "tokenizer_descriptor_inventory": [
            dict(item) for item in tokenizer_descriptor_inventory
        ],
        "tokenizer_descriptor_inventory_hash": tokenizer_descriptor_inventory_hash,
        "generator_git_commit": generator_git_commit,
        "requirement_hash": requirement_hash,
        "preprocessing_version": GLUE_PREPROCESSING_VERSION,
        "preprocessing_config_hash": preprocessing_config_hash,
        "format": "hf_load_from_disk",
        "columns": list(OUTPUT_COLUMNS),
        "max_length": MAX_LENGTH,
        "pad_token": PAD_TOKEN,
        "pad_token_id": PAD_TOKEN_ID,
        "derived_splits": list(requirement["preprocessing"]["derived_splits"]),
        "split_counts": {
            split: requirement["split_counts"][split]
            for split in requirement["preprocessing"]["derived_splits"]
        },
        "map_fingerprints": dict(fingerprints),
    }


def _write_sidecar(root: Path, base: Mapping[str, Any]) -> None:
    payload_inventory = [
        dict(item) for item in _file_inventory(root) if item["path"] != SIDECAR_NAME
    ]
    metadata = {**deepcopy(dict(base)), "payload_inventory": payload_inventory}
    atomic_write_bytes(root / SIDECAR_NAME, canonical_json_bytes(metadata))


def _load_and_validate_sidecar(root: Path, base: Mapping[str, Any]) -> dict[str, Any]:
    sidecar_path = root / SIDECAR_NAME
    _regular_file(sidecar_path, code="GLUE_SIDECAR_MISSING_OR_INVALID")
    raw = sidecar_path.read_bytes()
    decoded = ensure_json_object(loads_strict_json(raw), field="GLUE build sidecar")
    if raw != canonical_json_bytes(decoded):
        raise _BuildFailure("GLUE_SIDECAR_NOT_CANONICAL", str(sidecar_path))
    expected_fields = set(base) | {"payload_inventory"}
    if set(decoded) != expected_fields:
        raise _BuildFailure("GLUE_SIDECAR_FIELDS_INVALID", str(sidecar_path))
    for name, expected in base.items():
        if decoded.get(name) != expected:
            raise _BuildFailure("GLUE_SIDECAR_IDENTITY_MISMATCH", name)
    observed_payload = [
        dict(item) for item in _file_inventory(root) if item["path"] != SIDECAR_NAME
    ]
    if decoded["payload_inventory"] != observed_payload:
        raise _BuildFailure("GLUE_PAYLOAD_INVENTORY_MISMATCH", str(root))
    return dict(decoded)


def _validate_saved_dataset(
    *,
    dependencies: _Dependencies,
    root: Path,
    requirement: Mapping[str, Any],
    sidecar_base: Mapping[str, Any],
    fingerprints: Mapping[str, str],
) -> None:
    before = _stat_snapshot(root)
    _load_and_validate_sidecar(root, sidecar_base)
    loader = getattr(dependencies.datasets, "load_from_disk", None)
    if not callable(loader):
        raise _BuildFailure("DATASETS_LOAD_FROM_DISK_UNAVAILABLE", "datasets")
    loaded = loader(str(root))
    keys = getattr(loaded, "keys", None)
    if not callable(keys):
        raise _BuildFailure("DATASET_DICT_INVALID", str(root))
    expected_splits = tuple(requirement["preprocessing"]["derived_splits"])
    if set(keys()) != set(expected_splits):
        raise _BuildFailure("DATASET_SPLITS_MISMATCH", str(root))
    label_count = len(requirement["label_mapping"])
    for split in expected_splits:
        persisted = _persisted_dataset_fingerprint(root, split)
        if persisted != fingerprints[split]:
            raise _BuildFailure("DATASET_FINGERPRINT_MISMATCH", split)
        try:
            dataset = loaded[split]
        except (KeyError, TypeError) as error:
            raise _BuildFailure("DATASET_SPLIT_MISSING", split) from error
        _validate_dataset_split(
            dataset,
            split=split,
            expected_rows=requirement["split_counts"][split],
            expected_fingerprint=fingerprints[split],
            label_count=label_count,
            check_loaded_fingerprint=False,
        )
    if _stat_snapshot(root) != before:
        raise _BuildFailure("DATASET_VALIDATION_MUTATED_TARGET", str(root))


def _persisted_dataset_fingerprint(root: Path, split: str) -> str:
    state = root / split / "state.json"
    _regular_file(state, code="GLUE_STATE_MISSING_OR_INVALID")
    decoded = ensure_json_object(
        loads_strict_json(state.read_bytes()),
        field="dataset state",
    )
    fingerprint = decoded.get("_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise _BuildFailure("GLUE_STATE_FINGERPRINT_INVALID", split)
    return fingerprint


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    _file_inventory(root)
    directories: list[Path] = []
    for current_text, directory_names, file_names in os.walk(
        root, topdown=False, followlinks=False
    ):
        current = Path(current_text)
        directories.append(current)
        for name in file_names:
            path = current / name
            _regular_file(path, code="DATASET_FILE_INVALID")
            # Windows' ``_commit`` rejects a read-only descriptor; staging is
            # builder-owned, so opening read/write solely for ``fsync`` is safe.
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        for name in directory_names:
            if _is_link_like(current / name):
                raise _BuildFailure("DATASET_LINK_FORBIDDEN", str(current / name))
    for directory in directories:
        _fsync_directory(directory)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise _BuildFailure("STAGING_TARGET_CROSS_DEVICE", str(source))
    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform.startswith("linux"):
        import ctypes

        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise _BuildFailure("ATOMIC_NOREPLACE_UNAVAILABLE", sys.platform)
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        raise OSError(error_number, os.strerror(error_number), target)
    if sys.platform == "darwin":
        import ctypes

        library = ctypes.CDLL(None, use_errno=True)
        renamex = getattr(library, "renamex_np", None)
        if renamex is None:
            raise _BuildFailure("ATOMIC_NOREPLACE_UNAVAILABLE", sys.platform)
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        result = renamex(os.fsencode(source), os.fsencode(target), 0x00000004)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        raise OSError(error_number, os.strerror(error_number), target)
    raise _BuildFailure("ATOMIC_NOREPLACE_UNAVAILABLE", sys.platform)


def _discard_owned_staging(staging: Path, tmp_root: Path) -> None:
    if staging.parent != tmp_root or not staging.name.startswith("glue-derived-"):
        raise RuntimeError("REFUSING_TO_REMOVE_UNOWNED_STAGING")
    if staging.exists():
        shutil.rmtree(staging)
    _fsync_directory(tmp_root)


@contextmanager
def _offline_socket_guard(cache_root: Path) -> Iterator[_NetworkCounter]:
    counter = _NetworkCounter()
    loopback_sockets: set[int] = set()

    def _external(*_args: Any, **_kwargs: Any) -> Any:
        counter.attempts += 1
        raise _NetworkEgressBlocked("NETWORK_EGRESS_BLOCKED")

    def _loopback_host(host: object) -> bool:
        if host is None:
            return True
        if not isinstance(host, str):
            return False
        lowered = host.casefold()
        if lowered in {"", "localhost", "ip6-localhost", "ip6-loopback"}:
            return True
        try:
            return ipaddress.ip_address(lowered).is_loopback
        except ValueError:
            return False

    def _loopback_endpoint(address: object) -> bool:
        if isinstance(address, tuple) and address:
            return _loopback_host(address[0])
        if isinstance(address, str):
            # A bare string is an AF_UNIX socket path: purely local.
            return True
        return False

    def _peer_loopback(instance: socket.socket) -> bool:
        try:
            return _loopback_endpoint(instance.getpeername())
        except OSError:
            return False

    def _wrap_connect(original: Any) -> Any:
        def connect(
            instance: socket.socket,
            address: object,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if _loopback_endpoint(address):
                result = original(instance, address, *args, **kwargs)
                if _peer_loopback(instance):
                    loopback_sockets.add(instance.fileno())
                return result
            return _external()

        return connect

    def _wrap_send(original: Any) -> Any:
        def send(instance: socket.socket, *args: Any, **kwargs: Any) -> Any:
            if instance.fileno() in loopback_sockets or _peer_loopback(instance):
                return original(instance, *args, **kwargs)
            return _external()

        return send

    def _wrap_sendto(original: Any) -> Any:
        def sendto(instance: socket.socket, *args: Any, **kwargs: Any) -> Any:
            address = args[1] if len(args) > 1 else kwargs.get("address")
            if _loopback_endpoint(address):
                return original(instance, *args, **kwargs)
            return _external()

        return sendto

    def _wrap_resolver(original: Any) -> Any:
        def resolve(*args: Any, **kwargs: Any) -> Any:
            host = args[0] if args else kwargs.get("host")
            if _loopback_host(host):
                return original(*args, **kwargs)
            return _external()

        return resolve

    def _wrap_byaddr(original: Any) -> Any:
        def byaddr(*args: Any, **kwargs: Any) -> Any:
            address = args[0] if args else kwargs.get("ip_address")
            if _loopback_host(address):
                return original(*args, **kwargs)
            return _external()

        return byaddr

    socket_methods = (
        "connect",
        "connect_ex",
        "send",
        "sendall",
        "sendto",
        "sendmsg",
    )
    module_functions = (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
    )
    with _PROCESS_OFFLINE_LOCK:
        previous_environment = {
            name: os.environ.get(name)
            for name in (
                *_OFFLINE_ENVIRONMENT,
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "TRANSFORMERS_CACHE",
                "HUGGINGFACE_HUB_CACHE",
            )
        }
        previous_methods = {
            name: getattr(socket.socket, name)
            for name in socket_methods
            if hasattr(socket.socket, name)
        }
        previous_functions = {
            name: getattr(socket, name)
            for name in module_functions
            if hasattr(socket, name)
        }
        try:
            cache_root.mkdir(parents=True, exist_ok=False)
            os.environ.update(_OFFLINE_ENVIRONMENT)
            os.environ["HF_HOME"] = str(cache_root / "hf-home")
            os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
            os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
            for name in previous_methods:
                original = previous_methods[name]
                if name in {"connect", "connect_ex"}:
                    setattr(socket.socket, name, _wrap_connect(original))
                elif name in {"send", "sendall", "sendmsg"}:
                    setattr(socket.socket, name, _wrap_send(original))
                elif name == "sendto":
                    setattr(socket.socket, name, _wrap_sendto(original))
            for name in previous_functions:
                original = previous_functions[name]
                if name == "gethostbyaddr":
                    setattr(socket, name, _wrap_byaddr(original))
                else:
                    setattr(socket, name, _wrap_resolver(original))
            yield counter
        finally:
            for name, method in previous_methods.items():
                setattr(socket.socket, name, method)
            for name, function in previous_functions.items():
                setattr(socket, name, function)
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def build_glue_derived_dataset(
    data_root: str | Path,
    raw_task_root: str | Path,
    raw_asset_id: str,
    tokenizer_root: str | Path,
    tokenizer_asset_id: str,
    requirement: Mapping[str, Any],
    target_dir: str | Path,
    *,
    tokenizer_requirement: Mapping[str, Any],
    generator_git_commit: str,
) -> GlueDerivedBuildResult:
    """Build or strictly validate one immutable Stage 0 GLUE derived dataset.

    Any failure after staging allocation leaves that exact directory intact and
    raises :class:`GlueDerivedBuildError`.  An existing target is never deleted
    or overwritten; it is accepted only after a complete read-only validation.
    Tokenizer lineage binds only the frozen descriptor inventory supplied via
    ``tokenizer_requirement`` and verifies those files before and after work.
    """

    staging: Path | None = None
    target_display = str(Path(target_dir).absolute())
    task_for_report = (
        requirement.get("task")
        if isinstance(requirement, Mapping) and isinstance(requirement.get("task"), str)
        else None
    )
    network_counter = _NetworkCounter()
    try:
        normalized_requirement = _validate_requirement(requirement)
        try:
            expected_tokenizer_inventory = normalize_tokenizer_descriptor_inventory(
                tokenizer_requirement
            )
        except ValueError as error:
            raise _BuildFailure("TOKENIZER_REQUIREMENT_INVALID", str(error)) from error
        task = normalized_requirement["task"]
        task_for_report = task
        normalized_raw_asset_id = _validate_identity(
            raw_asset_id, field="RAW_ASSET_ID", pattern=_SHA256
        )
        normalized_tokenizer_asset_id = _validate_identity(
            tokenizer_asset_id, field="TOKENIZER_ASSET_ID", pattern=_SHA256
        )
        normalized_commit = _validate_identity(
            generator_git_commit, field="GENERATOR_GIT_COMMIT", pattern=_GIT_COMMIT
        )
        root = _approved_data_root(data_root)
        raw_root = _existing_directory_under_root(
            root, raw_task_root, field="RAW_TASK_ROOT"
        )
        tokenizer_directory = _existing_directory_under_root(
            root, tokenizer_root, field="TOKENIZER_ROOT"
        )
        target = _target_under_root(root, target_dir)
        target_display = str(target)
        if _paths_overlap(raw_root, tokenizer_directory):
            raise _BuildFailure("SOURCE_ROOTS_OVERLAP", str(raw_root))
        if _paths_overlap(raw_root, target) or _paths_overlap(tokenizer_directory, target):
            raise _BuildFailure("TARGET_SOURCE_OVERLAP", str(target))

        target.parent.mkdir(parents=True, exist_ok=True)
        target = _target_under_root(root, target)
        tmp_root = _candidate_under_root(root, "tmp", field="TMP_ROOT")
        if tmp_root.exists() and not tmp_root.is_dir():
            raise _BuildFailure("TMP_ROOT_NOT_DIRECTORY", str(tmp_root))
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_root = _existing_directory_under_root(root, tmp_root, field="TMP_ROOT")
        if _paths_overlap(tmp_root, raw_root) or _paths_overlap(tmp_root, tokenizer_directory):
            raise _BuildFailure("TMP_SOURCE_OVERLAP", str(tmp_root))
        try:
            target.relative_to(tmp_root)
        except ValueError:
            pass
        else:
            raise _BuildFailure("TARGET_INSIDE_TMP_FORBIDDEN", str(target))

        raw_inventory, raw_files_hash = _selected_raw_inventory(
            raw_root, normalized_requirement
        )
        tokenizer_inventory, tokenizer_inventory_hash = _verified_tokenizer_inventory(
            tokenizer_directory,
            expected_tokenizer_inventory,
        )
        preprocessing_hash = glue_preprocessing_config_hash(normalized_requirement)
        requirement_hash = canonical_json_hash(normalized_requirement)
        fingerprints = {
            split: _mapping_fingerprint(
                requirement=normalized_requirement,
                split=split,
                raw_asset_id=normalized_raw_asset_id,
                tokenizer_asset_id=normalized_tokenizer_asset_id,
                tokenizer_descriptor_inventory_hash=tokenizer_inventory_hash,
                generator_git_commit=normalized_commit,
                preprocessing_config_hash=preprocessing_hash,
            )
            for split in normalized_requirement["preprocessing"]["derived_splits"]
        }
        sidecar_base = _base_sidecar(
            requirement=normalized_requirement,
            raw_asset_id=normalized_raw_asset_id,
            tokenizer_asset_id=normalized_tokenizer_asset_id,
            raw_files_hash=raw_files_hash,
            tokenizer_descriptor_inventory=tokenizer_inventory,
            tokenizer_descriptor_inventory_hash=tokenizer_inventory_hash,
            generator_git_commit=normalized_commit,
            preprocessing_config_hash=preprocessing_hash,
            requirement_hash=requirement_hash,
            fingerprints=fingerprints,
        )

        staging = tmp_root / f"glue-derived-{task}-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700, exist_ok=False)
        payload_root = staging / "payload"
        cache_root = staging / "cache"
        status = "built"
        with _offline_socket_guard(cache_root) as guarded_counter:
            network_counter = guarded_counter
            dependencies = _load_dependencies()
            if target.exists() or _is_link_like(target):
                if _is_link_like(target) or not target.is_dir():
                    raise _BuildFailure("TARGET_CONFLICT", str(target))
                _validate_saved_dataset(
                    dependencies=dependencies,
                    root=target,
                    requirement=normalized_requirement,
                    sidecar_base=sidecar_base,
                    fingerprints=fingerprints,
                )
                _reverify_tokenizer_inventory(
                    tokenizer_directory,
                    expected_tokenizer_inventory,
                    tokenizer_inventory,
                    tokenizer_inventory_hash,
                )
                status = "reused"
            else:
                dataset_dict = _build_splits(
                    dependencies=dependencies,
                    raw_root=raw_root,
                    tokenizer_root=tokenizer_directory,
                    cache_root=cache_root,
                    requirement=normalized_requirement,
                    raw_asset_id=normalized_raw_asset_id,
                    tokenizer_asset_id=normalized_tokenizer_asset_id,
                    generator_git_commit=normalized_commit,
                    preprocessing_config_hash=preprocessing_hash,
                    fingerprints=fingerprints,
                )
                saver = getattr(dataset_dict, "save_to_disk", None)
                if not callable(saver):
                    raise _BuildFailure("DATASET_DICT_SAVE_UNAVAILABLE", task)
                saver(str(payload_root), num_proc=1)
                if not payload_root.is_dir() or _is_link_like(payload_root):
                    raise _BuildFailure("DATASET_SAVE_FAILED", str(payload_root))
                _write_sidecar(payload_root, sidecar_base)
                _validate_saved_dataset(
                    dependencies=dependencies,
                    root=payload_root,
                    requirement=normalized_requirement,
                    sidecar_base=sidecar_base,
                    fingerprints=fingerprints,
                )
                after_raw_inventory, after_raw_hash = _selected_raw_inventory(
                    raw_root, normalized_requirement
                )
                if after_raw_inventory != raw_inventory or after_raw_hash != raw_files_hash:
                    raise _BuildFailure("GLUE_RAW_SOURCE_CHANGED_DURING_BUILD", str(raw_root))
                _reverify_tokenizer_inventory(
                    tokenizer_directory,
                    expected_tokenizer_inventory,
                    tokenizer_inventory,
                    tokenizer_inventory_hash,
                )
                if guarded_counter.attempts:
                    raise _BuildFailure(
                        "NETWORK_EGRESS_BLOCKED", str(guarded_counter.attempts)
                    )
                _fsync_tree(payload_root)
                try:
                    _rename_directory_noreplace(payload_root, target)
                    _fsync_directory(target.parent)
                except FileExistsError:
                    if _is_link_like(target) or not target.is_dir():
                        raise _BuildFailure("TARGET_CONFLICT", str(target)) from None
                    _validate_saved_dataset(
                        dependencies=dependencies,
                        root=target,
                        requirement=normalized_requirement,
                        sidecar_base=sidecar_base,
                        fingerprints=fingerprints,
                    )
                    _reverify_tokenizer_inventory(
                        tokenizer_directory,
                        expected_tokenizer_inventory,
                        tokenizer_inventory,
                        tokenizer_inventory_hash,
                    )
                    status = "reused"
            if guarded_counter.attempts:
                raise _BuildFailure(
                    "NETWORK_EGRESS_BLOCKED", str(guarded_counter.attempts)
                )

        inventory = _file_inventory(target)
        _discard_owned_staging(staging, tmp_root)
        staging = None
        result = GlueDerivedBuildResult(
            status=status,
            task=task,
            raw_asset_id=normalized_raw_asset_id,
            tokenizer_asset_id=normalized_tokenizer_asset_id,
            tokenizer_descriptor_inventory=tokenizer_inventory,
            tokenizer_descriptor_inventory_hash=tokenizer_inventory_hash,
            target_path=str(target),
            target_ref=target.relative_to(root).as_posix(),
            generator_git_commit=normalized_commit,
            preprocessing_version=GLUE_PREPROCESSING_VERSION,
            preprocessing_config_hash=preprocessing_hash,
            requirement_hash=requirement_hash,
            derived_splits=tuple(
                normalized_requirement["preprocessing"]["derived_splits"]
            ),
            split_counts={
                split: normalized_requirement["split_counts"][split]
                for split in normalized_requirement["preprocessing"]["derived_splits"]
            },
            map_fingerprints=fingerprints,
            file_inventory=inventory,
            network_attempts=network_counter.attempts,
        )
        return result
    except GlueDerivedBuildError:
        raise
    except Exception as error:
        if network_counter.attempts:
            code = "NETWORK_EGRESS_BLOCKED"
            message = f"blocked socket attempts={network_counter.attempts}"
        elif isinstance(error, _BuildFailure):
            code = error.code
            message = error.message
        elif isinstance(error, DependencyUnavailable):
            code = "DEPENDENCY_UNAVAILABLE"
            message = str(error)
        elif isinstance(error, OSError):
            code = "LOCAL_IO_ERROR"
            message = f"{type(error).__name__}:{error}"
        else:
            code = "GLUE_DERIVED_BUILD_FAILED"
            message = f"{type(error).__name__}:{error}"
        report = GlueDerivedBuildFailureReport(
            code=code,
            message=message,
            task=task_for_report,
            target_path=target_display,
            staging_path=None if staging is None else str(staging),
            network_attempts=network_counter.attempts,
        )
        raise GlueDerivedBuildError(report) from error


# Explicit long-form alias for callers that prefer the Stage name in the API.
build_stage0_glue_derived_dataset = build_glue_derived_dataset


__all__ = [
    "FAILURE_SCHEMA_VERSION",
    "GlueDerivedBuildError",
    "GlueDerivedBuildFailureReport",
    "GlueDerivedBuildResult",
    "MAP_FINGERPRINT_SCHEMA_VERSION",
    "OUTPUT_COLUMNS",
    "RESULT_SCHEMA_VERSION",
    "SIDECAR_NAME",
    "SIDECAR_SCHEMA_VERSION",
    "build_glue_derived_dataset",
    "build_stage0_glue_derived_dataset",
    "normalize_tokenizer_descriptor_inventory",
]
