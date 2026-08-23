"""Strict Stage 2 asset, checkpoint and frozen-data contracts.

S2.3 owns identities and manifests only.  It deliberately does not download
weights or read a dataset directory.  A manifest may therefore describe a
missing checkpoint (``state=blocked``) without turning an incomplete asset into
an input eligible for a formal run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Literal, Mapping, Sequence

from ..contracts.jsonio import JSONValue, canonical_json_hash


ASSET_SCHEMA_VERSION = "stage2-asset-resolution-v1"
CHECKPOINT_SCHEMA_VERSION = "stage2-checkpoint-manifest-v1"
DATA_RANGE_SCHEMA_VERSION = "stage2-data-range-manifest-v1"
MODEL_NAMES = ("pythia-14m", "pythia-31m-deduped")
TRAINING_STAGES = ("initialization", "early", "mid_late")
TARGET_FRACTIONS = {
    "initialization": 0.0,
    "early": 0.01,
    "mid_late": 0.50,
}
FORMAL_CHECKPOINT_SELECTION = {
    ("pythia-14m", "initialization"): (0, "56079904bb80b7f36d3b794089f146e7a4d6efae"),
    ("pythia-14m", "early"): (1000, "5b020995bfc7aee2931b0f35bd70cf7ee8b1db62"),
    ("pythia-14m", "mid_late"): (71000, "6a9156279d41c80dea69f043e25818cb3e596f56"),
    ("pythia-31m-deduped", "initialization"): (0, "73628c85dd9d12d43c07be77ebcf10cef5fd9660"),
    ("pythia-31m-deduped", "early"): (1000, "dd4d3eab2b004272fee4a3d321064fa65e5e1ee6"),
    ("pythia-31m-deduped", "mid_late"): (71000, "aeafbd5e62a3e5cd6e9f4106167f31e7fec47b41"),
}
FORMAL_TOTAL_TRAINING_STEPS = 143000
FORMAL_DATASET_ID = "dbbfeb12bab4027b386bd97d604d8134699e96f79e309cceacff7999a55b5dad"
FORMAL_DATASET_REVISION = "4647773ea142ab1ff5694602fa104bbf49088408"
FORMAL_DATA_MANIFEST_SHA256 = "d53e9365bd2da3aab0ea220d496aa793175ba7690daa2180299940a2bd6ca4c9"
FORMAL_DATA_FILES = {
    "document-00000-of-00020.bin": (30000000000, "1ce355bd2683627d0ff689f8578115cf3df84bd1edf3410e6aca9705d31fc6ea"),
    "document.idx": (1757184042, "1d9fdd760295eb2007a4874440b27c559ca722239fa2814aa8a2ee6724b7852f"),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _revision(value: object, *, field: str = "revision") -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{field} must be an immutable hexadecimal revision")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _positive_int(value: object, *, field: str, zero_ok: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (
        value < 0 if zero_ok else value <= 0
    ):
        raise TypeError(f"{field} must be an integer")
    return value


def _relative_ref(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if "\\" in text or text.startswith("/") or any(
        part in {"", ".", ".."} for part in text.split("/")
    ):
        raise ValueError(f"{field} must be a safe POSIX relative reference")
    return text


@dataclass(frozen=True, slots=True)
class CheckpointFile:
    """One immutable file in a checkpoint or model asset."""

    path: str
    size_bytes: int
    sha256: str
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_ref(self.path, field="path"))
        object.__setattr__(
            self,
            "size_bytes",
            _positive_int(self.size_bytes, field="size_bytes", zero_ok=True),
        )
        object.__setattr__(self, "sha256", _sha(self.sha256, field="sha256"))
        object.__setattr__(self, "role", _text(self.role, field="role"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "role": self.role,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CheckpointFile":
        if set(value) != {"path", "size_bytes", "sha256", "role"}:
            raise ValueError("checkpoint file fields mismatch")
        return cls(
            path=value["path"],  # type: ignore[arg-type]
            size_bytes=value["size_bytes"],  # type: ignore[arg-type]
            sha256=value["sha256"],  # type: ignore[arg-type]
            role=value["role"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """A selected training checkpoint, including blocked/missing states."""

    model_id: str
    training_stage: Literal["initialization", "early", "mid_late"]
    checkpoint_id: str
    training_step: int
    total_training_steps: int
    target_fraction: float
    repository: str
    revision: str | None
    root_ref: str
    state: Literal["ready", "blocked"]
    files: tuple[CheckpointFile, ...] = ()
    manifest_ref: str | None = None
    manifest_sha256: str | None = None
    parameter_registry_hash: str | None = None
    config_sha256: str | None = None
    tokenizer_sha256: str | None = None
    load_status: Literal["not_run", "passed", "blocked"] = "not_run"
    load_evidence_ref: str | None = None
    load_evidence_sha256: str | None = None
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_NAMES:
            raise ValueError(f"unknown Stage 2 model: {self.model_id!r}")
        if self.training_stage not in TRAINING_STAGES:
            raise ValueError(f"unknown training stage: {self.training_stage!r}")
        object.__setattr__(self, "checkpoint_id", _text(self.checkpoint_id, field="checkpoint_id"))
        object.__setattr__(
            self,
            "training_step",
            _positive_int(self.training_step, field="training_step", zero_ok=True),
        )
        object.__setattr__(
            self,
            "total_training_steps",
            _positive_int(self.total_training_steps, field="total_training_steps"),
        )
        if not isinstance(self.target_fraction, (int, float)) or isinstance(
            self.target_fraction, bool
        ) or not 0.0 <= float(self.target_fraction) <= 1.0:
            raise ValueError("target_fraction must be in [0, 1]")
        expected_fraction = TARGET_FRACTIONS[self.training_stage]
        if abs(float(self.target_fraction) - expected_fraction) > 1e-12:
            raise ValueError("target_fraction does not match the frozen training stage")
        object.__setattr__(self, "repository", _text(self.repository, field="repository"))
        if self.revision is not None:
            object.__setattr__(self, "revision", _revision(self.revision))
        object.__setattr__(self, "root_ref", _relative_ref(self.root_ref, field="root_ref"))
        if self.state not in {"ready", "blocked"}:
            raise ValueError("checkpoint state must be ready or blocked")
        files = tuple(self.files)
        if any(not isinstance(item, CheckpointFile) for item in files):
            raise TypeError("files must contain CheckpointFile values")
        if len({item.path for item in files}) != len(files):
            raise ValueError("checkpoint files must be unique")
        object.__setattr__(self, "files", files)
        for name in (
            "manifest_sha256",
            "parameter_registry_hash",
            "config_sha256",
            "tokenizer_sha256",
            "load_evidence_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _sha(value, field=name)
        for name in ("manifest_ref", "load_evidence_ref", "missing_reason"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, field=name))
        if self.state == "ready":
            if self.revision is None or not files or self.missing_reason is not None:
                raise ValueError("ready checkpoint requires revision, files and no missing_reason")
            if self.load_status != "passed":
                raise ValueError("ready checkpoint requires passed offline load evidence")
            if self.load_evidence_ref is None or self.load_evidence_sha256 is None:
                raise ValueError("ready checkpoint requires load evidence reference and hash")
            if (
                self.manifest_ref is None
                or self.manifest_sha256 is None
                or self.parameter_registry_hash is None
                or self.config_sha256 is None
                or self.tokenizer_sha256 is None
            ):
                raise ValueError(
                    "ready checkpoint requires manifest, registry, config and tokenizer hashes"
                )
            roles = {item.role for item in files}
            if not {"weights", "config", "tokenizer"}.issubset(roles):
                raise ValueError(
                    "ready checkpoint files require weights, config and tokenizer roles"
                )
        else:
            if self.missing_reason is None:
                raise ValueError("blocked checkpoint requires missing_reason")
            if self.load_status not in {"not_run", "blocked"}:
                raise ValueError("blocked checkpoint cannot have passed load status")

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_id": self.model_id,
            "training_stage": self.training_stage,
            "checkpoint_id": self.checkpoint_id,
            "training_step": self.training_step,
            "total_training_steps": self.total_training_steps,
            "target_fraction": float(self.target_fraction),
            "repository": self.repository,
            "revision": self.revision,
            "root_ref": self.root_ref,
            "state": self.state,
            "files": [item.to_dict() for item in self.files],
            "manifest_ref": self.manifest_ref,
            "manifest_sha256": self.manifest_sha256,
            "parameter_registry_hash": self.parameter_registry_hash,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "load_status": self.load_status,
            "load_evidence_ref": self.load_evidence_ref,
            "load_evidence_sha256": self.load_evidence_sha256,
            "missing_reason": self.missing_reason,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CheckpointRecord":
        required = {
            "schema_version", "model_id", "training_stage", "checkpoint_id",
            "training_step", "total_training_steps", "target_fraction", "repository",
            "revision", "root_ref", "state", "files", "manifest_ref",
            "manifest_sha256", "parameter_registry_hash", "config_sha256",
            "tokenizer_sha256", "load_status", "load_evidence_ref",
            "load_evidence_sha256", "missing_reason",
        }
        if set(value) != required or value["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint manifest fields or schema mismatch")
        files = value["files"]
        if not isinstance(files, list) or not all(isinstance(item, Mapping) for item in files):
            raise TypeError("checkpoint files must be an object array")
        return cls(
            model_id=value["model_id"],  # type: ignore[arg-type]
            training_stage=value["training_stage"],  # type: ignore[arg-type]
            checkpoint_id=value["checkpoint_id"],  # type: ignore[arg-type]
            training_step=value["training_step"],  # type: ignore[arg-type]
            total_training_steps=value["total_training_steps"],  # type: ignore[arg-type]
            target_fraction=value["target_fraction"],  # type: ignore[arg-type]
            repository=value["repository"],  # type: ignore[arg-type]
            revision=value["revision"],  # type: ignore[arg-type]
            root_ref=value["root_ref"],  # type: ignore[arg-type]
            state=value["state"],  # type: ignore[arg-type]
            files=tuple(CheckpointFile.from_mapping(item) for item in files),
            manifest_ref=value["manifest_ref"],  # type: ignore[arg-type]
            manifest_sha256=value["manifest_sha256"],  # type: ignore[arg-type]
            parameter_registry_hash=value["parameter_registry_hash"],  # type: ignore[arg-type]
            config_sha256=value["config_sha256"],  # type: ignore[arg-type]
            tokenizer_sha256=value["tokenizer_sha256"],  # type: ignore[arg-type]
            load_status=value["load_status"],  # type: ignore[arg-type]
            load_evidence_ref=value["load_evidence_ref"],  # type: ignore[arg-type]
            load_evidence_sha256=value["load_evidence_sha256"],  # type: ignore[arg-type]
            missing_reason=value["missing_reason"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DataFile:
    path: str
    size_bytes: int
    sha256: str
    role: Literal["token_shard", "index"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_ref(self.path, field="data file path"))
        object.__setattr__(self, "size_bytes", _positive_int(self.size_bytes, field="data file size", zero_ok=True))
        object.__setattr__(self, "sha256", _sha(self.sha256, field="data file sha256"))
        if self.role not in {"token_shard", "index"}:
            raise ValueError("data file role must be token_shard or index")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256, "role": self.role}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DataFile":
        if set(value) != {"path", "size_bytes", "sha256", "role"}:
            raise ValueError("data file fields mismatch")
        return cls(value["path"], value["size_bytes"], value["sha256"], value["role"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DataRangeManifest:
    """The exact Stage 2 empirical universe and its two-file allowlist."""

    dataset_id: str
    revision: str
    manifest_ref: str
    manifest_sha256: str
    files: tuple[DataFile, ...]
    sample_id_min: int = 0
    sample_id_max_exclusive: int = 524_288
    source_tokens_per_record: int = 2049
    input_sequence_length: int = 2048
    effective_target_tokens: int = 2048
    reader_schema: str = "pythia-mmap-v1"
    sampling_design: str = "uniform_with_replacement"
    non_overlapping_windows: bool = True
    excluded_objects: tuple[str, ...] = (
        "document-00001-of-00020.bin",
        "document-00002-of-00020.bin",
        "document-00003-of-00020.bin",
        "document-00004-of-00020.bin",
        "document-00005-of-00020.bin.part",
        "*.lock",
        "*.meta",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _text(self.dataset_id, field="dataset_id"))
        object.__setattr__(self, "revision", _revision(self.revision))
        object.__setattr__(self, "manifest_ref", _relative_ref(self.manifest_ref, field="manifest_ref"))
        object.__setattr__(self, "manifest_sha256", _sha(self.manifest_sha256, field="manifest_sha256"))
        files = tuple(self.files)
        if len(files) != 2 or {item.role for item in files} != {"token_shard", "index"}:
            raise ValueError("Stage 2 data range requires exactly one shard and one index")
        expected = {"document-00000-of-00020.bin", "document.idx"}
        if {item.path for item in files} != expected:
            raise ValueError("Stage 2 allowlist must contain only shard 0 and document.idx")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "sample_id_min", _positive_int(self.sample_id_min, field="sample_id_min", zero_ok=True))
        object.__setattr__(self, "sample_id_max_exclusive", _positive_int(self.sample_id_max_exclusive, field="sample_id_max_exclusive"))
        if self.sample_id_min >= self.sample_id_max_exclusive:
            raise ValueError("sample id interval must be non-empty")
        for name in ("source_tokens_per_record", "input_sequence_length", "effective_target_tokens"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), field=name))
        object.__setattr__(self, "reader_schema", _text(self.reader_schema, field="reader_schema"))
        if self.sampling_design != "uniform_with_replacement":
            raise ValueError("Stage 2 data range must use uniform_with_replacement")
        if self.non_overlapping_windows is not True:
            raise ValueError("non_overlapping_windows must be true")
        if any("document-00000-of-00020.bin" in item or "document.idx" in item for item in self.excluded_objects):
            raise ValueError("excluded_objects must not exclude allowlisted files")

    @property
    def sample_count(self) -> int:
        return self.sample_id_max_exclusive - self.sample_id_min

    @property
    def digest(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {
            "schema_version": DATA_RANGE_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "manifest_ref": self.manifest_ref,
            "manifest_sha256": self.manifest_sha256,
            "files": [item.to_dict() for item in self.files],
            "sample_id_min": self.sample_id_min,
            "sample_id_max_exclusive": self.sample_id_max_exclusive,
            "sample_count": self.sample_count,
            "source_tokens_per_record": self.source_tokens_per_record,
            "input_sequence_length": self.input_sequence_length,
            "effective_target_tokens": self.effective_target_tokens,
            "reader_schema": self.reader_schema,
            "sampling_design": self.sampling_design,
            "non_overlapping_windows": self.non_overlapping_windows,
            "excluded_objects": list(self.excluded_objects),
        }
        if include_hash:
            result["data_range_hash"] = self.digest
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DataRangeManifest":
        required = {
            "schema_version", "dataset_id", "revision", "manifest_ref", "manifest_sha256",
            "files", "sample_id_min", "sample_id_max_exclusive", "sample_count",
            "source_tokens_per_record", "input_sequence_length", "effective_target_tokens",
            "reader_schema", "sampling_design", "non_overlapping_windows", "excluded_objects",
            "data_range_hash",
        }
        if set(value) != required or value["schema_version"] != DATA_RANGE_SCHEMA_VERSION:
            raise ValueError("data range manifest fields or schema mismatch")
        files = value["files"]
        excluded = value["excluded_objects"]
        if not isinstance(files, list) or not all(isinstance(item, Mapping) for item in files):
            raise TypeError("data range files must be an object array")
        if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
            raise TypeError("excluded_objects must be a string array")
        result = cls(
            dataset_id=value["dataset_id"], revision=value["revision"], manifest_ref=value["manifest_ref"],
            manifest_sha256=value["manifest_sha256"], files=tuple(DataFile.from_mapping(item) for item in files),
            sample_id_min=value["sample_id_min"], sample_id_max_exclusive=value["sample_id_max_exclusive"],
            source_tokens_per_record=value["source_tokens_per_record"], input_sequence_length=value["input_sequence_length"],
            effective_target_tokens=value["effective_target_tokens"], reader_schema=value["reader_schema"],
            sampling_design=value["sampling_design"], non_overlapping_windows=value["non_overlapping_windows"],
            excluded_objects=tuple(excluded),
        )
        if value["sample_count"] != result.sample_count:
            raise ValueError("sample_count does not match interval")
        if value["data_range_hash"] != result.digest:
            raise ValueError("data_range_hash does not match content")
        return result


@dataclass(frozen=True, slots=True)
class ManifestRepair:
    target: str
    original_sha256: str
    canonical_sha256: str
    original_size_bytes: int
    canonical_size_bytes: int
    original_encoding: str = "utf-8-bom"
    canonical_encoding: str = "utf-8"
    replaced_atomically: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _relative_ref(self.target, field="repair target"))
        _sha(self.original_sha256, field="original_sha256")
        _sha(self.canonical_sha256, field="canonical_sha256")
        if self.original_encoding != "utf-8-bom" or self.canonical_encoding != "utf-8":
            raise ValueError("manifest repair encoding contract mismatch")
        if self.replaced_atomically is not True:
            raise ValueError("manifest repair must be atomically published")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "target": self.target, "original_sha256": self.original_sha256,
            "canonical_sha256": self.canonical_sha256, "original_size_bytes": self.original_size_bytes,
            "canonical_size_bytes": self.canonical_size_bytes, "original_encoding": self.original_encoding,
            "canonical_encoding": self.canonical_encoding, "replaced_atomically": self.replaced_atomically,
        }


@dataclass(frozen=True, slots=True)
class AssetResolutionManifest:
    """Combined S2.3 asset matrix and data-range qualification summary."""

    scope: Literal["local_fixture", "formal"]
    checkpoints: tuple[CheckpointRecord, ...]
    data_range: DataRangeManifest
    producer_commit: str
    execution_commit: str
    consumer_commit: str | None = None
    manifest_repairs: tuple[ManifestRepair, ...] = ()
    excluded_active_objects: tuple[str, ...] = ("*.part", "*.lock", "*.meta")

    def __post_init__(self) -> None:
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("scope must be local_fixture or formal")
        records = tuple(self.checkpoints)
        if len({(item.model_id, item.training_stage) for item in records}) != len(records):
            raise ValueError("checkpoint matrix contains duplicate model/stage")
        expected = {(model, stage) for model in MODEL_NAMES for stage in TRAINING_STAGES}
        if {(item.model_id, item.training_stage) for item in records} != expected:
            raise ValueError("checkpoint matrix must contain both models and all three stages")
        object.__setattr__(self, "checkpoints", tuple(sorted(records, key=lambda item: (MODEL_NAMES.index(item.model_id), TRAINING_STAGES.index(item.training_stage)))))
        object.__setattr__(self, "producer_commit", _revision(self.producer_commit, field="producer_commit"))
        object.__setattr__(self, "execution_commit", _revision(self.execution_commit, field="execution_commit"))
        if self.consumer_commit is not None:
            object.__setattr__(self, "consumer_commit", _revision(self.consumer_commit, field="consumer_commit"))
        if any(not item or item.startswith("/") or "\\" in item for item in self.excluded_active_objects):
            raise ValueError("excluded active objects must be descriptive relative names")

    @property
    def checkpoint_matrix_complete(self) -> bool:
        return all(item.ready for item in self.checkpoints)

    @property
    def status(self) -> Literal["READY", "BLOCKED"]:
        return "READY" if self.checkpoint_matrix_complete else "BLOCKED"

    @property
    def digest(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {
            "schema_version": ASSET_SCHEMA_VERSION,
            "scope": self.scope,
            "status": self.status,
            "checkpoint_matrix_complete": self.checkpoint_matrix_complete,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "data_range": self.data_range.to_dict(),
            "manifest_repairs": [item.to_dict() for item in self.manifest_repairs],
            "excluded_active_objects": list(self.excluded_active_objects),
            "producer_commit": self.producer_commit,
            "execution_commit": self.execution_commit,
            "consumer_commit": self.consumer_commit,
        }
        if include_hash:
            result["asset_resolution_hash"] = self.digest
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AssetResolutionManifest":
        required = {
            "schema_version", "scope", "status", "checkpoint_matrix_complete", "checkpoints",
            "data_range", "manifest_repairs", "excluded_active_objects", "producer_commit",
            "execution_commit", "consumer_commit", "asset_resolution_hash",
        }
        if set(value) != required or value["schema_version"] != ASSET_SCHEMA_VERSION:
            raise ValueError("asset resolution manifest fields or schema mismatch")
        checkpoints = value["checkpoints"]
        repairs = value["manifest_repairs"]
        excluded = value["excluded_active_objects"]
        if not isinstance(checkpoints, list) or not all(isinstance(item, Mapping) for item in checkpoints):
            raise TypeError("checkpoints must be an object array")
        if not isinstance(value["data_range"], Mapping):
            raise TypeError("data_range must be an object")
        if not isinstance(repairs, list) or not all(isinstance(item, Mapping) for item in repairs):
            raise TypeError("manifest_repairs must be an object array")
        if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
            raise TypeError("excluded_active_objects must be a string array")
        result = cls(
            scope=value["scope"],
            checkpoints=tuple(CheckpointRecord.from_mapping(item) for item in checkpoints),
            data_range=DataRangeManifest.from_mapping(value["data_range"]),
            producer_commit=value["producer_commit"], execution_commit=value["execution_commit"],
            consumer_commit=value["consumer_commit"],
            manifest_repairs=tuple(ManifestRepair(**item) for item in repairs),
            excluded_active_objects=tuple(excluded),
        )  # type: ignore[arg-type]
        if value["status"] != result.status or value["checkpoint_matrix_complete"] != result.checkpoint_matrix_complete:
            raise ValueError("asset resolution status does not match checkpoint states")
        if value["asset_resolution_hash"] != result.digest:
            raise ValueError("asset_resolution_hash does not match content")
        return result


def validate_formal_asset_identity(manifest: AssetResolutionManifest) -> None:
    """Reject a structurally valid formal manifest with scientific drift."""

    if manifest.scope != "formal" or manifest.status != "READY":
        raise ValueError("formal asset manifest must be READY")
    observed = {
        (item.model_id, item.training_stage): (item.training_step, item.revision)
        for item in manifest.checkpoints
    }
    if observed != FORMAL_CHECKPOINT_SELECTION:
        raise ValueError("formal checkpoint selection/revision drift")
    if any(
        item.total_training_steps != FORMAL_TOTAL_TRAINING_STEPS
        or item.repository != f"EleutherAI/{item.model_id}"
        for item in manifest.checkpoints
    ):
        raise ValueError("formal checkpoint repository/total-step drift")
    data = manifest.data_range
    observed_files = {item.path: (item.size_bytes, item.sha256) for item in data.files}
    if (
        data.dataset_id != FORMAL_DATASET_ID
        or data.revision != FORMAL_DATASET_REVISION
        or data.manifest_sha256 != FORMAL_DATA_MANIFEST_SHA256
        or observed_files != FORMAL_DATA_FILES
        or (data.sample_id_min, data.sample_id_max_exclusive) != (0, 524_288)
        or (data.source_tokens_per_record, data.input_sequence_length, data.effective_target_tokens)
        != (2049, 2048, 2048)
        or data.sampling_design != "uniform_with_replacement"
        or not data.non_overlapping_windows
    ):
        raise ValueError("formal data range identity drift")

def build_data_range_from_prefix(
    *,
    dataset_id: str,
    revision: str,
    manifest_ref: str,
    manifest_sha256: str,
    shard_sha256: str,
    shard_size_bytes: int,
    index_sha256: str,
    index_size_bytes: int,
) -> DataRangeManifest:
    """Build the frozen prefix manifest without inspecting or globbing a directory."""

    return DataRangeManifest(
        dataset_id=dataset_id,
        revision=revision,
        manifest_ref=manifest_ref,
        manifest_sha256=manifest_sha256,
        files=(
            DataFile("document-00000-of-00020.bin", shard_size_bytes, shard_sha256, "token_shard"),
            DataFile("document.idx", index_size_bytes, index_sha256, "index"),
        ),
    )


def build_checkpoint_matrix(records: Iterable[CheckpointRecord]) -> tuple[CheckpointRecord, ...]:
    """Validate and return the canonical two-model by three-stage ordering."""

    values = tuple(records)
    AssetResolutionManifest(
        scope="local_fixture",
        checkpoints=values,
        data_range=build_data_range_from_prefix(
            dataset_id="fixture", revision="0" * 40, manifest_ref="fixture/data.json",
            manifest_sha256="0" * 64, shard_sha256="1" * 64, shard_size_bytes=1,
            index_sha256="2" * 64, index_size_bytes=1,
        ),
        producer_commit="0" * 40,
        execution_commit="0" * 40,
    )
    return tuple(sorted(values, key=lambda item: (MODEL_NAMES.index(item.model_id), TRAINING_STAGES.index(item.training_stage))))


__all__ = [
    "ASSET_SCHEMA_VERSION", "CHECKPOINT_SCHEMA_VERSION", "DATA_RANGE_SCHEMA_VERSION",
    "MODEL_NAMES", "TRAINING_STAGES", "TARGET_FRACTIONS", "FORMAL_CHECKPOINT_SELECTION",
    "FORMAL_TOTAL_TRAINING_STEPS", "FORMAL_DATASET_ID", "FORMAL_DATASET_REVISION",
    "FORMAL_DATA_MANIFEST_SHA256", "FORMAL_DATA_FILES", "AssetResolutionManifest",
    "CheckpointFile", "CheckpointRecord", "DataFile", "DataRangeManifest", "ManifestRepair",
    "build_checkpoint_matrix", "build_data_range_from_prefix", "validate_formal_asset_identity",
]
