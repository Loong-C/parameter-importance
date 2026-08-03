"""Stage 0 G3 asset identities and frozen Pile/GLUE data budgets.

The requirements artifact is deliberately separate from an asset manifest:
it describes what must exist before acquisition begins, while manifests and
qualification reports describe what was actually admitted.  The loader is
strict JSON (no BOM, duplicate keys, or non-finite values) and validates the
cross-field arithmetic that JSON Schema cannot express.
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


SCHEMA_VERSION: Final = "stage0-asset-requirements-v1"
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT: Final = re.compile(r"^[0-9a-f]{40}$")
_TOP_LEVEL: Final = frozenset(
    {
        "schema_version",
        "artifact_hash",
        "status",
        "created_at",
        "generator_git_commit",
        "source_plan_refs",
        "models",
        "tokenizer",
        "pile",
        "glue",
        "gate_matrix",
        "stage6_route",
    }
)
_MODEL_NAMES: Final = frozenset(
    {
        "pythia-14m-step0",
        "pythia-31m-deduped-step0",
        "pythia-160m-deduped-step0",
        "pythia-160m-deduped-step512",
        "pythia-410m-deduped-step0",
    }
)
_MAIN_PILE_INTERVALS: Final = (
    "train",
    "validation",
    "sampling_universe",
    "probe",
    "reserve",
)
_EXPECTED_PILE_WORKLOAD_MATRIX: Final = frozenset(
    {
        (1, "train"),
        (1, "validation"),
        (2, "sampling_universe"),
        (3, "probe"),
        (4, "train"),
        (4, "validation"),
        (5, "train"),
        (5, "validation"),
    }
)
_PILE_WORKLOAD_FIELDS: Final = frozenset(
    {
        "stage",
        "split",
        "max_steps",
        "global_batch_size",
        "required_unique_records",
        "required_target_tokens",
    }
)
_PILE_REFERENCE_ORACLE_REF: Final = "manifests/batch-viewer-comparison.json"
_PILE_REFERENCE_SOURCE_REF: Final = (
    "source/pythia-a19eecb807ec2c79a39ebf18108816e6ffffc1d5"
)
_LEGACY_31M_MANIFEST_REFS: Final = (
    "models/pythia-31m-deduped-step0/model-manifest.json",
    "manifests/pythia-31m-deduped-step0.json",
)
_FORMAL_31M_MANIFEST_REF: Final = "manifests/model/pythia-31m.json"
_EXPECTED_GATE_MATRIX: Final = {
    "stage0.G3-S1": {
        "model_names": ["pythia-14m-step0"],
        "require_tokenizer": True,
        "pile_intervals": ["debug"],
        "glue_tasks": [],
        "require_glue_derived": False,
    },
    "stage0.G3-S2": {
        "model_names": ["pythia-31m-deduped-step0"],
        "require_tokenizer": True,
        "pile_intervals": ["sampling_universe"],
        "glue_tasks": [],
        "require_glue_derived": False,
    },
    "stage0.G3-S4": {
        "model_names": [
            "pythia-160m-deduped-step0",
            "pythia-160m-deduped-step512",
        ],
        "require_tokenizer": True,
        "pile_intervals": ["train", "validation"],
        "glue_tasks": ["sst2"],
        "require_glue_derived": True,
    },
    "stage0.G3-S5": {
        "model_names": ["pythia-410m-deduped-step0"],
        "require_tokenizer": True,
        "pile_intervals": ["train", "validation"],
        "glue_tasks": [],
        "require_glue_derived": False,
    },
    "stage0.G3-S6": {
        "model_names": ["pythia-410m-deduped-step0"],
        "require_tokenizer": True,
        "pile_intervals": [],
        "glue_tasks": ["sst2", "mnli", "rte"],
        "require_glue_derived": True,
    },
}


class AssetRequirementsError(ValueError):
    """Raised when the Stage 0 asset freeze is incomplete or inconsistent."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetRequirementsError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AssetRequirementsError(f"{field} keys must be strings")
    return dict(value)


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssetRequirementsError(f"{field} must be an array")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AssetRequirementsError(f"{field} must be an integer >= {minimum}")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AssetRequirementsError(f"{field} must be normalized non-empty text")
    if any(ord(character) < 32 for character in value):
        raise AssetRequirementsError(f"{field} contains a control character")
    return value


def _digest(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if _SHA256.fullmatch(text) is None:
        raise AssetRequirementsError(f"{field} must be a lowercase SHA-256")
    return text


def _validate_file(value: Any, *, field: str) -> dict[str, Any]:
    item = _mapping(value, field=field)
    required = {"path", "size_bytes", "sha256", "role"}
    if set(item) != required:
        raise AssetRequirementsError(f"{field} fields must be {sorted(required)}")
    validate_asset_path(_text(item["path"], field=f"{field}.path"))
    _integer(item["size_bytes"], field=f"{field}.size_bytes")
    _digest(item["sha256"], field=f"{field}.sha256")
    _text(item["role"], field=f"{field}.role")
    return item


def _validate_model(value: Any, *, index: int) -> dict[str, Any]:
    model = _mapping(value, field=f"models[{index}]")
    name = _text(model.get("name"), field=f"models[{index}].name")
    revision = _text(model.get("revision"), field=f"models[{index}].revision")
    if _GIT_COMMIT.fullmatch(revision) is None:
        raise AssetRequirementsError(f"models[{index}].revision must be a Git commit")
    parameter_count = _integer(
        model.get("parameter_count"),
        field=f"models[{index}].parameter_count",
        minimum=1,
    )
    tensor_count = _integer(
        model.get("tensor_count"),
        field=f"models[{index}].tensor_count",
        minimum=1,
    )
    if tensor_count > parameter_count:
        raise AssetRequirementsError(f"models[{index}] tensor count exceeds parameters")
    dtype_counts = _mapping(
        model.get("dtype_counts"),
        field=f"models[{index}].dtype_counts",
    )
    if not dtype_counts or sum(
        _integer(count, field=f"models[{index}].dtype_counts.{dtype}", minimum=1)
        for dtype, count in dtype_counts.items()
    ) != parameter_count:
        raise AssetRequirementsError(
            f"models[{index}] dtype counts must sum to parameter_count"
        )
    files = [
        _validate_file(item, field=f"models[{index}].files[{file_index}]")
        for file_index, item in enumerate(
            _list(model.get("files"), field=f"models[{index}].files")
        )
    ]
    roles = {item["role"] for item in files}
    if roles != {"config", "weights"} or len({item["path"] for item in files}) != 2:
        raise AssetRequirementsError(
            f"models[{index}] must contain exactly config and weights"
        )
    checkpoint = _text(
        model.get("checkpoint"),
        field=f"models[{index}].checkpoint",
    )
    initialization_kind = _text(
        model.get("initialization_kind"),
        field=f"models[{index}].initialization_kind",
    )
    if (checkpoint == "step0") != (initialization_kind == "base_initialization"):
        raise AssetRequirementsError(
            f"models[{index}] step0/base-initialization identity mismatch"
        )
    if _integer(
        model.get("max_position_embeddings"),
        field=f"models[{index}].max_position_embeddings",
        minimum=1,
    ) != 2048:
        raise AssetRequirementsError(f"models[{index}] context must be 2048")
    if name not in _MODEL_NAMES:
        raise AssetRequirementsError(f"unexpected Stage 0 model: {name}")
    diagnostic = model.get("legacy_manifest_diagnostic")
    if name == "pythia-31m-deduped-step0":
        legacy = _mapping(
            diagnostic,
            field=f"models[{index}].legacy_manifest_diagnostic",
        )
        required_legacy_fields = {
            "refs",
            "size_bytes",
            "sha256",
            "condition",
            "replacement_manifest_ref",
        }
        if set(legacy) != required_legacy_fields:
            raise AssetRequirementsError(
                f"models[{index}].legacy_manifest_diagnostic fields must be "
                f"{sorted(required_legacy_fields)}"
            )
        refs = _list(
            legacy["refs"],
            field=f"models[{index}].legacy_manifest_diagnostic.refs",
        )
        if tuple(refs) != _LEGACY_31M_MANIFEST_REFS:
            raise AssetRequirementsError("31M legacy manifest references drifted")
        for ref_index, ref in enumerate(refs):
            validate_asset_path(
                _text(
                    ref,
                    field=(
                        f"models[{index}].legacy_manifest_diagnostic.refs"
                        f"[{ref_index}]"
                    ),
                )
            )
        _integer(
            legacy["size_bytes"],
            field=f"models[{index}].legacy_manifest_diagnostic.size_bytes",
            minimum=1,
        )
        _digest(
            legacy["sha256"],
            field=f"models[{index}].legacy_manifest_diagnostic.sha256",
        )
        if legacy["condition"] != "utf8_bom_strict_json_rejected":
            raise AssetRequirementsError("31M legacy manifest condition drifted")
        replacement_ref = validate_asset_path(
            _text(
                legacy["replacement_manifest_ref"],
                field=(
                    f"models[{index}].legacy_manifest_diagnostic"
                    ".replacement_manifest_ref"
                ),
            )
        )
        if replacement_ref != _FORMAL_31M_MANIFEST_REF or replacement_ref in refs:
            raise AssetRequirementsError("31M replacement manifest reference drifted")
    elif "legacy_manifest_diagnostic" in model:
        raise AssetRequirementsError(
            "legacy manifest diagnostic is only valid for the 31M model"
        )
    return model


def _validate_tokenizer(value: Any) -> None:
    tokenizer = _mapping(value, field="tokenizer")
    if tokenizer.get("name") != "pythia-tokenizer":
        raise AssetRequirementsError("tokenizer.name must be pythia-tokenizer")
    if tokenizer.get("repository") != "EleutherAI/pythia-410m-deduped":
        raise AssetRequirementsError("tokenizer repository must match the 410M base")
    revision = _text(tokenizer.get("revision"), field="tokenizer.revision")
    if _GIT_COMMIT.fullmatch(revision) is None:
        raise AssetRequirementsError("tokenizer.revision must be a Git commit")
    expected_scalars = {
        "tokenizer_class": "GPTNeoXTokenizerFast",
        "vocab_size": 50254,
        "token_count_with_added_tokens": 50277,
        "vocab_mapping_sha256": (
            "728db3c8825c64b8de98049c042221ba64b16a89348ada66f8d63c3d12dea84f"
        ),
    }
    for field, expected in expected_scalars.items():
        if tokenizer.get(field) != expected:
            raise AssetRequirementsError(f"tokenizer.{field} drifted")
    if tokenizer.get("special_token_ids") != {"bos": 0, "eos": 0, "unk": 0}:
        raise AssetRequirementsError("tokenizer special-token identity drifted")
    if tokenizer.get("glue_padding_token") != {
        "token": "<|padding|>",
        "token_id": 1,
        "policy": "explicit_existing_added_token",
    }:
        raise AssetRequirementsError("tokenizer GLUE padding policy drifted")
    files = _list(tokenizer.get("files"), field="tokenizer.files")
    validated = [
        _validate_file(item, field=f"tokenizer.files[{index}]")
        for index, item in enumerate(files)
    ]
    if {item["role"] for item in validated} != {
        "special_tokens",
        "tokenizer_model",
        "tokenizer_config",
    }:
        raise AssetRequirementsError("tokenizer file roles are incomplete")


def _validate_pile(value: Any) -> None:
    pile = _mapping(value, field="pile")
    revision = _text(pile.get("revision"), field="pile.revision")
    if _GIT_COMMIT.fullmatch(revision) is None:
        raise AssetRequirementsError("pile.revision must be a Git commit")
    _validate_file(pile.get("index"), field="pile.index")
    shards = _list(pile.get("selected_shards"), field="pile.selected_shards")
    previous_stop = 0
    for index, raw in enumerate(shards):
        shard = _mapping(raw, field=f"pile.selected_shards[{index}]")
        base = {key: shard.get(key) for key in ("path", "size_bytes", "sha256", "role")}
        _validate_file(base, field=f"pile.selected_shards[{index}]")
        ordinal = _integer(
            shard.get("ordinal"),
            field=f"pile.selected_shards[{index}].ordinal",
        )
        start = _integer(
            shard.get("byte_start"),
            field=f"pile.selected_shards[{index}].byte_start",
        )
        stop = _integer(
            shard.get("byte_stop"),
            field=f"pile.selected_shards[{index}].byte_stop",
            minimum=1,
        )
        if ordinal != index or start != previous_stop or stop - start != shard["size_bytes"]:
            raise AssetRequirementsError("Pile shards must be ordinal and byte contiguous")
        previous_stop = stop
    if not shards:
        raise AssetRequirementsError("Pile requires at least one selected shard")

    index_contract = _mapping(pile.get("index_contract"), field="pile.index_contract")
    if index_contract != {
        "magic_hex": "4d4d49444944580000",
        "version": 1,
        "dtype_code": 8,
        "dtype_bytes": 2,
        "sequence_count": 146432000,
        "document_count": 1,
    }:
        raise AssetRequirementsError("Pile MMIDIDX identity drifted")
    causal = _mapping(pile.get("causal_lm_contract"), field="pile.causal_lm_contract")
    expected_causal = {
        "labels_alignment": "pre_shifted_next_token",
        "source_tokens_per_record": 2049,
        "input_sequence_length": 2048,
        "target_sequence_length": 2048,
        "input_slice": [0, 2048],
        "target_slice": [1, 2049],
        "attention_mask_policy": "all_one_for_fixed_full_record",
        "effective_target_tokens_per_record": 2048,
        "loss_adapter_id": "pre-shifted-next-token-cross-entropy-v1",
    }
    if causal != expected_causal:
        raise AssetRequirementsError("Pile causal-LM alignment contract drifted")

    cursor_stop = _integer(
        pile.get("required_cursor_stop"),
        field="pile.required_cursor_stop",
        minimum=1,
    )
    if pile.get("required_raw_tokens") != cursor_stop * 2049:
        raise AssetRequirementsError("Pile raw-token budget arithmetic mismatch")
    if pile.get("required_target_tokens") != cursor_stop * 2048:
        raise AssetRequirementsError("Pile target-token budget arithmetic mismatch")
    if cursor_stop * 2049 * index_contract["dtype_bytes"] > previous_stop:
        raise AssetRequirementsError("selected Pile shards do not cover the frozen cursor")

    intervals = {
        _text(item.get("name"), field=f"pile.cursor_intervals[{index}].name"): item
        for index, raw in enumerate(
            _list(pile.get("cursor_intervals"), field="pile.cursor_intervals")
        )
        for item in [_mapping(raw, field=f"pile.cursor_intervals[{index}]")]
    }
    if set(intervals) != {"debug", *_MAIN_PILE_INTERVALS}:
        raise AssetRequirementsError("Pile cursor interval set is incomplete")
    previous_stop = 0
    interval_lengths: dict[str, int] = {}
    for name in _MAIN_PILE_INTERVALS:
        interval = intervals[name]
        start = _integer(interval.get("start"), field=f"pile.interval.{name}.start")
        stop = _integer(
            interval.get("stop"),
            field=f"pile.interval.{name}.stop",
            minimum=1,
        )
        _text(interval.get("purpose"), field=f"pile.interval.{name}.purpose")
        if start != previous_stop or stop <= start:
            raise AssetRequirementsError("main Pile intervals must form one partition")
        interval_lengths[name] = stop - start
        previous_stop = stop
    if previous_stop != cursor_stop:
        raise AssetRequirementsError("Pile interval partition must end at required_cursor_stop")
    debug = intervals["debug"]
    if debug.get("start") != 0 or not 0 < debug.get("stop", 0) <= intervals["train"]["stop"]:
        raise AssetRequirementsError("Pile debug interval must be a train prefix")

    observed_workloads: set[tuple[int, str]] = set()
    for index, raw in enumerate(_list(pile.get("workloads"), field="pile.workloads")):
        workload = _mapping(raw, field=f"pile.workloads[{index}]")
        if set(workload) != _PILE_WORKLOAD_FIELDS:
            raise AssetRequirementsError(
                f"pile.workloads[{index}] fields must be "
                f"{sorted(_PILE_WORKLOAD_FIELDS)}"
            )
        stage = _integer(
            workload.get("stage"),
            field=f"pile.workloads[{index}].stage",
            minimum=1,
        )
        split = _text(workload.get("split"), field=f"pile.workloads[{index}].split")
        identity = (stage, split)
        if identity not in _EXPECTED_PILE_WORKLOAD_MATRIX:
            raise AssetRequirementsError(
                f"Pile workload {index} has an unexpected stage/split identity"
            )
        if identity in observed_workloads:
            raise AssetRequirementsError(
                f"Pile workload {index} duplicates stage/split {identity!r}"
            )
        observed_workloads.add(identity)
        records = _integer(
            workload.get("required_unique_records"),
            field=f"pile.workloads[{index}].required_unique_records",
            minimum=1,
        )
        if split not in interval_lengths or records > interval_lengths[split]:
            raise AssetRequirementsError(f"Pile workload {index} exceeds split {split}")
        if workload.get("required_target_tokens") != records * 2048:
            raise AssetRequirementsError(f"Pile workload {index} token arithmetic mismatch")
        steps, batch = workload.get("max_steps"), workload.get("global_batch_size")
        if (steps is None) != (batch is None):
            raise AssetRequirementsError(f"Pile workload {index} step/batch fields mismatch")
        if (split == "train") != (steps is not None):
            raise AssetRequirementsError(
                f"Pile workload {index} train/step policy mismatch"
            )
        if steps is not None and (
            _integer(steps, field=f"pile.workloads[{index}].max_steps", minimum=1)
            * _integer(
                batch,
                field=f"pile.workloads[{index}].global_batch_size",
                minimum=1,
            )
            != records
        ):
            raise AssetRequirementsError(f"Pile workload {index} record arithmetic mismatch")
    if observed_workloads != _EXPECTED_PILE_WORKLOAD_MATRIX:
        raise AssetRequirementsError(
            "Pile workload stage/split matrix is incomplete or duplicated"
        )

    reference_reader = _mapping(
        pile.get("reference_reader"),
        field="pile.reference_reader",
    )
    if reference_reader != {
        "repository": "EleutherAI/pythia",
        "revision": "a19eecb807ec2c79a39ebf18108816e6ffffc1d5",
    }:
        raise AssetRequirementsError("Pile official reference reader drifted")
    oracle = _mapping(
        pile.get("reference_reader_oracle"),
        field="pile.reference_reader_oracle",
    )
    required_oracle_fields = {
        "artifact_ref",
        "artifact_sha256",
        "official_source_ref",
    }
    if set(oracle) != required_oracle_fields:
        raise AssetRequirementsError(
            "pile.reference_reader_oracle fields must be "
            f"{sorted(required_oracle_fields)}"
        )
    artifact_ref = validate_asset_path(
        _text(
            oracle["artifact_ref"],
            field="pile.reference_reader_oracle.artifact_ref",
        )
    )
    source_ref = validate_asset_path(
        _text(
            oracle["official_source_ref"],
            field="pile.reference_reader_oracle.official_source_ref",
        )
    )
    if artifact_ref != _PILE_REFERENCE_ORACLE_REF:
        raise AssetRequirementsError("Pile reference-reader oracle ref drifted")
    if source_ref != _PILE_REFERENCE_SOURCE_REF:
        raise AssetRequirementsError("Pile official reader source ref drifted")
    _digest(
        oracle["artifact_sha256"],
        field="pile.reference_reader_oracle.artifact_sha256",
    )
    if _integer(
        pile.get("reference_batch_size"),
        field="pile.reference_batch_size",
        minimum=1,
    ) != 1024:
        raise AssetRequirementsError("Pile reference batch size must be 1024")
    references = _mapping(
        pile.get("reference_batch_sha256"),
        field="pile.reference_batch_sha256",
    )
    if set(references) != {"0", "1", "511"}:
        raise AssetRequirementsError("Pile reference batches must be 0, 1 and 511")
    for key, digest in references.items():
        _digest(digest, field=f"pile.reference_batch_sha256.{key}")
    _digest(
        pile.get("last_required_record_sha256"),
        field="pile.last_required_record_sha256",
    )


def _validate_glue(value: Any) -> None:
    tasks: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_list(value, field="glue")):
        task = _mapping(raw, field=f"glue[{index}]")
        name = _text(task.get("task"), field=f"glue[{index}].task")
        if name in tasks:
            raise AssetRequirementsError(f"duplicate GLUE task: {name}")
        tasks[name] = task
        revision = _text(task.get("revision"), field=f"glue[{index}].revision")
        if _GIT_COMMIT.fullmatch(revision) is None:
            raise AssetRequirementsError(f"glue[{index}].revision must be a Git commit")
        raw_files = [
            _validate_file(item, field=f"glue[{index}].raw_files[{file_index}]")
            for file_index, item in enumerate(
                _list(task.get("raw_files"), field=f"glue[{index}].raw_files")
            )
        ]
        if any("/" in item["path"] for item in raw_files):
            raise AssetRequirementsError(
                f"GLUE {name} raw path must be flat and relative to its task asset root"
            )
        split_counts = _mapping(
            task.get("split_counts"),
            field=f"glue[{index}].split_counts",
        )
        if {item["role"] for item in raw_files} != set(split_counts):
            raise AssetRequirementsError(f"GLUE {name} file roles/splits differ")
        labels = _mapping(task.get("label_mapping"), field=f"glue[{index}].label_mapping")
        if sorted(labels.values()) != list(range(len(labels))):
            raise AssetRequirementsError(f"GLUE {name} labels must be contiguous from zero")
        preprocessing = _mapping(
            task.get("preprocessing"),
            field=f"glue[{index}].preprocessing",
        )
        expected = {
            "tokenizer_name": "pythia-tokenizer",
            "max_length": 512,
            "truncation": True,
            "padding": "max_length",
            "pad_token": "<|padding|>",
        }
        if any(preprocessing.get(key) != item for key, item in expected.items()):
            raise AssetRequirementsError(f"GLUE {name} preprocessing drifted")
        derived = _list(
            preprocessing.get("derived_splits"),
            field=f"glue[{index}].preprocessing.derived_splits",
        )
        if not derived or any(split.startswith("test") for split in derived):
            raise AssetRequirementsError(f"GLUE {name} derived splits include unlabeled test")
    if set(tasks) != {"sst2", "mnli", "rte"}:
        raise AssetRequirementsError("GLUE requirements must contain SST-2, MNLI and RTE")


def requirements_artifact_hash(value: Mapping[str, Any]) -> str:
    """Return the canonical hash with the self-referential field removed."""

    payload = deepcopy(dict(value))
    payload.pop("artifact_hash", None)
    return canonical_json_hash(payload)


def validate_stage0_asset_requirements(value: Mapping[str, Any]) -> None:
    artifact = _mapping(value, field="asset requirements")
    missing = _TOP_LEVEL - set(artifact)
    extra = set(artifact) - _TOP_LEVEL
    if missing or extra:
        raise AssetRequirementsError(
            f"asset requirement fields missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if artifact["schema_version"] != SCHEMA_VERSION or artifact["status"] != "FROZEN":
        raise AssetRequirementsError("unsupported asset requirement schema or status")
    declared_hash = _digest(artifact["artifact_hash"], field="artifact_hash")
    if declared_hash != requirements_artifact_hash(artifact):
        raise AssetRequirementsError("asset requirements artifact_hash mismatch")
    created_at = _text(artifact["created_at"], field="created_at")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise AssetRequirementsError("created_at must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise AssetRequirementsError("created_at must include a timezone")
    if _GIT_COMMIT.fullmatch(
        _text(artifact["generator_git_commit"], field="generator_git_commit")
    ) is None:
        raise AssetRequirementsError("generator_git_commit must be a Git commit")
    refs = _list(artifact["source_plan_refs"], field="source_plan_refs")
    if len(refs) != len(set(refs)):
        raise AssetRequirementsError("source_plan_refs must be unique")
    for index, ref in enumerate(refs):
        validate_asset_path(_text(ref, field=f"source_plan_refs[{index}]"))

    models = [
        _validate_model(item, index=index)
        for index, item in enumerate(_list(artifact["models"], field="models"))
    ]
    names = [model["name"] for model in models]
    if len(names) != len(set(names)) or set(names) != _MODEL_NAMES:
        raise AssetRequirementsError("Stage 0 model matrix is incomplete or duplicated")
    _validate_tokenizer(artifact["tokenizer"])
    _validate_pile(artifact["pile"])
    _validate_glue(artifact["glue"])
    gate_matrix = _mapping(artifact["gate_matrix"], field="gate_matrix")
    if gate_matrix != _EXPECTED_GATE_MATRIX:
        raise AssetRequirementsError(
            "G3 gate matrix must bind every planned sub-gate to its exact assets"
        )
    route = _mapping(artifact["stage6_route"], field="stage6_route")
    if route != {
        "architecture": "pythia-410m-sequence-classification",
        "base_model_name": "pythia-410m-deduped-step0",
        "direct_supervision_input": "base_initialization",
        "pretrained_finetune_input": "stage5_checkpoint_with_same_base_initialization",
        "classifier_head_initialization": "deterministic_seeded_runtime_head",
    }:
        raise AssetRequirementsError("Stage 6 must use the frozen 410M route")


def load_stage0_asset_requirements(path: str | Path) -> dict[str, Any]:
    """Load and validate a Stage 0 asset requirement artifact."""

    raw = Path(path).read_bytes()
    value = ensure_json_object(
        loads_strict_json(raw),
        field="asset requirements",
    )
    result = dict(value)
    validate_stage0_asset_requirements(result)
    return result


__all__ = [
    "AssetRequirementsError",
    "SCHEMA_VERSION",
    "load_stage0_asset_requirements",
    "requirements_artifact_hash",
    "validate_stage0_asset_requirements",
]
