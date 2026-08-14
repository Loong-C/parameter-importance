from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

import pytest

import param_importance_nlp.assets as assets_module
from param_importance_nlp.assets import (
    AssetActorRole,
    AssetEncodingError,
    AssetFile,
    AssetNotReadyError,
    AssetState,
    AssetValidationError,
    AssetVerificationError,
    admit_g3_qualification,
    build_g3_candidate_manifest,
    build_g3_qualification,
    build_manifest,
    compute_asset_id,
    compute_candidate_id,
    inspect_manifest_contract,
    load_manifest,
    publish_manifest_atomic,
    resolve_qualified_asset,
    resolve_ready_asset,
    transition_manifest,
    validate_asset_path,
    validate_g3_manifest,
    validate_g3_qualification,
    validate_qualified_ready_manifest,
    validate_manifest,
    validate_state_transition,
    verify_only,
)
from param_importance_nlp.atomic import sha256_file, stable_json_bytes, stable_json_hash


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata(asset_type: str = "model") -> dict[str, object]:
    if asset_type == "model":
        return {
            "architecture": "FixtureLM",
            "parameter_count": 42,
            "dtype": "float32",
            "initialization_id": "seed:7",
        }
    if asset_type == "tokenizer":
        return {
            "tokenizer_class": "FixtureTokenizer",
            "vocab_size": 128,
            "special_tokens": {"pad_token": "<pad>", "unk_token": "<unk>"},
            "normalization": "NFC",
        }
    if asset_type == "dataset":
        return {
            "splits": {
                "train": {"sample_count": 10, "fields": ["text", "label"]},
                "validation": {"sample_count": 2, "fields": ["text", "label"]},
            },
            "preprocessing_version": "fixture-preprocess-v1",
        }
    if asset_type == "source":
        return {"source_kind": "git", "license": "Apache-2.0"}
    raise AssertionError(asset_type)


_GIT_COMMIT = "a" * 40
_TOKENIZER_ASSET_ID = _digest(b"tokenizer-asset")
_PARENT_DATASET_ID = _digest(b"parent-dataset")


def _g3_preprocessing(
    *,
    tokenizer_asset_id: str | None = None,
    parent_asset_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "version": "fixture-preprocess-v1",
        "config_hash": _digest(b"preprocess-config"),
        "code_git_commit": _GIT_COMMIT,
        "tokenizer_asset_id": tokenizer_asset_id,
        "parent_asset_ids": parent_asset_ids or [],
    }


def _g3_causal_lm_mapping() -> dict[str, object]:
    return {
        "labels_alignment": "pre_shifted_next_token",
        "source_tokens_per_record": 2049,
        "input_sequence_length": 2048,
        "label_sequence_length": 2048,
        "input_slice": [0, 2048],
        "label_slice": [1, 2049],
        "attention_mask_policy": "all_one_for_fixed_full_record",
        "effective_target_tokens": 2048,
        "loss_adapter_id": "pre-shifted-next-token-cross-entropy-v1",
    }


def _g3_task_contract() -> dict[str, object]:
    return {
        "task": "sst2",
        "text_fields": ["sentence"],
        "label_mapping": {"negative": 0, "positive": 1},
        "unlabeled_test_policy": (
            "retain_in_raw_exclude_from_derived_training_and_evaluation"
        ),
    }


def _g3_asset_contract(
    asset_type: str,
    *,
    dataset_storage: str = "hf",
) -> tuple[list[AssetFile], dict[str, object]]:
    if asset_type == "model":
        config_hash = _digest(b"model-config")
        return (
            [
                AssetFile("config.json", 12, config_hash, "config"),
                AssetFile("weights.bin", len(b"weights"), _digest(b"weights"), "weights"),
            ],
            {
                "contract_version": "stage0-model-metadata-v1",
                "architecture": "FixtureLM",
                "parameter_count": 42,
                "tensor_count": 4,
                "dtype": "float32",
                "dtype_counts": {"float32": 42},
                "max_position_embeddings": 2048,
                "config_path": "config.json",
                "config_sha256": config_hash,
                "initialization_id": "seed:7",
                "initialization_kind": "base_initialization",
                "training_step": 0,
                "parent_model_asset_id": None,
            },
        )
    if asset_type == "tokenizer":
        config_hash = _digest(b"tokenizer-config")
        return (
            [
                AssetFile("tokenizer.json", 16, config_hash, "config"),
                AssetFile("vocab.json", 24, _digest(b"vocab"), "vocab"),
            ],
            {
                "contract_version": "stage0-tokenizer-metadata-v1",
                "tokenizer_class": "FixtureTokenizer",
                "implementation_version": "fixture-tokenizers-1.0",
                "vocab_size": 50254,
                "token_count_with_added_tokens": 50277,
                "vocab_mapping_sha256": (
                    "728db3c8825c64b8de98049c042221ba64b16a89348ada66f8d63c3d12dea84f"
                ),
                "glue_padding_policy": "explicit_existing_added_token",
                "special_tokens": {
                    "eos_token": {"token": "<eos>", "token_id": 0},
                    "glue_padding_token": {
                        "token": "<|padding|>",
                        "token_id": 1,
                    },
                },
                "normalization": "NFC",
                "config_path": "tokenizer.json",
                "config_sha256": config_hash,
                "model_max_length": 2048,
            },
        )
    if asset_type == "source":
        return (
            [AssetFile("source.tar", 64, _digest(b"source"), "source")],
            {
                "contract_version": "stage0-source-metadata-v1",
                "source_kind": "git",
                "license": "Apache-2.0",
                "upstream_locator": "git:example/project@0123456789abcdef",
            },
        )
    if asset_type != "dataset":
        raise AssertionError(asset_type)

    common: dict[str, object] = {
        "contract_version": "stage0-dataset-metadata-v1",
        "raw_revision": "pile-fixture-revision-1",
        "preprocessing_version": "fixture-preprocess-v1",
    }
    if dataset_storage == "raw_pile":
        idx_hash = _digest(b"idx")
        shard_zero_hash = _digest(b"shard-zero")
        shard_one_hash = _digest(b"shard-one")
        shard_size = 4098 * 262_144
        files = [
            AssetFile("document-00001.bin", shard_size, shard_one_hash, "tokens"),
            AssetFile("document.idx", 128, idx_hash, "index"),
            AssetFile("document-00000.bin", shard_size, shard_zero_hash, "tokens"),
        ]
        metadata = common | {
            "dataset_kind": "raw_indexed_mmap",
            "splits": {
                "train": {
                    "sample_count": 2,
                    "fields": ["tokens"],
                    "cursor": {"start": 0, "stop": 2},
                }
            },
            "preprocessing": _g3_preprocessing(
                parent_asset_ids=[_PARENT_DATASET_ID]
            ),
            "storage": {
                "kind": "pythia_mmap_shards",
                "idx": {
                    "path": "document.idx",
                    "sha256": idx_hash,
                    "magic": "MMIDIDX",
                    "version": 1,
                    "dtype_code": 8,
                    "itemsize_bytes": 2,
                    "sequence_count": 524_288,
                    "document_count": 1,
                },
                "tokens_per_record": 2049,
                "global_byte_coverage": {
                    "start": 0,
                    "stop": shard_size * 2,
                },
                "required_cursor_stop": 524_288,
                "causal_lm_mapping": _g3_causal_lm_mapping(),
                "reference_reader": {
                    "repository": "EleutherAI/pythia",
                    "revision": "a19eecb807ec2c79a39ebf18108816e6ffffc1d5",
                },
                "reference_batch_size": 1024,
                "reference_batch_sha256": {
                    "0": _digest(b"raw-batch-0"),
                    "1": _digest(b"raw-batch-1"),
                    "511": _digest(b"raw-batch-511"),
                },
                "last_required_record_sha256": _digest(
                    b"last-required-record"
                ),
                "cross_shard_policy": (
                    "explicit_ordered_global_byte_stream_selected_prefix_"
                    "must_be_fully_covered"
                ),
                "shards": [
                    {
                        "ordinal": 0,
                        "path": "document-00000.bin",
                        "size_bytes": shard_size,
                        "sha256": shard_zero_hash,
                        "byte_start": 0,
                        "byte_stop": shard_size,
                    },
                    {
                        "ordinal": 1,
                        "path": "document-00001.bin",
                        "size_bytes": shard_size,
                        "sha256": shard_one_hash,
                        "byte_start": shard_size,
                        "byte_stop": shard_size * 2,
                    },
                ],
            },
        }
        return files, metadata
    if dataset_storage == "raw_glue":
        train_hash = _digest(b"raw-glue-train")
        validation_hash = _digest(b"raw-glue-validation")
        test_hash = _digest(b"raw-glue-test")
        files = [
            AssetFile("test.parquet", 13, test_hash, "test"),
            AssetFile("train.parquet", 14, train_hash, "train"),
            AssetFile(
                "validation.parquet",
                19,
                validation_hash,
                "validation",
            ),
        ]
        metadata = common | {
            "dataset_kind": "hf_raw_parquet",
            "splits": {
                "train": {
                    "sample_count": 2,
                    "fields": ["sentence", "label"],
                    "cursor": {"start": 0, "stop": 2},
                },
                "validation": {
                    "sample_count": 1,
                    "fields": ["sentence", "label"],
                    "cursor": {"start": 0, "stop": 1},
                },
                "test": {
                    "sample_count": 1,
                    "fields": ["sentence", "label"],
                    "cursor": {"start": 0, "stop": 1},
                },
            },
            "preprocessing": _g3_preprocessing(
                parent_asset_ids=[_PARENT_DATASET_ID]
            ),
            "task_contract": _g3_task_contract(),
            "storage": {
                "kind": "hf_raw_parquet_files",
                "splits": {
                    "train": [
                        {"path": "train.parquet", "sha256": train_hash}
                    ],
                    "validation": [
                        {
                            "path": "validation.parquet",
                            "sha256": validation_hash,
                        }
                    ],
                    "test": [
                        {"path": "test.parquet", "sha256": test_hash}
                    ],
                },
            },
        }
        return files, metadata
    if dataset_storage == "derived_pile":
        files = [
            AssetFile("dataset/state.json", 32, _digest(b"state"), "dataset_state")
        ]
        metadata = common | {
            "dataset_kind": "derived_pretokenized",
            "splits": {
                "train": {
                    "sample_count": 2,
                    "fields": ["input_ids", "labels", "attention_mask"],
                    "cursor": {"start": 0, "stop": 2},
                }
            },
            "preprocessing": _g3_preprocessing(
                tokenizer_asset_id=_TOKENIZER_ASSET_ID,
                parent_asset_ids=[_PARENT_DATASET_ID],
            ),
            "storage": {
                "kind": "pile_derived_pretokenized",
                "format": "huggingface_load_from_disk",
                "causal_lm_mapping": _g3_causal_lm_mapping(),
                "source_row_mapping": {
                    "parent_asset_id": _PARENT_DATASET_ID,
                    "source_split": "train",
                    "source_cursor_start": 0,
                    "derived_row_start": 0,
                    "row_count": 2,
                },
                "converter_version": "pile-converter-v1",
                "converter_git_commit": _GIT_COMMIT,
            },
        }
        return files, metadata
    if dataset_storage == "derived_glue":
        files = [
            AssetFile("dataset/state.json", 32, _digest(b"state"), "dataset_state")
        ]
        metadata = common | {
            "dataset_kind": "derived_pretokenized",
            "splits": {
                "train": {
                    "sample_count": 2,
                    "fields": ["input_ids", "attention_mask", "labels"],
                    "cursor": {"start": 0, "stop": 2},
                },
                "validation": {
                    "sample_count": 1,
                    "fields": ["input_ids", "attention_mask", "labels"],
                    "cursor": {"start": 0, "stop": 1},
                },
            },
            "preprocessing": _g3_preprocessing(
                tokenizer_asset_id=_TOKENIZER_ASSET_ID,
                parent_asset_ids=[_PARENT_DATASET_ID],
            ),
            "task_contract": _g3_task_contract(),
            "storage": {
                "kind": "hf_load_from_disk",
                "format_version": "datasets-4",
            },
        }
        return files, metadata
    files = [AssetFile("dataset/state.json", 32, _digest(b"state"), "dataset_state")]
    metadata = common | {
        "dataset_kind": "hf_dataset",
        "splits": {
            "train": {
                "sample_count": 2,
                "fields": ["text", "label"],
                "cursor": {"start": 0, "stop": 2},
            }
        },
        "preprocessing": _g3_preprocessing(),
        "storage": {
            "kind": "hf_load_from_disk",
            "format_version": "datasets-4",
        },
    }
    return files, metadata


def _g3_manifest(
    asset_type: str = "model",
    *,
    dataset_storage: str = "hf",
    created_at: str = "2026-07-19T00:00:00Z",
    evidence_sha256: str | None = None,
) -> dict[str, object]:
    files, metadata = _g3_asset_contract(
        asset_type, dataset_storage=dataset_storage
    )
    return build_g3_candidate_manifest(
        asset_type=asset_type,
        name=f"g3-{asset_type}-{dataset_storage}",
        source=f"fixture:{asset_type}:{dataset_storage}",
        revision="0123456789abcdef",
        files=files,
        actor="test-fetcher",
        actor_role=AssetActorRole.FETCHER,
        actor_instance_id="test-host:fetcher:1",
        evidence_ref="evidence/acquisition.json",
        evidence_sha256=evidence_sha256 or _digest(b"acquisition"),
        generator_version="tests/1",
        generator_git_commit=_GIT_COMMIT,
        metadata=metadata,
        created_at=created_at,
    )


def _build_g3_from_contract(
    asset_type: str,
    files: list[AssetFile],
    metadata: dict[str, object],
) -> dict[str, object]:
    return build_g3_candidate_manifest(
        asset_type=asset_type,
        name=f"custom-g3-{asset_type}",
        source=f"fixture:custom:{asset_type}",
        revision="0123456789abcdef",
        files=files,
        actor="test-fetcher",
        actor_role="fetcher",
        actor_instance_id="test-host:fetcher:1",
        evidence_ref="evidence/fetch-start.json",
        evidence_sha256=_digest(b"fetch-start"),
        generator_version="tests/1",
        generator_git_commit=_GIT_COMMIT,
        metadata=metadata,
        created_at="2026-07-19T00:00:00Z",
    )


_REQUIREMENTS_REF = "configs/stage0/g3-asset-requirements-v1.json"
_REQUIREMENTS_SHA256 = _digest(b"frozen-g3-requirements")
_ACQUISITION_REF = "evidence/acquisition.json"
_ACQUISITION_SHA256 = _digest(b"acquisition")
_VERIFICATION_REF = "evidence/model-verification-v2.json"
_VERIFICATION_SHA256 = _digest(b"model-verification-v2")
_QUALIFICATION_REF = "evidence/model-qualification-v1.json"


def _verified_g3_manifest() -> dict[str, object]:
    manifest = _g3_manifest()
    downloaded = transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor="test-fetcher",
        actor_role="fetcher",
        actor_instance_id="test-host:fetcher:1",
        evidence_ref=_ACQUISITION_REF,
        evidence_sha256=_ACQUISITION_SHA256,
        summary="fetch completed",
        at="2026-07-19T00:00:01Z",
    )
    return transition_manifest(
        downloaded,
        AssetState.VERIFIED,
        actor="test-verifier",
        actor_role="verifier",
        actor_instance_id="test-host:verifier:1",
        evidence_ref=_VERIFICATION_REF,
        evidence_sha256=_VERIFICATION_SHA256,
        summary="full SHA-256 and typed verification passed",
        at="2026-07-19T00:00:02Z",
    )


def _qualification_checks() -> list[dict[str, object]]:
    return [
        {
            "check_id": "manifest-profile",
            "status": "PASS",
            "evidence_ref": "evidence/checks/manifest-profile.json",
            "evidence_sha256": _digest(b"manifest-profile-check"),
            "summary": "G3 manifest profile is complete",
        },
        {
            "check_id": "semantic-load",
            "status": "PASS",
            "evidence_ref": "evidence/checks/semantic-load.json",
            "evidence_sha256": _digest(b"semantic-load-check"),
            "summary": "asset loaded under the frozen offline contract",
        },
    ]


def _qualification(
    verified: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = verified or _verified_g3_manifest()
    report = build_g3_qualification(
        manifest,
        requirements_ref=_REQUIREMENTS_REF,
        requirements_sha256=_REQUIREMENTS_SHA256,
        acquisition_ref=_ACQUISITION_REF,
        acquisition_sha256=_ACQUISITION_SHA256,
        verification_ref=_VERIFICATION_REF,
        verification_sha256=_VERIFICATION_SHA256,
        checks=_qualification_checks(),
        checked_at="2026-07-19T00:00:03Z",
        generator_git_commit="b" * 40,
    )
    return manifest, report


def _rehash_qualification(value: dict[str, object]) -> None:
    payload = deepcopy(value)
    del payload["artifact_hash"]
    value["artifact_hash"] = stable_json_hash(payload)


def _manifest(
    tmp_path: Path,
    *,
    declared_size: int | None = None,
    declared_hash: str | None = None,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "asset-root"
    root.mkdir(parents=True)
    payload = b"immutable fixture\n"
    (root / "weights.bin").write_bytes(payload)
    manifest = build_manifest(
        asset_type="model",
        name="fixture-step0",
        source="huggingface:example/fixture",
        revision="0123456789abcdef",
        files=[
            AssetFile(
                path="weights.bin",
                size_bytes=len(payload) if declared_size is None else declared_size,
                sha256=_digest(payload) if declared_hash is None else declared_hash,
                role="weights",
            )
        ],
        actor="test-fetcher",
        actor_role=AssetActorRole.FETCHER,
        evidence_ref="evidence/fetch-start.json",
        generator_version="tests/1",
        metadata=_metadata(),
        created_at="2026-07-19T00:00:00Z",
    )
    return root, manifest


def _transition_to(
    manifest: dict[str, object], target: AssetState
) -> dict[str, object]:
    sequence = [
        (
            AssetState.DOWNLOADED,
            AssetActorRole.FETCHER,
            "evidence/fetch-complete.json",
            "2026-07-19T00:00:01Z",
        ),
        (
            AssetState.VERIFIED,
            AssetActorRole.VERIFIER,
            "evidence/verification.json",
            "2026-07-19T00:00:02Z",
        ),
        (
            AssetState.READY,
            AssetActorRole.GATE,
            "evidence/gate.json",
            "2026-07-19T00:00:03Z",
        ),
    ]
    updated = manifest
    for state, actor_role, evidence_ref, at in sequence:
        updated = transition_manifest(
            updated,
            state,
            actor="test-actor",
            actor_role=actor_role,
            evidence_ref=evidence_ref,
            summary=f"entered {state.value}",
            at=at,
        )
        if state is target:
            return updated
    raise AssertionError(target)


def _manifest_directory(tmp_path: Path) -> Path:
    path = tmp_path / "manifests"
    path.mkdir(parents=True)
    return path


def test_asset_state_machine_requires_order_role_and_terminal_invalid() -> None:
    validate_state_transition(
        AssetState.DOWNLOADING,
        AssetState.DOWNLOADED,
        actor_role=AssetActorRole.FETCHER,
    )
    validate_state_transition(
        AssetState.DOWNLOADED,
        AssetState.VERIFIED,
        actor_role=AssetActorRole.VERIFIER,
    )
    validate_state_transition(
        AssetState.VERIFIED,
        AssetState.READY,
        actor_role=AssetActorRole.GATE,
    )
    with pytest.raises(AssetValidationError, match="not authorized"):
        validate_state_transition(
            AssetState.DOWNLOADING,
            AssetState.DOWNLOADED,
            actor_role=AssetActorRole.VERIFIER,
        )
    with pytest.raises(AssetValidationError, match="Forbidden"):
        validate_state_transition(
            AssetState.DOWNLOADING,
            AssetState.READY,
            actor_role=AssetActorRole.GATE,
        )
    with pytest.raises(AssetValidationError, match="Forbidden"):
        validate_state_transition(
            AssetState.INVALID,
            AssetState.DOWNLOADING,
            actor_role=AssetActorRole.FETCHER,
        )


def test_history_enforces_roles_and_evidence(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    wrong_role = deepcopy(downloaded)
    wrong_role["state_history"][-1]["actor_role"] = "gate"
    with pytest.raises(AssetValidationError, match="not authorized"):
        validate_manifest(wrong_role)

    with pytest.raises(AssetValidationError, match="evidence_ref"):
        transition_manifest(
            downloaded,
            AssetState.VERIFIED,
            actor="test-verifier",
            actor_role=AssetActorRole.VERIFIER,
            evidence_ref=None,
            summary="missing evidence",
        )

    missing_key = deepcopy(downloaded)
    del missing_key["state_history"][-1]["evidence_ref"]
    with pytest.raises(AssetValidationError, match="missing"):
        validate_manifest(missing_key)


def test_invalid_manifest_cannot_flip_back_to_success(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    invalid = transition_manifest(
        downloaded,
        AssetState.INVALID,
        actor="test-verifier",
        actor_role=AssetActorRole.VERIFIER,
        evidence_ref="evidence/hash-failure.json",
        summary="hash failed",
        at="2026-07-19T00:00:02Z",
    )
    with pytest.raises(AssetValidationError, match="Forbidden"):
        transition_manifest(
            invalid,
            AssetState.VERIFIED,
            actor="test-verifier",
            actor_role=AssetActorRole.VERIFIER,
            evidence_ref="evidence/illegal-reversal.json",
            summary="illegal reversal",
        )


def test_asset_id_is_stable_across_file_order_and_state(tmp_path: Path) -> None:
    first = AssetFile("b.bin", 2, _digest(b"bb"))
    second = AssetFile("a.bin", 1, _digest(b"a"))
    identity = {
        "asset_type": "model",
        "name": "ordered-fixture",
        "source": "huggingface:example/ordered",
        "revision": "deadbeef",
        "metadata": _metadata(),
    }
    left = compute_asset_id(**identity, files=[first, second])
    right = compute_asset_id(**identity, files=[second, first])
    assert left == right

    changed_metadata = deepcopy(_metadata())
    changed_metadata["initialization_id"] = "seed:8"
    assert left != compute_asset_id(
        **{**identity, "metadata": changed_metadata}, files=[first, second]
    )

    _, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    assert downloaded["asset_id"] == manifest["asset_id"]


def test_legacy_builder_and_inspection_remain_profile_free(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    assert manifest["asset_id"] == (
        "5de17de847a90ca42bcde5fd15edb54134f6bb0c52195184cace578659b50301"
    )
    assert _digest(stable_json_bytes(manifest)) == (
        "e5f940056a3517fefb0a2ea7715934a8de08cbfa31ef852228d085be7cd67f9b"
    )
    assert set(manifest) == {
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
    assert set(manifest["state_history"][0]) == {
        "from",
        "to",
        "at",
        "actor",
        "actor_role",
        "evidence_ref",
        "summary",
    }
    inspection = inspect_manifest_contract(manifest)
    assert inspection["manifest_valid"] is True
    assert inspection["g3_eligible"] is False
    assert inspection["contract_profile"] is None
    with pytest.raises(AssetValidationError, match="G3 eligibility"):
        validate_g3_manifest(manifest)


@pytest.mark.parametrize(
    ("asset_type", "dataset_storage"),
    [
        ("model", "hf"),
        ("tokenizer", "hf"),
        ("source", "hf"),
        ("dataset", "hf"),
        ("dataset", "raw_glue"),
        ("dataset", "raw_pile"),
        ("dataset", "derived_pile"),
        ("dataset", "derived_glue"),
    ],
)
def test_g3_builder_accepts_strong_typed_profiles(
    asset_type: str, dataset_storage: str
) -> None:
    manifest = _g3_manifest(asset_type, dataset_storage=dataset_storage)
    validate_g3_manifest(manifest)
    assert manifest["contract_profile"] == "stage0-g3-v1"
    assert manifest["candidate_id"] == compute_candidate_id(manifest)
    assert inspect_manifest_contract(manifest) == {
        "schema_version": "stage0-asset-contract-inspection-v1",
        "manifest_valid": True,
        "contract_profile": "stage0-g3-v1",
        "g3_eligible": True,
        "violations": [],
    }


def test_g3_identity_fields_and_history_are_all_or_none(tmp_path: Path) -> None:
    _, legacy = _manifest(tmp_path)
    partial_top = deepcopy(legacy)
    partial_top["candidate_id"] = "0" * 64
    with pytest.raises(AssetValidationError, match="all-or-none"):
        validate_manifest(partial_top)

    partial_history = deepcopy(legacy)
    partial_history["state_history"][0]["actor_instance_id"] = "host:1"
    with pytest.raises(AssetValidationError, match="extra"):
        validate_manifest(partial_history)

    g3 = _g3_manifest()
    del g3["state_history"][0]["evidence_sha256"]
    with pytest.raises(AssetValidationError, match="missing"):
        validate_manifest(g3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generator_git_commit", "A" * 40, "Git digest"),
        ("generator_git_commit", "0" * 39, "Git digest"),
        ("history_evidence_sha256", "0" * 63, "64 lowercase"),
        ("history_summary", "", "non-empty"),
    ],
)
def test_g3_rejects_invalid_provenance_fields(
    field: str, value: str, message: str
) -> None:
    manifest = _g3_manifest()
    if field == "history_evidence_sha256":
        manifest["state_history"][0]["evidence_sha256"] = value
    elif field == "history_summary":
        manifest["state_history"][0]["summary"] = value
    else:
        manifest[field] = value
    with pytest.raises(AssetValidationError, match=message):
        validate_manifest(manifest)


def test_g3_candidate_id_is_stable_across_transitions_but_not_candidates() -> None:
    manifest = _g3_manifest()
    downloaded = transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor="test-fetcher",
        actor_role="fetcher",
        actor_instance_id="test-host:fetcher:1",
        evidence_ref="evidence/fetch-complete.json",
        evidence_sha256=_digest(b"fetch-complete"),
        summary="fetch completed",
        at="2026-07-19T00:00:01Z",
    )
    verified = transition_manifest(
        downloaded,
        AssetState.VERIFIED,
        actor="test-verifier",
        actor_role="verifier",
        actor_instance_id="test-host:verifier:1",
        evidence_ref="evidence/verification.json",
        evidence_sha256=_digest(b"verification"),
        summary="full hash verification passed",
        at="2026-07-19T00:00:02Z",
    )
    assert verified["asset_id"] == manifest["asset_id"]
    assert verified["candidate_id"] == manifest["candidate_id"]
    assert compute_candidate_id(verified) == manifest["candidate_id"]

    replacement_candidate = _g3_manifest(
        created_at="2026-07-19T00:01:00Z",
        evidence_sha256=_digest(b"replacement-fetch-start"),
    )
    assert replacement_candidate["asset_id"] == manifest["asset_id"]
    assert replacement_candidate["candidate_id"] != manifest["candidate_id"]

    with pytest.raises(AssetValidationError, match="qualification admission"):
        transition_manifest(
            verified,
            AssetState.READY,
            actor="test-gate",
            actor_role="gate",
            actor_instance_id="test-host:gate:1",
            evidence_ref="evidence/qualification.json",
            evidence_sha256=_digest(b"qualification"),
            summary="qualification passed",
            at="2026-07-19T00:00:03Z",
        )


def test_g3_history_timestamps_are_not_before_creation_and_are_monotonic() -> None:
    manifest = _g3_manifest(created_at="2026-07-19T00:00:05Z")
    with pytest.raises(AssetValidationError, match="monotonic"):
        transition_manifest(
            manifest,
            AssetState.DOWNLOADED,
            actor="test-fetcher",
            actor_role="fetcher",
            actor_instance_id="test-host:fetcher:1",
            evidence_ref="evidence/fetch-complete.json",
            evidence_sha256=_digest(b"fetch-complete"),
            summary="fetch completed",
            at="2026-07-19T00:00:04Z",
        )

    downloaded = transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor="test-fetcher",
        actor_role="fetcher",
        actor_instance_id="test-host:fetcher:1",
        evidence_ref="evidence/fetch-complete.json",
        evidence_sha256=_digest(b"fetch-complete"),
        summary="fetch completed",
        at="2026-07-19T00:00:06Z",
    )
    with pytest.raises(AssetValidationError, match="monotonic"):
        transition_manifest(
            downloaded,
            AssetState.VERIFIED,
            actor="test-verifier",
            actor_role="verifier",
            actor_instance_id="test-host:verifier:1",
            evidence_ref="evidence/verification.json",
            evidence_sha256=_digest(b"verification"),
            summary="verified",
            at="2026-07-19T00:00:05Z",
        )


def test_g3_transition_requires_enriched_evidence_but_legacy_rejects_it(
    tmp_path: Path,
) -> None:
    g3 = _g3_manifest()
    with pytest.raises(AssetValidationError, match="actor_instance_id"):
        transition_manifest(
            g3,
            AssetState.DOWNLOADED,
            actor="test-fetcher",
            actor_role="fetcher",
            evidence_ref="evidence/fetch-complete.json",
            summary="fetch completed",
        )


def test_g3_qualification_admission_and_qualified_resolution(tmp_path: Path) -> None:
    verified, qualification = _qualification()
    validate_g3_qualification(qualification)
    assert set(qualification) == {
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
    assert qualification["formal"] is True
    assert qualification["asset_id"] == verified["asset_id"]
    assert qualification["candidate_id"] == verified["candidate_id"]
    assert qualification["verified_manifest_sha256"] == stable_json_hash(verified)
    assert qualification["requirements_sha256"] == _REQUIREMENTS_SHA256
    assert qualification["verification_sha256"] == _VERIFICATION_SHA256
    assert qualification["network_attempts"] == 0
    assert all(
        set(check)
        == {"check_id", "status", "evidence_ref", "evidence_sha256", "summary"}
        for check in qualification["checks"]
    )

    ready = admit_g3_qualification(
        verified,
        qualification,
        qualification_ref=_QUALIFICATION_REF,
        requirements_artifact_hash=_REQUIREMENTS_SHA256,
        actor="test-gate",
        actor_instance_id="test-host:gate:1",
        at="2026-07-19T00:00:04Z",
    )
    final_event = ready["state_history"][-1]
    assert ready["state"] == "ready"
    assert final_event["actor_role"] == "gate"
    assert final_event["evidence_ref"] == _QUALIFICATION_REF
    assert final_event["evidence_sha256"] == qualification["artifact_hash"]

    validated = validate_qualified_ready_manifest(
        ready,
        qualification,
        qualification_ref=_QUALIFICATION_REF,
        requirements_artifact_hash=_REQUIREMENTS_SHA256,
    )
    assert validated == ready

    asset_root = tmp_path / "g3-model"
    asset_root.mkdir()
    (asset_root / "config.json").write_bytes(b"model-config")
    (asset_root / "weights.bin").write_bytes(b"weights")
    with pytest.raises(AssetNotReadyError, match="resolve_qualified_asset"):
        resolve_ready_asset(ready, asset_root)
    resolved = resolve_qualified_asset(
        ready,
        asset_root,
        qualification,
        qualification_ref=_QUALIFICATION_REF,
        requirements_artifact_hash=_REQUIREMENTS_SHA256,
    )
    assert resolved.asset_id == ready["asset_id"]
    assert resolved.path_for("weights.bin") == (asset_root / "weights.bin").resolve()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("formal_false", "formal must be true"),
        ("network", "network_attempts"),
        ("missing_checks", "non-empty"),
        ("failed_check", "must be PASS"),
        ("duplicate_check", "must be unique"),
    ],
)
def test_g3_qualification_rejects_forged_or_incomplete_pass(
    mutation: str, message: str
) -> None:
    _, qualification = _qualification()
    if mutation == "formal_false":
        qualification["formal"] = False
    elif mutation == "network":
        qualification["network_attempts"] = 1
    elif mutation == "missing_checks":
        qualification["checks"] = []
    elif mutation == "failed_check":
        qualification["checks"][0]["status"] = "FAIL"
    else:
        qualification["checks"][1]["check_id"] = qualification["checks"][0][
            "check_id"
        ]
    _rehash_qualification(qualification)
    with pytest.raises(AssetValidationError, match=message):
        validate_g3_qualification(qualification)


def test_g3_qualification_artifact_hash_prevents_unhashed_pass_forgery() -> None:
    _, qualification = _qualification()
    qualification["checks"][0]["summary"] = "forged after qualification"
    with pytest.raises(AssetValidationError, match="artifact_hash"):
        validate_g3_qualification(qualification)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("asset_id", "asset_id"),
        ("candidate_id", "candidate_id"),
        ("verified_manifest_sha256", "verified_manifest_sha256"),
        ("verification_ref", "verification evidence"),
        ("verification_sha256", "verification evidence"),
    ],
)
def test_g3_admission_rejects_qualification_for_another_identity(
    field: str, message: str
) -> None:
    verified, qualification = _qualification()
    qualification[field] = (
        "evidence/other-verification.json"
        if field == "verification_ref"
        else "f" * 64
    )
    _rehash_qualification(qualification)
    with pytest.raises(AssetValidationError, match=message):
        admit_g3_qualification(
            verified,
            qualification,
            qualification_ref=_QUALIFICATION_REF,
            requirements_artifact_hash=_REQUIREMENTS_SHA256,
            actor="test-gate",
            actor_instance_id="test-host:gate:1",
            at="2026-07-19T00:00:04Z",
        )


def test_g3_admission_rejects_wrong_requirements_artifact_hash() -> None:
    verified, qualification = _qualification()
    with pytest.raises(AssetValidationError, match="requirements artifact_hash"):
        admit_g3_qualification(
            verified,
            qualification,
            qualification_ref=_QUALIFICATION_REF,
            requirements_artifact_hash="e" * 64,
            actor="test-gate",
            actor_instance_id="test-host:gate:1",
            at="2026-07-19T00:00:04Z",
        )


def test_g3_qualification_builder_rejects_wrong_verification_and_network() -> None:
    verified = _verified_g3_manifest()
    common = {
        "requirements_ref": _REQUIREMENTS_REF,
        "requirements_sha256": _REQUIREMENTS_SHA256,
        "checks": _qualification_checks(),
        "checked_at": "2026-07-19T00:00:03Z",
        "generator_git_commit": "b" * 40,
    }
    with pytest.raises(AssetValidationError, match="VERIFIED history"):
        build_g3_qualification(
            verified,
            verification_ref="evidence/wrong-verification.json",
            verification_sha256=_VERIFICATION_SHA256,
            **common,
        )
    with pytest.raises(AssetValidationError, match="network_attempts"):
        build_g3_qualification(
            verified,
            verification_ref=_VERIFICATION_REF,
            verification_sha256=_VERIFICATION_SHA256,
            network_attempts=1,
            **common,
        )


def test_qualified_resolver_rejects_history_report_mismatch(tmp_path: Path) -> None:
    verified, qualification = _qualification()
    ready = admit_g3_qualification(
        verified,
        qualification,
        qualification_ref=_QUALIFICATION_REF,
        requirements_artifact_hash=_REQUIREMENTS_SHA256,
        actor="test-gate",
        actor_instance_id="test-host:gate:1",
        at="2026-07-19T00:00:04Z",
    )
    asset_root = tmp_path / "g3-model"
    asset_root.mkdir()
    (asset_root / "config.json").write_bytes(b"model-config")
    (asset_root / "weights.bin").write_bytes(b"weights")

    wrong_hash = deepcopy(ready)
    wrong_hash["state_history"][-1]["evidence_sha256"] = "f" * 64
    with pytest.raises(AssetNotReadyError, match="qualification hash"):
        resolve_qualified_asset(
            wrong_hash,
            asset_root,
            qualification,
            qualification_ref=_QUALIFICATION_REF,
            requirements_artifact_hash=_REQUIREMENTS_SHA256,
        )
    with pytest.raises(AssetNotReadyError, match="qualification_ref"):
        resolve_qualified_asset(
            ready,
            asset_root,
            qualification,
            qualification_ref="evidence/other-qualification.json",
            requirements_artifact_hash=_REQUIREMENTS_SHA256,
        )
    with pytest.raises(AssetValidationError, match="requirements artifact_hash"):
        resolve_qualified_asset(
            ready,
            asset_root,
            qualification,
            qualification_ref=_QUALIFICATION_REF,
            requirements_artifact_hash="e" * 64,
        )

    _, legacy = _manifest(tmp_path)
    with pytest.raises(AssetValidationError, match="require contract_profile"):
        transition_manifest(
            legacy,
            AssetState.DOWNLOADED,
            actor="test-fetcher",
            actor_role="fetcher",
            actor_instance_id="test-host:fetcher:1",
            evidence_ref="evidence/fetch-complete.json",
            evidence_sha256=_digest(b"fetch-complete"),
            summary="fetch completed",
        )


@pytest.mark.parametrize("asset_type", ["model", "tokenizer", "dataset", "source"])
def test_typed_metadata_accepts_each_asset_contract(asset_type: str) -> None:
    manifest = build_manifest(
        asset_type=asset_type,
        name=f"fixture-{asset_type}",
        source=f"fixture:{asset_type}",
        revision="v1.2.3",
        files=[AssetFile("artifact.bin", 1, _digest(b"x"))],
        actor="test-fetcher",
        actor_role="fetcher",
        evidence_ref="evidence/fetch.json",
        generator_version="tests/1",
        metadata=_metadata(asset_type),
        created_at="2026-07-19T00:00:00Z",
    )
    validate_manifest(manifest)


@pytest.mark.parametrize(
    ("asset_type", "field", "bad_value"),
    [
        ("model", "parameter_count", 0),
        ("tokenizer", "special_tokens", []),
        ("dataset", "splits", {}),
        ("source", "license", ""),
    ],
)
def test_typed_metadata_rejects_missing_or_wrong_minimum_fields(
    asset_type: str,
    field: str,
    bad_value: object,
) -> None:
    metadata = _metadata(asset_type)
    metadata[field] = bad_value
    with pytest.raises(AssetValidationError, match=f"metadata.{field}"):
        build_manifest(
            asset_type=asset_type,
            name=f"bad-{asset_type}",
            source=f"fixture:{asset_type}",
            revision="v1.2.3",
            files=[AssetFile("artifact.bin", 1, _digest(b"x"))],
            actor="test-fetcher",
            actor_role="fetcher",
            evidence_ref=None,
            generator_version="tests/1",
            metadata=metadata,
        )


def test_g3_model_and_tokenizer_config_identity_binds_manifest_files() -> None:
    model_files, model_metadata = _g3_asset_contract("model")
    bad_model = deepcopy(model_metadata)
    bad_model["config_sha256"] = _digest(b"wrong-config")
    with pytest.raises(AssetValidationError, match="must match manifest.files"):
        _build_g3_from_contract("model", model_files, bad_model)

    tokenizer_files, tokenizer_metadata = _g3_asset_contract("tokenizer")
    bad_tokenizer = deepcopy(tokenizer_metadata)
    bad_tokenizer["config_path"] = "missing-tokenizer.json"
    with pytest.raises(AssetValidationError, match="reference an entry"):
        _build_g3_from_contract("tokenizer", tokenizer_files, bad_tokenizer)


def test_g3_tokenizer_freezes_total_token_count_vocab_mapping_and_padding() -> None:
    files, metadata = _g3_asset_contract("tokenizer")
    manifest = _build_g3_from_contract("tokenizer", files, metadata)
    assert manifest["metadata"]["token_count_with_added_tokens"] == 50277
    assert manifest["metadata"]["vocab_mapping_sha256"] == (
        "728db3c8825c64b8de98049c042221ba64b16a89348ada66f8d63c3d12dea84f"
    )
    assert (
        manifest["metadata"]["glue_padding_policy"]
        == "explicit_existing_added_token"
    )
    assert manifest["metadata"]["special_tokens"]["glue_padding_token"] == {
        "token": "<|padding|>",
        "token_id": 1,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_token_count", "token_count_with_added_tokens"),
        ("missing_vocab_hash", "vocab_mapping_sha256"),
        ("bad_vocab_hash", "vocab_mapping_sha256"),
        ("count_below_vocab", "may not be smaller"),
        ("special_token_out_of_range", "must be smaller"),
        ("bad_padding_policy", "glue_padding_policy"),
        ("missing_padding_token", "glue_padding_token"),
        ("wrong_padding_identity", "glue_padding_token"),
    ],
)
def test_g3_tokenizer_vocab_identity_fields_are_strict(
    mutation: str, message: str
) -> None:
    files, metadata = _g3_asset_contract("tokenizer")
    if mutation == "missing_token_count":
        del metadata["token_count_with_added_tokens"]
    elif mutation == "missing_vocab_hash":
        del metadata["vocab_mapping_sha256"]
    elif mutation == "bad_vocab_hash":
        metadata["vocab_mapping_sha256"] = "not-a-sha256"
    elif mutation == "count_below_vocab":
        metadata["token_count_with_added_tokens"] = 50253
    elif mutation == "special_token_out_of_range":
        metadata["special_tokens"]["glue_padding_token"]["token_id"] = 50277
    elif mutation == "bad_padding_policy":
        metadata["glue_padding_policy"] = "add_new_token_at_runtime"
    elif mutation == "missing_padding_token":
        del metadata["special_tokens"]["glue_padding_token"]
    else:
        metadata["special_tokens"]["glue_padding_token"]["token"] = "<pad>"
    with pytest.raises(AssetValidationError, match=message):
        _build_g3_from_contract("tokenizer", files, metadata)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tensor_count", 0, "tensor_count"),
        ("dtype_counts", {}, "non-empty"),
        ("dtype_counts", {"float32": 41}, "must sum"),
        ("dtype_counts", {"float32": 0, "float16": 42}, "positive"),
        ("dtype", "mixed", "sole dtype_counts key"),
        ("max_position_embeddings", 0, "max_position_embeddings"),
        ("training_step", 1, "base_initialization"),
        ("parent_model_asset_id", "1" * 64, "base_initialization"),
        ("unexpected", "annotation", "extra"),
    ],
)
def test_g3_model_metadata_is_strict(
    field: str, value: object, message: str
) -> None:
    files, metadata = _g3_asset_contract("model")
    metadata[field] = value
    with pytest.raises(AssetValidationError, match=message):
        _build_g3_from_contract("model", files, metadata)


def test_g3_model_metadata_accepts_canonical_mixed_dtype_summary() -> None:
    files, metadata = _g3_asset_contract("model")
    metadata["dtype"] = "mixed"
    metadata["dtype_counts"] = {"float16": 12, "float32": 30}
    manifest = _build_g3_from_contract("model", files, metadata)
    assert manifest["metadata"]["dtype"] == "mixed"
    assert manifest["metadata"]["dtype_counts"] == {
        "float16": 12,
        "float32": 30,
    }


def test_g3_model_metadata_rejects_noncanonical_multi_dtype_summary() -> None:
    files, metadata = _g3_asset_contract("model")
    metadata["dtype_counts"] = {"float16": 12, "float32": 30}
    with pytest.raises(AssetValidationError, match="multiple dtypes"):
        _build_g3_from_contract("model", files, metadata)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("ordinal_gap", "ordinals"),
        ("byte_gap", "contiguous"),
        ("size_mismatch", "match manifest.files"),
        ("hash_mismatch", "must match manifest.files"),
        ("path_missing", "reference an entry"),
    ],
)
def test_g3_raw_pile_shards_are_explicit_contiguous_and_file_bound(
    mutation: str, message: str
) -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_pile")
    shards = metadata["storage"]["shards"]
    if mutation == "ordinal_gap":
        shards[1]["ordinal"] = 2
    elif mutation == "byte_gap":
        shards[1]["byte_start"] += 1
    elif mutation == "size_mismatch":
        shards[0]["size_bytes"] += 1
    elif mutation == "hash_mismatch":
        shards[0]["sha256"] = "0" * 64
    else:
        shards[0]["path"] = "document-99999.bin"
    with pytest.raises(AssetValidationError, match=message):
        _build_g3_from_contract("dataset", files, metadata)


def test_g3_raw_pile_shard_order_is_metadata_not_manifest_file_order() -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_pile")
    manifest = _build_g3_from_contract("dataset", list(reversed(files)), metadata)
    assert [item["path"] for item in manifest["files"]] == sorted(
        item.path for item in files
    )
    assert [
        item["path"] for item in manifest["metadata"]["storage"]["shards"]
    ] == ["document-00000.bin", "document-00001.bin"]


def test_g3_raw_pile_accepts_selected_prefix_of_full_index() -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_pile")
    metadata["storage"]["idx"]["sequence_count"] = 146_432_000
    manifest = _build_g3_from_contract("dataset", files, metadata)
    assert manifest["metadata"]["storage"]["idx"]["sequence_count"] == 146_432_000
    assert manifest["metadata"]["storage"]["required_cursor_stop"] == 524_288


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("labels_alignment", "self_shifted"),
        ("source_tokens_per_record", 2048),
        ("input_sequence_length", 2049),
        ("label_sequence_length", 2047),
        ("input_slice", [0, 2049]),
        ("label_slice", [1, 2048]),
        ("effective_target_tokens", 2047),
        ("loss_adapter_id", "causal_lm_self_shifted_v1"),
    ],
)
def test_g3_raw_pile_freezes_2049_to_preshifted_2048_contract(
    field: str, value: object
) -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_pile")
    metadata["storage"]["causal_lm_mapping"][field] = value
    with pytest.raises(AssetValidationError, match="causal_lm_mapping"):
        _build_g3_from_contract("dataset", files, metadata)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_map", "causal_lm_mapping"),
        ("missing_reference", "reference_batch_sha256"),
        ("missing_511", "exactly batch keys"),
        ("uppercase_reference", "lowercase hex"),
        ("noncanonical_key", "exactly batch keys"),
        ("out_of_range_key", "exactly batch keys"),
        ("wrong_reader", "reference_reader"),
        ("wrong_batch_size", "reference_batch_size"),
        ("bad_last", "last_required_record_sha256"),
    ],
)
def test_g3_raw_pile_reference_batch_and_last_record_evidence_are_strict(
    mutation: str, message: str
) -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_pile")
    storage = metadata["storage"]
    if mutation == "missing_map":
        del storage["causal_lm_mapping"]
    elif mutation == "missing_reference":
        del storage["reference_batch_sha256"]
    elif mutation == "missing_511":
        del storage["reference_batch_sha256"]["511"]
    elif mutation == "uppercase_reference":
        storage["reference_batch_sha256"]["0"] = storage[
            "reference_batch_sha256"
        ]["0"].upper()
    elif mutation == "noncanonical_key":
        storage["reference_batch_sha256"]["0511"] = _digest(b"bad-key")
    elif mutation == "out_of_range_key":
        storage["reference_batch_sha256"]["512"] = _digest(b"out-of-range")
    elif mutation == "wrong_reader":
        storage["reference_reader"]["revision"] = "0" * 40
    elif mutation == "wrong_batch_size":
        storage["reference_batch_size"] = 1
    elif mutation == "bad_last":
        storage["last_required_record_sha256"] = "F" * 64
    with pytest.raises(AssetValidationError, match=message):
        _build_g3_from_contract("dataset", files, metadata)


@pytest.mark.parametrize("shortfall", ["one_byte", "one_record"])
def test_g3_raw_pile_rejects_prefix_shorter_than_required_cursor(
    shortfall: str,
) -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_pile")
    metadata["storage"]["idx"]["sequence_count"] = 146_432_000
    if shortfall == "one_byte":
        last = files[0]
        files[0] = AssetFile(
            last.path,
            last.size_bytes - 1,
            last.sha256,
            last.role,
        )
        metadata["storage"]["shards"][1]["size_bytes"] -= 1
        metadata["storage"]["shards"][1]["byte_stop"] -= 1
        metadata["storage"]["global_byte_coverage"]["stop"] -= 1
    else:
        metadata["storage"]["required_cursor_stop"] = 524_289
    with pytest.raises(AssetValidationError, match="too short"):
        _build_g3_from_contract("dataset", files, metadata)


def test_g3_raw_pile_rejects_temporary_shard_paths() -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_pile")
    bad_files = list(files)
    original = bad_files[0]
    bad_files[0] = AssetFile(
        f"{original.path}.part", original.size_bytes, original.sha256, original.role
    )
    with pytest.raises(AssetValidationError, match="temporary"):
        _build_g3_from_contract("dataset", bad_files, metadata)


def test_g3_raw_glue_parquet_files_bind_each_split_and_top_level_hash() -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_glue")
    manifest = _build_g3_from_contract("dataset", files, metadata)
    assert manifest["metadata"]["dataset_kind"] == "hf_raw_parquet"
    assert manifest["metadata"]["storage"]["kind"] == "hf_raw_parquet_files"

    wrong_hash = deepcopy(metadata)
    wrong_hash["storage"]["splits"]["train"][0]["sha256"] = "f" * 64
    with pytest.raises(AssetValidationError, match="must match manifest.files"):
        _build_g3_from_contract("dataset", files, wrong_hash)

    wrong_role_files = list(files)
    train = next(item for item in files if item.role == "train")
    wrong_role_files[wrong_role_files.index(train)] = AssetFile(
        train.path, train.size_bytes, train.sha256, "validation"
    )
    with pytest.raises(AssetValidationError, match="role must equal split"):
        _build_g3_from_contract("dataset", wrong_role_files, metadata)


@pytest.mark.parametrize("dataset_storage", ["raw_glue", "derived_glue"])
def test_g3_glue_task_contract_is_explicit_and_bound(dataset_storage: str) -> None:
    files, metadata = _g3_asset_contract(
        "dataset", dataset_storage=dataset_storage
    )
    manifest = _build_g3_from_contract("dataset", files, metadata)
    assert manifest["metadata"]["task_contract"] == _g3_task_contract()


@pytest.mark.parametrize("dataset_storage", ["raw_glue", "derived_glue"])
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "requires metadata.task_contract"),
        ("extra", "extra"),
        ("bad_task", "must be sst2, mnli, or rte"),
        ("duplicate_text", "text_fields must be unique"),
        ("noncontiguous_labels", "unique and contiguous"),
        ("bad_policy", "unlabeled_test_policy"),
    ],
)
def test_g3_glue_task_contract_is_strict(
    dataset_storage: str, mutation: str, message: str
) -> None:
    files, metadata = _g3_asset_contract(
        "dataset", dataset_storage=dataset_storage
    )
    if mutation == "missing":
        del metadata["task_contract"]
    elif mutation == "extra":
        metadata["task_contract"]["annotation"] = "not contractual"
    elif mutation == "bad_task":
        metadata["task_contract"]["task"] = "cola"
    elif mutation == "duplicate_text":
        metadata["task_contract"]["text_fields"] = ["sentence", "sentence"]
    elif mutation == "noncontiguous_labels":
        metadata["task_contract"]["label_mapping"] = {
            "negative": 0,
            "positive": 2,
        }
    else:
        metadata["task_contract"]["unlabeled_test_policy"] = "retain_everywhere"
    with pytest.raises(AssetValidationError, match=message):
        _build_g3_from_contract("dataset", files, metadata)


def test_g3_raw_glue_split_fields_cover_task_text_fields() -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="raw_glue")
    metadata["splits"]["validation"]["fields"] = ["label"]
    with pytest.raises(AssetValidationError, match="include every task_contract"):
        _build_g3_from_contract("dataset", files, metadata)


def test_g3_derived_glue_excludes_unlabeled_test_splits() -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="derived_glue")
    metadata["splits"]["test"] = {
        "sample_count": 1,
        "fields": ["input_ids", "attention_mask", "labels"],
        "cursor": {"start": 0, "stop": 1},
    }
    with pytest.raises(AssetValidationError, match="may not include unlabeled test"):
        _build_g3_from_contract("dataset", files, metadata)


def test_g3_derived_glue_requires_pretokenized_split_fields() -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="derived_glue")
    metadata["splits"]["validation"]["fields"] = ["input_ids", "labels"]
    with pytest.raises(AssetValidationError, match="attention_mask"):
        _build_g3_from_contract("dataset", files, metadata)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("config_hash", "config_hash"),
        ("code_git_commit", "code_git_commit"),
        ("tokenizer", "tokenizer_asset_id"),
        ("parents", "parent_asset_ids"),
    ],
)
def test_g3_derived_glue_binds_preprocessing_identity(
    mutation: str, message: str
) -> None:
    files, metadata = _g3_asset_contract("dataset", dataset_storage="derived_glue")
    if mutation == "config_hash":
        metadata["preprocessing"]["config_hash"] = "not-a-hash"
    elif mutation == "code_git_commit":
        metadata["preprocessing"]["code_git_commit"] = "main"
    elif mutation == "tokenizer":
        metadata["preprocessing"]["tokenizer_asset_id"] = None
    else:
        metadata["preprocessing"]["parent_asset_ids"] = []
    with pytest.raises(AssetValidationError, match=message):
        _build_g3_from_contract("dataset", files, metadata)


@pytest.mark.parametrize("dataset_storage", ["hf", "raw_pile", "derived_pile"])
def test_g3_non_glue_datasets_reject_task_contract(dataset_storage: str) -> None:
    files, metadata = _g3_asset_contract(
        "dataset", dataset_storage=dataset_storage
    )
    metadata["task_contract"] = _g3_task_contract()
    with pytest.raises(AssetValidationError, match="only valid for raw or derived GLUE"):
        _build_g3_from_contract("dataset", files, metadata)


def test_g3_raw_parquet_and_derived_load_from_disk_cannot_masquerade() -> None:
    raw_files, raw_metadata = _g3_asset_contract(
        "dataset", dataset_storage="raw_glue"
    )
    raw_as_derived = deepcopy(raw_metadata)
    raw_as_derived["storage"] = {
        "kind": "hf_load_from_disk",
        "format_version": "datasets-4",
    }
    with pytest.raises(AssetValidationError, match="metadata.storage|hf_raw_parquet"):
        _build_g3_from_contract("dataset", raw_files, raw_as_derived)

    derived_files, derived_metadata = _g3_asset_contract("dataset")
    derived_as_raw = deepcopy(derived_metadata)
    derived_as_raw["dataset_kind"] = "derived_pretokenized"
    derived_as_raw["preprocessing"]["tokenizer_asset_id"] = _TOKENIZER_ASSET_ID
    derived_as_raw["preprocessing"]["parent_asset_ids"] = [_PARENT_DATASET_ID]
    derived_as_raw["storage"] = raw_metadata["storage"]
    with pytest.raises(AssetValidationError, match="metadata.storage|hf_load_from_disk"):
        _build_g3_from_contract("dataset", derived_files, derived_as_raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("labels_alignment", "self_shifted"),
        ("source_tokens_per_record", 2048),
        ("input_sequence_length", 2049),
        ("label_sequence_length", 2047),
        ("input_slice", [0, 2049]),
        ("label_slice", [1, 2048]),
        ("effective_target_tokens", 2047),
        ("loss_adapter_id", "causal_lm_self_shifted_v1"),
    ],
)
def test_g3_derived_pile_freezes_2049_to_preshifted_2048_contract(
    field: str, value: object
) -> None:
    files, metadata = _g3_asset_contract(
        "dataset", dataset_storage="derived_pile"
    )
    metadata["storage"]["causal_lm_mapping"][field] = value
    with pytest.raises(AssetValidationError, match="derived Pile|causal_lm_mapping"):
        _build_g3_from_contract("dataset", files, metadata)


def test_g3_replacement_cannot_rewrite_profile_candidate_or_generator_commit() -> None:
    previous = _g3_manifest()
    advanced = transition_manifest(
        previous,
        AssetState.DOWNLOADED,
        actor="test-fetcher",
        actor_role="fetcher",
        actor_instance_id="test-host:fetcher:1",
        evidence_ref="evidence/fetch-complete.json",
        evidence_sha256=_digest(b"fetch-complete"),
        summary="fetch completed",
        at="2026-07-19T00:00:01Z",
    )
    replacements = {
        "contract_profile": "rewritten-profile",
        "candidate_id": "f" * 64,
        "generator_git_commit": "b" * 40,
    }
    for field, value in replacements.items():
        rewritten = deepcopy(advanced)
        rewritten[field] = value
        with pytest.raises(AssetValidationError, match="cannot rewrite"):
            assets_module._validate_replacement(previous, rewritten)


@pytest.mark.parametrize("revision", ["", "unknown", "LATEST", "main", " master "])
def test_manifest_rejects_generic_or_blank_revision(revision: str) -> None:
    with pytest.raises(AssetValidationError, match="revision"):
        compute_asset_id(
            asset_type="model",
            name="bad-revision",
            source="fixture:model",
            revision=revision,
            files=[AssetFile("weights.bin", 1, _digest(b"x"))],
            metadata=_metadata(),
        )


@pytest.mark.parametrize(
    "path",
    [
        "../escape.bin",
        "/absolute.bin",
        "nested\\windows.bin",
        "weights.bin.part",
        "weights.part.meta",
        "asset.lock",
        "tmp/weights.bin",
        "weights.tmp-123",
        "temp/weights.bin",
    ],
)
def test_asset_paths_reject_traversal_and_unfinished_objects(path: str) -> None:
    with pytest.raises(AssetValidationError):
        validate_asset_path(path)


def test_manifest_rejects_case_colliding_paths() -> None:
    with pytest.raises(AssetValidationError, match="case-colliding"):
        compute_asset_id(
            asset_type="dataset",
            name="collision",
            source="huggingface:example/collision",
            revision="deadbeef",
            files=[
                AssetFile("Data.bin", 1, _digest(b"a")),
                AssetFile("data.bin", 1, _digest(b"b")),
            ],
            metadata=_metadata("dataset"),
        )


def test_manifest_source_rejects_query_or_signed_url() -> None:
    with pytest.raises(AssetValidationError, match="signed URL"):
        compute_asset_id(
            asset_type="model",
            name="unsafe-source",
            source="https://example.invalid/model?token=secret",
            revision="deadbeef",
            files=[AssetFile("weights.bin", 1, _digest(b"x"))],
            metadata=_metadata(),
        )


def test_verify_only_checks_size_and_sha_without_changing_state(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    before = deepcopy(downloaded)
    report = verify_only(downloaded, root)
    assert report == {
        "schema_version": "stage0-asset-verification-v1",
        "asset_id": downloaded["asset_id"],
        "state": "downloaded",
        "files_checked": 1,
        "bytes_checked": len(b"immutable fixture\n"),
        "ok": True,
    }
    assert downloaded == before


def test_verify_only_rejects_downloading_and_invalid(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    with pytest.raises(AssetVerificationError, match="acquisition must finish"):
        verify_only(manifest, root)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    invalid = transition_manifest(
        downloaded,
        AssetState.INVALID,
        actor="test-verifier",
        actor_role="verifier",
        evidence_ref="evidence/rejection.json",
        summary="rejected",
    )
    with pytest.raises(AssetVerificationError, match="acquisition must finish"):
        verify_only(invalid, root)


def test_verify_only_reports_size_and_hash_mismatch(tmp_path: Path) -> None:
    size_root, bad_size = _manifest(tmp_path / "size", declared_size=999)
    bad_size = _transition_to(bad_size, AssetState.DOWNLOADED)
    with pytest.raises(AssetVerificationError, match="Size mismatch"):
        verify_only(bad_size, size_root)

    hash_root, bad_hash = _manifest(tmp_path / "hash", declared_hash="0" * 64)
    bad_hash = _transition_to(bad_hash, AssetState.DOWNLOADED)
    with pytest.raises(AssetVerificationError, match="SHA-256 mismatch"):
        verify_only(bad_hash, hash_root)


def test_ready_only_resolver_rejects_other_states_and_detects_drift(
    tmp_path: Path,
) -> None:
    root, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    with pytest.raises(AssetNotReadyError, match="requires state=ready"):
        resolve_ready_asset(downloaded, root)

    ready = _transition_to(manifest, AssetState.READY)
    resolved = resolve_ready_asset(ready, root)
    assert resolved.path_for("weights.bin") == (root / "weights.bin").resolve()
    assert resolved.asset_id == ready["asset_id"]

    with pytest.raises(TypeError):
        resolve_ready_asset(ready, root, verify_hashes=False)  # type: ignore[call-arg]

    (root / "weights.bin").write_bytes(b"tampered fixture!\n")
    assert (root / "weights.bin").stat().st_size == len(b"immutable fixture\n")
    with pytest.raises(AssetVerificationError, match="SHA-256 mismatch"):
        resolve_ready_asset(ready, root)


def test_atomic_manifest_publication_is_canonical_and_no_clobber(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "fixture.json"
    assert publish_manifest_atomic(
        target, manifest, manifest_root=manifest_root
    ) == target
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n")
    assert load_manifest(target) == manifest
    assert not list(target.parent.glob(".*.tmp-*"))
    with pytest.raises(FileExistsError, match="already exists"):
        publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
    assert not list(target.parent.glob(".*.tmp-*"))


def test_atomic_replacement_requires_cas_and_only_advances_history(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "candidate.json"
    publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)

    with pytest.raises(AssetValidationError, match="requires expected"):
        publish_manifest_atomic(
            target,
            downloaded,
            manifest_root=manifest_root,
            allow_replace=True,
        )

    publish_manifest_atomic(
        target,
        downloaded,
        manifest_root=manifest_root,
        allow_replace=True,
        expected_previous_sha256=sha256_file(target),
    )
    assert load_manifest(target)["state"] == "downloaded"

    invalid = transition_manifest(
        downloaded,
        AssetState.INVALID,
        actor="test-verifier",
        actor_role="verifier",
        evidence_ref="evidence/invalid.json",
        summary="invalid candidate",
    )
    publish_manifest_atomic(
        target,
        invalid,
        manifest_root=manifest_root,
        allow_replace=True,
        expected_previous_sha256=sha256_file(target),
    )
    ready_from_original = _transition_to(manifest, AssetState.READY)
    with pytest.raises(AssetValidationError, match="terminal"):
        publish_manifest_atomic(
            target,
            ready_from_original,
            manifest_root=manifest_root,
            allow_replace=True,
            expected_previous_sha256=sha256_file(target),
        )


def test_ready_manifest_is_immutable_at_path_but_invalidation_gets_new_path(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    ready = _transition_to(manifest, AssetState.READY)
    invalidated = transition_manifest(
        ready,
        AssetState.INVALID,
        actor="test-gate",
        actor_role="gate",
        evidence_ref="evidence/post-admission-audit.json",
        summary="post-admission audit failed",
    )
    manifest_root = _manifest_directory(tmp_path)
    ready_path = manifest_root / "ready.json"
    publish_manifest_atomic(ready_path, ready, manifest_root=manifest_root)
    with pytest.raises(AssetValidationError, match="new path"):
        publish_manifest_atomic(
            ready_path,
            invalidated,
            manifest_root=manifest_root,
            allow_replace=True,
            expected_previous_sha256=sha256_file(ready_path),
        )
    assert load_manifest(ready_path)["state"] == "ready"

    invalidation_path = manifest_root / "ready.invalidated.json"
    publish_manifest_atomic(
        invalidation_path, invalidated, manifest_root=manifest_root
    )
    assert load_manifest(invalidation_path)["state"] == "invalid"


def test_replacement_rejects_stale_cas_digest(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "candidate.json"
    publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
    stale_digest = sha256_file(target)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    publish_manifest_atomic(
        target,
        downloaded,
        manifest_root=manifest_root,
        allow_replace=True,
        expected_previous_sha256=stale_digest,
    )
    verified = _transition_to(manifest, AssetState.VERIFIED)
    with pytest.raises(AssetValidationError, match="Stale"):
        publish_manifest_atomic(
            target,
            verified,
            manifest_root=manifest_root,
            allow_replace=True,
            expected_previous_sha256=stale_digest,
        )


def test_concurrent_replacements_allow_exactly_one_cas_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "candidate.json"
    publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
    expected = sha256_file(target)
    first = transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor="fetcher-a",
        actor_role="fetcher",
        evidence_ref="evidence/fetch-a.json",
        summary="fetch A complete",
    )
    second = transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor="fetcher-b",
        actor_role="fetcher",
        evidence_ref="evidence/fetch-b.json",
        summary="fetch B complete",
    )

    original_lock = assets_module._advisory_manifest_lock
    barrier = threading.Barrier(2)

    @contextmanager
    def synchronized_lock(lock_target: Path):
        barrier.wait(timeout=5)
        with original_lock(lock_target):
            yield

    monkeypatch.setattr(
        assets_module, "_advisory_manifest_lock", synchronized_lock
    )

    def publish(candidate: dict[str, object]) -> Path:
        return publish_manifest_atomic(
            target,
            candidate,
            manifest_root=manifest_root,
            allow_replace=True,
            expected_previous_sha256=expected,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, candidate) for candidate in (first, second)]
        successes = 0
        failures: list[BaseException] = []
        for future in futures:
            try:
                future.result(timeout=10)
                successes += 1
            except BaseException as error:
                failures.append(error)

    assert successes == 1
    assert len(failures) == 1
    assert isinstance(failures[0], AssetValidationError)
    assert "Concurrent" in str(failures[0]) or "Stale" in str(failures[0])
    assert load_manifest(target)["state"] == "downloaded"


def test_manifest_advisory_lock_rejects_a_hard_link(tmp_path: Path) -> None:
    _asset_root, manifest = _manifest(tmp_path / "asset")
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "candidate.json"
    victim = tmp_path / "unrelated-empty-file"
    victim.write_bytes(b"")
    lock_path = manifest_root / f".{target.name}.publish.lock"
    lock_path.hardlink_to(victim)

    with pytest.raises(AssetValidationError, match="single-link regular file"):
        publish_manifest_atomic(
            target,
            manifest,
            manifest_root=manifest_root,
            allow_replace=True,
        )

    assert victim.read_bytes() == b""


def test_publication_requires_approved_existing_non_symlink_root(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(AssetValidationError, match="escapes approved"):
        publish_manifest_atomic(
            outside / "escape.json", manifest, manifest_root=manifest_root
        )
    with pytest.raises(AssetValidationError, match="parent must already exist"):
        publish_manifest_atomic(
            manifest_root / "missing" / "candidate.json",
            manifest,
            manifest_root=manifest_root,
        )

    real = manifest_root / "real"
    real.mkdir()
    link = manifest_root / "linked"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a directory symlink is not permitted on this platform")
    with pytest.raises(AssetValidationError, match="symlinks or junctions"):
        publish_manifest_atomic(
            link / "candidate.json", manifest, manifest_root=manifest_root
        )


def test_strict_loader_rejects_bom_duplicate_keys_and_non_finite_json(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + canonical)
    with pytest.raises(AssetEncodingError, match="BOM"):
        load_manifest(bom)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"key":1,"key":2}', encoding="utf-8")
    with pytest.raises(AssetEncodingError, match="Duplicate"):
        load_manifest(duplicate)

    non_finite = tmp_path / "nan.json"
    non_finite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(AssetEncodingError, match="Non-finite"):
        load_manifest(non_finite)


def test_manifest_rejects_broken_history_asset_id_and_missing_metadata(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    broken_history = deepcopy(manifest)
    broken_history["state"] = "ready"
    with pytest.raises(AssetValidationError, match="final state_history"):
        validate_manifest(broken_history)

    broken_id = deepcopy(manifest)
    broken_id["asset_id"] = "0" * 64
    with pytest.raises(AssetValidationError, match="asset_id mismatch"):
        validate_manifest(broken_id)

    missing_metadata = deepcopy(manifest)
    del missing_metadata["metadata"]
    with pytest.raises(AssetValidationError, match="metadata"):
        validate_manifest(missing_metadata)


def test_schema_document_captures_roles_evidence_and_typed_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "schemas" / "stage0-asset-manifest-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schema_version"]["const"] == (
        "stage0-asset-manifest-v1"
    )
    assert "metadata" in schema["required"]
    transition_required = schema["$defs"]["stateTransition"]["required"]
    assert {"actor_role", "evidence_ref"} <= set(transition_required)
    assert {"modelMetadata", "tokenizerMetadata", "datasetMetadata", "sourceMetadata"} <= set(
        schema["$defs"]
    )
    assert schema["properties"]["contract_profile"]["const"] == "stage0-g3-v1"
    assert {"actor_instance_id", "evidence_sha256"} <= set(
        schema["$defs"]["g3StateTransition"]["allOf"][1]["required"]
    )
    assert {
        "g3ModelMetadata",
        "g3TokenizerMetadata",
        "g3DatasetMetadata",
        "g3SourceMetadata",
        "g3RawPileStorage",
        "g3DerivedPileStorage",
        "g3RawParquetStorage",
        "g3TaskContract",
    } <= set(schema["$defs"])
    model_required = set(schema["$defs"]["g3ModelMetadata"]["required"])
    assert {"dtype_counts", "max_position_embeddings"} <= model_required
    tokenizer_required = set(schema["$defs"]["g3TokenizerMetadata"]["required"])
    assert "glue_padding_policy" in tokenizer_required
    assert schema["$defs"]["g3TokenizerMetadata"]["properties"][
        "glue_padding_policy"
    ]["const"] == "explicit_existing_added_token"
    mapping = schema["$defs"]["g3CausalLmMapping"]["properties"]
    assert mapping["source_tokens_per_record"]["const"] == 2049
    assert mapping["input_sequence_length"]["const"] == 2048
    assert mapping["label_sequence_length"]["const"] == 2048
    assert mapping["labels_alignment"]["const"] == "pre_shifted_next_token"
    assert mapping["attention_mask_policy"]["const"] == (
        "all_one_for_fixed_full_record"
    )
    assert mapping["loss_adapter_id"]["const"] == (
        "pre-shifted-next-token-cross-entropy-v1"
    )
    raw_storage = schema["$defs"]["g3RawPileStorage"]
    assert {
        "required_cursor_stop",
        "causal_lm_mapping",
        "reference_reader",
        "reference_batch_size",
        "reference_batch_sha256",
        "last_required_record_sha256",
    } <= set(raw_storage["required"])
    assert raw_storage["properties"]["cross_shard_policy"]["const"] == (
        "explicit_ordered_global_byte_stream_selected_prefix_must_be_fully_covered"
    )
    assert set(
        raw_storage["properties"]["reference_batch_sha256"]["required"]
    ) == {"0", "1", "511"}
    assert schema["$defs"]["g3TaskContract"]["properties"][
        "unlabeled_test_policy"
    ]["const"] == (
        "retain_in_raw_exclude_from_derived_training_and_evaluation"
    )
