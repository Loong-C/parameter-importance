from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import pytest

from param_importance_nlp.asset_layout import layout_artifact_hash
from param_importance_nlp.asset_requirements import requirements_artifact_hash
from param_importance_nlp.assets import (
    AssetFile,
    admit_g3_qualification,
    build_g3_candidate_manifest,
    build_g3_qualification,
    transition_manifest,
)
from param_importance_nlp.contracts import (
    GateRecord,
    canonical_json_bytes,
    canonical_json_hash,
    ensure_json_object,
    loads_strict_json,
)
from param_importance_nlp.g3_gate import (
    G3GateAggregationError,
    GATE_IDS,
    GLUE_PREPROCESSING_PAYLOAD_SCHEMA_VERSION,
    GLUE_PREPROCESSING_VERSION,
    QUALIFICATION_CHECK_IDS_BY_KIND,
    evaluate_stage0_g3,
    g3_candidate_artifact_ref,
    g3_resolution_artifact_hash,
    glue_preprocessing_config_hash,
    glue_preprocessing_config_payload,
    load_legacy_model_manifest_diagnostic,
    load_pile_reference_reader_oracle,
    qualification_check_ids_for,
    validate_stage0_g3_resolution,
)
from param_importance_nlp.g3_semantic_evidence import (
    SEMANTIC_EVIDENCE_SCHEMA_VERSION,
    semantic_evidence_artifact_hash,
)
from param_importance_nlp.g3_lifecycle_evidence import (
    _candidate_from_acquisition_entry,
    _expected_derived_map_fingerprints,
    g3_acquisition_report_ref,
    g3_downloaded_candidate_ref,
    g3_verification_report_ref,
)
from tests.test_g3_asset_publication import _fake_probe


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_REQUIREMENTS = _ROOT / "configs/stage0/g3-asset-requirements-v1.json"
_SOURCE_LAYOUT = _ROOT / "configs/stage0/g3-asset-layout-v1.json"
_CHECKED_AT = "2026-08-03T08:00:00Z"
_GIT_COMMIT = "a" * 40


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    return dict(
        ensure_json_object(
            loads_strict_json(path.read_bytes()),
            field=str(path),
        )
    )


def _payload(logical_name: str, object_path: str) -> bytes:
    return f"fixture:{logical_name}:{object_path}".encode("utf-8")


def _replace_descriptor_payload(
    descriptor: dict[str, Any],
    payload: bytes,
) -> None:
    descriptor["size_bytes"] = len(payload)
    descriptor["sha256"] = _digest(payload)


def _shrink_requirements(
    requirements: dict[str, Any],
) -> dict[tuple[str, str, str], bytes]:
    payloads: dict[tuple[str, str, str], bytes] = {}
    for model in requirements["models"]:
        for descriptor in model["files"]:
            payload = _payload(model["name"], descriptor["path"])
            _replace_descriptor_payload(descriptor, payload)
            payloads[("model", model["name"], descriptor["path"])] = payload

    tokenizer = requirements["tokenizer"]
    tokenizer_payloads = {
        "special_tokens_map.json": canonical_json_bytes(
            {
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            }
        ),
        "tokenizer.json": canonical_json_bytes(
            {"version": "1.0", "normalizer": None}
        ),
        "tokenizer_config.json": canonical_json_bytes(
            {"tokenizer_class": tokenizer["tokenizer_class"]}
        ),
    }
    for descriptor in tokenizer["files"]:
        payload = tokenizer_payloads[descriptor["path"]]
        _replace_descriptor_payload(descriptor, payload)
        payloads[("tokenizer", tokenizer["name"], descriptor["path"])] = payload

    pile = requirements["pile"]
    index_payload = _payload("pile-selected-prefix", pile["index"]["path"])
    _replace_descriptor_payload(pile["index"], index_payload)
    payloads[("pile", "pile", pile["index"]["path"])] = index_payload
    pile["required_cursor_stop"] = 513
    pile["required_raw_tokens"] = 513 * 2049
    pile["required_target_tokens"] = 513 * 2048
    interval_bounds = {
        "debug": (0, 1),
        "train": (0, 1),
        "validation": (1, 2),
        "sampling_universe": (2, 3),
        "probe": (3, 511),
        "reserve": (511, 513),
    }
    for interval in pile["cursor_intervals"]:
        interval["start"], interval["stop"] = interval_bounds[interval["name"]]
    for workload in pile["workloads"]:
        workload["required_unique_records"] = 1
        workload["required_target_tokens"] = 2048
        if workload["max_steps"] is not None:
            workload["max_steps"] = 1
            workload["global_batch_size"] = 1
    shard_payload = b"\0" * (
        513 * 2049 * pile["index_contract"]["dtype_bytes"]
    )
    shard = pile["selected_shards"][0]
    _replace_descriptor_payload(shard, shard_payload)
    shard["byte_start"] = 0
    shard["byte_stop"] = len(shard_payload)
    payloads[("pile", "pile", shard["path"])] = shard_payload

    oracle = pile["reference_reader_oracle"]
    oracle_payload = canonical_json_bytes(
        {
            "official_source": f"/fixture/{oracle['official_source_ref']}",
            "documents": 1,
            "batches": [
                {
                    "step": int(step),
                    "shape": [pile["reference_batch_size"], 2049],
                    "independent_sha256": digest,
                    "official_batch_viewer_sha256": digest,
                    "equal": True,
                }
                for step, digest in pile["reference_batch_sha256"].items()
            ],
            "status": "ok",
        }
    )
    oracle["artifact_sha256"] = _digest(oracle_payload)
    payloads[("control", "", oracle["artifact_ref"])] = oracle_payload

    legacy_model = next(
        item
        for item in requirements["models"]
        if item["name"] == "pythia-31m-deduped-step0"
    )
    legacy = legacy_model["legacy_manifest_diagnostic"]
    legacy_payload = b"\xef\xbb\xbf" + canonical_json_bytes(
        {"legacy_model": legacy_model["name"], "fixture": True}
    )
    legacy["size_bytes"] = len(legacy_payload)
    legacy["sha256"] = _digest(legacy_payload)
    for reference in legacy["refs"]:
        payloads[("control", "", reference)] = legacy_payload

    for task in requirements["glue"]:
        for descriptor in task["raw_files"]:
            payload = _payload(f"glue-{task['task']}-raw", descriptor["path"])
            _replace_descriptor_payload(descriptor, payload)
            payloads[("glue_raw", task["task"], descriptor["path"])] = payload

    requirements["artifact_hash"] = requirements_artifact_hash(requirements)
    return payloads


def _requirement_for(
    requirements: dict[str, Any],
    kind: str,
    name: str,
) -> dict[str, Any]:
    if kind == "model":
        return next(item for item in requirements["models"] if item["name"] == name)
    if kind == "tokenizer":
        return requirements["tokenizer"]
    if kind == "pile":
        return requirements["pile"]
    return next(item for item in requirements["glue"] if item["task"] == name)


def _preprocessing(
    version: str,
    config_hash: str,
    *,
    tokenizer_asset_id: str | None = None,
    parent_asset_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": version,
        "config_hash": config_hash,
        "code_git_commit": _GIT_COMMIT,
        "tokenizer_asset_id": tokenizer_asset_id,
        "parent_asset_ids": parent_asset_ids or [],
    }


def _model_metadata(
    requirement: dict[str, Any],
    *,
    parent_asset_id: str | None,
    wrong_semantics: bool,
) -> dict[str, Any]:
    config = next(item for item in requirement["files"] if item["role"] == "config")
    step = int(requirement["checkpoint"][4:])
    return {
        "contract_version": "stage0-model-metadata-v1",
        "architecture": (
            "WrongFixtureArchitecture"
            if wrong_semantics
            else requirement["architecture"]
        ),
        "parameter_count": requirement["parameter_count"],
        "tensor_count": requirement["tensor_count"],
        "dtype": next(iter(requirement["dtype_counts"])),
        "dtype_counts": requirement["dtype_counts"],
        "max_position_embeddings": requirement["max_position_embeddings"],
        "config_path": config["path"],
        "config_sha256": config["sha256"],
        "initialization_id": (
            f"{requirement['repository']}@{requirement['revision']}:"
            f"{requirement['checkpoint']}"
        ),
        "initialization_kind": (
            "base_initialization" if step == 0 else "trained_checkpoint"
        ),
        "training_step": step,
        "parent_model_asset_id": parent_asset_id,
    }


def _tokenizer_metadata(requirement: dict[str, Any]) -> dict[str, Any]:
    config = next(
        item for item in requirement["files"] if item["role"] == "tokenizer_config"
    )
    special_tokens = {
        f"{name}_token": {
            "token": "<|endoftext|>",
            "token_id": token_id,
        }
        for name, token_id in requirement["special_token_ids"].items()
    }
    special_tokens["glue_padding_token"] = {
        "token": requirement["glue_padding_token"]["token"],
        "token_id": requirement["glue_padding_token"]["token_id"],
    }
    return {
        "contract_version": "stage0-tokenizer-metadata-v1",
        "tokenizer_class": requirement["tokenizer_class"],
        "implementation_version": "tokenizer-json-1.0",
        "vocab_size": requirement["vocab_size"],
        "token_count_with_added_tokens": requirement[
            "token_count_with_added_tokens"
        ],
        "vocab_mapping_sha256": requirement["vocab_mapping_sha256"],
        "glue_padding_policy": requirement["glue_padding_token"]["policy"],
        "special_tokens": special_tokens,
        "normalization": "identity",
        "config_path": config["path"],
        "config_sha256": config["sha256"],
        "model_max_length": 2048,
    }


def _pile_metadata(requirement: dict[str, Any]) -> dict[str, Any]:
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
    causal = requirement["causal_lm_contract"]
    causal_mapping = {
        "labels_alignment": causal["labels_alignment"],
        "source_tokens_per_record": causal["source_tokens_per_record"],
        "input_sequence_length": causal["input_sequence_length"],
        "label_sequence_length": causal["target_sequence_length"],
        "input_slice": causal["input_slice"],
        "label_slice": causal["target_slice"],
        "attention_mask_policy": causal["attention_mask_policy"],
        "effective_target_tokens": causal["effective_target_tokens_per_record"],
        "loss_adapter_id": causal["loss_adapter_id"],
    }
    return {
        "contract_version": "stage0-dataset-metadata-v1",
        "dataset_kind": "raw_indexed_mmap",
        "raw_revision": requirement["revision"],
        "splits": splits,
        "preprocessing_version": "pile-raw-fixture-v1",
        "preprocessing": _preprocessing(
            "pile-raw-fixture-v1",
            canonical_json_hash(requirement["causal_lm_contract"]),
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
            "causal_lm_mapping": causal_mapping,
            "reference_reader": requirement["reference_reader"],
            "reference_batch_size": requirement["reference_batch_size"],
            "reference_batch_sha256": requirement["reference_batch_sha256"],
            "last_required_record_sha256": requirement[
                "last_required_record_sha256"
            ],
            "cross_shard_policy": requirement["cross_shard_policy"],
            "shards": shards,
        },
    }


def _glue_splits(requirement: dict[str, Any], *, derived: bool) -> dict[str, Any]:
    names = (
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
        for name in names
    }


def _glue_metadata(
    requirement: dict[str, Any],
    *,
    derived: bool,
    tokenizer_asset_id: str | None,
    parent_asset_id: str | None,
    wrong_lineage: bool,
) -> dict[str, Any]:
    version = GLUE_PREPROCESSING_VERSION
    config_hash = glue_preprocessing_config_hash(requirement)
    task_contract = {
        "task": requirement["task"],
        "text_fields": requirement["text_fields"],
        "label_mapping": requirement["label_mapping"],
        "unlabeled_test_policy": requirement["unlabeled_test_policy"],
    }
    if derived:
        metadata = {
            "contract_version": "stage0-dataset-metadata-v1",
            "dataset_kind": "derived_pretokenized",
            "raw_revision": requirement["revision"],
            "splits": _glue_splits(requirement, derived=True),
            "preprocessing_version": version,
            "preprocessing": _preprocessing(
                version,
                config_hash,
                tokenizer_asset_id=(
                    "f" * 64 if wrong_lineage else tokenizer_asset_id
                ),
                parent_asset_ids=[
                    "e" * 64 if wrong_lineage else parent_asset_id
                ],
            ),
            "task_contract": task_contract,
            "storage": {
                "kind": "hf_load_from_disk",
                "format_version": "fixture-datasets-v1",
            },
        }
        return metadata
    storage_splits = {
        split: [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in requirement["raw_files"]
            if item["role"] == split
        ]
        for split in requirement["split_counts"]
    }
    return {
        "contract_version": "stage0-dataset-metadata-v1",
        "dataset_kind": "hf_raw_parquet",
        "raw_revision": requirement["revision"],
        "splits": _glue_splits(requirement, derived=False),
        "preprocessing_version": version,
        "preprocessing": _preprocessing(version, config_hash),
        "task_contract": task_contract,
        "storage": {
            "kind": "hf_raw_parquet_files",
            "splits": storage_splits,
        },
    }


@dataclass(frozen=True)
class _Fixture:
    requirements_path: Path
    layout_path: Path
    data_root: Path
    object_paths: dict[tuple[str, str], list[Path]]
    verification_paths: dict[tuple[str, str], Path]
    semantic_paths: dict[tuple[str, str], Path]
    candidate_paths: dict[tuple[str, str], Path]


def _materialize_fixture(
    tmp_path: Path,
    *,
    wrong_semantic_model: str | None = None,
    wrong_pile_batch_hash: bool = False,
    wrong_lineage_derived: str | None = None,
    wrong_qualification_checks_for: str | None = None,
    wrong_semantic_binding_for: str | None = None,
    bad_semantic_details: str | None = None,
    bad_tokenizer_identity: str | None = None,
) -> _Fixture:
    requirements = deepcopy(_load_object(_SOURCE_REQUIREMENTS))
    payloads = _shrink_requirements(requirements)
    layout = deepcopy(_load_object(_SOURCE_LAYOUT))
    layout["requirements_sha256"] = requirements["artifact_hash"]
    layout["artifact_hash"] = layout_artifact_hash(layout)

    control = tmp_path / "control"
    control.mkdir()
    requirements_path = control / "requirements.json"
    layout_path = control / "layout.json"
    requirements_path.write_bytes(canonical_json_bytes(requirements))
    layout_path.write_bytes(canonical_json_bytes(layout))
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    for (kind, _name, relative), payload in payloads.items():
        if kind != "control":
            continue
        target = data_root.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    pile_reference_reader_oracle = load_pile_reference_reader_oracle(
        data_root, requirements["pile"]
    )
    legacy_manifest_replacement = load_legacy_model_manifest_diagnostic(
        data_root,
        next(
            item
            for item in requirements["models"]
            if item["name"] == "pythia-31m-deduped-step0"
        ),
    )

    ready_manifests: dict[tuple[str, str], dict[str, Any]] = {}
    object_paths: dict[tuple[str, str], list[Path]] = {}
    verification_paths: dict[tuple[str, str], Path] = {}
    semantic_paths: dict[tuple[str, str], Path] = {}
    candidate_paths: dict[tuple[str, str], Path] = {}
    for entry in layout["entries"]:
        kind = entry["kind"]
        requirement_name = entry["requirement_name"]
        requirement = _requirement_for(requirements, kind, requirement_name)
        if kind == "model":
            parent_id = None
            if requirement["checkpoint"] != "step0":
                parent = next(
                    item
                    for item in requirements["models"]
                    if item["repository"] == requirement["repository"]
                    and item["checkpoint"] == "step0"
                )
                parent_id = ready_manifests[("model", parent["name"])]["asset_id"]
            metadata = _model_metadata(
                requirement,
                parent_asset_id=parent_id,
                wrong_semantics=entry["logical_name"] == wrong_semantic_model,
            )
            descriptors = requirement["files"]
            asset_type = "model"
        elif kind == "tokenizer":
            metadata = _tokenizer_metadata(requirement)
            if bad_tokenizer_identity == "implementation":
                metadata["implementation_version"] = "tokenizer-json-9.9"
            elif bad_tokenizer_identity == "normalization":
                metadata["normalization"] = "tokenizer-json:" + "f" * 64
            elif bad_tokenizer_identity == "token_text":
                metadata["special_tokens"]["bos_token"]["token"] = (
                    "<wrong-bos>"
                )
            elif bad_tokenizer_identity is not None:
                raise AssertionError("unknown bad tokenizer identity mode")
            descriptors = requirement["files"]
            asset_type = "tokenizer"
        elif kind == "pile":
            metadata = _pile_metadata(requirement)
            if wrong_pile_batch_hash:
                metadata["storage"]["reference_batch_sha256"]["0"] = "f" * 64
            descriptors = [
                requirement["index"],
                *requirement["selected_shards"],
            ]
            asset_type = "dataset"
        elif kind == "glue_raw":
            metadata = _glue_metadata(
                requirement,
                derived=False,
                tokenizer_asset_id=None,
                parent_asset_id=None,
                wrong_lineage=False,
            )
            descriptors = requirement["raw_files"]
            asset_type = "dataset"
        else:
            raw = ready_manifests[("glue_raw", requirement_name)]
            tokenizer = ready_manifests[("tokenizer", "pythia-tokenizer")]
            metadata = _glue_metadata(
                requirement,
                derived=True,
                tokenizer_asset_id=tokenizer["asset_id"],
                parent_asset_id=raw["asset_id"],
                wrong_lineage=(
                    entry["logical_name"] == wrong_lineage_derived
                ),
            )
            derived_payload = _payload(entry["logical_name"], "dataset/state.json")
            descriptors = [
                {
                    "path": "dataset/state.json",
                    "size_bytes": len(derived_payload),
                    "sha256": _digest(derived_payload),
                    "role": "dataset_state",
                }
            ]
            payloads[(kind, requirement_name, "dataset/state.json")] = derived_payload
            asset_type = "dataset"

        files = [
            AssetFile(
                descriptor["path"],
                descriptor["size_bytes"],
                descriptor["sha256"],
                descriptor["role"],
            )
            for descriptor in descriptors
        ]
        acquisition_ref = f"evidence/{entry['logical_name']}-acquisition.json"
        acquisition_sha = _digest(
            f"{entry['logical_name']}:acquisition".encode("utf-8")
        )
        candidate = build_g3_candidate_manifest(
            asset_type=asset_type,
            name=entry["logical_name"],
            source=requirement["repository"],
            revision=requirement["revision"],
            files=files,
            actor="fixture-fetcher",
            actor_role="fetcher",
            actor_instance_id="fixture-host:fetcher:1",
            evidence_ref=acquisition_ref,
            evidence_sha256=acquisition_sha,
            generator_version="tests/g3-gate-v1",
            generator_git_commit=_GIT_COMMIT,
            metadata=metadata,
            created_at="2026-08-03T07:00:00Z",
        )
        downloaded = transition_manifest(
            candidate,
            "downloaded",
            actor="fixture-fetcher",
            actor_role="fetcher",
            actor_instance_id="fixture-host:fetcher:1",
            evidence_ref=acquisition_ref,
            evidence_sha256=acquisition_sha,
            summary="fixture acquisition completed",
            at="2026-08-03T07:00:01Z",
        )
        verification_ref = f"evidence/{entry['logical_name']}-verification.json"
        verification_report = {
            "schema_version": "stage0-asset-verification-v1",
            "asset_id": downloaded["asset_id"],
            "state": "downloaded",
            "files_checked": len(descriptors),
            "bytes_checked": sum(item["size_bytes"] for item in descriptors),
            "ok": True,
        }
        verification_bytes = canonical_json_bytes(verification_report)
        verification_sha = _digest(verification_bytes)
        verified = transition_manifest(
            downloaded,
            "verified",
            actor="fixture-verifier",
            actor_role="verifier",
            actor_instance_id="fixture-host:verifier:1",
            evidence_ref=verification_ref,
            evidence_sha256=verification_sha,
            summary="fixture size, hash and semantic verification passed",
            at="2026-08-03T07:00:02Z",
        )
        expected_check_ids = qualification_check_ids_for(kind)
        semantic_check_ids = tuple(
            check_id
            for check_id in expected_check_ids
            if check_id != "full_file_integrity"
        )
        semantic_ref = f"evidence/{entry['logical_name']}-semantic.json"
        details_by_check = _fake_probe(
            kind,
            requirement,
            data_root,
            verified,
            legacy_manifest_replacement=(
                legacy_manifest_replacement
                if kind == "model"
                and "legacy_manifest_diagnostic" in requirement
                else None
            ),
            reference_reader_oracle=(
                pile_reference_reader_oracle
                if kind == "pile"
                else None
            ),
        )
        if (
            entry["logical_name"] == "pythia-14m-step0"
            and bad_semantic_details is not None
        ):
            model_details = details_by_check["model_semantic_contract"]
            if bad_semantic_details == "missing":
                model_details.pop("architecture")
            elif bad_semantic_details == "fixture":
                details_by_check["model_semantic_contract"] = {
                    "fixture": True
                }
            elif bad_semantic_details == "wrong":
                model_details["architecture"] = "WrongFixtureArchitecture"
            else:
                raise AssertionError("unknown bad semantic-details mode")
        semantic_evidence: dict[str, Any] = {
            "schema_version": SEMANTIC_EVIDENCE_SCHEMA_VERSION,
            "formal": True,
            "asset_id": verified["asset_id"],
            "candidate_id": (
                "f" * 64
                if entry["logical_name"] == wrong_semantic_binding_for
                else verified["candidate_id"]
            ),
            "logical_name": entry["logical_name"],
            "kind": kind,
            "requirements_ref": layout["requirements_ref"],
            "requirements_sha256": requirements["artifact_hash"],
            "checks": [
                {
                    "check_id": check_id,
                    "status": "PASS",
                    "summary": f"fixture {check_id} passed",
                    "details": details_by_check[check_id],
                }
                for check_id in semantic_check_ids
            ],
            "network_attempts": 0,
            "checked_at": "2026-08-03T07:00:03Z",
            "generator_git_commit": _GIT_COMMIT,
        }
        semantic_evidence["artifact_hash"] = semantic_evidence_artifact_hash(
            semantic_evidence
        )
        semantic_bytes = canonical_json_bytes(semantic_evidence)
        semantic_sha = _digest(semantic_bytes)

        qualification_check_ids = expected_check_ids
        if entry["logical_name"] == wrong_qualification_checks_for:
            qualification_check_ids = qualification_check_ids[:-1]
        qualification = build_g3_qualification(
            verified,
            requirements_ref=layout["requirements_ref"],
            requirements_sha256=requirements["artifact_hash"],
            acquisition_ref=acquisition_ref,
            acquisition_sha256=acquisition_sha,
            verification_ref=verification_ref,
            verification_sha256=verification_sha,
            checks=[
                {
                    "check_id": check_id,
                    "status": "PASS",
                    "evidence_ref": (
                        verification_ref
                        if check_id == "full_file_integrity"
                        else semantic_ref
                    ),
                    "evidence_sha256": (
                        verification_sha
                        if check_id == "full_file_integrity"
                        else semantic_sha
                    ),
                    "summary": f"fixture {check_id} passed",
                }
                for check_id in qualification_check_ids
            ],
            checked_at="2026-08-03T07:00:03Z",
            generator_git_commit=_GIT_COMMIT,
        )
        ready = admit_g3_qualification(
            verified,
            qualification,
            qualification_ref=entry["qualification_ref"],
            requirements_artifact_hash=requirements["artifact_hash"],
            actor="fixture-gate",
            actor_instance_id="fixture-host:gate:1",
            at="2026-08-03T07:00:04Z",
        )
        ready_manifests[(kind, requirement_name)] = ready

        asset_root = data_root.joinpath(*entry["asset_root_ref"].split("/"))
        asset_root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for descriptor in descriptors:
            object_path = asset_root.joinpath(*descriptor["path"].split("/"))
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_path.write_bytes(
                payloads[(kind, requirement_name, descriptor["path"])]
            )
            paths.append(object_path)
        object_paths[(kind, requirement_name)] = paths

        manifest_path = data_root.joinpath(*entry["manifest_ref"].split("/"))
        qualification_path = data_root.joinpath(
            *entry["qualification_ref"].split("/")
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        qualification_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(canonical_json_bytes(ready))
        qualification_path.write_bytes(canonical_json_bytes(qualification))
        verification_path = data_root.joinpath(*verification_ref.split("/"))
        verification_path.parent.mkdir(parents=True, exist_ok=True)
        verification_path.write_bytes(verification_bytes)
        verification_paths[(kind, requirement_name)] = verification_path
        semantic_path = data_root.joinpath(*semantic_ref.split("/"))
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_path.write_bytes(semantic_bytes)
        semantic_paths[(kind, requirement_name)] = semantic_path
        candidate_ref = g3_candidate_artifact_ref(
            entry["logical_name"], verified["candidate_id"]
        )
        candidate_path = data_root.joinpath(*candidate_ref.split("/"))
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_bytes(canonical_json_bytes(verified))
        candidate_paths[(kind, requirement_name)] = candidate_path

    # Upgrade the legacy per-entry fixture construction above into the current
    # single acquisition / single independent verify-only lifecycle.  Keeping
    # the first pass preserves all semantic-mutation knobs while this second
    # pass gives every READY manifest the exact production history shape.
    download_report_ref = "evidence/fixture-download-report.json"
    download_report_path = data_root.joinpath(*download_report_ref.split("/"))
    download_report_path.parent.mkdir(parents=True, exist_ok=True)
    download_report_path.write_bytes(canonical_json_bytes({"fixture": True}))
    acquisition_entries: list[dict[str, Any]] = []
    canonical_names = {
        "pythia-410m-deduped-step0",
        "pythia-tokenizer",
        "glue-mnli-raw",
        "glue-rte-raw",
    }
    for entry in layout["entries"]:
        identity = (entry["kind"], entry["requirement_name"])
        old_ready = ready_manifests[identity]
        mode = (
            "derived-build"
            if entry["kind"] == "glue_derived"
            else "canonical-plan"
            if entry["logical_name"] in canonical_names
            else "existing-import"
        )
        requirement = _requirement_for(
            requirements, entry["kind"], entry["requirement_name"]
        )
        if mode == "canonical-plan":
            descriptor = old_ready["files"][0]
            source_evidence = {
                "schema_version": "stage0-g3-canonical-plan-source-v1",
                "object_count": 1,
                "objects": [
                    {
                        "object_id": f"huggingface/fixture/g3/{entry['logical_name']}",
                        "spec_ref": f"configs/stage0/http-objects/{entry['logical_name']}.json",
                        "revision": old_ready["revision"],
                        "size_bytes": descriptor["size_bytes"],
                        "sha256": descriptor["sha256"],
                        "result_status": "already_ready",
                        "network_accessed": False,
                    }
                ],
            }
        elif mode == "existing-import":
            source_evidence = {
                "schema_version": "stage0-g3-existing-import-source-v1",
                "requirements_ref": layout["requirements_ref"],
                "requirements_sha256": requirements["artifact_hash"],
                "declared_file_count": len(old_ready["files"]),
                "declared_bytes": sum(
                    item["size_bytes"] for item in old_ready["files"]
                ),
            }
        else:
            preprocessing = old_ready["metadata"]["preprocessing"]
            tokenizer_inventory = [
                deepcopy(item) for item in requirements["tokenizer"]["files"]
            ]
            derived_splits = requirement["preprocessing"]["derived_splits"]
            source_evidence = {
                "schema_version": "stage0-g3-derived-build-source-v1",
                "task": entry["requirement_name"],
                "raw_asset_id": preprocessing["parent_asset_ids"][0],
                "tokenizer_asset_id": preprocessing["tokenizer_asset_id"],
                "tokenizer_descriptor_inventory": tokenizer_inventory,
                "tokenizer_descriptor_inventory_hash": canonical_json_hash(
                    tokenizer_inventory
                ),
                "target_ref": entry["asset_root_ref"],
                "generator_git_commit": _GIT_COMMIT,
                "preprocessing_version": preprocessing["version"],
                "preprocessing_config_hash": preprocessing["config_hash"],
                "requirement_hash": canonical_json_hash(requirement),
                "derived_splits": deepcopy(derived_splits),
                "split_counts": {
                    split: requirement["split_counts"][split]
                    for split in derived_splits
                },
                "map_fingerprints": _expected_derived_map_fingerprints(
                    requirement,
                    raw_asset_id=preprocessing["parent_asset_ids"][0],
                    tokenizer_asset_id=preprocessing["tokenizer_asset_id"],
                    tokenizer_descriptor_inventory_hash=canonical_json_hash(
                        tokenizer_inventory
                    ),
                    generator_git_commit=_GIT_COMMIT,
                    preprocessing_config_hash=preprocessing["config_hash"],
                ),
                "file_inventory": [
                    {
                        "path": item["path"],
                        "size_bytes": item["size_bytes"],
                    }
                    for item in old_ready["files"]
                ],
                "network_attempts": 0,
            }
        acquisition_entries.append(
            {
                "logical_name": entry["logical_name"],
                "kind": entry["kind"],
                "requirement_name": entry["requirement_name"],
                "asset_root_ref": entry["asset_root_ref"],
                "mode": mode,
                "asset_type": old_ready["asset_type"],
                "name": old_ready["name"],
                "source": old_ready["source"],
                "revision": old_ready["revision"],
                "asset_id": old_ready["asset_id"],
                "files": deepcopy(old_ready["files"]),
                "metadata": deepcopy(old_ready["metadata"]),
                "source_evidence": source_evidence,
            }
        )
    acquisition_payload: dict[str, Any] = {
        "schema_version": "stage0-g3-acquisition-report-v1",
        "formal": True,
        "status": "PASS",
        "started_at": "2026-08-03T07:00:00Z",
        "completed_at": "2026-08-03T07:00:01Z",
        "actor": "fixture-fetcher",
        "actor_role": "fetcher",
        "actor_instance_id": "fixture-host:fetcher:1",
        "source_git_commit": _GIT_COMMIT,
        "requirements_ref": layout["requirements_ref"],
        "requirements_sha256": requirements["artifact_hash"],
        "layout_ref": "configs/stage0/g3-asset-layout-v1.json",
        "layout_sha256": layout["artifact_hash"],
        "download_plan_ref": "configs/stage0/g3-download-plan-v1.json",
        "download_plan_sha256": "d" * 64,
        "download_report_ref": download_report_ref,
        "download_report_sha256": _digest(download_report_path.read_bytes()),
        "runtime_urls_persisted": False,
        "entry_count": 13,
        "entries": acquisition_entries,
    }
    acquisition = acquisition_payload | {
        "artifact_hash": canonical_json_hash(acquisition_payload)
    }
    acquisition_ref = g3_acquisition_report_ref(acquisition["artifact_hash"])
    acquisition_path = data_root.joinpath(*acquisition_ref.split("/"))
    acquisition_path.parent.mkdir(parents=True, exist_ok=True)
    acquisition_path.write_bytes(canonical_json_bytes(acquisition))

    downloaded_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    verification_entries: list[dict[str, Any]] = []
    for entry, acquisition_entry in zip(layout["entries"], acquisition_entries):
        identity = (entry["kind"], entry["requirement_name"])
        downloaded = _candidate_from_acquisition_entry(
            acquisition, acquisition_entry
        )
        downloaded_by_identity[identity] = downloaded
        downloaded_ref = g3_downloaded_candidate_ref(
            entry["logical_name"], downloaded["candidate_id"]
        )
        downloaded_path = data_root.joinpath(*downloaded_ref.split("/"))
        downloaded_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_bytes = canonical_json_bytes(downloaded)
        downloaded_path.write_bytes(downloaded_bytes)
        observations = [
            {
                "path": descriptor["path"],
                "expected_size": descriptor["size_bytes"],
                "expected_sha256": descriptor["sha256"],
                "observed_size": descriptor["size_bytes"],
                "observed_sha256": descriptor["sha256"],
                "status": "PASS",
            }
            for descriptor in downloaded["files"]
        ]
        verification_entries.append(
            {
                "logical_name": entry["logical_name"],
                "asset_id": downloaded["asset_id"],
                "candidate_id": downloaded["candidate_id"],
                "downloaded_manifest_ref": downloaded_ref,
                "downloaded_manifest_sha256": _digest(downloaded_bytes),
                "files": observations,
                "files_checked": len(observations),
                "bytes_checked": sum(
                    item["observed_size"] for item in observations
                ),
                "status": "PASS",
            }
        )
    verification_payload: dict[str, Any] = {
        "schema_version": "stage0-g3-verify-only-report-v1",
        "formal": True,
        "status": "PASS",
        "checked_at": "2026-08-03T07:00:02Z",
        "actor": "fixture-verifier",
        "actor_role": "verifier",
        "actor_instance_id": "fixture-host:verifier:1",
        "generator_git_commit": _GIT_COMMIT,
        "requirements_ref": layout["requirements_ref"],
        "requirements_sha256": requirements["artifact_hash"],
        "layout_ref": "configs/stage0/g3-asset-layout-v1.json",
        "layout_sha256": layout["artifact_hash"],
        "acquisition_ref": acquisition_ref,
        "acquisition_sha256": acquisition["artifact_hash"],
        "network_attempts": 0,
        "entry_count": 13,
        "entries": verification_entries,
    }
    verification = verification_payload | {
        "artifact_hash": canonical_json_hash(verification_payload)
    }
    verification_ref = g3_verification_report_ref(acquisition["artifact_hash"])
    verification_path = data_root.joinpath(*verification_ref.split("/"))
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_bytes(canonical_json_bytes(verification))

    for entry in layout["entries"]:
        identity = (entry["kind"], entry["requirement_name"])
        downloaded = downloaded_by_identity[identity]
        verified = transition_manifest(
            downloaded,
            "verified",
            actor="fixture-verifier",
            actor_role="verifier",
            actor_instance_id="fixture-host:verifier:1",
            evidence_ref=verification_ref,
            evidence_sha256=verification["artifact_hash"],
            summary="independent verify-only process matched every declared file",
            at="2026-08-03T07:00:02Z",
        )
        old_semantic_path = semantic_paths[identity]
        semantic_evidence = _load_object(old_semantic_path)
        semantic_evidence["asset_id"] = verified["asset_id"]
        semantic_evidence["candidate_id"] = (
            "f" * 64
            if entry["logical_name"] == wrong_semantic_binding_for
            else verified["candidate_id"]
        )
        semantic_evidence["artifact_hash"] = semantic_evidence_artifact_hash(
            semantic_evidence
        )
        semantic_bytes = canonical_json_bytes(semantic_evidence)
        old_semantic_path.write_bytes(semantic_bytes)
        semantic_sha = _digest(semantic_bytes)
        old_qualification_path = data_root.joinpath(
            *entry["qualification_ref"].split("/")
        )
        old_qualification = _load_object(old_qualification_path)
        check_ids = [item["check_id"] for item in old_qualification["checks"]]
        semantic_ref = next(
            item["evidence_ref"]
            for item in old_qualification["checks"]
            if item["check_id"] != "full_file_integrity"
        )
        qualification = build_g3_qualification(
            verified,
            requirements_ref=layout["requirements_ref"],
            requirements_sha256=requirements["artifact_hash"],
            acquisition_ref=acquisition_ref,
            acquisition_sha256=acquisition["artifact_hash"],
            verification_ref=verification_ref,
            verification_sha256=verification["artifact_hash"],
            checks=[
                {
                    "check_id": check_id,
                    "status": "PASS",
                    "evidence_ref": (
                        verification_ref
                        if check_id == "full_file_integrity"
                        else semantic_ref
                    ),
                    "evidence_sha256": (
                        verification["artifact_hash"]
                        if check_id == "full_file_integrity"
                        else semantic_sha
                    ),
                    "summary": f"fixture {check_id} passed",
                }
                for check_id in check_ids
            ],
            checked_at="2026-08-03T07:00:03Z",
            generator_git_commit=_GIT_COMMIT,
        )
        ready = admit_g3_qualification(
            verified,
            qualification,
            qualification_ref=entry["qualification_ref"],
            requirements_artifact_hash=requirements["artifact_hash"],
            actor="fixture-gate",
            actor_instance_id="fixture-host:gate:1",
            at="2026-08-03T07:00:04Z",
        )
        ready_manifests[identity] = ready
        manifest_path = data_root.joinpath(*entry["manifest_ref"].split("/"))
        manifest_path.write_bytes(canonical_json_bytes(ready))
        old_qualification_path.write_bytes(canonical_json_bytes(qualification))
        candidate_ref = g3_candidate_artifact_ref(
            entry["logical_name"], verified["candidate_id"]
        )
        candidate_path = data_root.joinpath(*candidate_ref.split("/"))
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_bytes(canonical_json_bytes(verified))
        candidate_paths[identity] = candidate_path
        verification_paths[identity] = verification_path

    return _Fixture(
        requirements_path=requirements_path,
        layout_path=layout_path,
        data_root=data_root,
        object_paths=object_paths,
        verification_paths=verification_paths,
        semantic_paths=semantic_paths,
        candidate_paths=candidate_paths,
    )


def _gates_by_id(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {gate["gate_id"]: gate for gate in audit["gates"]}


def test_shared_glue_preprocessing_hash_payload_is_task_bound() -> None:
    requirement = _load_object(_SOURCE_REQUIREMENTS)["glue"][1]

    payload = glue_preprocessing_config_payload(requirement)

    assert payload == {
        "schema_version": GLUE_PREPROCESSING_PAYLOAD_SCHEMA_VERSION,
        "task": requirement["task"],
        "text_fields": requirement["text_fields"],
        "label_mapping": requirement["label_mapping"],
        "unlabeled_test_policy": requirement["unlabeled_test_policy"],
        "preprocessing": requirement["preprocessing"],
    }
    assert glue_preprocessing_config_hash(requirement) == canonical_json_hash(
        payload
    )


def test_qualification_check_ids_are_exact_and_public() -> None:
    assert dict(QUALIFICATION_CHECK_IDS_BY_KIND) == {
        "model": (
            "full_file_integrity",
            "model_semantic_contract",
            "offline_model_load",
        ),
        "tokenizer": (
            "full_file_integrity",
            "offline_tokenizer_load",
            "tokenizer_semantic_contract",
        ),
        "pile": (
            "full_file_integrity",
            "pile_causal_lm_contract",
            "pile_cursor_coverage",
            "pile_index_contract",
        ),
        "glue_raw": (
            "full_file_integrity",
            "glue_task_contract",
            "offline_glue_raw_load",
        ),
        "glue_derived": (
            "full_file_integrity",
            "glue_preprocessing_lineage",
            "glue_task_contract",
            "offline_glue_derived_load",
        ),
    }
    assert qualification_check_ids_for("pile") == (
        "full_file_integrity",
        "pile_causal_lm_contract",
        "pile_cursor_coverage",
        "pile_index_contract",
    )


def test_g3_aggregator_resolves_all_entries_and_emits_deterministic_passes(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)

    first = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )
    second = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    assert first == second
    assert first["status"] == "PASS"
    assert len(first["entries"]) == 13
    assert all(entry["status"] == "PASS" for entry in first["entries"])
    assert all(
        entry["checks"]["semantic_evidence_matches"] is True
        and entry["semantic_evidence_artifact_hash"] is not None
        for entry in first["entries"]
    )
    assert tuple(gate["gate_id"] for gate in first["gates"]) == GATE_IDS
    assert all(gate["status"] == "PASS" for gate in first["gates"])
    assert all(
        GateRecord.from_mapping(gate).status.value == "PASS"
        for gate in first["gates"]
    )
    gates = _gates_by_id(first)
    assert "manifests/batch-viewer-comparison.json" in gates[
        "stage0.G3-S5"
    ]["evidence_refs"]
    assert (
        "models/pythia-31m-deduped-step0/model-manifest.json"
        in gates["stage0.G3-S2"]["evidence_refs"]
    )
    assert (
        "manifests/pythia-31m-deduped-step0.json"
        in gates["stage0.G3-S2"]["evidence_refs"]
    )
    assert first["artifact_hash"] == g3_resolution_artifact_hash(first)
    validate_stage0_g3_resolution(first)


def test_g3_aggregator_blocks_incomplete_qualification_check_set(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(
        tmp_path,
        wrong_qualification_checks_for="pythia-14m-step0",
    )

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pythia-14m-step0"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["qualification_check_set_matches"] is False
    assert entry["checks"]["semantic_evidence_matches"] is False
    assert entry["reasons"] == [
        "QUALIFICATION_CHECK_SET_MISMATCH",
        "SEMANTIC_EVIDENCE_INVALID",
    ]
    assert _gates_by_id(audit)["stage0.G3-S1"]["status"] == "BLOCKED"


def test_g3_aggregator_loads_and_hash_verifies_verification_report(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    verification_path = fixture.verification_paths[("tokenizer", "pythia-tokenizer")]
    verification_path.write_bytes(verification_path.read_bytes() + b" ")

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pythia-tokenizer"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["verification_report_matches"] is False
    assert entry["reasons"] == ["VERIFICATION_REPORT_INVALID"]
    assert _gates_by_id(audit)["stage0.G3-S6"]["status"] == "BLOCKED"


def test_g3_aggregator_blocks_missing_semantic_evidence(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    fixture.semantic_paths[("model", "pythia-14m-step0")].unlink()

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pythia-14m-step0"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["semantic_evidence_matches"] is False
    assert entry["reasons"] == ["SEMANTIC_EVIDENCE_MISSING"]
    assert _gates_by_id(audit)["stage0.G3-S1"]["status"] == "BLOCKED"


def test_g3_aggregator_blocks_missing_verified_candidate_artifact(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    fixture.candidate_paths[("model", "pythia-14m-step0")].unlink()

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pythia-14m-step0"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["candidate_artifact_matches"] is False
    assert entry["reasons"] == ["CANDIDATE_ARTIFACT_MISSING"]
    assert _gates_by_id(audit)["stage0.G3-S1"]["status"] == "BLOCKED"


def test_g3_aggregator_blocks_tampered_semantic_evidence(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    semantic_path = fixture.semantic_paths[("tokenizer", "pythia-tokenizer")]
    semantic_path.write_bytes(semantic_path.read_bytes() + b" ")

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pythia-tokenizer"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["semantic_evidence_matches"] is False
    assert entry["reasons"] == ["SEMANTIC_EVIDENCE_INVALID"]
    assert _gates_by_id(audit)["stage0.G3-S6"]["status"] == "BLOCKED"


@pytest.mark.parametrize("source", ("pile_oracle", "legacy_manifest"))
def test_g3_aggregator_blocks_missing_source_provenance_evidence(
    tmp_path: Path,
    source: str,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    requirements = _load_object(fixture.requirements_path)
    if source == "pile_oracle":
        reference = requirements["pile"]["reference_reader_oracle"][
            "artifact_ref"
        ]
        logical_name = "pile-selected-prefix"
    else:
        model = next(
            item
            for item in requirements["models"]
            if item["name"] == "pythia-31m-deduped-step0"
        )
        reference = model["legacy_manifest_diagnostic"]["refs"][0]
        logical_name = "pythia-31m-deduped-step0"
    fixture.data_root.joinpath(*reference.split("/")).unlink()

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == logical_name
    )
    assert entry["checks"]["semantic_evidence_matches"] is False
    assert entry["status"] == "BLOCKED"


def test_pile_reference_oracle_rejects_self_consistent_wrong_result(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    requirements = _load_object(fixture.requirements_path)
    pile = requirements["pile"]
    oracle = pile["reference_reader_oracle"]
    path = fixture.data_root.joinpath(*oracle["artifact_ref"].split("/"))
    value = _load_object(path)
    value["batches"][1]["equal"] = False
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    oracle["artifact_sha256"] = _digest(raw)

    with pytest.raises(
        G3GateAggregationError,
        match="PILE_REFERENCE_ORACLE_BATCH_CONTENT_INVALID",
    ):
        load_pile_reference_reader_oracle(fixture.data_root, pile)


def test_legacy_manifest_diagnostic_requires_bom_after_rehash(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    requirements = _load_object(fixture.requirements_path)
    model = next(
        item
        for item in requirements["models"]
        if item["name"] == "pythia-31m-deduped-step0"
    )
    diagnostic = model["legacy_manifest_diagnostic"]
    raw = canonical_json_bytes({"legacy": True})
    diagnostic["size_bytes"] = len(raw)
    diagnostic["sha256"] = _digest(raw)
    for reference in diagnostic["refs"]:
        fixture.data_root.joinpath(*reference.split("/")).write_bytes(raw)

    with pytest.raises(
        G3GateAggregationError,
        match="LEGACY_MODEL_MANIFEST_BOM_MISSING",
    ):
        load_legacy_model_manifest_diagnostic(fixture.data_root, model)


def test_g3_aggregator_blocks_wrongly_bound_semantic_evidence(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(
        tmp_path,
        wrong_semantic_binding_for="pythia-31m-deduped-step0",
    )

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item
        for item in audit["entries"]
        if item["logical_name"] == "pythia-31m-deduped-step0"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["semantic_evidence_matches"] is False
    assert entry["reasons"] == ["SEMANTIC_EVIDENCE_INVALID"]
    assert _gates_by_id(audit)["stage0.G3-S2"]["status"] == "BLOCKED"


@pytest.mark.parametrize("mode", ("missing", "fixture", "wrong"))
def test_g3_aggregator_blocks_unbound_semantic_observation_details(
    tmp_path: Path,
    mode: str,
) -> None:
    fixture = _materialize_fixture(tmp_path, bad_semantic_details=mode)

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pythia-14m-step0"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["semantic_evidence_matches"] is False
    assert entry["reasons"] == ["SEMANTIC_EVIDENCE_INVALID"]
    assert _gates_by_id(audit)["stage0.G3-S1"]["status"] == "BLOCKED"


@pytest.mark.parametrize(
    "mode",
    ("implementation", "normalization", "token_text"),
)
def test_g3_aggregator_binds_tokenizer_observations_to_local_files(
    tmp_path: Path,
    mode: str,
) -> None:
    fixture = _materialize_fixture(tmp_path, bad_tokenizer_identity=mode)

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pythia-tokenizer"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["semantic_evidence_matches"] is False
    assert "SEMANTIC_EVIDENCE_INVALID" in entry["reasons"]
    assert _gates_by_id(audit)["stage0.G3-S6"]["status"] == "BLOCKED"


def test_g3_aggregator_turns_file_hash_failure_into_blocked_not_pass(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path)
    target = fixture.object_paths[("model", "pythia-14m-step0")][1]
    payload = target.read_bytes()
    target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    gates = _gates_by_id(audit)
    assert audit["status"] == "BLOCKED"
    assert gates["stage0.G3-S1"]["status"] == "BLOCKED"
    assert gates["stage0.G3-S2"]["status"] == "PASS"
    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pythia-14m-step0"
    )
    assert entry["status"] == "BLOCKED"
    assert entry["reasons"] == ["QUALIFIED_RESOLUTION_FAILED"]


def test_g3_aggregator_blocks_self_consistent_manifest_semantic_drift(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(
        tmp_path,
        wrong_semantic_model="pythia-410m-deduped-step0",
    )

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    gates = _gates_by_id(audit)
    assert gates["stage0.G3-S5"]["status"] == "BLOCKED"
    assert gates["stage0.G3-S6"]["status"] == "BLOCKED"
    entry = next(
        item
        for item in audit["entries"]
        if item["logical_name"] == "pythia-410m-deduped-step0"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["semantic_metadata_matches"] is False
    assert "REQUIREMENT_SEMANTIC_METADATA_MISMATCH" in entry["reasons"]


def test_g3_aggregator_blocks_pile_reference_batch_semantic_drift(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(tmp_path, wrong_pile_batch_hash=True)

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    entry = next(
        item for item in audit["entries"] if item["logical_name"] == "pile-selected-prefix"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["semantic_metadata_matches"] is False
    assert entry["reasons"] == [
        "REQUIREMENT_SEMANTIC_METADATA_MISMATCH",
        "SEMANTIC_EVIDENCE_INVALID",
    ]
    assert _gates_by_id(audit)["stage0.G3-S5"]["status"] == "BLOCKED"


def test_g3_aggregator_blocks_derived_asset_with_wrong_parent_lineage(
    tmp_path: Path,
) -> None:
    fixture = _materialize_fixture(
        tmp_path,
        wrong_lineage_derived="glue-rte-pretokenized",
    )

    audit = evaluate_stage0_g3(
        fixture.requirements_path,
        fixture.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )

    gates = _gates_by_id(audit)
    assert gates["stage0.G3-S6"]["status"] == "BLOCKED"
    # The acquisition attestation is one immutable thirteen-entry artifact.
    # A forged derived lineage invalidates that shared source claim, so every
    # gate that consumes the acquisition report must fail closed as well.
    assert gates["stage0.G3-S5"]["status"] == "BLOCKED"
    entry = next(
        item
        for item in audit["entries"]
        if item["logical_name"] == "glue-rte-pretokenized"
    )
    assert entry["checks"]["qualified_resolution"] is True
    assert entry["checks"]["cross_asset_binding_matches"] is False
    assert entry["reasons"] == [
        "ACQUISITION_REPORT_INVALID",
        "DERIVED_ASSET_LINEAGE_MISMATCH",
        "VERIFICATION_REPORT_INVALID",
    ]


def test_g3_aggregator_rejects_invalid_control_plane_without_a_gate(
    tmp_path: Path,
) -> None:
    requirements_path = tmp_path / "requirements.json"
    layout_path = tmp_path / "layout.json"
    data_root = tmp_path / "data-root"
    requirements_path.write_bytes(b'{"schema_version":"wrong"}\n')
    layout_path.write_bytes(_SOURCE_LAYOUT.read_bytes())
    data_root.mkdir()

    with pytest.raises(G3GateAggregationError, match="G3_CONTROL_PLANE_INVALID"):
        evaluate_stage0_g3(
            requirements_path,
            layout_path,
            data_root,
            checked_at=_CHECKED_AT,
        )
