from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path, PurePosixPath
import socket
import struct
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from param_importance_nlp.asset_download_plan import download_plan_artifact_hash
from param_importance_nlp.asset_layout import layout_artifact_hash
from param_importance_nlp.asset_requirements import requirements_artifact_hash
from param_importance_nlp.assets import validate_g3_manifest, validate_g3_qualification
from param_importance_nlp.contracts import (
    canonical_json_bytes,
    canonical_json_hash,
    ensure_json_object,
    load_canonical_json,
    loads_strict_json,
)
from param_importance_nlp.g3_asset_publication import (
    G3AssetPublicationError,
    NetworkEgressAttempt,
    gate_stage0_g3_assets_from_evidence,
    validate_semantic_evidence,
)
from param_importance_nlp.g3_gate import (
    GATE_IDS,
    evaluate_stage0_g3,
    glue_preprocessing_config_hash,
    qualification_check_ids_for,
)
import param_importance_nlp.g3_asset_publication as publication_module
from param_importance_nlp.g3_lifecycle_evidence import (
    attest_stage0_g3_acquisition,
    g3_downloaded_candidate_ref,
    verify_stage0_g3_acquisition,
)
import param_importance_nlp.g3_lifecycle_evidence as lifecycle_module
from param_importance_nlp.glue_builder import (
    GLUE_PREPROCESSING_VERSION,
    normalize_tokenizer_descriptor_inventory,
)


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_REQUIREMENTS = _ROOT / "configs/stage0/g3-asset-requirements-v1.json"
_SOURCE_LAYOUT = _ROOT / "configs/stage0/g3-asset-layout-v1.json"
_CHECKED_AT = "2026-08-03T08:00:00Z"
_GIT_COMMIT = "a" * 40
_REQUIREMENTS_REF = "configs/stage0/g3-asset-requirements-v1.json"
_LAYOUT_REF = "configs/stage0/g3-asset-layout-v1.json"
_PLAN_REF = "configs/stage0/g3-download-plan-v1.json"
_DOWNLOAD_REPORT_REF = "operations/g3/download-report.json"
_STARTED_AT = "2026-08-03T07:00:00Z"
_COMPLETED_AT = "2026-08-03T07:00:01Z"
_VERIFIED_AT = "2026-08-03T07:00:02Z"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    return dict(
        ensure_json_object(loads_strict_json(path.read_bytes()), field=str(path))
    )


def _replace_payload(descriptor: dict[str, Any], payload: bytes) -> None:
    descriptor["size_bytes"] = len(payload)
    descriptor["sha256"] = _digest(payload)


def _shrink_requirements(
    requirements: dict[str, Any],
) -> dict[tuple[str, str, str], bytes]:
    payloads: dict[tuple[str, str, str], bytes] = {}
    for model in requirements["models"]:
        for descriptor in model["files"]:
            payload = f"fixture:{model['name']}:{descriptor['path']}".encode()
            _replace_payload(descriptor, payload)
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
        _replace_payload(descriptor, payload)
        payloads[("tokenizer", tokenizer["name"], descriptor["path"])] = payload

    pile = requirements["pile"]
    index_payload = b"fixture-pile-index"
    _replace_payload(pile["index"], index_payload)
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
    shard_payload = b"\0" * (513 * 2049 * pile["index_contract"]["dtype_bytes"])
    shard = pile["selected_shards"][0]
    _replace_payload(shard, shard_payload)
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
            payload = (
                f"fixture:glue-{task['task']}-raw:{descriptor['path']}".encode()
            )
            _replace_payload(descriptor, payload)
            payloads[("glue_raw", task["task"], descriptor["path"])] = payload
    requirements["artifact_hash"] = requirements_artifact_hash(requirements)
    return payloads


@dataclass(frozen=True)
class _Fixture:
    requirements_path: Path
    layout_path: Path
    data_root: Path
    layout: dict[str, Any]


def _fixture(tmp_path: Path) -> _Fixture:
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
        target = data_root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    for entry in layout["entries"]:
        asset_root = data_root.joinpath(*PurePosixPath(entry["asset_root_ref"]).parts)
        asset_root.mkdir(parents=True, exist_ok=True)
        key = (entry["kind"], entry["requirement_name"])
        if entry["kind"] == "glue_derived":
            derived = asset_root / "dataset" / "state.json"
            derived.parent.mkdir(parents=True, exist_ok=True)
            derived.write_bytes(
                canonical_json_bytes(
                    {"task": entry["requirement_name"], "fixture": True}
                )
            )
            continue
        for (kind, name, relative), payload in payloads.items():
            if (kind, name) != key:
                continue
            target = asset_root.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return _Fixture(requirements_path, layout_path, data_root, layout)


def _fake_probe(
    kind: str,
    requirement: dict[str, Any],
    _asset_root: Path,
    manifest: dict[str, Any],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    if kind == "model":
        config = next(
            item for item in requirement["files"] if item["role"] == "config"
        )
        architecture = requirement["architecture"]
        return {
            "model_semantic_contract": {
                "architecture": architecture,
                "max_position_embeddings": requirement[
                    "max_position_embeddings"
                ],
                "config_sha256": config["sha256"],
                "legacy_manifest_replacement": kwargs.get(
                    "legacy_manifest_replacement"
                ),
            },
            "offline_model_load": {
                "config_class": (
                    f"{architecture.removesuffix('ForCausalLM')}Config"
                ),
                "model_class": architecture,
                "parameter_count": requirement["parameter_count"],
                "tensor_count": requirement["tensor_count"],
                "max_position_embeddings": requirement[
                    "max_position_embeddings"
                ],
                "local_files_only": True,
            },
        }
    if kind == "tokenizer":
        metadata = manifest["metadata"]
        padding = requirement["glue_padding_token"]
        return {
            "offline_tokenizer_load": {
                "tokenizer_class": requirement["tokenizer_class"],
                "vocab_size": requirement["vocab_size"],
                "token_count_with_added_tokens": requirement[
                    "token_count_with_added_tokens"
                ],
                "vocab_mapping_sha256": requirement[
                    "vocab_mapping_sha256"
                ],
            },
            "tokenizer_semantic_contract": {
                "implementation_version": metadata[
                    "implementation_version"
                ],
                "normalization": metadata["normalization"],
                "special_tokens": {
                    f"{name}_token": metadata["special_tokens"][
                        f"{name}_token"
                    ]
                    for name in requirement["special_token_ids"]
                },
                "glue_padding_token": {
                    "token": padding["token"],
                    "token_id": padding["token_id"],
                    "policy": padding["policy"],
                },
            },
        }
    if kind == "pile":
        causal = requirement["causal_lm_contract"]
        required_stop = requirement["required_cursor_stop"]
        mapping = {
            "labels_alignment": causal["labels_alignment"],
            "source_tokens_per_record": causal["source_tokens_per_record"],
            "input_sequence_length": causal["input_sequence_length"],
            "label_sequence_length": causal["target_sequence_length"],
            "input_slice": causal["input_slice"],
            "label_slice": causal["target_slice"],
            "attention_mask_policy": causal["attention_mask_policy"],
            "effective_target_tokens": causal[
                "effective_target_tokens_per_record"
            ],
            "loss_adapter_id": causal["loss_adapter_id"],
        }
        return {
            "pile_causal_lm_contract": {
                "mapping": mapping,
                "reference_reader": requirement["reference_reader"],
                "reference_batch_size": requirement["reference_batch_size"],
                "reference_batch_sha256": requirement[
                    "reference_batch_sha256"
                ],
                "reference_reader_oracle": kwargs.get(
                    "reference_reader_oracle"
                ),
                "last_required_record": {
                    "record_index": required_stop - 1,
                    "sha256": requirement["last_required_record_sha256"],
                },
            },
            "pile_cursor_coverage": {
                "required_cursor_stop": required_stop,
                "last_required_byte_stop": (
                    required_stop
                    * causal["source_tokens_per_record"]
                    * requirement["index_contract"]["dtype_bytes"]
                ),
                "selected_coverage_stop": requirement["selected_shards"][-1][
                    "byte_stop"
                ],
            },
            "pile_index_contract": requirement["index_contract"],
        }

    task_contract = {
        "task": requirement["task"],
        "text_fields": requirement["text_fields"],
        "label_mapping": requirement["label_mapping"],
        "unlabeled_test_policy": requirement["unlabeled_test_policy"],
    }
    labels = sorted(requirement["label_mapping"].values())
    if kind == "glue_raw":
        return {
            "glue_task_contract": task_contract,
            "offline_glue_raw_load": {
                "splits": {
                    split: {
                        "sample_count": count,
                        "fields": sorted(
                            {*requirement["text_fields"], "label"}
                        ),
                        "labels": (
                            [-1] if split.startswith("test") else labels
                        ),
                    }
                    for split, count in requirement["split_counts"].items()
                }
            },
        }
    preprocessing = manifest["metadata"]["preprocessing"]
    return {
        "glue_preprocessing_lineage": {
            "config_hash": glue_preprocessing_config_hash(requirement),
            "tokenizer_asset_id": preprocessing["tokenizer_asset_id"],
            "parent_asset_ids": preprocessing["parent_asset_ids"],
        },
        "glue_task_contract": task_contract,
        "offline_glue_derived_load": {
            "splits": {
                split: {
                    "sample_count": requirement["split_counts"][split],
                    "fields": ["attention_mask", "input_ids", "labels"],
                    "labels": labels,
                    "sequence_length": requirement["preprocessing"][
                        "max_length"
                    ],
                }
                for split in requirement["preprocessing"]["derived_splits"]
            }
        },
    }


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_ref(root: Path, reference: str, value: dict[str, Any]) -> Path:
    target = root.joinpath(*PurePosixPath(reference).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(value))
    return target


def _requirement_for(
    requirements: dict[str, Any], kind: str, requirement_name: str
) -> dict[str, Any]:
    if kind == "model":
        return next(
            item for item in requirements["models"] if item["name"] == requirement_name
        )
    if kind == "tokenizer":
        return requirements["tokenizer"]
    if kind == "pile":
        return requirements["pile"]
    return next(
        item for item in requirements["glue"] if item["task"] == requirement_name
    )


def _plan_descriptors(
    requirements: dict[str, Any], layout: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    canonical_plan_assets = {
        "pythia-410m-deduped-step0",
        "pythia-tokenizer",
        "glue-mnli-raw",
        "glue-rte-raw",
    }
    values: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in layout["entries"]:
        if entry["logical_name"] not in canonical_plan_assets:
            continue
        requirement = _requirement_for(
            requirements, entry["kind"], entry["requirement_name"]
        )
        descriptors = (
            requirement["raw_files"]
            if entry["kind"] == "glue_raw"
            else requirement["files"]
        )
        values.extend((entry, descriptor) for descriptor in descriptors)
    assert len(values) == 13
    return values


def _fake_derived_builder(
    data_root: str | Path,
    _raw_task_root: str | Path,
    raw_asset_id: str,
    _tokenizer_root: str | Path,
    tokenizer_asset_id: str,
    requirement: dict[str, Any],
    target_dir: str | Path,
    *,
    tokenizer_requirement: dict[str, Any],
    generator_git_commit: str,
) -> SimpleNamespace:
    root = Path(data_root)
    target_ref = Path(target_dir).as_posix()
    state = root.joinpath(*PurePosixPath(target_ref).parts) / "dataset/state.json"
    payload = state.read_bytes()
    tokenizer_inventory = normalize_tokenizer_descriptor_inventory(
        tokenizer_requirement
    )
    tokenizer_inventory_hash = canonical_json_hash(
        [dict(item) for item in tokenizer_inventory]
    )
    derived_splits = tuple(requirement["preprocessing"]["derived_splits"])
    preprocessing_config_hash = glue_preprocessing_config_hash(requirement)
    return SimpleNamespace(
        task=requirement["task"],
        raw_asset_id=raw_asset_id,
        tokenizer_asset_id=tokenizer_asset_id,
        tokenizer_descriptor_inventory=tokenizer_inventory,
        tokenizer_descriptor_inventory_hash=tokenizer_inventory_hash,
        target_ref=target_ref,
        generator_git_commit=generator_git_commit,
        preprocessing_version=GLUE_PREPROCESSING_VERSION,
        preprocessing_config_hash=preprocessing_config_hash,
        requirement_hash=canonical_json_hash(requirement),
        derived_splits=derived_splits,
        split_counts={
            split: requirement["split_counts"][split] for split in derived_splits
        },
        map_fingerprints=lifecycle_module._expected_derived_map_fingerprints(
            requirement,
            raw_asset_id=raw_asset_id,
            tokenizer_asset_id=tokenizer_asset_id,
            tokenizer_descriptor_inventory_hash=tokenizer_inventory_hash,
            generator_git_commit=generator_git_commit,
            preprocessing_config_hash=preprocessing_config_hash,
        ),
        file_inventory=(
            {"path": "dataset/state.json", "size_bytes": len(payload)},
        ),
        network_attempts=0,
    )


@dataclass
class _LifecycleFixture:
    fixture: _Fixture
    source_root: Path
    requirements_path: Path
    layout_path: Path
    download_plan_path: Path
    commit: str
    acquisition: Any | None = None
    verification: Any | None = None
    publications: tuple[Any, ...] = ()


def _prepare_lifecycle_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _LifecycleFixture:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    fixture = _fixture(fixture_root)
    requirements = _load_object(fixture.requirements_path)
    layout = _load_object(fixture.layout_path)
    source = tmp_path / "source"
    source.mkdir()

    for reference in lifecycle_module.G3_CRITICAL_SOURCE_REFS:
        target = source.joinpath(*PurePosixPath(reference).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_ROOT.joinpath(*PurePosixPath(reference).parts).read_bytes())

    requirements_path = _write_ref(source, _REQUIREMENTS_REF, requirements)
    layout_path = _write_ref(source, _LAYOUT_REF, layout)
    plan_entries: list[dict[str, Any]] = []
    report_objects: list[dict[str, Any]] = []
    for index, (entry, descriptor) in enumerate(
        _plan_descriptors(requirements, layout)
    ):
        spec_ref = f"configs/stage0/http-objects/fixture-{index:02d}.json"
        object_id = (
            f"huggingface/fixture/g3/{entry['logical_name']}/{descriptor['path']}"
        )
        requirement = _requirement_for(
            requirements, entry["kind"], entry["requirement_name"]
        )
        spec = {
            "schema_version": "stage0-http-object-spec-v1",
            "source_id": object_id,
            "revision": requirement["revision"],
            "expected_size": descriptor["size_bytes"],
            "expected_sha256": descriptor["sha256"],
        }
        _write_ref(source, spec_ref, spec)
        plan_entries.append(
            {
                "object_id": object_id,
                "spec_ref": spec_ref,
                "asset_root_ref": entry["asset_root_ref"],
                "final_path": descriptor["path"],
            }
        )
        report_objects.append(
            {
                "object_id": object_id,
                "asset_root_ref": entry["asset_root_ref"],
                "final_path": descriptor["path"],
                "result": {
                    "schema_version": "stage0-asset-acquisition-result-v1",
                    "status": "already_ready",
                    "source_id": object_id,
                    "revision": requirement["revision"],
                    "size_bytes": descriptor["size_bytes"],
                    "sha256": descriptor["sha256"],
                    "attempts": 0,
                    "resumed": False,
                    "network_accessed": False,
                },
            }
        )
    plan: dict[str, Any] = {
        "schema_version": "stage0-g3-download-plan-v1",
        "created_at": _STARTED_AT,
        "generator_git_commit": _GIT_COMMIT,
        "requirements_ref": _REQUIREMENTS_REF,
        "requirements_sha256": requirements["artifact_hash"],
        "layout_ref": _LAYOUT_REF,
        "layout_sha256": layout["artifact_hash"],
        "entries": plan_entries,
    }
    plan["artifact_hash"] = download_plan_artifact_hash(plan)
    plan_path = _write_ref(source, _PLAN_REF, plan)
    download_payload: dict[str, Any] = {
        "schema_version": "stage0-g3-download-report-v1",
        "status": "PASS",
        "started_at": _STARTED_AT,
        "plan_sha256": plan["artifact_hash"],
        "objects": report_objects,
        "runtime_urls_persisted": False,
    }
    _write_ref(
        fixture.data_root,
        _DOWNLOAD_REPORT_REF,
        download_payload | {"artifact_hash": canonical_json_hash(download_payload)},
    )

    _git(source, "init", "-q")
    _git(source, "config", "user.name", "G3 Lifecycle Test")
    _git(source, "config", "user.email", "g3-lifecycle@example.invalid")
    _git(source, "config", "core.autocrlf", "false")
    _git(source, "add", "--", ".")
    _git(source, "commit", "--no-gpg-sign", "-q", "-m", "freeze G3 fixture")
    commit = _git(source, "rev-parse", "HEAD")

    for module_name, reference in lifecycle_module.G3_CRITICAL_MODULE_ORIGINS:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(
            module,
            "__file__",
            str(source.joinpath(*PurePosixPath(reference).parts)),
        )
    monkeypatch.setattr(
        lifecycle_module, "build_glue_derived_dataset", _fake_derived_builder
    )
    return _LifecycleFixture(
        fixture=fixture,
        source_root=source,
        requirements_path=requirements_path,
        layout_path=layout_path,
        download_plan_path=plan_path,
        commit=commit,
    )


def _attest_lifecycle(state: _LifecycleFixture) -> Any:
    state.acquisition = attest_stage0_g3_acquisition(
        source_root=state.source_root,
        data_root=state.fixture.data_root,
        requirements=state.requirements_path,
        layout=state.layout_path,
        download_plan=state.download_plan_path,
        requirements_ref=_REQUIREMENTS_REF,
        layout_ref=_LAYOUT_REF,
        download_plan_ref=_PLAN_REF,
        download_report_ref=_DOWNLOAD_REPORT_REF,
        actor_instance_id="fixture:fetcher:1",
        source_git_commit=state.commit,
        started_at=_STARTED_AT,
        completed_at=_COMPLETED_AT,
    )
    return state.acquisition


def _verify_lifecycle(state: _LifecycleFixture) -> Any:
    assert state.acquisition is not None
    state.verification = verify_stage0_g3_acquisition(
        source_root=state.source_root,
        data_root=state.fixture.data_root,
        requirements=state.requirements_path,
        layout=state.layout_path,
        download_plan=state.download_plan_path,
        acquisition_ref=state.acquisition.acquisition_ref,
        actor_instance_id="fixture:verifier:1",
        generator_git_commit=state.commit,
        checked_at=_VERIFIED_AT,
    )
    return state.verification


def _gate_lifecycle(
    state: _LifecycleFixture,
    *,
    checked_at: str = _CHECKED_AT,
    gate_actor_instance_id: str = "fixture:gate:1",
) -> tuple[Any, ...]:
    assert state.acquisition is not None
    assert state.verification is not None
    state.publications = gate_stage0_g3_assets_from_evidence(
        state.requirements_path,
        state.layout_path,
        state.download_plan_path,
        state.source_root,
        state.fixture.data_root,
        acquisition_ref=state.acquisition.acquisition_ref,
        verification_ref=state.verification.verification_ref,
        generator_git_commit=state.commit,
        checked_at=checked_at,
        gate_actor_instance_id=gate_actor_instance_id,
    )
    return state.publications


def _verified_lifecycle_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _LifecycleFixture:
    state = _prepare_lifecycle_fixture(tmp_path, monkeypatch)
    _attest_lifecycle(state)
    _verify_lifecycle(state)
    return state


def _published_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _LifecycleFixture:
    state = _verified_lifecycle_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(publication_module, "_run_semantic_probe", _fake_probe)
    results = _gate_lifecycle(state)
    assert len(results) == 13
    return state


def test_pile_probe_hashes_official_sized_batches_and_last_record_separately(
    tmp_path: Path,
) -> None:
    record_count = 1025
    tokens_per_record = 2049
    batch_size = 1024
    records = [
        struct.pack("<H", record_index) * tokens_per_record
        for record_index in range(record_count)
    ]
    logical_bytes = b"".join(records)
    pointers = [
        record_index * tokens_per_record * 2
        for record_index in range(record_count)
    ]
    index_bytes = (
        b"MMIDIDX\x00\x00"
        + struct.pack("<Q", 1)
        + struct.pack("<B", 8)
        + struct.pack("<QQ", record_count, 1)
        + struct.pack(f"<{record_count}i", *([tokens_per_record] * record_count))
        + struct.pack(f"<{record_count}q", *pointers)
        + struct.pack("<q", record_count)
    )
    index_path = tmp_path / "document.idx"
    shard_path = tmp_path / "document-00000-of-00020.bin"
    index_path.write_bytes(index_bytes)
    shard_path.write_bytes(logical_bytes)
    expected_batch_hash = _digest(b"".join(records[:batch_size]))
    expected_last_hash = _digest(records[-1])
    requirement = {
        "index": {"path": index_path.name},
        "index_contract": {
            "magic_hex": "4d4d49444944580000",
            "version": 1,
            "dtype_code": 8,
            "dtype_bytes": 2,
            "sequence_count": record_count,
            "document_count": 1,
        },
        "selected_shards": [
            {
                "path": shard_path.name,
                "byte_start": 0,
                "byte_stop": len(logical_bytes),
            }
        ],
        "causal_lm_contract": {
            "labels_alignment": "pre_shifted_next_token",
            "source_tokens_per_record": tokens_per_record,
            "input_sequence_length": 2048,
            "target_sequence_length": 2048,
            "input_slice": [0, 2048],
            "target_slice": [1, 2049],
            "attention_mask_policy": "all_one_for_fixed_full_record",
            "effective_target_tokens_per_record": 2048,
            "loss_adapter_id": "pre-shifted-next-token-cross-entropy-v1",
        },
        "required_cursor_stop": record_count,
        "reference_reader": {
            "repository": "EleutherAI/pythia",
            "revision": "a19eecb807ec2c79a39ebf18108816e6ffffc1d5",
        },
        "reference_batch_size": batch_size,
        "reference_batch_sha256": {"0": expected_batch_hash},
        "last_required_record_sha256": expected_last_hash,
    }

    details = publication_module._probe_pile(requirement, tmp_path)
    causal = details["pile_causal_lm_contract"]
    assert causal["reference_batch_size"] == 1024
    assert causal["reference_batch_sha256"] == {"0": expected_batch_hash}
    assert expected_batch_hash != _digest(records[0])
    assert causal["last_required_record"] == {
        "record_index": 1024,
        "sha256": expected_last_hash,
    }


def test_publication_runs_all_thirteen_lifecycles_and_passes_g3_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _published_fixture(tmp_path, monkeypatch)
    fixture = state.fixture
    results = state.publications
    assert state.acquisition is not None
    assert state.verification is not None
    assert len(results) == 13
    assert [item.logical_name for item in results] == [
        entry["logical_name"] for entry in fixture.layout["entries"]
    ]
    assert {item.status for item in results} == {"published"}
    assert {item.state for item in results} == {"ready"}

    for result, entry in zip(results, fixture.layout["entries"], strict=True):
        assert result.candidate_id in result.candidate_ref
        assert result.candidate_id in result.semantic_evidence_ref
        assert result.verification_ref == state.verification.verification_ref
        ready = load_canonical_json(fixture.data_root / entry["manifest_ref"])
        verified = load_canonical_json(fixture.data_root / result.candidate_ref)
        acquisition = load_canonical_json(
            fixture.data_root / state.acquisition.acquisition_ref
        )
        acquisition_entry = next(
            item
            for item in acquisition["entries"]
            if item["logical_name"] == entry["logical_name"]
        )
        downloaded_ref = g3_downloaded_candidate_ref(
            entry["logical_name"], result.candidate_id
        )
        downloaded = load_canonical_json(fixture.data_root / downloaded_ref)
        qualification = load_canonical_json(
            fixture.data_root / entry["qualification_ref"]
        )
        verification_path = fixture.data_root / result.verification_ref
        semantic_path = fixture.data_root / result.semantic_evidence_ref
        semantic = load_canonical_json(semantic_path)
        validate_g3_manifest(ready)
        validate_g3_manifest(downloaded)
        validate_g3_manifest(verified)
        validate_g3_qualification(qualification)
        validate_semantic_evidence(
            semantic,
            expected_check_ids=tuple(
                check_id
                for check_id in qualification_check_ids_for(entry["kind"])
                if check_id != "full_file_integrity"
            ),
        )
        assert ready["state"] == "ready"
        assert downloaded["state"] == "downloaded"
        assert verified["state"] == "verified"
        assert downloaded["asset_id"] == acquisition_entry["asset_id"]
        assert verified["candidate_id"] == downloaded["candidate_id"]
        assert [event["to"] for event in ready["state_history"]] == [
            "downloading",
            "downloaded",
            "verified",
            "ready",
        ]
        assert [event["actor_instance_id"] for event in ready["state_history"]] == [
            "fixture:fetcher:1",
            "fixture:fetcher:1",
            "fixture:verifier:1",
            "fixture:gate:1",
        ]
        assert tuple(check["check_id"] for check in qualification["checks"]) == (
            qualification_check_ids_for(entry["kind"])
        )
        verification_report = load_canonical_json(verification_path)
        semantic_sha = _digest(semantic_path.read_bytes())
        for check in qualification["checks"]:
            if check["check_id"] == "full_file_integrity":
                assert check["evidence_ref"] == result.verification_ref
                assert check["evidence_sha256"] == verification_report["artifact_hash"]
            else:
                assert check["evidence_ref"] == result.semantic_evidence_ref
                assert check["evidence_sha256"] == semantic_sha
        semantic_details = {
            check["check_id"]: check["details"]
            for check in semantic["checks"]
        }
        if entry["kind"] == "pile":
            assert semantic_details["pile_causal_lm_contract"][
                "reference_reader_oracle"
            ]["artifact_ref"] == "manifests/batch-viewer-comparison.json"
        if entry["logical_name"] == "pythia-31m-deduped-step0":
            assert semantic_details["model_semantic_contract"][
                "legacy_manifest_replacement"
            ]["condition"] == "utf8_bom_strict_json_rejected"

    forbidden_roots = (
        str(state.source_root),
        str(fixture.data_root),
    )
    for artifact in (fixture.data_root / "manifests").rglob("*.json"):
        text = artifact.read_text(encoding="utf-8")
        assert all(root not in text for root in forbidden_roots)

    audit = evaluate_stage0_g3(
        state.requirements_path,
        state.layout_path,
        fixture.data_root,
        checked_at=_CHECKED_AT,
    )
    assert audit["status"] == "PASS"
    assert tuple(item["gate_id"] for item in audit["gates"]) == GATE_IDS


def test_acquisition_and_independent_verify_cover_exact_thirteen_asset_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_lifecycle_fixture(tmp_path, monkeypatch)
    assert state.acquisition is not None
    assert state.verification is not None
    acquisition = load_canonical_json(
        state.fixture.data_root / state.acquisition.acquisition_ref
    )
    verification = load_canonical_json(
        state.fixture.data_root / state.verification.verification_ref
    )
    assert acquisition["entry_count"] == verification["entry_count"] == 13
    assert [entry["mode"] for entry in acquisition["entries"]].count(
        "canonical-plan"
    ) == 4
    assert [entry["mode"] for entry in acquisition["entries"]].count(
        "existing-import"
    ) == 6
    assert [entry["mode"] for entry in acquisition["entries"]].count(
        "derived-build"
    ) == 3
    assert verification["status"] == "PASS"
    assert acquisition["actor_instance_id"] != verification["actor_instance_id"]
    assert all(
        not (state.fixture.data_root / entry["manifest_ref"]).exists()
        for entry in state.fixture.layout["entries"]
    )


def test_publication_rerun_accepts_only_resolved_existing_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _published_fixture(tmp_path, monkeypatch)
    results = _gate_lifecycle(
        state,
        checked_at="2026-08-03T09:00:00Z",
    )
    assert len(results) == 13
    assert {item.status for item in results} == {"existing_ready"}


def test_publication_rerun_replays_current_semantic_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _published_fixture(tmp_path, monkeypatch)

    def drifted_probe(
        kind: str,
        requirement: dict[str, Any],
        asset_root: Path,
        manifest: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        details = _fake_probe(
            kind,
            requirement,
            asset_root,
            manifest,
            **kwargs,
        )
        for value in details.values():
            value["environment_drift"] = True
        return details

    monkeypatch.setattr(
        publication_module,
        "_run_semantic_probe",
        drifted_probe,
    )
    with pytest.raises(
        G3AssetPublicationError,
        match="current semantic probe differs",
    ):
        _gate_lifecycle(
            state,
            checked_at="2026-08-03T09:00:00Z",
        )


def test_existing_ready_requires_its_exact_candidate_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _published_fixture(tmp_path, monkeypatch)
    first = state.publications[0]
    (state.fixture.data_root / first.candidate_ref).unlink()

    with pytest.raises(
        G3AssetPublicationError,
        match="VERIFIED candidate cannot be replayed",
    ):
        _gate_lifecycle(
            state,
            checked_at="2026-08-03T09:00:00Z",
        )


def test_publication_never_clobbers_different_semantic_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _published_fixture(tmp_path, monkeypatch)
    first = state.fixture.layout["entries"][0]
    ready_path = state.fixture.data_root / first["manifest_ref"]
    ready_bytes = ready_path.read_bytes()
    semantic_ref = state.publications[0].semantic_evidence_ref
    semantic = state.fixture.data_root / semantic_ref
    original = canonical_json_bytes({"forged": True})
    semantic.write_bytes(original)
    with pytest.raises((FileExistsError, G3AssetPublicationError)):
        _gate_lifecycle(state, checked_at="2026-08-03T09:00:00Z")
    assert semantic.read_bytes() == original
    assert ready_path.read_bytes() == ready_bytes


def test_crash_after_qualification_recovers_from_run_scoped_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_lifecycle_fixture(tmp_path, monkeypatch)
    first = state.fixture.layout["entries"][0]
    monkeypatch.setattr(publication_module, "_run_semantic_probe", _fake_probe)
    publish = publication_module._publish_no_clobber

    def crash_after_qualification(
        root: Path,
        reference: str,
        value: dict[str, Any],
    ) -> tuple[Path, bool, str]:
        result = publish(root, reference, value)
        if reference == first["qualification_ref"]:
            raise RuntimeError("simulated crash after qualification exposure")
        return result

    monkeypatch.setattr(
        publication_module,
        "_publish_no_clobber",
        crash_after_qualification,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        _gate_lifecycle(state)
    assert (state.fixture.data_root / first["qualification_ref"]).is_file()
    assert not (state.fixture.data_root / first["manifest_ref"]).exists()

    monkeypatch.setattr(publication_module, "_publish_no_clobber", publish)
    resumed = _gate_lifecycle(
        state,
        checked_at="2026-08-03T09:00:00Z",
    )
    assert resumed[0].status == "recovered_ready"
    assert resumed[0].candidate_id in resumed[0].candidate_ref
    assert resumed[0].candidate_id in resumed[0].semantic_evidence_ref
    assert all(item.state == "ready" for item in resumed)


def test_publication_fails_closed_on_incomplete_semantic_check_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_lifecycle_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        publication_module,
        "_run_semantic_probe",
        lambda *_args, **_kwargs: {},
    )
    first = state.fixture.layout["entries"][0]
    with pytest.raises(G3AssetPublicationError, match="check set"):
        _gate_lifecycle(state)
    assert not (state.fixture.data_root / first["qualification_ref"]).exists()
    assert not (state.fixture.data_root / first["manifest_ref"]).exists()
    monkeypatch.setattr(publication_module, "_run_semantic_probe", _fake_probe)
    resumed = _gate_lifecycle(state)
    assert len(resumed) == 13
    assert all(item.state == "ready" for item in resumed)


def test_publication_socket_guard_blocks_probe_egress_before_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_lifecycle_fixture(tmp_path, monkeypatch)

    def egress(*_args: Any, **_kwargs: Any) -> dict[str, dict[str, Any]]:
        socket.create_connection(("example.com", 443))
        raise AssertionError("socket guard did not block egress")

    monkeypatch.setattr(publication_module, "_run_semantic_probe", egress)
    first = state.fixture.layout["entries"][0]
    with pytest.raises(NetworkEgressAttempt):
        _gate_lifecycle(state)
    assert not (state.fixture.data_root / first["qualification_ref"]).exists()
    assert not (state.fixture.data_root / first["manifest_ref"]).exists()


@pytest.mark.parametrize("source", ("pile_oracle", "legacy_manifest"))
def test_publication_blocks_invalid_source_provenance_before_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    state = _verified_lifecycle_fixture(tmp_path, monkeypatch)
    fixture = state.fixture
    requirements = _load_object(state.requirements_path)
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
    path = fixture.data_root.joinpath(*PurePosixPath(reference).parts)
    path.write_bytes(path.read_bytes() + b"tampered")
    entry = next(
        item
        for item in fixture.layout["entries"]
        if item["logical_name"] == logical_name
    )

    monkeypatch.setattr(publication_module, "_run_semantic_probe", _fake_probe)
    with pytest.raises(
        G3AssetPublicationError,
        match="source-provenance evidence",
    ):
        _gate_lifecycle(state)

    assert not (fixture.data_root / entry["qualification_ref"]).exists()
    assert not (fixture.data_root / entry["manifest_ref"]).exists()


def test_model_probe_performs_a_real_local_only_transformers_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirements = _load_object(_SOURCE_REQUIREMENTS)
    requirement = requirements["models"][0]
    asset_root = tmp_path / "model"
    asset_root.mkdir()
    config_descriptor = next(
        item for item in requirement["files"] if item["role"] == "config"
    )
    (asset_root / config_descriptor["path"]).write_bytes(
        canonical_json_bytes(
            {
                "architectures": [requirement["architecture"]],
                "max_position_embeddings": requirement[
                    "max_position_embeddings"
                ],
            }
        )
    )
    monkeypatch.setattr(
        publication_module,
        "_safetensors_statistics",
        lambda _path: {
            "tensor_count": requirement["tensor_count"],
            "parameter_count": requirement["parameter_count"],
            "dtype_counts": requirement["dtype_counts"],
            "header_length": 1,
            "file_size": 1,
        },
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Config:
        max_position_embeddings = requirement["max_position_embeddings"]

    class _Parameter:
        def numel(self) -> int:
            return requirement["parameter_count"]

    class _State:
        tensor_count = requirement["tensor_count"]

        def __len__(self) -> int:
            return self.tensor_count

    class _Model:
        def parameters(self) -> list[_Parameter]:
            return [_Parameter()]

        def state_dict(self) -> _State:
            return _State()

    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> _Config:
            calls.append((path, kwargs))
            return _Config()

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> _Model:
            calls.append((path, kwargs))
            return _Model()

    transformers = ModuleType("transformers")
    transformers.AutoConfig = _AutoConfig  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = _AutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    details = publication_module._probe_model(requirement, asset_root)

    assert len(calls) == 2
    assert all(path == str(asset_root) for path, _kwargs in calls)
    assert all(
        kwargs == {"local_files_only": True, "trust_remote_code": False}
        for _path, kwargs in calls
    )
    assert details["offline_model_load"]["parameter_count"] == requirement[
        "parameter_count"
    ]
    # The frozen Pythia contract binds the serialized tensor inventory to the
    # loaded state_dict inventory; no extra tied-weight alias is admissible.
    assert details["offline_model_load"]["tensor_count"] == requirement[
        "tensor_count"
    ]
    assert details["offline_model_load"]["local_files_only"] is True

    _State.tensor_count -= 1
    with pytest.raises(G3AssetPublicationError, match="offline model identity"):
        publication_module._probe_model(requirement, asset_root)
