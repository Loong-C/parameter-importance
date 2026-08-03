from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

from param_importance_nlp.contracts import DependencyUnavailable, canonical_json_hash
import param_importance_nlp.glue_builder as glue_builder
from param_importance_nlp.glue_builder import (
    GlueDerivedBuildError,
    build_glue_derived_dataset,
)


_RAW_ROWS: dict[str, list[dict[str, Any]]] = {}


class _FakeDataset:
    map_calls: list[dict[str, Any]] = []
    parquet_calls: list[dict[str, Any]] = []

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        fingerprint: str = "raw-fixture",
    ) -> None:
        self.rows = rows
        self._fingerprint = fingerprint
        self.column_names = list(rows[0]) if rows else []

    @classmethod
    def from_parquet(cls, paths: list[str], **kwargs: Any) -> "_FakeDataset":
        cls.parquet_calls.append(
            {
                "paths": list(paths),
                "kwargs": dict(kwargs),
                "offline": {
                    name: os.environ.get(name)
                    for name in (
                        "HF_HUB_OFFLINE",
                        "HF_DATASETS_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                    )
                },
            }
        )
        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(_RAW_ROWS[Path(path).name])
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def iter(self, *, batch_size: int):
        for start in range(0, len(self.rows), batch_size):
            rows = self.rows[start : start + batch_size]
            yield {
                column: [row[column] for row in rows]
                for column in self.column_names
            }

    def map(self, function: Any, **kwargs: Any) -> "_FakeDataset":
        type(self).map_calls.append(dict(kwargs))
        result_rows: list[dict[str, Any]] = []
        batch_size = kwargs["batch_size"]
        for start in range(0, len(self.rows), batch_size):
            rows = self.rows[start : start + batch_size]
            batch = {
                column: [row[column] for row in rows]
                for column in self.column_names
            }
            encoded = function(batch)
            for index in range(len(rows)):
                result_rows.append(
                    {column: encoded[column][index] for column in encoded}
                )
        return _FakeDataset(
            result_rows,
            fingerprint=kwargs["new_fingerprint"],
        )


class _FakeDatasetDict(dict[str, _FakeDataset]):
    save_calls: list[dict[str, Any]] = []

    def save_to_disk(self, path: str, **kwargs: Any) -> None:
        type(self).save_calls.append({"path": path, "kwargs": dict(kwargs)})
        root = Path(path)
        root.mkdir(parents=True)
        (root / "dataset_dict.json").write_text(
            json.dumps({"splits": list(self)}, sort_keys=True),
            encoding="utf-8",
        )
        for split, dataset in self.items():
            split_root = root / split
            split_root.mkdir()
            (split_root / "rows.json").write_text(
                json.dumps(dataset.rows, sort_keys=True),
                encoding="utf-8",
            )
            (split_root / "state.json").write_text(
                json.dumps({"fingerprint": dataset._fingerprint}),
                encoding="utf-8",
            )


class _FakeTokenizer:
    load_calls: list[dict[str, Any]] = []
    tokenize_calls: list[dict[str, Any]] = []
    connect_on_load = False
    subprocess_on_load = False
    mutate_on_load: Path | None = None

    def __init__(self) -> None:
        self._pad_token: str | None = None

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "_FakeTokenizer":
        cls.load_calls.append(
            {
                "path": path,
                "kwargs": dict(kwargs),
                "offline": {
                    name: os.environ.get(name)
                    for name in (
                        "HF_HUB_OFFLINE",
                        "HF_DATASETS_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                    )
                },
            }
        )
        if cls.connect_on_load:
            socket.create_connection(("example.invalid", 443))
        if cls.subprocess_on_load:
            subprocess.run([sys.executable, "-c", "pass"], check=True)
        if cls.mutate_on_load is not None:
            payload = bytearray(cls.mutate_on_load.read_bytes())
            payload[0] ^= 1
            cls.mutate_on_load.write_bytes(payload)
        return cls()

    def convert_tokens_to_ids(self, token: str) -> int:
        return 1 if token == "<|padding|>" else 2

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        self._pad_token = value

    @property
    def pad_token_id(self) -> int | None:
        return 1 if self._pad_token == "<|padding|>" else None

    def __call__(self, *values: list[str], **kwargs: Any) -> dict[str, Any]:
        type(self).tokenize_calls.append(
            {"argument_count": len(values), "kwargs": dict(kwargs)}
        )
        row_count = len(values[0])
        active = 2 if len(values) == 2 else 1
        return {
            "input_ids": [
                [10 + index for index in range(active)]
                + [1] * (glue_builder.MAX_LENGTH - active)
                for _ in range(row_count)
            ],
            "attention_mask": [
                [1] * active + [0] * (glue_builder.MAX_LENGTH - active)
                for _ in range(row_count)
            ],
            # The builder must deliberately discard tokenizer extras.
            "token_type_ids": [[0] * glue_builder.MAX_LENGTH for _ in range(row_count)],
        }


def _fake_dependencies() -> glue_builder._Dependencies:
    datasets = ModuleType("fake_datasets")
    datasets.Dataset = _FakeDataset  # type: ignore[attr-defined]
    datasets.DatasetDict = _FakeDatasetDict  # type: ignore[attr-defined]

    def load_from_disk(path: str) -> _FakeDatasetDict:
        root = Path(path)
        split_names = json.loads(
            (root / "dataset_dict.json").read_text(encoding="utf-8")
        )["splits"]
        loaded: dict[str, _FakeDataset] = {}
        for split in split_names:
            split_root = root / split
            rows = json.loads((split_root / "rows.json").read_text(encoding="utf-8"))
            fingerprint = json.loads(
                (split_root / "state.json").read_text(encoding="utf-8")
            )["fingerprint"]
            loaded[split] = _FakeDataset(rows, fingerprint=fingerprint)
        return _FakeDatasetDict(loaded)

    datasets.load_from_disk = load_from_disk  # type: ignore[attr-defined]
    transformers = ModuleType("fake_transformers")
    transformers.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    return glue_builder._Dependencies(datasets=datasets, transformers=transformers)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_raw_file(root: Path, name: str, payload: bytes, role: str) -> dict[str, Any]:
    (root / name).write_bytes(payload)
    return {
        "path": name,
        "size_bytes": len(payload),
        "sha256": _digest(payload),
        "role": role,
    }


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw" / "sst2"
    tokenizer_root = data_root / "tokenizers" / "pythia"
    raw_root.mkdir(parents=True)
    tokenizer_root.mkdir(parents=True)
    tokenizer_files = [
        _write_raw_file(
            tokenizer_root,
            "special_tokens_map.json",
            b'{"bos_token":"<|endoftext|>"}',
            "special_tokens",
        ),
        _write_raw_file(
            tokenizer_root,
            "tokenizer.json",
            b'{"padding":1}',
            "tokenizer_model",
        ),
        _write_raw_file(
            tokenizer_root,
            "tokenizer_config.json",
            b'{"tokenizer_class":"GPTNeoXTokenizerFast"}',
            "tokenizer_config",
        ),
    ]
    (tokenizer_root / "config.json").write_text(
        '{"model_type":"gpt_neox"}', encoding="utf-8"
    )
    tokenizer_requirement = {
        "name": "pythia-tokenizer",
        "files": tokenizer_files,
    }
    raw_files = [
        _write_raw_file(raw_root, "train.parquet", b"train-local", "train"),
        _write_raw_file(
            raw_root,
            "validation.parquet",
            b"validation-local",
            "validation",
        ),
        _write_raw_file(raw_root, "test.parquet", b"test-local", "test"),
    ]
    _RAW_ROWS.clear()
    _RAW_ROWS.update(
        {
            "train.parquet": [
                {"sentence": "good", "label": 1, "idx": 0},
                {"sentence": "bad", "label": 0, "idx": 1},
            ],
            "validation.parquet": [
                {"sentence": "fine", "label": 1, "idx": 2}
            ],
            "test.parquet": [{"sentence": "hidden", "label": -1, "idx": 3}],
        }
    )
    requirement = {
        "task": "sst2",
        "repository": "nyu-mll/glue",
        "revision": "a" * 40,
        "raw_files": raw_files,
        "split_counts": {"train": 2, "validation": 1, "test": 1},
        "text_fields": ["sentence"],
        "label_mapping": {"negative": 0, "positive": 1},
        "unlabeled_test_policy": (
            "retain_in_raw_exclude_from_derived_training_and_evaluation"
        ),
        "preprocessing": {
            "tokenizer_name": "pythia-tokenizer",
            "max_length": 512,
            "truncation": True,
            "padding": "max_length",
            "pad_token": "<|padding|>",
            "derived_splits": ["train", "validation"],
        },
    }
    return data_root, raw_root, tokenizer_root, tokenizer_requirement, requirement


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _FakeDataset.map_calls.clear()
    _FakeDataset.parquet_calls.clear()
    _FakeDatasetDict.save_calls.clear()
    _FakeTokenizer.load_calls.clear()
    _FakeTokenizer.tokenize_calls.clear()
    _FakeTokenizer.connect_on_load = False
    _FakeTokenizer.subprocess_on_load = False
    _FakeTokenizer.mutate_on_load = None


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    requirement_mutator: Any | None = None,
):
    data_root, raw_root, tokenizer_root, tokenizer_requirement, requirement = _fixture(
        tmp_path
    )
    if requirement_mutator is not None:
        requirement_mutator(requirement)
    monkeypatch.setattr(glue_builder, "_load_dependencies", _fake_dependencies)
    result = build_glue_derived_dataset(
        data_root,
        raw_root,
        "b" * 64,
        tokenizer_root,
        "c" * 64,
        requirement,
        data_root / "derived" / "sst2",
        tokenizer_requirement=tokenizer_requirement,
        generator_git_commit="d" * 40,
    )
    return (
        result,
        data_root,
        raw_root,
        tokenizer_root,
        tokenizer_requirement,
        requirement,
    )


def test_builds_only_derived_splits_offline_with_fixed_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, data_root, _, _, tokenizer_requirement, _ = _build(
        tmp_path, monkeypatch
    )

    assert result.status == "built"
    assert result.derived_splits == ("train", "validation")
    assert result.split_counts == {"train": 2, "validation": 1}
    assert result.network_attempts == 0
    expected_tokenizer_inventory = (
        glue_builder.normalize_tokenizer_descriptor_inventory(tokenizer_requirement)
    )
    assert result.tokenizer_descriptor_inventory == expected_tokenizer_inventory
    assert result.tokenizer_descriptor_inventory_hash == canonical_json_hash(
        [dict(item) for item in expected_tokenizer_inventory]
    )
    with pytest.raises(
        ValueError,
        match="GLUE_BUILD_RESULT_TOKENIZER_INVENTORY_HASH_INVALID",
    ):
        replace(result, tokenizer_descriptor_inventory_hash="0" * 64)
    assert set(result.map_fingerprints) == {"train", "validation"}
    assert all(len(value) == 64 for value in result.map_fingerprints.values())
    assert [Path(call["paths"][0]).name for call in _FakeDataset.parquet_calls] == [
        "train.parquet",
        "validation.parquet",
    ]
    assert all(call["kwargs"]["num_proc"] == 1 for call in _FakeDataset.parquet_calls)
    assert all(
        call["offline"]
        == {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        for call in _FakeDataset.parquet_calls
    )
    assert _FakeTokenizer.load_calls[0]["kwargs"] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert all(call["kwargs"]["max_length"] == 512 for call in _FakeTokenizer.tokenize_calls)
    assert all(call["kwargs"]["truncation"] is True for call in _FakeTokenizer.tokenize_calls)
    assert all(call["kwargs"]["padding"] == "max_length" for call in _FakeTokenizer.tokenize_calls)
    assert all(call["num_proc"] == 1 for call in _FakeDataset.map_calls)
    assert [call["new_fingerprint"] for call in _FakeDataset.map_calls] == [
        result.map_fingerprints["train"],
        result.map_fingerprints["validation"],
    ]
    assert _FakeDatasetDict.save_calls[0]["kwargs"] == {"num_proc": 1}

    target = Path(result.target_path)
    assert target.is_dir()
    sidecar_path = target / glue_builder.SIDECAR_NAME
    assert sidecar_path.is_file()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == glue_builder.SIDECAR_SCHEMA_VERSION
    assert "tokenizer_directory_hash" not in sidecar
    assert sidecar["tokenizer_descriptor_inventory"] == [
        dict(item) for item in expected_tokenizer_inventory
    ]
    assert (
        sidecar["tokenizer_descriptor_inventory_hash"]
        == result.tokenizer_descriptor_inventory_hash
    )
    assert {item["path"] for item in result.file_inventory} == {
        "dataset_dict.json",
        "stage0-glue-derived-build.json",
        "train/rows.json",
        "train/state.json",
        "validation/rows.json",
        "validation/state.json",
    }
    assert not any((data_root / "tmp").iterdir())
    assert result.to_dict() == json.loads(json.dumps(result.to_dict()))


def test_existing_target_is_read_only_validated_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        first,
        data_root,
        raw_root,
        tokenizer_root,
        tokenizer_requirement,
        requirement,
    ) = _build(tmp_path, monkeypatch)
    target = Path(first.target_path)
    before = {
        path.relative_to(target).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in target.rglob("*")
        if path.is_file()
    }
    _FakeDataset.map_calls.clear()
    _FakeDataset.parquet_calls.clear()
    # The tokenizer shares its root with model files and a persistent
    # acquisition lock.  Neither is tokenizer lineage.
    (tokenizer_root / "config.json").write_text(
        '{"model_type":"changed-but-unrelated"}', encoding="utf-8"
    )
    acquire_lock = tokenizer_root / "acquire.lock"
    acquire_lock.write_text("held", encoding="utf-8")
    second = build_glue_derived_dataset(
        data_root,
        raw_root,
        "b" * 64,
        tokenizer_root,
        "c" * 64,
        requirement,
        target,
        tokenizer_requirement=tokenizer_requirement,
        generator_git_commit="d" * 40,
    )
    acquire_lock.unlink()
    third = build_glue_derived_dataset(
        data_root,
        raw_root,
        "b" * 64,
        tokenizer_root,
        "c" * 64,
        requirement,
        target,
        tokenizer_requirement=tokenizer_requirement,
        generator_git_commit="d" * 40,
    )

    assert second.status == third.status == "reused"
    assert second.to_dict() | {"status": "built"} == first.to_dict()
    assert third.to_dict() | {"status": "built"} == first.to_dict()
    assert not _FakeDataset.map_calls
    assert not _FakeDataset.parquet_calls
    assert before == {
        path.relative_to(target).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in target.rglob("*")
        if path.is_file()
    }
    assert not any((data_root / "tmp").iterdir())


@pytest.mark.parametrize(
    "descriptor_name",
    [
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
def test_selected_tokenizer_descriptor_drift_blocks_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptor_name: str,
) -> None:
    (
        first,
        data_root,
        raw_root,
        tokenizer_root,
        tokenizer_requirement,
        requirement,
    ) = _build(tmp_path, monkeypatch)
    descriptor_path = tokenizer_root / descriptor_name
    drifted = bytearray(descriptor_path.read_bytes())
    drifted[0] ^= 1
    descriptor_path.write_bytes(drifted)

    with pytest.raises(GlueDerivedBuildError) as captured:
        build_glue_derived_dataset(
            data_root,
            raw_root,
            "b" * 64,
            tokenizer_root,
            "c" * 64,
            requirement,
            first.target_path,
            tokenizer_requirement=tokenizer_requirement,
            generator_git_commit="d" * 40,
        )

    assert captured.value.report.code == "TOKENIZER_DESCRIPTOR_HASH_MISMATCH"
    assert captured.value.report.staging_path is None


def test_tokenizer_descriptor_drift_during_build_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, raw_root, tokenizer_root, tokenizer_requirement, requirement = _fixture(
        tmp_path
    )
    monkeypatch.setattr(glue_builder, "_load_dependencies", _fake_dependencies)
    _FakeTokenizer.mutate_on_load = tokenizer_root / "tokenizer_config.json"
    target = data_root / "derived" / "sst2"

    with pytest.raises(GlueDerivedBuildError) as captured:
        build_glue_derived_dataset(
            data_root,
            raw_root,
            "b" * 64,
            tokenizer_root,
            "c" * 64,
            requirement,
            target,
            tokenizer_requirement=tokenizer_requirement,
            generator_git_commit="d" * 40,
        )

    assert captured.value.report.code == "TOKENIZER_SOURCE_CHANGED_DURING_BUILD"
    assert captured.value.report.staging_path is not None
    assert Path(captured.value.report.staging_path).is_dir()
    assert not target.exists()


def test_network_attempt_is_counted_blocked_and_preserves_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, raw_root, tokenizer_root, tokenizer_requirement, requirement = _fixture(
        tmp_path
    )
    monkeypatch.setattr(glue_builder, "_load_dependencies", _fake_dependencies)
    _FakeTokenizer.connect_on_load = True
    previous = os.environ.get("HF_HUB_OFFLINE")

    with pytest.raises(GlueDerivedBuildError) as captured:
        build_glue_derived_dataset(
            data_root,
            raw_root,
            "b" * 64,
            tokenizer_root,
            "c" * 64,
            requirement,
            data_root / "derived" / "sst2",
            tokenizer_requirement=tokenizer_requirement,
            generator_git_commit="d" * 40,
        )

    report = captured.value.report
    assert report.code == "NETWORK_EGRESS_BLOCKED"
    assert report.network_attempts == 1
    assert report.staging_path is not None
    assert Path(report.staging_path).is_dir()
    assert not (data_root / "derived" / "sst2").exists()
    assert os.environ.get("HF_HUB_OFFLINE") == previous


def test_child_process_attempt_is_counted_blocked_and_preserves_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, raw_root, tokenizer_root, tokenizer_requirement, requirement = _fixture(
        tmp_path
    )
    monkeypatch.setattr(glue_builder, "_load_dependencies", _fake_dependencies)
    _FakeTokenizer.subprocess_on_load = True
    original_popen = subprocess.Popen

    with pytest.raises(GlueDerivedBuildError) as captured:
        build_glue_derived_dataset(
            data_root,
            raw_root,
            "b" * 64,
            tokenizer_root,
            "c" * 64,
            requirement,
            data_root / "derived" / "sst2",
            tokenizer_requirement=tokenizer_requirement,
            generator_git_commit="d" * 40,
        )

    report = captured.value.report
    assert report.code == "NETWORK_EGRESS_BLOCKED"
    assert report.network_attempts == 1
    assert report.staging_path is not None
    assert Path(report.staging_path).is_dir()
    assert not (data_root / "derived" / "sst2").exists()
    assert subprocess.Popen is original_popen


def test_missing_optional_dependency_preserves_precise_staging_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, raw_root, tokenizer_root, tokenizer_requirement, requirement = _fixture(
        tmp_path
    )

    def missing() -> glue_builder._Dependencies:
        raise DependencyUnavailable(
            "datasets",
            feature="stage0_glue_derived_builder",
            install_extra="server",
        )

    monkeypatch.setattr(glue_builder, "_load_dependencies", missing)
    with pytest.raises(GlueDerivedBuildError) as captured:
        build_glue_derived_dataset(
            data_root,
            raw_root,
            "b" * 64,
            tokenizer_root,
            "c" * 64,
            requirement,
            data_root / "derived" / "sst2",
            tokenizer_requirement=tokenizer_requirement,
            generator_git_commit="d" * 40,
        )

    report = captured.value.report
    assert report.code == "DEPENDENCY_UNAVAILABLE"
    assert report.staging_path is not None
    assert Path(report.staging_path).is_dir()
    assert report.to_dict() == json.loads(json.dumps(report.to_dict()))


def test_out_of_range_label_fails_before_publication_and_keeps_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, raw_root, tokenizer_root, tokenizer_requirement, requirement = _fixture(
        tmp_path
    )
    _RAW_ROWS["train.parquet"][0]["label"] = 2
    monkeypatch.setattr(glue_builder, "_load_dependencies", _fake_dependencies)

    with pytest.raises(GlueDerivedBuildError) as captured:
        build_glue_derived_dataset(
            data_root,
            raw_root,
            "b" * 64,
            tokenizer_root,
            "c" * 64,
            requirement,
            data_root / "derived" / "sst2",
            tokenizer_requirement=tokenizer_requirement,
            generator_git_commit="d" * 40,
        )

    assert captured.value.report.code == "GLUE_LABEL_OUT_OF_RANGE"
    assert captured.value.report.staging_path is not None
    assert Path(captured.value.report.staging_path).is_dir()
    assert not (data_root / "derived" / "sst2").exists()
