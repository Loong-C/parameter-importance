"""离线 Hugging Face 数据与任务指标适配层的 fake/local tiny 测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from param_importance_nlp.contracts import DependencyUnavailable
from param_importance_nlp.providers import huggingface_offline as offline
from param_importance_nlp.providers import training as provider_training
from param_importance_nlp.providers.fixed_state_torch import (
    TorchFixedStateGradientProvider,
)
from param_importance_nlp.providers.protocols import FrozenSampleResolver
from param_importance_nlp.providers.training import (
    DatasetAdapter,
    TaskEvaluator,
    TorchModelAdapter,
    TrainingMicrobatch,
)


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]], *, fingerprint: str) -> None:
        self.rows = rows
        self.column_names = list(rows[0])
        self._fingerprint = fingerprint

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class _FakeDatasetsModule:
    def __init__(self, loaded: object) -> None:
        self.loaded = loaded
        self.calls: list[str] = []

    def load_from_disk(self, path: str):
        self.calls.append(path)
        return self.loaded


def _asset_directory(tmp_path: Path, name: str = "dataset") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "dataset_info.json").write_text('{"fixture":true}\n', encoding="utf-8")
    (root / "data.arrow").write_bytes(b"tiny-local-arrow-fixture")
    return root


def _pile_rows(count: int = 8) -> list[dict[str, object]]:
    return [
        {
            "input_ids": [index % 3, 1, 2, 0],
            "attention_mask": [1, 1, 1, 1],
            "labels": [index % 2, 1, 0, -100 if index % 2 else 1],
        }
        for index in range(count)
    ]


def _glue_rows(count: int = 6) -> list[dict[str, object]]:
    return [
        {
            "input_ids": [index + 1, 2, 0, 0],
            "attention_mask": [1, 1, 0, 0],
            "token_type_ids": [0, 0, 0, 0],
            "label": index % 2,
        }
        for index in range(count)
    ]


def test_pile_load_from_disk_resolver_cursor_and_asset_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _asset_directory(tmp_path)
    dataset = _FakeDataset(_pile_rows(), fingerprint="pile-fingerprint")
    fake_module = _FakeDatasetsModule(dataset)
    dependency_calls: list[tuple[str, str]] = []

    def fake_dependency(name: str, *, feature: str):
        dependency_calls.append((name, feature))
        return fake_module

    monkeypatch.setattr(offline, "require_optional_dependency", fake_dependency)
    asset_hash = offline.hash_local_directory(root)
    adapter = offline.PretokenizedPileDatasetAdapter(
        root,
        dataset_id="pile-fixture-v1",
        microbatch_size=2,
        microbatches_per_step=1,
        sampling_design="uniform_with_replacement_over_frozen_rows",
        # 有效 token 数可与梯度相关；fixture 不伪造加权 U 无偏前提。
        weights_exogenous=False,
        common_mean_assumption=False,
        expected_asset_hash=asset_hash,
        allowed_root=tmp_path,
    )

    assert dependency_calls == [("datasets", "offline_pile_load_from_disk")]
    assert fake_module.calls == [str(root.resolve())]
    assert isinstance(adapter, DatasetAdapter)
    assert isinstance(adapter, FrozenSampleResolver)
    assert adapter.asset_hash == asset_hash
    assert adapter.row_count == 8
    assert adapter.sequence_length == 4
    resolved = adapter.resolve(adapter.sample_ids[0])
    assert resolved.payload["input_ids"].shape == (1, 4)
    assert resolved.payload["labels"].shape == (1, 4)
    assert resolved.sample_ids == (adapter.sample_ids[0],)

    provider = TorchFixedStateGradientProvider(
        TorchModelAdapter(_TinyLM(), task_type="causal_lm"),
        adapter,
        fixed_state_id="offline-pile-tiny-model",
    )
    provider_digest = provider.state_digest()
    gradient = provider.gradient(
        [SimpleNamespace(sample_id=adapter.sample_ids[0], draw_id="pile-draw-0")]
    )
    assert gradient.statistical_weight > 0
    assert gradient.sample_ids == (adapter.sample_ids[0],)
    assert provider.state_digest() == provider_digest

    first_cursor = adapter.cursor(seed=19)
    first_step = first_cursor.next_microbatches()
    state = first_cursor.state_dict()
    expected_second = first_cursor.next_microbatches()
    resumed = adapter.cursor(seed=19)
    resumed.load_state_dict(state)
    actual_second = resumed.next_microbatches()
    assert first_step[0].sample_ids != expected_second[0].sample_ids
    assert actual_second[0].sample_ids == expected_second[0].sample_ids
    assert torch.equal(
        actual_second[0].payload["input_ids"],
        expected_second[0].payload["input_ids"],
    )

    digest = adapter.state_digest()
    adapter.assert_unchanged(digest)
    (root / "dataset_info.json").write_text('{"fixture":false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="DIRECTORY_STAT_CHANGED"):
        adapter.assert_unchanged(digest)


def test_accessed_row_mutation_and_asset_hash_mismatch_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _asset_directory(tmp_path)
    dataset = _FakeDataset(_pile_rows(), fingerprint="pile-fingerprint")
    fake_module = _FakeDatasetsModule(dataset)
    dependency_calls = 0

    def fake_dependency(name: str, *, feature: str):
        nonlocal dependency_calls
        dependency_calls += 1
        return fake_module

    monkeypatch.setattr(offline, "require_optional_dependency", fake_dependency)
    with pytest.raises(ValueError, match="ASSET_HASH_MISMATCH"):
        offline.PretokenizedPileDatasetAdapter(
            root,
            dataset_id="bad-hash",
            microbatch_size=1,
            microbatches_per_step=1,
            sampling_design="frozen",
            weights_exogenous=False,
            common_mean_assumption=False,
            expected_asset_hash="0" * 64,
        )
    assert dependency_calls == 0  # hash 错误时不解析可选 HF 依赖

    adapter = offline.PretokenizedPileDatasetAdapter(
        root,
        dataset_id="row-mutation",
        microbatch_size=1,
        microbatches_per_step=1,
        sampling_design="frozen",
        weights_exogenous=False,
        common_mean_assumption=False,
    )
    dataset.rows[0]["input_ids"] = [9, 9, 9, 9]
    with pytest.raises(RuntimeError, match="ROW_MUTATED"):
        adapter.resolve(adapter.sample_ids[0])


def test_glue_sst2_split_label_normalization_and_mnli_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _asset_directory(tmp_path)
    train = _FakeDataset(_glue_rows(), fingerprint="sst2-train")
    validation = _FakeDataset(_glue_rows(), fingerprint="sst2-validation")
    fake_module = _FakeDatasetsModule({"train": train, "validation": validation})
    monkeypatch.setattr(
        offline,
        "require_optional_dependency",
        lambda name, *, feature: fake_module,
    )
    adapter = offline.PretokenizedGlueDatasetAdapter(
        root,
        task_name="SST-2",
        split="validation",
        dataset_id="sst2-fixture-v1",
        microbatch_size=2,
        microbatches_per_step=1,
    )
    assert adapter.task_name == "sst2"
    assert adapter.loss_unit == "sample"
    assert adapter.weights_exogenous is True
    single = adapter.resolve(adapter.sample_ids[1])
    assert single.payload["labels"].shape == (1,)
    assert single.payload["token_type_ids"].shape == (1, 4)
    batch = adapter.cursor(seed=4).next_microbatches()[0]
    assert batch.payload["input_ids"].shape == (2, 4)
    assert batch.payload["labels"].shape == (2,)

    invalid_mnli = _FakeDataset(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "label": 3,
            }
        ],
        fingerprint="mnli-invalid",
    )
    monkeypatch.setattr(
        offline,
        "require_optional_dependency",
        lambda name, *, feature: _FakeDatasetsModule(
            {"validation_matched": invalid_mnli}
        ),
    )
    with pytest.raises(ValueError, match="GLUE_LABEL_OUT_OF_RANGE"):
        offline.PretokenizedGlueDatasetAdapter(
            root,
            task_name="mnli",
            split="validation_matched",
            dataset_id="mnli-invalid",
            microbatch_size=1,
            microbatches_per_step=1,
        )


def test_remote_or_missing_path_fails_before_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def forbidden_dependency(name: str, *, feature: str):
        nonlocal calls
        calls += 1
        raise AssertionError("optional dependency should not be imported")

    monkeypatch.setattr(offline, "require_optional_dependency", forbidden_dependency)
    common = dict(
        dataset_id="offline-only",
        microbatch_size=1,
        microbatches_per_step=1,
        sampling_design="frozen",
        weights_exogenous=False,
        common_mean_assumption=False,
    )
    with pytest.raises(ValueError, match="REMOTE_REFERENCE_FORBIDDEN"):
        offline.PretokenizedPileDatasetAdapter("https://huggingface.co/datasets/x", **common)
    with pytest.raises(FileNotFoundError, match="DIRECTORY_NOT_FOUND"):
        offline.PretokenizedPileDatasetAdapter(tmp_path / "missing", **common)
    assert calls == 0

    root = _asset_directory(tmp_path, "valid-but-dependency-missing")

    def missing_dependency(name: str, *, feature: str):
        raise DependencyUnavailable(name, feature=feature, install_extra="server")

    monkeypatch.setattr(offline, "require_optional_dependency", missing_dependency)
    with pytest.raises(DependencyUnavailable, match="DEPENDENCY_UNAVAILABLE:datasets"):
        offline.PretokenizedPileDatasetAdapter(root, **common)


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        score = self.scale * input_ids.to(dtype=torch.float32).sum(dim=1)
        return {"logits": torch.stack((-score, score), dim=-1)}


class _TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids, attention_mask=None):
        score = self.scale * input_ids.to(dtype=torch.float32)
        return {"logits": torch.stack((score, -score), dim=-1)}


def test_task_metric_evaluator_dispatches_glue_and_pile_without_state_drift() -> None:
    classifier = _TinyClassifier()
    classification_batch = TrainingMicrobatch(
        "classification-eval",
        {
            "input_ids": torch.tensor([[1, 0], [2, 0]], dtype=torch.int64),
            "attention_mask": torch.ones((2, 2), dtype=torch.int64),
            "labels": torch.tensor([1, 0], dtype=torch.int64),
        },
        ("a", "b"),
    )
    sst2 = offline.HuggingFaceTaskMetricEvaluator("sst-2")
    assert isinstance(sst2, TaskEvaluator)
    before_parameter = classifier.scale.detach().clone()
    before_rng = torch.random.get_rng_state().clone()
    metrics = sst2.evaluate(
        TorchModelAdapter(classifier, task_type="sequence_classification"),
        (classification_batch,),
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["sst2_accuracy"] == pytest.approx(0.5)
    torch.testing.assert_close(classifier.scale, before_parameter)
    assert torch.equal(torch.random.get_rng_state(), before_rng)

    lm = _TinyLM()
    lm_batch = TrainingMicrobatch(
        "pile-eval",
        {
            "input_ids": torch.tensor([[0, 1, 0]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
            "labels": torch.tensor([[0, 1, 0]], dtype=torch.int64),
        },
        ("pile-row",),
    )
    pile_metrics = offline.HuggingFaceTaskMetricEvaluator("pile").evaluate(
        TorchModelAdapter(lm, task_type="causal_lm"),
        (lm_batch,),
    )
    assert pile_metrics["pile_loss"] == pytest.approx(pile_metrics["loss"])
    assert pile_metrics["pile_perplexity"] == pytest.approx(
        pile_metrics["perplexity"]
    )


def test_existing_hf_model_and_tokenizer_boundaries_force_local_files_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = _asset_directory(tmp_path, "local-model")
    model_calls: list[tuple[str, dict[str, object]]] = []
    tokenizer_calls: list[tuple[str, dict[str, object]]] = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            model_calls.append((path, dict(kwargs)))
            return _TinyLM()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            tokenizer_calls.append((path, dict(kwargs)))
            return object()

    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )
    monkeypatch.setattr(
        provider_training,
        "require_optional_dependency",
        lambda name, *, feature: fake_transformers,
    )
    provider_training.OfflineHuggingFaceModelAdapter.from_local_directory(
        model_root, task_type="causal_lm"
    )
    provider_training.OfflineTokenizer.from_local_directory(model_root)
    assert model_calls == [
        (str(model_root.resolve()), {"local_files_only": True})
    ]
    assert tokenizer_calls == [
        (str(model_root.resolve()), {"local_files_only": True})
    ]
