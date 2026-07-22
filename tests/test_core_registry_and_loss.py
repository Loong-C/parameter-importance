from __future__ import annotations

import torch
import torch.nn.functional as functional

from param_importance_nlp.core import (
    CoreContractError,
    ParameterRegistry,
    RegistryError,
    TensorMap,
    TensorMapError,
    causal_lm_loss,
    sequence_classification_loss,
)


class _AliasedModel(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.left = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
        self.right = torch.nn.Module()
        self.right.weight = self.left.weight
        self.frozen = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)


def test_registry_alias_eligibility_and_three_hash_boundaries() -> None:
    model32 = _AliasedModel(dtype=torch.float32)
    optimizer32 = torch.optim.SGD([{"params": [model32.left.weight], "lr": 0.1, "momentum": 0.9}])
    registry32 = ParameterRegistry.from_model(model32, optimizer32)

    assert registry32.canonical_name("right.weight") == "left.weight"
    assert registry32.record("left.weight").aliases == ("right.weight",)
    assert registry32.eligible_names == ("left.weight",)
    assert registry32.record("frozen").eligible is False
    assert registry32.record("frozen").eligibility_reason == "requires_grad_false"

    # dtype/device 只影响 runtime layout，不能改变“研究哪些坐标”的身份。
    model64 = _AliasedModel(dtype=torch.float64)
    optimizer64 = torch.optim.SGD([{"params": [model64.left.weight], "lr": 0.7, "momentum": 0.9}])
    registry64 = ParameterRegistry.from_model(model64, optimizer64)
    assert registry32.coordinate_registry_hash == registry64.coordinate_registry_hash
    assert registry32.optimizer_contract_hash == registry64.optimizer_contract_hash
    assert registry32.runtime_layout_hash != registry64.runtime_layout_hash


def test_registry_rejects_cross_group_duplicate_and_overlapping_storage() -> None:
    model = _AliasedModel()
    # torch optimizer 本身也拒绝跨组重复；直接构造后篡改只用于验证 registry 防线。
    optimizer = torch.optim.SGD([model.left.weight], lr=0.1)
    optimizer.param_groups.append({**optimizer.param_groups[0], "params": [model.left.weight]})
    try:
        ParameterRegistry.from_model(model, optimizer)
    except RegistryError as exc:
        assert "跨 optimizer 参数组" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("跨组重复参数必须被拒绝")

    base = torch.arange(6.0)
    overlap_model = torch.nn.Module()
    overlap_model.first = torch.nn.Parameter(base[:4])
    overlap_model.second = torch.nn.Parameter(base[2:])
    overlap_optimizer = torch.optim.SGD(overlap_model.parameters(), lr=0.1)
    try:
        ParameterRegistry.from_model(overlap_model, overlap_optimizer)
    except RegistryError as exc:
        assert "storage 区间重叠" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("不同 Parameter 的重叠 storage 必须被拒绝")


def test_registry_rejects_current_sparse_gradient() -> None:
    embedding = torch.nn.Embedding(8, 3, sparse=True)
    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
    embedding(torch.tensor([1, 2])).sum().backward()
    assert embedding.weight.grad is not None and embedding.weight.grad.is_sparse
    try:
        ParameterRegistry.from_model(embedding, optimizer)
    except RegistryError as exc:
        assert "sparse gradient" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("sparse gradient 必须被拒绝")


def test_tensor_map_alias_normalization_shape_and_finite_guards() -> None:
    model = _AliasedModel()
    optimizer = torch.optim.SGD([model.left.weight], lr=0.1)
    registry = ParameterRegistry.from_model(model, optimizer)
    tensors = TensorMap({"right.weight": torch.ones(2, 2)}, registry=registry)
    assert tuple(tensors) == ("left.weight",)
    assert tensors["right.weight"].shape == (2, 2)

    try:
        TensorMap({"left.weight": torch.ones(4)}, registry=registry)
    except RegistryError:
        pass
    else:  # pragma: no cover
        raise AssertionError("shape mismatch 必须失败")
    try:
        TensorMap({"x": torch.tensor([float("nan")])})
    except CoreContractError:
        pass
    else:  # pragma: no cover
        raise AssertionError("非有限 TensorMap 必须失败")

    left = TensorMap({"a": torch.ones(2)})
    right = TensorMap({"b": torch.ones(2)})
    try:
        left.assert_compatible(right)
    except TensorMapError:
        pass
    else:  # pragma: no cover
        raise AssertionError("不同坐标的 TensorMap 必须失败")


def test_causal_lm_loss_shift_mask_and_gradient() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [4.0, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[2, 0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    batch = causal_lm_loss(logits, labels, attention_mask)
    expected = functional.cross_entropy(logits[:, :2].reshape(-1, 3), labels[:, 1:3].reshape(-1), reduction="sum")
    assert batch.effective_count == 2
    assert batch.statistical_unit == "target_token"
    torch.testing.assert_close(batch.loss_numerator, expected)
    batch.mean_loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0


def test_classification_loss_and_weighted_merge() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1, -100])
    batch = sequence_classification_loss(logits, labels)
    expected = functional.cross_entropy(logits[:2], labels[:2], reduction="sum")
    assert batch.effective_count == 2
    torch.testing.assert_close(batch.loss_numerator, expected)

    other = sequence_classification_loss(torch.tensor([[0.0, 2.0]]), torch.tensor([1]))
    merged = batch.merge(other)
    assert merged.effective_count == 3
    torch.testing.assert_close(merged.mean_loss, (batch.loss_numerator + other.loss_numerator) / 3)

    try:
        causal_lm_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.long))
    except CoreContractError:
        pass
    else:  # pragma: no cover
        raise AssertionError("长度 1 的 causal LM 必须失败")

