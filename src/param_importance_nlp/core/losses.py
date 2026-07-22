"""与模型供应商无关的任务损失适配器。

适配器返回 ``loss_numerator``、``effective_count`` 与由两者定义的 mean loss，
从而让 microbatch 合并严格按统计单元（分类样本或有效目标 token）加权。调用方
不得先对不等有效计数的 microbatch mean 做无条件平均。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .errors import CoreContractError, NumericalError


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_binary_mask(mask: torch.Tensor, *, name: str) -> None:
    if mask.dtype != torch.bool and mask.dtype not in _INTEGER_DTYPES:
        raise CoreContractError(f"{name} 必须是 bool 或整数 0/1 张量")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise CoreContractError(f"{name} 只能包含 0/1")


@dataclass(frozen=True, slots=True)
class LossBatch:
    """一个 batch 的可加损失充分量。

    Parameters
    ----------
    loss_numerator:
        所有有效统计单元 loss 的和，必须是保留 autograd 图的标量张量。
    effective_count:
        有效样本数或有效目标 token 数，必须为正整数。
    statistical_unit:
        ``"sample"`` 或 ``"target_token"`` 等明确单位。
    """

    loss_numerator: torch.Tensor
    effective_count: int
    statistical_unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.loss_numerator, torch.Tensor) or self.loss_numerator.ndim != 0:
            raise CoreContractError("loss_numerator 必须是标量 torch.Tensor")
        if not isinstance(self.effective_count, int) or isinstance(self.effective_count, bool):
            raise CoreContractError("effective_count 必须是整数")
        if self.effective_count <= 0:
            raise CoreContractError("batch 没有有效损失单元")
        if not self.statistical_unit:
            raise CoreContractError("statistical_unit 不能为空")
        if not bool(torch.isfinite(self.loss_numerator.detach())):
            raise NumericalError("loss_numerator 为 NaN/Inf")

    @property
    def mean_loss(self) -> torch.Tensor:
        """按有效统计单元计算 mean，保留反向传播图。"""

        return self.loss_numerator / self.effective_count

    def merge(self, other: "LossBatch") -> "LossBatch":
        """合并两个 batch，而不是错误地平均两个 mean loss。"""

        if self.statistical_unit != other.statistical_unit:
            raise CoreContractError("不同 statistical_unit 的 LossBatch 不能合并")
        return LossBatch(
            self.loss_numerator + other.loss_numerator,
            self.effective_count + other.effective_count,
            self.statistical_unit,
        )


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> LossBatch:
    """计算标准 causal-LM shift loss。

    ``logits`` 形状为 ``[batch, sequence, vocabulary]``，``labels`` 为
    ``[batch, sequence]``。位置 ``j`` 的 logit 预测 ``labels[j+1]``，因此首个
    label 没有对应目标，最后一个 logit 不参与损失。``attention_mask`` 若提供，
    按目标位置右移后与 ``labels != ignore_index`` 取交集。

    返回的 numerator 是有效目标 token 的交叉熵之和，单位为 ``target_token``。
    该实现只依赖 Torch，不会导入 Transformers。
    """

    if logits.ndim != 3:
        raise CoreContractError("causal LM logits 必须为 [B, L, V]")
    if labels.ndim != 2 or tuple(labels.shape) != tuple(logits.shape[:2]):
        raise CoreContractError("causal LM labels 必须为 [B, L] 且与 logits 前两维一致")
    if labels.dtype not in _INTEGER_DTYPES:
        raise CoreContractError("causal LM labels 必须是整数张量")
    if logits.shape[1] < 2:
        raise CoreContractError("causal LM 序列长度必须至少为 2")
    if not logits.is_floating_point():
        raise CoreContractError("logits 必须是浮点张量")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(ignore_index)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(labels.shape):
            raise CoreContractError("attention_mask 必须与 labels 同 shape")
        _validate_binary_mask(attention_mask, name="attention_mask")
        valid = valid & attention_mask[:, 1:].to(dtype=torch.bool)
    effective_count = int(valid.sum().item())
    if effective_count <= 0:
        raise CoreContractError("causal LM batch 没有有效目标 token")

    flat_logits = shifted_logits.reshape(-1, shifted_logits.shape[-1])
    flat_labels = shifted_labels.reshape(-1)
    flat_valid = valid.reshape(-1)
    numerator = functional.cross_entropy(
        flat_logits[flat_valid],
        flat_labels[flat_valid].to(dtype=torch.long),
        reduction="sum",
    )
    return LossBatch(numerator, effective_count, "target_token")


def sequence_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> LossBatch:
    """计算单标签 sequence classification 的样本平均交叉熵。

    ``logits`` 必须为 ``[batch, classes]``，``labels`` 为 ``[batch]``。``mask``
    可排除 padding/占位样本；``ignore_index`` 也会排除对应标签。返回 numerator
    与有效样本数，使多个 microbatch 可以精确合并。
    """

    if logits.ndim != 2:
        raise CoreContractError("classification logits 必须为 [B, C]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise CoreContractError("classification labels 必须为 [B]")
    if labels.dtype not in _INTEGER_DTYPES:
        raise CoreContractError("classification labels 必须是整数张量")
    if logits.shape[1] < 2:
        raise CoreContractError("classification 至少需要两个类别")
    if not logits.is_floating_point():
        raise CoreContractError("logits 必须是浮点张量")

    valid = labels.ne(ignore_index)
    if mask is not None:
        if mask.ndim != 1 or tuple(mask.shape) != tuple(labels.shape):
            raise CoreContractError("classification mask 必须与 labels 同 shape")
        _validate_binary_mask(mask, name="classification mask")
        valid = valid & mask.to(dtype=torch.bool)
    effective_count = int(valid.sum().item())
    if effective_count <= 0:
        raise CoreContractError("classification batch 没有有效样本")
    numerator = functional.cross_entropy(
        logits[valid], labels[valid].to(dtype=torch.long), reduction="sum"
    )
    return LossBatch(numerator, effective_count, "sample")
