"""无需外部资产的 tiny Torch 训练 fixture。

这些模型只用于本机合同、恢复和流水线测试，artifact 必须标记
``run_intent=local_fixture``，不得替代任何 Pythia/formal 结论。模型没有 dropout，
使“开启重要性观测不改变参数更新”的逐位比较具有清晰解释。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .training import InMemoryDatasetAdapter, TorchModelAdapter, TrainingMicrobatch


class TinyCausalLM(torch.nn.Module):
    """embedding 加线性词表头的最小 causal-LM logits 生成器。"""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        if vocab_size < 4 or hidden_size <= 0:
            raise ValueError("TINY_CAUSAL_LM_DIMENSION_INVALID")
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        return self.lm_head(self.embedding(input_ids))


class TinySequenceClassifier(torch.nn.Module):
    """masked mean pooling 加线性分类头。"""

    def __init__(self, vocab_size: int, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        if vocab_size < 4 or hidden_size <= 0 or num_labels < 2:
            raise ValueError("TINY_CLASSIFIER_DIMENSION_INVALID")
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        if attention_mask is None:
            return self.classifier(hidden.mean(dim=1))
        weight = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denominator = weight.sum(dim=1).clamp_min(1)
        return self.classifier((hidden * weight).sum(dim=1) / denominator)


@dataclass(frozen=True, slots=True)
class TinyTrainingFixture:
    """已经绑定模型适配器和确定性数据集的本机运行资源。"""

    model: TorchModelAdapter
    dataset: InMemoryDatasetAdapter


def build_tiny_training_fixture(
    *,
    task_type: str,
    seed: int,
    steps: int,
    microbatches_per_step: int = 2,
    microbatch_size: int = 2,
    sequence_length: int = 6,
    vocab_size: int = 17,
    hidden_size: int = 8,
    num_labels: int = 3,
) -> TinyTrainingFixture:
    """构造给定 seed 的完整 batch 计划；不读取全局 RNG 状态。"""

    for name, value in (
        ("steps", steps),
        ("microbatches_per_step", microbatches_per_step),
        ("microbatch_size", microbatch_size),
        ("sequence_length", sequence_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"TINY_FIXTURE_POSITIVE_INTEGER_REQUIRED:{name}")
    if sequence_length < 2:
        raise ValueError("TINY_FIXTURE_SEQUENCE_TOO_SHORT")
    # fork_rng 使初始化也不改变调用者的全局 CPU/CUDA RNG。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if task_type == "causal_lm":
            module: torch.nn.Module = TinyCausalLM(vocab_size, hidden_size)
        elif task_type == "sequence_classification":
            module = TinySequenceClassifier(vocab_size, hidden_size, num_labels)
        else:
            raise ValueError("TINY_FIXTURE_TASK_TYPE_UNSUPPORTED")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    plan: list[tuple[TrainingMicrobatch, ...]] = []
    for step in range(steps):
        micros: list[TrainingMicrobatch] = []
        for micro in range(microbatches_per_step):
            input_ids = torch.randint(
                0,
                vocab_size,
                (microbatch_size, sequence_length),
                generator=generator,
            )
            attention_mask = torch.ones_like(input_ids)
            if task_type == "causal_lm":
                labels = input_ids.clone()
            else:
                labels = (input_ids.sum(dim=1) % num_labels).to(dtype=torch.int64)
            sample_ids = tuple(
                f"tiny-{task_type}-s{step:04d}-m{micro:03d}-i{index:03d}"
                for index in range(microbatch_size)
            )
            micros.append(
                TrainingMicrobatch(
                    f"tiny-s{step:04d}-m{micro:03d}",
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    },
                    sample_ids,
                    {"scope": "local_fixture"},
                )
            )
        plan.append(tuple(micros))
    return TinyTrainingFixture(
        TorchModelAdapter(module, task_type=task_type),
        InMemoryDatasetAdapter(f"tiny-{task_type}-seed-{seed}", tuple(plan)),
    )


__all__ = [
    "TinyCausalLM",
    "TinySequenceClassifier",
    "TinyTrainingFixture",
    "build_tiny_training_fixture",
]
