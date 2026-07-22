"""训练模型、批数据与任务评估的依赖反转接口。

训练引擎只依赖这里的协议，因此 tiny Torch fixture 与服务器上的 Pythia、
Pile、GLUE 适配器共享同一 step 生命周期。任何 Hugging Face 入口都要求本地
路径并使用 ``local_files_only=True``；导入本模块不会访问网络或下载模型。
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, runtime_checkable

import torch

from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from ..core.losses import LossBatch, causal_lm_loss, sequence_classification_loss
from .optional import require_optional_dependency


def _clone_payload(payload: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """复制 batch 张量并验证共同的第一维，防止预取缓冲区复用污染记录。"""

    if not payload:
        raise ValueError("TRAINING_MICROBATCH_PAYLOAD_EMPTY")
    copied: dict[str, torch.Tensor] = {}
    batch_size: int | None = None
    for name, value in payload.items():
        if not isinstance(name, str) or not name:
            raise TypeError("TRAINING_MICROBATCH_FIELD_NAME_INVALID")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"TRAINING_MICROBATCH_FIELD_NOT_TENSOR:{name}")
        if value.ndim == 0:
            raise ValueError(f"TRAINING_MICROBATCH_FIELD_MISSING_BATCH_DIM:{name}")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif int(value.shape[0]) != batch_size:
            raise ValueError("TRAINING_MICROBATCH_BATCH_DIM_MISMATCH")
        copied[name] = value.detach().clone()
    if batch_size is None or batch_size <= 0:
        raise ValueError("TRAINING_MICROBATCH_BATCH_SIZE_INVALID")
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class TrainingMicrobatch:
    """一个可重放的训练 microbatch。

    ``sample_ids`` 保留原始顺序和重复项，便于核验 sampler/cursor 恢复。payload
    只允许带 batch 维的 dense Torch tensor，并在构造时防御性复制。
    """

    batch_id: str
    payload: Mapping[str, torch.Tensor]
    sample_ids: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("TRAINING_MICROBATCH_ID_EMPTY")
        copied = _clone_payload(self.payload)
        size = int(next(iter(copied.values())).shape[0])
        if len(self.sample_ids) != size or not all(
            isinstance(value, str) and value for value in self.sample_ids
        ):
            raise ValueError("TRAINING_MICROBATCH_SAMPLE_IDS_INVALID")
        object.__setattr__(self, "payload", copied)
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="TrainingMicrobatch.metadata"),
        )

    def to(self, device: torch.device | str) -> "TrainingMicrobatch":
        """显式搬运到目标 device；不修改 cursor 中的源 batch。"""

        return TrainingMicrobatch(
            self.batch_id,
            {name: value.to(device=device) for name, value in self.payload.items()},
            self.sample_ids,
            self.metadata,
        )


@runtime_checkable
class BatchCursor(Protocol):
    """按 optimizer attempt 返回一组有序 microbatch 的可恢复游标。"""

    def next_microbatches(self) -> tuple[TrainingMicrobatch, ...]: ...

    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


@runtime_checkable
class DatasetAdapter(Protocol):
    """从冻结数据资产构造独立 cursor；不得隐式联网。"""

    @property
    def dataset_id(self) -> str: ...

    def cursor(self, *, seed: int, rank: int = 0, world_size: int = 1) -> BatchCursor: ...


@runtime_checkable
class ModelAdapter(Protocol):
    """把供应商模型输出规范成项目自己的 :class:`LossBatch`。"""

    @property
    def module(self) -> torch.nn.Module: ...

    @property
    def task_type(self) -> str: ...

    def loss(self, microbatch: TrainingMicrobatch) -> LossBatch: ...


@runtime_checkable
class TaskEvaluator(Protocol):
    """只读评估器；调用前后不得改变模型参数、buffer 或 RNG。"""

    def evaluate(
        self,
        model: ModelAdapter,
        microbatches: Sequence[TrainingMicrobatch],
    ) -> Mapping[str, float]: ...


class DeterministicBatchCursor:
    """内存 batch 计划的严格 cursor，供 fixture 与 adapter 合约测试复用。"""

    def __init__(self, steps: Sequence[Sequence[TrainingMicrobatch]]) -> None:
        if not steps or any(not step for step in steps):
            raise ValueError("BATCH_CURSOR_STEPS_EMPTY")
        self._steps = tuple(tuple(step) for step in steps)
        self._index = 0

    def next_microbatches(self) -> tuple[TrainingMicrobatch, ...]:
        if self._index >= len(self._steps):
            raise StopIteration("BATCH_CURSOR_EXHAUSTED")
        value = self._steps[self._index]
        self._index += 1
        return value

    def state_dict(self) -> Mapping[str, object]:
        return {"schema_version": "batch-cursor-state-v1", "index": self._index}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"schema_version", "index"}:
            raise ValueError("BATCH_CURSOR_STATE_FIELDS_MISMATCH")
        if state["schema_version"] != "batch-cursor-state-v1":
            raise ValueError("BATCH_CURSOR_STATE_VERSION_UNSUPPORTED")
        index = state["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("BATCH_CURSOR_STATE_INDEX_NOT_INTEGER")
        if not 0 <= index <= len(self._steps):
            raise ValueError("BATCH_CURSOR_STATE_INDEX_OUT_OF_RANGE")
        self._index = index


class PrefetchBatchCursor:
    """可 checkpoint 的有序后台预取游标。

    ``num_workers`` 个线程可以提前构造后续 optimizer step，但底层 cursor 的读取
    仍按提交 ticket 严格串行，因而 worker 调度不会改变 sample 顺序。checkpoint
    会等待已提交预取完成，把尚未消费的完整 microbatch（含 tensor）写入安全状态
    树，并同时保存已经前移的源 cursor 状态；恢复时先重放 pending，再从该源状态
    继续，既不丢样本也不重复样本。

    这里使用线程而不是任意 pickle 的多进程 DataLoader。离线 HF 数据已经驻留在
    只读本地 dataset 对象中，线程预取避免重新序列化供应商对象，也符合 Windows
    fresh-process 恢复边界。
    """

    def __init__(
        self,
        source: BatchCursor,
        *,
        num_workers: int,
        prefetch_factor: int,
        persistent_workers: bool,
    ) -> None:
        if not isinstance(source, BatchCursor):
            raise TypeError("PREFETCH_CURSOR_SOURCE_PROTOCOL_INVALID")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (num_workers, prefetch_factor)
        ):
            raise ValueError("PREFETCH_CURSOR_POSITIVE_OPTIONS_REQUIRED")
        if type(persistent_workers) is not bool:
            raise TypeError("PREFETCH_CURSOR_PERSISTENT_WORKERS_NOT_BOOLEAN")
        self.source = source
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self._capacity = num_workers * prefetch_factor
        self._ready: deque[tuple[TrainingMicrobatch, ...]] = deque()
        self._futures: deque[Future[tuple[TrainingMicrobatch, ...]]] = deque()
        self._condition = threading.Condition()
        self._submitted_ticket = 0
        self._read_ticket = 0
        self._source_exhausted = False
        self._executor: ThreadPoolExecutor | None = None
        self._ensure_executor()
        self._fill()

    def _ensure_executor(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="param-importance-prefetch",
            )

    def _ordered_read(self, ticket: int) -> tuple[TrainingMicrobatch, ...]:
        # Condition 只冻结底层 cursor 的次序；batch 构造完成后 future 仍可与模型
        # forward/backward 重叠。异常也必须推进 ticket，否则后续 worker 会死锁。
        with self._condition:
            self._condition.wait_for(lambda: ticket == self._read_ticket)
            try:
                return self.source.next_microbatches()
            finally:
                self._read_ticket += 1
                self._condition.notify_all()

    def _fill(self) -> None:
        if self._source_exhausted:
            return
        self._ensure_executor()
        assert self._executor is not None
        missing = self._capacity - len(self._ready) - len(self._futures)
        for _ in range(max(missing, 0)):
            ticket = self._submitted_ticket
            self._submitted_ticket += 1
            self._futures.append(self._executor.submit(self._ordered_read, ticket))

    def _take_future(self) -> tuple[TrainingMicrobatch, ...]:
        if not self._futures:
            self._fill()
        if not self._futures:
            raise StopIteration("PREFETCH_BATCH_CURSOR_EXHAUSTED")
        future = self._futures.popleft()
        try:
            return future.result()
        except StopIteration as error:
            self._source_exhausted = True
            for pending in self._futures:
                try:
                    pending.result()
                except StopIteration:
                    pass
            self._futures.clear()
            raise StopIteration("PREFETCH_BATCH_CURSOR_EXHAUSTED") from error

    def next_microbatches(self) -> tuple[TrainingMicrobatch, ...]:
        if self._ready:
            value = self._ready.popleft()
        else:
            value = self._take_future()
        self._fill()
        return value

    @staticmethod
    def _batch_to_state(batch: TrainingMicrobatch) -> Mapping[str, object]:
        return {
            "batch_id": batch.batch_id,
            "payload": {
                name: tensor.detach().to(device="cpu").clone()
                for name, tensor in batch.payload.items()
            },
            "sample_ids": list(batch.sample_ids),
            "metadata": thaw_json_value(batch.metadata),
        }

    @staticmethod
    def _batch_from_state(value: object) -> TrainingMicrobatch:
        expected = {"batch_id", "payload", "sample_ids", "metadata"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("PREFETCH_CURSOR_PENDING_BATCH_FIELDS_INVALID")
        payload = value["payload"]
        sample_ids = value["sample_ids"]
        metadata = value["metadata"]
        if not isinstance(payload, Mapping) or not all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in payload.items()
        ):
            raise TypeError("PREFETCH_CURSOR_PENDING_PAYLOAD_INVALID")
        if not isinstance(sample_ids, list) or not all(
            isinstance(item, str) and item for item in sample_ids
        ):
            raise TypeError("PREFETCH_CURSOR_PENDING_SAMPLE_IDS_INVALID")
        if not isinstance(metadata, Mapping):
            raise TypeError("PREFETCH_CURSOR_PENDING_METADATA_INVALID")
        return TrainingMicrobatch(
            value["batch_id"],  # type: ignore[arg-type]
            dict(payload),  # type: ignore[arg-type]
            tuple(sample_ids),
            metadata,
        )

    def _drain_futures(self) -> None:
        while self._futures:
            try:
                self._ready.append(self._take_future())
            except StopIteration:
                break

    def state_dict(self) -> Mapping[str, object]:
        # 预取会让 source 指针领先于“已消费”指针；pending 是二者之间的精确差。
        # 此方法不再 refill，因此在同一 attempt 内重复调用是幂等的。
        self._drain_futures()
        if not self.persistent_workers:
            self._shutdown_executor()
        return {
            "schema_version": "prefetch-batch-cursor-state-v1",
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "source_state": dict(self.source.state_dict()),
            "pending": [
                [self._batch_to_state(batch) for batch in step]
                for step in self._ready
            ],
            "source_exhausted": self._source_exhausted,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "schema_version", "num_workers", "prefetch_factor",
            "persistent_workers", "source_state", "pending", "source_exhausted",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("PREFETCH_CURSOR_STATE_FIELDS_INVALID")
        identities = {
            "schema_version": "prefetch-batch-cursor-state-v1",
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
        }
        if any(state[name] != expected_value for name, expected_value in identities.items()):
            raise ValueError("PREFETCH_CURSOR_STATE_IDENTITY_MISMATCH")
        source_state = state["source_state"]
        pending = state["pending"]
        exhausted = state["source_exhausted"]
        if not isinstance(source_state, Mapping):
            raise TypeError("PREFETCH_CURSOR_SOURCE_STATE_INVALID")
        if not isinstance(pending, list) or not all(isinstance(step, list) for step in pending):
            raise TypeError("PREFETCH_CURSOR_PENDING_NOT_ARRAY")
        if type(exhausted) is not bool:
            raise TypeError("PREFETCH_CURSOR_EXHAUSTED_NOT_BOOLEAN")
        self._shutdown_executor()
        self.source.load_state_dict(source_state)
        self._ready = deque(
            tuple(self._batch_from_state(batch) for batch in step) for step in pending
        )
        self._futures.clear()
        self._source_exhausted = exhausted
        # source state 已包含 pending 之后的位置；新 ticket 只需保持本进程内部顺序。
        self._submitted_ticket = 0
        self._read_ticket = 0
        self._ensure_executor()
        self._fill()

    def _shutdown_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def close(self) -> None:
        """等待后台读取结束并释放线程；不修改已预取的业务顺序。"""

        self._drain_futures()
        self._shutdown_executor()

    def __del__(self) -> None:  # pragma: no cover - 仅作解释器退出防线
        try:
            self._shutdown_executor()
        except Exception:
            pass


def configure_batch_cursor(
    source: BatchCursor,
    *,
    num_workers: int,
    prefetch_factor: int | None,
    persistent_workers: bool,
) -> BatchCursor:
    """把严格 v2 data_loader 选项应用到 cursor；``num_workers=0`` 保持直连。"""

    if num_workers == 0:
        if prefetch_factor is not None or persistent_workers:
            raise ValueError("DIRECT_CURSOR_CANNOT_ENABLE_PREFETCH_OPTIONS")
        return source
    if prefetch_factor is None:
        raise ValueError("PREFETCH_CURSOR_FACTOR_REQUIRED")
    return PrefetchBatchCursor(
        source,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )


@dataclass(frozen=True, slots=True)
class InMemoryDatasetAdapter:
    """已经切分好 microbatch 的小型确定性数据集。"""

    dataset_id: str
    steps: tuple[tuple[TrainingMicrobatch, ...], ...]

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.steps or any(not step for step in self.steps):
            raise ValueError("IN_MEMORY_DATASET_INVALID")

    def cursor(self, *, seed: int, rank: int = 0, world_size: int = 1) -> BatchCursor:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("DATASET_SEED_NOT_INTEGER")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("DATASET_RANK_WORLD_SIZE_INVALID")
        sharded = self.steps[rank::world_size]
        if not sharded:
            raise ValueError("DATASET_SHARD_EMPTY")
        return DeterministicBatchCursor(sharded)


def _extract_logits(outputs: object) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, Mapping) and isinstance(outputs.get("logits"), torch.Tensor):
        return outputs["logits"]
    logits = getattr(outputs, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError("MODEL_OUTPUT_MISSING_LOGITS")


class TorchModelAdapter:
    """任意 Torch module 的 causal-LM/classification 损失适配器。"""

    def __init__(self, module: torch.nn.Module, *, task_type: str) -> None:
        if task_type not in {"causal_lm", "sequence_classification"}:
            raise ValueError("MODEL_TASK_TYPE_UNSUPPORTED")
        self._module = module
        self._task_type = task_type

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    @property
    def task_type(self) -> str:
        return self._task_type

    def logits(self, microbatch: TrainingMicrobatch) -> torch.Tensor:
        payload = dict(microbatch.payload)
        labels = payload.pop("labels", None)
        payload.pop("classification_mask", None)
        outputs = self._module(**payload)
        if labels is None:
            raise ValueError("TRAINING_BATCH_LABELS_REQUIRED")
        return _extract_logits(outputs)

    def loss(self, microbatch: TrainingMicrobatch) -> LossBatch:
        labels = microbatch.payload.get("labels")
        if labels is None:
            raise ValueError("TRAINING_BATCH_LABELS_REQUIRED")
        logits = self.logits(microbatch)
        if self._task_type == "causal_lm":
            return causal_lm_loss(
                logits, labels, microbatch.payload.get("attention_mask")
            )
        return sequence_classification_loss(
            logits, labels, microbatch.payload.get("classification_mask")
        )


class CausalLMEvaluator:
    """按有效 target token 汇总 validation loss/perplexity。"""

    def evaluate(
        self,
        model: ModelAdapter,
        microbatches: Sequence[TrainingMicrobatch],
    ) -> Mapping[str, float]:
        if model.task_type != "causal_lm":
            raise ValueError("CAUSAL_LM_EVALUATOR_TASK_MISMATCH")
        was_training = model.module.training
        numerator = 0.0
        count = 0
        model.module.eval()
        try:
            with torch.no_grad():
                for batch in microbatches:
                    loss = model.loss(batch)
                    numerator += float(loss.loss_numerator.detach().cpu().item())
                    count += loss.effective_count
        finally:
            model.module.train(was_training)
        if count <= 0:
            raise ValueError("EVALUATION_EFFECTIVE_COUNT_ZERO")
        mean = numerator / count
        # canonical artifact 禁止 Infinity；溢出不是“非常大的有效数”，因此不能
        # 写入 inf 或任意截断值伪装精确结果。
        try:
            perplexity = math.exp(mean)
        except OverflowError as exc:
            raise OverflowError("EVALUATION_PERPLEXITY_UNDEFINED_OVERFLOW") from exc
        if not math.isfinite(perplexity):
            raise OverflowError("EVALUATION_PERPLEXITY_UNDEFINED_OVERFLOW")
        return {"loss": mean, "perplexity": perplexity}


class ClassificationEvaluator:
    """计算样本加权 loss 与 accuracy。"""

    def evaluate(
        self,
        model: ModelAdapter,
        microbatches: Sequence[TrainingMicrobatch],
    ) -> Mapping[str, float]:
        if model.task_type != "sequence_classification":
            raise ValueError("CLASSIFICATION_EVALUATOR_TASK_MISMATCH")
        if not isinstance(model, TorchModelAdapter):
            raise TypeError("CLASSIFICATION_EVALUATOR_REQUIRES_LOGITS_ADAPTER")
        was_training = model.module.training
        numerator = 0.0
        count = 0
        correct = 0
        model.module.eval()
        try:
            with torch.no_grad():
                for batch in microbatches:
                    labels = batch.payload["labels"]
                    mask = labels.ne(-100)
                    extra_mask = batch.payload.get("classification_mask")
                    if extra_mask is not None:
                        mask = mask & extra_mask.to(dtype=torch.bool)
                    logits = model.logits(batch)
                    loss = sequence_classification_loss(logits, labels, extra_mask)
                    numerator += float(loss.loss_numerator.detach().cpu().item())
                    count += loss.effective_count
                    correct += int((logits.argmax(dim=-1)[mask] == labels[mask]).sum().item())
        finally:
            model.module.train(was_training)
        if count <= 0:
            raise ValueError("EVALUATION_EFFECTIVE_COUNT_ZERO")
        return {"loss": numerator / count, "accuracy": correct / count}


class OfflineHuggingFaceModelAdapter(TorchModelAdapter):
    """从已验证本地目录加载 Pythia/HF 模型，绝不使用 Hub fallback。"""

    @classmethod
    def from_local_directory(
        cls,
        path: str | Path,
        *,
        task_type: str,
        num_labels: int | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> "OfflineHuggingFaceModelAdapter":
        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"OFFLINE_MODEL_DIRECTORY_NOT_FOUND:{root}")
        transformers = require_optional_dependency(
            "transformers", feature="offline_huggingface_model"
        )
        common: dict[str, object] = {"local_files_only": True}
        if torch_dtype is not None:
            common["torch_dtype"] = torch_dtype
        if task_type == "causal_lm":
            model = transformers.AutoModelForCausalLM.from_pretrained(str(root), **common)
        elif task_type == "sequence_classification":
            if num_labels is None or num_labels < 2:
                raise ValueError("CLASSIFICATION_NUM_LABELS_INVALID")
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(root), num_labels=num_labels, **common
            )
        else:
            raise ValueError("MODEL_TASK_TYPE_UNSUPPORTED")
        return cls(model, task_type=task_type)


class OfflineTokenizer:
    """严格本地 tokenizer 边界；显式构造前不会导入 Transformers。"""

    @staticmethod
    def from_local_directory(path: str | Path) -> object:
        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"OFFLINE_TOKENIZER_DIRECTORY_NOT_FOUND:{root}")
        transformers = require_optional_dependency(
            "transformers", feature="offline_huggingface_tokenizer"
        )
        return transformers.AutoTokenizer.from_pretrained(str(root), local_files_only=True)


class PretokenizedJsonlDatasetAdapter:
    """读取本地预分词 JSONL，并冻结为确定性 microbatch 计划。

    每行包含 ``sample_id`` 及声明的 tensor 字段数组。padding/truncation 必须在
    资产构建阶段完成；同一 microbatch 不能依赖 runner 猜测 padding 语义。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_id: str,
        microbatch_size: int,
        microbatches_per_step: int,
        tensor_fields: Sequence[str],
    ) -> None:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"PRETOKENIZED_JSONL_NOT_FOUND:{source}")
        if microbatch_size <= 0 or microbatches_per_step <= 0:
            raise ValueError("DATASET_BATCH_SIZE_INVALID")
        fields = tuple(tensor_fields)
        if not fields or "labels" not in fields or len(set(fields)) != len(fields):
            raise ValueError("DATASET_TENSOR_FIELDS_INVALID")
        rows: list[tuple[str, dict[str, torch.Tensor]]] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"DATASET_JSON_INVALID:{line_number}") from exc
            if not isinstance(value, dict) or not isinstance(value.get("sample_id"), str):
                raise ValueError(f"DATASET_ROW_ID_INVALID:{line_number}")
            if set(value) != {"sample_id", *fields}:
                raise ValueError(f"DATASET_ROW_FIELDS_MISMATCH:{line_number}")
            tensors: dict[str, torch.Tensor] = {}
            for field_name in fields:
                raw = value[field_name]
                if not isinstance(raw, list):
                    raise TypeError(f"DATASET_ROW_FIELD_NOT_ARRAY:{line_number}:{field_name}")
                tensors[field_name] = torch.tensor(raw, dtype=torch.long)
            rows.append((value["sample_id"], tensors))
        group = microbatch_size * microbatches_per_step
        if len(rows) < group:
            raise ValueError("DATASET_NOT_ENOUGH_ROWS_FOR_ONE_STEP")
        steps: list[tuple[TrainingMicrobatch, ...]] = []
        for step_start in range(0, len(rows) - group + 1, group):
            micros: list[TrainingMicrobatch] = []
            for micro_index in range(microbatches_per_step):
                start = step_start + micro_index * microbatch_size
                chunk = rows[start : start + microbatch_size]
                try:
                    payload = {
                        field_name: torch.stack([row[1][field_name] for row in chunk])
                        for field_name in fields
                    }
                except RuntimeError as exc:
                    raise ValueError("DATASET_ROW_SHAPE_MISMATCH") from exc
                micros.append(
                    TrainingMicrobatch(
                        f"step-{len(steps):08d}-micro-{micro_index:04d}",
                        payload,
                        tuple(row[0] for row in chunk),
                    )
                )
            steps.append(tuple(micros))
        self._delegate = InMemoryDatasetAdapter(dataset_id, tuple(steps))

    @property
    def dataset_id(self) -> str:
        return self._delegate.dataset_id

    def cursor(self, *, seed: int, rank: int = 0, world_size: int = 1) -> BatchCursor:
        return self._delegate.cursor(seed=seed, rank=rank, world_size=world_size)


__all__ = [
    "BatchCursor",
    "CausalLMEvaluator",
    "ClassificationEvaluator",
    "DatasetAdapter",
    "DeterministicBatchCursor",
    "InMemoryDatasetAdapter",
    "ModelAdapter",
    "OfflineHuggingFaceModelAdapter",
    "OfflineTokenizer",
    "PretokenizedJsonlDatasetAdapter",
    "TaskEvaluator",
    "TorchModelAdapter",
    "TrainingMicrobatch",
]
