r"""严格离线的 Hugging Face ``load_from_disk`` 数据与任务指标适配层。

本模块只认本地目录，唯一 HF 加载入口是 ``datasets.load_from_disk``；
不调用 ``load_dataset``、Hub API 或任何 URL fallback。``datasets`` 也不在模块
导入时解析，只有显式构造 adapter 时才经过
``require_optional_dependency`` 延迟导入。模型侧的 ``from_pretrained`` 仍由
``OfflineHuggingFaceModelAdapter`` 负责，该适配器已固定
``local_files_only=True``。

Pile 与 GLUE 输入必须在资产构建阶段完成 tokenize/pad。适配器会扫描
全部行，拒绝变长、非整数 token、非 0/1 mask、越界 label 和未标注
test split。目录内容 SHA-256、HF fingerprint、规范化行 hash 与文件 stat
manifest 共同组成 resolver 固定状态。
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
import hashlib
import math
from numbers import Integral
from pathlib import Path
import random
import re
from types import MappingProxyType

import torch

from ..contracts.jsonio import canonical_json_bytes
from .fixed_state_torch import _restore_snapshot, _snapshot_drift, _take_snapshot
from .optional import require_optional_dependency
from .training import (
    BatchCursor,
    CausalLMEvaluator,
    ModelAdapter,
    TrainingMicrobatch,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GLUE_LABEL_COUNTS = {"sst2": 2, "mnli": 3, "rte": 2}
_GLUE_SPLITS = {
    "sst2": frozenset({"train", "validation"}),
    "mnli": frozenset({"train", "validation_matched", "validation_mismatched"}),
    "rte": frozenset({"train", "validation"}),
}


def _confusion_metrics(
    confusion: torch.Tensor,
    *,
    binary_positive_f1: bool,
) -> dict[str, float]:
    """由整数 confusion matrix 计算 accuracy/F1/MCC，不引入 sklearn 依赖。

    当分母为零时不返回相应指标。调用方若在配置中请求了该指标，会得到明确的
    ``EVALUATION_METRIC_NOT_PRODUCED``，而不是用 epsilon 或 0 伪造有效数字。
    """

    matrix = confusion.to(dtype=torch.float64, device="cpu")
    total = float(matrix.sum().item())
    if total <= 0:
        raise ValueError("GLUE_CONFUSION_MATRIX_EMPTY")
    result = {"accuracy": float(matrix.diagonal().sum().item()) / total}
    f1_values: list[float] = []
    class_indices = (1,) if binary_positive_f1 else range(matrix.shape[0])
    f1_defined = True
    for index in class_indices:
        true_positive = float(matrix[index, index].item())
        false_positive = float(matrix[:, index].sum().item()) - true_positive
        false_negative = float(matrix[index, :].sum().item()) - true_positive
        denominator = 2.0 * true_positive + false_positive + false_negative
        if denominator <= 0:
            f1_defined = False
            break
        f1_values.append(2.0 * true_positive / denominator)
    if f1_defined and f1_values:
        result["f1"] = sum(f1_values) / len(f1_values)

    true_marginal = matrix.sum(dim=1)
    predicted_marginal = matrix.sum(dim=0)
    correct = matrix.diagonal().sum()
    numerator = correct * total - (true_marginal * predicted_marginal).sum()
    left = total * total - (predicted_marginal * predicted_marginal).sum()
    right = total * total - (true_marginal * true_marginal).sum()
    denominator = torch.sqrt(left * right)
    if float(denominator.item()) > 0:
        result["matthews_correlation"] = float((numerator / denominator).item())
    return result


def _evaluate_glue_batches(
    model: ModelAdapter,
    batches: Sequence[TrainingMicrobatch],
    *,
    num_labels: int,
) -> tuple[float, torch.Tensor]:
    """返回样本加权 loss 与 confusion matrix；仅做只读 forward。"""

    if not hasattr(model, "logits"):
        raise TypeError("GLUE_EVALUATION_REQUIRES_LOGITS_ADAPTER")
    numerator = 0.0
    count = 0
    confusion = torch.zeros((num_labels, num_labels), dtype=torch.int64)
    was_training = model.module.training
    model.module.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                labels = batch.payload["labels"]
                mask = labels.ne(-100)
                extra_mask = batch.payload.get("classification_mask")
                if extra_mask is not None:
                    mask = mask & extra_mask.to(dtype=torch.bool)
                logits = model.logits(batch)  # type: ignore[attr-defined]
                loss = model.loss(batch)
                numerator += float(loss.loss_numerator.detach().cpu().item())
                count += loss.effective_count
                truth = labels[mask].detach().to(dtype=torch.int64, device="cpu")
                prediction = logits.argmax(dim=-1)[mask].detach().to(
                    dtype=torch.int64, device="cpu"
                )
                for true_value, predicted_value in zip(
                    truth.tolist(), prediction.tolist(), strict=True
                ):
                    confusion[true_value, predicted_value] += 1
    finally:
        model.module.train(was_training)
    if count <= 0 or int(confusion.sum().item()) != count:
        raise ValueError("GLUE_EVALUATION_EFFECTIVE_COUNT_INVALID")
    return numerator / count, confusion


def _length_prefixed(digest: "hashlib._Hash", value: bytes) -> None:  # type: ignore[name-defined]
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _validate_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} 必须是小写 SHA-256")
    return value


def _resolve_local_directory(
    path: str | Path,
    *,
    allowed_root: str | Path | None,
) -> Path:
    raw = str(path)
    if "://" in raw or raw.lower().startswith("hf:"):
        raise ValueError("OFFLINE_DATASET_REMOTE_REFERENCE_FORBIDDEN")
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("OFFLINE_DATASET_ROOT_SYMLINK_FORBIDDEN")
    root = candidate.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OFFLINE_DATASET_DIRECTORY_NOT_FOUND:{root}")
    if allowed_root is not None:
        boundary = Path(allowed_root).resolve()
        if not boundary.is_dir():
            raise FileNotFoundError(f"OFFLINE_DATASET_ALLOWED_ROOT_NOT_FOUND:{boundary}")
        try:
            root.relative_to(boundary)
        except ValueError as exc:
            raise ValueError("OFFLINE_DATASET_PATH_OUTSIDE_ALLOWED_ROOT") from exc
    return root


def _directory_entries(root: Path) -> tuple[Path, ...]:
    entries = tuple(sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()))
    symlinks = [entry for entry in entries if entry.is_symlink()]
    if symlinks:
        raise ValueError(
            "OFFLINE_DATASET_SYMLINK_FORBIDDEN:"
            + symlinks[0].relative_to(root).as_posix()
        )
    files = tuple(entry for entry in entries if entry.is_file())
    if not files:
        raise ValueError("OFFLINE_DATASET_DIRECTORY_EMPTY")
    return files


def _directory_stat_manifest(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in _directory_entries(root)
    )


def hash_local_directory(path: str | Path) -> str:
    """流式计算本地目录全部普通文件的内容 SHA-256。

    相对路径、文件长度与内容均使用长度前缀，防止连接歧义。
    symlink 被拒绝，因为它可以在验证后改指到允许根之外。
    """

    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OFFLINE_DATASET_DIRECTORY_NOT_FOUND:{root}")
    digest = hashlib.sha256()
    for file_path in _directory_entries(root):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        _length_prefixed(digest, relative)
        _length_prefixed(digest, str(file_path.stat().st_size).encode("ascii"))
        file_digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                file_digest.update(chunk)
        _length_prefixed(digest, file_digest.digest())
    return digest.hexdigest()


def _normalize_task_name(value: str) -> str:
    normalized = value.lower().replace("-", "").replace("_", "")
    aliases = {"sst2": "sst2", "mnli": "mnli", "rte": "rte", "pile": "pile"}
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"OFFLINE_TASK_UNSUPPORTED:{value}") from exc


def _integer_sequence(
    value: object,
    *,
    field_name: str,
    row_index: int,
) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"DATASET_FIELD_NOT_1D:{row_index}:{field_name}")
        raw_values = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_values = list(value)
    else:
        raise TypeError(f"DATASET_FIELD_NOT_INTEGER_SEQUENCE:{row_index}:{field_name}")
    if not raw_values:
        raise ValueError(f"DATASET_FIELD_SEQUENCE_EMPTY:{row_index}:{field_name}")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in raw_values):
        raise TypeError(f"DATASET_FIELD_NOT_INTEGER_SEQUENCE:{row_index}:{field_name}")
    return tuple(int(item) for item in raw_values)


def _dataset_columns(dataset: object) -> tuple[str, ...]:
    columns = getattr(dataset, "column_names", None)
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise TypeError("OFFLINE_DATASET_COLUMN_NAMES_UNAVAILABLE")
    normalized = tuple(str(value) for value in columns)
    if not normalized or any(not value for value in normalized) or len(set(normalized)) != len(
        normalized
    ):
        raise ValueError("OFFLINE_DATASET_COLUMN_NAMES_INVALID")
    return normalized


def _dataset_length(dataset: object) -> int:
    try:
        length = len(dataset)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("OFFLINE_DATASET_LENGTH_UNAVAILABLE") from exc
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("OFFLINE_DATASET_LENGTH_INVALID")
    return length


class _OfflineDatasetCursor:
    """冻结 HF 行索引的确定性 shuffle/shard/batch cursor。"""

    def __init__(
        self,
        adapter: "_PretokenizedHFDatasetAdapter",
        *,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise TypeError("OFFLINE_DATASET_SEED_NOT_NONNEGATIVE_INTEGER")
        if (
            isinstance(rank, bool)
            or isinstance(world_size, bool)
            or not isinstance(rank, int)
            or not isinstance(world_size, int)
            or world_size <= 0
            or not 0 <= rank < world_size
        ):
            raise ValueError("OFFLINE_DATASET_RANK_WORLD_SIZE_INVALID")
        order = list(range(adapter.row_count))
        random.Random(seed).shuffle(order)
        self._indices = tuple(order[rank::world_size])
        self._adapter = adapter
        self._seed = seed
        self._rank = rank
        self._world_size = world_size
        self._position = 0
        self._step = 0
        self._group_size = adapter.microbatch_size * adapter.microbatches_per_step
        self._resolver_digest = adapter.state_digest()
        if len(self._indices) < self._group_size:
            raise ValueError("OFFLINE_DATASET_SHARD_TOO_SMALL_FOR_ONE_STEP")
        self._order_hash = hashlib.sha256(
            canonical_json_bytes(list(self._indices))
        ).hexdigest()

    def next_microbatches(self) -> tuple[TrainingMicrobatch, ...]:
        self._adapter.assert_unchanged(self._resolver_digest)
        end = self._position + self._group_size
        if end > len(self._indices):
            raise StopIteration("OFFLINE_DATASET_CURSOR_EXHAUSTED")
        step_indices = self._indices[self._position : end]
        microbatches: list[TrainingMicrobatch] = []
        for micro_index in range(self._adapter.microbatches_per_step):
            begin = micro_index * self._adapter.microbatch_size
            indices = step_indices[begin : begin + self._adapter.microbatch_size]
            microbatches.append(
                self._adapter._microbatch_from_indices(
                    indices,
                    batch_id=(
                        f"{self._adapter.dataset_id}:seed-{self._seed}:rank-{self._rank}:"
                        f"step-{self._step:012d}:micro-{micro_index:04d}"
                    ),
                )
            )
        self._position = end
        self._step += 1
        return tuple(microbatches)

    def state_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": "offline-hf-cursor-state-v1",
            "resolver_digest": self._resolver_digest,
            "seed": self._seed,
            "rank": self._rank,
            "world_size": self._world_size,
            "position": self._position,
            "step": self._step,
            "order_hash": self._order_hash,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {
            "schema_version",
            "resolver_digest",
            "seed",
            "rank",
            "world_size",
            "position",
            "step",
            "order_hash",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("OFFLINE_HF_CURSOR_STATE_FIELDS_MISMATCH")
        identities = {
            "schema_version": "offline-hf-cursor-state-v1",
            "resolver_digest": self._resolver_digest,
            "seed": self._seed,
            "rank": self._rank,
            "world_size": self._world_size,
            "order_hash": self._order_hash,
        }
        if any(state[name] != expected for name, expected in identities.items()):
            raise ValueError("OFFLINE_HF_CURSOR_STATE_IDENTITY_MISMATCH")
        position, step = state["position"], state["step"]
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or isinstance(step, bool)
            or not isinstance(step, int)
            or position < 0
            or step < 0
            or position > len(self._indices)
            or position % self._group_size != 0
            or step != position // self._group_size
        ):
            raise ValueError("OFFLINE_HF_CURSOR_STATE_POSITION_INVALID")
        self._position = position
        self._step = step


class _PretokenizedHFDatasetAdapter:
    """Pile/GLUE 的共享离线实现；不作为公共构造入口。"""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_id: str,
        split: str,
        task_name: str,
        microbatch_size: int,
        microbatches_per_step: int,
        expected_asset_hash: str | None,
        allowed_root: str | Path | None,
        sampling_design: str,
        weights_exogenous: bool,
        common_mean_assumption: bool,
    ) -> None:
        if not dataset_id or not split or not sampling_design:
            raise ValueError("OFFLINE_DATASET_ID_SPLIT_OR_SAMPLING_DESIGN_EMPTY")
        if (
            isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, int)
            or isinstance(microbatches_per_step, bool)
            or not isinstance(microbatches_per_step, int)
            or microbatch_size <= 0
            or microbatches_per_step <= 0
        ):
            raise ValueError("OFFLINE_DATASET_BATCH_SIZE_INVALID")
        if type(weights_exogenous) is not bool or type(common_mean_assumption) is not bool:
            raise TypeError("OFFLINE_DATASET_WEIGHT_ASSUMPTION_NOT_BOOLEAN")
        task = _normalize_task_name(task_name)
        root = _resolve_local_directory(path, allowed_root=allowed_root)
        before_stat = _directory_stat_manifest(root)
        asset_hash = hash_local_directory(root)
        if expected_asset_hash is not None and _validate_hash(
            expected_asset_hash, field_name="expected_asset_hash"
        ) != asset_hash:
            raise ValueError("OFFLINE_DATASET_ASSET_HASH_MISMATCH")
        datasets = require_optional_dependency(
            "datasets", feature=f"offline_{task}_load_from_disk"
        )
        loader = getattr(datasets, "load_from_disk", None)
        if not callable(loader):
            raise TypeError("DATASETS_LOAD_FROM_DISK_UNAVAILABLE")
        loaded = loader(str(root))
        if _directory_stat_manifest(root) != before_stat:
            raise RuntimeError("OFFLINE_DATASET_LOADER_MUTATED_ASSET_DIRECTORY")
        if isinstance(loaded, Mapping):
            if split not in loaded:
                raise ValueError(f"OFFLINE_DATASET_SPLIT_NOT_FOUND:{split}")
            dataset = loaded[split]
        else:
            dataset = loaded
        row_count = _dataset_length(dataset)
        columns = _dataset_columns(dataset)

        self._root = root
        self._dataset = dataset
        self._dataset_id = dataset_id
        self._split = split
        self._task_name = task
        self._microbatch_size = microbatch_size
        self._microbatches_per_step = microbatches_per_step
        self._sampling_design = sampling_design
        self._weights_exogenous = weights_exogenous
        self._common_mean_assumption = common_mean_assumption
        self._asset_hash = asset_hash
        self._asset_stat = before_stat
        self._row_count = row_count
        self._columns = columns
        self._fingerprint = str(getattr(dataset, "_fingerprint", "<unavailable>"))
        self._sample_prefix = f"{dataset_id}:{split}:"
        self._sample_ids = tuple(
            f"{self._sample_prefix}{index:012d}" for index in range(row_count)
        )

        if task == "pile":
            required = {"input_ids", "attention_mask", "labels"}
            self._label_column = "labels"
            self._input_fields = ("input_ids", "attention_mask")
            self._loss_unit = "target_token"
            self._statistical_unit = "pretokenized_pile_sequence_draw_group_mean"
            self._weight_unit = "effective_target_tokens"
            self._num_labels = None
        else:
            required = {"input_ids", "attention_mask"}
            labels_present = [name for name in ("labels", "label") if name in columns]
            if len(labels_present) != 1:
                raise ValueError("GLUE_DATASET_REQUIRES_EXACTLY_ONE_LABEL_COLUMN")
            self._label_column = labels_present[0]
            optional = ("token_type_ids",) if "token_type_ids" in columns else ()
            self._input_fields = ("input_ids", "attention_mask", *optional)
            self._loss_unit = "sample"
            self._statistical_unit = f"glue_{task}_example_draw_group_mean"
            self._weight_unit = "effective_samples"
            self._num_labels = _GLUE_LABEL_COUNTS[task]
            if split not in _GLUE_SPLITS[task]:
                raise ValueError(f"GLUE_SPLIT_UNSUPPORTED:{task}:{split}")
        missing = required - set(columns)
        if missing:
            raise ValueError(f"OFFLINE_DATASET_REQUIRED_COLUMNS_MISSING:{sorted(missing)}")

        row_hashes: list[str] = []
        sequence_length: int | None = None
        for index in range(row_count):
            normalized = self._normalize_row(index)
            observed_length = len(normalized["input_ids"])  # type: ignore[arg-type]
            if sequence_length is None:
                sequence_length = observed_length
            elif observed_length != sequence_length:
                raise ValueError(f"OFFLINE_DATASET_VARIABLE_SEQUENCE_LENGTH:{index}")
            row_hashes.append(hashlib.sha256(canonical_json_bytes(normalized)).hexdigest())
        assert sequence_length is not None
        self._sequence_length = sequence_length
        self._row_hashes = tuple(row_hashes)
        self._row_root_hash = hashlib.sha256(
            canonical_json_bytes(row_hashes)
        ).hexdigest()

    def _raw_row(self, index: int) -> Mapping[str, object]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.row_count:
            raise IndexError("OFFLINE_DATASET_ROW_INDEX_OUT_OF_RANGE")
        row = self._dataset[index]  # type: ignore[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"OFFLINE_DATASET_ROW_NOT_MAPPING:{index}")
        return row

    def _normalize_row(self, index: int) -> dict[str, object]:
        row = self._raw_row(index)
        inputs = _integer_sequence(row.get("input_ids"), field_name="input_ids", row_index=index)
        if any(value < 0 for value in inputs):
            raise ValueError(f"DATASET_INPUT_ID_NEGATIVE:{index}")
        attention = _integer_sequence(
            row.get("attention_mask"), field_name="attention_mask", row_index=index
        )
        if len(attention) != len(inputs) or any(value not in {0, 1} for value in attention):
            raise ValueError(f"DATASET_ATTENTION_MASK_INVALID:{index}")
        if not any(attention):
            raise ValueError(f"DATASET_ATTENTION_MASK_ALL_ZERO:{index}")
        normalized: dict[str, object] = {
            "input_ids": list(inputs),
            "attention_mask": list(attention),
        }
        if "token_type_ids" in self._input_fields:
            token_types = _integer_sequence(
                row.get("token_type_ids"),
                field_name="token_type_ids",
                row_index=index,
            )
            if len(token_types) != len(inputs) or any(value < 0 for value in token_types):
                raise ValueError(f"DATASET_TOKEN_TYPE_IDS_INVALID:{index}")
            normalized["token_type_ids"] = list(token_types)
        raw_label = row.get(self._label_column)
        if self.task_name == "pile":
            labels = _integer_sequence(raw_label, field_name="labels", row_index=index)
            if len(labels) != len(inputs) or any(
                value != -100 and value < 0 for value in labels
            ):
                raise ValueError(f"PILE_LABELS_INVALID:{index}")
            if len(labels) < 2 or not any(
                label != -100 and attention[position] == 1
                for position, label in enumerate(labels[1:], start=1)
            ):
                raise ValueError(f"PILE_EFFECTIVE_TARGET_COUNT_ZERO:{index}")
            normalized["labels"] = list(labels)
        else:
            if isinstance(raw_label, bool) or not isinstance(raw_label, Integral):
                raise TypeError(f"GLUE_LABEL_NOT_INTEGER:{index}")
            label = int(raw_label)
            assert self._num_labels is not None
            if not 0 <= label < self._num_labels:
                raise ValueError(f"GLUE_LABEL_OUT_OF_RANGE:{index}")
            normalized["labels"] = label
        return normalized

    def _checked_normalized_row(self, index: int) -> dict[str, object]:
        row = self._normalize_row(index)
        observed = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        if observed != self._row_hashes[index]:
            raise RuntimeError(f"OFFLINE_DATASET_ROW_MUTATED:{index}")
        return row

    def _payload_from_index(self, index: int) -> dict[str, torch.Tensor]:
        row = self._checked_normalized_row(index)
        payload: dict[str, torch.Tensor] = {}
        for name in self._input_fields:
            payload[name] = torch.tensor(row[name], dtype=torch.int64)
        payload["labels"] = torch.tensor(row["labels"], dtype=torch.int64)
        return payload

    def _microbatch_from_indices(
        self,
        indices: Sequence[int],
        *,
        batch_id: str,
    ) -> TrainingMicrobatch:
        if not indices:
            raise ValueError("OFFLINE_DATASET_MICROBATCH_INDICES_EMPTY")
        rows = [self._payload_from_index(index) for index in indices]
        fields = tuple(rows[0])
        if any(tuple(row) != fields for row in rows):
            raise RuntimeError("OFFLINE_DATASET_PAYLOAD_FIELDS_DRIFTED")
        payload = {
            name: torch.stack([row[name] for row in rows]) for name in fields
        }
        return TrainingMicrobatch(
            batch_id,
            payload,
            tuple(self._sample_ids[index] for index in indices),
            {
                "asset_hash": self.asset_hash,
                "dataset_id": self.dataset_id,
                "dataset_split": self.split,
                "task_name": self.task_name,
                "row_indices": list(indices),
            },
        )

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def resolver_id(self) -> str:
        return f"offline-hf:{self.dataset_id}:{self.split}:{self.asset_hash[:16]}"

    @property
    def split(self) -> str:
        return self._split

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def asset_hash(self) -> str:
        return self._asset_hash

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def microbatch_size(self) -> int:
        return self._microbatch_size

    @property
    def microbatches_per_step(self) -> int:
        return self._microbatches_per_step

    @property
    def sample_ids(self) -> tuple[Hashable, ...]:
        return self._sample_ids

    @property
    def loss_unit(self) -> str:
        return self._loss_unit

    @property
    def statistical_unit(self) -> str:
        return self._statistical_unit

    @property
    def weight_unit(self) -> str:
        return self._weight_unit

    @property
    def sampling_design(self) -> str:
        return self._sampling_design

    @property
    def weights_exogenous(self) -> bool:
        return self._weights_exogenous

    @property
    def common_mean_assumption(self) -> bool:
        return self._common_mean_assumption

    def resolve(self, sample_id: Hashable) -> TrainingMicrobatch:
        if not isinstance(sample_id, str) or not sample_id.startswith(self._sample_prefix):
            raise KeyError(f"OFFLINE_DATASET_SAMPLE_ID_UNKNOWN:{sample_id!r}")
        suffix = sample_id[len(self._sample_prefix) :]
        if len(suffix) != 12 or not suffix.isdigit():
            raise KeyError(f"OFFLINE_DATASET_SAMPLE_ID_UNKNOWN:{sample_id!r}")
        index = int(suffix)
        if index >= self.row_count or self._sample_ids[index] != sample_id:
            raise KeyError(f"OFFLINE_DATASET_SAMPLE_ID_UNKNOWN:{sample_id!r}")
        return self._microbatch_from_indices(
            (index,), batch_id=f"resolved:{sample_id}"
        )

    def cursor(self, *, seed: int, rank: int = 0, world_size: int = 1) -> BatchCursor:
        return _OfflineDatasetCursor(
            self, seed=seed, rank=rank, world_size=world_size
        )

    def state_digest(self) -> str:
        payload = {
            "schema_version": "offline-hf-resolver-state-v1",
            "resolver_id": self.resolver_id,
            "dataset_id": self.dataset_id,
            "task_name": self.task_name,
            "split": self.split,
            "asset_hash": self.asset_hash,
            "dataset_fingerprint": self._fingerprint,
            "columns": list(self._columns),
            "row_count": self.row_count,
            "sequence_length": self.sequence_length,
            "row_root_hash": self._row_root_hash,
            "microbatch_size": self.microbatch_size,
            "microbatches_per_step": self.microbatches_per_step,
            "loss_unit": self.loss_unit,
            "statistical_unit": self.statistical_unit,
            "weight_unit": self.weight_unit,
            "sampling_design": self.sampling_design,
            "weights_exogenous": self.weights_exogenous,
            "common_mean_assumption": self.common_mean_assumption,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def assert_unchanged(self, expected_digest: str) -> None:
        _validate_hash(expected_digest, field_name="expected resolver digest")
        if _directory_stat_manifest(self._root) != self._asset_stat:
            raise RuntimeError("OFFLINE_DATASET_DIRECTORY_STAT_CHANGED")
        if _dataset_length(self._dataset) != self.row_count:
            raise RuntimeError("OFFLINE_DATASET_ROW_COUNT_CHANGED")
        if _dataset_columns(self._dataset) != self._columns:
            raise RuntimeError("OFFLINE_DATASET_COLUMNS_CHANGED")
        if str(getattr(self._dataset, "_fingerprint", "<unavailable>")) != self._fingerprint:
            raise RuntimeError("OFFLINE_DATASET_FINGERPRINT_CHANGED")
        if self.state_digest() != expected_digest:
            raise RuntimeError("OFFLINE_DATASET_RESOLVER_STATE_CHANGED")

    def verify_asset_hash(self) -> None:
        """显式重扫全目录，用于 formal Gate 或长任务边界。"""

        if hash_local_directory(self._root) != self.asset_hash:
            raise RuntimeError("OFFLINE_DATASET_ASSET_CONTENT_CHANGED")


class PretokenizedPileDatasetAdapter(_PretokenizedHFDatasetAdapter):
    """预分词、定长 Pile causal-LM 数据适配器。"""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_id: str,
        split: str = "train",
        microbatch_size: int,
        microbatches_per_step: int,
        sampling_design: str,
        weights_exogenous: bool,
        common_mean_assumption: bool,
        expected_asset_hash: str | None = None,
        allowed_root: str | Path | None = None,
    ) -> None:
        super().__init__(
            path,
            dataset_id=dataset_id,
            split=split,
            task_name="pile",
            microbatch_size=microbatch_size,
            microbatches_per_step=microbatches_per_step,
            expected_asset_hash=expected_asset_hash,
            allowed_root=allowed_root,
            sampling_design=sampling_design,
            weights_exogenous=weights_exogenous,
            common_mean_assumption=common_mean_assumption,
        )


class PretokenizedGlueDatasetAdapter(_PretokenizedHFDatasetAdapter):
    """GLUE SST-2/MNLI/RTE 预分词分类数据适配器。"""

    def __init__(
        self,
        path: str | Path,
        *,
        task_name: str,
        split: str,
        dataset_id: str,
        microbatch_size: int,
        microbatches_per_step: int,
        sampling_design: str = "uniform_with_replacement_over_frozen_glue_rows",
        expected_asset_hash: str | None = None,
        allowed_root: str | Path | None = None,
    ) -> None:
        normalized = _normalize_task_name(task_name)
        if normalized not in _GLUE_LABEL_COUNTS:
            raise ValueError("GLUE_TASK_UNSUPPORTED")
        super().__init__(
            path,
            dataset_id=dataset_id,
            split=split,
            task_name=normalized,
            microbatch_size=microbatch_size,
            microbatches_per_step=microbatches_per_step,
            expected_asset_hash=expected_asset_hash,
            allowed_root=allowed_root,
            sampling_design=sampling_design,
            # GLUE 每个已标注 example 的 effective count 恒为 1。
            weights_exogenous=True,
            common_mean_assumption=True,
        )


class HuggingFaceTaskMetricEvaluator:
    """Pile/SST-2/MNLI/RTE 的只读 loss 与主任务指标适配器。"""

    def __init__(self, task_name: str, *, split: str | None = None) -> None:
        self.task_name = _normalize_task_name(task_name)
        self.split = split
        if self.task_name == "mnli" and split is not None and split not in {
            "validation_matched",
            "validation_mismatched",
            "train",
        }:
            raise ValueError("MNLI_METRIC_SPLIT_UNSUPPORTED")

    def evaluate(
        self,
        model: ModelAdapter,
        microbatches: Sequence[TrainingMicrobatch],
    ) -> Mapping[str, float]:
        if not isinstance(model, ModelAdapter):
            raise TypeError("TASK_METRIC_MODEL_ADAPTER_REQUIRED")
        batches = tuple(microbatches)
        if not batches:
            raise ValueError("TASK_METRIC_MICROBATCHES_EMPTY")
        expected_task_type = (
            "causal_lm" if self.task_name == "pile" else "sequence_classification"
        )
        if model.task_type != expected_task_type:
            raise ValueError("TASK_METRIC_MODEL_TASK_TYPE_MISMATCH")
        snapshot = _take_snapshot(model.module)
        metrics: Mapping[str, float] | None = None
        error: BaseException | None = None
        try:
            if self.task_name == "pile":
                base = CausalLMEvaluator().evaluate(model, batches)
                metrics = {
                    "loss": float(base["loss"]),
                    "perplexity": float(base["perplexity"]),
                    "pile_loss": float(base["loss"]),
                    "pile_perplexity": float(base["perplexity"]),
                }
            else:
                mean_loss, confusion = _evaluate_glue_batches(
                    model,
                    batches,
                    num_labels=_GLUE_LABEL_COUNTS[self.task_name],
                )
                base = _confusion_metrics(
                    confusion,
                    binary_positive_f1=self.task_name in {"sst2", "rte"},
                )
                metric_name = f"{self.task_name}_accuracy"
                if self.task_name == "mnli" and self.split in {
                    "validation_matched",
                    "validation_mismatched",
                }:
                    suffix = self.split.removeprefix("validation_")
                    metric_name = f"mnli_{suffix}_accuracy"
                metrics = {
                    "loss": mean_loss,
                    "accuracy": float(base["accuracy"]),
                    metric_name: float(base["accuracy"]),
                    **{
                        name: float(value)
                        for name, value in base.items()
                        if name not in {"accuracy"}
                    },
                }
        except BaseException as exc:
            error = exc
        finally:
            drift = _snapshot_drift(model.module, snapshot)
            _restore_snapshot(model.module, snapshot)
        if drift:
            raise RuntimeError("TASK_METRIC_MODEL_STATE_MUTATED:" + ";".join(drift)) from error
        if error is not None:
            raise error
        assert metrics is not None
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("TASK_METRIC_NONFINITE_RESULT")
        return MappingProxyType(dict(metrics))


__all__ = [
    "HuggingFaceTaskMetricEvaluator",
    "PretokenizedGlueDatasetAdapter",
    "PretokenizedPileDatasetAdapter",
    "hash_local_directory",
]
