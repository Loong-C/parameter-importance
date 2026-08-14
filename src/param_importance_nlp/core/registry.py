"""稳定参数坐标注册表。

本模块把模型中的 ``Parameter`` 对象转换成可持久化的坐标合同。注册表明确
分离三类身份：

``coordinate_registry_hash``
    只描述“研究的是哪些坐标”，包括规范名称、别名、形状、顺序、静态参数组
    ID 与分析标签；设备、dtype 和动态学习率不会污染该哈希。
``optimizer_contract_hash``
    描述 optimizer 类型、参数到组的映射及静态超参数。学习率是逐 step 读取的
    动态量，因此排除 ``lr`` 与 ``initial_lr``。
``runtime_layout_hash``
    描述当前 device、dtype、layout 和 stride，用于发现运行布局漂移。

共享同一个 ``Parameter`` 对象的多个逻辑名称会合并成一个坐标，首个模型遍历
名称作为 canonical name，其余名称作为 alias。不同对象只要底层 storage 区间
重叠就拒绝，因为这会让逐坐标累计发生难以察觉的重复计数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from .errors import RegistryError


REGISTRY_SCHEMA_VERSION = "parameter-registry-v1"
_SHA256_LENGTH = 64


def _canonical_json(value: object) -> str:
    """返回用于身份哈希的严格、稳定 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_scalar(value: Any) -> Any:
    """把 optimizer 超参数规范化为可哈希的原始 JSON 值。

    未知对象会被拒绝，避免把 ``repr``（可能含内存地址）偷偷写进合同。
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegistryError("optimizer 静态超参数必须为有限数")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in sorted(value.items())}
    raise RegistryError(f"不支持写入 optimizer 合同的超参数类型: {type(value).__name__}")


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(f"{field} 必须是非负有限数")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RegistryError(f"{field} 必须是非负有限数")
    return result


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RegistryError(f"{field} 必须是小写 SHA-256")
    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RegistryError(f"{field} 必须是非空规范字符串")
    return value


def _infer_module_type(name: str) -> str:
    """从 canonical name 生成保守的分析标签。

    标签只用于分组呈现，不参与参数坐标身份的推断。未知命名约定统一落到
    ``other``，避免把一个启发式猜测当成 eligibility 规则。
    """

    lowered = name.casefold()
    if any(token in lowered for token in ("embedding", "embed_tokens", "word_embeddings")):
        return "embedding"
    if any(token in lowered for token in ("attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(token in lowered for token in ("mlp", "ffn", "feed_forward", "fc1", "fc2", "gate_proj", "up_proj", "down_proj")):
        return "mlp"
    if any(token in lowered for token in ("layernorm", "layer_norm", "norm", ".ln", "ln_")):
        return "norm"
    if any(token in lowered for token in ("lm_head", "classifier", "classification_head", ".head", "head.")):
        return "head"
    return "other"


def _infer_parameter_role(name: str) -> str:
    lowered = name.casefold()
    if lowered.endswith(".bias") or lowered == "bias":
        return "bias"
    if lowered.endswith(".weight") or lowered == "weight":
        return "weight"
    return "other"


def _storage_span(tensor: torch.Tensor) -> tuple[str, int, int, int] | None:
    """计算张量可能触及的 storage 字节闭区间。

    PyTorch 参数通常连续，但注册表不能依赖这一偶然事实。这里根据 shape、stride
    和 storage_offset 计算最小/最大元素偏移，因而也能检查切片 view。空张量没有
    可重叠字节，返回 ``None``。
    """

    if tensor.numel() == 0:
        return None
    if tensor.layout is not torch.strided:
        raise RegistryError(f"仅支持稠密 strided 参数，实际 layout={tensor.layout}")
    storage = tensor.untyped_storage()
    base = int(storage.data_ptr())
    minimum = int(tensor.storage_offset())
    maximum = minimum
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent = (int(size) - 1) * int(stride)
        if extent >= 0:
            maximum += extent
        else:
            minimum += extent
    item_size = int(tensor.element_size())
    return (str(tensor.device), base, base + minimum * item_size, base + (maximum + 1) * item_size)


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    """单个规范参数张量的冻结坐标说明。"""

    canonical_name: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    order: int
    eligible: bool
    eligibility_reason: str
    group_id: str | None
    tags: Mapping[str, str] = field(default_factory=dict)
    dtype: str = ""
    device: str = ""
    layout: str = ""
    stride: tuple[int, ...] = ()
    # Learning rate is a step-time value and deliberately does not participate
    # in coordinate_registry_hash or optimizer_contract_hash.  It is nevertheless
    # recorded in the manifest so a consumer can audit the actual group mapping.
    learning_rate: float | None = None
    weight_decay: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", MappingProxyType(dict(sorted(self.tags.items()))))
        if not self.canonical_name or self.canonical_name != self.canonical_name.strip():
            raise RegistryError("canonical_name 必须是非空规范字符串")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise RegistryError(f"参数 {self.canonical_name!r} 的 shape 非法")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise RegistryError(f"参数 {self.canonical_name!r} 的 order 非法")
        if self.learning_rate is not None:
            _finite_nonnegative(self.learning_rate, field=f"{self.canonical_name}.learning_rate")
        if self.weight_decay is not None:
            _finite_nonnegative(self.weight_decay, field=f"{self.canonical_name}.weight_decay")

    @property
    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= size
        return result

    def to_manifest(self) -> dict[str, Any]:
        """返回不含运行时 tensor 引用的稳定 wire record。"""

        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "shape": list(self.shape),
            "numel": self.numel,
            "order": self.order,
            "eligible": self.eligible,
            "eligibility_reason": self.eligibility_reason,
            "group_id": self.group_id,
            "tags": dict(self.tags),
            "dtype": self.dtype,
            "device": self.device,
            "layout": self.layout,
            "stride": list(self.stride),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


class ParameterRegistry(Sequence[ParameterRecord]):
    """模型参数的稳定、有序注册表。

    推荐通过 :meth:`from_model` 构造。默认只允许 optimizer 中出现模型已命名的
    参数，并拒绝跨参数组重复、重叠 storage、稀疏参数或当前稀疏梯度。冻结参数与
    未进入 optimizer 的参数仍会出现在记录中，但 ``eligible=False``；纯估计器只
    消费 :attr:`eligible_records`。
    """

    def __init__(
        self,
        records: Iterable[ParameterRecord],
        *,
        optimizer_type: str,
        optimizer_groups: Sequence[Mapping[str, Any]],
        parameters: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        frozen_records = tuple(records)
        names = [record.canonical_name for record in frozen_records]
        if len(names) != len(set(names)):
            raise RegistryError("canonical parameter name 必须唯一")
        if [record.order for record in frozen_records] != list(range(len(frozen_records))):
            raise RegistryError("ParameterRecord.order 必须从 0 连续递增")
        self._records = frozen_records
        self._by_name = {record.canonical_name: record for record in frozen_records}
        alias_map: dict[str, str] = {}
        for record in frozen_records:
            for name in (record.canonical_name, *record.aliases):
                previous = alias_map.setdefault(name, record.canonical_name)
                if previous != record.canonical_name:
                    raise RegistryError(f"参数名称 {name!r} 同时指向多个规范坐标")
        self._alias_to_canonical = MappingProxyType(alias_map)
        self.optimizer_type = optimizer_type
        self.optimizer_groups = tuple(MappingProxyType(dict(group)) for group in optimizer_groups)
        self._parameters = MappingProxyType(dict(parameters or {}))

        coordinate_payload = [
            {
                "canonical_name": r.canonical_name,
                "aliases": list(r.aliases),
                "shape": list(r.shape),
                "order": r.order,
                "eligible": r.eligible,
                "eligibility_reason": r.eligibility_reason,
                "group_id": r.group_id,
                "tags": dict(r.tags),
            }
            for r in frozen_records
        ]
        optimizer_payload = {
            "optimizer_type": optimizer_type,
            # Learning rate is recorded per ParameterRecord for step-time
            # auditing.  It is intentionally excluded from this contract hash;
            # static options and parameter-to-group identity remain here.
            "groups": [
                {
                    "group_id": str(group.get("group_id")),
                    "static_options": dict(group.get("static_options", {})),
                }
                for group in self.optimizer_groups
            ],
            "parameter_groups": {
                r.canonical_name: r.group_id for r in frozen_records if r.group_id is not None
            },
        }
        runtime_payload = [
            {
                "canonical_name": r.canonical_name,
                "dtype": r.dtype,
                "device": r.device,
                "layout": r.layout,
                "stride": list(r.stride),
            }
            for r in frozen_records
        ]
        self.coordinate_registry_hash = _sha256_json(coordinate_payload)
        self.optimizer_contract_hash = _sha256_json(optimizer_payload)
        self.runtime_layout_hash = _sha256_json(runtime_payload)

    @property
    def registry_hash(self) -> str:
        """兼容下游 provider 的通用 registry hash 名称。"""

        return self.coordinate_registry_hash

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        tags: Mapping[str, Mapping[str, str]] | None = None,
    ) -> "ParameterRegistry":
        """从模型与 optimizer 建立注册表。

        ``remove_duplicate=False`` 是别名识别的关键：若两个模块属性引用同一
        Parameter，PyTorch 默认遍历会隐藏后一个名称。canonical name 采用别名
        集合中的字典序最小名称，所有独立坐标再按 canonical name 排序，因此
        参数注册顺序变化不会改变坐标身份。
        """

        try:
            named = list(model.named_parameters(remove_duplicate=False))
        except TypeError:  # pragma: no cover - 兼容非常旧的 torch
            named = list(model.named_parameters())
        if not named:
            raise RegistryError("模型没有任何命名参数")

        names_by_object: dict[int, list[str]] = {}
        object_by_id: dict[int, torch.nn.Parameter] = {}
        for name, parameter in named:
            if not name:
                raise RegistryError("参数名称不能为空")
            identity = id(parameter)
            if identity not in names_by_object:
                names_by_object[identity] = []
                object_by_id[identity] = parameter
            names_by_object[identity].append(name)

        canonical_by_object = {
            identity: min(names)
            for identity, names in names_by_object.items()
        }
        for names in names_by_object.values():
            names.sort()
        object_order = sorted(
            names_by_object,
            key=lambda identity: canonical_by_object[identity],
        )

        # PyTorch 的稀疏 Embedding 参数本身仍是 dense strided storage，只有 backward
        # 才产生 sparse COO gradient。若只检查 ``parameter.layout``，在第一次
        # backward 之前会错误放行，因此同时审计已知模块的 ``sparse`` 合同。
        sparse_gradient_parameter_ids: set[int] = set()
        for module in model.modules():
            if bool(getattr(module, "sparse", False)):
                weight = getattr(module, "weight", None)
                if isinstance(weight, torch.nn.Parameter):
                    sparse_gradient_parameter_ids.add(id(weight))

        group_by_object: dict[int, int] = {}
        optimizer_groups: list[dict[str, Any]] = []
        model_object_ids = set(names_by_object)
        for group_index, group in enumerate(optimizer.param_groups):
            group_id = f"group_{group_index:04d}"
            learning_rate = _finite_nonnegative(
                group.get("lr"), field=f"optimizer.param_groups[{group_index}].lr"
            )
            weight_decay = _finite_nonnegative(
                group.get("weight_decay", 0.0),
                field=f"optimizer.param_groups[{group_index}].weight_decay",
            )
            parameter_names: list[str] = []
            for parameter in group.get("params", []):
                identity = id(parameter)
                if identity not in model_object_ids:
                    raise RegistryError("optimizer 包含不属于模型命名参数的对象")
                if identity in group_by_object:
                    previous = group_by_object[identity]
                    if previous != group_index:
                        raise RegistryError("同一 Parameter 不能跨 optimizer 参数组重复")
                    raise RegistryError("同一 Parameter 在 optimizer 参数组内重复")
                group_by_object[identity] = group_index
                parameter_names.append(canonical_by_object[identity])
            static_options = {
                str(key): _json_scalar(value)
                for key, value in sorted(group.items())
                if key not in {"params", "lr", "initial_lr"}
            }
            optimizer_groups.append(
                {
                    "group_id": group_id,
                    "parameter_names": sorted(parameter_names),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "static_options": static_options,
                }
            )

        # 不同 Parameter 对象的 storage 区间不得重叠；同对象别名已在前面合并。
        spans: list[tuple[int, tuple[str, int, int, int], str]] = []
        for identity in object_order:
            parameter = object_by_id[identity]
            span = _storage_span(parameter)
            if span is None:
                continue
            canonical_name = canonical_by_object[identity]
            for other_identity, other_span, other_name in spans:
                same_storage = span[0] == other_span[0] and span[1] == other_span[1]
                overlaps = max(span[2], other_span[2]) < min(span[3], other_span[3])
                if same_storage and overlaps and other_identity != identity:
                    raise RegistryError(
                        f"不同 Parameter 的 storage 区间重叠: {other_name!r} 与 {canonical_name!r}"
                    )
            spans.append((identity, span, canonical_name))

        tag_map = tags or {}
        records: list[ParameterRecord] = []
        parameters: dict[str, torch.Tensor] = {}
        for order, identity in enumerate(object_order):
            parameter = object_by_id[identity]
            all_names = names_by_object[identity]
            canonical = canonical_by_object[identity]
            group_index = group_by_object.get(identity)
            if parameter.layout is not torch.strided:
                raise RegistryError(f"参数 {canonical!r} 不是稠密 strided 张量")
            if identity in sparse_gradient_parameter_ids:
                raise RegistryError(f"参数 {canonical!r} 的模块合同会生成 sparse gradient")
            if parameter.grad is not None and parameter.grad.layout is not torch.strided:
                raise RegistryError(f"参数 {canonical!r} 当前具有 sparse gradient")
            if not parameter.requires_grad:
                eligible = False
                reason = "requires_grad_false"
            elif group_index is None:
                eligible = False
                reason = "not_in_optimizer"
            else:
                eligible = True
                reason = "eligible_dense_unique_storage"
            inferred_tags = {
                "module": canonical.rsplit(".", 1)[0] if "." in canonical else "<root>",
                "layer": canonical.split(".", 1)[0],
                "module_type": _infer_module_type(canonical),
                "parameter_role": _infer_parameter_role(canonical),
            }
            inferred_tags.update({str(k): str(v) for k, v in tag_map.get(canonical, {}).items()})
            records.append(
                ParameterRecord(
                    canonical_name=canonical,
                    aliases=tuple(all_names[1:]),
                    shape=tuple(int(size) for size in parameter.shape),
                    order=order,
                    eligible=eligible,
                    eligibility_reason=reason,
                    group_id=None if group_index is None else f"group_{group_index:04d}",
                    tags=inferred_tags,
                    dtype=str(parameter.dtype),
                    device=str(parameter.device),
                    layout=str(parameter.layout),
                    stride=tuple(int(value) for value in parameter.stride()),
                    learning_rate=(
                        None
                        if group_index is None
                        else float(optimizer_groups[group_index]["learning_rate"])
                    ),
                    weight_decay=(
                        None
                        if group_index is None
                        else float(optimizer_groups[group_index]["weight_decay"])
                    ),
                )
            )
            parameters[canonical] = parameter

        optimizer_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        return cls(
            records,
            optimizer_type=optimizer_type,
            optimizer_groups=optimizer_groups,
            parameters=parameters,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int | slice) -> ParameterRecord | tuple[ParameterRecord, ...]:
        return self._records[index]

    def __iter__(self) -> Iterator[ParameterRecord]:
        return iter(self._records)

    @property
    def eligible_records(self) -> tuple[ParameterRecord, ...]:
        return tuple(record for record in self._records if record.eligible)

    @property
    def eligible_names(self) -> tuple[str, ...]:
        return tuple(record.canonical_name for record in self.eligible_records)

    def canonical_name(self, name: str) -> str:
        """把 alias 解析为 canonical name；未知名称立即失败。"""

        try:
            return self._alias_to_canonical[name]
        except KeyError as exc:
            raise RegistryError(f"未知参数名称或 alias: {name!r}") from exc

    def record(self, name: str) -> ParameterRecord:
        return self._by_name[self.canonical_name(name)]

    def parameter(self, name: str) -> torch.Tensor:
        canonical = self.canonical_name(name)
        try:
            return self._parameters[canonical]
        except KeyError as exc:
            raise RegistryError("该注册表没有绑定运行时 Parameter 对象") from exc

    def validate_gradient(self, name: str, gradient: torch.Tensor) -> None:
        """验证单个梯度的坐标、shape、layout 与有限性。"""

        record = self.record(name)
        if not record.eligible:
            raise RegistryError(f"参数 {record.canonical_name!r} 不属于 eligible set")
        if tuple(gradient.shape) != record.shape:
            raise RegistryError(
                f"梯度 shape 不匹配: {record.canonical_name} 期望 {record.shape}，实际 {tuple(gradient.shape)}"
            )
        if gradient.layout is not torch.strided:
            raise RegistryError(f"参数 {record.canonical_name!r} 的 sparse gradient 不受支持")

    def validate_model_gradients(
        self,
        *,
        expected_missing: Iterable[str] = (),
    ) -> dict[str, tuple[str, ...]]:
        """分类当前模型的 ``grad is None``，并对异常缺失 fail-closed。

        ``expected_missing`` 只允许由上层根据冻结/路由合同显式声明；registry 不
        会把未知缺失梯度静默填零。返回值可直接写入调试 manifest。
        """

        expected = {self.canonical_name(name) for name in expected_missing}
        unknown_expected = expected - set(self.eligible_names)
        if unknown_expected:
            raise RegistryError(f"expected_missing 包含未知或非 eligible 参数: {sorted(unknown_expected)}")
        present: list[str] = []
        missing: list[str] = []
        for record in self.eligible_records:
            gradient = self.parameter(record.canonical_name).grad  # type: ignore[union-attr]
            if gradient is None:
                missing.append(record.canonical_name)
                if record.canonical_name not in expected:
                    raise RegistryError(f"参数 {record.canonical_name!r} 异常缺失梯度")
                continue
            self.validate_gradient(record.canonical_name, gradient)
            present.append(record.canonical_name)
        return {
            "present": tuple(present),
            "expected_missing": tuple(sorted(set(missing).intersection(expected))),
        }

    def to_manifest(self) -> dict[str, Any]:
        """生成可保存、可审计且绑定三类 hash 的 registry manifest。"""

        groups: list[dict[str, Any]] = []
        for group in self.optimizer_groups:
            groups.append(
                {
                    "group_id": str(group.get("group_id")),
                    "parameter_names": list(group.get("parameter_names", [])),
                    "learning_rate": group.get("learning_rate"),
                    "weight_decay": group.get("weight_decay", 0.0),
                    "static_options": dict(group.get("static_options", {})),
                }
            )
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "coordinate_registry_hash": self.coordinate_registry_hash,
            "optimizer_contract_hash": self.optimizer_contract_hash,
            "runtime_layout_hash": self.runtime_layout_hash,
            "optimizer_type": self.optimizer_type,
            "records": [record.to_manifest() for record in self._records],
            "optimizer_groups": groups,
        }

    @classmethod
    def from_manifest(
        cls,
        value: Mapping[str, Any],
        *,
        parameters: Mapping[str, torch.Tensor] | None = None,
    ) -> "ParameterRegistry":
        """严格读取 registry manifest，并核对声明的 hash 不可被手工改写。"""

        required = {
            "schema_version",
            "coordinate_registry_hash",
            "optimizer_contract_hash",
            "runtime_layout_hash",
            "optimizer_type",
            "records",
            "optimizer_groups",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise RegistryError("parameter registry manifest 字段集合无效")
        if value["schema_version"] != REGISTRY_SCHEMA_VERSION:
            raise RegistryError("parameter registry manifest schema_version 不受支持")
        records_value = value["records"]
        if not isinstance(records_value, list) or not records_value:
            raise RegistryError("parameter registry manifest.records 必须为非空数组")
        records: list[ParameterRecord] = []
        record_fields = {
            "canonical_name", "aliases", "shape", "numel", "order", "eligible",
            "eligibility_reason", "group_id", "tags", "dtype", "device", "layout",
            "stride", "learning_rate", "weight_decay",
        }
        for index, raw in enumerate(records_value):
            if not isinstance(raw, Mapping) or set(raw) != record_fields:
                raise RegistryError(f"registry.records[{index}] 字段集合无效")
            canonical = _require_string(raw["canonical_name"], field=f"records[{index}].canonical_name")
            aliases = raw["aliases"]
            if not isinstance(aliases, list) or any(
                not isinstance(alias, str) or not alias or alias != alias.strip()
                for alias in aliases
            ):
                raise RegistryError(f"records[{index}].aliases 必须是规范字符串数组")
            shape = raw["shape"]
            if not isinstance(shape, list) or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            ):
                raise RegistryError(f"records[{index}].shape 非法")
            expected_numel = math.prod(shape) if shape else 1
            if raw["numel"] != expected_numel:
                raise RegistryError(f"records[{index}].numel 与 shape 不一致")
            stride = raw["stride"]
            if not isinstance(stride, list) or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in stride
            ):
                raise RegistryError(f"records[{index}].stride 非法")
            if len(stride) != len(shape):
                raise RegistryError(f"records[{index}].stride 与 shape 维数不一致")
            if not isinstance(raw["eligible"], bool):
                raise RegistryError(f"records[{index}].eligible 必须是 bool")
            group_id = raw["group_id"]
            if group_id is not None:
                group_id = _require_string(group_id, field=f"records[{index}].group_id")
            tags = raw["tags"]
            if not isinstance(tags, Mapping) or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in tags.items()
            ):
                raise RegistryError(f"records[{index}].tags 必须是 string -> string object")
            dtype = raw["dtype"]
            device = raw["device"]
            layout = raw["layout"]
            for field_name, field_value in (("dtype", dtype), ("device", device), ("layout", layout)):
                if not isinstance(field_value, str):
                    raise RegistryError(f"records[{index}].{field_name} 必须是字符串")
            records.append(
                ParameterRecord(
                    canonical_name=canonical,
                    aliases=tuple(aliases),
                    shape=tuple(shape),
                    order=raw["order"],
                    eligible=raw["eligible"],
                    eligibility_reason=_require_string(
                        raw["eligibility_reason"], field=f"records[{index}].eligibility_reason"
                    ),
                    group_id=group_id,
                    tags=dict(tags),
                    dtype=dtype,
                    device=device,
                    layout=layout,
                    stride=tuple(stride),
                    learning_rate=(
                        None
                        if raw["learning_rate"] is None
                        else _finite_nonnegative(raw["learning_rate"], field=f"records[{index}].learning_rate")
                    ),
                    weight_decay=(
                        None
                        if raw["weight_decay"] is None
                        else _finite_nonnegative(raw["weight_decay"], field=f"records[{index}].weight_decay")
                    ),
                )
            )
        groups_value = value["optimizer_groups"]
        if not isinstance(groups_value, list):
            raise RegistryError("parameter registry manifest.optimizer_groups 必须为数组")
        groups: list[dict[str, Any]] = []
        group_fields = {"group_id", "parameter_names", "learning_rate", "weight_decay", "static_options"}
        for index, raw in enumerate(groups_value):
            if not isinstance(raw, Mapping) or set(raw) != group_fields:
                raise RegistryError(f"optimizer_groups[{index}] 字段集合无效")
            group_id = _require_string(raw["group_id"], field=f"optimizer_groups[{index}].group_id")
            names = raw["parameter_names"]
            if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
                raise RegistryError(f"optimizer_groups[{index}].parameter_names 非法")
            static_options = raw["static_options"]
            if not isinstance(static_options, Mapping):
                raise RegistryError(f"optimizer_groups[{index}].static_options 必须是 object")
            groups.append(
                {
                    "group_id": group_id,
                    "parameter_names": list(names),
                    "learning_rate": _finite_nonnegative(
                        raw["learning_rate"], field=f"optimizer_groups[{index}].learning_rate"
                    ),
                    "weight_decay": _finite_nonnegative(
                        raw["weight_decay"], field=f"optimizer_groups[{index}].weight_decay"
                    ),
                    "static_options": _json_scalar(dict(static_options)),
                }
            )
        registry = cls(
            records,
            optimizer_type=_require_string(value["optimizer_type"], field="optimizer_type"),
            optimizer_groups=groups,
            parameters=parameters,
        )
        group_by_name: dict[str, str] = {}
        group_by_id = {str(group["group_id"]): group for group in groups}
        if len(group_by_id) != len(groups):
            raise RegistryError("optimizer_groups.group_id 必须唯一")
        for group in groups:
            group_id = str(group["group_id"])
            for name in group["parameter_names"]:
                canonical = registry.canonical_name(name)
                if canonical in group_by_name:
                    raise RegistryError(f"参数 {canonical!r} 在 manifest optimizer_groups 中重复")
                record = registry.record(canonical)
                if not record.eligible or record.group_id != group_id:
                    raise RegistryError(
                        f"参数 {canonical!r} 与 manifest optimizer group 映射不一致"
                    )
                if record.learning_rate != group["learning_rate"]:
                    raise RegistryError(f"参数 {canonical!r} learning_rate 映射不一致")
                if record.weight_decay != group["weight_decay"]:
                    raise RegistryError(f"参数 {canonical!r} weight_decay 映射不一致")
                group_by_name[canonical] = group_id
        for record in registry:
            if record.eligible and group_by_name.get(record.canonical_name) != record.group_id:
                raise RegistryError(
                    f"eligible 参数 {record.canonical_name!r} 缺少 optimizer group 映射"
                )
            if not record.eligible and record.group_id is not None:
                raise RegistryError(
                    f"非 eligible 参数 {record.canonical_name!r} 不应绑定 optimizer group"
                )
        for field_name in (
            "coordinate_registry_hash", "optimizer_contract_hash", "runtime_layout_hash"
        ):
            declared = _require_sha256(value[field_name], field=field_name)
            if declared != getattr(registry, field_name):
                raise RegistryError(f"registry manifest {field_name} 与内容不一致")
        return registry

    def save(self, path: str | Path) -> Path:
        """以 canonical JSON 保存 registry manifest。"""

        from ..contracts.jsonio import write_canonical_json

        return write_canonical_json(path, self.to_manifest())

    @classmethod
    def load(cls, path: str | Path) -> "ParameterRegistry":
        """从 canonical JSON 加载无运行时 Parameter 绑定的 registry。"""

        from ..contracts.jsonio import load_canonical_json

        value = load_canonical_json(path)
        if not isinstance(value, Mapping):
            raise RegistryError("parameter registry manifest 顶层必须是 object")
        return cls.from_manifest(value)

    def validate_against_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        require_runtime_layout: bool = False,
    ) -> None:
        """把已保存 manifest 与当前模型/optimizer 做精确身份核对。"""

        if optimizer is None:
            raise RegistryError("validate_against_model 必须提供 optimizer")
        observed = type(self).from_model(model, optimizer)
        for field_name in ("coordinate_registry_hash", "optimizer_contract_hash"):
            if getattr(self, field_name) != getattr(observed, field_name):
                raise RegistryError(f"{field_name} 与当前模型/optimizer 不一致")
        if self._optimizer_runtime_payload() != observed._optimizer_runtime_payload():
            raise RegistryError("optimizer 参数组的学习率、weight_decay 或参数映射不一致")
        if require_runtime_layout and self.runtime_layout_hash != observed.runtime_layout_hash:
            raise RegistryError("runtime_layout_hash 与当前运行布局不一致")

    def _optimizer_runtime_payload(self) -> list[dict[str, Any]]:
        """返回用于校验 step-time optimizer 映射的规范 payload。"""

        return [
            {
                "group_id": str(group.get("group_id")),
                "parameter_names": sorted(str(name) for name in group.get("parameter_names", [])),
                "learning_rate": float(group.get("learning_rate")),
                "weight_decay": float(group.get("weight_decay", 0.0)),
            }
            for group in self.optimizer_groups
        ]

    def bind_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        require_runtime_layout: bool = False,
    ) -> "ParameterRegistry":
        """验证 manifest 后返回带当前 Parameter 引用的 registry。"""

        self.validate_against_model(
            model, optimizer, require_runtime_layout=require_runtime_layout
        )
        return type(self).from_model(model, optimizer)
