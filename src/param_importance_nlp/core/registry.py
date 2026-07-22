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
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from .errors import RegistryError


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", MappingProxyType(dict(sorted(self.tags.items()))))

    @property
    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= size
        return result


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
            "groups": [dict(group) for group in self.optimizer_groups],
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
        Parameter，PyTorch 默认遍历会隐藏后一个名称。canonical name 采用模型
        的确定性遍历顺序，而非对象地址或哈希表顺序。
        """

        try:
            named = list(model.named_parameters(remove_duplicate=False))
        except TypeError:  # pragma: no cover - 兼容非常旧的 torch
            named = list(model.named_parameters())
        if not named:
            raise RegistryError("模型没有任何命名参数")

        names_by_object: dict[int, list[str]] = {}
        object_by_id: dict[int, torch.nn.Parameter] = {}
        object_order: list[int] = []
        for name, parameter in named:
            if not name:
                raise RegistryError("参数名称不能为空")
            identity = id(parameter)
            if identity not in names_by_object:
                names_by_object[identity] = []
                object_by_id[identity] = parameter
                object_order.append(identity)
            names_by_object[identity].append(name)

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
            static_options = {
                str(key): _json_scalar(value)
                for key, value in sorted(group.items())
                if key not in {"params", "lr", "initial_lr"}
            }
            optimizer_groups.append({"group_id": group_id, "static_options": static_options})

        # 不同 Parameter 对象的 storage 区间不得重叠；同对象别名已在前面合并。
        spans: list[tuple[int, tuple[str, int, int, int], str]] = []
        for identity in object_order:
            parameter = object_by_id[identity]
            span = _storage_span(parameter)
            if span is None:
                continue
            canonical_name = names_by_object[identity][0]
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
            canonical = all_names[0]
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
