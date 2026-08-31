"""JSON 元数据的递归不可变表示与可序列化还原。

``dataclass(frozen=True)`` 只禁止字段重新赋值，并不会冻结字段内部的 dict/list。
若 artifact hash 在构造后仍可通过嵌套容器修改，就会出现“对象内容已变、身份摘要
未变”或摘要随读取时机漂移。公共 artifact 因此在构造边界调用
:func:`freeze_json_value`，序列化时调用 :func:`thaw_json_value`。
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping


def freeze_json_value(value: Any, *, field: str = "metadata") -> Any:
    """递归复制并冻结严格 JSON 值；未知对象和非有限数 fail-closed。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} 包含 NaN/Inf")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} 的 object key 必须是字符串")
            frozen[key] = freeze_json_value(item, field=f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{field} 包含不支持的 JSON 类型: {type(value).__name__}")


def thaw_json_value(value: Any) -> Any:
    """把冻结表示递归还原为 canonical JSON writer 可接受的 dict/list。"""

    if isinstance(value, Mapping):
        return {str(key): thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value


def freeze_json_mapping(
    value: Mapping[str, Any], *, field: str = "metadata"
) -> Mapping[str, Any]:
    """类型收窄的 metadata 入口。"""

    frozen = freeze_json_value(value, field=field)
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = ["freeze_json_mapping", "freeze_json_value", "thaw_json_value"]
