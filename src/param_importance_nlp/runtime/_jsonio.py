"""运行时内部使用的规范 JSON 读取边界。

项目写出的 artifact 必须是 UTF-8、无 BOM、无重复键且不含 NaN/Infinity。
标准库 ``json.loads`` 默认会接受后三类输入，因此不能直接作为 checkpoint、
事件或 tensor manifest 的信任边界。此模块只负责语法层；具体 schema 仍由各
调用模块检查。外部 legacy manifest 的 BOM 兼容导入属于 ``contracts`` 层，
不能借此函数放宽内部 artifact。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_canonical_json_bytes(payload: bytes, *, source: str) -> Any:
    """严格解码规范 JSON；错误使用稳定机器码前缀。"""

    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"CANONICAL_JSON_BOM_FORBIDDEN:{source}")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CANONICAL_JSON_INVALID_UTF8:{source}") from exc

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"CANONICAL_JSON_DUPLICATE_KEY:{key}:{source}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"CANONICAL_JSON_NONFINITE:{token}:{source}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except ValueError:
        raise
    except json.JSONDecodeError as exc:  # pragma: no cover - JSONDecodeError 是 ValueError
        raise ValueError(f"CANONICAL_JSON_INVALID:{source}") from exc


def load_canonical_json(path: str | Path) -> Any:
    """从文件读取严格规范 JSON。"""

    target = Path(path)
    return load_canonical_json_bytes(target.read_bytes(), source=str(target))
