"""严格 JSON 读取、canonical 编码和 legacy 导入边界。

本项目把 JSON 分成两个明确边界：

* 内部 artifact 必须是 UTF-8、无 BOM、无重复键、无 ``NaN``/``Infinity``，并且
  字节必须与 :func:`canonical_json_bytes` 的输出完全一致；
* 外部 legacy JSON 仅允许额外出现 UTF-8 BOM 或非 canonical 空白。导入器先执行
  同样的重复键与有限数检查，再产生新的 canonical 字节，绝不会放宽内部 loader。

canonical 表示使用 Unicode 原文、对象键字典序、紧凑分隔符和单个尾随换行。
稳定哈希始终对这组字节计算 SHA-256，因此输入映射的插入顺序和平台换行符不会
改变 artifact 身份。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, TypeAlias

from param_importance_nlp.atomic import atomic_write_bytes

from .errors import CanonicalJSONError


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_UTF8_BOM = b"\xef\xbb\xbf"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """由 ``json.loads`` 调用，在对象仍保留原始键序时拒绝重复键。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"JSON 对象包含重复键：{key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    """拒绝 Python JSON 解码器默认接受、但标准 JSON 不允许的常量。"""

    raise CanonicalJSONError(f"JSON 包含非有限数：{token}")


def _validate_json_value(value: Any, *, path: str = "$") -> JSONValue:
    """递归验证值可无歧义地编码成项目 JSON 数据模型。

    这里故意拒绝 tuple、任意 Mapping 子类键以及 dataclass 等“看起来可序列化”的
    对象。调用方应先显式执行 ``to_dict()``，从而让公开 artifact 的 wire shape 不会
    因 JSON encoder 的隐式行为而改变。
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # ``allow_nan=False`` 也会检查，但在这里可以给出准确字段路径。
        if value != value or value in (float("inf"), float("-inf")):
            raise CanonicalJSONError(f"{path} 必须是有限浮点数")
        return value
    if isinstance(value, list):
        return [
            _validate_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(f"{path} 的对象键必须是字符串")
            normalized[key] = _validate_json_value(item, path=f"{path}.{key}")
        return normalized
    raise CanonicalJSONError(
        f"{path} 的类型 {type(value).__name__} 不属于 JSON 数据模型"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """返回项目唯一的 canonical UTF-8 JSON 字节。

    输出始终以一个 ``LF`` 结尾，便于命令行审阅；哈希函数也包含该换行。对象键
    排序只改变表示，不改变数组顺序，因为数组顺序通常承载参数 registry 或实验
    矩阵语义。
    """

    normalized = _validate_json_value(value)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:  # pragma: no cover - 防御性兜底
        raise CanonicalJSONError("值无法编码为 canonical JSON") from error
    return (encoded + "\n").encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    """计算 canonical JSON 字节的 SHA-256 小写十六进制摘要。"""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def loads_strict_json(payload: bytes, *, allow_bom: bool = False) -> JSONValue:
    """解析严格 JSON，但不要求输入空白已经 canonical 化。

    ``allow_bom`` 只供本模块的 legacy 导入器调用。业务代码应使用
    :func:`load_canonical_json`，不能把该开关暴露为“遇到错误就重试”的后门。
    """

    if not isinstance(payload, bytes):
        raise TypeError("payload 必须是 bytes")
    if payload.startswith(_UTF8_BOM):
        if not allow_bom:
            raise CanonicalJSONError("内部 JSON artifact 禁止 UTF-8 BOM")
        payload = payload[len(_UTF8_BOM) :]
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CanonicalJSONError("JSON artifact 不是严格 UTF-8") from error
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except CanonicalJSONError:
        raise
    except json.JSONDecodeError as error:
        raise CanonicalJSONError(
            f"JSON 语法错误：line={error.lineno}, column={error.colno}"
        ) from error
    return _validate_json_value(decoded)


def load_canonical_json(path: str | Path) -> JSONValue:
    """读取并验证内部 canonical artifact。

    除语义安全检查外，本函数还比较原始字节与重新编码结果。这会拒绝漂亮打印、
    CRLF、未排序键和多余空白，保证同一 artifact 在任何机器上只有一种字节身份。
    """

    source = Path(path)
    payload = source.read_bytes()
    decoded = loads_strict_json(payload)
    expected = canonical_json_bytes(decoded)
    if payload != expected:
        raise CanonicalJSONError(
            f"{source} 不是 canonical JSON；请通过 canonical 发布器重新生成"
        )
    return decoded


def write_canonical_json(path: str | Path, value: Any) -> Path:
    """以原子替换方式发布 canonical artifact，并返回目标路径。"""

    target = Path(path)
    atomic_write_bytes(target, canonical_json_bytes(value))
    return target


def import_legacy_json(
    source: str | Path,
    target: str | Path | None = None,
) -> JSONValue:
    """读取 legacy JSON，并可选择发布到一个新的 canonical 路径。

    legacy 边界只兼容 UTF-8 BOM 与格式空白；重复键、非有限数和非 UTF-8 编码仍然
    立即失败。为避免“原地修复”掩盖来源，指定 ``target`` 时不允许与 ``source``
    解析到同一路径。
    """

    source_path = Path(source)
    decoded = loads_strict_json(source_path.read_bytes(), allow_bom=True)
    if target is not None:
        target_path = Path(target)
        if source_path.resolve() == target_path.resolve():
            raise CanonicalJSONError("legacy 导入必须发布到新路径，不能原地覆盖来源")
        write_canonical_json(target_path, decoded)
    return decoded


def ensure_json_object(value: JSONValue, *, field: str = "artifact") -> dict[str, JSONValue]:
    """把通用 JSON 结果收窄为对象，供各合同 ``from_mapping`` 入口复用。"""

    if not isinstance(value, dict):
        raise CanonicalJSONError(f"{field} 顶层必须是 JSON object")
    return value
