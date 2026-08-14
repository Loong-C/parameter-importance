"""Stage 1 使用的资产 manifest 读取边界。

该入口刻意不复用 Stage 0 的“内部 canonical 文件”读取器：Stage 1 接受外部
交付的 UTF-8 与 UTF-8 BOM JSON，但仍使用同一套严格 JSON、字段、revision、文件
声明和 asset_id 校验。实际 provider 读取前，可再传入 ``asset_root`` 执行文件大小
与 SHA-256 验证。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from . import assets
from .contracts.jsonio import CanonicalJSONError, loads_strict_json


class Stage1ManifestError(ValueError):
    """Stage 1 manifest 无法安全进入 provider。"""


class Stage1ManifestReadError(Stage1ManifestError):
    """来源文件不可读。"""


class Stage1ManifestEncodingError(Stage1ManifestError):
    """来源不是严格 UTF-8 JSON（允许 BOM，但不允许其他编码）。"""


class Stage1ManifestValidationError(Stage1ManifestError):
    """JSON 已解析，但 manifest 合同或调用方期望不满足。"""


def read_stage1_manifest_bytes(path: str | Path) -> bytes:
    """仅执行文件读取，不解析或验证内容。"""

    source = Path(path)
    try:
        return source.read_bytes()
    except OSError as error:
        raise Stage1ManifestReadError(f"无法读取 stage1 manifest: {source}") from error


def _validate_expectations(
    manifest: Mapping[str, Any],
    *,
    source: str,
    expected_revision: str | None,
    expected_files: Iterable[str] | None,
) -> None:
    if expected_revision is not None and manifest.get("revision") != expected_revision:
        raise Stage1ManifestValidationError(
            f"{source}: revision 不匹配，期望 {expected_revision!r}，"
            f"实际 {manifest.get('revision')!r}"
        )
    if expected_files is not None:
        expected = tuple(expected_files)
        if any(not isinstance(item, str) or not item for item in expected):
            raise Stage1ManifestValidationError(
                f"{source}: expected_files 必须是非空字符串数组"
            )
        declared = tuple(item["path"] for item in manifest["files"])
        if set(declared) != set(expected) or len(declared) != len(expected):
            raise Stage1ManifestValidationError(
                f"{source}: manifest files 与允许文件集合不一致；"
                f"declared={sorted(declared)}, expected={sorted(expected)}"
            )


def parse_stage1_manifest_bytes(
    payload: bytes,
    *,
    source: str = "<bytes>",
    expected_revision: str | None = None,
    expected_files: Iterable[str] | None = None,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """解析并验证 Stage 1 manifest。

    解析层允许 UTF-8 BOM；未知编码、重复键、非有限数、非法字段、revision 漂移、
    未声明的替代文件和物理文件 size/hash 不匹配都会 fail-closed。
    """

    if not isinstance(payload, bytes):
        raise Stage1ManifestEncodingError(f"{source}: manifest payload 必须是 bytes")
    try:
        decoded = loads_strict_json(payload, allow_bom=True)
    except (CanonicalJSONError, UnicodeError, ValueError) as error:
        raise Stage1ManifestEncodingError(
            f"{source}: 不是可接受的 UTF-8 JSON manifest"
        ) from error
    if not isinstance(decoded, dict):
        raise Stage1ManifestValidationError(f"{source}: manifest 顶层必须是 object")
    try:
        assets.validate_manifest(decoded)
    except assets.AssetManifestError as error:
        raise Stage1ManifestValidationError(
            f"{source}: Stage 0 asset manifest 合同校验失败"
        ) from error
    _validate_expectations(
        decoded,
        source=source,
        expected_revision=expected_revision,
        expected_files=expected_files,
    )
    if asset_root is not None:
        try:
            assets.verify_only(decoded, asset_root)
        except assets.AssetManifestError as error:
            raise Stage1ManifestValidationError(
                f"{source}: asset_root 文件 size/hash 校验失败"
            ) from error
    return dict(decoded)


def load_stage1_asset_manifest(
    path: str | Path,
    *,
    expected_revision: str | None = None,
    expected_files: Iterable[str] | None = None,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """按“读文件 -> parse JSON -> 合同校验 -> 可选物理校验”分层加载。"""

    source = Path(path)
    return parse_stage1_manifest_bytes(
        read_stage1_manifest_bytes(source),
        source=str(source),
        expected_revision=expected_revision,
        expected_files=expected_files,
        asset_root=asset_root,
    )


__all__ = [
    "Stage1ManifestEncodingError",
    "Stage1ManifestError",
    "Stage1ManifestReadError",
    "Stage1ManifestValidationError",
    "load_stage1_asset_manifest",
    "parse_stage1_manifest_bytes",
    "read_stage1_manifest_bytes",
]
