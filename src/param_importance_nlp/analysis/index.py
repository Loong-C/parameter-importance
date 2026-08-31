"""Stage 9 冻结派生表索引。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Mapping

from ..contracts.jsonio import JSONValue, canonical_json_hash


_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DerivedTableIndex:
    """把表名映射到 canonical source commit 与内容 hash。

    索引自身不保存数值，因此任何表格、图片或报告仍必须重新读取并核对源表。
    """

    index_id: str
    scope: str
    tables: Mapping[str, Mapping[str, str]]
    formal_eligible: bool

    def __post_init__(self) -> None:
        if not self.index_id or self.scope not in {"local_fixture", "formal"}:
            raise ValueError("DERIVED_TABLE_INDEX_ID_OR_SCOPE_INVALID")
        if self.formal_eligible != (self.scope == "formal"):
            raise ValueError("DERIVED_TABLE_INDEX_FORMAL_ELIGIBILITY_MISMATCH")
        normalized: dict[str, Mapping[str, str]] = {}
        for name, value in self.tables.items():
            if not isinstance(name, str) or not name or not isinstance(value, Mapping):
                raise TypeError("DERIVED_TABLE_INDEX_ENTRY_INVALID")
            if set(value) != {"source_ref", "content_hash", "schema_version"}:
                raise ValueError("DERIVED_TABLE_INDEX_ENTRY_FIELDS_INVALID")
            reference = value["source_ref"]
            path = PurePosixPath(reference)
            if (
                not reference
                or "\\" in reference
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError("DERIVED_TABLE_INDEX_SOURCE_REF_INVALID")
            if _HASH.fullmatch(value["content_hash"]) is None:
                raise ValueError("DERIVED_TABLE_INDEX_CONTENT_HASH_INVALID")
            if not value["schema_version"]:
                raise ValueError("DERIVED_TABLE_INDEX_SCHEMA_EMPTY")
            normalized[name] = MappingProxyType(dict(value))
        if not normalized:
            raise ValueError("DERIVED_TABLE_INDEX_EMPTY")
        object.__setattr__(self, "tables", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "schema_version": "derived-table-index-v1",
            "index_id": self.index_id,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "tables": {
                name: dict(value) for name, value in sorted(self.tables.items())
            },
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DerivedTableIndex":
        expected = {
            "schema_version", "index_id", "scope", "formal_eligible", "tables",
            "artifact_hash",
        }
        if set(value) != expected or value.get("schema_version") != "derived-table-index-v1":
            raise ValueError("DERIVED_TABLE_INDEX_FIELDS_OR_VERSION_INVALID")
        tables = value["tables"]
        if not isinstance(tables, Mapping) or not all(
            isinstance(item, Mapping) for item in tables.values()
        ):
            raise TypeError("DERIVED_TABLE_INDEX_TABLES_INVALID")
        result = cls(
            index_id=value["index_id"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            tables=tables,  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
        )
        if value["artifact_hash"] != result.to_dict()["artifact_hash"]:
            raise ValueError("DERIVED_TABLE_INDEX_HASH_MISMATCH")
        return result


__all__ = ["DerivedTableIndex"]
