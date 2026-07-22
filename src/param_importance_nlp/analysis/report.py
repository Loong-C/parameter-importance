"""只从冻结、hash 绑定源表构建确定性 Stage 9 报告。

报告 ID、内容摘要与 Markdown 都不包含当前时间、随机顺序或本机绝对路径。
相同源表和指标在两次独立运行中会产生同一 SHA-256。源表未冻结、hash 不符
或包含 NaN/Inf 时立即拒绝，禁止在报告中手工覆盖实验数字。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .metrics import MetricResult


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _deep_freeze(value: object) -> object:
    """递归冻结源表单元格中的 JSON 容器。"""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class FrozenSourceTable:
    """一份可重建分析的 canonical 小型源表。"""

    name: str
    schema_version: str
    rows: tuple[Mapping[str, object], ...]
    content_hash: str
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.schema_version:
            raise ValueError("source table name/schema_version 不能为空")
        if not self.rows:
            raise ValueError("source table rows 不能为空")
        if not self.frozen:
            raise ValueError("分析只能消费 frozen source table")
        for row_index, row in enumerate(self.rows):
            if not isinstance(row, Mapping) or not row:
                raise TypeError(f"source table row {row_index} 必须是非空 object")
            if any(not isinstance(column, str) or not column for column in row):
                raise TypeError(f"source table row {row_index} 的列名必须是非空字符串")
        canonical_rows = [dict(row) for row in self.rows]
        expected = _digest(
            {
                "name": self.name,
                "schema_version": self.schema_version,
                "rows": canonical_rows,
            }
        )
        if self.content_hash != expected:
            raise ValueError("source table content_hash 与内容不一致")
        object.__setattr__(
            self,
            "rows",
            tuple(_deep_freeze(copy.deepcopy(dict(row))) for row in self.rows),
        )

    @classmethod
    def from_rows(
        cls,
        *,
        name: str,
        schema_version: str,
        rows: Sequence[Mapping[str, object]],
    ) -> "FrozenSourceTable":
        copied = tuple(copy.deepcopy(dict(row)) for row in rows)
        content_hash = _digest(
            {"name": name, "schema_version": schema_version, "rows": list(copied)}
        )
        return cls(name, schema_version, copied, content_hash, True)

    @property
    def columns(self) -> tuple[str, ...]:
        """返回每一行都实际存在的 canonical 列集合。"""

        shared: set[str] | None = None
        for row in self.rows:
            shared = set(row) if shared is None else shared.intersection(row)
        return tuple(sorted(shared or ()))

    def to_dict(self) -> dict[str, object]:
        """返回包含完整 rows 的冻结源表 wire object。"""

        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "rows": [_json_ready(row) for row in self.rows],
            "content_hash": self.content_hash,
            "frozen": self.frozen,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FrozenSourceTable":
        """严格加载既有冻结表；不替缺少 hash 的 rows 现场自封。"""

        required = {"name", "schema_version", "rows", "content_hash", "frozen"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("FROZEN_SOURCE_TABLE_FIELDS_MISMATCH")
        for field_name in ("name", "schema_version", "content_hash"):
            if not isinstance(value[field_name], str):
                raise TypeError(f"FROZEN_SOURCE_TABLE_FIELD_NOT_STRING:{field_name}")
        rows = value["rows"]
        if not isinstance(rows, list) or not all(
            isinstance(row, Mapping) for row in rows
        ):
            raise TypeError("FROZEN_SOURCE_TABLE_ROWS_NOT_OBJECT_ARRAY")
        if not isinstance(value["frozen"], bool):
            raise TypeError("FROZEN_SOURCE_TABLE_FROZEN_NOT_BOOLEAN")
        return cls(
            name=value["name"],
            schema_version=value["schema_version"],
            rows=tuple(rows),
            content_hash=value["content_hash"],
            frozen=value["frozen"],
        )


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """报告中保存的源表身份，不含可手工修改的派生数字。"""

    name: str
    schema_version: str
    content_hash: str
    row_count: int
    frozen: bool

    def __post_init__(self) -> None:
        if not self.name or not self.schema_version or self.row_count <= 0:
            raise ValueError("SourceArtifact 字段不完整")
        if not _valid_sha256(self.content_hash):
            raise ValueError("SourceArtifact content_hash 必须是 SHA-256")
        if not self.frozen:
            raise ValueError("AnalysisReport 不接受未冻结 source")


@dataclass(frozen=True, slots=True)
class MetricSourceBinding:
    """一个标量指标对冻结源表及确定派生方法的结构化绑定。

    ``derivation_id`` 应是版本化、机器可识别的算法名称，例如
    ``stage9.pearson.v1``；``input_columns`` 列出重建该指标所需的全部源列。
    指标函数自身产生的补充 metadata 可以继续保留，但不能替代此绑定。
    """

    source_name: str
    source_hash: str
    derivation_id: str
    input_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("METRIC_SOURCE_NAME_INVALID")
        if not _valid_sha256(self.source_hash):
            raise ValueError("METRIC_SOURCE_HASH_INVALID")
        if not isinstance(self.derivation_id, str) or not self.derivation_id:
            raise ValueError("METRIC_DERIVATION_ID_INVALID")
        columns = tuple(self.input_columns)
        if not columns or any(
            not isinstance(column, str) or not column for column in columns
        ):
            raise ValueError("METRIC_INPUT_COLUMNS_INVALID")
        if len(columns) != len(set(columns)):
            raise ValueError("METRIC_INPUT_COLUMNS_DUPLICATED")
        object.__setattr__(self, "input_columns", columns)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "source_hash": self.source_hash,
            "derivation_id": self.derivation_id,
            "input_columns": list(self.input_columns),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MetricSourceBinding":
        required = {
            "source_name",
            "source_hash",
            "derivation_id",
            "input_columns",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("METRIC_SOURCE_BINDING_FIELDS_MISMATCH")
        for field_name in ("source_name", "source_hash", "derivation_id"):
            if not isinstance(value[field_name], str):
                raise TypeError(f"METRIC_SOURCE_BINDING_FIELD_NOT_STRING:{field_name}")
        columns = value["input_columns"]
        if not isinstance(columns, list) or not all(
            isinstance(column, str) for column in columns
        ):
            raise TypeError("METRIC_SOURCE_BINDING_COLUMNS_NOT_STRING_ARRAY")
        return cls(
            source_name=value["source_name"],
            source_hash=value["source_hash"],
            derivation_id=value["derivation_id"],
            input_columns=tuple(columns),
        )


def _binding_from_metric(name: str, result: MetricResult) -> MetricSourceBinding:
    value = result.metadata.get("source_binding")
    if not isinstance(value, Mapping):
        raise ValueError(f"metric {name!r} 缺少结构化 source_binding")
    normalized = _json_ready(value)
    assert isinstance(normalized, dict)
    return MetricSourceBinding.from_mapping(normalized)


def _validate_metric_sources(
    source_artifacts: Sequence[SourceArtifact],
    metrics: Mapping[str, MetricResult],
) -> None:
    sources = {source.name: source for source in source_artifacts}
    for name, result in metrics.items():
        if not isinstance(name, str) or not name:
            raise TypeError("AnalysisReport metric 名必须是非空字符串")
        if not isinstance(result, MetricResult):
            raise TypeError(f"metric {name!r} 必须是 MetricResult")
        binding = _binding_from_metric(name, result)
        source = sources.get(binding.source_name)
        if source is None:
            raise ValueError(
                f"metric {name!r} 引用了未登记 source {binding.source_name!r}"
            )
        if source.content_hash != binding.source_hash:
            raise ValueError(f"metric {name!r} 的 source_hash 与冻结源不一致")


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Stage 9 确定性分析报告合同。"""

    report_id: str
    source_artifacts: tuple[SourceArtifact, ...]
    metrics: Mapping[str, MetricResult]
    report_hash: str
    schema_version: str = "analysis-report-v1"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.report_id or not self.source_artifacts:
            raise ValueError("report_id 与 source_artifacts 不能为空")
        if self.schema_version != "analysis-report-v1":
            raise ValueError("不支持的 AnalysisReport schema")
        if len({source.name for source in self.source_artifacts}) != len(
            self.source_artifacts
        ):
            raise ValueError("source artifact name 必须唯一")
        normalized_metrics: dict[str, MetricResult] = {}
        for name, result in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise TypeError("AnalysisReport metric 名必须是非空字符串")
            if not isinstance(result, MetricResult):
                raise TypeError(f"metric {name!r} 必须是 MetricResult")
            metadata = _json_ready(result.metadata)
            if not isinstance(metadata, dict):  # pragma: no cover - MetricResult 已限定 Mapping
                raise TypeError(f"metric {name!r}.metadata 必须是 object")
            frozen_metadata = _deep_freeze(metadata)
            assert isinstance(frozen_metadata, Mapping)
            normalized_metrics[name] = MetricResult(
                defined=result.defined,
                value=result.value,
                reason=result.reason,
                metadata=frozen_metadata,
            )
        object.__setattr__(self, "metrics", MappingProxyType(normalized_metrics))
        _validate_metric_sources(self.source_artifacts, self.metrics)
        if not _valid_sha256(self.report_hash):
            raise ValueError("report_hash 必须是 SHA-256")
        payload = _report_payload(
            report_id=self.report_id,
            source_artifacts=self.source_artifacts,
            metrics=self.metrics,
            schema_version=self.schema_version,
            metadata=self.metadata,
        )
        if _digest(payload) != self.report_hash:
            raise ValueError("report_hash 与报告内容不一致")
        frozen_report_metadata = _deep_freeze(copy.deepcopy(dict(self.metadata)))
        assert isinstance(frozen_report_metadata, Mapping)
        object.__setattr__(self, "metadata", frozen_report_metadata)

    @property
    def digest(self) -> str:
        return self.report_hash

    def to_dict(self) -> dict[str, object]:
        """返回可直接交给 strict canonical JSON writer 的字典。"""

        return _report_payload(
            report_id=self.report_id,
            source_artifacts=self.source_artifacts,
            metrics=self.metrics,
            schema_version=self.schema_version,
            metadata=self.metadata,
        ) | {"report_hash": self.report_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AnalysisReport":
        """严格加载报告并复算 hash，不允许把任意 rows 现场自封为冻结源。"""

        required = {
            "schema_version",
            "report_id",
            "source_artifacts",
            "metrics",
            "metadata",
            "report_hash",
        }
        if set(value) != required:
            raise ValueError(
                "AnalysisReport 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        for name in ("schema_version", "report_id", "report_hash"):
            if not isinstance(value[name], str):
                raise TypeError(f"AnalysisReport {name} 必须是字符串")
        source_values = value["source_artifacts"]
        if not isinstance(source_values, list) or not source_values:
            raise TypeError("AnalysisReport source_artifacts 必须是非空 object 数组")
        sources: list[SourceArtifact] = []
        source_fields = {
            "name",
            "schema_version",
            "content_hash",
            "row_count",
            "frozen",
        }
        for item in source_values:
            if not isinstance(item, Mapping) or set(item) != source_fields:
                raise ValueError("AnalysisReport source artifact 字段集合不匹配")
            if any(
                not isinstance(item[name], str)
                for name in ("name", "schema_version", "content_hash")
            ):
                raise TypeError("source artifact 身份字段必须是字符串")
            if isinstance(item["row_count"], bool) or not isinstance(
                item["row_count"], int
            ):
                raise TypeError("source artifact row_count 必须是整数")
            if not isinstance(item["frozen"], bool):
                raise TypeError("source artifact frozen 必须是布尔值")
            sources.append(
                SourceArtifact(
                    name=item["name"],
                    schema_version=item["schema_version"],
                    content_hash=item["content_hash"],
                    row_count=item["row_count"],
                    frozen=item["frozen"],
                )
            )
        metric_values = value["metrics"]
        if not isinstance(metric_values, Mapping) or not metric_values:
            raise TypeError("AnalysisReport metrics 必须是非空 object")
        metric_fields = {"defined", "value", "reason", "metadata"}
        metrics: dict[str, MetricResult] = {}
        for name, item in metric_values.items():
            if not isinstance(name, str) or not name:
                raise TypeError("AnalysisReport metric 名必须是非空字符串")
            if not isinstance(item, Mapping) or set(item) != metric_fields:
                raise ValueError(f"metric {name!r} 字段集合不匹配")
            if not isinstance(item["defined"], bool):
                raise TypeError(f"metric {name!r}.defined 必须是布尔值")
            if item["value"] is not None and (
                isinstance(item["value"], bool)
                or not isinstance(item["value"], (int, float))
            ):
                raise TypeError(f"metric {name!r}.value 必须是数值或 null")
            if item["reason"] is not None and not isinstance(item["reason"], str):
                raise TypeError(f"metric {name!r}.reason 必须是字符串或 null")
            if not isinstance(item["metadata"], Mapping):
                raise TypeError(f"metric {name!r}.metadata 必须是 object")
            binding_value = item["metadata"].get("source_binding")
            if not isinstance(binding_value, Mapping):
                raise ValueError(f"metric {name!r} 缺少结构化 source_binding")
            binding = MetricSourceBinding.from_mapping(binding_value)
            normalized_metadata = dict(item["metadata"])
            normalized_metadata["source_binding"] = binding.to_dict()
            metrics[name] = MetricResult(
                defined=item["defined"],
                value=(
                    None if item["value"] is None else float(item["value"])
                ),
                reason=item["reason"],
                metadata=normalized_metadata,
            )
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("AnalysisReport metadata 必须是 object")
        return cls(
            report_id=value["report_id"],
            source_artifacts=tuple(sources),
            metrics=metrics,
            report_hash=value["report_hash"],
            schema_version=value["schema_version"],
            metadata=value["metadata"],
        )

    def render_markdown(self) -> str:
        """从结构化字段确定性生成 Markdown，不接受手工数字注入。"""

        lines = [f"# 分析报告 `{self.report_id}`", "", "## 冻结源", ""]
        lines.extend(
            f"- `{source.name}`：{source.row_count} 行，SHA-256 `{source.content_hash}`"
            for source in sorted(self.source_artifacts, key=lambda item: item.name)
        )
        lines.extend(["", "## 指标", ""])
        for name, result in sorted(self.metrics.items()):
            binding = _binding_from_metric(name, result)
            provenance = (
                f"`{binding.source_name}` / `{binding.derivation_id}` / "
                f"列 `{', '.join(binding.input_columns)}`"
            )
            if result.defined:
                assert result.value is not None
                lines.append(f"- `{name}`：{result.value:.12g}（{provenance}）")
            else:
                lines.append(
                    f"- `{name}`：未定义（`{result.reason}`；{provenance}）"
                )
        lines.extend(["", f"报告 SHA-256：`{self.report_hash}`", ""])
        return "\n".join(lines)


def _metric_payload(result: MetricResult) -> dict[str, object]:
    return {
        "defined": result.defined,
        "value": result.value,
        "reason": result.reason,
        "metadata": _json_ready(result.metadata),
    }


def _source_payload(source: SourceArtifact) -> dict[str, object]:
    return {
        "name": source.name,
        "schema_version": source.schema_version,
        "content_hash": source.content_hash,
        "row_count": source.row_count,
        "frozen": source.frozen,
    }


def _report_payload(
    *,
    report_id: str,
    source_artifacts: Sequence[SourceArtifact],
    metrics: Mapping[str, MetricResult],
    schema_version: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "report_id": report_id,
        "source_artifacts": [
            _source_payload(source)
            for source in sorted(source_artifacts, key=lambda item: item.name)
        ],
        "metrics": {
            name: _metric_payload(result) for name, result in sorted(metrics.items())
        },
        "metadata": _json_ready(metadata),
    }


class AnalysisReportBuilder:
    """只接受冻结 source 且拒绝无来源手工数字的报告构建器。"""

    def __init__(self, *, report_id: str) -> None:
        if not report_id:
            raise ValueError("report_id 不能为空")
        self.report_id = report_id
        self._sources: dict[str, FrozenSourceTable] = {}
        self._metrics: dict[str, MetricResult] = {}

    def add_source(self, table: FrozenSourceTable) -> None:
        if not isinstance(table, FrozenSourceTable):
            raise TypeError("AnalysisReportBuilder 只接受 FrozenSourceTable")
        existing = self._sources.get(table.name)
        if existing is not None and existing.content_hash != table.content_hash:
            raise ValueError(f"同名 source {table.name!r} 的内容 hash 冲突")
        self._sources[table.name] = table

    def add_metric(
        self,
        name: str,
        result: MetricResult,
        *,
        source: FrozenSourceTable,
        derivation_id: str,
        input_columns: Sequence[str],
    ) -> None:
        """登记一个可从冻结表重建的结构化指标。

        Args:
            name: 报告内唯一指标名。
            result: 已由确定算法计算出的标量或显式未定义结果。
            source: 已先通过 :meth:`add_source` 登记的冻结源表。
            derivation_id: 版本化算法身份；禁止用自然语言备注替代。
            input_columns: 重建时读取的完整列集合，且必须存在于 source 每一行。

        指标函数原有 metadata 会保留；``source_binding`` 由 builder 独占写入，
        调用方不能伪造或覆盖。
        """

        if not name:
            raise ValueError("metric name 不能为空")
        if not isinstance(result, MetricResult):
            raise TypeError("metric result 必须是 MetricResult")
        if not isinstance(source, FrozenSourceTable):
            raise TypeError("metric source 必须是 FrozenSourceTable")
        registered = self._sources.get(source.name)
        if registered is None:
            raise ValueError(f"metric source {source.name!r} 尚未登记")
        if registered.content_hash != source.content_hash:
            raise ValueError(f"metric source {source.name!r} hash 与登记值冲突")
        binding = MetricSourceBinding(
            source_name=source.name,
            source_hash=source.content_hash,
            derivation_id=derivation_id,
            input_columns=tuple(input_columns),
        )
        missing_columns = sorted(set(binding.input_columns) - set(source.columns))
        if missing_columns:
            raise ValueError(
                f"metric {name!r} 引用了 source 中不完整的列：{missing_columns}"
            )
        metadata = dict(result.metadata)
        if "source_binding" in metadata:
            raise ValueError("metric source_binding 只能由 AnalysisReportBuilder 写入")
        metadata["source_binding"] = binding.to_dict()
        bound_result = MetricResult(
            defined=result.defined,
            value=result.value,
            reason=result.reason,
            metadata=metadata,
        )
        if name in self._metrics and self._metrics[name] != bound_result:
            raise ValueError(f"metric {name!r} 被重复定义")
        self._metrics[name] = bound_result

    def build(self, *, metadata: Mapping[str, object] | None = None) -> AnalysisReport:
        if not self._sources:
            raise ValueError("报告至少需要一个冻结 source table")
        if not self._metrics:
            raise ValueError("报告至少需要一个结构化 metric")
        sources = tuple(
            SourceArtifact(
                name=table.name,
                schema_version=table.schema_version,
                content_hash=table.content_hash,
                row_count=len(table.rows),
                frozen=table.frozen,
            )
            for table in sorted(self._sources.values(), key=lambda item: item.name)
        )
        payload = _report_payload(
            report_id=self.report_id,
            source_artifacts=sources,
            metrics=self._metrics,
            schema_version="analysis-report-v1",
            metadata=metadata or {},
        )
        report_hash = _digest(payload)
        return AnalysisReport(
            report_id=self.report_id,
            source_artifacts=sources,
            metrics=self._metrics,
            report_hash=report_hash,
            metadata=metadata or {},
        )
