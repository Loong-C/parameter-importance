"""由冻结源表驱动、可重建且 hash 绑定的 Stage 9 图表合同。

图表分成两个边界：:class:`ChartSpec` 只描述科学数据选择与视觉类型，
:class:`ChartArtifact` 再记录规范是否仅发布为 spec，或由哪个可选 renderer 产生了
哪一组字节。输出文件路径、当前时间和主机信息均不属于任何字段，因此不会进入
hash。Matplotlib 只在显式调用 renderer 时延迟导入；读取或验证 spec 不要求安装
绘图库。

本模块只接受 :class:`~param_importance_nlp.analysis.report.FrozenSourceTable`。
筛选是结构化谓词而不是任意 Python 回调，排序字段与方向也完整进入 spec hash，
从而相同 source hash 与 spec 必然重建出相同的逻辑行序列。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Mapping, Sequence

from param_importance_nlp.atomic import atomic_write_bytes
from param_importance_nlp.contracts import DependencyUnavailable
from param_importance_nlp.contracts.jsonio import canonical_json_bytes, canonical_json_hash

from .report import FrozenSourceTable


_CHART_TYPES = frozenset({"line", "scatter", "bar"})
_FILTER_OPERATORS = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "in"})
_VOLATILE_OPTION_KEYS = frozenset(
    {
        "path",
        "output_path",
        "directory",
        "cwd",
        "created_at",
        "timestamp",
        "time",
        "date",
        "hostname",
    }
)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _json_value(value: object, *, field_path: str = "$") -> object:
    """复制为有限 JSON 数据模型，拒绝 renderer 私有对象悄悄进入 hash。"""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"CHART_NONFINITE_JSON_VALUE:{field_path}")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"CHART_JSON_KEY_NOT_STRING:{field_path}")
            result[key] = _json_value(item, field_path=f"{field_path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, field_path=f"{field_path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"CHART_UNSUPPORTED_JSON_TYPE:{field_path}:{type(value).__name__}")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _normalize_filters(
    filters: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    normalized: list[Mapping[str, object]] = []
    for index, predicate in enumerate(filters):
        if not isinstance(predicate, Mapping) or set(predicate) != {
            "column",
            "operator",
            "value",
        }:
            raise ValueError(f"CHART_FILTER_FIELDS_MISMATCH:{index}")
        column = predicate["column"]
        operator = predicate["operator"]
        if not isinstance(column, str) or not column:
            raise TypeError(f"CHART_FILTER_COLUMN_INVALID:{index}")
        if not isinstance(operator, str) or operator not in _FILTER_OPERATORS:
            raise ValueError(f"CHART_FILTER_OPERATOR_INVALID:{index}:{operator}")
        value = _json_value(predicate["value"], field_path=f"$.filters[{index}].value")
        if operator == "in":
            if not isinstance(value, list) or not value:
                raise ValueError(f"CHART_FILTER_IN_REQUIRES_NONEMPTY_ARRAY:{index}")
        elif isinstance(value, (dict, list)):
            raise TypeError(f"CHART_FILTER_SCALAR_REQUIRED:{index}")
        normalized.append(
            MappingProxyType(
                {
                    "column": column,
                    "operator": operator,
                    "value": _freeze_json(value),
                }
            )
        )
    return tuple(normalized)


def _filter_payload(filters: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "column": str(predicate["column"]),
            "operator": str(predicate["operator"]),
            "value": _json_value(predicate["value"]),
        }
        for predicate in filters
    ]


def _columns_in_table(table: FrozenSourceTable) -> frozenset[str]:
    columns: set[str] | None = None
    for row in table.rows:
        row_columns = set(row)
        columns = row_columns if columns is None else columns.intersection(row_columns)
    return frozenset(columns or ())


def _validate_spec_against_table(spec: "ChartSpec", table: object) -> FrozenSourceTable:
    if not isinstance(table, FrozenSourceTable):
        raise TypeError("CHART_REQUIRES_FROZEN_SOURCE_TABLE")
    if (
        table.name != spec.source_name
        or table.schema_version != spec.source_schema_version
        or table.content_hash != spec.source_hash
    ):
        raise ValueError("CHART_SOURCE_IDENTITY_MISMATCH")
    available = _columns_in_table(table)
    referenced = {spec.x_column, *spec.y_columns, *spec.sort_columns}
    referenced.update(str(predicate["column"]) for predicate in spec.filters)
    missing = sorted(referenced - available)
    if missing:
        raise ValueError(f"CHART_SOURCE_COLUMNS_MISSING:{missing}")
    return table


def _compare_filter(value: object, operator: str, expected: object) -> bool:
    if operator == "eq":
        return value == expected
    if operator == "ne":
        return value != expected
    if operator == "in":
        assert isinstance(expected, tuple)
        return value in expected
    try:
        if operator == "lt":
            return value < expected  # type: ignore[operator]
        if operator == "le":
            return value <= expected  # type: ignore[operator]
        if operator == "gt":
            return value > expected  # type: ignore[operator]
        if operator == "ge":
            return value >= expected  # type: ignore[operator]
    except TypeError as exc:
        raise ValueError(f"CHART_FILTER_VALUES_NOT_COMPARABLE:{operator}") from exc
    raise AssertionError(f"未处理的 filter operator：{operator}")


def _sort_token(value: object) -> tuple[int, object]:
    """为常见 JSON 标量给出跨行稳定全序；复杂值禁止作为排序键。"""

    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("CHART_SORT_NONFINITE_VALUE")
        return (2, numeric)
    if isinstance(value, str):
        return (3, value)
    raise TypeError(f"CHART_SORT_COMPLEX_VALUE:{type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """绑定冻结源表与全部行变换语义的 canonical 图表规范。

    ``filters`` 采用 AND 语义并按给定顺序进入 hash；``sort_columns`` 形成复合
    排序键。source 原始行序也属于 :class:`FrozenSourceTable` 的 content hash，
    因而排序键相同的行仍有确定的稳定次序。
    """

    chart_id: str
    source_name: str
    source_schema_version: str
    source_hash: str
    chart_type: str
    x_column: str
    y_columns: tuple[str, ...]
    filters: tuple[Mapping[str, object], ...]
    sort_columns: tuple[str, ...]
    sort_descending: bool
    spec_hash: str
    schema_version: str = "analysis-chart-spec-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "analysis-chart-spec-v1":
            raise ValueError("CHART_SPEC_SCHEMA_MISMATCH")
        for field_name in (
            "chart_id",
            "source_name",
            "source_schema_version",
            "x_column",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"CHART_SPEC_STRING_FIELD_INVALID:{field_name}")
        if not _valid_sha256(self.source_hash) or not _valid_sha256(self.spec_hash):
            raise ValueError("CHART_SPEC_HASH_INVALID")
        if self.chart_type not in _CHART_TYPES:
            raise ValueError(f"CHART_TYPE_UNSUPPORTED:{self.chart_type}")
        y_columns = tuple(self.y_columns)
        sort_columns = tuple(self.sort_columns)
        if not y_columns or any(not isinstance(column, str) or not column for column in y_columns):
            raise ValueError("CHART_Y_COLUMNS_INVALID")
        if len(y_columns) != len(set(y_columns)) or self.x_column in y_columns:
            raise ValueError("CHART_COLUMNS_DUPLICATED")
        if any(not isinstance(column, str) or not column for column in sort_columns):
            raise ValueError("CHART_SORT_COLUMNS_INVALID")
        if len(sort_columns) != len(set(sort_columns)):
            raise ValueError("CHART_SORT_COLUMNS_DUPLICATED")
        if not isinstance(self.sort_descending, bool):
            raise TypeError("CHART_SORT_DESCENDING_NOT_BOOLEAN")
        object.__setattr__(self, "y_columns", y_columns)
        object.__setattr__(self, "sort_columns", sort_columns)
        object.__setattr__(self, "filters", _normalize_filters(self.filters))
        if canonical_json_hash(self._payload_without_hash()) != self.spec_hash:
            raise ValueError("CHART_SPEC_HASH_MISMATCH")

    @classmethod
    def from_table(
        cls,
        table: FrozenSourceTable,
        *,
        chart_id: str,
        chart_type: str,
        x_column: str,
        y_columns: Sequence[str],
        filters: Sequence[Mapping[str, object]] = (),
        sort_columns: Sequence[str] = (),
        sort_descending: bool = False,
    ) -> "ChartSpec":
        """从已冻结表生成 spec；任意 mapping 或未冻结 rows 都会被拒绝。"""

        if not isinstance(table, FrozenSourceTable):
            raise TypeError("CHART_REQUIRES_FROZEN_SOURCE_TABLE")
        normalized_filters = _normalize_filters(filters)
        payload = {
            "schema_version": "analysis-chart-spec-v1",
            "chart_id": chart_id,
            "source_name": table.name,
            "source_schema_version": table.schema_version,
            "source_hash": table.content_hash,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_columns": list(y_columns),
            "filters": _filter_payload(normalized_filters),
            "sort_columns": list(sort_columns),
            "sort_descending": sort_descending,
        }
        spec = cls(
            chart_id=chart_id,
            source_name=table.name,
            source_schema_version=table.schema_version,
            source_hash=table.content_hash,
            chart_type=chart_type,
            x_column=x_column,
            y_columns=tuple(y_columns),
            filters=normalized_filters,
            sort_columns=tuple(sort_columns),
            sort_descending=sort_descending,
            spec_hash=canonical_json_hash(payload),
        )
        _validate_spec_against_table(spec, table)
        # 在 spec 发布前执行一次逻辑物化，可尽早发现不可比较 filter/sort 值。
        spec.materialize(table)
        return spec

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "chart_id": self.chart_id,
            "source_name": self.source_name,
            "source_schema_version": self.source_schema_version,
            "source_hash": self.source_hash,
            "chart_type": self.chart_type,
            "x_column": self.x_column,
            "y_columns": list(self.y_columns),
            "filters": _filter_payload(self.filters),
            "sort_columns": list(self.sort_columns),
            "sort_descending": self.sort_descending,
        }

    def to_dict(self) -> dict[str, object]:
        value = self._payload_without_hash()
        value["spec_hash"] = self.spec_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ChartSpec":
        required = {
            "schema_version",
            "chart_id",
            "source_name",
            "source_schema_version",
            "source_hash",
            "chart_type",
            "x_column",
            "y_columns",
            "filters",
            "sort_columns",
            "sort_descending",
            "spec_hash",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("CHART_SPEC_FIELDS_MISMATCH")
        for field_name in (
            "schema_version",
            "chart_id",
            "source_name",
            "source_schema_version",
            "source_hash",
            "chart_type",
            "x_column",
            "spec_hash",
        ):
            if not isinstance(value[field_name], str):
                raise TypeError(f"CHART_SPEC_FIELD_NOT_STRING:{field_name}")
        y_columns = value["y_columns"]
        sort_columns = value["sort_columns"]
        filters = value["filters"]
        if not isinstance(y_columns, list) or not all(
            isinstance(column, str) for column in y_columns
        ):
            raise TypeError("CHART_SPEC_Y_COLUMNS_NOT_STRING_ARRAY")
        if not isinstance(sort_columns, list) or not all(
            isinstance(column, str) for column in sort_columns
        ):
            raise TypeError("CHART_SPEC_SORT_COLUMNS_NOT_STRING_ARRAY")
        if not isinstance(filters, list) or not all(
            isinstance(predicate, Mapping) for predicate in filters
        ):
            raise TypeError("CHART_SPEC_FILTERS_NOT_OBJECT_ARRAY")
        if not isinstance(value["sort_descending"], bool):
            raise TypeError("CHART_SPEC_SORT_DESCENDING_NOT_BOOLEAN")
        return cls(
            chart_id=value["chart_id"],
            source_name=value["source_name"],
            source_schema_version=value["source_schema_version"],
            source_hash=value["source_hash"],
            chart_type=value["chart_type"],
            x_column=value["x_column"],
            y_columns=tuple(y_columns),
            filters=tuple(filters),
            sort_columns=tuple(sort_columns),
            sort_descending=value["sort_descending"],
            spec_hash=value["spec_hash"],
            schema_version=value["schema_version"],
        )

    def materialize(
        self,
        table: FrozenSourceTable,
    ) -> tuple[Mapping[str, object], ...]:
        """按 spec 筛选、排序并投影列，返回不可变逻辑图表数据。"""

        source = _validate_spec_against_table(self, table)
        rows: list[Mapping[str, object]] = []
        for row in source.rows:
            include = all(
                _compare_filter(
                    row[str(predicate["column"])],
                    str(predicate["operator"]),
                    predicate["value"],
                )
                for predicate in self.filters
            )
            if include:
                rows.append(row)
        if self.sort_columns:
            rows = sorted(
                rows,
                key=lambda row: tuple(
                    _sort_token(row[column]) for column in self.sort_columns
                ),
                reverse=self.sort_descending,
            )
        projected: list[Mapping[str, object]] = []
        columns = (self.x_column, *self.y_columns)
        for row in rows:
            x_value = row[self.x_column]
            if not isinstance(x_value, (str, int, float)) or isinstance(x_value, bool):
                raise TypeError("CHART_X_VALUE_MUST_BE_STRING_OR_NUMBER")
            if isinstance(x_value, float) and not math.isfinite(x_value):
                raise ValueError("CHART_X_VALUE_NONFINITE")
            for column in self.y_columns:
                value = row[column]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"CHART_Y_VALUE_NOT_NUMERIC:{column}")
                if not math.isfinite(float(value)):
                    raise ValueError(f"CHART_Y_VALUE_NONFINITE:{column}")
            projected.append(
                MappingProxyType({column: _freeze_json(row[column]) for column in columns})
            )
        return tuple(projected)


def _reject_volatile_renderer_options(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.casefold() in _VOLATILE_OPTION_KEYS:
                raise ValueError(f"CHART_RENDER_OPTION_VOLATILE_FIELD:{path}.{key}")
            _reject_volatile_renderer_options(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_volatile_renderer_options(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and (
        PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    ):
        raise ValueError(f"CHART_RENDER_OPTION_ABSOLUTE_PATH:{path}")


@dataclass(frozen=True, slots=True)
class ChartArtifact:
    """canonical spec 或其渲染字节的路径无关身份。

    ``output_format='spec'`` 时不含渲染字节 hash；PNG artifact 则同时绑定 spec、
    renderer ID、稳定选项和实际文件 SHA-256。调用方可把同一 artifact 写到不同
    目录，路径不会改变 ``artifact_hash``。
    """

    spec: ChartSpec
    renderer_id: str
    output_format: str
    render_options: Mapping[str, object]
    content_sha256: str | None
    artifact_hash: str
    schema_version: str = "analysis-chart-artifact-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "analysis-chart-artifact-v1":
            raise ValueError("CHART_ARTIFACT_SCHEMA_MISMATCH")
        if not isinstance(self.spec, ChartSpec):
            raise TypeError("CHART_ARTIFACT_SPEC_REQUIRED")
        if not isinstance(self.renderer_id, str) or not self.renderer_id:
            raise ValueError("CHART_RENDERER_ID_INVALID")
        if self.output_format not in {"spec", "png"}:
            raise ValueError("CHART_OUTPUT_FORMAT_UNSUPPORTED")
        if self.output_format == "spec":
            if self.content_sha256 is not None or self.renderer_id != "canonical-spec-only":
                raise ValueError("CHART_SPEC_ONLY_ARTIFACT_FIELDS_INVALID")
        elif not _valid_sha256(self.content_sha256):
            raise ValueError("CHART_CONTENT_HASH_INVALID")
        if not _valid_sha256(self.artifact_hash):
            raise ValueError("CHART_ARTIFACT_HASH_INVALID")
        options = _json_value(self.render_options, field_path="$.render_options")
        assert isinstance(options, dict)
        _reject_volatile_renderer_options(options)
        canonical_json_bytes(options)
        object.__setattr__(self, "render_options", _freeze_json(options))
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("CHART_ARTIFACT_HASH_MISMATCH")

    @classmethod
    def from_spec(cls, spec: ChartSpec) -> "ChartArtifact":
        """发布不依赖绘图库的 canonical spec artifact。"""

        if not isinstance(spec, ChartSpec):
            raise TypeError("CHART_ARTIFACT_SPEC_REQUIRED")
        payload = {
            "schema_version": "analysis-chart-artifact-v1",
            "spec": spec.to_dict(),
            "renderer_id": "canonical-spec-only",
            "output_format": "spec",
            "render_options": {},
            "content_sha256": None,
        }
        return cls(
            spec=spec,
            renderer_id="canonical-spec-only",
            output_format="spec",
            render_options={},
            content_sha256=None,
            artifact_hash=canonical_json_hash(payload),
        )

    @classmethod
    def from_rendered_bytes(
        cls,
        spec: ChartSpec,
        payload: bytes,
        *,
        renderer_id: str,
        output_format: str,
        render_options: Mapping[str, object],
    ) -> "ChartArtifact":
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("CHART_RENDERED_BYTES_EMPTY")
        content_sha256 = hashlib.sha256(payload).hexdigest()
        base = {
            "schema_version": "analysis-chart-artifact-v1",
            "spec": spec.to_dict(),
            "renderer_id": renderer_id,
            "output_format": output_format,
            "render_options": _json_value(render_options),
            "content_sha256": content_sha256,
        }
        return cls(
            spec=spec,
            renderer_id=renderer_id,
            output_format=output_format,
            render_options=render_options,
            content_sha256=content_sha256,
            artifact_hash=canonical_json_hash(base),
        )

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "renderer_id": self.renderer_id,
            "output_format": self.output_format,
            "render_options": _json_value(self.render_options),
            "content_sha256": self.content_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        value = self._payload_without_hash()
        value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ChartArtifact":
        required = {
            "schema_version",
            "spec",
            "renderer_id",
            "output_format",
            "render_options",
            "content_sha256",
            "artifact_hash",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("CHART_ARTIFACT_FIELDS_MISMATCH")
        spec_value = value["spec"]
        if not isinstance(spec_value, Mapping):
            raise TypeError("CHART_ARTIFACT_SPEC_NOT_OBJECT")
        for field_name in (
            "schema_version",
            "renderer_id",
            "output_format",
            "artifact_hash",
        ):
            if not isinstance(value[field_name], str):
                raise TypeError(f"CHART_ARTIFACT_FIELD_NOT_STRING:{field_name}")
        if value["content_sha256"] is not None and not isinstance(
            value["content_sha256"], str
        ):
            raise TypeError("CHART_ARTIFACT_CONTENT_HASH_NOT_STRING_OR_NULL")
        if not isinstance(value["render_options"], Mapping):
            raise TypeError("CHART_ARTIFACT_RENDER_OPTIONS_NOT_OBJECT")
        return cls(
            spec=ChartSpec.from_mapping(spec_value),
            renderer_id=value["renderer_id"],
            output_format=value["output_format"],
            render_options=value["render_options"],
            content_sha256=value["content_sha256"],
            artifact_hash=value["artifact_hash"],
            schema_version=value["schema_version"],
        )


def render_matplotlib_chart(
    spec: ChartSpec,
    table: FrozenSourceTable,
    output_path: str | Path,
    *,
    dpi: int = 120,
) -> ChartArtifact:
    """用延迟导入的 Matplotlib/Agg 渲染确定性 PNG。

    ``output_path`` 只决定字节发布位置，从不写入返回 artifact。renderer 固定
    figure 尺寸、DPI、字体和 PNG metadata，并通过内存缓冲区保存，避免文件名
    被绘图库嵌入产物。不同 Matplotlib 版本仍可能产生不同字节，因此版本号属于
    ``renderer_id``，而不是被误认为同一 renderer。
    """

    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("CHART_RENDER_DPI_INVALID")
    rows = spec.materialize(table)
    if not rows:
        raise ValueError("CHART_RENDER_NO_ROWS_AFTER_FILTER")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - 取决于本机可选 analysis extra
        raise DependencyUnavailable(
            "matplotlib",
            feature="stage9_chart_renderer",
            install_extra="analysis",
        ) from exc

    options = {"dpi": dpi, "figsize_inches": [6.4, 4.0], "backend": "Agg"}
    rc = {
        "font.family": "DejaVu Sans",
        "figure.figsize": options["figsize_inches"],
        "axes.grid": True,
        "axes.unicode_minus": False,
        "svg.hashsalt": spec.spec_hash,
    }
    with matplotlib.rc_context(rc):
        figure, axis = plt.subplots()
        x_values = [row[spec.x_column] for row in rows]
        if spec.chart_type == "bar":
            positions = list(range(len(rows)))
            width = 0.8 / len(spec.y_columns)
            for series_index, column in enumerate(spec.y_columns):
                offset = (series_index - (len(spec.y_columns) - 1) / 2.0) * width
                axis.bar(
                    [position + offset for position in positions],
                    [float(row[column]) for row in rows],
                    width=width,
                    label=column,
                )
            axis.set_xticks(positions, [str(value) for value in x_values])
        else:
            for column in spec.y_columns:
                y_values = [float(row[column]) for row in rows]
                if spec.chart_type == "line":
                    axis.plot(x_values, y_values, marker="o", label=column)
                else:
                    axis.scatter(x_values, y_values, label=column)
        axis.set_xlabel(spec.x_column)
        axis.set_ylabel(", ".join(spec.y_columns))
        axis.set_title(spec.chart_id)
        if len(spec.y_columns) > 1:
            axis.legend()
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=dpi,
            metadata={"Software": "param-importance-nlp"},
        )
        plt.close(figure)
    payload = buffer.getvalue()
    atomic_write_bytes(output_path, payload)
    return ChartArtifact.from_rendered_bytes(
        spec,
        payload,
        renderer_id=f"matplotlib:{matplotlib.__version__}:Agg",
        output_format="png",
        render_options=options,
    )
