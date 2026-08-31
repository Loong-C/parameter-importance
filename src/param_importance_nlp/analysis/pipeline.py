"""Stage 9 跨阶段 ETL、统计、图表、表格与 AnalysisBundle。

本模块接收的最小上游单位是 :class:`StageArtifactRows`：它既绑定原始 artifact
hash，也绑定确定版本的 row adapter。多个 Stage 4--8 输入合并时会自动补充来源
列，随后发布为 :class:`BoundSourceTable`。所有统计、表格和 heatmap 都继续引用该
source artifact hash，禁止从无来源的 Python list 直接生成论文数字。

配对统计以显式 pair ID 为统计单位，要求 baseline/candidate 的 pair 集完全一致；
缺失配对不是“自动丢行”，而是合同错误。多重比较使用 Holm step-down 调整，
不添加 epsilon。formal bundle 必须拥有独立授权 hash，且每个输入表都必须由 formal
rows 产生；local fixture 无法在 bundle 层升级资格。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
import hashlib
import io
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..atomic import atomic_write_bytes
from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from ..contracts.jsonio import canonical_json_bytes, canonical_json_hash, write_canonical_json
from .charts import ChartArtifact
from .metrics import mean_confidence_interval
from .report import AnalysisReport, FrozenSourceTable


def _require_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} 必须是小写 SHA-256")
    return value


def _json_sort_key(value: object) -> bytes:
    return canonical_json_bytes(thaw_json_value(value))


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数")
    return number


def _require_exact_fields(
    value: Mapping[str, object],
    required: set[str],
    *,
    artifact_name: str,
) -> None:
    """拒绝缺失字段与未知字段。

    artifact loader 不做“最佳努力”升级；schema 迁移必须由显式 adapter
    完成，否则旧结果可能在无审计记录的情况下改变语义。
    """

    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{artifact_name} fields mismatch")


@dataclass(frozen=True, slots=True)
class StageArtifactRows:
    """版本化 adapter 从一个上游 artifact 提取的冻结行集合。"""

    artifact_id: str
    source_schema_version: str
    source_artifact_hash: str
    stage: int
    adapter_id: str
    rows: tuple[Mapping[str, object], ...]
    scope: str
    formal_eligible: bool
    artifact_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "stage-artifact-rows-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "stage-artifact-rows-v1":
            raise ValueError("StageArtifactRows schema_version 不受支持")
        for name in ("artifact_id", "source_schema_version", "adapter_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} 不能为空")
        _require_hash(self.source_artifact_hash, field_name="source_artifact_hash")
        _require_hash(self.artifact_hash, field_name="artifact_hash")
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= 9:
            raise ValueError("stage 必须位于 0..9")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("rows scope 只能是 local_fixture/formal")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("rows formal_eligible 与 scope 不一致")
        if not self.rows:
            raise ValueError("StageArtifactRows rows 不能为空")
        normalized_rows: list[Mapping[str, object]] = []
        for index, row in enumerate(self.rows):
            if not isinstance(row, Mapping) or not row:
                raise TypeError(f"row {index} 必须是非空 object")
            normalized_rows.append(freeze_json_mapping(row))
        object.__setattr__(self, "rows", tuple(normalized_rows))
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("StageArtifactRows artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "source_schema_version": self.source_schema_version,
            "source_artifact_hash": self.source_artifact_hash,
            "stage": self.stage,
            "adapter_id": self.adapter_id,
            "rows": [thaw_json_value(row) for row in self.rows],
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "metadata": thaw_json_value(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        source_schema_version: str,
        source_artifact_hash: str,
        stage: int,
        adapter_id: str,
        rows: Sequence[Mapping[str, object]],
        scope: str,
        formal_eligible: bool,
        metadata: Mapping[str, object] | None = None,
    ) -> "StageArtifactRows":
        values = {
            "schema_version": "stage-artifact-rows-v1",
            "artifact_id": artifact_id,
            "source_schema_version": source_schema_version,
            "source_artifact_hash": source_artifact_hash,
            "stage": stage,
            "adapter_id": adapter_id,
            "rows": [dict(row) for row in rows],
            "scope": scope,
            "formal_eligible": formal_eligible,
            "metadata": dict(metadata or {}),
        }
        return cls(
            artifact_id=artifact_id,
            source_schema_version=source_schema_version,
            source_artifact_hash=source_artifact_hash,
            stage=stage,
            adapter_id=adapter_id,
            rows=tuple(rows),
            scope=scope,
            formal_eligible=formal_eligible,
            metadata=metadata or {},
            artifact_hash=canonical_json_hash(values),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "StageArtifactRows":
        """严格加载已发布的 Stage rows，并重算自身 hash。"""

        required = {
            "schema_version",
            "artifact_id",
            "source_schema_version",
            "source_artifact_hash",
            "stage",
            "adapter_id",
            "rows",
            "scope",
            "formal_eligible",
            "metadata",
            "artifact_hash",
        }
        _require_exact_fields(value, required, artifact_name="StageArtifactRows")
        rows = value["rows"]
        metadata = value["metadata"]
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise TypeError("StageArtifactRows rows 必须是 object array")
        if not isinstance(metadata, Mapping):
            raise TypeError("StageArtifactRows metadata 必须是 object")
        return cls(
            artifact_id=value["artifact_id"],  # type: ignore[arg-type]
            source_schema_version=value["source_schema_version"],  # type: ignore[arg-type]
            source_artifact_hash=value["source_artifact_hash"],  # type: ignore[arg-type]
            stage=value["stage"],  # type: ignore[arg-type]
            adapter_id=value["adapter_id"],  # type: ignore[arg-type]
            rows=tuple(rows),
            scope=value["scope"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            metadata=metadata,
            artifact_hash=value["artifact_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class BoundSourceTable:
    """带 scope、父 artifact 和自身 hash 的 :class:`FrozenSourceTable`。"""

    table: FrozenSourceTable
    role: str
    scope: str
    formal_eligible: bool
    parent_artifact_hashes: tuple[str, ...]
    derivation_id: str
    artifact_hash: str
    schema_version: str = "bound-frozen-source-table-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "bound-frozen-source-table-v1":
            raise ValueError("BoundSourceTable schema_version 不受支持")
        if not isinstance(self.table, FrozenSourceTable):
            raise TypeError("BoundSourceTable.table 必须是 FrozenSourceTable")
        if not self.role or not self.derivation_id:
            raise ValueError("table role/derivation_id 不能为空")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("table formal_eligible 与 scope 不一致")
        parents = tuple(self.parent_artifact_hashes)
        if not parents or len(parents) != len(set(parents)):
            raise ValueError("parent artifact hashes 必须非空且唯一")
        for value in (*parents, self.artifact_hash):
            _require_hash(value, field_name="table artifact hash")
        object.__setattr__(self, "parent_artifact_hashes", parents)
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("BoundSourceTable artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "table": self.table.to_dict(),
            "role": self.role,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "parent_artifact_hashes": list(self.parent_artifact_hashes),
            "derivation_id": self.derivation_id,
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def create(
        cls,
        table: FrozenSourceTable,
        *,
        role: str,
        scope: str,
        parent_artifact_hashes: Sequence[str],
        derivation_id: str,
    ) -> "BoundSourceTable":
        payload = {
            "schema_version": "bound-frozen-source-table-v1",
            "table": table.to_dict(),
            "role": role,
            "scope": scope,
            "formal_eligible": scope == "formal",
            "parent_artifact_hashes": list(parent_artifact_hashes),
            "derivation_id": derivation_id,
        }
        return cls(
            table=table,
            role=role,
            scope=scope,
            formal_eligible=scope == "formal",
            parent_artifact_hashes=tuple(parent_artifact_hashes),
            derivation_id=derivation_id,
            artifact_hash=canonical_json_hash(payload),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "BoundSourceTable":
        """严格加载带 lineage 的冻结源表。"""

        required = {
            "schema_version",
            "table",
            "role",
            "scope",
            "formal_eligible",
            "parent_artifact_hashes",
            "derivation_id",
            "artifact_hash",
        }
        _require_exact_fields(value, required, artifact_name="BoundSourceTable")
        table = value["table"]
        parents = value["parent_artifact_hashes"]
        if not isinstance(table, Mapping):
            raise TypeError("BoundSourceTable table 必须是 object")
        if not isinstance(parents, list):
            raise TypeError("parent_artifact_hashes 必须是 array")
        return cls(
            table=FrozenSourceTable.from_mapping(table),
            role=value["role"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            parent_artifact_hashes=tuple(parents),  # type: ignore[arg-type]
            derivation_id=value["derivation_id"],  # type: ignore[arg-type]
            artifact_hash=value["artifact_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


class CrossStageSourceBuilder:
    """把若干经过验证的 Stage rows 合并为一张来源可追溯的冻结表。"""

    _RESERVED = frozenset(
        {
            "source_stage",
            "source_artifact_id",
            "source_artifact_hash",
            "source_schema_version",
            "source_adapter_id",
            "source_row_index",
        }
    )

    def __init__(
        self,
        *,
        scope: str,
        formal_authorization_hash: str | None = None,
    ) -> None:
        if scope not in {"local_fixture", "formal"}:
            raise ValueError("builder scope 只能是 local_fixture/formal")
        if scope == "formal":
            _require_hash(formal_authorization_hash, field_name="formal_authorization_hash")
        elif formal_authorization_hash is not None:
            raise ValueError("fixture source builder 不得携带 formal authorization")
        self.scope = scope
        self.formal_authorization_hash = formal_authorization_hash
        self._artifacts: dict[str, StageArtifactRows] = {}

    def add(self, artifact: StageArtifactRows) -> None:
        if artifact.scope != self.scope or artifact.formal_eligible is not (self.scope == "formal"):
            raise ValueError("StageArtifactRows scope 与 builder 不一致")
        if any(self._RESERVED.intersection(row) for row in artifact.rows):
            raise ValueError("upstream rows 伪造了 ETL reserved source columns")
        existing = self._artifacts.get(artifact.artifact_id)
        if existing is not None and existing.artifact_hash != artifact.artifact_hash:
            raise ValueError("同名 StageArtifactRows hash 冲突")
        self._artifacts[artifact.artifact_id] = artifact

    def build(self, *, name: str, schema_version: str) -> BoundSourceTable:
        if not self._artifacts:
            raise ValueError("CrossStageSourceBuilder 至少需要一个 artifact")
        rows: list[Mapping[str, object]] = []
        ordered = sorted(
            self._artifacts.values(),
            key=lambda item: (item.stage, item.artifact_id, item.artifact_hash),
        )
        for artifact in ordered:
            for row_index, row in enumerate(artifact.rows):
                rows.append(
                    {
                        **thaw_json_value(row),
                        "source_stage": artifact.stage,
                        "source_artifact_id": artifact.artifact_id,
                        "source_artifact_hash": artifact.source_artifact_hash,
                        "source_schema_version": artifact.source_schema_version,
                        "source_adapter_id": artifact.adapter_id,
                        "source_row_index": row_index,
                    }
                )
        table = FrozenSourceTable.from_rows(
            name=name,
            schema_version=schema_version,
            rows=rows,
        )
        return BoundSourceTable.create(
            table,
            role="cross_stage_source",
            scope=self.scope,
            parent_artifact_hashes=tuple(item.artifact_hash for item in ordered),
            derivation_id="stage9.cross_stage_etl.v1",
        )


def _group_rows(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> list[tuple[tuple[object, ...], list[Mapping[str, object]]]]:
    groups: dict[bytes, tuple[tuple[object, ...], list[Mapping[str, object]]]] = {}
    for row in rows:
        try:
            key = tuple(row[column] for column in columns)
        except KeyError as exc:
            raise ValueError(f"group column 不存在: {exc.args[0]}") from exc
        encoded = canonical_json_bytes([thaw_json_value(item) for item in key])
        groups.setdefault(encoded, (key, []))[1].append(row)
    return [groups[key] for key in sorted(groups)]


def grouped_statistics(
    source: BoundSourceTable,
    *,
    group_columns: Sequence[str],
    value_column: str,
    confidence: float = 0.95,
    output_name: str = "grouped_statistics",
) -> BoundSourceTable:
    """按明确统计单位计算 n/mean/sample-std/Student-t CI。"""

    if not value_column:
        raise ValueError("value_column 不能为空")
    columns = tuple(group_columns)
    if len(columns) != len(set(columns)) or value_column in columns:
        raise ValueError("group columns 重复或包含 value column")
    result_rows: list[Mapping[str, object]] = []
    for key, rows in _group_rows(source.table.rows, columns):
        values = [
            _finite_number(row.get(value_column), field_name=value_column) for row in rows
        ]
        mean_result, lower_result, upper_result = mean_confidence_interval(
            values, confidence=confidence
        )
        row: dict[str, object] = dict(zip(columns, key, strict=True))
        row.update(
            {
                "value_column": value_column,
                "n": len(values),
                "mean": float(np.mean(values)),
                "sample_std": (
                    None if len(values) < 2 else float(np.std(values, ddof=1))
                ),
                "ci_defined": mean_result.defined,
                "ci_reason": mean_result.reason,
                "confidence": confidence,
                "ci_lower": lower_result.value,
                "ci_upper": upper_result.value,
            }
        )
        result_rows.append(row)
    table = FrozenSourceTable.from_rows(
        name=output_name,
        schema_version="grouped-statistics-v1",
        rows=result_rows,
    )
    return BoundSourceTable.create(
        table,
        role="derived_grouped_statistics",
        scope=source.scope,
        parent_artifact_hashes=(source.artifact_hash,),
        derivation_id="stage9.grouped_student_t.v1",
    )


def _holm_adjust(p_values: Sequence[tuple[bytes, float]]) -> dict[bytes, float]:
    ordered = sorted(p_values, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[bytes, float] = {}
    running = 0.0
    for index, (identity, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)
        adjusted[identity] = running
    return adjusted


def paired_statistics_with_holm(
    source: BoundSourceTable,
    *,
    group_columns: Sequence[str],
    pair_id_column: str,
    condition_column: str,
    value_column: str,
    baseline_condition: object,
    confidence: float = 0.95,
    familywise_alpha: float = 0.05,
    output_name: str = "paired_comparisons",
) -> BoundSourceTable:
    """执行严格配对 Student-t 比较并在全部比较上应用 Holm 校正。"""

    if not 0 < familywise_alpha < 1:
        raise ValueError("familywise_alpha 必须位于 (0,1)")
    groups = tuple(group_columns)
    forbidden = {pair_id_column, condition_column, value_column}
    if not all((pair_id_column, condition_column, value_column)):
        raise ValueError("pair/condition/value column 不能为空")
    if len(groups) != len(set(groups)) or set(groups).intersection(forbidden):
        raise ValueError("group columns 与统计列重叠或重复")
    pending: list[dict[str, object]] = []
    identities: list[bytes] = []
    p_values: list[tuple[bytes, float]] = []
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:  # pragma: no cover - analysis profile 固定包含 SciPy
        raise RuntimeError("paired_statistics_with_holm 需要 scipy") from exc

    for group_key, group_rows in _group_rows(source.table.rows, groups):
        by_condition: dict[bytes, tuple[object, dict[bytes, float]]] = {}
        for row in group_rows:
            for column in (pair_id_column, condition_column, value_column):
                if column not in row:
                    raise ValueError(f"paired statistics 缺列: {column}")
            condition = row[condition_column]
            condition_key = _json_sort_key(condition)
            pair_key = _json_sort_key(row[pair_id_column])
            slot = by_condition.setdefault(condition_key, (condition, {}))[1]
            if pair_key in slot:
                raise ValueError("同一 group/condition/pair ID 出现重复观测")
            slot[pair_key] = _finite_number(row[value_column], field_name=value_column)
        baseline_key = _json_sort_key(baseline_condition)
        if baseline_key not in by_condition:
            raise ValueError("某个 group 缺少 baseline condition")
        baseline_pairs = by_condition[baseline_key][1]
        for condition_key in sorted(by_condition):
            if condition_key == baseline_key:
                continue
            condition, candidate_pairs = by_condition[condition_key]
            if set(candidate_pairs) != set(baseline_pairs):
                raise ValueError("paired comparison 的 baseline/candidate pair 集不一致")
            pair_order = sorted(baseline_pairs)
            differences = np.asarray(
                [candidate_pairs[key] - baseline_pairs[key] for key in pair_order],
                dtype=np.float64,
            )
            n = int(differences.size)
            mean_difference = float(np.mean(differences))
            std_difference = None if n < 2 else float(np.std(differences, ddof=1))
            comparison_defined = n >= 2
            reason = None if comparison_defined else "INSUFFICIENT_PAIRS"
            t_statistic: float | None = None
            p_value: float | None = None
            ci_lower: float | None = None
            ci_upper: float | None = None
            if comparison_defined:
                assert std_difference is not None
                _mean, lower, upper = mean_confidence_interval(
                    differences, confidence=confidence
                )
                ci_lower, ci_upper = lower.value, upper.value
                if std_difference == 0.0:
                    if mean_difference == 0.0:
                        t_statistic, p_value = 0.0, 1.0
                    else:
                        # 数学上的 |t|=inf 不能进入 canonical JSON；p=0 是其极限值。
                        t_statistic, p_value = None, 0.0
                        reason = "ZERO_VARIANCE_NONZERO_MEAN"
                else:
                    t_statistic = mean_difference / (std_difference / math.sqrt(n))
                    p_value = float(2.0 * student_t.sf(abs(t_statistic), n - 1))
            identity = canonical_json_bytes(
                [thaw_json_value(item) for item in (*group_key, condition)]
            )
            row = dict(zip(groups, group_key, strict=True))
            row.update(
                {
                    "baseline_condition": thaw_json_value(baseline_condition),
                    "candidate_condition": thaw_json_value(condition),
                    "pair_id_column": pair_id_column,
                    "value_column": value_column,
                    "n_pairs": n,
                    "mean_difference": mean_difference,
                    "sample_std_difference": std_difference,
                    "confidence": confidence,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "comparison_defined": comparison_defined,
                    "reason": reason,
                    "t_statistic": t_statistic,
                    "degrees_of_freedom": None if n < 2 else n - 1,
                    "p_value": p_value,
                    "p_value_holm": None,
                    "familywise_alpha": familywise_alpha,
                    "reject_h0_holm": None,
                }
            )
            identities.append(identity)
            pending.append(row)
            if p_value is not None:
                p_values.append((identity, p_value))
    adjusted = _holm_adjust(p_values)
    for identity, row in zip(identities, pending, strict=True):
        if identity in adjusted:
            row["p_value_holm"] = adjusted[identity]
            row["reject_h0_holm"] = adjusted[identity] <= familywise_alpha
    if not pending:
        raise ValueError("paired statistics 没有 candidate comparison")
    table = FrozenSourceTable.from_rows(
        name=output_name,
        schema_version="paired-statistics-holm-v1",
        rows=pending,
    )
    return BoundSourceTable.create(
        table,
        role="derived_paired_statistics",
        scope=source.scope,
        parent_artifact_hashes=(source.artifact_hash,),
        derivation_id="stage9.paired_student_t_holm.v1",
    )


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("table cell 包含非有限数")
        return f"{value:.12g}"
    if isinstance(value, (int, str)):
        return str(value)
    raise TypeError("table renderer 只接受标量 JSON cell")


def _markdown_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


@dataclass(frozen=True, slots=True)
class TableSpec:
    table_artifact_hash: str
    columns: tuple[str, ...]
    formats: tuple[str, ...] = ("csv", "markdown", "latex")
    caption: str = ""
    spec_hash: str = ""
    schema_version: str = "analysis-table-spec-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "analysis-table-spec-v1":
            raise ValueError("TableSpec schema_version 不受支持")
        _require_hash(self.table_artifact_hash, field_name="table_artifact_hash")
        if not self.columns or len(self.columns) != len(set(self.columns)):
            raise ValueError("table columns 必须非空且唯一")
        if not self.formats or set(self.formats) - {"csv", "markdown", "latex"}:
            raise ValueError("table formats 只能是 csv/markdown/latex")
        if len(self.formats) != len(set(self.formats)):
            raise ValueError("table formats 不能重复")
        expected = canonical_json_hash(self._payload_without_hash())
        if self.spec_hash:
            if self.spec_hash != expected:
                raise ValueError("TableSpec spec_hash 不匹配")
        else:
            object.__setattr__(self, "spec_hash", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "table_artifact_hash": self.table_artifact_hash,
            "columns": list(self.columns),
            "formats": list(self.formats),
            "caption": self.caption,
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"spec_hash": self.spec_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TableSpec":
        """严格加载表格渲染规格。"""

        required = {
            "schema_version",
            "table_artifact_hash",
            "columns",
            "formats",
            "caption",
            "spec_hash",
        }
        _require_exact_fields(value, required, artifact_name="TableSpec")
        columns, formats = value["columns"], value["formats"]
        if not isinstance(columns, list) or not isinstance(formats, list):
            raise TypeError("TableSpec columns/formats 必须是 array")
        return cls(
            table_artifact_hash=value["table_artifact_hash"],  # type: ignore[arg-type]
            columns=tuple(columns),  # type: ignore[arg-type]
            formats=tuple(formats),  # type: ignore[arg-type]
            caption=value["caption"],  # type: ignore[arg-type]
            spec_hash=value["spec_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TableArtifact:
    spec: TableSpec
    contents: Mapping[str, str]
    content_hashes: Mapping[str, str]
    artifact_hash: str
    schema_version: str = "analysis-table-artifact-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "analysis-table-artifact-v1":
            raise ValueError("TableArtifact schema_version 不受支持")
        if set(self.contents) != set(self.spec.formats) or set(self.content_hashes) != set(
            self.spec.formats
        ):
            raise ValueError("table contents/hashes 未精确覆盖 formats")
        for name, content in self.contents.items():
            if not isinstance(content, str):
                raise TypeError("table content 必须是 UTF-8 字符串")
            expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if self.content_hashes[name] != expected:
                raise ValueError("table content hash 不匹配")
        object.__setattr__(self, "contents", MappingProxyType(dict(self.contents)))
        object.__setattr__(self, "content_hashes", MappingProxyType(dict(self.content_hashes)))
        _require_hash(self.artifact_hash, field_name="table artifact_hash")
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("TableArtifact artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "contents": dict(self.contents),
            "content_hashes": dict(self.content_hashes),
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TableArtifact":
        """加载表格 artifact，重新校验每种文本的 UTF-8 hash。"""

        required = {
            "schema_version",
            "spec",
            "contents",
            "content_hashes",
            "artifact_hash",
        }
        _require_exact_fields(value, required, artifact_name="TableArtifact")
        spec, contents, hashes = value["spec"], value["contents"], value["content_hashes"]
        if not all(isinstance(item, Mapping) for item in (spec, contents, hashes)):
            raise TypeError("TableArtifact spec/contents/content_hashes 必须是 object")
        return cls(
            spec=TableSpec.from_mapping(spec),
            contents=contents,  # type: ignore[arg-type]
            content_hashes=hashes,  # type: ignore[arg-type]
            artifact_hash=value["artifact_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )

    def publish(self, root: str | Path, *, stem: str) -> tuple[Path, ...]:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        suffixes = {"csv": ".csv", "markdown": ".md", "latex": ".tex"}
        paths: list[Path] = []
        for name in self.spec.formats:
            path = directory / f"{stem}{suffixes[name]}"
            atomic_write_bytes(path, self.contents[name].encode("utf-8"))
            paths.append(path)
        write_canonical_json(directory / f"{stem}.table.json", self.to_dict())
        return tuple(paths)


def render_table(source: BoundSourceTable, spec: TableSpec) -> TableArtifact:
    """从 hash 绑定表确定性生成 CSV、Markdown 与 LaTeX。"""

    if spec.table_artifact_hash != source.artifact_hash:
        raise ValueError("TableSpec 未绑定当前 BoundSourceTable")
    missing = set(spec.columns) - set(source.table.columns)
    if missing:
        raise ValueError(f"table columns 不完整: {sorted(missing)}")
    rows = [[_format_cell(row[column]) for column in spec.columns] for row in source.table.rows]
    contents: dict[str, str] = {}
    if "csv" in spec.formats:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(spec.columns)
        writer.writerows(rows)
        contents["csv"] = buffer.getvalue()
    if "markdown" in spec.formats:
        header = "| " + " | ".join(_markdown_escape(item) for item in spec.columns) + " |"
        separator = "| " + " | ".join("---" for _ in spec.columns) + " |"
        body = [
            "| " + " | ".join(_markdown_escape(item) for item in row) + " |" for row in rows
        ]
        prefix = [] if not spec.caption else [f"### {spec.caption}", ""]
        contents["markdown"] = "\n".join([*prefix, header, separator, *body, ""])
    if "latex" in spec.formats:
        alignment = "l" * len(spec.columns)
        lines = [f"\\begin{{tabular}}{{{alignment}}}", r"\toprule"]
        lines.append(" & ".join(_latex_escape(item) for item in spec.columns) + r" \\")
        lines.append(r"\midrule")
        lines.extend(" & ".join(_latex_escape(item) for item in row) + r" \\" for row in rows)
        lines.extend([r"\bottomrule", r"\end{tabular}", ""])
        contents["latex"] = "\n".join(lines)
    hashes = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in contents.items()
    }
    payload = {
        "schema_version": "analysis-table-artifact-v1",
        "spec": spec.to_dict(),
        "contents": contents,
        "content_hashes": hashes,
    }
    return TableArtifact(spec, contents, hashes, canonical_json_hash(payload))


@dataclass(frozen=True, slots=True)
class HeatmapSpec:
    table_artifact_hash: str
    source_content_hash: str
    heatmap_id: str
    x_column: str
    y_column: str
    value_column: str
    spec_hash: str = ""
    schema_version: str = "analysis-heatmap-spec-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "analysis-heatmap-spec-v1":
            raise ValueError("HeatmapSpec schema_version 不受支持")
        for value in (self.table_artifact_hash, self.source_content_hash):
            _require_hash(value, field_name="heatmap source hash")
        if not all((self.heatmap_id, self.x_column, self.y_column, self.value_column)):
            raise ValueError("heatmap identity/columns 不能为空")
        if len({self.x_column, self.y_column, self.value_column}) != 3:
            raise ValueError("heatmap x/y/value columns 必须不同")
        expected = canonical_json_hash(self._payload_without_hash())
        if self.spec_hash and self.spec_hash != expected:
            raise ValueError("HeatmapSpec spec_hash 不匹配")
        if not self.spec_hash:
            object.__setattr__(self, "spec_hash", expected)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "table_artifact_hash": self.table_artifact_hash,
            "source_content_hash": self.source_content_hash,
            "heatmap_id": self.heatmap_id,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "value_column": self.value_column,
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"spec_hash": self.spec_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HeatmapSpec":
        """严格加载 heatmap 规格。"""

        required = {
            "schema_version",
            "table_artifact_hash",
            "source_content_hash",
            "heatmap_id",
            "x_column",
            "y_column",
            "value_column",
            "spec_hash",
        }
        _require_exact_fields(value, required, artifact_name="HeatmapSpec")
        return cls(
            table_artifact_hash=value["table_artifact_hash"],  # type: ignore[arg-type]
            source_content_hash=value["source_content_hash"],  # type: ignore[arg-type]
            heatmap_id=value["heatmap_id"],  # type: ignore[arg-type]
            x_column=value["x_column"],  # type: ignore[arg-type]
            y_column=value["y_column"],  # type: ignore[arg-type]
            value_column=value["value_column"],  # type: ignore[arg-type]
            spec_hash=value["spec_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class HeatmapArtifact:
    spec: HeatmapSpec
    renderer_id: str
    content_sha256: str
    render_options: Mapping[str, object]
    artifact_hash: str
    schema_version: str = "analysis-heatmap-artifact-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "analysis-heatmap-artifact-v1":
            raise ValueError("HeatmapArtifact schema_version 不受支持")
        if not self.renderer_id:
            raise ValueError("heatmap renderer_id 不能为空")
        for value in (self.content_sha256, self.artifact_hash):
            _require_hash(value, field_name="heatmap artifact hash")
        object.__setattr__(self, "render_options", freeze_json_mapping(self.render_options))
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("HeatmapArtifact artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "renderer_id": self.renderer_id,
            "content_sha256": self.content_sha256,
            "render_options": thaw_json_value(self.render_options),
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HeatmapArtifact":
        """严格加载 heatmap 产物元数据。"""

        required = {
            "schema_version",
            "spec",
            "renderer_id",
            "content_sha256",
            "render_options",
            "artifact_hash",
        }
        _require_exact_fields(value, required, artifact_name="HeatmapArtifact")
        spec, options = value["spec"], value["render_options"]
        if not isinstance(spec, Mapping) or not isinstance(options, Mapping):
            raise TypeError("HeatmapArtifact spec/render_options 必须是 object")
        return cls(
            spec=HeatmapSpec.from_mapping(spec),
            renderer_id=value["renderer_id"],  # type: ignore[arg-type]
            content_sha256=value["content_sha256"],  # type: ignore[arg-type]
            render_options=options,
            artifact_hash=value["artifact_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


def render_heatmap(
    source: BoundSourceTable,
    spec: HeatmapSpec,
    output_path: str | Path,
    *,
    dpi: int = 120,
) -> HeatmapArtifact:
    """渲染完整矩形的确定性 heatmap；重复或缺失 cell 一律拒绝。"""

    if spec.table_artifact_hash != source.artifact_hash or spec.source_content_hash != source.table.content_hash:
        raise ValueError("HeatmapSpec 未绑定当前 source table")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("heatmap dpi 必须是正整数")
    for column in (spec.x_column, spec.y_column, spec.value_column):
        if column not in source.table.columns:
            raise ValueError(f"heatmap column 不存在: {column}")
    x_values = sorted({row[spec.x_column] for row in source.table.rows}, key=_json_sort_key)
    y_values = sorted({row[spec.y_column] for row in source.table.rows}, key=_json_sort_key)
    cells: dict[tuple[bytes, bytes], float] = {}
    for row in source.table.rows:
        key = (_json_sort_key(row[spec.x_column]), _json_sort_key(row[spec.y_column]))
        if key in cells:
            raise ValueError("heatmap 存在重复 x/y cell")
        cells[key] = _finite_number(row[spec.value_column], field_name=spec.value_column)
    if len(cells) != len(x_values) * len(y_values):
        raise ValueError("heatmap 必须是完整矩形，不能静默填补缺失 cell")
    matrix = np.asarray(
        [
            [cells[(_json_sort_key(x), _json_sort_key(y))] for x in x_values]
            for y in y_values
        ],
        dtype=np.float64,
    )
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("render_heatmap 需要 matplotlib analysis extra") from exc
    options = {"dpi": dpi, "figsize_inches": [6.4, 4.8], "backend": "Agg"}
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "figure.figsize": options["figsize_inches"],
            "axes.unicode_minus": False,
        }
    ):
        figure, axis = plt.subplots()
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
        axis.set_xticks(range(len(x_values)), [str(value) for value in x_values])
        axis.set_yticks(range(len(y_values)), [str(value) for value in y_values])
        axis.set_xlabel(spec.x_column)
        axis.set_ylabel(spec.y_column)
        axis.set_title(spec.heatmap_id)
        figure.colorbar(image, ax=axis, label=spec.value_column)
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=dpi,
            metadata={"Software": "param-importance-nlp"},
        )
        plt.close(figure)
    content = buffer.getvalue()
    atomic_write_bytes(output_path, content)
    content_hash = hashlib.sha256(content).hexdigest()
    payload = {
        "schema_version": "analysis-heatmap-artifact-v1",
        "spec": spec.to_dict(),
        "renderer_id": f"matplotlib:{matplotlib.__version__}:Agg",
        "content_sha256": content_hash,
        "render_options": options,
    }
    return HeatmapArtifact(
        spec=spec,
        renderer_id=payload["renderer_id"],
        content_sha256=content_hash,
        render_options=options,
        artifact_hash=canonical_json_hash(payload),
    )


def analysis_producer_source_hash() -> str:
    """返回当前 ETL/表图实现源码 hash，供派生产物绑定算法实现。"""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class CompositeFigureArtifact:
    """heatmap、误差条与 facet 小图的路径无关组合产物。"""

    source_table_hash: str
    source_content_hash: str
    producer_source_hash: str
    spec_hash: str
    renderer_id: str
    files: tuple[Mapping[str, object], ...]
    render_options: Mapping[str, object]
    artifact_hash: str
    schema_version: str = "analysis-composite-figure-artifact-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "analysis-composite-figure-artifact-v1":
            raise ValueError("CompositeFigureArtifact schema_version 不受支持")
        for value in (
            self.source_table_hash,
            self.source_content_hash,
            self.producer_source_hash,
            self.spec_hash,
            self.artifact_hash,
        ):
            _require_hash(value, field_name="composite figure hash")
        if not self.renderer_id or len(self.files) != 3:
            raise ValueError("composite figure 必须包含三个 renderer 输出")
        kinds: set[str] = set()
        normalized: list[Mapping[str, object]] = []
        for item in self.files:
            if set(item) != {"kind", "content_sha256"}:
                raise ValueError("composite figure file fields 不匹配")
            kind = item["kind"]
            if kind not in {"heatmap", "errorbar", "facet"} or kind in kinds:
                raise ValueError("composite figure kind 非法或重复")
            _require_hash(item["content_sha256"], field_name="figure content hash")
            kinds.add(str(kind))
            normalized.append(freeze_json_mapping(item))
        object.__setattr__(self, "files", tuple(normalized))
        object.__setattr__(self, "render_options", freeze_json_mapping(self.render_options))
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("CompositeFigureArtifact artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_table_hash": self.source_table_hash,
            "source_content_hash": self.source_content_hash,
            "producer_source_hash": self.producer_source_hash,
            "spec_hash": self.spec_hash,
            "renderer_id": self.renderer_id,
            "files": [thaw_json_value(item) for item in self.files],
            "render_options": thaw_json_value(self.render_options),
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CompositeFigureArtifact":
        """严格加载组合图 artifact，并复核源码、规格与文件内容 hash 字段。"""

        required = {
            "schema_version",
            "source_table_hash",
            "source_content_hash",
            "producer_source_hash",
            "spec_hash",
            "renderer_id",
            "files",
            "render_options",
            "artifact_hash",
        }
        _require_exact_fields(value, required, artifact_name="CompositeFigureArtifact")
        files = value["files"]
        render_options = value["render_options"]
        if not isinstance(files, list) or not all(
            isinstance(item, Mapping) for item in files
        ):
            raise TypeError("CompositeFigureArtifact files 必须是 object array")
        if not isinstance(render_options, Mapping):
            raise TypeError("CompositeFigureArtifact render_options 必须是 object")
        return cls(
            source_table_hash=value["source_table_hash"],  # type: ignore[arg-type]
            source_content_hash=value["source_content_hash"],  # type: ignore[arg-type]
            producer_source_hash=value["producer_source_hash"],  # type: ignore[arg-type]
            spec_hash=value["spec_hash"],  # type: ignore[arg-type]
            renderer_id=value["renderer_id"],  # type: ignore[arg-type]
            files=tuple(files),
            render_options=render_options,
            artifact_hash=value["artifact_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


def render_composite_figure_set(
    source: BoundSourceTable,
    output_directory: str | Path,
    *,
    value_column: str = "value",
    condition_column: str = "condition",
    pair_column: str = "replicate",
    x_column: str | None = None,
    dpi: int = 120,
) -> CompositeFigureArtifact:
    """从同一冻结源表确定性渲染 heatmap、Student-t 误差条和 facet。

    三张图共用 source/artifact hash 与源码 hash。heatmap 要求
    ``condition × pair`` 是完整矩形；误差条按 condition 聚合；facet 每个
    condition 一个 panel。缺列、重复 cell 或非有限数均 fail-closed。
    """

    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("composite figure dpi 必须为正整数")
    required = {value_column, condition_column, pair_column}
    if not required.issubset(source.table.columns):
        raise ValueError("composite figure source columns 缺失")
    x_name = x_column or ("ratio" if "ratio" in source.table.columns else pair_column)
    if x_name not in source.table.columns:
        raise ValueError("composite figure x column 缺失")
    rows = [dict(row) for row in source.table.rows]
    for row in rows:
        _finite_number(row[value_column], field_name=value_column)
    conditions = sorted({row[condition_column] for row in rows}, key=_json_sort_key)
    pairs = sorted({row[pair_column] for row in rows}, key=_json_sort_key)
    cells: dict[tuple[bytes, bytes], float] = {}
    for row in rows:
        key = (_json_sort_key(row[condition_column]), _json_sort_key(row[pair_column]))
        if key in cells:
            raise ValueError("composite heatmap 存在重复 condition/pair")
        cells[key] = _finite_number(row[value_column], field_name=value_column)
    if len(cells) != len(conditions) * len(pairs):
        raise ValueError("composite heatmap condition/pair 不是完整矩形")
    spec = {
        "schema_version": "analysis-composite-figure-spec-v1",
        "source_table_hash": source.artifact_hash,
        "source_content_hash": source.table.content_hash,
        "producer_source_hash": analysis_producer_source_hash(),
        "value_column": value_column,
        "condition_column": condition_column,
        "pair_column": pair_column,
        "x_column": x_name,
        "panels": ["heatmap", "errorbar", "facet"],
    }
    spec_hash = canonical_json_hash(spec)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("render_composite_figure_set 需要 matplotlib") from exc
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    options = {
        "dpi": dpi,
        "backend": "Agg",
        "figsize_inches": [6.4, 4.0],
        "font_family": "DejaVu Sans",
    }
    rc = {
        "font.family": options["font_family"],
        "figure.figsize": options["figsize_inches"],
        "axes.unicode_minus": False,
        "axes.grid": True,
        "svg.hashsalt": spec_hash,
    }
    file_rows: list[Mapping[str, object]] = []

    def save(kind: str, figure: object) -> None:
        buffer = io.BytesIO()
        figure.savefig(  # type: ignore[attr-defined]
            buffer,
            format="png",
            dpi=dpi,
            metadata={"Software": "param-importance-nlp"},
        )
        plt.close(figure)
        content = buffer.getvalue()
        atomic_write_bytes(output / f"{kind}.png", content)
        file_rows.append(
            {"kind": kind, "content_sha256": hashlib.sha256(content).hexdigest()}
        )

    with matplotlib.rc_context(rc):
        matrix = np.asarray(
            [
                [
                    cells[(_json_sort_key(condition), _json_sort_key(pair))]
                    for pair in pairs
                ]
                for condition in conditions
            ],
            dtype=np.float64,
        )
        heatmap_figure, heatmap_axis = plt.subplots()
        image = heatmap_axis.imshow(matrix, aspect="auto", interpolation="nearest")
        heatmap_axis.set_xticks(range(len(pairs)), [str(value) for value in pairs])
        heatmap_axis.set_yticks(range(len(conditions)), [str(value) for value in conditions])
        heatmap_axis.set_xlabel(pair_column)
        heatmap_axis.set_ylabel(condition_column)
        heatmap_axis.set_title("Stage 9 heatmap")
        heatmap_figure.colorbar(image, ax=heatmap_axis, label=value_column)
        heatmap_figure.tight_layout()
        save("heatmap", heatmap_figure)

        means: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        for condition in conditions:
            values = [
                float(row[value_column])
                for row in rows
                if row[condition_column] == condition
            ]
            mean_result, lower, upper = mean_confidence_interval(values)
            mean_value = float(np.mean(values))
            means.append(mean_value)
            lower_errors.append(0.0 if not mean_result.defined else mean_value - float(lower.value))
            upper_errors.append(0.0 if not mean_result.defined else float(upper.value) - mean_value)
        error_figure, error_axis = plt.subplots()
        positions = np.arange(len(conditions), dtype=np.float64)
        error_axis.errorbar(
            positions,
            means,
            yerr=np.asarray([lower_errors, upper_errors]),
            fmt="o",
            capsize=4,
        )
        error_axis.set_xticks(positions, [str(value) for value in conditions])
        error_axis.set_xlabel(condition_column)
        error_axis.set_ylabel(value_column)
        error_axis.set_title("Stage 9 mean and confidence interval")
        error_figure.tight_layout()
        save("errorbar", error_figure)

        facet_figure, axes = plt.subplots(
            1,
            len(conditions),
            figsize=(max(4.0, 3.2 * len(conditions)), 3.6),
            squeeze=False,
        )
        for index, condition in enumerate(conditions):
            axis = axes[0, index]
            selected = sorted(
                (row for row in rows if row[condition_column] == condition),
                key=lambda row: _json_sort_key(row[x_name]),
            )
            axis.plot(
                [row[x_name] for row in selected],
                [float(row[value_column]) for row in selected],
                marker="o",
            )
            axis.set_title(str(condition))
            axis.set_xlabel(x_name)
            axis.set_ylabel(value_column)
        facet_figure.tight_layout()
        save("facet", facet_figure)

    payload = {
        "schema_version": "analysis-composite-figure-artifact-v1",
        "source_table_hash": source.artifact_hash,
        "source_content_hash": source.table.content_hash,
        "producer_source_hash": spec["producer_source_hash"],
        "spec_hash": spec_hash,
        "renderer_id": f"matplotlib:{matplotlib.__version__}:Agg",
        "files": file_rows,
        "render_options": options,
    }
    return CompositeFigureArtifact(
        source_table_hash=source.artifact_hash,
        source_content_hash=source.table.content_hash,
        producer_source_hash=str(spec["producer_source_hash"]),
        spec_hash=spec_hash,
        renderer_id=str(payload["renderer_id"]),
        files=tuple(file_rows),
        render_options=options,
        artifact_hash=canonical_json_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """可独立审计的 Stage 9 源表、报告、图表和表格集合。"""

    bundle_id: str
    scope: str
    formal_eligible: bool
    formal_authorization_hash: str | None
    tables: tuple[BoundSourceTable, ...]
    reports: tuple[AnalysisReport, ...]
    chart_artifacts: tuple[Mapping[str, object], ...]
    heatmap_artifacts: tuple[HeatmapArtifact, ...]
    table_artifacts: tuple[TableArtifact, ...]
    artifact_hash: str
    schema_version: str = "analysis-bundle-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "analysis-bundle-v1":
            raise ValueError("AnalysisBundle schema_version 不受支持")
        if not self.bundle_id:
            raise ValueError("bundle_id 不能为空")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("bundle scope 非法")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("bundle formal_eligible 与 scope 不一致")
        if self.scope == "formal":
            _require_hash(self.formal_authorization_hash, field_name="formal_authorization_hash")
        elif self.formal_authorization_hash is not None:
            raise ValueError("fixture bundle 不得携带 formal authorization")
        if not self.tables:
            raise ValueError("AnalysisBundle 至少需要一张表")
        if any(table.scope != self.scope for table in self.tables):
            raise ValueError("AnalysisBundle 混入其他 scope 的表")
        table_content_hashes = {table.table.content_hash for table in self.tables}
        table_artifact_hashes = {table.artifact_hash for table in self.tables}
        for report in self.reports:
            if any(
                source.content_hash not in table_content_hashes
                for source in report.source_artifacts
            ):
                raise ValueError("report 引用了 bundle 外 source")
        normalized_charts: list[Mapping[str, object]] = []
        for chart in self.chart_artifacts:
            if not isinstance(chart, Mapping):
                raise TypeError("chart artifact 必须是 mapping")
            spec = chart.get("spec")
            if not isinstance(spec, Mapping) or spec.get("source_hash") not in table_content_hashes:
                raise ValueError("chart 引用了 bundle 外 source")
            normalized_charts.append(freeze_json_mapping(chart))
        for heatmap in self.heatmap_artifacts:
            if heatmap.spec.table_artifact_hash not in table_artifact_hashes:
                raise ValueError("heatmap 引用了 bundle 外 table")
        for table_artifact in self.table_artifacts:
            if table_artifact.spec.table_artifact_hash not in table_artifact_hashes:
                raise ValueError("rendered table 引用了 bundle 外 table")
        object.__setattr__(self, "chart_artifacts", tuple(normalized_charts))
        _require_hash(self.artifact_hash, field_name="bundle artifact_hash")
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("AnalysisBundle artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "formal_authorization_hash": self.formal_authorization_hash,
            "tables": [table.to_dict() for table in self.tables],
            "reports": [report.to_dict() for report in self.reports],
            "chart_artifacts": [thaw_json_value(chart) for chart in self.chart_artifacts],
            "heatmap_artifacts": [artifact.to_dict() for artifact in self.heatmap_artifacts],
            "table_artifacts": [artifact.to_dict() for artifact in self.table_artifacts],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AnalysisBundle":
        """
        严格重载 bundle 及其嵌套 artifact。

        PNG 本体与表格文件不被偷偷重生；本 loader 只校验 bundle
        manifest 中已绑定的 hash 与完整 lineage。
        """

        required = {
            "schema_version",
            "bundle_id",
            "scope",
            "formal_eligible",
            "formal_authorization_hash",
            "tables",
            "reports",
            "chart_artifacts",
            "heatmap_artifacts",
            "table_artifacts",
            "artifact_hash",
        }
        _require_exact_fields(value, required, artifact_name="AnalysisBundle")
        sequence_fields = {
            name: value[name]
            for name in (
                "tables",
                "reports",
                "chart_artifacts",
                "heatmap_artifacts",
                "table_artifacts",
            )
        }
        if any(
            not isinstance(items, list)
            or not all(isinstance(item, Mapping) for item in items)
            for items in sequence_fields.values()
        ):
            raise TypeError("AnalysisBundle nested artifacts 必须是 object array")
        charts = tuple(
            ChartArtifact.from_mapping(item).to_dict()
            for item in sequence_fields["chart_artifacts"]
        )
        return cls(
            bundle_id=value["bundle_id"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            formal_authorization_hash=value["formal_authorization_hash"],  # type: ignore[arg-type]
            tables=tuple(
                BoundSourceTable.from_mapping(item)
                for item in sequence_fields["tables"]
            ),
            reports=tuple(
                AnalysisReport.from_mapping(item)
                for item in sequence_fields["reports"]
            ),
            chart_artifacts=charts,
            heatmap_artifacts=tuple(
                HeatmapArtifact.from_mapping(item)
                for item in sequence_fields["heatmap_artifacts"]
            ),
            table_artifacts=tuple(
                TableArtifact.from_mapping(item)
                for item in sequence_fields["table_artifacts"]
            ),
            artifact_hash=value["artifact_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )

    def publish(self, root: str | Path) -> Path:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        tables_dir = directory / "tables"
        reports_dir = directory / "reports"
        tables_dir.mkdir(exist_ok=True)
        reports_dir.mkdir(exist_ok=True)
        for index, table in enumerate(self.tables):
            write_canonical_json(
                tables_dir / f"{index:03d}-{table.table.name}.json", table.to_dict()
            )
        for index, report in enumerate(self.reports):
            write_canonical_json(
                reports_dir / f"{index:03d}-{report.report_id}.json", report.to_dict()
            )
            atomic_write_bytes(
                reports_dir / f"{index:03d}-{report.report_id}.md",
                report.render_markdown().encode("utf-8"),
            )
        rendered_dir = directory / "rendered-tables"
        for index, artifact in enumerate(self.table_artifacts):
            artifact.publish(rendered_dir, stem=f"table-{index:03d}")
        target = directory / "analysis-bundle.json"
        write_canonical_json(target, self.to_dict())
        return target


class AnalysisBundleBuilder:
    """fail-closed 构建 :class:`AnalysisBundle`。"""

    def __init__(
        self,
        *,
        bundle_id: str,
        scope: str,
        formal_authorization_hash: str | None = None,
    ) -> None:
        if not bundle_id:
            raise ValueError("bundle_id 不能为空")
        if scope not in {"local_fixture", "formal"}:
            raise ValueError("scope 非法")
        if scope == "formal":
            _require_hash(formal_authorization_hash, field_name="formal_authorization_hash")
        elif formal_authorization_hash is not None:
            raise ValueError("fixture bundle builder 不得携带 formal authorization")
        self.bundle_id = bundle_id
        self.scope = scope
        self.formal_authorization_hash = formal_authorization_hash
        self.tables: dict[str, BoundSourceTable] = {}
        self.reports: dict[str, AnalysisReport] = {}
        self.charts: dict[str, Mapping[str, object]] = {}
        self.heatmaps: dict[str, HeatmapArtifact] = {}
        self.rendered_tables: dict[str, TableArtifact] = {}

    def add_table(self, table: BoundSourceTable) -> None:
        if table.scope != self.scope:
            raise ValueError("bundle table scope 不一致")
        existing = self.tables.get(table.artifact_hash)
        if existing is not None and existing.to_dict() != table.to_dict():
            raise ValueError("bundle table hash 冲突")
        self.tables[table.artifact_hash] = table

    def add_report(self, report: AnalysisReport) -> None:
        self.reports[report.report_hash] = report

    def add_chart(self, chart: ChartArtifact | Mapping[str, object]) -> None:
        value = chart.to_dict() if isinstance(chart, ChartArtifact) else dict(chart)
        artifact_hash = value.get("artifact_hash")
        _require_hash(artifact_hash, field_name="chart artifact_hash")
        self.charts[str(artifact_hash)] = freeze_json_mapping(value)

    def add_heatmap(self, artifact: HeatmapArtifact) -> None:
        self.heatmaps[artifact.artifact_hash] = artifact

    def add_rendered_table(self, artifact: TableArtifact) -> None:
        self.rendered_tables[artifact.artifact_hash] = artifact

    def build(self) -> AnalysisBundle:
        tables = tuple(self.tables[key] for key in sorted(self.tables))
        reports = tuple(self.reports[key] for key in sorted(self.reports))
        charts = tuple(self.charts[key] for key in sorted(self.charts))
        heatmaps = tuple(self.heatmaps[key] for key in sorted(self.heatmaps))
        rendered = tuple(self.rendered_tables[key] for key in sorted(self.rendered_tables))
        payload = {
            "schema_version": "analysis-bundle-v1",
            "bundle_id": self.bundle_id,
            "scope": self.scope,
            "formal_eligible": self.scope == "formal",
            "formal_authorization_hash": self.formal_authorization_hash,
            "tables": [table.to_dict() for table in tables],
            "reports": [report.to_dict() for report in reports],
            "chart_artifacts": [thaw_json_value(chart) for chart in charts],
            "heatmap_artifacts": [artifact.to_dict() for artifact in heatmaps],
            "table_artifacts": [artifact.to_dict() for artifact in rendered],
        }
        return AnalysisBundle(
            bundle_id=self.bundle_id,
            scope=self.scope,
            formal_eligible=self.scope == "formal",
            formal_authorization_hash=self.formal_authorization_hash,
            tables=tables,
            reports=reports,
            chart_artifacts=charts,
            heatmap_artifacts=heatmaps,
            table_artifacts=rendered,
            artifact_hash=canonical_json_hash(payload),
        )


__all__ = [
    "AnalysisBundle",
    "AnalysisBundleBuilder",
    "BoundSourceTable",
    "CrossStageSourceBuilder",
    "HeatmapArtifact",
    "HeatmapSpec",
    "StageArtifactRows",
    "TableArtifact",
    "TableSpec",
    "grouped_statistics",
    "paired_statistics_with_holm",
    "render_heatmap",
    "render_table",
]
