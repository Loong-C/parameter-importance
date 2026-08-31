"""Stage 2/3 跨子任务结果索引。

研究波次会产生大量不可变 shard 与报告；结果索引只保存逻辑 commit 引用和它们的
内容 hash，不复制梯度或节点张量。它使下游 Stage 4 能消费一个完整、可审核的研究
结论边界，同时保持 fixture/formal 资格不可混用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Mapping

from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from ..contracts.jsonio import JSONValue, canonical_json_hash


_HASH = re.compile(r"^[0-9a-f]{64}$")


def _ref(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} 不是 POSIX 逻辑引用")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} 发生路径逃逸")
    return path.as_posix()


def _hash(value: str | None, *, field: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{field} 必须是小写 SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class EstimatorStudyResult:
    """Stage 2 reference/pilot/confirmatory/decision 的完整索引。"""

    study_id: str
    scope: str
    reference_ref: str
    pilot_ref: str
    confirmatory_ref: str
    decision_ref: str
    decision_hash: str
    formal_eligible: bool
    metadata: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.study_id:
            raise ValueError("ESTIMATOR_STUDY_ID_EMPTY")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("ESTIMATOR_STUDY_SCOPE_INVALID")
        for field in ("reference_ref", "pilot_ref", "confirmatory_ref", "decision_ref"):
            object.__setattr__(self, field, _ref(getattr(self, field), field=field))
        _hash(self.decision_hash, field="decision_hash")
        if self.formal_eligible != (self.scope == "formal"):
            raise ValueError("ESTIMATOR_STUDY_FORMAL_ELIGIBILITY_MISMATCH")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "schema_version": "estimator-study-result-v1",
            "study_id": self.study_id,
            "scope": self.scope,
            "reference_ref": self.reference_ref,
            "pilot_ref": self.pilot_ref,
            "confirmatory_ref": self.confirmatory_ref,
            "decision_ref": self.decision_ref,
            "decision_hash": self.decision_hash,
            "formal_eligible": self.formal_eligible,
            "metadata": thaw_json_value(self.metadata),
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EstimatorStudyResult":
        expected = {
            "schema_version", "study_id", "scope", "reference_ref", "pilot_ref",
            "confirmatory_ref", "decision_ref", "decision_hash", "formal_eligible",
            "metadata", "artifact_hash",
        }
        if set(value) != expected or value.get("schema_version") != "estimator-study-result-v1":
            raise ValueError("ESTIMATOR_STUDY_FIELDS_OR_VERSION_INVALID")
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("ESTIMATOR_STUDY_METADATA_INVALID")
        result = cls(
            study_id=value["study_id"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            reference_ref=value["reference_ref"],  # type: ignore[arg-type]
            pilot_ref=value["pilot_ref"],  # type: ignore[arg-type]
            confirmatory_ref=value["confirmatory_ref"],  # type: ignore[arg-type]
            decision_ref=value["decision_ref"],  # type: ignore[arg-type]
            decision_hash=value["decision_hash"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            metadata=metadata,  # type: ignore[arg-type]
        )
        if value["artifact_hash"] != result.to_dict()["artifact_hash"]:
            raise ValueError("ESTIMATOR_STUDY_HASH_MISMATCH")
        return result


@dataclass(frozen=True, slots=True)
class PathStudyResult:
    """Stage 3 endpoint/probe/reference/matrix/recommendation 的完整索引。"""

    study_id: str
    scope: str
    endpoint_index_ref: str
    probe_panel_ref: str
    reference_ref: str
    matrix_ref: str
    recommendation_ref: str
    recommendation_hash: str
    formal_eligible: bool
    metadata: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.study_id:
            raise ValueError("PATH_STUDY_ID_EMPTY")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("PATH_STUDY_SCOPE_INVALID")
        for field in (
            "endpoint_index_ref", "probe_panel_ref", "reference_ref", "matrix_ref",
            "recommendation_ref",
        ):
            object.__setattr__(self, field, _ref(getattr(self, field), field=field))
        _hash(self.recommendation_hash, field="recommendation_hash")
        if self.formal_eligible != (self.scope == "formal"):
            raise ValueError("PATH_STUDY_FORMAL_ELIGIBILITY_MISMATCH")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "schema_version": "path-study-result-v1",
            "study_id": self.study_id,
            "scope": self.scope,
            "endpoint_index_ref": self.endpoint_index_ref,
            "probe_panel_ref": self.probe_panel_ref,
            "reference_ref": self.reference_ref,
            "matrix_ref": self.matrix_ref,
            "recommendation_ref": self.recommendation_ref,
            "recommendation_hash": self.recommendation_hash,
            "formal_eligible": self.formal_eligible,
            "metadata": thaw_json_value(self.metadata),
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PathStudyResult":
        expected = {
            "schema_version", "study_id", "scope", "endpoint_index_ref",
            "probe_panel_ref", "reference_ref", "matrix_ref", "recommendation_ref",
            "recommendation_hash", "formal_eligible", "metadata", "artifact_hash",
        }
        if set(value) != expected or value.get("schema_version") != "path-study-result-v1":
            raise ValueError("PATH_STUDY_FIELDS_OR_VERSION_INVALID")
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("PATH_STUDY_METADATA_INVALID")
        result = cls(
            study_id=value["study_id"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            endpoint_index_ref=value["endpoint_index_ref"],  # type: ignore[arg-type]
            probe_panel_ref=value["probe_panel_ref"],  # type: ignore[arg-type]
            reference_ref=value["reference_ref"],  # type: ignore[arg-type]
            matrix_ref=value["matrix_ref"],  # type: ignore[arg-type]
            recommendation_ref=value["recommendation_ref"],  # type: ignore[arg-type]
            recommendation_hash=value["recommendation_hash"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            metadata=metadata,  # type: ignore[arg-type]
        )
        if value["artifact_hash"] != result.to_dict()["artifact_hash"]:
            raise ValueError("PATH_STUDY_HASH_MISMATCH")
        return result


__all__ = ["EstimatorStudyResult", "PathStudyResult"]
