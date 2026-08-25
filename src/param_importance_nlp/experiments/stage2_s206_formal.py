"""Strict S2.6/G2.4b preparation helpers.

This module is intentionally separate from the existing TaskRuntime runner.  It
prepares the 24-cell pilot schedule, gives each cell a disjoint interval in the
single frozen ``pilot`` stream, reduces only operational/blinded fields, and
binds a formal matrix to a newly issued G2.4b GateRecord.  It never calls a
provider and never creates confirmatory draws during pilot reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..contracts.errors import FormalRunRejected
from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from ..contracts.stage23 import (
    ArtifactQualification,
    FormalExecutionEvidence,
    require_accepted_gate,
)
from ..contracts.status import GateRecord, GateStatus
from .sampling import (
    CANDIDATE_BATCH_SIZES,
    CANDIDATE_MICROBATCH_COUNTS,
    Draw,
    RepetitionMapping,
    SamplingPlan,
)
from .stage2_pilot import (
    ANCHOR_IDS,
    AnchorPilotResult,
    CandidateEvaluation,
    CostSemantics,
    required_repetitions,
    scan_candidates,
)


GLOBAL_PILOT_MAPPING_SCHEMA = "stage2-formal-global-pilot-mapping-v1"
BLINDED_PILOT_REPORT_SCHEMA = "stage2-formal-blinded-pilot-report-v1"
FORMAL_MATRIX_SCHEMA = "stage2-formal-pilot-matrix-freeze-v1"
G24B_GATE_SCHEMA = "stage2-g24b-gate-v1"
FORMAL_CONFIRMATORY_MAPPING_SCHEMA = "stage2-formal-confirmatory-mapping-v1"

PILOT_REPETITIONS = 50
PILOT_M_VALUES = (2, 4, 8, 16, 32)
PILOT_B_VALUES = tuple(CANDIDATE_BATCH_SIZES)
PILOT_STREAM = "pilot"
CONFIRMATORY_STREAM = "confirmatory"

APPROVED_GPU_UUIDS = (
    "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
)
EXCLUDED_PCI = "0000:50:00.0"

CELL_GPU_BINDINGS = {
    ANCHOR_IDS[0]: APPROVED_GPU_UUIDS[0],
    ANCHOR_IDS[1]: APPROVED_GPU_UUIDS[1],
    ANCHOR_IDS[2]: APPROVED_GPU_UUIDS[2],
    ANCHOR_IDS[3]: APPROVED_GPU_UUIDS[3],
    ANCHOR_IDS[4]: APPROVED_GPU_UUIDS[0],
    ANCHOR_IDS[5]: APPROVED_GPU_UUIDS[1],
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


class S206PreparationBlocked(RuntimeError):
    """Raised when S2.6 preparation cannot prove its immutable inputs."""


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} must be a safe identifier")
    return value


def _canonical_hash_without_artifact(value: Mapping[str, object]) -> str:
    return canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    )


def _cell_component(anchor_id: str, batch_size: int) -> str:
    return f"{anchor_id.replace('.', '__')}__b{batch_size:03d}"


def _cell_order() -> tuple[tuple[str, int], ...]:
    return tuple((anchor_id, batch_size) for anchor_id in ANCHOR_IDS for batch_size in PILOT_B_VALUES)


@dataclass(frozen=True, slots=True)
class PilotCellMapping:
    """One anchor/B slice of the globally disjoint pilot draw stream."""

    anchor_id: str
    batch_size: int
    repetitions: int
    stream_start: int
    mappings: tuple[RepetitionMapping, ...]

    def __post_init__(self) -> None:
        if self.anchor_id not in ANCHOR_IDS:
            raise ValueError("pilot anchor is not preregistered")
        if self.batch_size not in PILOT_B_VALUES:
            raise ValueError("pilot B is not preregistered")
        if self.repetitions != PILOT_REPETITIONS:
            raise ValueError("S2.6 pilot repetitions must be exactly 50")
        if self.stream_start < 0:
            raise ValueError("pilot stream_start must be non-negative")
        if len(self.mappings) != self.repetitions:
            raise ValueError("pilot mapping count does not match repetitions")
        expected_start = self.stream_start
        expected_ids: set[str] = set()
        for index, mapping in enumerate(self.mappings):
            expected_id = f"pilot-{self.anchor_id.replace('.', '-')}-b{self.batch_size:03d}-r{index:04d}"
            if mapping.repetition_id != expected_id:
                raise ValueError("pilot repetition_id is not deterministic")
            if mapping.batch_size != self.batch_size or mapping.m_values != PILOT_M_VALUES:
                raise ValueError("pilot mapping B/M contract mismatch")
            if any(draw.stream != PILOT_STREAM for draw in mapping.draws):
                raise ValueError("pilot mapping consumed a non-pilot draw")
            if mapping.draws[0].position != expected_start:
                raise ValueError("pilot mapping stream interval is not contiguous")
            expected_start += self.batch_size
            expected_ids.update(draw.draw_id for draw in mapping.draws)
        if len(expected_ids) != self.repetitions * self.batch_size:
            raise ValueError("pilot cell draw IDs are not unique")

    @property
    def stream_end(self) -> int:
        return self.stream_start + self.repetitions * self.batch_size

    @property
    def draw_ids(self) -> tuple[str, ...]:
        return tuple(draw.draw_id for mapping in self.mappings for draw in mapping.draws)

    @property
    def sample_collision_count(self) -> int:
        draws = tuple(draw for mapping in self.mappings for draw in mapping.draws)
        return len(draws) - len({draw.sample_id for draw in draws})

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "batch_size": self.batch_size,
            "repetitions": self.repetitions,
            "stream": PILOT_STREAM,
            "stream_start": self.stream_start,
            "stream_end": self.stream_end,
            "mapping_ids": [mapping.repetition_id for mapping in self.mappings],
            "mappings": [mapping.to_dict() for mapping in self.mappings],
            "draw_id_count": len(self.draw_ids),
            "draw_id_hash": canonical_json_hash(list(self.draw_ids)),
            "sample_collision_count": self.sample_collision_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PilotCellMapping":
        required = {
            "anchor_id", "batch_size", "repetitions", "stream", "stream_start",
            "stream_end", "mapping_ids", "mappings", "draw_id_count",
            "draw_id_hash", "sample_collision_count",
        }
        if set(value) != required or value.get("stream") != PILOT_STREAM:
            raise ValueError("pilot cell mapping fields/stream mismatch")
        raw_mappings = value["mappings"]
        if not isinstance(raw_mappings, list) or not all(isinstance(item, Mapping) for item in raw_mappings):
            raise TypeError("pilot cell mappings must be objects")
        result = cls(
            anchor_id=value["anchor_id"],  # type: ignore[arg-type]
            batch_size=value["batch_size"],  # type: ignore[arg-type]
            repetitions=value["repetitions"],  # type: ignore[arg-type]
            stream_start=value["stream_start"],  # type: ignore[arg-type]
            mappings=tuple(RepetitionMapping.from_manifest(dict(item)) for item in raw_mappings),
        )
        if value["stream_end"] != result.stream_end or value["mapping_ids"] != [item.repetition_id for item in result.mappings]:
            raise ValueError("pilot cell mapping derived fields mismatch")
        if value["draw_id_count"] != len(result.draw_ids) or value["draw_id_hash"] != canonical_json_hash(list(result.draw_ids)):
            raise ValueError("pilot cell mapping draw hash mismatch")
        if value["sample_collision_count"] != result.sample_collision_count:
            raise ValueError("pilot cell mapping collision count mismatch")
        return result


@dataclass(frozen=True, slots=True)
class GlobalPilotMappingManifest:
    """The complete 24-cell pilot mapping, with no overlapping positions."""

    mapping_id: str
    sampling_plan_hash: str
    cells: tuple[PilotCellMapping, ...]
    stream: str = PILOT_STREAM
    schema_version: str = GLOBAL_PILOT_MAPPING_SCHEMA

    def __post_init__(self) -> None:
        _safe_id(self.mapping_id, "mapping_id")
        _hash(self.sampling_plan_hash, "sampling_plan_hash")
        if self.schema_version != GLOBAL_PILOT_MAPPING_SCHEMA or self.stream != PILOT_STREAM:
            raise ValueError("global pilot mapping schema/stream mismatch")
        expected_cells = _cell_order()
        actual_cells = tuple((item.anchor_id, item.batch_size) for item in self.cells)
        if actual_cells != expected_cells:
            raise ValueError("global pilot mapping must contain all 24 cells in canonical order")
        expected_start = 0
        draw_ids: set[str] = set()
        for cell in self.cells:
            if cell.stream_start != expected_start:
                raise ValueError("global pilot stream intervals must be contiguous")
            expected_start = cell.stream_end
            draw_ids.update(cell.draw_ids)
        if len(draw_ids) != expected_start:
            raise ValueError("global pilot draw IDs must be globally unique")

    @property
    def total_draw_count(self) -> int:
        return sum(cell.repetitions * cell.batch_size for cell in self.cells)

    @property
    def pilot_draw_ids(self) -> tuple[str, ...]:
        return tuple(draw_id for cell in self.cells for draw_id in cell.draw_ids)

    @property
    def sample_collision_count(self) -> int:
        draws = tuple(
            draw
            for cell in self.cells
            for mapping in cell.mappings
            for draw in mapping.draws
        )
        return len(draws) - len({draw.sample_id for draw in draws})

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "mapping_id": self.mapping_id,
            "stream": self.stream,
            "sampling_plan_hash": self.sampling_plan_hash,
            "cell_count": len(self.cells),
            "cells": [cell.to_dict() for cell in self.cells],
            "total_draw_count": self.total_draw_count,
            "pilot_draw_id_unique": True,
            "pilot_draw_id_hash": canonical_json_hash(list(self.pilot_draw_ids)),
            "sample_id_collision_count": self.sample_collision_count,
            "confirmatory_draws_generated": False,
            "formal_eligible": False,
            "qualification_gate_hash": None,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "GlobalPilotMappingManifest":
        required = {
            "schema_version", "mapping_id", "stream", "sampling_plan_hash", "cell_count",
            "cells", "total_draw_count", "pilot_draw_id_unique", "pilot_draw_id_hash",
            "sample_id_collision_count", "confirmatory_draws_generated",
            "formal_eligible", "qualification_gate_hash", "artifact_hash",
        }
        if set(value) != required or value.get("schema_version") != GLOBAL_PILOT_MAPPING_SCHEMA:
            raise ValueError("global pilot mapping fields/schema mismatch")
        raw_cells = value["cells"]
        if not isinstance(raw_cells, list) or not all(isinstance(item, Mapping) for item in raw_cells):
            raise TypeError("global pilot cells must be objects")
        result = cls(
            mapping_id=value["mapping_id"],  # type: ignore[arg-type]
            sampling_plan_hash=value["sampling_plan_hash"],  # type: ignore[arg-type]
            cells=tuple(PilotCellMapping.from_mapping(dict(item)) for item in raw_cells),
            stream=value["stream"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
        if value["artifact_hash"] != result.artifact_hash:
            raise ValueError("global pilot mapping artifact hash mismatch")
        if value["cell_count"] != len(result.cells) or value["total_draw_count"] != result.total_draw_count:
            raise ValueError("global pilot mapping counts mismatch")
        if value["pilot_draw_id_unique"] is not True or value["pilot_draw_id_hash"] != canonical_json_hash(list(result.pilot_draw_ids)):
            raise ValueError("global pilot draw identity mismatch")
        if value["confirmatory_draws_generated"] is not False or value["formal_eligible"] is not False or value["qualification_gate_hash"] is not None:
            raise FormalRunRejected("PILOT_MAPPING_SCOPE_UPGRADED")
        return result


def build_global_pilot_mapping(
    sampling: SamplingPlan,
    *,
    mapping_id: str = "s206-formal-pilot-mapping",
    repetitions: int = PILOT_REPETITIONS,
) -> GlobalPilotMappingManifest:
    """Build all 24 cells from one contiguous pilot stream interval."""

    if repetitions != PILOT_REPETITIONS:
        raise ValueError("S2.6 requires exactly 50 pilot repetitions per anchor/B")
    if not isinstance(sampling, SamplingPlan):
        raise TypeError("sampling must be a SamplingPlan")
    cells: list[PilotCellMapping] = []
    offset = 0
    for anchor_id, batch_size in _cell_order():
        draws = sampling.draws(PILOT_STREAM, repetitions * batch_size, start=offset)
        mappings = tuple(
            RepetitionMapping.create(
                repetition_id=(
                    f"pilot-{anchor_id.replace('.', '-')}-b{batch_size:03d}-r{index:04d}"
                ),
                draws=draws[index * batch_size : (index + 1) * batch_size],
                m_values=PILOT_M_VALUES,
            )
            for index in range(repetitions)
        )
        cells.append(
            PilotCellMapping(
                anchor_id=anchor_id,
                batch_size=batch_size,
                repetitions=repetitions,
                stream_start=offset,
                mappings=mappings,
            )
        )
        offset += repetitions * batch_size
    return GlobalPilotMappingManifest(
        mapping_id=mapping_id,
        sampling_plan_hash=sampling.digest,
        cells=tuple(cells),
    )


@dataclass(frozen=True, slots=True)
class BlindPilotMeasurement:
    """One operational row; scientific means and directions are impossible here."""

    anchor_id: str
    batch_size: int
    microbatch_count: int
    repetitions: int
    anchors_runnable: bool
    finite: bool
    state_unchanged: bool
    m2_equivalent: bool
    mean_gradient_consistent: bool
    aggregation_overhead_ratio: float
    variance_by_endpoint: Mapping[str, float]
    delta_sci_by_endpoint: Mapping[str, float]
    reference_half_width_by_endpoint: Mapping[str, float]
    storage_bytes: int
    gpu_hours: float
    resource_within_budget: bool
    cost_io_quiescent: bool

    def __post_init__(self) -> None:
        if self.anchor_id not in ANCHOR_IDS:
            raise ValueError("measurement anchor is not preregistered")
        if self.batch_size not in PILOT_B_VALUES or self.microbatch_count not in PILOT_M_VALUES:
            raise ValueError("measurement B/M is not preregistered")
        if self.batch_size % self.microbatch_count:
            raise ValueError("measurement M must divide B")
        if self.repetitions != PILOT_REPETITIONS:
            raise ValueError("measurement repetitions must be exactly 50")
        if not 0 <= self.aggregation_overhead_ratio <= 1:
            raise ValueError("aggregation overhead must be in [0,1]")
        if self.storage_bytes < 0 or self.gpu_hours < 0:
            raise ValueError("storage/gpu-hours must be non-negative")
        endpoints = set(self.variance_by_endpoint)
        if not endpoints or endpoints != set(self.delta_sci_by_endpoint) or endpoints != set(self.reference_half_width_by_endpoint):
            raise ValueError("sizing endpoint sets must match and be non-empty")
        for endpoint in sorted(endpoints):
            variance = float(self.variance_by_endpoint[endpoint])
            delta = float(self.delta_sci_by_endpoint[endpoint])
            half_width = float(self.reference_half_width_by_endpoint[endpoint])
            if not all(map(lambda value: value == value and abs(value) != float("inf"), (variance, delta, half_width))):
                raise ValueError("sizing values must be finite")
            if variance < 0 or delta <= 0 or half_width < 0 or half_width > delta / 4:
                raise ValueError("sizing values violate the preregistered precision contract")

    @property
    def required_repetitions_by_endpoint(self) -> dict[str, int]:
        return {
            endpoint: required_repetitions(
                estimator_variance=float(self.variance_by_endpoint[endpoint]),
                delta_sci=float(self.delta_sci_by_endpoint[endpoint]),
                reference_half_width=float(self.reference_half_width_by_endpoint[endpoint]),
            )
            for endpoint in sorted(self.variance_by_endpoint)
        }

    @property
    def required_repetitions(self) -> int:
        return max(self.required_repetitions_by_endpoint.values())

    @property
    def operational_ready(self) -> bool:
        return all(
            (
                self.anchors_runnable,
                self.finite,
                self.state_unchanged,
                self.m2_equivalent,
                self.mean_gradient_consistent,
                self.resource_within_budget,
                self.cost_io_quiescent,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "repetitions": self.repetitions,
            "anchors_runnable": self.anchors_runnable,
            "finite": self.finite,
            "state_unchanged": self.state_unchanged,
            "m2_equivalent": self.m2_equivalent,
            "mean_gradient_consistent": self.mean_gradient_consistent,
            "aggregation_overhead_ratio": self.aggregation_overhead_ratio,
            "variance_by_endpoint": dict(self.variance_by_endpoint),
            "delta_sci_by_endpoint": dict(self.delta_sci_by_endpoint),
            "reference_half_width_by_endpoint": dict(self.reference_half_width_by_endpoint),
            "required_repetitions_by_endpoint": self.required_repetitions_by_endpoint,
            "required_repetitions": self.required_repetitions,
            "storage_bytes": self.storage_bytes,
            "gpu_hours": self.gpu_hours,
            "resource_within_budget": self.resource_within_budget,
            "cost_io_quiescent": self.cost_io_quiescent,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "BlindPilotMeasurement":
        required = {
            "anchor_id", "batch_size", "microbatch_count", "repetitions",
            "anchors_runnable", "finite", "state_unchanged", "m2_equivalent",
            "mean_gradient_consistent", "aggregation_overhead_ratio",
            "variance_by_endpoint", "delta_sci_by_endpoint",
            "reference_half_width_by_endpoint", "required_repetitions_by_endpoint",
            "required_repetitions", "storage_bytes", "gpu_hours",
            "resource_within_budget", "cost_io_quiescent",
        }
        if set(value) != required:
            raise ValueError("blind pilot measurement fields mismatch")
        for field in ("variance_by_endpoint", "delta_sci_by_endpoint", "reference_half_width_by_endpoint"):
            if not isinstance(value[field], Mapping):
                raise TypeError(f"{field} must be an object")
        result = cls(
            anchor_id=value["anchor_id"],  # type: ignore[arg-type]
            batch_size=value["batch_size"],  # type: ignore[arg-type]
            microbatch_count=value["microbatch_count"],  # type: ignore[arg-type]
            repetitions=value["repetitions"],  # type: ignore[arg-type]
            anchors_runnable=value["anchors_runnable"],  # type: ignore[arg-type]
            finite=value["finite"],  # type: ignore[arg-type]
            state_unchanged=value["state_unchanged"],  # type: ignore[arg-type]
            m2_equivalent=value["m2_equivalent"],  # type: ignore[arg-type]
            mean_gradient_consistent=value["mean_gradient_consistent"],  # type: ignore[arg-type]
            aggregation_overhead_ratio=value["aggregation_overhead_ratio"],  # type: ignore[arg-type]
            variance_by_endpoint=dict(value["variance_by_endpoint"]),  # type: ignore[arg-type]
            delta_sci_by_endpoint=dict(value["delta_sci_by_endpoint"]),  # type: ignore[arg-type]
            reference_half_width_by_endpoint=dict(value["reference_half_width_by_endpoint"]),  # type: ignore[arg-type]
            storage_bytes=value["storage_bytes"],  # type: ignore[arg-type]
            gpu_hours=value["gpu_hours"],  # type: ignore[arg-type]
            resource_within_budget=value["resource_within_budget"],  # type: ignore[arg-type]
            cost_io_quiescent=value["cost_io_quiescent"],  # type: ignore[arg-type]
        )
        if value["required_repetitions_by_endpoint"] != result.required_repetitions_by_endpoint or value["required_repetitions"] != result.required_repetitions:
            raise ValueError("blind pilot sizing fields mismatch")
        return result


@dataclass(frozen=True, slots=True)
class BlindedPilotReport:
    """A candidate-only report suitable for matrix selection."""

    report_id: str
    mapping_hash: str
    sampling_plan_hash: str
    measurements: tuple[BlindPilotMeasurement, ...]
    anchor_rows: tuple[AnchorPilotResult, ...]
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    cost_semantics: CostSemantics
    status: str
    schema_version: str = BLINDED_PILOT_REPORT_SCHEMA

    def __post_init__(self) -> None:
        _safe_id(self.report_id, "report_id")
        _hash(self.mapping_hash, "mapping_hash")
        _hash(self.sampling_plan_hash, "sampling_plan_hash")
        if self.status not in {"READY_FOR_QUALIFICATION", "BLOCKED"}:
            raise ValueError("invalid blinded pilot status")
        if self.schema_version != BLINDED_PILOT_REPORT_SCHEMA:
            raise ValueError("invalid blinded pilot schema")

    @property
    def selected(self) -> CandidateEvaluation | None:
        return next((item for item in self.candidate_evaluations if item.selected), None)

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "mapping_hash": self.mapping_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "status": self.status,
            "measurement_count": len(self.measurements),
            "anchor_row_count": len(self.anchor_rows),
            "measurements": [item.to_dict() for item in self.measurements],
            "anchor_rows": [item.to_dict() for item in self.anchor_rows],
            "candidate_evaluations": [item.to_dict() for item in self.candidate_evaluations],
            "cost_semantics": self.cost_semantics.to_dict(),
            "scientific_values_masked": True,
            "formal_eligible": False,
            "qualification_gate_hash": None,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value


def reduce_blinded_pilot(
    mapping: GlobalPilotMappingManifest,
    measurements: Sequence[BlindPilotMeasurement],
    *,
    cost_semantics: CostSemantics,
    report_id: str = "s206-formal-blinded-pilot",
    r_max: int = 1000,
) -> BlindedPilotReport:
    """Reduce exactly 24×5 operational rows and apply the preregistered scan."""

    if mapping.stream != PILOT_STREAM:
        raise S206PreparationBlocked("PILOT_MAPPING_STREAM_INVALID")
    expected = {(anchor_id, batch_size, microbatch_count) for anchor_id, batch_size in _cell_order() for microbatch_count in PILOT_M_VALUES}
    observed = {(item.anchor_id, item.batch_size, item.microbatch_count) for item in measurements}
    if len(measurements) != len(expected) or observed != expected:
        raise S206PreparationBlocked("PILOT_MEASUREMENT_GRID_INCOMPLETE_OR_DUPLICATE")
    by_key = {(item.anchor_id, item.batch_size, item.microbatch_count): item for item in measurements}
    anchor_rows: list[AnchorPilotResult] = []
    for anchor_id, batch_size, microbatch_count in sorted(expected, key=lambda item: (ANCHOR_IDS.index(item[0]), item[1], item[2])):
        item = by_key[(anchor_id, batch_size, microbatch_count)]
        if not item.operational_ready:
            raise S206PreparationBlocked(f"PILOT_OPERATIONAL_CHECK_FAILED:{anchor_id}:B{batch_size}:M{microbatch_count}")
        anchor_rows.append(
            AnchorPilotResult(
                anchor_id=anchor_id,
                batch_size=batch_size,
                microbatch_count=microbatch_count,
                repetitions=item.repetitions,
                anchors_runnable=item.anchors_runnable,
                finite=item.finite,
                aggregation_overhead_ratio=item.aggregation_overhead_ratio,
                required_repetitions=item.required_repetitions,
                storage_bytes=item.storage_bytes,
                gpu_hours=item.gpu_hours,
                resource_within_budget=item.resource_within_budget,
                cost_io_quiescent=item.cost_io_quiescent,
            )
        )
    evaluations = scan_candidates(anchor_rows, required_anchor_ids=ANCHOR_IDS, r_max=r_max)
    selected = [item for item in evaluations if item.selected]
    if len(selected) > 1:
        raise S206PreparationBlocked("PILOT_MULTIPLE_SELECTED_PAIRS")
    status = "READY_FOR_QUALIFICATION" if selected and selected[0].microbatch_count > 2 else "BLOCKED"
    if not cost_semantics.cost_io_quiescent:
        status = "BLOCKED"
    if any(not bool(getattr(cost_semantics, name).get("defined")) for name in (
        "scientific_equal_sample_cost",
        "isolated_estimator_cost",
        "online_training_incremental_cost",
    )):
        status = "BLOCKED"
    return BlindedPilotReport(
        report_id=report_id,
        mapping_hash=mapping.artifact_hash,
        sampling_plan_hash=mapping.sampling_plan_hash,
        measurements=tuple(measurements),
        anchor_rows=tuple(anchor_rows),
        candidate_evaluations=tuple(evaluations),
        cost_semantics=cost_semantics,
        status=status,
    )


@dataclass(frozen=True, slots=True)
class FormalMatrixFreeze:
    """Formal matrix candidate qualified by the G2.4b GateRecord."""

    freeze_id: str
    pilot_report_hash: str
    pilot_mapping_hash: str
    sampling_plan_hash: str
    anchor_ids: tuple[str, ...]
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    b_primary: int
    m_primary: int
    r_primary: int
    completion_denominator: int
    cost_semantics: CostSemantics
    qualification_gate_hash: str
    execution_evidence_hash: str
    status: str = "FORMAL_FROZEN"
    scope: str = "formal"
    formal_eligible: bool = True
    schema_version: str = FORMAL_MATRIX_SCHEMA

    def __post_init__(self) -> None:
        _safe_id(self.freeze_id, "freeze_id")
        for field, value in (
            ("pilot_report_hash", self.pilot_report_hash),
            ("pilot_mapping_hash", self.pilot_mapping_hash),
            ("sampling_plan_hash", self.sampling_plan_hash),
            ("qualification_gate_hash", self.qualification_gate_hash),
            ("execution_evidence_hash", self.execution_evidence_hash),
        ):
            _hash(value, field)
        if self.scope != "formal" or not self.formal_eligible or self.status != "FORMAL_FROZEN":
            raise FormalRunRejected("S206_FORMAL_MATRIX_NOT_QUALIFIED")
        if self.schema_version != FORMAL_MATRIX_SCHEMA:
            raise ValueError("invalid formal matrix schema")
        if self.b_primary not in PILOT_B_VALUES or self.m_primary not in (4, 8, 16, 32) or self.b_primary % self.m_primary:
            raise ValueError("formal matrix B/M invalid")
        if self.r_primary < 200 or self.r_primary > 1000:
            raise ValueError("formal matrix R outside [200,1000]")
        if self.completion_denominator != len(self.anchor_ids) * self.r_primary:
            raise ValueError("formal matrix completion denominator mismatch")

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "freeze_id": self.freeze_id,
            "scope": self.scope,
            "status": self.status,
            "anchor_ids": list(self.anchor_ids),
            "candidate_evaluations": [item.to_dict() for item in self.candidate_evaluations],
            "b_primary": self.b_primary,
            "m_primary": self.m_primary,
            "r_primary": self.r_primary,
            "completion_denominator": self.completion_denominator,
            "cost_semantics": list((
                "scientific_equal_sample_cost",
                "isolated_estimator_cost",
                "online_training_incremental_cost",
            )),
            "cost_observations": self.cost_semantics.to_dict(),
            "cost_io_quiescent": self.cost_semantics.cost_io_quiescent,
            "pilot_draw_stream": PILOT_STREAM,
            "confirmatory_draw_stream": CONFIRMATORY_STREAM,
            "pilot_report_hash": self.pilot_report_hash,
            "pilot_mapping_hash": self.pilot_mapping_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "execution_evidence_hash": self.execution_evidence_hash,
            "formal_eligible": self.formal_eligible,
            "qualification_gate_hash": self.qualification_gate_hash,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value


def build_g24b_gate(
    report: BlindedPilotReport,
    execution: FormalExecutionEvidence,
    *,
    evidence_refs: Sequence[str],
    checked_at: datetime | None = None,
) -> GateRecord:
    """Create the single-writer G2.4b GateRecord, never an implicit PASS."""

    reasons: list[str] = []
    try:
        execution.require_for_stage(2)
    except FormalRunRejected as error:
        reasons.append(f"UPSTREAM_EXECUTION_NOT_ELIGIBLE:{error}")
    if report.status != "READY_FOR_QUALIFICATION":
        reasons.append(f"PILOT_REPORT_NOT_READY:{report.status}")
    if report.selected is None:
        reasons.append("NO_UNIQUE_SELECTED_B_M")
    if report.selected is not None and report.selected.microbatch_count <= 2:
        reasons.append("PRIMARY_M_MUST_BE_GREATER_THAN_TWO")
    if len(report.anchor_rows) != len(ANCHOR_IDS) * len(PILOT_B_VALUES) * len(PILOT_M_VALUES):
        reasons.append("PILOT_OPERATIONAL_ROW_COUNT_INVALID")
    refs = tuple(dict.fromkeys(str(ref) for ref in evidence_refs if str(ref)))
    if not refs:
        reasons.append("G24B_EVIDENCE_REFS_MISSING")
    now = (checked_at or datetime.now(timezone.utc)).isoformat()
    if reasons:
        return GateRecord(
            gate_id="stage2.G2.4b",
            stage=2,
            status=GateStatus.BLOCKED,
            checked_at=now,
            measured={"pilot_report_hash": report.artifact_hash},
            threshold={"required_status": "READY_FOR_QUALIFICATION"},
            evidence_refs=refs or ("s206-preparation-blocked",),
            reasons=tuple(reasons),
        )
    selected = report.selected
    assert selected is not None
    return GateRecord(
        gate_id="stage2.G2.4b",
        stage=2,
        status=GateStatus.PASS,
        checked_at=now,
        measured={
            "pilot_report_hash": report.artifact_hash,
            "pilot_mapping_hash": report.mapping_hash,
            "operational_row_count": len(report.anchor_rows),
            "selected_batch_size": selected.batch_size,
            "selected_microbatch_count": selected.microbatch_count,
            "required_repetitions": selected.r_required,
        },
        threshold={
            "anchor_count": len(ANCHOR_IDS),
            "candidate_batch_sizes": list(PILOT_B_VALUES),
            "candidate_microbatch_counts": list(PILOT_M_VALUES),
            "pilot_repetitions": PILOT_REPETITIONS,
            "r_min": 200,
            "r_max": 1000,
            "aggregation_overhead_limit": 0.25,
        },
        evidence_refs=refs,
    )


def qualify_formal_matrix(
    report: BlindedPilotReport,
    execution: FormalExecutionEvidence,
    gate: GateRecord,
    *,
    freeze_id: str = "s206-formal-matrix-freeze",
) -> FormalMatrixFreeze:
    """Bind the selected pair/R to an accepted G2.4b GateRecord."""

    execution.require_for_stage(2)
    accepted = require_accepted_gate(gate, stage=2)
    if accepted.gate_id != "stage2.G2.4b":
        raise FormalRunRejected("S206_WRONG_QUALIFICATION_GATE")
    if report.status != "READY_FOR_QUALIFICATION" or report.selected is None:
        raise FormalRunRejected("S206_PILOT_REPORT_NOT_READY")
    selected = report.selected
    if selected.microbatch_count <= 2:
        raise FormalRunRejected("S206_PRIMARY_M_MUST_BE_GREATER_THAN_TWO")
    qualification = ArtifactQualification.from_gate(scope="formal", gate=accepted, stage=2)
    assert qualification.formal_eligible and qualification.qualification_gate_hash
    return FormalMatrixFreeze(
        freeze_id=freeze_id,
        pilot_report_hash=report.artifact_hash,
        pilot_mapping_hash=report.mapping_hash,
        sampling_plan_hash=report.sampling_plan_hash,
        anchor_ids=ANCHOR_IDS,
        candidate_evaluations=report.candidate_evaluations,
        b_primary=selected.batch_size,
        m_primary=selected.microbatch_count,
        r_primary=selected.r_required,
        completion_denominator=len(ANCHOR_IDS) * selected.r_required,
        cost_semantics=report.cost_semantics,
        qualification_gate_hash=qualification.qualification_gate_hash,
        execution_evidence_hash=execution.artifact_hash,
    )


@dataclass(frozen=True, slots=True)
class FormalConfirmatoryCellMapping:
    anchor_id: str
    mappings: tuple[RepetitionMapping, ...]

    def __post_init__(self) -> None:
        if self.anchor_id not in ANCHOR_IDS:
            raise ValueError("confirmatory anchor is not preregistered")
        if not self.mappings:
            raise ValueError("confirmatory cell mappings cannot be empty")
        if any(mapping.m_values[0] != 2 for mapping in self.mappings):
            raise ValueError("confirmatory mappings must retain M=2 equivalence")
        if len({mapping.repetition_id for mapping in self.mappings}) != len(self.mappings):
            raise ValueError("confirmatory repetition IDs must be unique per cell")

    @property
    def draw_ids(self) -> tuple[str, ...]:
        return tuple(draw.draw_id for mapping in self.mappings for draw in mapping.draws)

    @property
    def sample_collision_count(self) -> int:
        draws = tuple(draw for mapping in self.mappings for draw in mapping.draws)
        return len(draws) - len({draw.sample_id for draw in draws})

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "repetition_count": len(self.mappings),
            "mappings": [mapping.to_dict() for mapping in self.mappings],
            "sample_collision_count": self.sample_collision_count,
        }


@dataclass(frozen=True, slots=True)
class FormalConfirmatoryMappingManifest:
    mapping_id: str
    freeze_hash: str
    sampling_plan_hash: str
    pilot_draw_ids: tuple[str, ...]
    cells: tuple[FormalConfirmatoryCellMapping, ...]
    qualification_gate_hash: str
    stream: str = CONFIRMATORY_STREAM
    scope: str = "formal"
    formal_eligible: bool = True
    schema_version: str = FORMAL_CONFIRMATORY_MAPPING_SCHEMA

    def __post_init__(self) -> None:
        _safe_id(self.mapping_id, "mapping_id")
        for field, value in (
            ("freeze_hash", self.freeze_hash),
            ("sampling_plan_hash", self.sampling_plan_hash),
            ("qualification_gate_hash", self.qualification_gate_hash),
        ):
            _hash(value, field)
        if self.scope != "formal" or not self.formal_eligible or self.stream != CONFIRMATORY_STREAM:
            raise FormalRunRejected("S206_CONFIRMATORY_MAPPING_NOT_FORMAL")
        if self.schema_version != FORMAL_CONFIRMATORY_MAPPING_SCHEMA:
            raise ValueError("invalid formal confirmatory mapping schema")
        if tuple(item.anchor_id for item in self.cells) != ANCHOR_IDS:
            raise ValueError("confirmatory mapping must cover all six anchors")
        pilot_ids = set(self.pilot_draw_ids)
        confirmatory_ids = [draw_id for cell in self.cells for draw_id in cell.draw_ids]
        if len(confirmatory_ids) != len(set(confirmatory_ids)):
            raise ValueError("confirmatory draw IDs must be globally unique")
        if pilot_ids.intersection(confirmatory_ids):
            raise ValueError("pilot and confirmatory draw namespaces overlap")

    @property
    def sample_collision_count(self) -> int:
        return sum(cell.sample_collision_count for cell in self.cells)

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        confirmatory_ids = [draw_id for cell in self.cells for draw_id in cell.draw_ids]
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "mapping_id": self.mapping_id,
            "scope": self.scope,
            "stream": self.stream,
            "freeze_hash": self.freeze_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "pilot_draw_id_count": len(self.pilot_draw_ids),
            "pilot_draw_id_hash": canonical_json_hash(list(self.pilot_draw_ids)),
            "confirmatory_draw_id_count": len(confirmatory_ids),
            "confirmatory_draw_id_hash": canonical_json_hash(confirmatory_ids),
            "cells": [cell.to_dict() for cell in self.cells],
            "draw_id_unique": True,
            "sample_id_collision_count": self.sample_collision_count,
            "complete": True,
            "qualification_gate_hash": self.qualification_gate_hash,
            "formal_eligible": self.formal_eligible,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value


def build_formal_confirmatory_mapping(
    matrix: FormalMatrixFreeze,
    sampling: SamplingPlan,
    *,
    pilot_draw_ids: Sequence[str],
    gate: GateRecord,
    mapping_id: str = "s206-formal-confirmatory-mapping",
) -> FormalConfirmatoryMappingManifest:
    """Generate confirmatory draws only after a formal matrix is qualified."""

    accepted = require_accepted_gate(gate, stage=2)
    if accepted.gate_id != "stage2.G2.4b" or accepted.artifact_hash != matrix.qualification_gate_hash:
        raise FormalRunRejected("S206_CONFIRMATORY_GATE_HASH_MISMATCH")
    if not matrix.formal_eligible or matrix.status != "FORMAL_FROZEN":
        raise FormalRunRejected("S206_CONFIRMATORY_MATRIX_NOT_FROZEN")
    if sampling.digest != matrix.sampling_plan_hash:
        raise ValueError("confirmatory sampling plan hash mismatch")
    pilot_ids = tuple(pilot_draw_ids)
    if len(set(pilot_ids)) != len(pilot_ids):
        raise ValueError("pilot draw IDs must be unique")
    cells: list[FormalConfirmatoryCellMapping] = []
    offset = 0
    count_per_anchor = matrix.r_primary * matrix.b_primary
    for anchor_id in ANCHOR_IDS:
        draws = sampling.draws(CONFIRMATORY_STREAM, count_per_anchor, start=offset)
        mappings = tuple(
            RepetitionMapping.create(
                repetition_id=(
                    f"confirm-{anchor_id.replace('.', '-')}-b{matrix.b_primary:03d}-r{index:04d}"
                ),
                draws=draws[index * matrix.b_primary : (index + 1) * matrix.b_primary],
                m_values=(2, matrix.m_primary),
            )
            for index in range(matrix.r_primary)
        )
        cells.append(FormalConfirmatoryCellMapping(anchor_id=anchor_id, mappings=mappings))
        offset += count_per_anchor
    return FormalConfirmatoryMappingManifest(
        mapping_id=mapping_id,
        freeze_hash=matrix.artifact_hash,
        sampling_plan_hash=sampling.digest,
        pilot_draw_ids=pilot_ids,
        cells=tuple(cells),
        qualification_gate_hash=matrix.qualification_gate_hash,
    )


def _load_root_json(root: Path, reference: str, *, field: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise S206PreparationBlocked(f"{field}:INVALID_REFERENCE")
    candidate = (root / reference).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise S206PreparationBlocked(f"{field}:PATH_ESCAPE") from error
    try:
        value = load_canonical_json(candidate)
    except (OSError, TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"{field}:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(value, dict):
        raise S206PreparationBlocked(f"{field}:OBJECT_REQUIRED")
    return candidate.relative_to(root.resolve()).as_posix(), dict(value)


def _validate_hashed_object(value: Mapping[str, Any], *, field: str) -> None:
    if value.get("artifact_hash") != _canonical_hash_without_artifact(value):
        raise S206PreparationBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")


def _validate_g23(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != "stage2-g23-reference-evaluation-v1":
        raise S206PreparationBlocked("G23_SCHEMA_INVALID")
    if value.get("status") != "PASS" or value.get("formal_eligible") is not True:
        raise S206PreparationBlocked("G23_STRICT_PASS_REQUIRED")
    if value.get("required_cell_count") != 6 or value.get("complete_cell_count") != 6:
        raise S206PreparationBlocked("G23_SIX_CELL_COMPLETENESS_REQUIRED")
    cells = value.get("cells")
    if not isinstance(cells, list) or tuple(item.get("cell_id") for item in cells if isinstance(item, Mapping)) != ANCHOR_IDS:
        raise S206PreparationBlocked("G23_ANCHOR_SET_INVALID")
    if any(
        not isinstance(item, Mapping)
        or item.get("status") != "PASS"
        or item.get("formal_eligible") is False
        for item in cells
    ):
        raise S206PreparationBlocked("G23_CELL_PASS_REQUIRED")
    _validate_hashed_object(value, field="G23")


def _validate_g24a(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != "stage2-g24a-formal-evaluation-v1":
        raise S206PreparationBlocked("G24A_SCHEMA_INVALID")
    if value.get("gate_id") != "stage2.G2.4a" or value.get("status") != "PASS" or value.get("formal_eligible") is not True:
        raise S206PreparationBlocked("G24A_STRICT_PASS_REQUIRED")
    if value.get("cell_count") != 6:
        raise S206PreparationBlocked("G24A_SIX_CELL_COMPLETENESS_REQUIRED")
    results = value.get("results")
    if not isinstance(results, list) or len(results) != 6:
        raise S206PreparationBlocked("G24A_RESULTS_INVALID")
    if any(not isinstance(item, Mapping) or item.get("status") != "PASS" or item.get("formal_eligible") is not True for item in results):
        raise S206PreparationBlocked("G24A_CELL_PASS_REQUIRED")
    _validate_hashed_object(value, field="G24A")


def _validate_s204_root(root: Path) -> None:
    components = tuple(anchor_id.replace(".", "__") for anchor_id in ANCHOR_IDS)
    for component in components:
        candidates = sorted((root / component).rglob("final-status.json"))
        complete: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                value = load_canonical_json(candidate)
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(value, Mapping) and value.get("status") == "COMPLETE" and value.get("formal_eligible") is True:
                complete.append(dict(value))
        if len(complete) != 1:
            raise S206PreparationBlocked(f"S204_STATUS_NOT_UNIQUE:{component}:{len(complete)}")
        refs = complete[0].get("artifact_refs")
        if not isinstance(refs, Mapping) or set(refs) != {"reference_result", "reference_convergence_report", "gate_record"}:
            raise S206PreparationBlocked(f"S204_ARTIFACT_SET_INVALID:{component}")


def validate_gpu_inventory(inventory: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate a normalized nvidia-smi inventory without selecting devices."""

    by_uuid = {str(item.get("uuid")): item for item in inventory}
    missing = [uuid for uuid in APPROVED_GPU_UUIDS if uuid not in by_uuid]
    if missing:
        raise S206PreparationBlocked(f"APPROVED_GPU_MISSING:{','.join(missing)}")
    for item in inventory:
        uuid = str(item.get("uuid"))
        pci = str(item.get("pci_bus_id", "")).lower()
        if uuid in APPROVED_GPU_UUIDS and pci == EXCLUDED_PCI.lower():
            raise S206PreparationBlocked("APPROVED_GPU_BOUND_TO_EXCLUDED_PCI")
        if uuid == "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c" or pci == EXCLUDED_PCI.lower():
            if uuid in CELL_GPU_BINDINGS.values():
                raise S206PreparationBlocked("EXCLUDED_GPU_SELECTED")
    return {
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_pci": EXCLUDED_PCI,
        "inventory_count": len(inventory),
        "selected_bindings": dict(CELL_GPU_BINDINGS),
    }


@dataclass(frozen=True, slots=True)
class S206PreflightSpec:
    data_root: Path
    s204_root: str
    g23_ref: str
    g24a_ref: str


def strict_preflight(
    spec: S206PreflightSpec,
    *,
    gpu_inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fail closed before any formal S2.6 process is started."""

    root = spec.data_root.resolve()
    g23_ref, g23 = _load_root_json(root, spec.g23_ref, field="g23_ref")
    g24a_ref, g24a = _load_root_json(root, spec.g24a_ref, field="g24a_ref")
    _validate_g23(g23)
    _validate_g24a(g24a)
    s204_root = (root / spec.s204_root).resolve()
    try:
        s204_root.relative_to(root)
    except ValueError as error:
        raise S206PreparationBlocked("s204_root:PATH_ESCAPE") from error
    _validate_s204_root(s204_root)
    gpu = validate_gpu_inventory(gpu_inventory)
    return {
        "schema_version": "stage2-s206-formal-preflight-v1",
        "status": "READY",
        "formal_eligible": True,
        "g23_ref": g23_ref,
        "g23_hash": str(g23["artifact_hash"]),
        "g24a_ref": g24a_ref,
        "g24a_hash": str(g24a["artifact_hash"]),
        "s204_root": spec.s204_root,
        "anchor_ids": list(ANCHOR_IDS),
        "pilot_batch_sizes": list(PILOT_B_VALUES),
        "pilot_microbatch_counts": list(PILOT_M_VALUES),
        "pilot_repetitions": PILOT_REPETITIONS,
        "confirmatory_draws_generated": False,
        "gpu": gpu,
    }


__all__ = [
    "APPROVED_GPU_UUIDS",
    "ANCHOR_IDS",
    "BLINDED_PILOT_REPORT_SCHEMA",
    "CELL_GPU_BINDINGS",
    "CONFIRMATORY_STREAM",
    "EXCLUDED_PCI",
    "FORMAL_CONFIRMATORY_MAPPING_SCHEMA",
    "FORMAL_MATRIX_SCHEMA",
    "G24B_GATE_SCHEMA",
    "GLOBAL_PILOT_MAPPING_SCHEMA",
    "GlobalPilotMappingManifest",
    "BlindPilotMeasurement",
    "BlindedPilotReport",
    "FormalConfirmatoryCellMapping",
    "FormalConfirmatoryMappingManifest",
    "FormalMatrixFreeze",
    "PilotCellMapping",
    "PILOT_B_VALUES",
    "PILOT_M_VALUES",
    "PILOT_REPETITIONS",
    "S206PreflightSpec",
    "S206PreparationBlocked",
    "build_formal_confirmatory_mapping",
    "build_global_pilot_mapping",
    "build_g24b_gate",
    "qualify_formal_matrix",
    "reduce_blinded_pilot",
    "strict_preflight",
    "validate_gpu_inventory",
]
