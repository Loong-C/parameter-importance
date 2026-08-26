"""Strict G2.6-to-S2.9 accuracy sidecar production.

S2.9's Pareto reducer intentionally consumes a small, stable metric shape.  A
G2.6 statistics summary is not accepted directly because its ranking metrics
are nested and its U method retains the ``u_m<M>`` name.  This module performs
that conversion only from a complete, hash-bound, formal G2.6 output bundle.
"""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from .stage2_s204_ids import EXPECTED_CELL_IDS


S209_ACCURACY_SCHEMA = "stage2-s209-g27a-accuracy-sidecar-v1"
S208_ANALYSIS_SCHEMA = "stage2-s208-g26-analysis-v1"
S208_GATE_SCHEMA = "stage2-s208-g26-gate-v1"
S208_QUALITY_SCHEMA = "stage2-s208-quality-gates-v1"
S208_FAMILY_SCHEMA = "stage2-s208-confirmatory-family-v1"
S208_HYPOTHESIS_SCHEMA = "stage2-s208-hypothesis-decisions-v1"
S208_LINEAGE_SCHEMA = "stage2-s208-lineage-v1"

_SOURCES = {
    "g26_gate": "g2.6-gate.json",
    "quality_gates": "quality_gates.json",
    "hypothesis_decisions": "hypothesis_decisions.json",
    "statistics_long_table": "statistics_long_table.json",
    "statistics_summary": "statistics_summary.json",
    "family_decisions": "confirmatory_family_decisions.json",
    "lineage_manifest": "lineage_manifest.json",
}


class S209AccuracyBlocked(RuntimeError):
    """Raised when G2.6 evidence cannot safely authorize the sidecar."""


def _safe_ref(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise S209AccuracyBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise S209AccuracyBlocked(f"{field}:PATH_ESCAPE")
    target = (root / Path(*logical.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise S209AccuracyBlocked(f"{field}:PATH_ESCAPE") from error
    return target


def _load(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = load_canonical_json(path)
    except Exception as error:
        raise S209AccuracyBlocked(f"{field}:CANONICAL_READ_FAILED") from error
    if not isinstance(value, Mapping):
        raise S209AccuracyBlocked(f"{field}:OBJECT_REQUIRED")
    return dict(value)


def _identity(value: Mapping[str, Any], *, field: str) -> tuple[str, str]:
    """Verify declared hash or bind legacy hash-less objects canonically."""

    declared = value.get("artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    computed = canonical_json_hash(body if declared is not None else value)
    if declared is not None:
        if not isinstance(declared, str) or len(declared) != 64 or declared != computed:
            raise S209AccuracyBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
        return declared, "artifact_hash"
    # G2.6 hypothesis decisions were published before artifact_hash became
    # mandatory.  The complete canonical object is still an immutable input.
    return computed, "canonical_content"


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise S209AccuracyBlocked(f"{field}:NUMBER_REQUIRED")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise S209AccuracyBlocked(f"{field}:NUMBER_REQUIRED") from error
    if not math.isfinite(result):
        raise S209AccuracyBlocked(f"{field}:NONFINITE")
    return result


def _source_map(root_ref: str) -> dict[str, str]:
    logical = PurePosixPath(root_ref)
    return {
        name: (logical / filename).as_posix()
        for name, filename in _SOURCES.items()
    }


def _validate_g26_bundle(
    values: Mapping[str, Mapping[str, Any]],
    *,
    identities: Mapping[str, str],
) -> None:
    gate = values["g26_gate"]
    if gate.get("schema_version") != S208_GATE_SCHEMA or gate.get("gate_id") != "stage2.G2.6" or gate.get("status") != "PASS" or gate.get("quality_gate_dependency") is not True:
        raise S209AccuracyBlocked("G26_GATE_PASS_FORMAL_REQUIRED")
    quality = values["quality_gates"]
    if quality.get("schema_version") != S208_QUALITY_SCHEMA or quality.get("gate_id") != "stage2.G2.6" or quality.get("status") != "PASS" or quality.get("formal_eligible") is not True:
        raise S209AccuracyBlocked("G26_QUALITY_PASS_FORMAL_REQUIRED")
    records = quality.get("gates")
    if not isinstance(records, list) or not records or any(not isinstance(item, Mapping) or item.get("status") != "PASS" for item in records):
        raise S209AccuracyBlocked("G26_QUALITY_RECORDS_INCOMPLETE")
    family = values["family_decisions"]
    if family.get("schema_version") != S208_FAMILY_SCHEMA or family.get("primary_cells") != list(EXPECTED_CELL_IDS):
        raise S209AccuracyBlocked("G26_FAMILY_SCHEMA_OR_CELL_ORDER_INVALID")
    hypothesis = values["hypothesis_decisions"]
    if hypothesis.get("schema_version") != S208_HYPOTHESIS_SCHEMA or hypothesis.get("quality_gate_dependency") != "passed":
        raise S209AccuracyBlocked("G26_HYPOTHESIS_PASS_REQUIRED")
    lineage = values["lineage_manifest"]
    if lineage.get("schema_version") != S208_LINEAGE_SCHEMA or lineage.get("producer_task") != "stage2.08_statistics_and_robustness":
        raise S209AccuracyBlocked("G26_LINEAGE_SCHEMA_INVALID")
    derived = lineage.get("derived_artifacts")
    # The producer's lineage list names derived analysis artifacts; the gate
    # itself is published alongside that list but is not listed as derived.
    required_derived = set(_SOURCES.values()) - {"g2.6-gate.json"}
    if not isinstance(derived, list) or not required_derived.issubset(set(derived)):
        raise S209AccuracyBlocked("G26_LINEAGE_DERIVED_ARTIFACTS_INCOMPLETE")
    measured = gate.get("measured")
    quality_metrics = quality.get("gates")
    if not isinstance(measured, Mapping):
        raise S209AccuracyBlocked("G26_GATE_MEASURED_REQUIRED")
    if isinstance(quality_metrics, list) and not all(isinstance(item, Mapping) for item in quality_metrics):
        raise S209AccuracyBlocked("G26_QUALITY_RECORD_INVALID")
    # The lineage and quality/audit objects must agree on the frozen source
    # identities whenever those fields are present.
    for field in ("matrix_hash", "raw_manifest_hash"):
        lineage_value = lineage.get(field)
        measured_value = measured.get(field)
        if lineage_value is not None and measured_value is not None and lineage_value != measured_value:
            raise S209AccuracyBlocked(f"G26_{field.upper()}_LINEAGE_MISMATCH")
    if identities.get("lineage_manifest") is None:
        raise S209AccuracyBlocked("G26_LINEAGE_HASH_REQUIRED")


def _method_name(value: Any, *, microbatch_count: int | None) -> str:
    if value in {"raw", "double"}:
        return str(value)
    expected = f"u_m{microbatch_count}" if microbatch_count is not None else None
    if expected is not None and value == expected:
        return "u"
    raise S209AccuracyBlocked(f"G26_ACCURACY_METHOD_UNKNOWN:{value}")


def _accuracy_rows(
    summary: Mapping[str, Any],
    long_table: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summary_rows = summary.get("rows")
    long_rows = long_table.get("rows")
    if not isinstance(summary_rows, list) or not summary_rows:
        raise S209AccuracyBlocked("G26_STATISTICS_SUMMARY_ROWS_REQUIRED")
    if not isinstance(long_rows, list) or not long_rows:
        raise S209AccuracyBlocked("G26_STATISTICS_LONG_ROWS_REQUIRED")
    methods = {(str(row.get("cell_id")), str(row.get("method"))) for row in summary_rows if isinstance(row, Mapping)}
    if len(summary_rows) != len(methods) or any(cell not in EXPECTED_CELL_IDS for cell, _ in methods):
        raise S209AccuracyBlocked("G26_STATISTICS_SUMMARY_DUPLICATE_OR_UNKNOWN_CELL")
    by_cell: dict[str, list[Mapping[str, Any]]] = {cell: [] for cell in EXPECTED_CELL_IDS}
    inferred_m: int | None = None
    for row in summary_rows:
        if not isinstance(row, Mapping):
            raise S209AccuracyBlocked("G26_STATISTICS_SUMMARY_ROW_INVALID")
        cell = row.get("cell_id")
        if cell not in EXPECTED_CELL_IDS:
            raise S209AccuracyBlocked(f"G26_ACCURACY_CELL_UNKNOWN:{cell}")
        microbatch = row.get("microbatch_count")
        if isinstance(microbatch, int) and not isinstance(microbatch, bool) and microbatch > 0:
            if inferred_m is None:
                inferred_m = microbatch
            elif inferred_m != microbatch:
                raise S209AccuracyBlocked("G26_MICROBATCH_COUNT_DRIFT")
        by_cell[str(cell)].append(row)
    if any(len(items) != 3 for items in by_cell.values()):
        raise S209AccuracyBlocked("G26_ACCURACY_METHOD_SET_INCOMPLETE")
    ranking: dict[tuple[str, str], Mapping[str, Any]] = {}
    bias: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in long_rows:
        if not isinstance(row, Mapping):
            raise S209AccuracyBlocked("G26_STATISTICS_LONG_ROW_INVALID")
        cell = row.get("cell_id")
        if cell not in EXPECTED_CELL_IDS:
            raise S209AccuracyBlocked(f"G26_LONG_CELL_UNKNOWN:{cell}")
        method = _method_name(row.get("method"), microbatch_count=inferred_m)
        key = (str(cell), method)
        view = row.get("reference_view")
        if view == "ranking_repetition_mean":
            if key in ranking:
                raise S209AccuracyBlocked(f"G26_LONG_RANKING_DUPLICATE:{key}")
            ranking[key] = row
        elif view == "bias":
            if key in bias:
                raise S209AccuracyBlocked(f"G26_LONG_BIAS_DUPLICATE:{key}")
            bias[key] = row
    output: list[dict[str, Any]] = []
    for cell in EXPECTED_CELL_IDS:
        for source in by_cell[cell]:
            method = _method_name(source.get("method"), microbatch_count=inferred_m)
            key = (cell, method)
            rank = ranking.get(key)
            bias_row = bias.get(key)
            if rank is None or bias_row is None:
                raise S209AccuracyBlocked(f"G26_LONG_METRIC_ROW_MISSING:{cell}:{method}")
            ranking_mean = source.get("ranking_repeat_mean")
            if not isinstance(ranking_mean, Mapping):
                raise S209AccuracyBlocked(f"G26_RANKING_SUMMARY_MISSING:{cell}:{method}")
            if ranking.get(key, {}).get("spearman") != ranking_mean.get("spearman") or ranking.get(key, {}).get("overlap_at_0.01") != ranking_mean.get("overlap_at_0.01"):
                raise S209AccuracyBlocked(f"G26_RANKING_SUMMARY_LONG_MISMATCH:{cell}:{method}")
            if bias_row.get("mse_observed") != source.get("mse_observed"):
                raise S209AccuracyBlocked(f"G26_MSE_SUMMARY_LONG_MISMATCH:{cell}:{method}")
            row = {
                "row_id": f"{cell}:{method}",
                "cell_id": cell,
                "method": method,
                "corrected_nmse": _finite(source.get("corrected_nmse"), field=f"accuracy.{cell}.{method}.corrected_nmse"),
                "mse": _finite(source.get("mse_observed"), field=f"accuracy.{cell}.{method}.mse"),
                "spearman": _finite(ranking_mean.get("spearman"), field=f"accuracy.{cell}.{method}.spearman"),
                "overlap_1pct": _finite(ranking_mean.get("overlap_at_0.01"), field=f"accuracy.{cell}.{method}.overlap_1pct"),
            }
            if not -1 <= row["spearman"] <= 1 or not 0 <= row["overlap_1pct"] <= 1:
                raise S209AccuracyBlocked(f"G26_RANKING_METRIC_DOMAIN_INVALID:{cell}:{method}")
            output.append(row)
    if {(row["cell_id"], row["method"]) for row in output} != {(cell, method) for cell in EXPECTED_CELL_IDS for method in ("raw", "double", "u")}:
        raise S209AccuracyBlocked("G26_ACCURACY_COMPLETE_CELL_METHOD_GRID_REQUIRED")
    return output


def produce_s209_accuracy_sidecar(
    *,
    data_root: str | Path,
    g26_root_ref: str,
    output_ref: str,
    run_id: str,
) -> dict[str, Any]:
    """Produce one immutable S2.9 accuracy sidecar from a sealed G2.6 root."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise S209AccuracyBlocked("RUN_ID_REQUIRED")
    root = Path(data_root).resolve()
    g26_root = _safe_ref(root, g26_root_ref, field="g26_root_ref")
    output = _safe_ref(root, output_ref, field="output_ref")
    refs = _source_map(g26_root_ref)
    values = {name: _load(_safe_ref(root, ref, field=name), field=name) for name, ref in refs.items()}
    identity_details = {name: _identity(value, field=name) for name, value in values.items()}
    identities = {name: details[0] for name, details in identity_details.items()}
    _validate_g26_bundle(values, identities=identities)
    summary = values["statistics_summary"]
    long_table = values["statistics_long_table"]
    for name, payload in (("statistics_summary", summary), ("statistics_long_table", long_table)):
        if payload.get("schema_version") != S208_ANALYSIS_SCHEMA:
            raise S209AccuracyBlocked(f"{name}:SCHEMA_INVALID")
    rows = _accuracy_rows(summary, long_table)
    body: dict[str, Any] = {
        "schema_version": S209_ACCURACY_SCHEMA,
        "task_id": "stage2.09_cost_and_system_validation",
        "run_id": run_id,
        "source_g26_root_ref": g26_root_ref,
        "source_artifacts": {
            name: {"ref": refs[name], "hash": identities[name], "hash_kind": identity_details[name][1]}
            for name in sorted(refs)
        },
        "rows": rows,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = _load(output, field="output")
        if existing != body:
            raise S209AccuracyBlocked("OUTPUT_ALREADY_EXISTS_WITH_DIFFERENT_CONTENT")
    else:
        write_canonical_json(output, body)
    return body


__all__ = [
    "S209_ACCURACY_SCHEMA",
    "S209AccuracyBlocked",
    "produce_s209_accuracy_sidecar",
]
