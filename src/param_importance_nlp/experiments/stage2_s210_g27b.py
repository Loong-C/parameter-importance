"""Detached S2.10/G2.7b visualization, reporting, and estimator decision.

This module is deliberately a consumer, not an experiment runner.  It reads
sealed S2.8 and S2.9 JSON artifacts, verifies their content identities, and
publishes a new append-only report directory.  Chart artifacts are
source-backed specifications (the optional renderer is intentionally outside
the formal decision path), so a chart can never hide a failed unit or silently
recompute a statistic from raw tensors.

The public entry point is :func:`run_s210_g27b`.  Malformed or forged inputs
raise :class:`S210G27BBlocked`; valid but blocked upstream gates result in a
machine-readable BLOCKED report and never a formal estimator conclusion.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.status import GateRecord, GateStatus
from .stage2_s204_ids import EXPECTED_CELL_IDS


S210_SCHEMA = "stage2-s210-g27b-report-v1"
S210_GATE_SCHEMA = "stage2-s210-g27b-gate-v1"
S210_DECISION_SCHEMA = "estimator-decision-v1"
S210_SOURCE_SCHEMA = "stage2-s210-source-table-v1"
S210_CHART_SCHEMA = "stage2-s210-chart-spec-v1"
S210_LINEAGE_SCHEMA = "stage2-s210-lineage-v1"
S210_TASK_ID = "stage2.10_visualization_reporting_and_decision"
S210_GATE_ID = "stage2.G2.7b"
S210_PRIMARY_CELLS = tuple(EXPECTED_CELL_IDS)
S210_METHODS = ("raw", "double", "u")
S210_COST_RATIO = 1.25
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class S210G27BBlocked(RuntimeError):
    """Raised for malformed, forged, or identity-inconsistent inputs."""


def _finite(value: Any, *, field: str = "value") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise S210G27BBlocked(f"{field}:NONFINITE")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, field=f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise S210G27BBlocked(f"{field}:NON_STRING_KEY")
            _finite(item, field=f"{field}.{key}")
        return
    raise S210G27BBlocked(f"{field}:NOT_JSON_VALUE")


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S210G27BBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _load(value: Mapping[str, Any] | str | Path | None, *, field: str) -> dict[str, Any]:
    if value is None:
        raise S210G27BBlocked(f"{field}:INPUT_REQUIRED")
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        try:
            loaded = load_canonical_json(value)
        except Exception as error:  # pragma: no cover - stable public error below
            raise S210G27BBlocked(f"{field}:CANONICAL_READ_FAILED") from error
        if not isinstance(loaded, Mapping):
            raise S210G27BBlocked(f"{field}:OBJECT_REQUIRED")
        payload = dict(loaded)
    _finite(payload, field=field)
    return payload


def _body(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in payload.items() if key != "artifact_hash"}


def _identity(payload: Mapping[str, Any], *, field: str, required: bool = True) -> str:
    """Verify declared artifact_hash, returning the content identity.

    A few S2.8 producer objects (notably ``hypothesis_decisions.json``) were
    specified before artifact_hash became mandatory.  They are still bound to
    their canonical content hash; all producer objects that declare a hash are
    checked against it and never trusted merely because their status is PASS.
    """

    declared = payload.get("artifact_hash")
    if declared is None:
        if required:
            raise S210G27BBlocked(f"{field}:ARTIFACT_HASH_REQUIRED")
        return canonical_json_hash(payload)
    digest = _sha(declared, field=f"{field}.artifact_hash")
    if digest != canonical_json_hash(_body(payload)):
        raise S210G27BBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return digest


def _write_once(root: Path, name: str, value: Mapping[str, Any]) -> str:
    target = root / name
    if target.exists():
        raise S210G27BBlocked(f"OUTPUT_ALREADY_EXISTS:{name}")
    write_canonical_json(target, value)
    return name


def _write_bytes_once(root: Path, name: str, payload: bytes) -> str:
    target = root / name
    if target.exists():
        raise S210G27BBlocked(f"OUTPUT_ALREADY_EXISTS:{name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return name


@dataclass(frozen=True)
class _GateView:
    gate_id: str
    status: GateStatus
    measured: Mapping[str, Any]
    artifact_hash: str

    def effective_status(self) -> GateStatus:
        return self.status


def _gate(value: Mapping[str, Any] | str | Path | None, *, gate_id: str, field: str) -> tuple[_GateView, str]:
    payload = _load(value, field=field)
    # S2.8 predates the generic GateRecord and publishes a producer-specific
    # gate schema.  It remains accepted only as this exact, content-addressed
    # PASS/BLOCKED view; no generic status is inferred from report text.
    if payload.get("schema_version") == "stage2-s208-g26-gate-v1":
        if payload.get("gate_id") != gate_id or payload.get("stage") != 2:
            raise S210G27BBlocked(f"{field}:GATE_ID_MISMATCH")
        digest = _identity(payload, field=field)
        status = payload.get("status")
        if status not in {item.value for item in GateStatus}:
            raise S210G27BBlocked(f"{field}:STATUS_INVALID")
        measured = payload.get("measured")
        if not isinstance(measured, Mapping):
            raise S210G27BBlocked(f"{field}:MEASURED_REQUIRED")
        return _GateView(gate_id, GateStatus(status), dict(measured), digest), digest
    try:
        record = GateRecord.from_mapping(payload)
    except Exception as error:
        raise S210G27BBlocked(f"{field}:INVALID_GATE_RECORD") from error
    if record.gate_id != gate_id:
        raise S210G27BBlocked(f"{field}:GATE_ID_MISMATCH")
    digest = _identity(payload, field=field)
    return _GateView(record.gate_id, record.status, dict(record.measured) if isinstance(record.measured, Mapping) else {}, digest), digest


@dataclass(frozen=True)
class _Input:
    name: str
    schema_version: str
    payload: Mapping[str, Any]
    content_hash: str

    def descriptor(self, *, row_count: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "frozen": True,
        }
        if row_count is not None:
            result["row_count"] = int(row_count)
        return result


def _input(
    value: Mapping[str, Any] | str | Path | None,
    *,
    name: str,
    field: str,
    required_hash: bool = True,
) -> _Input:
    payload = _load(value, field=field)
    content_hash = _identity(payload, field=field, required=required_hash)
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise S210G27BBlocked(f"{field}:SCHEMA_VERSION_REQUIRED")
    return _Input(name, schema, payload, content_hash)


def _as_rows(value: Mapping[str, Any], *, field: str) -> list[dict[str, Any]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise S210G27BBlocked(f"{field}:ROWS_REQUIRED")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise S210G27BBlocked(f"{field}.rows[{index}]:OBJECT_REQUIRED")
        _finite(row, field=f"{field}.rows[{index}]")
        result.append(dict(row))
    return result


def _require_upstream_pass(
    quality: _Input,
    family: _Input,
    g26_gate: GateRecord,
    cost: _Input,
    g27a_gate: GateRecord,
) -> list[str]:
    reasons: list[str] = []
    if g26_gate.effective_status() is not GateStatus.PASS:
        reasons.append("G2.6_NOT_PASS")
    if g27a_gate.effective_status() is not GateStatus.PASS:
        reasons.append("G2.7A_NOT_PASS")
    if quality.payload.get("status") != "PASS" or quality.payload.get("formal_eligible") is not True:
        reasons.append("G2.6_QUALITY_NOT_FORMAL_PASS")
    checks = quality.payload.get("gates")
    if not isinstance(checks, list) or not checks or any(
        not isinstance(item, Mapping) or item.get("status") != "PASS" for item in checks
    ):
        reasons.append("G2.6_QUALITY_GATE_SET_INCOMPLETE")
    if family.payload.get("primary_cells") != list(S210_PRIMARY_CELLS):
        reasons.append("G2.6_PRIMARY_CELL_SET_MISMATCH")
    if family.payload.get("multiplicity") != "intersection_union_across_six_primary_cells":
        reasons.append("G2.6_INTERSECTION_UNION_REQUIRED")
    if cost.payload.get("status") != "PASS" or cost.payload.get("formal_eligible") is not True:
        reasons.append("G2.7A_COST_NOT_FORMAL_PASS")
    if cost.payload.get("cost_io_quiescent") is not True:
        reasons.append("G2.7A_COST_IO_NOT_QUIESCENT")
    four_gpu = cost.payload.get("four_gpu_anchor")
    if not isinstance(four_gpu, Mapping) or four_gpu.get("status") != "PASS":
        reasons.append("G2.7A_FOUR_GPU_ANCHOR_REQUIRED")
    health = cost.payload.get("health_snapshot")
    if not isinstance(health, Mapping) or health.get("healthy") is not True or health.get("cost_io_quiescent") is not True:
        reasons.append("G2.7A_HEALTH_SNAPSHOT_NOT_PASS")
    pareto = cost.payload.get("pareto")
    if not isinstance(pareto, Mapping) or pareto.get("status") != "PASS":
        reasons.append("G2.7A_PARETO_NOT_PASS")
    consistency = cost.payload.get("consistency")
    if not isinstance(consistency, Mapping) or consistency.get("all_pass") is not True:
        reasons.append("G2.7A_CONSISTENCY_NOT_PASS")
    return sorted(set(reasons))


def _method_id(method: Any, *, field: str) -> str:
    if method in {"raw", "double"}:
        return str(method)
    if isinstance(method, str) and re.fullmatch(r"u_m[0-9]+", method):
        if int(method[3:]) < 2:
            raise S210G27BBlocked(f"{field}:MICROBATCH_COUNT_INVALID")
        return method
    raise S210G27BBlocked(f"{field}:METHOD_INVALID")


def _validate_statistics(
    table: _Input,
    *,
    expected_b: int,
    expected_m: int,
) -> tuple[list[dict[str, Any]], str]:
    rows = _as_rows(table.payload, field=table.name)
    seen: set[tuple[str, str, str]] = set()
    cells: set[str] = set()
    methods: set[str] = set()
    for index, row in enumerate(rows):
        cell = row.get("cell_id")
        if cell not in S210_PRIMARY_CELLS:
            raise S210G27BBlocked(f"{table.name}.rows[{index}]:PRIMARY_CELL_REQUIRED")
        if not isinstance(row.get("model"), str) or not isinstance(row.get("training_stage"), str):
            raise S210G27BBlocked(f"{table.name}.rows[{index}]:MODEL_STAGE_REQUIRED")
        method = _method_id(row.get("method"), field=f"{table.name}.rows[{index}].method")
        batch = row.get("batch_size")
        micro = row.get("microbatch_count")
        if isinstance(batch, bool) or not isinstance(batch, int) or batch != expected_b:
            raise S210G27BBlocked(f"{table.name}.rows[{index}]:PRIMARY_BATCH_MISMATCH")
        if isinstance(micro, bool) or not isinstance(micro, int) or micro != expected_m:
            raise S210G27BBlocked(f"{table.name}.rows[{index}]:PRIMARY_MICROBATCH_MISMATCH")
        view = row.get("reference_view")
        if not isinstance(view, str) or not view:
            raise S210G27BBlocked(f"{table.name}.rows[{index}]:REFERENCE_VIEW_REQUIRED")
        key = (str(cell), method, view)
        if key in seen:
            # Repetition aggregates intentionally have a separate view name;
            # duplicate keys would make a chart non-reproducible.
            raise S210G27BBlocked(f"{table.name}.rows[{index}]:DUPLICATE_SOURCE_KEY")
        seen.add(key)
        cells.add(str(cell))
        methods.add(method)
        if row.get("scope") != "parameter":
            raise S210G27BBlocked(f"{table.name}.rows[{index}]:PARAMETER_SCOPE_REQUIRED")
    if cells != set(S210_PRIMARY_CELLS):
        raise S210G27BBlocked(f"{table.name}:SIX_PRIMARY_CELLS_REQUIRED")
    u_methods = {method for method in methods if method.startswith("u_m")}
    if methods != {"raw", "double"} | u_methods or u_methods != {f"u_m{expected_m}"}:
        raise S210G27BBlocked(f"{table.name}:PRIMARY_METHOD_SET_MISMATCH")
    return rows, table.content_hash


def _extract_primary(matrix: Mapping[str, Any], family: Mapping[str, Any], cost: Mapping[str, Any]) -> tuple[int, int]:
    b = family.get("b_primary")
    m = family.get("m_primary")
    if isinstance(b, bool) or not isinstance(b, int) or b <= 0:
        raise S210G27BBlocked("G2.6_FAMILY_B_PRIMARY_REQUIRED")
    if isinstance(m, bool) or not isinstance(m, int) or m < 2:
        raise S210G27BBlocked("G2.6_FAMILY_M_PRIMARY_REQUIRED")
    if isinstance(matrix, Mapping) and matrix:
        for key, expected in (("b_primary", b), ("m_primary", m)):
            if key in matrix and matrix[key] != expected:
                raise S210G27BBlocked(f"MATRIX_{key.upper()}_MISMATCH")
    frozen = cost.get("frozen_inputs")
    if isinstance(frozen, Mapping):
        if frozen.get("batch_size") != b or frozen.get("microbatch_count") != m:
            raise S210G27BBlocked("G2.7A_PRIMARY_B_M_MISMATCH")
    return b, m


def _family_qualification(family: Mapping[str, Any], *, method: str, expected_m: int) -> tuple[bool, list[str]]:
    method_id = method if method != "u" else f"u_m{expected_m}"
    global_rows = family.get("global")
    if not isinstance(global_rows, Mapping):
        return False, [f"{method.upper()}_GLOBAL_QUALIFICATION_MISSING"]
    value = global_rows.get(method_id)
    if not isinstance(value, Mapping) or value.get("bias_qualified") is not True:
        return False, [f"{method.upper()}_BIAS_NOT_QUALIFIED"]
    rows = family.get("rows")
    if not isinstance(rows, list):
        return False, ["G2.6_FAMILY_ROWS_MISSING"]
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("method") == method_id]
    if len(selected) != 18 or {row.get("cell_id") for row in selected} != set(S210_PRIMARY_CELLS):
        return False, [f"{method.upper()}_FAMILY_ROWS_INCOMPLETE"]
    if any(row.get("state") != "PASS" for row in selected):
        return False, [f"{method.upper()}_BIAS_ENDPOINT_NOT_PASS"]
    return True, []


def _u_noninferiority(family: Mapping[str, Any], *, expected_m: int) -> tuple[bool, list[str]]:
    method_id = f"u_m{expected_m}"
    rows = family.get("noninferiority_rows")
    expected_endpoints = {
        "corrected_parameter_nmse_noninferiority",
        "parameter_spearman_noninferiority",
        "parameter_overlap_at_1_percent_noninferiority",
    }
    if not isinstance(rows, list):
        return False, ["U_NONINFERIORITY_ROWS_MISSING"]
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("method") == method_id]
    if len(selected) != 18:
        return False, ["U_NONINFERIORITY_FAMILY_INCOMPLETE"]
    if {row.get("cell_id") for row in selected} != set(S210_PRIMARY_CELLS) or {row.get("endpoint") for row in selected} != expected_endpoints:
        return False, ["U_NONINFERIORITY_ENDPOINT_SET_INCOMPLETE"]
    if any(row.get("state") != "PASS" for row in selected):
        return False, ["U_NONINFERIORITY_NOT_PASS"]
    global_rows = family.get("noninferiority_global")
    if isinstance(global_rows, Mapping) and any(
        not isinstance(global_rows.get(endpoint), Mapping) or global_rows[endpoint].get("all_cells") is not True
        for endpoint in expected_endpoints
    ):
        return False, ["U_NONINFERIORITY_GLOBAL_NOT_PASS"]
    return True, []


def _online_cost_ok(cost: Mapping[str, Any], *, method: str) -> tuple[bool, list[str]]:
    online = cost.get("online_training_incremental_cost")
    ratios = online.get("ratios") if isinstance(online, Mapping) else None
    if not isinstance(ratios, Mapping) or ratios.get("source") != "online_training_incremental_cost":
        return False, ["ONLINE_COST_RATIOS_MISSING"]
    if ratios.get("threshold") != S210_COST_RATIO:
        return False, ["ONLINE_COST_THRESHOLD_NOT_FROZEN"]
    methods = ratios.get("methods")
    method_id = method if method != "u" else "u"
    entry = methods.get(method_id) if isinstance(methods, Mapping) else None
    if not isinstance(entry, Mapping) or not entry:
        return False, [f"ONLINE_COST_{method.upper()}_MISSING"]
    values: list[float] = []
    for key, value in entry.items():
        if key == "source":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False, [f"ONLINE_COST_{method.upper()}_NONFINITE"]
        values.append(float(value))
    if not values or any(value > S210_COST_RATIO for value in values):
        return False, [f"ONLINE_COST_{method.upper()}_OVER_BOUND"]
    return True, []


def _value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_hash: str,
    source_run_id: str,
    predicate: Any,
    table_name: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not predicate(row):
            continue
        output = dict(row)
        output["effect"] = _value(row, "signed_bias", "absolute_bias", "mse_observed", "corrected_nmse", "slope")
        bootstrap = row.get("bootstrap")
        if isinstance(bootstrap, Mapping):
            output["interval_lower"] = _value(bootstrap, "quantile_0.05", "quantile_0.025")
            output["interval_upper"] = _value(bootstrap, "quantile_0.95", "quantile_0.975")
        else:
            output["interval_lower"] = _value(row, "interval_lower")
            output["interval_upper"] = _value(row, "interval_upper")
        output["source_artifact_hash"] = source_hash
        output["source_run_id"] = row.get("run_id", source_run_id)
        output["source_table"] = table_name
        selected.append(output)
    if not selected:
        raise S210G27BBlocked(f"{table_name}:EMPTY_SOURCE_TABLE")
    return selected


def _provenance_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    microbatch_count: int,
    default_scope: str,
    default_aggregate: str,
) -> list[dict[str, Any]]:
    """Ensure every source row carries the review-time comparison identity."""

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        cell = item.get("cell_id")
        if isinstance(cell, str) and ":" in cell:
            model, stage = cell.split(":", 1)
            item.setdefault("model", model)
            item.setdefault("training_stage", stage)
            item.setdefault("checkpoint", stage)
        item.setdefault("batch_size", batch_size)
        item.setdefault("microbatch_count", microbatch_count)
        item.setdefault("scope", default_scope)
        item.setdefault("aggregate", default_aggregate)
        item.setdefault("repetition", None)
        item.setdefault("method", "raw" if default_scope == "raw_calibration" else item.get("method"))
        result.append(item)
    return result


def _table(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    body = {
        "schema_version": S210_SOURCE_SCHEMA,
        "source_name": name,
        "rows": [dict(row) for row in rows],
        "row_count": len(rows),
        "source_backed": True,
    }
    return body | {"artifact_hash": canonical_json_hash(body)}


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _table_csv(table: Mapping[str, Any]) -> bytes:
    rows = table["rows"]
    assert isinstance(rows, list)
    columns: list[str] = []
    for row in rows:
        assert isinstance(row, Mapping)
        for key in row:
            if key not in columns:
                columns.append(str(key))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    return buffer.getvalue().encode("utf-8")


def _table_markdown(table: Mapping[str, Any]) -> bytes:
    rows = table["rows"]
    assert isinstance(rows, list)
    columns: list[str] = []
    for row in rows:
        assert isinstance(row, Mapping)
        for key in row:
            if key not in columns:
                columns.append(str(key))
    def cell(value: Any) -> str:
        return _csv_value(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(key)) for key in columns) + " |" for row in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _chart(chart_id: str, table: Mapping[str, Any], *, x: str, ys: Sequence[str], chart_type: str = "line") -> dict[str, Any]:
    rows = table.get("rows")
    assert isinstance(rows, list)
    run_ids = sorted({str(row.get("source_run_id")) for row in rows if row.get("source_run_id") is not None})
    body = {
        "schema_version": S210_CHART_SCHEMA,
        "chart_id": chart_id,
        "source_name": table["source_name"],
        "source_schema_version": table["schema_version"],
        "source_hash": table["artifact_hash"],
        "chart_type": chart_type,
        "x_column": x,
        "y_columns": list(ys),
        "filters": [],
        "sort_columns": [x],
        "sort_descending": False,
        "source_row_count": len(rows),
        "source_run_ids": run_ids,
        "renderer": "source-spec-v1",
        "rebuild_rule": "read_source_table_only",
    }
    return body | {"artifact_hash": canonical_json_hash(body)}


def _decision(
    *,
    b: int,
    m: int,
    repetitions: int,
    family: Mapping[str, Any],
    cost: Mapping[str, Any],
    upstream_reasons: Sequence[str],
    upstream: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reasons = list(upstream_reasons)
    double_ok, double_reasons = _family_qualification(family, method="double", expected_m=m)
    u_ok, u_reasons = _family_qualification(family, method="u", expected_m=m)
    u_noninf, noninf_reasons = _u_noninferiority(family, expected_m=m)
    double_cost, double_cost_reasons = _online_cost_ok(cost, method="double")
    u_cost, u_cost_reasons = _online_cost_ok(cost, method="u")
    reasons.extend(double_reasons + u_reasons)
    if not u_noninf:
        reasons.extend(noninf_reasons)

    selected: str | None = None
    rule = "return_to_stage1_or_reference"
    follow_up: list[str] = []
    if not upstream_reasons:
        if double_ok and u_ok:
            if u_noninf and u_cost:
                selected = "u"
                rule = "u_primary_double_calibration"
            elif double_cost:
                selected = "double"
                rule = "double_primary_u_noninferiority_or_cost_failed"
            else:
                reasons.extend(double_cost_reasons + u_cost_reasons)
        elif double_ok and double_cost:
            selected = "double"
            rule = "double_primary_only_qualified"
        elif u_ok and u_cost:
            selected = "u"
            rule = "u_provisional_revalidate_double"
            follow_up.append("revalidate_double_before_claiming_full_calibration")
        else:
            reasons.extend(double_cost_reasons + u_cost_reasons)
    if selected is None and not reasons:
        reasons.append("NO_ESTIMATOR_CANDIDATE_QUALIFIED")

    status = "SELECTED" if selected is not None and not upstream_reasons else "BLOCKED"
    decision_body: dict[str, Any] = {
        "schema_version": S210_DECISION_SCHEMA,
        "decision_id": "stage2-s210-estimator-decision",
        "selected_estimator": selected,
        "scope": "formal",
        "status": status,
        "state": "SELECTED" if selected is not None else "UNFROZEN",
        "batch_size": b if selected is not None else None,
        "microbatch_count": m if selected is not None else None,
        "repetitions": repetitions if selected is not None else None,
        "gate_id": S210_GATE_ID,
        "gate_status": "PASS" if selected is not None else "BLOCKED",
        "artifact_ref": "estimator_decision.json" if selected is not None else None,
        "metadata": {
            "decision_rule": rule,
            "qualified_estimators": [method for method, ok in (("double", double_ok), ("u", u_ok)) if ok],
            "bias_qualified": {"double": double_ok, "u": u_ok},
            "u_noninferiority_pass": u_noninf,
            "online_cost_within_1_25": {"double": double_cost, "u": u_cost},
            "required_primary_cells": list(S210_PRIMARY_CELLS),
            "confirmatory_family": "intersection_union_across_six_primary_cells",
            "upstream_artifacts": dict(upstream),
            "reasons": sorted(set(reasons)),
            "follow_up": follow_up,
        },
    }
    decision_body["artifact_hash"] = canonical_json_hash(decision_body)
    explanation = {
        "schema_version": "stage2-s210-decision-explanation-v1",
        "decision_rule": rule,
        "selected_estimator": selected,
        "primary": {"batch_size": b, "microbatch_count": m, "repetitions": repetitions},
        "checks": {
            "double_bias_qualified": double_ok,
            "u_bias_qualified": u_ok,
            "u_noninferiority": u_noninf,
            "double_online_cost": double_cost,
            "u_online_cost": u_cost,
        },
        "reasons": sorted(set(reasons)),
        "follow_up": follow_up,
        "formal_conclusion": selected is not None,
        "artifact_hash": "",
    }
    explanation["artifact_hash"] = canonical_json_hash(_body(explanation))
    return decision_body, explanation


def _render_report_markdown(report: Mapping[str, Any], decision: Mapping[str, Any]) -> bytes:
    metadata = decision.get("metadata") if isinstance(decision.get("metadata"), Mapping) else {}
    selected = decision.get("selected_estimator") or "none"
    reasons = metadata.get("reasons", []) if isinstance(metadata, Mapping) else []
    lines = [
        "# Stage 2.10 estimator decision report",
        "",
        f"- status: `{report.get('status')}`",
        f"- formal_eligible: `{report.get('formal_eligible')}`",
        f"- selected_estimator: `{selected}`",
        f"- decision_rule: `{metadata.get('decision_rule', 'unknown')}`",
        "",
        "## Machine decision",
        "",
        "The machine-readable `estimator_decision.json` is authoritative; this text is a rendered view.",
        "",
        f"- batch_size: `{decision.get('batch_size')}`",
        f"- microbatch_count: `{decision.get('microbatch_count')}`",
        f"- repetitions: `{decision.get('repetitions')}`",
        f"- required primary cells: `{len(metadata.get('required_primary_cells', [])) if isinstance(metadata, Mapping) else 0}`",
        "",
        "## Blockers and follow-up",
        "",
    ]
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    follow_up = metadata.get("follow_up", []) if isinstance(metadata, Mapping) else []
    if follow_up:
        lines.extend(f"- follow-up: `{item}`" for item in follow_up)
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            "This report concerns the fixed-state local gradient target `mu^2`; it is not a path-integral contribution and does not claim an actual AdamW update effect.",
            "",
            "All figures are source-backed specifications and can be rebuilt from the listed source tables.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_s210_g27b(
    *,
    g26_analysis: Mapping[str, Any] | str | Path | None = None,
    g26_gate: Mapping[str, Any] | str | Path | None = None,
    g26_quality_gates: Mapping[str, Any] | str | Path | None = None,
    g26_hypothesis_decisions: Mapping[str, Any] | str | Path | None = None,
    g26_statistics_long_table: Mapping[str, Any] | str | Path | None = None,
    g26_statistics_summary: Mapping[str, Any] | str | Path | None = None,
    g26_raw_calibration: Mapping[str, Any] | str | Path | None = None,
    g26_family_decisions: Mapping[str, Any] | str | Path | None = None,
    g27a_report: Mapping[str, Any] | str | Path | None = None,
    g27a_gate: Mapping[str, Any] | str | Path | None = None,
    matrix: Mapping[str, Any] | str | Path | None = None,
    output_root: str | Path | None = None,
    run_id: str = "s210-g27b",
    checked_at: str | None = None,
    **aliases: Any,
) -> dict[str, Any]:
    """Consume sealed G2.6/G2.7a evidence and publish S2.10 artifacts.

    ``g26_analysis`` and ``g27a_report`` may be complete in-memory reports;
    individual artifact arguments are preferred for server use.  A new
    ``output_root`` must be empty (or absent), preventing accidental overwrite
    of an earlier analysis.
    """

    # Accept descriptive aliases used by detached server wrappers without
    # weakening the required source identities.
    g26_quality_gates = g26_quality_gates or aliases.pop("quality_gates", None)
    g26_hypothesis_decisions = g26_hypothesis_decisions or aliases.pop("hypothesis_decisions", None)
    g26_statistics_long_table = g26_statistics_long_table or aliases.pop("statistics_long_table", None)
    g26_statistics_summary = g26_statistics_summary or aliases.pop("statistics_summary", None)
    g26_raw_calibration = g26_raw_calibration or aliases.pop("raw_calibration", None)
    g26_family_decisions = g26_family_decisions or aliases.pop("confirmatory_family_decisions", None)
    g27a_report = g27a_report or aliases.pop("cost_report", None)
    g27a_gate = g27a_gate or aliases.pop("cost_gate", None)
    if aliases:
        raise S210G27BBlocked(f"UNSUPPORTED_ARGUMENTS:{','.join(sorted(aliases))}")
    if not run_id or not isinstance(run_id, str):
        raise S210G27BBlocked("RUN_ID_REQUIRED")

    analysis = _load(g26_analysis, field="g26_analysis") if g26_analysis is not None else {}
    cost = _load(g27a_report, field="g27a_report") if g27a_report is not None else {}
    if not g26_quality_gates and analysis:
        g26_quality_gates = analysis.get("quality_gates")
    if not g26_hypothesis_decisions and analysis:
        g26_hypothesis_decisions = analysis.get("hypothesis_decisions")
    if not g26_statistics_long_table and analysis:
        source = analysis.get("statistics_long_table")
        if isinstance(source, list):
            g26_statistics_long_table = {"schema_version": S210_SOURCE_SCHEMA, "rows": source, "artifact_hash": canonical_json_hash({"schema_version": S210_SOURCE_SCHEMA, "rows": source})}
        else:
            g26_statistics_long_table = source
    if not g26_statistics_summary and analysis:
        source = analysis.get("statistics_summary")
        if isinstance(source, list):
            g26_statistics_summary = {"schema_version": S210_SOURCE_SCHEMA, "rows": source, "artifact_hash": canonical_json_hash({"schema_version": S210_SOURCE_SCHEMA, "rows": source})}
        else:
            g26_statistics_summary = source
    if not g26_family_decisions and analysis:
        g26_family_decisions = analysis.get("confirmatory_family_decisions")
    if not g26_raw_calibration and analysis:
        source = analysis.get("raw_calibration")
        if isinstance(source, list):
            g26_raw_calibration = {"schema_version": S210_SOURCE_SCHEMA, "rows": source, "artifact_hash": canonical_json_hash({"schema_version": S210_SOURCE_SCHEMA, "rows": source})}
        else:
            g26_raw_calibration = source
    if not g26_gate and analysis:
        g26_gate = analysis.get("g2_6_gate")
    if not g27a_gate and cost:
        g27a_gate = cost.get("gate")

    quality = _input(g26_quality_gates, name="g26_quality_gates", field="g26_quality_gates")
    hypothesis = _input(g26_hypothesis_decisions, name="g26_hypothesis_decisions", field="g26_hypothesis_decisions", required_hash=False)
    long_table = _input(g26_statistics_long_table, name="g26_statistics_long_table", field="g26_statistics_long_table")
    family = _input(g26_family_decisions, name="g26_family_decisions", field="g26_family_decisions")
    summary = None
    if g26_statistics_summary is not None:
        summary = _input(g26_statistics_summary, name="g26_statistics_summary", field="g26_statistics_summary", required_hash=False)
    cost_input = _input(g27a_report, name="g27a_report", field="g27a_report")
    g26_record, g26_gate_hash = _gate(g26_gate, gate_id="stage2.G2.6", field="g26_gate")
    g27a_record, g27a_gate_hash = _gate(g27a_gate, gate_id="stage2.G2.7a", field="g27a_gate")
    hypothesis_hash = hypothesis.content_hash  # retained in lineage; family rows drive the decision

    family_payload = family.payload
    b, m = _extract_primary(_load(matrix, field="matrix") if matrix is not None else {}, family_payload, cost_input.payload)
    stat_rows, _ = _validate_statistics(long_table, expected_b=b, expected_m=m)
    repetitions = max(int(row["repetitions"]) for row in stat_rows if isinstance(row.get("repetitions"), int) and row["repetitions"] > 0)
    if repetitions < 2:
        raise S210G27BBlocked("STATISTICS_REPETITIONS_LT_TWO")
    reasons = _require_upstream_pass(quality, family, g26_record, cost_input, g27a_record)
    nested_cost_gate = cost_input.payload.get("gate")
    if isinstance(nested_cost_gate, Mapping):
        nested_hash = _identity(nested_cost_gate, field="g27a_report.gate")
        if nested_hash != g27a_gate_hash:
            reasons.append("G2.7A_NESTED_GATE_HASH_MISMATCH")

    # Cross-stage content bindings.  If either producer exposes the shared
    # matrix/raw identities, they must agree; absence is not silently filled.
    g26_measured = g26_record.measured if isinstance(g26_record.measured, Mapping) else {}
    frozen = cost_input.payload.get("frozen_inputs")
    if isinstance(frozen, Mapping) and isinstance(g26_measured, Mapping):
        if g26_measured.get("matrix_hash") is not None and frozen.get("matrix_hash") is not None and g26_measured.get("matrix_hash") != frozen.get("matrix_hash"):
            reasons.append("MATRIX_HASH_CROSS_STAGE_MISMATCH")
        if g26_measured.get("raw_manifest_hash") is not None and frozen.get("raw_manifest_hash") is not None and g26_measured.get("raw_manifest_hash") != frozen.get("raw_manifest_hash"):
            reasons.append("RAW_MANIFEST_HASH_CROSS_STAGE_MISMATCH")
    reasons = sorted(set(reasons))

    source_run_id = str(cost_input.payload.get("run_id") or g26_measured.get("raw_manifest_hash") or long_table.content_hash)
    tables: dict[str, dict[str, Any]] = {}
    tables["bias_variance_mse"] = _table(
        "bias_variance_mse",
        _provenance_rows(_source_rows(stat_rows, source_hash=long_table.content_hash, source_run_id=source_run_id, predicate=lambda row: row.get("reference_view") == "bias", table_name="bias_variance_mse"), batch_size=b, microbatch_count=m, default_scope="parameter", default_aggregate="repetition_mean"),
    )
    tables["ranking"] = _table(
        "ranking",
        _provenance_rows(_source_rows(stat_rows, source_hash=long_table.content_hash, source_run_id=source_run_id, predicate=lambda row: row.get("reference_view") in {"ranking", "ranking_repetition_mean"}, table_name="ranking"), batch_size=b, microbatch_count=m, default_scope="parameter", default_aggregate="repetition_mean"),
    )
    tables["negative_value"] = _table(
        "negative_value",
        _provenance_rows(_source_rows(stat_rows, source_hash=long_table.content_hash, source_run_id=source_run_id, predicate=lambda row: row.get("reference_view") == "bias", table_name="negative_value"), batch_size=b, microbatch_count=m, default_scope="parameter", default_aggregate="repetition_mean"),
    )
    calibration_payload = analysis.get("raw_calibration") if isinstance(analysis, Mapping) else None
    if g26_raw_calibration is not None:
        calibration_input = _input(g26_raw_calibration, name="g26_raw_calibration", field="g26_raw_calibration", required_hash=True)
        calibration_rows = _as_rows(calibration_input.payload, field="g26_raw_calibration")
    elif isinstance(calibration_payload, list) and calibration_payload:
        calibration_input = None
        calibration_rows = [dict(row) for row in calibration_payload if isinstance(row, Mapping)]
    else:
        calibration_input = None
        calibration_rows = []
    if calibration_rows:
        calibration_hash = calibration_input.content_hash if calibration_input is not None else long_table.content_hash
        tables["raw_calibration"] = _table("raw_calibration", _provenance_rows(_source_rows(calibration_rows, source_hash=calibration_hash, source_run_id=source_run_id, predicate=lambda _: True, table_name="raw_calibration"), batch_size=b, microbatch_count=m, default_scope="raw_calibration", default_aggregate="cell"))
    else:
        tables["raw_calibration"] = _table("raw_calibration", _provenance_rows([{"cell_id": cell, "status": "NOT_PRESENT_IN_G26_SOURCE", "source_artifact_hash": long_table.content_hash, "source_run_id": source_run_id, "source_table": "raw_calibration"} for cell in S210_PRIMARY_CELLS], batch_size=b, microbatch_count=m, default_scope="raw_calibration", default_aggregate="cell"))
        reasons.append("G2.6_RAW_CALIBRATION_SOURCE_MISSING")
    pareto = cost_input.payload.get("pareto")
    pareto_rows = pareto.get("rows") if isinstance(pareto, Mapping) else None
    if not isinstance(pareto_rows, list) or not pareto_rows:
        raise S210G27BBlocked("G2.7A_PARETO_ROWS_REQUIRED")
    pareto_source = [dict(row) for row in pareto_rows if isinstance(row, Mapping)]
    if not pareto_source:
        raise S210G27BBlocked("G2.7A_PARETO_ROWS_INVALID")
    tables["cost_pareto"] = _table("cost_pareto", _provenance_rows(_source_rows(pareto_source, source_hash=cost_input.content_hash, source_run_id=str(cost_input.payload.get("run_id", source_run_id)), predicate=lambda _: True, table_name="cost_pareto"), batch_size=b, microbatch_count=m, default_scope="online_training_incremental_cost", default_aggregate="pareto"))

    upstream = {
        "g26_gate": g26_gate_hash,
        "g26_quality_gates": quality.content_hash,
        "g26_hypothesis_decisions": hypothesis_hash,
        "g26_statistics_long_table": long_table.content_hash,
        "g26_family_decisions": family.content_hash,
        "g27a_report": cost_input.content_hash,
        "g27a_gate": g27a_gate_hash,
    }
    decision, explanation = _decision(b=b, m=m, repetitions=repetitions, family=family_payload, cost=cost_input.payload, upstream_reasons=reasons, upstream=upstream)
    final_status = "PASS" if decision["selected_estimator"] is not None and decision["gate_status"] == "PASS" else "BLOCKED"
    formal_eligible = final_status == "PASS"
    charts: list[dict[str, Any]] = []
    if final_status == "PASS":
        charts = [
            _chart("bias-variance-mse", tables["bias_variance_mse"], x="batch_size", ys=("signed_bias", "variance_s2", "mse_observed")),
            _chart("ranking-quality", tables["ranking"], x="batch_size", ys=("spearman", "pearson")),
            _chart("signed-negative-diagnostic", tables["negative_value"], x="batch_size", ys=("negative_fraction", "negative_mass", "positive_mass")),
            _chart("raw-calibration", tables["raw_calibration"], x="batch_size", ys=("slope", "intercept")),
            _chart("cost-pareto", tables["cost_pareto"], x="wall_seconds", ys=("corrected_nmse", "spearman"), chart_type="scatter"),
        ]
    source_descriptors = [
        {"name": name, "schema_version": value["schema_version"], "content_hash": value["artifact_hash"], "row_count": value["row_count"], "frozen": True}
        for name, value in sorted(tables.items())
    ]
    lineage_body = {
        "schema_version": S210_LINEAGE_SCHEMA,
        "task_id": S210_TASK_ID,
        "run_id": run_id,
        "producer_commit": "record_at_execution",
        "consumer_commit": "record_at_execution",
        "input_artifacts": upstream,
        "source_tables": source_descriptors,
        "chart_specs": [chart["artifact_hash"] for chart in charts],
        "decision_hash": decision["artifact_hash"],
        "formal_eligible": formal_eligible,
    }
    lineage = lineage_body | {"artifact_hash": canonical_json_hash(lineage_body)}
    report_body = {
        "schema_version": S210_SCHEMA,
        "task_id": S210_TASK_ID,
        "report_id": run_id,
        "status": final_status,
        "formal_eligible": formal_eligible,
        "primary": {"batch_size": b, "microbatch_count": m, "repetitions": repetitions, "cells": list(S210_PRIMARY_CELLS)},
        "upstream_artifacts": upstream,
        "source_tables": source_descriptors,
        "chart_specs": charts,
        "decision_ref": "estimator_decision.json",
        "decision_hash": decision["artifact_hash"],
        "lineage_ref": "lineage_manifest.json",
        "limitations": ["fixed_state_mu_squared_target_only", "conditional_on_frozen_empirical_distribution", "path_integral_and_actual_adamw_update_out_of_scope"],
        "reasons": sorted(set(explanation["reasons"])),
    }
    report = report_body | {"artifact_hash": canonical_json_hash(report_body)}
    gate_measured = {
        "g26_gate_hash": g26_gate_hash,
        "g27a_gate_hash": g27a_gate_hash,
        "g26_quality_hash": quality.content_hash,
        "g26_family_hash": family.content_hash,
        "g27a_report_hash": cost_input.content_hash,
        "decision_hash": decision["artifact_hash"],
        "source_table_count": len(tables),
        "chart_spec_count": len(charts),
        "formal_eligible": formal_eligible,
    }
    gate_status = GateStatus.PASS if final_status == "PASS" else GateStatus.BLOCKED
    gate = GateRecord(
        gate_id=S210_GATE_ID,
        stage=2,
        status=gate_status,
        checked_at=checked_at or datetime.now(timezone.utc).isoformat(),
        measured=gate_measured,
        threshold={"required_upstream": ["stage2.G2.6", "stage2.G2.7a"], "six_primary_cell_intersection_union": True, "source_backed_visualization": True, "formal_provisional_forbidden": True},
        evidence_refs=("estimator_decision.json", "report.json", "lineage_manifest.json"),
        reasons=tuple(explanation["reasons"]) if gate_status is not GateStatus.PASS else (),
    )
    if output_root is not None:
        destination = Path(output_root)
        if destination.exists() and any(destination.iterdir()):
            raise S210G27BBlocked("OUTPUT_ROOT_MUST_BE_NEW_AND_EMPTY")
        destination.mkdir(parents=True, exist_ok=False) if not destination.exists() else None
        files: list[str] = []
        files.append(_write_once(destination, "estimator_decision.json", decision))
        files.append(_write_once(destination, "decision_explanation.json", explanation))
        files.append(_write_once(destination, "g2.7b-gate.json", gate.to_dict()))
        files.append(_write_once(destination, "report.json", report))
        files.append(_write_once(destination, "lineage_manifest.json", lineage))
        files.append(_write_bytes_once(destination, "report.md", _render_report_markdown(report, decision)))
        for name, table in sorted(tables.items()):
            files.append(_write_once(destination, f"sources/{name}.json", table))
            files.append(_write_bytes_once(destination, f"sources/{name}.csv", _table_csv(table)))
            files.append(_write_bytes_once(destination, f"sources/{name}.md", _table_markdown(table)))
        for chart in charts:
            files.append(_write_once(destination, f"charts/{chart['chart_id']}.json", chart))
        report["output_files"] = files
    else:
        report["output_files"] = []
    report["analysis_hash"] = canonical_json_hash(report)
    return {"status": final_status, "formal_eligible": formal_eligible, "report": report, "decision": decision, "gate": gate.to_dict(), "lineage": lineage, "tables": tables, "charts": charts, "output_files": report["output_files"], "analysis_hash": report["analysis_hash"]}


orchestrate_s210_g27b = run_s210_g27b
validate_g27b = run_s210_g27b


__all__ = [
    "S210_CHART_SCHEMA",
    "S210_DECISION_SCHEMA",
    "S210_GATE_ID",
    "S210_GATE_SCHEMA",
    "S210_LINEAGE_SCHEMA",
    "S210_SCHEMA",
    "S210_SOURCE_SCHEMA",
    "S210G27BBlocked",
    "orchestrate_s210_g27b",
    "run_s210_g27b",
    "validate_g27b",
]
