"""Stage 3 formal reporting materialization.

This module is deliberately a small, data-only boundary.  It accepts frozen
tables and the hash-bound Stage 3.07 raw path-result payload, writes reproducible
JSON/CSV tables, and renders each declared chart in both PNG and SVG.  It never
creates observations or fills missing values; malformed or incomplete raw
results fail closed before any formal reporting payload is returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import io
from pathlib import Path
import re

import numpy as np

from ..analysis import (
    AnalysisReport,
    ChartArtifact,
    ChartSpec,
    FrozenSourceTable,
    render_matplotlib_chart,
)
from ..atomic import atomic_write_bytes
from ..contracts.jsonio import JSONValue, canonical_json_bytes, write_canonical_json
from .stage3_raw_storage import derive_candidate_views, load_raw_aggregate


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _file_record(root: Path, path: Path) -> dict[str, JSONValue]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError("STAGE3_REPORTING_FILE_OUTSIDE_WORKSPACE") from error
    payload = resolved.read_bytes()
    return {
        "path": relative,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _csv_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json_bytes(value).decode("utf-8")
    return str(value)


def _table_csv(table: FrozenSourceTable) -> bytes:
    columns = sorted({str(column) for row in table.rows for column in row})
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in table.rows:
        writer.writerow([_csv_scalar(row.get(column)) for column in columns])
    return buffer.getvalue().encode("utf-8")


def _safe_table_name(name: str) -> str:
    if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
        raise ValueError("STAGE3_REPORTING_TABLE_NAME_INVALID")
    return name


def _path_vector_rows(
    raw: Mapping[str, object],
) -> tuple[list[dict[str, JSONValue]], tuple[str, ...]]:
    signed = raw.get("signed")
    if not isinstance(signed, Mapping) or not signed:
        raise ValueError("STAGE3_REPORTING_RAW_SIGNED_VECTOR_MISSING")
    rows: list[dict[str, JSONValue]] = []
    coordinate_ids: list[str] = []
    positive = raw.get("positive")
    negative = raw.get("negative_mass")
    absolute = raw.get("absolute")
    def vector(value: object) -> list[float]:
        if isinstance(value, np.ndarray):
            result = value.reshape(-1).tolist()
        else:
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None and isinstance(value, torch.Tensor):
                result = value.detach().cpu().to(torch.float64).reshape(-1).tolist()
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                result = list(value)
            else:
                raise ValueError("STAGE3_REPORTING_RAW_SIGNED_VECTOR_INVALID")
        return [float(item) for item in result]

    for parameter, values in sorted(signed.items()):
        if not isinstance(parameter, str):
            raise ValueError("STAGE3_REPORTING_RAW_SIGNED_VECTOR_INVALID")
        signed_values = vector(values)
        arrays: dict[str, Sequence[object]] = {"signed": signed_values}
        for key, source in (("positive", positive), ("negative_mass", negative), ("absolute", absolute)):
            if isinstance(source, Mapping) and parameter in source:
                candidate = vector(source[parameter])
                if len(candidate) != len(signed_values):
                    raise ValueError("STAGE3_REPORTING_RAW_VECTOR_SHAPE_MISMATCH")
                arrays[key] = candidate
        for index, signed_value in enumerate(signed_values):
            numeric = float(signed_value)
            if not np.isfinite(numeric):
                raise ValueError("STAGE3_REPORTING_RAW_VECTOR_NONFINITE")
            coordinate = f"{parameter}[{index}]"
            coordinate_ids.append(coordinate)
            row: dict[str, JSONValue] = {
                "coordinate_id": coordinate,
                "parameter": parameter,
                "coordinate_index": index,
                "signed": numeric,
            }
            for key, array in arrays.items():
                if key == "signed":
                    continue
                value = float(array[index])
                if not np.isfinite(value):
                    raise ValueError("STAGE3_REPORTING_RAW_VECTOR_NONFINITE")
                row[key] = value
            rows.append(row)
    return rows, tuple(coordinate_ids)


def _candidate_results(raw: Mapping[str, object]) -> list[tuple[str, str, Mapping[str, object]]]:
    candidates = raw.get("candidate_results")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("STAGE3_REPORTING_RAW_CANDIDATES_MISSING")
    active_unit = raw.get("active_unit_id", "formal")
    if not isinstance(active_unit, str) or not active_unit:
        raise ValueError("STAGE3_REPORTING_RAW_UNIT_INVALID")
    result: list[tuple[str, str, Mapping[str, object]]] = []
    # S3.07's final matrix payload is unit-local.  A future aggregate may
    # additionally wrap the same wire under unit IDs; support both forms while
    # keeping the raw payload itself as the authority.
    for name, value in sorted(candidates.items()):
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ValueError("STAGE3_REPORTING_RAW_CANDIDATE_INVALID")
        if isinstance(value.get("candidate_results"), Mapping):
            nested = value["candidate_results"]
            for nested_name, nested_value in sorted(nested.items()):
                if not isinstance(nested_name, str) or not isinstance(nested_value, Mapping):
                    raise ValueError("STAGE3_REPORTING_RAW_NESTED_CANDIDATE_INVALID")
                result.append((name, nested_name, nested_value))
        else:
            result.append((active_unit, name, value))
    if not result:
        raise ValueError("STAGE3_REPORTING_RAW_CANDIDATES_MISSING")
    return result


def _aggregate_reporting_tables(
    raw: Mapping[str, object],
    *,
    workspace_root: Path,
) -> tuple[FrozenSourceTable, FrozenSourceTable]:
    aggregate_ref = raw.get("raw_aggregate_ref")
    aggregate_hash = raw.get("raw_aggregate_hash")
    if not isinstance(aggregate_ref, str) or not isinstance(aggregate_hash, str):
        raise ValueError("STAGE3_REPORTING_RAW_AGGREGATE_BINDING_MISSING")
    aggregate, loaded = load_raw_aggregate(
        root=workspace_root,
        aggregate_ref=aggregate_ref,
        aggregate_hash=aggregate_hash,
        require_complete=True,
    )
    vector_rows: list[dict[str, JSONValue]] = []
    curve_rows: list[dict[str, JSONValue]] = []
    for unit_id in aggregate["required_unit_ids"]:  # type: ignore[index]
        shard, state, _bundle = loaded[unit_id]
        shard_entry = aggregate["unit_shards"][unit_id]  # type: ignore[index]
        reference = state["reference"]
        reference_signed = reference["signed"]  # type: ignore[index]
        ref_arrays = {
            name: np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=np.float64).reshape(-1)
            for name, value in reference_signed.items()  # type: ignore[union-attr]
        }
        summaries = shard["rule_summaries"]
        candidates = state["candidates"]
        derived_by_rule = {
            str(rule_name): derive_candidate_views(candidate["signed"])  # type: ignore[index]
            for rule_name, candidate in candidates.items()  # type: ignore[union-attr]
        }
        for parameter_index, parameter in enumerate(sorted(ref_arrays)):
            reference_values = ref_arrays[parameter]
            if reference_values.size == 0 or not np.all(np.isfinite(reference_values)):
                raise ValueError("STAGE3_REPORTING_RAW_REFERENCE_VECTOR_INVALID")
            for rule_name in aggregate["candidate_rule_names"]:  # type: ignore[index]
                candidate = candidates[rule_name]  # type: ignore[index]
                signed_value = candidate["signed"][parameter]  # type: ignore[index]
                values = np.asarray(signed_value.detach().cpu().numpy() if hasattr(signed_value, "detach") else signed_value, dtype=np.float64).reshape(-1)
                if values.shape != reference_values.shape or not np.all(np.isfinite(values)):
                    raise ValueError("STAGE3_REPORTING_RAW_COORDINATE_ID_DRIFT")
                delta = values - reference_values
                row: dict[str, JSONValue] = {
                    "unit_id": str(unit_id),
                    "rule_name": str(rule_name),
                    "parameter": parameter,
                    "parameter_index": parameter_index,
                    "coordinate_count": int(values.size),
                    "signed_mean": float(values.mean()),
                    "signed_l1": float(np.abs(values).sum()),
                    "reference_mean": float(reference_values.mean()),
                    "reference_l1": float(np.abs(reference_values).sum()),
                    "signed_error_mean": float(delta.mean()),
                    "signed_error_l1": float(np.abs(delta).sum()),
                    "max_abs_error": float(np.abs(delta).max()),
                    "raw_shard_ref": str(shard_entry["shard_ref"]),
                    "raw_bundle_ref": str(shard["bundle_ref"]),
                }
                derived_views = derived_by_rule[str(rule_name)]
                for mass_name in ("positive", "negative_mass", "absolute"):
                    mass = derived_views[mass_name]
                    if parameter in mass:
                        mass_value = mass[parameter]
                        mass_array = np.asarray(
                            mass_value.detach().cpu().numpy()
                            if hasattr(mass_value, "detach")
                            else mass_value,
                            dtype=np.float64,
                        ).reshape(-1)
                        if mass_array.shape != values.shape or not np.all(np.isfinite(mass_array)):
                            raise ValueError("STAGE3_REPORTING_RAW_VECTOR_SHAPE_MISMATCH")
                        row[f"{mass_name}_l1"] = float(np.abs(mass_array).sum())
                vector_rows.append(row)
        for rule_name in aggregate["candidate_rule_names"]:  # type: ignore[index]
            summary = summaries[rule_name]  # type: ignore[index]
            if not isinstance(summary, Mapping):
                raise ValueError("STAGE3_REPORTING_RAW_RULE_SUMMARY_INVALID")
            alphas = summary.get("node_alphas")
            losses = summary.get("node_losses")
            if not isinstance(alphas, list) or not isinstance(losses, list) or len(alphas) != len(losses) or not alphas:
                raise ValueError("STAGE3_REPORTING_RAW_PATH_CURVE_MISSING")
            for alpha, loss in zip(alphas, losses):
                if loss is None or not np.isfinite(float(alpha)) or not np.isfinite(float(loss)):
                    raise ValueError("STAGE3_REPORTING_RAW_PATH_CURVE_NONFINITE")
                curve_rows.append({"unit_id": str(unit_id), "rule_name": str(rule_name), "alpha": float(alpha), "loss": float(loss)})
    vector_rows.sort(key=lambda row: (str(row["unit_id"]), str(row["rule_name"]), str(row["parameter"])))
    curve_rows.sort(key=lambda row: (str(row["unit_id"]), str(row["rule_name"]), float(row["alpha"])))
    return (
        FrozenSourceTable.from_rows(
            name="stage3_formal_parameter_vectors",
            schema_version="stage3-formal-parameter-summary-table-v1",
            rows=vector_rows,
        ),
        FrozenSourceTable.from_rows(
            name="stage3_formal_path_curves",
            schema_version="stage3-formal-path-curve-table-v1",
            rows=curve_rows,
        ),
    )


def build_raw_reporting_tables(
    raw: Mapping[str, object], *, workspace_root: Path | None = None
) -> tuple[FrozenSourceTable, FrozenSourceTable]:
    """Validate raw vectors and expose parameter/path-curve tables.

    The function intentionally requires node losses for every candidate.  A
    formal report without the measured path curve would silently claim a
    diagnostic that was never recorded.
    """

    if raw.get("schema_version") != "stage3-task-path-results-v1" or raw.get("scope") != "formal":
        raise ValueError("STAGE3_REPORTING_RAW_SCOPE_INVALID")
    if raw.get("raw_aggregate_ref") is not None:
        if workspace_root is None:
            raise ValueError("STAGE3_REPORTING_RAW_WORKSPACE_ROOT_REQUIRED")
        return _aggregate_reporting_tables(raw, workspace_root=workspace_root.resolve())
    reference = raw.get("independent_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("STAGE3_REPORTING_RAW_REFERENCE_MISSING")
    reference_rows, reference_ids = _path_vector_rows(reference)
    vector_rows: list[dict[str, JSONValue]] = []
    curve_rows: list[dict[str, JSONValue]] = []
    for unit_id, rule_name, candidate in _candidate_results(raw):
        rows, candidate_ids = _path_vector_rows(candidate)
        if candidate_ids != reference_ids:
            raise ValueError("STAGE3_REPORTING_RAW_COORDINATE_ID_DRIFT")
        for row in rows:
            row = dict(row)
            row.update({"unit_id": unit_id, "rule_name": rule_name})
            vector_rows.append(row)
        rule = candidate.get("rule")
        if not isinstance(rule, Mapping):
            raise ValueError("STAGE3_REPORTING_RAW_RULE_MISSING")
        nodes = rule.get("nodes")
        losses = candidate.get("node_losses")
        if not isinstance(nodes, list) or not isinstance(losses, list) or len(nodes) != len(losses) or not nodes:
            raise ValueError("STAGE3_REPORTING_RAW_PATH_CURVE_MISSING")
        for alpha, loss in zip(nodes, losses):
            if loss is None:
                raise ValueError("STAGE3_REPORTING_RAW_PATH_LOSS_UNDEFINED")
            alpha_value, loss_value = float(alpha), float(loss)
            if not np.isfinite(alpha_value) or not np.isfinite(loss_value):
                raise ValueError("STAGE3_REPORTING_RAW_PATH_CURVE_NONFINITE")
            curve_rows.append({"unit_id": unit_id, "rule_name": rule_name, "alpha": alpha_value, "loss": loss_value})
    # Reference rows are deliberately not discarded: the report's lineage
    # records the coordinate basis against which every candidate is compared.
    vector_rows.sort(key=lambda row: (str(row["unit_id"]), str(row["rule_name"]), str(row["coordinate_id"])))
    curve_rows.sort(key=lambda row: (str(row["unit_id"]), str(row["rule_name"]), float(row["alpha"])))
    vector_table = FrozenSourceTable.from_rows(
        name="stage3_formal_parameter_vectors",
        schema_version="stage3-formal-parameter-vector-table-v1",
        rows=vector_rows,
    )
    curve_table = FrozenSourceTable.from_rows(
        name="stage3_formal_path_curves",
        schema_version="stage3-formal-path-curve-table-v1",
        rows=curve_rows,
    )
    return vector_table, curve_table


def write_reporting_bundle(
    *,
    workspace_root: Path,
    output_root: Path,
    tables: Sequence[FrozenSourceTable],
    raw_formal_results: Mapping[str, object],
    report: AnalysisReport,
    chart_specs: Sequence[ChartSpec],
) -> Mapping[str, object]:
    """Write the reproducible small-file bundle and return its file bindings."""

    base = output_root / "reporting"
    base.mkdir(parents=True, exist_ok=True)
    table_records: list[dict[str, JSONValue]] = []
    for table in tables:
        stem = _safe_table_name(table.name)
        json_path = base / "tables" / f"{stem}.json"
        csv_path = base / "tables" / f"{stem}.csv"
        write_canonical_json(json_path, table.to_dict())
        atomic_write_bytes(csv_path, _table_csv(table))
        table_records.append({
            "name": table.name,
            "schema_version": table.schema_version,
            "content_hash": table.content_hash,
            "json": _file_record(workspace_root, json_path),
            "csv": _file_record(workspace_root, csv_path),
        })

    raw_path = base / "tables" / "stage3_formal_path_results.raw.json"
    write_canonical_json(raw_path, dict(raw_formal_results))
    raw_shard_manifest_record: Mapping[str, object] | None = None
    if raw_formal_results.get("raw_aggregate_ref") is not None:
        aggregate, _loaded = load_raw_aggregate(
            root=workspace_root.resolve(),
            aggregate_ref=raw_formal_results.get("raw_aggregate_ref"),
            aggregate_hash=raw_formal_results.get("raw_aggregate_hash"),
            require_complete=True,
        )
        shard_manifest_path = base / "tables" / "stage3_formal_raw_shard_manifest.json"
        write_canonical_json(shard_manifest_path, aggregate)
        raw_shard_manifest_record = _file_record(workspace_root, shard_manifest_path)
    report_path = base / "analysis_report.json"
    markdown_path = base / "analysis_report.md"
    write_canonical_json(report_path, report.to_dict())
    atomic_write_bytes(markdown_path, report.render_markdown().encode("utf-8"))

    artifacts: list[dict[str, JSONValue]] = []
    figures: list[dict[str, JSONValue]] = []
    for spec in chart_specs:
        stem = _safe_table_name(spec.chart_id)
        png_path = base / "figures" / f"{stem}.png"
        svg_path = base / "figures" / f"{stem}.svg"
        png = render_matplotlib_chart(spec, next(table for table in tables if table.name == spec.source_name), png_path, output_format="png")
        svg = render_matplotlib_chart(spec, next(table for table in tables if table.name == spec.source_name), svg_path, output_format="svg")
        artifacts.extend((png.to_dict(), svg.to_dict()))
        figures.append({
            "id": spec.chart_id,
            "source_table": spec.source_name,
            "source_hash": spec.source_hash,
            "png": _file_record(workspace_root, png_path),
            "svg": _file_record(workspace_root, svg_path),
        })

    return {
        "tables": table_records,
        "raw_formal_path_results": {
            "ref": _file_record(workspace_root, raw_path)["path"],
            "sha256": _file_record(workspace_root, raw_path)["sha256"],
            "file": _file_record(workspace_root, raw_path),
        },
        "raw_shard_manifest": raw_shard_manifest_record,
        "analysis_report": _file_record(workspace_root, report_path),
        "report_markdown": _file_record(workspace_root, markdown_path),
        "artifacts": artifacts,
        "figures": figures,
    }


__all__ = ["build_raw_reporting_tables", "write_reporting_bundle"]
