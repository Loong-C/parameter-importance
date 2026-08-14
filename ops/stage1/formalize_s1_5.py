#!/usr/bin/env python3
"""Publish immutable CPU-only S1.5 / G1-EST evidence.

The command verifies the exact S1.4 G1-GRAD handoff before executing only the
deterministic estimator fixture.  It never opens model/data assets, starts
training, probes CUDA, or writes outside the caller-supplied project data
root.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
TASK_ID = "stage1.05_estimators"
GATE_ID = "G1-EST"
FIXTURE_ID = "stage1-s15-estimator-fixture-v1"
INDEX_SCHEMA = "stage1-s1-5-formalization-index-v1"
VALIDATION_SCHEMA = "stage1-s1-5-validation-v1"
EXPECTED_S1_4_INDEX_SHA256 = "346c86daeb8cc61c9b4145891fc195fd3ffa29ae7a8ee28d6e670c27a0ee62c0"
EXPECTED_S1_4_GATE_ARTIFACT_HASH = "56e8e5d2128eb1f5c26d20ae7cfbcc4d01d1470cb5e55ede075f253e4f85b7d6"
EXPECTED_S1_4_PRODUCER_COMMIT = "92e3fa5ec286afa43c51be691895f9a7210199ff"


class Stage1S15FormalError(RuntimeError):
    """An S1.5 formal attempt is not eligible for publication."""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(["git", "-C", str(repository), *arguments], capture_output=True, check=False, text=True, timeout=30)
    if completed.returncode != 0:
        raise Stage1S15FormalError(f"S1_5_GIT_COMMAND_FAILED:{arguments[0]}")
    return completed.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _with_artifact_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash
    payload = dict(value)
    payload["artifact_hash"] = canonical_json_hash(value)
    return payload


def _logical_path(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S15FormalError(f"S1_5_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S15FormalError(f"S1_5_LOGICAL_REF_ESCAPE:{field}")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S15FormalError(f"S1_5_LOGICAL_REF_ESCAPE:{field}") from error
    return candidate


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise Stage1S15FormalError(f"S1_5_DIGEST_INVALID:{field}")
    return value


def _role_path(data_root: Path, index_path: Path, reference: object, expected_sha: object, *, field: str) -> Path:
    from param_importance_nlp.atomic import sha256_file
    if not isinstance(reference, str):
        raise Stage1S15FormalError(f"S1_5_S1_4_ROLE_REF_INVALID:{field}")
    candidate = _logical_path(data_root, reference, field=field)
    if not candidate.is_file():
        candidate = (index_path.parent / reference).resolve()
        try:
            candidate.relative_to(data_root.resolve())
        except ValueError as error:
            raise Stage1S15FormalError(f"S1_5_S1_4_ROLE_REF_ESCAPE:{field}") from error
    if not candidate.is_file():
        raise Stage1S15FormalError(f"S1_5_S1_4_ROLE_MISSING:{field}")
    if sha256_file(candidate) != _require_digest(expected_sha, field=field):
        raise Stage1S15FormalError(f"S1_5_S1_4_ROLE_SHA256_MISMATCH:{field}")
    return candidate


def _assert_s1_4_consumer_compatibility(
    report: Mapping[str, Any],
    table: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    repository_root: Path,
) -> set[str]:
    """Rebuild S1.4 at the consumer commit without invalidating valid history.

    S1.5 intentionally extends the shared public export module and Stage 1 task
    runner.  Those byte changes make a literal replay of the historical S1.4
    source-hash map impossible, even though the S1.4 loss/gradient implementation
    and all numerical payloads remain unchanged.  We therefore allow only those
    two downstream-owned source drifts, rebuild S1.4 with the current checkout,
    and compare every semantic field while excluding only source/self/cross hashes.
    """

    from param_importance_nlp.stage1_gradient_scale import build_stage1_s14_evidence

    rebuilt = build_stage1_s14_evidence(
        repository_root,
        producer_commit=str(report.get("producer_commit")),
        scope=str(report.get("scope")),
        upstream_evidence=report.get("upstream") if isinstance(report.get("upstream"), Mapping) else None,
    )
    rebuilt_report = rebuilt["gradient_scale_report"]
    rebuilt_table = rebuilt["comparison_table"]
    rebuilt_gate = rebuilt["gate_record"]
    historical_sources = report.get("implementation_source_sha256")
    current_sources = rebuilt_report.get("implementation_source_sha256")
    if not isinstance(historical_sources, Mapping) or not isinstance(current_sources, Mapping) or set(historical_sources) != set(current_sources):
        raise Stage1S15FormalError("S1_5_S1_4_SOURCE_MAP_INVALID")
    drifted = {name for name in historical_sources if historical_sources[name] != current_sources[name]}
    allowed_drift = {
        "src/param_importance_nlp/core/__init__.py",
        "src/param_importance_nlp/experiments/stage01_task_runners.py",
    }
    if not drifted <= allowed_drift:
        raise Stage1S15FormalError(f"S1_5_S1_4_UNAUTHORIZED_SOURCE_DRIFT:{','.join(sorted(drifted - allowed_drift))}")

    historical_report = dict(report)
    current_report = dict(rebuilt_report)
    for value in (historical_report, current_report):
        value.pop("implementation_source_sha256", None)
        value.pop("report_hash", None)
    historical_table = dict(table)
    current_table = dict(rebuilt_table)
    for value in (historical_table, current_table):
        value.pop("report_hash", None)
        value.pop("table_hash", None)
    historical_gate = dict(gate)
    current_gate = dict(rebuilt_gate)
    for value in (historical_gate, current_gate):
        value.pop("report_hash", None)
        value.pop("comparison_table_hash", None)
        value.pop("artifact_hash", None)
    if historical_report != current_report or historical_table != current_table or historical_gate != current_gate:
        raise Stage1S15FormalError("S1_5_S1_4_CONSUMER_SEMANTIC_REBUILD_MISMATCH")
    return drifted


def _load_s1_4_handoff(data_root: Path, index_ref: str) -> dict[str, str]:
    """Load and replay every immutable role of the one eligible S1.4 Gate."""

    from param_importance_nlp.atomic import sha256_file
    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
    from param_importance_nlp.stage1_gradient_scale import validate_stage1_s14_evidence

    index_path = _logical_path(data_root, index_ref, field="s1_4_index_ref")
    if not index_path.is_file() or sha256_file(index_path) != EXPECTED_S1_4_INDEX_SHA256:
        raise Stage1S15FormalError("S1_5_S1_4_INDEX_NOT_CURRENT")
    index = load_canonical_json(index_path)
    if not isinstance(index, Mapping):
        raise Stage1S15FormalError("S1_5_S1_4_INDEX_NOT_OBJECT")
    body = dict(index)
    if body.pop("artifact_hash", None) != canonical_json_hash(body):
        raise Stage1S15FormalError("S1_5_S1_4_INDEX_HASH_INVALID")
    if (
        index.get("schema_version") != "stage1-s1-4-formalization-index-v1"
        or index.get("status") != "PASS"
        or index.get("task_id") != "stage1.04_loss_and_gradient_scale"
        or index.get("gate_id") != "G1-GRAD"
        or index.get("generator_git_commit") != EXPECTED_S1_4_PRODUCER_COMMIT
        or index.get("gate_artifact_hash") != EXPECTED_S1_4_GATE_ARTIFACT_HASH
        or index.get("next_task_id") != TASK_ID
    ):
        raise Stage1S15FormalError("S1_5_S1_4_HANDOFF_NOT_READY")
    specs = {
        "gradient_scale_report": ("gradient_scale_report_ref", "gradient_scale_report_sha256"),
        "comparison_table": ("comparison_table_ref", "comparison_table_sha256"),
        "gate_record": ("gate_record_ref", "gate_record_sha256"),
        "replay": ("replay_ref", "replay_sha256"),
        "validation": ("validation_ref", "validation_sha256"),
    }
    paths = {role: _role_path(data_root, index_path, index.get(ref), index.get(sha), field=role) for role, (ref, sha) in specs.items()}
    roles = {role: load_canonical_json(path) for role, path in paths.items()}
    report = roles["gradient_scale_report"]
    table = roles["comparison_table"]
    gate = roles["gate_record"]
    replay = roles["replay"]
    validation = roles["validation"]
    if not all(isinstance(value, Mapping) for value in (report, table, gate, replay, validation)):
        raise Stage1S15FormalError("S1_5_S1_4_ROLE_NOT_OBJECT")
    assert isinstance(report, Mapping) and isinstance(table, Mapping) and isinstance(gate, Mapping) and isinstance(replay, Mapping) and isinstance(validation, Mapping)
    if (
        report.get("status") != "PASS" or report.get("gate_status") != "PASS"
        or gate.get("status") != "PASS" or gate.get("artifact_hash") != EXPECTED_S1_4_GATE_ARTIFACT_HASH
        or index.get("gradient_scale_report_hash") != report.get("report_hash")
        or index.get("comparison_table_hash") != table.get("table_hash")
        or index.get("replay_hash") != replay.get("replay_hash")
        or validation.get("status") != "PASS"
    ):
        raise Stage1S15FormalError("S1_5_S1_4_ROLE_BINDING_INVALID")
    replay_body = dict(replay)
    replay_hash = replay_body.pop("replay_hash", None)
    validation_body = dict(validation)
    validation_hash = validation_body.pop("artifact_hash", None)
    if replay_hash != canonical_json_hash(replay_body) or validation_hash != canonical_json_hash(validation_body):
        raise Stage1S15FormalError("S1_5_S1_4_AUXILIARY_SELF_HASH_INVALID")
    if (
        validation.get("producer_commit") != EXPECTED_S1_4_PRODUCER_COMMIT
        or validation.get("consumer_commit") != EXPECTED_S1_4_PRODUCER_COMMIT
        or validation.get("role_sha256") != {
            "gradient_scale_report": sha256_file(paths["gradient_scale_report"]),
            "comparison_table": sha256_file(paths["comparison_table"]),
            "gate_record": sha256_file(paths["gate_record"]),
        }
        or validation.get("replay_sha256") != sha256_file(paths["replay"])
        or validation.get("replay_hash") != replay.get("replay_hash")
    ):
        raise Stage1S15FormalError("S1_5_S1_4_VALIDATION_BINDING_INVALID")
    try:
        validate_stage1_s14_evidence({"gradient_scale_report": report, "comparison_table": table, "gate_record": gate})
        _assert_s1_4_consumer_compatibility(
            report,
            table,
            gate,
            repository_root=Path(__file__).resolve().parents[2],
        )
    except Exception as error:
        if isinstance(error, Stage1S15FormalError):
            raise
        raise Stage1S15FormalError("S1_5_S1_4_CONSUMER_REVALIDATION_FAILED") from error
    return {
        "s1_4_index_ref": index_ref, "s1_4_index_sha256": EXPECTED_S1_4_INDEX_SHA256,
        "s1_4_index_artifact_hash": str(index["artifact_hash"]), "s1_4_gate_artifact_hash": EXPECTED_S1_4_GATE_ARTIFACT_HASH,
        "s1_4_gradient_scale_report_sha256": sha256_file(paths["gradient_scale_report"]),
        "s1_4_comparison_table_sha256": sha256_file(paths["comparison_table"]),
        "s1_4_gate_record_sha256": sha256_file(paths["gate_record"]),
        "s1_4_replay_sha256": sha256_file(paths["replay"]), "s1_4_validation_sha256": sha256_file(paths["validation"]),
    }


def _run_regression(repository_root: Path, work_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--basetemp", str(work_dir / "pytest-tmp"), "tests/test_stage1_s15_estimators.py", "tests/test_stage1_s15_handoff_and_charts.py", "tests/test_stage01_task_runners.py", "tests/test_core_estimators_and_accumulator.py", "tests/test_stage1_s14_handoff_and_charts.py"]
    completed = subprocess.run(command, cwd=repository_root, capture_output=True, check=False, text=True, timeout=timeout_seconds)
    if completed.returncode != 0:
        raise Stage1S15FormalError(f"S1_5_SERVER_REGRESSION_FAILED:returncode={completed.returncode}")
    return {"schema_version": "stage1-s1-5-regression-v1", "command": command, "returncode": 0, "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")), "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")), "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-2000:]}


def _direct_checks(evidence: Mapping[str, Any], upstream: Mapping[str, str], *, repository_root: Path | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    from param_importance_nlp.stage1_estimators import _source_hashes, replay_stage1_s15_evidence, validate_stage1_s15_evidence
    validate_stage1_s15_evidence(evidence, source_root=repository_root)
    replay = replay_stage1_s15_evidence(evidence, source_root=repository_root)
    report = evidence["estimator_report"]
    table = evidence["comparison_table"]
    gate = evidence["gate_record"]
    assert isinstance(report, Mapping) and isinstance(table, Mapping) and isinstance(gate, Mapping)
    checks: list[dict[str, str]] = []
    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            raise Stage1S15FormalError(f"S1_5_DIRECT_CHECK_FAILED:{check_id}")
    profiles = report["profiles"]
    check("formal_gate_record", report.get("gate_status") == "PASS" and gate.get("status") == "PASS", str(gate.get("status")))
    check("profiles_and_tensor_rows", {profile["profile"] for profile in profiles} == {"T64_ORACLE", "T32_SINGLE"} and all(row.get("passed") is True for row in table["rows"]), str(len(table["rows"])))
    check("raw_clip_lr_contract", all(
        all(all(value is True for value in tensor_values.values()) for tensor_values in profile["raw_clip_boundary"].values())
        and profile["parameter_to_group"] == {"bias": "group_0000", "head": "group_0001", "weight": "group_0000"}
        and profile["learning_rates"] == {"group_0000": 0.25, "group_0001": 0.05}
        for profile in profiles
    ), "raw core, raw score, and clipped score are separately replayed per tensor")
    check("double_and_m2_contract", all(profile["double_sampling"]["sample_id_overlap_allowed"] is True and profile["double_sampling"]["a_b_exchange_passed"] is True and profile["double_sampling"]["m2_equals_double_passed"] is True for profile in profiles), "provenance permits overlapping IDs but rejects shared objects/state")
    check("u_weighted_and_negative_boundaries", all(all(float(value) < 0 for values in profile["negative_u"].values() for value in values) and all(profile["rejections"].values()) for profile in profiles), "negative U preserved; malformed boundaries rejected")
    check("estimator_matrix_and_claims", all(
        len(profile["comparisons"]) == 20
        and profile["weighted_statistical_contract"] == {
            "statistical_unit": "microbatch",
            "weight_unit": "effective_target_tokens",
            "sampling_design": "with_replacement_product_sampling",
            "weights_exogenous": True,
            "common_mean_assumption": True,
        }
        and profile["estimator_public_fields"]["weighted_u"]["unbiasedness_claim"]
        == "unbiased_fixed_state_under_declared_sampling_assumptions"
        and profile["estimator_public_fields"]["weighted_u_assumptions_false"]["unbiasedness_claim"]
        == "no_unbiasedness_claim"
        for profile in profiles
    ), "20 comparison objects bind every public estimator field and statistical claim")
    check("chart_route_matrix", all(
        {row["route_id"] for row in profile["scatter_rows"]}
        == {
            "equal_u_ordered_vs_oracle",
            "equal_u_unordered_vs_oracle",
            "equal_u_streaming_vs_oracle",
        }
        and all({"raw_core", "raw_score", "raw_clipped_score", "u_core", "u_score"} <= set(row) for row in profile["scaling_rows"])
        for profile in profiles
    ), "ordered, unordered, and streaming U routes plus raw/U scaling are exact report projections")
    check("sufficient_statistics_replay", all(len(profile["statistics_comparisons"]) == 4 for profile in replay["profiles"]), "S1/S2/G1/G2/N1/N2 replayed")
    check("s1_4_handoff_closed", upstream["s1_4_index_sha256"] == EXPECTED_S1_4_INDEX_SHA256 and upstream["s1_4_gate_artifact_hash"] == EXPECTED_S1_4_GATE_ARTIFACT_HASH, upstream["s1_4_index_ref"])
    check("offline_replay", replay.get("status") == "PASS", str(replay.get("replay_hash")))
    if repository_root is not None:
        check("implementation_source_binding", report.get("implementation_source_sha256") == _source_hashes(repository_root), "formal consumer reread every implementation/schema/test source digest")
    return checks, replay


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise Stage1S15FormalError(f"S1_5_IMMUTABLE_TARGET_EXISTS:{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_svg(path: Path, *, title: str, source_name: str, body: Iterable[str], width: int = 1120, height: int = 640) -> None:
    if path.exists():
        raise Stage1S15FormalError(f"S1_5_IMMUTABLE_TARGET_EXISTS:{path}")
    path.write_text("\n".join(['<?xml version="1.0" encoding="UTF-8"?>', f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-source="{escape(source_name)}">', f"<title>{escape(title)}</title>", f"<desc>Dependency-free SVG rendered from {escape(source_name)}.</desc>", '<rect width="100%" height="100%" fill="white"/>', *body, "</svg>", ""]), encoding="utf-8")


def _scatter_svg(path: Path, rows: list[Mapping[str, Any]]) -> None:
    values = [float(row["candidate"]) for row in rows] + [float(row["reference"]) for row in rows]
    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum): maximum = minimum + 1.0
    left, top, right, bottom = 90.0, 70.0, 1040.0, 560.0
    body = [f'<text x="60" y="34" font-size="18">ordered, unordered, and streaming U vs independent oracle (y=x)</text>', f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>', f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>', f'<line class="identity" x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="#777" stroke-dasharray="6 4"/>']
    route_colors = {
        "equal_u_ordered_vs_oracle": "#1565c0",
        "equal_u_unordered_vs_oracle": "#2e7d32",
        "equal_u_streaming_vs_oracle": "#c62828",
    }
    for row in rows:
        x = left + (float(row["reference"]) - minimum) / (maximum - minimum) * (right - left)
        y = bottom - (float(row["candidate"]) - minimum) / (maximum - minimum) * (bottom - top)
        route_id = str(row["route_id"])
        if route_id not in route_colors:
            raise Stage1S15FormalError(f"S1_5_SCATTER_ROUTE_INVALID:{route_id}")
        body.append(f'<circle class="scatter-point" cx="{x:.4f}" cy="{y:.4f}" r="3" fill="{route_colors[route_id]}" data-route="{escape(route_id)}" data-profile="{escape(str(row["profile"]))}" data-parameter="{escape(str(row["parameter_name"]))}" data-coordinate="{row["coordinate"]}"/>')
    _write_svg(path, title="three U implementations vs independent oracle", source_name="u-identity-scatter.csv", body=body)


def _heatmap_svg(path: Path, rows: list[Mapping[str, Any]]) -> None:
    names = list(dict.fromkeys(str(row["parameter_name"]) for row in rows))
    routes = list(dict.fromkeys((str(row["profile"]), str(row["comparison_id"])) for row in rows))
    width, left, top, cell_w, cell_h = max(1120, 240 + 26 * len(routes)), 200.0, 105.0, 24.0, 48.0
    body = ['<text x="60" y="34" font-size="18">per-tensor estimator error heatmap</text>']
    for row in rows:
        x = left + routes.index((str(row["profile"]), str(row["comparison_id"]))) * cell_w
        y = top + names.index(str(row["parameter_name"])) * cell_h
        value = abs(float(row["scaled_max_error"] or 0.0))
        shade = int(245 - min(value / 1e-5, 1.0) * 200)
        body.append(f'<rect class="heatmap-cell" x="{x}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" fill="rgb(255,{shade},{shade})" data-profile="{escape(str(row["profile"]))}" data-route="{escape(str(row["comparison_id"]))}" data-tensor="{escape(str(row["parameter_name"]))}" data-error="{value:.17g}"/>')
    _write_svg(path, title="per-tensor estimator error heatmap", source_name="tensor-errors.csv", body=body, width=width, height=max(420, int(top + cell_h * len(names) + 80)))


def _scaling_svg(path: Path, rows: list[Mapping[str, Any]]) -> None:
    maximum = max(
        [
            abs(float(row[field]))
            for row in rows
            for field in ("raw_core", "raw_score", "raw_clipped_score", "u_core", "u_score")
        ]
        + [1.0]
    )
    body = ['<text x="60" y="34" font-size="18">core to score learning-rate / clip scaling</text>']
    for index, row in enumerate(rows):
        x = 70 + index * 96
        raw_height = abs(float(row["raw_core"])) / maximum * 430
        score_height = abs(float(row["raw_score"])) / maximum * 430
        clipped_height = abs(float(row["raw_clipped_score"])) / maximum * 430
        u_core_height = abs(float(row["u_core"])) / maximum * 430
        u_score_height = abs(float(row["u_score"])) / maximum * 430
        metadata = f'data-profile="{escape(str(row["profile"]))}" data-parameter="{escape(str(row["parameter_name"]))}" data-coordinate="{row["coordinate"]}"'
        body.extend([
            f'<rect class="raw-core-bar" x="{x}" y="{540 - raw_height:.3f}" width="14" height="{raw_height:.3f}" fill="#455a64" {metadata}/>',
            f'<rect class="raw-score-bar" x="{x + 17}" y="{540 - score_height:.3f}" width="14" height="{score_height:.3f}" fill="#1565c0" {metadata}/>',
            f'<rect class="raw-clipped-bar" x="{x + 34}" y="{540 - clipped_height:.3f}" width="14" height="{clipped_height:.3f}" fill="#c62828" {metadata}/>',
            f'<rect class="u-core-bar" x="{x + 51}" y="{540 - u_core_height:.3f}" width="14" height="{u_core_height:.3f}" fill="#6a1b9a" {metadata}/>',
            f'<rect class="u-score-bar" x="{x + 68}" y="{540 - u_score_height:.3f}" width="14" height="{u_score_height:.3f}" fill="#ef6c00" {metadata}/>',
        ])
    _write_svg(path, title="raw and U core-to-score scaling", source_name="scaling.csv", body=body, width=max(1120, 100 + len(rows) * 96))


def _negative_svg(path: Path, rows: list[Mapping[str, Any]]) -> None:
    body = ['<text x="60" y="34" font-size="18">constructed negative U coordinates (not clamped)</text>']
    for index, row in enumerate(rows):
        body.append(f'<text class="negative-u-row" x="70" y="{72 + index * 24}" font-size="14" data-profile="{escape(str(row["profile"]))}" data-parameter="{escape(str(row["parameter_name"]))}" data-coordinate="{row["coordinate"]}" data-u-core="{float(row["u_core"]):.17g}">{escape(str(row["profile"]))} {escape(str(row["parameter_name"]))}[{row["coordinate"]}] = {float(row["u_core"]):.7g}</text>')
    _write_svg(path, title="constructed negative U", source_name="negative-u.csv", body=body, height=max(300, 100 + len(rows) * 25))


def write_chart_data(work_dir: Path, evidence: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    from param_importance_nlp.atomic import sha256_file
    table = evidence["comparison_table"]
    assert isinstance(table, Mapping)
    rows, scatter, scaling, negative = table["rows"], table["scatter_rows"], table["scaling_rows"], table["negative_u_rows"]
    _write_csv(work_dir / "tensor-errors.csv", ["profile", "comparison_id", "object_id", "parameter_name", "passed", "branch", "comparison_dtype", "natural_scale", "atol", "rtol", "normalized_l2_limit", "near_zero_threshold", "absolute_threshold", "max_absolute_error", "scaled_max_error", "normalized_l2_error", "nonfinite_count", "worst_parameter"], rows)
    _write_csv(work_dir / "u-identity-scatter.csv", ["profile", "route_id", "parameter_name", "coordinate", "candidate", "reference"], scatter)
    _write_csv(work_dir / "scaling.csv", ["profile", "parameter_name", "parameter_group", "coordinate", "learning_rate", "clip_factor", "raw_core", "raw_score", "raw_clipped_score", "u_core", "u_score"], scaling)
    _write_csv(work_dir / "negative-u.csv", ["profile", "parameter_name", "coordinate", "u_core"], negative)
    _scatter_svg(work_dir / "u-identity-scatter.svg", scatter)
    _heatmap_svg(work_dir / "tensor-error-heatmap.svg", rows)
    _scaling_svg(work_dir / "scaling.svg", scaling)
    _negative_svg(work_dir / "negative-u.svg", negative)
    csv_sha = {path.name: sha256_file(path) for path in sorted(work_dir.glob("*.csv"))}
    svg_sha = {path.name: sha256_file(path) for path in sorted(work_dir.glob("*.svg"))}
    if set(csv_sha) != {"tensor-errors.csv", "u-identity-scatter.csv", "scaling.csv", "negative-u.csv"} or set(svg_sha) != {"u-identity-scatter.svg", "tensor-error-heatmap.svg", "scaling.svg", "negative-u.svg"}:
        raise Stage1S15FormalError("S1_5_CHART_SET_INVALID")
    return csv_sha, svg_sha


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Stage1S15FormalError(f"S1_5_IMMUTABLE_TARGET_EXISTS:{path}")
    from param_importance_nlp.contracts.jsonio import write_canonical_json
    write_canonical_json(path, dict(value))


def execute(*, repository: str | Path, data_root: str | Path, s1_4_index_ref: str, attempt_id: str, timeout_seconds: int = 1200) -> dict[str, str]:
    repository_root, data_root_path = Path(repository).resolve(strict=True), Path(data_root).resolve(strict=True)
    source_root = repository_root / "src"
    if str(source_root) not in sys.path: sys.path.insert(0, str(source_root))
    commit = _git(repository_root, "rev-parse", "HEAD")
    if _COMMIT_RE.fullmatch(commit) is None or _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S15FormalError("S1_5_FORMAL_REQUIRES_CLEAN_WORKTREE")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", attempt_id) is None:
        raise Stage1S15FormalError("S1_5_ATTEMPT_ID_INVALID")
    upstream = _load_s1_4_handoff(data_root_path, s1_4_index_ref)
    evidence_dir = data_root_path / "evidence" / "stage1" / "s1-5-formal" / commit / attempt_id
    work_dir = data_root_path / "tmp" / "stage1-s1-5" / commit / attempt_id
    if evidence_dir.exists() or work_dir.exists():
        raise Stage1S15FormalError("S1_5_ATTEMPT_ALREADY_EXISTS")
    work_dir.mkdir(parents=True, exist_ok=False)
    regression = _run_regression(repository_root, work_dir, timeout_seconds)
    from param_importance_nlp.atomic import sha256_file
    from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
    from param_importance_nlp.stage1_estimators import build_stage1_s15_evidence
    evidence = build_stage1_s15_evidence(repository_root, producer_commit=commit, scope="formal", upstream_evidence=upstream)
    checks, replay = _direct_checks(evidence, upstream, repository_root=repository_root)
    filenames = {"estimator_report": "estimator-report.json", "oracle_report": "oracle-report.json", "tensor_bundle": "tensor-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-est-record.json"}
    paths: dict[str, Path] = {}
    for role, filename in filenames.items():
        paths[role] = work_dir / filename
        _write_new(paths[role], evidence[role])
    reloaded = {role: load_canonical_json(path) for role, path in paths.items()}
    _, replay_after_roundtrip = _direct_checks(reloaded, upstream, repository_root=repository_root)
    replay_path = work_dir / "replay-validation.json"
    _write_new(replay_path, replay_after_roundtrip)
    csv_sha, svg_sha = write_chart_data(work_dir, reloaded)
    validation = _with_artifact_hash({"schema_version": VALIDATION_SCHEMA, "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "execution_scope": "formal_server_cpu", "fixture_id": FIXTURE_ID, "producer_commit": commit, "consumer_commit": commit, "upstream": upstream, "regression": regression, "direct_checks": checks, "role_sha256": {role: sha256_file(path) for role, path in paths.items()}, "csv_sha256": csv_sha, "svg_sha256": svg_sha, "replay_sha256": sha256_file(replay_path), "replay_hash": replay_after_roundtrip["replay_hash"]})
    validation_path = work_dir / "validation.json"
    _write_new(validation_path, validation)
    index = _with_artifact_hash({"schema_version": INDEX_SCHEMA, "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "generator_git_commit": commit, "consumer_git_commit": commit, "git_branch": _git(repository_root, "branch", "--show-current"), "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), **upstream, "role_refs": {role: filename for role, filename in filenames.items()}, "role_sha256": {role: sha256_file(path) for role, path in paths.items()}, "gate_artifact_hash": evidence["gate_record"]["artifact_hash"], "csv_sha256": csv_sha, "svg_sha256": svg_sha, "validation_ref": "validation.json", "validation_sha256": sha256_file(validation_path), "replay_ref": "replay-validation.json", "replay_sha256": sha256_file(replay_path), "replay_hash": replay_after_roundtrip["replay_hash"], "next_task_id": "stage1.06_training_integration_and_accumulators"})
    staging_dir = evidence_dir.parent / f".{attempt_id}.publishing"
    if staging_dir.exists(): raise Stage1S15FormalError("S1_5_PUBLISH_STAGING_ALREADY_EXISTS")
    staging_dir.mkdir(parents=True, exist_ok=False)
    for source in (*paths.values(), replay_path, validation_path, *sorted(work_dir.glob("*.csv")), *sorted(work_dir.glob("*.svg"))): (staging_dir / source.name).write_bytes(source.read_bytes())
    index_path = staging_dir / "index.json"
    write_canonical_json(index_path, index)
    loaded = load_canonical_json(index_path)
    if not isinstance(loaded, Mapping) or loaded.get("artifact_hash") != index["artifact_hash"]: raise Stage1S15FormalError("S1_5_INDEX_RELOAD_FAILED")
    os.replace(staging_dir, evidence_dir)
    published = evidence_dir / "index.json"
    if not published.is_file(): raise Stage1S15FormalError("S1_5_PUBLISH_RELOAD_FAILED")
    shutil.rmtree(work_dir)
    return {"index_ref": published.relative_to(data_root_path).as_posix(), "validation_ref": (evidence_dir / "validation.json").relative_to(data_root_path).as_posix(), "gate_record_ref": (evidence_dir / "g1-est-record.json").relative_to(data_root_path).as_posix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-4-index-ref", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args(argv)
    print(execute(repository=args.repository, data_root=args.data_root, s1_4_index_ref=args.s1_4_index_ref, attempt_id=args.attempt_id, timeout_seconds=args.timeout_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
