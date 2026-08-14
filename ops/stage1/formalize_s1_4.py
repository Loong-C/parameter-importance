#!/usr/bin/env python3
"""Publish immutable CPU-only S1.4 / G1-GRAD evidence.

The formal command closes the exact S1.3 v2 handoff before it calculates the
loss-reduction fixture.  It never starts training, probes CUDA, or writes
outside the project ``DATA_ROOT`` passed by the caller.
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
TASK_ID = "stage1.04_loss_and_gradient_scale"
GATE_ID = "G1-GRAD"
FIXTURE_ID = "stage1-s14-loss-gradient-fixture-v1"
INDEX_SCHEMA = "stage1-s1-4-formalization-index-v1"
VALIDATION_SCHEMA = "stage1-s1-4-validation-v1"
S1_3_INDEX_SCHEMA = "stage1-s1-3-formalization-index-v2"
S1_3_FIXTURE_SCHEMA = "stage1-fixture-manifest-v2"
S1_3_BUNDLE_SCHEMA = "stage1-oracle-bundle-v2"
S1_3_REPORT_SCHEMA = "stage1-oracle-validation-report-v2"
EXPECTED_S1_3_INDEX_SHA256 = "51eb16bf87d73d68f6c1da49b7635fa42bd0456e9305f7263326a794b9b2f2ab"
EXPECTED_S1_2_INDEX_SHA256 = "c0b77a662364cf5ac3c95499e68759534770c6251bd2de1f6d5f0f553093be29"


class Stage1S14FormalError(RuntimeError):
    """S1.4 evidence cannot be safely published."""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise Stage1S14FormalError(f"S1_4_GIT_COMMAND_FAILED:{arguments[0]}")
    return completed.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _logical_path(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S14FormalError(f"S1_4_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S14FormalError(f"S1_4_LOGICAL_REF_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S14FormalError(f"S1_4_LOGICAL_REF_ESCAPE:{field}") from error
    return path


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Stage1S14FormalError(f"S1_4_IMMUTABLE_TARGET_EXISTS:{path}")
    from param_importance_nlp.contracts.jsonio import write_canonical_json

    write_canonical_json(path, dict(value))


def _with_artifact_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash

    payload = dict(value)
    payload["artifact_hash"] = canonical_json_hash(value)
    return payload


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise Stage1S14FormalError(f"S1_4_DIGEST_INVALID:{field}")
    return value


def _load_s1_3_handoff(
    data_root: Path,
    index_ref: str,
    *,
    expected_index_sha256: str = EXPECTED_S1_3_INDEX_SHA256,
) -> dict[str, Any]:
    """Load every hash-bound role of only the current S1.3 v2 handoff."""

    from param_importance_nlp.atomic import sha256_file
    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
    from param_importance_nlp.stage1_fixtures import validate_stage1_s13_evidence

    index_path = _logical_path(data_root, index_ref, field="s1_3_index_ref")
    if not index_path.is_file():
        raise Stage1S14FormalError("S1_4_S1_3_INDEX_MISSING")
    observed_index_sha = sha256_file(index_path)
    if observed_index_sha != expected_index_sha256:
        raise Stage1S14FormalError("S1_4_S1_3_INDEX_NOT_CURRENT_V2")
    index = load_canonical_json(index_path)
    if not isinstance(index, dict):
        raise Stage1S14FormalError("S1_4_S1_3_INDEX_NOT_OBJECT")
    index_body = dict(index)
    if index_body.pop("artifact_hash", None) != canonical_json_hash(index_body):
        raise Stage1S14FormalError("S1_4_S1_3_INDEX_HASH_INVALID")
    if (
        index.get("schema_version") != S1_3_INDEX_SCHEMA
        or index.get("status") != "PASS"
        or index.get("gate_id") != "G1-ORACLE"
        or index.get("task_id") != "stage1.03_fixtures_and_oracles"
        or index.get("generator_git_commit") != "79f88d47c534840df950efac42481a317d66faba"
        or index.get("s1_2_index_sha256") != EXPECTED_S1_2_INDEX_SHA256
        or index.get("next_task_id") != TASK_ID
    ):
        raise Stage1S14FormalError("S1_4_S1_3_HANDOFF_NOT_READY")

    def role_path(reference: object, expected_sha: object, *, field: str) -> Path:
        if not isinstance(reference, str):
            raise Stage1S14FormalError(f"S1_4_S1_3_ROLE_REF_INVALID:{field}")
        candidate = _logical_path(data_root, reference, field=field)
        if not candidate.is_file():
            candidate = (index_path.parent / reference).resolve()
            try:
                candidate.relative_to(data_root.resolve())
            except ValueError as error:
                raise Stage1S14FormalError(f"S1_4_S1_3_ROLE_REF_ESCAPE:{field}") from error
        if not candidate.is_file():
            raise Stage1S14FormalError(f"S1_4_S1_3_ROLE_MISSING:{field}")
        if sha256_file(candidate) != _require_digest(expected_sha, field=field):
            raise Stage1S14FormalError(f"S1_4_S1_3_ROLE_SHA256_MISMATCH:{field}")
        return candidate

    role_specs = {
        "fixture_manifest": ("fixture_manifest_ref", "fixture_manifest_sha256"),
        "oracle_bundle": ("oracle_bundle_ref", "oracle_bundle_sha256"),
        "oracle_validation_report": (
            "oracle_validation_report_ref",
            "oracle_validation_report_sha256",
        ),
        "replay": ("replay_ref", "replay_sha256"),
        "validation": ("validation_ref", "validation_sha256"),
    }
    role_paths = {
        role: role_path(index.get(ref_field), index.get(sha_field), field=role)
        for role, (ref_field, sha_field) in role_specs.items()
    }
    roles = {role: load_canonical_json(path) for role, path in role_paths.items()}
    if not all(isinstance(value, Mapping) for value in roles.values()):
        raise Stage1S14FormalError("S1_4_S1_3_ROLE_NOT_OBJECT")
    fixture = roles["fixture_manifest"]
    bundle = roles["oracle_bundle"]
    report = roles["oracle_validation_report"]
    replay = roles["replay"]
    validation = roles["validation"]
    assert isinstance(fixture, Mapping) and isinstance(bundle, Mapping)
    assert isinstance(report, Mapping) and isinstance(replay, Mapping) and isinstance(validation, Mapping)
    fixture_body = dict(fixture)
    bundle_body = dict(bundle)
    report_body = dict(report)
    validation_body = dict(validation)
    if (
        fixture_body.pop("manifest_hash", None) != canonical_json_hash(fixture_body)
        or bundle_body.pop("bundle_hash", None) != canonical_json_hash(bundle_body)
        or report_body.pop("report_hash", None) != canonical_json_hash(report_body)
        or validation_body.pop("artifact_hash", None) != canonical_json_hash(validation_body)
    ):
        raise Stage1S14FormalError("S1_4_S1_3_ROLE_SELF_HASH_INVALID")
    if (
        fixture.get("schema_version") != S1_3_FIXTURE_SCHEMA
        or bundle.get("schema_version") != S1_3_BUNDLE_SCHEMA
        or report.get("schema_version") != S1_3_REPORT_SCHEMA
        or report.get("status") != "PASS"
        or validation.get("status") != "PASS"
        or fixture.get("manifest_hash") != bundle.get("fixture_manifest_hash")
        or bundle.get("bundle_hash") != report.get("oracle_bundle_hash")
        or index.get("oracle_bundle_hash") != bundle.get("bundle_hash")
        or index.get("frozen_gradient_input_hash") != bundle.get("frozen_gradient_input_hash")
        or bundle.get("frozen_gradient_input_hash") != report.get("frozen_gradient_input_hash")
        or replay.get("replay_hash") != index.get("replay_hash")
        or report.get("pythia_14m", {}).get("consumed_by_gate") is not False
    ):
        raise Stage1S14FormalError("S1_4_S1_3_ROLE_BINDING_INVALID")
    try:
        replay_result = validate_stage1_s13_evidence(
            {
                "fixture_manifest": fixture,
                "oracle_bundle": bundle,
                "oracle_validation_report": report,
            }
        )
    except Exception as error:  # the exact inner cause is retained in the failed attempt log
        raise Stage1S14FormalError("S1_4_S1_3_OFFLINE_REPLAY_FAILED") from error
    if replay_result.get("replay_hash") != index.get("replay_hash"):
        raise Stage1S14FormalError("S1_4_S1_3_REPLAY_HASH_MISMATCH")
    return {
        "s1_3_index_ref": index_ref,
        "s1_3_index_sha256": observed_index_sha,
        "s1_3_gate_artifact_hash": index["artifact_hash"],
        "s1_3_fixture_manifest_sha256": sha256_file(role_paths["fixture_manifest"]),
        "s1_3_oracle_bundle_sha256": sha256_file(role_paths["oracle_bundle"]),
        "s1_3_oracle_validation_report_sha256": sha256_file(role_paths["oracle_validation_report"]),
        "s1_3_replay_sha256": sha256_file(role_paths["replay"]),
        "s1_3_validation_sha256": sha256_file(role_paths["validation"]),
        "s1_3_frozen_gradient_input_hash": bundle["frozen_gradient_input_hash"],
    }


def _run_regression(repository_root: Path, work_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "--basetemp",
        str(work_dir / "pytest-tmp"),
        "tests/test_stage1_s14_loss_gradient_scale.py",
        "tests/test_stage1_s14_handoff_and_charts.py",
        "tests/test_core_registry_and_loss.py",
        "tests/test_stage01_task_runners.py",
        "tests/test_stage1_s13_fixtures.py",
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise Stage1S14FormalError(
            f"S1_4_SERVER_REGRESSION_FAILED:returncode={completed.returncode}"
        )
    return {
        "schema_version": "stage1-s1-4-regression-v1",
        "command": command,
        "returncode": completed.returncode,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _direct_checks(
    repository_root: Path, evidence: Mapping[str, Any], upstream: Mapping[str, str]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    from param_importance_nlp.stage1_gradient_scale import (
        replay_stage1_s14_evidence,
        validate_stage1_s14_evidence,
    )

    validate_stage1_s14_evidence(evidence)
    replay = replay_stage1_s14_evidence(evidence, repository_root)
    report = evidence["gradient_scale_report"]
    table = evidence["comparison_table"]
    gate = evidence["gate_record"]
    assert isinstance(report, Mapping) and isinstance(table, Mapping) and isinstance(gate, Mapping)
    checks: list[dict[str, str]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            raise Stage1S14FormalError(f"S1_4_DIRECT_CHECK_FAILED:{check_id}")

    checks_by_profile = {
        profile["profile"]: profile
        for profile in report["profiles"]
        if isinstance(profile, Mapping)
    }
    check("formal_gate_record", report.get("gate_status") == "PASS" and gate.get("status") == "PASS", str(gate.get("status")))
    check("both_required_profiles", set(checks_by_profile) == {"T64_ORACLE", "T32_SINGLE"}, str(sorted(checks_by_profile)))
    check(
        "all_tensor_rows_pass",
        all(row.get("passed") is True for row in table["rows"]),
        str(len(table["rows"])),
    )
    check(
        "negative_control_detected",
        all(profile["negative_control"]["gradient"]["passed"] is False and profile["negative_control"]["loss"]["passed"] is False for profile in checks_by_profile.values()),
        "deliberately equal-weighted unequal 2:6 route",
    )
    check(
        "sample_order_and_zero_token_fail_closed",
        report["sample_contract"]["zero_effective_token_rejected"] is True
        and report["sample_contract"]["ignored_target_locations"]
        == [{"sample_id": "s14-sample-0", "target_position": 1, "attention_mask": 1, "label": -100}],
        str(report["sample_contract"]["ordered_sample_ids"]),
    )
    check(
        "rng_and_accumulation_boundaries",
        all(
            profile["rng"]["exact_equivalence"]["cpu_rng_before"] == profile["rng"]["exact_equivalence"]["cpu_rng_between"] == profile["rng"]["exact_equivalence"]["cpu_rng_after"]
            and all(row["clear_was_complete"] is True for row in profile["accumulation"])
            for profile in checks_by_profile.values()
        ),
        "eval RNG unchanged; every accumulation route clears .grad",
    )
    check(
        "s1_3_v2_handoff_closed",
        upstream["s1_3_index_sha256"] == EXPECTED_S1_3_INDEX_SHA256,
        upstream["s1_3_index_ref"],
    )
    check(
        "offline_deterministic_replay",
        replay.get("status") == "PASS" and len(replay.get("comparison_hashes", {})) == 56,
        str(replay.get("replay_hash")),
    )
    return checks, replay


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise Stage1S14FormalError(f"S1_4_IMMUTABLE_TARGET_EXISTS:{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_svg(path: Path, *, title: str, source_name: str, body: Iterable[str], width: int = 1120, height: int = 640) -> None:
    """Write a dependency-free SVG with an explicit CSV provenance binding."""

    if path.exists():
        raise Stage1S14FormalError(f"S1_4_IMMUTABLE_TARGET_EXISTS:{path}")
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-source="{escape(source_name)}">',
                f"<title>{escape(title)}</title>",
                f"<desc>Dependency-free SVG rendered from {escape(source_name)}.</desc>",
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text class="chart-title" x="60" y="30" font-size="18" font-weight="bold">{escape(title)}</text>',
                *body,
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _scale(values: list[float], low: float, high: float) -> tuple[float, float]:
    minimum = min(values) if values else 0.0
    maximum = max(values) if values else 1.0
    if math.isclose(minimum, maximum):
        margin = max(abs(minimum) * 0.1, 1e-12)
        minimum -= margin
        maximum += margin
    return minimum, maximum


def _point(value: float, minimum: float, maximum: float, low: float, high: float) -> float:
    return low + (value - minimum) * (high - low) / (maximum - minimum)


def _scatter_svg(path: Path, rows: list[Mapping[str, Any]]) -> None:
    values = [float(row["candidate"]) for row in rows] + [float(row["reference"]) for row in rows]
    minimum, maximum = _scale(values, 0.0, 1.0)
    left, top, right, bottom = 90.0, 70.0, 1050.0, 560.0
    body = [
        f'<line class="axis axis-x" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
        f'<line class="axis axis-y" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        f'<line id="y-equals-x" class="y-equals-x" x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="#777" stroke-dasharray="6 4"/>',
        f'<text x="{right - 210}" y="{bottom + 42}" font-size="13">reference gradient (x)</text>',
        f'<text x="18" y="{top + 20}" font-size="13" transform="rotate(-90 18 {top + 20})">candidate gradient (y)</text>',
        f'<text x="{left}" y="{bottom + 20}" font-size="11">{minimum:.3g}</text>',
        f'<text x="{right - 35}" y="{bottom + 20}" font-size="11">{maximum:.3g}</text>',
    ]
    colours = {"equal_microbatch_m2_gradient": "#1565c0", "token_weighted_microbatch_gradient": "#2e7d32"}
    for row in rows:
        x = _point(float(row["reference"]), minimum, maximum, left, right)
        y = bottom - _point(float(row["candidate"]), minimum, maximum, 0.0, bottom - top)
        body.append(
            f'<circle class="scatter-point" cx="{x:.4f}" cy="{y:.4f}" r="2.3" fill="{colours.get(str(row["route_id"]), "#333")}" '
            f'data-profile="{escape(str(row["profile"]))}" data-route="{escape(str(row["route_id"]))}" '
            f'data-parameter="{escape(str(row["parameter_name"]))}" data-coordinate="{int(row["coordinate"])}"/>'
        )
    _write_svg(path, title="Full-batch versus reconstructed gradients (y=x)", source_name="gradient-scatter.csv", body=body)


def _route_error_svg(path: Path, rows: list[Mapping[str, Any]]) -> None:
    categories = ["equal_microbatch_m2_gradient", "token_weighted_microbatch_gradient", "negative_control_unweighted_unequal_microbatch_gradient"]
    labels = {categories[0]: "equal", categories[1]: "token-weighted", categories[2]: "intentional equal negative-control"}
    maximum = max([float(row["max_absolute_error"]) for row in rows] or [1.0])
    if maximum <= 0.0:
        maximum = 1.0
    left, top, right, bottom = 100.0, 80.0, 1050.0, 545.0
    body = [
        f'<line class="axis axis-x" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
        f'<line class="axis axis-y" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        '<text x="20" y="300" font-size="13" transform="rotate(-90 20 300)">global max absolute error</text>',
    ]
    colours = {categories[0]: "#1565c0", categories[1]: "#2e7d32", categories[2]: "#c62828"}
    profiles = ["T64_ORACLE", "T32_SINGLE"]
    for category_index, category in enumerate(categories):
        base_x = 210.0 + category_index * 285.0
        body.append(f'<text class="route-label" x="{base_x - 50:.1f}" y="{bottom + 34}" font-size="11">{escape(labels[category])}</text>')
        for profile_index, profile in enumerate(profiles):
            selected = next((row for row in rows if row["profile"] == profile and row["comparison_id"] == category), None)
            if selected is None:
                raise Stage1S14FormalError(f"S1_4_ROUTE_CHART_ROUTE_MISSING:{profile}:{category}")
            value = float(selected["max_absolute_error"])
            x = base_x + (profile_index - 0.5) * 34.0
            y = bottom - value / maximum * (bottom - top)
            body.append(f'<line class="route-stem" x1="{x:.3f}" y1="{bottom}" x2="{x:.3f}" y2="{y:.3f}" stroke="{colours[category]}"/>')
            body.append(
                f'<circle class="route-mark" cx="{x:.3f}" cy="{y:.3f}" r="6" fill="{colours[category]}" '
                f'data-profile="{profile}" data-route="{category}" data-error="{value:.17g}"/>'
            )
            body.append(f'<text x="{x - 16:.3f}" y="{bottom + 16}" font-size="9">{profile[-3:]}</text>')
    _write_svg(path, title="Equal / token-weighted / intentional-negative route errors", source_name="tensor-errors.csv", body=body)


def _heatmap_svg(path: Path, rows: list[Mapping[str, Any]]) -> None:
    tensor_names = sorted({str(row["parameter_name"]) for row in rows})
    routes = [(str(row["profile"]), str(row["comparison_id"])) for row in rows]
    unique_routes = list(dict.fromkeys(routes))
    width = max(1120, 240 + len(unique_routes) * 29)
    left, top, cell_w, cell_h = 205.0, 120.0, 26.0, 56.0
    values = [abs(float(row["scaled_max_error"])) for row in rows if row["scaled_max_error"] is not None]
    maximum = max(values or [1.0])
    body = ['<text x="60" y="76" font-size="12">tensor × route cells; colour is scaled max error</text>']
    for row_index, name in enumerate(tensor_names):
        y = top + row_index * cell_h
        body.append(f'<text x="{left - 8}" y="{y + 30}" font-size="10" text-anchor="end">{escape(name)}</text>')
    for column_index, (profile, comparison_id) in enumerate(unique_routes):
        x = left + column_index * cell_w
        body.append(f'<text class="heatmap-route-label" x="{x + 8}" y="{top - 8}" font-size="8" transform="rotate(-55 {x + 8} {top - 8})">{escape(profile + ":" + comparison_id)}</text>')
    for row in rows:
        row_index = tensor_names.index(str(row["parameter_name"]))
        column_index = unique_routes.index((str(row["profile"]), str(row["comparison_id"])))
        value = abs(float(row["scaled_max_error"])) if row["scaled_max_error"] is not None else 0.0
        intensity = int(245 - min(value / maximum, 1.0) * 205)
        x, y = left + column_index * cell_w, top + row_index * cell_h
        body.append(
            f'<rect class="heatmap-cell" x="{x}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" fill="rgb(255,{intensity},{intensity})" '
            f'data-profile="{escape(str(row["profile"]))}" data-route="{escape(str(row["comparison_id"]))}" data-tensor="{escape(str(row["parameter_name"]))}" data-value="{value:.17g}"/>'
        )
        body.append(f'<text class="heatmap-value" x="{x + 2}" y="{y + 31}" font-size="7">{value:.2g}</text>')
    legend_x = width - 150
    body.extend([
        f'<defs><linearGradient id="heatmap-gradient"><stop offset="0%" stop-color="rgb(255,245,245)"/><stop offset="100%" stop-color="rgb(255,40,40)"/></linearGradient></defs>',
        f'<rect id="heatmap-color-scale" x="{legend_x}" y="{top + len(tensor_names) * cell_h + 25}" width="100" height="14" fill="url(#heatmap-gradient)"/>',
        f'<text x="{legend_x}" y="{top + len(tensor_names) * cell_h + 56}" font-size="10">0 … {maximum:.2g}</text>',
    ])
    _write_svg(path, title="Per-tensor error heatmap", source_name="tensor-error-heatmap.csv", body=body, width=width, height=max(640, int(top + len(tensor_names) * cell_h + 105)))


def _accumulation_svg(path: Path, rows: list[Mapping[str, Any]]) -> None:
    categories = list(dict.fromkeys((int(row["microbatch_count"]), str(row["split_id"])) for row in rows))
    left, top, right, bottom = 95.0, 80.0, 1050.0, 550.0
    values = [float(row["normalized_l2_error"] or 0.0) for row in rows]
    minimum, maximum = _scale(values, 0.0, 1.0)
    body = [
        f'<line class="axis axis-x" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
        f'<line class="axis axis-y" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        '<text x="18" y="310" font-size="13" transform="rotate(-90 18 310)">normalized L2 error</text>',
    ]
    colours = {"T64_ORACLE": "#1565c0", "T32_SINGLE": "#2e7d32"}
    for index, (microbatch_count, split_id) in enumerate(categories):
        x = left + (index + 0.5) * (right - left) / len(categories)
        body.append(f'<text class="accumulation-label" x="{x - 38:.2f}" y="{bottom + 28}" font-size="10">M={microbatch_count} {escape(split_id.replace("accumulation_", ""))}</text>')
    for profile in ("T64_ORACLE", "T32_SINGLE"):
        points: list[str] = []
        for index, category in enumerate(categories):
            row = next(item for item in rows if item["profile"] == profile and (int(item["microbatch_count"]), str(item["split_id"])) == category)
            value = float(row["normalized_l2_error"] or 0.0)
            x = left + (index + 0.5) * (right - left) / len(categories)
            y = bottom - _point(value, minimum, maximum, 0.0, bottom - top)
            points.append(f"{x:.3f},{y:.3f}")
            body.append(f'<circle class="accumulation-point" cx="{x:.3f}" cy="{y:.3f}" r="5" fill="{colours[profile]}" data-profile="{profile}" data-m="{category[0]}" data-route="{escape(category[1])}" data-normalized-l2="{value:.17g}"/>')
        body.append(f'<polyline class="accumulation-curve" fill="none" stroke="{colours[profile]}" stroke-width="2" points="{" ".join(points)}" data-profile="{profile}"/>')
    _write_svg(path, title="Accumulation error by M and route", source_name="accumulation-errors.csv", body=body)


def _write_chart_data(work_dir: Path, evidence: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    from param_importance_nlp.atomic import sha256_file

    table = evidence["comparison_table"]
    report = evidence["gradient_scale_report"]
    assert isinstance(table, Mapping) and isinstance(report, Mapping)
    rows = table["rows"]
    scatter = table["scatter_rows"]
    accumulation = table["accumulation_rows"]
    assert isinstance(rows, list) and isinstance(scatter, list) and isinstance(accumulation, list)
    route_ids = ("equal_microbatch_m2_gradient", "token_weighted_microbatch_gradient")
    route_rows: list[dict[str, Any]] = []
    for profile in report["profiles"]:
        assert isinstance(profile, Mapping)
        comparisons = {item["comparison_id"]: item for item in profile["comparisons"] if isinstance(item, Mapping)}
        for comparison_id in route_ids:
            comparison = comparisons.get(comparison_id)
            if not isinstance(comparison, Mapping) or not isinstance(comparison.get("global"), Mapping):
                raise Stage1S14FormalError(f"S1_4_ROUTE_CHART_COMPARISON_MISSING:{comparison_id}")
            global_error = comparison["global"]
            route_rows.append(
                {
                    "profile": profile["profile"],
                    "comparison_id": comparison_id,
                    "max_absolute_error": global_error["max_absolute_error"],
                    "normalized_l2_error": global_error["normalized_l2_error"],
                    "expected": "PASS",
                }
            )
        negative = profile["negative_control"]
        assert isinstance(negative, Mapping) and isinstance(negative["gradient"], Mapping)
        global_error = negative["gradient"]["global"]
        assert isinstance(global_error, Mapping)
        route_rows.append({"profile": profile["profile"], "comparison_id": "negative_control_unweighted_unequal_microbatch_gradient", "max_absolute_error": global_error["max_absolute_error"], "normalized_l2_error": global_error["normalized_l2_error"], "expected": "FAIL"})
    _write_csv(work_dir / "tensor-errors.csv", ["profile", "comparison_id", "max_absolute_error", "normalized_l2_error", "expected"], route_rows)
    _write_csv(work_dir / "gradient-scatter.csv", ["profile", "route_id", "parameter_name", "coordinate", "candidate", "reference"], scatter)
    _write_csv(work_dir / "tensor-error-heatmap.csv", ["profile", "comparison_id", "parameter_name", "scaled_max_error", "normalized_l2_error"], rows)
    _write_csv(work_dir / "accumulation-errors.csv", ["profile", "split_id", "microbatch_count", "effective_counts", "clear_was_complete", "comparison_id", "normalized_l2_error"], accumulation)
    _scatter_svg(work_dir / "gradient-scatter.svg", scatter)
    _route_error_svg(work_dir / "route-errors.svg", route_rows)
    _heatmap_svg(work_dir / "tensor-error-heatmap.svg", rows)
    _accumulation_svg(work_dir / "accumulation-errors.svg", accumulation)
    csv_sha = {path.name: sha256_file(path) for path in sorted(work_dir.glob("*.csv"))}
    svg_sha = {path.name: sha256_file(path) for path in sorted(work_dir.glob("*.svg"))}
    if set(csv_sha) != {"tensor-errors.csv", "gradient-scatter.csv", "tensor-error-heatmap.csv", "accumulation-errors.csv"}:
        raise Stage1S14FormalError("S1_4_CSV_SET_INVALID")
    if set(svg_sha) != {"gradient-scatter.svg", "route-errors.svg", "tensor-error-heatmap.svg", "accumulation-errors.svg"}:
        raise Stage1S14FormalError("S1_4_SVG_SET_INVALID")
    return csv_sha, svg_sha


def _publish_staging_directory(staging_dir: Path, evidence_dir: Path, *, expected_artifact_hash: str) -> Path:
    """Atomically publish and return the final, live index path.

    Returning the staging path after ``os.replace`` creates a dangling handoff
    reference, so reload the index through its published location before the
    caller reports success.
    """

    from param_importance_nlp.contracts.jsonio import load_canonical_json

    staged_index = staging_dir / "index.json"
    staged = load_canonical_json(staged_index)
    if not isinstance(staged, Mapping) or staged.get("artifact_hash") != expected_artifact_hash:
        raise Stage1S14FormalError("S1_4_STAGING_INDEX_INVALID")
    os.replace(staging_dir, evidence_dir)
    published_index = evidence_dir / "index.json"
    published = load_canonical_json(published_index)
    if not isinstance(published, Mapping) or published.get("artifact_hash") != expected_artifact_hash:
        raise Stage1S14FormalError("S1_4_PUBLISHED_INDEX_RELOAD_FAILED")
    return published_index


def _require_exact_digest_map(value: Mapping[str, str], expected: set[str], *, field: str) -> None:
    if set(value) != expected or not all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in value.values()):
        raise Stage1S14FormalError(f"S1_4_EXACT_DIGEST_MAP_INVALID:{field}")


def execute(
    *,
    repository: str | Path,
    data_root: str | Path,
    s1_3_index_ref: str,
    attempt_id: str,
    timeout_seconds: int,
) -> dict[str, str]:
    repository_root = Path(repository).resolve(strict=True)
    data_root_path = Path(data_root).resolve(strict=True)
    source_root = repository_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    commit = _git(repository_root, "rev-parse", "HEAD")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise Stage1S14FormalError("S1_4_REPOSITORY_COMMIT_INVALID")
    if _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S14FormalError("S1_4_FORMAL_REQUIRES_CLEAN_WORKTREE")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", attempt_id) is None:
        raise Stage1S14FormalError("S1_4_ATTEMPT_ID_INVALID")
    upstream = _load_s1_3_handoff(data_root_path, s1_3_index_ref)
    evidence_dir = data_root_path / "evidence" / "stage1" / "s1-4-formal" / commit / attempt_id
    work_dir = data_root_path / "tmp" / "stage1-s1-4" / commit / attempt_id
    if evidence_dir.exists():
        raise Stage1S14FormalError(f"S1_4_ATTEMPT_ALREADY_EXISTS:{evidence_dir}")
    if work_dir.exists():
        raise Stage1S14FormalError(f"S1_4_WORK_ATTEMPT_ALREADY_EXISTS:{work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)
    regression = _run_regression(repository_root, work_dir, timeout_seconds)
    from param_importance_nlp.atomic import sha256_file
    from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
    from param_importance_nlp.stage1_gradient_scale import build_stage1_s14_evidence

    evidence = build_stage1_s14_evidence(
        repository_root,
        producer_commit=commit,
        scope="formal",
        upstream_evidence=upstream,
    )
    direct_checks, replay = _direct_checks(repository_root, evidence, upstream)
    role_filenames = {
        "gradient_scale_report": "gradient-scale-report.json",
        "comparison_table": "comparison-table.json",
        "gate_record": "g1-grad-record.json",
    }
    role_paths: dict[str, Path] = {}
    for role, filename in role_filenames.items():
        path = work_dir / filename
        _write_new(path, evidence[role])
        role_paths[role] = path
    reloaded = {role: load_canonical_json(path) for role, path in role_paths.items()}
    _, replay_after_roundtrip = _direct_checks(repository_root, reloaded, upstream)
    replay_path = work_dir / "replay-validation.json"
    _write_new(replay_path, replay_after_roundtrip)
    csv_sha, svg_sha = _write_chart_data(work_dir, reloaded)
    _require_exact_digest_map(
        csv_sha,
        {"tensor-errors.csv", "gradient-scatter.csv", "tensor-error-heatmap.csv", "accumulation-errors.csv"},
        field="csv_sha256",
    )
    _require_exact_digest_map(
        svg_sha,
        {"gradient-scatter.svg", "route-errors.svg", "tensor-error-heatmap.svg", "accumulation-errors.svg"},
        field="svg_sha256",
    )
    validation = _with_artifact_hash(
        {
            "schema_version": VALIDATION_SCHEMA,
            "status": "PASS",
            "gate_id": GATE_ID,
            "task_id": TASK_ID,
            "execution_scope": "formal_server_cpu",
            "fixture_id": FIXTURE_ID,
            "producer_commit": commit,
            "consumer_commit": commit,
            "upstream": {
                key: upstream[key]
                for key in (
                    "s1_3_index_ref",
                    "s1_3_index_sha256",
                    "s1_3_gate_artifact_hash",
                    "s1_3_frozen_gradient_input_hash",
                )
            },
            "regression": regression,
            "direct_checks": direct_checks,
            "role_sha256": {role: sha256_file(path) for role, path in role_paths.items()},
            "csv_sha256": csv_sha,
            "svg_sha256": svg_sha,
            "replay_sha256": sha256_file(replay_path),
            "replay_hash": replay_after_roundtrip["replay_hash"],
        }
    )
    validation_path = work_dir / "validation.json"
    _write_new(validation_path, validation)
    index = _with_artifact_hash(
        {
            "schema_version": INDEX_SCHEMA,
            "status": "PASS",
            "gate_id": GATE_ID,
            "task_id": TASK_ID,
            "fixture_id": FIXTURE_ID,
            "generator_git_commit": commit,
            "consumer_git_commit": commit,
            "git_branch": _git(repository_root, "branch", "--show-current"),
            "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **upstream,
            "gradient_scale_report_ref": role_filenames["gradient_scale_report"],
            "gradient_scale_report_sha256": sha256_file(role_paths["gradient_scale_report"]),
            "gradient_scale_report_hash": evidence["gradient_scale_report"]["report_hash"],
            "comparison_table_ref": role_filenames["comparison_table"],
            "comparison_table_sha256": sha256_file(role_paths["comparison_table"]),
            "comparison_table_hash": evidence["comparison_table"]["table_hash"],
            "gate_record_ref": role_filenames["gate_record"],
            "gate_record_sha256": sha256_file(role_paths["gate_record"]),
            "gate_artifact_hash": evidence["gate_record"]["artifact_hash"],
            "csv_sha256": csv_sha,
            "svg_sha256": svg_sha,
            "validation_ref": "validation.json",
            "validation_sha256": sha256_file(validation_path),
            "replay_ref": "replay-validation.json",
            "replay_sha256": sha256_file(replay_path),
            "replay_hash": replay_after_roundtrip["replay_hash"],
            "next_task_id": "stage1.05_estimators",
        }
    )
    staging_dir = evidence_dir.parent / f".{attempt_id}.publishing"
    if staging_dir.exists():
        raise Stage1S14FormalError(f"S1_4_PUBLISH_STAGING_ALREADY_EXISTS:{staging_dir}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    for source in (*role_paths.values(), replay_path, validation_path, *sorted(work_dir.glob("*.csv")), *sorted(work_dir.glob("*.svg"))):
        (staging_dir / source.name).write_bytes(source.read_bytes())
    index_path = staging_dir / "index.json"
    write_canonical_json(index_path, index)
    loaded_index = load_canonical_json(index_path)
    if not isinstance(loaded_index, Mapping) or loaded_index.get("artifact_hash") != index["artifact_hash"]:
        raise Stage1S14FormalError("S1_4_INDEX_RELOAD_FAILED")
    published_index_path = _publish_staging_directory(
        staging_dir, evidence_dir, expected_artifact_hash=index["artifact_hash"]
    )
    shutil.rmtree(work_dir)
    return {
        "index_ref": published_index_path.relative_to(data_root_path).as_posix(),
        "validation_ref": (evidence_dir / "validation.json").relative_to(data_root_path).as_posix(),
        "gate_record_ref": (evidence_dir / "g1-grad-record.json").relative_to(data_root_path).as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-3-index-ref", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    arguments = parser.parse_args(argv)
    print(
        execute(
            repository=arguments.repository,
            data_root=arguments.data_root,
            s1_3_index_ref=arguments.s1_3_index_ref,
            attempt_id=arguments.attempt_id,
            timeout_seconds=arguments.timeout_seconds,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
