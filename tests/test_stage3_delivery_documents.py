from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "ops" / "stage3" / "render_stage3_delivery_documents.py"
if not RENDERER.is_file():
    RENDERER = ROOT / ".agent-temp" / "render_stage3_delivery_documents.py"
SPEC = importlib.util.spec_from_file_location(
    "stage3_delivery_documents",
    RENDERER,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _figure(root: Path, name: str, color: str) -> dict[str, object]:
    png = root / "inputs" / f"{name}.png"
    svg = root / "inputs" / f"{name}.svg"
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR42mNkYGD4z8DAwMgABYwMDAAAMgIBBIgq9QAAAABJRU5ErkJggg=="
        )
    )
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560"><title>{name}</title><rect width="1000" height="560" fill="white"/><polyline points="110,450 300,320 520,350 720,180 900,120" fill="none" stroke="{color}" stroke-width="12"/></svg>\n',
        encoding="utf-8",
        newline="\n",
    )
    return {"id": name, "png": _record(root, png), "svg": _record(root, svg)}


def test_hash_bound_cjk_report_and_slides_are_reproducible(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    PdfReader = pytest.importorskip("pypdf").PdfReader
    figures = [
        _figure(tmp_path, "stage3-cost-error", "#0b6e75"),
        _figure(tmp_path, "stage3-path-loss", "#d97706"),
    ]
    analysis = {
        "schema_version": "analysis-report-v1",
        "metadata": {"scope": "formal", "formal_eligible": True},
        "metrics": {
            "mean_normalized_l1_error": {"value": 0.0125},
            "nodes_error_pearson": {"value": -0.88},
            "raw_path_loss_mean": {"value": 2.75},
            **{
                f"formal_metric_{index:02d}": {"value": index / 1000.0}
                for index in range(4, 17)
            },
        },
    }
    charts = {
        "schema_version": "stage3-task-chart-artifacts-v1",
        "scope": "formal",
        "formal_eligible": True,
        "rendered_figures": figures,
    }
    handoff = {
        "schema_version": "stage3-task-handoff-manifest-v1",
        "scope": "formal",
        "formal_eligible": True,
        "default_rule": "simpson",
        "fallback_rule": "gauss_legendre_8",
        "passing_rules": ["simpson", "gauss_legendre_8"],
        "formal_stage_complete": False,
        "completion_boundary": "PENDING_G3_8_DELIVERY_ACCEPTANCE",
    }
    gates = {
        "schema_version": "stage3-task-gate-summary-v1",
        "scope": "formal",
        "stage3.G3-6": "PASS",
        "stage3.G3-7": "PASS",
        "stage3.G3-8": "NOT_RUN",
        "formal_exit_gate": "NOT_RUN",
    }
    source_refs = {role: f"results/stage3/s310/commits/{role}.json" for role in MODULE.REQUIRED_KINDS}
    identities = {
        role: {
            "task_id": MODULE.TASK_ID,
            "artifact_kind": role,
            "config_hash": "1" * 64,
            "artifact_hash": f"{index + 2:064x}",
            "payload_hash": f"{index + 10:064x}",
        }
        for index, role in enumerate(MODULE.REQUIRED_KINDS)
    }
    font = next(
        (
            candidate
            for candidate in (
                Path(r"C:\Windows\Fonts\msyh.ttc"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            )
            if candidate.is_file()
        ),
        None,
    )
    if font is None:
        pytest.skip("no usable TrueType font")
    workspace_font = tmp_path / "assets" / f"formal-cjk{font.suffix}"
    workspace_font.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(font, workspace_font)

    manifests = []
    for name in ("output/pdf/first", "output/pdf/second"):
        manifests.append(
            MODULE.render_stage3_delivery_documents(
                workspace_root=tmp_path,
                output_dir=name,
                analysis_report=analysis,
                chart_artifacts=charts,
                handoff_manifest=handoff,
                gate_summary=gates,
                source_refs=source_refs,
                input_identities=identities,
                producer_commit="a" * 40,
                font_file=workspace_font,
            )
        )

    first = manifests[0]["files"]
    second = manifests[1]["files"]
    assert first["chinese_report"]["pdf"]["sha256"] == second["chinese_report"]["pdf"]["sha256"]
    assert first["beamer"]["pdf"]["sha256"] == second["beamer"]["pdf"]["sha256"]
    assert len(first["beamer"]["backups"]) == 4
    assert manifests[0]["completion_boundary"] == "PENDING_G3_8_DELIVERY_ACCEPTANCE"
    assert manifests[0]["font"] == {
        "name": "Stage3CJK",
        "path": workspace_font.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(workspace_font.read_bytes()).hexdigest(),
        "size": workspace_font.stat().st_size,
    }

    report_path = tmp_path / first["chinese_report"]["pdf"]["path"]
    slides_path = tmp_path / first["beamer"]["pdf"]["path"]
    assert len(PdfReader(str(report_path)).pages) >= 4
    slide_pages = PdfReader(str(slides_path)).pages
    assert len(slide_pages) >= 12
    backup_metric_pages = [
        page.extract_text() or ""
        for page in slide_pages
        if any(
            tag in (page.extract_text() or "")
            for tag in ("BACKUP 2/5", "BACKUP 3/5")
        )
    ]
    assert len(backup_metric_pages) == 2
    backup_metric_text = "\n".join(backup_metric_pages)
    assert all(
        name in backup_metric_text
        for name in (
            "mean_normalized_l1_error",
            "nodes_error_pearson",
            "raw_path_loss_mean",
            *(f"formal_metric_{index:02d}" for index in range(4, 17)),
        )
    )
    assert report_path.read_bytes().startswith(b"%PDF-")
    assert slides_path.read_bytes().startswith(b"%PDF-")


def test_renderer_rejects_completed_or_unverified_inputs(tmp_path: Path) -> None:
    figure = _figure(tmp_path, "overview", "#0b6e75")
    analysis = {"metadata": {"scope": "formal", "formal_eligible": True}, "metrics": {"m": {"value": 1.0}}}
    charts = {"scope": "formal", "formal_eligible": True, "rendered_figures": [figure]}
    handoff = {"scope": "formal", "formal_eligible": True, "default_rule": "simpson", "fallback_rule": "gl8", "formal_stage_complete": True, "completion_boundary": "COMPLETE"}
    gates = {"scope": "formal", "stage3.G3-6": "PASS", "stage3.G3-7": "PASS", "stage3.G3-8": "PASS", "formal_exit_gate": "PASS"}
    with pytest.raises(ValueError, match="pending independent G3-8"):
        MODULE._validate_semantics(analysis, charts, handoff, gates)

    tampered = dict(figure)
    tampered["png"] = dict(figure["png"], sha256="f" * 64)
    with pytest.raises(ValueError, match="hash drift"):
        MODULE._figure_inputs(tmp_path, {"rendered_figures": [tampered]})

    outside = tmp_path.parent / "outside-font.ttf"
    outside.write_bytes(b"not-a-font")
    with pytest.raises(ValueError, match="inside workspace_root"):
        MODULE._register_font(tmp_path, outside)


def test_metric_rows_preserve_undefined_reason_from_real_wire_shape() -> None:
    rows = MODULE._metric_rows(
        {
            "metrics": {
                "defined": {
                    "defined": True,
                    "value": 0.25,
                    "reason": None,
                    "metadata": {},
                },
                "undefined": {
                    "defined": False,
                    "value": None,
                    "reason": "constant input",
                    "metadata": {},
                },
            }
        }
    )
    assert rows == [
        ("defined", "0.25"),
        ("undefined", "undefined (constant input)"),
    ]
