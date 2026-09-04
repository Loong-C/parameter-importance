"""Render the human-facing Stage 3 delivery documents from formal S3.10 commits.

The renderer is intentionally downstream-only: it accepts the four immutable
S3.10 task commits, verifies their shared formal identity and the recorded
figure bytes, and then renders Chinese report and Beamer-equivalent PDFs.  It
does not calculate scientific results or change the G3-8 status.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if not (_REPOSITORY_ROOT / "src").is_dir():
    _REPOSITORY_ROOT = Path.cwd().resolve()
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

JSONValue = Any


DOCUMENT_SCHEMA = "stage3-delivery-documents-v1"
TASK_ID = "stage3.10_reports_visualizations_and_handoff"
REQUIRED_KINDS = (
    "analysis_report",
    "chart_artifacts",
    "handoff_manifest",
    "gate_summary",
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_json_hash(value: object) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_canonical_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ValueError(f"immutable output differs: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _safe_ref(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "?" in value or "://" in value or "\\" in value:
        raise ValueError(f"{field} must be a stable POSIX workspace ref")
    ref = PurePosixPath(value)
    if ref.is_absolute() or not ref.parts or any(part in {"", ".", ".."} for part in ref.parts):
        raise ValueError(f"{field} escapes the workspace")
    return ref.as_posix()


def _workspace_file(root: Path, value: object, *, field: str) -> tuple[Path, dict[str, JSONValue]]:
    record = _mapping(value, field=field)
    if set(record) != {"path", "sha256", "size"}:
        raise ValueError(f"{field} must contain exactly path/sha256/size")
    ref = _safe_ref(record["path"], field=f"{field}.path")
    digest = record["sha256"]
    size = record["size"]
    if not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None:
        raise ValueError(f"{field}.sha256 is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{field}.size is invalid")
    path = root.joinpath(*PurePosixPath(ref).parts)
    current = root
    for part in PurePosixPath(ref).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} contains a symlink")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes the workspace") from error
    payload = resolved.read_bytes()
    if len(payload) != size:
        raise ValueError(f"{field} size drift")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{field} hash drift")
    return resolved, {"path": ref, "sha256": digest, "size": size}


def _output_record(root: Path, path: Path) -> dict[str, JSONValue]:
    resolved = path.resolve(strict=True)
    try:
        ref = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("generated document escapes workspace") from error
    payload = resolved.read_bytes()
    return {"path": ref, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def _tex(value: object) -> str:
    text = str(value)
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
    return "".join(replacements.get(char, char) for char in text)


def _short(value: object, limit: int = 100) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _metric_rows(report: Mapping[str, object]) -> list[tuple[str, str]]:
    raw = report.get("metrics")
    rows: list[tuple[str, str]] = []
    if isinstance(raw, Mapping):
        for name, value in sorted(raw.items()):
            if isinstance(value, Mapping):
                if value.get("defined") is False:
                    reason = value.get("reason")
                    candidate = f"undefined ({reason})" if reason is not None else "undefined"
                else:
                    candidate = value.get(
                        "value", value.get("estimate", value.get("result", value))
                    )
            else:
                candidate = value
            rows.append((str(name), _short(candidate, 80)))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for index, value in enumerate(raw):
            item = _mapping(value, field=f"analysis_report.metrics[{index}]")
            name = item.get("metric_id", item.get("name", f"metric-{index + 1}"))
            candidate = item.get("value", item.get("estimate", item.get("result", item)))
            rows.append((str(name), _short(candidate, 80)))
    if not rows:
        raise ValueError("analysis_report.metrics is empty")
    return rows


def _validate_semantics(
    analysis: Mapping[str, object],
    charts: Mapping[str, object],
    handoff: Mapping[str, object],
    gates: Mapping[str, object],
) -> None:
    metadata = _mapping(analysis.get("metadata"), field="analysis_report.metadata")
    if metadata.get("scope") != "formal" or metadata.get("formal_eligible") is not True:
        raise ValueError("analysis report is not formal eligible")
    if charts.get("scope") != "formal" or charts.get("formal_eligible") is not True:
        raise ValueError("chart artifacts are not formal eligible")
    if handoff.get("scope") != "formal" or handoff.get("formal_eligible") is not True:
        raise ValueError("handoff manifest is not formal eligible")
    if handoff.get("formal_stage_complete") is not False or handoff.get("completion_boundary") != "PENDING_G3_8_DELIVERY_ACCEPTANCE":
        raise ValueError("handoff must remain pending independent G3-8")
    if gates.get("scope") != "formal":
        raise ValueError("gate summary is not formal")
    expected = {"stage3.G3-6": "PASS", "stage3.G3-7": "PASS", "stage3.G3-8": "NOT_RUN", "formal_exit_gate": "NOT_RUN"}
    for key, value in expected.items():
        if gates.get(key) != value:
            raise ValueError(f"gate summary has unexpected {key}")
    if not isinstance(handoff.get("default_rule"), str) or not isinstance(handoff.get("fallback_rule"), str):
        raise ValueError("handoff rule selection is missing")
    _metric_rows(analysis)


def _figure_inputs(root: Path, charts: Mapping[str, object]) -> list[dict[str, object]]:
    raw = charts.get("rendered_figures")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("chart_artifacts.rendered_figures is empty")
    figures: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        item = _mapping(value, field=f"rendered_figures[{index}]")
        figure_id = item.get("id")
        if not isinstance(figure_id, str) or not figure_id or figure_id in seen:
            raise ValueError("rendered figure id is invalid or duplicated")
        png_path, png = _workspace_file(root, item.get("png"), field=f"rendered_figures[{index}].png")
        _svg_path, svg = _workspace_file(root, item.get("svg"), field=f"rendered_figures[{index}].svg")
        figures.append({"id": figure_id, "png_path": png_path, "png": png, "svg": svg})
        seen.add(figure_id)
    return figures


def _register_font(root: Path, font_file: Path | None) -> tuple[str, dict[str, JSONValue]]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if font_file is None:
        raise ValueError("font_file is required for the formal document renderer")
    candidate = font_file if font_file.is_absolute() else root / font_file
    candidate = Path(os.path.abspath(candidate))
    try:
        ref = candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("font_file must stay inside workspace_root") from error
    current = root
    for part in PurePosixPath(ref).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("font_file contains a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("font_file must stay inside workspace_root") from error
    if resolved.suffix.casefold() not in {".ttf", ".otf", ".ttc"}:
        raise ValueError("font_file must be a TTF, OTF, or TTC file")
    payload = resolved.read_bytes()
    if not payload:
        raise ValueError("font file is empty")
    font_name = "Stage3CJK"
    pdfmetrics.registerFont(TTFont(font_name, str(resolved), subfontIndex=0))
    return font_name, {
        "name": font_name,
        "path": ref,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _paragraph(text: str, style: Any) -> Any:
    from reportlab.platypus import Paragraph

    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return Paragraph(escaped, style)


def _scaled_image(path: Path, *, max_width: float, max_height: float) -> Any:
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image

    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _write_report_pdf(
    path: Path,
    *,
    font_name: str,
    metrics: Sequence[tuple[str, str]],
    handoff: Mapping[str, object],
    gates: Mapping[str, object],
    figures: Sequence[Mapping[str, object]],
    source_refs: Mapping[str, str],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

    class InvariantCanvas(Canvas):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["invariant"] = 1
            super().__init__(*args, **kwargs)

    styles = getSampleStyleSheet()
    title = ParagraphStyle("cjk-title", parent=styles["Title"], fontName=font_name, fontSize=23, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#17324d"))
    heading = ParagraphStyle("cjk-heading", parent=styles["Heading1"], fontName=font_name, fontSize=15, leading=21, textColor=colors.HexColor("#0b6e75"), spaceBefore=7, spaceAfter=6)
    body = ParagraphStyle("cjk-body", parent=styles["BodyText"], fontName=font_name, fontSize=9.5, leading=15, wordWrap="CJK", textColor=colors.HexColor("#263238"))
    small = ParagraphStyle("cjk-small", parent=body, fontSize=7.2, leading=10)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=17 * mm, title="Stage 3 正式实验分析报告", author="parameter-importance formal pipeline")

    def footer(canvas: Canvas, document: Any) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 7)
        canvas.setFillColor(colors.HexColor("#607d8b"))
        canvas.drawString(18 * mm, 9 * mm, "Stage 3 formal delivery - hash-bound source")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    story: list[Any] = [
        Spacer(1, 18 * mm),
        _paragraph("参数重要性 Stage 3 正式实验分析报告", title),
        Spacer(1, 8 * mm),
        _paragraph("数据边界：S3.10 四个不可变正式提交；本文档不重新计算任何科学结果。", body),
        Spacer(1, 5 * mm),
        _paragraph("结论状态", heading),
        _paragraph(
            f"默认积分规则：{handoff['default_rule']}；回退规则：{handoff['fallback_rule']}。"
            "G3-6 与 G3-7 已通过；Stage 3 仍等待独立 G3-8 交付验收，本文档不会提前宣称完成。",
            body,
        ),
        _paragraph("Gate 状态", heading),
    ]
    gate_rows = [[_paragraph("Gate", small), _paragraph("状态", small)]] + [
        [_paragraph(key, small), _paragraph(str(gates[key]), small)]
        for key in ("stage3.G3-6", "stage3.G3-7", "stage3.G3-8", "formal_exit_gate")
    ]
    gate_table = Table(gate_rows, colWidths=[80 * mm, 55 * mm], repeatRows=1)
    gate_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eef0")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#90a4ae")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.extend([gate_table, _paragraph("正式指标", heading)])
    metric_rows = [[_paragraph("指标", small), _paragraph("正式值/记录", small)]] + [[_paragraph(name, small), _paragraph(value, small)] for name, value in metrics]
    metric_table = Table(metric_rows, colWidths=[70 * mm, 92 * mm], repeatRows=1)
    metric_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b0bec5")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.extend([metric_table, PageBreak()])
    for index, figure in enumerate(figures):
        story.extend([
            _paragraph(f"图 {index + 1}：{figure['id']}", heading),
            _scaled_image(Path(figure["png_path"]), max_width=165 * mm, max_height=205 * mm),
            Spacer(1, 3 * mm),
            _paragraph(f"PNG SHA-256: {figure['png']['sha256']}", small),
            _paragraph(f"SVG SHA-256: {figure['svg']['sha256']}", small),
            PageBreak(),
        ])
    story.extend([_paragraph("谱系与重放入口", heading)])
    for role, ref in sorted(source_refs.items()):
        story.append(_paragraph(f"{role}: {ref}", small))
    story.extend([
        _paragraph("解释边界", heading),
        _paragraph("所有数值来自 S3.10 的哈希绑定正式输出。图表引用的 PNG/SVG 已按记录的字节数和 SHA-256 重新核验。G3-8 必须由独立消费者在文档、小文件清单、大资产清单、回放证据和 Git 同步证据齐备后发布。", body),
    ])
    doc.build(story, onFirstPage=footer, onLaterPages=footer, canvasmaker=InvariantCanvas)


def _wrapped_lines(text: str, *, width_chars: int) -> list[str]:
    import textwrap

    return textwrap.wrap(
        text,
        width=width_chars,
        break_long_words=True,
        replace_whitespace=False,
    ) or [""]


def _draw_wrapped(canvas: Any, text: str, *, x: float, y: float, width_chars: int, leading: float, font_name: str, font_size: float) -> float:
    canvas.setFont(font_name, font_size)
    for line in _wrapped_lines(text, width_chars=width_chars):
        canvas.drawString(x, y, line)
        y -= leading
    return y


def _paginate_slide_bullets(
    bullets: Sequence[str],
    *,
    width_chars: int,
    available_height: float,
    leading: float,
    spacing: float,
) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    used = 0.0
    for bullet in bullets:
        required = len(_wrapped_lines(bullet, width_chars=width_chars)) * leading + spacing
        if required > available_height:
            raise ValueError("one slide bullet exceeds the safe content height")
        if current and used + required > available_height:
            pages.append(current)
            current = []
            used = 0.0
        current.append(bullet)
        used += required
    if current or not pages:
        pages.append(current)
    return pages


def _write_slides_pdf(
    path: Path,
    *,
    font_name: str,
    metrics: Sequence[tuple[str, str]],
    handoff: Mapping[str, object],
    gates: Mapping[str, object],
    figures: Sequence[Mapping[str, object]],
    source_refs: Mapping[str, str],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as canvas_module

    page = (13.333 * inch, 7.5 * inch)
    pdf = canvas_module.Canvas(str(path), pagesize=page, invariant=1, pageCompression=1)
    pdf.setTitle("Stage 3 正式实验汇报")

    def frame(title: str, bullets: Sequence[str], image: Path | None = None, tag: str = "FORMAL") -> None:
        width, height = page
        pdf.setFillColor(colors.HexColor("#f5f7fa")); pdf.rect(0, 0, width, height, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor("#17324d")); pdf.rect(0, height - 0.76 * inch, width, 0.76 * inch, fill=1, stroke=0)
        pdf.setFillColor(colors.white); pdf.setFont(font_name, 22); pdf.drawString(0.48 * inch, height - 0.50 * inch, title)
        pdf.setFillColor(colors.HexColor("#0b6e75")); pdf.setFont(font_name, 8); pdf.drawRightString(width - 0.45 * inch, height - 0.48 * inch, tag)
        text_width = 70 if image is None else 44
        y = height - 1.20 * inch
        bottom_safe = 0.58 * inch
        pdf.setFillColor(colors.HexColor("#263238"))
        for bullet in bullets:
            required = len(_wrapped_lines(bullet, width_chars=text_width)) * 0.29 * inch + 0.14 * inch
            if y - required < bottom_safe:
                raise ValueError(f"slide content exceeds safe footer boundary: {title}")
            pdf.setFont(font_name, 16); pdf.drawString(0.58 * inch, y, "•")
            y = _draw_wrapped(pdf, bullet, x=0.88 * inch, y=y, width_chars=text_width, leading=0.29 * inch, font_name=font_name, font_size=13)
            y -= 0.14 * inch
        if image is not None:
            iw, ih = ImageReader(str(image)).getSize()
            max_w, max_h = 5.75 * inch, 5.55 * inch
            scale = min(max_w / iw, max_h / ih)
            pdf.drawImage(str(image), width - 0.45 * inch - iw * scale, 0.48 * inch, width=iw * scale, height=ih * scale, preserveAspectRatio=True, mask="auto")
        pdf.setFillColor(colors.HexColor("#78909c")); pdf.setFont(font_name, 7); pdf.drawString(0.48 * inch, 0.24 * inch, "parameter-importance / Stage 3 / immutable S3.10 sources")
        pdf.drawRightString(width - 0.45 * inch, 0.24 * inch, str(pdf.getPageNumber()))
        pdf.showPage()

    frame("Stage 3 正式实验汇报", ["99 个正式路径积分单元与独立 Gate 链", "报告由 S3.10 哈希绑定提交和已验证图表生成", "完成边界：等待独立 G3-8 交付验收"])
    frame("结论与方法选择", [f"默认规则：{handoff['default_rule']}", f"回退规则：{handoff['fallback_rule']}", f"通过规则：{_short(handoff.get('passing_rules', []), 160)}", "成本语义：measured callback cost and unique nodes"])
    metric_bullets = [f"{name}: {value}" for name, value in metrics]
    metric_pages = _paginate_slide_bullets(
        metric_bullets,
        width_chars=70,
        available_height=(7.5 - 1.20 - 0.58) * inch,
        leading=0.29 * inch,
        spacing=0.14 * inch,
    )
    frame("正式指标", metric_pages[0])
    for figure in figures:
        frame(str(figure["id"]), ["正式 S3.10 图表", f"PNG: {str(figure['png']['sha256'])[:16]}...", f"SVG: {str(figure['svg']['sha256'])[:16]}..."], Path(figure["png_path"]))
    frame("Gate 与完成边界", [f"{key}: {gates[key]}" for key in ("stage3.G3-6", "stage3.G3-7", "stage3.G3-8", "formal_exit_gate")])
    frame("下一步", ["物化小文件交付清单与服务器大资产清单", "完成本地 CPU、锁定服务器与冻结端点无缓存三层语义回放", "验证分支、提交、推送、远端、服务器 clean HEAD 与多端同步", "由独立 G3-8 消费者验收并发布 Stage4 handoff"])
    backup_total = 3 + len(metric_pages)
    frame("备份：核心谱系", [f"{role}: {_short(ref, 130)}" for role, ref in sorted(source_refs.items())], tag=f"BACKUP 1/{backup_total}")
    for index, bullets in enumerate(metric_pages, start=1):
        suffix = f" ({index}/{len(metric_pages)})" if len(metric_pages) > 1 else ""
        frame(
            f"备份：指标定义{suffix}",
            bullets,
            tag=f"BACKUP {index + 1}/{backup_total}",
        )
    frame("备份：图表字节绑定", [f"{figure['id']}: PNG {figure['png']['sha256'][:20]}... / SVG {figure['svg']['sha256'][:20]}..." for figure in figures], tag=f"BACKUP {backup_total - 1}/{backup_total}")
    frame("备份：验收边界", ["G3-8 不生成科学结果，只消费已完成交付", "所有文件须重新计算大小与 SHA-256", "大资产保持在服务器 DATA_ROOT，Git 仅保存代码与小文件", "Stage4 仅在 G3-8 PASS 后接收 handoff"], tag=f"BACKUP {backup_total}/{backup_total}")
    pdf.save()


def _write_tex_sources(
    output: Path,
    *,
    metrics: Sequence[tuple[str, str]],
    handoff: Mapping[str, object],
    gates: Mapping[str, object],
    figures: Sequence[Mapping[str, object]],
    source_refs: Mapping[str, str],
) -> tuple[Path, Path, Path, list[Path]]:
    report_tex = output / "stage3-formal-report-zh.tex"
    beamer_tex = output / "stage3-formal-beamer.tex"
    notes = output / "stage3-formal-speaker-notes.md"
    backups = [output / f"stage3-formal-backup-{index:02d}.tex" for index in range(1, 5)]

    figure_tex = []
    for index, figure in enumerate(figures, start=1):
        png_ref = os.path.relpath(Path(figure["png_path"]), start=output).replace("\\", "/")
        figure_tex.append(f"\\section{{图 {index}: {_tex(figure['id'])}}}\n\\begin{{center}}\\includegraphics[width=0.94\\linewidth]{{\\detokenize{{{png_ref}}}}}\\end{{center}}\n\\small PNG: \\texttt{{{figure['png']['sha256']}}}\\\\ SVG: \\texttt{{{figure['svg']['sha256']}}}\n")
    metric_lines = "\n".join(f"{_tex(name)} & {_tex(value)} \\\\" for name, value in metrics)
    source_lines = "\n".join(f"\\item \\texttt{{{_tex(role)}}}: \\path{{{ref}}}" for role, ref in sorted(source_refs.items()))
    report_tex.write_text(
        "\\documentclass[UTF8,11pt]{ctexart}\n\\usepackage[a4paper,margin=2cm]{geometry}\n\\usepackage{booktabs,longtable,graphicx,xcolor,hyperref}\n\\title{参数重要性 Stage 3 正式实验分析报告}\n\\author{parameter-importance formal pipeline}\n\\date{}\n\\begin{document}\n\\maketitle\n\\section{结论状态}\n"
        f"默认规则：\\texttt{{{_tex(handoff['default_rule'])}}}；回退规则：\\texttt{{{_tex(handoff['fallback_rule'])}}}。G3-6 与 G3-7 已通过；G3-8 尚未运行，Stage 3 尚未越过最终交付边界。\n"
        "\\section{Gate 状态}\n\\begin{longtable}{ll}\\toprule Gate & 状态 \\\\ \\midrule\n"
        + "\n".join(f"{_tex(key)} & {_tex(gates[key])} \\\\" for key in ("stage3.G3-6", "stage3.G3-7", "stage3.G3-8", "formal_exit_gate"))
        + "\n\\bottomrule\\end{longtable}\n\\section{正式指标}\n\\begin{longtable}{p{0.42\\linewidth}p{0.50\\linewidth}}\\toprule 指标 & 正式值/记录 \\\\ \\midrule\n"
        + metric_lines
        + "\n\\bottomrule\\end{longtable}\n"
        + "\n".join(figure_tex)
        + "\\section{谱系与重放入口}\n\\begin{itemize}\n"
        + source_lines
        + "\n\\end{itemize}\n\\section{解释边界}\n所有数值来自 S3.10 的不可变正式提交，图表文件已按大小与 SHA-256 复核。独立 G3-8 只有在文档、清单、三层回放和 Git 多端同步证据齐备后才可发布 Stage4 handoff。\n\\end{document}\n",
        encoding="utf-8",
        newline="\n",
    )

    backup_frames = [
        ("核心谱系", [f"{role}: {ref}" for role, ref in sorted(source_refs.items())]),
        ("指标定义", [f"{name}: {value}" for name, value in metrics]),
        ("图表字节绑定", [f"{figure['id']}: PNG {figure['png']['sha256']} / SVG {figure['svg']['sha256']}" for figure in figures]),
        ("验收边界", ["G3-8 不生成科学结果，只消费已完成交付", "所有文件重新计算大小与 SHA-256", "大资产留在服务器 DATA_ROOT", "G3-8 PASS 后才发布 Stage4 handoff"]),
    ]
    for path, (title, bullets) in zip(backups, backup_frames):
        path.write_text("\\begin{frame}[allowframebreaks]{" + _tex(title) + "}\n\\begin{itemize}\n" + "\n".join(f"\\item {_tex(item)}" for item in bullets) + "\n\\end{itemize}\n\\end{frame}\n", encoding="utf-8", newline="\n")
    figure_frames = "\n".join(
        "\\begin{frame}{" + _tex(figure["id"]) + "}\n\\centering\\includegraphics[width=0.92\\linewidth,height=0.78\\textheight,keepaspectratio]{\\detokenize{" + os.path.relpath(Path(figure["png_path"]), start=output).replace("\\", "/") + "}}\n\\end{frame}"
        for figure in figures
    )
    beamer_tex.write_text(
        "\\documentclass[UTF8,aspectratio=169]{ctexbeamer}\n\\usepackage{graphicx}\n\\usetheme{default}\n\\definecolor{stageblue}{HTML}{17324D}\n\\setbeamercolor{structure}{fg=stageblue}\n\\title{Stage 3 正式实验汇报}\n\\subtitle{99 个路径积分单元、独立 Gate 与交付边界}\n\\author{parameter-importance formal pipeline}\n\\date{}\n\\begin{document}\n\\frame{\\titlepage}\n"
        "\\begin{frame}{结论与方法选择}\\begin{itemize}\n"
        f"\\item 默认规则：\\texttt{{{_tex(handoff['default_rule'])}}}\n\\item 回退规则：\\texttt{{{_tex(handoff['fallback_rule'])}}}\n\\item G3-6/G3-7 已通过，G3-8 尚待独立验收\n\\end{{itemize}}\\end{{frame}}\n"
        "\\begin{frame}{正式指标}\\begin{itemize}\n"
        + "\n".join(f"\\item {_tex(name)}: {_tex(value)}" for name, value in metrics)
        + "\n\\end{itemize}\\end{frame}\n"
        + figure_frames
        + "\n\\begin{frame}{下一步}\\begin{enumerate}\\item 物化交付与大资产清单\\item 完成三层语义回放\\item 验证 Git 多端同步\\item 独立 G3-8 发布 Stage4 handoff\\end{enumerate}\\end{frame}\n"
        + "\n".join(f"\\input{{{path.name}}}" for path in backups)
        + "\n\\end{document}\n",
        encoding="utf-8",
        newline="\n",
    )
    notes.write_text(
        "# Stage 3 正式汇报讲稿\n\n"
        "- 开场：说明全部内容来自不可变 S3.10 正式提交，文档没有重新计算或手工改数。\n"
        f"- 方法：默认规则 `{handoff['default_rule']}`，回退规则 `{handoff['fallback_rule']}`。\n"
        "- Gate：G3-6 与 G3-7 已通过；G3-8 和 formal exit gate 仍未运行。\n"
        "- 图表：逐图说明成本-误差、完整性残差、墙钟时间、原始向量与路径损失的正式来源。\n"
        "- 边界：小文件、大资产、三层回放和 Git 多端同步全部验收后，才能发布 Stage4 handoff。\n"
        "- 备份页：核心谱系、指标定义、图表字节绑定、验收边界。\n",
        encoding="utf-8",
        newline="\n",
    )
    return report_tex, beamer_tex, notes, backups


def render_stage3_delivery_documents(
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    analysis_report: Mapping[str, object],
    chart_artifacts: Mapping[str, object],
    handoff_manifest: Mapping[str, object],
    gate_summary: Mapping[str, object],
    source_refs: Mapping[str, str],
    input_identities: Mapping[str, Mapping[str, object]],
    producer_commit: str,
    font_file: str | Path | None,
) -> Mapping[str, object]:
    root = Path(workspace_root).resolve(strict=True)
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    out = Path(os.path.abspath(out))
    try:
        out.relative_to(root)
    except ValueError as error:
        raise ValueError("output_dir must stay inside workspace_root") from error
    if not _GIT_RE.fullmatch(producer_commit):
        raise ValueError("producer_commit must be a 40-character Git commit")
    if set(source_refs) != set(REQUIRED_KINDS) or set(input_identities) != set(REQUIRED_KINDS):
        raise ValueError("exactly four S3.10 input roles are required")
    _validate_semantics(analysis_report, chart_artifacts, handoff_manifest, gate_summary)
    figures = _figure_inputs(root, chart_artifacts)
    metrics = _metric_rows(analysis_report)
    out.mkdir(parents=True, exist_ok=True)
    font_name, font_record = _register_font(
        root,
        None if font_file is None else Path(font_file),
    )
    report_tex, beamer_tex, notes, backups = _write_tex_sources(out, metrics=metrics, handoff=handoff_manifest, gates=gate_summary, figures=figures, source_refs=source_refs)
    report_pdf = out / "stage3-formal-report-zh.pdf"
    beamer_pdf = out / "stage3-formal-beamer.pdf"
    _write_report_pdf(report_pdf, font_name=font_name, metrics=metrics, handoff=handoff_manifest, gates=gate_summary, figures=figures, source_refs=source_refs)
    _write_slides_pdf(beamer_pdf, font_name=font_name, metrics=metrics, handoff=handoff_manifest, gates=gate_summary, figures=figures, source_refs=source_refs)

    files = {
        "chinese_report": {"tex": _output_record(root, report_tex), "pdf": _output_record(root, report_pdf)},
        "beamer": {
            "tex": _output_record(root, beamer_tex),
            "pdf": _output_record(root, beamer_pdf),
            "notes": [_output_record(root, notes)],
            "backups": [_output_record(root, path) for path in backups],
        },
    }
    body: dict[str, JSONValue] = {
        "schema_version": DOCUMENT_SCHEMA,
        "status": "PASS",
        "scope": "formal",
        "formal_eligible": True,
        "producer_commit": producer_commit,
        "renderer": {"name": "reportlab", "version": importlib.metadata.version("reportlab"), "invariant_pdf": True},
        "font": font_record,
        "inputs": {role: dict(input_identities[role]) for role in REQUIRED_KINDS},
        "source_refs": {role: source_refs[role] for role in REQUIRED_KINDS},
        "figure_inputs": [{"id": figure["id"], "png": figure["png"], "svg": figure["svg"]} for figure in figures],
        "files": files,
        "completion_boundary": "PENDING_G3_8_DELIVERY_ACCEPTANCE",
    }
    manifest = body | {"artifact_hash": canonical_json_hash(body)}
    write_canonical_json(out / "stage3-formal-document-manifest.json", manifest)
    return manifest


def _git_commit(repository: Path) -> str:
    value = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if _GIT_RE.fullmatch(value) is None:
        raise ValueError("Git HEAD is not a full commit")
    return value


def _load_inputs(root: Path, refs: Mapping[str, str], expected_config_hash: str) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]]]:
    from param_importance_nlp.analysis import AnalysisReport, ChartArtifact
    from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact

    payloads: dict[str, Mapping[str, object]] = {}
    identities: dict[str, Mapping[str, object]] = {}
    for role in REQUIRED_KINDS:
        loaded = load_committed_task_artifact(root, refs[role], require_formal=True)
        identity = loaded.identity
        if identity.task_id != TASK_ID or identity.artifact_kind != role or identity.config_hash != expected_config_hash or identity.formal_eligible is not True or loaded.run_intent != "formal":
            raise ValueError(f"{role} formal S3.10 identity mismatch")
        payloads[role] = loaded.payload
        identities[role] = {
            "task_id": identity.task_id,
            "artifact_kind": identity.artifact_kind,
            "config_hash": identity.config_hash,
            "artifact_hash": identity.artifact_hash,
            "payload_hash": canonical_json_hash(loaded.payload),
        }
    report = AnalysisReport.from_mapping(payloads["analysis_report"])
    chart_payload = payloads["chart_artifacts"]
    handoff = payloads["handoff_manifest"]
    gates = payloads["gate_summary"]
    _validate_semantics(report.to_dict(), chart_payload, handoff, gates)
    if handoff.get("report_hash") != report.report_hash:
        raise ValueError("handoff report hash does not match the live-reloaded report")
    source_table_hash = handoff.get("source_table_hash")
    if (
        not isinstance(source_table_hash, str)
        or chart_payload.get("source_table_hash") != source_table_hash
        or source_table_hash
        not in {source.content_hash for source in report.source_artifacts}
    ):
        raise ValueError("S3.10 report/chart/handoff source table identity mismatch")
    if (
        gates.get("stage3.G3-7_hash") != handoff.get("stage3_g37_gate_hash")
        or gates.get("stage3.G3-7_ref") != handoff.get("stage3_g37_gate_ref")
    ):
        raise ValueError("S3.10 gate summary and handoff G3-7 identity mismatch")

    raw_chart_artifacts = chart_payload.get("artifacts")
    raw_figures = chart_payload.get("rendered_figures")
    if (
        not isinstance(raw_chart_artifacts, list)
        or not raw_chart_artifacts
        or not isinstance(raw_figures, list)
        or not raw_figures
    ):
        raise ValueError("S3.10 rendered chart inventory is incomplete")
    parsed_charts = [
        ChartArtifact.from_mapping(
            _mapping(value, field=f"chart_artifacts.artifacts[{index}]")
        )
        for index, value in enumerate(raw_chart_artifacts)
    ]
    chart_hashes: dict[tuple[str, str], str] = {}
    for chart in parsed_charts:
        key = (chart.spec.chart_id, chart.output_format)
        if (
            chart.output_format not in {"png", "svg"}
            or chart.content_sha256 is None
            or key in chart_hashes
        ):
            raise ValueError("S3.10 rendered ChartArtifact set is invalid")
        chart_hashes[key] = chart.content_sha256
    figure_hashes: dict[tuple[str, str], str] = {}
    for index, value in enumerate(raw_figures):
        figure = _mapping(value, field=f"chart_artifacts.rendered_figures[{index}]")
        figure_id = figure.get("id")
        if not isinstance(figure_id, str) or not figure_id:
            raise ValueError("S3.10 rendered figure id is invalid")
        for output_format in ("png", "svg"):
            record = _mapping(
                figure.get(output_format),
                field=f"chart_artifacts.rendered_figures[{index}].{output_format}",
            )
            digest = record.get("sha256")
            key = (figure_id, output_format)
            if (
                not isinstance(digest, str)
                or _HASH_RE.fullmatch(digest) is None
                or key in figure_hashes
            ):
                raise ValueError("S3.10 rendered figure hash inventory is invalid")
            figure_hashes[key] = digest
    if chart_hashes != figure_hashes:
        raise ValueError("S3.10 ChartArtifact hashes do not match PNG/SVG file records")
    return payloads, identities


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-config-hash", required=True)
    parser.add_argument("--font-file", type=Path, required=True)
    for role in REQUIRED_KINDS:
        parser.add_argument(f"--{role.replace('_', '-')}-ref", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if _HASH_RE.fullmatch(args.expected_config_hash) is None:
        raise ValueError("expected config hash is invalid")
    refs = {role: getattr(args, f"{role}_ref") for role in REQUIRED_KINDS}
    root = args.workspace_root.resolve(strict=True)
    payloads, identities = _load_inputs(root, refs, args.expected_config_hash)
    manifest = render_stage3_delivery_documents(
        workspace_root=root,
        output_dir=args.output_dir,
        analysis_report=payloads["analysis_report"],
        chart_artifacts=payloads["chart_artifacts"],
        handoff_manifest=payloads["handoff_manifest"],
        gate_summary=payloads["gate_summary"],
        source_refs=refs,
        input_identities=identities,
        producer_commit=_git_commit(args.repository_root.resolve(strict=True)),
        font_file=args.font_file,
    )
    print(json.dumps({"status": "PASS", "artifact_hash": manifest["artifact_hash"], "completion_boundary": manifest["completion_boundary"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
