"""Compose the path-only Stage 3 G3-8 delivery inventory from live artifacts.

The composer reads the four formal S3.10 commits plus the document renderer's
self-hashed manifest and assembles every path required by the strict delivery
manifest materializer.  It deliberately does not hash delivery files itself;
that remains the independent next boundary in
``materialize_stage3_delivery_manifest.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import (  # noqa: E402
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.experiments.stage3_g38_publisher import (  # noqa: E402
    REQUIRED_STAGE3_G38_GIT_ROLES,
    REQUIRED_STAGE3_G38_REPLAY_LAYERS,
    REQUIRED_STAGE3_G38_STAGE310_KINDS,
)
from param_importance_nlp.runtime import publish_canonical_immutable  # noqa: E402
from param_importance_nlp.runtime.task_artifacts import (  # noqa: E402
    load_committed_task_artifact,
)


INVENTORY_SCHEMA = "stage3-g38-delivery-inventory-source-v1"
DOCUMENT_SCHEMA = "stage3-delivery-documents-v1"
SOURCE_SNAPSHOT_SCHEMA = "stage3-g38-delivery-source-snapshot-v1"
STAGE310_TASK_ID = "stage3.10_reports_visualizations_and_handoff"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_FORBIDDEN_RE = re.compile(r"(?:fixture|synthetic|g3[-_.]?9|future)", re.IGNORECASE)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None or _FORBIDDEN_RE.search(value):
        raise ValueError(f"{field} is not a safe formal identifier")
    return value


def _safe_ref(value: object, *, field: str, suffixes: tuple[str, ...] | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "?" in value
        or "://" in value
        or "\\" in value
        or _FORBIDDEN_RE.search(value)
    ):
        raise ValueError(f"{field} is not a stable formal workspace ref")
    ref = PurePosixPath(value)
    if ref.is_absolute() or not ref.parts or any(part in {"", ".", ".."} for part in ref.parts):
        raise ValueError(f"{field} escapes the workspace")
    normalized = ref.as_posix()
    if suffixes is not None and not normalized.casefold().endswith(suffixes):
        raise ValueError(f"{field} must end with {suffixes}")
    return normalized


def _record_path(value: object, *, field: str, suffixes: tuple[str, ...] | None = None) -> str:
    record = _mapping(value, field=field)
    path = record.get("path")
    return _safe_ref(path, field=f"{field}.path", suffixes=suffixes)


def _binding_map(values: Sequence[str], *, expected: Sequence[str], field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, value in enumerate(values):
        if "=" not in value:
            raise ValueError(f"{field}[{index}] must be role=ref")
        role, ref = value.split("=", 1)
        if role in result:
            raise ValueError(f"{field} duplicates role {role}")
        result[role] = _safe_ref(ref, field=f"{field}.{role}", suffixes=(".json",))
    if set(result) != set(expected):
        raise ValueError(f"{field} must contain exactly {sorted(expected)}")
    return {role: result[role] for role in expected}


def _verify_document_manifest(
    document: Mapping[str, object],
    *,
    stage3_10_refs: Mapping[str, str],
    stage3_10_identities: Mapping[str, Mapping[str, object]],
    rendered_figures: object,
) -> None:
    if (
        document.get("schema_version") != DOCUMENT_SCHEMA
        or document.get("status") != "PASS"
        or document.get("scope") != "formal"
        or document.get("formal_eligible") is not True
        or document.get("completion_boundary") != "PENDING_G3_8_DELIVERY_ACCEPTANCE"
    ):
        raise ValueError("document manifest is not a formal pre-G3-8 PASS artifact")
    supplied = document.get("artifact_hash")
    if not isinstance(supplied, str) or _HASH_RE.fullmatch(supplied) is None:
        raise ValueError("document manifest artifact_hash is invalid")
    if canonical_json_hash({key: value for key, value in document.items() if key != "artifact_hash"}) != supplied:
        raise ValueError("document manifest artifact_hash does not match its content")
    refs = _mapping(document.get("source_refs"), field="document.source_refs")
    identities = _mapping(document.get("inputs"), field="document.inputs")
    if set(refs) != set(REQUIRED_STAGE3_G38_STAGE310_KINDS) or set(identities) != set(REQUIRED_STAGE3_G38_STAGE310_KINDS):
        raise ValueError("document manifest does not bind exactly four S3.10 roles")
    for role in REQUIRED_STAGE3_G38_STAGE310_KINDS:
        if refs[role] != stage3_10_refs[role]:
            raise ValueError(f"document manifest source ref drift: {role}")
        recorded = _mapping(identities[role], field=f"document.inputs.{role}")
        live = stage3_10_identities[role]
        for key in ("task_id", "artifact_kind", "config_hash", "artifact_hash", "payload_hash"):
            if recorded.get(key) != live.get(key):
                raise ValueError(f"document manifest input identity drift: {role}.{key}")

    document_figures = document.get("figure_inputs")
    if (
        not isinstance(document_figures, list)
        or not isinstance(rendered_figures, list)
        or not document_figures
        or len(document_figures) != len(rendered_figures)
    ):
        raise ValueError("document manifest figure inputs do not match S3.10")
    bound_figures: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(document_figures):
        figure = _mapping(value, field=f"document.figure_inputs[{index}]")
        figure_id = _safe_id(figure.get("id"), field=f"document.figure_inputs[{index}].id")
        if figure_id in bound_figures:
            raise ValueError(f"document manifest duplicates figure {figure_id}")
        bound_figures[figure_id] = figure
    live_figures: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(rendered_figures):
        figure = _mapping(value, field=f"chart_artifacts.rendered_figures[{index}]")
        figure_id = _safe_id(figure.get("id"), field=f"chart_artifacts.rendered_figures[{index}].id")
        if figure_id in live_figures:
            raise ValueError(f"S3.10 duplicates rendered figure {figure_id}")
        live_figures[figure_id] = figure
    if set(bound_figures) != set(live_figures):
        raise ValueError("document manifest figure ids drift from S3.10")
    for figure_id, recorded in bound_figures.items():
        live = live_figures[figure_id]
        for output_format in ("png", "svg"):
            recorded_file = _mapping(
                recorded.get(output_format),
                field=f"document.figure_inputs.{figure_id}.{output_format}",
            )
            live_file = _mapping(
                live.get(output_format),
                field=f"chart_artifacts.rendered_figures.{figure_id}.{output_format}",
            )
            for key in ("path", "size", "sha256"):
                if recorded_file.get(key) != live_file.get(key):
                    raise ValueError(
                        f"document manifest figure record drift: {figure_id}.{output_format}.{key}"
                    )


def _verify_delivery_source_manifest(
    source: Mapping[str, object],
    *,
    workspace_root: Path,
    source_ref: str,
    analysis_scripts: Sequence[str],
    worklog: str,
) -> None:
    expected = {
        "schema_version",
        "snapshot_id",
        "scope",
        "status",
        "formal_eligible",
        "producer_commit",
        "repository_branch",
        "snapshot_root",
        "analysis_scripts",
        "worklog",
        "artifact_hash",
    }
    if set(source) != expected:
        raise ValueError("delivery source manifest fields are invalid")
    if (
        source.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA
        or source.get("scope") != "formal"
        or source.get("status") != "PASS"
        or source.get("formal_eligible") is not True
        or not isinstance(source.get("producer_commit"), str)
        or _GIT_RE.fullmatch(str(source.get("producer_commit"))) is None
    ):
        raise ValueError("delivery source manifest is not a formal PASS snapshot")
    supplied = source.get("artifact_hash")
    if (
        not isinstance(supplied, str)
        or _HASH_RE.fullmatch(supplied) is None
        or canonical_json_hash(
            {key: value for key, value in source.items() if key != "artifact_hash"}
        )
        != supplied
    ):
        raise ValueError("delivery source manifest artifact_hash is invalid")
    snapshot_root = _safe_ref(source.get("snapshot_root"), field="source.snapshot_root")
    if PurePosixPath(snapshot_root).parts[0] not in {"evidence", "results"}:
        raise ValueError("delivery source snapshot must be in stable evidence/results")
    manifest_ref = _safe_ref(
        source_ref,
        field="delivery_source_manifest_ref",
        suffixes=(".json",),
    )
    if PurePosixPath(manifest_ref).parent.as_posix() != snapshot_root:
        raise ValueError("delivery source manifest ref does not match snapshot_root")
    stored_manifest = load_canonical_json(
        workspace_root.joinpath(*PurePosixPath(manifest_ref).parts)
    )
    if stored_manifest != source:
        raise ValueError("delivery source manifest mapping differs from its file")

    def record_ref(
        value: object,
        *,
        field: str,
        suffixes: tuple[str, ...],
    ) -> str:
        record = _mapping(value, field=field)
        if set(record) != {"source_ref", "snapshot_ref", "size", "sha256"}:
            raise ValueError(f"{field} fields are invalid")
        _safe_ref(record.get("source_ref"), field=f"{field}.source_ref", suffixes=suffixes)
        snapshot_ref = _safe_ref(
            record.get("snapshot_ref"),
            field=f"{field}.snapshot_ref",
            suffixes=suffixes,
        )
        try:
            PurePosixPath(snapshot_ref).relative_to(PurePosixPath(snapshot_root))
        except ValueError as error:
            raise ValueError(f"{field}.snapshot_ref escapes snapshot_root") from error
        size = record.get("size")
        digest = record.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or _HASH_RE.fullmatch(digest) is None
        ):
            raise ValueError(f"{field} byte identity is invalid")
        current = workspace_root
        for part in PurePosixPath(snapshot_ref).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"{field}.snapshot_ref contains a symlink")
        try:
            resolved = workspace_root.joinpath(
                *PurePosixPath(snapshot_ref).parts
            ).resolve(strict=True)
            resolved.relative_to(workspace_root)
        except (OSError, ValueError) as error:
            raise ValueError(f"{field}.snapshot_ref is not a workspace file") from error
        payload = resolved.read_bytes()
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"{field} snapshot byte identity drift")
        return snapshot_ref

    raw_scripts = source.get("analysis_scripts")
    if not isinstance(raw_scripts, list) or not raw_scripts:
        raise ValueError("delivery source manifest has no analysis scripts")
    recorded_scripts = [
        record_ref(value, field=f"source.analysis_scripts[{index}]", suffixes=(".py",))
        for index, value in enumerate(raw_scripts)
    ]
    if recorded_scripts != list(analysis_scripts):
        raise ValueError("delivery source manifest analysis script refs drift")
    recorded_worklog = record_ref(
        source.get("worklog"),
        field="source.worklog",
        suffixes=(".md",),
    )
    if recorded_worklog != worklog:
        raise ValueError("delivery source manifest worklog ref drift")


def compose_stage3_delivery_inventory(
    *,
    workspace_root: str | Path,
    manifest_id: str,
    output: str | Path,
    stage3_10_payloads: Mapping[str, Mapping[str, object]],
    stage3_10_refs: Mapping[str, str],
    stage3_10_identities: Mapping[str, Mapping[str, object]],
    document_manifest: Mapping[str, object],
    document_manifest_ref: str,
    delivery_source_manifest: Mapping[str, object],
    delivery_source_manifest_ref: str,
    analysis_scripts: Sequence[str],
    replay_reports: Mapping[str, str],
    server_large_artifact_manifest: str,
    git_sync: Mapping[str, str],
    worklog: str,
) -> Mapping[str, JSONValue]:
    root = Path(workspace_root).resolve(strict=True)
    manifest_name = _safe_id(manifest_id, field="manifest_id")
    roles = set(REQUIRED_STAGE3_G38_STAGE310_KINDS)
    if set(stage3_10_payloads) != roles or set(stage3_10_refs) != roles or set(stage3_10_identities) != roles:
        raise ValueError("exactly four S3.10 roles are required")
    normalized_refs = {
        role: _safe_ref(
            stage3_10_refs[role],
            field=f"stage3_10_refs.{role}",
            suffixes=(".json",),
        )
        for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
    }
    for role in REQUIRED_STAGE3_G38_STAGE310_KINDS:
        identity = stage3_10_identities[role]
        if identity.get("task_id") != STAGE310_TASK_ID or identity.get("artifact_kind") != role:
            raise ValueError(f"stage3_10_identities.{role} is not the expected S3.10 identity")
        for key in ("config_hash", "artifact_hash", "payload_hash"):
            value = identity.get(key)
            if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
                raise ValueError(f"stage3_10_identities.{role}.{key} is invalid")
    if len({identity["config_hash"] for identity in stage3_10_identities.values()}) != 1:
        raise ValueError("the four S3.10 identities do not share one config hash")
    rendered_figures = stage3_10_payloads["chart_artifacts"].get("rendered_figures")
    _verify_document_manifest(
        document_manifest,
        stage3_10_refs=normalized_refs,
        stage3_10_identities=stage3_10_identities,
        rendered_figures=rendered_figures,
    )
    document_ref = _safe_ref(document_manifest_ref, field="document_manifest_ref", suffixes=(".json",))
    script_refs = [
        _safe_ref(value, field=f"analysis_scripts[{index}]", suffixes=(".py",))
        for index, value in enumerate(analysis_scripts)
    ]
    if not script_refs or len(script_refs) != len(set(script_refs)):
        raise ValueError("analysis_scripts must be a non-empty unique list")
    if set(replay_reports) != set(REQUIRED_STAGE3_G38_REPLAY_LAYERS):
        raise ValueError("replay_reports must contain exactly three required layers")
    replay_refs = {
        role: _safe_ref(replay_reports[role], field=f"replay_reports.{role}", suffixes=(".json",))
        for role in REQUIRED_STAGE3_G38_REPLAY_LAYERS
    }
    if set(git_sync) != set(REQUIRED_STAGE3_G38_GIT_ROLES):
        raise ValueError("git_sync must contain exactly six required roles")
    git_refs = {
        role: _safe_ref(git_sync[role], field=f"git_sync.{role}", suffixes=(".json",))
        for role in REQUIRED_STAGE3_G38_GIT_ROLES
    }
    large_ref = _safe_ref(server_large_artifact_manifest, field="server_large_artifact_manifest", suffixes=(".json", ".jsonl", ".txt"))
    worklog_ref = _safe_ref(worklog, field="worklog", suffixes=(".md", ".txt", ".json"))
    source_manifest_ref = _safe_ref(
        delivery_source_manifest_ref,
        field="delivery_source_manifest_ref",
        suffixes=(".json",),
    )
    _verify_delivery_source_manifest(
        delivery_source_manifest,
        workspace_root=root,
        source_ref=source_manifest_ref,
        analysis_scripts=script_refs,
        worklog=worklog_ref,
    )

    charts = stage3_10_payloads["chart_artifacts"]
    handoff = stage3_10_payloads["handoff_manifest"]
    reporting = _mapping(handoff.get("reporting_files"), field="handoff.reporting_files")
    tables = reporting.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError("S3.10 reporting bundle has no source tables")
    csv_tables: list[str] = []
    json_tables: list[str] = []
    for index, value in enumerate(tables):
        table = _mapping(value, field=f"reporting.tables[{index}]")
        csv_tables.append(_record_path(table.get("csv"), field=f"reporting.tables[{index}].csv", suffixes=(".csv",)))
        json_tables.append(_record_path(table.get("json"), field=f"reporting.tables[{index}].json", suffixes=(".json",)))
    raw = _mapping(reporting.get("raw_formal_path_results"), field="reporting.raw_formal_path_results")
    raw_file = raw.get("file", raw)
    json_tables.append(_record_path(raw_file, field="reporting.raw_formal_path_results.file", suffixes=(".json",)))
    raw_shards = reporting.get("raw_shard_manifest")
    if isinstance(raw_shards, Mapping):
        json_tables.append(_record_path(raw_shards, field="reporting.raw_shard_manifest", suffixes=(".json",)))
    json_tables.append(document_ref)
    json_tables.append(source_manifest_ref)
    csv_tables = list(dict.fromkeys(csv_tables))
    json_tables = list(dict.fromkeys(json_tables))

    raw_figures = rendered_figures
    if not isinstance(raw_figures, list) or not raw_figures:
        raise ValueError("S3.10 chart artifacts have no rendered figures")
    figures: list[dict[str, JSONValue]] = []
    figure_ids: set[str] = set()
    for index, value in enumerate(raw_figures):
        figure = _mapping(value, field=f"rendered_figures[{index}]")
        figure_id = _safe_id(figure.get("id"), field=f"rendered_figures[{index}].id")
        if figure_id in figure_ids:
            raise ValueError(f"duplicate rendered figure: {figure_id}")
        figures.append({
            "id": figure_id,
            "png": _record_path(figure.get("png"), field=f"rendered_figures[{index}].png", suffixes=(".png",)),
            "svg": _record_path(figure.get("svg"), field=f"rendered_figures[{index}].svg", suffixes=(".svg",)),
        })
        figure_ids.add(figure_id)

    document_files = _mapping(document_manifest.get("files"), field="document.files")
    report = _mapping(document_files.get("chinese_report"), field="document.files.chinese_report")
    beamer = _mapping(document_files.get("beamer"), field="document.files.beamer")
    notes = beamer.get("notes")
    backups = beamer.get("backups")
    if not isinstance(notes, list) or not notes or not isinstance(backups, list) or not 3 <= len(backups) <= 5:
        raise ValueError("document manifest notes/backups are incomplete")
    body: dict[str, JSONValue] = {
        "schema_version": INVENTORY_SCHEMA,
        "manifest_id": manifest_name,
        "source_tables": {"csv": csv_tables, "json": json_tables},
        "analysis_scripts": script_refs,
        "figures": figures,
        "chinese_report": {
            "tex": _record_path(report.get("tex"), field="document.chinese_report.tex", suffixes=(".tex",)),
            "pdf": _record_path(report.get("pdf"), field="document.chinese_report.pdf", suffixes=(".pdf",)),
        },
        "beamer": {
            "tex": _record_path(beamer.get("tex"), field="document.beamer.tex", suffixes=(".tex",)),
            "pdf": _record_path(beamer.get("pdf"), field="document.beamer.pdf", suffixes=(".pdf",)),
            "notes": [
                _record_path(value, field=f"document.beamer.notes[{index}]", suffixes=(".md", ".txt"))
                for index, value in enumerate(notes)
            ],
            "backups": [
                _record_path(value, field=f"document.beamer.backups[{index}]", suffixes=(".tex",))
                for index, value in enumerate(backups)
            ],
        },
        "replay_reports": replay_refs,
        "server_large_artifact_manifest": large_ref,
        "git_sync": git_refs,
        "worklog": worklog_ref,
    }
    target = Path(output)
    if not target.is_absolute():
        target = root / target
    target = Path(os.path.abspath(target))
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("output must stay inside workspace_root") from error
    publish_canonical_immutable(target, body)
    return body


def _load_stage310(root: Path, refs: Mapping[str, str]) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]]]:
    payloads: dict[str, Mapping[str, object]] = {}
    identities: dict[str, Mapping[str, object]] = {}
    for role in REQUIRED_STAGE3_G38_STAGE310_KINDS:
        loaded = load_committed_task_artifact(root, refs[role], require_formal=True)
        identity = loaded.identity
        if identity.task_id != STAGE310_TASK_ID or identity.artifact_kind != role or identity.formal_eligible is not True or loaded.run_intent != "formal":
            raise ValueError(f"{role} is not the expected formal S3.10 commit")
        payloads[role] = loaded.payload
        identities[role] = {
            "task_id": identity.task_id,
            "artifact_kind": identity.artifact_kind,
            "config_hash": identity.config_hash,
            "artifact_hash": identity.artifact_hash,
            "payload_hash": canonical_json_hash(loaded.payload),
        }
    if len({identity["config_hash"] for identity in identities.values()}) != 1:
        raise ValueError("the four S3.10 commits do not share one config identity")
    return payloads, identities


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--document-manifest", required=True)
    parser.add_argument("--delivery-source-manifest", required=True)
    parser.add_argument("--analysis-script", action="append", default=[], required=True)
    for role in REQUIRED_STAGE3_G38_STAGE310_KINDS:
        parser.add_argument(f"--{role.replace('_', '-')}-ref", required=True)
    parser.add_argument("--replay-report", action="append", default=[], required=True, help="layer=workspace-ref")
    parser.add_argument("--git-evidence", action="append", default=[], required=True, help="role=workspace-ref")
    parser.add_argument("--server-large-artifact-manifest", required=True)
    parser.add_argument("--worklog", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.workspace_root.resolve(strict=True)
    refs = {
        role: _safe_ref(getattr(args, f"{role}_ref"), field=f"stage3_10_refs.{role}", suffixes=(".json",))
        for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
    }
    payloads, identities = _load_stage310(root, refs)
    document_ref = _safe_ref(args.document_manifest, field="document_manifest", suffixes=(".json",))
    document = _mapping(load_canonical_json(root.joinpath(*PurePosixPath(document_ref).parts)), field="document_manifest")
    source_manifest_ref = _safe_ref(
        args.delivery_source_manifest,
        field="delivery_source_manifest",
        suffixes=(".json",),
    )
    source_manifest = _mapping(
        load_canonical_json(root.joinpath(*PurePosixPath(source_manifest_ref).parts)),
        field="delivery_source_manifest",
    )
    inventory = compose_stage3_delivery_inventory(
        workspace_root=root,
        manifest_id=args.manifest_id,
        output=args.output,
        stage3_10_payloads=payloads,
        stage3_10_refs=refs,
        stage3_10_identities=identities,
        document_manifest=document,
        document_manifest_ref=document_ref,
        delivery_source_manifest=source_manifest,
        delivery_source_manifest_ref=source_manifest_ref,
        analysis_scripts=args.analysis_script,
        replay_reports=_binding_map(args.replay_report, expected=REQUIRED_STAGE3_G38_REPLAY_LAYERS, field="replay_report"),
        server_large_artifact_manifest=args.server_large_artifact_manifest,
        git_sync=_binding_map(args.git_evidence, expected=REQUIRED_STAGE3_G38_GIT_ROLES, field="git_evidence"),
        worklog=args.worklog,
    )
    target = args.output
    if target.is_absolute():
        output_ref = target.resolve().relative_to(root).as_posix()
    else:
        output_ref = PurePosixPath(target.as_posix()).as_posix()
    print(json.dumps({"status": "PASS", "output_ref": output_ref, "manifest_id": inventory["manifest_id"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
