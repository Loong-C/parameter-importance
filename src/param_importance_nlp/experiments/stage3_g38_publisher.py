"""Independent Stage 3 G3-8 delivery acceptance publisher.

G3-8 is deliberately a *consumer* of the Stage 3.10 delivery.  It does not
generate a report, repair a manifest, or infer a gate result.  Before it
publishes anything it reloads every input through the formal task-commit
loader and verifies the small-file inventory byte-for-byte.  A receipt and a
G3-8 ``GateRecord`` are then published as two separate immutable task
artifacts.

The public entry point is :func:`publish_stage3_g38` (or
``Stage3G38Publisher().publish``).  The publisher accepts a mapping of the
eight prerequisite gate commit refs and a mapping/sequence of exactly four
Stage3.10 commit refs.  A few named aliases are accepted for integrations
that use ``report_ref``/``visualization_ref`` style names, but all aliases are
normalised before the deterministic publication-config hash is calculated.

This module intentionally has no CLI or catalog registration.  Keeping the
acceptance boundary independent prevents a producer, a report, or the
G3-8 receipt from being used as its own evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from ..contracts.errors import FormalRunRejected
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.status import GateRecord, GateStatus
from ..runtime.task_artifacts import LoadedTaskArtifact, TaskArtifactStore, load_committed_task_artifact


STAGE3_G38_PUBLICATION_SCHEMA = "stage3-g38-publication-v1"
STAGE3_G38_DELIVERY_MANIFEST_SCHEMA = "stage3-g38-delivery-manifest-v1"
STAGE3_G38_PUBLICATION_CONFIG_SCHEMA = "stage3-g38-publication-config-v1"
STAGE3_G38_TASK_ID = "stage3.10_g3_8_delivery_acceptance"
STAGE3_G38_GATE_ID = "stage3.G3-8"
STAGE3_G38_GATE_ARTIFACT_KIND = "gate_record"
# Match the Stage3.08 publisher's ``g36_publication`` naming while keeping a
# descriptive alias below for callers that refer to this output as a receipt.
STAGE3_G38_RECEIPT_ARTIFACT_KIND = "g38_publication"
STAGE3_G38_DELIVERY_RECEIPT_ARTIFACT_KIND = STAGE3_G38_RECEIPT_ARTIFACT_KIND

REQUIRED_STAGE3_G38_GATE_IDS: tuple[str, ...] = tuple(
    f"stage3.G3-{index}" for index in range(8)
)
REQUIRED_STAGE3_G38_DELIVERY_ROLES: tuple[str, ...] = (
    "source_tables",
    "analysis_scripts",
    "figures",
    "chinese_report",
    "beamer",
    "replay_reports",
    "server_large_artifact_manifest",
    "git_sync",
    "worklog",
)
REQUIRED_STAGE3_G38_GIT_ROLES: tuple[str, ...] = (
    "branch",
    "commit",
    "push",
    "remote",
    "server_clean_head",
    "sync",
)
STAGE3_G38_GIT_SYNC_EVIDENCE_SCHEMA = "stage3-g38-git-sync-evidence-v1"
REQUIRED_STAGE3_G38_AGENT_DOCUMENTS: tuple[str, ...] = (
    "Agent/git.md",
    "Agent/local.md",
    "Agent/remote_access.md",
    "Agent/server.md",
    "Agent/sync.md",
    "Agent/worklogs.md",
)
STAGE3_G38_LARGE_ARTIFACT_MANIFEST_SCHEMA = "stage3-g38-large-artifact-manifest-v1"
REQUIRED_STAGE3_G38_LARGE_ARTIFACT_ROLES: tuple[str, ...] = (
    "stage3_formal_endpoints",
    "stage3_formal_probes",
    "stage3_formal_configs",
    "s307_formal_cache",
    "s307_formal_output",
    "stage3_formal_results",
    "stage3_formal_evidence",
)
REQUIRED_STAGE3_G38_REPLAY_LAYERS: tuple[str, ...] = (
    "local_cpu",
    "server_locked",
    "frozen_endpoint_uncached",
)
STAGE3_G38_REPLAY_REPORT_SCHEMA = "stage3-g38-replay-report-v1"
_REPLAY_CACHE_MODES = {
    "local_cpu": "not_applicable",
    "server_locked": "locked_environment",
    "frozen_endpoint_uncached": "uncached",
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_FORBIDDEN_RE = re.compile(r"(?:fixture|synthetic)", re.IGNORECASE)
_FUTURE_RE = re.compile(
    r"(?:g3[-_.]?9|stage3[./_-]?g3[-_.]?9|future)",
    re.IGNORECASE,
)

REQUIRED_STAGE3_G38_STAGE310_KINDS: tuple[str, ...] = (
    "analysis_report",
    "chart_artifacts",
    "handoff_manifest",
    "gate_summary",
)
STAGE3_G38_STAGE310_TASK_ID = "stage3.10_reports_visualizations_and_handoff"
STAGE3_G38_MANIFEST_TASK_ID = "stage3.10_delivery_manifest_authority"
STAGE3_G38_MANIFEST_ARTIFACT_KIND = "delivery_manifest"


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} 必须是小写 SHA-256")
    return value


def _safe_ref(value: object, *, field: str, reject_future: bool = True) -> str:
    if not isinstance(value, str) or not value or "?" in value or "://" in value:
        raise ValueError(f"{field} 必须是稳定相对引用")
    if "\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"{field} 必须使用 workspace 内 POSIX 相对引用")
    lowered = value.casefold()
    if _FORBIDDEN_RE.search(value):
        raise FormalRunRejected(f"{field} 禁止 fixture/synthetic 引用")
    if reject_future and _FUTURE_RE.search(value):
        raise FormalRunRejected(f"{field} 禁止 self/future 引用")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} 路径逃逸 workspace")
    return path.as_posix()


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field} 不是安全标识")
    if _FORBIDDEN_RE.search(value) or _FUTURE_RE.search(value):
        raise FormalRunRejected(f"{field} 禁止 fixture/synthetic/self/future 标识")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} 必须是 object")
    return value


def _normalise_hash_object(value: Mapping[str, object], *, field: str) -> str:
    supplied = value.get("artifact_hash")
    _hash(supplied, field=f"{field}.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(body) != supplied:
        raise FormalRunRejected(f"{field}.artifact_hash 与内容不一致")
    return str(supplied)


def _load_formal_commit(
    workspace_root: Path,
    reference: object,
    *,
    field: str,
    config_hash: str | None,
    expected_kind: str | None = None,
    expected_task_id: str | None = None,
) -> LoadedTaskArtifact:
    ref = _safe_ref(reference, field=field)
    try:
        loaded = load_committed_task_artifact(workspace_root, ref, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise FormalRunRejected(f"STAGE3_G38_{field.upper()}_FORMAL_COMMIT_INVALID") from error
    identity = loaded.identity
    if config_hash is not None and identity.config_hash != config_hash:
        raise FormalRunRejected(f"STAGE3_G38_{field.upper()}_CONFIG_HASH_MISMATCH")
    if identity.formal_eligible is not True or loaded.run_intent != "formal":
        raise FormalRunRejected(f"STAGE3_G38_{field.upper()}_FORMAL_REQUIRED")
    if expected_kind is not None and identity.artifact_kind != expected_kind:
        raise FormalRunRejected(
            f"STAGE3_G38_{field.upper()}_ARTIFACT_KIND_INVALID:{identity.artifact_kind}"
        )
    if expected_task_id is not None and identity.task_id != expected_task_id:
        raise FormalRunRejected(
            f"STAGE3_G38_{field.upper()}_TASK_ID_INVALID:{identity.task_id}"
        )
    # ``task_id`` may legitimately contain ``stage3.10`` (the current
    # producer stage), so use the identifier validator for envelope identity;
    # only actual source references are subject to the no-future rule.
    _safe_id(identity.task_id, field=f"{field}.task_id")
    if not isinstance(identity.artifact_kind, str) or not identity.artifact_kind:
        raise FormalRunRejected(f"{field}.artifact_kind invalid")
    for item in loaded.source_refs:
        _safe_ref(item, field=f"{field}.source_ref", reject_future=True)
    return loaded


def _file_mapping(value: object, *, field: str) -> Mapping[str, object]:
    raw = _mapping(value, field=field)
    path = raw.get("path", raw.get("ref", raw.get("file", raw.get("workspace_path"))))
    digest = raw.get("sha256", raw.get("file_sha256", raw.get("hash")))
    size = raw.get("size", raw.get("bytes", raw.get("size_bytes")))
    allowed = {"path", "ref", "file", "workspace_path", "sha256", "file_sha256", "hash", "size", "bytes", "size_bytes", "role", "source_refs"}
    if set(raw) - allowed:
        unknown = sorted(set(raw) - allowed)
        raise ValueError(f"{field} 未知字段:{unknown}")
    path_text = _safe_ref(path, field=f"{field}.path", reject_future=False)
    _hash(digest, field=f"{field}.sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{field}.size 必须是非负整数")
    result: dict[str, object] = {"path": path_text, "sha256": str(digest), "size": size}
    if "role" in raw:
        if not isinstance(raw["role"], str) or not raw["role"]:
            raise ValueError(f"{field}.role 无效")
        result["role"] = raw["role"]
    if "source_refs" in raw:
        refs = raw["source_refs"]
        if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
            raise ValueError(f"{field}.source_refs 必须是字符串数组")
        result["source_refs"] = [
            _safe_ref(item, field=f"{field}.source_refs", reject_future=True) for item in refs
        ]
    return result


def _require_suffix(record: Mapping[str, object], suffixes: tuple[str, ...], *, field: str) -> None:
    path = str(record["path"]).casefold()
    if not path.endswith(suffixes):
        raise ValueError(f"{field} 文件扩展名必须是 {suffixes}")


def _records(value: object, *, field: str) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        # A single file record has a path/ref key; otherwise mappings are a
        # convenient machine-readable name -> record inventory.
        if any(key in value for key in ("path", "ref", "file")):
            return [_file_mapping(value, field=field)]
        values = list(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(f"{field} 必须是 file record 数组或 object")
    if not values:
        raise ValueError(f"{field} 不能为空")
    return [_file_mapping(item, field=f"{field}[{index}]") for index, item in enumerate(values)]


def _record_tuple(value: object, *, field: str) -> tuple[dict[str, object], ...]:
    return tuple(dict(item) for item in _records(value, field=field))


def _record_wire(value: Mapping[str, object]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {
        "path": str(value["path"]),
        "sha256": str(value["sha256"]),
        "size": int(value["size"]),
    }
    if "role" in value:
        result["role"] = str(value["role"])
    if "source_refs" in value:
        result["source_refs"] = list(value["source_refs"])  # type: ignore[arg-type]
    return result


@dataclass(frozen=True, slots=True)
class Stage3G38DeliveryManifest:
    """Canonical, machine-readable Stage 3.10 small-file inventory."""

    manifest_id: str
    publication_config_hash: str | None
    csv_tables: tuple[Mapping[str, object], ...]
    json_tables: tuple[Mapping[str, object], ...]
    analysis_scripts: tuple[Mapping[str, object], ...]
    figures: tuple[Mapping[str, object], ...]
    chinese_report_tex: Mapping[str, object]
    chinese_report_pdf: Mapping[str, object]
    beamer_tex: Mapping[str, object]
    beamer_pdf: Mapping[str, object]
    beamer_notes: tuple[Mapping[str, object], ...]
    beamer_backups: tuple[Mapping[str, object], ...]
    replay_reports: Mapping[str, Mapping[str, object]]
    server_large_artifact_manifest: Mapping[str, object]
    git_sync: Mapping[str, Mapping[str, object]]
    worklog: Mapping[str, object]
    status: str = "PASS"
    formal_eligible: bool = True
    schema_version: str = STAGE3_G38_DELIVERY_MANIFEST_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage3G38DeliveryManifest":
        raw = _mapping(value, field="delivery_manifest")
        if raw.get("schema_version") != STAGE3_G38_DELIVERY_MANIFEST_SCHEMA:
            raise ValueError("STAGE3_G38_DELIVERY_MANIFEST_SCHEMA_UNSUPPORTED")
        if raw.get("scope") != "formal":
            raise FormalRunRejected("STAGE3_G38_DELIVERY_MANIFEST_FORMAL_SCOPE_REQUIRED")
        manifest_id = _safe_id(raw.get("manifest_id"), field="manifest_id")
        status = raw.get("status", "PASS")
        eligible = raw.get("formal_eligible", True)
        if status != "PASS" or eligible is not True:
            raise FormalRunRejected("STAGE3_G38_DELIVERY_MANIFEST_NOT_FORMAL_PASS")
        supplied = raw.get("artifact_hash")
        _hash(supplied, field="delivery_manifest.artifact_hash")
        if canonical_json_hash({key: item for key, item in raw.items() if key != "artifact_hash"}) != supplied:
            raise FormalRunRejected("STAGE3_G38_DELIVERY_MANIFEST_HASH_MISMATCH")

        # Producers may put the role inventory below ``artifacts``.  The
        # canonical wire emitted by this class remains flat, but accepting
        # this one envelope keeps the consumer independent from a cosmetic
        # Stage3.10 report-layout choice.
        artifact_inventory = raw.get("artifacts")
        inventory = artifact_inventory if isinstance(artifact_inventory, Mapping) else {}
        def pick(*names: str) -> object:
            for name in names:
                if name in raw:
                    return raw[name]
                if name in inventory:
                    return inventory[name]
            return None

        source_tables = pick("source_tables", "tables")
        if isinstance(source_tables, Mapping):
            csv_raw = source_tables.get("csv", source_tables.get("csv_tables"))
            json_raw = source_tables.get("json", source_tables.get("json_tables"))
            if csv_raw is None or json_raw is None:
                raise ValueError("delivery_manifest.source_tables 必须含 csv/json")
            csv_tables, json_tables = _records(csv_raw, field="source_tables.csv"), _records(json_raw, field="source_tables.json")
        else:
            all_tables = _records(source_tables if source_tables is not None else pick("csv_tables", "source_csv"), field="source_tables")
            extra_json = pick("json_tables", "source_json")
            if extra_json is not None:
                csv_tables, json_tables = all_tables, _records(extra_json, field="json_tables")
            else:
                csv_tables = [item for item in all_tables if str(item["path"]).casefold().endswith(".csv")]
                json_tables = [item for item in all_tables if str(item["path"]).casefold().endswith(".json")]
        if not csv_tables or not json_tables:
            raise ValueError("delivery_manifest 必须同时列出 CSV 与 JSON 源表")
        for index, item in enumerate(csv_tables):
            _require_suffix(item, (".csv",), field=f"source_tables.csv[{index}]")
        for index, item in enumerate(json_tables):
            _require_suffix(item, (".json",), field=f"source_tables.json[{index}]")

        figures_raw = pick("figures", "visualizations")
        if isinstance(figures_raw, Mapping):
            figures_raw = [dict(item, id=key) if isinstance(item, Mapping) else item for key, item in figures_raw.items()]
        if not isinstance(figures_raw, Sequence) or isinstance(figures_raw, (str, bytes)) or not figures_raw:
            raise ValueError("delivery_manifest.figures 必须是非空数组")
        figures: list[dict[str, object]] = []
        for index, item in enumerate(figures_raw):
            figure = _mapping(item, field=f"figures[{index}]")
            png = _file_mapping(figure.get("png"), field=f"figures[{index}].png")
            svg = _file_mapping(figure.get("svg"), field=f"figures[{index}].svg")
            _require_suffix(png, (".png",), field=f"figures[{index}].png")
            _require_suffix(svg, (".svg",), field=f"figures[{index}].svg")
            figure_id = figure.get("id", f"figure-{index + 1}")
            if not isinstance(figure_id, str) or not figure_id:
                raise ValueError(f"figures[{index}].id 无效")
            out: dict[str, object] = {"id": figure_id, "png": png, "svg": svg}
            if "source_table_refs" in figure:
                refs = figure["source_table_refs"]
                if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                    raise ValueError(f"figures[{index}].source_table_refs 无效")
                out["source_table_refs"] = [_safe_ref(ref, field="figure.source_table_refs", reject_future=True) for ref in refs]
            figures.append(out)

        report_raw = _mapping(pick("chinese_report", "report"), field="chinese_report")
        beamer_raw = _mapping(pick("beamer", "slides"), field="beamer")
        report_tex = _file_mapping(report_raw.get("tex", report_raw.get("tex_file", pick("chinese_report_tex", "report_tex"))), field="chinese_report.tex")
        report_pdf = _file_mapping(report_raw.get("pdf", report_raw.get("pdf_file", pick("chinese_report_pdf", "report_pdf"))), field="chinese_report.pdf")
        beamer_tex = _file_mapping(beamer_raw.get("tex", beamer_raw.get("tex_file")), field="beamer.tex")
        beamer_pdf = _file_mapping(beamer_raw.get("pdf", beamer_raw.get("pdf_file")), field="beamer.pdf")
        _require_suffix(report_tex, (".tex",), field="chinese_report.tex")
        _require_suffix(report_pdf, (".pdf",), field="chinese_report.pdf")
        _require_suffix(beamer_tex, (".tex",), field="beamer.tex")
        _require_suffix(beamer_pdf, (".pdf",), field="beamer.pdf")
        notes = _record_tuple(beamer_raw.get("notes", beamer_raw.get("notes_file", pick("beamer_notes"))), field="beamer.notes")
        backups = _record_tuple(beamer_raw.get("backups", beamer_raw.get("backup_slides", beamer_raw.get("backup"))), field="beamer.backups")
        if not 3 <= len(backups) <= 5:
            raise ValueError("beamer.backups 必须是 3–5 个机器字段")

        replay_raw = _mapping(pick("replay_reports", "replays", "replay"), field="replay_reports")
        replay: dict[str, Mapping[str, object]] = {}
        for layer in REQUIRED_STAGE3_G38_REPLAY_LAYERS:
            aliases = {
                "local_cpu": ("local_cpu", "cpu", "local"),
                "server_locked": ("server_locked", "server", "locked_server"),
                "frozen_endpoint_uncached": ("frozen_endpoint_uncached", "endpoint", "uncached_endpoint", "frozen_endpoint"),
            }[layer]
            item = next((replay_raw.get(alias) for alias in aliases if alias in replay_raw), None)
            if item is None:
                raise ValueError(f"replay_reports 缺少 {layer}")
            if isinstance(item, Mapping) and not any(key in item for key in ("path", "ref")) and "file" in item:
                item = item["file"]
            replay[layer] = _file_mapping(item, field=f"replay_reports.{layer}")
            _require_suffix(replay[layer], (".json",), field=f"replay_reports.{layer}")

        large = _file_mapping(pick("server_large_artifact_manifest", "large_artifact_manifest", "server_manifest"), field="server_large_artifact_manifest")
        git_raw = _mapping(pick("git_sync", "git_evidence", "sync_evidence", "git"), field="git_sync")
        git: dict[str, Mapping[str, object]] = {}
        for role in REQUIRED_STAGE3_G38_GIT_ROLES:
            item = next((git_raw.get(alias) for alias in (role, role.replace("_", "-"), role.replace("_", "")) if alias in git_raw), None)
            if item is None:
                raise ValueError(f"git_sync 缺少 {role}")
            git[role] = _file_mapping(item, field=f"git_sync.{role}")
        worklog = _file_mapping(pick("worklog", "worklog_ref"), field="worklog")
        _require_suffix(large, (".json", ".jsonl", ".txt"), field="server_large_artifact_manifest")
        _require_suffix(worklog, (".md", ".txt", ".json"), field="worklog")
        config_hash = raw.get("publication_config_hash")
        if config_hash is not None:
            _hash(config_hash, field="delivery_manifest.publication_config_hash")
        return cls(
            manifest_id=manifest_id,
            publication_config_hash=(None if config_hash is None else str(config_hash)),
            csv_tables=tuple(csv_tables),
            json_tables=tuple(json_tables),
            analysis_scripts=_record_tuple(pick("analysis_scripts", "analysis_script"), field="analysis_scripts"),
            figures=tuple(figures),
            chinese_report_tex=report_tex,
            chinese_report_pdf=report_pdf,
            beamer_tex=beamer_tex,
            beamer_pdf=beamer_pdf,
            beamer_notes=notes,
            beamer_backups=backups,
            replay_reports=replay,
            server_large_artifact_manifest=large,
            git_sync=git,
            worklog=worklog,
            status=str(status),
            formal_eligible=bool(eligible),
        )

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "scope": "formal",
            "status": self.status,
            "formal_eligible": self.formal_eligible,
            "publication_config_hash": self.publication_config_hash,
            "source_tables": {"csv": [_record_wire(item) for item in self.csv_tables], "json": [_record_wire(item) for item in self.json_tables]},
            "analysis_scripts": [_record_wire(item) for item in self.analysis_scripts],
            "figures": [{key: (_record_wire(value) if key in {"png", "svg"} else value) for key, value in item.items()} for item in self.figures],
            "chinese_report": {"tex": _record_wire(self.chinese_report_tex), "pdf": _record_wire(self.chinese_report_pdf)},
            "beamer": {"tex": _record_wire(self.beamer_tex), "pdf": _record_wire(self.beamer_pdf), "notes": [_record_wire(item) for item in self.beamer_notes], "backups": [_record_wire(item) for item in self.beamer_backups]},
            "replay_reports": {key: _record_wire(value) for key, value in self.replay_reports.items()},
            "server_large_artifact_manifest": _record_wire(self.server_large_artifact_manifest),
            "git_sync": {key: _record_wire(value) for key, value in self.git_sync.items()},
            "worklog": _record_wire(self.worklog),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def file_records(self) -> tuple[Mapping[str, object], ...]:
        values: list[Mapping[str, object]] = [*self.csv_tables, *self.json_tables, *self.analysis_scripts]
        for figure in self.figures:
            values.extend((figure["png"], figure["svg"]))  # type: ignore[arg-type]
        values.extend((self.chinese_report_tex, self.chinese_report_pdf, self.beamer_tex, self.beamer_pdf))
        values.extend(self.beamer_notes)
        values.extend(self.beamer_backups)
        values.extend(self.replay_reports.values())
        values.append(self.server_large_artifact_manifest)
        values.extend(self.git_sync.values())
        values.append(self.worklog)
        return tuple(values)


def _verify_manifest_files(workspace_root: Path, manifest: Stage3G38DeliveryManifest) -> None:
    root = workspace_root.resolve()
    seen: set[str] = set()
    for index, record in enumerate(manifest.file_records()):
        ref = _safe_ref(record.get("path"), field=f"delivery_manifest.files[{index}].path", reject_future=False)
        if ref in seen:
            raise FormalRunRejected(f"STAGE3_G38_DUPLICATE_DELIVERY_FILE:{ref}")
        seen.add(ref)
        candidate = root.joinpath(*PurePosixPath(ref).parts)
        try:
            current = root
            for part in PurePosixPath(ref).parts:
                current = current / part
                if current.is_symlink():
                    raise FormalRunRejected(f"STAGE3_G38_DELIVERY_FILE_SYMLINK:{ref}")
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except FormalRunRejected:
            raise
        except (OSError, ValueError) as error:
            raise FormalRunRejected(f"STAGE3_G38_DELIVERY_FILE_NOT_IN_WORKSPACE:{ref}") from error
        if not resolved.is_file():
            raise FormalRunRejected(f"STAGE3_G38_DELIVERY_FILE_NOT_REGULAR:{ref}")
        expected_size = record.get("size")
        expected_hash = record.get("sha256")
        actual_size = resolved.stat().st_size
        if actual_size != expected_size:
            raise FormalRunRejected(f"STAGE3_G38_DELIVERY_FILE_SIZE_MISMATCH:{ref}")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise FormalRunRejected(f"STAGE3_G38_DELIVERY_FILE_SHA256_MISMATCH:{ref}")


def _verify_reporting_bundle(
    workspace_root: Path,
    report_payload: Mapping[str, object],
    charts_payload: Mapping[str, object],
) -> None:
    """Reload and validate S3.10's semantic report/figure bundle.

    G3-8 remains a consumer: it does not generate figures or repair reports.
    It does, however, re-run the strict hash-bound parsers and verify that each
    declared chart has both rendered formats bound to the corresponding files.
    """

    from ..analysis import AnalysisReport, ChartArtifact

    try:
        report = AnalysisReport.from_mapping(report_payload)
    except (TypeError, ValueError) as error:
        raise FormalRunRejected("STAGE3_G38_ANALYSIS_REPORT_INVALID") from error
    if report.metadata.get("scope") != "formal" or report.metadata.get("formal_eligible") is not True:
        raise FormalRunRejected("STAGE3_G38_ANALYSIS_REPORT_NOT_FORMAL")
    source_hashes = {source.content_hash for source in report.source_artifacts}
    source_table_hash = charts_payload.get("source_table_hash")
    if not isinstance(source_table_hash, str) or source_table_hash not in source_hashes:
        raise FormalRunRejected("STAGE3_G38_CHART_SOURCE_NOT_IN_REPORT")
    raw_artifacts = charts_payload.get("artifacts")
    figures = charts_payload.get("rendered_figures")
    if (
        not isinstance(raw_artifacts, list)
        or not raw_artifacts
        or not isinstance(figures, list)
        or not figures
    ):
        raise FormalRunRejected("STAGE3_G38_CHARTS_REQUIRE_PNG_AND_SVG")
    parsed: dict[tuple[str, str], ChartArtifact] = {}
    for value in raw_artifacts:
        if not isinstance(value, Mapping):
            raise FormalRunRejected("STAGE3_G38_CHART_ARTIFACT_INVALID")
        try:
            artifact = ChartArtifact.from_mapping(value)
        except (TypeError, ValueError) as error:
            raise FormalRunRejected("STAGE3_G38_CHART_ARTIFACT_INVALID") from error
        if artifact.output_format not in {"png", "svg"}:
            raise FormalRunRejected("STAGE3_G38_RENDERED_CHART_REQUIRED")
        if artifact.spec.source_hash not in source_hashes:
            raise FormalRunRejected("STAGE3_G38_CHART_SOURCE_HASH_DRIFT")
        key = (artifact.spec.chart_id, artifact.output_format)
        if key in parsed:
            raise FormalRunRejected("STAGE3_G38_DUPLICATE_RENDERED_CHART")
        parsed[key] = artifact

    figure_ids: set[str] = set()
    for index, value in enumerate(figures):
        if not isinstance(value, Mapping):
            raise FormalRunRejected(f"STAGE3_G38_RENDERED_FIGURE_INVALID:{index}")
        chart_id = value.get("id")
        source_hash = value.get("source_hash")
        if not isinstance(chart_id, str) or not chart_id or source_hash not in source_hashes:
            raise FormalRunRejected(f"STAGE3_G38_RENDERED_FIGURE_IDENTITY_INVALID:{index}")
        if chart_id in figure_ids:
            raise FormalRunRejected(f"STAGE3_G38_DUPLICATE_RENDERED_FIGURE:{chart_id}")
        figure_ids.add(chart_id)
        for output_format in ("png", "svg"):
            record = value.get(output_format)
            if not isinstance(record, Mapping):
                raise FormalRunRejected(f"STAGE3_G38_RENDERED_FIGURE_FORMAT_MISSING:{chart_id}:{output_format}")
            artifact = parsed.get((chart_id, output_format))
            if artifact is None or artifact.content_sha256 != record.get("sha256"):
                raise FormalRunRejected(f"STAGE3_G38_RENDERED_FIGURE_ARTIFACT_MISMATCH:{chart_id}:{output_format}")
            _verify_file_record(workspace_root, record, field=f"charts.{chart_id}.{output_format}")
    chart_ids = {chart_id for chart_id, _ in parsed}
    if figure_ids != chart_ids or any(
        (chart_id, "png") not in parsed or (chart_id, "svg") not in parsed
        for chart_id in chart_ids
    ):
        raise FormalRunRejected("STAGE3_G38_CHART_FORMAT_PAIR_INCOMPLETE")


def _verify_file_record(
    workspace_root: Path,
    record: Mapping[str, object],
    *,
    field: str,
) -> None:
    """Verify one small-file record using the same workspace boundary as G3-8."""

    ref = _safe_ref(record.get("path"), field=f"{field}.path", reject_future=False)
    size = record.get("size")
    digest = record.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0 or not _HASH_RE.fullmatch(str(digest)):
        raise FormalRunRejected(f"STAGE3_G38_FILE_RECORD_INVALID:{field}")
    root = workspace_root.resolve()
    candidate = root.joinpath(*PurePosixPath(ref).parts)
    try:
        current = root
        for part in PurePosixPath(ref).parts:
            current = current / part
            if current.is_symlink():
                raise FormalRunRejected(f"STAGE3_G38_FILE_SYMLINK:{ref}")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except FormalRunRejected:
        raise
    except (OSError, ValueError) as error:
        raise FormalRunRejected(f"STAGE3_G38_FILE_NOT_IN_WORKSPACE:{ref}") from error
    if not resolved.is_file():
        raise FormalRunRejected(f"STAGE3_G38_FILE_NOT_REGULAR:{ref}")
    if resolved.stat().st_size != size:
        raise FormalRunRejected(f"STAGE3_G38_FILE_HASH_OR_SIZE_MISMATCH:{ref}")
    hasher = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise FormalRunRejected(f"STAGE3_G38_FILE_HASH_OR_SIZE_MISMATCH:{ref}")


def _verify_replay_input_file(
    workspace_root: Path,
    *,
    ref: str,
    digest: str,
    identity: str,
) -> None:
    """Bind one replay-declared input digest to its actual workspace bytes."""

    root = workspace_root.resolve()
    candidate = root.joinpath(*PurePosixPath(ref).parts)
    try:
        current = root
        for part in PurePosixPath(ref).parts:
            current = current / part
            if current.is_symlink():
                raise FormalRunRejected(
                    f"STAGE3_G38_REPLAY_INPUT_FILE_SYMLINK:{identity}"
                )
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except FormalRunRejected:
        raise
    except (OSError, ValueError) as error:
        raise FormalRunRejected(
            f"STAGE3_G38_REPLAY_INPUT_FILE_NOT_IN_WORKSPACE:{identity}"
        ) from error
    if not resolved.is_file():
        raise FormalRunRejected(
            f"STAGE3_G38_REPLAY_INPUT_FILE_NOT_REGULAR:{identity}"
        )
    hasher = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise FormalRunRejected(
            f"STAGE3_G38_REPLAY_INPUT_FILE_HASH_MISMATCH:{identity}"
        )


def _utc_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise FormalRunRejected(f"STAGE3_G38_REPLAY_TIMESTAMP_INVALID:{field}")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise FormalRunRejected(f"STAGE3_G38_REPLAY_TIMESTAMP_INVALID:{field}") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FormalRunRejected(f"STAGE3_G38_REPLAY_TIMESTAMP_NOT_UTC:{field}")
    return parsed


def validate_stage3_replay_reports(
    workspace_root: str | Path,
    manifest: Stage3G38DeliveryManifest,
) -> None:
    """Verify that all three hash-bound replay files prove a real PASS run."""

    root = Path(workspace_root).resolve()
    required = {
        "schema_version",
        "replay_id",
        "layer",
        "scope",
        "status",
        "formal_eligible",
        "implementation_commit",
        "environment_hash",
        "command",
        "returncode",
        "started_at",
        "completed_at",
        "cache_mode",
        "test_summary",
        "input_refs",
        "input_hashes",
        "evidence_files",
        "artifact_hash",
    }
    for layer in REQUIRED_STAGE3_G38_REPLAY_LAYERS:
        record = manifest.replay_reports[layer]
        _verify_file_record(root, record, field=f"replay_reports.{layer}")
        ref = _safe_ref(
            record.get("path"),
            field=f"replay_reports.{layer}.path",
            reject_future=False,
        )
        try:
            payload = _mapping(
                load_canonical_json(root.joinpath(*PurePosixPath(ref).parts)),
                field=f"replay_reports.{layer}",
            )
        except (OSError, TypeError, ValueError) as error:
            raise FormalRunRejected(
                f"STAGE3_G38_REPLAY_REPORT_JSON_INVALID:{layer}"
            ) from error
        if set(payload) != required:
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_REPORT_FIELDS_INVALID:{layer}")
        if (
            payload.get("schema_version") != STAGE3_G38_REPLAY_REPORT_SCHEMA
            or payload.get("layer") != layer
            or payload.get("scope") != "formal"
            or payload.get("status") != "PASS"
            or payload.get("formal_eligible") is not True
        ):
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_REPORT_NOT_FORMAL_PASS:{layer}")
        _safe_id(payload.get("replay_id"), field=f"replay_reports.{layer}.replay_id")
        commit = payload.get("implementation_commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_COMMIT_INVALID:{layer}")
        _hash(payload.get("environment_hash"), field=f"replay_reports.{layer}.environment_hash")
        command = payload.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_COMMAND_INVALID:{layer}")
        returncode = payload.get("returncode")
        if isinstance(returncode, bool) or returncode != 0:
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_RETURNCODE_NOT_ZERO:{layer}")
        started = _utc_timestamp(payload.get("started_at"), field=f"{layer}.started_at")
        completed = _utc_timestamp(payload.get("completed_at"), field=f"{layer}.completed_at")
        if completed < started:
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_TIMESTAMP_ORDER_INVALID:{layer}")
        if payload.get("cache_mode") != _REPLAY_CACHE_MODES[layer]:
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_CACHE_MODE_INVALID:{layer}")

        summary = _mapping(payload.get("test_summary"), field=f"replay_reports.{layer}.test_summary")
        if set(summary) != {"collected", "passed", "failed", "errors", "skipped"}:
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_TEST_SUMMARY_FIELDS_INVALID:{layer}")
        counts = tuple(summary[name] for name in ("collected", "passed", "failed", "errors", "skipped"))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_TEST_SUMMARY_INVALID:{layer}")
        collected, passed, failed, errors, skipped = counts
        if collected <= 0 or passed != collected or any(value != 0 for value in (failed, errors, skipped)):
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_TEST_SUMMARY_NOT_PASS:{layer}")

        input_refs = _mapping(payload.get("input_refs"), field=f"replay_reports.{layer}.input_refs")
        input_hashes = _mapping(payload.get("input_hashes"), field=f"replay_reports.{layer}.input_hashes")
        if not input_refs or set(input_refs) != set(input_hashes):
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_INPUT_BINDING_INVALID:{layer}")
        for name, source_ref in input_refs.items():
            if not isinstance(name, str):
                raise FormalRunRejected(f"STAGE3_G38_REPLAY_INPUT_NAME_INVALID:{layer}")
            _safe_id(name, field=f"replay_reports.{layer}.input_refs")
            ref = _safe_ref(source_ref, field=f"replay_reports.{layer}.input_refs.{name}")
            digest = _hash(
                input_hashes[name],
                field=f"replay_reports.{layer}.input_hashes.{name}",
            )
            _verify_replay_input_file(
                root,
                ref=ref,
                digest=digest,
                identity=f"{layer}:{name}",
            )

        evidence = payload.get("evidence_files")
        if not isinstance(evidence, list) or not evidence:
            raise FormalRunRejected(f"STAGE3_G38_REPLAY_EVIDENCE_FILES_MISSING:{layer}")
        evidence_seen: set[str] = set()
        for index, item in enumerate(evidence):
            parsed_record = _file_mapping(item, field=f"replay_reports.{layer}.evidence_files[{index}]")
            evidence_ref = str(parsed_record["path"])
            if evidence_ref in evidence_seen:
                raise FormalRunRejected(f"STAGE3_G38_REPLAY_EVIDENCE_FILE_DUPLICATE:{layer}")
            evidence_seen.add(evidence_ref)
            _verify_file_record(
                root,
                parsed_record,
                field=f"replay_reports.{layer}.evidence_files[{index}]",
            )
        _normalise_hash_object(payload, field=f"replay_reports.{layer}")


def _git_branch(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", value) is None
        or value.endswith("/")
        or "//" in value
        or ".." in value
        or _FORBIDDEN_RE.search(value)
        or _FUTURE_RE.search(value)
    ):
        raise FormalRunRejected(f"STAGE3_G38_GIT_BRANCH_INVALID:{field}")
    return value


def validate_stage3_git_sync_evidence(
    workspace_root: str | Path,
    manifest: Stage3G38DeliveryManifest,
) -> None:
    """Verify six command-backed Git roles and one consistent three-end snapshot."""

    root = Path(workspace_root).resolve()
    required = {
        "schema_version",
        "evidence_id",
        "role",
        "scope",
        "status",
        "formal_eligible",
        "checked_at",
        "branch",
        "local_commit",
        "remote_commit",
        "server_commit",
        "remote_name",
        "local_delivery_worktree_clean",
        "server_worktree_clean",
        "agent_document_hashes",
        "command",
        "returncode",
        "stdout_log",
        "artifact_hash",
    }
    expected_snapshot: tuple[object, ...] | None = None
    for role in REQUIRED_STAGE3_G38_GIT_ROLES:
        record = manifest.git_sync[role]
        _verify_file_record(root, record, field=f"git_sync.{role}")
        ref = _safe_ref(
            record.get("path"),
            field=f"git_sync.{role}.path",
            reject_future=False,
        )
        try:
            payload = _mapping(
                load_canonical_json(root.joinpath(*PurePosixPath(ref).parts)),
                field=f"git_sync.{role}",
            )
        except (OSError, TypeError, ValueError) as error:
            raise FormalRunRejected(f"STAGE3_G38_GIT_EVIDENCE_JSON_INVALID:{role}") from error
        if set(payload) != required:
            raise FormalRunRejected(f"STAGE3_G38_GIT_EVIDENCE_FIELDS_INVALID:{role}")
        if (
            payload.get("schema_version") != STAGE3_G38_GIT_SYNC_EVIDENCE_SCHEMA
            or payload.get("role") != role
            or payload.get("scope") != "formal"
            or payload.get("status") != "PASS"
            or payload.get("formal_eligible") is not True
        ):
            raise FormalRunRejected(f"STAGE3_G38_GIT_EVIDENCE_NOT_FORMAL_PASS:{role}")
        _safe_id(payload.get("evidence_id"), field=f"git_sync.{role}.evidence_id")
        _utc_timestamp(payload.get("checked_at"), field=f"git_sync.{role}.checked_at")
        branch = _git_branch(payload.get("branch"), field=role)
        commits: list[str] = []
        for name in ("local_commit", "remote_commit", "server_commit"):
            commit = payload.get(name)
            if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                raise FormalRunRejected(f"STAGE3_G38_GIT_COMMIT_INVALID:{role}:{name}")
            commits.append(commit)
        if len(set(commits)) != 1:
            raise FormalRunRejected(f"STAGE3_G38_GIT_HEAD_MISMATCH:{role}")
        remote_name = _safe_id(payload.get("remote_name"), field=f"git_sync.{role}.remote_name")
        if (
            payload.get("local_delivery_worktree_clean") is not True
            or payload.get("server_worktree_clean") is not True
        ):
            raise FormalRunRejected(f"STAGE3_G38_GIT_WORKTREE_NOT_CLEAN:{role}")
        document_hashes = _mapping(
            payload.get("agent_document_hashes"),
            field=f"git_sync.{role}.agent_document_hashes",
        )
        if set(document_hashes) != set(REQUIRED_STAGE3_G38_AGENT_DOCUMENTS):
            raise FormalRunRejected(f"STAGE3_G38_AGENT_DOCUMENT_SET_INVALID:{role}")
        normalized_documents: tuple[tuple[str, str], ...] = tuple(
            (name, _hash(document_hashes[name], field=f"git_sync.{role}.{name}"))
            for name in REQUIRED_STAGE3_G38_AGENT_DOCUMENTS
        )
        command = payload.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise FormalRunRejected(f"STAGE3_G38_GIT_COMMAND_INVALID:{role}")
        returncode = payload.get("returncode")
        if isinstance(returncode, bool) or returncode != 0:
            raise FormalRunRejected(f"STAGE3_G38_GIT_RETURNCODE_NOT_ZERO:{role}")
        log_record = _file_mapping(payload.get("stdout_log"), field=f"git_sync.{role}.stdout_log")
        _verify_file_record(root, log_record, field=f"git_sync.{role}.stdout_log")
        _normalise_hash_object(payload, field=f"git_sync.{role}")

        snapshot: tuple[object, ...] = (
            branch,
            *commits,
            remote_name,
            normalized_documents,
        )
        if expected_snapshot is None:
            expected_snapshot = snapshot
        elif snapshot != expected_snapshot:
            raise FormalRunRejected(f"STAGE3_G38_GIT_ROLE_SNAPSHOT_DRIFT:{role}")


def _directory_file_refs(root: Path, root_ref: str) -> set[str]:
    path = root.joinpath(*PurePosixPath(root_ref).parts)
    try:
        current = root
        for part in PurePosixPath(root_ref).parts:
            current = current / part
            if current.is_symlink():
                raise FormalRunRejected(f"STAGE3_G38_LARGE_ROOT_SYMLINK:{root_ref}")
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except FormalRunRejected:
        raise
    except (OSError, ValueError) as error:
        raise FormalRunRejected(f"STAGE3_G38_LARGE_ROOT_INVALID:{root_ref}") from error
    if not resolved.is_dir():
        raise FormalRunRejected(f"STAGE3_G38_LARGE_ROOT_NOT_DIRECTORY:{root_ref}")

    files: set[str] = set()

    def walk_error(error: OSError) -> None:
        raise FormalRunRejected(f"STAGE3_G38_LARGE_ROOT_WALK_FAILED:{root_ref}") from error

    for directory, dirnames, filenames in os.walk(resolved, topdown=True, onerror=walk_error, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            if (directory_path / name).is_symlink():
                raise FormalRunRejected(f"STAGE3_G38_LARGE_TREE_SYMLINK:{root_ref}")
        for name in filenames:
            candidate = directory_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise FormalRunRejected(f"STAGE3_G38_LARGE_TREE_NONREGULAR:{root_ref}")
            files.add(candidate.relative_to(root).as_posix())
    return files


def validate_stage3_large_artifact_manifest(
    workspace_root: str | Path,
    manifest: Stage3G38DeliveryManifest,
) -> None:
    """Verify the complete, per-file hash inventory of every Stage 3 large root."""

    root = Path(workspace_root).resolve()
    record = manifest.server_large_artifact_manifest
    _verify_file_record(root, record, field="server_large_artifact_manifest")
    ref = _safe_ref(
        record.get("path"),
        field="server_large_artifact_manifest.path",
        reject_future=False,
    )
    try:
        payload = _mapping(
            load_canonical_json(root.joinpath(*PurePosixPath(ref).parts)),
            field="server_large_artifact_manifest",
        )
    except (OSError, TypeError, ValueError) as error:
        raise FormalRunRejected("STAGE3_G38_LARGE_MANIFEST_JSON_INVALID") from error
    required = {
        "schema_version",
        "manifest_id",
        "scope",
        "status",
        "formal_eligible",
        "generated_at",
        "source_refs",
        "source_hashes",
        "artifact_roots",
        "file_count",
        "total_size",
        "artifact_hash",
    }
    if set(payload) != required:
        raise FormalRunRejected("STAGE3_G38_LARGE_MANIFEST_FIELDS_INVALID")
    if (
        payload.get("schema_version") != STAGE3_G38_LARGE_ARTIFACT_MANIFEST_SCHEMA
        or payload.get("scope") != "formal"
        or payload.get("status") != "PASS"
        or payload.get("formal_eligible") is not True
    ):
        raise FormalRunRejected("STAGE3_G38_LARGE_MANIFEST_NOT_FORMAL_PASS")
    _safe_id(payload.get("manifest_id"), field="server_large_artifact_manifest.manifest_id")
    _utc_timestamp(payload.get("generated_at"), field="server_large_artifact_manifest.generated_at")
    source_refs = _mapping(payload.get("source_refs"), field="server_large_artifact_manifest.source_refs")
    source_hashes = _mapping(payload.get("source_hashes"), field="server_large_artifact_manifest.source_hashes")
    if not source_refs or set(source_refs) != set(source_hashes):
        raise FormalRunRejected("STAGE3_G38_LARGE_MANIFEST_SOURCE_BINDING_INVALID")
    for name, source_ref in source_refs.items():
        if not isinstance(name, str):
            raise FormalRunRejected("STAGE3_G38_LARGE_MANIFEST_SOURCE_NAME_INVALID")
        _safe_id(name, field="server_large_artifact_manifest.source_refs")
        _safe_ref(source_ref, field=f"server_large_artifact_manifest.source_refs.{name}")
        _hash(source_hashes[name], field=f"server_large_artifact_manifest.source_hashes.{name}")

    groups = payload.get("artifact_roots")
    if not isinstance(groups, list) or not groups:
        raise FormalRunRejected("STAGE3_G38_LARGE_MANIFEST_ROOTS_MISSING")
    seen_roles: set[str] = set()
    seen_roots: list[PurePosixPath] = []
    seen_files: set[str] = set()
    total_count = 0
    total_size = 0
    for index, raw_group in enumerate(groups):
        group = _mapping(raw_group, field=f"artifact_roots[{index}]")
        if set(group) != {"role", "root_ref", "files", "file_count", "total_size", "collection_hash"}:
            raise FormalRunRejected(f"STAGE3_G38_LARGE_GROUP_FIELDS_INVALID:{index}")
        role = _safe_id(group.get("role"), field=f"artifact_roots[{index}].role")
        if role in seen_roles:
            raise FormalRunRejected(f"STAGE3_G38_LARGE_ROLE_DUPLICATE:{role}")
        seen_roles.add(role)
        root_ref = _safe_ref(
            group.get("root_ref"),
            field=f"artifact_roots[{index}].root_ref",
            reject_future=False,
        )
        root_path = PurePosixPath(root_ref)
        if any(root_path == prior or root_path in prior.parents or prior in root_path.parents for prior in seen_roots):
            raise FormalRunRejected(f"STAGE3_G38_LARGE_ROOT_OVERLAP:{root_ref}")
        seen_roots.append(root_path)
        files = group.get("files")
        if not isinstance(files, list) or not files:
            raise FormalRunRejected(f"STAGE3_G38_LARGE_GROUP_FILES_MISSING:{role}")
        declared: set[str] = set()
        normalized_files: list[dict[str, JSONValue]] = []
        previous_ref: str | None = None
        group_size = 0
        for file_index, raw_file in enumerate(files):
            item = _mapping(raw_file, field=f"artifact_roots[{index}].files[{file_index}]")
            if set(item) != {"path", "sha256", "size"}:
                raise FormalRunRejected(f"STAGE3_G38_LARGE_FILE_FIELDS_INVALID:{role}:{file_index}")
            parsed = _file_mapping(item, field=f"artifact_roots[{index}].files[{file_index}]")
            file_ref = str(parsed["path"])
            try:
                PurePosixPath(file_ref).relative_to(root_path)
            except ValueError as error:
                raise FormalRunRejected(f"STAGE3_G38_LARGE_FILE_OUTSIDE_ROOT:{file_ref}") from error
            if previous_ref is not None and file_ref <= previous_ref:
                raise FormalRunRejected(f"STAGE3_G38_LARGE_FILES_NOT_SORTED:{role}")
            previous_ref = file_ref
            if file_ref in seen_files:
                raise FormalRunRejected(f"STAGE3_G38_LARGE_FILE_DUPLICATE:{file_ref}")
            seen_files.add(file_ref)
            declared.add(file_ref)
            _verify_file_record(root, parsed, field=f"artifact_roots[{index}].files[{file_index}]")
            group_size += int(parsed["size"])
            normalized_files.append({
                "path": file_ref,
                "sha256": str(parsed["sha256"]),
                "size": int(parsed["size"]),
            })
        actual = _directory_file_refs(root, root_ref)
        if actual != declared:
            raise FormalRunRejected(f"STAGE3_G38_LARGE_DIRECTORY_CLOSURE_MISMATCH:{role}")
        group_count = group.get("file_count")
        declared_group_size = group.get("total_size")
        if (
            isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or isinstance(declared_group_size, bool)
            or not isinstance(declared_group_size, int)
            or group_count != len(normalized_files)
            or declared_group_size != group_size
        ):
            raise FormalRunRejected(f"STAGE3_G38_LARGE_GROUP_TOTAL_MISMATCH:{role}")
        collection_body: dict[str, JSONValue] = {
            "role": role,
            "root_ref": root_ref,
            "files": normalized_files,
            "file_count": len(normalized_files),
            "total_size": group_size,
        }
        if group.get("collection_hash") != canonical_json_hash(collection_body):
            raise FormalRunRejected(f"STAGE3_G38_LARGE_GROUP_HASH_MISMATCH:{role}")
        total_count += len(normalized_files)
        total_size += group_size
    if not set(REQUIRED_STAGE3_G38_LARGE_ARTIFACT_ROLES).issubset(seen_roles):
        raise FormalRunRejected("STAGE3_G38_LARGE_REQUIRED_ROLES_MISSING")
    manifest_count = payload.get("file_count")
    declared_total_size = payload.get("total_size")
    if (
        isinstance(manifest_count, bool)
        or not isinstance(manifest_count, int)
        or isinstance(declared_total_size, bool)
        or not isinstance(declared_total_size, int)
        or manifest_count != total_count
        or declared_total_size != total_size
    ):
        raise FormalRunRejected("STAGE3_G38_LARGE_MANIFEST_TOTAL_MISMATCH")
    _normalise_hash_object(payload, field="server_large_artifact_manifest")


def _status_from_payload(payload: Mapping[str, object]) -> str | None:
    status = payload.get("status")
    if isinstance(status, str):
        return status
    qualification = payload.get("qualification")
    if isinstance(qualification, Mapping) and isinstance(qualification.get("status"), str):
        return str(qualification["status"])
    return None


def _require_qualified_recommendation(payload: Mapping[str, object]) -> None:
    status = _status_from_payload(payload)
    nested = payload.get("recommendation")
    nested_status = _status_from_payload(nested) if isinstance(nested, Mapping) else None
    candidate = payload if status == "QUALIFIED" else nested
    if status != "QUALIFIED" and nested_status != "QUALIFIED":
        raise FormalRunRejected("STAGE3_G38_QUALIFIED_RECOMMENDATION_REQUIRED")
    if payload.get("scope", "formal") != "formal":
        raise FormalRunRejected("STAGE3_G38_RECOMMENDATION_FORMAL_SCOPE_REQUIRED")
    if isinstance(candidate, Mapping):
        if candidate.get("formal_eligible") is not True:
            raise FormalRunRejected("STAGE3_G38_RECOMMENDATION_NOT_FORMAL_ELIGIBLE")
        supplied = candidate.get("artifact_hash")
        if supplied is not None:
            _hash(supplied, field="recommendation.artifact_hash")
            if canonical_json_hash({key: item for key, item in candidate.items() if key != "artifact_hash"}) != supplied:
                raise FormalRunRejected("STAGE3_G38_RECOMMENDATION_HASH_MISMATCH")


def _require_finalization_pass(payload: Mapping[str, object]) -> None:
    if payload.get("status") != "PASS" or payload.get("formal_eligible") is not True:
        raise FormalRunRejected("STAGE3_G38_FINALIZATION_NOT_FORMAL_PASS")
    recommendation = payload.get("recommendation")
    method = payload.get("method_selection")
    recommendation_status = _status_from_payload(recommendation) if isinstance(recommendation, Mapping) else None
    method_status = method.get("status") if isinstance(method, Mapping) else None
    if recommendation_status != "QUALIFIED" and method_status != "QUALIFIED":
        raise FormalRunRejected("STAGE3_G38_FINALIZATION_QUALIFIED_RECOMMENDATION_MISSING")
    supplied = payload.get("artifact_hash")
    if supplied is not None:
        _hash(supplied, field="finalization.artifact_hash")
        if canonical_json_hash({key: item for key, item in payload.items() if key != "artifact_hash"}) != supplied:
            raise FormalRunRejected("STAGE3_G38_FINALIZATION_HASH_MISMATCH")


def _gate_refs(value: object) -> dict[str, str]:
    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for gate_id, reference in value.items():
            if not isinstance(gate_id, str):
                raise TypeError("gate_refs key 必须是字符串")
            normalized_id = gate_id if gate_id.startswith("stage3.") else f"stage3.{gate_id}"
            result[normalized_id] = _safe_ref(reference, field=f"gate_refs.{gate_id}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = {}
        for index, reference in enumerate(value):
            result[f"stage3.G3-{index}"] = _safe_ref(reference, field=f"gate_refs[{index}]")
        return result
    raise TypeError("gate_refs 必须是 stage3.G3-0..G3-7 到 commit ref 的 mapping")


def _execution_is_g37_extension(
    base: FormalExecutionEvidence,
    current: FormalExecutionEvidence,
) -> bool:
    if (
        base.run_intent != "formal"
        or current.run_intent != "formal"
        or base.contract_freeze_hash != current.contract_freeze_hash
        or base.asset_manifest_hashes != current.asset_manifest_hashes
        or dict(base.metadata) != dict(current.metadata)
    ):
        return False
    base_gates = {gate.gate_id: gate for gate in base.prerequisite_gates}
    current_gates = {gate.gate_id: gate for gate in current.prerequisite_gates}
    if set(base_gates) != set(REQUIRED_STAGE3_G38_GATE_IDS[:6]):
        return False
    if set(current_gates) != set(REQUIRED_STAGE3_G38_GATE_IDS):
        return False
    if any(
        current_gates[gate_id].artifact_hash != gate.artifact_hash
        for gate_id, gate in base_gates.items()
    ):
        return False
    return all(
        current_gates[gate_id].stage == 3
        and current_gates[gate_id].status is GateStatus.PASS
        and current_gates[gate_id].effective_status() is GateStatus.PASS
        for gate_id in REQUIRED_STAGE3_G38_GATE_IDS[6:]
    )


def _stage310_refs(kwargs: Mapping[str, object]) -> dict[str, str]:
    value = kwargs.get("stage3_10_refs", kwargs.get("stage3_10_artifact_refs", kwargs.get("formal_stage3_10_refs")))
    if value is None:
        aliases = {
            "report": "report_ref",
            "visualizations": "visualizations_ref",
            "handoff": "handoff_ref",
            "gate_summary": "gate_summary_ref",
        }
        if all(name in kwargs for name in aliases.values()):
            value = {key: kwargs[name] for key, name in aliases.items()}
    if not isinstance(value, Mapping):
        raise TypeError("stage3_10_refs 必须按 canonical artifact kind 提供 mapping")
    aliases = {
        "report": "analysis_report",
        "visualizations": "chart_artifacts",
        "handoff": "handoff_manifest",
    }
    result = {
        aliases.get(str(key), str(key)): _safe_ref(
            ref, field=f"stage3_10_refs.{key}"
        )
        for key, ref in value.items()
    }
    if (
        set(result) != set(REQUIRED_STAGE3_G38_STAGE310_KINDS)
        or len(set(result.values())) != len(REQUIRED_STAGE3_G38_STAGE310_KINDS)
    ):
        raise FormalRunRejected(
            "STAGE3_G38_REQUIRES_EXACT_CANONICAL_STAGE3_10_COMMITS"
        )
    return {key: result[key] for key in REQUIRED_STAGE3_G38_STAGE310_KINDS}


def _output_ref(output_dir: str, kind: str) -> str:
    logical = _safe_ref(output_dir, field="output_dir", reject_future=False)
    return f"{logical}/commits/{kind}.json"


@dataclass(frozen=True, slots=True)
class Stage3G38Publication:
    """Immutable G3-8 gate plus its complete acceptance receipt."""

    publication_id: str
    task_id: str
    config_hash: str
    input_config_hash: str
    input_config_hashes: Mapping[str, str]
    publication_config_hash: str
    status: str
    formal_eligible: bool
    execution_evidence_ref: str
    execution_evidence_hash: str
    g3_7_publication_ref: str
    g3_7_publication_hash: str
    gate_refs: Mapping[str, str]
    gate_hashes: Mapping[str, str]
    stage3_10_refs: Mapping[str, str]
    stage3_10_hashes: Mapping[str, str]
    recommendation_ref: str
    recommendation_hash: str
    finalization_ref: str
    finalization_hash: str
    delivery_manifest_ref: str
    delivery_manifest_hash: str
    g3_8_ref: str
    g3_8_hash: str
    g3_8_gate: GateRecord
    source_artifact_refs: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    schema_version: str = STAGE3_G38_PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_G38_PUBLICATION_SCHEMA:
            raise ValueError("STAGE3_G38_PUBLICATION_SCHEMA_UNSUPPORTED")
        _safe_id(self.publication_id, field="publication_id")
        _safe_id(self.task_id, field="task_id")
        for field, value in (
            ("config_hash", self.config_hash),
            ("input_config_hash", self.input_config_hash),
            ("publication_config_hash", self.publication_config_hash),
            ("execution_evidence_hash", self.execution_evidence_hash),
            ("g3_7_publication_hash", self.g3_7_publication_hash),
            ("recommendation_hash", self.recommendation_hash),
            ("finalization_hash", self.finalization_hash),
            ("delivery_manifest_hash", self.delivery_manifest_hash),
            ("g3_8_hash", self.g3_8_hash),
        ):
            _hash(value, field=field)
        if not self.input_config_hashes or any(not isinstance(key, str) for key in self.input_config_hashes):
            raise ValueError("STAGE3_G38_PUBLICATION_INPUT_CONFIG_HASHES_INVALID")
        for key, value in self.input_config_hashes.items():
            _hash(value, field=f"input_config_hashes.{key}")
        _safe_ref(self.execution_evidence_ref, field="execution_evidence_ref")
        _safe_ref(self.g3_7_publication_ref, field="g3_7_publication_ref")
        if self.status not in {"PASS", "BLOCKED"} or self.formal_eligible is not (self.status == "PASS"):
            raise FormalRunRejected("STAGE3_G38_PUBLICATION_STATUS_ELIGIBILITY_MISMATCH")
        refs = tuple(self.source_artifact_refs)
        if not refs or len(refs) != len(set(refs)):
            raise ValueError("STAGE3_G38_PUBLICATION_SOURCE_REFS_INVALID")
        for index, ref in enumerate(refs):
            _safe_ref(ref, field=f"source_artifact_refs[{index}]")
        if self.g3_8_ref in refs:
            raise FormalRunRejected("STAGE3_G38_PUBLICATION_SELF_BINDING")
        if not {
            self.execution_evidence_ref,
            self.g3_7_publication_ref,
        }.issubset(set(refs)):
            raise FormalRunRejected("STAGE3_G38_PUBLICATION_AUTHORITY_REFS_UNBOUND")
        if not isinstance(self.g3_8_gate, GateRecord) or self.g3_8_gate.gate_id != STAGE3_G38_GATE_ID:
            raise ValueError("STAGE3_G38_PUBLICATION_GATE_INVALID")
        if self.g3_8_gate.artifact_hash != self.g3_8_hash:
            raise ValueError("STAGE3_G38_PUBLICATION_GATE_HASH_MISMATCH")
        if self.g3_8_gate.status.value != self.status:
            raise FormalRunRejected("STAGE3_G38_PUBLICATION_GATE_STATUS_MISMATCH")
        if self.status == "PASS" and self.reasons:
            raise FormalRunRejected("STAGE3_G38_PUBLICATION_PASS_HAS_REASONS")
        if self.status == "BLOCKED" and not self.reasons:
            raise FormalRunRejected("STAGE3_G38_PUBLICATION_BLOCKED_REASON_REQUIRED")
        gate_refs = dict(self.gate_refs)
        gate_hashes = dict(self.gate_hashes)
        if set(gate_refs) != set(REQUIRED_STAGE3_G38_GATE_IDS) or set(gate_hashes) != set(REQUIRED_STAGE3_G38_GATE_IDS):
            raise FormalRunRejected("STAGE3_G38_PUBLICATION_GATE_COVERAGE_INVALID")
        for gate_id in REQUIRED_STAGE3_G38_GATE_IDS:
            _safe_ref(gate_refs[gate_id], field=f"gate_refs.{gate_id}")
            _hash(gate_hashes[gate_id], field=f"gate_hashes.{gate_id}")
        stage_refs = dict(self.stage3_10_refs)
        stage_hashes = dict(self.stage3_10_hashes)
        if len(stage_refs) != 4 or len(stage_refs) != len(set(stage_refs)) or set(stage_refs) != set(stage_hashes):
            raise FormalRunRejected("STAGE3_G38_PUBLICATION_STAGE3_10_COVERAGE_INVALID")
        for role in stage_refs:
            _safe_ref(stage_refs[role], field=f"stage3_10_refs.{role}")
            _hash(stage_hashes[role], field=f"stage3_10_hashes.{role}")
        for field, value in (
            ("recommendation_ref", self.recommendation_ref),
            ("finalization_ref", self.finalization_ref),
            ("delivery_manifest_ref", self.delivery_manifest_ref),
            ("g3_8_ref", self.g3_8_ref),
        ):
            _safe_ref(value, field=field, reject_future=False)

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "task_id": self.task_id,
            "config_hash": self.config_hash,
            "input_config_hash": self.input_config_hash,
            "input_config_hashes": dict(self.input_config_hashes),
            "publication_config_hash": self.publication_config_hash,
            "status": self.status,
            "scope": "formal",
            "formal_eligible": self.formal_eligible,
            "execution_evidence_ref": self.execution_evidence_ref,
            "execution_evidence_hash": self.execution_evidence_hash,
            "g3_7_publication_ref": self.g3_7_publication_ref,
            "g3_7_publication_hash": self.g3_7_publication_hash,
            "gate_refs": dict(self.gate_refs),
            "gate_hashes": dict(self.gate_hashes),
            "stage3_10_refs": dict(self.stage3_10_refs),
            "stage3_10_hashes": dict(self.stage3_10_hashes),
            "recommendation_ref": self.recommendation_ref,
            "recommendation_hash": self.recommendation_hash,
            "finalization_ref": self.finalization_ref,
            "finalization_hash": self.finalization_hash,
            "delivery_manifest_ref": self.delivery_manifest_ref,
            "delivery_manifest_hash": self.delivery_manifest_hash,
            "g3_8_ref": self.g3_8_ref,
            "g3_8_hash": self.g3_8_hash,
            "g3_8_gate": self.g3_8_gate.to_dict(),
            "source_artifact_refs": list(self.source_artifact_refs),
            "reasons": list(self.reasons),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage3G38Publication":
        raw = _mapping(value, field="g3_8_publication")
        required = {
            "schema_version", "publication_id", "task_id", "config_hash", "input_config_hash", "input_config_hashes", "publication_config_hash", "status", "scope", "formal_eligible",
            "execution_evidence_ref", "execution_evidence_hash", "g3_7_publication_ref", "g3_7_publication_hash",
            "gate_refs", "gate_hashes", "stage3_10_refs", "stage3_10_hashes", "recommendation_ref", "recommendation_hash",
            "finalization_ref", "finalization_hash", "delivery_manifest_ref", "delivery_manifest_hash", "g3_8_ref", "g3_8_hash",
            "g3_8_gate", "source_artifact_refs", "reasons", "artifact_hash",
        }
        if set(raw) != required:
            raise ValueError("STAGE3_G38_PUBLICATION_FIELDS_MISMATCH")
        supplied = raw.get("artifact_hash")
        _hash(supplied, field="artifact_hash")
        if raw.get("scope") != "formal":
            raise ValueError("STAGE3_G38_PUBLICATION_SCOPE_INVALID")
        if canonical_json_hash({key: item for key, item in raw.items() if key != "artifact_hash"}) != supplied:
            raise ValueError("STAGE3_G38_PUBLICATION_HASH_MISMATCH")
        gate = raw.get("g3_8_gate")
        if not isinstance(gate, Mapping):
            raise TypeError("STAGE3_G38_PUBLICATION_GATE_REQUIRED")
        records = ("gate_refs", "gate_hashes", "input_config_hashes", "stage3_10_refs", "stage3_10_hashes")
        if any(not isinstance(raw[name], Mapping) for name in records) or not isinstance(raw["source_artifact_refs"], list) or not isinstance(raw["reasons"], list):
            raise TypeError("STAGE3_G38_PUBLICATION_ARRAYS_INVALID")
        return cls(
            publication_id=raw["publication_id"], task_id=raw["task_id"], config_hash=raw["config_hash"], input_config_hash=raw["input_config_hash"], input_config_hashes=dict(raw["input_config_hashes"]), publication_config_hash=raw["publication_config_hash"],
            status=raw["status"], formal_eligible=raw["formal_eligible"], execution_evidence_ref=raw["execution_evidence_ref"], execution_evidence_hash=raw["execution_evidence_hash"],
            g3_7_publication_ref=raw["g3_7_publication_ref"], g3_7_publication_hash=raw["g3_7_publication_hash"], gate_refs=dict(raw["gate_refs"]), gate_hashes=dict(raw["gate_hashes"]),
            stage3_10_refs=dict(raw["stage3_10_refs"]), stage3_10_hashes=dict(raw["stage3_10_hashes"]), recommendation_ref=raw["recommendation_ref"], recommendation_hash=raw["recommendation_hash"],
            finalization_ref=raw["finalization_ref"], finalization_hash=raw["finalization_hash"], delivery_manifest_ref=raw["delivery_manifest_ref"], delivery_manifest_hash=raw["delivery_manifest_hash"],
            g3_8_ref=raw["g3_8_ref"], g3_8_hash=raw["g3_8_hash"], g3_8_gate=GateRecord.from_mapping(dict(gate)), source_artifact_refs=tuple(raw["source_artifact_refs"]), reasons=tuple(raw["reasons"]),
            schema_version=raw["schema_version"],
        )


def validate_stage3_g38_handoff_authority(
    workspace_root: str | Path,
    *,
    gate_ref: str,
    publication_ref: str,
) -> dict[str, JSONValue]:
    """Reload and bind the canonical G3-8 gate/receipt for Stage 4 entry.

    This is intentionally a lightweight *consumer* check.  G3-8 already
    streamed the full large-artifact inventory when it published the gate;
    Stage 4 reopens every immutable control-plane authority and verifies all
    hashes/config identities without rehashing terabytes before each task.
    A self-consistent GateRecord from another producer is therefore
    insufficient: both canonical G3-8 commits and their complete cross-links
    must still be live.
    """

    root = Path(workspace_root).resolve()
    canonical_gate_ref = _safe_ref(
        gate_ref, field="stage4_g3_8_gate_ref", reject_future=False
    )
    canonical_publication_ref = _safe_ref(
        publication_ref,
        field="stage4_g3_8_publication_ref",
        reject_future=False,
    )
    if canonical_gate_ref == canonical_publication_ref:
        raise FormalRunRejected("STAGE3_G38_HANDOFF_GATE_RECEIPT_REF_COLLISION")

    gate_artifact = _load_formal_commit(
        root,
        canonical_gate_ref,
        field="stage4_g3_8_gate",
        config_hash=None,
        expected_kind=STAGE3_G38_GATE_ARTIFACT_KIND,
        expected_task_id=STAGE3_G38_TASK_ID,
    )
    publication_artifact = _load_formal_commit(
        root,
        canonical_publication_ref,
        field="stage4_g3_8_publication",
        config_hash=None,
        expected_kind=STAGE3_G38_RECEIPT_ARTIFACT_KIND,
        expected_task_id=STAGE3_G38_TASK_ID,
    )
    try:
        gate = GateRecord.from_mapping(dict(gate_artifact.payload))
        publication = Stage3G38Publication.from_mapping(
            dict(publication_artifact.payload)
        )
    except (TypeError, ValueError) as error:
        raise FormalRunRejected("STAGE3_G38_HANDOFF_PAYLOAD_INVALID") from error
    if (
        gate.gate_id != STAGE3_G38_GATE_ID
        or gate.stage != 3
        or gate.status is not GateStatus.PASS
        or gate.effective_status() is not GateStatus.PASS
        or publication.status != "PASS"
        or publication.formal_eligible is not True
        or publication.task_id != STAGE3_G38_TASK_ID
    ):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_NOT_LIVE_PASS")
    if (
        gate_artifact.identity.config_hash
        != publication_artifact.identity.config_hash
        or publication.config_hash != publication.publication_config_hash
        or publication.publication_config_hash
        != gate_artifact.identity.config_hash
    ):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_PUBLICATION_CONFIG_MISMATCH")
    if (
        publication.g3_8_ref != canonical_gate_ref
        or publication.g3_8_hash != gate.artifact_hash
        or publication.g3_8_gate.to_dict() != gate.to_dict()
    ):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_GATE_BINDING_MISMATCH")

    gate_ref_map = {
        gate_id: publication.gate_refs[gate_id]
        for gate_id in REQUIRED_STAGE3_G38_GATE_IDS
    }
    gate_hash_map = {
        gate_id: publication.gate_hashes[gate_id]
        for gate_id in REQUIRED_STAGE3_G38_GATE_IDS
    }
    loaded_gates: dict[str, LoadedTaskArtifact] = {}
    gate_records: dict[str, GateRecord] = {}
    for gate_id in REQUIRED_STAGE3_G38_GATE_IDS:
        loaded = _load_formal_commit(
            root,
            gate_ref_map[gate_id],
            field=f"stage4_{gate_id.replace('.', '_')}",
            config_hash=None,
            expected_kind=STAGE3_G38_GATE_ARTIFACT_KIND,
        )
        try:
            record = GateRecord.from_mapping(dict(loaded.payload))
        except (TypeError, ValueError) as error:
            raise FormalRunRejected(
                f"STAGE3_G38_HANDOFF_PREREQUISITE_GATE_INVALID:{gate_id}"
            ) from error
        if (
            record.gate_id != gate_id
            or record.stage != 3
            or record.status is not GateStatus.PASS
            or record.effective_status() is not GateStatus.PASS
            or record.artifact_hash != gate_hash_map[gate_id]
        ):
            raise FormalRunRejected(
                f"STAGE3_G38_HANDOFF_PREREQUISITE_GATE_DRIFT:{gate_id}"
            )
        loaded_gates[gate_id] = loaded
        gate_records[gate_id] = record

    if set(publication.stage3_10_refs) != set(REQUIRED_STAGE3_G38_STAGE310_KINDS):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_STAGE310_COVERAGE_INVALID")
    stage_ref_map = {
        role: publication.stage3_10_refs[role]
        for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
    }
    stage_loaded = {
        role: _load_formal_commit(
            root,
            stage_ref_map[role],
            field=f"stage4_stage3_10_{role}",
            config_hash=None,
            expected_kind=role,
            expected_task_id=STAGE3_G38_STAGE310_TASK_ID,
        )
        for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
    }
    if len({item.identity.config_hash for item in stage_loaded.values()}) != 1:
        raise FormalRunRejected("STAGE3_G38_HANDOFF_STAGE310_CONFIG_MISMATCH")
    for role, loaded in stage_loaded.items():
        if loaded.identity.artifact_hash != publication.stage3_10_hashes[role]:
            raise FormalRunRejected(
                f"STAGE3_G38_HANDOFF_STAGE310_HASH_DRIFT:{role}"
            )
        status = _status_from_payload(loaded.payload)
        if (
            status in {"BLOCKED", "FAIL", "FAILED", "NOT_RUN"}
            or loaded.payload.get("formal_eligible") is False
        ):
            raise FormalRunRejected(
                f"STAGE3_G38_HANDOFF_STAGE310_NOT_PASS:{role}"
            )

    from .stage3_g37_publisher import (
        STAGE3_G37_FINALIZATION_ARTIFACT_KIND,
        STAGE3_G37_PUBLICATION_ARTIFACT_KIND,
        STAGE3_G37_RECOMMENDATION_ARTIFACT_KIND,
        STAGE3_G37_TASK_ID,
        Stage3G37Publication,
    )

    execution_artifact = _load_formal_commit(
        root,
        publication.execution_evidence_ref,
        field="stage4_execution_evidence",
        config_hash=None,
        expected_kind="formal_execution_evidence",
    )
    execution = FormalExecutionEvidence.from_mapping(execution_artifact.payload)
    execution.require_for_stage(3)
    execution_gates = {
        item.gate_id: item for item in execution.prerequisite_gates
    }
    if (
        execution.artifact_hash != publication.execution_evidence_hash
        or set(execution_gates) != set(REQUIRED_STAGE3_G38_GATE_IDS)
        or any(
            execution_gates[gate_id].artifact_hash
            != gate_records[gate_id].artifact_hash
            for gate_id in REQUIRED_STAGE3_G38_GATE_IDS
        )
    ):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_EXECUTION_EVIDENCE_DRIFT")

    g37_artifact = _load_formal_commit(
        root,
        publication.g3_7_publication_ref,
        field="stage4_g3_7_publication",
        config_hash=None,
        expected_kind=STAGE3_G37_PUBLICATION_ARTIFACT_KIND,
        expected_task_id=STAGE3_G37_TASK_ID,
    )
    g37 = Stage3G37Publication.from_mapping(dict(g37_artifact.payload))
    if (
        g37.status != "PASS"
        or g37.formal_eligible is not True
        or g37.artifact_hash != publication.g3_7_publication_hash
        or g37.g3_7_ref != gate_ref_map["stage3.G3-7"]
        or g37.g3_7_hash != gate_hash_map["stage3.G3-7"]
        or g37.recommendation_ref != publication.recommendation_ref
        or g37.finalization_ref != publication.finalization_ref
    ):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_G37_AUTHORITY_DRIFT")

    recommendation = _load_formal_commit(
        root,
        publication.recommendation_ref,
        field="stage4_recommendation",
        config_hash=g37.publication_config_hash,
        expected_kind=STAGE3_G37_RECOMMENDATION_ARTIFACT_KIND,
        expected_task_id=STAGE3_G37_TASK_ID,
    )
    finalization = _load_formal_commit(
        root,
        publication.finalization_ref,
        field="stage4_finalization",
        config_hash=g37.publication_config_hash,
        expected_kind=STAGE3_G37_FINALIZATION_ARTIFACT_KIND,
        expected_task_id=STAGE3_G37_TASK_ID,
    )
    _require_qualified_recommendation(recommendation.payload)
    _require_finalization_pass(finalization.payload)
    recommendation_hash = str(recommendation.payload.get("artifact_hash"))
    finalization_hash = str(finalization.payload.get("artifact_hash"))
    if (
        recommendation_hash != publication.recommendation_hash
        or recommendation_hash != g37.recommendation_hash
        or finalization_hash != publication.finalization_hash
        or finalization_hash != g37.finalization_hash
    ):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_G37_OUTPUT_HASH_DRIFT")

    manifest_artifact = _load_formal_commit(
        root,
        publication.delivery_manifest_ref,
        field="stage4_delivery_manifest",
        config_hash=None,
        expected_kind=STAGE3_G38_MANIFEST_ARTIFACT_KIND,
        expected_task_id=STAGE3_G38_MANIFEST_TASK_ID,
    )
    manifest = Stage3G38DeliveryManifest.from_mapping(
        dict(manifest_artifact.payload)
    )
    if manifest_artifact.identity.artifact_hash != publication.delivery_manifest_hash:
        raise FormalRunRejected("STAGE3_G38_HANDOFF_DELIVERY_MANIFEST_HASH_DRIFT")

    expected_input_config_hashes = {
        **{
            gate_id: loaded_gates[gate_id].identity.config_hash
            for gate_id in REQUIRED_STAGE3_G38_GATE_IDS
        },
        **{
            f"stage3_10.{role}": stage_loaded[role].identity.config_hash
            for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
        },
        "execution_evidence": execution_artifact.identity.config_hash,
        "g3_7_publication": g37_artifact.identity.config_hash,
        "recommendation": recommendation.identity.config_hash,
        "finalization": finalization.identity.config_hash,
        "delivery_manifest": manifest_artifact.identity.config_hash,
    }
    if dict(publication.input_config_hashes) != expected_input_config_hashes:
        raise FormalRunRejected("STAGE3_G38_HANDOFF_INPUT_CONFIG_IDENTITY_DRIFT")
    input_hash_values = set(expected_input_config_hashes.values())
    expected_input_config_hash = (
        next(iter(input_hash_values))
        if len(input_hash_values) == 1
        else canonical_json_hash(
            {
                "schema_version": "stage3-g38-input-config-set-v1",
                "config_hashes": expected_input_config_hashes,
            }
        )
    )
    if publication.input_config_hash != expected_input_config_hash:
        raise FormalRunRejected("STAGE3_G38_HANDOFF_INPUT_CONFIG_HASH_DRIFT")

    publication_config = {
        "schema_version": STAGE3_G38_PUBLICATION_CONFIG_SCHEMA,
        "input_config_hash": expected_input_config_hash,
        "input_config_hashes": expected_input_config_hashes,
        "gate_refs": gate_ref_map,
        "gate_hashes": gate_hash_map,
        "stage3_10_refs": stage_ref_map,
        "stage3_10_hashes": {
            role: stage_loaded[role].identity.artifact_hash
            for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
        },
        "execution_evidence_ref": publication.execution_evidence_ref,
        "execution_evidence_hash": execution.artifact_hash,
        "g3_7_publication_ref": publication.g3_7_publication_ref,
        "g3_7_publication_hash": g37.artifact_hash,
        "recommendation_ref": publication.recommendation_ref,
        "recommendation_hash": recommendation_hash,
        "finalization_ref": publication.finalization_ref,
        "finalization_hash": finalization_hash,
        "delivery_manifest_ref": publication.delivery_manifest_ref,
    }
    computed_publication_config_hash = canonical_json_hash(publication_config)
    if computed_publication_config_hash != publication.publication_config_hash:
        raise FormalRunRejected("STAGE3_G38_HANDOFF_PUBLICATION_CONFIG_HASH_DRIFT")

    expected_sources = tuple(
        dict.fromkeys(
            (
                *gate_ref_map.values(),
                *stage_ref_map.values(),
                publication.execution_evidence_ref,
                publication.g3_7_publication_ref,
                publication.recommendation_ref,
                publication.finalization_ref,
                publication.delivery_manifest_ref,
            )
        )
    )
    if (
        set(publication.source_artifact_refs) != set(expected_sources)
        or len(publication.source_artifact_refs) != len(expected_sources)
        or set(gate.evidence_refs) != set(expected_sources)
        or len(gate.evidence_refs) != len(expected_sources)
        or gate_artifact.source_refs != gate.evidence_refs
        or publication_artifact.source_refs
        != (*publication.source_artifact_refs, canonical_gate_ref)
    ):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_SOURCE_LINEAGE_DRIFT")

    handoff = stage_loaded["handoff_manifest"].payload
    if (
        handoff.get("scope") != "formal"
        or handoff.get("formal_eligible") is not True
        or handoff.get("formal_stage_complete") is not False
        or handoff.get("completion_boundary")
        != "PENDING_G3_8_DELIVERY_ACCEPTANCE"
        or handoff.get("stage3_g37_publication_ref")
        != publication.g3_7_publication_ref
        or handoff.get("stage3_g37_publication_hash") != g37.artifact_hash
        or handoff.get("stage3_g37_gate_ref")
        != gate_ref_map["stage3.G3-7"]
        or handoff.get("stage3_g37_gate_hash")
        != gate_hash_map["stage3.G3-7"]
        or handoff.get("stage3_g37_recommendation_ref")
        != publication.recommendation_ref
        or handoff.get("stage3_g37_recommendation_hash")
        != recommendation_hash
        or handoff.get("stage3_g37_finalization_ref")
        != publication.finalization_ref
        or handoff.get("stage3_g37_finalization_hash") != finalization_hash
        or handoff.get("stage3_finalization") != dict(finalization.payload)
        or handoff.get("execution_evidence_ref")
        != publication.execution_evidence_ref
        or handoff.get("execution_evidence_hash") != execution.artifact_hash
    ):
        raise FormalRunRejected("STAGE3_G38_HANDOFF_MANIFEST_AUTHORITY_DRIFT")

    expected_measured = {
        "g3_0_through_g3_7_live_pass": True,
        "stage3_10_formal_commit_count": 4,
        "delivery_file_count": len(manifest.file_records()),
        "delivery_manifest_hash": manifest_artifact.identity.artifact_hash,
        "publication_config_hash": computed_publication_config_hash,
    }
    expected_threshold = {
        "all_prerequisite_gates_pass": True,
        "stage3_10_formal_commits": 4,
        "delivery_manifest_complete": True,
        "small_file_hash_and_size_match": True,
        "qualified_recommendation": True,
        "formal_finalization": True,
    }
    if gate.measured != expected_measured or gate.threshold != expected_threshold:
        raise FormalRunRejected("STAGE3_G38_HANDOFF_GATE_SEMANTICS_DRIFT")

    lineage_refs = (
        canonical_publication_ref,
        canonical_gate_ref,
        stage_ref_map["handoff_manifest"],
        publication.recommendation_ref,
        publication.finalization_ref,
        publication.delivery_manifest_ref,
    )
    audit: dict[str, JSONValue] = {
        "schema_version": "stage3-g38-handoff-audit-v1",
        "scope": "formal",
        "status": "PASS",
        "formal_eligible": True,
        "publication_ref": canonical_publication_ref,
        "publication_hash": publication.artifact_hash,
        "publication_commit_hash": publication_artifact.identity.artifact_hash,
        "publication_config_hash": computed_publication_config_hash,
        "g3_8_ref": canonical_gate_ref,
        "g3_8_hash": gate.artifact_hash,
        "g3_8_commit_hash": gate_artifact.identity.artifact_hash,
        "handoff_manifest_ref": stage_ref_map["handoff_manifest"],
        "handoff_manifest_hash": stage_loaded[
            "handoff_manifest"
        ].identity.artifact_hash,
        "recommendation_ref": publication.recommendation_ref,
        "recommendation_hash": recommendation_hash,
        "finalization_ref": publication.finalization_ref,
        "finalization_hash": finalization_hash,
        "delivery_manifest_ref": publication.delivery_manifest_ref,
        "delivery_manifest_hash": manifest_artifact.identity.artifact_hash,
        "source_artifact_refs": list(expected_sources),
        "lineage_refs": list(lineage_refs),
    }
    audit["audit_hash"] = canonical_json_hash(audit)
    return audit


class Stage3G38Publisher:
    """Reload formal inputs, verify delivery, then publish G3-8 and receipt."""

    def publish(
        self,
        *,
        workspace_root: str | Path,
        output_dir: str,
        config_hash: str | None = None,
        gate_refs: Mapping[str, str] | Sequence[str] | None = None,
        stage3_10_refs: Mapping[str, str] | Sequence[str] | None = None,
        execution_evidence_ref: str | None = None,
        g3_7_publication_ref: str | None = None,
        recommendation_ref: str | None = None,
        finalization_ref: str | None = None,
        delivery_manifest_ref: str | None = None,
        publication_id: str | None = None,
        publication_config_hash: str | None = None,
        checked_at: str | None = None,
        task_id: str = STAGE3_G38_TASK_ID,
        **aliases: object,
    ) -> Stage3G38Publication:
        """Publish the PASS G3-8 gate only after a complete formal preflight.

        ``aliases`` is limited to backwards-compatible spelling variants;
        unknown aliases are rejected so a typo cannot silently weaken the
        acceptance boundary.
        """

        allowed_aliases = {
            "prerequisite_gate_refs", "gates", "stage3_10_artifact_refs", "formal_stage3_10_refs",
            "qualified_recommendation_ref", "recommendation_commit_ref", "quadrature_decision_ref", "finalization_commit_ref",
            "stage3_finalization_ref", "manifest_ref", "delivery_manifest_commit_ref",
            "formal_execution_evidence_ref", "execution_ref",
            "g37_publication_ref", "stage3_g37_publication_ref",
            "analysis_report_ref", "chart_artifacts_ref", "handoff_manifest_ref", "gate_summary_ref",
        }
        allowed_aliases.update({f"g3_{index}_ref" for index in range(8)})
        allowed_aliases.update({f"g3_{index}_gate_ref" for index in range(8)})
        unknown = set(aliases) - allowed_aliases
        if unknown:
            raise TypeError(f"STAGE3_G38_UNKNOWN_ARGUMENTS:{sorted(unknown)}")
        if gate_refs is None:
            gate_refs = aliases.get("prerequisite_gate_refs", aliases.get("gates"))  # type: ignore[assignment]
        if gate_refs is None:
            individual_gates = {
                f"stage3.G3-{index}": aliases.get(f"g3_{index}_ref", aliases.get(f"g3_{index}_gate_ref"))
                for index in range(8)
            }
            if all(reference is not None for reference in individual_gates.values()):
                gate_refs = individual_gates  # type: ignore[assignment]
        if stage3_10_refs is None:
            stage3_10_refs = aliases.get("stage3_10_artifact_refs", aliases.get("formal_stage3_10_refs"))  # type: ignore[assignment]
        if stage3_10_refs is None and all(name in aliases for name in ("analysis_report_ref", "chart_artifacts_ref", "handoff_manifest_ref", "gate_summary_ref")):
            stage3_10_refs = {
                "analysis_report": aliases["analysis_report_ref"],
                "chart_artifacts": aliases["chart_artifacts_ref"],
                "handoff_manifest": aliases["handoff_manifest_ref"],
                "gate_summary": aliases["gate_summary_ref"],
            }
        if recommendation_ref is None:
            recommendation_ref = aliases.get("qualified_recommendation_ref", aliases.get("recommendation_commit_ref", aliases.get("quadrature_decision_ref")))  # type: ignore[assignment]
        finalization_ref = finalization_ref or aliases.get("finalization_commit_ref", aliases.get("stage3_finalization_ref"))  # type: ignore[assignment]
        delivery_manifest_ref = delivery_manifest_ref or aliases.get("manifest_ref", aliases.get("delivery_manifest_commit_ref"))  # type: ignore[assignment]
        execution_evidence_ref = execution_evidence_ref or aliases.get(
            "formal_execution_evidence_ref", aliases.get("execution_ref")
        )  # type: ignore[assignment]
        g3_7_publication_ref = g3_7_publication_ref or aliases.get(
            "g37_publication_ref", aliases.get("stage3_g37_publication_ref")
        )  # type: ignore[assignment]
        if config_hash is not None:
            _hash(config_hash, field="config_hash")
        _safe_id(task_id, field="task_id")
        if (
            gate_refs is None
            or stage3_10_refs is None
            or delivery_manifest_ref is None
            or execution_evidence_ref is None
            or g3_7_publication_ref is None
        ):
            raise FormalRunRejected("STAGE3_G38_REQUIRED_FORMAL_COMMITS_MISSING")
        root = Path(workspace_root).resolve()
        gate_ref_map = _gate_refs(gate_refs)
        if set(gate_ref_map) != set(REQUIRED_STAGE3_G38_GATE_IDS):
            raise FormalRunRejected("STAGE3_G38_REQUIRES_G3_0_THROUGH_G3_7")
        stage_ref_map = _stage310_refs({"stage3_10_refs": stage3_10_refs})
        manifest_ref = _safe_ref(delivery_manifest_ref, field="delivery_manifest_ref")
        execution_evidence_ref = _safe_ref(
            execution_evidence_ref, field="execution_evidence_ref"
        )
        g3_7_publication_ref = _safe_ref(
            g3_7_publication_ref, field="g3_7_publication_ref"
        )
        expected_gate_ref = _output_ref(output_dir, STAGE3_G38_GATE_ARTIFACT_KIND)
        expected_receipt_ref = _output_ref(output_dir, STAGE3_G38_RECEIPT_ARTIFACT_KIND)
        explicit_input_refs = (
            *gate_ref_map.values(),
            *stage_ref_map.values(),
            manifest_ref,
            execution_evidence_ref,
            g3_7_publication_ref,
        )
        if recommendation_ref is not None:
            recommendation_ref = _safe_ref(recommendation_ref, field="recommendation_ref")
            explicit_input_refs += (recommendation_ref,)
        if finalization_ref is not None:
            finalization_ref = _safe_ref(finalization_ref, field="finalization_ref")
            explicit_input_refs += (finalization_ref,)
        if any(ref in {expected_gate_ref, expected_receipt_ref} for ref in explicit_input_refs):
            raise FormalRunRejected("STAGE3_G38_SELF_OR_FUTURE_INPUT_REF")

        loaded_gates: dict[str, LoadedTaskArtifact] = {}
        gate_records: dict[str, GateRecord] = {}
        for gate_id in REQUIRED_STAGE3_G38_GATE_IDS:
            loaded = _load_formal_commit(root, gate_ref_map[gate_id], field=gate_id.replace(".", "_"), config_hash=None, expected_kind=STAGE3_G38_GATE_ARTIFACT_KIND)
            try:
                record = GateRecord.from_mapping(dict(loaded.payload))
            except (TypeError, ValueError) as error:
                raise FormalRunRejected(f"STAGE3_G38_GATE_PAYLOAD_INVALID:{gate_id}") from error
            if record.gate_id != gate_id or record.stage != 3 or record.status is not GateStatus.PASS or record.effective_status() is not GateStatus.PASS:
                raise FormalRunRejected(f"STAGE3_G38_GATE_NOT_LIVE_PASS:{gate_id}")
            loaded_gates[gate_id], gate_records[gate_id] = loaded, record

        from .stage3_g37_publisher import (
            STAGE3_G37_FINALIZATION_ARTIFACT_KIND,
            STAGE3_G37_PUBLICATION_ARTIFACT_KIND,
            STAGE3_G37_RECOMMENDATION_ARTIFACT_KIND,
            STAGE3_G37_TASK_ID,
            Stage3G37Publication,
        )

        execution_artifact = _load_formal_commit(
            root,
            execution_evidence_ref,
            field="execution_evidence",
            config_hash=None,
            expected_kind="formal_execution_evidence",
        )
        execution = FormalExecutionEvidence.from_mapping(execution_artifact.payload)
        execution.require_for_stage(3)
        execution_gates = {
            gate.gate_id: gate for gate in execution.prerequisite_gates
        }
        if set(execution_gates) != set(REQUIRED_STAGE3_G38_GATE_IDS):
            raise FormalRunRejected(
                "STAGE3_G38_EXECUTION_REQUIRES_EXACT_G3_0_THROUGH_G3_7"
            )
        if any(
            execution_gates[gate_id].artifact_hash
            != gate_records[gate_id].artifact_hash
            for gate_id in REQUIRED_STAGE3_G38_GATE_IDS
        ):
            raise FormalRunRejected("STAGE3_G38_EXECUTION_GATE_HASH_MISMATCH")

        g37_publication_artifact = _load_formal_commit(
            root,
            g3_7_publication_ref,
            field="g3_7_publication",
            config_hash=None,
            expected_kind=STAGE3_G37_PUBLICATION_ARTIFACT_KIND,
            expected_task_id=STAGE3_G37_TASK_ID,
        )
        g37_publication = Stage3G37Publication.from_mapping(
            dict(g37_publication_artifact.payload)
        )
        if (
            g37_publication.status != "PASS"
            or not g37_publication.formal_eligible
            or g37_publication.g3_7_ref != gate_ref_map["stage3.G3-7"]
            or g37_publication.g3_7_hash
            != gate_records["stage3.G3-7"].artifact_hash
            or g37_publication.g3_6_ref != gate_ref_map["stage3.G3-6"]
            or g37_publication.g3_6_hash
            != gate_records["stage3.G3-6"].artifact_hash
        ):
            raise FormalRunRejected("STAGE3_G38_G37_PUBLICATION_NOT_LIVE_PASS")
        g37_base_execution_artifact = _load_formal_commit(
            root,
            g37_publication.execution_evidence_ref,
            field="g3_7_base_execution",
            config_hash=None,
            expected_kind="formal_execution_evidence",
        )
        g37_base_execution = FormalExecutionEvidence.from_mapping(
            g37_base_execution_artifact.payload
        )
        if (
            g37_base_execution.artifact_hash
            != g37_publication.execution_evidence_hash
            or not _execution_is_g37_extension(g37_base_execution, execution)
        ):
            raise FormalRunRejected(
                "STAGE3_G38_EXECUTION_NOT_APPEND_ONLY_G37_EXTENSION"
            )

        stage_loaded = {
            role: _load_formal_commit(
                root,
                stage_ref_map[role],
                field=f"stage3_10_{role}",
                config_hash=None,
                expected_kind=role,
                expected_task_id=STAGE3_G38_STAGE310_TASK_ID,
            )
            for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
        }
        stage310_config_hashes = {
            loaded.identity.config_hash for loaded in stage_loaded.values()
        }
        if len(stage310_config_hashes) != 1:
            raise FormalRunRejected("STAGE3_G38_STAGE310_CONFIG_HASH_MISMATCH")
        for role, loaded in stage_loaded.items():
            status = _status_from_payload(loaded.payload)
            if status in {"BLOCKED", "FAIL", "FAILED", "NOT_RUN"} or loaded.payload.get("formal_eligible") is False:
                raise FormalRunRejected(f"STAGE3_G38_STAGE3_10_ARTIFACT_NOT_PASS:{role}")
        handoff = stage_loaded["handoff_manifest"].payload
        summary = stage_loaded["gate_summary"].payload
        charts = stage_loaded["chart_artifacts"].payload
        _verify_reporting_bundle(
            root,
            stage_loaded["analysis_report"].payload,
            charts,
        )
        raw_aggregate_ref = handoff.get("stage3_07_raw_aggregate_ref")
        raw_aggregate_hash = handoff.get("stage3_07_raw_aggregate_hash")
        if not isinstance(raw_aggregate_ref, str) or not isinstance(raw_aggregate_hash, str):
            raise FormalRunRejected("STAGE3_G38_RAW_AGGREGATE_BINDING_MISSING")
        try:
            from .stage3_raw_storage import iter_raw_aggregate_units

            for _unit_id, _shard, _state, _bundle in iter_raw_aggregate_units(
                root=root,
                aggregate_ref=raw_aggregate_ref,
                aggregate_hash=raw_aggregate_hash,
                require_complete=True,
            ):
                # The iterator validates and releases one TensorBundle per unit;
                # G3-8 only needs the aggregate's fail-closed validity here.
                del _unit_id, _shard, _state, _bundle
        except (OSError, TypeError, ValueError) as error:
            raise FormalRunRejected("STAGE3_G38_RAW_AGGREGATE_INVALID") from error
        report_metadata = stage_loaded["analysis_report"].payload.get("metadata")
        if (
            not isinstance(report_metadata, Mapping)
            or report_metadata.get("stage3_07_raw_aggregate_ref") != raw_aggregate_ref
            or report_metadata.get("stage3_07_raw_aggregate_hash") != raw_aggregate_hash
        ):
            raise FormalRunRejected("STAGE3_G38_RAW_AGGREGATE_REPORT_BINDING_INVALID")
        if (
            handoff.get("scope") != "formal"
            or handoff.get("formal_eligible") is not True
            or handoff.get("formal_stage_complete") is not False
            or handoff.get("completion_boundary")
            != "PENDING_G3_8_DELIVERY_ACCEPTANCE"
            or handoff.get("stage3_g37_publication_ref")
            != g3_7_publication_ref
            or handoff.get("stage3_g37_publication_hash")
            != g37_publication.artifact_hash
            or handoff.get("stage3_g37_gate_ref")
            != gate_ref_map["stage3.G3-7"]
            or handoff.get("stage3_g37_gate_hash")
            != gate_records["stage3.G3-7"].artifact_hash
            or handoff.get("execution_evidence_ref") != execution_evidence_ref
            or handoff.get("execution_evidence_hash") != execution.artifact_hash
        ):
            raise FormalRunRejected("STAGE3_G38_HANDOFF_AUTHORITY_BINDING_INVALID")
        if (
            summary.get("scope") != "formal"
            or summary.get("stage3.G3-6") != "PASS"
            or summary.get("stage3.G3-7") != "PASS"
            or summary.get("stage3.G3-8") != "NOT_RUN"
            or summary.get("formal_exit_gate") != "NOT_RUN"
            or summary.get("stage3.G3-7_ref")
            != gate_ref_map["stage3.G3-7"]
            or summary.get("stage3.G3-7_hash")
            != gate_records["stage3.G3-7"].artifact_hash
        ):
            raise FormalRunRejected("STAGE3_G38_GATE_SUMMARY_INVALID")
        if charts.get("scope") != "formal" or charts.get("formal_eligible") is not True:
            raise FormalRunRejected("STAGE3_G38_CHART_ARTIFACTS_NOT_FORMAL")

        recommendation_ref = recommendation_ref or g37_publication.recommendation_ref
        finalization_ref = finalization_ref or g37_publication.finalization_ref
        if (
            recommendation_ref != g37_publication.recommendation_ref
            or finalization_ref != g37_publication.finalization_ref
            or recommendation_ref is None
            or finalization_ref is None
        ):
            raise FormalRunRejected("STAGE3_G38_G37_OUTPUT_REFS_MISMATCH")
        recommendation = _load_formal_commit(
            root,
            recommendation_ref,
            field="recommendation",
            config_hash=g37_publication.publication_config_hash,
            expected_kind=STAGE3_G37_RECOMMENDATION_ARTIFACT_KIND,
            expected_task_id=STAGE3_G37_TASK_ID,
        )
        finalization = _load_formal_commit(
            root,
            finalization_ref,
            field="finalization",
            config_hash=g37_publication.publication_config_hash,
            expected_kind=STAGE3_G37_FINALIZATION_ARTIFACT_KIND,
            expected_task_id=STAGE3_G37_TASK_ID,
        )
        recommendation_payload = recommendation.payload
        finalization_payload = finalization.payload
        _require_qualified_recommendation(recommendation_payload)
        _require_finalization_pass(finalization_payload)
        recommendation_hash = str(recommendation_payload.get("artifact_hash"))
        finalization_hash = str(finalization_payload.get("artifact_hash"))
        if (
            recommendation_hash != g37_publication.recommendation_hash
            or finalization_hash != g37_publication.finalization_hash
            or g37_publication.recommendation != dict(recommendation_payload)
            or g37_publication.finalization != dict(finalization_payload)
            or handoff.get("stage3_g37_recommendation_ref") != recommendation_ref
            or handoff.get("stage3_g37_recommendation_hash")
            != recommendation_hash
            or handoff.get("stage3_g37_finalization_ref") != finalization_ref
            or handoff.get("stage3_g37_finalization_hash") != finalization_hash
            or handoff.get("stage3_finalization") != dict(finalization_payload)
        ):
            raise FormalRunRejected("STAGE3_G38_G37_OUTPUT_PAYLOAD_MISMATCH")
        finalization_g37_ref = finalization_payload.get("g3_7_ref")
        if finalization_g37_ref is not None and finalization_g37_ref != gate_ref_map["stage3.G3-7"]:
            raise FormalRunRejected("STAGE3_G38_FINALIZATION_G3_7_REF_MISMATCH")

        manifest_commit = _load_formal_commit(
            root,
            manifest_ref,
            field="delivery_manifest",
            config_hash=None,
            expected_kind=STAGE3_G38_MANIFEST_ARTIFACT_KIND,
            expected_task_id=STAGE3_G38_MANIFEST_TASK_ID,
        )
        manifest = Stage3G38DeliveryManifest.from_mapping(dict(manifest_commit.payload))
        _verify_manifest_files(root, manifest)
        validate_stage3_replay_reports(root, manifest)
        validate_stage3_git_sync_evidence(root, manifest)
        validate_stage3_large_artifact_manifest(root, manifest)

        # Upstream publishers may intentionally derive a new publication
        # config hash (G3-6/G3-7), while the Stage3.10 runner retains the
        # experiment config hash.  Do not incorrectly demand one hash across
        # those independent formal commits; bind every identity below.
        input_config_hashes: dict[str, str] = {
            **{gate_id: loaded_gates[gate_id].identity.config_hash for gate_id in REQUIRED_STAGE3_G38_GATE_IDS},
            **{f"stage3_10.{role}": loaded.identity.config_hash for role, loaded in stage_loaded.items()},
            "execution_evidence": execution_artifact.identity.config_hash,
            "g3_7_publication": g37_publication_artifact.identity.config_hash,
            "recommendation": recommendation.identity.config_hash,
            "finalization": finalization.identity.config_hash,
            "delivery_manifest": manifest_commit.identity.config_hash,
        }
        for field, value in input_config_hashes.items():
            _hash(value, field=f"input_config_hashes.{field}")
        input_config_hash = next(iter(set(input_config_hashes.values()))) if len(set(input_config_hashes.values())) == 1 else canonical_json_hash({"schema_version": "stage3-g38-input-config-set-v1", "config_hashes": input_config_hashes})

        base_config: dict[str, JSONValue] = {
            "schema_version": STAGE3_G38_PUBLICATION_CONFIG_SCHEMA,
            "input_config_hash": input_config_hash,
            "input_config_hashes": input_config_hashes,
            "gate_refs": {key: gate_ref_map[key] for key in REQUIRED_STAGE3_G38_GATE_IDS},
            "gate_hashes": {key: gate_records[key].artifact_hash for key in REQUIRED_STAGE3_G38_GATE_IDS},
            "stage3_10_refs": dict(stage_ref_map),
            "stage3_10_hashes": {key: loaded.identity.artifact_hash for key, loaded in stage_loaded.items()},
            "execution_evidence_ref": execution_evidence_ref,
            "execution_evidence_hash": execution.artifact_hash,
            "g3_7_publication_ref": g3_7_publication_ref,
            "g3_7_publication_hash": g37_publication.artifact_hash,
            "recommendation_ref": recommendation_ref,
            "recommendation_hash": recommendation_hash,
            "finalization_ref": finalization_ref,
            "finalization_hash": finalization_hash,
            "delivery_manifest_ref": manifest_ref,
        }
        computed_publication_config_hash = canonical_json_hash(base_config)
        if publication_config_hash is not None:
            _hash(publication_config_hash, field="publication_config_hash")
            if publication_config_hash != computed_publication_config_hash:
                raise FormalRunRejected("STAGE3_G38_PUBLICATION_CONFIG_HASH_MISMATCH")
        if manifest.publication_config_hash is not None and manifest.publication_config_hash != computed_publication_config_hash:
            raise FormalRunRejected("STAGE3_G38_MANIFEST_PUBLICATION_CONFIG_HASH_MISMATCH")
        if publication_id is None:
            publication_id = f"stage3-g38-{computed_publication_config_hash[:24]}"
        publication_id = _safe_id(publication_id, field="publication_id")
        timestamp = checked_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            gate = GateRecord(
                gate_id=STAGE3_G38_GATE_ID,
                stage=3,
                status=GateStatus.PASS,
                checked_at=timestamp,
                measured={
                    "g3_0_through_g3_7_live_pass": True,
                    "stage3_10_formal_commit_count": 4,
                    "delivery_file_count": len(manifest.file_records()),
                    "delivery_manifest_hash": manifest_commit.identity.artifact_hash,
                    "publication_config_hash": computed_publication_config_hash,
                },
                threshold={
                    "all_prerequisite_gates_pass": True,
                    "stage3_10_formal_commits": 4,
                    "delivery_manifest_complete": True,
                    "small_file_hash_and_size_match": True,
                    "qualified_recommendation": True,
                    "formal_finalization": True,
                },
                evidence_refs=(
                    *[gate_ref_map[gate_id] for gate_id in REQUIRED_STAGE3_G38_GATE_IDS],
                    *stage_ref_map.values(),
                    execution_evidence_ref,
                    g3_7_publication_ref,
                    recommendation_ref,
                    finalization_ref,
                    manifest_ref,
                ),
            )
        except (TypeError, ValueError) as error:
            raise FormalRunRejected("STAGE3_G38_GATE_CONSTRUCTION_FAILED") from error
        gate_commit = TaskArtifactStore(root, output_dir).publish(
            task_id=task_id,
            artifact_kind=STAGE3_G38_GATE_ARTIFACT_KIND,
            config_hash=computed_publication_config_hash,
            run_intent="formal",
            payload=gate.to_dict(),
            formal_eligible=True,
            source_refs=gate.evidence_refs,
        )
        if gate_commit.commit_ref != expected_gate_ref:
            raise RuntimeError("STAGE3_G38_GATE_COMMIT_REF_DRIFT")
        source_refs = tuple(dict.fromkeys((
            *gate_ref_map.values(),
            *stage_ref_map.values(),
            execution_evidence_ref,
            g3_7_publication_ref,
            recommendation_ref,
            finalization_ref,
            manifest_ref,
        )))
        publication = Stage3G38Publication(
            publication_id=publication_id,
            task_id=task_id,
            config_hash=computed_publication_config_hash,
            input_config_hash=input_config_hash,
            input_config_hashes=input_config_hashes,
            publication_config_hash=computed_publication_config_hash,
            status="PASS",
            formal_eligible=True,
            execution_evidence_ref=execution_evidence_ref,
            execution_evidence_hash=execution.artifact_hash,
            g3_7_publication_ref=g3_7_publication_ref,
            g3_7_publication_hash=g37_publication.artifact_hash,
            gate_refs=dict(gate_ref_map),
            gate_hashes={key: gate_records[key].artifact_hash for key in REQUIRED_STAGE3_G38_GATE_IDS},
            stage3_10_refs=dict(stage_ref_map),
            stage3_10_hashes={key: loaded.identity.artifact_hash for key, loaded in stage_loaded.items()},
            recommendation_ref=recommendation_ref,
            recommendation_hash=recommendation_hash,
            finalization_ref=finalization_ref,
            finalization_hash=finalization_hash,
            delivery_manifest_ref=manifest_ref,
            delivery_manifest_hash=manifest_commit.identity.artifact_hash,
            g3_8_ref=gate_commit.commit_ref,
            # The nested GateRecord hash is the semantic gate identity (the
            # task envelope has a separate artifact hash used only to reload
            # the commit).
            g3_8_hash=gate.artifact_hash,
            g3_8_gate=gate,
            source_artifact_refs=source_refs,
        )
        receipt_commit = TaskArtifactStore(root, output_dir).publish(
            task_id=task_id,
            artifact_kind=STAGE3_G38_RECEIPT_ARTIFACT_KIND,
            config_hash=computed_publication_config_hash,
            run_intent="formal",
            payload=publication.to_dict(),
            formal_eligible=True,
            source_refs=tuple((*source_refs, gate_commit.commit_ref)),
        )
        if receipt_commit.commit_ref != expected_receipt_ref:
            raise RuntimeError("STAGE3_G38_RECEIPT_COMMIT_REF_DRIFT")
        return publication


def publish_stage3_g38(**kwargs: object) -> Stage3G38Publication:
    """Functional wrapper for :class:`Stage3G38Publisher`."""

    return Stage3G38Publisher().publish(**kwargs)  # type: ignore[arg-type]


def publish_stage3_delivery_manifest(
    *,
    workspace_root: str | Path,
    output_dir: str,
    config_hash: str,
    manifest: Mapping[str, object],
    stage3_10_refs: Mapping[str, str],
    source_refs: Sequence[str] = (),
) -> object:
    """Publish the independent S3.10 delivery-manifest authority.

    The producer only accepts a completed, hash-bound file inventory.  It does
    not create missing report files and requires all four S3.10 commits to be
    present in the immutable source reference set, leaving G3-8 to consume this
    authority independently.
    """

    if not isinstance(stage3_10_refs, Mapping):
        raise FormalRunRejected("STAGE3_G38_STAGE310_REFS_REQUIRED")
    normalized_refs = {
        str(key): _safe_ref(value, field=f"stage3_10_refs.{key}")
        for key, value in stage3_10_refs.items()
    }
    if set(normalized_refs) != set(REQUIRED_STAGE3_G38_STAGE310_KINDS):
        raise FormalRunRejected("STAGE3_G38_REQUIRES_EXACT_CANONICAL_STAGE3_10_COMMITS")
    refs = tuple(_safe_ref(ref, field="source_refs") for ref in source_refs)
    if not set(normalized_refs.values()).issubset(set(refs)):
        raise FormalRunRejected("STAGE3_G38_MANIFEST_MISSING_STAGE310_SOURCES")
    root = Path(workspace_root).resolve()
    parsed = Stage3G38DeliveryManifest.from_mapping(manifest)
    _verify_manifest_files(root, parsed)
    validate_stage3_replay_reports(root, parsed)
    validate_stage3_git_sync_evidence(root, parsed)
    validate_stage3_large_artifact_manifest(root, parsed)
    published = TaskArtifactStore(root, output_dir).publish(
        task_id=STAGE3_G38_MANIFEST_TASK_ID,
        artifact_kind=STAGE3_G38_MANIFEST_ARTIFACT_KIND,
        config_hash=config_hash,
        run_intent="formal",
        payload=parsed.to_dict(),
        formal_eligible=True,
        source_refs=tuple(dict.fromkeys((*refs, *normalized_refs.values()))),
    )
    return published


__all__ = [
    "REQUIRED_STAGE3_G38_DELIVERY_ROLES",
    "REQUIRED_STAGE3_G38_GATE_IDS",
    "REQUIRED_STAGE3_G38_GIT_ROLES",
    "REQUIRED_STAGE3_G38_AGENT_DOCUMENTS",
    "REQUIRED_STAGE3_G38_LARGE_ARTIFACT_ROLES",
    "REQUIRED_STAGE3_G38_REPLAY_LAYERS",
    "STAGE3_G38_DELIVERY_MANIFEST_SCHEMA",
    "STAGE3_G38_DELIVERY_RECEIPT_ARTIFACT_KIND",
    "STAGE3_G38_GATE_ARTIFACT_KIND",
    "STAGE3_G38_GATE_ID",
    "STAGE3_G38_PUBLICATION_CONFIG_SCHEMA",
    "STAGE3_G38_PUBLICATION_SCHEMA",
    "STAGE3_G38_GIT_SYNC_EVIDENCE_SCHEMA",
    "STAGE3_G38_LARGE_ARTIFACT_MANIFEST_SCHEMA",
    "STAGE3_G38_REPLAY_REPORT_SCHEMA",
    "STAGE3_G38_RECEIPT_ARTIFACT_KIND",
    "STAGE3_G38_TASK_ID",
    "Stage3G38DeliveryManifest",
    "Stage3G38Publication",
    "Stage3G38Publisher",
    "publish_stage3_delivery_manifest",
    "publish_stage3_g38",
    "validate_stage3_g38_handoff_authority",
    "validate_stage3_git_sync_evidence",
    "validate_stage3_large_artifact_manifest",
    "validate_stage3_replay_reports",
]
