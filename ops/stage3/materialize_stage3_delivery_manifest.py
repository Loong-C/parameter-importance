"""Materialize a hash-bound Stage 3.10 delivery inventory.

The input is a path-only inventory.  Every listed file must already exist
inside the workspace; this command computes its size and SHA-256 and publishes
the canonical manifest immutably.  It never creates report or replay content.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import (
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.experiments.stage3_g38_publisher import (
    REQUIRED_STAGE3_G38_GIT_ROLES,
    REQUIRED_STAGE3_G38_REPLAY_LAYERS,
    STAGE3_G38_DELIVERY_MANIFEST_SCHEMA,
    Stage3G38DeliveryManifest,
)
from param_importance_nlp.runtime import publish_canonical_immutable


INPUT_SCHEMA = "stage3-g38-delivery-inventory-source-v1"
_ROLE_KEYS = {
    "manifest_id",
    "source_tables",
    "analysis_scripts",
    "figures",
    "chinese_report",
    "beamer",
    "replay_reports",
    "server_large_artifact_manifest",
    "git_sync",
    "worklog",
}


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    field: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{field} missing fields: {missing}")
    if unknown:
        raise ValueError(f"{field} unknown fields: {unknown}")


def _file_record(
    root: Path,
    value: object,
    *,
    field: str,
    seen: set[str],
) -> dict[str, JSONValue]:
    if isinstance(value, str):
        spec: Mapping[str, object] = {"path": value}
    else:
        spec = _mapping(value, field=field)
    _exact_keys(
        spec,
        required={"path"},
        optional={"role", "source_refs"},
        field=field,
    )
    raw_ref = spec["path"]
    if (
        not isinstance(raw_ref, str)
        or not raw_ref
        or "?" in raw_ref
        or "://" in raw_ref
        or "\\" in raw_ref
    ):
        raise ValueError(f"{field}.path must be a stable POSIX workspace ref")
    ref = PurePosixPath(raw_ref)
    if ref.is_absolute() or not ref.parts or any(part in {"", ".", ".."} for part in ref.parts):
        raise ValueError(f"{field}.path escapes the workspace")
    normalized = ref.as_posix()
    if normalized in seen:
        raise ValueError(f"duplicate delivery file: {normalized}")

    candidate = root.joinpath(*ref.parts)
    current = root
    for part in ref.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field}.path contains a symlink: {normalized}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{field}.path is not an existing workspace file") from error
    if not resolved.is_file():
        raise ValueError(f"{field}.path is not a regular file")

    payload = resolved.read_bytes()
    record: dict[str, JSONValue] = {
        "path": normalized,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    role = spec.get("role")
    if role is not None:
        if not isinstance(role, str) or not role:
            raise ValueError(f"{field}.role must be a non-empty string")
        record["role"] = role
    refs = spec.get("source_refs")
    if refs is not None:
        if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
            raise ValueError(f"{field}.source_refs must be a string list")
        record["source_refs"] = refs
    seen.add(normalized)
    return record


def _records(
    root: Path,
    value: object,
    *,
    field: str,
    seen: set[str],
) -> list[dict[str, JSONValue]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError(f"{field} must be a non-empty list")
    return [
        _file_record(root, item, field=f"{field}[{index}]", seen=seen)
        for index, item in enumerate(value)
    ]


def materialize_stage3_delivery_manifest(
    *,
    workspace_root: str | Path,
    inventory: Mapping[str, object],
    output: str | Path,
) -> Stage3G38DeliveryManifest:
    root = Path(workspace_root).resolve(strict=True)
    _exact_keys(
        inventory,
        required={"schema_version", *_ROLE_KEYS},
        optional={"publication_config_hash"},
        field="inventory",
    )
    if inventory.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("STAGE3_G38_DELIVERY_INVENTORY_SCHEMA_UNSUPPORTED")
    seen: set[str] = set()
    source_tables = _mapping(inventory["source_tables"], field="source_tables")
    _exact_keys(source_tables, required={"csv", "json"}, field="source_tables")

    figures_value = inventory["figures"]
    if (
        not isinstance(figures_value, Sequence)
        or isinstance(figures_value, (str, bytes))
        or not figures_value
    ):
        raise TypeError("figures must be a non-empty list")
    figures: list[dict[str, JSONValue]] = []
    for index, value in enumerate(figures_value):
        figure = _mapping(value, field=f"figures[{index}]")
        _exact_keys(
            figure,
            required={"id", "png", "svg"},
            optional={"source_table_refs"},
            field=f"figures[{index}]",
        )
        item: dict[str, JSONValue] = {
            "id": figure["id"],  # validated by the canonical parser
            "png": _file_record(root, figure["png"], field=f"figures[{index}].png", seen=seen),
            "svg": _file_record(root, figure["svg"], field=f"figures[{index}].svg", seen=seen),
        }
        if "source_table_refs" in figure:
            item["source_table_refs"] = figure["source_table_refs"]  # type: ignore[assignment]
        figures.append(item)

    report = _mapping(inventory["chinese_report"], field="chinese_report")
    _exact_keys(report, required={"tex", "pdf"}, field="chinese_report")
    beamer = _mapping(inventory["beamer"], field="beamer")
    _exact_keys(beamer, required={"tex", "pdf", "notes", "backups"}, field="beamer")
    replay = _mapping(inventory["replay_reports"], field="replay_reports")
    _exact_keys(replay, required=set(REQUIRED_STAGE3_G38_REPLAY_LAYERS), field="replay_reports")
    git_sync = _mapping(inventory["git_sync"], field="git_sync")
    _exact_keys(git_sync, required=set(REQUIRED_STAGE3_G38_GIT_ROLES), field="git_sync")

    body: dict[str, JSONValue] = {
        "schema_version": STAGE3_G38_DELIVERY_MANIFEST_SCHEMA,
        "manifest_id": inventory["manifest_id"],  # type: ignore[assignment]
        "scope": "formal",
        "status": "PASS",
        "formal_eligible": True,
        "publication_config_hash": inventory.get("publication_config_hash"),  # type: ignore[assignment]
        "source_tables": {
            "csv": _records(root, source_tables["csv"], field="source_tables.csv", seen=seen),
            "json": _records(root, source_tables["json"], field="source_tables.json", seen=seen),
        },
        "analysis_scripts": _records(root, inventory["analysis_scripts"], field="analysis_scripts", seen=seen),
        "figures": figures,
        "chinese_report": {
            "tex": _file_record(root, report["tex"], field="chinese_report.tex", seen=seen),
            "pdf": _file_record(root, report["pdf"], field="chinese_report.pdf", seen=seen),
        },
        "beamer": {
            "tex": _file_record(root, beamer["tex"], field="beamer.tex", seen=seen),
            "pdf": _file_record(root, beamer["pdf"], field="beamer.pdf", seen=seen),
            "notes": _records(root, beamer["notes"], field="beamer.notes", seen=seen),
            "backups": _records(root, beamer["backups"], field="beamer.backups", seen=seen),
        },
        "replay_reports": {
            role: _file_record(root, replay[role], field=f"replay_reports.{role}", seen=seen)
            for role in REQUIRED_STAGE3_G38_REPLAY_LAYERS
        },
        "server_large_artifact_manifest": _file_record(
            root,
            inventory["server_large_artifact_manifest"],
            field="server_large_artifact_manifest",
            seen=seen,
        ),
        "git_sync": {
            role: _file_record(root, git_sync[role], field=f"git_sync.{role}", seen=seen)
            for role in REQUIRED_STAGE3_G38_GIT_ROLES
        },
        "worklog": _file_record(root, inventory["worklog"], field="worklog", seen=seen),
    }
    parsed = Stage3G38DeliveryManifest.from_mapping(
        body | {"artifact_hash": canonical_json_hash(body)}
    )
    target = Path(output)
    if not target.is_absolute():
        target = root / target
    absolute_target = Path(os.path.abspath(target))
    try:
        absolute_target.relative_to(root)
    except ValueError as error:
        raise ValueError("output must stay inside workspace_root") from error
    publish_canonical_immutable(absolute_target, parsed.to_dict())
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hash existing Stage 3 delivery files into an immutable G3-8 inventory"
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    source = load_canonical_json(arguments.inventory)
    if not isinstance(source, Mapping):
        raise TypeError("inventory must be an object")
    manifest = materialize_stage3_delivery_manifest(
        workspace_root=arguments.workspace_root,
        inventory=source,
        output=arguments.output,
    )
    target = Path(arguments.output)
    if target.is_absolute():
        manifest_ref = target.resolve().relative_to(arguments.workspace_root.resolve()).as_posix()
    else:
        manifest_ref = PurePosixPath(target.as_posix()).as_posix()
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_ref": manifest_ref,
                "artifact_hash": manifest.artifact_hash,
                "file_count": len(manifest.file_records()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
