"""Receive one frozen G3 object from the approved lab-pc SSH relay.

The command never accepts a URL.  It resolves a stable object ID through the
tracked download plan and object specification, emits a URL-free resume
handshake, and delegates all writes to the shared locked acquisition primitive.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys
from typing import Any

from param_importance_nlp.asset_acquisition import (
    AcquisitionPolicy,
    AssetAcquisitionError,
    AssetObjectSpec,
    StreamReceptionPlan,
    receive_streamed_asset,
    resolve_approved_asset_target,
)
from param_importance_nlp.asset_download_plan import load_g3_download_plan
from param_importance_nlp.asset_layout import load_stage0_asset_layout
from param_importance_nlp.asset_requirements import load_stage0_asset_requirements
from param_importance_nlp.assets import validate_asset_path
from param_importance_nlp.contracts import (
    canonical_json_bytes,
    ensure_json_object,
    load_canonical_json,
)
from param_importance_nlp.storage import is_within, require_data_root


PROTOCOL_VERSION = "stage0-g3-ssh-stream-v1"


class G3StreamReceiverError(ValueError):
    """Raised when a relay request is outside the frozen G3 control plane."""


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    return parser


def _source_file(source_root: Path, value: str | Path, *, field: str) -> Path:
    supplied = Path(value)
    target = supplied if supplied.is_absolute() else source_root / supplied
    if _is_link_like(target):
        raise G3StreamReceiverError(f"{field} may not be link-like")
    try:
        resolved = target.resolve(strict=True)
    except OSError as error:
        raise G3StreamReceiverError(f"{field} is missing") from error
    if not resolved.is_file() or not is_within(resolved, source_root):
        raise G3StreamReceiverError(f"{field} is outside the tracked source root")
    relative = resolved.relative_to(source_root)
    current = source_root
    for part in relative.parts:
        current = current / part
        if _is_link_like(current):
            raise G3StreamReceiverError(f"{field} traverses a link")
    return resolved


def _asset_root(data_root: Path, reference: str) -> Path:
    relative = PurePosixPath(validate_asset_path(reference))
    target = data_root.joinpath(*relative.parts)
    if not is_within(target, data_root):
        raise G3StreamReceiverError("asset root escapes DATA_ROOT")
    current = data_root
    for part in relative.parts:
        current = current / part
        if current.exists() and (_is_link_like(current) or not current.is_dir()):
            raise G3StreamReceiverError("asset root has an unsafe path component")
    if not target.parent.is_dir() or _is_link_like(target.parent):
        raise G3StreamReceiverError("asset root parent is missing or link-like")
    target.mkdir(mode=0o750, exist_ok=True)
    if _is_link_like(target) or not target.is_dir():
        raise G3StreamReceiverError("asset root is not a real directory")
    return target


def _emit(value: dict[str, Any], *, stream: Any) -> None:
    payload = canonical_json_bytes(value).decode("utf-8")
    stream.write(payload if payload.endswith("\n") else payload + "\n")
    stream.flush()


def _resolve_request(arguments: argparse.Namespace) -> tuple[AssetObjectSpec, Path]:
    try:
        source_root = arguments.source_root.resolve(strict=True)
    except OSError as error:
        raise G3StreamReceiverError("source root is missing") from error
    if not source_root.is_dir() or _is_link_like(source_root):
        raise G3StreamReceiverError("source root must be a real directory")
    requirements_path = _source_file(
        source_root, arguments.requirements, field="requirements"
    )
    layout_path = _source_file(source_root, arguments.layout, field="layout")
    plan_path = _source_file(source_root, arguments.plan, field="plan")
    requirements = load_stage0_asset_requirements(requirements_path)
    layout = load_stage0_asset_layout(layout_path, requirements=requirements)
    plan = load_g3_download_plan(
        plan_path,
        requirements=requirements,
        layout=layout,
    )
    matches = [
        dict(entry)
        for entry in plan["entries"]
        if entry["object_id"] == arguments.object_id
    ]
    if len(matches) != 1:
        raise G3StreamReceiverError("object_id is not unique in the frozen plan")
    entry = matches[0]
    spec_path = _source_file(source_root, entry["spec_ref"], field="object spec")
    spec_value = ensure_json_object(
        load_canonical_json(spec_path), field="object spec"
    )
    spec = AssetObjectSpec.from_mapping(spec_value)
    if spec.source_id != entry["object_id"]:
        raise G3StreamReceiverError("object spec does not match object_id")
    data_root = require_data_root(arguments.data_root)
    root = _asset_root(data_root, entry["asset_root_ref"])
    target = resolve_approved_asset_target(root, entry["final_path"])
    return spec, target


def main() -> int:
    arguments = _parser().parse_args()
    try:
        spec, target = _resolve_request(arguments)

        def ready(plan: StreamReceptionPlan) -> None:
            _emit(
                {
                    "schema_version": PROTOCOL_VERSION,
                    "phase": "READY",
                    "object_id": arguments.object_id,
                    "reception": plan.to_dict(),
                    "runtime_urls_persisted": False,
                },
                stream=sys.stdout,
            )

        result = receive_streamed_asset(
            spec,
            target,
            sys.stdin.buffer,
            on_ready=ready,
            policy=AcquisitionPolicy(
                max_attempts=1,
                request_timeout_seconds=60.0,
                overall_timeout_seconds=6 * 60 * 60,
                initial_backoff_seconds=0.0,
                max_backoff_seconds=0.0,
                lock_timeout_seconds=60.0,
                lock_poll_interval_seconds=0.1,
                chunk_size=4 * 1024 * 1024,
            ),
        )
        _emit(
            {
                "schema_version": PROTOCOL_VERSION,
                "phase": "COMPLETE",
                "object_id": arguments.object_id,
                "result": result.to_dict(),
                "runtime_urls_persisted": False,
            },
            stream=sys.stdout,
        )
        return 0
    except AssetAcquisitionError as error:
        _emit(
            {
                "schema_version": PROTOCOL_VERSION,
                "phase": "FAILED",
                "object_id": arguments.object_id,
                "failure": error.report.to_dict(),
                "runtime_urls_persisted": False,
            },
            stream=sys.stderr,
        )
        return 2
    except (G3StreamReceiverError, OSError, TypeError, ValueError) as error:
        _emit(
            {
                "schema_version": PROTOCOL_VERSION,
                "phase": "REJECTED",
                "object_id": arguments.object_id,
                "error_type": type(error).__name__,
                "runtime_urls_persisted": False,
            },
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
