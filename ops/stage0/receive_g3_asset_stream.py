"""Receive one frozen G3 object from the approved lab-pc SSH relay.

The command never accepts a URL.  It resolves a stable object ID through the
tracked download plan and object specification, emits a URL-free resume
handshake, and delegates all writes to the shared locked acquisition primitive.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path, PurePosixPath
import sys
import time
from typing import Any

from param_importance_nlp.asset_acquisition import (
    AcquisitionPolicy,
    AssetAcquisitionError,
    AssetObjectSpec,
    StreamReceptionPlan,
    inspect_legacy_acquisition_state,
    migrate_legacy_acquisition_state,
    receive_streamed_asset,
    resolve_approved_asset_target,
)
from param_importance_nlp.asset_download_plan import (
    G3RelayBinding,
    build_g3_relay_binding,
    load_g3_download_plan,
    resolve_source_git_commit,
)
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


class _ReceptionPlanComplete(RuntimeError):
    """Internal control flow used to close a plan-only receiver session."""

    def __init__(self, plan: StreamReceptionPlan) -> None:
        self.plan = plan
        super().__init__("reception plan complete")


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
    parser.add_argument("--revision")
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--requirements-sha256")
    parser.add_argument("--layout-sha256")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--spec-ref")
    parser.add_argument("--spec-sha256")
    parser.add_argument("--asset-root-ref")
    parser.add_argument("--final-path")
    parser.add_argument("--generator-git-commit")
    parser.add_argument("--source-git-commit")
    parser.add_argument("--route", choices=("lab-direct", "local-via-lab"))
    parser.add_argument("--overall-timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument(
        "--expected-offset",
        type=int,
        help="Fail closed if the locked receiver offset differs from this value.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Emit the locked reception plan and exit without reading object bytes.",
    )
    parser.add_argument(
        "--legacy-state-action",
        choices=("none", "inspect", "migrate"),
        default="none",
        help=(
            "Explicitly inspect or migrate the former asset-adjacent checkpoint; "
            "normal receive never consumes legacy state."
        ),
    )
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


def _asset_root(
    data_root: Path,
    reference: str,
    *,
    create: bool,
) -> Path:
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
    if create:
        target.mkdir(mode=0o750, exist_ok=True)
    if _is_link_like(target) or not target.is_dir():
        raise G3StreamReceiverError("asset root is not a real directory")
    return target


def _emit(value: dict[str, Any], *, stream: Any) -> None:
    payload = canonical_json_bytes(value).decode("utf-8")
    stream.write(payload if payload.endswith("\n") else payload + "\n")
    stream.flush()


def _resolve_frozen_object(
    arguments: argparse.Namespace,
    *,
    route: str,
    create_asset_root: bool = True,
    deadline: float | None = None,
) -> tuple[AssetObjectSpec, Path, G3RelayBinding, Path]:
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
    root = _asset_root(
        data_root,
        entry["asset_root_ref"],
        create=create_asset_root,
    )
    target = resolve_approved_asset_target(root, entry["final_path"])
    try:
        remaining = (
            10.0
            if deadline is None
            else min(10.0, deadline - time.monotonic())
        )
        if remaining <= 0:
            raise G3StreamReceiverError("receiver deadline expired")
        source_git_commit = resolve_source_git_commit(
            source_root,
            timeout_seconds=remaining,
        )
        binding = build_g3_relay_binding(
            plan=plan,
            entry=entry,
            spec=spec,
            source_git_commit=source_git_commit,
            route=route,
        )
    except ValueError as error:
        raise G3StreamReceiverError("frozen relay binding is invalid") from error
    return spec, target, binding, data_root


def _requested_binding(arguments: argparse.Namespace) -> G3RelayBinding:
    fields = {
        "schema_version": "stage0-g3-relay-binding-v1",
        "object_id": arguments.object_id,
        "revision": arguments.revision,
        "expected_size": arguments.expected_size,
        "expected_sha256": arguments.expected_sha256,
        "requirements_sha256": arguments.requirements_sha256,
        "layout_sha256": arguments.layout_sha256,
        "plan_sha256": arguments.plan_sha256,
        "spec_ref": arguments.spec_ref,
        "spec_sha256": arguments.spec_sha256,
        "asset_root_ref": arguments.asset_root_ref,
        "final_path": arguments.final_path,
        "generator_git_commit": arguments.generator_git_commit,
        "source_git_commit": arguments.source_git_commit,
        "route": arguments.route,
    }
    try:
        return G3RelayBinding.from_mapping(fields)
    except (TypeError, ValueError) as error:
        raise G3StreamReceiverError("relay request binding is invalid") from error


def _resolve_request(
    arguments: argparse.Namespace,
    *,
    deadline: float | None = None,
) -> tuple[AssetObjectSpec, Path, G3RelayBinding, Path]:
    requested = _requested_binding(arguments)
    spec, target, frozen, data_root = _resolve_frozen_object(
        arguments,
        route=requested.route,
        deadline=deadline,
    )
    if requested != frozen:
        raise G3StreamReceiverError("relay request does not match the frozen binding")
    return spec, target, frozen, data_root


def _check_expected_offset(
    arguments: argparse.Namespace,
    plan: StreamReceptionPlan,
) -> None:
    expected = getattr(arguments, "expected_offset", None)
    if expected is None:
        return
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected < 0
        or expected > plan.expected_size
        or plan.offset != expected
    ):
        raise G3StreamReceiverError("locked reception offset does not match request")


def main() -> int:
    arguments = _parser().parse_args()
    deadline: float | None = None
    try:
        if (
            isinstance(arguments.overall_timeout_seconds, bool)
            or not math.isfinite(arguments.overall_timeout_seconds)
            or arguments.overall_timeout_seconds <= 0
        ):
            raise G3StreamReceiverError("overall timeout is invalid")
        if arguments.plan_only and arguments.legacy_state_action != "none":
            raise G3StreamReceiverError("plan-only cannot inspect legacy state")
        deadline = time.monotonic() + arguments.overall_timeout_seconds

        if arguments.legacy_state_action != "none":
            spec, target, binding, data_root = _resolve_frozen_object(
                arguments,
                route=arguments.route or "local-via-lab",
                create_asset_root=arguments.legacy_state_action == "migrate",
                deadline=deadline,
            )
            if arguments.legacy_state_action == "inspect":
                result = inspect_legacy_acquisition_state(
                    spec,
                    target,
                    data_root=data_root,
                ).to_dict()
                operation = "legacy_state_inspection"
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise G3StreamReceiverError("receiver deadline expired")
                result = migrate_legacy_acquisition_state(
                    spec,
                    target,
                    data_root=data_root,
                    policy=AcquisitionPolicy(
                        max_attempts=1,
                        request_timeout_seconds=60.0,
                        overall_timeout_seconds=remaining,
                        initial_backoff_seconds=0.0,
                        max_backoff_seconds=0.0,
                        lock_timeout_seconds=min(
                            60.0,
                            remaining,
                        ),
                        lock_poll_interval_seconds=0.1,
                        chunk_size=4 * 1024 * 1024,
                    ),
                ).to_dict()
                operation = "legacy_state_migration"
            _emit(
                {
                    "schema_version": PROTOCOL_VERSION,
                    "phase": "COMPLETE",
                    "operation": operation,
                    "object_id": arguments.object_id,
                    "binding": binding.to_dict(),
                    "result": result,
                    "runtime_urls_persisted": False,
                },
                stream=sys.stdout,
            )
            return 0

        spec, target, binding, data_root = _resolve_request(
            arguments,
            deadline=deadline,
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise G3StreamReceiverError("receiver deadline expired")

        def ready(plan: StreamReceptionPlan) -> None:
            _check_expected_offset(arguments, plan)
            _emit(
                {
                    "schema_version": PROTOCOL_VERSION,
                    "phase": "READY",
                    "object_id": arguments.object_id,
                    "binding": binding.to_dict(),
                    "reception": plan.to_dict(),
                    "runtime_urls_persisted": False,
                },
                stream=sys.stdout,
            )
            if arguments.plan_only:
                raise _ReceptionPlanComplete(plan)

        try:
            result = receive_streamed_asset(
                spec,
                target,
                sys.stdin.buffer,
                on_ready=ready,
                policy=AcquisitionPolicy(
                    max_attempts=1,
                    request_timeout_seconds=60.0,
                    overall_timeout_seconds=remaining,
                    initial_backoff_seconds=0.0,
                    max_backoff_seconds=0.0,
                    lock_timeout_seconds=60.0,
                    lock_poll_interval_seconds=0.1,
                    chunk_size=4 * 1024 * 1024,
                ),
                data_root=data_root,
            )
        except _ReceptionPlanComplete as complete:
            _emit(
                {
                    "schema_version": PROTOCOL_VERSION,
                    "phase": "PLAN_COMPLETE",
                    "object_id": arguments.object_id,
                    "binding": binding.to_dict(),
                    "reception": complete.plan.to_dict(),
                    "runtime_urls_persisted": False,
                },
                stream=sys.stdout,
            )
            return 0
        _emit(
            {
                "schema_version": PROTOCOL_VERSION,
                "phase": "COMPLETE",
                "object_id": arguments.object_id,
                "binding": binding.to_dict(),
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
                "failure": {
                    **error.report.to_dict(),
                    "code": f"ACQUISITION_{error.report.code.value}",
                },
                "runtime_urls_persisted": False,
            },
            stream=sys.stdout,
        )
        return 2
    except (G3StreamReceiverError, OSError, TypeError, ValueError) as error:
        _emit(
            {
                "schema_version": PROTOCOL_VERSION,
                "phase": "FAILED",
                "object_id": arguments.object_id,
                "failure": {
                    "status": "failed",
                    "code": "REQUEST_REJECTED",
                    "detail_type": type(error).__name__,
                },
                "runtime_urls_persisted": False,
            },
            stream=sys.stdout,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
