"""Dispatch the URL-free G3 stream relay through the documented SSH aliases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Any

from param_importance_nlp.asset_acquisition import AssetObjectSpec
from param_importance_nlp.asset_download_plan import (
    G3RelayBinding,
    build_g3_relay_binding,
    load_g3_download_plan,
    resolve_source_git_commit,
)
from param_importance_nlp.asset_layout import load_stage0_asset_layout
from param_importance_nlp.asset_requirements import load_stage0_asset_requirements
from param_importance_nlp.contracts import ensure_json_object, load_canonical_json
from param_importance_nlp.storage import is_within


LAB_ALIAS = "lab-pc"
_RELAY_SCRIPT_REF = "ops/stage0/relay_g3_object_from_lab.py"
_PROTOCOL_VERSION = "stage0-g3-ssh-stream-v1"
_SUCCESS_STATUSES = frozenset(
    {"downloaded", "already_ready", "published_by_peer"}
)


class G3RelayDispatchError(ValueError):
    """Raised before dispatch when the local freeze is incomplete or unsafe."""


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument(
        "--relay-process",
        choices=("local", "lab"),
        default="local",
        help=(
            "Run the in-memory HTTP relay locally (default) or feed it to "
            "python - on lab-pc; both routes use only documented SSH aliases."
        ),
    )
    parser.add_argument(
        "--overall-timeout-seconds",
        type=float,
        default=6 * 60 * 60,
        help="One monotonic deadline for the complete selected dispatch.",
    )
    return parser


def _source_file(source_root: Path, value: str | Path, *, field: str) -> Path:
    supplied = Path(value)
    target = supplied if supplied.is_absolute() else source_root / supplied
    if _is_link_like(target):
        raise G3RelayDispatchError(f"{field} may not be link-like")
    try:
        resolved = target.resolve(strict=True)
    except OSError as error:
        raise G3RelayDispatchError(f"{field} is missing") from error
    if not resolved.is_file() or not is_within(resolved, source_root):
        raise G3RelayDispatchError(f"{field} is outside source root")
    relative = resolved.relative_to(source_root)
    current = source_root
    for part in relative.parts:
        current = current / part
        if _is_link_like(current):
            raise G3RelayDispatchError(f"{field} traverses a link")
    return resolved


def _load_specs(
    arguments: argparse.Namespace,
    *,
    git_timeout_seconds: float = 10.0,
) -> tuple[Path, bytes, list[tuple[AssetObjectSpec, G3RelayBinding]]]:
    try:
        source_root = arguments.source_root.resolve(strict=True)
    except OSError as error:
        raise G3RelayDispatchError("source root is missing") from error
    if not source_root.is_dir() or _is_link_like(source_root):
        raise G3RelayDispatchError("source root must be a real directory")
    requirements_path = _source_file(
        source_root, arguments.requirements, field="requirements"
    )
    layout_path = _source_file(source_root, arguments.layout, field="layout")
    plan_path = _source_file(source_root, arguments.plan, field="plan")
    relay_path = _source_file(source_root, _RELAY_SCRIPT_REF, field="relay script")
    requirements = load_stage0_asset_requirements(requirements_path)
    layout = load_stage0_asset_layout(layout_path, requirements=requirements)
    plan = load_g3_download_plan(
        plan_path,
        requirements=requirements,
        layout=layout,
    )
    requirements_ref = requirements_path.relative_to(source_root).as_posix()
    if requirements_ref != plan["requirements_ref"]:
        raise G3RelayDispatchError("requirements path does not match the plan")
    if layout_path.relative_to(source_root).as_posix() != plan["layout_ref"]:
        raise G3RelayDispatchError("layout path does not match the plan")
    try:
        source_git_commit = resolve_source_git_commit(
            source_root,
            timeout_seconds=git_timeout_seconds,
        )
    except ValueError as error:
        raise G3RelayDispatchError("source Git identity is unavailable") from error
    relay_process = getattr(arguments, "relay_process", "local")
    route = {
        "local": "local-via-lab",
        "lab": "lab-direct",
    }.get(relay_process)
    if route is None:
        raise G3RelayDispatchError("relay_process is invalid")
    selected = list(arguments.object_id)
    if len(selected) != len(set(selected)):
        raise G3RelayDispatchError("object_id selections must be unique")
    known = {entry["object_id"] for entry in plan["entries"]}
    if any(value not in known for value in selected):
        raise G3RelayDispatchError("selected object_id is outside the frozen plan")
    entries = [
        dict(entry)
        for entry in plan["entries"]
        if not selected or entry["object_id"] in selected
    ]
    specs: list[tuple[AssetObjectSpec, G3RelayBinding]] = []
    for entry in entries:
        spec_path = _source_file(
            source_root,
            source_root.joinpath(*PurePosixPath(entry["spec_ref"]).parts),
            field="object spec",
        )
        value = ensure_json_object(load_canonical_json(spec_path), field="object spec")
        spec = AssetObjectSpec.from_mapping(value)
        if spec.source_id != entry["object_id"]:
            raise G3RelayDispatchError("object spec does not match the plan")
        try:
            binding = build_g3_relay_binding(
                plan=plan,
                entry=entry,
                spec=spec,
                source_git_commit=source_git_commit,
                route=route,
            )
        except ValueError as error:
            raise G3RelayDispatchError("relay binding is invalid") from error
        specs.append((spec, binding))
    if not specs:
        raise G3RelayDispatchError("relay selection is empty")
    return relay_path, relay_path.read_bytes(), specs


def _relay_arguments(
    binding: G3RelayBinding,
    *,
    overall_timeout_seconds: float,
) -> list[str]:
    return [
        "--object-id",
        binding.object_id,
        "--revision",
        binding.revision,
        "--expected-size",
        str(binding.expected_size),
        "--expected-sha256",
        binding.expected_sha256,
        "--requirements-sha256",
        binding.requirements_sha256,
        "--layout-sha256",
        binding.layout_sha256,
        "--plan-sha256",
        binding.plan_sha256,
        "--spec-ref",
        binding.spec_ref,
        "--spec-sha256",
        binding.spec_sha256,
        "--asset-root-ref",
        binding.asset_root_ref,
        "--final-path",
        binding.final_path,
        "--generator-git-commit",
        binding.generator_git_commit,
        "--source-git-commit",
        binding.source_git_commit,
        "--route",
        binding.route,
        "--overall-timeout-seconds",
        str(overall_timeout_seconds),
    ]


def _lab_command(
    binding: G3RelayBinding,
    *,
    overall_timeout_seconds: float,
) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=180",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=12",
        LAB_ALIAS,
        "python",
        "-",
        *_relay_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
        ),
    ]
    if any("://" in argument or "?" in argument for argument in command):
        raise G3RelayDispatchError("dispatcher argv may not contain runtime URLs")
    return command


def _local_command(
    relay_path: Path,
    binding: G3RelayBinding,
    *,
    overall_timeout_seconds: float,
) -> list[str]:
    command = [
        sys.executable,
        str(relay_path),
        *_relay_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
        ),
        "--server-alias",
        "sophgo13-via-lab",
    ]
    if any("://" in argument or "?" in argument for argument in command):
        raise G3RelayDispatchError("dispatcher argv may not contain runtime URLs")
    return command


def _parse_relay_result(
    payload: bytes,
    *,
    binding: G3RelayBinding,
    returncode: int,
) -> dict[str, Any]:
    lines = payload.splitlines()
    if len(lines) != 1:
        raise G3RelayDispatchError("relay emitted an invalid protocol transcript")
    try:
        value = json.loads(lines[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G3RelayDispatchError("relay emitted invalid protocol JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != _PROTOCOL_VERSION
        or value.get("runtime_urls_persisted") is not False
        or value.get("object_id") != binding.object_id
    ):
        raise G3RelayDispatchError("relay protocol envelope does not match")
    if value.get("phase") == "FAILED":
        raise G3RelayDispatchError("relay returned a structured failure")
    if value.get("phase") != "COMPLETE" or returncode != 0:
        raise G3RelayDispatchError("relay did not complete successfully")
    if set(value) != {
        "schema_version",
        "phase",
        "object_id",
        "binding",
        "result",
        "runtime_urls_persisted",
    }:
        raise G3RelayDispatchError("relay completion fields are not exact")
    if value.get("binding") != binding.to_dict():
        raise G3RelayDispatchError("relay completion binding does not match")
    result = value.get("result")
    result_fields = {
        "schema_version",
        "status",
        "source_id",
        "revision",
        "size_bytes",
        "sha256",
        "attempts",
        "resumed",
        "network_accessed",
    }
    if (
        not isinstance(result, dict)
        or set(result) != result_fields
        or result.get("schema_version") != "stage0-asset-acquisition-result-v1"
        or result.get("status") not in _SUCCESS_STATUSES
        or result.get("source_id") != binding.object_id
        or result.get("revision") != binding.revision
        or result.get("size_bytes") != binding.expected_size
        or result.get("sha256") != binding.expected_sha256
        or isinstance(result.get("attempts"), bool)
        or not isinstance(result.get("attempts"), int)
        or result["attempts"] < 0
        or not isinstance(result.get("resumed"), bool)
        or not isinstance(result.get("network_accessed"), bool)
    ):
        raise G3RelayDispatchError("relay completion result does not match")
    return result


def dispatch(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    overall_timeout_seconds = getattr(
        arguments,
        "overall_timeout_seconds",
        6 * 60 * 60,
    )
    if (
        isinstance(overall_timeout_seconds, bool)
        or not isinstance(overall_timeout_seconds, (int, float))
        or not math.isfinite(float(overall_timeout_seconds))
        or overall_timeout_seconds <= 0
    ):
        raise G3RelayDispatchError("overall timeout is invalid")
    deadline = time.monotonic() + float(overall_timeout_seconds)
    load_remaining = deadline - time.monotonic()
    if load_remaining <= 0:
        raise G3RelayDispatchError("relay dispatch deadline expired")
    relay_path, relay_source, specs = _load_specs(
        arguments,
        git_timeout_seconds=min(10.0, load_remaining),
    )
    relay_process = getattr(arguments, "relay_process", "local")
    if relay_process not in {"local", "lab"}:
        raise G3RelayDispatchError("relay_process is invalid")
    results: list[dict[str, Any]] = []
    for index, (spec, binding) in enumerate(specs, start=1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise G3RelayDispatchError("relay dispatch deadline expired")
        print(
            f"relay={index}/{len(specs)} object_id={spec.source_id} "
            f"expected_size={spec.expected_size} route={binding.route}",
            flush=True,
        )
        try:
            if relay_process == "lab":
                completed = subprocess.run(
                    _lab_command(
                        binding,
                        overall_timeout_seconds=remaining,
                    ),
                    input=relay_source,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    timeout=remaining,
                )
            else:
                completed = subprocess.run(
                    _local_command(
                        relay_path,
                        binding,
                        overall_timeout_seconds=remaining,
                    ),
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    timeout=remaining,
                )
        except subprocess.TimeoutExpired as error:
            raise G3RelayDispatchError("relay dispatch deadline expired") from error
        protocol_result = _parse_relay_result(
            completed.stdout,
            binding=binding,
            returncode=completed.returncode,
        )
        result = {
            "object_id": spec.source_id,
            "expected_size": spec.expected_size,
            "expected_sha256": spec.expected_sha256,
            "returncode": completed.returncode,
            "route": binding.route,
            "source_git_commit": binding.source_git_commit,
            "plan_sha256": binding.plan_sha256,
            "result_status": protocol_result["status"],
        }
        results.append(result)
    return results


def main() -> int:
    arguments = _parser().parse_args()
    try:
        results = dispatch(arguments)
    except (G3RelayDispatchError, OSError, RuntimeError) as error:
        print(
            f"status=FAILED error_type={type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(f"status=PASS objects={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
