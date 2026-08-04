"""Dispatch the URL-free G3 stream relay through the documented SSH aliases."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
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
SERVER_ALIAS = "sophgo13-via-lab"
_SERVER_REPO = "/home/sophgo13/cjl/parameter-importance"
_SERVER_DATA_ROOT = "/home/sophgo13/cjl/storage/parameter-importance"
_SERVER_PYTHON = f"{_SERVER_DATA_ROOT}/envs/parameter-importance/bin/python"
_RELAY_SCRIPT_REF = "ops/stage0/relay_g3_object_from_lab.py"
_PROTOCOL_VERSION = "stage0-g3-ssh-stream-v1"
_ENDPOINT_PROFILES = ("official", "hf-mirror")
_LAB_PYTHON_PROFILES = {
    "path": "python",
    "cjl-python312": r"C:\Users\cjl\Apps\Python312\python.exe",
}
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
        choices=("local", "lab", "lab-pipe"),
        default="local",
        help=(
            "Run the duplex relay locally (default), on lab-pc, or use the "
            "Windows-safe lab-pipe fallback with a native byte pipeline."
        ),
    )
    parser.add_argument(
        "--overall-timeout-seconds",
        type=float,
        default=6 * 60 * 60,
        help="One monotonic deadline for the complete selected dispatch.",
    )
    parser.add_argument(
        "--endpoint-profile",
        choices=_ENDPOINT_PROFILES,
        default="official",
        help="Named in-memory relay endpoint; no runtime URL enters argv or evidence.",
    )
    parser.add_argument(
        "--lab-python-profile",
        choices=tuple(_LAB_PYTHON_PROFILES),
        default="path",
        help=(
            "Named lab-pc Python interpreter profile; the controlled host's "
            "user Python is selected without accepting an arbitrary command."
        ),
    )
    parser.add_argument(
        "--lab-pipe-max-attempts",
        type=int,
        default=6,
        help=(
            "Bounded native-pipe sessions per object. Every retry reacquires "
            "the locked server offset before resuming."
        ),
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
        "lab-pipe": "lab-direct",
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


def _binding_arguments(
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


def _relay_arguments(
    binding: G3RelayBinding,
    *,
    overall_timeout_seconds: float,
    endpoint_profile: str,
) -> list[str]:
    return [
        *_binding_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
        ),
        "--endpoint-profile",
        endpoint_profile,
    ]


def _lab_command(
    binding: G3RelayBinding,
    *,
    overall_timeout_seconds: float,
    endpoint_profile: str = "official",
    lab_python_profile: str = "path",
) -> list[str]:
    lab_python = _LAB_PYTHON_PROFILES.get(lab_python_profile)
    if lab_python is None:
        raise G3RelayDispatchError("lab_python_profile is invalid")
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
        lab_python,
        "-",
        *_relay_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
            endpoint_profile=endpoint_profile,
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
    endpoint_profile: str = "official",
) -> list[str]:
    command = [
        sys.executable,
        str(relay_path),
        *_relay_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
            endpoint_profile=endpoint_profile,
        ),
        "--server-alias",
        "sophgo13-via-lab",
    ]
    if any("://" in argument or "?" in argument for argument in command):
        raise G3RelayDispatchError("dispatcher argv may not contain runtime URLs")
    return command


def _ssh_options(*, no_stdin: bool = False) -> list[str]:
    options = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=180",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=12",
    ]
    if no_stdin:
        options.append("-n")
    return options


def _receiver_remote_arguments(
    binding: G3RelayBinding,
    *,
    overall_timeout_seconds: float,
    plan_only: bool = False,
    expected_offset: int | None = None,
) -> list[str]:
    command = [
        "env",
        f"PYTHONPATH={_SERVER_REPO}/src:{_SERVER_REPO}",
        _SERVER_PYTHON,
        "-m",
        "ops.stage0.receive_g3_asset_stream",
        "--source-root",
        _SERVER_REPO,
        "--data-root",
        _SERVER_DATA_ROOT,
        "--requirements",
        "configs/stage0/g3-asset-requirements-v1.json",
        "--layout",
        "configs/stage0/g3-asset-layout-v1.json",
        "--plan",
        "configs/stage0/g3-download-plan-v1.json",
        *_binding_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
        ),
    ]
    if plan_only:
        command.append("--plan-only")
    if expected_offset is not None:
        command.extend(("--expected-offset", str(expected_offset)))
    return command


def _receiver_remote_command(
    binding: G3RelayBinding,
    *,
    overall_timeout_seconds: float,
    plan_only: bool = False,
    expected_offset: int | None = None,
) -> str:
    return shlex.join(
        _receiver_remote_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
            plan_only=plan_only,
            expected_offset=expected_offset,
        )
    )


def _plan_command(
    binding: G3RelayBinding,
    *,
    overall_timeout_seconds: float,
) -> list[str]:
    return [
        *_ssh_options(no_stdin=True),
        SERVER_ALIAS,
        _receiver_remote_command(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
            plan_only=True,
        ),
    ]


def _cmd_atom(value: str) -> str:
    if (
        not value
        or any(character in value for character in "\x00\r\n&|<>^%!()")
        or "://" in value
        or "?" in value
    ):
        raise G3RelayDispatchError("lab-pipe command atom is unsafe")
    return value


def _pipe_command(
    relay_path: Path,
    binding: G3RelayBinding,
    *,
    offset: int,
    overall_timeout_seconds: float,
    endpoint_profile: str,
    lab_python_profile: str,
) -> list[str]:
    if isinstance(offset, bool) or not 0 <= offset <= binding.expected_size:
        raise G3RelayDispatchError("lab-pipe offset is invalid")
    lab_python = _LAB_PYTHON_PROFILES.get(lab_python_profile)
    if lab_python is None:
        raise G3RelayDispatchError("lab_python_profile is invalid")
    comspec = Path(os.environ.get("COMSPEC", ""))
    if not comspec.is_absolute() or comspec.name.casefold() != "cmd.exe":
        raise G3RelayDispatchError("lab-pipe requires the controlled Windows cmd.exe")
    left = [
        *_ssh_options(),
        LAB_ALIAS,
        lab_python,
        "-",
        *_relay_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
            endpoint_profile=endpoint_profile,
        ),
        "--relay-mode",
        "emit",
        "--emit-offset",
        str(offset),
        "--max-attempts",
        "1",
    ]
    right = [
        *_ssh_options(),
        SERVER_ALIAS,
        *_receiver_remote_arguments(
            binding,
            overall_timeout_seconds=overall_timeout_seconds,
            expected_offset=offset,
        ),
    ]
    atoms = [str(relay_path), *right]
    if offset < binding.expected_size:
        atoms.extend(left)
    for atom in atoms:
        _cmd_atom(atom)
    if offset == binding.expected_size:
        pipeline = " | ".join(("type NUL", subprocess.list2cmdline(right)))
    else:
        pipeline = " | ".join(
            (
                subprocess.list2cmdline(["type", str(relay_path)]),
                subprocess.list2cmdline(left),
                subprocess.list2cmdline(right),
            )
        )
    return [str(comspec), "/d", "/s", "/c", pipeline]


def _protocol_values(payload: bytes, *, count: int) -> list[dict[str, Any]]:
    lines = payload.splitlines()
    if len(lines) != count:
        raise G3RelayDispatchError("relay emitted an invalid protocol transcript")
    values: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise G3RelayDispatchError("relay emitted invalid protocol JSON") from error
        if not isinstance(value, dict):
            raise G3RelayDispatchError("relay protocol envelope does not match")
        values.append(value)
    return values


def _validate_protocol_base(value: dict[str, Any], binding: G3RelayBinding) -> None:
    if (
        value.get("schema_version") != _PROTOCOL_VERSION
        or value.get("runtime_urls_persisted") is not False
        or value.get("object_id") != binding.object_id
    ):
        raise G3RelayDispatchError("relay protocol envelope does not match")


def _validate_acquisition_result(
    result: object,
    *,
    binding: G3RelayBinding,
) -> dict[str, Any]:
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


def _validate_reception(
    reception: object,
    *,
    binding: G3RelayBinding,
) -> dict[str, Any]:
    expected = {
        "schema_version": "stage0-asset-stream-reception-plan-v1",
        "source_id": binding.object_id,
        "revision": binding.revision,
        "expected_size": binding.expected_size,
        "expected_sha256": binding.expected_sha256,
    }
    if (
        not isinstance(reception, dict)
        or set(reception) != {*expected, "offset", "already_ready"}
        or any(reception.get(key) != value for key, value in expected.items())
        or isinstance(reception.get("offset"), bool)
        or not isinstance(reception.get("offset"), int)
        or not 0 <= reception["offset"] <= binding.expected_size
        or not isinstance(reception.get("already_ready"), bool)
        or (reception["already_ready"] and reception["offset"] != binding.expected_size)
    ):
        raise G3RelayDispatchError("receiver reception plan does not match")
    return reception


def _parse_relay_result(
    payload: bytes,
    *,
    binding: G3RelayBinding,
    returncode: int,
) -> dict[str, Any]:
    value = _protocol_values(payload, count=1)[0]
    _validate_protocol_base(value, binding)
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
    return _validate_acquisition_result(value.get("result"), binding=binding)


def _parse_plan_result(
    payload: bytes,
    *,
    binding: G3RelayBinding,
    returncode: int,
) -> dict[str, Any]:
    ready, complete = _protocol_values(payload, count=2)
    for value in (ready, complete):
        _validate_protocol_base(value, binding)
        if value.get("binding") != binding.to_dict():
            raise G3RelayDispatchError("receiver plan binding does not match")
    if returncode != 0 or ready.get("phase") != "READY" or complete.get("phase") != "PLAN_COMPLETE":
        raise G3RelayDispatchError("receiver plan did not complete successfully")
    expected_fields = {
        "schema_version",
        "phase",
        "object_id",
        "binding",
        "reception",
        "runtime_urls_persisted",
    }
    if set(ready) != expected_fields or set(complete) != expected_fields:
        raise G3RelayDispatchError("receiver plan fields are not exact")
    reception = _validate_reception(ready.get("reception"), binding=binding)
    if complete.get("reception") != reception:
        raise G3RelayDispatchError("receiver plan changed before completion")
    return reception


def _parse_pipe_result(
    payload: bytes,
    *,
    binding: G3RelayBinding,
    returncode: int,
    expected_offset: int,
) -> dict[str, Any]:
    ready, complete = _protocol_values(payload, count=2)
    for value in (ready, complete):
        _validate_protocol_base(value, binding)
    if ready.get("phase") != "READY":
        raise G3RelayDispatchError("receiver pipe handshake is invalid")
    reception = _validate_reception(ready.get("reception"), binding=binding)
    if reception["offset"] != expected_offset or reception["already_ready"]:
        raise G3RelayDispatchError("receiver pipe offset changed")
    if set(ready) != {
        "schema_version", "phase", "object_id", "binding", "reception",
        "runtime_urls_persisted",
    } or ready.get("binding") != binding.to_dict():
        raise G3RelayDispatchError("receiver pipe handshake fields are not exact")
    if complete.get("phase") == "FAILED":
        raise G3RelayDispatchError("receiver pipe returned a structured failure")
    if returncode != 0 or complete.get("phase") != "COMPLETE":
        raise G3RelayDispatchError("receiver pipe did not complete successfully")
    if set(complete) != {
        "schema_version", "phase", "object_id", "binding", "result",
        "runtime_urls_persisted",
    } or complete.get("binding") != binding.to_dict():
        raise G3RelayDispatchError("receiver pipe completion fields are not exact")
    return _validate_acquisition_result(complete.get("result"), binding=binding)


def _dispatch_lab_pipe(
    relay_path: Path,
    binding: G3RelayBinding,
    *,
    deadline: float,
    endpoint_profile: str,
    lab_python_profile: str,
    max_attempts: int,
) -> tuple[dict[str, Any], int]:
    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise G3RelayDispatchError("relay dispatch deadline expired")
        try:
            plan = subprocess.run(
                _plan_command(
                    binding,
                    overall_timeout_seconds=remaining,
                ),
                check=False,
                stdout=subprocess.PIPE,
                stderr=None,
                timeout=remaining,
            )
            reception = _parse_plan_result(
                plan.stdout,
                binding=binding,
                returncode=plan.returncode,
            )
            print(
                f"object_id={binding.object_id} phase=PLAN "
                f"attempt={attempt}/{max_attempts} "
                f"offset={reception['offset']} "
                f"already_ready={str(reception['already_ready']).lower()}",
                file=sys.stderr,
                flush=True,
            )
            if reception["already_ready"]:
                return (
                    {
                        "schema_version": "stage0-asset-acquisition-result-v1",
                        "status": "already_ready",
                        "source_id": binding.object_id,
                        "revision": binding.revision,
                        "size_bytes": binding.expected_size,
                        "sha256": binding.expected_sha256,
                        "attempts": 0,
                        "resumed": False,
                        "network_accessed": False,
                    },
                    0,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise G3RelayDispatchError("relay dispatch deadline expired")
            completed = subprocess.run(
                _pipe_command(
                    relay_path,
                    binding,
                    offset=int(reception["offset"]),
                    overall_timeout_seconds=remaining,
                    endpoint_profile=endpoint_profile,
                    lab_python_profile=lab_python_profile,
                ),
                check=False,
                stdout=subprocess.PIPE,
                stderr=None,
                timeout=remaining,
            )
            return (
                _parse_pipe_result(
                    completed.stdout,
                    binding=binding,
                    returncode=completed.returncode,
                    expected_offset=int(reception["offset"]),
                ),
                completed.returncode,
            )
        except G3RelayDispatchError:
            if attempt >= max_attempts:
                raise
            print(
                f"object_id={binding.object_id} phase=RETRY "
                f"completed_attempt={attempt}/{max_attempts}",
                file=sys.stderr,
                flush=True,
            )
    raise G3RelayDispatchError("lab-pipe attempt budget exhausted")


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
    if relay_process not in {"local", "lab", "lab-pipe"}:
        raise G3RelayDispatchError("relay_process is invalid")
    endpoint_profile = getattr(arguments, "endpoint_profile", "official")
    if endpoint_profile not in _ENDPOINT_PROFILES:
        raise G3RelayDispatchError("endpoint_profile is invalid")
    lab_python_profile = getattr(arguments, "lab_python_profile", "path")
    if lab_python_profile not in _LAB_PYTHON_PROFILES:
        raise G3RelayDispatchError("lab_python_profile is invalid")
    lab_pipe_max_attempts = getattr(arguments, "lab_pipe_max_attempts", 6)
    if (
        isinstance(lab_pipe_max_attempts, bool)
        or not isinstance(lab_pipe_max_attempts, int)
        or lab_pipe_max_attempts < 1
    ):
        raise G3RelayDispatchError("lab_pipe_max_attempts is invalid")
    results: list[dict[str, Any]] = []
    for index, (spec, binding) in enumerate(specs, start=1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise G3RelayDispatchError("relay dispatch deadline expired")
        print(
            f"relay={index}/{len(specs)} object_id={spec.source_id} "
            f"expected_size={spec.expected_size} route={binding.route} "
            f"relay_process={relay_process} "
            f"endpoint_profile={endpoint_profile} "
            f"lab_python_profile={lab_python_profile}",
            flush=True,
        )
        completed_returncode = 0
        try:
            if relay_process == "lab":
                completed = subprocess.run(
                    _lab_command(
                        binding,
                        overall_timeout_seconds=remaining,
                        endpoint_profile=endpoint_profile,
                        lab_python_profile=lab_python_profile,
                    ),
                    input=relay_source,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    timeout=remaining,
                )
                protocol_result = _parse_relay_result(
                    completed.stdout,
                    binding=binding,
                    returncode=completed.returncode,
                )
                completed_returncode = completed.returncode
            elif relay_process == "lab-pipe":
                protocol_result, completed_returncode = _dispatch_lab_pipe(
                    relay_path,
                    binding,
                    deadline=deadline,
                    endpoint_profile=endpoint_profile,
                    lab_python_profile=lab_python_profile,
                    max_attempts=lab_pipe_max_attempts,
                )
            else:
                completed = subprocess.run(
                    _local_command(
                        relay_path,
                        binding,
                        overall_timeout_seconds=remaining,
                        endpoint_profile=endpoint_profile,
                    ),
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    timeout=remaining,
                )
                protocol_result = _parse_relay_result(
                    completed.stdout,
                    binding=binding,
                    returncode=completed.returncode,
                )
                completed_returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            raise G3RelayDispatchError("relay dispatch deadline expired") from error
        result = {
            "object_id": spec.source_id,
            "expected_size": spec.expected_size,
            "expected_sha256": spec.expected_sha256,
            "returncode": completed_returncode,
            "route": binding.route,
            "source_git_commit": binding.source_git_commit,
            "plan_sha256": binding.plan_sha256,
            "result_status": protocol_result["status"],
            "endpoint_profile": endpoint_profile,
            "lab_python_profile": lab_python_profile,
            "relay_process": relay_process,
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
