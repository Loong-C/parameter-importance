"""Stream one frozen G3 object from an approved control host into sophgo13.

This script is intentionally self-contained so the local dispatcher can feed
it to ``python -`` on lab-pc or run it directly on the local workstation.  It
derives the immutable Hugging Face endpoint only in process memory, never
writes asset bytes on the relay host, and gives the server receiver sole
ownership of locks, resume state, hashing, and publish.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROTOCOL_VERSION = "stage0-g3-ssh-stream-v1"
BINDING_SCHEMA_VERSION = "stage0-g3-relay-binding-v1"
SERVER_ALIAS = "sophgo13"
LOCAL_SERVER_ALIAS = "sophgo13-via-lab"
APPROVED_SERVER_ALIASES = (SERVER_ALIAS, LOCAL_SERVER_ALIAS)
ENDPOINT_PROFILES = ("official", "hf-mirror")
_ENDPOINT_ORIGINS = {
    "official": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}
SERVER_REPO = "/home/sophgo13/cjl/parameter-importance"
SERVER_DATA_ROOT = "/home/sophgo13/cjl/storage/parameter-importance"
SERVER_PYTHON = f"{SERVER_DATA_ROOT}/envs/parameter-importance/bin/python"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_SAFE_FAILURE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_SUCCESS_STATUSES = frozenset(
    {"downloaded", "already_ready", "published_by_peer"}
)
_BINDING_ARGUMENTS = (
    "requirements_sha256",
    "layout_sha256",
    "plan_sha256",
    "spec_ref",
    "spec_sha256",
    "asset_root_ref",
    "final_path",
    "generator_git_commit",
    "source_git_commit",
    "route",
)


class RelayError(RuntimeError):
    """A redacted relay failure that never includes a runtime URL."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--requirements-sha256", required=True)
    parser.add_argument("--layout-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--spec-ref", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--asset-root-ref", required=True)
    parser.add_argument("--final-path", required=True)
    parser.add_argument("--generator-git-commit", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument(
        "--route",
        choices=("lab-direct", "local-via-lab"),
        required=True,
    )
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--overall-timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument(
        "--endpoint-profile",
        choices=ENDPOINT_PROFILES,
        default="official",
        help="Named in-memory endpoint origin; runtime URLs are never persisted.",
    )
    parser.add_argument(
        "--server-alias",
        choices=APPROVED_SERVER_ALIASES,
        default=SERVER_ALIAS,
    )
    return parser


def _validate(arguments: argparse.Namespace) -> None:
    if (
        not isinstance(arguments.object_id, str)
        or not arguments.object_id.startswith("huggingface/")
        or "://" in arguments.object_id
        or "?" in arguments.object_id
        or arguments.object_id != arguments.object_id.strip()
    ):
        raise RelayError("OBJECT_ID_INVALID")
    parts = arguments.object_id[len("huggingface/") :].split("/")
    if len(parts) < 3 or any(not part for part in parts):
        raise RelayError("OBJECT_ID_INVALID")
    if _REVISION.fullmatch(arguments.revision) is None:
        raise RelayError("REVISION_INVALID")
    if (
        isinstance(arguments.expected_size, bool)
        or arguments.expected_size < 0
        or _SHA256.fullmatch(arguments.expected_sha256) is None
    ):
        raise RelayError("EXPECTED_IDENTITY_INVALID")
    if (
        isinstance(arguments.max_attempts, bool)
        or arguments.max_attempts < 1
        or isinstance(arguments.chunk_size, bool)
        or arguments.chunk_size < 1
    ):
        raise RelayError("POLICY_INVALID")
    for value in (
        arguments.request_timeout_seconds,
        arguments.overall_timeout_seconds,
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise RelayError("POLICY_INVALID")
    for field in (
        "requirements_sha256",
        "layout_sha256",
        "plan_sha256",
        "spec_sha256",
    ):
        if _SHA256.fullmatch(getattr(arguments, field, "")) is None:
            raise RelayError("BINDING_INVALID")
    for field in ("generator_git_commit", "source_git_commit"):
        if _REVISION.fullmatch(getattr(arguments, field, "")) is None:
            raise RelayError("BINDING_INVALID")
    for field in ("spec_ref", "asset_root_ref", "final_path"):
        value = getattr(arguments, field, "")
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or value.startswith("/")
            or "\\" in value
            or ":" in value
            or "?" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise RelayError("BINDING_INVALID")
    route = getattr(arguments, "route", "")
    expected_alias = {
        "lab-direct": SERVER_ALIAS,
        "local-via-lab": LOCAL_SERVER_ALIAS,
    }.get(route)
    if expected_alias is None or expected_alias != getattr(
        arguments,
        "server_alias",
        None,
    ):
        raise RelayError("ROUTE_ALIAS_MISMATCH")
    if getattr(arguments, "endpoint_profile", "official") not in ENDPOINT_PROFILES:
        raise RelayError("ENDPOINT_PROFILE_INVALID")


def _binding(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "object_id": arguments.object_id,
        "revision": arguments.revision,
        "expected_size": arguments.expected_size,
        "expected_sha256": arguments.expected_sha256,
        **{field: getattr(arguments, field) for field in _BINDING_ARGUMENTS},
    }


def _runtime_url(
    object_id: str,
    revision: str,
    *,
    endpoint_profile: str = "official",
) -> str:
    try:
        origin = _ENDPOINT_ORIGINS[endpoint_profile]
    except KeyError as error:
        raise RelayError("ENDPOINT_PROFILE_INVALID") from error
    owner, repository, *path_parts = object_id[len("huggingface/") :].split("/")
    encoded_path = "/".join(quote(part, safe="") for part in path_parts)
    return (
        f"{origin}/{quote(owner, safe='')}/"
        f"{quote(repository, safe='')}/resolve/{quote(revision, safe='')}/"
        f"{encoded_path}"
    )


def _receiver_command(
    arguments: argparse.Namespace,
) -> list[str]:
    server_alias = arguments.server_alias
    if server_alias not in APPROVED_SERVER_ALIASES:
        raise RelayError("SERVER_ALIAS_INVALID")
    remote = shlex.join(
        [
            "env",
            f"PYTHONPATH={SERVER_REPO}/src:{SERVER_REPO}",
            SERVER_PYTHON,
            "-m",
            "ops.stage0.receive_g3_asset_stream",
            "--source-root",
            SERVER_REPO,
            "--data-root",
            SERVER_DATA_ROOT,
            "--requirements",
            "configs/stage0/g3-asset-requirements-v1.json",
            "--layout",
            "configs/stage0/g3-asset-layout-v1.json",
            "--plan",
            "configs/stage0/g3-download-plan-v1.json",
            "--object-id",
            arguments.object_id,
            "--revision",
            arguments.revision,
            "--expected-size",
            str(arguments.expected_size),
            "--expected-sha256",
            arguments.expected_sha256,
            "--requirements-sha256",
            arguments.requirements_sha256,
            "--layout-sha256",
            arguments.layout_sha256,
            "--plan-sha256",
            arguments.plan_sha256,
            "--spec-ref",
            arguments.spec_ref,
            "--spec-sha256",
            arguments.spec_sha256,
            "--asset-root-ref",
            arguments.asset_root_ref,
            "--final-path",
            arguments.final_path,
            "--generator-git-commit",
            arguments.generator_git_commit,
            "--source-git-commit",
            arguments.source_git_commit,
            "--route",
            arguments.route,
            "--overall-timeout-seconds",
            str(arguments.overall_timeout_seconds),
        ]
    )
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=180",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=12",
        server_alias,
        remote,
    ]


def _bounded_call(
    operation: Callable[[], Any],
    *,
    deadline: float,
    timeout_code: str = "OVERALL_TIMEOUT",
) -> Any:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RelayError(timeout_code)
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcome.put((True, operation()))
        except BaseException as error:
            outcome.put((False, error))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        succeeded, value = outcome.get(timeout=remaining)
    except queue.Empty:
        raise RelayError(timeout_code) from None
    if not succeeded:
        raise value
    return value


def _protocol_line(
    stream: BinaryIO,
    *,
    expected_phase: str,
    deadline: float,
) -> dict[str, Any]:
    line = _bounded_call(stream.readline, deadline=deadline)
    if not line:
        raise RelayError(f"RECEIVER_{expected_phase}_MISSING")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelayError(f"RECEIVER_{expected_phase}_INVALID") from error
    if not isinstance(value, dict) or value.get("schema_version") != PROTOCOL_VERSION:
        raise RelayError(f"RECEIVER_{expected_phase}_INVALID")
    if value.get("runtime_urls_persisted") is not False:
        raise RelayError(f"RECEIVER_{expected_phase}_INVALID")
    phase = value.get("phase")
    if phase == "FAILED":
        allowed_failure_fields = (
            frozenset(
                {
                    "schema_version",
                    "phase",
                    "object_id",
                    "failure",
                    "runtime_urls_persisted",
                }
            ),
            frozenset(
                {
                    "schema_version",
                    "phase",
                    "object_id",
                    "binding",
                    "failure",
                    "runtime_urls_persisted",
                }
            ),
        )
        if frozenset(value) not in allowed_failure_fields:
            raise RelayError("RECEIVER_FAILED_INVALID")
        failure = value.get("failure")
        code = failure.get("code") if isinstance(failure, dict) else None
        if (
            not isinstance(failure, dict)
            or failure.get("status") != "failed"
            or not isinstance(code, str)
            or _SAFE_FAILURE_CODE.fullmatch(code) is None
        ):
            raise RelayError("RECEIVER_FAILED_INVALID")
        raise RelayError(f"RECEIVER_FAILED_{code}")
    if phase != expected_phase:
        raise RelayError(f"RECEIVER_{expected_phase}_INVALID")
    return value


def _validated_offset(
    handshake: dict[str, Any], arguments: argparse.Namespace
) -> tuple[int, bool]:
    if set(handshake) != {
        "schema_version",
        "phase",
        "object_id",
        "binding",
        "reception",
        "runtime_urls_persisted",
    }:
        raise RelayError("RECEIVER_HANDSHAKE_INVALID")
    if handshake.get("object_id") != arguments.object_id:
        raise RelayError("RECEIVER_OBJECT_ID_MISMATCH")
    if handshake.get("binding") != _binding(arguments):
        raise RelayError("RECEIVER_BINDING_MISMATCH")
    reception = handshake.get("reception")
    if not isinstance(reception, dict):
        raise RelayError("RECEIVER_HANDSHAKE_INVALID")
    expected = {
        "schema_version": "stage0-asset-stream-reception-plan-v1",
        "source_id": arguments.object_id,
        "revision": arguments.revision,
        "expected_size": arguments.expected_size,
        "expected_sha256": arguments.expected_sha256,
    }
    if any(reception.get(key) != value for key, value in expected.items()):
        raise RelayError("RECEIVER_IDENTITY_MISMATCH")
    if set(reception) != {*expected, "offset", "already_ready"}:
        raise RelayError("RECEIVER_HANDSHAKE_INVALID")
    offset = reception.get("offset")
    already_ready = reception.get("already_ready")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset <= arguments.expected_size
        or not isinstance(already_ready, bool)
        or (already_ready and offset != arguments.expected_size)
    ):
        raise RelayError("RECEIVER_OFFSET_INVALID")
    return offset, already_ready


def _validate_response(response: Any, *, offset: int, expected_size: int) -> int:
    status = int(getattr(response, "status", response.getcode()))
    raw_length = response.headers.get("Content-Length")
    try:
        content_length = int(raw_length)
    except (TypeError, ValueError) as error:
        raise RelayError("HTTP_CONTENT_LENGTH_INVALID") from error
    remaining = expected_size - offset
    if content_length != remaining:
        raise RelayError("HTTP_CONTENT_LENGTH_INVALID")
    if offset == 0:
        if status != 200:
            raise RelayError("HTTP_RANGE_SEMANTICS_INVALID")
        return content_length
    raw_range = response.headers.get("Content-Range")
    match = _CONTENT_RANGE.fullmatch(raw_range.strip()) if raw_range else None
    if status != 206 or match is None:
        raise RelayError("HTTP_RANGE_SEMANTICS_INVALID")
    start, end, total = (int(value) for value in match.groups())
    if (
        start != offset
        or end != expected_size - 1
        or total != expected_size
        or end - start + 1 != content_length
    ):
        raise RelayError("HTTP_RANGE_SEMANTICS_INVALID")
    return content_length


def _close_receiver_input(
    process: subprocess.Popen[bytes],
    *,
    deadline: float | None = None,
) -> None:
    if process.stdin is not None and not process.stdin.closed:
        try:
            if deadline is None:
                process.stdin.close()
            else:
                _bounded_call(process.stdin.close, deadline=deadline)
        except (OSError, RelayError):
            pass


def _finish_receiver(
    process: subprocess.Popen[bytes],
    *,
    arguments: argparse.Namespace,
    deadline: float,
) -> dict[str, Any]:
    if process.stdout is None:
        raise RelayError("RECEIVER_STDOUT_MISSING")
    complete = _protocol_line(
        process.stdout,
        expected_phase="COMPLETE",
        deadline=deadline,
    )
    if set(complete) != {
        "schema_version",
        "phase",
        "object_id",
        "binding",
        "result",
        "runtime_urls_persisted",
    }:
        raise RelayError("RECEIVER_COMPLETE_INVALID")
    if complete.get("object_id") != arguments.object_id:
        raise RelayError("RECEIVER_OBJECT_ID_MISMATCH")
    if complete.get("binding") != _binding(arguments):
        raise RelayError("RECEIVER_BINDING_MISMATCH")
    result = complete.get("result")
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
        or result.get("source_id") != arguments.object_id
        or result.get("revision") != arguments.revision
        or result.get("size_bytes") != arguments.expected_size
        or result.get("sha256") != arguments.expected_sha256
        or isinstance(result.get("attempts"), bool)
        or not isinstance(result.get("attempts"), int)
        or result["attempts"] < 0
        or not isinstance(result.get("resumed"), bool)
        or not isinstance(result.get("network_accessed"), bool)
    ):
        raise RelayError("RECEIVER_RESULT_MISMATCH")
    return complete


def _write_receiver(
    process: subprocess.Popen[bytes],
    chunk: bytes,
    *,
    deadline: float,
) -> None:
    if process.stdin is None:
        raise RelayError("RECEIVER_PIPE_UNAVAILABLE")

    def write() -> None:
        assert process.stdin is not None
        process.stdin.write(chunk)
        process.stdin.flush()

    try:
        _bounded_call(write, deadline=deadline)
    except RelayError:
        raise
    except (BrokenPipeError, OSError) as error:
        raise RelayError("RECEIVER_PIPE_CLOSED") from error


def _stream_response(
    response: Any,
    process: subprocess.Popen[bytes],
    *,
    arguments: argparse.Namespace,
    expected_bytes: int,
    deadline: float,
) -> int:
    received = 0
    progress_interval = max(256 * 1024 * 1024, arguments.chunk_size)
    next_progress = min(progress_interval, expected_bytes)
    while received < expected_bytes:
        read_size = min(arguments.chunk_size, expected_bytes - received)
        try:
            chunk = _bounded_call(
                lambda: response.read(read_size),
                deadline=deadline,
            )
        except http.client.IncompleteRead as error:
            partial = bytes(error.partial or b"")
            if len(partial) > read_size:
                raise RelayError("HTTP_TRANSFER_EXCESS") from None
            if partial:
                _write_receiver(process, partial, deadline=deadline)
                received += len(partial)
            raise RelayError("HTTP_TRANSFER_INCOMPLETE") from None
        except RelayError:
            raise
        except (TimeoutError, OSError) as error:
            raise RelayError("HTTP_NETWORK_ERROR") from error
        if not chunk:
            raise RelayError("HTTP_TRANSFER_INCOMPLETE")
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise RelayError("HTTP_RESPONSE_INVALID")
        payload = bytes(chunk)
        if len(payload) > read_size:
            raise RelayError("HTTP_TRANSFER_EXCESS")
        _write_receiver(process, payload, deadline=deadline)
        received += len(payload)
        if received >= next_progress or received == expected_bytes:
            print(
                f"object_id={arguments.object_id} phase=STREAM "
                f"received_bytes={received} expected_bytes={expected_bytes}",
                file=sys.stderr,
                flush=True,
            )
            next_progress = min(expected_bytes, next_progress + progress_interval)
    try:
        excess = _bounded_call(lambda: response.read(1), deadline=deadline)
    except http.client.IncompleteRead as error:
        excess = error.partial
    except RelayError:
        raise
    except (TimeoutError, OSError) as error:
        raise RelayError("HTTP_NETWORK_ERROR") from error
    if excess:
        raise RelayError("HTTP_TRANSFER_EXCESS")
    return received


def _wait_receiver(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RelayError("OVERALL_TIMEOUT")
    try:
        return process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        raise RelayError("OVERALL_TIMEOUT") from None


def _emit(value: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()


def _relay_once(
    arguments: argparse.Namespace,
    *,
    deadline: float,
) -> dict[str, Any]:
    process = subprocess.Popen(
        _receiver_command(arguments),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    try:
        if process.stdin is None or process.stdout is None:
            raise RelayError("RECEIVER_PIPE_UNAVAILABLE")
        handshake = _protocol_line(
            process.stdout,
            expected_phase="READY",
            deadline=deadline,
        )
        offset, already_ready = _validated_offset(handshake, arguments)
        print(
            f"object_id={arguments.object_id} phase=READY "
            f"offset={offset} already_ready={str(already_ready).lower()}",
            file=sys.stderr,
            flush=True,
        )
        if not already_ready and offset < arguments.expected_size:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise RelayError("OVERALL_TIMEOUT")
            headers = {
                "Accept-Encoding": "identity",
                "User-Agent": "param-importance-lab-stream-relay/1",
            }
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(
                _runtime_url(
                    arguments.object_id,
                    arguments.revision,
                    endpoint_profile=getattr(
                        arguments,
                        "endpoint_profile",
                        "official",
                    ),
                ),
                headers=headers,
                method="GET",
            )
            timeout = min(arguments.request_timeout_seconds, remaining_time)
            try:
                response = _bounded_call(
                    lambda: urlopen(request, timeout=timeout),
                    deadline=deadline,
                )
            except HTTPError as error:
                error.close()
                raise RelayError(f"HTTP_STATUS_{error.code}") from None
            except (URLError, TimeoutError, OSError) as error:
                raise RelayError("HTTP_NETWORK_ERROR") from error
            with response:
                expected_bytes = _validate_response(
                    response,
                    offset=offset,
                    expected_size=arguments.expected_size,
                )
                _stream_response(
                    response,
                    process,
                    arguments=arguments,
                    expected_bytes=expected_bytes,
                    deadline=deadline,
                )
        _close_receiver_input(process, deadline=deadline)
        complete = _finish_receiver(
            process,
            arguments=arguments,
            deadline=deadline,
        )
        return_code = _wait_receiver(process, deadline=deadline)
        if return_code != 0:
            raise RelayError("RECEIVER_EXIT_NONZERO")
        assert process.stdout is not None
        if process.stdout.read(1):
            raise RelayError("RECEIVER_TRAILING_OUTPUT")
        return complete
    except Exception:
        _close_receiver_input(process, deadline=time.monotonic() + 1.0)
        try:
            remaining = max(0.0, deadline - time.monotonic())
            process.wait(timeout=min(remaining, 5.0))
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()


def main() -> int:
    arguments = _parser().parse_args()
    try:
        _validate(arguments)
    except RelayError as error:
        _emit(
            {
                "schema_version": PROTOCOL_VERSION,
                "phase": "FAILED",
                "object_id": getattr(arguments, "object_id", "invalid"),
                "failure": {"status": "failed", "code": error.code},
                "runtime_urls_persisted": False,
            }
        )
        return 2
    deadline = time.monotonic() + arguments.overall_timeout_seconds
    for attempt in range(1, arguments.max_attempts + 1):
        try:
            complete = _relay_once(arguments, deadline=deadline)
            result = complete["result"]
            _emit(
                {
                    "schema_version": PROTOCOL_VERSION,
                    "phase": "COMPLETE",
                    "object_id": arguments.object_id,
                    "binding": _binding(arguments),
                    "result": result,
                    "runtime_urls_persisted": False,
                }
            )
            return 0
        except (RelayError, OSError, subprocess.SubprocessError) as error:
            code = error.code if isinstance(error, RelayError) else type(error).__name__
            print(
                f"object_id={arguments.object_id} phase=RETRY "
                f"attempt={attempt} code={code}",
                file=sys.stderr,
                flush=True,
            )
            if attempt >= arguments.max_attempts or time.monotonic() >= deadline:
                failure_code = re.sub(r"[^A-Z0-9_]", "_", code.upper())[:128]
                _emit(
                    {
                        "schema_version": PROTOCOL_VERSION,
                        "phase": "FAILED",
                        "object_id": arguments.object_id,
                        "binding": _binding(arguments),
                        "failure": {
                            "status": "failed",
                            "code": failure_code,
                            "attempts": attempt,
                        },
                        "runtime_urls_persisted": False,
                    }
                )
                return 2
            time.sleep(
                min(
                    2 ** (attempt - 1),
                    30.0,
                    max(0.0, deadline - time.monotonic()),
                )
            )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
