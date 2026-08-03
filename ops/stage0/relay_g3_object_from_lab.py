"""Stream one frozen G3 object from an approved control host into sophgo13.

This script is intentionally self-contained so the local dispatcher can feed
it to ``python -`` on lab-pc or run it directly on the local workstation.  It
derives the immutable Hugging Face endpoint only in process memory, never
writes asset bytes on the relay host, and gives the server receiver sole
ownership of locks, resume state, hashing, and publish.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROTOCOL_VERSION = "stage0-g3-ssh-stream-v1"
SERVER_ALIAS = "sophgo13"
LOCAL_SERVER_ALIAS = "sophgo13-via-lab"
APPROVED_SERVER_ALIASES = (SERVER_ALIAS, LOCAL_SERVER_ALIAS)
SERVER_REPO = "/home/sophgo13/cjl/parameter-importance"
SERVER_DATA_ROOT = "/home/sophgo13/cjl/storage/parameter-importance"
SERVER_PYTHON = f"{SERVER_DATA_ROOT}/envs/parameter-importance/bin/python"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


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
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--overall-timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
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


def _runtime_url(object_id: str, revision: str) -> str:
    owner, repository, *path_parts = object_id[len("huggingface/") :].split("/")
    encoded_path = "/".join(quote(part, safe="") for part in path_parts)
    return (
        f"https://huggingface.co/{quote(owner, safe='')}/"
        f"{quote(repository, safe='')}/resolve/{quote(revision, safe='')}/"
        f"{encoded_path}"
    )


def _receiver_command(
    object_id: str,
    *,
    server_alias: str = SERVER_ALIAS,
) -> list[str]:
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
            object_id,
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


def _protocol_line(stream: BinaryIO, *, expected_phase: str) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise RelayError(f"RECEIVER_{expected_phase}_MISSING")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelayError(f"RECEIVER_{expected_phase}_INVALID") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PROTOCOL_VERSION
        or value.get("phase") != expected_phase
        or value.get("runtime_urls_persisted") is not False
    ):
        raise RelayError(f"RECEIVER_{expected_phase}_INVALID")
    return value


def _validated_offset(
    handshake: dict[str, Any], arguments: argparse.Namespace
) -> tuple[int, bool]:
    if handshake.get("object_id") != arguments.object_id:
        raise RelayError("RECEIVER_OBJECT_ID_MISMATCH")
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


def _close_receiver_input(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except OSError:
            pass


def _finish_receiver(
    process: subprocess.Popen[bytes],
    *,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    if process.stdout is None:
        raise RelayError("RECEIVER_STDOUT_MISSING")
    complete = _protocol_line(process.stdout, expected_phase="COMPLETE")
    if complete.get("object_id") != arguments.object_id:
        raise RelayError("RECEIVER_OBJECT_ID_MISMATCH")
    result = complete.get("result")
    if (
        not isinstance(result, dict)
        or result.get("source_id") != arguments.object_id
        or result.get("revision") != arguments.revision
        or result.get("size_bytes") != arguments.expected_size
        or result.get("sha256") != arguments.expected_sha256
    ):
        raise RelayError("RECEIVER_RESULT_MISMATCH")
    if process.stdout.read(1):
        raise RelayError("RECEIVER_TRAILING_OUTPUT")
    return complete


def _relay_once(
    arguments: argparse.Namespace,
    *,
    deadline: float,
) -> dict[str, Any]:
    process = subprocess.Popen(
        _receiver_command(
            arguments.object_id,
            server_alias=arguments.server_alias,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    try:
        if process.stdin is None or process.stdout is None:
            raise RelayError("RECEIVER_PIPE_UNAVAILABLE")
        handshake = _protocol_line(process.stdout, expected_phase="READY")
        offset, already_ready = _validated_offset(handshake, arguments)
        print(
            f"object_id={arguments.object_id} phase=READY "
            f"offset={offset} already_ready={str(already_ready).lower()}",
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
                _runtime_url(arguments.object_id, arguments.revision),
                headers=headers,
                method="GET",
            )
            timeout = min(arguments.request_timeout_seconds, remaining_time)
            try:
                response = urlopen(request, timeout=timeout)
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
                received = 0
                while received < expected_bytes:
                    if time.monotonic() >= deadline:
                        raise RelayError("OVERALL_TIMEOUT")
                    try:
                        chunk = response.read(
                            min(arguments.chunk_size, expected_bytes - received)
                        )
                    except (TimeoutError, OSError) as error:
                        raise RelayError("HTTP_NETWORK_ERROR") from error
                    if not chunk:
                        raise RelayError("HTTP_TRANSFER_INCOMPLETE")
                    try:
                        process.stdin.write(chunk)
                        process.stdin.flush()
                    except (BrokenPipeError, OSError) as error:
                        raise RelayError("RECEIVER_PIPE_CLOSED") from error
                    received += len(chunk)
                if response.read(1):
                    raise RelayError("HTTP_TRANSFER_EXCESS")
        _close_receiver_input(process)
        complete = _finish_receiver(process, arguments=arguments)
        return_code = process.wait(timeout=180)
        if return_code != 0:
            raise RelayError("RECEIVER_EXIT_NONZERO")
        return complete
    except Exception:
        _close_receiver_input(process)
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()


def main() -> int:
    arguments = _parser().parse_args()
    try:
        _validate(arguments)
    except RelayError as error:
        print(f"phase=REJECTED code={error.code}", file=sys.stderr, flush=True)
        return 2
    deadline = time.monotonic() + arguments.overall_timeout_seconds
    for attempt in range(1, arguments.max_attempts + 1):
        try:
            complete = _relay_once(arguments, deadline=deadline)
            result = complete["result"]
            print(
                f"object_id={arguments.object_id} phase=COMPLETE "
                f"status={result['status']} size_bytes={result['size_bytes']} "
                f"sha256={result['sha256']}",
                flush=True,
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
                return 2
            time.sleep(min(2 ** (attempt - 1), 30.0, max(0.0, deadline - time.monotonic())))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
