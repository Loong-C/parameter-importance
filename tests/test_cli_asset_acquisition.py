from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import threading
from typing import Iterator

import pytest

from param_importance_nlp.asset_acquisition import part_path_for
from param_importance_nlp.cli import main
from param_importance_nlp.contracts import canonical_json_bytes


_URL_ENV = "PARAM_IMPORTANCE_TEST_RUNTIME_URL"


@dataclass
class _HTTPState:
    payload: bytes
    forced_status: int | None = None
    ranges: list[str | None] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    guard: threading.Lock = field(default_factory=threading.Lock)


class _RangeHandler(BaseHTTPRequestHandler):
    server: "_FixtureServer"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        state = self.server.state
        range_header = self.headers.get("Range")
        with state.guard:
            state.ranges.append(range_header)
            state.targets.append(self.path)
        if state.forced_status is not None:
            self.send_response(state.forced_status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        offset = 0
        if range_header is not None:
            assert range_header.startswith("bytes=") and range_header.endswith("-")
            offset = int(range_header[6:-1])
            body = state.payload[offset:]
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {offset}-{len(state.payload) - 1}/{len(state.payload)}",
            )
        else:
            body = state.payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: _HTTPState) -> None:
        self.state = state
        super().__init__(("127.0.0.1", 0), _RangeHandler)


@contextmanager
def _serve(state: _HTTPState) -> Iterator[str]:
    server = _FixtureServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/object"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _write_spec(path: Path, payload: bytes) -> Path:
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "stage0-http-object-spec-v1",
                "source_id": "fixture-public-source",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "expected_size": len(payload),
                "expected_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    )
    return path


def _arguments(
    spec: Path,
    asset_root: Path,
    *,
    final_path: str = "models/object.bin",
    output: Path | None = None,
    stdin: bool = False,
    max_attempts: int = 2,
) -> list[str]:
    result = [
        "asset",
        "fetch-http",
        "--spec",
        str(spec),
        "--asset-root",
        str(asset_root),
        "--final-path",
        final_path,
        "--max-attempts",
        str(max_attempts),
        "--request-timeout-seconds",
        "1",
        "--overall-timeout-seconds",
        "3",
        "--initial-backoff-seconds",
        "0",
        "--max-backoff-seconds",
        "0",
        "--lock-timeout-seconds",
        "1",
        "--lock-poll-interval-seconds",
        "0.005",
        "--chunk-size",
        "64",
    ]
    result.extend(["--url-stdin"] if stdin else ["--url-env", _URL_ENV])
    if output is not None:
        result.extend(["--output", str(output)])
    return result


@pytest.mark.parametrize(
    "runtime_arguments",
    [
        ["--url", "https://example.invalid/object?token=ARGV_SECRET"],
        ["https://example.invalid/object?token=BARE_ARGV_SECRET"],
    ],
)
def test_fetch_http_rejects_runtime_url_in_argv_without_echoing_it(
    runtime_arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["asset", "fetch-http", *runtime_arguments])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.err == ""
    assert "example.invalid" not in captured.out
    assert "ARGV_SECRET" not in captured.out
    report = json.loads(captured.out)
    assert report["code"] == "RUNTIME_URL_ARGV_FORBIDDEN"


def test_fetch_http_rejects_relative_path_escape_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = b"path boundary fixture"
    root = tmp_path / "approved-assets"
    root.mkdir()
    spec = _write_spec(tmp_path / "spec.json", payload)
    state = _HTTPState(payload)
    with _serve(state) as base_url:
        monkeypatch.setenv(_URL_ENV, f"{base_url}?token=PATH_SECRET")
        exit_code = main(_arguments(spec, root, final_path="../escape.bin"))

    captured = capsys.readouterr()
    assert exit_code != 0
    assert json.loads(captured.out)["code"] == "TARGET_PATH_INVALID"
    assert "PATH_SECRET" not in captured.out + captured.err
    assert not (tmp_path / "escape.bin").exists()
    assert state.ranges == []


def test_fetch_http_success_publishes_canonical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = b"successful immutable HTTP object"
    root = tmp_path / "approved-assets"
    root.mkdir()
    spec = _write_spec(tmp_path / "spec.json", payload)
    output = tmp_path / "result.json"
    state = _HTTPState(payload)
    with _serve(state) as base_url:
        monkeypatch.setenv(_URL_ENV, f"{base_url}?token=SUCCESS_SECRET")
        exit_code = main(_arguments(spec, root, output=output))

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert report["schema_version"] == "stage0-asset-acquisition-result-v1"
    assert report["status"] == "downloaded"
    assert report["network_accessed"] is True
    assert (root / "models" / "object.bin").read_bytes() == payload
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "SUCCESS_SECRET" not in captured.out + output.read_text(encoding="utf-8")
    assert state.ranges == [None]


def test_fetch_http_resumes_existing_part_using_stdin_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = b"resume this fixed object from a validated byte boundary"
    root = tmp_path / "approved-assets"
    target = root / "models" / "object.bin"
    target.parent.mkdir(parents=True)
    prefix = payload[:11]
    part_path_for(target).write_bytes(prefix)
    spec = _write_spec(tmp_path / "spec.json", payload)
    state = _HTTPState(payload)
    with _serve(state) as base_url:
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(f"{base_url}?token=STDIN_SECRET\n"),
        )
        exit_code = main(_arguments(spec, root, stdin=True))

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["resumed"] is True
    assert target.read_bytes() == payload
    assert not part_path_for(target).exists()
    assert state.ranges == [f"bytes={len(prefix)}-"]
    assert "STDIN_SECRET" not in captured.out + captured.err


def test_fetch_http_failure_is_canonical_redacted_and_preserves_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = b"failure redaction fixture"
    root = tmp_path / "approved-assets"
    root.mkdir()
    spec = _write_spec(tmp_path / "spec.json", payload)
    output = tmp_path / "failure.json"
    state = _HTTPState(payload, forced_status=503)
    with _serve(state) as base_url:
        monkeypatch.setenv(
            _URL_ENV,
            f"{base_url}?token=FAILURE_TOKEN&signature=FAILURE_SIGNATURE",
        )
        exit_code = main(
            _arguments(spec, root, output=output, max_attempts=1)
        )

    captured = capsys.readouterr()
    rendered = captured.out + captured.err + output.read_text(encoding="utf-8")
    report = json.loads(captured.out)
    assert exit_code != 0
    assert captured.err == ""
    assert report["schema_version"] == "stage0-asset-acquisition-failure-v1"
    assert report["status"] == "failed"
    assert report["code"] == "HTTP_STATUS"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "FAILURE_TOKEN" not in rendered
    assert "FAILURE_SIGNATURE" not in rendered
    assert base_url not in rendered
    assert part_path_for(root / "models" / "object.bin").exists()


def test_fetch_http_existing_output_is_no_clobber_and_prevents_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = b"no clobber fixture"
    root = tmp_path / "approved-assets"
    root.mkdir()
    spec = _write_spec(tmp_path / "spec.json", payload)
    output = tmp_path / "result.json"
    sentinel = b"user-owned-output"
    output.write_bytes(sentinel)
    state = _HTTPState(payload)
    with _serve(state) as base_url:
        monkeypatch.setenv(_URL_ENV, f"{base_url}?token=NO_CLOBBER_SECRET")
        exit_code = main(_arguments(spec, root, output=output))

    captured = capsys.readouterr()
    assert exit_code != 0
    assert json.loads(captured.out)["code"] == "OUTPUT_EXISTS"
    assert output.read_bytes() == sentinel
    assert state.ranges == []
    assert not (root / "models" / "object.bin").exists()
    assert "NO_CLOBBER_SECRET" not in captured.out + captured.err


def test_fetch_http_verified_final_is_idempotent_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = b"already complete fixture"
    root = tmp_path / "approved-assets"
    root.mkdir()
    spec = _write_spec(tmp_path / "spec.json", payload)
    state = _HTTPState(payload)
    with _serve(state) as base_url:
        monkeypatch.setenv(_URL_ENV, base_url)
        assert main(_arguments(spec, root)) == 0
    capsys.readouterr()

    monkeypatch.setenv(_URL_ENV, "http://127.0.0.1:1/unreachable?token=IDEMPOTENT_SECRET")
    exit_code = main(_arguments(spec, root))

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["status"] == "already_ready"
    assert report["attempts"] == 0
    assert report["network_accessed"] is False
    assert state.ranges == [None]
    assert "IDEMPOTENT_SECRET" not in captured.out + captured.err


def test_asset_help_keeps_legacy_commands_and_lists_fetch_http(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured_exit:
        main(["asset", "--help"])

    captured = capsys.readouterr()
    assert captured_exit.value.code == 0
    assert "fetch-http" in captured.out
    assert "acquire" in captured.out
    assert "verify" in captured.out
