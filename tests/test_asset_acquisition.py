from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
import time
from typing import Iterator

import pytest

import param_importance_nlp.asset_acquisition as acquisition
from param_importance_nlp.asset_acquisition import (
    AcquisitionFailureCode,
    AcquisitionPolicy,
    AcquisitionStatus,
    AssetAcquisitionError,
    AssetObjectSpec,
    acquire_http_asset,
    part_path_for,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _spec(payload: bytes, *, digest: str | None = None) -> AssetObjectSpec:
    return AssetObjectSpec(
        source_id="fixture-public-source",
        revision="0123456789abcdef0123456789abcdef01234567",
        expected_size=len(payload),
        expected_sha256=digest or _sha256(payload),
    )


def _policy(
    *,
    max_attempts: int = 3,
    lock_timeout: float = 1.0,
    overall_timeout: float = 3.0,
    initial_backoff: float = 0.0,
    max_backoff: float = 0.0,
) -> AcquisitionPolicy:
    return AcquisitionPolicy(
        max_attempts=max_attempts,
        request_timeout_seconds=1.0,
        overall_timeout_seconds=overall_timeout,
        initial_backoff_seconds=initial_backoff,
        max_backoff_seconds=max_backoff,
        lock_timeout_seconds=lock_timeout,
        lock_poll_interval_seconds=0.005,
        chunk_size=64,
    )


@dataclass
class _HTTPState:
    payload: bytes
    interrupt_first_after: int | None = None
    bad_content_range: bool = False
    delay_seconds: float = 0.0
    forced_status: int | None = None
    ranges: list[str | None] = field(default_factory=list)
    request_targets: list[str] = field(default_factory=list)
    request_started: threading.Event = field(default_factory=threading.Event)
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
            state.request_targets.append(self.path)
            request_number = len(state.ranges)
        state.request_started.set()
        if state.delay_seconds:
            time.sleep(state.delay_seconds)
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
            declared_start = offset + (1 if state.bad_content_range else 0)
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {declared_start}-{len(state.payload) - 1}/{len(state.payload)}",
            )
        else:
            body = state.payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        if state.interrupt_first_after is not None and request_number == 1:
            cutoff = state.interrupt_first_after
            self.wfile.write(body[:cutoff])
            self.wfile.flush()
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return
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


def test_interrupted_transfer_resumes_with_validated_range(tmp_path: Path) -> None:
    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
    state = _HTTPState(payload, interrupt_first_after=7)
    target = tmp_path / "model.bin"
    with _serve(state) as base_url:
        runtime_url = f"{base_url}?token=DO_NOT_PERSIST&signature=SECRET_VALUE"
        result = acquire_http_asset(
            _spec(payload),
            runtime_url,
            target,
            policy=_policy(max_attempts=2),
        )

    assert result.status is AcquisitionStatus.DOWNLOADED
    assert result.attempts == 2
    assert result.resumed is True
    assert result.network_accessed is True
    assert state.ranges == [None, "bytes=7-"]
    assert target.read_bytes() == payload
    assert not part_path_for(target).exists()
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "DO_NOT_PERSIST" not in serialized
    assert "SECRET_VALUE" not in serialized
    assert "runtime_url" not in result.to_dict()


def test_existing_prefix_is_not_trusted_without_final_hash(tmp_path: Path) -> None:
    payload = b"the authoritative immutable payload"
    target = tmp_path / "dataset.bin"
    part_path_for(target).write_bytes(b"BAD!!")
    state = _HTTPState(payload)
    with _serve(state) as runtime_url:
        result = acquire_http_asset(
            _spec(payload),
            runtime_url,
            target,
            policy=_policy(max_attempts=2),
        )

    assert result.status is AcquisitionStatus.DOWNLOADED
    assert state.ranges == ["bytes=5-", None]
    assert target.read_bytes() == payload
    assert not part_path_for(target).exists()


def test_invalid_content_range_preserves_prefix_and_reports_safely(
    tmp_path: Path,
) -> None:
    payload = b"range validation fixture"
    target = tmp_path / "range.bin"
    prefix = payload[:6]
    part_path_for(target).write_bytes(prefix)
    state = _HTTPState(payload, bad_content_range=True)
    with _serve(state) as base_url:
        runtime_url = f"{base_url}?token=RANGE_TOKEN&sig=RANGE_SIGNATURE"
        with pytest.raises(AssetAcquisitionError) as captured:
            acquire_http_asset(
                _spec(payload),
                runtime_url,
                target,
                policy=_policy(max_attempts=1),
            )

    report = captured.value.report
    assert report.code is AcquisitionFailureCode.CONTENT_RANGE_INVALID
    assert report.attempts == 1
    assert report.http_status == 206
    assert part_path_for(target).read_bytes() == prefix
    assert not target.exists()
    rendered = json.dumps(report.to_dict(), sort_keys=True) + repr(captured.value)
    assert "RANGE_TOKEN" not in rendered
    assert "RANGE_SIGNATURE" not in rendered
    assert base_url not in rendered
    assert captured.value.__cause__ is None


def test_wrong_hash_is_never_published_and_retries_are_bounded(
    tmp_path: Path,
) -> None:
    payload = b"served bytes"
    wrong_digest = _sha256(b"wrong bytes!")
    assert len(b"wrong bytes!") == len(payload)
    state = _HTTPState(payload)
    target = tmp_path / "wrong-hash.bin"
    with _serve(state) as runtime_url:
        with pytest.raises(AssetAcquisitionError) as captured:
            acquire_http_asset(
                _spec(payload, digest=wrong_digest),
                runtime_url,
                target,
                policy=_policy(max_attempts=2),
            )

    report = captured.value.report
    assert report.code is AcquisitionFailureCode.HASH_MISMATCH
    assert report.attempts == 2
    assert report.exhausted is True
    assert len(state.ranges) == 2
    assert not target.exists()


def test_object_lock_rejects_a_competing_writer(tmp_path: Path) -> None:
    payload = b"serialized writer fixture"
    state = _HTTPState(payload, delay_seconds=0.2)
    target = tmp_path / "locked.bin"
    with _serve(state) as runtime_url, ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            acquire_http_asset,
            _spec(payload),
            runtime_url,
            target,
            policy=_policy(max_attempts=1, lock_timeout=1.0),
        )
        assert state.request_started.wait(timeout=1)
        with pytest.raises(AssetAcquisitionError) as captured:
            acquire_http_asset(
                _spec(payload),
                runtime_url,
                target,
                policy=_policy(
                    max_attempts=1,
                    lock_timeout=0.04,
                    overall_timeout=1.0,
                ),
            )
        first_result = first.result(timeout=2)

    assert captured.value.report.code is AcquisitionFailureCode.LOCK_TIMEOUT
    assert first_result.status is AcquisitionStatus.DOWNLOADED
    assert target.read_bytes() == payload
    assert len(state.ranges) == 1


def test_ready_final_is_idempotent_without_an_http_request(tmp_path: Path) -> None:
    payload = b"idempotent final object"
    state = _HTTPState(payload)
    target = tmp_path / "ready.bin"
    with _serve(state) as runtime_url:
        first = acquire_http_asset(
            _spec(payload), runtime_url, target, policy=_policy(max_attempts=1)
        )
        request_count = len(state.ranges)
        second = acquire_http_asset(
            _spec(payload),
            "http://127.0.0.1:1/unreachable?token=NOT_STORED",
            target,
            policy=_policy(max_attempts=1),
        )

    assert first.status is AcquisitionStatus.DOWNLOADED
    assert second.status is AcquisitionStatus.ALREADY_READY
    assert second.attempts == 0
    assert second.network_accessed is False
    assert len(state.ranges) == request_count == 1
    assert target.read_bytes() == payload


def test_atomic_publication_never_clobbers_a_racing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"verified candidate bytes"
    rival = b"rival process content"
    assert len(payload) != len(rival)
    state = _HTTPState(payload)
    target = tmp_path / "race.bin"

    def racing_link(
        _source: str | bytes | Path,
        destination: str | bytes | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        del follow_symlinks
        Path(destination).write_bytes(rival)
        raise FileExistsError(errno.EEXIST, "racing target")

    monkeypatch.setattr(acquisition.os, "link", racing_link)
    with _serve(state) as runtime_url:
        with pytest.raises(AssetAcquisitionError) as captured:
            acquire_http_asset(
                _spec(payload),
                runtime_url,
                target,
                policy=_policy(max_attempts=1),
            )

    assert captured.value.report.code is AcquisitionFailureCode.TARGET_CONFLICT
    assert target.read_bytes() == rival
    assert part_path_for(target).read_bytes() == payload


def test_retry_backoff_respects_the_overall_deadline(tmp_path: Path) -> None:
    payload = b"timeout fixture"
    state = _HTTPState(payload, forced_status=503)
    target = tmp_path / "bounded.bin"
    started = time.monotonic()
    with _serve(state) as runtime_url:
        with pytest.raises(AssetAcquisitionError) as captured:
            acquire_http_asset(
                _spec(payload),
                runtime_url,
                target,
                policy=_policy(
                    max_attempts=20,
                    overall_timeout=0.08,
                    initial_backoff=0.05,
                    max_backoff=0.05,
                ),
            )
    elapsed = time.monotonic() - started

    assert captured.value.report.attempts < 20
    assert captured.value.report.code in {
        AcquisitionFailureCode.HTTP_STATUS,
        AcquisitionFailureCode.OVERALL_TIMEOUT,
    }
    assert elapsed < 1.0
    assert not target.exists()


def test_spec_rejects_runtime_urls_and_generic_revisions() -> None:
    digest = _sha256(b"x")
    with pytest.raises(ValueError, match="stable identifier"):
        AssetObjectSpec("public-source?token=secret", "abc123", 1, digest)
    with pytest.raises(ValueError, match="immutable"):
        AssetObjectSpec("public-source", "latest", 1, digest)
    with pytest.raises(ValueError, match="max_attempts"):
        AcquisitionPolicy(max_attempts=0)
