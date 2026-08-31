from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ops.stage0.promote_environment_after_gpu_gate import (
    PromotionError,
    normalize_uuid,
    parse_marker,
    parse_excluded_uuid_parameter,
    verify_sha256sums,
)


def test_normalize_uuid_accepts_pytorch_and_nvml_forms() -> None:
    plain = "5c672d04-4f83-3cc0-80d0-0108b1b63267"
    expected = "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267"
    assert normalize_uuid(plain) == expected
    assert normalize_uuid(expected) == expected


@pytest.mark.parametrize("value", ["", "GPU-", "GPU-not-a-uuid", "../uuid"])
def test_normalize_uuid_rejects_malformed_values(value: str) -> None:
    with pytest.raises(PromotionError, match="invalid GPU UUID"):
        normalize_uuid(value)


def test_parse_excluded_uuid_parameter_accepts_driver_quoted_form() -> None:
    value = (
        ' "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267,'
        'GPU-e78c55cd-db97-b761-f559-dc6eae3be81d"'
    )
    assert parse_excluded_uuid_parameter(value) == [
        "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
        "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    ]


def test_parse_excluded_uuid_parameter_rejects_unbalanced_quotes() -> None:
    with pytest.raises(PromotionError, match="malformed quoted"):
        parse_excluded_uuid_parameter('"GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267')


def test_parse_marker_rejects_duplicate_keys(tmp_path: Path) -> None:
    marker = tmp_path / "SUCCESS"
    marker.write_text("status=PASS\nstatus=FAIL\n", encoding="utf-8")
    with pytest.raises(PromotionError, match="duplicate"):
        parse_marker(marker)


def test_verify_sha256sums_checks_required_regular_files(tmp_path: Path) -> None:
    payload = b"evidence\n"
    (tmp_path / "report.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  report.json\n", encoding="utf-8"
    )
    assert verify_sha256sums(tmp_path, {"report.json"}) == {"report.json": digest}


def test_verify_sha256sums_normalizes_standard_dot_slash_prefix(tmp_path: Path) -> None:
    payload = b"evidence\n"
    (tmp_path / "report.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  ./report.json\n", encoding="utf-8"
    )
    assert verify_sha256sums(tmp_path, {"report.json"}) == {"report.json": digest}


def test_verify_sha256sums_rejects_traversal(tmp_path: Path) -> None:
    digest = "0" * 64
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  ../outside\n", encoding="utf-8"
    )
    with pytest.raises(PromotionError, match="unsafe"):
        verify_sha256sums(tmp_path, set())
