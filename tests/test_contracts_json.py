from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_hash,
    import_legacy_json,
    load_canonical_json,
    loads_strict_json,
    write_canonical_json,
)


def test_canonical_json_has_one_stable_utf8_representation() -> None:
    first = {"中文": "值", "a": [3, 2, 1], "nested": {"z": False, "b": None}}
    second = {"nested": {"b": None, "z": False}, "a": [3, 2, 1], "中文": "值"}

    encoded = canonical_json_bytes(first)

    assert encoded == canonical_json_bytes(second)
    assert encoded.endswith(b"\n")
    assert b"\r" not in encoded
    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert canonical_json_hash(first) == canonical_json_hash(second)


def test_canonical_loader_round_trip_and_rejects_pretty_json(tmp_path: Path) -> None:
    target = write_canonical_json(tmp_path / "artifact.json", {"z": 1, "a": 2})
    assert load_canonical_json(target) == {"a": 2, "z": 1}

    pretty = tmp_path / "pretty.json"
    pretty.write_text('{\n  "a": 2,\n  "z": 1\n}\n', encoding="utf-8", newline="")
    with pytest.raises(CanonicalJSONError, match="不是 canonical JSON"):
        load_canonical_json(pretty)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xef\xbb\xbf{\"a\":1}\n", "BOM"),
        (b'{"a":1,"a":2}', "重复键"),
        (b'{"a":NaN}', "非有限数"),
        (b'{"a":Infinity}', "非有限数"),
        (b"\xff", "UTF-8"),
    ],
)
def test_strict_json_rejects_unsafe_encodings_and_values(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(CanonicalJSONError, match=message):
        loads_strict_json(payload)


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string key"},
        {"tuple": (1, 2)},
        {"nan": float("nan")},
        {"inf": float("inf")},
        {"object": object()},
    ],
)
def test_canonical_encoder_rejects_implicit_or_nonfinite_types(value: object) -> None:
    with pytest.raises(CanonicalJSONError):
        canonical_json_bytes(value)


def test_legacy_bom_import_is_separate_and_republishes_canonical(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.json"
    source.write_bytes(b"\xef\xbb\xbf" + b'{ "z": 1, "a": 2 }')
    target = tmp_path / "canonical.json"

    decoded = import_legacy_json(source, target)

    assert decoded == {"a": 2, "z": 1}
    assert source.read_bytes().startswith(b"\xef\xbb\xbf")
    assert target.read_bytes() == b'{"a":2,"z":1}\n'
    assert load_canonical_json(target) == decoded


def test_legacy_import_never_relaxes_duplicate_or_nonfinite_checks(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b"\xef\xbb\xbf" + b'{"a":1,"a":2}')
    with pytest.raises(CanonicalJSONError, match="重复键"):
        import_legacy_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":-Infinity}', encoding="utf-8")
    with pytest.raises(CanonicalJSONError, match="非有限数"):
        import_legacy_json(nonfinite)


def test_legacy_import_refuses_in_place_rewrite(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    source.write_text('{"a":1}', encoding="utf-8")
    with pytest.raises(CanonicalJSONError, match="不能原地覆盖"):
        import_legacy_json(source, source)
