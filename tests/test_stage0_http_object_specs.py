from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

from param_importance_nlp.asset_acquisition import AssetObjectSpec
from param_importance_nlp.contracts import (
    canonical_json_bytes,
    ensure_json_object,
    loads_strict_json,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FREEZE_PATH = _REPOSITORY_ROOT / "configs/stage0/g3-asset-requirements-v1.json"
_SPEC_ROOT = _REPOSITORY_ROOT / "configs/stage0/http-objects"
_EXPECTED_FILENAMES = {
    "glue-mnli-test-matched.json",
    "glue-mnli-test-mismatched.json",
    "glue-mnli-train.json",
    "glue-mnli-validation-matched.json",
    "glue-mnli-validation-mismatched.json",
    "glue-rte-test.json",
    "glue-rte-train.json",
    "glue-rte-validation.json",
    "pythia-410m-deduped-step0-config.json",
    "pythia-410m-deduped-step0-model-safetensors.json",
    "pythia-tokenizer-special-tokens-map.json",
    "pythia-tokenizer-tokenizer-config.json",
    "pythia-tokenizer-tokenizer.json",
}
_URL_PATTERN = re.compile(r"(?i)\b(?:https?|ftp)://")
_SECRET_QUERY_PATTERN = re.compile(
    r"(?i)(?:[?&]|\b)(?:access_?token|api_?key|auth|authorization|cookie|"
    r"credential|password|secret|sig|signature)="
)


def _logical_source_id(repository: str, object_path: str) -> str:
    return f"huggingface/{repository}/{object_path}"


def _expected_object(
    *,
    repository: str,
    revision: str,
    descriptor: Mapping[str, Any],
    upstream_path: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "stage0-http-object-spec-v1",
        "source_id": _logical_source_id(
            repository,
            upstream_path or descriptor["path"],
        ),
        "revision": revision,
        "expected_size": descriptor["size_bytes"],
        "expected_sha256": descriptor["sha256"],
    }


def _freeze_objects() -> dict[str, dict[str, object]]:
    freeze = ensure_json_object(
        loads_strict_json(_FREEZE_PATH.read_bytes()),
        field="stage0 G3 asset requirements",
    )
    selected: list[dict[str, object]] = []

    model = next(
        item
        for item in freeze["models"]
        if item["name"] == "pythia-410m-deduped-step0"
    )
    selected.extend(
        _expected_object(
            repository=model["repository"],
            revision=model["revision"],
            descriptor=descriptor,
        )
        for descriptor in model["files"]
    )

    tokenizer = freeze["tokenizer"]
    selected.extend(
        _expected_object(
            repository=tokenizer["repository"],
            revision=tokenizer["revision"],
            descriptor=descriptor,
        )
        for descriptor in tokenizer["files"]
    )

    glue_tasks = {item["task"]: item for item in freeze["glue"]}
    for task_name in ("mnli", "rte"):
        task = glue_tasks[task_name]
        selected.extend(
            _expected_object(
                repository=task["repository"],
                revision=task["revision"],
                descriptor=descriptor,
                upstream_path=f"{task_name}/{descriptor['path']}",
            )
            for descriptor in task["raw_files"]
        )

    return {item["source_id"]: item for item in selected}


def test_stage0_http_object_specs_are_complete_canonical_and_frozen() -> None:
    spec_paths = sorted(_SPEC_ROOT.glob("*.json"))
    assert {path.name for path in spec_paths} == _EXPECTED_FILENAMES
    assert len(spec_paths) == 13

    expected = _freeze_objects()
    assert len(expected) == 13
    observed: dict[str, dict[str, object]] = {}
    identities: list[tuple[str, str, int, str]] = []

    for path in spec_paths:
        raw = path.read_bytes()
        value = dict(
            ensure_json_object(
                loads_strict_json(raw),
                field=str(path),
            )
        )
        assert raw == canonical_json_bytes(value), f"not canonical: {path.name}"
        assert AssetObjectSpec.from_mapping(value).to_dict() == value

        source_id = value["source_id"]
        assert source_id not in observed, f"duplicate source_id: {source_id}"
        observed[source_id] = value
        identities.append(
            (
                source_id,
                value["revision"],
                value["expected_size"],
                value["expected_sha256"],
            )
        )

    assert observed == expected
    assert len(identities) == len(set(identities)) == 13


def test_stage0_http_object_specs_contain_no_url_or_secret_query() -> None:
    for path in sorted(_SPEC_ROOT.glob("*.json")):
        raw = path.read_bytes()
        value = ensure_json_object(loads_strict_json(raw), field=str(path))
        source_id = value["source_id"]
        parsed = urlsplit(source_id)

        assert parsed.scheme == ""
        assert parsed.netloc == ""
        assert parsed.query == ""
        assert parsed.fragment == ""
        assert "?" not in source_id
        rendered = raw.decode("utf-8")
        assert _URL_PATTERN.search(rendered) is None
        assert _SECRET_QUERY_PATTERN.search(rendered) is None
