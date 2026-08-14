#!/usr/bin/env python3
"""Fail-closed formal publisher for S1.6 / G1-STEP CPU evidence.

This command intentionally has no best-effort mode.  It refuses a dirty
checkout, verifies the immutable S1.5 index and every referenced role, replays
S1.5 numerical semantics from the current consumer checkout, and only then
creates a new immutable S1.6 evidence directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping


TASK_ID = "stage1.06_training_integration_and_accumulators"
GATE_ID = "G1-STEP"
FIXTURE_ID = "stage1-s16-training-fixture-v1"
EXPECTED_INDEX_SHA256 = "890970c5886821377ca0409ca910184da99063432d3ef35fd3e3713a5e5514c5"
EXPECTED_INDEX_ARTIFACT_HASH = "69a8281fe0186d0574e9b0faebecfb4dedba44ab960367010c622007ad96deb1"
EXPECTED_GATE_HASH = "4bee73ef5053d78f77d688f94fd1737cc744cbcf2bb4774f8916fc70e466887a"
EXPECTED_S1_5_PRODUCER = "36a792b6a89045ae49c32225038a6c10d5082d2c"
EXPECTED_S1_5_REPORT_HASH = "7635dd8c3b69ca80f1ac7d0d5a9956602ad31160a26f7eda0101c34a0373bd87"
EXPECTED_S1_5_ORACLE_HASH = "41f1b2e654cdafbb29de45aecd8db3fd11dee3d8f6c260acb574f7803c639b4b"
EXPECTED_S1_5_BUNDLE_HASH = "55c8b576e9bc71078fbbd214838e493505a2c362ad3eaeacc4720a824d9860f0"
EXPECTED_S1_5_TABLE_HASH = "8fbade2df149deb7f61eeb64ea3aa76f2b3b6dcceee199e906ed359999b6ee65"
EXPECTED_S1_5_REPLAY_HASH = "b8160d409a1f65d30b33f89f60a08c7792916d43b7962ee7efd7a6ab963bc7dc"
EXPECTED_S1_5_VALIDATION_ARTIFACT_HASH = "a81d1b286c818a30d9eeab9113ef491dcd7902f91544663ad66fa8a787c56d91"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class Stage1S16FormalError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repository), *args], text=True, capture_output=True, check=False, timeout=30)
    if done.returncode:
        raise Stage1S16FormalError(f"S1_6_GIT_FAILED:{args[0]}")
    return done.stdout.strip()


def _path(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S16FormalError(f"S1_6_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S16FormalError(f"S1_6_LOGICAL_REF_ESCAPE:{field}")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S16FormalError(f"S1_6_LOGICAL_REF_ESCAPE:{field}") from error
    return candidate


def _with_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash
    body = dict(value)
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _audit_s15_producer_diff(repository: Path) -> tuple[str, ...]:
    """Reject an S1.5-owned drift while permitting only named S1.6 consumers."""
    changed = tuple(filter(None, _git(repository, "diff", "--name-only", EXPECTED_S1_5_PRODUCER, "HEAD").splitlines()))
    allowed_exact = {
        # The sole post-producer S1.5 refresh is an operator log; no S1.5
        # implementation, fixture, or schema may change beneath this handoff.
        "worklogs/2026-08-14-s1.5-estimators.md",
        # S1.6-owned consumer surface.  Keep this exact rather than a broad
        # directory prefix: a new path requires an explicit handoff review.
        "ops/stage1/formalize_s1_6.py",
        "src/param_importance_nlp/core/accumulator.py",
        "src/param_importance_nlp/core/baselines.py",
        "src/param_importance_nlp/runtime/optimizer.py",
        "src/param_importance_nlp/runtime/training.py",
        "tests/test_stage1_s16_training_integration.py",
        "tests/test_stage1_s16_handoff_and_charts.py",
        "tests/test_runtime_training_engine.py",
        "tests/test_core_estimators_and_accumulator.py",
        "tests/test_stage79_run_ready_completion.py",
    }
    # Only freshly introduced S1.6 artifact families receive a constrained
    # naming rule.  Existing runtime/core files and named regression tests are
    # exact paths above; e.g. ``runtime/training.py.extra`` is never accepted.
    allowed_s16_family = (
        re.compile(r"fixtures/stage1/stage1-s16-[a-z0-9-]+-v1\.json"),
        re.compile(r"schemas/stage1/s1-6-[a-z0-9-]+-v1\.json"),
        re.compile(r"src/param_importance_nlp/stage1_training_(?:integration|oracle)\.py"),
        re.compile(r"tests/test_stage1_s16_[a-z0-9_]+\.py"),
    )
    unauthorized = [
        path for path in changed
        if path not in allowed_exact and not any(rule.fullmatch(path) for rule in allowed_s16_family)
    ]
    if unauthorized:
        raise Stage1S16FormalError("S1_6_S1_5_PRODUCER_DIFF_UNAUTHORIZED:" + ",".join(unauthorized))
    s15_owned = ("fixtures/stage1/stage1-s15-", "schemas/stage1/s1-5-", "src/param_importance_nlp/stage1_estimators.py")
    if any(path.startswith(s15_owned) for path in changed):
        raise Stage1S16FormalError("S1_6_S1_5_OWNED_SOURCE_DRIFT")
    return changed


def _load_s15(data_root: Path, index_ref: str, repository: Path) -> dict[str, Any]:
    """Verify immutable index/roles, then rebuild the S1.5 numeric semantics.

    Source maps themselves are immutable role payload.  Any current source-map
    drift is rejected, rather than treating all future S1.6 files as an
    allowlist.  This deliberately makes the formal run fail until a reviewer
    explicitly resolves a shared-core change.
    """
    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
    from param_importance_nlp.stage1_estimators import build_stage1_s15_evidence, validate_stage1_s15_evidence

    producer_diff = _audit_s15_producer_diff(repository)
    index_path = _path(data_root, index_ref, field="s1_5_index_ref")
    if not index_path.is_file() or _sha(index_path) != EXPECTED_INDEX_SHA256:
        raise Stage1S16FormalError("S1_6_S1_5_INDEX_NOT_CURRENT")
    index = load_canonical_json(index_path)
    if not isinstance(index, Mapping):
        raise Stage1S16FormalError("S1_6_S1_5_INDEX_NOT_OBJECT")
    body = dict(index); supplied = body.pop("artifact_hash", None)
    if supplied != canonical_json_hash(body):
        raise Stage1S16FormalError("S1_6_S1_5_INDEX_HASH_INVALID")
    if (
        index.get("schema_version") != "stage1-s1-5-formalization-index-v1"
        or index.get("status") != "PASS"
        or index.get("gate_id") != "G1-EST"
        or index.get("task_id") != "stage1.05_estimators"
        or index.get("generator_git_commit") != EXPECTED_S1_5_PRODUCER
        or index.get("consumer_git_commit") != EXPECTED_S1_5_PRODUCER
        or index.get("artifact_hash") != EXPECTED_INDEX_ARTIFACT_HASH
        or index.get("gate_artifact_hash") != EXPECTED_GATE_HASH
        or index.get("replay_hash") != EXPECTED_S1_5_REPLAY_HASH
        or index.get("next_task_id") != TASK_ID
    ):
        raise Stage1S16FormalError("S1_6_S1_5_HANDOFF_NOT_READY")
    refs = index.get("role_refs"); hashes = index.get("role_sha256")
    wanted = {"estimator_report", "oracle_report", "tensor_bundle", "comparison_table", "gate_record"}
    if not isinstance(refs, Mapping) or not isinstance(hashes, Mapping) or set(refs) != wanted or set(hashes) != wanted:
        raise Stage1S16FormalError("S1_6_S1_5_ROLE_SET_INVALID")
    roles: dict[str, Any] = {}
    for role in sorted(wanted):
        candidate = _path(data_root, refs[role], field=role)
        if not candidate.is_file():
            candidate = (index_path.parent / str(refs[role])).resolve()
        if not candidate.is_file() or not isinstance(hashes[role], str) or _DIGEST.fullmatch(hashes[role]) is None or _sha(candidate) != hashes[role]:
            raise Stage1S16FormalError(f"S1_6_S1_5_ROLE_INVALID:{role}")
        roles[role] = load_canonical_json(candidate)
    auxiliary: dict[str, Any] = {}
    for role, ref_field, sha_field in (
        ("replay", "replay_ref", "replay_sha256"),
        ("validation", "validation_ref", "validation_sha256"),
    ):
        candidate = _path(data_root, index.get(ref_field), field=role)
        if not candidate.is_file(): candidate = (index_path.parent / str(index.get(ref_field))).resolve()
        expected = index.get(sha_field)
        if not candidate.is_file() or not isinstance(expected, str) or _DIGEST.fullmatch(expected) is None or _sha(candidate) != expected:
            raise Stage1S16FormalError(f"S1_6_S1_5_AUXILIARY_INVALID:{role}")
        auxiliary[role] = load_canonical_json(candidate)
    replay, validation = auxiliary["replay"], auxiliary["validation"]
    if not isinstance(replay, Mapping) or not isinstance(validation, Mapping):
        raise Stage1S16FormalError("S1_6_S1_5_AUXILIARY_NOT_OBJECT")
    replay_body = dict(replay); replay_hash = replay_body.pop("replay_hash", None)
    validation_body = dict(validation); validation_hash = validation_body.pop("artifact_hash", None)
    if replay_hash != canonical_json_hash(replay_body) or validation_hash != canonical_json_hash(validation_body):
        raise Stage1S16FormalError("S1_6_S1_5_AUXILIARY_SELF_HASH_INVALID")
    if (
        validation.get("status") != "PASS"
        or validation.get("artifact_hash") != EXPECTED_S1_5_VALIDATION_ARTIFACT_HASH
        or validation.get("role_sha256") != dict(hashes)
        or validation.get("replay_sha256") != index.get("replay_sha256")
        or validation.get("replay_hash") != index.get("replay_hash")
        or index.get("replay_hash") != replay.get("replay_hash")
    ):
        raise Stage1S16FormalError("S1_6_S1_5_AUXILIARY_BINDING_INVALID")
    try:
        # First validate the historical roles against their own frozen hashes;
        # passing a current checkout here would wrongly confuse a producer
        # revision with its consumer's later worklog/implementation state.
        validate_stage1_s15_evidence(roles)
    except Exception as error:
        raise Stage1S16FormalError("S1_6_S1_5_HISTORICAL_VALIDATION_FAILED") from error
    report = roles["estimator_report"]
    if not isinstance(report, Mapping):
        raise Stage1S16FormalError("S1_6_S1_5_REPORT_INVALID")
    oracle, bundle, table, gate = (
        roles["oracle_report"],
        roles["tensor_bundle"],
        roles["comparison_table"],
        roles["gate_record"],
    )
    if not all(isinstance(value, Mapping) for value in (oracle, bundle, table, gate)):
        raise Stage1S16FormalError("S1_6_S1_5_ROLE_NOT_OBJECT")
    if (
        report.get("report_hash") != EXPECTED_S1_5_REPORT_HASH
        or oracle.get("oracle_hash") != EXPECTED_S1_5_ORACLE_HASH
        or bundle.get("bundle_hash") != EXPECTED_S1_5_BUNDLE_HASH
        or table.get("table_hash") != EXPECTED_S1_5_TABLE_HASH
        or gate.get("artifact_hash") != EXPECTED_GATE_HASH
    ):
        raise Stage1S16FormalError("S1_6_S1_5_ROLE_SEMANTIC_BINDING_INVALID")
    rebuilt = build_stage1_s15_evidence(
        repository, producer_commit=str(report.get("producer_commit")), scope=str(report.get("scope")),
        upstream_evidence=report.get("upstream") if isinstance(report.get("upstream"), Mapping) else None,
    )
    historical_map = report.get("implementation_source_sha256")
    rebuilt_map = rebuilt["estimator_report"].get("implementation_source_sha256")
    if not isinstance(historical_map, Mapping) or historical_map != rebuilt_map:
        raise Stage1S16FormalError("S1_6_S1_5_SOURCE_MAP_DRIFT")
    # Compare every numerical role after dropping only its self-hash/source map.
    for role in wanted:
        old, new = dict(roles[role]), dict(rebuilt[role])
        for value in (old, new):
            value.pop("report_hash", None); value.pop("table_hash", None); value.pop("artifact_hash", None)
            value.pop("implementation_source_sha256", None)
        if old != new:
            raise Stage1S16FormalError(f"S1_6_S1_5_NUMERIC_REPLAY_MISMATCH:{role}")
    return {
        "s1_5_index_ref": index_ref, "s1_5_index_sha256": EXPECTED_INDEX_SHA256,
        "s1_5_index_artifact_hash": str(index["artifact_hash"]),
        "s1_5_generator_commit": str(index["generator_git_commit"]), "s1_5_consumer_commit": str(index["consumer_git_commit"]),
        "s1_5_gate_artifact_hash": EXPECTED_GATE_HASH,
        "s1_5_role_refs": {str(key): str(value) for key, value in refs.items()},
        "s1_5_role_sha256": {str(key): str(value) for key, value in hashes.items()},
        "s1_5_report_hash": EXPECTED_S1_5_REPORT_HASH,
        "s1_5_oracle_hash": EXPECTED_S1_5_ORACLE_HASH,
        "s1_5_bundle_hash": EXPECTED_S1_5_BUNDLE_HASH,
        "s1_5_table_hash": EXPECTED_S1_5_TABLE_HASH,
        "s1_5_replay_ref": str(index["replay_ref"]), "s1_5_replay_sha256": str(index["replay_sha256"]), "s1_5_replay_hash": str(index["replay_hash"]),
        "s1_5_validation_ref": str(index["validation_ref"]), "s1_5_validation_sha256": str(index["validation_sha256"]), "s1_5_validation_artifact_hash": str(validation["artifact_hash"]),
        "s1_5_numeric_replay_role_count": len(wanted),
        "s1_5_producer_diff_paths": list(producer_diff),
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    from param_importance_nlp.contracts.jsonio import write_canonical_json
    if path.exists():
        raise Stage1S16FormalError("S1_6_IMMUTABLE_TARGET_EXISTS")
    write_canonical_json(path, dict(value))


def _schema_registry(repository: Path) -> dict[str, Mapping[str, Any]]:
    """Load only the frozen local S1.6 schemas (no network/schema fallback)."""

    from param_importance_nlp.contracts.jsonio import loads_strict_json

    registry: dict[str, Mapping[str, Any]] = {}
    for path in sorted((repository / "schemas" / "stage1").glob("s1-6-*.json")):
        try:
            # Schemas are source inputs rather than canonical artifacts, but
            # duplicate keys are still an ambiguity and must be rejected.
            loaded = loads_strict_json(path.read_bytes())
        except Exception as error:
            raise Stage1S16FormalError(f"S1_6_SCHEMA_PARSE_INVALID:{path.name}") from error
        identifier = loaded.get("$id") if isinstance(loaded, Mapping) else None
        if not isinstance(identifier, str) or not identifier:
            raise Stage1S16FormalError(f"S1_6_SCHEMA_ID_INVALID:{path.name}")
        registry[identifier] = loaded
        registry[path.name] = loaded
    if len({key for key in registry if key.startswith("https://")}) != 8:
        raise Stage1S16FormalError("S1_6_SCHEMA_REGISTRY_INCOMPLETE")
    return registry


def _schema_resolve(reference: str, document: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Small fail-closed Draft-2020 subset sufficient for our frozen schemas.

    The project intentionally has no runtime dependency on ``jsonschema``.  We
    therefore resolve local refs and enforce every keyword used by the eight
    S1.6 schemas; unknown remote refs never acquire permissive semantics.
    """

    base, separator, fragment = reference.partition("#")
    target: object
    if base:
        target = registry.get(base) or registry.get(PurePosixPath(base).name)
        if target is None:
            raise Stage1S16FormalError(f"S1_6_SCHEMA_REF_UNKNOWN:{reference}")
    else:
        target = document
    if separator and fragment:
        if not fragment.startswith("/"):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_REF_FRAGMENT_INVALID:{reference}")
        for token in fragment.lstrip("/").split("/"):
            if not isinstance(target, Mapping):
                raise Stage1S16FormalError(f"S1_6_SCHEMA_REF_NOT_OBJECT:{reference}")
            target = target.get(token.replace("~1", "/").replace("~0", "~"))
    if not isinstance(target, Mapping):
        raise Stage1S16FormalError(f"S1_6_SCHEMA_REF_NOT_OBJECT:{reference}")
    return target


def _validate_schema(value: object, schema: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]], *, document: Mapping[str, Any] | None = None, path: str = "$") -> None:
    """Strict, deterministic instance validator for the frozen local schemas."""

    owner = schema if document is None else document
    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_REF_INVALID:{path}")
        resolved = _schema_resolve(reference, owner, registry)
        ref_document = owner if not reference.partition("#")[0] else (registry.get(reference.partition("#")[0]) or registry.get(PurePosixPath(reference.partition("#")[0]).name))
        if not isinstance(ref_document, Mapping):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_REF_UNKNOWN:{reference}")
        _validate_schema(value, resolved, registry, document=ref_document, path=path)
        return
    for member in schema.get("allOf", []):
        if not isinstance(member, Mapping): raise Stage1S16FormalError(f"S1_6_SCHEMA_ALLOF_INVALID:{path}")
        _validate_schema(value, member, registry, document=owner, path=path)
    if "anyOf" in schema:
        members = schema["anyOf"]
        if not isinstance(members, list) or not any(
            _schema_valid(value, member, registry, owner, path) for member in members if isinstance(member, Mapping)
        ):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_ANYOF_INVALID:{path}")
    if "const" in schema and value != schema["const"]:
        raise Stage1S16FormalError(f"S1_6_SCHEMA_CONST_INVALID:{path}")
    if "enum" in schema and value not in schema["enum"]:
        raise Stage1S16FormalError(f"S1_6_SCHEMA_ENUM_INVALID:{path}")
    declared = schema.get("type")
    if declared is not None:
        options = declared if isinstance(declared, list) else [declared]
        if not any(_schema_type(value, candidate) for candidate in options):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_TYPE_INVALID:{path}")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        if not isinstance(required, list) or any(key not in value for key in required):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_REQUIRED_INVALID:{path}")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping): raise Stage1S16FormalError(f"S1_6_SCHEMA_PROPERTIES_INVALID:{path}")
        additional = schema.get("additionalProperties", True)
        extras = set(value) - set(properties)
        if additional is False and extras:
            raise Stage1S16FormalError(f"S1_6_SCHEMA_EXTRA_FIELD:{path}")
        if isinstance(additional, Mapping):
            for key in extras:
                _validate_schema(value[key], additional, registry, document=owner, path=f"{path}.{key}")
        property_names = schema.get("propertyNames")
        if isinstance(property_names, Mapping):
            for key in value:
                _validate_schema(key, property_names, registry, document=owner, path=f"{path}.<key>")
        for keyword, failed in (
            ("minProperties", isinstance(schema.get("minProperties"), int) and len(value) < schema["minProperties"]),
            ("maxProperties", isinstance(schema.get("maxProperties"), int) and len(value) > schema["maxProperties"]),
        ):
            if failed: raise Stage1S16FormalError(f"S1_6_SCHEMA_{keyword.upper()}_INVALID:{path}")
        for key, member in properties.items():
            if key in value:
                if not isinstance(member, Mapping): raise Stage1S16FormalError(f"S1_6_SCHEMA_PROPERTY_INVALID:{path}.{key}")
                _validate_schema(value[key], member, registry, document=owner, path=f"{path}.{key}")
    if isinstance(value, list):
        minimum, maximum = schema.get("minItems"), schema.get("maxItems")
        if (isinstance(minimum, int) and len(value) < minimum) or (isinstance(maximum, int) and len(value) > maximum):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_ITEM_COUNT_INVALID:{path}")
        if schema.get("uniqueItems") is True and len({json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) for item in value}) != len(value):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_ITEMS_NOT_UNIQUE:{path}")
        prefix = schema.get("prefixItems")
        if prefix is not None:
            if not isinstance(prefix, list) or len(value) < len(prefix): raise Stage1S16FormalError(f"S1_6_SCHEMA_PREFIX_ITEMS_INVALID:{path}")
            for index, member in enumerate(prefix):
                if not isinstance(member, Mapping): raise Stage1S16FormalError(f"S1_6_SCHEMA_PREFIX_INVALID:{path}[{index}]")
                _validate_schema(value[index], member, registry, document=owner, path=f"{path}[{index}]")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value): _validate_schema(item, item_schema, registry, document=owner, path=f"{path}[{index}]")
        contains = schema.get("contains")
        if isinstance(contains, Mapping) and not any(_schema_valid(item, contains, registry, owner, f"{path}[{index}]") for index, item in enumerate(value)):
            raise Stage1S16FormalError(f"S1_6_SCHEMA_CONTAINS_INVALID:{path}")
    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]: raise Stage1S16FormalError(f"S1_6_SCHEMA_STRING_LENGTH_INVALID:{path}")
        if isinstance(schema.get("pattern"), str) and re.fullmatch(schema["pattern"], value) is None: raise Stage1S16FormalError(f"S1_6_SCHEMA_PATTERN_INVALID:{path}")
        if schema.get("format") == "date-time":
            try: datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error: raise Stage1S16FormalError(f"S1_6_SCHEMA_DATE_TIME_INVALID:{path}") from error
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)): raise Stage1S16FormalError(f"S1_6_SCHEMA_NONFINITE_INVALID:{path}")
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]: raise Stage1S16FormalError(f"S1_6_SCHEMA_MINIMUM_INVALID:{path}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]: raise Stage1S16FormalError(f"S1_6_SCHEMA_MAXIMUM_INVALID:{path}")
        if isinstance(schema.get("exclusiveMinimum"), (int, float)) and value <= schema["exclusiveMinimum"]: raise Stage1S16FormalError(f"S1_6_SCHEMA_EXCLUSIVE_MINIMUM_INVALID:{path}")
        if isinstance(schema.get("exclusiveMaximum"), (int, float)) and value >= schema["exclusiveMaximum"]: raise Stage1S16FormalError(f"S1_6_SCHEMA_EXCLUSIVE_MAXIMUM_INVALID:{path}")


def _schema_type(value: object, declared: object) -> bool:
    return {
        "object": isinstance(value, Mapping), "array": isinstance(value, list), "string": isinstance(value, str),
        "boolean": type(value) is bool, "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool), "null": value is None,
    }.get(declared, False)


def _schema_valid(value: object, schema: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]], document: Mapping[str, Any], path: str) -> bool:
    try:
        _validate_schema(value, schema, registry, document=document, path=path)
    except Stage1S16FormalError:
        return False
    return True


def _validate_role_schemas(repository: Path, objects: Mapping[str, Mapping[str, Any]]) -> None:
    registry = _schema_registry(repository)
    filenames = {
        "step_report": "s1-6-step-report-v1.json", "oracle_bundle": "s1-6-oracle-bundle-v1.json",
        "trace_bundle": "s1-6-trace-bundle-v1.json", "comparison_table": "s1-6-comparison-table-v1.json",
        "gate_record": "s1-6-gate-record-v1.json", "validation": "s1-6-validation-v1.json",
        "index": "s1-6-formalization-index-v1.json",
    }
    for role, instance in objects.items():
        filename = filenames.get(role)
        schema = registry.get(filename)
        if filename is None or schema is None:
            raise Stage1S16FormalError(f"S1_6_SCHEMA_ROLE_UNKNOWN:{role}")
        _validate_schema(instance, schema, registry, document=schema, path=role)


def _fixture_schema_and_hash(repository: Path) -> bool:
    """Validate the serialized fixture itself rather than borrowing a gate bit."""

    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json

    try:
        from param_importance_nlp.stage1_training_oracle import FROZEN_FIXTURE_HASH
        fixture = json.loads((repository / "fixtures/stage1/stage1-s16-training-fixture-v1.json").read_text(encoding="utf-8"))
        if not isinstance(fixture, Mapping): return False
        schema = _schema_registry(repository).get("s1-6-training-fixture-v1.json")
        if schema is None: return False
        _validate_schema(fixture, schema, _schema_registry(repository), document=schema, path="fixture")
        body = dict(fixture); supplied = body.pop("fixture_hash", None)
        return isinstance(supplied, str) and supplied == FROZEN_FIXTURE_HASH and supplied == canonical_json_hash(body)
    except (OSError, ValueError, Stage1S16FormalError):
        return False


def _oracle_isolation(repository: Path, oracle_bundle: Mapping[str, Any]) -> bool:
    """AST-level proof that the independent oracle imports no production core."""

    try:
        tree = ast.parse((repository / "src/param_importance_nlp/stage1_training_oracle.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): imports.append("." * node.level + (node.module or ""))
    forbidden = ("param_importance_nlp", ".contracts", ".core", ".runtime", ".stage1_training_integration")
    return bool(oracle_bundle.get("independent_implementation") is True) and not any(name.startswith(forbidden) for name in imports)


def _nontrivial_clip_contract(evidence: Mapping[str, Any]) -> bool:
    """Recompute the frozen clip identities from persisted trace/oracle roles."""

    try:
        trace = evidence["trace_bundle"]
        oracle_bundle = evidence["oracle_bundle"]
        if not isinstance(trace, Mapping) or not isinstance(oracle_bundle, Mapping): return False
        clip = trace["clip_training_engine_trace"]
        oracle = oracle_bundle["oracle"]
        if not isinstance(clip, Mapping) or not isinstance(oracle, Mapping): return False
        expected = oracle["clip_oracle"]
        gradient, post = clip["gradient_event"], clip["parameter_post_event"]
        accumulator, interval, record = clip["accumulator_state"], clip["accumulator_interval_delta"], clip["record"]
        if not all(isinstance(value, Mapping) for value in (expected, gradient, post, accumulator, interval, record)): return False
        factor = float(expected["clip_factor"])
        close = lambda a, b: abs(float(a) - float(b)) <= 1e-12
        for name in ("left", "right"):
            if not close(gradient["mean_gradient"][name][0], expected["mean_gradient"][name][0]): return False
            if not close(gradient["optimizer_gradient"][name][0], expected["optimizer_gradient"][name][0]): return False
            if not close(post["data_delta"][name][0], expected["data_delta"][name][0]): return False
            if not close(accumulator["raw"][name][0], expected["raw_unclipped"][name][0]): return False
            if not close(accumulator["raw_clipped"][name][0], expected["raw_clipped"][name][0]): return False
            if not close(accumulator["signed"][name][0], expected["raw_clipped"][name][0]): return False
            if not close(interval["raw"][name][0], expected["raw_unclipped"][name][0]): return False
            if not close(interval["raw_clipped"][name][0], expected["raw_clipped"][name][0]): return False
            if not close(interval["signed"][name][0], expected["raw_clipped"][name][0]): return False
            if not close(interval["positive"][name][0], expected["raw_clipped"][name][0]): return False
            if not close(interval["negative_mass"][name][0], 0.0): return False
            if not close(interval["absolute"][name][0], expected["raw_clipped"][name][0]): return False
            if not close(float(expected["raw_clipped"][name][0]), float(expected["raw_unclipped"][name][0]) * factor): return False
        return close(record["global_gradient_norm"], 5.0) and close(record["clip_factor"], factor)
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _wire_close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if isinstance(left, bool) or isinstance(right, bool): return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_wire_close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_wire_close(a, b, tolerance) for a, b in zip(left, right, strict=True))
    return left == right


def _adamw_contract(evidence: Mapping[str, Any]) -> bool:
    """Re-evaluate AdamW moments and v3 movement views from saved roles."""

    try:
        trace, oracle_bundle = evidence["trace_bundle"], evidence["oracle_bundle"]
        if not isinstance(trace, Mapping) or not isinstance(oracle_bundle, Mapping): return False
        actual, expected, checks = trace["production_adamw_trace"], oracle_bundle["oracle"]["adamw_trace"], trace["adamw_checks"]
        if not all(isinstance(value, list) and len(value) == 4 for value in (actual, expected, checks)): return False
        for candidate, reference, check in zip(actual, expected, checks, strict=True):
            if not isinstance(candidate, Mapping) or not isinstance(reference, Mapping) or not isinstance(check, Mapping): return False
            if not _wire_close(candidate, reference) or not all(value is True for key, value in check.items() if key not in {"case_id", "step"}): return False
        decoupled = [row for row in actual if isinstance(row, Mapping) and row.get("case_id") == "decoupled_weight_decay"]
        if len(decoupled) != 2: return False
        after = decoupled[-1]["accumulator_after"]
        if not isinstance(after, Mapping): return False
        get = lambda field: float(after[field]["weight"][0])
        data = sum(float(row["data_delta"]) for row in decoupled)
        decay = sum(float(row["weight_decay_delta"]) for row in decoupled)
        total = sum(float(row["total_delta"]) for row in decoupled)
        return (
            _wire_close(get("data_movement"), sum(abs(float(row["data_delta"])) for row in decoupled))
            and _wire_close(get("total_movement"), sum(abs(float(row["total_delta"])) for row in decoupled))
            and _wire_close(get("weight_decay_movement"), sum(abs(float(row["weight_decay_delta"])) for row in decoupled))
            and _wire_close(get("net_data_movement"), abs(data)) and _wire_close(get("net_weight_decay_movement"), abs(decay))
            and _wire_close(get("total_endpoint_movement"), abs(float(decoupled[-1]["parameter_post"]) - float(decoupled[0]["parameter_pre"])))
            and _wire_close(get("actual_update_raw_importance"), -sum(float(row["data_delta"]) * float(row["gradient"]) for row in decoupled))
            and _wire_close(total, data + decay) and abs(get("net_data_movement") - get("total_endpoint_movement")) > 1e-15
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _accumulator_mass_contract(evidence: Mapping[str, Any]) -> bool:
    try:
        trace = evidence["trace_bundle"]
        if not isinstance(trace, Mapping): return False
        views = [row["accumulator_after"] for row in trace["production_sgd_trace"]]
        views.append(trace["production_sgd_summary"])
        for view in views:
            if not isinstance(view, Mapping): return False
            for name in ("left", "right"):
                for signed, positive, negative, absolute in zip(view["signed"][name], view["positive"][name], view["negative_mass"][name], view["absolute"][name], strict=True):
                    if not (_wire_close(signed, float(positive) - float(negative)) and _wire_close(absolute, float(positive) + float(negative)) and float(positive) >= 0 and float(negative) >= 0 and float(absolute) >= 0): return False
        return _wire_close(trace["production_sgd_trace"][-1]["accumulator_after"], trace["production_sgd_summary"]["v3_roundtrip"])
    except (KeyError, TypeError, ValueError):
        return False


def _skip_contract(evidence: Mapping[str, Any]) -> bool:
    try:
        trace = evidence["trace_bundle"]["nonfinite_skip_trace"]
        if not isinstance(trace, Mapping) or trace["scale_before"] != [8.0, 16.0, 8.0] or trace["scale_after"] != [16.0, 8.0, 16.0]: return False
        if trace["skip_observation"] != {"parameters_unchanged": True, "optimizer_unchanged": True, "accumulator_long_term_unchanged": True, "scaler_present": True}: return False
        reference = trace["reference_comparison"]
        if not isinstance(reference, Mapping) or not reference or not all(value is True for value in reference.values()): return False
        lifecycle = trace["lifecycle"]
        labels = ["01_parameters_optimizer_lr_pre","02_clear_transient_gradients","03_freeze_loss_scale","04_local_unscaled_statistics","05_global_mean_gradient_loss","06_install_scaled_optimizer_gradient","07_formal_unscale","08_global_finite_decision","10_stage_preclip_scores","11_install_clipped_gradient","12_single_optimizer_or_scaler_step","13_freeze_score_mass_payload","14_parameter_post","15_decompose_data_decay_delta","16_commit_all_long_term_views","17_scheduler_success_counter","18_publish_step_log_state"]
        skip = labels[:8] + ["09_skip_discard_and_scaler","17_skip_control_counter","18_publish_step_log_state"]
        chunks = [lifecycle[:17], lifecycle[17:28], lifecycle[28:]]
        if [len(chunk) for chunk in chunks] != [17, 11, 17] or [[item["boundary"] for item in chunk] for chunk in chunks] != [labels, skip, labels]: return False
        events = trace["events"]
        gradients = [event for event in events if event["boundary"] == "gradient_ready"]
        skipped = next(event for event in events if event["boundary"] == "skip")
        commits = [event for event in events if event["boundary"] == "attempt_commit"]
        if len(gradients) != 2 or gradients[-1]["microbatch_ids"] != trace["reference"]["next_batch_ids"] or gradients[-1]["sample_ids"] != trace["reference"]["next_sample_ids"]: return False
        return len(skipped["microbatch_ids"]) == 2 and len(skipped["sample_ids"]) == 4 and [event["cursor_state"].get("index") for event in (commits[:1] + [skipped] + commits[1:])] == [1, 2, 3]
    except (KeyError, StopIteration, TypeError, IndexError):
        return False


def _sgd_engine_contract(evidence: Mapping[str, Any]) -> bool:
    try:
        trace, oracle_bundle = evidence["trace_bundle"], evidence["oracle_bundle"]
        if not isinstance(trace, Mapping) or not isinstance(oracle_bundle, Mapping): return False
        production, oracle = trace["production_sgd_trace"], oracle_bundle["oracle"]["sgd_trace"]
        engine = trace["production_engine_sgd_trace"]
        if (
            not isinstance(production, list)
            or not isinstance(oracle, list)
            or not isinstance(engine, Mapping)
            or len(production) != 2
            or len(oracle) != 2
        ):
            return False
        if not all(_wire_close(actual, expected) for actual, expected in zip(production, oracle, strict=True)): return False
        gradients, posts, commits = engine["gradient_events"], engine["parameter_post_events"], engine["commit_events"]
        if not all(isinstance(value, list) and len(value) == 2 for value in (gradients, posts, commits)): return False
        return all(
            _wire_close(gradient["mean_gradient"], analytic["mean_gradient"])
            and _wire_close(post["data_delta"], analytic["data_delta"])
            and _wire_close(post["weight_decay_delta"], analytic["weight_decay_delta"])
            and _wire_close(post["total_delta"], analytic["total_delta"])
            and _wire_close(commit["accumulator_after"], analytic["accumulator_after"])
            for gradient, post, commit, analytic in zip(gradients, posts, commits, production, strict=True)
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _actual_update_contract(evidence: Mapping[str, Any]) -> bool:
    try:
        trace = evidence["trace_bundle"]
        if not isinstance(trace, Mapping): return False
        steps = trace["production_sgd_trace"]
        summary = trace["production_sgd_summary"]
        if not isinstance(steps, list) or len(steps) != 2 or not isinstance(summary, Mapping): return False
        totals = {name: 0.0 for name in ("left", "right")}
        for step in steps:
            for name in totals:
                expected = -float(step["data_delta"][name][0]) * float(step["mean_gradient"][name][0])
                if not _wire_close(step["actual_update_raw_importance"][name][0], expected): return False
                totals[name] += expected
        return all(_wire_close(summary["actual_update_raw_importance"][name][0], total) for name, total in totals.items()) and summary.get("actual_update_raw_importance_available") is True
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _tiny_parity_contract(evidence: Mapping[str, Any]) -> bool:
    try:
        tiny = evidence["trace_bundle"]["tiny_transformer_trace"]
        if not isinstance(tiny, Mapping): return False
        # The two executions intentionally differ only in score-observer output:
        # ``estimator_name`` and the three boundary observations that contain
        # score/accumulator payload hashes.  Every training-control datum remains
        # byte-for-byte equivalent (the actual equality below permits the
        # registered T32 tolerance solely for numeric wire values).
        observed_records, control_records = tiny["observed_records"], tiny["control_records"]
        if not isinstance(observed_records, list) or not isinstance(control_records, list): return False
        if len(observed_records) != 2 or len(control_records) != 2: return False
        for observed, control in zip(observed_records, control_records, strict=True):
            if not isinstance(observed, Mapping) or not isinstance(control, Mapping): return False
            observed_control = {key: value for key, value in observed.items() if key != "estimator_name"}
            control_control = {key: value for key, value in control.items() if key != "estimator_name"}
            if not _wire_close(observed_control, control_control): return False
        for suffix in ("events", "parameters", "optimizer_state", "scheduler_state", "scaler_state", "scale_before", "scale_after"):
            if not _wire_close(tiny[f"observed_{suffix}"], tiny[f"control_{suffix}"]): return False
        observed_lifecycle, control_lifecycle = tiny["observed_lifecycle"], tiny["control_lifecycle"]
        if not isinstance(observed_lifecycle, list) or not isinstance(control_lifecycle, list): return False
        if len(observed_lifecycle) != 34 or len(control_lifecycle) != 34: return False
        observer_only = {
            "10_stage_preclip_scores": {"estimator", "staged_scores_hash", "raw_unclipped_hash"},
            "13_freeze_score_mass_payload": {"score_payload_hash"},
            "16_commit_all_long_term_views": {"accumulator_hash"},
        }
        for observed, control in zip(observed_lifecycle, control_lifecycle, strict=True):
            if not isinstance(observed, Mapping) or not isinstance(control, Mapping): return False
            if tuple(observed.get(key) for key in ("sequence", "boundary", "attempt_index", "global_step")) != tuple(control.get(key) for key in ("sequence", "boundary", "attempt_index", "global_step")): return False
            boundary = observed.get("boundary")
            if boundary not in observer_only and not _wire_close(observed.get("observation"), control.get("observation")): return False
            if boundary in observer_only:
                left, right = observed.get("observation"), control.get("observation")
                if not isinstance(left, Mapping) or not isinstance(right, Mapping): return False
                ignored = observer_only[boundary]
                if not _wire_close({key: value for key, value in left.items() if key not in ignored}, {key: value for key, value in right.items() if key not in ignored}): return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


_CSV_FIELDS = ("section", "case_id", "step", "field", "coordinate", "index", "actual", "reference", "absolute_error", "passed")


def _chart_projections(evidence: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    """Exact, semantic chart projections from the persisted trace role."""

    trace_role = evidence["trace_bundle"]
    if not isinstance(trace_role, Mapping) or not isinstance(trace_role.get("production_sgd_trace"), list):
        raise Stage1S16FormalError("S1_6_CHART_TRACE_MISSING")
    identities: list[Mapping[str, Any]] = []
    movements: list[Mapping[str, Any]] = []
    masses: list[Mapping[str, Any]] = []
    for step_trace in trace_role["production_sgd_trace"]:
        if not isinstance(step_trace, Mapping) or not isinstance(step_trace.get("step"), int):
            raise Stage1S16FormalError("S1_6_CHART_STEP_TRACE_INVALID")
        after, interval = step_trace.get("accumulator_after"), step_trace.get("accumulator_interval_delta")
        if not isinstance(after, Mapping) or not isinstance(interval, Mapping):
            raise Stage1S16FormalError("S1_6_CHART_ACCUMULATOR_TRACE_INVALID")
        for coordinate in ("left", "right"):
            for index, signed in enumerate(after["signed"][coordinate]):
                signed_residual = float(signed) - (float(after["positive"][coordinate][index]) - float(after["negative_mass"][coordinate][index]))
                absolute_residual = float(after["absolute"][coordinate][index]) - (float(after["positive"][coordinate][index]) + float(after["negative_mass"][coordinate][index]))
                for field, residual in (("signed_identity_residual", signed_residual), ("absolute_identity_residual", absolute_residual)):
                    identities.append({"section": "accumulator_identity", "step": step_trace["step"], "field": field, "coordinate": coordinate, "index": index, "actual": residual, "reference": 0.0, "absolute_error": abs(residual), "passed": abs(residual) <= 1e-12})
                for field in ("raw", "signed", "positive", "negative_mass"):
                    value = float(after[field][coordinate][index])
                    masses.append({"section": "score_mass", "step": step_trace["step"], "field": field, "coordinate": coordinate, "index": index, "actual": value, "reference": value, "absolute_error": 0.0, "passed": True})
                for field in ("data_movement", "weight_decay_movement", "total_movement"):
                    value = float(interval[field][coordinate][index])
                    if value < 0.0: raise Stage1S16FormalError("S1_6_CHART_MOVEMENT_NEGATIVE")
                    movements.append({"section": "movement_stack", "step": step_trace["step"], "field": field, "coordinate": coordinate, "index": index, "actual": value, "reference": value, "absolute_error": 0.0, "passed": True})
    tiny = trace_role.get("tiny_transformer_trace")
    if not isinstance(tiny, Mapping) or not isinstance(tiny.get("observed_events"), list) or not isinstance(tiny.get("control_events"), list):
        raise Stage1S16FormalError("S1_6_CHART_TINY_TRACE_INVALID")
    observed = [row for row in tiny["observed_events"] if isinstance(row, Mapping) and row.get("boundary") == "parameter_post"]
    control = [row for row in tiny["control_events"] if isinstance(row, Mapping) and row.get("boundary") == "parameter_post"]
    if len(observed) != 2 or len(control) != 2: raise Stage1S16FormalError("S1_6_CHART_TINY_POST_COUNT_INVALID")
    trajectory: list[Mapping[str, Any]] = []
    for step, (left, right) in enumerate(zip(observed, control, strict=True), start=1):
        params_left, params_right = left.get("parameters_post"), right.get("parameters_post")
        if not isinstance(params_left, Mapping) or not isinstance(params_right, Mapping) or set(params_left) != set(params_right): raise Stage1S16FormalError("S1_6_CHART_TINY_PARAMETER_SET_INVALID")
        maximum = max(abs(float(a) - float(b)) for name in params_left for a, b in zip(params_left[name], params_right[name], strict=True))
        trajectory.append({"section": "tiny_parameter_parity", "step": step, "field": "max_parameter_error", "coordinate": "all", "index": 0, "actual": maximum, "reference": 0.0, "absolute_error": maximum, "passed": maximum == 0.0})
    return {"accumulator-identities.csv": identities, "movement-stack.csv": movements, "trajectory-parity.csv": trajectory, "score-masses.csv": masses}


def _csv_cell(value: object) -> str:
    return "" if value is None else str(value)


def _verify_charts(work: Path, evidence: Mapping[str, Any], csv_hashes: Mapping[str, str], svg_hashes: Mapping[str, str]) -> bool:
    """Re-read exact tabular projections and SVG geometry after serialization."""

    projections = _chart_projections(evidence)
    if set(csv_hashes) != set(projections) or set(svg_hashes) != {name.replace(".csv", ".svg") for name in projections}:
        return False
    for name, selected in projections.items():
        csv_path, svg_path = work / name, work / name.replace(".csv", ".svg")
        if not csv_path.is_file() or not svg_path.is_file() or _sha(csv_path) != csv_hashes[name] or _sha(svg_path) != svg_hashes[svg_path.name]:
            return False
        with csv_path.open(encoding="utf-8", newline="") as handle:
            actual = list(csv.DictReader(handle))
        expected = [{field: _csv_cell(row.get(field)) for field in _CSV_FIELDS} for row in selected]
        if actual != expected:
            return False
        svg = svg_path.read_text(encoding="utf-8")
        if f'data-source="{name}"' not in svg or '<line class="x-axis"' not in svg or '<line class="y-axis"' not in svg:
            return False
        mark_class = "stack-bar" if name == "movement-stack.csv" else "identity-point" if name == "accumulator-identities.csv" else "mass-point" if name == "score-masses.csv" else "trace-mark"
        if svg.count(f'class="{mark_class}"') != len(selected):
            return False
        if any(f'data-row="{index}"' not in svg for index in range(len(selected))):
            return False
        if name == "trajectory-parity.csv" and ("<polyline class=\"trajectory\"" not in svg or "points=\"\"" in svg):
            return False
        if name == "accumulator-identities.csv" and svg.count('class="identity-series"') != 4:
            return False
        if name == "score-masses.csv" and svg.count('class="mass-series"') != 8:
            return False
    return True


def _charts(work: Path, evidence: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    projections = _chart_projections(evidence)
    csv_hashes: dict[str, str] = {}; svg_hashes: dict[str, str] = {}
    for name, selected in projections.items():
        path = work / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
            writer.writeheader(); writer.writerows(selected)
        csv_hashes[name] = _sha(path)
        svg = work / name.replace(".csv", ".svg")
        marks: list[str] = ['<line class="x-axis" x1="55" y1="170" x2="760" y2="170" stroke="black"/>', '<line class="y-axis" x1="55" y1="25" x2="55" y2="170" stroke="black"/>']
        values = [float(row["actual"]) for row in selected] or [0.0]
        group_positions: dict[tuple[object, object], int] = {}
        if name == "movement-stack.csv":
            for row in selected:
                group_positions.setdefault((row["step"], row["coordinate"]), len(group_positions))
            totals = {group: sum(float(row["actual"]) for row in selected if (row["step"], row["coordinate"]) == group) for group in group_positions}
            scale = max(max(totals.values(), default=0.0), 1.0)
        else:
            scale = max(max(abs(value) for value in values), 1.0)
        points: list[str] = []
        series: dict[tuple[object, object], list[str]] = {}
        cumulative: dict[tuple[object, object], float] = {}
        for index, row in enumerate(selected):
            if name == "movement-stack.csv":
                x = 65 + group_positions[(row["step"], row["coordinate"])] * (680 / max(len(group_positions), 1))
            else:
                x = 65 + index * (680 / max(len(selected), 1))
            value = float(row["actual"]); y = 165 - value / scale * 120
            points.append(f"{x:.3f},{y:.3f}")
            if name == "movement-stack.csv":
                key = (row["step"], row["coordinate"]); start = cumulative.get(key, 0.0); cumulative[key] = start + value
                height = value / scale * 120; stack_y = 165 - (start + value) / scale * 120
                marks.append(f'<rect class="stack-bar" x="{x:.3f}" y="{stack_y:.3f}" width="8" height="{height:.3f}" data-row="{index}"/>')
            elif name == "accumulator-identities.csv":
                key = (row["field"], row["coordinate"]); series.setdefault(key, []).append(f"{x:.3f},{y:.3f}")
                marks.append(f'<circle class="identity-point" cx="{x:.3f}" cy="{y:.3f}" r="3" data-row="{index}"/>')
            elif name == "score-masses.csv":
                key = (row["field"], row["coordinate"]); series.setdefault(key, []).append(f"{x:.3f},{y:.3f}")
                marks.append(f'<circle class="mass-point" cx="{x:.3f}" cy="{y:.3f}" r="3" data-row="{index}"/>')
            else: marks.append(f'<circle class="trace-mark" cx="{x:.3f}" cy="{y:.3f}" r="3" data-row="{index}"/>')
        if name == "trajectory-parity.csv": marks.append(f'<polyline class="trajectory" points="{" ".join(points)}" fill="none" stroke="#1565c0"/>')
        if name == "accumulator-identities.csv":
            for (field, coordinate), series_points in series.items(): marks.append(f'<polyline class="identity-series" data-series="{field}:{coordinate}" points="{" ".join(series_points)}" fill="none" stroke="#1565c0"/>')
        if name == "score-masses.csv":
            for (field, coordinate), series_points in series.items(): marks.append(f'<polyline class="mass-series" data-series="{field}:{coordinate}" points="{" ".join(series_points)}" fill="none" stroke="#1565c0"/>')
        svg.write_text(f'<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://www.w3.org/2000/svg" width="800" height="200" data-source="{name}"><title>{name}</title><desc>Exact projection of {name}</desc>{"".join(marks)}</svg>\n', encoding="utf-8")
        svg_hashes[svg.name] = _sha(svg)
    return csv_hashes, svg_hashes


def execute(*, repository: str | Path, data_root: str | Path, s1_5_index_ref: str, attempt_id: str, timeout_seconds: int = 1800) -> dict[str, str]:
    repository_root, root = Path(repository).resolve(strict=True), Path(data_root).resolve(strict=True)
    if str(repository_root / "src") not in sys.path: sys.path.insert(0, str(repository_root / "src"))
    commit = _git(repository_root, "rev-parse", "HEAD")
    if _COMMIT.fullmatch(commit) is None or _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S16FormalError("S1_6_FORMAL_REQUIRES_CLEAN_WORKTREE")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", attempt_id) is None:
        raise Stage1S16FormalError("S1_6_ATTEMPT_ID_INVALID")
    upstream = _load_s15(root, s1_5_index_ref, repository_root)
    s16_upstream = {key: upstream[key] for key in ("s1_5_index_ref", "s1_5_index_sha256", "s1_5_gate_artifact_hash", "s1_5_replay_sha256")}
    target = root / "evidence" / "stage1" / "s1-6-formal" / commit / attempt_id
    work = root / "tmp" / "stage1-s1-6" / commit / attempt_id
    if target.exists() or work.exists(): raise Stage1S16FormalError("S1_6_ATTEMPT_ALREADY_EXISTS")
    work.mkdir(parents=True)
    command = [sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--basetemp", str(work / "pytest-tmp"), "tests/test_stage1_s16_training_integration.py", "tests/test_stage1_s16_handoff_and_charts.py", "tests/test_runtime_training_engine.py", "tests/test_core_estimators_and_accumulator.py", "tests/test_stage79_run_ready_completion.py"]
    completed = subprocess.run(command, cwd=repository_root, text=True, capture_output=True, check=False, timeout=timeout_seconds)
    if completed.returncode: raise Stage1S16FormalError("S1_6_REGRESSION_FAILED")
    regression = {"schema_version": "stage1-s1-6-regression-v1", "command": command, "returncode": 0, "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(), "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(), "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-2000:]}
    from param_importance_nlp.stage1_training_integration import build_stage1_s16_evidence, replay_stage1_s16_evidence, validate_stage1_s16_evidence
    evidence = build_stage1_s16_evidence(repository_root, producer_commit=commit, scope="formal", upstream_evidence=s16_upstream)
    replay = replay_stage1_s16_evidence(evidence, source_root=repository_root)
    filenames = {"step_report": "step-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-step-record.json"}
    paths = {role: work / name for role, name in filenames.items()}
    for role, path in paths.items(): _write(path, evidence[role])
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    persisted_evidence = {role: load_canonical_json(path) for role, path in paths.items()}
    if not all(isinstance(value, Mapping) for value in persisted_evidence.values()):
        raise Stage1S16FormalError("S1_6_PERSISTED_ROLE_NOT_OBJECT")
    reloaded_evidence: dict[str, Mapping[str, Any]] = {role: value for role, value in persisted_evidence.items() if isinstance(value, Mapping)}
    _validate_role_schemas(repository_root, reloaded_evidence)
    validate_stage1_s16_evidence(reloaded_evidence, source_root=repository_root)
    replay_path = work / "replay-validation.json"; _write(replay_path, replay)
    persisted_replay = load_canonical_json(replay_path)
    if not isinstance(persisted_replay, Mapping) or replay_stage1_s16_evidence(reloaded_evidence, source_root=repository_root) != persisted_replay:
        raise Stage1S16FormalError("S1_6_PERSISTED_REPLAY_MISMATCH")
    evidence, replay = reloaded_evidence, persisted_replay
    csv_hashes, svg_hashes = _charts(work, evidence)
    requirements = evidence["gate_record"].get("requirements")
    if not isinstance(requirements, Mapping): raise Stage1S16FormalError("S1_6_GATE_REQUIREMENTS_MISSING")
    derived = {
        "s1_5_handoff_closed": upstream["s1_5_index_sha256"] == EXPECTED_INDEX_SHA256 and upstream["s1_5_gate_artifact_hash"] == EXPECTED_GATE_HASH,
        "s1_5_consumer_semantic_replay": upstream.get("s1_5_numeric_replay_role_count") == 5,
        "fixture_schema_and_hash": _fixture_schema_and_hash(repository_root),
        "independent_oracle_isolation": _oracle_isolation(repository_root, evidence["oracle_bundle"]),
        "sgd_multi_group_replay": bool(requirements.get("sgd_offline_replay")) and bool(requirements.get("multi_group_actual_learning_rates")) and bool(requirements.get("sgd_training_engine_integration")) and _sgd_engine_contract(evidence),
        "nontrivial_clip_exactly_once": bool(requirements.get("nontrivial_clip_exactly_once")) and _nontrivial_clip_contract(evidence),
        "adamw_delta_decomposition": bool(requirements.get("adamw_data_decay_total_decomposition")) and _adamw_contract(evidence),
        "accumulator_identities": bool(requirements.get("signed_mass_identities")) and _accumulator_mass_contract(evidence),
        "actual_update_diagnostic": bool(requirements.get("actual_update_diagnostic_boundary")) and _actual_update_contract(evidence),
        "skip_atomicity": bool(requirements.get("skip_discards_staged_long_term_state")) and _skip_contract(evidence),
        "tiny_transformer_observation_parity": bool(requirements.get("statistics_do_not_perturb_training_path")) and _tiny_parity_contract(evidence),
        "chart_projection_exact": _verify_charts(work, evidence, csv_hashes, svg_hashes),
        "offline_replay": replay.get("status") == "PASS" and replay.get("source_report_hash") == evidence["step_report"].get("report_hash"),
    }
    checks = [{"check_id": check, "status": "PASS" if passed else "FAIL", "detail": "recomputed from persisted role/chart input"} for check, passed in derived.items()]
    if not all(derived.values()): raise Stage1S16FormalError("S1_6_DIRECT_CHECK_FAILED")
    validation = _with_hash({"schema_version": "stage1-s1-6-validation-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "execution_scope": "formal_server_cpu", "fixture_id": FIXTURE_ID, "producer_commit": commit, "consumer_commit": commit, "upstream": s16_upstream, "regression": regression, "direct_checks": checks, "role_sha256": {role: _sha(path) for role, path in paths.items()}, "csv_sha256": csv_hashes, "svg_sha256": svg_hashes, "replay_sha256": _sha(replay_path), "replay_hash": replay["replay_hash"]})
    _validate_role_schemas(repository_root, {"validation": validation})
    validation_path = work / "validation.json"; _write(validation_path, validation)
    index = _with_hash({"schema_version": "stage1-s1-6-formalization-index-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "generator_git_commit": commit, "consumer_git_commit": commit, "git_branch": _git(repository_root, "branch", "--show-current"), "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), **s16_upstream, "s1_5_handoff": upstream, "role_refs": filenames, "role_sha256": {role: _sha(path) for role, path in paths.items()}, "gate_artifact_hash": evidence["gate_record"]["artifact_hash"], "csv_sha256": csv_hashes, "svg_sha256": svg_hashes, "validation_ref": "validation.json", "validation_sha256": _sha(validation_path), "replay_ref": "replay-validation.json", "replay_sha256": _sha(replay_path), "replay_hash": replay["replay_hash"], "next_task_id": "stage1.07_single_gpu_pythia14m"})
    _validate_role_schemas(repository_root, {"index": index})
    staging = target.parent / f".{attempt_id}.publishing"; staging.mkdir(parents=True)
    for path in [*paths.values(), replay_path, validation_path, *(work / name for name in csv_hashes), *(work / name for name in svg_hashes)]: shutil.copy2(path, staging / path.name)
    _write(staging / "index.json", index); os.replace(staging, target)
    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
    published_index = load_canonical_json(target / "index.json")
    if not isinstance(published_index, Mapping): raise Stage1S16FormalError("S1_6_PUBLISH_INDEX_NOT_OBJECT")
    index_body = dict(published_index); published_hash = index_body.pop("artifact_hash", None)
    if published_hash != canonical_json_hash(index_body) or published_hash != index["artifact_hash"]: raise Stage1S16FormalError("S1_6_PUBLISH_INDEX_HASH_INVALID")
    _validate_role_schemas(repository_root, {"index": published_index})
    for role, filename in filenames.items():
        if _sha(target / filename) != published_index["role_sha256"][role]: raise Stage1S16FormalError(f"S1_6_PUBLISH_ROLE_HASH_INVALID:{role}")
        published_role = load_canonical_json(target / filename)
        if not isinstance(published_role, Mapping): raise Stage1S16FormalError(f"S1_6_PUBLISH_ROLE_NOT_OBJECT:{role}")
        _validate_role_schemas(repository_root, {role: published_role})
    if _sha(target / "validation.json") != published_index["validation_sha256"] or _sha(target / "replay-validation.json") != published_index["replay_sha256"]: raise Stage1S16FormalError("S1_6_PUBLISH_AUXILIARY_HASH_INVALID")
    published_validation = load_canonical_json(target / "validation.json")
    if not isinstance(published_validation, Mapping): raise Stage1S16FormalError("S1_6_PUBLISH_VALIDATION_NOT_OBJECT")
    _validate_role_schemas(repository_root, {"validation": published_validation})
    validation_body = dict(published_validation); validation_hash = validation_body.pop("artifact_hash", None)
    if validation_hash != canonical_json_hash(validation_body): raise Stage1S16FormalError("S1_6_PUBLISH_VALIDATION_SELF_HASH_INVALID")
    published_replay = load_canonical_json(target / "replay-validation.json")
    if not isinstance(published_replay, Mapping): raise Stage1S16FormalError("S1_6_PUBLISH_REPLAY_NOT_OBJECT")
    replay_body = dict(published_replay); replay_hash = replay_body.pop("replay_hash", None)
    if replay_hash != canonical_json_hash(replay_body): raise Stage1S16FormalError("S1_6_PUBLISH_REPLAY_SELF_HASH_INVALID")
    if (
        published_validation.get("role_sha256") != published_index.get("role_sha256")
        or published_validation.get("csv_sha256") != published_index.get("csv_sha256")
        or published_validation.get("svg_sha256") != published_index.get("svg_sha256")
        or published_validation.get("replay_sha256") != published_index.get("replay_sha256")
        or published_validation.get("replay_hash") != published_index.get("replay_hash")
        or published_replay.get("source_report_hash") != reloaded_evidence["step_report"].get("report_hash")
        or published_replay.get("source_oracle_hash") != reloaded_evidence["oracle_bundle"].get("oracle_hash")
        or published_replay.get("source_trace_hash") != reloaded_evidence["trace_bundle"].get("trace_hash")
        or published_replay.get("source_table_hash") != reloaded_evidence["comparison_table"].get("table_hash")
        or published_replay.get("source_gate_artifact_hash") != reloaded_evidence["gate_record"].get("artifact_hash")
    ): raise Stage1S16FormalError("S1_6_PUBLISH_CROSS_BINDING_INVALID")
    if not _verify_charts(target, evidence, published_index["csv_sha256"], published_index["svg_sha256"]): raise Stage1S16FormalError("S1_6_PUBLISH_CHART_PROJECTION_INVALID")
    # Keep the validated staging material for server-side audit.  Failed and
    # successful attempts are both immutable operator evidence; this command
    # never deletes a caller-owned temporary directory.
    return {"index_ref": (target / "index.json").relative_to(root).as_posix(), "validation_ref": (target / "validation.json").relative_to(root).as_posix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-5-index-ref", required=True); parser.add_argument("--attempt-id", required=True); parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    print(execute(repository=args.repository, data_root=args.data_root, s1_5_index_ref=args.s1_5_index_ref, attempt_id=args.attempt_id, timeout_seconds=args.timeout_seconds)); return 0


if __name__ == "__main__": raise SystemExit(main())
