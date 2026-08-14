#!/usr/bin/env python3
"""Publish formal CPU evidence for Stage 1 S1.3 / G1-ORACLE.

This command is intentionally CPU-only.  It consumes the immutable S1.2 handoff,
generates the committed deterministic fixture set, validates the three role
payloads after a canonical save/load roundtrip, and publishes a new immutable
attempt under ``$DATA_ROOT/evidence``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
TASK_ID = "stage1.03_fixtures_and_oracles"
INDEX_SCHEMA = "stage1-s1-3-formalization-index-v1"
VALIDATION_SCHEMA = "stage1-s1-3-validation-v1"
REQUIRED_UPSTREAM_GATES = ("stage1.G1-ENTRY", "stage1.G1-CONTRACT", "stage1.G1-REGISTRY")


class Stage1S13FormalError(RuntimeError):
    """S1.3 formal evidence cannot be safely published."""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise Stage1S13FormalError(f"S1_3_GIT_COMMAND_FAILED:{arguments[0]}")
    return completed.stdout.strip()


def _logical_path(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S13FormalError(f"S1_3_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S13FormalError(f"S1_3_LOGICAL_REF_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S13FormalError(f"S1_3_LOGICAL_REF_ESCAPE:{field}") from error
    return path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Stage1S13FormalError(f"S1_3_IMMUTABLE_TARGET_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    from param_importance_nlp.contracts.jsonio import write_canonical_json

    write_canonical_json(path, dict(value))


def _with_artifact_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash

    payload = dict(value)
    payload["artifact_hash"] = canonical_json_hash(value)
    return payload


def _load_s1_2_handoff(data_root: Path, index_ref: str) -> dict[str, Any]:
    from param_importance_nlp.atomic import sha256_file
    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json

    index_path = _logical_path(data_root, index_ref, field="s1_2_index_ref")
    raw = load_canonical_json(index_path)
    if not isinstance(raw, dict):
        raise Stage1S13FormalError("S1_3_S1_2_INDEX_NOT_OBJECT")
    body = dict(raw)
    declared_artifact_hash = body.pop("artifact_hash", None)
    if declared_artifact_hash != canonical_json_hash(body):
        raise Stage1S13FormalError("S1_3_S1_2_INDEX_HASH_INVALID")
    if (
        raw.get("schema_version") != "stage1-s1-2-formalization-index-v1"
        or raw.get("status") != "PASS"
        or raw.get("gate_id") != "G1-REGISTRY"
        or raw.get("task_id") != "stage1.02_architecture_and_parameter_registry"
        or raw.get("next_task_id") != TASK_ID
    ):
        raise Stage1S13FormalError("S1_3_S1_2_HANDOFF_NOT_READY")
    producer_commit = raw.get("generator_git_commit")
    if not isinstance(producer_commit, str) or _COMMIT_RE.fullmatch(producer_commit) is None:
        raise Stage1S13FormalError("S1_3_S1_2_PRODUCER_COMMIT_INVALID")
    upstream_gate_hashes = raw.get("s1_1_gate_artifact_hashes")
    if not isinstance(upstream_gate_hashes, dict):
        raise Stage1S13FormalError("S1_3_S1_1_GATE_HASHES_MISSING")
    for gate_id in ("stage1.G1-ENTRY", "stage1.G1-CONTRACT"):
        digest = upstream_gate_hashes.get(gate_id)
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise Stage1S13FormalError(f"S1_3_UPSTREAM_GATE_INVALID:{gate_id}")

    report_ref = raw.get("report_ref")
    report_sha256 = raw.get("report_sha256")
    if not isinstance(report_ref, str) or not isinstance(report_sha256, str):
        raise Stage1S13FormalError("S1_3_S1_2_REPORT_REF_MISSING")
    report_path = _logical_path(data_root, report_ref, field="s1_2_report_ref")
    if not report_path.is_file():
        # S1.2 stores short artifact names in its index; those names are
        # relative to the immutable S1.2 attempt directory, not DATA_ROOT.
        report_path = (index_path.parent / report_ref).resolve()
        try:
            report_path.relative_to(data_root.resolve())
        except ValueError as error:
            raise Stage1S13FormalError("S1_3_S1_2_REPORT_REF_ESCAPE") from error
    if report_sha256 != sha256_file(report_path):
        raise Stage1S13FormalError("S1_3_S1_2_REPORT_SHA256_MISMATCH")
    report = load_canonical_json(report_path)
    if not isinstance(report, dict) or report.get("status") != "PASS" or report.get("gate_id") != "G1-REGISTRY":
        raise Stage1S13FormalError("S1_3_S1_2_REPORT_NOT_PASS")

    raw["_index_path"] = index_path
    raw["_index_sha256"] = sha256_file(index_path)
    raw["_report_path"] = report_path
    raw["_report_sha256"] = report_sha256
    raw["_upstream_evidence"] = {
        "s1_2_index_ref": index_ref,
        "s1_2_index_sha256": raw["_index_sha256"],
        "s1_2_index_artifact_hash": declared_artifact_hash,
        "s1_2_report_ref": report_ref,
        "s1_2_report_sha256": report_sha256,
        "s1_2_gate_artifact_hash": declared_artifact_hash,
        "s1_1_gate_artifact_hashes": {
            gate_id: upstream_gate_hashes[gate_id]
            for gate_id in ("stage1.G1-ENTRY", "stage1.G1-CONTRACT")
        },
    }
    return raw


def _run_regression(repository_root: Path, work_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    test_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "--basetemp",
        str(work_dir / "pytest-tmp"),
        "tests/test_stage1_s13_fixtures.py",
        "tests/test_stage01_task_runners.py",
        "tests/test_core_oracles.py",
        "tests/test_stage1_architecture_registry.py",
        "tests/test_core_estimators_and_accumulator.py",
        "-k",
        "s13 or fixtures_and_oracles or core_oracles or architecture_registry or estimator_runner",
    ]
    completed = subprocess.run(
        test_command,
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise Stage1S13FormalError(
            f"S1_3_SERVER_REGRESSION_FAILED:returncode={completed.returncode}"
        )
    return {
        "schema_version": "stage1-s1-3-regression-v1",
        "command": test_command,
        "returncode": completed.returncode,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _direct_checks(repository_root: Path, evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash
    from param_importance_nlp.stage1_fixtures import validate_stage1_s13_evidence

    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            raise Stage1S13FormalError(f"S1_3_DIRECT_CHECK_FAILED:{check_id}")

    manifest = evidence["fixture_manifest"]
    bundle = evidence["oracle_bundle"]
    report = evidence["oracle_validation_report"]
    manifest_body = dict(manifest)
    declared_manifest_hash = manifest_body.pop("manifest_hash", None)
    check(
        "manifest_self_hash",
        declared_manifest_hash == canonical_json_hash(manifest_body),
        str(declared_manifest_hash),
    )
    check(
        "role_hash_bindings",
        bundle["fixture_manifest_hash"] == declared_manifest_hash
        and report["oracle_bundle_hash"] == bundle["bundle_hash"],
        f"manifest={declared_manifest_hash};bundle={bundle['bundle_hash']}",
    )
    check(
        "oracle_bundle_is_offline",
        bundle["offline_recompute"] == {
            "algorithm": "explicit_fp64_loops_over_serialized_inputs",
            "formal_estimator_imported": False,
            "recomputed_from_saved_values": True,
        },
        str(bundle["offline_recompute"]),
    )
    source = (repository_root / "src/param_importance_nlp/stage1_fixtures.py").read_text(encoding="utf-8")
    check("oracle_has_no_production_estimator_import", "core.estimators" not in source, "source import scan")
    check("g1_oracle_checks_pass", report["status"] == "PASS" and report["check_count"] == report["passed_check_count"], str(report["check_count"]))
    check("pythia_descriptor_not_consumed", report["pythia_14m"]["consumed_by_gate"] is False, str(report["pythia_14m"]))
    replay = validate_stage1_s13_evidence(evidence)
    check("offline_serialized_replay", replay["report_status"] == "PASS", str(replay["replay_hash"]))
    return checks


def execute(
    *,
    repository: str | Path,
    data_root: str | Path,
    s1_2_index_ref: str,
    attempt_id: str,
    timeout_seconds: int,
) -> dict[str, str]:
    repository_root = Path(repository).resolve(strict=True)
    data_root_path = Path(data_root).resolve(strict=True)
    source_root = repository_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    commit = _git(repository_root, "rev-parse", "HEAD")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise Stage1S13FormalError("S1_3_REPOSITORY_COMMIT_INVALID")
    if _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S13FormalError("S1_3_FORMAL_REQUIRES_CLEAN_WORKTREE")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", attempt_id) is None:
        raise Stage1S13FormalError("S1_3_ATTEMPT_ID_INVALID")
    upstream = _load_s1_2_handoff(data_root_path, s1_2_index_ref)

    evidence_dir = data_root_path / "evidence" / "stage1" / "s1-3-formal" / commit / attempt_id
    if evidence_dir.exists():
        raise Stage1S13FormalError(f"S1_3_ATTEMPT_ALREADY_EXISTS:{evidence_dir}")
    work_dir = data_root_path / "tmp" / "stage1-s1-3" / commit / attempt_id
    if work_dir.exists():
        raise Stage1S13FormalError(f"S1_3_WORK_ATTEMPT_ALREADY_EXISTS:{work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)

    regression = _run_regression(repository_root, work_dir, timeout_seconds)
    from param_importance_nlp.stage1_fixtures import build_stage1_s13_evidence

    evidence = build_stage1_s13_evidence(
        repository_root,
        producer_commit=commit,
        scope="formal",
        upstream_evidence=upstream["_upstream_evidence"],
    )
    direct_checks = _direct_checks(repository_root, evidence)

    role_files = {
        "fixture_manifest": ("fixture-manifest.json", evidence["fixture_manifest"]),
        "oracle_bundle": ("oracle-bundle.json", evidence["oracle_bundle"]),
        "oracle_validation_report": ("oracle-validation-report.json", evidence["oracle_validation_report"]),
    }
    role_paths: dict[str, Path] = {}
    for role, (filename, payload) in role_files.items():
        path = work_dir / filename
        _write_new(path, payload)
        role_paths[role] = path

    from param_importance_nlp.atomic import sha256_file
    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json

    loaded_evidence = {
        role: load_canonical_json(path)
        for role, path in role_paths.items()
    }
    replay = _direct_checks(repository_root, loaded_evidence)
    replay_validation = validate_stage1_s13_evidence(loaded_evidence)
    replay_path = work_dir / "oracle-replay-validation.json"
    _write_new(replay_path, replay_validation)

    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "gate_id": "G1-ORACLE",
        "task_id": TASK_ID,
        "execution_scope": "formal_server_cpu",
        "producer_commit": commit,
        "consumer_commit": commit,
        "upstream": upstream["_upstream_evidence"],
        "regression": regression,
        "direct_checks": direct_checks,
        "role_sha256": {role: sha256_file(path) for role, path in role_paths.items()},
        "replay_sha256": sha256_file(replay_path),
        "replay_hash": replay_validation["replay_hash"],
    }
    validation = _with_artifact_hash(validation)
    validation_path = work_dir / "validation.json"
    _write_new(validation_path, validation)

    index = {
        "schema_version": INDEX_SCHEMA,
        "status": "PASS",
        "gate_id": "G1-ORACLE",
        "task_id": TASK_ID,
        "generator_git_commit": commit,
        "consumer_git_commit": commit,
        "git_branch": _git(repository_root, "branch", "--show-current"),
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "s1_2_index_ref": s1_2_index_ref,
        "s1_2_index_sha256": upstream["_index_sha256"],
        "s1_2_gate_artifact_hash": upstream["_upstream_evidence"]["s1_2_gate_artifact_hash"],
        "fixture_manifest_ref": "fixture-manifest.json",
        "fixture_manifest_sha256": sha256_file(role_paths["fixture_manifest"]),
        "oracle_bundle_ref": "oracle-bundle.json",
        "oracle_bundle_sha256": sha256_file(role_paths["oracle_bundle"]),
        "oracle_validation_report_ref": "oracle-validation-report.json",
        "oracle_validation_report_sha256": sha256_file(role_paths["oracle_validation_report"]),
        "oracle_bundle_hash": evidence["oracle_bundle"]["bundle_hash"],
        "validation_ref": "validation.json",
        "validation_sha256": sha256_file(validation_path),
        "replay_ref": "oracle-replay-validation.json",
        "replay_sha256": sha256_file(replay_path),
        "replay_hash": replay_validation["replay_hash"],
        "next_task_id": "stage1.04_loss_and_gradient_scale",
    }
    index = _with_artifact_hash(index)

    evidence_dir.mkdir(parents=True, exist_ok=False)
    for source in (*role_paths.values(), replay_path, validation_path):
        (evidence_dir / source.name).write_bytes(source.read_bytes())
    index_path = evidence_dir / "index.json"
    write_canonical_json(index_path, index)
    loaded_index = load_canonical_json(index_path)
    if not isinstance(loaded_index, dict) or loaded_index.get("artifact_hash") != index["artifact_hash"]:
        raise Stage1S13FormalError("S1_3_INDEX_RELOAD_FAILED")
    shutil.rmtree(work_dir)
    return {
        "index_ref": index_path.relative_to(data_root_path).as_posix(),
        "oracle_bundle_ref": (evidence_dir / "oracle-bundle.json").relative_to(data_root_path).as_posix(),
        "validation_ref": (evidence_dir / "validation.json").relative_to(data_root_path).as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-2-index-ref", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    arguments = parser.parse_args(argv)
    result = execute(
        repository=arguments.repository,
        data_root=arguments.data_root,
        s1_2_index_ref=arguments.s1_2_index_ref,
        attempt_id=arguments.attempt_id,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
