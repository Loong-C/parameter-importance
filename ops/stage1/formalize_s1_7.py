"""Fail-closed formalizer for the real single-A100 Pythia-14M S1.7 gate.

The command never chooses a GPU.  The operator supplies an exact approved GPU
UUID, which must appear in a current, hash-bound capability record.  Full Pile
hashing and fixture extraction happen before the project lease; the leased
child sees that UUID only as logical ``cuda:0``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


TASK_ID = "stage1.07_single_gpu_pythia14m"
GATE_ID = "G1-SINGLE"
FIXTURE_ID = "stage1-s17-pythia14m-pile16-v1"
EXPECTED_S1_6_INDEX_SHA256 = "9b25d016bdfb82a75d46b3fa89432211314e72e9a63522c772c2da1a764e69bc"
EXPECTED_S1_6_GATE_HASH = "872775ca7b74d5948019bb47ee070ec5f5cf81dc2127ca16d805944787a61a64"
EXPECTED_S1_6_PRODUCER = "ceb202ec3278c60111a31d82692b36eb98de6a40"
HISTORICAL_G3_PRODUCER = "54b1c7f87eda0533b29622b39cc8a7ec90646d0b"
EXPECTED_G3_RESOLUTION = "evidence/stage0/tasks/04-a3bc369bcb6f7dd2ba7dbd83a59d518d64d4431e355150c92d8a0cda02cb2a92/commits/asset_resolution.json"
EXPECTED_G3_COMMIT_ARTIFACT_HASH = "418e9a60c25edfc275fe459b333bf4893912d03b9331b08dc9afb3e1560ea5cd"
EXPECTED_G3_PAYLOAD_HASH = "a3bc369bcb6f7dd2ba7dbd83a59d518d64d4431e355150c92d8a0cda02cb2a92"
EXPECTED_GPU_CAPABILITY_REF = "evidence/stage0/bootstrap/a15f0e2970b7cae6951dd606ebd396a8df68255c/commits/capability_cuda.json"
EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH = "a536e191cd59318325289d238db727f8939767e384bfccd961ae7ca1c6a11ce4"
EXPECTED_GPU_CAPABILITY_FILE_SHA256 = "1d5f28369f4119c1e46072a687d217e2b2ad2de0bd02269acd42f14083c14b1f"
# The only permitted historical-producer-to-consumer drift.  This is the
# SHA-256 of the exact Git binary patch for the three names below with
# ``--full-index``.  Full object IDs are essential: Git otherwise abbreviates
# index lines according to the local object database, yielding different bytes
# for the same semantic patch on separate hosts.  Checking names alone would
# silently accept later semantic changes.
EXPECTED_HISTORICAL_G3_PATCH_SHA256 = "308db1c1e38135e5a65d37fa92566ac9cd5136220b4ffbb143a7e2f323d1ee0b"
HISTORICAL_G3_CRITICAL_SOURCE_REFS = (
    "src/param_importance_nlp/assets.py",
    "src/param_importance_nlp/contracts/__init__.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
)
EXPECTED_MODEL_CONFIG_VOCAB_SIZE = 50_304
EXPECTED_TOKENIZER_RUNTIME_VOCAB_SIZE = 50_277
PYTHIA_ACTIVE_DROPOUT_FIELDS = ("attention_dropout", "hidden_dropout")
EXPECTED_RUNTIME_ASSETS = {
    "model": {"logical_name": "pythia-14m-step0", "asset_id": "11dd681a22649a451b9be53c255bb4e9f83207c3f22f75f1eec53a33b7776fd2", "revision": "56079904bb80b7f36d3b794089f146e7a4d6efae", "ready_manifest_sha256": "7d3404906f3dd00c0d0314863f706c5df01f1db1fc0e0b4cf501353b88963d1e", "parameter_count": 14067712, "config_vocab_size": EXPECTED_MODEL_CONFIG_VOCAB_SIZE},
    "tokenizer": {"logical_name": "pythia-tokenizer", "asset_id": "b5eebc43fe88687e5bf692761f1db25f91e8d6f9a8cceaa2342d2624ac1f652d", "revision": "e361f9afd54b3e7856879eead5326d36ff6f32d7", "ready_manifest_sha256": "ea59f3f8e37321208701326b2ea88b7491450a88eae870775beeff027d102794", "vocab_size": EXPECTED_TOKENIZER_RUNTIME_VOCAB_SIZE},
    "pile": {"logical_name": "pile-selected-prefix", "asset_id": "dbbfeb12bab4027b386bd97d604d8134699e96f79e309cceacff7999a55b5dad", "revision": "4647773ea142ab1ff5694602fa104bbf49088408", "ready_manifest_sha256": "345cd0f49d35ad9543daa3f95118013c55bdd729ed87fdec3c7a7c93ae449f8b"},
}
EXPECTED_PILE_HASHED_BYTES = 31_757_184_042
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
REPLAY_CHECK_IDS = {
    "full_vs_offline_mean", "online_vs_offline_mean", "raw_vs_oracle",
    "double_vs_oracle", "explicit_u_vs_oracle", "streaming_u_vs_oracle",
}
ACCUMULATOR_FIELDS = {
    "positive", "negative_mass", "raw", "raw_clipped", "data_movement",
    "data_displacement", "total_movement", "total_displacement",
    "weight_decay_movement", "weight_decay_displacement",
    "actual_update_raw_importance", "magnitude", "initial_parameters",
    "last_parameters",
}
GATE_CHECK_IDS = (
    "s1_6_handoff", "qualified_assets", "historical_producer_replay",
    "approved_gpu_uuid", "gpu_preflight_twice", "cuda_minimum_allocation",
    "offline_guard", "registry_reload", "fixed_state_t32",
    "independent_oracle", "statistics_path_parity", "two_steps",
    "safetensors_manifest", "resource_evidence", "failure_markers",
    "charts_exact", "catalog_decision_exempt",
)


class Stage1S17FormalError(RuntimeError):
    """A formal-only S1.7 prerequisite or evidence check failed closed."""


class Stage1S17ManualInterventionRequired(Stage1S17FormalError):
    """A child fingerprint changed, so no signal or lease release is safe."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    done = subprocess.run(["git", "-C", str(repository), *arguments], text=True, capture_output=True, check=False, timeout=30)
    if done.returncode:
        raise Stage1S17FormalError(f"S17_GIT_FAILED:{arguments[0]}")
    return done.stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    """Return Git's unmodified bytes for hash-bound producer attestations."""

    done = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if done.returncode:
        raise Stage1S17FormalError(f"S17_GIT_FAILED:{arguments[0]}")
    return bytes(done.stdout)


def _path(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S17FormalError(f"S17_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S17FormalError(f"S17_LOGICAL_REF_ESCAPE:{field}")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S17FormalError(f"S17_LOGICAL_REF_ESCAPE:{field}") from error
    return candidate


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage1S17FormalError(f"S17_OBJECT_INVALID:{field}")
    return dict(value)


def _with_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value); body["artifact_hash"] = _canonical(body)
    return body


def _canonical(value: object) -> str:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash
    return canonical_json_hash(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    from param_importance_nlp.contracts.jsonio import write_canonical_json
    if path.exists():
        raise Stage1S17FormalError(f"S17_IMMUTABLE_TARGET_EXISTS:{path.name}")
    write_canonical_json(path, dict(value))


def _validate_role_schemas(repository: Path, objects: Mapping[str, Mapping[str, Any]]) -> None:
    """Use the project's strict stdlib schema subset against all eight roles."""

    from param_importance_nlp.contracts.jsonio import loads_strict_json
    formalizer_path = repository / "ops" / "stage1" / "formalize_s1_6.py"
    spec = importlib.util.spec_from_file_location("_s17_schema_subset", formalizer_path)
    if spec is None or spec.loader is None:
        raise Stage1S17FormalError("S17_SCHEMA_VALIDATOR_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    registry: dict[str, Mapping[str, Any]] = {}
    paths = sorted((repository / "schemas" / "stage1").glob("s1-7-*.json"))
    if len(paths) != 8:
        raise Stage1S17FormalError("S17_SCHEMA_REGISTRY_INCOMPLETE")
    for path in paths:
        try:
            loaded = loads_strict_json(path.read_bytes())
        except Exception as error:
            raise Stage1S17FormalError(f"S17_SCHEMA_PARSE_INVALID:{path.name}") from error
        if not isinstance(loaded, Mapping) or not isinstance(loaded.get("$id"), str):
            raise Stage1S17FormalError(f"S17_SCHEMA_ID_INVALID:{path.name}")
        registry[path.name] = loaded; registry[str(loaded["$id"])] = loaded
    filenames = {
        "fixture_manifest": "s1-7-fixture-manifest-v1.json", "gradient_bundle": "s1-7-gradient-bundle-v1.json",
        "single_gpu_report": "s1-7-single-gpu-report-v1.json", "comparison_table": "s1-7-comparison-table-v1.json",
        "gate_record": "s1-7-gate-record-v1.json", "replay": "s1-7-replay-validation-v1.json",
        "validation": "s1-7-validation-v1.json", "index": "s1-7-formalization-index-v1.json",
    }
    for role, value in objects.items():
        filename = filenames.get(role); schema = registry.get(filename or "")
        if schema is None:
            raise Stage1S17FormalError(f"S17_SCHEMA_ROLE_UNKNOWN:{role}")
        try:
            module._validate_schema(value, schema, registry, document=schema, path=role)
        except Exception as error:
            raise Stage1S17FormalError(f"S17_SCHEMA_VALIDATION_FAILED:{role}") from error


def _validate_validation_checks(value: Mapping[str, Any]) -> None:
    """Validation is a fixed, one-to-one narration of the 17 gate checks."""

    checks = value.get("checks")
    if not isinstance(checks, list) or len(checks) != len(GATE_CHECK_IDS):
        raise Stage1S17FormalError("S17_VALIDATION_CHECK_COUNT_INVALID")
    observed: list[str] = []
    for item in checks:
        row = _mapping(item, field="validation.check")
        check_id = row.get("check_id")
        if not isinstance(check_id, str) or row.get("status") != "PASS" or not isinstance(row.get("detail"), str) or not row["detail"]:
            raise Stage1S17FormalError("S17_VALIDATION_CHECK_ROW_INVALID")
        observed.append(check_id)
    if tuple(observed) != GATE_CHECK_IDS or len(set(observed)) != len(observed):
        raise Stage1S17FormalError("S17_VALIDATION_CHECK_IDS_INVALID")


def _self_hash(value: Mapping[str, Any], field: str = "artifact_hash") -> bool:
    body = dict(value); declared = body.pop(field, None)
    return isinstance(declared, str) and declared == _canonical(body)


def _audit_s16_consumer_diff(repository: Path) -> tuple[str, ...]:
    """Permit only named S1.7 consumer code and the post-S1.6 worklog.

    This intentionally makes a shared S1.6/runtime change block reuse until a
    reviewer explicitly replays S1.6 rather than treating an old gate as a
    generic dependency cache.
    """

    changed = tuple(filter(None, _git(repository, "diff", "--name-only", EXPECTED_S1_6_PRODUCER, "HEAD").splitlines()))
    allowed_exact = {
        "worklogs/2026-08-15-s1.6-training-integration-and-accumulators.md",
        "configs/run-ready/layers/formal-stage1-pythia14m.yaml",
    }
    allowed_family = (
        re.compile(r"ops/stage1/(?:formalize_s1_7|run_s1_7_worker)\.py"),
        re.compile(r"src/param_importance_nlp/stage1_single_gpu(?:_oracle)?\.py"),
        re.compile(r"schemas/stage1/s1-7-[a-z0-9-]+-v1\.json"),
        re.compile(r"tests/test_stage1_s17_[a-z0-9_]+\.py"),
    )
    rejected = [name for name in changed if name not in allowed_exact and not any(rule.fullmatch(name) for rule in allowed_family)]
    if rejected:
        raise Stage1S17FormalError("S17_S16_CONSUMER_DIFF_UNAUTHORIZED:" + ",".join(rejected))
    s16_owned = ("fixtures/stage1/stage1-s16-", "schemas/stage1/s1-6-", "src/param_importance_nlp/stage1_training_")
    if any(name.startswith(s16_owned) for name in changed):
        raise Stage1S17FormalError("S17_S16_OWNED_SOURCE_DRIFT")
    return changed


def _candidate_from_index(root: Path, index_path: Path, reference: object, *, field: str) -> Path:
    candidate = _path(root, reference, field=field)
    if candidate.is_file():
        return candidate
    if not isinstance(reference, str):
        raise Stage1S17FormalError(f"S17_ROLE_REF_INVALID:{field}")
    relative = (index_path.parent / reference).resolve()
    try:
        relative.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S17FormalError(f"S17_ROLE_REF_ESCAPE:{field}") from error
    return relative


def load_s1_6(data_root: Path, index_ref: str, repository: Path) -> dict[str, Any]:
    """Bind exactly the mandated current S1.6 index and every role hash."""

    from param_importance_nlp.contracts.jsonio import load_canonical_json
    _audit_s16_consumer_diff(repository)
    index_path = _path(data_root, index_ref, field="s1_6_index_ref")
    if not index_path.is_file() or _sha(index_path) != EXPECTED_S1_6_INDEX_SHA256:
        raise Stage1S17FormalError("S17_S16_INDEX_NOT_CURRENT")
    index = _mapping(load_canonical_json(index_path), field="s1_6.index")
    if not _self_hash(index) or (
        index.get("schema_version") != "stage1-s1-6-formalization-index-v1"
        or index.get("status") != "PASS" or index.get("gate_id") != "G1-STEP"
        or index.get("task_id") != "stage1.06_training_integration_and_accumulators"
        or index.get("generator_git_commit") != EXPECTED_S1_6_PRODUCER
        or index.get("consumer_git_commit") != EXPECTED_S1_6_PRODUCER
        or index.get("gate_artifact_hash") != EXPECTED_S1_6_GATE_HASH
        or index.get("next_task_id") != TASK_ID
    ):
        raise Stage1S17FormalError("S17_S16_HANDOFF_NOT_READY")
    expected_roles = {"step_report", "oracle_bundle", "trace_bundle", "comparison_table", "gate_record"}
    refs, hashes = index.get("role_refs"), index.get("role_sha256")
    if not isinstance(refs, Mapping) or not isinstance(hashes, Mapping) or set(refs) != expected_roles or set(hashes) != expected_roles:
        raise Stage1S17FormalError("S17_S16_ROLE_SET_INVALID")
    roles: dict[str, Mapping[str, Any]] = {}
    for role in sorted(expected_roles):
        candidate = _candidate_from_index(data_root, index_path, refs[role], field=f"s1_6.{role}")
        expected = hashes[role]
        if not candidate.is_file() or not isinstance(expected, str) or _DIGEST.fullmatch(expected) is None or _sha(candidate) != expected:
            raise Stage1S17FormalError(f"S17_S16_ROLE_HASH_INVALID:{role}")
        loaded = load_canonical_json(candidate)
        roles[role] = _mapping(loaded, field=f"s1_6.{role}")
    gate = roles["gate_record"]
    if not _self_hash(gate) or gate.get("status") != "PASS" or gate.get("artifact_hash") != EXPECTED_S1_6_GATE_HASH:
        raise Stage1S17FormalError("S17_S16_GATE_RECORD_INVALID")
    # This validates the historical numerical role relation without accepting a
    # different S1.6 attempt or merely trusting an index bit.
    try:
        from param_importance_nlp.stage1_training_integration import validate_stage1_s16_evidence
        validate_stage1_s16_evidence(roles, source_root=repository)
    except Exception as error:
        raise Stage1S17FormalError("S17_S16_HISTORICAL_VALIDATION_FAILED") from error
    return {
        "s1_6_index_ref": index_ref, "s1_6_index_sha256": EXPECTED_S1_6_INDEX_SHA256,
        "s1_6_index_artifact_hash": str(index["artifact_hash"]), "s1_6_generator_commit": EXPECTED_S1_6_PRODUCER,
        "s1_6_consumer_commit": EXPECTED_S1_6_PRODUCER, "s1_6_gate_artifact_hash": EXPECTED_S1_6_GATE_HASH,
        "s1_6_role_refs": {str(key): str(value) for key, value in refs.items()},
        "s1_6_role_sha256": {str(key): str(value) for key, value in hashes.items()},
    }


def _load_capability(data_root: Path, reference: str, approved_uuid: str) -> dict[str, Any]:
    """Load an immutable formal task-output commit, never a loose JSON payload."""

    from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact
    if reference != EXPECTED_GPU_CAPABILITY_REF:
        raise Stage1S17FormalError("S17_GPU_CAPABILITY_REF_NOT_FROZEN")
    commit_file = _path(data_root, reference, field="gpu_capability_ref")
    if not commit_file.is_file() or _sha(commit_file) != EXPECTED_GPU_CAPABILITY_FILE_SHA256:
        raise Stage1S17FormalError("S17_GPU_CAPABILITY_COMMIT_FILE_DRIFT")
    try:
        loaded = load_committed_task_artifact(data_root, reference, require_formal=True)
    except Exception as error:
        raise Stage1S17FormalError("S17_GPU_CAPABILITY_COMMIT_INVALID") from error
    if loaded.identity.task_id != "stage0.01_baseline_and_safety" or loaded.identity.artifact_kind != "capability_cuda" or loaded.identity.artifact_hash != EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH:
        raise Stage1S17FormalError("S17_GPU_CAPABILITY_IDENTITY_INVALID")
    payload = _mapping(loaded.payload, field="gpu_capability.payload")
    try:
        from param_importance_nlp.contracts.runtime_evidence import RuntimeCapabilityEvidence
        evidence = RuntimeCapabilityEvidence.from_mapping(payload)
    except Exception as error:
        raise Stage1S17FormalError("S17_GPU_CAPABILITY_PAYLOAD_INVALID") from error
    if evidence.capability != "cuda" or evidence.status != "VERIFIED":
        raise Stage1S17FormalError("S17_GPU_CAPABILITY_NOT_VERIFIED")
    metadata = _mapping(evidence.metadata, field="gpu_capability.payload.metadata")
    allowed = metadata.get("allowed_gpu_uuids")
    if not isinstance(allowed, list) or not allowed or any(not isinstance(item, str) or not item.startswith("GPU-") for item in allowed):
        raise Stage1S17FormalError("S17_GPU_CAPABILITY_UUID_SET_INVALID")
    if approved_uuid not in allowed:
        raise Stage1S17FormalError("S17_GPU_UUID_NOT_OPERATOR_APPROVED")
    return {
        "commit_ref": loaded.identity.commit_ref,
        "object_ref": loaded.identity.object_ref,
        "task_id": loaded.identity.task_id,
        "artifact_kind": loaded.identity.artifact_kind,
        "artifact_hash": loaded.identity.artifact_hash,
        "config_hash": loaded.identity.config_hash,
        "source_refs": list(loaded.source_refs),
        "allowed_gpu_uuids": allowed,
    }


def _run(command: Sequence[str], *, timeout: int = 30) -> str:
    done = subprocess.run(list(command), text=True, capture_output=True, check=False, timeout=timeout)
    if done.returncode:
        raise Stage1S17FormalError(f"S17_COMMAND_FAILED:{Path(command[0]).name}:{done.stderr[-500:]}")
    return done.stdout


def discover_gpu(approved_uuid: str) -> dict[str, Any]:
    """Discover current UUID→physical mapping and reject occupied/non-A100 GPU."""

    output = _run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 7 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise Stage1S17FormalError("S17_GPU_DISCOVERY_PARSE_INVALID")
        rows.append({"physical_index": int(fields[0]), "uuid": fields[1], "name": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "utilization_percent": int(fields[5]), "temperature_c": int(fields[6])})
    selected = next((item for item in rows if item["uuid"] == approved_uuid), None)
    if selected is None or not str(selected["name"]).startswith("NVIDIA A100") or int(selected["memory_total_mib"]) < 70000 or int(selected["memory_used_mib"]) != 0 or int(selected["utilization_percent"]) != 0:
        raise Stage1S17FormalError("S17_GPU_HEALTH_OR_IDLE_PRECONDITION_FAILED")
    applications = _run(["nvidia-smi", f"--id={approved_uuid}", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if applications.strip():
        raise Stage1S17FormalError("S17_GPU_EXTERNAL_PROCESS_PRESENT")
    health = _run(["nvidia-smi", f"--id={approved_uuid}", "-q", "-d", "PERFORMANCE,ECC"])
    if "Unknown Error" in health or "Retired Pages" in health and "Pending                        : Yes" in health:
        raise Stage1S17FormalError("S17_GPU_HEALTH_QUERY_FAILED")
    return {"discovery": rows, "selected": selected, "health_sha256": hashlib.sha256(health.encode("utf-8")).hexdigest(), "external_compute_processes": []}


def _min_cuda_probe(repository: Path, uuid: str, work: Path) -> dict[str, Any]:
    script = "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; torch.cuda.set_device(0); x=torch.empty((1024,1024),device='cuda',dtype=torch.float32); torch.cuda.synchronize(); print(torch.cuda.get_device_properties(0).name, x.nbytes)"
    environment = dict(os.environ); environment["CUDA_VISIBLE_DEVICES"] = uuid
    done = subprocess.run([sys.executable, "-c", script], cwd=repository, text=True, capture_output=True, check=False, timeout=60, env=environment)
    (work / "cuda-min-probe.stdout.txt").write_text(done.stdout, encoding="utf-8")
    (work / "cuda-min-probe.stderr.txt").write_text(done.stderr, encoding="utf-8")
    if done.returncode or "NVIDIA A100" not in done.stdout:
        raise Stage1S17FormalError("S17_GPU_MIN_CUDA_PROBE_FAILED")
    return {"command": [sys.executable, "-c", "torch.empty(1024,1024,cuda)"], "stdout_sha256": _sha(work / "cuda-min-probe.stdout.txt"), "stderr_sha256": _sha(work / "cuda-min-probe.stderr.txt")}


def _safe_fixture_safetensors(work: Path, dataset: object) -> tuple[str, dict[str, Any]]:
    """Extract 16 verified Pile rows; full mmap hash occurred at dataset open."""

    try:
        from safetensors.torch import save_file
        import torch
    except ImportError as error:  # pragma: no cover
        raise Stage1S17FormalError("S17_SAFETENSORS_UNAVAILABLE") from error
    values: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for index in range(16):
        raw = dataset.raw_record(index)  # PythiaIndexedDataset already enforces global interval.
        tensor = torch.from_numpy(raw.astype("int64", copy=False)).clone().contiguous()
        if tuple(tensor.shape) != (2049,):
            raise Stage1S17FormalError("S17_PILE_RECORD_SHAPE_DRIFT")
        values[f"record_{index:012d}"] = tensor
        hashes[str(index)] = hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()
    target = work / "fixture-inputs.safetensors"
    save_file(values, str(target), metadata={"schema_version": "stage1-s1-7-pile-fixture-tokens-v1"})
    return target.name, hashes


def _validate_fixture_manifest(value: object) -> dict[str, Any]:
    """Formal fixture parser shared by the positive and negative paths."""

    fixture = _mapping(value, field="fixture")
    expected = {"schema_version", "fixture_id", "assets", "asset_identity", "batching", "records", "token_sha256", "execution_contract", "fixture_hash"}
    if set(fixture) != expected or fixture.get("schema_version") != "stage1-s1-7-fixture-manifest-v1" or fixture.get("fixture_id") != FIXTURE_ID:
        raise Stage1S17FormalError("S17_FIXTURE_FIELDS_INVALID")
    if fixture.get("fixture_hash") != _canonical({key: item for key, item in fixture.items() if key != "fixture_hash"}):
        raise Stage1S17FormalError("S17_FIXTURE_HASH_INVALID")
    provenances = _mapping(fixture.get("assets"), field="fixture.assets")
    if set(provenances) != set(EXPECTED_RUNTIME_ASSETS):
        raise Stage1S17FormalError("S17_FIXTURE_PROVENANCE_ROLE_SET_INVALID")
    for role, frozen in EXPECTED_RUNTIME_ASSETS.items():
        provenance = _mapping(provenances.get(role), field=f"fixture.assets.{role}")
        if (
            provenance.get("logical_asset_id") != frozen["logical_name"]
            or provenance.get("asset_id") != frozen["asset_id"]
            or provenance.get("ready_manifest_sha256") != frozen["ready_manifest_sha256"]
            or provenance.get("g3_resolution_ref") != EXPECTED_G3_RESOLUTION
            or provenance.get("g3_resolution_artifact_hash") != EXPECTED_G3_PAYLOAD_HASH
            or provenance.get("source_git_commit") != HISTORICAL_G3_PRODUCER
        ):
            raise Stage1S17FormalError(f"S17_FIXTURE_PROVENANCE_IDENTITY_INVALID:{role}")
    if _mapping(fixture.get("batching"), field="fixture.batching") != {"global_batch_size": 4, "microbatch_size": 1, "accumulation_steps": 4, "world_size": 1}:
        raise Stage1S17FormalError("S17_FIXTURE_BATCHING_INVALID")
    if _mapping(fixture.get("records"), field="fixture.records") != {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7], "training": [[8, 9, 10, 11], [12, 13, 14, 15]]}:
        raise Stage1S17FormalError("S17_FIXTURE_RECORDS_INVALID")
    hashes = _mapping(fixture.get("token_sha256"), field="fixture.token_sha256")
    if set(hashes) != {str(index) for index in range(16)} or any(not isinstance(item, str) or _DIGEST.fullmatch(item) is None for item in hashes.values()):
        raise Stage1S17FormalError("S17_FIXTURE_TOKEN_HASHES_INVALID")
    asset_identity = _mapping(fixture.get("asset_identity"), field="fixture.asset_identity")
    for role, expected_identity in EXPECTED_RUNTIME_ASSETS.items():
        if _mapping(asset_identity.get(role), field=f"fixture.asset_identity.{role}") != expected_identity:
            raise Stage1S17FormalError(f"S17_FIXTURE_ASSET_IDENTITY_INVALID:{role}")
    model_identity = _mapping(asset_identity["model"], field="fixture.asset_identity.model")
    tokenizer_identity = _mapping(asset_identity["tokenizer"], field="fixture.asset_identity.tokenizer")
    if model_identity.get("parameter_count") != 14067712 or model_identity.get("config_vocab_size") != EXPECTED_MODEL_CONFIG_VOCAB_SIZE or tokenizer_identity.get("vocab_size") != EXPECTED_TOKENIZER_RUNTIME_VOCAB_SIZE:
        raise Stage1S17FormalError("S17_FIXTURE_ASSET_DIMENSION_IDENTITY_INVALID")
    contract = _mapping(fixture.get("execution_contract"), field="fixture.execution_contract")
    expected_contract = {"model_mode", "dropout_probabilities", "random_layer_policy", "precision", "loss", "optimizer", "gradient_clip_max_norm", "scheduler", "statistical_contract", "determinism"}
    if set(contract) != expected_contract or contract.get("model_mode") != "train" or contract.get("random_layer_policy") != "all_pythia_dropout_probabilities_zero" or contract.get("scheduler") is not None or contract.get("gradient_clip_max_norm") != 1.0:
        raise Stage1S17FormalError("S17_FIXTURE_EXECUTION_CONTRACT_INVALID")
    dropout = _mapping(contract.get("dropout_probabilities"), field="fixture.dropout")
    if set(dropout) != set(PYTHIA_ACTIVE_DROPOUT_FIELDS) or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0.0 for value in dropout.values()):
        raise Stage1S17FormalError("S17_FIXTURE_DROPOUT_INVALID")
    if _mapping(contract.get("statistical_contract"), field="fixture.statistical_contract") != {"estimator_name": "u", "statistical_unit": "microbatch_mean_gradient", "weight_unit": "effective_target_tokens", "sampling_design": "ordered_disjoint_microbatches", "weights_exogenous": True, "common_mean_assumption": True}:
        raise Stage1S17FormalError("S17_FIXTURE_STATISTICAL_CONTRACT_INVALID")
    if _mapping(contract.get("determinism"), field="fixture.determinism") != {"model_seed": 1707, "training_seed": 2707, "deterministic_algorithms": True, "allow_tf32": False, "cublas_workspace_config": ":4096:8"}:
        raise Stage1S17FormalError("S17_FIXTURE_DETERMINISM_INVALID")
    return fixture


def _historical_checkout(repository: Path, work: Path) -> Path:
    """Create a temp-local historical producer clone without mutating main Git."""

    checkout = work / "historical-g3"
    if checkout.exists():
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_TARGET_EXISTS")
    try:
        _run(["git", "-C", str(repository), "cat-file", "-e", f"{HISTORICAL_G3_PRODUCER}^{{commit}}"])
        _run(["git", "clone", "--shared", "--no-checkout", str(repository), str(checkout)], timeout=120)
        _run(["git", "-C", str(checkout), "checkout", "--detach", HISTORICAL_G3_PRODUCER], timeout=120)
    except Exception as error:
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_UNAVAILABLE") from error
    if _git(checkout, "rev-parse", "HEAD") != HISTORICAL_G3_PRODUCER or _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_CHECKOUT_INVALID")
    return checkout


def _historical_source_attestation(repository: Path, checkout: Path) -> dict[str, Any]:
    is_ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", HISTORICAL_G3_PRODUCER, "HEAD"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if is_ancestor.returncode != 0:
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_NOT_ANCESTOR")
    changed = tuple(filter(None, _git(repository, "diff", "--name-only", HISTORICAL_G3_PRODUCER, "HEAD", "--", *HISTORICAL_G3_CRITICAL_SOURCE_REFS).splitlines()))
    expected_changed = {
        "src/param_importance_nlp/assets.py", "src/param_importance_nlp/contracts/__init__.py",
        "src/param_importance_nlp/experiments/stage01_task_runners.py",
    }
    if set(changed) != expected_changed:
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_DIFF_ATTESTATION_FAILED")
    patch = _git_bytes(repository, "diff", "--binary", "--full-index", HISTORICAL_G3_PRODUCER, "HEAD", "--", *HISTORICAL_G3_CRITICAL_SOURCE_REFS)
    patch_sha256 = hashlib.sha256(patch).hexdigest()
    if patch_sha256 != EXPECTED_HISTORICAL_G3_PATCH_SHA256:
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_PATCH_DRIFT")
    historical_hashes = {path: _sha(checkout / path) for path in HISTORICAL_G3_CRITICAL_SOURCE_REFS}
    consumer_hashes = {path: _sha(repository / path) for path in HISTORICAL_G3_CRITICAL_SOURCE_REFS}
    return _with_hash({
        "schema_version": "stage1-s1-7-historical-producer-attestation-v1",
        "status": "PASS",
        "historical_producer_commit": HISTORICAL_G3_PRODUCER,
        "consumer_commit": _git(repository, "rev-parse", "HEAD"),
        "historical_producer_is_ancestor": True,
        "critical_source_diff": list(changed),
        "critical_patch_sha256": patch_sha256,
        "historical_source_sha256": historical_hashes,
        "consumer_source_sha256": consumer_hashes,
    })


def _historical_asset_replay(checkout: Path, data_root: Path, work: Path, resolution_ref: str) -> dict[str, Any]:
    if resolution_ref != EXPECTED_G3_RESOLUTION:
        raise Stage1S17FormalError("S17_G3_RESOLUTION_NOT_CURRENT_A3BC")
    output = work / "historical-g3-replay.json"
    fixture_path = work / "fixture-inputs.safetensors"
    script = r'''import hashlib,json,os,sys,time
from pathlib import Path
import torch
from safetensors.torch import save_file
from param_importance_nlp.g3_runtime_assets import FormalG3RuntimeAssets
from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact
from param_importance_nlp.contracts.jsonio import canonical_json_hash
root=Path(sys.argv[1]); resolution=sys.argv[2]; output=Path(sys.argv[3]); fixture=Path(sys.argv[4]); repo=Path.cwd()
def canonical(value): return canonical_json_hash(value)
def sha_file(path):
 digest=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda: handle.read(1024*1024),b''): digest.update(chunk)
 return digest.hexdigest()
committed=load_committed_task_artifact(root,resolution,require_formal=True)
assets=FormalG3RuntimeAssets.load(root,resolution,requirements_path=repo/'configs/stage0/g3-asset-requirements-v1.json',layout_path=repo/'configs/stage0/g3-asset-layout-v1.json')
start=time.monotonic(); model=assets.resolve('pythia-14m-step0',expected_kind='model'); tokenizer=assets.resolve('pythia-tokenizer',expected_kind='tokenizer'); pile=assets.resolve('pile-selected-prefix',expected_kind='pile'); resolve_seconds=time.monotonic()-start
assets.validate_pile_budget(stage=1,split='train',requested_records=16)
start=time.monotonic(); dataset=assets.pythia_dataset(pile,split='train')
try:
 values={}; hashes={}
 for i in range(16):
  raw=dataset.raw_record(i); value=torch.from_numpy(raw.astype('int64',copy=False)).clone().contiguous()
  if tuple(value.shape)!=(2049,): raise RuntimeError('S17_PILE_RECORD_SHAPE_DRIFT')
  values[f'record_{i:012d}']=value; hashes[str(i)]=hashlib.sha256(value.numpy().tobytes(order='C')).hexdigest()
 save_file(values,str(fixture),metadata={'schema_version':'stage1-s1-7-pile-fixture-tokens-v1'})
finally: dataset.close()
model_config=json.loads((model.resolved.root/'config.json').read_text(encoding='utf-8'))
active_dropout_fields=('attention_dropout','hidden_dropout')
if any(key not in model_config or isinstance(model_config[key],bool) or not isinstance(model_config[key],(int,float)) for key in active_dropout_fields): raise RuntimeError('S17_PYTHIA_ACTIVE_DROPOUT_WIRE_INVALID')
dropout={key:float(model_config[key]) for key in active_dropout_fields}
if any(value!=0.0 for value in dropout.values()): raise RuntimeError('S17_PYTHIA_ACTIVE_DROPOUT_VALUE_INVALID')
model_vocab_size=model_config.get('vocab_size')
tokenizer_metadata=tokenizer.manifest.get('metadata')
tokenizer_vocab_size=None if not isinstance(tokenizer_metadata,dict) else tokenizer_metadata.get('token_count_with_added_tokens')
if isinstance(model_vocab_size,bool) or not isinstance(model_vocab_size,int) or model_vocab_size!=int(sys.argv[6]) or isinstance(tokenizer_vocab_size,bool) or not isinstance(tokenizer_vocab_size,int) or tokenizer_vocab_size!=int(sys.argv[7]): raise RuntimeError('S17_PYTHIA_VOCAB_WIRE_INVALID')
manifest_files=pile.manifest.get('files'); storage=pile.manifest['metadata']['storage']; resolution_entries=assets.resolution.get('entries')
if not isinstance(manifest_files,list) or len(manifest_files)!=2 or not isinstance(resolution_entries,list): raise RuntimeError('S17_PILE_HASHED_BYTES_WIRE_INVALID')
expected_file_roles={(storage['idx']['path'],'index')}
shards=storage.get('shards')
if not isinstance(shards,list) or len(shards)!=1: raise RuntimeError('S17_PILE_HASHED_BYTES_WIRE_INVALID')
expected_file_roles.add((shards[0]['path'],'token_shard')); observed_file_roles=set(); pile_hashed_bytes=0
for descriptor in manifest_files:
 if not isinstance(descriptor,dict) or set(descriptor)!={'path','role','sha256','size_bytes'}: raise RuntimeError('S17_PILE_HASHED_BYTES_WIRE_INVALID')
 size=descriptor['size_bytes']; digest=descriptor['sha256']; identity=(descriptor['path'],descriptor['role'])
 if isinstance(size,bool) or not isinstance(size,int) or size<=0 or not isinstance(digest,str) or len(digest)!=64 or any(character not in '0123456789abcdef' for character in digest) or identity in observed_file_roles: raise RuntimeError('S17_PILE_HASHED_BYTES_WIRE_INVALID')
 observed_file_roles.add(identity); pile_hashed_bytes+=size
pile_entries=[entry for entry in resolution_entries if isinstance(entry,dict) and entry.get('logical_name')=='pile-selected-prefix']
expected_pile_hashed_bytes=int(sys.argv[5])
if observed_file_roles!=expected_file_roles or len(pile_entries)!=1 or isinstance(pile_entries[0].get('bytes_checked'),bool) or pile_entries[0].get('bytes_checked')!=pile_hashed_bytes or pile_hashed_bytes!=expected_pile_hashed_bytes: raise RuntimeError('S17_PILE_HASHED_BYTES_BINDING_INVALID')
payload={'schema_version':'stage1-s1-7-historical-g3-replay-v1','status':'PASS','model':model.provenance(),'tokenizer':tokenizer.provenance(),'pile':pile.provenance(),'asset_identity':{'model':{'logical_name':'pythia-14m-step0','asset_id':model.resolved.asset_id,'revision':model.resolved.revision,'ready_manifest_sha256':model.ready_manifest_sha256,'parameter_count':model.manifest['metadata']['parameter_count'],'config_vocab_size':model_vocab_size,'root':str(model.resolved.root)},'tokenizer':{'logical_name':'pythia-tokenizer','asset_id':tokenizer.resolved.asset_id,'revision':tokenizer.resolved.revision,'ready_manifest_sha256':tokenizer.ready_manifest_sha256,'root':str(tokenizer.resolved.root),'vocab_size':tokenizer_vocab_size},'pile':{'logical_name':'pile-selected-prefix','asset_id':pile.resolved.asset_id,'revision':pile.resolved.revision,'ready_manifest_sha256':pile.ready_manifest_sha256}},'resolution_commit_artifact_hash':committed.identity.artifact_hash,'resolution_artifact_hash':assets.resolution_artifact_hash,'fixture_file':fixture.name,'fixture_file_sha256':sha_file(fixture),'token_sha256':hashes,'dropout_probabilities':dropout,'resolve_hash_seconds':resolve_seconds,'dataset_rehash_seconds':time.monotonic()-start,'qualified_resolution_hashed_bytes':pile_hashed_bytes,'dataset_rehash_bytes':pile_hashed_bytes,'pile_hash_passes':2,'network_policy':{'hf_hub_offline':os.environ.get('HF_HUB_OFFLINE')=='1','transformers_offline':os.environ.get('TRANSFORMERS_OFFLINE')=='1','datasets_offline':os.environ.get('HF_DATASETS_OFFLINE')=='1','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')=='','cuda_is_available':bool(torch.cuda.is_available()),'operations':['committed-resolution-parse','qualified-local-manifest-parse','local-pile-mmap-hash-and-fixture-extraction'],'external_attempts':[]}}
payload['replay_hash']=canonical(payload)
output.write_text(json.dumps(payload,sort_keys=True,separators=(',',':')),encoding='utf-8')'''
    environment = dict(os.environ)
    environment.update({"PYTHONPATH": str(checkout / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run([sys.executable, "-c", script, str(data_root), resolution_ref, str(output), str(fixture_path), str(EXPECTED_PILE_HASHED_BYTES), str(EXPECTED_MODEL_CONFIG_VOCAB_SIZE), str(EXPECTED_TOKENIZER_RUNTIME_VOCAB_SIZE)], cwd=checkout, text=True, capture_output=True, check=False, timeout=3600, env=environment)
    stdout_path = work / "historical-g3-replay.stdout.txt"
    stderr_path = work / "historical-g3-replay.stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode or not output.is_file() or not fixture_path.is_file():
        raise Stage1S17FormalError(
            "S17_HISTORICAL_PRODUCER_REPLAY_FAILED:"
            f"returncode={completed.returncode}:"
            f"stdout_sha256={_sha(stdout_path)}:stderr_sha256={_sha(stderr_path)}:"
            f"output_present={output.is_file()}:fixture_present={fixture_path.is_file()}"
        )
    replay = _mapping(json.loads(output.read_text(encoding="utf-8")), field="historical_g3_replay")
    expected = {"schema_version", "status", "model", "tokenizer", "pile", "asset_identity", "resolution_commit_artifact_hash", "resolution_artifact_hash", "fixture_file", "fixture_file_sha256", "token_sha256", "dropout_probabilities", "resolve_hash_seconds", "dataset_rehash_seconds", "qualified_resolution_hashed_bytes", "dataset_rehash_bytes", "pile_hash_passes", "network_policy", "replay_hash"}
    if set(replay) != expected or replay.get("schema_version") != "stage1-s1-7-historical-g3-replay-v1" or replay.get("status") != "PASS" or replay.get("replay_hash") != _canonical({key: value for key, value in replay.items() if key != "replay_hash"}):
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_REPLAY_SCHEMA_INVALID")
    if replay.get("resolution_commit_artifact_hash") != EXPECTED_G3_COMMIT_ARTIFACT_HASH or replay.get("resolution_artifact_hash") != EXPECTED_G3_PAYLOAD_HASH:
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_RESOLUTION_PAYLOAD_DRIFT")
    network = _mapping(replay.get("network_policy"), field="historical.network_policy")
    if network.get("external_attempts") != [] or network.get("cuda_is_available") is not False or not all(network.get(key) is True for key in ("hf_hub_offline", "transformers_offline", "datasets_offline", "cuda_visible_devices")):
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_OFFLINE_POLICY_INVALID")
    if (
        replay.get("qualified_resolution_hashed_bytes") != EXPECTED_PILE_HASHED_BYTES
        or replay.get("dataset_rehash_bytes") != EXPECTED_PILE_HASHED_BYTES
        or replay.get("pile_hash_passes") != 2
    ):
        raise Stage1S17FormalError("S17_HISTORICAL_PRODUCER_HASH_BYTES_INVALID")
    return replay


def _asset_fixture(repository: Path, data_root: Path, work: Path, resolution_ref: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the a3bc assets with the exact historical G3 producer."""

    checkout = _historical_checkout(repository, work)
    attestation = _historical_source_attestation(repository, checkout)
    _write(work / "historical-producer-attestation.json", attestation)
    replay = _historical_asset_replay(checkout, data_root, work, resolution_ref)
    identity = _mapping(replay.get("asset_identity"), field="historical.asset_identity")
    for role, expected in EXPECTED_RUNTIME_ASSETS.items():
        observed = _mapping(identity.get(role), field=f"historical.asset_identity.{role}")
        if any(observed.get(field) != value for field, value in expected.items()):
            raise Stage1S17FormalError(f"S17_FROZEN_ASSET_IDENTITY_DRIFT:{role}")
    model = _mapping(identity["model"], field="historical.model")
    if model.get("parameter_count") != 14067712 or model.get("config_vocab_size") != EXPECTED_MODEL_CONFIG_VOCAB_SIZE:
        raise Stage1S17FormalError("S17_FROZEN_MODEL_PARAMETER_COUNT_DRIFT")
    fixture: dict[str, Any] = {"schema_version": "stage1-s1-7-fixture-manifest-v1", "fixture_id": FIXTURE_ID, "assets": {role: replay[role] for role in ("model", "tokenizer", "pile")}, "asset_identity": {"model": {key: model[key] for key in ("logical_name", "asset_id", "revision", "ready_manifest_sha256", "parameter_count", "config_vocab_size")}, "tokenizer": {key: _mapping(identity["tokenizer"], field="historical.tokenizer")[key] for key in ("logical_name", "asset_id", "revision", "ready_manifest_sha256", "vocab_size")}, "pile": {key: _mapping(identity["pile"], field="historical.pile")[key] for key in ("logical_name", "asset_id", "revision", "ready_manifest_sha256")}}, "batching": {"global_batch_size": 4, "microbatch_size": 1, "accumulation_steps": 4, "world_size": 1}, "records": {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7], "training": [[8, 9, 10, 11], [12, 13, 14, 15]]}, "token_sha256": replay["token_sha256"], "execution_contract": {"model_mode": "train", "dropout_probabilities": replay["dropout_probabilities"], "random_layer_policy": "all_pythia_dropout_probabilities_zero", "precision": {"compute": "float32", "gradient": "float32", "statistics": "float32", "reference": "float64", "amp": False}, "loss": {"task_type": "causal_lm", "reduction": "mean", "valid_tokens_per_microbatch": 2048, "ignore_index": -100}, "optimizer": {"type": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8, "foreach": False, "fused": False}, "gradient_clip_max_norm": 1.0, "scheduler": None, "statistical_contract": {"estimator_name": "u", "statistical_unit": "microbatch_mean_gradient", "weight_unit": "effective_target_tokens", "sampling_design": "ordered_disjoint_microbatches", "weights_exogenous": True, "common_mean_assumption": True}, "determinism": {"model_seed": 1707, "training_seed": 2707, "deterministic_algorithms": True, "allow_tf32": False, "cublas_workspace_config": ":4096:8"}}}
    fixture["fixture_hash"] = _canonical(fixture); _validate_fixture_manifest(fixture)
    _validate_role_schemas(repository, {"fixture_manifest": fixture})
    worker_assets = {"model": {"parameter_count": 14067712, "config_vocab_size": model["config_vocab_size"], **{field: model[field] for field in ("revision", "asset_id", "ready_manifest_sha256")}}, "tokenizer": {field: _mapping(identity["tokenizer"], field="historical.tokenizer")[field] for field in ("asset_id", "revision", "ready_manifest_sha256")}, "pile": {**{field: _mapping(identity["pile"], field="historical.pile")[field] for field in ("asset_id", "revision", "ready_manifest_sha256")}, "full_hash_seconds": float(replay["resolve_hash_seconds"]) + float(replay["dataset_rehash_seconds"]), "qualified_resolution_hash_seconds": float(replay["resolve_hash_seconds"]), "dataset_rehash_seconds": float(replay["dataset_rehash_seconds"]), "full_hash_bytes": int(replay["qualified_resolution_hashed_bytes"]) + int(replay["dataset_rehash_bytes"]), "qualified_resolution_hashed_bytes": int(replay["qualified_resolution_hashed_bytes"]), "dataset_rehash_bytes": int(replay["dataset_rehash_bytes"]), "fixture_file_sha256": replay["fixture_file_sha256"], "hash_passes": replay["pile_hash_passes"]}}
    return fixture, {"fixture_file": replay["fixture_file"], "model_root": model["root"], "tokenizer_root": _mapping(identity["tokenizer"], field="historical.tokenizer")["root"], "worker_assets": worker_assets, "g3_resolution_ref": resolution_ref, "g3_resolution_artifact_hash": replay["resolution_artifact_hash"], "historical_producer": attestation, "historical_replay_sha256": _sha(work / "historical-g3-replay.json")}


def _pid_fingerprint(pid: int, run_token: str) -> dict[str, Any]:
    stat = Path(f"/proc/{pid}/stat")
    start_ticks = ppid = process_group = session = None
    if stat.is_file():
        values = stat.read_text(encoding="utf-8").split()
        if len(values) > 21:
            ppid, process_group, session, start_ticks = values[3], values[4], values[5], values[21]
    cmdline = Path(f"/proc/{pid}/cmdline")
    executable = Path(f"/proc/{pid}/exe")
    if not cmdline.is_file() or not executable.exists():
        raise ProcessLookupError(pid)
    try:
        executable_ref = os.readlink(executable)
    except OSError as error:
        raise ProcessLookupError(pid) from error
    return {
        "uid": getattr(os, "getuid", lambda: None)(),
        "pid": pid,
        "pgid": os.getpgid(pid),
        "ppid": ppid,
        "process_group": process_group,
        "sid": os.getsid(pid),
        "session": session,
        "start_ticks": start_ticks,
        "exe": executable_ref,
        "cmdline_sha256": hashlib.sha256(cmdline.read_bytes()).hexdigest(),
        "run_token": run_token,
    }


def _terminate_exact_worker(process: subprocess.Popen[str], expected: Mapping[str, Any], work: Path) -> None:
    """Terminate only an exact session after a timeout; never use broad pkill."""

    try:
        current = _pid_fingerprint(process.pid, str(expected["run_token"]))
    except ProcessLookupError:
        return
    if current != dict(expected) or current["pgid"] != process.pid:
        _write(work / "manual-intervention-required.json", _with_hash({"schema_version": "stage1-s1-7-manual-intervention-v1", "status": "BLOCKED", "expected_fingerprint": dict(expected), "observed_fingerprint": current, "action": "NO_SIGNAL_NO_LEASE_RELEASE"}))
        raise Stage1S17ManualInterventionRequired("S17_WORKER_FINGERPRINT_DRIFT_DO_NOT_SIGNAL")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        current = _pid_fingerprint(process.pid, str(expected["run_token"]))
    except ProcessLookupError:
        return
    if current != dict(expected):
        _write(work / "manual-intervention-required.json", _with_hash({"schema_version": "stage1-s1-7-manual-intervention-v1", "status": "BLOCKED", "expected_fingerprint": dict(expected), "observed_fingerprint": current, "action": "NO_KILL_NO_LEASE_RELEASE"}))
        raise Stage1S17ManualInterventionRequired("S17_WORKER_FINGERPRINT_DRIFT_AFTER_TERM")
    os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        raise Stage1S17ManualInterventionRequired("S17_WORKER_KILL_TIMEOUT_MANUAL_REVIEW") from error


def _worker(repository: Path, work: Path, plan: Mapping[str, Any], uuid: str, timeout_seconds: int, lease: object) -> Mapping[str, Any]:
    from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
    plan_path = work / "worker-plan.json"; write_canonical_json(plan_path, dict(plan))
    environment = dict(os.environ)
    cache_root = Path(str(plan["cache_root"])).resolve(strict=True)
    environment.update({"CUDA_VISIBLE_DEVICES": uuid, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PARAM_IMPORTANCE_OFFLINE_GUARD": "1", "PARAM_IMPORTANCE_NETWORK_AUDIT_DIR": str(work / "network"), "HF_HOME": str(cache_root / "hf"), "TRANSFORMERS_CACHE": str(cache_root / "hf" / "transformers"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    command = [sys.executable, str(repository / "ops" / "stage1" / "run_s1_7_worker.py"), "--plan", str(plan_path)]
    process = subprocess.Popen(command, cwd=repository, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment, start_new_session=True)
    fingerprint = _pid_fingerprint(process.pid, str(plan["run_token"]))
    _write(work / "worker-start.json", _with_hash({"schema_version": "stage1-s1-7-worker-start-v1", "fingerprint": fingerprint, "command": command}))
    started = time.monotonic()
    while process.poll() is None:
        if time.monotonic() - started > timeout_seconds:
            _terminate_exact_worker(process, fingerprint, work)
            raise Stage1S17FormalError("S17_WORKER_TIMEOUT_TERMINATED_EXACT_GROUP")
        lease.heartbeat()
        time.sleep(10)
    stdout, stderr = process.communicate(timeout=30)
    (work / "worker.stdout.txt").write_text(stdout, encoding="utf-8")
    (work / "worker.stderr.txt").write_text(stderr, encoding="utf-8")
    if process.returncode:
        raise Stage1S17FormalError("S17_WORKER_FAILED")
    output = work / "worker-report.json"
    if not output.is_file():
        raise Stage1S17FormalError("S17_WORKER_REPORT_MISSING")
    report = _mapping(load_canonical_json(output), field="worker_report")
    if not _self_hash(report) or report.get("status") != "PASS" or report.get("task_id") != TASK_ID:
        raise Stage1S17FormalError("S17_WORKER_REPORT_INVALID")
    guard_ref = report.get("offline_guard_ref")
    if not isinstance(guard_ref, str) or not guard_ref:
        raise Stage1S17FormalError("S17_WORKER_OFFLINE_GUARD_REF_MISSING")
    guard_path = work / "network" / guard_ref
    guard = _mapping(load_canonical_json(guard_path), field="offline_guard")
    if not _self_hash(guard) or guard.get("status") != "COMPLETE" or guard.get("external_attempts") != []:
        raise Stage1S17FormalError("S17_WORKER_OFFLINE_GUARD_FAILED")
    return report


def _load_arrays(work: Path, report: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], Mapping[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover
        raise Stage1S17FormalError("S17_SAFETENSORS_UNAVAILABLE") from error
    manifest = _mapping(report.get("arrays"), field="arrays")
    if not _self_hash(manifest) or manifest.get("schema_version") != "stage1-s1-7-safetensors-manifest-v1":
        raise Stage1S17FormalError("S17_ARRAY_MANIFEST_HASH_INVALID")
    file = work / str(manifest.get("file"))
    if manifest.get("file") != "s1-7-arrays.safetensors" or not file.is_file() or _sha(file) != manifest.get("file_sha256") or file.stat().st_size != manifest.get("file_size_bytes"):
        raise Stage1S17FormalError("S17_ARRAY_FILE_IDENTITY_INVALID")
    maps = _mapping(manifest.get("maps"), field="arrays.maps"); tensors = _mapping(manifest.get("tensors"), field="arrays.tensors")
    values: dict[str, Any] = {}
    with safe_open(str(file), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors) or handle.metadata() != {"schema_version": "stage1-s1-7-safetensors-manifest-v1"}:
            raise Stage1S17FormalError("S17_ARRAY_FILE_KEY_OR_METADATA_DRIFT")
        for key in sorted(tensors):
            entry = _mapping(tensors[key], field=f"arrays.tensors.{key}"); tensor = handle.get_tensor(key)
            actual = hashlib.sha256(tensor.contiguous().numpy().tobytes(order="C")).hexdigest()
            if entry.get("sha256") != actual or entry.get("dtype") != str(tensor.dtype) or entry.get("shape") != list(tensor.shape):
                raise Stage1S17FormalError(f"S17_ARRAY_TENSOR_IDENTITY_INVALID:{key}")
            values[key] = tensor
    decoded: dict[str, dict[str, Any]] = {}
    mapped_keys: list[str] = []
    for phase, keys in maps.items():
        if not isinstance(keys, list) or not keys:
            raise Stage1S17FormalError("S17_ARRAY_MAP_INVALID")
        prefix = f"{phase}::"; decoded[str(phase)] = {}
        for key in keys:
            if not isinstance(key, str) or not key.startswith(prefix) or key not in values:
                raise Stage1S17FormalError("S17_ARRAY_MAP_KEY_INVALID")
            mapped_keys.append(key)
            tensor = values[key]
            if tensor.dtype != __import__("torch").float32 and str(phase) != "training_accumulator":
                raise Stage1S17FormalError(f"S17_ARRAY_PHASE_DTYPE_INVALID:{phase}")
            if str(phase) == "training_accumulator" and tensor.dtype != __import__("torch").float64:
                raise Stage1S17FormalError("S17_ACCUMULATOR_DTYPE_INVALID")
            if not bool(__import__("torch").isfinite(tensor).all()):
                raise Stage1S17FormalError(f"S17_ARRAY_NONFINITE:{phase}")
            decoded[str(phase)][key[len(prefix):]] = tensor
    if len(mapped_keys) != len(set(mapped_keys)) or set(mapped_keys) != set(values):
        raise Stage1S17FormalError("S17_ARRAY_MAP_TENSOR_COVERAGE_INVALID")
    required = {"full_gradient", "online_mean_gradient", "raw", "double", "explicit_u", "streaming_u", "local_a_0", "local_a_1", "local_a_2", "local_a_3", "local_b_0", "local_b_1", "local_b_2", "local_b_3", "training_accumulator"}
    if set(decoded) != required:
        raise Stage1S17FormalError("S17_ARRAY_PHASE_SET_INVALID")
    _validate_retained_tensor_bytes(report, decoded)
    registry = _mapping(report.get("registry"), field="registry")
    records = registry.get("records")
    if not isinstance(records, list): raise Stage1S17FormalError("S17_REGISTRY_RECORDS_INVALID")
    eligible = {str(_mapping(item, field="registry.record")["canonical_name"]) for item in records if _mapping(item, field="registry.record").get("eligible") is True}
    expected_fields = {"positive", "negative_mass", "raw", "raw_clipped", "data_movement", "data_displacement", "total_movement", "total_displacement", "weight_decay_movement", "weight_decay_displacement", "actual_update_raw_importance", "magnitude", "initial_parameters", "last_parameters"}
    accumulator = decoded["training_accumulator"]
    observed: dict[str, set[str]] = {}
    for key in accumulator:
        field, separator, parameter = key.partition(":")
        if not separator: raise Stage1S17FormalError("S17_ACCUMULATOR_KEY_INVALID")
        observed.setdefault(field, set()).add(parameter)
    if set(observed) != expected_fields or any(names != eligible for names in observed.values()):
        raise Stage1S17FormalError("S17_ACCUMULATOR_REGISTRY_COVERAGE_INVALID")
    record_by_name = {str(_mapping(item, field="registry.record")["canonical_name"]): _mapping(item, field="registry.record") for item in records}
    nonnegative = {"positive", "negative_mass", "raw_clipped", "data_movement", "total_movement", "weight_decay_movement", "magnitude"}
    for field, names in observed.items():
        for name in names:
            tensor = accumulator[f"{field}:{name}"]
            record = record_by_name[name]
            if tuple(tensor.shape) != tuple(record["shape"]):
                raise Stage1S17FormalError(f"S17_ACCUMULATOR_SHAPE_DRIFT:{field}:{name}")
            if field in nonnegative and bool((tensor < 0).any()):
                raise Stage1S17FormalError(f"S17_ACCUMULATOR_NEGATIVE_VALUE:{field}:{name}")
    for movement, displacement in (
        ("data_movement", "data_displacement"),
        ("total_movement", "total_displacement"),
        ("weight_decay_movement", "weight_decay_displacement"),
    ):
        for name in eligible:
            if bool((accumulator[f"{displacement}:{name}"].abs() > accumulator[f"{movement}:{name}"] + 1.0e-12).any()):
                raise Stage1S17FormalError(f"S17_ACCUMULATOR_MOVEMENT_IDENTITY_FAILED:{name}")
    for name in eligible:
        endpoint = accumulator[f"last_parameters:{name}"] - accumulator[f"initial_parameters:{name}"]
        displacement = accumulator[f"total_displacement:{name}"]
        if not bool(__import__("torch").allclose(endpoint, displacement, atol=1.0e-5, rtol=1.0e-5)):
            raise Stage1S17FormalError(f"S17_ACCUMULATOR_ENDPOINT_IDENTITY_FAILED:{name}")
    import torch
    def scalar(field: str) -> float:
        return sum(float(accumulator[f"{field}:{name}"].to(torch.float64).sum().item()) for name in sorted(eligible))
    def scalar_abs(field: str) -> float:
        return sum(float(accumulator[f"{field}:{name}"].to(torch.float64).abs().sum().item()) for name in sorted(eligible))
    projected = {
        "signed": scalar("positive") - scalar("negative_mass"),
        "positive": scalar("positive"),
        "negative_mass": scalar("negative_mass"),
        "absolute": scalar("positive") + scalar("negative_mass"),
        "raw": scalar("raw"),
        "raw_clipped": scalar("raw_clipped"),
        "data_movement": scalar("data_movement"),
        "net_data_movement": scalar_abs("data_displacement"),
        "total_endpoint_movement": sum(float((accumulator[f"last_parameters:{name}"] - accumulator[f"initial_parameters:{name}"]).to(torch.float64).abs().sum().item()) for name in sorted(eligible)),
        "weight_decay_movement": scalar("weight_decay_movement"),
        "magnitude": scalar("magnitude"),
        "actual_update_raw_importance": scalar("actual_update_raw_importance"),
    }
    on_trace = _mapping(_mapping(report.get("training"), field="training").get("statistics_on"), field="training.statistics_on")
    commits = [
        _mapping(row, field="training.observer") for row in on_trace.get("observer_rows", [])
        if isinstance(row, Mapping) and row.get("boundary") == "attempt_commit"
    ]
    if len(commits) != 2:
        raise Stage1S17FormalError("S17_ACCUMULATOR_COMMIT_SNAPSHOT_MISSING")
    snapshot = _mapping(commits[-1].get("importance_snapshot"), field="training.importance_snapshot")
    summaries = _mapping(snapshot.get("scalar_summaries"), field="training.scalar_summaries")
    required_summaries = set(projected)
    if not required_summaries <= set(summaries) or any(
        not isinstance(summaries[name], (int, float)) or isinstance(summaries[name], bool)
        or abs(float(summaries[name]) - projected[name]) > 1.0e-7 * max(1.0, abs(projected[name]))
        for name in required_summaries
    ):
        raise Stage1S17FormalError("S17_ACCUMULATOR_SCALAR_SUMMARY_DRIFT")
    return decoded, manifest


def _decoded_tensor_bytes(decoded: Mapping[str, Mapping[str, Any]]) -> int:
    """Measure the exact decoded safetensors payload, never a producer claim."""

    total = 0
    for phase, tensors in decoded.items():
        if not isinstance(phase, str) or not isinstance(tensors, Mapping):
            raise Stage1S17FormalError("S17_ARRAY_DECODED_STRUCTURE_INVALID")
        for name, tensor in tensors.items():
            if not isinstance(name, str):
                raise Stage1S17FormalError("S17_ARRAY_DECODED_STRUCTURE_INVALID")
            try:
                numel, element_size = tensor.numel(), tensor.element_size()
            except AttributeError as error:
                raise Stage1S17FormalError("S17_ARRAY_DECODED_TENSOR_INVALID") from error
            if type(numel) is not int or type(element_size) is not int or numel < 0 or element_size <= 0:
                raise Stage1S17FormalError("S17_ARRAY_DECODED_TENSOR_INVALID")
            total += numel * element_size
    if total <= 0:
        raise Stage1S17FormalError("S17_ARRAY_DECODED_BYTES_INVALID")
    return total


def _validate_retained_tensor_bytes(report: Mapping[str, Any], decoded: Mapping[str, Mapping[str, Any]]) -> int:
    resources = _mapping(report.get("resources"), field="resources")
    declared = resources.get("retained_tensor_bytes")
    measured = _decoded_tensor_bytes(decoded)
    if type(declared) is not int or declared != measured:
        raise Stage1S17FormalError("S17_RETAINED_TENSOR_BYTES_RECOMPUTE_FAILED")
    return measured


def _oracle_replay(arrays: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Import the independent oracle while production-estimator imports are denied."""

    blocked = ("param_importance_nlp.core.estimators", "param_importance_nlp.runtime.training", "param_importance_nlp.stage1_single_gpu")
    import builtins
    original = builtins.__import__
    module_name = "param_importance_nlp.stage1_single_gpu_oracle"
    sys.modules.pop(module_name, None)
    def guarded(name: str, globals: object = None, locals: object = None, fromlist: object = (), level: int = 0):
        if name.startswith(blocked): raise ImportError("S17_ORACLE_IMPORT_ISOLATION")
        return original(name, globals, locals, fromlist, level)
    try:
        builtins.__import__ = guarded
        oracle = importlib.import_module(module_name)
    finally:
        builtins.__import__ = original
    local_a = [arrays[f"local_a_{index}"] for index in range(4)]
    local_b = [arrays[f"local_b_{index}"] for index in range(4)]
    replay = oracle.raw_double_and_u(local_a, local_b)
    checks = {
        "full_vs_offline_mean": oracle.max_error(arrays["full_gradient"], replay["mean_a"]),
        "online_vs_offline_mean": oracle.max_error(arrays["online_mean_gradient"], replay["mean_a"]),
        "raw_vs_oracle": oracle.max_error(arrays["raw"], replay["raw"]),
        "double_vs_oracle": oracle.max_error(arrays["double"], replay["double"]),
        "explicit_u_vs_oracle": oracle.max_error(arrays["explicit_u"], replay["explicit_u"]),
        "streaming_u_vs_oracle": oracle.max_error(arrays["streaming_u"], replay["streaming_u"]),
    }
    if set(checks) != REPLAY_CHECK_IDS or any(value.get("within_t32") is not True for value in checks.values()):
        raise Stage1S17FormalError("S17_INDEPENDENT_ORACLE_T32_FAILED")
    body: dict[str, Any] = {"schema_version": "stage1-s1-7-offline-replay-v1", "status": "PASS", "oracle_import_isolated": True, "checks": checks}
    body["replay_hash"] = _canonical(body)
    return body


def _comparison_table(report: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    fixed = _mapping(report.get("fixed_state"), field="fixed_state")
    comparisons = _mapping(fixed.get("comparisons"), field="fixed_state.comparisons")
    registry = _mapping(report.get("registry"), field="registry")
    registry_records = registry.get("records")
    if not isinstance(registry_records, list):
        raise Stage1S17FormalError("S17_COMPARISON_REGISTRY_RECORDS_INVALID")
    tags_by_parameter: dict[str, Mapping[str, Any]] = {}
    for record in registry_records:
        item = _mapping(record, field="comparison.registry_record")
        name, tags = item.get("canonical_name"), item.get("tags")
        if not isinstance(name, str) or not isinstance(tags, Mapping):
            raise Stage1S17FormalError("S17_COMPARISON_REGISTRY_TAGS_INVALID")
        if item.get("eligible") is True:
            tags_by_parameter[name] = dict(tags)
    rows: list[dict[str, Any]] = []
    all_checks = [("full_vs_online", comparisons.get("full_vs_online")), ("explicit_u_vs_streaming_u", comparisons.get("explicit_u_vs_streaming_u")), *sorted(_mapping(replay.get("checks"), field="replay.checks").items())]
    for metric, value in all_checks:
        item = _mapping(value, field=f"comparison.{metric}")
        if item.get("within_t32") is not True:
            raise Stage1S17FormalError(f"S17_COMPARISON_T32_FAILED:{metric}")
        rows.append({"scope": "global", "metric": metric, "module": "__global__", "layer": "__global__", "max_abs_error": float(item["max_abs_error"]), "max_scaled_error": float(item["max_scaled_error"]), "normalized_l2_error": float(item["normalized_l2_error"]), "parameter": str(item["parameter"]), "index": list(item["index"]), "violation_count": int(item["violation_count"]), "status": "PASS"})
        per_tensor = _mapping(item.get("per_tensor"), field=f"comparison.{metric}.per_tensor")
        if set(per_tensor) != set(tags_by_parameter):
            raise Stage1S17FormalError(f"S17_COMPARISON_PER_TENSOR_COVERAGE_INVALID:{metric}")
        for parameter, evidence in sorted(per_tensor.items()):
            tensor = _mapping(evidence, field=f"comparison.{metric}.per_tensor.{parameter}")
            if tensor.get("within_t32") is not True:
                raise Stage1S17FormalError(f"S17_COMPARISON_PER_TENSOR_T32_FAILED:{metric}:{parameter}")
            tags = tags_by_parameter[parameter]
            rows.append({"scope": "per_tensor", "metric": metric, "module": str(tags.get("module")), "layer": str(tags.get("layer")), "max_abs_error": float(tensor["max_abs_error"]), "max_scaled_error": float(tensor["max_scaled_error"]), "normalized_l2_error": float(tensor["normalized_l2_error"]), "parameter": parameter, "index": [], "violation_count": int(tensor["violation_count"]), "status": "PASS"})
    body: dict[str, Any] = {"schema_version": "stage1-s1-7-comparison-table-v1", "fixture_hash": _mapping(report["fixture"], field="fixture")["fixture_hash"], "rows": rows}
    body["table_hash"] = _canonical(body)
    return body


def _csv(name: str, rows: Sequence[Mapping[str, object]]) -> str:
    import csv
    from io import StringIO
    out = StringIO(newline="")
    keys = list(rows[0])
    writer = csv.DictWriter(out, fieldnames=keys, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue()


def _heatmap_rows(table: Mapping[str, Any]) -> list[dict[str, object]]:
    """Project every per-tensor metric onto both module and layer groups."""

    raw_rows = table.get("rows")
    if not isinstance(raw_rows, list):
        raise Stage1S17FormalError("S17_CHART_TABLE_ROWS_INVALID")
    maxima: dict[tuple[str, str, str], float] = {}
    for raw in raw_rows:
        row = _mapping(raw, field="comparison.row")
        if row.get("scope") != "per_tensor":
            continue
        metric, module, layer = row.get("metric"), row.get("module"), row.get("layer")
        scaled = row.get("max_scaled_error")
        if not all(isinstance(value, str) and value for value in (metric, module, layer)) or not isinstance(scaled, (int, float)) or isinstance(scaled, bool) or not math.isfinite(float(scaled)) or float(scaled) < 0.0:
            raise Stage1S17FormalError("S17_CHART_HEATMAP_INPUT_INVALID")
        for scope, group in (("module", module), ("layer", layer)):
            key = (scope, group, metric)
            maxima[key] = max(maxima.get(key, 0.0), float(scaled))
    if not maxima:
        raise Stage1S17FormalError("S17_CHART_HEATMAP_EMPTY")
    return [
        {"scope": scope, "group": group, "metric": metric, "max_scaled_error": value}
        for (scope, group, metric), value in sorted(maxima.items())
    ]


def _chart_rows(report: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, list[dict[str, object]]]:
    fixed_rows = [{"metric": row["metric"], "max_abs_error": row["max_abs_error"], "parameter": row["parameter"]} for row in table["rows"] if row["scope"] == "global"]
    heatmap_rows = _heatmap_rows(table)
    training = _mapping(report["training"], field="training")
    parity = training["per_step_parity"]
    if not isinstance(parity, list): raise Stage1S17FormalError("S17_TRAINING_PARITY_ROWS_INVALID")
    trajectory = []
    for item in parity:
        row = _mapping(item, field="training.parity")
        summaries = _mapping(row["accumulator_summaries"], field="training.accumulator_summaries")
        trajectory.append({"step": row["step"], "loss": row["loss"], "gradient_norm": row["gradient_norm"], "clip_factor": row["clip_factor"], "total_update_norm": row["total_update_norm"], "u_signed": summaries["signed"], "raw": summaries["raw"], "positive": summaries["positive"], "negative_mass": summaries["negative_mass"], "absolute": summaries["absolute"], "data_movement": summaries["data_movement"], "magnitude": summaries["magnitude"], "status": row["status"]})
    parameter_errors = [{"step": row["step"], "parameter_max_abs_error": _mapping(row["parameter_post_error"], field="training.parameter_post_error")["max_abs_error"], "optimizer_max_abs_error": _mapping(row["optimizer_state_error"], field="training.optimizer_state_error")["max_abs_error"]} for row in parity]
    timeline = [{key: item.get(key) for key in ("phase", "cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "cpu_rss_bytes")} for item in _mapping(report["resources"], field="resources")["timeline"]]
    return {"gradient-parity.csv": fixed_rows, "training-parity.csv": trajectory, "parameter-error.csv": parameter_errors, "resource-timeline.csv": timeline, "module-metric-heatmap.csv": heatmap_rows}


def _charts(work: Path, report: Mapping[str, Any], table: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    projections = _chart_rows(report, table); csv_hashes: dict[str, str] = {}; svg_hashes: dict[str, str] = {}
    for name, rows in projections.items():
        payload = _csv(name, rows); csv_path = work / name; csv_path.write_text(payload, encoding="utf-8"); csv_hashes[name] = _sha(csv_path)
        svg = work / name.replace(".csv", ".svg")
        svg.write_text(_svg_projection(name, rows), encoding="utf-8")
        svg_hashes[svg.name] = _sha(svg)
    return csv_hashes, svg_hashes


def _svg_projection(name: str, rows: Sequence[Mapping[str, object]]) -> str:
    """Deterministic SVG geometry, calculated only from the saved CSV rows."""

    height = 160
    axes = '<line x1="30" y1="140" x2="780" y2="140" /><line x1="30" y1="20" x2="30" y2="140" />'
    if name == "gradient-parity.csv":
        circles = "".join(f'<circle cx="{40 + index * 90}" cy="{140 - min(120, float(row["max_abs_error"]) * 1.0e7):.6f}" r="4" data-metric="{row["metric"]}" />' for index, row in enumerate(rows))
        geometry = f'{axes}{circles}'
    elif name == "module-metric-heatmap.csv":
        maximum = max(float(row["max_scaled_error"]) for row in rows) or 1.0
        rectangles = "".join(f'<rect x="{20 + (index % 10) * 76}" y="{20 + (index // 10) * 38}" width="72" height="34" fill="rgb({int(255 * float(row["max_scaled_error"]) / maximum)},0,0)" data-scope="{row["scope"]}" data-group="{row["group"]}" data-metric="{row["metric"]}" data-max-scaled-error="{row["max_scaled_error"]}" />' for index, row in enumerate(rows))
        geometry = rectangles
        height = 20 + 38 * math.ceil(len(rows) / 10) + 20
    elif name == "training-parity.csv":
        points = " ".join(f'{30 + index * 700 / max(1, len(rows) - 1):.6f},{140 - min(120, float(row["loss"]) * 10):.6f}' for index, row in enumerate(rows))
        geometry = f'{axes}<polyline fill="none" points="{points}" data-series="loss" />'
    elif name == "parameter-error.csv":
        maximum = max(max(float(row["parameter_max_abs_error"]), float(row["optimizer_max_abs_error"])) for row in rows) or 1.0
        parameter_points = " ".join(f'{30 + index * 700 / max(1, len(rows) - 1):.6f},{140 - 120 * float(row["parameter_max_abs_error"]) / maximum:.6f}' for index, row in enumerate(rows))
        optimizer_points = " ".join(f'{30 + index * 700 / max(1, len(rows) - 1):.6f},{140 - 120 * float(row["optimizer_max_abs_error"]) / maximum:.6f}' for index, row in enumerate(rows))
        geometry = f'{axes}<polyline fill="none" points="{parameter_points}" data-series="parameters" /><polyline fill="none" points="{optimizer_points}" data-series="optimizer" />'
    elif name == "resource-timeline.csv":
        maximum = max(max(1, int(row["cuda_peak_allocated_bytes"] or 0)) for row in rows)
        points = " ".join(f'{30 + index * 700 / max(1, len(rows) - 1):.6f},{140 - 120 * int(row["cuda_peak_allocated_bytes"] or 0) / maximum:.6f}' for index, row in enumerate(rows))
        geometry = f'{axes}<polyline fill="none" points="{points}" data-series="cuda_peak_allocated_bytes" />'
    else:
        raise Stage1S17FormalError("S17_SVG_PROJECTION_UNKNOWN")
    return f'<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://www.w3.org/2000/svg" width="800" height="{height}" viewBox="0 0 800 {height}" data-source="{name}" data-row-count="{len(rows)}"><title>{name}</title><desc>Exact projection of {name}</desc>{geometry}</svg>\n'


def _verify_charts(root: Path, report: Mapping[str, Any], table: Mapping[str, Any], csv_hashes: Mapping[str, str], svg_hashes: Mapping[str, str]) -> bool:
    projections = _chart_rows(report, table)
    if set(csv_hashes) != set(projections) or set(svg_hashes) != {name.replace(".csv", ".svg") for name in projections}: return False
    for name, rows in projections.items():
        if _sha(root / name) != csv_hashes[name] or (root / name).read_text(encoding="utf-8") != _csv(name, rows): return False
        svg = root / name.replace(".csv", ".svg")
        text = svg.read_text(encoding="utf-8")
        if _sha(svg) != svg_hashes[svg.name] or text != _svg_projection(name, rows): return False
    return True


def _validate_report_context(
    report: Mapping[str, Any], *, execution_commit: str, run_token: str,
    fixture: Mapping[str, Any], assets: Mapping[str, Any], approved_gpu_uuid: str,
    physical_gpu_index: int,
) -> None:
    """Bind the worker report to this exact plan and UUID-isolated discovery."""

    if (
        report.get("execution_commit") != execution_commit
        or report.get("run_token") != run_token
        or _mapping(report.get("fixture"), field="report.fixture") != dict(fixture)
        or _mapping(report.get("assets"), field="report.assets") != dict(assets)
    ):
        raise Stage1S17FormalError("S17_WORKER_REPORT_CONTEXT_INVALID")
    device = _mapping(report.get("device"), field="report.device")
    if (
        device.get("logical_device") != 0
        or device.get("cuda_visible_devices") != approved_gpu_uuid
        or device.get("physical_gpu_uuid") != approved_gpu_uuid
        or device.get("physical_gpu_index_at_discovery") != physical_gpu_index
        or device.get("cuda_device_count") != 1
        or not isinstance(device.get("device_name"), str)
        or not str(device["device_name"]).startswith("NVIDIA A100")
    ):
        raise Stage1S17FormalError("S17_WORKER_REPORT_DEVICE_CONTEXT_INVALID")


def _strict_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise Stage1S17FormalError(f"S17_RESOURCE_BOUND_INTEGER_INVALID:{field}")
    return value


def _validate_temporary_state_bounds(value: object) -> Mapping[str, int]:
    """Recompute the fixed allocation rule from persisted memory samples."""

    bounds = _mapping(value, field="training.temporary_state_bounds")
    required = {
        "allocated_before", "allocated_after", "allocated_growth_limit",
        "reserved_before", "reserved_after",
        "reserved_growth_explained_as_allocator_cache",
    }
    if set(bounds) != required:
        raise Stage1S17FormalError("S17_RESOURCE_BOUND_FIELDS_INVALID")
    parsed = {field: _strict_nonnegative_int(bounds[field], field=field) for field in required}
    expected_limit = max(16 * 1024**2, math.ceil(parsed["allocated_before"] * 0.02))
    if (
        parsed["allocated_growth_limit"] != expected_limit
        or parsed["allocated_after"] > parsed["allocated_before"] + expected_limit
        or parsed["reserved_growth_explained_as_allocator_cache"]
        != max(0, parsed["reserved_after"] - parsed["reserved_before"])
    ):
        raise Stage1S17FormalError("S17_RESOURCE_BOUND_RECOMPUTE_FAILED")
    return parsed


def _validate_trace_observer_contract(trace: Mapping[str, Any], *, statistics: bool) -> None:
    """Freeze the two-step observer wire, including tracker-only snapshots."""

    rows = trace.get("observer_rows")
    expected_boundaries = ["gradient_ready", "parameter_post", "attempt_commit"] * 2
    if (
        not isinstance(rows, list)
        or [row.get("boundary") if isinstance(row, Mapping) else None for row in rows]
        != expected_boundaries
    ):
        raise Stage1S17FormalError("S17_TRAINING_OBSERVER_BOUNDARY_SEQUENCE_INVALID")
    fields = trace.get("accumulator_fields")
    if not isinstance(fields, list) or set(fields) != (ACCUMULATOR_FIELDS if statistics else set()) or len(fields) != (len(ACCUMULATOR_FIELDS) if statistics else 0):
        raise Stage1S17FormalError("S17_TRAINING_ACCUMULATOR_FIELDS_INVALID")
    commits = [row for row in rows if isinstance(row, Mapping) and row.get("boundary") == "attempt_commit"]
    if len(commits) != 2:
        raise Stage1S17FormalError("S17_TRAINING_OBSERVER_COMMIT_COUNT_INVALID")
    for successful_steps, commit in enumerate(commits, start=1):
        snapshot = commit.get("importance_snapshot")
        if not statistics:
            if snapshot is not None:
                raise Stage1S17FormalError("S17_TRAINING_OFF_SNAPSHOT_INVALID")
            continue
        value = _mapping(snapshot, field="training.importance_snapshot")
        if value.get("successful_steps") != successful_steps or value.get("skipped_steps") != 0:
            raise Stage1S17FormalError("S17_TRAINING_ON_SNAPSHOT_STEP_INVALID")


def _validate_registry_audit(report: Mapping[str, Any]) -> None:
    """Require the persisted audit to prove the fresh-model registry reload."""

    registry = _mapping(report["registry"], field="registry")
    audit = _mapping(report["registry_audit"], field="registry_audit")
    if (
        audit.get("eligible_numel") != audit.get("model_trainable_numel")
        or audit.get("coordinate_registry_hash") != audit.get("reload_coordinate_registry_hash")
        or audit.get("fresh_reload_coordinate_registry_hash") != registry.get("coordinate_registry_hash")
        or audit.get("fresh_reload_optimizer_contract_hash") != registry.get("optimizer_contract_hash")
        or audit.get("fresh_reload_runtime_layout_hash") != registry.get("runtime_layout_hash")
        or audit.get("shared_weight_alias_contract") != "registry_from_model_remove_duplicate_false"
    ):
        raise Stage1S17FormalError("S17_REGISTRY_AUDIT_INVALID")


def _validate_report(
    report: Mapping[str, Any], *, execution_commit: str, run_token: str,
    fixture: Mapping[str, Any], assets: Mapping[str, Any], approved_gpu_uuid: str,
    physical_gpu_index: int,
) -> None:
    expected = {"schema_version", "status", "task_id", "execution_commit", "run_token", "fixture", "device", "assets", "registry", "registry_audit", "fixed_state", "training", "arrays", "resources", "offline_guard_ref", "tokenizer_class", "offline_provider_loads", "artifact_hash"}
    if set(report) != expected or not _self_hash(report) or report.get("schema_version") != "stage1-s1-7-worker-report-v1" or report.get("status") != "PASS" or report.get("task_id") != TASK_ID:
        raise Stage1S17FormalError("S17_WORKER_REPORT_SCHEMA_INVALID")
    _validate_report_context(
        report, execution_commit=execution_commit, run_token=run_token,
        fixture=fixture, assets=assets, approved_gpu_uuid=approved_gpu_uuid,
        physical_gpu_index=physical_gpu_index,
    )
    fixture = _mapping(report["fixture"], field="report.fixture")
    if fixture.get("fixture_id") != FIXTURE_ID or fixture.get("fixture_hash") != _canonical({key: item for key, item in fixture.items() if key != "fixture_hash"}):
        raise Stage1S17FormalError("S17_WORKER_FIXTURE_INVALID")
    training = _mapping(report["training"], field="training")
    if training.get("bitwise_final_state_equal") is not True:
        raise Stage1S17FormalError("S17_TRAINING_PARITY_INVALID")
    on = _mapping(training.get("statistics_on"), field="training.statistics_on")
    off = _mapping(training.get("statistics_off"), field="training.statistics_off")
    if on.get("temporary_state_bounded") is not True or not isinstance(on.get("observer_rows"), list) or not isinstance(off.get("observer_rows"), list) or len(on["observer_rows"]) != 6 or len(off["observer_rows"]) != 6 or not isinstance(training.get("per_step_parity"), list) or len(training["per_step_parity"]) != 2:
        raise Stage1S17FormalError("S17_TRAINING_STEP_EVIDENCE_INVALID")
    for trace, estimator in ((off, None), (on, "u")):
        records = trace.get("records")
        if (
            not isinstance(records, list) or len(records) != 2
            or any(
                not isinstance(record, Mapping)
                or record.get("status") != "COMMITTED"
                or record.get("estimator_name") != estimator
                for record in records
            )
        ):
            raise Stage1S17FormalError("S17_TRAINING_RECORD_WIRE_INVALID")
    _validate_trace_observer_contract(off, statistics=False)
    _validate_trace_observer_contract(on, statistics=True)
    component_fields = {"parameters_sha256", "buffers_sha256", "model_modes_sha256", "torch_cpu_rng_sha256", "torch_cuda_rng_sha256", "python_rng_sha256", "numpy_rng_sha256", "scheduler", "optimizer_tensors_sha256", "optimizer_groups_sha256"}
    for left, right in zip(off["observer_rows"], on["observer_rows"], strict=True):
        left_row, right_row = _mapping(left, field="statistics_off.observer"), _mapping(right, field="statistics_on.observer")
        if left_row.get("boundary") == "attempt_commit":
            left_components = _mapping(left_row.get("runtime_state_components"), field="statistics_off.runtime_components")
            right_components = _mapping(right_row.get("runtime_state_components"), field="statistics_on.runtime_components")
            if set(left_components) != component_fields or left_components != right_components:
                raise Stage1S17FormalError("S17_TRAINING_COMPONENT_PARITY_INVALID")
    for row in training["per_step_parity"]:
        parity = _mapping(row, field="training.per_step_parity")
        for field in ("parameter_post_error", "optimizer_state_error"):
            if _mapping(parity.get(field), field=f"training.{field}").get("within_t32") is not True:
                raise Stage1S17FormalError("S17_TRAINING_POST_STEP_TENSOR_PARITY_INVALID")
    fixed = _mapping(report["fixed_state"], field="fixed_state")
    before_components = _mapping(fixed.get("state_before_components"), field="fixed_state.before_components")
    after_components = _mapping(fixed.get("state_after_components"), field="fixed_state.after_components")
    expected_components = {"parameters_sha256", "buffers_sha256", "model_modes_sha256", "torch_cpu_rng_sha256", "torch_cuda_rng_sha256", "python_rng_sha256", "numpy_rng_sha256", "scheduler", "optimizer_tensors_sha256", "optimizer_groups_sha256"}
    if set(before_components) != expected_components or before_components != after_components or fixed.get("state_before_sha256") != fixed.get("state_after_sha256") or fixed.get("microbatch_count") != 4:
        raise Stage1S17FormalError("S17_FIXED_STATE_RESTORATION_INVALID")
    comparisons = _mapping(fixed.get("comparisons"), field="fixed_state.comparisons")
    if set(comparisons) != {"full_vs_online", "explicit_u_vs_streaming_u"} or any(_mapping(item, field="fixed_state.comparison").get("within_t32") is not True for item in comparisons.values()):
        raise Stage1S17FormalError("S17_FIXED_STATE_T32_INVALID")
    resources = _mapping(report["resources"], field="resources")
    required_resources = {"timeline", "fixed_and_training_seconds", "fixed_gradient_seconds", "training_seconds", "safetensors_serialization_seconds", "gradient_dump_bytes", "retained_tensor_bytes", "wall_seconds"}
    if set(resources) != required_resources or any(not isinstance(resources.get(field), (int, float)) or isinstance(resources.get(field), bool) or float(resources[field]) < 0.0 for field in ("fixed_and_training_seconds", "fixed_gradient_seconds", "training_seconds", "safetensors_serialization_seconds", "wall_seconds")) or not isinstance(resources.get("retained_tensor_bytes"), int) or resources["retained_tensor_bytes"] <= 0:
        raise Stage1S17FormalError("S17_RESOURCE_REPORT_INVALID")
    _validate_temporary_state_bounds(off.get("temporary_state_bounds"))
    _validate_temporary_state_bounds(on.get("temporary_state_bounds"))
    if float(resources["fixed_gradient_seconds"]) + float(resources["training_seconds"]) > float(resources["fixed_and_training_seconds"]) + 1.0e-6 or float(resources["fixed_and_training_seconds"]) > float(resources["wall_seconds"]) + 1.0e-6:
        raise Stage1S17FormalError("S17_RESOURCE_TIMELINE_INVALID")
    loads = _mapping(report["offline_provider_loads"], field="offline_provider_loads")
    if loads != {"model": 3, "tokenizer": 3, "all_inside_guard": True}:
        raise Stage1S17FormalError("S17_OFFLINE_PROVIDER_LOAD_EVIDENCE_INVALID")
    audit = _mapping(report["registry_audit"], field="registry_audit")
    _validate_registry_audit(report)


def _task_catalog_decision_exempt() -> bool:
    """The config keeps its global default; this task does not consume Stage 2."""

    from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG

    return DEFAULT_TASK_CATALOG.get(TASK_ID).formal_eligibility.requires_estimator_decision is False


def _success_marker(root: Path, *, gate_hash: str, validation_sha256: str) -> None:
    """Create the terminal success marker only after all PASS checks finish."""

    if (root / "failed.json").exists() or (root / "success.json").exists():
        raise Stage1S17FormalError("S17_ATTEMPT_MARKER_CONFLICT")
    _write(root / "success.json", _with_hash({
        "schema_version": "stage1-s1-7-attempt-success-v1", "status": "PASS",
        "completed_at": _now(), "gate_artifact_hash": gate_hash,
        "validation_sha256": validation_sha256, "failed_marker_present": False,
    }))


def _negative_checks(
    work: Path, fixture: Mapping[str, Any], report: Mapping[str, Any], *,
    execution_commit: str, run_token: str, assets: Mapping[str, Any],
    approved_gpu_uuid: str, physical_gpu_index: int,
) -> bool:
    """Exercise the real malformed-input and marker rejection paths."""

    malformed = dict(fixture)
    malformed["records"] = {"a": [0], "b": [4, 5, 6, 7], "training": [[8, 9, 10, 11], [12, 13, 14, 15]]}
    malformed["fixture_hash"] = _canonical({key: item for key, item in malformed.items() if key != "fixture_hash"})
    try:
        _validate_fixture_manifest(malformed)
    except Stage1S17FormalError:
        rejected_fixture = True
    else:
        rejected_fixture = False
    arrays = _mapping(report["arrays"], field="arrays")
    tampered_report = dict(report); tampered_arrays = dict(arrays); tampered_arrays["file_sha256"] = "0" * 64; tampered_report["arrays"] = tampered_arrays
    try:
        _load_arrays(work, tampered_report)
    except Stage1S17FormalError:
        rejected_hash = True
    else:
        rejected_hash = False
    # Rehashing a modified report is not a bypass: its fields must still bind
    # to the plan and the exact GPU discovery that launched this worker.
    tampered_context = dict(report)
    tampered_context["run_token"] = "0" * 64
    tampered_context["artifact_hash"] = _canonical({key: value for key, value in tampered_context.items() if key != "artifact_hash"})
    try:
        _validate_report(
            tampered_context, execution_commit=execution_commit,
            run_token=run_token, fixture=fixture, assets=assets,
            approved_gpu_uuid=approved_gpu_uuid,
            physical_gpu_index=physical_gpu_index,
        )
    except Stage1S17FormalError:
        rejected_context = True
    else:
        rejected_context = False
    probe = work / "negative-marker-probe"
    probe.mkdir(exist_ok=False)
    _failure_marker(probe, Stage1S17FormalError("expected marker test"), phase="negative_probe")
    try:
        _success_marker(probe, gate_hash="0" * 64, validation_sha256="1" * 64)
    except Stage1S17FormalError as error:
        marker_mutual_exclusion = str(error) == "S17_ATTEMPT_MARKER_CONFLICT"
    else:
        marker_mutual_exclusion = False
    marker = _with_hash({"schema_version": "stage1-s1-7-negative-check-v1", "status": "EXPECTED_FAILURE", "checks": {"fixture_m1_rejected": rejected_fixture, "array_hash_tamper_rejected": rejected_hash, "report_context_joint_rehash_rejected": rejected_context, "marker_mutual_exclusion_rejected": marker_mutual_exclusion, "success_marker_absent_before_publish": not (work / "success.json").exists()}})
    _write(work / "negative-checks.json", marker)
    return all(marker["checks"].values())


def _resource_summary(work: Path) -> dict[str, Any]:
    """Measure actual attempt writes without conflating tensors and files.

    The historical producer checkout is a pre-lease, local source replay.  It
    is reported separately rather than being smuggled into the experiment's
    safetensors or retained-array accounting.
    """

    historical_root = work / "historical-g3"
    files: list[tuple[str, int]] = []
    historical_bytes = 0
    for path in sorted(work.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        try:
            logical = path.relative_to(work).as_posix()
        except ValueError as error:  # pragma: no cover - defensive resolve guard
            raise Stage1S17FormalError("S17_RESOURCE_PATH_ESCAPE") from error
        if path.is_relative_to(historical_root):
            historical_bytes += size
        elif logical != "resource-summary.json":
            files.append((logical, size))
    total = sum(size for _, size in files)
    return _with_hash({
        "schema_version": "stage1-s1-7-resource-summary-v1",
        "status": "PASS",
        "scope": "pre-gate-attempt-files-excluding-resource-summary",
        "attempt_file_total_bytes": total,
        "attempt_file_count": len(files),
        "historical_checkout_bytes": historical_bytes,
        "historical_checkout_classification": "prelease_local_historical_producer_source_replay",
        "large_files": [
            {"path": path, "bytes": size}
            for path, size in sorted(files, key=lambda item: (-item[1], item[0]))[:20]
        ],
    })


def _validate_resource_summary(work: Path, value: Mapping[str, Any]) -> None:
    expected = _resource_summary(work)
    if dict(value) != expected:
        raise Stage1S17FormalError("S17_RESOURCE_SUMMARY_RECOMPUTE_FAILED")


def _failure_marker(work: Path, error: BaseException, *, phase: str) -> None:
    if not work.exists() or (work / "success.json").exists():
        return
    if not (work / "failed.json").exists():
        _write(work / "failed.json", _with_hash({"schema_version": "stage1-s1-7-attempt-failure-v1", "status": "FAILED", "phase": phase, "error_type": type(error).__name__, "error": str(error)[:1000], "failed_at": _now(), "success_marker_present": False}))


def execute(*, repository: str | Path, data_root: str | Path, s1_6_index_ref: str, g3_resolution_ref: str, gpu_capability_ref: str, approved_gpu_uuid: str, attempt_id: str, lease_owner: str, timeout_seconds: int = 2700) -> dict[str, str]:
    from param_importance_nlp.contracts.jsonio import load_canonical_json

    repository_root, root = Path(repository).resolve(strict=True), Path(data_root).resolve(strict=True)
    if str(repository_root / "src") not in sys.path: sys.path.insert(0, str(repository_root / "src"))
    commit = _git(repository_root, "rev-parse", "HEAD")
    if _COMMIT.fullmatch(commit) is None or _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S17FormalError("S17_FORMAL_REQUIRES_CLEAN_WORKTREE")
    if not _ATTEMPT.fullmatch(attempt_id) or not isinstance(lease_owner, str) or not lease_owner:
        raise Stage1S17FormalError("S17_ATTEMPT_OR_OWNER_INVALID")
    target = root / "evidence" / "stage1" / "s1-7-formal" / commit / attempt_id
    work = root / "tmp" / "stage1-s1-7" / commit / attempt_id
    if target.exists() or work.exists(): raise Stage1S17FormalError("S17_ATTEMPT_ALREADY_EXISTS")
    work.mkdir(parents=True)
    lease = None; lease_release_attempted = False; staging: Path | None = None; phase = "bootstrap"
    try:
        upstream = load_s1_6(root, s1_6_index_ref, repository_root)
        phase = "asset_full_hash_before_lease"; fixture, asset_plan = _asset_fixture(repository_root, root, work, g3_resolution_ref)
        _write(work / "fixture-manifest.json", fixture)
        capability = _load_capability(root, gpu_capability_ref, approved_gpu_uuid)
        phase = "gpu_preflight"; gpu = discover_gpu(approved_gpu_uuid)
        stat = os.statvfs(root); free_bytes = stat.f_bavail * stat.f_frsize
        cache_root = (root / "cache").resolve(strict=True)
        if not cache_root.is_dir() or cache_root.parent != root or free_bytes < 100 * 1024**3 or (root / "tmp" / "stage1-s1-7" / commit / attempt_id).exists() is False:
            raise Stage1S17FormalError("S17_DISK_PREFLIGHT_FAILED")
        preflight = _with_hash({"schema_version": "stage1-s1-7-preflight-v1", "status": "PASS", "execution_commit": commit, "approved_gpu_uuid": approved_gpu_uuid, "gpu_capability": capability, "gpu": gpu, "fixture_hash": fixture["fixture_hash"], "fixture_file_sha256": asset_plan["worker_assets"]["pile"]["fixture_file_sha256"], "g3_historical_producer": asset_plan["historical_producer"], "g3_historical_replay_sha256": asset_plan["historical_replay_sha256"], "g3_resolution_commit_artifact_hash": EXPECTED_G3_COMMIT_ARTIFACT_HASH, "g3_resolution_payload_hash": asset_plan["g3_resolution_artifact_hash"], "data_free_bytes": free_bytes, "full_hash_completed_before_lease": True})
        _write(work / "preflight.json", preflight)
        from param_importance_nlp.runtime.operations import GpuLeaseIdentity, ProjectGpuLease
        environment_hash = _canonical({"preflight": preflight["artifact_hash"], "g3_resolution": asset_plan["g3_resolution_artifact_hash"]})
        identity = GpuLeaseIdentity(run_id=f"s17-{attempt_id}", lease_id=f"s17-{attempt_id}", gpu_uuids=(approved_gpu_uuid,), owner=lease_owner, config_hash=_canonical(fixture), environment_hash=environment_hash)
        phase = "lease_acquire"; lease = ProjectGpuLease(root, identity); lease.acquire(); lease.heartbeat()
        # Recheck immediately after acquiring our project lease; it remains an
        # advisory lease and never authorizes use of a newly appearing process.
        phase = "post_lease_gpu_recheck"; post_lease_gpu = discover_gpu(approved_gpu_uuid)
        _write(work / "post-lease-gpu.json", _with_hash({"schema_version": "stage1-s1-7-post-lease-gpu-v1", "status": "PASS", "gpu": post_lease_gpu, "uuid_still_approved": post_lease_gpu["selected"]["uuid"] == approved_gpu_uuid}))
        probe = _min_cuda_probe(repository_root, approved_gpu_uuid, work)
        _write(work / "cuda-min-probe.json", _with_hash({"schema_version": "stage1-s1-7-cuda-min-probe-v1", "status": "PASS", "probe": probe, "logical_device": 0, "approved_gpu_uuid": approved_gpu_uuid}))
        lease.heartbeat()
        run_token = hashlib.sha256(f"{commit}:{attempt_id}:{time.time_ns()}".encode()).hexdigest()
        plan_body: dict[str, Any] = {"schema_version": "stage1-s1-7-worker-plan-v1", "task_id": TASK_ID, "execution_commit": commit, "approved_gpu_uuid": approved_gpu_uuid, "physical_gpu_index": gpu["selected"]["physical_index"], "fixture": fixture, "fixture_safetensors_ref": asset_plan["fixture_file"], "fixture_safetensors_sha256": asset_plan["worker_assets"]["pile"]["fixture_file_sha256"], "model_root": asset_plan["model_root"], "tokenizer_root": asset_plan["tokenizer_root"], "cache_root": str(cache_root), "assets": asset_plan["worker_assets"], "output_ref": str((work / "worker-report.json").resolve()), "run_token": run_token}
        plan = _with_hash(plan_body)
        expected_report_context = {
            "execution_commit": commit, "run_token": run_token,
            "fixture": fixture, "assets": asset_plan["worker_assets"],
            "approved_gpu_uuid": approved_gpu_uuid,
            "physical_gpu_index": gpu["selected"]["physical_index"],
        }
        _write(work / "attempt-start.json", _with_hash({"schema_version": "stage1-s1-7-attempt-start-v1", "status": "STARTED", "run_token": run_token, "parent_fingerprint": _pid_fingerprint(os.getpid(), run_token), "lease": identity.to_dict(), "preflight_sha256": _sha(work / "preflight.json")}))
        phase = "real_gpu_worker"; worker_report = _worker(repository_root, work, plan, approved_gpu_uuid, timeout_seconds, lease)
        guard_ref = worker_report.get("offline_guard_ref")
        if not isinstance(guard_ref, str) or not guard_ref or not (work / "network" / guard_ref).is_file():
            raise Stage1S17FormalError("S17_NETWORK_AUDIT_MISSING")
        shutil.copy2(work / "network" / guard_ref, work / "network-audit.json")
        post_worker_gpu = discover_gpu(approved_gpu_uuid)
        _write(work / "post-worker-gpu.json", _with_hash({"schema_version": "stage1-s1-7-post-worker-gpu-v1", "status": "PASS", "gpu": post_worker_gpu, "no_residual_compute_processes": True}))
        # The lease covers only the GPU phase.  Offline replay/publication can
        # still fail closed after this record without mislabelling it a full
        # formal-gate success.
        lease_release_attempted = True
        history = lease.release(outcome="GPU_PHASE_SUCCESS")
        lease = None
        shutil.copy2(history, work / "lease-history.json")
        _validate_report(worker_report, **expected_report_context)
        arrays, array_manifest = _load_arrays(work, worker_report)
        _write(work / "arrays-manifest.json", array_manifest)
        phase = "offline_replay"; replay = _oracle_replay(arrays); _write(work / "replay-validation.json", replay)
        table = _comparison_table(worker_report, replay); _write(work / "comparison-table.json", table)
        csv_hashes, svg_hashes = _charts(work, worker_report, table)
        if not _verify_charts(work, worker_report, table, csv_hashes, svg_hashes): raise Stage1S17FormalError("S17_CHART_PROJECTION_INVALID")
        if not _negative_checks(work, fixture, worker_report, **{key: value for key, value in expected_report_context.items() if key != "fixture"}): raise Stage1S17FormalError("S17_NEGATIVE_CHECK_FAILED")
        _write(work / "resource-summary.json", _resource_summary(work))
        persisted_preflight = _mapping(load_canonical_json(work / "preflight.json"), field="persisted.preflight")
        persisted_post_lease = _mapping(load_canonical_json(work / "post-lease-gpu.json"), field="persisted.post_lease")
        persisted_probe = _mapping(load_canonical_json(work / "cuda-min-probe.json"), field="persisted.probe")
        persisted_post_worker = _mapping(load_canonical_json(work / "post-worker-gpu.json"), field="persisted.post_worker")
        persisted_network = _mapping(load_canonical_json(work / "network-audit.json"), field="persisted.network")
        persisted_resource_summary = _mapping(load_canonical_json(work / "resource-summary.json"), field="persisted.resource_summary")
        _validate_resource_summary(work, persisted_resource_summary)
        persisted_worker_report = _mapping(load_canonical_json(work / "worker-report.json"), field="persisted.worker_report")
        _validate_report(persisted_worker_report, **expected_report_context)
        resources = _mapping(persisted_worker_report["resources"], field="resources")
        training = _mapping(persisted_worker_report["training"], field="training")
        fixed = _mapping(persisted_worker_report["fixed_state"], field="fixed_state")
        registry_audit = _mapping(persisted_worker_report["registry_audit"], field="registry_audit")
        fixed_comparisons = _mapping(fixed["comparisons"], field="fixed.comparisons")
        replay_checks = _mapping(replay["checks"], field="replay.checks")
        if set(replay_checks) != REPLAY_CHECK_IDS:
            raise Stage1S17FormalError("S17_REPLAY_CHECK_SET_INVALID")
        persisted_bounds = _validate_temporary_state_bounds(
            _mapping(training["statistics_on"], field="statistics_on").get("temporary_state_bounds")
        )
        check_map: dict[str, tuple[bool, str]] = {
            "s1_6_handoff": (upstream["s1_6_gate_artifact_hash"] == EXPECTED_S1_6_GATE_HASH, "exact current S1.6 index and gate hash"),
            "qualified_assets": (asset_plan["worker_assets"]["pile"]["full_hash_seconds"] > 0 and int(asset_plan["worker_assets"]["pile"]["full_hash_bytes"]) == int(asset_plan["worker_assets"]["pile"]["qualified_resolution_hashed_bytes"]) + int(asset_plan["worker_assets"]["pile"]["dataset_rehash_bytes"]) and (work / asset_plan["fixture_file"]).is_file() and _sha(work / asset_plan["fixture_file"]) == asset_plan["worker_assets"]["pile"]["fixture_file_sha256"], "full Pile open/hash bytes completed before lease and fixture hash reread"),
            "historical_producer_replay": (
                _self_hash(_mapping(load_canonical_json(work / "historical-producer-attestation.json"), field="historical.attestation"))
                and _mapping(load_canonical_json(work / "historical-producer-attestation.json"), field="historical.attestation").get("critical_patch_sha256") == EXPECTED_HISTORICAL_G3_PATCH_SHA256
                and _mapping(load_canonical_json(work / "historical-g3-replay.json"), field="historical.replay").get("status") == "PASS"
                and _mapping(load_canonical_json(work / "historical-g3-replay.json"), field="historical.replay").get("replay_hash") == _canonical({key: value for key, value in _mapping(load_canonical_json(work / "historical-g3-replay.json"), field="historical.replay").items() if key != "replay_hash"}),
                "historical 54b1 producer replay and exact three-file patch attestation reread",
            ),
            "approved_gpu_uuid": (approved_gpu_uuid in capability["allowed_gpu_uuids"] and gpu["selected"]["uuid"] == approved_gpu_uuid and post_lease_gpu["selected"]["uuid"] == approved_gpu_uuid and post_worker_gpu["selected"]["uuid"] == approved_gpu_uuid, "approved UUID retained across discovery and GPU phase"),
            "gpu_preflight_twice": (_self_hash(persisted_preflight) and _self_hash(persisted_post_lease) and _self_hash(persisted_post_worker) and persisted_post_lease.get("uuid_still_approved") is True and persisted_post_worker.get("no_residual_compute_processes") is True, "persisted pre/post lease and post-worker idle evidence"),
            "cuda_minimum_allocation": (_self_hash(persisted_probe) and persisted_probe.get("status") == "PASS" and _mapping(persisted_probe.get("probe"), field="probe").get("stdout_sha256") == _sha(work / "cuda-min-probe.stdout.txt") and _mapping(persisted_probe.get("probe"), field="probe").get("stderr_sha256") == _sha(work / "cuda-min-probe.stderr.txt"), "persisted UUID-isolated CUDA allocation probe and logs"),
            "offline_guard": (_self_hash(persisted_network) and persisted_network.get("status") == "COMPLETE" and persisted_network.get("external_attempts") == [], "persisted network audit reports no external attempts"),
            "registry_reload": (
                registry_audit.get("coordinate_registry_hash") == registry_audit.get("reload_coordinate_registry_hash")
                and registry_audit.get("fresh_reload_coordinate_registry_hash") == _mapping(persisted_worker_report["registry"], field="registry").get("coordinate_registry_hash")
                and registry_audit.get("fresh_reload_optimizer_contract_hash") == _mapping(persisted_worker_report["registry"], field="registry").get("optimizer_contract_hash")
                and registry_audit.get("fresh_reload_runtime_layout_hash") == _mapping(persisted_worker_report["registry"], field="registry").get("runtime_layout_hash")
                and registry_audit.get("eligible_numel") == registry_audit.get("model_trainable_numel"),
                "fresh model reload registry coordinate/optimizer/runtime hashes and trainable-numel coverage",
            ),
            "fixed_state_t32": (all(_mapping(item, field="fixed.comparison").get("within_t32") is True for item in fixed_comparisons.values()), "fixed-state production comparisons pass frozen T32"),
            "independent_oracle": (replay["oracle_import_isolated"] is True and all(_mapping(item, field="replay.check").get("within_t32") is True for item in replay_checks.values()), "array-only import-isolated oracle reread all fixed maps"),
            "statistics_path_parity": (training.get("bitwise_final_state_equal") is True, "stats-off/on final state bitwise equal"),
            "two_steps": (isinstance(training.get("per_step_parity"), list) and len(training["per_step_parity"]) == 2, "exactly two committed training steps"),
            "safetensors_manifest": (array_manifest.get("file_sha256") == _sha(work / "s1-7-arrays.safetensors") and array_manifest.get("file_size_bytes") == (work / "s1-7-arrays.safetensors").stat().st_size, "safetensors file and manifest reread"),
            "resource_evidence": (
                isinstance(resources.get("timeline"), list)
                and len(resources["timeline"]) == 3
                and resources.get("gradient_dump_bytes") == (work / "s1-7-arrays.safetensors").stat().st_size
                and _self_hash(persisted_resource_summary)
                and persisted_resource_summary.get("status") == "PASS"
                and persisted_resource_summary.get("scope") == "pre-gate-attempt-files-excluding-resource-summary"
                and isinstance(persisted_resource_summary.get("attempt_file_total_bytes"), int)
                and int(persisted_resource_summary["attempt_file_total_bytes"]) >= int(resources["gradient_dump_bytes"])
                and isinstance(persisted_resource_summary.get("historical_checkout_bytes"), int)
                and persisted_resource_summary.get("historical_checkout_classification") == "prelease_local_historical_producer_source_replay"
                and isinstance(resources.get("retained_tensor_bytes"), int)
                and resources["retained_tensor_bytes"] > 0
                and len(fixed.get("microbatch_resources", [])) == 8
                and all(isinstance(_mapping(row, field="microbatch_resource").get("backward_seconds"), (int, float)) and float(_mapping(row, field="microbatch_resource")["backward_seconds"]) >= 0.0 for row in fixed["microbatch_resources"])
                and all(isinstance(resources.get(field), (int, float)) and float(resources[field]) >= 0.0 for field in ("fixed_and_training_seconds", "fixed_gradient_seconds", "training_seconds", "safetensors_serialization_seconds", "wall_seconds"))
                and float(resources["fixed_and_training_seconds"]) <= float(resources["wall_seconds"])
                and float(resources["fixed_gradient_seconds"]) + float(resources["training_seconds"]) <= float(resources["fixed_and_training_seconds"]) + 1.0e-6
                and persisted_bounds["allocated_after"] <= persisted_bounds["allocated_before"] + persisted_bounds["allocated_growth_limit"]
                and _mapping(training["statistics_on"], field="statistics_on").get("temporary_state_bounded") is True,
                "recomputed fixed/training timings, safetensors bytes and allocated step-local bound",
            ),
            "failure_markers": (not (work / "failed.json").exists() and (work / "negative-marker-probe" / "failed.json").is_file(), "real fixture/loader/marker rejection paths completed"),
            "charts_exact": (_verify_charts(work, worker_report, table, csv_hashes, svg_hashes), "CSV-to-SVG geometric projection reread"),
            "catalog_decision_exempt": (_task_catalog_decision_exempt(), "catalog declares no estimator-decision consumption for S1.7"),
        }
        if tuple(check_map) != GATE_CHECK_IDS or not all(passed for passed, _ in check_map.values()):
            raise Stage1S17FormalError("S17_GATE_REQUIREMENT_FAILED")
        requirements = {check_id: passed for check_id, (passed, _) in check_map.items()}
        gate = _with_hash({"schema_version": "stage1-s1-7-gate-record-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "requirements": requirements})
        _write(work / "g1-single-record.json", gate)
        role_files = {"fixture_manifest": "fixture-manifest.json", "single_gpu_report": "worker-report.json", "gradient_bundle": "arrays-manifest.json", "comparison_table": "comparison-table.json", "gate_record": "g1-single-record.json"}
        role_sha = {role: _sha(work / name) for role, name in role_files.items()}
        validation = _with_hash({"schema_version": "stage1-s1-7-validation-v1", "status": "PASS", "task_id": TASK_ID, "gate_id": GATE_ID, "producer_commit": commit, "checks": [{"check_id": check_id, "status": "PASS", "detail": detail} for check_id, (_, detail) in check_map.items()], "role_sha256": role_sha, "replay_sha256": _sha(work / "replay-validation.json"), "csv_sha256": csv_hashes, "svg_sha256": svg_hashes})
        _validate_validation_checks(validation)
        _write(work / "validation.json", validation)
        staging = target.parent / f".{attempt_id}.publishing"
        if staging.exists(): raise Stage1S17FormalError("S17_PUBLISH_STAGING_COLLISION")
        staging.mkdir(parents=True)
        reproduction_files = {
            "fixture_inputs": "fixture-inputs.safetensors", "gradient_arrays": "s1-7-arrays.safetensors",
            "historical_producer_attestation": "historical-producer-attestation.json", "historical_g3_replay": "historical-g3-replay.json",
            "historical_g3_replay_stdout": "historical-g3-replay.stdout.txt", "historical_g3_replay_stderr": "historical-g3-replay.stderr.txt",
            "preflight": "preflight.json", "post_lease_gpu": "post-lease-gpu.json", "post_worker_gpu": "post-worker-gpu.json",
            "cuda_min_probe": "cuda-min-probe.json", "cuda_min_probe_stdout": "cuda-min-probe.stdout.txt", "cuda_min_probe_stderr": "cuda-min-probe.stderr.txt",
            "lease_history": "lease-history.json", "network_audit": "network-audit.json", "attempt_start": "attempt-start.json",
            "worker_plan": "worker-plan.json", "worker_start": "worker-start.json", "worker_stdout": "worker.stdout.txt", "worker_stderr": "worker.stderr.txt",
            "negative_checks": "negative-checks.json", "resource_summary": "resource-summary.json",
        }
        publish_files = set(role_files.values()) | {"replay-validation.json", "validation.json"} | set(reproduction_files.values()) | set(csv_hashes) | set(svg_hashes)
        for name in sorted(publish_files):
            shutil.copy2(work / name, staging / name)
        reproduction_sha = {role: _sha(staging / name) for role, name in reproduction_files.items()}
        index = _with_hash({"schema_version": "stage1-s1-7-formalization-index-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "generator_git_commit": commit, "consumer_git_commit": commit, "git_branch": _git(repository_root, "branch", "--show-current"), "checked_at": _now(), "s1_6_index_ref": s1_6_index_ref, "s1_6_index_sha256": EXPECTED_S1_6_INDEX_SHA256, "s1_6_gate_artifact_hash": EXPECTED_S1_6_GATE_HASH, "s1_6_handoff": upstream, "role_refs": role_files, "role_sha256": role_sha, "reproduction_role_refs": reproduction_files, "reproduction_role_sha256": reproduction_sha, "gate_artifact_hash": gate["artifact_hash"], "csv_sha256": csv_hashes, "svg_sha256": svg_hashes, "validation_ref": "validation.json", "validation_sha256": _sha(work / "validation.json"), "replay_ref": "replay-validation.json", "replay_sha256": _sha(work / "replay-validation.json"), "next_task_ids": ["stage1.08_ddp_and_gradient_accumulation", "stage1.09_precision_clipping_and_optimizer_boundaries"]})
        _validate_role_schemas(repository_root, {"fixture_manifest": fixture, "gradient_bundle": array_manifest, "single_gpu_report": worker_report, "comparison_table": table, "gate_record": gate, "replay": replay, "validation": validation, "index": index})
        _write(staging / "index.json", index)
        staged_index = _mapping(load_canonical_json(staging / "index.json"), field="staged.index")
        staged_validation = _mapping(load_canonical_json(staging / "validation.json"), field="staged.validation")
        _validate_validation_checks(staged_validation)
        if (
            not _self_hash(staged_index)
            or staged_index != index
            or not _self_hash(staged_validation)
            or not _verify_charts(staging, worker_report, table, csv_hashes, svg_hashes)
            or any(_sha(staging / role_files[role]) != digest for role, digest in role_sha.items())
            or any(_sha(staging / reproduction_files[role]) != digest for role, digest in reproduction_sha.items())
        ):
            raise Stage1S17FormalError("S17_PUBLISH_VERIFICATION_FAILED")
        # The terminal marker is deliberately not an index role: creating it
        # earlier would allow a later schema/hash failure to strand a false
        # success in staging.  Its own two hashes bind it to the verified gate
        # and validation records.
        _success_marker(staging, gate_hash=str(gate["artifact_hash"]), validation_sha256=_sha(work / "validation.json"))
        success = _mapping(load_canonical_json(staging / "success.json"), field="staged.success")
        if not _self_hash(success) or success.get("gate_artifact_hash") != gate["artifact_hash"] or success.get("validation_sha256") != _sha(work / "validation.json"):
            raise Stage1S17FormalError("S17_SUCCESS_MARKER_VERIFICATION_FAILED")
        os.replace(staging, target)
        staging = None
        return {"index_ref": (target / "index.json").relative_to(root).as_posix(), "validation_ref": (target / "validation.json").relative_to(root).as_posix()}
    except BaseException as error:
        if staging is not None and staging.exists() and (staging / "success.json").is_file():
            # A failed rename or post-marker verification must not leave an
            # apparently successful publish candidate for later discovery.
            (staging / "success.json").unlink()
        _failure_marker(work, error, phase=phase)
        if lease is not None:
            if isinstance(error, Stage1S17ManualInterventionRequired):
                # Preserve the current lease record for an operator; releasing
                # it after an unknown process identity would invite collision.
                lease.close()
            elif lease_release_attempted:
                # A release call may have completed remotely before raising;
                # never issue a second release against the same lease ID.
                lease.close()
            else:
                lease_release_attempted = True
                try: lease.release(outcome="FAILED")
                except Exception: lease.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-6-index-ref", required=True); parser.add_argument("--g3-resolution-ref", required=True); parser.add_argument("--gpu-capability-ref", required=True)
    parser.add_argument("--approved-gpu-uuid", required=True); parser.add_argument("--attempt-id", required=True); parser.add_argument("--lease-owner", required=True); parser.add_argument("--timeout-seconds", type=int, default=2700)
    args = parser.parse_args(argv)
    print(execute(repository=args.repository, data_root=args.data_root, s1_6_index_ref=args.s1_6_index_ref, g3_resolution_ref=args.g3_resolution_ref, gpu_capability_ref=args.gpu_capability_ref, approved_gpu_uuid=args.approved_gpu_uuid, attempt_id=args.attempt_id, lease_owner=args.lease_owner, timeout_seconds=args.timeout_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
