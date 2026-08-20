"""Fail-closed formalizer for S1.8 real DDP/no-sync accumulation evidence.

The command is deliberately inert until an operator supplies four approved GPU
UUIDs.  It never accepts a physical GPU index, refreshes topology immediately
before the lease, runs an independent pre-route scale oracle before constructing
the fixture, and publishes no success marker after any controlled failure.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


TASK_ID = "stage1.08_ddp_and_gradient_accumulation"
GATE_ID = "G1-DDP"
EXPECTED_S1_7_PRODUCER = "dcc92506947c3ea30bed75542e006a26d5a2af1b"
EXPECTED_S1_7_INDEX_SHA256 = "4ca26c82d3e6246e0b99c7fc7a35882f712fc1142fa8f3fe9f5191bce64c2a7f"
EXPECTED_S1_7_ARTIFACT_HASH = "21b14bdec009bee827dea5d604b363c6ce46ce55c06334d0409a2dc4400292cb"
EXPECTED_G1_SINGLE_HASH = "0c8d91dc010533a5c99229fe0c8577e10278f41d0f3fd754d885749c511e7f37"
EXPECTED_MODEL_READY_SHA256 = "7d3404906f3dd00c0d0314863f706c5df01f1db1fc0e0b4cf501353b88963d1e"
HISTORICAL_G3_PRODUCER = "54b1c7f87eda0533b29622b39cc8a7ec90646d0b"
EXPECTED_HISTORICAL_G3_PATCH_SHA256 = "308db1c1e38135e5a65d37fa92566ac9cd5136220b4ffbb143a7e2f323d1ee0b"
HISTORICAL_G3_CRITICAL_SOURCE_REFS = (
    "src/param_importance_nlp/assets.py",
    "src/param_importance_nlp/contracts/__init__.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
)
HISTORICAL_G3_REPRODUCTION_ROLES = {
    "historical_producer_attestation": "historical-producer-attestation.json",
    "historical_g3_replay": "historical-g3-replay.json",
}
EXPECTED_S1_7_HISTORICAL_PRODUCER_ATTESTATION_SHA256 = "c28bcf52bd268ce34fe56e509686c6f374bd80a0f1f6d584c6387123479e230a"
EXPECTED_S1_7_HISTORICAL_G3_REPLAY_SHA256 = "69e74a2adea8cbc4539e85f09cd25f453780fb9f471906b96be3805194c1278b"
EXPECTED_G3_RESOLUTION_ARTIFACT_HASH = "418e9a60c25edfc275fe459b333bf4893912d03b9331b08dc9afb3e1560ea5cd"
EXPECTED_G3_RESOLUTION_PAYLOAD_HASH = "a3bc369bcb6f7dd2ba7dbd83a59d518d64d4431e355150c92d8a0cda02cb2a92"
EXPECTED_MODEL_IDENTITY = {
    "logical_name": "pythia-14m-step0",
    "asset_id": "11dd681a22649a451b9be53c255bb4e9f83207c3f22f75f1eec53a33b7776fd2",
    "revision": "56079904bb80b7f36d3b794089f146e7a4d6efae",
    "ready_manifest_sha256": EXPECTED_MODEL_READY_SHA256,
    "parameter_count": 14067712,
    "config_vocab_size": 50_304,
}
EXPECTED_TOKENIZER_IDENTITY = {
    "logical_name": "pythia-tokenizer",
    "asset_id": "b5eebc43fe88687e5bf692761f1db25f91e8d6f9a8cceaa2342d2624ac1f652d",
    "revision": "e361f9afd54b3e7856879eead5326d36ff6f32d7",
    "ready_manifest_sha256": "ea59f3f8e37321208701326b2ea88b7491450a88eae870775beeff027d102794",
    "vocab_size": 50_277,
}
EXPECTED_PILE_IDENTITY = {
    "logical_name": "pile-selected-prefix",
    "asset_id": "dbbfeb12bab4027b386bd97d604d8134699e96f79e309cceacff7999a55b5dad",
    "revision": "4647773ea142ab1ff5694602fa104bbf49088408",
    "ready_manifest_sha256": "345cd0f49d35ad9543daa3f95118013c55bdd729ed87fdec3c7a7c93ae449f8b",
}
EXPECTED_HISTORICAL_PILE_HASHED_BYTES = 31_757_184_042
S1_7_INDEX_HANDOFF_KEYS = frozenset({
    "index_ref", "index_sha256", "index_artifact_hash", "producer_commit",
    "gate_artifact_hash", "fixture_hash", "token_file_sha256", "model_resolution_ref",
    "model_provenance", "pile_provenance", "token_sha256", "role_refs", "role_sha256",
    "historical_producer_attestation_ref", "historical_producer_attestation_sha256",
    "historical_g3_replay_ref", "historical_g3_replay_sha256",
    "historical_g3_attestation_artifact_hash", "historical_g3_historical_producer_commit",
    "historical_g3_critical_patch_sha256", "historical_g3_historical_source_sha256",
    "historical_g3_replay_hash", "historical_g3_current_consumer_commit",
    "historical_g3_current_consumer_source_sha256",
})
EXPECTED_PILE_LOGICAL_ASSET_ID = "pile-selected-prefix"
EXPECTED_PILE_READY_SHA256 = "345cd0f49d35ad9543daa3f95118013c55bdd729ed87fdec3c7a7c93ae449f8b"
EXPECTED_PILE_STORAGE_KIND = "pythia_mmap_shards"
EXPECTED_PILE_PROVENANCE = {
    "acquisition_ref": "manifests/evidence/g3/acquisition/9a51688c24143b98f56151551265efc8a9a5ad9767517de65cf915ef7b667b5a.json",
    "acquisition_sha256": "9a51688c24143b98f56151551265efc8a9a5ad9767517de65cf915ef7b667b5a",
    "asset_id": "dbbfeb12bab4027b386bd97d604d8134699e96f79e309cceacff7999a55b5dad",
    "asset_root_ref": "datasets/pile-deduped-pythia-preshuffled",
    "directory_content_sha256": None,
    "g3_resolution_artifact_hash": "a3bc369bcb6f7dd2ba7dbd83a59d518d64d4431e355150c92d8a0cda02cb2a92",
    "g3_resolution_ref": "evidence/stage0/tasks/04-a3bc369bcb6f7dd2ba7dbd83a59d518d64d4431e355150c92d8a0cda02cb2a92/commits/asset_resolution.json",
    "logical_asset_id": EXPECTED_PILE_LOGICAL_ASSET_ID,
    "manifest_ref": "manifests/data/pile-pretokenized.json",
    "qualification_artifact_hash": "b86c8b6eec6f915e62d568cfbc3cb4493df3ec0397a1adbfe2947941b7ec686a",
    "qualification_ref": "manifests/qualifications/pile-selected-prefix.json",
    "ready_manifest_sha256": EXPECTED_PILE_READY_SHA256,
    "schema_version": "qualified-g3-runtime-provenance-v1",
    "source_git_commit": "54b1c7f87eda0533b29622b39cc8a7ec90646d0b",
    "storage_kind": EXPECTED_PILE_STORAGE_KIND,
    "verification_ref": "manifests/evidence/g3/verification/9a51688c24143b98f56151551265efc8a9a5ad9767517de65cf915ef7b667b5a.json",
    "verification_sha256": "2bc057850d5e7daf5da03c5c1e60c18c1281e7117fd13b677b964db95e20e605",
}
EXPECTED_GPU_CAPABILITY_REF = "evidence/stage0/bootstrap/a15f0e2970b7cae6951dd606ebd396a8df68255c/commits/capability_cuda.json"
EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH = "a536e191cd59318325289d238db727f8939767e384bfccd961ae7ca1c6a11ce4"
EXPECTED_GPU_CAPABILITY_FILE_SHA256 = "1d5f28369f4119c1e46072a687d217e2b2ad2de0bd02269acd42f14083c14b1f"
ROUTE_WORLD = {"A": 1, "B": 1, "C": 2, "D": 4}
PERMUTATIONS = ("rank_swap", "local_reverse")
_DIGEST = set("0123456789abcdef")
GPU_UUID_RE = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_USED_LOOPBACK_RENDEZVOUS_PORTS: set[int] = set()
# A /proc entry can disappear immediately before Popen observes the launcher's
# exit.  One bounded wait distinguishes that benign reaping race from a live
# process whose token/identity can no longer be audited.
LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS = 1.0
# This is the already-established post-poll terminal join bound.  It is not a
# broader token-loss grace: only a separately re-attested Linux owner-exit
# candidate may use it, revalidating at the one-second confirmation cadence.
TERMINAL_PROCESS_JOIN_TIMEOUT_SECONDS = 30.0
# GPU contexts can outlive a successfully reaped worker briefly.  The formal
# contract is therefore not a one-shot observation: it needs three complete,
# consecutive, exact-idle inventories before a release or reacquire boundary.
# The 60-second operational bound is versioned in gpu-quiescence-v2.  It does
# not relax any idle predicate: a frozen-server CPU-only management-query
# measurement observed a 12.280219650001527-second two-query sample, so three
# samples plus the two fixed one-second cadences require 38.84065895000458s.
# The fixed 21.15934104999542-second margin is a static operating budget, not
# dynamically fitted from a formal attempt.
GPU_QUIESCENCE_TIMEOUT_SECONDS = 60.0
GPU_QUIESCENCE_SAMPLE_INTERVAL_SECONDS = 1.0
GPU_QUIESCENCE_REQUIRED_CONSECUTIVE_EXACT_IDLE_SAMPLES = 3
GPU_QUIESCENCE_SCHEMA_VERSION = "stage1-s1-8-gpu-quiescence-v2"
GPU_QUIESCENCE_OPERATIONAL_TIMEOUT_BASIS = {
    "measurement_method": "frozen_linux_cpu_only_nvidia_smi_management_queries",
    "combined_inventory_recovery_seconds": 6.166325362000862,
    "compute_apps_seconds": 6.113894288000665,
    "two_query_sample_seconds": 12.280219650001527,
    "three_samples_plus_two_cadences_seconds": 38.84065895000458,
    "fixed_timeout_seconds": 60.0,
    "fixed_margin_seconds": 21.15934104999542,
    "dynamic_fitting": False,
}
GPU_QUIESCENCE_ROLES = {
    "prelease": "prelease-gpu-quiescence.json",
    "post_worker": "post-worker-gpu-quiescence.json",
    "post_release": "post-release-gpu-quiescence.json",
    "reacquire_preflight": "reacquire-preflight-gpu-quiescence.json",
}
# Elastic workers can become zombies while their recently-used session remains
# enumerable.  Recheck that narrow /proc transition exactly once before
# treating a session member as foreign process ownership.
SESSION_MEMBER_REVALIDATION_SECONDS = 0.05
PILE_DOWNLOADER_CMDLINE_SIGNATURES = (
    b"server_xet_download.sh",
    b"pile-full-download",
)
GATE_CHECK_IDS = (
    "s1_7_handoff", "consumer_diff", "approved_four_uuid_topology", "nccl_smoke", "pre_route_scale_oracle",
    "real_routes_equal_and_weighted", "rank_partition_and_no_sync", "manual_collective_contract",
    "independent_fp64_replay", "optimizer_and_accumulator", "rank_and_order_permutations",
    "ordinary_sync_negative", "controlled_rank_failure", "array_manifest", "resource_summary", "lease_release_reacquire",
    "charts_no_a_u_reference", "schemas_and_atomic_publication",
)
IMPLEMENTATION_SOURCE_FILES = (
    "ops/stage1/formalize_s1_8.py", "ops/stage1/run_s1_8_worker.py", "ops/stage1/run_s1_8_scale_oracle.py", "ops/stage1/run_s1_8_nccl_smoke.py",
    "ops/stage1/formalize_s1_6.py",
    "src/param_importance_nlp/stage1_ddp.py", "src/param_importance_nlp/stage1_ddp_oracle.py", "src/param_importance_nlp/stage1_ddp_scale_oracle.py",
    "src/param_importance_nlp/atomic.py", "src/param_importance_nlp/contracts/errors.py", "src/param_importance_nlp/contracts/jsonio.py", "src/param_importance_nlp/contracts/runtime_evidence.py",
    "src/param_importance_nlp/core/accumulator.py", "src/param_importance_nlp/core/errors.py", "src/param_importance_nlp/core/tensors.py",
    "src/param_importance_nlp/g3_runtime_assets.py", "src/param_importance_nlp/runtime/operations.py", "src/param_importance_nlp/runtime/task_artifacts.py",
    "configs/stage0/g3-asset-requirements-v1.json", "configs/stage0/g3-asset-layout-v1.json",
    "schemas/stage1/s1-8-array-bundle-v1.json", "schemas/stage1/s1-8-comparison-table-v1.json", "schemas/stage1/s1-8-ddp-report-v1.json", "schemas/stage1/s1-8-ddp-report-v2.json", "schemas/stage1/s1-8-ddp-report-v3.json", "schemas/stage1/s1-8-ddp-report-v4.json", "schemas/stage1/s1-8-fixture-manifest-v1.json", "schemas/stage1/s1-8-fixture-manifest-v2.json", "schemas/stage1/s1-8-fixture-manifest-v3.json", "schemas/stage1/s1-8-formalization-index-v1.json", "schemas/stage1/s1-8-formalization-index-v2.json", "schemas/stage1/s1-8-formalization-index-v3.json", "schemas/stage1/s1-8-formalization-index-v4.json", "schemas/stage1/s1-8-gate-record-v1.json", "schemas/stage1/s1-8-gpu-quiescence-v1.json", "schemas/stage1/s1-8-gpu-quiescence-v2.json", "schemas/stage1/s1-8-replay-validation-v1.json", "schemas/stage1/s1-8-replay-validation-v2.json", "schemas/stage1/s1-8-safetensors-manifest-v1.json", "schemas/stage1/s1-8-validation-v1.json", "schemas/stage1/s1-8-validation-v2.json", "schemas/stage1/s1-8-validation-v3.json", "schemas/stage1/s1-8-validation-v4.json", "schemas/stage1/s1-8-worker-report-v1.json",
)


def _fixed_reproduction_roles() -> dict[str, tuple[str, str]]:
    """Return the complete, deterministic publication closure for a PASS run.

    The keys, flattened names, and source-relative paths are part of the v4
    wire contract.  No glob-discovered file is allowed to add a role merely
    because it happened to exist in an attempt directory.
    """

    roles: dict[str, tuple[str, str]] = {
        "fixture_inputs": ("fixture-inputs.safetensors", "fixture-inputs.safetensors"),
        "s1_7_historical_producer_attestation": ("s1-7-historical-producer-attestation.json", "s1-7-historical-producer-attestation.json"),
        "s1_7_historical_g3_replay": ("s1-7-historical-g3-replay.json", "s1-7-historical-g3-replay.json"),
        "model_qualified_resolution": ("model-qualified-resolution.json", "model-qualified-resolution.json"),
        "offline_policy": ("offline-policy.json", "offline-policy.json"),
        "nccl_transport_protocol": ("nccl-transport-protocol.json", "nccl-transport-protocol.json"),
        "pre_route_scale_plan": ("pre-route-scale-plan.json", "pre-route-scale-plan.json"),
        "pre_route_scale": ("pre-route-scale.json", "pre-route-scale.json"),
        "preflight": ("preflight.json", "preflight.json"),
        "prelease_gpu_quiescence": ("prelease-gpu-quiescence.json", "prelease-gpu-quiescence.json"),
        "post_lease_gpu": ("post-lease-gpu.json", "post-lease-gpu.json"),
        "post_worker_gpu": ("post-worker-gpu.json", "post-worker-gpu.json"),
        "post_release_gpu": ("post-release-gpu.json", "post-release-gpu.json"),
        "post_worker_gpu_quiescence": ("post-worker-gpu-quiescence.json", "post-worker-gpu-quiescence.json"),
        "post_release_gpu_quiescence": ("post-release-gpu-quiescence.json", "post-release-gpu-quiescence.json"),
        "reacquire_preflight_gpu_quiescence": ("reacquire-preflight-gpu-quiescence.json", "reacquire-preflight-gpu-quiescence.json"),
        "resource_summary": ("resource-summary.json", "resource-summary.json"),
        "nccl_smoke_report": ("nccl-smoke-report.json", "nccl-smoke-report.json"),
        "nccl_smoke_process": ("nccl-smoke.process.json", "nccl-smoke.process.json"),
        "nccl_smoke_stdout": ("nccl-smoke.stdout.txt", "nccl-smoke.stdout.txt"),
        "nccl_smoke_stderr": ("nccl-smoke.stderr.txt", "nccl-smoke.stderr.txt"),
        "lease_history_first": ("lease-history-first.json", "lease-history-first.json"),
        "lease_history_reacquire": ("lease-history-reacquire.json", "lease-history-reacquire.json"),
        "attempt_start": ("attempt-start.json", "attempt-start.json"),
    }
    labels = (
        "nccl-smoke", "pre-route-scale", "A-identity-formal", "B-identity-formal",
        "C-identity-formal", "D-identity-formal", "D-rank_swap-formal",
        "D-local_reverse-formal", "D-identity-ordinary_sync_negative",
        "D-identity-inject_rank_failure",
    )
    for label in labels:
        for source_name in (f"{label}.process.json", f"{label}.stdout.txt", f"{label}.stderr.txt", f"{label}-process-tree-initial.json"):
            role = "run_root_" + hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16]
            roles[role] = ("run__root__" + source_name, source_name)
    route_runs = (
        ("A", "identity", "formal"), ("B", "identity", "formal"),
        ("C", "identity", "formal"), ("D", "identity", "formal"),
        ("D", "rank_swap", "formal"), ("D", "local_reverse", "formal"),
        ("D", "identity", "ordinary_sync_negative"), ("D", "identity", "inject_rank_failure"),
    )
    for route, permutation, mode in route_runs:
        route_directory = f"route-{route}-{permutation}-{mode}"
        source_relative = route_directory + "/worker-plan.json"
        role = "run_" + hashlib.sha256(source_relative.encode("utf-8")).hexdigest()[:16]
        roles[role] = ("run__" + source_relative.replace("/", "__"), source_relative)
    for route, permutation in (("A", "identity"), ("B", "identity"), ("C", "identity"), ("D", "identity"), ("D", "rank_swap"), ("D", "local_reverse")):
        route_directory = f"route-{route}-{permutation}-formal"
        for source_relative in (
            route_directory + "/route-output/route-report.json",
            route_directory + f"/route-output/route-{route}.safetensors",
        ):
            role = "run_" + hashlib.sha256(source_relative.encode("utf-8")).hexdigest()[:16]
            roles[role] = ("run__" + source_relative.replace("/", "__"), source_relative)
    if len(roles) != 84 or len({reference for reference, _ in roles.values()}) != len(roles):
        raise Stage1S18FormalError("S18_REPRODUCTION_ROLE_CLOSURE_INVALID")
    return roles


class Stage1S18FormalError(RuntimeError):
    """A formal prerequisite, worker observation, or publication check failed."""


class Stage1S18ManualInterventionRequired(Stage1S18FormalError):
    """A process fingerprint drift makes a signal/lease release unsafe."""


class _SessionMemberStatUnavailable(RuntimeError):
    """A session PID vanished before its stat record could be read."""


class _LauncherNaturalExitCandidate(RuntimeError):
    """Only the recorded launcher may be awaiting Popen exit confirmation."""


class _LauncherOwnerExitCandidate(_LauncherNaturalExitCandidate):
    """An exact, attested Linux procfs owner-exit transition needs terminal join."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_source_map(repository: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for reference in IMPLEMENTATION_SOURCE_FILES:
        file = repository / reference
        if not file.is_file():
            raise Stage1S18FormalError("S18_IMPLEMENTATION_SOURCE_MISSING:" + reference)
        result[reference] = _sha(file)
    return result


def _validate_implementation_source_map(repository: Path, source_map: Mapping[str, Any]) -> None:
    """Require candidate source hashes to bind to the current source bytes."""

    observed = _mapping(source_map, field="candidate.source_map")
    expected = _implementation_source_map(repository)
    if set(observed) != set(IMPLEMENTATION_SOURCE_FILES) or any(
        _require_sha256(value, field="source_map." + key) != value
        for key, value in observed.items()
    ):
        raise Stage1S18FormalError("S18_CANDIDATE_SOURCE_MAP_INVALID")
    if observed != expected:
        raise Stage1S18FormalError("S18_CANDIDATE_SOURCE_MAP_BYTE_DRIFT")


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage1S18FormalError(f"S18_FORMAL_OBJECT_INVALID:{field}")
    return dict(value)


def _exact_contract_value(value: object, expected: object) -> bool:
    """Compare frozen JSON values without accepting ``False == 0``."""

    if isinstance(expected, Mapping):
        return isinstance(value, Mapping) and set(value) == set(expected) and all(
            _exact_contract_value(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) == len(expected) and all(
            _exact_contract_value(item, reference) for item, reference in zip(value, expected, strict=True)
        )
    return type(value) is type(expected) and value == expected


def _require_pre_route_scale_conditioning(report: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Bind the scale report's actual AdamW configuration before fixture use."""

    if not _exact_contract_value(report.get("optimizer_conditioning"), expected):
        raise Stage1S18FormalError("S18_PRE_ROUTE_SCALE_OPTIMIZER_CONDITIONING_INVALID")


def _canonical(value: object) -> str:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash
    return canonical_json_hash(value)


def _with_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    if "artifact_hash" in body:
        raise Stage1S18FormalError("S18_FORMAL_ARTIFACT_HASH_DERIVED")
    body["artifact_hash"] = _canonical(body)
    return body


def _self_hash(value: Mapping[str, Any]) -> bool:
    body = dict(value); declared = body.pop("artifact_hash", None)
    return isinstance(declared, str) and declared == _canonical(body)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    from param_importance_nlp.contracts.jsonio import write_canonical_json
    if path.exists():
        raise Stage1S18FormalError(f"S18_FORMAL_IMMUTABLE_COLLISION:{path}")
    write_canonical_json(path, dict(value))


def _validate_output_schemas(repository: Path, objects: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate every published role with the project's strict schema subset."""

    from param_importance_nlp.contracts.jsonio import loads_strict_json
    validator_path = repository / "ops" / "stage1" / "formalize_s1_6.py"
    spec = importlib.util.spec_from_file_location("_s18_schema_subset", validator_path)
    if spec is None or spec.loader is None:
        raise Stage1S18FormalError("S18_SCHEMA_VALIDATOR_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    registry: dict[str, Mapping[str, Any]] = {}
    for path in sorted((repository / "schemas" / "stage1").glob("s1-8-*.json")):
        value = loads_strict_json(path.read_bytes())
        if not isinstance(value, Mapping) or not isinstance(value.get("$id"), str):
            raise Stage1S18FormalError("S18_SCHEMA_REGISTRY_INVALID:" + path.name)
        registry[path.name] = value; registry[str(value["$id"])] = value
    filenames = {
        "fixture_manifest": "s1-8-fixture-manifest-v3.json", "ddp_report": "s1-8-ddp-report-v4.json",
        "array_bundle": "s1-8-array-bundle-v1.json", "comparison_table": "s1-8-comparison-table-v1.json",
        "gate_record": "s1-8-gate-record-v1.json", "replay": "s1-8-replay-validation-v2.json",
        "validation": "s1-8-validation-v4.json", "index": "s1-8-formalization-index-v4.json",
        "worker_report": "s1-8-worker-report-v1.json", "safetensors_manifest": "s1-8-safetensors-manifest-v1.json",
        "gpu_quiescence": "s1-8-gpu-quiescence-v2.json",
    }
    for role, value in objects.items():
        schema = registry.get(filenames.get(role, ""))
        if schema is None:
            raise Stage1S18FormalError("S18_SCHEMA_ROLE_UNKNOWN:" + role)
        try:
            module._validate_schema(value, schema, registry, document=schema, path=role)
        except Exception as error:
            raise Stage1S18FormalError("S18_SCHEMA_VALIDATION_FAILED:" + role) from error


def _schema_prepublication_check(repository: Path, objects: Mapping[str, Mapping[str, Any]]) -> bool:
    """Validate real role instances and reject schemas with naked objects."""

    from param_importance_nlp.contracts.jsonio import loads_strict_json
    expected = {
        "s1-8-array-bundle-v1.json", "s1-8-comparison-table-v1.json", "s1-8-ddp-report-v1.json", "s1-8-ddp-report-v2.json", "s1-8-ddp-report-v3.json", "s1-8-ddp-report-v4.json",
        "s1-8-fixture-manifest-v1.json", "s1-8-fixture-manifest-v2.json", "s1-8-fixture-manifest-v3.json", "s1-8-formalization-index-v1.json", "s1-8-formalization-index-v2.json", "s1-8-formalization-index-v3.json", "s1-8-formalization-index-v4.json", "s1-8-gate-record-v1.json", "s1-8-gpu-quiescence-v1.json", "s1-8-gpu-quiescence-v2.json",
        "s1-8-replay-validation-v1.json", "s1-8-replay-validation-v2.json", "s1-8-safetensors-manifest-v1.json", "s1-8-validation-v1.json", "s1-8-validation-v2.json", "s1-8-validation-v3.json", "s1-8-validation-v4.json", "s1-8-worker-report-v1.json",
    }
    paths = {path.name: path for path in (repository / "schemas" / "stage1").glob("s1-8-*.json")}
    if set(paths) != expected:
        raise Stage1S18FormalError("S18_SCHEMA_SET_INVALID")
    def strict(node: object, *, location: str) -> None:
        if isinstance(node, Mapping):
            additional = node.get("additionalProperties")
            if node.get("type") == "object" and "properties" not in node and (not isinstance(additional, Mapping) or not additional):
                raise Stage1S18FormalError("S18_SCHEMA_BARE_OBJECT:" + location)
            for key, value in node.items(): strict(value, location=location + "." + str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node): strict(value, location=location + f"[{index}]")
    for name, path in paths.items(): strict(loads_strict_json(path.read_bytes()), location=name)
    _validate_output_schemas(repository, objects)
    return True


def _require_baseline_replay(*, replay_fn: Any, route_arrays: Mapping[str, Mapping[str, Any]], fixture: Mapping[str, Any], route_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Fail before any permutation when the canonical A/B/C/D baseline drifts."""

    result = replay_fn(route_arrays=route_arrays, fixture=fixture, route_reports=route_reports)
    if result.get("status") != "PASS":
        raise Stage1S18FormalError("S18_BASELINE_REPLAY_FAILED")
    return result


def _canonical_file_sha(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the exact canonical file bytes for ``value``."""

    from param_importance_nlp.contracts.jsonio import canonical_json_bytes
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _DIGEST:
        raise Stage1S18FormalError("S18_CANDIDATE_SHA256_INVALID:" + field)
    return value


def _nccl_transport_protocol() -> dict[str, object]:
    """Bind the Stage 0 G6/G10-qualified P2P-disabled NCCL contract."""

    from param_importance_nlp.stage1_ddp import NCCL_TRANSPORT_PROTOCOL
    protocol = dict(NCCL_TRANSPORT_PROTOCOL)
    if protocol != {
        "schema_version": "stage1-s1-8-nccl-transport-v1",
        "qualification_basis_gate_ids": ["stage0.G6", "stage0.G10"],
        "current_cuda_capability_artifact_hash": EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH,
        "nccl_p2p_disable": "1",
        "process_group_timeout_seconds": 180,
    }:
        raise Stage1S18FormalError("S18_NCCL_TRANSPORT_PROTOCOL_INVALID")
    return protocol


def _validate_worker_candidate_contract(route_key: str, report: Mapping[str, Any]) -> None:
    """Apply route-specific semantics that JSON Schema cannot express alone."""

    route, cases = report.get("route"), report.get("cases")
    if not isinstance(route, str) or route not in ROUTE_WORLD or not isinstance(cases, list) or [row.get("case") if isinstance(row, Mapping) else None for row in cases] != ["equal", "weighted"]:
        raise Stage1S18FormalError("S18_CANDIDATE_WORKER_CASE_CARDINALITY_INVALID:" + route_key)
    if (route_key in ROUTE_WORLD and route_key != route) or (route_key.startswith("D-") and route != "D") or report.get("world_size") != ROUTE_WORLD[route]:
        raise Stage1S18FormalError("S18_CANDIDATE_WORKER_ROUTE_WORLD_INVALID:" + route_key)
    if report.get("nccl_transport_protocol") != _nccl_transport_protocol():
        raise Stage1S18FormalError("S18_CANDIDATE_WORKER_NCCL_TRANSPORT_INVALID:" + route_key)
    visible, rank_uuids = report.get("visible_gpu_uuids"), report.get("rank_to_gpu_uuid")
    if not isinstance(visible, list) or not isinstance(rank_uuids, list) or visible != rank_uuids or len(visible) != ROUTE_WORLD[route] or len(set(visible)) != len(visible) or any(not isinstance(value, str) or GPU_UUID_RE.fullmatch(value) is None for value in visible):
        raise Stage1S18FormalError("S18_CANDIDATE_WORKER_UUID_INVALID:" + route_key)
    layout = _mapping(report.get("route_layout"), field="candidate.worker.route_layout")
    partitions = layout.get("rank_microbatch_ids")
    if layout.get("route") != route or layout.get("world_size") != ROUTE_WORLD[route] or not isinstance(partitions, list) or len(partitions) != ROUTE_WORLD[route] or sorted(item for part in partitions if isinstance(part, list) for item in part) != list(range(8)):
        raise Stage1S18FormalError("S18_CANDIDATE_WORKER_LAYOUT_INVALID:" + route_key)
    ordinary_collectives = [row.get("ordinary_ddp_gradient_collectives") if isinstance(row, Mapping) else None for row in cases]
    expected_ordinary_collectives = [1, 2] if route == "A" else [0, 0]
    if any(type(value) is not int for value in ordinary_collectives) or ordinary_collectives != expected_ordinary_collectives:
        raise Stage1S18FormalError("S18_CANDIDATE_WORKER_ORDINARY_DDP_COLLECTIVE_CONTRACT_INVALID:" + route_key)
    for ordinal, row_raw in enumerate(cases, start=1):
        row = _mapping(row_raw, field="candidate.worker.case")
        keys, accumulator = row.get("array_keys"), row.get("accumulator")
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise Stage1S18FormalError("S18_CANDIDATE_WORKER_ARRAY_KEY_INVALID:" + route_key)
        rank_records = row.get("rank_records")
        if not isinstance(rank_records, list) or len(rank_records) != ROUTE_WORLD[route] or sorted(_mapping(record, field="candidate.worker.rank").get("rank") for record in rank_records) != list(range(ROUTE_WORLD[route])):
            raise Stage1S18FormalError("S18_CANDIDATE_WORKER_RANK_CARDINALITY_INVALID:" + route_key)
        for rank, (record_raw, expected_ids) in enumerate(zip(rank_records, partitions, strict=True)):
            record = _mapping(record_raw, field="candidate.worker.rank")
            if record.get("rank") != rank or record.get("local_microbatch_ids") != expected_ids:
                raise Stage1S18FormalError("S18_CANDIDATE_WORKER_RANK_PARTITION_INVALID:" + route_key)
            gradients = record.get("local_gradient_checksums")
            expected_gradient_count = 1 if route == "A" else len(expected_ids)
            if (
                not isinstance(gradients, list)
                or len(gradients) != expected_gradient_count
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in gradients
                )
            ):
                raise Stage1S18FormalError("S18_CANDIDATE_WORKER_LOCAL_GRADIENT_CHECKSUM_INVALID:" + route_key)
        has_u, has_accumulator = any("/u_" in key for key in keys), any(key.startswith("accumulator/") for key in keys)
        if route == "A":
            if accumulator is not None or has_u or has_accumulator or any(key.startswith(("scores/", "stats/")) for key in keys) or not all(any(key.startswith(f"a-reference/{row['case']}/{field}/") for key in keys) for field in ("mean_gradient", "raw_core", "raw_score", "raw_score_clipped", "data_update", "data_movement", "total_update", "total_movement", "weight_decay_update", "weight_decay_movement", "actual_update_raw_importance", "magnitude")):
                raise Stage1S18FormalError("S18_CANDIDATE_A_U_OR_ACCUMULATOR_FORBIDDEN")
            continue
        summary = _mapping(accumulator, field="candidate.worker.accumulator")
        expected_summary = {"successful_steps", "skipped_steps", "signed_identity", "absolute_identity", "contribution_checksums", "cumulative_checksums"}
        if set(summary) != expected_summary or summary.get("successful_steps") != ordinal or summary.get("skipped_steps") != 0 or summary.get("signed_identity") is not True or summary.get("absolute_identity") is not True or not has_u or not has_accumulator:
            raise Stage1S18FormalError("S18_CANDIDATE_U_ACCUMULATOR_REQUIRED:" + route_key)
        contribution = _mapping(summary.get("contribution_checksums"), field="candidate.worker.contribution_checksums")
        cumulative = _mapping(summary.get("cumulative_checksums"), field="candidate.worker.cumulative_checksums")
        if set(contribution) != {"signed", "raw", "raw_clipped", "data_update", "total_update", "weight_decay_update", "actual_update_raw_importance"} or set(cumulative) != {"signed", "positive", "negative_mass", "absolute", "raw", "raw_clipped", "data_movement", "net_data_movement", "total_movement", "total_endpoint_movement", "weight_decay_movement", "net_weight_decay_movement", "actual_update_raw_importance", "magnitude"} or any(_require_sha256(value, field="accumulator_checksum") != value for value in (*contribution.values(), *cumulative.values())):
            raise Stage1S18FormalError("S18_CANDIDATE_ACCUMULATOR_CHECKSUM_SET_INVALID:" + route_key)
        statistic_keys = {"s1", "s2"} if row.get("case") == "equal" else {"g1", "g2"}
        if set(_mapping(row.get("global_statistic_checksums"), field="candidate.worker.statistics")) != statistic_keys or not all(any(key.startswith(f"stats/{row['case']}/{field}/") for key in keys) for field in statistic_keys) or not all(any(key.startswith(f"scores/{row['case']}/{field}/") for key in keys) for field in ("mean_gradient", "raw_core", "u_core", "raw_score", "u_score", "u_score_clipped")):
            raise Stage1S18FormalError("S18_CANDIDATE_WORKER_STATISTICS_OR_ARRAYS_INVALID:" + route_key)


def _validate_recorded_launch_tree(*, fingerprint: Mapping[str, Any], tree: Mapping[str, Any], expected_member_count: int | None) -> None:
    """Validate serialized ownership evidence without trusting process output.

    The formal report preserves first-captured PPIDs as historical ancestry,
    because elastic's ``start_new_session=True`` makes worker sessions distinct
    from torchrun and later re-parenting is legitimate.  This check deliberately
    does not require a common SID/PGID; it proves the stored tree instead.
    """

    required_tree = {"pgid", "sid", "members", "member_pids", "ancestry_depths"}
    required_fingerprint = {"pid", "ppid", "uid", "pgid", "sid", "start_ticks", "exe", "cmdline_sha256", "environment_run_token"}
    if set(tree) != required_tree or set(fingerprint) != required_fingerprint:
        raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_KEYS_INVALID")
    members, pids, depths = tree.get("members"), tree.get("member_pids"), tree.get("ancestry_depths")
    launcher_pid, launcher_uid, launcher_start = fingerprint.get("pid"), fingerprint.get("uid"), fingerprint.get("start_ticks")
    if not isinstance(members, list) or not isinstance(pids, list) or not isinstance(depths, Mapping) or not isinstance(launcher_pid, int) or not isinstance(launcher_uid, int) or not isinstance(launcher_start, str) or not launcher_start.isdigit():
        raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_SHAPE_INVALID")
    by_pid = {member.get("pid"): member for member in members if isinstance(member, Mapping) and set(member) == required_fingerprint and isinstance(member.get("pid"), int)}
    if len(by_pid) != len(members) or pids != sorted(by_pid) or set(depths) != {str(pid) for pid in by_pid}:
        raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_CARDINALITY_INVALID")
    launcher = by_pid.get(launcher_pid)
    if launcher is None or dict(launcher) != dict(fingerprint) or tree.get("pgid") != fingerprint.get("pgid") or tree.get("sid") != fingerprint.get("sid"):
        raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_LAUNCHER_INVALID")
    if expected_member_count is not None and (len(by_pid) != expected_member_count or len({member.get("sid") for member in by_pid.values()}) != expected_member_count):
        raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_MEMBER_COUNT_INVALID")
    for pid, member in by_pid.items():
        if member.get("uid") != launcher_uid or member.get("environment_run_token") != fingerprint.get("environment_run_token") or not isinstance(member.get("start_ticks"), str) or not member["start_ticks"].isdigit() or int(member["start_ticks"]) < int(launcher_start) or not isinstance(depths.get(str(pid)), int) or int(depths[str(pid)]) < 0:
            raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_MEMBER_INVALID")
        current, depth, seen = pid, 0, set()
        while current != launcher_pid:
            if current in seen:
                raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_ANCESTRY_CYCLE")
            seen.add(current)
            parent = by_pid[current].get("ppid")
            if not isinstance(parent, int) or parent not in by_pid:
                raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_ANCESTRY_INVALID")
            current, depth = parent, depth + 1
        if depth != depths[str(pid)]:
            raise Stage1S18FormalError("S18_CANDIDATE_PROCESS_TREE_DEPTH_INVALID")


def _validate_process_outcome_contract(outcome: Mapping[str, Any]) -> None:
    """Directly validate every launcher outcome before and after publication."""

    required = {"schema_version", "label", "command", "rendezvous_id", "rendezvous_endpoint", "rendezvous_handoff", "returncode", "fingerprint", "initial_tree", "known_tree", "termination_audit_ref", "stdout_sha256", "stderr_sha256", "expected_success", "residual_launch_tree", "artifact_hash"}
    if set(outcome) != required or outcome.get("schema_version") != "stage1-s1-8-process-outcome-v1":
        raise Stage1S18FormalError("S18_PROCESS_OUTCOME_KEYS_INVALID")
    command, fingerprint = outcome.get("command"), _mapping(outcome.get("fingerprint"), field="process.fingerprint")
    endpoint, handoff = outcome.get("rendezvous_endpoint"), outcome.get("rendezvous_handoff")
    expected_success, returncode = outcome.get("expected_success"), outcome.get("returncode")
    if not _self_hash(outcome) or not isinstance(command, list) or any(not isinstance(value, str) for value in command) or not isinstance(expected_success, bool) or not isinstance(returncode, int) or (returncode == 0) != expected_success or outcome.get("rendezvous_id") != fingerprint.get("environment_run_token") or outcome.get("residual_launch_tree") != {"session_members": [], "token_members": []}:
        raise Stage1S18FormalError("S18_PROCESS_OUTCOME_INVALID")
    distributed = "torch.distributed.run" in command
    ids = [index for index, value in enumerate(command) if value == "--rdzv-id"]
    endpoints = [index for index, value in enumerate(command) if value == "--rdzv-endpoint"]
    # A non-torchrun pre-route scale oracle is one launcher process.  Its
    # initial snapshot remains permitted to be a transient subset, but its
    # complete known tree must prove no extra child was launched.
    expected_member_count: int | None = 1
    if distributed:
        nproc_positions = [index for index, value in enumerate(command) if value == "--nproc_per_node"]
        if len(nproc_positions) != 1 or nproc_positions[0] + 1 >= len(command):
            raise Stage1S18FormalError("S18_PROCESS_NPROC_OUTCOME_INVALID")
        try:
            nproc_per_node = int(command[nproc_positions[0] + 1])
        except ValueError as error:
            raise Stage1S18FormalError("S18_PROCESS_NPROC_OUTCOME_INVALID") from error
        if nproc_per_node not in {1, 2, 4}:
            raise Stage1S18FormalError("S18_PROCESS_NPROC_OUTCOME_INVALID")
        expected_member_count = nproc_per_node + 1
        if not _is_nonzero_loopback_endpoint(endpoint) or handoff != {"reservation_held_to_popen": True, "single_attempt": True, "silent_retry": False} or len(ids) != 1 or ids[0] + 1 >= len(command) or command[ids[0] + 1] != outcome.get("rendezvous_id") or len(endpoints) != 1 or endpoints[0] + 1 >= len(command) or command[endpoints[0] + 1] != endpoint:
            raise Stage1S18FormalError("S18_PROCESS_RENDEZVOUS_OUTCOME_INVALID")
    elif endpoint is not None or handoff != {"reservation_held_to_popen": False, "single_attempt": True, "silent_retry": False} or ids or endpoints:
        raise Stage1S18FormalError("S18_PROCESS_NON_DISTRIBUTED_OUTCOME_INVALID")
    _validate_recorded_launch_tree(fingerprint=fingerprint, tree=_mapping(outcome.get("initial_tree"), field="process.initial_tree"), expected_member_count=None)
    _validate_recorded_launch_tree(fingerprint=fingerprint, tree=_mapping(outcome.get("known_tree"), field="process.known_tree"), expected_member_count=expected_member_count)


def _validate_gpu_quiescence_publication(
    *,
    repository: Path,
    ddp: Mapping[str, Any],
    validation: Mapping[str, Any],
    index: Mapping[str, Any],
    source_files: Mapping[str, Path],
) -> None:
    """Bind all three sampler artifacts into every current public role."""

    ddp_roles = _mapping(ddp.get("gpu_quiescence"), field="candidate.ddp.gpu_quiescence")
    validation_roles = _mapping(validation.get("gpu_quiescence"), field="candidate.validation.gpu_quiescence")
    refs = _mapping(index.get("reproduction_role_refs"), field="candidate.index.reproduction_refs")
    hashes = _mapping(index.get("reproduction_role_sha256"), field="candidate.index.reproduction_hashes")
    if set(ddp_roles) != set(GPU_QUIESCENCE_ROLES) or set(validation_roles) != set(GPU_QUIESCENCE_ROLES):
        raise Stage1S18FormalError("S18_CANDIDATE_GPU_QUIESCENCE_ROLE_SET_INVALID")
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    for phase, filename in GPU_QUIESCENCE_ROLES.items():
        source = source_files.get(filename)
        index_role = phase + "_gpu_quiescence"
        ddp_binding = _mapping(ddp_roles.get(phase), field="candidate.ddp.gpu_quiescence." + phase)
        validation_binding = _mapping(validation_roles.get(phase), field="candidate.validation.gpu_quiescence." + phase)
        if (
            set(ddp_binding) != {"ref", "sha256"}
            or set(validation_binding) != {"ref", "sha256"}
            or ddp_binding.get("ref") != filename
            or validation_binding.get("ref") != filename
            or refs.get(index_role) != filename
            or source is None
            or not source.is_file()
        ):
            raise Stage1S18FormalError("S18_CANDIDATE_GPU_QUIESCENCE_REFERENCE_INVALID:" + phase)
        digest = _sha(source)
        if (
            _require_sha256(ddp_binding.get("sha256"), field="candidate.ddp.gpu_quiescence." + phase) != digest
            or _require_sha256(validation_binding.get("sha256"), field="candidate.validation.gpu_quiescence." + phase) != digest
            or _require_sha256(hashes.get(index_role), field="candidate.index.gpu_quiescence." + phase) != digest
        ):
            raise Stage1S18FormalError("S18_CANDIDATE_GPU_QUIESCENCE_HASH_DRIFT:" + phase)
        record = _mapping(load_canonical_json(source), field="candidate.gpu_quiescence." + phase)
        _validate_output_schemas(repository, {"gpu_quiescence": record})
        if not _self_hash(record) or record.get("status") != "PASS" or record.get("phase") != phase:
            raise Stage1S18FormalError("S18_CANDIDATE_GPU_QUIESCENCE_RECORD_INVALID:" + phase)
        snapshot_name = {
            "prelease": "preflight.json",
            "post_worker": "post-worker-gpu.json",
            "post_release": "post-release-gpu.json",
            "reacquire_preflight": None,
        }[phase]
        if snapshot_name is not None:
            snapshot_source = source_files.get(snapshot_name)
            if snapshot_source is None or not snapshot_source.is_file():
                raise Stage1S18FormalError("S18_CANDIDATE_GPU_QUIESCENCE_SNAPSHOT_MISSING:" + phase)
            snapshot = _mapping(load_canonical_json(snapshot_source), field="candidate.gpu_snapshot." + phase)
            if record.get("final_gpu") != snapshot.get("gpu"):
                raise Stage1S18FormalError("S18_CANDIDATE_GPU_QUIESCENCE_FINAL_SAMPLE_DRIFT:" + phase)


def _candidate_publication_check(
    *,
    repository: Path,
    objects: Mapping[str, Mapping[str, Any]],
    worker_reports: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    source_files: Mapping[str, Path],
) -> bool:
    """Strictly validate the complete candidate before staging can succeed.

    This deliberately checks the in-memory objects *and* their prospective
    canonical file identities.  It is called before a PASS gate is emitted and
    again after the byte-identical staging copy.  A later role cannot turn a
    successful gate into a schema surprise.
    """

    expected_roles = {
        "fixture_manifest", "ddp_report", "array_bundle", "comparison_table",
        "gate_record", "replay", "validation", "index",
    }
    if set(objects) != expected_roles:
        raise Stage1S18FormalError("S18_CANDIDATE_ROLE_SET_INVALID")
    _schema_prepublication_check(repository, objects)
    if set(worker_reports) != {"A", "B", "C", "D", "D-rank_swap", "D-local_reverse"} or set(manifests) != set(worker_reports):
        raise Stage1S18FormalError("S18_CANDIDATE_WORKER_ROUTE_SET_INVALID")
    for route_key in sorted(worker_reports):
        report, manifest = worker_reports[route_key], manifests[route_key]
        _validate_output_schemas(repository, {"worker_report": report, "safetensors_manifest": manifest})
        if not _self_hash(report) or not _self_hash(manifest):
            raise Stage1S18FormalError("S18_CANDIDATE_WORKER_SELF_HASH_INVALID:" + route_key)
        if _mapping(report.get("arrays"), field="candidate.worker.arrays") != manifest:
            raise Stage1S18FormalError("S18_CANDIDATE_WORKER_MANIFEST_CROSSREF_INVALID:" + route_key)
        _validate_worker_candidate_contract(route_key, report)
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    for name, path in sorted(source_files.items()):
        if name.endswith(".process.json"):
            _validate_process_outcome_contract(_mapping(load_canonical_json(path), field="candidate.process." + name))

    fixture, ddp, bundle, table, gate, replay_record, validation, index = (
        objects["fixture_manifest"], objects["ddp_report"], objects["array_bundle"],
        objects["comparison_table"], objects["gate_record"], objects["replay"],
        objects["validation"], objects["index"],
    )
    if any(not _self_hash(value) for value in objects.values()):
        raise Stage1S18FormalError("S18_CANDIDATE_SELF_HASH_INVALID")
    role_names = {
        "fixture_manifest": "fixture-manifest.json", "ddp_report": "ddp-report.json",
        "array_bundle": "array-bundle.json", "comparison_table": "comparison-table.json",
        "gate_record": "g1-ddp-record.json",
    }
    role_refs, role_sha = _mapping(index.get("role_refs"), field="candidate.index.role_refs"), _mapping(index.get("role_sha256"), field="candidate.index.role_sha256")
    if role_refs != role_names or set(role_sha) != set(role_names):
        raise Stage1S18FormalError("S18_CANDIDATE_ROLE_REFERENCE_SET_INVALID")
    for role, name in role_names.items():
        digest = _canonical_file_sha(objects[role])
        if _require_sha256(role_sha.get(role), field="role_sha." + role) != digest:
            raise Stage1S18FormalError("S18_CANDIDATE_ROLE_HASH_DRIFT:" + role)
        if _mapping(validation.get("role_sha256"), field="candidate.validation.role_sha256").get(role) != digest:
            raise Stage1S18FormalError("S18_CANDIDATE_VALIDATION_ROLE_HASH_DRIFT:" + role)
        if name not in source_files and role != "gate_record":
            # The five public roles are not reproduction inputs.  Their exact
            # identities are checked above rather than requiring duplication.
            continue

    requirements = _mapping(gate.get("requirements"), field="candidate.gate.requirements")
    checks = validation.get("checks")
    if set(requirements) != set(GATE_CHECK_IDS) or any(value is not True for value in requirements.values()):
        raise Stage1S18FormalError("S18_CANDIDATE_GATE_REQUIREMENTS_INVALID")
    if not isinstance(checks, list) or [entry.get("check_id") if isinstance(entry, Mapping) else None for entry in checks] != list(GATE_CHECK_IDS):
        raise Stage1S18FormalError("S18_CANDIDATE_VALIDATION_CHECK_ORDER_INVALID")
    if any(not isinstance(entry, Mapping) or entry.get("status") != "PASS" for entry in checks):
        raise Stage1S18FormalError("S18_CANDIDATE_VALIDATION_CHECK_STATUS_INVALID")
    if index.get("gate_artifact_hash") != gate.get("artifact_hash") or index.get("validation_ref") != "validation.json" or index.get("validation_sha256") != _canonical_file_sha(validation) or index.get("replay_ref") != "replay-validation.json" or index.get("replay_sha256") != _canonical_file_sha(replay_record):
        raise Stage1S18FormalError("S18_CANDIDATE_INDEX_ROLE_CROSSREF_INVALID")
    fixture_binding = {"fixture_schema_version": fixture.get("schema_version"), "fixture_id": fixture.get("fixture_id")}
    if (
        ddp.get("fixture_hash") != fixture.get("fixture_hash")
        or any({key: value.get(key) for key in fixture_binding} != fixture_binding for value in (ddp, replay_record, validation, index))
        or table.get("status") != "PASS" or replay_record.get("status") != "PASS"
        or ddp.get("nccl_transport_protocol") != _nccl_transport_protocol() or index.get("nccl_transport_protocol") != _nccl_transport_protocol()
    ):
        raise Stage1S18FormalError("S18_CANDIDATE_FIXTURE_OR_REPLAY_CROSSREF_INVALID")
    negatives = _mapping(ddp.get("negative_controls"), field="candidate.ddp.negative_controls")
    if set(negatives) != {"ordinary_sync_negative", "inject_rank_failure"}:
        raise Stage1S18FormalError("S18_CANDIDATE_NEGATIVE_CONTROL_SET_INVALID")
    for name, raw in negatives.items():
        negative = _mapping(raw, field="candidate.ddp.negative." + name)
        process = _mapping(negative.get("process"), field="candidate.ddp.negative.process." + name)
        tree = _mapping(process.get("initial_tree"), field="candidate.ddp.negative.initial_tree." + name)
        known_tree = _mapping(process.get("known_tree"), field="candidate.ddp.negative.known_tree." + name)
        residual = _mapping(process.get("residual_launch_tree"), field="candidate.ddp.negative.residual_tree." + name)
        members = tree.get("members")
        known_members = known_tree.get("members")
        fingerprint = _mapping(process.get("fingerprint"), field="candidate.ddp.negative.fingerprint." + name)
        command = process.get("command")
        initial_pids = tree.get("member_pids")
        known_pids = known_tree.get("member_pids")
        initial_depths, known_depths = tree.get("ancestry_depths"), known_tree.get("ancestry_depths")
        endpoint = process.get("rendezvous_endpoint")
        if isinstance(command, list):
            rendezvous_id_positions = [index for index, value in enumerate(command) if value == "--rdzv-id"]
            endpoint_positions = [index for index, value in enumerate(command) if value == "--rdzv-endpoint"]
        else:
            rendezvous_id_positions, endpoint_positions = [], []
        if (
            not _is_nonzero_loopback_endpoint(endpoint)
            or process.get("rendezvous_id") != fingerprint.get("environment_run_token")
            or process.get("rendezvous_handoff") != {"reservation_held_to_popen": True, "single_attempt": True, "silent_retry": False}
            or not isinstance(members, list)
            or not isinstance(known_members, list)
            or initial_pids != [item.get("pid") for item in members if isinstance(item, Mapping)]
            or known_pids != [item.get("pid") for item in known_members if isinstance(item, Mapping)]
            or not isinstance(initial_depths, Mapping) or set(initial_depths) != {str(pid) for pid in initial_pids if isinstance(pid, int)}
            or not isinstance(known_depths, Mapping) or set(known_depths) != {str(pid) for pid in known_pids if isinstance(pid, int)}
            or not isinstance(initial_pids, list) or not isinstance(known_pids, list) or not set(initial_pids).issubset(set(known_pids))
            or len(rendezvous_id_positions) != 1 or rendezvous_id_positions[0] + 1 >= len(command) or command[rendezvous_id_positions[0] + 1] != process.get("rendezvous_id")
            or len(endpoint_positions) != 1 or endpoint_positions[0] + 1 >= len(command) or command[endpoint_positions[0] + 1] != endpoint
            or residual != {"session_members": [], "token_members": []}
        ):
            raise Stage1S18FormalError("S18_CANDIDATE_NEGATIVE_PROCESS_TREE_INVALID:" + name)
        _validate_recorded_launch_tree(fingerprint=fingerprint, tree=tree, expected_member_count=None)
        _validate_process_outcome_contract(process)
    if index.get("implementation_source_sha256") != ddp.get("implementation_source_sha256"):
        raise Stage1S18FormalError("S18_CANDIDATE_SOURCE_MAP_CROSSREF_INVALID")
    _validate_implementation_source_map(repository, index.get("implementation_source_sha256"))
    capability = _mapping(index.get("gpu_capability"), field="candidate.gpu_capability")
    handoff = _mapping(index.get("s1_7_handoff"), field="candidate.s1_7_handoff")
    if (
        set(handoff) != S1_7_INDEX_HANDOFF_KEYS
        or
        capability.get("artifact_hash") != EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH
        or handoff.get("producer_commit") != EXPECTED_S1_7_PRODUCER
        or handoff.get("index_sha256") != EXPECTED_S1_7_INDEX_SHA256
        or handoff.get("gate_artifact_hash") != EXPECTED_G1_SINGLE_HASH
        or handoff.get("historical_producer_attestation_ref") != HISTORICAL_G3_REPRODUCTION_ROLES["historical_producer_attestation"]
        or handoff.get("historical_producer_attestation_sha256") != EXPECTED_S1_7_HISTORICAL_PRODUCER_ATTESTATION_SHA256
        or handoff.get("historical_g3_replay_ref") != HISTORICAL_G3_REPRODUCTION_ROLES["historical_g3_replay"]
        or handoff.get("historical_g3_replay_sha256") != EXPECTED_S1_7_HISTORICAL_G3_REPLAY_SHA256
        or _require_sha256(handoff.get("historical_g3_replay_hash"), field="candidate.historical_g3.replay_hash") != handoff.get("historical_g3_replay_hash")
        or handoff.get("historical_g3_historical_producer_commit") != HISTORICAL_G3_PRODUCER
        or handoff.get("historical_g3_critical_patch_sha256") != EXPECTED_HISTORICAL_G3_PATCH_SHA256
        or set(_mapping(handoff.get("historical_g3_historical_source_sha256"), field="candidate.historical_g3.historical_sources")) != set(HISTORICAL_G3_CRITICAL_SOURCE_REFS)
        or set(_mapping(handoff.get("historical_g3_current_consumer_source_sha256"), field="candidate.historical_g3.consumer_sources")) != set(HISTORICAL_G3_CRITICAL_SOURCE_REFS)
    ):
        raise Stage1S18FormalError("S18_CANDIDATE_UPSTREAM_OR_CAPABILITY_DRIFT")

    refs, hashes = _mapping(index.get("reproduction_role_refs"), field="candidate.reproduction.refs"), _mapping(index.get("reproduction_role_sha256"), field="candidate.reproduction.hashes")
    expected_reproduction = _fixed_reproduction_roles()
    expected_refs = {role: published for role, (published, _source) in expected_reproduction.items()}
    if refs != expected_refs or set(hashes) != set(expected_refs) or set(source_files) != set(expected_refs.values()):
        raise Stage1S18FormalError("S18_CANDIDATE_REPRODUCTION_ROLE_CLOSURE_INVALID")
    if set(refs) != set(hashes) or not refs:
        raise Stage1S18FormalError("S18_CANDIDATE_REPRODUCTION_ROLE_SET_INVALID")
    published_names: set[str] = set()
    for role, name in refs.items():
        if not isinstance(name, str) or name in published_names or name not in source_files:
            raise Stage1S18FormalError("S18_CANDIDATE_REPRODUCTION_REF_INVALID:" + str(role))
        published_names.add(name)
        source = source_files[name]
        if not source.is_file() or _require_sha256(hashes.get(role), field="reproduction." + str(role)) != _sha(source):
            raise Stage1S18FormalError("S18_CANDIDATE_REPRODUCTION_HASH_DRIFT:" + str(role))
    _validate_gpu_quiescence_publication(
        repository=repository, ddp=ddp, validation=validation, index=index,
        source_files=source_files,
    )
    descriptors = _mapping(bundle.get("route_artifacts"), field="candidate.array_bundle.routes")
    if set(descriptors) != set(worker_reports):
        raise Stage1S18FormalError("S18_CANDIDATE_ARRAY_DESCRIPTOR_SET_INVALID")
    for route_key, raw in descriptors.items():
        descriptor = _mapping(raw, field="candidate.array_bundle." + route_key)
        artifact_ref, manifest_ref = descriptor.get("artifact_ref"), descriptor.get("manifest_ref")
        if not isinstance(artifact_ref, str) or not isinstance(manifest_ref, str) or artifact_ref not in source_files or manifest_ref not in source_files:
            raise Stage1S18FormalError("S18_CANDIDATE_ARRAY_REFERENCE_INVALID:" + route_key)
        manifest = manifests[route_key]
        if _sha(source_files[artifact_ref]) != descriptor.get("file_sha256") or source_files[artifact_ref].stat().st_size != descriptor.get("file_size_bytes") or _sha(source_files[manifest_ref]) != _canonical_file_sha(worker_reports[route_key]) or descriptor.get("manifest_hash") != manifest.get("artifact_hash"):
            raise Stage1S18FormalError("S18_CANDIDATE_ARRAY_DESCRIPTOR_DRIFT:" + route_key)
    return True


def _path(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S18FormalError(f"S18_FORMAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S18FormalError(f"S18_FORMAL_REF_ESCAPE:{field}")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S18FormalError(f"S18_FORMAL_REF_ESCAPE:{field}") from error
    return candidate


def _candidate(root: Path, index_path: Path, reference: object, *, field: str) -> Path:
    direct = _path(root, reference, field=field)
    if direct.is_file():
        return direct
    if not isinstance(reference, str):
        raise Stage1S18FormalError(f"S18_FORMAL_ROLE_REF_INVALID:{field}")
    relative = (index_path.parent / reference).resolve()
    try:
        relative.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S18FormalError(f"S18_FORMAL_ROLE_REF_ESCAPE:{field}") from error
    return relative


def _git(repository: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repository), *args], text=True, capture_output=True, timeout=30, check=False)
    if done.returncode:
        raise Stage1S18FormalError(f"S18_FORMAL_GIT_FAILED:{args[0]}")
    return done.stdout.strip()


def _git_bytes(repository: Path, *args: str) -> bytes:
    """Read exact Git bytes; historical patch identity must not be text-normalized."""

    done = subprocess.run(
        ["git", "-C", str(repository), *args], capture_output=True,
        timeout=30, check=False,
    )
    if done.returncode:
        raise Stage1S18FormalError(f"S18_FORMAL_GIT_FAILED:{args[0]}")
    return bytes(done.stdout)


def _current_historical_g3_compatibility(
    repository: Path, attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that today's three G3-critical sources are exactly S1.7-compatible.

    This deliberately does *not* ask the current G3 resolver to reinterpret a
    historical resolution.  The producer's immutable attestation supplies the
    only permitted historical patch, while this check makes later source drift
    fail closed before any CUDA discovery or lease.
    """

    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", HISTORICAL_G3_PRODUCER, "HEAD"],
        capture_output=True, timeout=30, check=False,
    )
    if ancestor.returncode != 0:
        raise Stage1S18FormalError("S18_HISTORICAL_G3_PRODUCER_NOT_ANCESTOR")
    changed = tuple(filter(None, _git(
        repository, "diff", "--name-only", HISTORICAL_G3_PRODUCER, "HEAD", "--",
        *HISTORICAL_G3_CRITICAL_SOURCE_REFS,
    ).splitlines()))
    s1_7_to_current_changed = tuple(filter(None, _git(
        repository, "diff", "--name-only", EXPECTED_S1_7_PRODUCER, "HEAD", "--",
        *HISTORICAL_G3_CRITICAL_SOURCE_REFS,
    ).splitlines()))
    patch_sha256 = hashlib.sha256(_git_bytes(
        repository, "diff", "--binary", "--full-index", HISTORICAL_G3_PRODUCER,
        "HEAD", "--", *HISTORICAL_G3_CRITICAL_SOURCE_REFS,
    )).hexdigest()
    source_hashes = {reference: _sha(repository / reference) for reference in HISTORICAL_G3_CRITICAL_SOURCE_REFS}
    expected_hashes = _mapping(attestation.get("consumer_source_sha256"), field="historical_g3.attestation.consumer_source_sha256")
    if (
        changed != HISTORICAL_G3_CRITICAL_SOURCE_REFS
        or s1_7_to_current_changed != ()
        or patch_sha256 != EXPECTED_HISTORICAL_G3_PATCH_SHA256
        or source_hashes != expected_hashes
    ):
        raise Stage1S18FormalError("S18_HISTORICAL_G3_CONSUMER_COMPATIBILITY_DRIFT")
    return {
        "current_consumer_commit": _git(repository, "rev-parse", "HEAD"),
        "historical_producer_is_ancestor": True,
        "critical_source_diff": list(changed),
        "s1_7_producer_to_current_critical_source_diff": [],
        "critical_patch_sha256": patch_sha256,
        "consumer_source_sha256": source_hashes,
    }


def _validate_s1_7_historical_g3_binding(
    *, repository: Path, handoff: Mapping[str, Any], attestation: Mapping[str, Any], replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly replay-bind S1.8 to the published S1.7 historical G3 proof."""

    attestation_expected = {
        "schema_version", "status", "historical_producer_commit", "consumer_commit",
        "historical_producer_is_ancestor", "critical_source_diff", "critical_patch_sha256",
        "historical_source_sha256", "consumer_source_sha256", "artifact_hash",
    }
    expected_hash_keys = set(HISTORICAL_G3_CRITICAL_SOURCE_REFS)
    historical_hashes = _mapping(attestation.get("historical_source_sha256"), field="historical_g3.attestation.historical_source_sha256")
    consumer_hashes = _mapping(attestation.get("consumer_source_sha256"), field="historical_g3.attestation.consumer_source_sha256")
    if (
        set(attestation) != attestation_expected
        or not _self_hash(attestation)
        or attestation.get("schema_version") != "stage1-s1-7-historical-producer-attestation-v1"
        or attestation.get("status") != "PASS"
        or attestation.get("historical_producer_commit") != HISTORICAL_G3_PRODUCER
        or attestation.get("consumer_commit") != EXPECTED_S1_7_PRODUCER
        or attestation.get("historical_producer_is_ancestor") is not True
        or attestation.get("critical_source_diff") != list(HISTORICAL_G3_CRITICAL_SOURCE_REFS)
        or attestation.get("critical_patch_sha256") != EXPECTED_HISTORICAL_G3_PATCH_SHA256
        or set(historical_hashes) != expected_hash_keys
        or set(consumer_hashes) != expected_hash_keys
        or any(_require_sha256(value, field="historical_g3.source_hash") != value for value in (*historical_hashes.values(), *consumer_hashes.values()))
    ):
        raise Stage1S18FormalError("S18_S17_HISTORICAL_G3_ATTESTATION_INVALID")

    replay_expected = {
        "schema_version", "status", "model", "tokenizer", "pile", "asset_identity",
        "resolution_commit_artifact_hash", "resolution_artifact_hash", "fixture_file",
        "fixture_file_sha256", "token_sha256", "dropout_probabilities", "resolve_hash_seconds",
        "dataset_rehash_seconds", "qualified_resolution_hashed_bytes", "dataset_rehash_bytes",
        "pile_hash_passes", "network_policy", "replay_hash",
    }
    network = _mapping(replay.get("network_policy"), field="historical_g3.replay.network_policy")
    identity = _mapping(replay.get("asset_identity"), field="historical_g3.replay.asset_identity")
    model_identity = _mapping(identity.get("model"), field="historical_g3.replay.asset_identity.model")
    tokenizer_identity = _mapping(identity.get("tokenizer"), field="historical_g3.replay.asset_identity.tokenizer")
    pile_identity = _mapping(identity.get("pile"), field="historical_g3.replay.asset_identity.pile")
    fixture_assets = _mapping(handoff.get("fixture_assets"), field="historical_g3.fixture.assets")
    tokens = handoff.get("token_sha256")
    if (
        set(replay) != replay_expected
        or replay.get("schema_version") != "stage1-s1-7-historical-g3-replay-v1"
        or replay.get("status") != "PASS"
        or replay.get("replay_hash") != _canonical({key: value for key, value in replay.items() if key != "replay_hash"})
        or replay.get("resolution_commit_artifact_hash") != EXPECTED_G3_RESOLUTION_ARTIFACT_HASH
        or replay.get("resolution_artifact_hash") != EXPECTED_G3_RESOLUTION_PAYLOAD_HASH
        or replay.get("fixture_file") != "fixture-inputs.safetensors"
        or replay.get("fixture_file_sha256") != handoff.get("token_file_sha256")
        or replay.get("token_sha256") != tokens
        or replay.get("model") != handoff.get("model_provenance")
        or replay.get("tokenizer") != fixture_assets.get("tokenizer")
        or replay.get("pile") != handoff.get("pile_provenance")
        or {key: model_identity.get(key) for key in EXPECTED_MODEL_IDENTITY} != EXPECTED_MODEL_IDENTITY
        or {key: tokenizer_identity.get(key) for key in EXPECTED_TOKENIZER_IDENTITY} != EXPECTED_TOKENIZER_IDENTITY
        or {key: pile_identity.get(key) for key in EXPECTED_PILE_IDENTITY} != EXPECTED_PILE_IDENTITY
        or not isinstance(model_identity.get("root"), str) or not model_identity["root"]
        or not isinstance(tokenizer_identity.get("root"), str) or not tokenizer_identity["root"]
        or replay.get("qualified_resolution_hashed_bytes") != EXPECTED_HISTORICAL_PILE_HASHED_BYTES
        or replay.get("dataset_rehash_bytes") != EXPECTED_HISTORICAL_PILE_HASHED_BYTES
        or replay.get("pile_hash_passes") != 2
        or network != {
            "hf_hub_offline": True, "transformers_offline": True,
            "datasets_offline": True, "cuda_visible_devices": True,
            "cuda_is_available": False,
            "operations": ["committed-resolution-parse", "qualified-local-manifest-parse", "local-pile-mmap-hash-and-fixture-extraction"],
            "external_attempts": [],
        }
    ):
        raise Stage1S18FormalError("S18_S17_HISTORICAL_G3_REPLAY_INVALID")
    compatibility = _current_historical_g3_compatibility(repository, attestation)
    return {
        "qualification_method": "s1_7_published_historical_g3_replay_consumer_binding",
        "historical_producer_attestation": {
            "ref": handoff["historical_producer_attestation_ref"],
            "sha256": handoff["historical_producer_attestation_sha256"],
            "artifact_hash": attestation["artifact_hash"],
            "historical_producer_commit": HISTORICAL_G3_PRODUCER,
            "critical_patch_sha256": EXPECTED_HISTORICAL_G3_PATCH_SHA256,
            "historical_source_sha256": historical_hashes,
        },
        "historical_g3_replay": {
            "ref": handoff["historical_g3_replay_ref"],
            "sha256": handoff["historical_g3_replay_sha256"],
            "replay_hash": replay["replay_hash"],
            "network_policy": network,
            "model_identity": model_identity,
            "pile_identity": pile_identity,
        },
        "current_consumer_compatibility": compatibility,
    }


def _index_safe_s1_7_handoff(handoff: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the immutable historical proof into the strict public index."""

    binding = _mapping(handoff.get("historical_g3_binding"), field="index.historical_g3_binding")
    attestation = _mapping(binding.get("historical_producer_attestation"), field="index.historical_g3_attestation")
    replay = _mapping(binding.get("historical_g3_replay"), field="index.historical_g3_replay")
    compatibility = _mapping(binding.get("current_consumer_compatibility"), field="index.historical_g3_consumer_compatibility")
    excluded = {
        "token_file", "fixture_assets", "historical_producer_attestation",
        "historical_g3_replay", "historical_g3_binding",
        "historical_producer_attestation_file", "historical_g3_replay_file",
    }
    value = {key: item for key, item in handoff.items() if key not in excluded}
    value.update({
        "historical_g3_attestation_artifact_hash": attestation["artifact_hash"],
        "historical_g3_historical_producer_commit": attestation["historical_producer_commit"],
        "historical_g3_critical_patch_sha256": attestation["critical_patch_sha256"],
        "historical_g3_historical_source_sha256": attestation["historical_source_sha256"],
        "historical_g3_replay_hash": replay["replay_hash"],
        "historical_g3_current_consumer_commit": compatibility["current_consumer_commit"],
        "historical_g3_current_consumer_source_sha256": compatibility["consumer_source_sha256"],
    })
    return value


def _audit_consumer_diff(repository: Path) -> tuple[str, ...]:
    """A historical S1.7 gate is invalidated by unrelated semantic drift."""

    if _git(repository, "status", "--porcelain=v1"):
        raise Stage1S18FormalError("S18_FORMAL_WORKTREE_NOT_CLEAN")
    changed = tuple(filter(None, _git(repository, "diff", "--name-only", EXPECTED_S1_7_PRODUCER, "HEAD").splitlines()))
    allowed = (
        ".gitignore",
        "worklogs/2026-08-15-s1.7-single-gpu-pythia14m.md",
        "ops/stage1/formalize_s1_8.py", "ops/stage1/run_s1_8_worker.py", "ops/stage1/run_s1_8_scale_oracle.py", "ops/stage1/run_s1_8_nccl_smoke.py",
    )
    def owned(name: str) -> bool:
        return name in allowed or (
            name.startswith("src/param_importance_nlp/stage1_ddp") and name.endswith(".py")
        ) or (name.startswith("schemas/stage1/s1-8-") and name.endswith(".json")) or (
            name.startswith("tests/test_stage1_s18_") and name.endswith(".py")
        )
    rejected = [name for name in changed if not owned(name)]
    if rejected:
        raise Stage1S18FormalError("S18_S17_CONSUMER_DIFF_UNAUTHORIZED:" + ",".join(rejected))
    return changed


def _validate_frozen_pile_provenance(model_asset: Mapping[str, Any], pile_asset: Mapping[str, Any]) -> None:
    """Bind the S1.7 Pile READY asset before any S1.8 lease exists."""

    if (
        dict(pile_asset) != EXPECTED_PILE_PROVENANCE
        or pile_asset.get("g3_resolution_ref") != model_asset.get("g3_resolution_ref")
        or pile_asset.get("g3_resolution_artifact_hash") != model_asset.get("g3_resolution_artifact_hash")
    ):
        raise Stage1S18FormalError("S18_S17_PILE_PROVENANCE_INVALID")


def load_s1_7_handoff(*, data_root: Path, index_ref: str, repository: Path) -> dict[str, Any]:
    """Consume only the exact approved S1.7 index, gate, roles and fixture."""

    from param_importance_nlp.contracts.jsonio import load_canonical_json
    _audit_consumer_diff(repository)
    index_path = _path(data_root, index_ref, field="s1_7_index_ref")
    if not index_path.is_file() or _sha(index_path) != EXPECTED_S1_7_INDEX_SHA256:
        raise Stage1S18FormalError("S18_S17_INDEX_HASH_NOT_CURRENT")
    index = _mapping(load_canonical_json(index_path), field="s1_7.index")
    if not _self_hash(index) or index.get("schema_version") != "stage1-s1-7-formalization-index-v1" or index.get("status") != "PASS" or index.get("task_id") != "stage1.07_single_gpu_pythia14m" or index.get("generator_git_commit") != EXPECTED_S1_7_PRODUCER or index.get("gate_artifact_hash") != EXPECTED_G1_SINGLE_HASH or index.get("artifact_hash") != EXPECTED_S1_7_ARTIFACT_HASH:
        raise Stage1S18FormalError("S18_S17_INDEX_IDENTITY_INVALID")
    next_ids = index.get("next_task_ids")
    if not isinstance(next_ids, list) or TASK_ID not in next_ids:
        raise Stage1S18FormalError("S18_S17_NEXT_TASK_NOT_AUTHORIZED")
    refs = _mapping(index.get("role_refs"), field="s1_7.role_refs")
    hashes = _mapping(index.get("role_sha256"), field="s1_7.role_sha256")
    expected_roles = {"fixture_manifest", "single_gpu_report", "gradient_bundle", "comparison_table", "gate_record"}
    if set(refs) != expected_roles or set(hashes) != expected_roles:
        raise Stage1S18FormalError("S18_S17_ROLE_SET_INVALID")
    roles: dict[str, Mapping[str, Any]] = {}
    for role in sorted(expected_roles):
        file = _candidate(data_root, index_path, refs[role], field=f"s1_7.{role}")
        expected = hashes[role]
        if not isinstance(expected, str) or len(expected) != 64 or not file.is_file() or _sha(file) != expected:
            raise Stage1S18FormalError(f"S18_S17_ROLE_HASH_INVALID:{role}")
        roles[role] = _mapping(load_canonical_json(file), field=f"s1_7.{role}")
    gate = roles["gate_record"]
    if not _self_hash(gate) or gate.get("status") != "PASS" or gate.get("artifact_hash") != EXPECTED_G1_SINGLE_HASH:
        raise Stage1S18FormalError("S18_S17_GATE_INVALID")
    reproduce_refs = _mapping(index.get("reproduction_role_refs"), field="s1_7.reproduction_role_refs")
    reproduce_hashes = _mapping(index.get("reproduction_role_sha256"), field="s1_7.reproduction_role_sha256")
    historical_roles: dict[str, dict[str, Any]] = {}
    historical_role_files: dict[str, Path] = {}
    for role, expected_name in HISTORICAL_G3_REPRODUCTION_ROLES.items():
        reference, digest = reproduce_refs.get(role), reproduce_hashes.get(role)
        file = _candidate(data_root, index_path, reference, field="s1_7." + role)
        expected_digest = (
            EXPECTED_S1_7_HISTORICAL_PRODUCER_ATTESTATION_SHA256
            if role == "historical_producer_attestation"
            else EXPECTED_S1_7_HISTORICAL_G3_REPLAY_SHA256
        )
        if (
            reference != expected_name
            or not isinstance(digest, str)
            or digest != expected_digest
            or not file.is_file()
            or _require_sha256(digest, field="s1_7." + role) != digest
            or _sha(file) != digest
        ):
            raise Stage1S18FormalError("S18_S17_HISTORICAL_G3_ROLE_HASH_INVALID:" + role)
        historical_roles[role] = _mapping(load_canonical_json(file), field="s1_7." + role)
        historical_role_files[role] = file
    token_ref, token_hash = reproduce_refs.get("fixture_inputs"), reproduce_hashes.get("fixture_inputs")
    token_file = _candidate(data_root, index_path, token_ref, field="s1_7.fixture_inputs")
    if not isinstance(token_hash, str) or not token_file.is_file() or _sha(token_file) != token_hash:
        raise Stage1S18FormalError("S18_S17_FIXTURE_INPUT_HASH_INVALID")
    fixture = roles["fixture_manifest"]
    fixture_hash = fixture.get("fixture_hash")
    if not isinstance(fixture_hash, str) or len(fixture_hash) != 64:
        raise Stage1S18FormalError("S18_S17_FIXTURE_HASH_INVALID")
    assets = _mapping(fixture.get("assets"), field="s1_7.fixture.assets")
    model_asset = _mapping(assets.get("model"), field="s1_7.fixture.assets.model")
    pile_asset = _mapping(assets.get("pile"), field="s1_7.fixture.assets.pile")
    resolution_ref = model_asset.get("g3_resolution_ref")
    if (
        not isinstance(resolution_ref, str)
        or model_asset.get("logical_asset_id") != "pythia-14m-step0"
        or model_asset.get("ready_manifest_sha256") != EXPECTED_MODEL_READY_SHA256
    ):
        raise Stage1S18FormalError("S18_S17_MODEL_PROVENANCE_INVALID")
    _validate_frozen_pile_provenance(model_asset, pile_asset)
    tokens = fixture.get("token_sha256")
    if not isinstance(tokens, Mapping) or set(tokens) != {str(index) for index in range(16)}:
        raise Stage1S18FormalError("S18_S17_TOKEN_HASH_SET_INVALID")
    handoff = {
        "index_ref": index_ref, "index_sha256": EXPECTED_S1_7_INDEX_SHA256, "index_artifact_hash": EXPECTED_S1_7_ARTIFACT_HASH,
        "producer_commit": EXPECTED_S1_7_PRODUCER, "gate_artifact_hash": EXPECTED_G1_SINGLE_HASH,
        "fixture_hash": fixture_hash, "token_file": token_file, "token_file_sha256": token_hash,
        "model_resolution_ref": resolution_ref,
        "model_provenance": model_asset,
        "pile_provenance": pile_asset,
        "fixture_assets": assets,
        "token_sha256": {str(key): str(value) for key, value in tokens.items()},
        "role_refs": {str(key): str(value) for key, value in refs.items()}, "role_sha256": {str(key): str(value) for key, value in hashes.items()},
        "historical_producer_attestation_ref": str(reproduce_refs["historical_producer_attestation"]),
        "historical_producer_attestation_sha256": str(reproduce_hashes["historical_producer_attestation"]),
        "historical_g3_replay_ref": str(reproduce_refs["historical_g3_replay"]),
        "historical_g3_replay_sha256": str(reproduce_hashes["historical_g3_replay"]),
        "historical_producer_attestation": historical_roles["historical_producer_attestation"],
        "historical_g3_replay": historical_roles["historical_g3_replay"],
        "historical_producer_attestation_file": historical_role_files["historical_producer_attestation"],
        "historical_g3_replay_file": historical_role_files["historical_g3_replay"],
    }
    handoff["historical_g3_binding"] = _validate_s1_7_historical_g3_binding(
        repository=repository, handoff=handoff,
        attestation=historical_roles["historical_producer_attestation"],
        replay=historical_roles["historical_g3_replay"],
    )
    return handoff


def _audit_pile_download_activity(handoff: Mapping[str, Any], *, proc_root: Path | None = None) -> dict[str, Any]:
    """Reject project Pile downloader activity without retaining command text."""

    pile = _mapping(handoff.get("pile_provenance"), field="pile.provenance")
    records: list[dict[str, object]] = []
    proc_root = Path("/proc") if proc_root is None else proc_root
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes()
                lowered = cmdline.lower()
                if not any(signature in lowered for signature in PILE_DOWNLOADER_CMDLINE_SIGNATURES):
                    continue
                stat = (entry / "stat").read_text(encoding="utf-8").split()
                records.append({"pid": int(entry.name), "uid": (entry / "stat").stat().st_uid, "pgid": os.getpgid(int(entry.name)), "start_ticks": stat[21], "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(), "role": "project_pile_downloader"})
            except (OSError, ProcessLookupError, IndexError):
                continue
    if records:
        raise Stage1S18FormalError("S18_PILE_DOWNLOAD_ACTIVITY_PRESENT")
    return {"status": "PASS", "pile_logical_asset_id": pile["logical_asset_id"], "pile_ready_manifest_sha256": pile["ready_manifest_sha256"], "active_count": 0, "process_fingerprints": []}


def _require_prelease_cuda_hidden() -> dict[str, object]:
    """Keep the formalizer parent CPU-only until an owned child is launched."""

    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if value != "":
        raise Stage1S18FormalError("S18_PRELEASE_CUDA_VISIBLE_DEVICES_NOT_EMPTY")
    return {"cuda_visible_devices": "", "parent_cuda_initialization": False}


def _frozen_model_and_cache_root(repository: Path, data_root: Path, handoff: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Recheck current model bytes through S1.7's immutable historical proof.

    The current qualified G3 resolver intentionally rejects a checkout whose
    historical critical-source patch differs from its own policy.  That policy
    is left untouched: this consumer first verifies the published S1.7
    historical producer attestation and then invokes the independently exposed
    READY/qualification byte verifier only after its current source hash has
    been proven identical to that attested S1.7 consumer source.
    """

    from param_importance_nlp.assets import load_manifest, resolve_qualified_asset
    from param_importance_nlp.contracts.jsonio import load_canonical_json

    cache = _path(data_root, "cache", field="controlled_cache_root")
    provenance = _mapping(handoff.get("model_provenance"), field="frozen_model.provenance")
    binding = _mapping(handoff.get("historical_g3_binding"), field="frozen_model.historical_g3_binding")
    replay = _mapping(handoff.get("historical_g3_replay"), field="frozen_model.historical_g3_replay")
    identity = _mapping(_mapping(replay.get("asset_identity"), field="frozen_model.replay.asset_identity").get("model"), field="frozen_model.replay.model")
    if (
        not cache.is_dir()
        or provenance != replay.get("model")
        or {key: identity.get(key) for key in EXPECTED_MODEL_IDENTITY} != EXPECTED_MODEL_IDENTITY
        or binding.get("qualification_method") != "s1_7_published_historical_g3_replay_consumer_binding"
    ):
        raise Stage1S18FormalError("S18_FROZEN_MODEL_OR_CACHE_INVALID")
    try:
        manifest = load_manifest(_path(data_root, provenance.get("manifest_ref"), field="frozen_model.manifest_ref"))
        qualification = _mapping(load_canonical_json(_path(data_root, provenance.get("qualification_ref"), field="frozen_model.qualification_ref")), field="frozen_model.qualification")
        requirements = _mapping(load_canonical_json(repository / "configs" / "stage0" / "g3-asset-requirements-v1.json"), field="frozen_model.requirements")
        resolved = resolve_qualified_asset(
            manifest,
            _path(data_root, provenance.get("asset_root_ref"), field="frozen_model.asset_root_ref"),
            qualification,
            qualification_ref=str(provenance["qualification_ref"]),
            requirements_artifact_hash=str(requirements["artifact_hash"]),
        )
    except (OSError, TypeError, ValueError) as error:
        raise Stage1S18FormalError("S18_FROZEN_MODEL_QUALIFIED_RESOLUTION_FAILED") from error
    if (
        _canonical(manifest) != EXPECTED_MODEL_READY_SHA256
        or resolved.asset_id != EXPECTED_MODEL_IDENTITY["asset_id"]
        or resolved.revision != EXPECTED_MODEL_IDENTITY["revision"]
        or str(resolved.root) != identity.get("root")
    ):
        raise Stage1S18FormalError("S18_FROZEN_MODEL_IDENTITY_OR_PROVENANCE_DRIFT")
    summary = {
        "provenance": provenance,
        "resolution_ref": handoff["model_resolution_ref"],
        "resolution_artifact_hash": EXPECTED_G3_RESOLUTION_PAYLOAD_HASH,
        "files_checked": len(resolved.files),
        "bytes_checked": sum(item.size_bytes for item in resolved.files),
        "content_hashes": {item.relative_path: item.sha256 for item in resolved.files},
        "historical_g3_binding": binding,
        "verification_path": "current_assets.resolve_qualified_asset_after_exact_s1_7_historical_consumer_source_binding",
    }
    return str(resolved.root), str(cache), summary


def _run(command: Sequence[str], *, timeout: int = 30) -> str:
    done = subprocess.run(list(command), text=True, capture_output=True, check=False, timeout=timeout)
    if done.returncode:
        raise Stage1S18FormalError(f"S18_FORMAL_COMMAND_FAILED:{Path(command[0]).name}:{done.stderr[-300:]}")
    return done.stdout


def _load_capability(data_root: Path, reference: str, approved: Sequence[str]) -> dict[str, Any]:
    """Consume only the committed, formal CUDA capability record used by S1.7."""

    from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact
    from param_importance_nlp.contracts.runtime_evidence import RuntimeCapabilityEvidence
    if reference != EXPECTED_GPU_CAPABILITY_REF:
        raise Stage1S18FormalError("S18_GPU_CAPABILITY_REF_NOT_FROZEN")
    file = _path(data_root, reference, field="gpu_capability_ref")
    if not file.is_file() or _sha(file) != EXPECTED_GPU_CAPABILITY_FILE_SHA256:
        raise Stage1S18FormalError("S18_GPU_CAPABILITY_FILE_DRIFT")
    try:
        artifact = load_committed_task_artifact(data_root, reference, require_formal=True)
        evidence = RuntimeCapabilityEvidence.from_mapping(_mapping(artifact.payload, field="gpu_capability.payload"))
    except (OSError, TypeError, ValueError) as error:
        raise Stage1S18FormalError("S18_GPU_CAPABILITY_INVALID") from error
    allowed = _mapping(evidence.metadata, field="gpu_capability.metadata").get("allowed_gpu_uuids")
    if artifact.identity.task_id != "stage0.01_baseline_and_safety" or artifact.identity.artifact_kind != "capability_cuda" or artifact.identity.artifact_hash != EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH or evidence.capability != "cuda" or evidence.status != "VERIFIED" or not isinstance(allowed, list) or any(uuid not in allowed for uuid in approved):
        raise Stage1S18FormalError("S18_GPU_CAPABILITY_UUID_NOT_QUALIFIED")
    return {"commit_ref": artifact.identity.commit_ref, "object_ref": artifact.identity.object_ref, "task_id": artifact.identity.task_id, "artifact_kind": artifact.identity.artifact_kind, "artifact_hash": artifact.identity.artifact_hash, "config_hash": artifact.identity.config_hash, "source_refs": list(artifact.source_refs), "allowed_gpu_uuids": allowed}


def _approved_gpu_uuid_values(approved_gpu_uuids: Sequence[str]) -> tuple[str, ...]:
    values = tuple(approved_gpu_uuids)
    if len(values) != 4 or len(set(values)) != 4 or any(not isinstance(value, str) or GPU_UUID_RE.fullmatch(value) is None for value in values):
        raise Stage1S18FormalError("S18_APPROVED_GPU_UUID_SET_INVALID")
    return values


def _parse_gpu_compute_apps(output: str, *, expected_uuids: Sequence[str]) -> list[dict[str, Any]]:
    """Parse selected compute applications without persisting command lines."""

    expected = set(_approved_gpu_uuid_values(expected_uuids))
    apps: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or GPU_UUID_RE.fullmatch(fields[0]) is None or not fields[1].isdigit() or int(fields[1]) <= 0 or not fields[2]:
            raise Stage1S18FormalError("S18_GPU_COMPUTE_APPS_PARSE_INVALID")
        if fields[0] in expected:
            # nvidia-smi returns the executable/process name, not cmdline.  Do
            # not query /proc here, so a command line can never enter evidence.
            apps.append({"gpu_uuid": fields[0], "pid": int(fields[1]), "process_name": fields[2]})
    return apps


def _parse_approved_gpu_inventory(output: str, *, expected_uuids: Sequence[str]) -> list[dict[str, Any]]:
    """Parse one exact inventory that includes the recovery action per UUID."""

    values = _approved_gpu_uuid_values(expected_uuids)
    inventory: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if (
            len(fields) != 9 or not fields[0].isdigit() or GPU_UUID_RE.fullmatch(fields[1]) is None
            or not all(field.isdigit() for field in fields[3:7]) or not fields[2] or not fields[7]
            or not fields[8] or fields[1] in inventory
        ):
            raise Stage1S18FormalError("S18_GPU_DISCOVERY_PARSE_INVALID")
        inventory[fields[1]] = {
            "physical_index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]),
            "utilization_percent": int(fields[5]), "temperature_c": int(fields[6]),
            "compute_capability": fields[7], "recovery_action": fields[8],
        }
    selected: list[dict[str, Any]] = []
    for uuid in values:
        row = inventory.get(uuid)
        if row is None:
            raise Stage1S18FormalError("S18_GPU_DISCOVERY_UUID_MISMATCH")
        selected.append(row)
    return selected


def _probe_approved_gpus(approved_gpu_uuids: Sequence[str]) -> dict[str, Any]:
    """Perform one complete structured selected-GPU observation.

    The inventory and Recovery Action are one supported ``nvidia-smi`` query,
    while compute apps remain an independent exact query.  This only reduces
    management-process startup; it does not alter the recorded fields or any
    immediate identity/recovery/PID failure semantics.
    """

    values = _approved_gpu_uuid_values(approved_gpu_uuids)
    selected = _parse_approved_gpu_inventory(
        _run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,compute_cap,gpu_recovery_action", "--format=csv,noheader,nounits"]),
        expected_uuids=values,
    )
    apps = _parse_gpu_compute_apps(
        _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"]),
        expected_uuids=values,
    )
    return {"selected": selected, "requested_uuid_order": list(values), "compute_apps": apps}


def _gpu_idle_violations(probe: Mapping[str, Any]) -> list[str]:
    """Return exact-idle violations for an already parsed selected probe."""

    selected = probe.get("selected")
    if not isinstance(selected, list):
        raise Stage1S18FormalError("S18_GPU_DISCOVERY_PARSE_INVALID")
    violations: list[str] = []
    for row in selected:
        if not isinstance(row, Mapping) or not isinstance(row.get("uuid"), str):
            raise Stage1S18FormalError("S18_GPU_DISCOVERY_PARSE_INVALID")
        uuid = str(row["uuid"])
        if (
            not str(row.get("name")).startswith("NVIDIA A100")
            or int(row.get("memory_total_mib", -1)) < 70000
            or int(row.get("temperature_c", 85)) >= 85
            or str(row.get("compute_capability")) != "8.0"
        ):
            violations.append("S18_GPU_NOT_IDLE_A100:" + uuid)
        if row.get("recovery_action") != "None":
            violations.append("S18_GPU_RECOVERY_ACTION_NOT_NONE:" + uuid + ":" + str(row.get("recovery_action")))
        if int(row.get("memory_used_mib", -1)) != 0 or int(row.get("utilization_percent", -1)) != 0:
            violations.append("S18_GPU_NOT_IDLE_A100:" + uuid)
    return violations


def _gpu_quiescence_immediate_violations(probe: Mapping[str, Any]) -> list[str]:
    """Classify failures that cannot become benign by waiting one cadence."""

    selected = probe.get("selected")
    apps = probe.get("compute_apps")
    if not isinstance(selected, list) or not isinstance(apps, list):
        raise Stage1S18FormalError("S18_GPU_DISCOVERY_PARSE_INVALID")
    immediate: list[str] = []
    for row in selected:
        if not isinstance(row, Mapping) or not isinstance(row.get("uuid"), str):
            raise Stage1S18FormalError("S18_GPU_DISCOVERY_PARSE_INVALID")
        uuid = str(row["uuid"])
        if (
            not str(row.get("name")).startswith("NVIDIA A100")
            or int(row.get("memory_total_mib", -1)) < 70000
            or int(row.get("temperature_c", 85)) >= 85
            or str(row.get("compute_capability")) != "8.0"
        ):
            immediate.append("S18_GPU_NOT_IDLE_A100:" + uuid)
        if row.get("recovery_action") != "None":
            immediate.append("S18_GPU_RECOVERY_ACTION_NOT_NONE:" + uuid + ":" + str(row.get("recovery_action")))
    for app in apps:
        if not isinstance(app, Mapping):
            raise Stage1S18FormalError("S18_GPU_COMPUTE_APPS_PARSE_INVALID")
        immediate.append("S18_GPU_COMPUTE_PROCESS_PRESENT:" + str(app.get("gpu_uuid")) + ":" + str(app.get("pid")))
    return immediate


def discover_approved_gpus(approved_gpu_uuids: Sequence[str]) -> dict[str, Any]:
    """Resolve exact UUIDs freshly and require a one-shot exact idle snapshot."""

    probe = _probe_approved_gpus(approved_gpu_uuids)
    violations = _gpu_idle_violations(probe)
    if violations:
        raise Stage1S18FormalError(violations[0])
    return probe


def _gpu_quiescence_sample(
    *,
    sample_index: int,
    started_monotonic: float,
    observed_monotonic: float,
    consecutive_exact_idle_samples: int,
    probe: Mapping[str, Any] | None,
    violations: Sequence[str],
) -> dict[str, Any]:
    """Make one safe, complete sampler observation record.

    The structured observation exposes only selected-GPU inventory and the
    process name reported by nvidia-smi.  It never reads or stores cmdline.
    """

    selected = [] if probe is None else probe.get("selected")
    apps = [] if probe is None else probe.get("compute_apps")
    requested = [] if probe is None else probe.get("requested_uuid_order")
    if not isinstance(selected, list) or not isinstance(apps, list) or not isinstance(requested, list):
        raise Stage1S18FormalError("S18_GPU_QUIESCENCE_OBSERVATION_INVALID")
    return {
        "sample_index": sample_index,
        "observed_at": _now(),
        # This is the actual post-probe completion time, including query and
        # parsing duration.  Do not clamp it: a value beyond the deadline is
        # evidence for the failed bounded wait.
        "monotonic_elapsed_seconds": observed_monotonic - started_monotonic,
        "requested_uuid_order": requested,
        "selected": selected,
        "compute_apps": apps,
        "violations": list(violations),
        "exact_selected_idle": not violations,
        "consecutive_exact_idle_samples": consecutive_exact_idle_samples,
    }


def _write_gpu_quiescence_record(
    *,
    work: Path,
    phase: str,
    status: str,
    started_at: str,
    samples: Sequence[Mapping[str, Any]],
    final_gpu: Mapping[str, Any] | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    if phase not in {"prelease", "post_worker", "post_release", "reacquire_preflight"} or status not in {"PASS", "FAILED"}:
        raise Stage1S18FormalError("S18_GPU_QUIESCENCE_RECORD_INVALID")
    value = _with_hash({
        "schema_version": GPU_QUIESCENCE_SCHEMA_VERSION,
        "status": status,
        "phase": phase,
        "started_at": started_at,
        "timeout_seconds": GPU_QUIESCENCE_TIMEOUT_SECONDS,
        "sample_interval_seconds": GPU_QUIESCENCE_SAMPLE_INTERVAL_SECONDS,
        "required_consecutive_exact_idle_samples": GPU_QUIESCENCE_REQUIRED_CONSECUTIVE_EXACT_IDLE_SAMPLES,
        "operational_timeout_basis": GPU_QUIESCENCE_OPERATIONAL_TIMEOUT_BASIS,
        "samples": list(samples),
        "final_gpu": None if final_gpu is None else dict(final_gpu),
        "failure_reason": failure_reason,
    })
    _write(work / (phase.replace("_", "-") + "-gpu-quiescence.json"), value)
    return value


def require_gpu_quiescence(
    approved_gpu_uuids: Sequence[str],
    *,
    work: Path,
    phase: str,
) -> dict[str, Any]:
    """Require a bounded, evidence-preserving exact-idle quiescence window."""

    _approved_gpu_uuid_values(approved_gpu_uuids)
    started_at, started_monotonic = _now(), time.monotonic()
    deadline = started_monotonic + GPU_QUIESCENCE_TIMEOUT_SECONDS
    samples: list[dict[str, Any]] = []
    consecutive = 0
    while True:
        try:
            probe = _probe_approved_gpus(approved_gpu_uuids)
        except Stage1S18FormalError as error:
            # A malformed/mismatched query cannot supply a complete inventory;
            # preserve the failed probe attempt and stop immediately.
            sample = _gpu_quiescence_sample(
                sample_index=len(samples), started_monotonic=started_monotonic, observed_monotonic=time.monotonic(),
                consecutive_exact_idle_samples=0, probe=None,
                violations=[str(error).split(":", 1)[0]],
            )
            samples.append(sample)
            _write_gpu_quiescence_record(work=work, phase=phase, status="FAILED", started_at=started_at, samples=samples, final_gpu=None, failure_reason=str(error).split(":", 1)[0])
            raise
        try:
            immediate = _gpu_quiescence_immediate_violations(probe)
            violations = _gpu_idle_violations(probe)
        except Stage1S18FormalError as error:
            sample = _gpu_quiescence_sample(
                sample_index=len(samples), started_monotonic=started_monotonic, observed_monotonic=time.monotonic(),
                consecutive_exact_idle_samples=0, probe=probe,
                violations=[str(error).split(":", 1)[0]],
            )
            samples.append(sample)
            _write_gpu_quiescence_record(work=work, phase=phase, status="FAILED", started_at=started_at, samples=samples, final_gpu=None, failure_reason=str(error).split(":", 1)[0])
            raise
        if immediate:
            sample = _gpu_quiescence_sample(
                sample_index=len(samples), started_monotonic=started_monotonic, observed_monotonic=time.monotonic(),
                consecutive_exact_idle_samples=0, probe=probe, violations=immediate,
            )
            samples.append(sample)
            _write_gpu_quiescence_record(work=work, phase=phase, status="FAILED", started_at=started_at, samples=samples, final_gpu=None, failure_reason=immediate[0])
            raise Stage1S18FormalError(immediate[0])
        if violations:
            consecutive = 0
        else:
            consecutive += 1
        completed_monotonic = time.monotonic()
        sample = _gpu_quiescence_sample(
            sample_index=len(samples), started_monotonic=started_monotonic, observed_monotonic=completed_monotonic,
            consecutive_exact_idle_samples=consecutive, probe=probe, violations=violations,
        )
        samples.append(sample)
        # A sample is admissible only if its complete nvidia-smi observation
        # finished by the frozen deadline.  Equality is allowed; once the
        # boundary is reached without the third sample, no further sample can
        # be admitted.
        if completed_monotonic > deadline:
            marker = "S18_GPU_QUIESCENCE_TIMEOUT"
            _write_gpu_quiescence_record(work=work, phase=phase, status="FAILED", started_at=started_at, samples=samples, final_gpu=None, failure_reason=marker)
            raise Stage1S18FormalError(marker)
        if consecutive == GPU_QUIESCENCE_REQUIRED_CONSECUTIVE_EXACT_IDLE_SAMPLES:
            return _write_gpu_quiescence_record(work=work, phase=phase, status="PASS", started_at=started_at, samples=samples, final_gpu=probe, failure_reason=None)
        if completed_monotonic >= deadline:
            marker = "S18_GPU_QUIESCENCE_TIMEOUT"
            _write_gpu_quiescence_record(work=work, phase=phase, status="FAILED", started_at=started_at, samples=samples, final_gpu=None, failure_reason=marker)
            raise Stage1S18FormalError(marker)
        time.sleep(min(GPU_QUIESCENCE_SAMPLE_INTERVAL_SECONDS, deadline - completed_monotonic))


def _parse_gpu_recovery_actions(output: str, *, expected_uuids: Sequence[str]) -> dict[str, str]:
    """Parse the structured ``nvidia-smi`` recovery-action query exactly."""

    expected = tuple(expected_uuids)
    if not expected or len(expected) != len(set(expected)) or any(GPU_UUID_RE.fullmatch(value) is None for value in expected):
        raise Stage1S18FormalError("S18_GPU_RECOVERY_ACTION_EXPECTED_UUIDS_INVALID")
    values: dict[str, str] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or GPU_UUID_RE.fullmatch(fields[0]) is None or not fields[1]:
            raise Stage1S18FormalError("S18_GPU_RECOVERY_ACTION_PARSE_INVALID")
        uuid, action = fields
        if uuid in values:
            raise Stage1S18FormalError("S18_GPU_RECOVERY_ACTION_UUID_DUPLICATE:" + uuid)
        values[uuid] = action
    selected = {uuid: values[uuid] for uuid in expected if uuid in values}
    if set(selected) != set(expected):
        raise Stage1S18FormalError("S18_GPU_RECOVERY_ACTION_UUID_MISMATCH")
    return selected


def _process_identity(pid: int) -> dict[str, Any]:
    """Read stable process identity without making an inheritance claim."""

    stat, cmdline, executable = Path(f"/proc/{pid}/stat"), Path(f"/proc/{pid}/cmdline"), Path(f"/proc/{pid}/exe")
    if not stat.is_file() or not cmdline.is_file() or not executable.exists():
        raise ProcessLookupError(pid)
    parts = stat.read_text(encoding="utf-8").split()
    if len(parts) <= 21:
        raise ProcessLookupError(pid)
    return {"pid": pid, "ppid": int(parts[3]), "uid": stat.stat().st_uid, "pgid": os.getpgid(pid), "sid": os.getsid(pid), "start_ticks": parts[21], "exe": os.readlink(executable), "cmdline_sha256": hashlib.sha256(cmdline.read_bytes()).hexdigest()}


def _parse_session_member_stat(raw: str, *, pid: int, uid: int) -> dict[str, Any]:
    """Parse `/proc/<pid>/stat` without treating its arbitrary comm as fields."""

    opening, closing = raw.find("("), raw.rfind(")")
    if opening <= 0 or closing <= opening or raw[:opening].strip() != str(pid):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_MEMBER_STATE_UNVERIFIABLE")
    fields = raw[closing + 1:].split()
    # After the parenthesized comm, indexes 0/2/3/19 are Linux stat fields
    # 3(state), 5(pgrp), 6(session), and 22(starttime), respectively.
    if len(fields) <= 19 or len(fields[0]) != 1 or not fields[2].isdigit() or not fields[3].isdigit() or not fields[19].isdigit():
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_MEMBER_STATE_UNVERIFIABLE")
    return {"pid": pid, "uid": uid, "pgid": int(fields[2]), "sid": int(fields[3]), "state": fields[0], "start_ticks": fields[19]}


def _session_member_stat(pid: int) -> dict[str, Any]:
    """Read the minimum race-safe state needed to classify a session member.

    A SID can retain a zombie after its environment/executable is no longer
    readable.  A fully parsed ``Z`` is excluded only for a known exact member.
    The caller separately recognizes the frozen-host's tightly bounded
    procfs-owner transition (the stat inode UID alone becomes zero) for an
    already token-bound exiting task.  Malformed records are ownership
    failures, while an unavailable record is reported distinctly so the caller
    can re-audit one already-known PID.
    """

    try:
        with Path(f"/proc/{pid}/stat").open("r", encoding="utf-8") as handle:
            raw, uid = handle.read(), os.fstat(handle.fileno()).st_uid
    except (OSError, UnicodeDecodeError) as error:
        raise _SessionMemberStatUnavailable(pid) from error
    return _parse_session_member_stat(raw, pid=pid, uid=uid)


def _same_session_stat_identity(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    """Compare stat-only fields that prove a known PID was not reused."""

    return all(expected.get(field) == observed.get(field) for field in ("pid", "uid", "pgid", "sid", "start_ticks"))


_SESSION_STAT_IDENTITY_FIELDS = ("pid", "uid", "pgid", "sid", "start_ticks")


def _session_stat_identity_differences(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, dict[str, object]]:
    """Return stable, no-command-text evidence for a stat identity drift."""

    return {
        field: {"expected": expected.get(field), "observed": observed.get(field)}
        for field in _SESSION_STAT_IDENTITY_FIELDS
        if expected.get(field) != observed.get(field)
    }


def _session_stat_identity_drift_error(member_pid: int, expected: Mapping[str, Any], observed: Mapping[str, Any]) -> Stage1S18ManualInterventionRequired:
    """Make state and changed fields visible without broadening ownership."""

    differences = _session_stat_identity_differences(expected, observed)
    state = observed.get("state")
    fields = ",".join(differences) if differences else "none"
    return Stage1S18ManualInterventionRequired(
        f"S18_PROCESS_SESSION_STAT_IDENTITY_DRIFT:pid={member_pid}:state={state}:fields={fields}"
    )


def _is_known_procfs_owner_exit_transition(
    *, expected: Mapping[str, Any], earlier: Mapping[str, Any], member_stat: Mapping[str, Any],
) -> bool:
    """Recognize the observed Linux exit-owner UID transition, and nothing else.

    A bounded frozen-host probe observed an exiting task's ``/proc/<pid>/stat``
    inode owner transition from the launch UID to root while PID, PGID, SID and
    start ticks remained exact (states ``R``/``Z`` only).  This proves neither
    a new identity nor ownership by itself; it only lets an *already
    token-bound* member take the narrowly defined exit path below.
    """

    differences = _session_stat_identity_differences(earlier, member_stat)
    prior_uid, expected_uid, observed_uid = earlier.get("uid"), expected.get("uid"), member_stat.get("uid")
    return (
        earlier.get("environment_run_token") == expected.get("environment_run_token")
        and isinstance(prior_uid, int)
        and not isinstance(prior_uid, bool)
        and isinstance(expected_uid, int)
        and not isinstance(expected_uid, bool)
        and prior_uid != 0
        and prior_uid == expected_uid
        and expected_uid != 0
        and observed_uid == 0
        and member_stat.get("state") in {"R", "Z"}
        and set(differences) == {"uid"}
    )


def _same_process_identity(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    """Compare a pre-token provisional identity without claiming token proof."""

    return all(expected.get(field) == observed.get(field) for field in ("pid", "uid", "pgid", "sid", "start_ticks", "exe", "cmdline_sha256"))


def _inherited_run_token(pid: int, run_token: str) -> str:
    """Require an exact S1.8 token inherited at child process creation."""

    environment = _process_environment(pid)
    token_values = [item.split(b"=", 1)[1].decode("ascii") for item in environment if item.startswith(b"PARAM_IMPORTANCE_S18_RUN_TOKEN=")]
    if token_values != [run_token]:
        raise ProcessLookupError(pid)
    return token_values[0]


def _process_environment(pid: int) -> list[bytes]:
    """Return raw environment entries for strict spawned-child attestation."""

    return Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")


def _fingerprint(pid: int, run_token: str) -> dict[str, Any]:
    """Fingerprint a spawned launcher/worker that inherited ``run_token``."""

    identity = _process_identity(pid)
    identity["environment_run_token"] = _inherited_run_token(pid, run_token)
    return identity


def _parent_fingerprint(pid: int, planned_run_token: str) -> dict[str, Any]:
    """Fingerprint the pre-existing parent without inventing token inheritance."""

    if _require_sha256(planned_run_token, field="parent.planned_run_token") != planned_run_token:
        raise Stage1S18FormalError("S18_PARENT_RUN_TOKEN_INVALID")
    identity = _process_identity(pid)
    return {**identity, "planned_run_token": planned_run_token, "token_inherited_at_exec": False}


def _session_members(sid: int) -> list[int]:
    """Enumerate the launch session without retaining process command text."""

    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getsid(int(entry.name)) == sid:
                members.append(int(entry.name))
        except ProcessLookupError:
            continue
    return sorted(members)


def _token_process_ids(run_token: str) -> list[int]:
    """Find only exact-token processes; all same-session members are audited separately."""

    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            _inherited_run_token(int(entry.name), run_token)
        except (OSError, ProcessLookupError, UnicodeDecodeError):
            continue
        result.append(int(entry.name))
    return sorted(result)


_FINGERPRINT_STABLE_FIELDS = (
    "pid", "uid", "pgid", "sid", "start_ticks", "exe", "cmdline_sha256", "environment_run_token",
)


def _same_live_fingerprint(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    """Compare fields that cannot legitimately change after a worker execs.

    PPID is deliberately excluded: elastic workers can be re-parented after
    the torchrun launcher exits.  The initial PPID is retained separately as
    historical ancestry evidence, while pidfd + start ticks protect signals
    from PID reuse.
    """

    return all(expected.get(field) == observed.get(field) for field in _FINGERPRINT_STABLE_FIELDS)


def _tree_depths(
    *, expected: Mapping[str, Any], observed: Mapping[int, Mapping[str, Any]], known_members: Mapping[int, Mapping[str, Any]],
) -> dict[int, int]:
    """Prove each token process was launched below the recorded launcher.

    A member's first captured parent is authoritative historical evidence.
    Thus a worker that has since been re-parented stays attributable, whereas
    an uncaptured orphan (or a newly injected token process) is rejected.
    """

    launcher_pid = expected.get("pid")
    if not isinstance(launcher_pid, int):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_AUDIT_EXPECTED_INVALID")
    ancestry: dict[int, Mapping[str, Any]] = {int(pid): value for pid, value in known_members.items() if isinstance(pid, int)}
    ancestry.update({int(pid): value for pid, value in observed.items()})
    # Preserve first-observed PPID for established workers rather than a
    # potentially re-parented value from this polling pass.
    ancestry.update({int(pid): value for pid, value in known_members.items() if isinstance(pid, int)})
    depths: dict[int, int] = {}
    for origin in observed:
        current, depth, seen = origin, 0, set()
        while current != launcher_pid:
            if current in seen:
                raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_ANCESTRY_CYCLE")
            seen.add(current)
            item = ancestry.get(current)
            parent = item.get("ppid") if isinstance(item, Mapping) else None
            if not isinstance(parent, int) or parent not in ancestry:
                raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_PARENT_DRIFT")
            current, depth = parent, depth + 1
        depths[origin] = depth
    return depths


def _provisional_session_member(
    *, member_pid: int, member_stat: Mapping[str, Any], expected: Mapping[str, Any],
    observed: Mapping[int, Mapping[str, Any]], known_members: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove a new session PID is a launcher descendant before one token retry.

    This is deliberately not an ownership attestation: the returned identity
    is not put into ``known_members`` and is never eligible for signalling.
    The next audit must discover the same PID through its inherited run token.
    """

    try:
        candidate = _process_identity(member_pid)
    except (OSError, ProcessLookupError):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_CANDIDATE_UNVERIFIABLE") from None
    expected_uid, expected_start = expected.get("uid"), expected.get("start_ticks")
    start = candidate.get("start_ticks")
    if candidate.get("uid") != expected_uid or not isinstance(expected_start, str) or not expected_start.isdigit() or not isinstance(start, str) or not start.isdigit() or int(start) < int(expected_start) or not _same_session_stat_identity(candidate, member_stat):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_CANDIDATE_IDENTITY_DRIFT")
    try:
        _tree_depths(expected=expected, observed={**observed, member_pid: candidate}, known_members=known_members)
    except Stage1S18ManualInterventionRequired as error:
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_CANDIDATE_ANCESTRY_DRIFT") from error
    return candidate


def _require_exact_attested_launcher(
    fingerprint: Mapping[str, Any], known_members: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Require the originally recorded launcher identity, including its token."""

    expected_pid = fingerprint.get("pid")
    attested = known_members.get(expected_pid) if isinstance(expected_pid, int) else None
    if not isinstance(expected_pid, int) or not isinstance(attested, Mapping) or dict(attested) != dict(fingerprint):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_LAUNCHER_EXIT_UNATTESTED")
    return attested


def _token_missing_launcher_owner_exit_candidate(
    expected: Mapping[str, Any], *, known_members: Mapping[int, Mapping[str, Any]], token_ids: Sequence[int],
) -> _LauncherOwnerExitCandidate | None:
    """Classify only the frozen-host's exact owner-exit state after token loss.

    A missing launcher token ordinarily means that live ownership can no
    longer be proven.  The lone exception is the frozen Linux observation
    where an already-attested launcher keeps its PID/PGID/SID/start ticks but
    procfs changes only its stat inode UID to zero while it is exiting.  No
    other token or same-session process may coexist with that exception.
    """

    expected_pid, expected_sid = expected.get("pid"), expected.get("sid")
    if not isinstance(expected_pid, int) or not isinstance(expected_sid, int):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_AUDIT_EXPECTED_INVALID")
    if token_ids:
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_LAUNCHER_OWNER_EXIT_TOKEN_MEMBERS_PRESENT")
    attested = _require_exact_attested_launcher(expected, known_members)
    try:
        session_members = _session_members(expected_sid)
    except OSError:
        # The expected PID may have been reaped between the token and session
        # scans; retain the existing one-second Popen confirmation path.
        return None
    if expected_pid not in session_members:
        return None
    if session_members != [expected_pid]:
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_LAUNCHER_OWNER_EXIT_SESSION_MEMBERS")
    try:
        member_stat = _session_member_stat(expected_pid)
    except _SessionMemberStatUnavailable:
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_LAUNCHER_OWNER_EXIT_STATE_UNVERIFIABLE") from None
    if not _is_known_procfs_owner_exit_transition(
        expected=expected,
        earlier=attested,
        member_stat=member_stat,
    ):
        raise Stage1S18ManualInterventionRequired(
            "S18_PROCESS_LAUNCHER_TOKEN_MISSING_LIVE_OR_IDENTITY_DRIFT"
        )
    return _LauncherOwnerExitCandidate(
        "S18_PROCESS_LAUNCHER_PROCFS_OWNER_EXIT_TOKEN_MISSING:"
        f"pid={expected_pid}:state={member_stat['state']}:fields=uid"
    )


def _audit_exact_process_group(
    expected: Mapping[str, Any], *, known_members: Mapping[int, Mapping[str, Any]] | None = None,
    _session_membership_rechecked: bool = False,
    _provisional_members: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every live token process belongs to this launch tree.

    ``torch.distributed.elastic`` starts workers in *new sessions*.  Therefore
    SID/PGID equality with the launcher is not an ownership test.  Ownership
    is the exact inherited token, same UID, post-launch start time, and the
    launcher-recorded PPID ancestry captured while the launcher is alive.  We
    nevertheless inspect every live member of every involved SID and reject a
    session containing an unrelated process.  A fully parsed zombie is
    excluded only if it was already fully token-bound and its stat identity
    either exactly matches that first fingerprint or exhibits the frozen
    UID-only procfs owner exit transition.  A newly visible direct descendant
    gets one provisional retry only after stat/UID/ancestry checks; it cannot
    enter the known tree or be signalled until exact-token discovery on that
    retry.  A never-token PID, including a zombie, is never promoted.
    """

    pgid, sid, token = expected.get("pgid"), expected.get("sid"), expected.get("environment_run_token")
    expected_pid, expected_uid, expected_start = expected.get("pid"), expected.get("uid"), expected.get("start_ticks")
    if not isinstance(pgid, int) or not isinstance(sid, int) or not isinstance(token, str) or not isinstance(expected_pid, int) or not isinstance(expected_uid, int) or not isinstance(expected_start, str) or not expected_start.isdigit():
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_AUDIT_EXPECTED_INVALID")
    previous = dict(known_members or {})
    provisional = {int(pid): dict(member) for pid, member in (dict(_provisional_members or {})).items() if isinstance(pid, int) and isinstance(member, Mapping)}
    if len(provisional) != len(_provisional_members or {}):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_PROVISIONAL_INVALID")
    token_ids = _token_process_ids(token)
    if expected_pid not in token_ids:
        owner_exit_candidate = _token_missing_launcher_owner_exit_candidate(
            expected,
            known_members=previous,
            token_ids=token_ids,
        )
        if owner_exit_candidate is not None:
            raise owner_exit_candidate
        raise ProcessLookupError(expected_pid)
    if _session_membership_rechecked and not set(provisional).issubset(token_ids):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_PROVISIONAL_TOKEN_MISSING")
    observed: dict[int, dict[str, Any]] = {}
    for pid in token_ids:
        try:
            fingerprint = _fingerprint(pid, token)
        except (OSError, ProcessLookupError):
            raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_MEMBER_UNVERIFIABLE") from None
        start_ticks = fingerprint.get("start_ticks")
        if fingerprint["uid"] != expected_uid or not isinstance(start_ticks, str) or not start_ticks.isdigit() or int(start_ticks) < int(expected_start):
            raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_MEMBER_IDENTITY_DRIFT")
        earlier = previous.get(pid)
        if earlier is not None and not _same_live_fingerprint(earlier, fingerprint):
            raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_MEMBER_IDENTITY_DRIFT")
        provisional_member = provisional.get(pid)
        if provisional_member is not None and not _same_process_identity(provisional_member, fingerprint):
            raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_PROVISIONAL_IDENTITY_DRIFT")
        observed[pid] = fingerprint
    launcher = observed.get(expected_pid)
    if launcher is None or not _same_live_fingerprint(expected, launcher):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_LAUNCHER_FINGERPRINT_DRIFT")
    depths = _tree_depths(expected=expected, observed=observed, known_members=previous)
    newly_provisional: dict[int, dict[str, Any]] = {}
    launcher_exit_candidate: str | None = None
    for member_sid in sorted({int(item["sid"]) for item in observed.values()}):
        for member_pid in _session_members(member_sid):
            earlier = previous.get(member_pid)
            try:
                member_stat = _session_member_stat(member_pid)
            except _SessionMemberStatUnavailable:
                if isinstance(earlier, Mapping) and not _session_membership_rechecked:
                    time.sleep(SESSION_MEMBER_REVALIDATION_SECONDS)
                    return _audit_exact_process_group(
                        expected,
                        known_members=previous,
                        _session_membership_rechecked=True,
                    )
                raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_MEMBER_STATE_UNVERIFIABLE") from None
            stat_identity_matches = isinstance(earlier, Mapping) and _same_session_stat_identity(earlier, member_stat)
            owner_exit_transition = isinstance(earlier, Mapping) and _is_known_procfs_owner_exit_transition(
                expected=expected,
                earlier=earlier,
                member_stat=member_stat,
            )
            if member_stat["state"] == "Z":
                # Linux can retain an unexecutable zombie whose procfs stat
                # inode owner has transitioned to root.  This is safe only
                # for a member attested in an earlier pass, with exactly the
                # frozen UID-only transition.  Unknown/provisional zombies
                # remain fail-closed regardless of how quickly they vanish.
                if not isinstance(earlier, Mapping):
                    raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_TOKEN_MEMBERSHIP_DRIFT")
                if stat_identity_matches or owner_exit_transition:
                    continue
                raise _session_stat_identity_drift_error(member_pid, earlier, member_stat)
            observed_member = observed.get(member_pid)
            if (
                not isinstance(earlier, Mapping)
                and isinstance(observed_member, Mapping)
                and not _same_session_stat_identity(observed_member, member_stat)
            ):
                # Exact-token discovery does not make a contradictory procfs
                # identity safe.  In particular, a UID-zero transition is an
                # exit exception only for a member attested in an earlier
                # complete pass; it can never promote a new/provisional PID.
                raise _session_stat_identity_drift_error(member_pid, observed_member, member_stat)
            # Check the minimal procfs identity before a fresh environ read:
            # otherwise a readable token could mask PID reuse or an owner
            # transition that must be confirmed as an exit, never live work.
            if not isinstance(earlier, Mapping):
                stat_identity_matches = False
            if isinstance(earlier, Mapping) and not stat_identity_matches:
                if owner_exit_transition:
                    if member_pid == expected_pid and _same_live_fingerprint(expected, earlier):
                        launcher_exit_candidate = (
                            "S18_PROCESS_LAUNCHER_PROCFS_OWNER_EXIT_CANDIDATE:"
                            f"pid={member_pid}:state={member_stat['state']}:fields=uid"
                        )
                        continue
                    # A non-zombie worker can pass through the same kernel
                    # owner transition.  It gets exactly one complete audit:
                    # recovery, disappearance, or a known zombie is okay;
                    # remaining live UID-zero state is not.
                    if not _session_membership_rechecked:
                        time.sleep(SESSION_MEMBER_REVALIDATION_SECONDS)
                        return _audit_exact_process_group(
                            expected,
                            known_members=previous,
                            _session_membership_rechecked=True,
                        )
                    raise Stage1S18ManualInterventionRequired(
                        "S18_PROCESS_SESSION_WORKER_OWNER_EXIT_UNRESOLVED:"
                        f"pid={member_pid}:state={member_stat['state']}"
                    )
                raise _session_stat_identity_drift_error(member_pid, earlier, member_stat)
            try:
                session_member = _fingerprint(member_pid, token)
            except (OSError, ProcessLookupError):
                session_member = None
            if session_member is None or member_pid not in observed or not _same_live_fingerprint(observed[member_pid], session_member):
                if not isinstance(earlier, Mapping):
                    if _session_membership_rechecked:
                        raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_TOKEN_MEMBERSHIP_DRIFT")
                    candidate = _provisional_session_member(
                        member_pid=member_pid,
                        member_stat=member_stat,
                        expected=expected,
                        observed=observed,
                        known_members=previous,
                    )
                    # Audit every newly seen member in this snapshot before
                    # the single bounded retry.  Otherwise a second worker in
                    # the same elastic session could evade the provisional
                    # token-recovery requirement merely because iteration
                    # reached its sibling first.
                    newly_provisional[member_pid] = candidate
                    continue
                if session_member is not None and not _same_live_fingerprint(earlier, session_member):
                    raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_TOKEN_MEMBERSHIP_DRIFT")
                # A successful launcher can exit between the token directory
                # scan and this direct environment read.  This exception is
                # deliberately unavailable to workers and to any identity
                # drift: the caller must still confirm the Popen launcher has
                # actually exited, then run the usual empty-residual check.
                if (
                    member_pid == expected_pid
                    and session_member is None
                    and _same_live_fingerprint(expected, earlier)
                ):
                    # Finish auditing every other member first.  A launcher
                    # exit can never excuse a foreign/unknown sibling in its
                    # session from the normal fail-closed checks.
                    launcher_exit_candidate = "S18_PROCESS_LAUNCHER_NATURAL_EXIT_CANDIDATE"
                    continue
                if not _session_membership_rechecked:
                    # Only a previously attested PID reaches this retry.  The
                    # retry starts from token discovery and redoes
                    # UID/start/ancestry checks; no session-only PID is added.
                    time.sleep(SESSION_MEMBER_REVALIDATION_SECONDS)
                    return _audit_exact_process_group(
                        expected,
                        known_members=previous,
                        _session_membership_rechecked=True,
                    )
                raise Stage1S18ManualInterventionRequired("S18_PROCESS_SESSION_TOKEN_MEMBERSHIP_DRIFT")
    if newly_provisional:
        time.sleep(SESSION_MEMBER_REVALIDATION_SECONDS)
        return _audit_exact_process_group(
            expected,
            known_members=previous,
            _session_membership_rechecked=True,
            _provisional_members={**provisional, **newly_provisional},
        )
    if launcher_exit_candidate is not None:
        if not _session_membership_rechecked:
            time.sleep(SESSION_MEMBER_REVALIDATION_SECONDS)
            return _audit_exact_process_group(
                expected,
                known_members=previous,
                _session_membership_rechecked=True,
            )
        raise _LauncherNaturalExitCandidate(launcher_exit_candidate)
    members = [observed[pid] for pid in sorted(observed)]
    return {
        "pgid": pgid, "sid": sid, "members": members, "member_pids": sorted(observed),
        "ancestry_depths": {str(pid): depths[pid] for pid in sorted(depths)},
    }


def _manual_intervention(work: Path, label: str, expected: Mapping[str, Any], *, reason: str, observed: object) -> None:
    _write(work / f"{label}-manual-intervention.json", _with_hash({"schema_version": "stage1-s1-8-manual-intervention-v1", "status": "BLOCKED", "reason": reason, "expected_fingerprint": dict(expected), "observed": observed, "action": "NO_SIGNAL_NO_LEASE_RELEASE"}))


def _tree_signal_order(members: Sequence[Mapping[str, Any]], ancestry_depths: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return children before parents, rejecting malformed or cyclic ancestry."""

    by_pid = {member.get("pid"): member for member in members if isinstance(member.get("pid"), int)}
    if len(by_pid) != len(members) or not by_pid:
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_SIGNAL_AUDIT_INVALID")
    if set(ancestry_depths) != {str(pid) for pid in by_pid} or any(not isinstance(value, int) or value < 0 for value in ancestry_depths.values()):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_SIGNAL_AUDIT_INVALID")
    return sorted(members, key=lambda member: (-int(ancestry_depths[str(member["pid"])]), int(member["pid"])))


def _signal_exact_member(member: Mapping[str, Any], signal_value: int) -> bool:
    """Signal one freshly re-attested process through a pidfd, never a raw PID.

    ``False`` means it naturally exited between audit and signalling.  A pidfd
    pins the kernel task identity, while re-reading the exact run token and
    start identity before and after opening it makes PID reuse fail closed.
    """

    pid, token = member.get("pid"), member.get("environment_run_token")
    if not isinstance(pid, int) or not isinstance(token, str):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_SIGNAL_MEMBER_INVALID")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_PIDFD_UNAVAILABLE")
    try:
        if not _same_live_fingerprint(member, _fingerprint(pid, token)):
            raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_SIGNAL_IDENTITY_DRIFT")
        descriptor = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        return False
    except OSError as error:
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_PIDFD_OPEN_FAILED") from error
    try:
        try:
            if not _same_live_fingerprint(member, _fingerprint(pid, token)):
                raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_SIGNAL_IDENTITY_DRIFT")
            signal.pidfd_send_signal(descriptor, signal_value, None, 0)
        except ProcessLookupError:
            return False
        except OSError as error:
            raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_PIDFD_SIGNAL_FAILED") from error
    finally:
        os.close(descriptor)
    return True


def _signal_exact_tree(audit: Mapping[str, Any], signal_value: int) -> list[int]:
    """Signal only freshly re-attested launch-tree members, child before parent."""

    members, depths = audit.get("members"), audit.get("ancestry_depths")
    if not isinstance(members, list) or not isinstance(depths, Mapping) or any(not isinstance(member, Mapping) for member in members):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_SIGNAL_AUDIT_INVALID")
    signalled: list[int] = []
    for member in _tree_signal_order([dict(member) for member in members], depths):
        if _signal_exact_member(member, signal_value):
            signalled.append(int(member["pid"]))
    return signalled


def _residual_launch_tree(expected: Mapping[str, Any], *, known_members: Mapping[int, Mapping[str, Any]] | None = None) -> dict[str, list[int]]:
    """Return token residuals and every session ever attested for this launch."""

    token = expected.get("environment_run_token")
    if not isinstance(token, str):
        return {"session_members": [], "token_members": []}
    token_members = _token_process_ids(token)
    session_members: set[int] = set()
    session_ids: set[int] = set()
    for pid in token_members:
        try:
            session_ids.add(os.getsid(pid))
        except ProcessLookupError:
            continue
    for member in (known_members or {}).values():
        if isinstance(member, Mapping) and isinstance(member.get("sid"), int):
            session_ids.add(int(member["sid"]))
    for member_sid in session_ids:
        session_members.update(_session_members(member_sid))
    return {"session_members": sorted(session_members), "token_members": token_members}


def _known_launch_tree(expected: Mapping[str, Any], known_members: Mapping[int, Mapping[str, Any]], ancestry_depths: Mapping[int, int]) -> dict[str, Any]:
    """Serialize the first-seen, token-bound process tree for reproduction."""

    pgid, sid = expected.get("pgid"), expected.get("sid")
    if not isinstance(pgid, int) or not isinstance(sid, int) or not known_members:
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_KNOWN_INVALID")
    members = [dict(known_members[pid]) for pid in sorted(known_members)]
    if set(ancestry_depths) != set(known_members):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_KNOWN_ANCESTRY_INVALID")
    return {
        "pgid": pgid, "sid": sid, "members": members, "member_pids": sorted(known_members),
        "ancestry_depths": {str(pid): int(ancestry_depths[pid]) for pid in sorted(ancestry_depths)},
    }


def _terminate_exact(process: subprocess.Popen[str], expected: Mapping[str, Any], work: Path, *, label: str, known_members: Mapping[int, Mapping[str, Any]] | None = None) -> None:
    try:
        audit = _audit_exact_process_group(expected, known_members=known_members)
    except Stage1S18ManualInterventionRequired as error:
        _manual_intervention(work, label, expected, reason=str(error), observed=_residual_launch_tree(expected, known_members=known_members))
        raise
    except ProcessLookupError:
        survivors = _residual_launch_tree(expected, known_members=known_members)
        if survivors["session_members"] or survivors["token_members"]:
            _manual_intervention(work, label, expected, reason="S18_LAUNCHER_GONE_PROCESS_TREE_REMAINS", observed=survivors)
            raise Stage1S18ManualInterventionRequired("S18_LAUNCHER_GONE_PROCESS_TREE_REMAINS")
        return
    _write(work / f"{label}-termination-audit.json", _with_hash({"schema_version": "stage1-s1-8-process-tree-audit-v1", "status": "PASS", "phase": "pre_signal", "expected_launcher": dict(expected), "observed_tree": audit}))
    _signal_exact_tree(audit, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            audit = _audit_exact_process_group(expected, known_members=known_members)
        except Stage1S18ManualInterventionRequired as error:
            _manual_intervention(work, label, expected, reason=str(error), observed=_residual_launch_tree(expected, known_members=known_members))
            raise
        _signal_exact_tree(audit, signal.SIGKILL)
        process.wait(timeout=30)
    residual = _residual_launch_tree(expected, known_members=known_members)
    if residual["session_members"] or residual["token_members"]:
        _manual_intervention(work, label, expected, reason="S18_PROCESS_TREE_RESIDUAL_AFTER_TERMINATION", observed=residual)
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_TREE_RESIDUAL_AFTER_TERMINATION")


def _reserve_loopback_rendezvous_endpoint() -> tuple[socket.socket, str]:
    """Reserve one fresh non-zero loopback port for the immediate torchrun exec.

    The reservation socket is held until the Popen boundary, is never reused
    by this formal attempt, and a collision after handoff is a launch failure
    rather than a retry on a possibly different endpoint.
    """

    for _ in range(32):
        reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            reservation.bind(("127.0.0.1", 0))
            port = int(reservation.getsockname()[1])
            if port == 0 or port in _USED_LOOPBACK_RENDEZVOUS_PORTS:
                reservation.close()
                continue
            _USED_LOOPBACK_RENDEZVOUS_PORTS.add(port)
            return reservation, f"127.0.0.1:{port}"
        except BaseException:
            reservation.close()
            raise
    raise Stage1S18FormalError("S18_RENDEZVOUS_ENDPOINT_ALLOCATION_EXHAUSTED")


def _prepare_rendezvous_command(command: Sequence[str], *, run_token: str | None = None) -> tuple[list[str], socket.socket | None, str | None]:
    """Replace only the frozen torchrun ``127.0.0.1:0`` placeholder.

    A distributed command must carry the launch token as its rendezvous ID so
    the process outcome has one immutable identity across torchrun and workers.
    """

    values = list(command)
    is_torchrun = "torch.distributed.run" in values
    endpoint_positions = [index for index, value in enumerate(values) if value == "--rdzv-endpoint"]
    if not is_torchrun:
        if endpoint_positions:
            raise Stage1S18FormalError("S18_NON_TORCHRUN_RENDEZVOUS_ARGUMENT")
        return values, None, None
    if len(endpoint_positions) != 1 or endpoint_positions[0] + 1 >= len(values) or values[endpoint_positions[0] + 1] != "127.0.0.1:0":
        raise Stage1S18FormalError("S18_TORCHRUN_RENDEZVOUS_PLACEHOLDER_INVALID")
    id_positions = [index for index, value in enumerate(values) if value == "--rdzv-id"]
    if run_token is None or len(id_positions) != 1 or id_positions[0] + 1 >= len(values) or values[id_positions[0] + 1] != run_token:
        raise Stage1S18FormalError("S18_TORCHRUN_RENDEZVOUS_ID_INVALID")
    reservation, endpoint = _reserve_loopback_rendezvous_endpoint()
    values[endpoint_positions[0] + 1] = endpoint
    return values, reservation, endpoint


def _is_nonzero_loopback_endpoint(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("127.0.0.1:"):
        return False
    try:
        port = int(value.rsplit(":", 1)[1])
    except ValueError:
        return False
    return 1 <= port <= 65535


def _confirm_attested_launcher_exit(
    process: subprocess.Popen[str],
    fingerprint: Mapping[str, Any],
    known_members: Mapping[int, Mapping[str, Any]],
    *,
    reason: str,
) -> None:
    """Confirm an already-attested launcher exit without weakening ownership.

    The poll loop may sample ``/proc`` after a short-lived launcher has gone
    away but before its ``Popen`` object has been observed as exited.  That is
    acceptable only after the exact launcher identity (including its inherited
    run token) was recorded in the initial tree.  A bounded ``wait`` freezes
    that decision, then an immediate residual scan proves that no token or
    session member survived the launcher.
    """

    _require_exact_attested_launcher(fingerprint, known_members)
    try:
        returncode = process.wait(timeout=LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise Stage1S18ManualInterventionRequired(
            "S18_PROCESS_LAUNCHER_NATURAL_EXIT_UNCONFIRMED:" + reason
        ) from error
    except OSError as error:
        raise Stage1S18ManualInterventionRequired(
            "S18_PROCESS_LAUNCHER_NATURAL_EXIT_WAIT_FAILED:" + reason
        ) from error
    if not isinstance(returncode, int):
        raise Stage1S18ManualInterventionRequired(
            "S18_PROCESS_LAUNCHER_NATURAL_EXIT_UNCONFIRMED:" + reason
        )
    residual = _residual_launch_tree(fingerprint, known_members=known_members)
    if residual["session_members"] or residual["token_members"]:
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_LAUNCHER_EXIT_RESIDUAL")


def _confirm_attested_launcher_owner_exit(
    process: subprocess.Popen[str],
    fingerprint: Mapping[str, Any],
    known_members: Mapping[int, Mapping[str, Any]],
    *,
    reason: str,
) -> None:
    """Join only a revalidated frozen-host owner-exit transition.

    This reuses the normal post-poll terminal join bound, but never turns it
    into a general token-loss timeout.  Each one-second incomplete wait must
    still show the same exact UID-only R/Z procfs transition; normal live
    token-loss, a sibling, or every identity drift remains Manual.
    """

    _require_exact_attested_launcher(fingerprint, known_members)
    run_token = fingerprint.get("environment_run_token")
    if not isinstance(run_token, str):
        raise Stage1S18ManualInterventionRequired("S18_PROCESS_AUDIT_EXPECTED_INVALID")
    deadline = time.monotonic() + TERMINAL_PROCESS_JOIN_TIMEOUT_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Stage1S18ManualInterventionRequired(
                "S18_PROCESS_LAUNCHER_OWNER_EXIT_UNCONFIRMED:" + reason
            )
        try:
            returncode = process.wait(timeout=min(LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            candidate = _token_missing_launcher_owner_exit_candidate(
                fingerprint,
                known_members=known_members,
                token_ids=_token_process_ids(run_token),
            )
            if candidate is None:
                raise Stage1S18ManualInterventionRequired(
                    "S18_PROCESS_LAUNCHER_OWNER_EXIT_REVALIDATION_DISAPPEARED:" + reason
                )
            continue
        except OSError as error:
            raise Stage1S18ManualInterventionRequired(
                "S18_PROCESS_LAUNCHER_OWNER_EXIT_WAIT_FAILED:" + reason
            ) from error
        if not isinstance(returncode, int):
            raise Stage1S18ManualInterventionRequired(
                "S18_PROCESS_LAUNCHER_OWNER_EXIT_UNCONFIRMED:" + reason
            )
        residual = _residual_launch_tree(fingerprint, known_members=known_members)
        if residual["session_members"] or residual["token_members"]:
            raise Stage1S18ManualInterventionRequired("S18_PROCESS_LAUNCHER_EXIT_RESIDUAL")
        return


def _audit_or_confirmed_launcher_exit(process: subprocess.Popen[str], fingerprint: Mapping[str, Any], known_members: Mapping[int, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Audit a live launcher, tolerating only a confirmed natural-exit race."""

    try:
        return _audit_exact_process_group(fingerprint, known_members=known_members)
    except _LauncherOwnerExitCandidate as audit_error:
        _confirm_attested_launcher_owner_exit(
            process,
            fingerprint,
            known_members,
            reason=str(audit_error),
        )
        return None
    except _LauncherNaturalExitCandidate as audit_error:
        _confirm_attested_launcher_exit(
            process,
            fingerprint,
            known_members,
            reason=str(audit_error),
        )
        return None
    except ProcessLookupError as audit_error:
        # `_audit_exact_process_group` raises ProcessLookupError(expected_pid)
        # only when the previously observed expected launcher vanished from the
        # token scan.  Do not reinterpret any other lookup failure as exit.
        if audit_error.args != (fingerprint.get("pid"),):
            raise Stage1S18ManualInterventionRequired(
                "S18_PROCESS_LAUNCHER_EXIT_AUDIT_TARGET_INVALID"
            ) from audit_error
        _confirm_attested_launcher_exit(
            process,
            fingerprint,
            known_members,
            reason="S18_PROCESS_LAUNCHER_TOKEN_OWNER_MISSING",
        )
        return None


def _launch(
    *, repository: Path, work: Path, label: str, command: Sequence[str], environment: Mapping[str, str],
    run_token: str, timeout_seconds: int, lease: object, expected_success: bool,
) -> dict[str, Any]:
    stdout_path, stderr_path = work / f"{label}.stdout.txt", work / f"{label}.stderr.txt"
    env = dict(environment); env["PARAM_IMPORTANCE_S18_RUN_TOKEN"] = run_token
    actual_command, reservation, rendezvous_endpoint = _prepare_rendezvous_command(command, run_token=run_token)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            if reservation is not None:
                reservation.close()
                reservation = None
            process = subprocess.Popen(actual_command, cwd=repository, text=True, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        finally:
            if reservation is not None:
                reservation.close()
        fingerprint = _fingerprint(process.pid, run_token)
        known_members: dict[int, Mapping[str, Any]] = {int(fingerprint["pid"]): fingerprint}
        known_depths: dict[int, int] = {int(fingerprint["pid"]): 0}
        try:
            initial_tree = _audit_exact_process_group(fingerprint, known_members=known_members)
        except _LauncherNaturalExitCandidate as error:
            # Without an initial complete token-bound tree there is no safe
            # ownership basis for treating this as a successful launch exit.
            # The poll-loop-only confirmation path is intentionally unavailable
            # until this first audit has been recorded.
            _manual_intervention(work, label, fingerprint, reason="S18_PROCESS_INITIAL_TREE_LAUNCHER_EXIT_UNCONFIRMED", observed=_residual_launch_tree(fingerprint, known_members=known_members))
            raise Stage1S18ManualInterventionRequired("S18_PROCESS_INITIAL_TREE_LAUNCHER_EXIT_UNCONFIRMED") from error
        except Stage1S18ManualInterventionRequired as error:
            _manual_intervention(work, label, fingerprint, reason=str(error), observed=_residual_launch_tree(fingerprint, known_members=known_members))
            raise
        for member in initial_tree["members"]:
            pid = int(member["pid"])
            known_members.setdefault(pid, dict(member))
            known_depths.setdefault(pid, int(initial_tree["ancestry_depths"][str(pid)]))
        _write(work / f"{label}-process-tree-initial.json", _with_hash({"schema_version": "stage1-s1-8-process-tree-audit-v1", "status": "PASS", "phase": "launch", "expected_launcher": fingerprint, "observed_tree": initial_tree}))
        started, next_heartbeat = time.monotonic(), time.monotonic()
        while process.poll() is None:
            now = time.monotonic()
            if now - started > timeout_seconds:
                _terminate_exact(process, fingerprint, work, label=label, known_members=known_members)
                raise Stage1S18FormalError(f"S18_{label.upper()}_TIMEOUT")
            try:
                observed_tree = _audit_or_confirmed_launcher_exit(process, fingerprint, known_members)
            except Stage1S18ManualInterventionRequired as error:
                _manual_intervention(work, label, fingerprint, reason=str(error), observed=_residual_launch_tree(fingerprint, known_members=known_members))
                raise
            if observed_tree is None:
                break
            for member in observed_tree["members"]:
                pid = int(member["pid"])
                known_members.setdefault(pid, dict(member))
                known_depths.setdefault(pid, int(observed_tree["ancestry_depths"][str(pid)]))
            if now >= next_heartbeat:
                lease.heartbeat()
                next_heartbeat = now + 2.0
            # Poll process ownership more frequently than the lease.  This
            # records elastic workers before a short negative control exits;
            # heartbeat traffic remains bounded at one update per two seconds.
            time.sleep(0.2)
        process.wait(timeout=TERMINAL_PROCESS_JOIN_TIMEOUT_SECONDS)
    residual = _residual_launch_tree(fingerprint, known_members=known_members)
    if residual["session_members"] or residual["token_members"]:
        _manual_intervention(work, label, fingerprint, reason=f"S18_{label.upper()}_PROCESS_TREE_RESIDUAL", observed=residual)
        raise Stage1S18ManualInterventionRequired(f"S18_{label.upper()}_PROCESS_TREE_RESIDUAL")
    outcome = _with_hash({"schema_version": "stage1-s1-8-process-outcome-v1", "label": label, "command": actual_command, "rendezvous_id": run_token, "rendezvous_endpoint": rendezvous_endpoint, "rendezvous_handoff": {"reservation_held_to_popen": rendezvous_endpoint is not None, "single_attempt": True, "silent_retry": False}, "returncode": process.returncode, "fingerprint": fingerprint, "initial_tree": initial_tree, "known_tree": _known_launch_tree(fingerprint, known_members, known_depths), "termination_audit_ref": f"{label}-termination-audit.json" if (work / f"{label}-termination-audit.json").is_file() else None, "stdout_sha256": _sha(stdout_path), "stderr_sha256": _sha(stderr_path), "expected_success": expected_success, "residual_launch_tree": {"session_members": [], "token_members": []}})
    _validate_process_outcome_contract(outcome)
    _write(work / f"{label}.process.json", outcome)
    if (process.returncode == 0) != expected_success:
        raise Stage1S18FormalError(f"S18_{label.upper()}_UNEXPECTED_EXIT")
    return outcome


def _load_arrays(route_dir: Path, report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - server runtime dependency
        raise Stage1S18FormalError("S18_SAFETENSORS_UNAVAILABLE") from error
    manifest = _mapping(report.get("arrays"), field="route.arrays")
    if not _self_hash(manifest) or manifest.get("schema_version") != "stage1-s1-8-safetensors-manifest-v1":
        raise Stage1S18FormalError("S18_ARRAY_MANIFEST_INVALID")
    file = route_dir / str(manifest.get("file"))
    if not file.is_file() or _sha(file) != manifest.get("file_sha256") or file.stat().st_size != manifest.get("file_size_bytes"):
        raise Stage1S18FormalError("S18_ARRAY_FILE_DRIFT")
    tensors = _mapping(manifest.get("tensors"), field="route.arrays.tensors")
    values: dict[str, Any] = {}
    with safe_open(str(file), framework="pt", device="cpu") as handle:
        if handle.metadata() != {"schema_version": "stage1-s1-8-safetensors-manifest-v1"} or set(handle.keys()) != set(tensors):
            raise Stage1S18FormalError("S18_ARRAY_METADATA_OR_KEY_DRIFT")
        for name in sorted(tensors):
            meta = _mapping(tensors[name], field=f"route.arrays.{name}")
            tensor = handle.get_tensor(name)
            raw = tensor.contiguous().numpy().tobytes(order="C")
            if meta != {"sha256": hashlib.sha256(raw).hexdigest(), "dtype": str(tensor.dtype), "shape": list(tensor.shape)}:
                raise Stage1S18FormalError(f"S18_ARRAY_TENSOR_MANIFEST_DRIFT:{name}")
            values[name] = tensor
    return values, manifest


def _staged_array_bundle_replay(repository: Path, staging: Path) -> None:
    """Resolve every published flattened array solely through its descriptor."""

    from param_importance_nlp.contracts.jsonio import load_canonical_json
    from param_importance_nlp.stage1_ddp_oracle import replay
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - formal host dependency
        raise Stage1S18FormalError("S18_SAFETENSORS_UNAVAILABLE") from error
    fixture = _mapping(load_canonical_json(staging / "fixture-manifest.json"), field="staged.fixture")
    ddp = _mapping(load_canonical_json(staging / "ddp-report.json"), field="staged.ddp")
    bundle = _mapping(load_canonical_json(staging / "array-bundle.json"), field="staged.array_bundle")
    descriptors = _mapping(bundle.get("route_artifacts"), field="staged.route_artifacts")
    if not _self_hash(bundle) or set(descriptors) != {"A", "B", "C", "D", "D-rank_swap", "D-local_reverse"}:
        raise Stage1S18FormalError("S18_STAGED_ARRAY_BUNDLE_IDENTITY_INVALID")
    reports = _mapping(ddp.get("baseline_routes"), field="staged.baseline_routes")
    permutations = _mapping(ddp.get("permutation_routes"), field="staged.permutation_routes")
    loaded: dict[str, Mapping[str, Any]] = {}
    for key, descriptor_raw in descriptors.items():
        descriptor = _mapping(descriptor_raw, field="staged.descriptor." + key)
        artifact = _path(staging, descriptor.get("artifact_ref"), field="staged.artifact." + key)
        report_path = _path(staging, descriptor.get("manifest_ref"), field="staged.manifest." + key)
        report = _mapping(load_canonical_json(report_path), field="staged.worker_report." + key)
        manifest = _mapping(report.get("arrays"), field="staged.worker_arrays." + key)
        if not artifact.is_file() or _sha(artifact) != descriptor.get("file_sha256") or artifact.stat().st_size != descriptor.get("file_size_bytes") or manifest.get("artifact_hash") != descriptor.get("manifest_hash") or manifest.get("file_sha256") != descriptor.get("file_sha256") or manifest.get("file_size_bytes") != descriptor.get("file_size_bytes"):
            raise Stage1S18FormalError("S18_STAGED_ARRAY_DESCRIPTOR_DRIFT:" + key)
        _validate_output_schemas(repository, {"worker_report": report, "safetensors_manifest": manifest})
        loaded[key] = load_file(str(artifact), device="cpu")
    result = replay(route_arrays={key: loaded[key] for key in ROUTE_WORLD}, fixture=fixture, route_reports={key: _mapping(reports.get(key), field="staged.baseline." + key) for key in ROUTE_WORLD})
    if result.get("status") != "PASS":
        raise Stage1S18FormalError("S18_STAGED_BASELINE_REPLAY_FAILED")
    for permutation in PERMUTATIONS:
        arrays = {key: loaded[key] for key in ROUTE_WORLD}; arrays["D"] = loaded["D-" + permutation]
        route_reports = {key: _mapping(reports.get(key), field="staged.baseline." + key) for key in ROUTE_WORLD}
        route_reports["D"] = _mapping(permutations.get(permutation), field="staged.permutation." + permutation)
        if replay(route_arrays=arrays, fixture=fixture, route_reports=route_reports).get("status") != "PASS":
            raise Stage1S18FormalError("S18_STAGED_PERMUTATION_REPLAY_FAILED:" + permutation)


def _finite_chart(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise Stage1S18FormalError("S18_CHART_NUMERIC_INVALID:" + field)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise Stage1S18FormalError("S18_CHART_NUMERIC_INVALID:" + field) from error
    if not math.isfinite(parsed):
        raise Stage1S18FormalError("S18_CHART_NUMERIC_INVALID:" + field)
    return parsed


def _write_chart_csv(work: Path, filename: str, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    """Write a strict data projection and prove the persisted CSV round-trips."""

    path = work / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise Stage1S18FormalError("S18_CHART_ROW_FIELD_DRIFT:" + filename)
            writer.writerow(dict(row))
    with path.open("r", encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    if not parsed or len(parsed) != len(rows):
        raise Stage1S18FormalError("S18_CHART_CSV_READBACK_INVALID:" + filename)
    return parsed


def _svg_from_csv(work: Path, *, csv_name: str, svg_name: str, title: str, value_column: str) -> None:
    """Render every mark from a re-read CSV value (not from in-memory rows)."""

    with (work / csv_name).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or value_column not in rows[0]:
        raise Stage1S18FormalError("S18_CHART_CSV_COLUMN_INVALID:" + csv_name)
    values = [_finite_chart(row[value_column], field=f"{csv_name}:{index}") for index, row in enumerate(rows)]
    low, high = min(values), max(values)
    span = high - low
    marks: list[str] = []
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        label = " | ".join(row.get(key, "") for key in tuple(row)[:3])
        if "A U" in label or (row.get("route") == "A" and "u_" in label.lower()):
            raise Stage1S18FormalError("S18_CHART_A_U_REFERENCE_FORBIDDEN")
        x = 60.0 if len(values) == 1 else 60.0 + 840.0 * index / (len(values) - 1)
        y = 120.0 if span == 0.0 else 210.0 - 170.0 * (value - low) / span
        marks.append(f'<circle cx="{x:.9g}" cy="{y:.9g}" r="3" data-row="{index}" data-value="{value:.17g}" data-label="{html.escape(label, quote=True)}"/>')
    output = ('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="260" '
              f'data-source="{html.escape(csv_name, quote=True)}" data-value-column="{html.escape(value_column, quote=True)}">'
              '<line x1="40" y1="210" x2="920" y2="210"/><line x1="40" y1="30" x2="40" y2="210"/>'
              f'<text x="45" y="22">{html.escape(title)}</text><text x="45" y="235">{html.escape(value_column)}</text>' + "".join(marks) + "</svg>")
    (work / svg_name).write_text(output, encoding="utf-8")
    rendered = (work / svg_name).read_text(encoding="utf-8")
    projected = [float(item) for item in re.findall(r'data-value="([^" ]+)"', rendered)]
    if len(projected) != len(values) or any(left != right for left, right in zip(projected, values, strict=True)):
        raise Stage1S18FormalError("S18_CHART_SVG_PROJECTION_DRIFT:" + svg_name)


def _tensor_max_abs(value: object, *, field: str) -> float:
    try:
        import torch
        if not isinstance(value, torch.Tensor):
            raise TypeError
        return _finite_chart(float(value.detach().to(torch.float64).abs().max().item()), field=field)
    except (ImportError, TypeError, ValueError, RuntimeError) as error:
        raise Stage1S18FormalError("S18_CHART_TENSOR_INVALID:" + field) from error


def _charts(work: Path, replay: Mapping[str, Any], ddp_report: Mapping[str, Any], route_arrays: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """Build five materially different projections; A never participates in U."""

    comparisons = replay.get("comparison_rows")
    baseline = ddp_report.get("baseline_routes")
    if not isinstance(comparisons, list) or not comparisons or not isinstance(baseline, Mapping) or set(route_arrays) != set(ROUTE_WORLD):
        raise Stage1S18FormalError("S18_CHART_INPUT_INVALID")
    scatter: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for route, arrays in sorted(route_arrays.items()):
        prefix = "a-reference/equal" if route == "A" else "scores/equal"
        mean = {key.removeprefix(prefix + "/mean_gradient/"): value for key, value in arrays.items() if key.startswith(prefix + "/mean_gradient/")}
        raw = {key.removeprefix(prefix + "/raw_core/"): value for key, value in arrays.items() if key.startswith(prefix + "/raw_core/")}
        pre = {key.removeprefix("pre/equal/"): value for key, value in arrays.items() if key.startswith("pre/equal/")}
        post = {key.removeprefix("post/weighted/"): value for key, value in arrays.items() if key.startswith("post/weighted/")}
        movement = {key.removeprefix("accumulator/weighted/cumulative/data_movement/"): value for key, value in arrays.items() if key.startswith("accumulator/weighted/cumulative/data_movement/")}
        if set(mean) != set(raw) or set(pre) != set(post):
            raise Stage1S18FormalError("S18_CHART_ARRAY_SET_DRIFT:" + route)
        for parameter in sorted(mean):
            scatter.append({"route": route, "parameter": parameter, "mean_gradient_max_abs": f"{_tensor_max_abs(mean[parameter], field=route + ':' + parameter + ':mean'):.17g}", "raw_core_max_abs": f"{_tensor_max_abs(raw[parameter], field=route + ':' + parameter + ':raw'):.17g}"})
        for parameter in sorted(pre):
            delta = _tensor_max_abs(post[parameter] - pre[parameter], field=route + ':' + parameter + ':delta')
            moved = _tensor_max_abs(movement[parameter], field=route + ':' + parameter + ':movement') if route != "A" and parameter in movement else 0.0
            deltas.append({"route": route, "parameter": parameter, "two_step_parameter_delta_max_abs": f"{delta:.17g}", "cumulative_data_movement_max_abs": f"{moved:.17g}"})
    heatmap = [{"comparison": str(row.get("comparison")), "parameter": str(row.get("parameter")), "max_scaled_error": f"{_finite_chart(row.get('max_scaled_error'), field='heatmap'):.17g}"} for row in comparisons if isinstance(row, Mapping)]
    series = [{"comparison": str(row.get("comparison")), "parameter": str(row.get("parameter")), "max_abs_error": f"{_finite_chart(row.get('max_abs_error'), field='series.abs'):.17g}", "normalized_l2_error": f"{_finite_chart(row.get('normalized_l2_error'), field='series.l2'):.17g}"} for row in comparisons if isinstance(row, Mapping)]
    ranks: list[dict[str, object]] = []
    for route in sorted(ROUTE_WORLD):
        report = _mapping(baseline.get(route), field="chart.report." + route)
        for case in report.get("cases", []):
            observed = _mapping(case, field="chart.case")
            for record in observed.get("rank_records", []):
                rank = _mapping(record, field="chart.rank")
                ranks.append({"route": route, "case": str(observed["case"]), "rank": str(rank["rank"]), "local_microbatch_ids": ";".join(str(item) for item in rank["local_microbatch_ids"]), "local_effective_tokens": str(rank["local_effective_tokens"]), "global_microbatch_count": str(observed["global_microbatch_count"]), "global_n1": str(observed["global_n1"]), "global_n2": str(observed["global_n2"])})
    specs = {
        "gradient-raw-scatter.csv": (("route", "parameter", "mean_gradient_max_abs", "raw_core_max_abs"), scatter, "raw_core_max_abs", "gradient-raw-scatter.svg", "mean-gradient / raw-core maxima"),
        "error-heatmap.csv": (("comparison", "parameter", "max_scaled_error"), heatmap, "max_scaled_error", "error-heatmap.svg", "comparison scaled-error heatmap"),
        "error-series.csv": (("comparison", "parameter", "max_abs_error", "normalized_l2_error"), series, "normalized_l2_error", "error-series.svg", "comparison normalized L2 error"),
        "rank-diagnostics.csv": (("route", "case", "rank", "local_microbatch_ids", "local_effective_tokens", "global_microbatch_count", "global_n1", "global_n2"), ranks, "global_n2", "rank-diagnostics.svg", "observed rank partition/token statistics"),
        "parameter-deltas.csv": (("route", "parameter", "two_step_parameter_delta_max_abs", "cumulative_data_movement_max_abs"), deltas, "two_step_parameter_delta_max_abs", "parameter-deltas.svg", "two-step parameter displacement"),
    }
    for csv_name, (columns, rows, value_column, svg_name, title) in specs.items():
        _write_chart_csv(work, csv_name, columns, rows)
        _svg_from_csv(work, csv_name=csv_name, svg_name=svg_name, title=title, value_column=value_column)
    return {name: _sha(work / name) for name in specs}, {spec[3]: _sha(work / spec[3]) for spec in specs.values()}


def _resource_summary(work: Path, *, estimated_peak_bytes: int, preflight: Mapping[str, Any], post_gpu: Mapping[str, Any]) -> dict[str, Any]:
    files = [path for path in work.rglob("*") if path.is_file() and path.name != "resource-summary.json"]
    return _with_hash({"schema_version": "stage1-s1-8-resource-summary-v1", "status": "PASS", "scope": "pre-staging-complete-attempt-files-excluding-resource-summary", "estimated_peak_bytes": estimated_peak_bytes, "actual_attempt_bytes": sum(path.stat().st_size for path in files), "free_bytes_after": shutil.disk_usage(work).free, "file_count": len(files), "preflight_gpu": preflight, "post_gpu": post_gpu})


def _release_lease_transaction(lease: object, *, outcome: str, work: Path, label: str) -> Path:
    """Release the exact held lease or leave an explicit review record.

    ``close`` only drops an OS lock and is never an acceptable substitute for
    removing the authenticated current lease record.
    """

    errors: list[str] = []
    for attempt in (1, 2):
        try:
            history = lease.release(outcome=outcome if attempt == 1 else outcome + "_RETRY")
            current = getattr(lease, "current_path", None)
            if not isinstance(history, Path) or not history.is_file() or (isinstance(current, Path) and current.exists()):
                raise Stage1S18FormalError("S18_LEASE_RELEASE_READBACK_INVALID")
            return history
        except BaseException as error:
            errors.append(f"{type(error).__name__}:{error}")
    _write(work / f"{label}-lease-release-failed.json", _with_hash({"schema_version": "stage1-s1-8-lease-release-failure-v1", "status": "FAILED", "outcome": outcome, "attempt_errors": errors, "current_record_present": bool(getattr(lease, "current_path", Path(".")).exists())}))
    raise Stage1S18ManualInterventionRequired("S18_LEASE_RELEASE_UNCONFIRMED")


def _require_no_current_lease_record(data_root: Path, identity: object) -> Path:
    """Reject the exact resource-key current record before lease acquisition.

    A current record is operator evidence, even if its owner is no longer
    alive.  The formalizer must neither overwrite nor remove it while trying
    to begin a new attempt.
    """

    resource_key = getattr(identity, "resource_key", None)
    if not isinstance(resource_key, str) or not re.fullmatch(r"[0-9a-f]{24}", resource_key):
        raise Stage1S18FormalError("S18_GPU_LEASE_RESOURCE_KEY_INVALID")
    current = data_root / "operations" / "gpu-leases" / "current" / f"{resource_key}.json"
    if current.exists() or current.is_symlink():
        raise Stage1S18FormalError("S18_GPU_LEASE_CURRENT_RECORD_REQUIRES_REVIEW")
    return current


def _finalize_failed_lease(
    lease: object | None, *, held: bool, release_attempted: bool,
    error: BaseException, work: Path,
) -> bool:
    """Release only an acquired lease; preserve the original acquire failure.

    ``ProjectGpuLease.acquire`` may reject an existing stale record after it
    has already cleaned up its OS descriptor.  That is not ownership of the
    record, so calling ``release`` would both mask the original cause and risk
    acting on somebody else's evidence.  A held non-manual failure still uses
    the exact release transaction and must read back its history.
    """

    if lease is None:
        return False
    if isinstance(error, Stage1S18ManualInterventionRequired) or release_attempted or not held:
        lease.close()
        return False
    _release_lease_transaction(lease, outcome="FAILED", work=work, label="failure")
    return True


def _route_plan(*, fixture: Mapping[str, Any], execution_commit: str, route: str, visible: Sequence[str], work: Path, data_root: Path, token_sha: str, model_root: str, cache_root: str, run_token: str, permutation: str = "identity", execution_mode: str = "formal") -> dict[str, Any]:
    from param_importance_nlp.stage1_ddp import WORKER_PLAN_SCHEMA, with_artifact_hash
    route_work = work / f"route-{route}-{permutation}-{execution_mode}"
    route_work.mkdir()
    return with_artifact_hash({"schema_version": WORKER_PLAN_SCHEMA, "task_id": TASK_ID, "execution_commit": execution_commit, "route": route, "fixture": dict(fixture), "fixture_tokens_ref": "../fixture-inputs.safetensors", "fixture_tokens_sha256": token_sha, "data_root": str(data_root), "model_root": model_root, "cache_root": cache_root, "run_token": run_token, "visible_gpu_uuids": list(visible), "nccl_transport_protocol": _nccl_transport_protocol(), "output_dir": "route-output", "permutation": permutation, "execution_mode": execution_mode})


def _offline_environment(cache_root: str) -> tuple[dict[str, str], dict[str, object]]:
    """Constrain every HuggingFace cache and record the no-network contract."""

    root = Path(cache_root).resolve(strict=True)
    hf_home, transformers = root / "hf", root / "transformers"
    hf_home.mkdir(exist_ok=True); transformers.mkdir(exist_ok=True)
    values = {"HF_HOME": str(hf_home), "TRANSFORMERS_CACHE": str(transformers), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PARAM_IMPORTANCE_OFFLINE_GUARD": "1"}
    return values, {"cache_root": str(root), "allowed_cache_roots": [str(hf_home), str(transformers)], "hf_hub_offline": True, "transformers_offline": True, "datasets_offline": True, "external_attempts": []}


def _run_route(*, repository: Path, work: Path, plan: Mapping[str, Any], timeout_seconds: int, lease: object, expect_success: bool) -> tuple[Mapping[str, Any] | None, Path, Mapping[str, Any]]:
    from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
    route = str(plan["route"]); permutation = str(plan["permutation"]); mode = str(plan["execution_mode"])
    route_work = work / f"route-{route}-{permutation}-{mode}"
    plan_path = route_work / "worker-plan.json"; write_canonical_json(plan_path, dict(plan))
    if plan.get("nccl_transport_protocol") != _nccl_transport_protocol():
        raise Stage1S18FormalError("S18_ROUTE_NCCL_TRANSPORT_PLAN_INVALID")
    environment = dict(os.environ); offline, _ = _offline_environment(str(plan["cache_root"])); environment.update(offline); environment.update({"CUDA_VISIBLE_DEVICES": ",".join(plan["visible_gpu_uuids"]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_P2P_DISABLE": "1"})
    command = [sys.executable, "-m", "torch.distributed.run", "--rdzv-id", str(plan["run_token"]), "--rdzv-endpoint", "127.0.0.1:0", "--nproc_per_node", str(ROUTE_WORLD[route]), str(repository / "ops" / "stage1" / "run_s1_8_worker.py"), "--plan", str(plan_path)]
    label = f"{route}-{permutation}-{mode}"
    outcome = _launch(repository=repository, work=work, label=label, command=command, environment=environment, run_token=str(plan["run_token"]), timeout_seconds=timeout_seconds, lease=lease, expected_success=expect_success)
    if not _is_nonzero_loopback_endpoint(outcome.get("rendezvous_endpoint")):
        raise Stage1S18FormalError("S18_ROUTE_RENDEZVOUS_ENDPOINT_INVALID")
    report_path = route_work / "route-output" / "route-report.json"
    if not expect_success:
        if report_path.exists() or (route_work / "route-output" / "success.json").exists():
            raise Stage1S18FormalError("S18_NEGATIVE_CONTROL_PUBLISHED_SUCCESS")
        return None, route_work, outcome
    if not report_path.is_file():
        raise Stage1S18FormalError("S18_ROUTE_REPORT_MISSING")
    report = _mapping(load_canonical_json(report_path), field=f"route.{route}.report")
    if not _self_hash(report) or report.get("status") != "PASS" or report.get("execution_mode") != "formal" or report.get("route") != route or report.get("permutation") != permutation or report.get("nccl_transport_protocol") != _nccl_transport_protocol():
        raise Stage1S18FormalError("S18_ROUTE_REPORT_INVALID")
    return report, route_work / "route-output", outcome


def _scale_oracle(*, repository: Path, work: Path, handoff: Mapping[str, Any], model_root: str, cache_root: str, uuid: str, execution_commit: str, run_token: str, timeout_seconds: int, lease: object) -> Mapping[str, Any]:
    from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
    from param_importance_nlp.stage1_ddp_scale_oracle import OPTIMIZER_CONDITIONING, PLAN_SCHEMA, REPORT_SCHEMA
    selected = {str(index): handoff["token_sha256"][str(index)] for index in range(8)}
    plan = {"schema_version": PLAN_SCHEMA, "task_id": TASK_ID, "execution_commit": execution_commit, "fixture_tokens_ref": "fixture-inputs.safetensors", "fixture_tokens_sha256": handoff["token_file_sha256"], "selected_token_sha256": selected, "upstream_token_sha256": handoff["token_sha256"], "cases": {"equal": {"label_ignore_suffixes": [0] * 8}, "weighted": {"label_ignore_suffixes": [0, 16, 32, 48, 64, 80, 96, 112]}}, "optimizer_conditioning": OPTIMIZER_CONDITIONING, "model_root": model_root, "output_file": "pre-route-scale.json", "run_token": run_token, "visible_gpu_uuids": [uuid]}
    body = dict(plan); body["artifact_hash"] = _canonical(body)
    plan_path = work / "pre-route-scale-plan.json"; write_canonical_json(plan_path, body)
    env = dict(os.environ); offline, _ = _offline_environment(cache_root); env.update(offline); env.update({"CUDA_VISIBLE_DEVICES": uuid, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    scale_process = _launch(repository=repository, work=work, label="pre-route-scale", command=[sys.executable, str(repository / "ops" / "stage1" / "run_s1_8_scale_oracle.py"), "--plan", str(plan_path)], environment=env, run_token=run_token, timeout_seconds=timeout_seconds, lease=lease, expected_success=True)
    if scale_process.get("rendezvous_endpoint") is not None:
        raise Stage1S18FormalError("S18_SCALE_NON_DISTRIBUTED_RENDEZVOUS_INVALID")
    report = _mapping(load_canonical_json(work / "pre-route-scale.json"), field="pre_route_scale")
    needed = {"schema_version", "status", "task_id", "execution_commit", "run_token", "fixture_tokens_sha256", "selected_token_sha256", "upstream_token_sha256", "visible_gpu_uuid", "method", "optimizer_conditioning", "unit_count", "unit_records", "maximum_unit_gradient_abs", "maximum_case", "maximum_microbatch_id", "maximum_parameter", "maximum_abs_data_update", "maximum_data_update_case", "maximum_data_update_parameter", "parameter_registry_hash", "case_pre_parameter_checksums", "case_post_parameter_checksums", "artifact_hash"}
    if set(report) != needed or not _self_hash(report) or report.get("schema_version") != REPORT_SCHEMA or report.get("status") != "PASS" or report.get("visible_gpu_uuid") != uuid or report.get("unit_count") != 16 or report.get("method") != "independent_pre_route_single_gpu_autograd_oracle" or not isinstance(report.get("maximum_unit_gradient_abs"), (int, float)) or float(report["maximum_unit_gradient_abs"]) <= 0 or not isinstance(report.get("maximum_abs_data_update"), (int, float)) or float(report["maximum_abs_data_update"]) <= 0:
        raise Stage1S18FormalError("S18_PRE_ROUTE_SCALE_REPORT_INVALID")
    _require_pre_route_scale_conditioning(report, OPTIMIZER_CONDITIONING)
    pre_checks, post_checks = report.get("case_pre_parameter_checksums"), report.get("case_post_parameter_checksums")
    if not isinstance(pre_checks, Mapping) or not isinstance(post_checks, Mapping) or set(pre_checks) != {"equal", "weighted"} or set(post_checks) != {"equal", "weighted"} or pre_checks.get("weighted") != post_checks.get("equal") or not isinstance(report.get("parameter_registry_hash"), str) or len(str(report["parameter_registry_hash"])) != 64:
        raise Stage1S18FormalError("S18_PRE_ROUTE_SCALE_STATE_SEQUENCE_INVALID")
    if report.get("selected_token_sha256") != selected or report.get("upstream_token_sha256") != handoff["token_sha256"]:
        raise Stage1S18FormalError("S18_PRE_ROUTE_SCALE_TOKEN_BINDING_INVALID")
    return report


def prelease_dry_chain(*, repository: Path, data_root: Path, s1_7_index_ref: str, gpu_capability_ref: str, approved_gpu_uuids: Sequence[str]) -> dict[str, object]:
    """Run every no-GPU formal prerequisite and stop before discovery/lease."""

    repository, data_root = repository.resolve(strict=True), data_root.resolve(strict=True)
    handoff = load_s1_7_handoff(data_root=data_root, index_ref=s1_7_index_ref, repository=repository)
    capability = _load_capability(data_root, gpu_capability_ref, approved_gpu_uuids)
    parent_cuda = _require_prelease_cuda_hidden()
    model_root, cache_root, qualification = _frozen_model_and_cache_root(repository, data_root, handoff)
    pile = _audit_pile_download_activity(handoff)
    return {"status": "PASS", "s1_7_handoff": {key: value for key, value in handoff.items() if key != "token_file"}, "gpu_capability": capability, "nccl_transport_protocol": _nccl_transport_protocol(), "parent_cuda": parent_cuda, "model_root": model_root, "cache_root": cache_root, "model_qualification": qualification, "pile_download_audit": pile}


def execute(*, repository: Path, data_root: Path, s1_7_index_ref: str, gpu_capability_ref: str, approved_gpu_uuids: Sequence[str], attempt_id: str, lease_owner: str, timeout_seconds: int = 3600) -> dict[str, str]:
    """Execute the formal producer only after explicit four-UUID approval."""

    from param_importance_nlp.contracts.jsonio import load_canonical_json
    from param_importance_nlp.runtime.operations import GpuLeaseIdentity, ProjectGpuLease
    from param_importance_nlp.stage1_ddp import build_fixture, build_route_layout, permute_route_layout, tensor_map_digest, validate_fixture
    from param_importance_nlp.stage1_ddp_oracle import replay
    repository, data_root = repository.resolve(strict=True), data_root.resolve(strict=True)
    if not attempt_id or not lease_owner or timeout_seconds <= 0:
        raise Stage1S18FormalError("S18_FORMAL_ARGUMENT_INVALID")
    commit = _git(repository, "rev-parse", "HEAD")
    handoff = load_s1_7_handoff(data_root=data_root, index_ref=s1_7_index_ref, repository=repository)
    capability = _load_capability(data_root, gpu_capability_ref, approved_gpu_uuids)
    _require_prelease_cuda_hidden()
    model_root, cache_root, model_qualification = _frozen_model_and_cache_root(repository, data_root, handoff)
    target = data_root / "evidence" / "stage1" / "s1-8-formal" / commit / attempt_id
    work = data_root / "tmp" / "stage1-s1-8" / attempt_id
    if target.exists() or work.exists():
        raise Stage1S18FormalError("S18_FORMAL_ATTEMPT_COLLISION")
    work.mkdir(parents=True)
    estimated_peak_bytes = 80 * 1024**3
    disk = shutil.disk_usage(work)
    if disk.free < estimated_peak_bytes:
        raise Stage1S18FormalError("S18_DISK_CAPACITY_INSUFFICIENT")
    shutil.copy2(handoff["token_file"], work / "fixture-inputs.safetensors")
    historical_role_copies = {
        "s1-7-historical-producer-attestation.json": (
            handoff["historical_producer_attestation_file"],
            handoff["historical_producer_attestation_sha256"],
        ),
        "s1-7-historical-g3-replay.json": (
            handoff["historical_g3_replay_file"],
            handoff["historical_g3_replay_sha256"],
        ),
    }
    for published, (source, digest) in historical_role_copies.items():
        if not isinstance(source, Path) or not source.is_file() or _sha(source) != digest:
            raise Stage1S18FormalError("S18_S17_HISTORICAL_G3_COPY_SOURCE_DRIFT:" + published)
        shutil.copy2(source, work / published)
        if _sha(work / published) != digest:
            raise Stage1S18FormalError("S18_S17_HISTORICAL_G3_COPY_WRITEBACK_DRIFT:" + published)
    _write(work / "model-qualified-resolution.json", _with_hash({"schema_version": "stage1-s1-8-model-qualified-resolution-v1", "status": "PASS", "model": model_qualification}))
    _, offline_policy = _offline_environment(cache_root)
    _write(work / "offline-policy.json", _with_hash({"schema_version": "stage1-s1-8-offline-policy-v1", "status": "PASS", "policy": offline_policy}))
    lease: ProjectGpuLease | None = None; lease_held = False; release_attempted = False; phase = "preflight"; staging: Path | None = None
    try:
        pile_download_audit = _audit_pile_download_activity(handoff)
        nccl_transport = _nccl_transport_protocol()
        _write(work / "nccl-transport-protocol.json", _with_hash({"schema_version": "stage1-s1-8-nccl-transport-binding-v1", "status": "PASS", "protocol": nccl_transport}))
        prelease_quiescence = require_gpu_quiescence(approved_gpu_uuids, work=work, phase="prelease")
        preflight = _mapping(prelease_quiescence.get("final_gpu"), field="prelease_quiescence.final_gpu")
        _write(work / "preflight.json", _with_hash({"schema_version": "stage1-s1-8-gpu-preflight-v1", "status": "PASS", "gpu": preflight, "capability": capability, "nccl_transport_protocol": nccl_transport, "pile_download_audit": pile_download_audit}))
        identity = GpuLeaseIdentity(run_id=f"stage1-s1-8-{attempt_id}", lease_id=f"s1-8-{attempt_id}", gpu_uuids=tuple(approved_gpu_uuids), owner=lease_owner, config_hash=_canonical({"task_id": TASK_ID, "commit": commit, "attempt_id": attempt_id}), environment_hash=_canonical(preflight))
        phase = "lease_preflight"; _require_no_current_lease_record(data_root, identity)
        phase = "lease_acquire"; lease = ProjectGpuLease(data_root, identity); lease.acquire(); lease_held = True; lease.heartbeat()
        post_lease = discover_approved_gpus(approved_gpu_uuids); _write(work / "post-lease-gpu.json", _with_hash({"schema_version": "stage1-s1-8-gpu-preflight-v1", "status": "PASS", "gpu": post_lease}))
        run_token = hashlib.sha256(f"{commit}:{attempt_id}:{time.time_ns()}".encode()).hexdigest()
        _write(work / "attempt-start.json", _with_hash({"schema_version": "stage1-s1-8-attempt-start-v1", "status": "STARTED", "run_token": run_token, "parent_fingerprint": _parent_fingerprint(os.getpid(), run_token), "lease": identity.to_dict()}))
        phase = "nccl_smoke"; smoke_env = dict(os.environ); smoke_env.update({"CUDA_VISIBLE_DEVICES": ",".join(approved_gpu_uuids), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_P2P_DISABLE": "1"})
        smoke = _launch(repository=repository, work=work, label="nccl-smoke", command=[sys.executable, "-m", "torch.distributed.run", "--rdzv-id", run_token, "--rdzv-endpoint", "127.0.0.1:0", "--nproc_per_node", "4", str(repository / "ops" / "stage1" / "run_s1_8_nccl_smoke.py"), "--report", str(work / "nccl-smoke-report.json")], environment=smoke_env, run_token=run_token, timeout_seconds=min(timeout_seconds, 300), lease=lease, expected_success=True)
        smoke_report = _mapping(load_canonical_json(work / "nccl-smoke-report.json"), field="nccl_smoke")
        if not _self_hash(smoke_report) or smoke_report.get("backend") != "nccl" or smoke_report.get("nccl_transport_protocol") != nccl_transport or smoke_report.get("world_size") != 4 or smoke_report.get("allocation_probe") != "torch.empty.float32.256" or smoke_report.get("rank_records") != [{"rank": index, "uuid": approved_gpu_uuids[index], "input": index + 1, "output": 10, "cuda_initialized": True, "allocation_bytes": 1024} for index in range(4)]:
            raise Stage1S18FormalError("S18_NCCL_SMOKE_REPORT_INVALID")
        phase = "pre_route_scale"; scale = _scale_oracle(repository=repository, work=work, handoff=handoff, model_root=model_root, cache_root=cache_root, uuid=approved_gpu_uuids[0], execution_commit=commit, run_token=run_token, timeout_seconds=timeout_seconds, lease=lease)
        fixture = build_fixture(token_sha256=handoff["token_sha256"], upstream_fixture_hash=str(handoff["fixture_hash"]), gradient_design_scale=float(scale["maximum_unit_gradient_abs"]), optimizer_delta_design_scale=float(scale["maximum_abs_data_update"]), pre_route_scale_oracle_hash=str(scale["artifact_hash"]), pre_route_parameter_registry_hash=str(scale["parameter_registry_hash"]), pre_route_case_state_checksums=_mapping(scale["case_pre_parameter_checksums"], field="scale.case_pre_parameter_checksums"))
        validate_fixture(fixture); _write(work / "fixture-manifest.json", fixture)
        baseline_reports: dict[str, Mapping[str, Any]] = {}; baseline_arrays: dict[str, Mapping[str, Any]] = {}; manifests: dict[str, Mapping[str, Any]] = {}
        for route, world in ROUTE_WORLD.items():
            phase = f"route_{route}"; plan = _route_plan(fixture=fixture, execution_commit=commit, route=route, visible=approved_gpu_uuids[:world], work=work, data_root=data_root, token_sha=str(handoff["token_file_sha256"]), model_root=model_root, cache_root=cache_root, run_token=run_token)
            report, route_dir, _ = _run_route(repository=repository, work=work, plan=plan, timeout_seconds=timeout_seconds, lease=lease, expect_success=True)
            if report is None: raise Stage1S18FormalError("S18_BASELINE_REPORT_NONE")
            expected_layout = fixture["routes"][route]
            if report.get("world_size") != world or report.get("visible_gpu_uuids") != list(approved_gpu_uuids[:world]) or report.get("rank_to_gpu_uuid") != list(approved_gpu_uuids[:world]) or report.get("route_layout") != expected_layout or report.get("parameter_registry_hash") != fixture["pre_route_gradient_scale"]["parameter_registry_hash"]:
                raise Stage1S18FormalError("S18_ROUTE_IDENTITY_OR_REGISTRY_DRIFT:" + route)
            arrays, manifest = _load_arrays(route_dir, report)
            _validate_output_schemas(repository, {"worker_report": report, "safetensors_manifest": manifest})
            baseline_reports[route] = report; baseline_arrays[route] = arrays; manifests[route] = manifest
        phase = "baseline_offline_replay"; result = _require_baseline_replay(replay_fn=replay, route_arrays=baseline_arrays, fixture=fixture, route_reports=baseline_reports)
        permutation_reports: dict[str, Mapping[str, Any]] = {}
        for permutation in PERMUTATIONS:
            phase = f"D_{permutation}"; plan = _route_plan(fixture=fixture, execution_commit=commit, route="D", visible=approved_gpu_uuids, work=work, data_root=data_root, token_sha=str(handoff["token_file_sha256"]), model_root=model_root, cache_root=cache_root, run_token=run_token, permutation=permutation)
            report, route_dir, _ = _run_route(repository=repository, work=work, plan=plan, timeout_seconds=timeout_seconds, lease=lease, expect_success=True)
            if report is None: raise Stage1S18FormalError("S18_PERMUTATION_REPORT_NONE")
            expected_permutation_layout = permute_route_layout(build_route_layout("D"), permutation=permutation).to_dict()
            if (
                not _self_hash(report) or report.get("status") != "PASS" or report.get("execution_commit") != commit
                or report.get("run_token") != run_token or report.get("fixture_hash") != fixture["fixture_hash"]
                or report.get("execution_mode") != "formal" or report.get("route") != "D" or report.get("permutation") != permutation
                or report.get("world_size") != 4 or report.get("visible_gpu_uuids") != list(approved_gpu_uuids)
                or report.get("rank_to_gpu_uuid") != list(approved_gpu_uuids) or report.get("route_layout") != expected_permutation_layout
                or report.get("parameter_registry_hash") != fixture["pre_route_gradient_scale"]["parameter_registry_hash"]
            ):
                raise Stage1S18FormalError("S18_PERMUTATION_ROUTE_IDENTITY_DRIFT")
            arrays, manifest = _load_arrays(route_dir, report)
            _validate_output_schemas(repository, {"worker_report": report, "safetensors_manifest": manifest})
            candidate_arrays = dict(baseline_arrays); candidate_reports = dict(baseline_reports); candidate_arrays["D"] = arrays; candidate_reports["D"] = report
            result = replay(route_arrays=candidate_arrays, fixture=fixture, route_reports=candidate_reports)
            if result.get("status") != "PASS": raise Stage1S18FormalError("S18_PERMUTATION_REPLAY_FAILED")
            permutation_reports[permutation] = report; manifests[f"D-{permutation}"] = manifest
        negative: dict[str, Mapping[str, Any]] = {}
        for mode, marker in (("ordinary_sync_negative", "S18_NO_SYNC_DDP_COLLECTIVE_DETECTED"), ("inject_rank_failure", "S18_INJECTED_RANK_FAILURE")):
            phase = mode; plan = _route_plan(fixture=fixture, execution_commit=commit, route="D", visible=approved_gpu_uuids, work=work, data_root=data_root, token_sha=str(handoff["token_file_sha256"]), model_root=model_root, cache_root=cache_root, run_token=run_token, execution_mode=mode)
            _, route_dir, outcome = _run_route(repository=repository, work=work, plan=plan, timeout_seconds=timeout_seconds, lease=lease, expect_success=False)
            stderr = (work / f"D-identity-{mode}.stderr.txt").read_text(encoding="utf-8")
            if marker not in stderr: raise Stage1S18FormalError("S18_NEGATIVE_MARKER_MISSING:" + mode)
            negative[mode] = {"process": outcome, "marker": marker, "route_work": str(route_dir.relative_to(work)), "success_marker_absent": True}
        phase = "post_gpu"
        post_worker_quiescence = require_gpu_quiescence(approved_gpu_uuids, work=work, phase="post_worker")
        post_gpu = _mapping(post_worker_quiescence.get("final_gpu"), field="post_worker_quiescence.final_gpu")
        _write(work / "post-worker-gpu.json", _with_hash({"schema_version": "stage1-s1-8-gpu-preflight-v1", "status": "PASS", "gpu": post_gpu}))
        first_history = _release_lease_transaction(lease, outcome="GPU_PHASE_SUCCESS", work=work, label="first"); release_attempted = True; lease_held = False; lease = None; shutil.copy2(first_history, work / "lease-history-first.json")
        phase = "post_release_gpu"
        post_release_quiescence = require_gpu_quiescence(approved_gpu_uuids, work=work, phase="post_release")
        post_release = _mapping(post_release_quiescence.get("final_gpu"), field="post_release_quiescence.final_gpu")
        _write(work / "post-release-gpu.json", _with_hash({"schema_version": "stage1-s1-8-gpu-preflight-v1", "status": "PASS", "gpu": post_release}))
        # The exact set must be reacquirable immediately after release; no GPU
        # worker is launched under the second lease.
        phase = "reacquire_preflight_gpu"
        reacquire_quiescence = require_gpu_quiescence(approved_gpu_uuids, work=work, phase="reacquire_preflight")
        second = ProjectGpuLease(data_root, identity); second_held = False
        try:
            second.acquire(); second_held = True; second.heartbeat(); second_history = _release_lease_transaction(second, outcome="RELEASE_REACQUIRE_CHECK", work=work, label="reacquire"); second_held = False
        except Stage1S18ManualInterventionRequired:
            # The transaction persisted a review marker and deliberately did
            # not pretend a current record was released.
            raise
        except BaseException:
            if second_held:
                # A non-process error after acquisition must still remove the
                # exact current record.  ``close`` alone is never a release.
                _release_lease_transaction(second, outcome="REACQUIRE_CHECK_FAILED", work=work, label="reacquire-failure")
                second_held = False
            else:
                second.close()
            raise
        shutil.copy2(second_history, work / "lease-history-reacquire.json")
        replay_record = _with_hash(dict(result)); _write(work / "replay-validation.json", replay_record)
        table = _with_hash({"schema_version": "stage1-s1-8-comparison-table-v1", "status": "PASS", "rows": result["comparison_rows"]}); _write(work / "comparison-table.json", table)
        source_hashes = _implementation_source_map(repository)
        gpu_quiescence_bindings = {
            phase: {"ref": filename, "sha256": _sha(work / filename)}
            for phase, filename in GPU_QUIESCENCE_ROLES.items()
        }
        ddp_report = _with_hash({"schema_version": "stage1-s1-8-ddp-report-v4", "status": "PASS", "task_id": TASK_ID, "fixture_hash": fixture["fixture_hash"], "fixture_schema_version": fixture["schema_version"], "fixture_id": fixture["fixture_id"], "nccl_transport_protocol": nccl_transport, "implementation_source_sha256": source_hashes, "baseline_routes": baseline_reports, "permutation_routes": permutation_reports, "negative_controls": negative, "gpu_quiescence": gpu_quiescence_bindings}); _write(work / "ddp-report.json", ddp_report)
        csv_hashes, svg_hashes = _charts(work, result, ddp_report, baseline_arrays)
        # It is intentionally written only after every worker output, replay,
        # report and chart exists, and excludes itself from the measured scope.
        _write(work / "resource-summary.json", _resource_summary(work, estimated_peak_bytes=estimated_peak_bytes, preflight=preflight, post_gpu=post_gpu))
        role_files = {"fixture_manifest": "fixture-manifest.json", "ddp_report": "ddp-report.json", "array_bundle": "array-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-ddp-record.json"}
        fixed_reproduction = _fixed_reproduction_roles()
        reproduction = {role: published for role, (published, _source) in fixed_reproduction.items()}
        reproduction_sources: dict[str, Path] = {published: work / source for published, source in fixed_reproduction.values()}
        expected_root_logs = {source for role, (_published, source) in fixed_reproduction.items() if role.startswith("run_root_")}
        actual_root_logs = {
            source.relative_to(work).as_posix()
            for source in (list(work.glob("*.process.json")) + list(work.glob("*.stdout.txt")) + list(work.glob("*.stderr.txt")) + list(work.glob("*-process-tree-initial.json")) + list(work.glob("*-termination-audit.json")))
        }
        expected_route_files = {source for role, (_published, source) in fixed_reproduction.items() if role.startswith("run_") and not role.startswith("run_root_")}
        actual_route_files = {
            source.relative_to(work).as_posix()
            for source in (list(work.glob("route-*/*")) + list(work.glob("route-*/*/*")))
            if source.is_file()
        }
        if actual_root_logs != expected_root_logs or actual_route_files != expected_route_files or any(not source.is_file() for source in reproduction_sources.values()):
            raise Stage1S18FormalError("S18_REPRODUCTION_ROLE_CLOSURE_INVALID")
        array_descriptors: dict[str, object] = {}
        route_paths = {**{route: work / f"route-{route}-identity-formal" / "route-output" for route in ROUTE_WORLD}, **{f"D-{permutation}": work / f"route-D-{permutation}-formal" / "route-output" for permutation in PERMUTATIONS}}
        if set(manifests) != set(route_paths):
            raise Stage1S18FormalError("S18_ARRAY_DESCRIPTOR_ROUTE_SET_INVALID")
        for route_key, directory in route_paths.items():
            manifest = manifests[route_key]; source = directory / str(manifest["file"])
            relative = source.relative_to(work).as_posix(); published = "run__" + relative.replace("/", "__")
            if reproduction_sources.get(published) != source:
                raise Stage1S18FormalError("S18_ARRAY_PUBLISHED_REF_MISSING:" + route_key)
            report_source = directory / "route-report.json"
            report_relative = report_source.relative_to(work).as_posix(); report_published = "run__" + report_relative.replace("/", "__")
            if reproduction_sources.get(report_published) != report_source:
                raise Stage1S18FormalError("S18_ARRAY_MANIFEST_PUBLISHED_REF_MISSING:" + route_key)
            array_descriptors[route_key] = {"artifact_ref": published, "manifest_ref": report_published, "file_sha256": manifest["file_sha256"], "file_size_bytes": manifest["file_size_bytes"], "manifest_hash": manifest["artifact_hash"]}
        arrays_bundle = _with_hash({"schema_version": "stage1-s1-8-array-bundle-v1", "status": "PASS", "route_artifacts": array_descriptors})
        _write(work / "array-bundle.json", arrays_bundle)
        all_worker_reports = {**baseline_reports, **{f"D-{permutation}": report for permutation, report in permutation_reports.items()}}
        base_requirements = {
            "s1_7_handoff": handoff["gate_artifact_hash"] == EXPECTED_G1_SINGLE_HASH,
            "consumer_diff": isinstance(_audit_consumer_diff(repository), tuple),
            "approved_four_uuid_topology": capability["artifact_hash"] == EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH and set(approved_gpu_uuids).issubset(set(capability["allowed_gpu_uuids"])) and preflight["requested_uuid_order"] == list(approved_gpu_uuids) == post_gpu["requested_uuid_order"],
            "nccl_smoke": smoke["returncode"] == 0 and smoke["residual_launch_tree"] == {"session_members": [], "token_members": []} and _is_nonzero_loopback_endpoint(smoke.get("rendezvous_endpoint")) and smoke_report.get("status") == "PASS" and smoke_report.get("nccl_transport_protocol") == nccl_transport,
            "pre_route_scale_oracle": (
                scale["case_pre_parameter_checksums"]["weighted"] == scale["case_post_parameter_checksums"]["equal"]
                and _exact_contract_value(scale.get("optimizer_conditioning"), fixture.get("optimizer_conditioning"))
                and _exact_contract_value(
                    {key: fixture.get("optimizer", {}).get(key) for key in ("learning_rate", "weight_decay", "betas", "eps", "foreach", "fused")},
                    {key: fixture.get("optimizer_conditioning", {}).get(key) for key in ("learning_rate", "weight_decay", "betas", "eps", "foreach", "fused")},
                )
            ),
            "real_routes_equal_and_weighted": set(baseline_reports) == set(ROUTE_WORLD) and all(len(report["cases"]) == 2 for report in baseline_reports.values()),
            "rank_partition_and_no_sync": [row["ordinary_ddp_gradient_collectives"] for row in baseline_reports["A"]["cases"]] == [1, 2] and all(all(row["ordinary_ddp_gradient_collectives"] == 0 for row in report["cases"]) for route, report in baseline_reports.items() if route != "A"),
            "manual_collective_contract": result.get("status") == "PASS",
            "independent_fp64_replay": replay_record.get("oracle_reference_dtype") == "torch.float64" and replay_record.get("production_candidate_dtype") == "torch.float32",
            "optimizer_and_accumulator": any("accumulator:cumulative:absolute" in key for key in result["checks"]),
            "rank_and_order_permutations": set(permutation_reports) == set(PERMUTATIONS),
            "ordinary_sync_negative": negative["ordinary_sync_negative"]["marker"] == "S18_NO_SYNC_DDP_COLLECTIVE_DETECTED",
            "controlled_rank_failure": negative["inject_rank_failure"]["marker"] == "S18_INJECTED_RANK_FAILURE",
            "array_manifest": all(_self_hash(value) for value in manifests.values()),
            "lease_release_reacquire": (
                (work / "lease-history-first.json").is_file() and (work / "lease-history-reacquire.json").is_file()
                and all(record.get("status") == "PASS" and _self_hash(record) for record in (prelease_quiescence, post_worker_quiescence, post_release_quiescence, reacquire_quiescence))
            ),
            "resource_summary": _self_hash(_mapping(load_canonical_json(work / "resource-summary.json"), field="resource_summary")),
            "charts_no_a_u_reference": all("A U" not in (work / name).read_text(encoding="utf-8") for name in svg_hashes),
        }
        if set(base_requirements) != set(GATE_CHECK_IDS) - {"schemas_and_atomic_publication"} or not all(base_requirements.values()):
            raise Stage1S18FormalError("S18_GATE_REQUIREMENT_FAILED")
        reproduction_sha = {role: _sha(reproduction_sources[name]) for role, name in reproduction.items()}
        handoff_for_index = _index_safe_s1_7_handoff(handoff)

        def build_candidate(schema_result: bool) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any]]:
            requirements = {**base_requirements, "schemas_and_atomic_publication": schema_result}
            gate_value = _with_hash({"schema_version": "stage1-s1-8-gate-record-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "requirements": requirements})
            role_hashes = {role: _canonical_file_sha(value) for role, value in {"fixture_manifest": fixture, "ddp_report": ddp_report, "array_bundle": arrays_bundle, "comparison_table": table, "gate_record": gate_value}.items()}
            validation_value = _with_hash({"schema_version": "stage1-s1-8-validation-v4", "status": "PASS", "task_id": TASK_ID, "gate_id": GATE_ID, "producer_commit": commit, "fixture_schema_version": fixture["schema_version"], "fixture_id": fixture["fixture_id"], "checks": [{"check_id": identifier, "status": "PASS", "detail": identifier.replace("_", " ")} for identifier in GATE_CHECK_IDS], "role_sha256": role_hashes, "gpu_quiescence": gpu_quiescence_bindings})
            index_value = _with_hash({"schema_version": "stage1-s1-8-formalization-index-v4", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "generator_git_commit": commit, "consumer_git_commit": commit, "fixture_schema_version": fixture["schema_version"], "fixture_id": fixture["fixture_id"], "gpu_capability": capability, "nccl_transport_protocol": nccl_transport, "implementation_source_sha256": source_hashes, "s1_7_handoff": handoff_for_index, "role_refs": role_files, "role_sha256": role_hashes, "reproduction_role_refs": reproduction, "reproduction_role_sha256": reproduction_sha, "gate_artifact_hash": gate_value["artifact_hash"], "validation_ref": "validation.json", "validation_sha256": _canonical_file_sha(validation_value), "replay_ref": "replay-validation.json", "replay_sha256": _canonical_file_sha(replay_record), "next_task_ids": ["stage1.10_checkpoint_resume_and_artifacts"]})
            values = {"fixture_manifest": fixture, "ddp_report": ddp_report, "array_bundle": arrays_bundle, "comparison_table": table, "gate_record": gate_value, "replay": replay_record, "validation": validation_value, "index": index_value}
            return values, role_hashes, gate_value, validation_value, index_value

        # The preliminary object is byte-identical to the final one if and
        # only if real schema/cross-reference validation succeeds.  This makes
        # the last gate predicate evidence-derived rather than a literal.
        candidate_objects, role_sha, gate, validation, index = build_candidate(True)
        schema_evidence = _candidate_publication_check(repository=repository, objects=candidate_objects, worker_reports=all_worker_reports, manifests=manifests, source_files=reproduction_sources)
        candidate_objects, role_sha, gate, validation, index = build_candidate(schema_evidence)
        if not _candidate_publication_check(repository=repository, objects=candidate_objects, worker_reports=all_worker_reports, manifests=manifests, source_files=reproduction_sources):
            raise Stage1S18FormalError("S18_CANDIDATE_SCHEMA_EVIDENCE_FALSE")
        _write(work / "g1-ddp-record.json", gate); _write(work / "validation.json", validation); _write(work / "index.json", index)
        if any(_sha(work / role_files[role]) != digest for role, digest in role_sha.items()) or _sha(work / "validation.json") != _canonical_file_sha(validation) or _sha(work / "index.json") != _canonical_file_sha(index):
            raise Stage1S18FormalError("S18_CANDIDATE_FILE_WRITEBACK_DRIFT")
        target.parent.mkdir(parents=True, exist_ok=True); staging = target.parent / f".{attempt_id}.publishing"
        if staging.exists(): raise Stage1S18FormalError("S18_PUBLISH_STAGING_COLLISION")
        staging.mkdir()
        publish = set(role_files.values()) | {"replay-validation.json", "validation.json", "index.json"} | set(csv_hashes) | set(svg_hashes) | set(reproduction_sources)
        for name in sorted(publish):
            source = reproduction_sources.get(name, work / name)
            if not source.is_file():
                raise Stage1S18FormalError("S18_PUBLISH_SOURCE_MISSING:" + name)
            shutil.copy2(source, staging / name)
        staged_objects = {"fixture_manifest": _mapping(load_canonical_json(staging / "fixture-manifest.json"), field="staged.fixture"), "ddp_report": _mapping(load_canonical_json(staging / "ddp-report.json"), field="staged.ddp"), "array_bundle": _mapping(load_canonical_json(staging / "array-bundle.json"), field="staged.bundle"), "comparison_table": _mapping(load_canonical_json(staging / "comparison-table.json"), field="staged.table"), "gate_record": _mapping(load_canonical_json(staging / "g1-ddp-record.json"), field="staged.gate"), "replay": _mapping(load_canonical_json(staging / "replay-validation.json"), field="staged.replay"), "validation": _mapping(load_canonical_json(staging / "validation.json"), field="staged.validation"), "index": _mapping(load_canonical_json(staging / "index.json"), field="staged.index")}
        staged_worker_reports: dict[str, Mapping[str, Any]] = {}
        for route_key, raw in _mapping(staged_objects["array_bundle"].get("route_artifacts"), field="staged.bundle.routes").items():
            descriptor = _mapping(raw, field="staged.bundle.route." + route_key)
            staged_worker_reports[route_key] = _mapping(load_canonical_json(_path(staging, descriptor.get("manifest_ref"), field="staged.manifest." + route_key)), field="staged.worker." + route_key)
        staged_manifests = {route_key: _mapping(report.get("arrays"), field="staged.arrays." + route_key) for route_key, report in staged_worker_reports.items()}
        staged_sources = {name: staging / name for name in reproduction_sources}
        _candidate_publication_check(repository=repository, objects=staged_objects, worker_reports=staged_worker_reports, manifests=staged_manifests, source_files=staged_sources)
        if (staging / "success.json").exists() or any(_sha(staging / role_files[role]) != digest for role, digest in role_sha.items()) or any(_sha(staging / reproduction[role]) != digest for role, digest in reproduction_sha.items()):
            raise Stage1S18FormalError("S18_STAGED_PUBLICATION_VERIFICATION_FAILED")
        _staged_array_bundle_replay(repository, staging)
        _write(staging / "success.json", _with_hash({"schema_version": "stage1-s1-8-success-v1", "status": "PASS", "gate_artifact_hash": gate["artifact_hash"], "validation_sha256": _sha(work / "validation.json")}))
        os.replace(staging, target); staging = None
        return {"index_ref": (target / "index.json").relative_to(data_root).as_posix(), "validation_ref": (target / "validation.json").relative_to(data_root).as_posix()}
    except BaseException as error:
        if staging is not None and (staging / "success.json").exists(): (staging / "success.json").unlink()
        _write(work / "failed.json", _with_hash({"schema_version": "stage1-s1-8-failure-v1", "status": "FAILED", "phase": phase, "error_type": type(error).__name__, "error": str(error)}))
        if _finalize_failed_lease(lease, held=lease_held, release_attempted=release_attempted, error=error, work=work):
            release_attempted = True
            lease_held = False
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-7-index-ref", required=True); parser.add_argument("--gpu-capability-ref", required=True); parser.add_argument("--approved-gpu-uuid", action="append", dest="uuids", required=True)
    parser.add_argument("--attempt-id", required=True); parser.add_argument("--lease-owner", required=True); parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    print(execute(repository=args.repository, data_root=args.data_root, s1_7_index_ref=args.s1_7_index_ref, gpu_capability_ref=args.gpu_capability_ref, approved_gpu_uuids=args.uuids, attempt_id=args.attempt_id, lease_owner=args.lease_owner, timeout_seconds=args.timeout_seconds))
    return 0


if __name__ == "__main__":  # pragma: no cover - executable entry point
    raise SystemExit(main())
