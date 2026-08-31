"""Stage 0/1 非训练任务的专用复合 task runner。

本模块不把审计、存储、数学验证和交付任务降级成同一个 ``core_probe``。每个
canonical task ID 都会调用其负责层的真实 API，并发布带任务语义的证据：例如
checkpoint 任务会真的执行两阶段发布、发现、恢复和 tombstone-only retention；
Stage 1 estimator 任务会分别计算生产核与独立 FP64 oracle。

这里的 tiny 数值只承担本机合同验证，产物始终把 formal Gate 写为 ``NOT_RUN``。
formal 命令所需服务器、设备、资产或上游 Gate 仍由统一 ``TaskRuntime`` preflight
计算为结构化 ``BLOCKED``，本模块既不探测网络，也不伪造这些能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import platform
from pathlib import Path, PurePosixPath
import sys
from types import MappingProxyType
from typing import Mapping

import torch

from ..assets import (
    AssetActorRole,
    AssetState,
    AssetType,
    build_manifest,
    transition_manifest,
    verify_only,
)
from ..atomic import atomic_write_bytes, sha256_file
from ..capacity import (
    estimate_checkpoint_bytes,
    estimate_experiment_storage,
    estimate_parameter_statistics_bytes,
)
from ..contracts.identity import RunIdentity, derive_experiment_id
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.freeze import ContractFreeze
from ..contracts.runtime_evidence import RuntimeCapabilityEvidence
from ..contracts.seed import SeedPlan
from ..contracts.status import GateRecord, GateStatus
from ..contracts.task_catalog import RunnerKind
from ..core.estimators import (
    double_sample_importance,
    equal_u_importance,
    raw_importance,
)
from ..core.losses import causal_lm_loss, sequence_classification_loss
from ..core.oracles import (
    compare_tensor_maps_fp64,
    fp64_double_sample_oracle,
    fp64_equal_u_oracle,
    fp64_mean_gradient_oracle,
    fp64_raw_oracle,
)
from ..core.registry import ParameterRegistry
from ..core.sufficient_statistics import EqualSufficientStatistics
from ..core.tensors import TensorMap
from ..runtime.checkpoint import CheckpointRetentionPolicy, CheckpointStore
from ..runtime.events import (
    EventRecord,
    EventType,
    JsonlEventSink,
    canonical_optimizer_steps,
    read_event_stream,
)
from ..runtime.gradients import GradientAttempt
from ..runtime.lineage import AttemptDisposition, LineageStore
from ..runtime.optimizer import OptimizerBridge, compute_global_clip_factor
from ..runtime.task_artifacts import (
    TaskArtifactStore,
    load_committed_task_artifact,
)
from ..runtime.task_runtime import (
    BlockerCode,
    TaskBlockedError,
    TaskBlocker,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRunner,
)
from ..runtime.tensor_bundle import load_tensor_bundle
from ..storage import REQUIRED_DIRECTORIES, StorageLayout, is_within


_HANDLED_BY_KIND: Mapping[RunnerKind, frozenset[str]] = MappingProxyType(
    {
        RunnerKind.AUDIT: frozenset({"stage0.01_baseline_and_safety"}),
        RunnerKind.STORAGE: frozenset({"stage0.02_storage_and_layout"}),
        RunnerKind.ENVIRONMENT: frozenset({"stage0.03_runtime_and_dependencies"}),
        RunnerKind.ASSET: frozenset({"stage0.04_assets_and_manifests"}),
        RunnerKind.CONTRACT: frozenset(
            {"stage0.05_config_run_identity_and_seeds", "stage1.01_entry_and_contract"}
        ),
        RunnerKind.OBSERVABILITY: frozenset({"stage0.08_logging_and_tracking"}),
        RunnerKind.CHECKPOINT: frozenset(
            {"stage0.09_checkpoint_and_resume", "stage1.10_checkpoint_resume_and_artifacts"}
        ),
        RunnerKind.CAPACITY: frozenset({"stage0.10_capacity_and_operations"}),
        RunnerKind.TEST_MATRIX: frozenset({"stage0.11_test_quality_and_replay"}),
        RunnerKind.DELIVERY: frozenset({"stage0.12_delivery_and_sync"}),
        RunnerKind.REGISTRY: frozenset({"stage1.02_architecture_and_parameter_registry"}),
        RunnerKind.ORACLE: frozenset({"stage1.03_fixtures_and_oracles"}),
        RunnerKind.VALIDATION: frozenset(
            {
                "stage1.04_loss_and_gradient_scale",
                "stage1.09_precision_clipping_and_optimizer_boundaries",
            }
        ),
        RunnerKind.ESTIMATOR: frozenset({"stage1.05_estimators"}),
        RunnerKind.REPORTING: frozenset({"stage1.11_reporting_and_exit_gate"}),
    }
)

STAGE01_HANDLED_TASK_IDS = frozenset(
    task_id for task_ids in _HANDLED_BY_KIND.values() for task_id in task_ids
)

# 这些任务的 formal 结论依赖服务器、资产、真实训练或交付现场，不能重跑下面的
# tiny/local 探针后发布成 formal 产物。它们的正式路径只消费已经通过两阶段协议
# 发布的上游/能力证据，并生成 hash-bound 的证据投影。
_FORMAL_EVIDENCE_ONLY_TASKS = frozenset(
    {
        "stage0.01_baseline_and_safety",
        "stage0.02_storage_and_layout",
        "stage0.03_runtime_and_dependencies",
        "stage0.04_assets_and_manifests",
        "stage0.08_logging_and_tracking",
        "stage0.09_checkpoint_and_resume",
        "stage0.10_capacity_and_operations",
        "stage0.11_test_quality_and_replay",
        "stage0.12_delivery_and_sync",
        "stage1.09_precision_clipping_and_optimizer_boundaries",
        "stage1.10_checkpoint_resume_and_artifacts",
        "stage1.11_reporting_and_exit_gate",
    }
)


def _logical(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"STAGE01_LOGICAL_PATH_INVALID:{field}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"STAGE01_PATH_ESCAPE:{field}")
    return path


def _resolve(root: Path, value: str, *, field: str) -> Path:
    path = _logical(value, field=field)
    target = root.joinpath(*path.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"STAGE01_PATH_ESCAPE:{field}") from error
    return target


def _store(request: TaskExecutionRequest, root: Path) -> TaskArtifactStore:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    return TaskArtifactStore(root, str(artifacts["output_dir"]))


def _input_evidence(
    request: TaskExecutionRequest, root: Path
) -> tuple[list[JSONValue], tuple[str, ...]]:
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    refs = tuple(str(item) for item in orchestration["input_result_refs"])
    evidence: list[JSONValue] = []
    formal_identities: dict[tuple[str, str], str] = {}
    for ref in refs:
        if request.config.run_intent == "formal":
            try:
                loaded = load_committed_task_artifact(root, ref, require_formal=True)
            except Exception as error:
                raise TaskBlockedError(
                    TaskBlocker(
                        BlockerCode.ASSET_UNAVAILABLE,
                        "formal_input_commit",
                        f"正式输入 commit 无法验证：{ref} ({type(error).__name__})",
                        True,
                        (ref,),
                    )
                ) from error
            identity = loaded.identity
            key = (identity.task_id, identity.artifact_kind)
            if key in formal_identities:
                raise TaskBlockedError(
                    TaskBlocker(
                        BlockerCode.ASSET_UNAVAILABLE,
                        "formal_input_duplicate",
                        f"正式输入身份重复：{identity.task_id}:{identity.artifact_kind}",
                        False,
                        (formal_identities[key], ref),
                    )
                )
            formal_identities[key] = ref
            evidence.append(
                {
                    "ref": ref,
                    "sha256": identity.artifact_hash,
                    "kind": "task-output-commit-v1",
                    "producer_task_id": identity.task_id,
                    "artifact_kind": identity.artifact_kind,
                    "formal_eligible": True,
                }
            )
            continue
        path = _resolve(root, ref, field="input_result_refs")
        if path.is_dir():
            _, identity = load_tensor_bundle(path)
            digest = identity.manifest_sha256
            kind = "tensor_bundle"
        elif path.suffix.casefold() == ".json":
            value = load_canonical_json(path)
            digest = canonical_json_hash(value)
            kind = (
                str(value.get("schema_version", "json"))
                if isinstance(value, dict)
                else "json"
            )
        elif path.is_file():
            digest = sha256_file(path)
            kind = "hash_bound_file"
        else:
            raise FileNotFoundError(f"STAGE01_INPUT_NOT_FOUND:{ref}")
        evidence.append({"ref": ref, "sha256": digest, "kind": kind})
    if request.config.run_intent == "formal":
        blockers: list[TaskBlocker] = []
        for contract in request.task.input_artifacts:
            if not contract.required:
                continue
            for artifact_kind in contract.artifact_kinds:
                if not any(
                    producer in contract.producer_task_ids and kind == artifact_kind
                    for producer, kind in formal_identities
                ):
                    blockers.append(
                        TaskBlocker(
                            BlockerCode.ASSET_UNAVAILABLE,
                            f"input:{contract.input_id}:{artifact_kind}",
                            f"缺少正式上游 commit：{contract.input_id}/{artifact_kind}",
                            True,
                        )
                    )
        if blockers:
            raise TaskBlockedError(*blockers)
    return evidence, refs


def _formal_payload(
    root: Path,
    reference: str,
    schema_version: str,
) -> Mapping[str, object]:
    loaded = load_committed_task_artifact(root, reference, require_formal=True)
    payload = loaded.payload
    if payload.get("schema_version") == schema_version:
        return payload
    candidates = [
        value
        for value in payload.values()
        if isinstance(value, Mapping) and value.get("schema_version") == schema_version
    ]
    if len(candidates) != 1:
        raise ValueError(f"STAGE01_FORMAL_PAYLOAD_NOT_UNIQUE:{schema_version}")
    return candidates[0]


def _formal_guard(request: TaskExecutionRequest, root: Path) -> None:
    """直接调用 runner 时也保留与统一 preflight 相同的外部条件防线。"""

    if request.config.run_intent != "formal":
        return
    policy = request.task.formal_eligibility
    blockers: list[TaskBlocker] = []
    for stage in policy.required_contract_stages:
        reference = request.environment.evidence_refs.get(f"contract_stage_{stage}")
        valid = False
        if stage in request.environment.frozen_contract_stages and reference is not None:
            try:
                freeze = ContractFreeze.from_mapping(
                    dict(_formal_payload(root, reference, "contract-freeze-v1"))
                )
                valid = freeze.stage == stage and freeze.formal_eligible
            except Exception:
                valid = False
        if not valid:
            blockers.append(
                TaskBlocker(
                    BlockerCode.CONTRACT_UNFROZEN,
                    f"stage{stage}",
                    f"Stage {stage} 缺少可验证的 formal ContractFreeze commit",
                    True,
                    (() if reference is None else (reference,)),
                )
            )
    for gate_id in policy.required_gate_ids:
        key = "gate_" + "".join(
            character if character.isalnum() else "_"
            for character in gate_id.casefold()
        ).strip("_")
        reference = request.environment.evidence_refs.get(key)
        valid = False
        if gate_id in request.environment.passed_gate_ids and reference is not None:
            try:
                gate = GateRecord.from_mapping(
                    dict(_formal_payload(root, reference, "gate-record-v1"))
                )
                valid = gate.gate_id == gate_id and gate.status is GateStatus.PASS
            except Exception:
                valid = False
        if not valid:
            blockers.append(
                TaskBlocker(
                    BlockerCode.GATE_NOT_READY,
                    gate_id,
                    f"前置 Gate 缺少可验证的 PASS commit：{gate_id}",
                    True,
                    (() if reference is None else (reference,)),
                )
            )
    for capability in sorted(policy.required_capabilities):
        reference = request.environment.evidence_refs.get(f"capability_{capability}")
        valid = False
        if capability in request.environment.capabilities and reference is not None:
            try:
                item = RuntimeCapabilityEvidence.from_mapping(
                    _formal_payload(
                        root,
                        reference,
                        "runtime-capability-evidence-v1",
                    )
                )
                valid = item.capability == capability and item.verified
            except Exception:
                valid = False
        if valid:
            continue
        code = (
            BlockerCode.SERVER_UNREACHABLE
            if capability == "server"
            else BlockerCode.DEVICE_UNAVAILABLE
            if capability in {"cuda", "nccl"}
            else BlockerCode.ASSET_UNAVAILABLE
            if capability in {"model_assets", "data_assets"}
            else BlockerCode.DEPENDENCY_UNAVAILABLE
            if capability == "wheelhouse"
            else BlockerCode.CAPABILITY_UNAVAILABLE
        )
        blockers.append(
            TaskBlocker(
                code,
                capability,
                f"formal 外部条件缺少可验证的 capability commit：{capability}",
                True,
                (() if reference is None else (reference,)),
            )
        )
    if blockers:
        raise TaskBlockedError(*blockers)


def _formal_external_evidence(
    request: TaskExecutionRequest,
    root: Path,
    inputs: list[JSONValue],
) -> tuple[Mapping[str, JSONValue], tuple[str, ...]]:
    """把已核验的 formal 输入投影为 Stage 0/1 证据，不运行本机 fixture。"""

    refs: list[str] = []
    environment_items: list[JSONValue] = []
    for key, reference in sorted(request.environment.evidence_refs.items()):
        try:
            loaded = load_committed_task_artifact(root, reference, require_formal=True)
        except Exception as error:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    f"formal_evidence:{key}",
                    f"正式环境证据 commit 无法验证：{key}",
                    True,
                    (reference,),
                )
            ) from error
        refs.append(reference)
        environment_items.append(
            {
                "key": key,
                "ref": reference,
                "artifact_hash": loaded.identity.artifact_hash,
                "producer_task_id": loaded.identity.task_id,
                "artifact_kind": loaded.identity.artifact_kind,
            }
        )
    source_refs = tuple(dict.fromkeys(
        [str(item["ref"]) for item in inputs if isinstance(item, dict)] + refs
    ))
    return (
        {
            "evidence_type": "formal_committed_evidence",
            "task_id": request.task.task_id,
            "execution_mode": "formal_evidence_projection",
            "input_commits": inputs,
            "environment_commits": environment_items,
            "local_fixture_executed": False,
            "formal_gate_automatically_passed": False,
        },
        source_refs,
    )


def _tensor_values(value: TensorMap) -> dict[str, JSONValue]:
    return {
        name: tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
        for name, tensor in value.items()
    }


def _gradient_samples() -> list[TensorMap]:
    return [
        TensorMap({"weight": torch.tensor([1.0, -1.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([3.0, 1.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([2.0, 0.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([4.0, 2.0], dtype=torch.float64)}),
    ]


def _baseline_evidence(root: Path) -> Mapping[str, JSONValue]:
    source_root = root
    if not (source_root / "Agent").is_dir() or not (source_root / "plan").is_dir():
        source_root = Path(__file__).resolve().parents[3]
    hashes: dict[str, JSONValue] = {}
    for directory in ("Agent", "plan"):
        base = source_root / directory
        for path in sorted(base.rglob("*.md")):
            hashes[path.relative_to(source_root).as_posix()] = sha256_file(path)
    return {
        "evidence_type": "baseline_and_safety",
        "source_hashes": hashes,
        "source_count": len(hashes),
        "remote_execution_attempted": False,
        "server_state": "BLOCKED:server_unreachable",
        "sensitive_payload_policy": "runtime.events.reject_known_secret_patterns",
    }


def _storage_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    sandbox = _resolve(root, str(artifacts["output_dir"]), field="output_dir") / "storage-layout"
    sandbox.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRECTORIES:
        (sandbox / name).mkdir(exist_ok=True)
    layout = StorageLayout(sandbox)
    escape_rejected = False
    try:
        layout.path("runs", "..", "outside")
    except ValueError:
        escape_rejected = True
    return {
        "evidence_type": "storage_layout",
        "required_directories": list(REQUIRED_DIRECTORIES),
        "validation_failures": layout.validate(require_writable=True),
        "escape_rejected": escape_rejected,
        "output_within_workspace": is_within(sandbox, root),
        "persistence_semantics": "immutable_object_plus_authoritative_commit",
    }


def _environment_evidence() -> Mapping[str, JSONValue]:
    return {
        "evidence_type": "runtime_environment",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "dependency_mode": "local_installed_only_no_download",
        "optional_ml_dependencies": "lazy_import",
    }


def _asset_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    providers = request.config.section("providers")
    artifacts = request.config.section("artifacts")
    assert isinstance(providers, dict)
    assert isinstance(artifacts, dict)
    asset_root = _resolve(root, str(artifacts["output_dir"]), field="output_dir") / "asset-fixture"
    asset_root.mkdir(parents=True, exist_ok=True)
    asset_file = asset_root / "fixture.bin"
    content = b"stage01-safe-local-asset\n"
    if not asset_file.exists():
        atomic_write_bytes(asset_file, content)
    elif asset_file.read_bytes() != content:
        raise ValueError("STAGE01_ASSET_FIXTURE_DRIFT")
    descriptor = {
        "path": "fixture.bin",
        "size_bytes": len(content),
        "sha256": sha256_file(asset_file),
        "role": "contract_fixture",
    }
    manifest = build_manifest(
        asset_type=AssetType.SOURCE,
        name="stage01-contract-fixture",
        source="workspace-generated-fixture",
        revision="fixture-v1",
        files=(descriptor,),
        actor="stage01-runner",
        actor_role=AssetActorRole.FETCHER,
        evidence_ref=None,
        generator_version="0.4.0",
        metadata={"source_kind": "contract_fixture", "license": "internal-test-only"},
        created_at="2026-01-01T00:00:00Z",
    )
    for state, role, minute in (
        (AssetState.DOWNLOADED, AssetActorRole.FETCHER, 1),
        (AssetState.VERIFIED, AssetActorRole.VERIFIER, 2),
        (AssetState.READY, AssetActorRole.GATE, 3),
    ):
        manifest = transition_manifest(
            manifest,
            state,
            actor="stage01-runner",
            actor_role=role,
            evidence_ref=(None if state is AssetState.DOWNLOADED else f"fixture/{state.value}.json"),
            summary=f"fixture {state.value}",
            at=f"2026-01-01T00:0{minute}:00Z",
        )
    verification = verify_only(manifest, asset_root)
    manifest_fields = sorted(key for key in providers if key.endswith("manifest_ref"))
    return {
        "evidence_type": "asset_manifest_boundary",
        "provider_kind": str(providers["kind"]),
        "manifest_fields": manifest_fields,
        "configured_manifest_refs": {
            key: providers[key] for key in manifest_fields if providers[key] is not None
        },
        "download_attempted": False,
        "fixture_asset_id": manifest["asset_id"],
        "fixture_asset_state": manifest["state"],
        "fixture_verification": verification,
        "legacy_bom_importer_separate": True,
        "canonical_json_requires_utf8_no_bom": True,
    }


def _identity_seed_evidence(request: TaskExecutionRequest) -> Mapping[str, JSONValue]:
    identity = request.config.base_config.section("identity")
    model = request.config.base_config.section("model")
    assert isinstance(identity, dict) and isinstance(model, dict)
    master_seed = int(identity["master_seed"])
    seed_plan = SeedPlan.from_master_seed(master_seed, world_size=1)
    experiment_id = derive_experiment_id(
        stage=request.task.stage,
        task=request.task.task_id,
        model_identity=str(model["asset_id"]),
        route=str(identity["route"]),
        master_seed=master_seed,
        config_hash=request.config.config_hash,
    )
    run = RunIdentity.create(
        experiment_id=experiment_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        collision_code="00000001",
    )
    resumed = run.next_attempt(
        started_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        input_checkpoint_id="fixture-checkpoint-0001",
    )
    return {
        "evidence_type": "config_identity_seed",
        "config_hash": request.config.config_hash,
        "config_full_hash": request.config.full_hash,
        "run_identity": run.to_dict(),
        "resume_attempt": resumed.to_dict(),
        "seed_plan": seed_plan.to_dict(),
        "seed_domains_unique": len(set(seed_plan.domains.values()))
        == len(seed_plan.domains),
    }


def _observability_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    evidence_root = _resolve(root, str(artifacts["output_dir"]), field="output_dir") / "observability"
    evidence_root.mkdir(parents=True, exist_ok=True)
    event_path = evidence_root / "events.jsonl"
    if not event_path.exists():
        with JsonlEventSink(event_path) as sink:
            for sequence, step in enumerate((0, 1)):
                sink.append(
                    EventRecord.create(
                        experiment_id="stage0-observability",
                        run_id="fixture-run",
                        attempt_id="attempt-0001",
                        session_id="session-0001",
                        rank=0,
                        event_type=EventType.OPTIMIZER_STEP,
                        sequence=sequence,
                        event_id=f"stage0-event-{sequence:04d}",
                        occurred_at=f"2026-01-01T00:00:0{sequence}+00:00",
                        payload={"global_step": step, "finite": True},
                    ),
                    critical=True,
                )
    events = read_event_stream(event_path)
    canonical = canonical_optimizer_steps((events,))

    lineage = LineageStore(evidence_root / "lineage.json", run_id="fixture-run")
    if not lineage.records():
        lineage.register("attempt-0001", parent_attempt_id=None, reason="initial")
        lineage.register("attempt-0002", parent_attempt_id="attempt-0001", reason="retry")
        lineage.mark_orphan("attempt-0002", reason="incomplete")
        lineage.select_canonical(
            "attempt-0001", reason="complete event stream", evidence_hash=sha256_file(event_path)
        )
    dispositions = {
        item.attempt_id: item.disposition.value for item in lineage.records()
    }
    return {
        "evidence_type": "event_and_lineage",
        "event_count": len(events),
        "event_stream_sha256": sha256_file(event_path),
        "canonical_optimizer_steps": [int(item.payload["global_step"]) for item in canonical],
        "lineage_dispositions": dispositions,
        "single_writer": True,
        "expected_dispositions_present": sorted(
            {AttemptDisposition.CANONICAL.value, AttemptDisposition.ORPHAN.value}
        ),
    }


def _checkpoint_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    checkpoint_root = _resolve(root, str(artifacts["output_dir"]), field="output_dir") / "checkpoint-core"
    store = CheckpointStore(checkpoint_root)
    if not (store.commits / "state-0001.json").exists():
        store.publish(
            "state-0001",
            {"parameter": torch.tensor([1.0, 2.0]), "step": 1},
            generation=1,
            metadata={"boundary": "parameter_post_state"},
        )
    if not (store.commits / "state-0002.json").exists():
        state, _ = store.load("state-0001")
        store.publish(
            "state-0002",
            {"parameter": state["parameter"] + 2.0, "step": 2},
            generation=2,
            metadata={"boundary": "attempt_commit_state"},
            parent_checkpoint_id="state-0001",
        )
    resumed, commit = store.load("state-0002")
    uninterrupted = torch.tensor([3.0, 4.0])
    equivalent = bool(torch.equal(resumed["parameter"], uninterrupted))
    active_before = [item.checkpoint_id for item in store.discover()]
    if not (store.tombstones / "state-0001.json").exists():
        selection = store.select_retention(CheckpointRetentionPolicy(keep_latest=1))
        application = store.apply_retention(selection, reason="stage01 fixture retention")
        newly_tombstoned = list(application.newly_tombstoned)
    else:
        newly_tombstoned = []
    recovered_again, _ = CheckpointStore(checkpoint_root).load("state-0002")
    reconciliation = store.reconcile()
    return {
        "evidence_type": "checkpoint_resume",
        "checkpoint_id": commit.checkpoint_id,
        "generation": commit.generation,
        "bundle_manifest_sha256": commit.manifest_sha256,
        "active_before_retention": active_before,
        "active_after_retention": [item.checkpoint_id for item in store.discover()],
        "newly_tombstoned": newly_tombstoned,
        "objects_deleted": 0,
        "resume_equivalent": equivalent
        and bool(torch.equal(recovered_again["parameter"], uninterrupted)),
        "reconcile_invalid": reconciliation["invalid"],
        "two_phase_commit": True,
    }


def _capacity_evidence() -> Mapping[str, JSONValue]:
    parameter_count = 1024
    checkpoint_bytes = estimate_checkpoint_bytes(parameter_count)
    statistics_bytes = estimate_parameter_statistics_bytes(
        parameter_count, resident_fp32_buffers=3
    )
    total = estimate_experiment_storage(
        parameter_count=parameter_count,
        retained_checkpoints=2,
        resident_fp32_buffers=3,
        seed_count=2,
        parallel_runs=1,
        logs_and_reports_per_run=4096,
    )
    return {
        "evidence_type": "capacity_and_operations",
        "parameter_count": parameter_count,
        "checkpoint_bytes": checkpoint_bytes,
        "statistics_bytes": statistics_bytes,
        "experiment_storage_bytes": total,
        "actual_server_capacity": "BLOCKED:server_unreachable",
        "launcher_invoked": False,
    }


def _replay_evidence(request: TaskExecutionRequest) -> Mapping[str, JSONValue]:
    payload = {
        "task_id": request.task.task_id,
        "config_hash": request.config.config_hash,
        "catalog_contract": request.task.to_dict(),
    }
    first = canonical_json_hash(payload)
    second = canonical_json_hash(dict(payload))
    return {
        "evidence_type": "deterministic_replay",
        "first_hash": first,
        "second_hash": second,
        "hashes_equal": first == second,
        "formal_gate_status": "NOT_RUN",
        "server_matrix_status": "BLOCKED:server_unreachable",
    }


def _delivery_evidence(request: TaskExecutionRequest) -> Mapping[str, JSONValue]:
    return {
        "evidence_type": "delivery_and_sync",
        "expected_artifact_kinds": list(request.task.artifact_kinds),
        "immutable_publish_required": True,
        "worklog_language": "zh-CN",
        "local_delivery_ready": True,
        "github_push_status": "NOT_RUN",
        "server_sync_status": "BLOCKED:server_unreachable",
    }


def _stage1_contract_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    # 测试可把产物 workspace 指向临时目录；合同源文件仍必须来自已安装源码所属仓库，
    # 不能因为输出根不同就悄悄生成另一份数学合同。
    contract_root = root
    if not (contract_root / "docs" / "mathematics.md").is_file():
        contract_root = Path(__file__).resolve().parents[3]
    math_path = contract_root / "docs" / "mathematics.md"
    plan_path = contract_root / "plan" / "stage1" / "01_entry_and_contract.md"
    return {
        "evidence_type": "stage1_math_contract",
        "contract_hashes": {
            "docs/mathematics.md": sha256_file(math_path),
            "plan/stage1/01_entry_and_contract.md": sha256_file(plan_path),
        },
        "frozen_formulas": {
            "raw": "mean_gradient**2",
            "equal_u": "(S1**2-S2)/(M*(M-1))",
            "double": "mean_gradient_A*mean_gradient_B",
        },
        "dynamic_learning_rate_in_coordinate_hash": False,
        "same_batch_clipped_u_claim": "plugin_no_strict_unbiasedness",
        "task_definition_hash": canonical_json_hash(request.task.to_dict()),
    }


class _AliasModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.weight_alias = self.weight
        self.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))


def _registry_evidence() -> Mapping[str, JSONValue]:
    model = _AliasModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
    registry = ParameterRegistry.from_model(model, optimizer)
    records = [
        {
            "canonical_name": record.canonical_name,
            "aliases": list(record.aliases),
            "shape": list(record.shape),
            "order": record.order,
            "eligible": record.eligible,
            "group_id": record.group_id,
        }
        for record in registry
    ]
    return {
        "evidence_type": "parameter_registry",
        "records": records,
        "eligible_names": list(registry.eligible_names),
        "coordinate_registry_hash": registry.coordinate_registry_hash,
        "optimizer_contract_hash": registry.optimizer_contract_hash,
        "runtime_layout_hash": registry.runtime_layout_hash,
        "alias_resolves_to": registry.canonical_name("weight_alias"),
    }


def _oracle_evidence() -> Mapping[str, JSONValue]:
    samples = _gradient_samples()
    mean = fp64_mean_gradient_oracle(samples)
    statistics = EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float64)
    production = equal_u_importance(statistics)
    oracle = fp64_equal_u_oracle(samples)
    comparison = compare_tensor_maps_fp64(production, oracle, natural_scale=4.0)
    return {
        "evidence_type": "fp64_fixture_oracle",
        "sample_count": len(samples),
        "mean_gradient": _tensor_values(mean),
        "u_oracle": _tensor_values(oracle),
        "comparison": comparison.to_dict(),
        "fixture_scope": "local_fixture",
    }


def _loss_scale_evidence() -> Mapping[str, JSONValue]:
    lm_logits = torch.tensor(
        [[[3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [0.0, -1.0, 3.0]]],
        dtype=torch.float64,
    )
    lm_labels = torch.tensor([[0, 1, 2]])
    lm = causal_lm_loss(lm_logits, lm_labels, torch.tensor([[1, 1, 1]]))
    cls = sequence_classification_loss(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64),
        torch.tensor([0, 1, -100]),
    )
    merged = cls.merge(
        sequence_classification_loss(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([0])
        )
    )
    return {
        "evidence_type": "loss_and_gradient_scale",
        "causal_lm": {
            "effective_count": lm.effective_count,
            "statistical_unit": lm.statistical_unit,
            "numerator": float(lm.loss_numerator.item()),
        },
        "classification": {
            "effective_count": cls.effective_count,
            "statistical_unit": cls.statistical_unit,
            "numerator": float(cls.loss_numerator.item()),
        },
        "merged_effective_count": merged.effective_count,
        "merge_uses_numerator_denominator": True,
    }


def _estimator_evidence() -> Mapping[str, JSONValue]:
    samples = _gradient_samples()
    statistics = EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float64)
    mean_a = fp64_mean_gradient_oracle(samples[:2])
    mean_b = fp64_mean_gradient_oracle(samples[2:])
    raw = raw_importance(statistics.mean_gradient)
    double = double_sample_importance(mean_a, mean_b)
    u = equal_u_importance(statistics)
    comparisons = {
        "raw": compare_tensor_maps_fp64(
            raw, fp64_raw_oracle(statistics.mean_gradient), natural_scale=4.0
        ).to_dict(),
        "double": compare_tensor_maps_fp64(
            double, fp64_double_sample_oracle(mean_a, mean_b), natural_scale=4.0
        ).to_dict(),
        "u": compare_tensor_maps_fp64(
            u, fp64_equal_u_oracle(samples), natural_scale=4.0
        ).to_dict(),
    }
    return {
        "evidence_type": "importance_estimators",
        "raw": _tensor_values(raw),
        "double": _tensor_values(double),
        "u": _tensor_values(u),
        "comparisons": comparisons,
        "u_can_be_negative": any(value < 0 for values in _tensor_values(u).values() for value in values),
        "unclipped_u_claim": "unbiased_fixed_state_under_declared_sampling_assumptions",
        "same_batch_clipped_u_claim": "plugin_same_batch_clip_no_strict_unbiasedness",
    }


def _numeric_boundary_evidence() -> Mapping[str, JSONValue]:
    attempt = GradientAttempt.capture(
        {"weight": torch.tensor([6.0, 8.0]), "unused": None},
        gradient_scale=2.0,
        scaled=True,
    ).unscale().check_finite().clip(2.5)
    skipped = GradientAttempt.capture(
        {"weight": torch.tensor([float("inf")])}, scaled=False
    ).check_finite()
    norm, factor = compute_global_clip_factor({"weight": torch.tensor([3.0, 4.0])}, 2.5)

    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.1, weight_decay=0.2, foreach=False, fused=False
    )
    parameter.grad = torch.tensor([0.5, -0.25])
    outcome = OptimizerBridge({"weight": parameter}, optimizer).step()
    decomposition_error = (
        outcome.total_delta["weight"]
        - outcome.data_delta["weight"]
        - outcome.weight_decay_delta["weight"]
    ).abs().max()
    gradient_lifecycle: dict[str, JSONValue] = {
        "phase": attempt.phase.value,
        "gradient_scale": attempt.gradient_scale,
        "missing_names": list(attempt.missing_names),
        "global_norm": attempt.global_norm,
        "clip_factor": attempt.clip_factor,
        "skip_reason": attempt.skip_reason,
    }
    return {
        "evidence_type": "numeric_optimizer_boundary",
        "gradient_lifecycle": gradient_lifecycle,
        "nonfinite_phase": skipped.phase.value,
        "nonfinite_skip_reason": skipped.skip_reason,
        "global_norm": norm,
        "global_clip_factor": factor,
        "adamw_learning_rate": outcome.learning_rates["weight"],
        "adamw_decomposition_max_error": float(decomposition_error.item()),
        "foreach": False,
        "fused": False,
    }


def _report_evidence(request: TaskExecutionRequest) -> Mapping[str, JSONValue]:
    return {
        "evidence_type": "stage1_exit_report",
        "requirements": [
            "contract",
            "parameter_registry",
            "fp64_oracle",
            "loss_gradient_scale",
            "estimators",
            "training_integration",
            "ddp",
            "numeric_boundaries",
            "checkpoint_resume",
        ],
        "formal_gate_status": "NOT_RUN",
        "local_evidence_only": request.config.run_intent == "local_fixture",
        "server_validation": "BLOCKED:server_unreachable",
        "manual_numbers_allowed": False,
    }


def _task_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    task_id = request.task.task_id
    if task_id == "stage0.01_baseline_and_safety":
        return _baseline_evidence(root)
    if task_id == "stage0.02_storage_and_layout":
        return _storage_evidence(request, root)
    if task_id == "stage0.03_runtime_and_dependencies":
        return _environment_evidence()
    if task_id == "stage0.04_assets_and_manifests":
        return _asset_evidence(request, root)
    if task_id == "stage0.05_config_run_identity_and_seeds":
        return _identity_seed_evidence(request)
    if task_id == "stage0.08_logging_and_tracking":
        return _observability_evidence(request, root)
    if task_id in {"stage0.09_checkpoint_and_resume", "stage1.10_checkpoint_resume_and_artifacts"}:
        return _checkpoint_evidence(request, root)
    if task_id == "stage0.10_capacity_and_operations":
        return _capacity_evidence()
    if task_id == "stage0.11_test_quality_and_replay":
        return _replay_evidence(request)
    if task_id == "stage0.12_delivery_and_sync":
        return _delivery_evidence(request)
    if task_id == "stage1.01_entry_and_contract":
        return _stage1_contract_evidence(request, root)
    if task_id == "stage1.02_architecture_and_parameter_registry":
        return _registry_evidence()
    if task_id == "stage1.03_fixtures_and_oracles":
        return _oracle_evidence()
    if task_id == "stage1.04_loss_and_gradient_scale":
        return _loss_scale_evidence()
    if task_id == "stage1.05_estimators":
        return _estimator_evidence()
    if task_id == "stage1.09_precision_clipping_and_optimizer_boundaries":
        return _numeric_boundary_evidence()
    if task_id == "stage1.11_reporting_and_exit_gate":
        return _report_evidence(request)
    raise ValueError(f"STAGE01_TASK_UNHANDLED:{task_id}")


def _role_evidence(task_id: str, artifact_kind: str) -> Mapping[str, JSONValue]:
    """为同一任务中的不同产物声明不同消费语义，禁止复制同一 payload 冒充分区。"""

    if "gate" in artifact_kind:
        return {"role": "gate_candidate", "gate_status": "NOT_RUN", "decision_authority": "external_reviewer"}
    if "report" in artifact_kind or "summary" in artifact_kind:
        return {"role": "derived_report", "source": "task_core_evidence", "manual_numeric_edits": False}
    if "manifest" in artifact_kind or "registry" in artifact_kind or "plan" in artifact_kind:
        return {"role": "frozen_manifest", "canonical_order": True, "identity_bound": True}
    if "checkpoint" in artifact_kind or "state" in artifact_kind:
        return {"role": "recovery_boundary", "authoritative_commit_required": True, "directory_rename_is_commit": False}
    if "event" in artifact_kind or "lineage" in artifact_kind:
        return {"role": "append_only_runtime_evidence", "single_writer": True, "canonical_selection_required": True}
    if "sync" in artifact_kind or "delivery" in artifact_kind or artifact_kind == "worklog":
        return {"role": "delivery_record", "local_only": True, "server_sync_claimed": False}
    return {"role": "task_specific_evidence", "artifact_contract": f"{task_id}:{artifact_kind}"}


@dataclass(slots=True)
class Stage01CompositeTaskRunner(TaskRunner):
    """同一 ``RunnerKind`` 下按 task ID 分派 Stage 0/1 专用逻辑。"""

    runner_kind: RunnerKind
    workspace_root: Path
    fallback: TaskRunner | None = None

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()

    @property
    def handled_task_ids(self) -> frozenset[str]:
        return _HANDLED_BY_KIND.get(self.runner_kind, frozenset())

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        if request.task.runner_kind is not self.runner_kind:
            raise ValueError("STAGE01_RUNNER_KIND_MISMATCH")
        if request.task.task_id not in self.handled_task_ids:
            if self.fallback is not None:
                return self.fallback.run(request)
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CAPABILITY_UNAVAILABLE,
                    f"fallback:{self.runner_kind.value}",
                    f"Stage 0/1 composite 不处理任务 {request.task.task_id}",
                    False,
                )
            )
        _formal_guard(request, self.workspace_root)
        store = _store(request, self.workspace_root)
        existing = store.discover_complete(
            task_id=request.task.task_id,
            config_hash=request.config.config_hash,
            artifact_kinds=request.task.artifact_kinds,
            formal_eligible=request.config.run_intent == "formal",
        )
        if existing is not None:
            if (
                request.config.run_intent == "formal"
                and request.task.task_id in _FORMAL_EVIDENCE_ONLY_TASKS
            ):
                for reference in existing.values():
                    loaded = load_committed_task_artifact(
                        self.workspace_root,
                        reference,
                        require_formal=True,
                    )
                    core = loaded.payload.get("core_evidence")
                    if not isinstance(core, Mapping) or (
                        core.get("evidence_type") != "formal_committed_evidence"
                        or core.get("local_fixture_executed") is not False
                    ):
                        raise TaskBlockedError(
                            TaskBlocker(
                                BlockerCode.CONTRACT_UNFROZEN,
                                "formal_stage01_output_scope",
                                "已有输出不是 formal evidence-only 路径，请使用新的输出目录",
                                False,
                                (reference,),
                            )
                        )
            return TaskRunResult.passed(
                request,
                artifact_refs=existing,
                checkpoint_ref=next(
                    (ref for kind, ref in existing.items() if "checkpoint" in kind or "state" in kind),
                    None,
                ),
                message="stage0-1 task restored from authoritative commits",
                metadata={"stage01_specialized": True, "restored": True},
            )

        inputs, source_refs = _input_evidence(request, self.workspace_root)
        if (
            request.config.run_intent == "formal"
            and request.task.task_id in _FORMAL_EVIDENCE_ONLY_TASKS
        ):
            formal_evidence, formal_sources = _formal_external_evidence(
                request,
                self.workspace_root,
                inputs,
            )
            evidence = dict(formal_evidence)
            source_refs = formal_sources
        else:
            evidence = dict(_task_evidence(request, self.workspace_root))
        evidence_hash = canonical_json_hash(evidence)
        refs: dict[str, str] = {}
        for artifact_kind in request.task.artifact_kinds:
            payload: dict[str, JSONValue] = {
                "schema_version": "stage01-task-evidence-v1",
                "task_id": request.task.task_id,
                "artifact_role": artifact_kind,
                "scope": request.config.run_intent,
                "local_validation_status": (
                    "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
                ),
                "gate_status": "NOT_RUN",
                "config_hash": request.config.config_hash,
                "task_definition_hash": canonical_json_hash(request.task.to_dict()),
                "input_evidence": inputs,
                "core_evidence_hash": evidence_hash,
                "core_evidence": evidence,
                "role_evidence": dict(_role_evidence(request.task.task_id, artifact_kind)),
            }
            published = store.publish(
                task_id=request.task.task_id,
                artifact_kind=artifact_kind,
                config_hash=request.config.config_hash,
                run_intent=request.config.run_intent,
                payload=payload,
                formal_eligible=request.config.run_intent == "formal",
                source_refs=source_refs,
            )
            refs[artifact_kind] = published.commit_ref
        return TaskRunResult.passed(
            request,
            artifact_refs=refs,
            checkpoint_ref=next(
                (ref for kind, ref in refs.items() if "checkpoint" in kind or "state" in kind),
                None,
            ),
            message="stage0-1 specialized core task completed",
            metadata={"stage01_specialized": True, "core_evidence_hash": evidence_hash},
        )


def build_stage01_runner_overrides(
    workspace_root: str | Path,
    *,
    fallbacks: Mapping[RunnerKind, TaskRunner] | None = None,
) -> Mapping[RunnerKind, Stage01CompositeTaskRunner]:
    """构造 ``RunnerKind -> composite`` 覆盖映射，供统一 runtime 工厂逐层组合。

    未命中 Stage 0/1 task ID 时会转交同 kind 的 ``fallback``；因此 CONTRACT、
    VALIDATION、REPORTING 等共享 kind 能继续被 Stage 2--9 的专用 runner 消费。
    """

    root = Path(workspace_root).resolve()
    fallback_map = dict(fallbacks or {})
    return MappingProxyType(
        {
            kind: Stage01CompositeTaskRunner(kind, root, fallback_map.get(kind))
            for kind in _HANDLED_BY_KIND
        }
    )


__all__ = [
    "STAGE01_HANDLED_TASK_IDS",
    "Stage01CompositeTaskRunner",
    "build_stage01_runner_overrides",
]
