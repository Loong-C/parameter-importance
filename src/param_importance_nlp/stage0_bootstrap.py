"""Bootstrap hash-bound formal evidence for the already completed G0--G2 gates.

The historical Stage 0 reports are useful source evidence, but a path to a JSON
file is not sufficient to unlock :class:`TaskRuntime`: formal preflight requires
an immutable task-output commit containing a strict ``ContractFreeze``,
``GateRecord`` or ``RuntimeCapabilityEvidence``.  This module validates the
reports and a current runtime snapshot, publishes those commits under DATA_ROOT,
then executes the canonical S0.1--S0.3 tasks in order.

It intentionally stops before G3.  Asset capabilities can only be published from
the independent acquisition/verification/gate-only reports produced by S0.4.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
from typing import Mapping

import torch

from .atomic import sha256_file
from .contracts import (
    ContractFreeze,
    ContractState,
    GateRecord,
    GateStatus,
    ResolvedConfig,
    RuntimeCapabilityEvidence,
    canonical_json_hash,
    load_canonical_json,
    loads_strict_json,
    write_canonical_json,
)
from .contracts.config_v2 import ResolvedConfigV2
from .contracts.jsonio import JSONValue
from .experiments import build_default_task_runtime
from .runtime import TaskArtifactStore, TaskRuntimeEnvironment


_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_REPOSITORY = Path("/home/sophgo13/cjl/parameter-importance")
_EXPECTED_DATA_ROOT = Path("/home/sophgo13/cjl/storage/parameter-importance")
_SOURCE_REPORT_REFS = (
    "reports/stage0/g0-baseline-20260719.json",
    "reports/stage0/g0-g-gpu-final-20260807.json",
    "reports/stage0/g1-storage-mechanism-20260804.json",
    "reports/stage0/g1-persistence-decision-20260719.json",
    "reports/stage0/g2-environment-final-20260804.json",
)
_BOOTSTRAP_TASK_IDS = (
    "stage0.01_baseline_and_safety",
    "stage0.02_storage_and_layout",
    "stage0.03_runtime_and_dependencies",
)


class Stage0BootstrapError(RuntimeError):
    """The existing G0--G2 evidence cannot safely unlock formal execution."""


@dataclass(frozen=True, slots=True)
class Stage0SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str
    worktree_clean: bool

    def __post_init__(self) -> None:
        root = Path(self.repository).resolve()
        if not root.is_dir():
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_SOURCE_ROOT_INVALID")
        if _GIT_COMMIT_RE.fullmatch(self.git_commit) is None:
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_GIT_COMMIT_INVALID")
        if not isinstance(self.git_branch, str) or not self.git_branch:
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_GIT_BRANCH_INVALID")
        if type(self.worktree_clean) is not bool:
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_WORKTREE_STATE_INVALID")
        object.__setattr__(self, "repository", root)


@dataclass(frozen=True, slots=True)
class Stage0RuntimeSnapshot:
    checked_at: str
    hostname: str
    boot_id: str
    kernel: str
    data_root: str
    python_prefix: str
    python_version: str
    torch_version: str
    torch_cuda_runtime: str | None
    cuda_device_count: int
    allowed_gpu_uuids: tuple[str, ...]
    git_verified: bool
    server_verified: bool
    wheelhouse_verified: bool
    cuda_verified: bool
    nccl_verified: bool

    def __post_init__(self) -> None:
        parsed = _parse_timestamp(self.checked_at, field="snapshot.checked_at")
        object.__setattr__(
            self,
            "checked_at",
            parsed.astimezone(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        for field_name in (
            "hostname",
            "boot_id",
            "kernel",
            "data_root",
            "python_prefix",
            "python_version",
            "torch_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise Stage0BootstrapError(
                    f"STAGE0_BOOTSTRAP_SNAPSHOT_FIELD_INVALID:{field_name}"
                )
        if self.torch_cuda_runtime is not None and not self.torch_cuda_runtime:
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_CUDA_RUNTIME_INVALID")
        if (
            isinstance(self.cuda_device_count, bool)
            or not isinstance(self.cuda_device_count, int)
            or self.cuda_device_count < 0
        ):
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_DEVICE_COUNT_INVALID")
        uuids = tuple(self.allowed_gpu_uuids)
        if any(not isinstance(item, str) or not item.startswith("GPU-") for item in uuids):
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_GPU_UUID_INVALID")
        if len(uuids) != len(set(uuids)):
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_GPU_UUID_DUPLICATE")
        object.__setattr__(self, "allowed_gpu_uuids", uuids)
        for field_name in (
            "git_verified",
            "server_verified",
            "wheelhouse_verified",
            "cuda_verified",
            "nccl_verified",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise Stage0BootstrapError(
                    f"STAGE0_BOOTSTRAP_SNAPSHOT_FLAG_INVALID:{field_name}"
                )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": "stage0-runtime-bootstrap-snapshot-v1",
            "checked_at": self.checked_at,
            "hostname": self.hostname,
            "boot_id": self.boot_id,
            "kernel": self.kernel,
            "data_root": self.data_root,
            "python_prefix": self.python_prefix,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_cuda_runtime": self.torch_cuda_runtime,
            "cuda_device_count": self.cuda_device_count,
            "allowed_gpu_uuids": list(self.allowed_gpu_uuids),
            "git_verified": self.git_verified,
            "server_verified": self.server_verified,
            "wheelhouse_verified": self.wheelhouse_verified,
            "cuda_verified": self.cuda_verified,
            "nccl_verified": self.nccl_verified,
        }


@dataclass(frozen=True, slots=True)
class Stage0BootstrapResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, Mapping[str, str]]
    index_ref: str
    config_refs: Mapping[str, str]


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise Stage0BootstrapError(f"STAGE0_BOOTSTRAP_TIMESTAMP_INVALID:{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage0BootstrapError(
            f"STAGE0_BOOTSTRAP_TIMESTAMP_INVALID:{field}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage0BootstrapError(f"STAGE0_BOOTSTRAP_TIMESTAMP_NAIVE:{field}")
    return parsed


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def inspect_stage0_source(repository: str | Path, *, expected_commit: str) -> Stage0SourceBinding:
    root = Path(repository).resolve(strict=True)
    if _GIT_COMMIT_RE.fullmatch(expected_commit) is None:
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_EXPECTED_COMMIT_INVALID")
    top = _run_git(root, "rev-parse", "--show-toplevel")
    head = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "branch", "--show-current")
    tracked = _run_git(root, "ls-files", "--error-unmatch", "--", *_SOURCE_REPORT_REFS)
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if any(item.returncode != 0 for item in (top, head, branch, tracked, status)):
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_SOURCE_GIT_PROBE_FAILED")
    if Path(top.stdout.strip()).resolve() != root:
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_SOURCE_GIT_ROOT_MISMATCH")
    if head.stdout.strip() != expected_commit:
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_SOURCE_HEAD_MISMATCH")
    clean = not status.stdout.strip()
    if not clean:
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_FORMAL_SOURCE_DIRTY")
    return Stage0SourceBinding(root, expected_commit, branch.stdout.strip(), clean)


def inspect_stage0_runtime(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    checked_at: str,
) -> Stage0RuntimeSnapshot:
    """Probe the current server/venv without network access or state changes."""

    root = Path(data_root).resolve(strict=True)
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    if not boot_path.is_file():
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_BOOT_ID_UNAVAILABLE")
    boot_id = boot_path.read_text(encoding="utf-8").strip()

    g2 = _load_report(binding.repository, "reports/stage0/g2-environment-final-20260804.json")
    candidate = _mapping(g2.get("candidate"), field="g2.candidate")
    expected_prefix = Path(_string(candidate.get("path"), field="g2.candidate.path")).resolve()
    prefix = Path(sys.prefix).resolve()
    wheelhouse_verified = (
        prefix == expected_prefix
        and platform.python_version() == candidate.get("python")
        and torch.__version__ == candidate.get("torch")
        and torch.version.cuda == candidate.get("torch_cuda_runtime")
    )
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheelhouse_verified = wheelhouse_verified and pip_check.returncode == 0

    g0g = _load_report(binding.repository, _SOURCE_REPORT_REFS[1])
    allowed = _sequence(g0g.get("allowed_gpus"), field="g0g.allowed_gpus")
    expected_uuids = tuple(
        _string(_mapping(item, field="g0g.allowed_gpu").get("uuid"), field="g0g.uuid")
        for item in allowed
    )
    cuda_verified = bool(torch.cuda.is_available()) and torch.cuda.device_count() == 4
    if cuda_verified:
        for index in range(4):
            tensor = torch.tensor([float(index + 1)], device=f"cuda:{index}")
            cuda_verified = cuda_verified and float(tensor.cpu().item()) == float(index + 1)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gate = _mapping(g0g.get("gate"), field="g0g.gate")
    smoke = _mapping(g0g.get("cuda_nccl_smoke"), field="g0g.cuda_nccl_smoke")
    nccl_verified = (
        cuda_verified
        and gate.get("boot_id") == boot_id
        and smoke.get("status") == "PASS"
        and smoke.get("world_size") == 4
        and smoke.get("torch") == torch.__version__
        and smoke.get("cuda_runtime") == torch.version.cuda
    )
    return Stage0RuntimeSnapshot(
        checked_at=checked_at,
        hostname=platform.node(),
        boot_id=boot_id,
        kernel=platform.release(),
        data_root=root.as_posix(),
        python_prefix=prefix.as_posix(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_cuda_runtime=torch.version.cuda,
        cuda_device_count=torch.cuda.device_count(),
        allowed_gpu_uuids=expected_uuids,
        git_verified=binding.worktree_clean,
        server_verified=(
            binding.repository == _EXPECTED_REPOSITORY
            and root == _EXPECTED_DATA_ROOT
            and root.is_dir()
        ),
        wheelhouse_verified=wheelhouse_verified,
        cuda_verified=cuda_verified,
        nccl_verified=nccl_verified,
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage0BootstrapError(f"STAGE0_BOOTSTRAP_REPORT_FIELD_INVALID:{field}")
    return value


def _sequence(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise Stage0BootstrapError(f"STAGE0_BOOTSTRAP_REPORT_FIELD_INVALID:{field}")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage0BootstrapError(f"STAGE0_BOOTSTRAP_REPORT_FIELD_INVALID:{field}")
    return value


def _load_report(repository: Path, reference: str) -> dict[str, JSONValue]:
    path = repository.joinpath(*PurePosixPath(reference).parts)
    # G0--G2 predate the canonical one-line publisher and are deliberately kept
    # byte-for-byte as historical evidence.  Parse them with the strict UTF-8 /
    # duplicate-key reader, then bind their original file SHA-256 in the new
    # canonical attestation instead of rewriting history in place.
    value = loads_strict_json(path.read_bytes())
    if not isinstance(value, dict):
        raise Stage0BootstrapError(f"STAGE0_BOOTSTRAP_REPORT_ROOT_INVALID:{reference}")
    return value


def _source_file_map(repository: Path) -> dict[str, str]:
    return {reference: sha256_file(repository / reference) for reference in _SOURCE_REPORT_REFS}


def _contract_hashes(repository: Path) -> tuple[dict[str, str], dict[str, str]]:
    schema_paths = sorted((repository / "schemas" / "shared").glob("*.json"))
    schema_paths += sorted((repository / "schemas" / "stage0").glob("*.json"))
    schema_paths += sorted((repository / "schemas").glob("stage0*.json"))
    source_paths = sorted((repository / "plan" / "stage0").glob("*.md"))
    source_paths += [repository / "plan" / "general_plan.md"]
    source_paths += sorted((repository / "src" / "param_importance_nlp").rglob("*.py"))
    for path in (*schema_paths, *source_paths):
        if not path.is_file() or path.is_symlink():
            raise Stage0BootstrapError("STAGE0_BOOTSTRAP_CONTRACT_SOURCE_INVALID")
    return (
        {
            path.relative_to(repository).as_posix(): sha256_file(path)
            for path in schema_paths
        },
        {
            path.relative_to(repository).as_posix(): sha256_file(path)
            for path in source_paths
        },
    )


def validate_existing_g0_g2(
    *,
    binding: Stage0SourceBinding,
    snapshot: Stage0RuntimeSnapshot,
) -> dict[str, dict[str, JSONValue]]:
    """Validate source reports against one current runtime snapshot."""

    repository = binding.repository
    g0 = _load_report(repository, _SOURCE_REPORT_REFS[0])
    g0g = _load_report(repository, _SOURCE_REPORT_REFS[1])
    g1 = _load_report(repository, _SOURCE_REPORT_REFS[2])
    g1d = _load_report(repository, _SOURCE_REPORT_REFS[3])
    g2 = _load_report(repository, _SOURCE_REPORT_REFS[4])
    if any(item.get("secrets_included") is not False for item in (g0, g0g, g2)):
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_SOURCE_REPORT_SECRET_FLAG_INVALID")

    g0_subgates = _mapping(g0.get("subgates"), field="g0.subgates")
    g0c = _mapping(g0_subgates.get("G0-C"), field="g0.G0-C")
    if g0c.get("status") != "PASS":
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_G0_C_NOT_PASS")
    g0c_evidence = _mapping(g0c.get("evidence"), field="g0.G0-C.evidence")
    roots = _mapping(g0c_evidence.get("authorized_roots"), field="g0.authorized_roots")
    if (
        roots.get("repository") != _EXPECTED_REPOSITORY.as_posix()
        or roots.get("data_root") != _EXPECTED_DATA_ROOT.as_posix()
        or roots.get("read_write") is not True
    ):
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_G0_C_ROOTS_INVALID")

    g0g_gate = _mapping(g0g.get("gate"), field="g0g.gate")
    health = _mapping(g0g.get("health"), field="g0g.health")
    smoke = _mapping(g0g.get("cuda_nccl_smoke"), field="g0g.cuda_nccl_smoke")
    allowed = _sequence(g0g.get("allowed_gpus"), field="g0g.allowed_gpus")
    allowed_uuids = tuple(
        _string(_mapping(item, field="g0g.allowed_gpu").get("uuid"), field="g0g.uuid")
        for item in allowed
    )
    if (
        g0g_gate.get("id") != "G0-G"
        or g0g_gate.get("status") != "PASS"
        or g0g_gate.get("boot_id") != snapshot.boot_id
        or g0g_gate.get("kernel") != snapshot.kernel
        or allowed_uuids != snapshot.allowed_gpu_uuids
        or health.get("nvml_device_count") != 4
        or health.get("pytorch_device_count") != 4
        or health.get("volatile_uncorrectable_ecc_each") != 0
        or health.get("aggregate_uncorrectable_ecc_each") != 0
        or health.get("compute_clients_after_validation") != 0
        or smoke.get("status") != "PASS"
        or smoke.get("world_size") != 4
    ):
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_G0_G_CURRENT_RUNTIME_MISMATCH")

    checked_at = _parse_timestamp(snapshot.checked_at, field="snapshot.checked_at")
    expires_at = _parse_timestamp(g1d.get("expires_at"), field="g1d.expires_at")
    if (
        g1.get("status") != "PASS"
        or g1.get("g1_d_status") != "PASS"
        or g1.get("g1_overall_status") != "PASS"
        or g1d.get("gate") != "G1-D"
        or g1d.get("status") != "PASS"
        or g1d.get("satisfaction") != "TIME_BOUNDED_RISK_ACCEPTANCE"
        or checked_at >= expires_at.astimezone(timezone.utc)
    ):
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_G1_NOT_CURRENT_PASS")

    g2_gate = _mapping(g2.get("gate"), field="g2.gate")
    g2_candidate = _mapping(g2.get("candidate"), field="g2.candidate")
    g2_gpu = _mapping(g2.get("gpu_gate_reference"), field="g2.gpu_gate_reference")
    if (
        g2.get("subtask_status") != "COMPLETE"
        or g2_gate.get("id") != "G2"
        or g2_gate.get("status") != "PASS"
        or g2_gate.get("training_eligible") is not True
        or g2_gpu.get("status") != "PASS"
        or g2_gpu.get("allowed_gpu_count") != 4
        or g2_candidate.get("environment_id") is None
        or g2_candidate.get("path") != snapshot.python_prefix
        or g2_candidate.get("torch") != snapshot.torch_version
        or g2_candidate.get("torch_cuda_runtime") != snapshot.torch_cuda_runtime
    ):
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_G2_RUNTIME_MISMATCH")

    return {
        "g0_c": {
            "source_status": "PASS",
            "authorized_repository": str(roots["repository"]),
            "authorized_data_root": str(roots["data_root"]),
        },
        "g0_g": {
            "source_status": "PASS",
            "boot_id": snapshot.boot_id,
            "gpu_count": 4,
            "allowed_gpu_uuids": list(allowed_uuids),
            "nccl_smoke_status": "PASS",
        },
        "g1": {
            "source_status": "PASS",
            "persistence_satisfaction": str(g1d["satisfaction"]),
            "persistence_expires_at": str(g1d["expires_at"]),
            "independent_failure_domain_copy_count": int(
                g1d["independent_failure_domain_copy_count"]
            ),
        },
        "g2": {
            "source_status": "PASS",
            "environment_id": str(g2_candidate["environment_id"]),
            "torch": snapshot.torch_version,
            "cuda_runtime": snapshot.torch_cuda_runtime,
        },
    }


def _publish(
    store: TaskArtifactStore,
    *,
    task_id: str,
    artifact_kind: str,
    config_hash: str,
    payload: Mapping[str, JSONValue],
    source_refs: tuple[str, ...] = (),
) -> str:
    return store.publish(
        task_id=task_id,
        artifact_kind=artifact_kind,
        config_hash=config_hash,
        run_intent="formal",
        payload=payload,
        formal_eligible=True,
        source_refs=source_refs,
    ).commit_ref


def _formal_base_config(repository: Path, task_id: str) -> ResolvedConfig:
    value = deepcopy(
        load_canonical_json(repository / "configs/local-fixtures/resolved-config-v1.json")
    )
    if not isinstance(value, dict):
        raise Stage0BootstrapError("STAGE0_BOOTSTRAP_BASE_CONFIG_INVALID")
    task_number = int(task_id.split(".", 1)[1].split("_", 1)[0])
    identity = _mapping(value.get("identity"), field="base.identity")
    runtime = _mapping(value.get("runtime"), field="base.runtime")
    identity.update(  # type: ignore[attr-defined]
        {
            "stage": 0,
            "task": task_id,
            "route": f"stage0-bootstrap-{task_number:02d}",
            "run_intent": "formal",
            "formal_eligible": True,
        }
    )
    runtime["allow_dirty_worktree"] = False  # type: ignore[index]
    return ResolvedConfig.from_mapping(value)


def build_stage0_formal_config(
    repository: str | Path,
    *,
    task_id: str,
    input_refs: tuple[str, ...],
    output_dir: str,
    base_overrides: Mapping[str, object] | None = None,
    v2_overrides: Mapping[str, object] | None = None,
) -> ResolvedConfigV2:
    """Build one canonical formal Stage 0 task configuration.

    The bootstrap and all later Stage 0 orchestrators use the same constructor
    so a task cannot silently acquire a different identity/runtime baseline.
    Task-specific runners still own their scientific execution semantics.
    """

    base = _formal_base_config(Path(repository).resolve(strict=True), task_id)
    if base_overrides is not None:
        base = ResolvedConfig.resolve(base.to_dict(), base_overrides)
    overrides: dict[str, object] = {
        "orchestration": {"input_result_refs": list(input_refs)},
        "artifacts": {"output_dir": output_dir},
    }
    if v2_overrides is not None:
        for section, raw_fields in v2_overrides.items():
            if not isinstance(section, str) or not isinstance(raw_fields, Mapping):
                raise Stage0BootstrapError("STAGE0_FORMAL_V2_OVERRIDE_INVALID")
            fields = dict(raw_fields)
            if section == "orchestration" and "input_result_refs" in fields:
                raise Stage0BootstrapError("STAGE0_FORMAL_INPUT_REFS_OVERRIDE_FORBIDDEN")
            if section == "artifacts" and "output_dir" in fields:
                raise Stage0BootstrapError("STAGE0_FORMAL_OUTPUT_DIR_OVERRIDE_FORBIDDEN")
            current = overrides.setdefault(section, {})
            if not isinstance(current, dict):  # pragma: no cover - constructor invariant
                raise Stage0BootstrapError("STAGE0_FORMAL_OVERRIDE_INTERNAL_INVALID")
            current.update(fields)
    return ResolvedConfigV2.resolve(
        base,
        task_id=task_id,
        overrides=overrides,
    )


def bootstrap_formal_stage0(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    snapshot: Stage0RuntimeSnapshot,
) -> Stage0BootstrapResult:
    root = Path(data_root).resolve(strict=True)
    validated = validate_existing_g0_g2(binding=binding, snapshot=snapshot)
    source_hashes = _source_file_map(binding.repository)
    schema_hashes, contract_source_hashes = _contract_hashes(binding.repository)
    attestation_payload: dict[str, JSONValue] = {
        "schema_version": "stage0-bootstrap-source-attestation-v1",
        "generator_git_commit": binding.git_commit,
        "git_branch": binding.git_branch,
        "worktree_clean": binding.worktree_clean,
        "source_report_sha256": source_hashes,
        "runtime_snapshot": snapshot.to_dict(),
        "validated_gates": validated,
    }
    config_hash = canonical_json_hash(attestation_payload)
    output_dir = f"evidence/stage0/bootstrap/{binding.git_commit}"
    store = TaskArtifactStore(root, output_dir)
    source_ref = _publish(
        store,
        task_id="stage0.01_baseline_and_safety",
        artifact_kind="source_attestation",
        config_hash=config_hash,
        payload=attestation_payload,
    )
    freeze = ContractFreeze(
        contract_id="stage0.contract.formal-v1",
        stage=0,
        scope="formal",
        state=ContractState.FROZEN,
        formula_version="stage0-infrastructure-formal-v1",
        config_hash=config_hash,
        schema_hashes=schema_hashes,
        source_hashes=contract_source_hashes,
        required_gate_ids=(
            "stage0.G0-C",
            "stage0.G0-G",
            "stage0.G1",
            "stage0.G2",
        ),
        frozen_at=snapshot.checked_at,
    )
    freeze_ref = _publish(
        store,
        task_id="stage0.01_baseline_and_safety",
        artifact_kind="contract_freeze",
        config_hash=config_hash,
        payload=freeze.to_dict(),
        source_refs=(source_ref,),
    )

    gate_specs = {
        "stage0.G0-C": ("gate_g0_c", validated["g0_c"]),
        "stage0.G0-G": ("gate_g0_g", validated["g0_g"]),
        "stage0.G1": ("gate_g1", validated["g1"]),
        "stage0.G2": ("gate_g2", validated["g2"]),
    }
    evidence_refs: dict[str, str] = {"contract_stage_0": freeze_ref}
    for gate_id, (artifact_kind, measured) in gate_specs.items():
        gate = GateRecord(
            gate_id=gate_id,
            stage=0,
            status=GateStatus.PASS,
            checked_at=snapshot.checked_at,
            measured=measured,
            threshold={"required_status": "PASS"},
            evidence_refs=(source_ref,),
        )
        reference = _publish(
            store,
            task_id="stage0.01_baseline_and_safety",
            artifact_kind=artifact_kind,
            config_hash=config_hash,
            payload=gate.to_dict(),
            source_refs=(source_ref,),
        )
        key = "gate_" + "".join(
            character if character.isalnum() else "_"
            for character in gate_id.casefold()
        ).strip("_")
        evidence_refs[key] = reference

    capability_specs: dict[str, tuple[bool, tuple[str, ...], dict[str, JSONValue]]] = {
        "git": (
            snapshot.git_verified,
            (source_ref,),
            {
                "git_commit": binding.git_commit,
                "git_branch": binding.git_branch,
                "worktree_clean": binding.worktree_clean,
            },
        ),
        "server": (
            snapshot.server_verified,
            (evidence_refs["gate_stage0_g0_c"],),
            {"hostname": snapshot.hostname, "data_root": snapshot.data_root},
        ),
        "wheelhouse": (
            snapshot.wheelhouse_verified,
            (evidence_refs["gate_stage0_g2"],),
            {
                "python_prefix": snapshot.python_prefix,
                "python": snapshot.python_version,
                "torch": snapshot.torch_version,
            },
        ),
        "cuda": (
            snapshot.cuda_verified,
            (evidence_refs["gate_stage0_g0_g"], evidence_refs["gate_stage0_g2"]),
            {
                "device_count": snapshot.cuda_device_count,
                "allowed_gpu_uuids": list(snapshot.allowed_gpu_uuids),
            },
        ),
        "nccl": (
            snapshot.nccl_verified,
            (evidence_refs["gate_stage0_g0_g"], evidence_refs["gate_stage0_g2"]),
            {"world_size": 4, "scope": "G0-G environment communication smoke"},
        ),
    }
    verified_capabilities: set[str] = set()
    for capability, (verified, refs, metadata) in capability_specs.items():
        record = RuntimeCapabilityEvidence(
            capability=capability,
            status="VERIFIED" if verified else "BLOCKED",
            checked_at=snapshot.checked_at,
            evidence_refs=refs,
            metadata=metadata,
        )
        reference = _publish(
            store,
            task_id="stage0.01_baseline_and_safety",
            artifact_kind=f"capability_{capability}",
            config_hash=config_hash,
            payload=record.to_dict(),
            source_refs=refs,
        )
        evidence_refs[f"capability_{capability}"] = reference
        if verified:
            verified_capabilities.add(capability)

    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(verified_capabilities),
        frozen_contract_stages=frozenset({0}),
        passed_gate_ids=frozenset(gate_specs),
        evidence_refs=evidence_refs,
    )
    environment_ref = f"{output_dir}/environment.json"
    write_canonical_json(root / environment_ref, environment.to_dict())

    runtime = build_default_task_runtime(root)
    previous_refs: tuple[str, ...] = ()
    outputs: dict[str, Mapping[str, str]] = {}
    config_refs: dict[str, str] = {}
    for task_id in _BOOTSTRAP_TASK_IDS:
        task_suffix = task_id.split(".", 1)[1].split("_", 1)[0]
        task_output = f"evidence/stage0/tasks/{task_suffix}-{binding.git_commit}"
        config = build_stage0_formal_config(
            binding.repository,
            task_id=task_id,
            input_refs=previous_refs,
            output_dir=task_output,
        )
        config_ref = f"{output_dir}/resolved-configs/{task_suffix}.json"
        write_canonical_json(root / config_ref, config.to_dict())
        config_refs[task_id] = config_ref
        result = runtime.execute(config, environment=environment)
        if result.status.value != "PASS" or not result.formal_eligible:
            raise Stage0BootstrapError(
                f"STAGE0_BOOTSTRAP_TASK_NOT_FORMAL_PASS:{task_id}:{result.status.value}"
            )
        outputs[task_id] = dict(result.artifact_refs)
        previous_refs = tuple(result.artifact_refs.values())

    index_payload: dict[str, JSONValue] = {
        "schema_version": "stage0-formal-bootstrap-index-v1",
        "generator_git_commit": binding.git_commit,
        "checked_at": snapshot.checked_at,
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "config_refs": dict(config_refs),
        "task_output_refs": {
            task_id: dict(refs) for task_id, refs in outputs.items()
        },
        "next_task_id": "stage0.04_assets_and_manifests",
        "next_input_refs": list(previous_refs),
    }
    index_payload["artifact_hash"] = canonical_json_hash(index_payload)
    index_ref = f"{output_dir}/index.json"
    write_canonical_json(root / index_ref, index_payload)
    return Stage0BootstrapResult(
        environment=environment,
        task_output_refs=outputs,
        index_ref=index_ref,
        config_refs=config_refs,
    )


__all__ = [
    "Stage0BootstrapError",
    "Stage0BootstrapResult",
    "Stage0RuntimeSnapshot",
    "Stage0SourceBinding",
    "bootstrap_formal_stage0",
    "build_stage0_formal_config",
    "inspect_stage0_runtime",
    "inspect_stage0_source",
    "validate_existing_g0_g2",
]
