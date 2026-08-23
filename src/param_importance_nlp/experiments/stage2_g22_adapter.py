"""The S2.3/G2.2 formal asset GateRecord adapter.

This module is deliberately a consumer, rather than another S2.3 runner.  It
only accepts the four formal task-output commits produced by the S2.3 runner
and the independently published server evidence.  In particular, a candidate
GateRecord, a local fixture, or a self-reported digest is never promoted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.status import GateRecord, GateStatus
from ..contracts.task_catalog import DEFAULT_TASK_CATALOG
from ..runtime.task_artifacts import LoadedTaskArtifact, TaskArtifactStore, load_committed_task_artifact
from .stage23_task_runners import (
    _predecessor_context,
    validate_formal_s203_payloads,
    validate_formal_s203_task_artifacts,
)
from .stage2_assets import (
    FORMAL_CHECKPOINT_SELECTION,
    FORMAL_TOTAL_TRAINING_STEPS,
    AssetResolutionManifest,
    DataRangeManifest,
    validate_formal_asset_identity,
)


GATE_ID = "stage2.G2.2"
TASK_ID = "stage2.03_assets_checkpoints_and_sampling"
ARTIFACT_KINDS = ("sampling_plan", "draw_manifest", "asset_resolution", "gate_record")
ADAPTER_SCHEMA_VERSION = "stage2-g2.2-gate-adapter-v1"

AUTHORITY_EVIDENCE_REF = "evidence/stage2/s203/g2.2-assets.json"
AUTHORITY_EVIDENCE_SHA256 = "b6805b6744374e7d05f193db4d72162176b930fa1db250bc00797b3ad30528a8"
ASSET_REF = "manifests/stage2/s203-asset-resolution.json"
ASSET_SHA256 = "1b1609ea97974a560b4da98707eecdbbbd97010e067e9899234079ab7b6cfb20"
DATA_REF = "manifests/stage2/s203-data-range.json"
DATA_SHA256 = "a6a633d30e55351679368ded522d4b697fc676e8f4ca7f63108406524459fb14"
SELECTION_REF = "manifests/stage2/s203-selection.json"
SELECTION_SHA256 = "1035e831f63862c0d15549fe24738ffd1432e8e165628ba64b40248461dea9bd"
ASSET_DIGEST = "f57decd5cf00e69e45ab2f02c994abb202f5c614e1441acb8aebcb1807ff76ee"
DATA_DIGEST = "df8eeac5178305d409cf6128ac5d5648567aae895592c79fa21542e84a28e0f1"
PRODUCER_COMMIT = "676ab436422b1a514bddfc1181d0645fed4de7be"

OFFLINE_SHA256 = {
    "evidence/stage2/s203/pythia-14m-step0-56079904bb80/offline-load.json": "41a42aff7ce9294a3a2cd0e074d215345e3ab760ca41a19b1bad9e0d362116c1",
    "evidence/stage2/s203/pythia-14m-step1000-5b020995bfc7/offline-load.json": "e0e41e9594c25eb85b92345a4cbc16385c8bbc52edca95a34db39bdfe3e24b92",
    "evidence/stage2/s203/pythia-14m-step71000-6a9156279d41/offline-load.json": "bb3301f2e51e550f29092d2f3514dc057216a9b7ee318604280d4a9308ee7e5d",
    "evidence/stage2/s203/pythia-31m-deduped-step0-73628c85dd9d/offline-load.json": "4eff6fea9a0984bf53cc57809a3f32db98558f4bcf1c06d6445e61a071137bb6",
    "evidence/stage2/s203/pythia-31m-deduped-step1000-dd4d3eab2b00/offline-load.json": "41a72b53b562750035d28bea01dafb426a06380d78c7a6559f24a40cb34a0ea2",
    "evidence/stage2/s203/pythia-31m-deduped-step71000-aeafbd5e62a3/offline-load.json": "c079ebab849a4d3fb94ab0b58b3384065b89d26ef39d97b0580d1a4f2ecd12b2",
}
FAILED_ATTEMPTS = (
    ("tmp/stage2/s203-hf-fixed-20260823-01/publish.log", "e9889afca02c454031dfcc842e12d37e04625c2a9cc3d4a89f385d60b43c48d4"),
    ("tmp/stage2/s203-hf-fixed-20260823-02/repair.log", "706f329e2466b99ac7ceb0ff8085b50248ce10f460a9dc963bdf9315827db30f"),
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATHS = (
    "src/param_importance_nlp/experiments/stage2_assets.py",
    "src/param_importance_nlp/experiments/sampling.py",
    "schemas/shared/stage2-asset-resolution-v1.json",
    "schemas/shared/stage2-checkpoint-manifest-v1.json",
    "schemas/shared/stage2-data-range-manifest-v1.json",
    "schemas/shared/stage2-draw-stream-manifest-v1.json",
    "schemas/shared/stage2-repetition-mapping-v1.json",
    "plan/stage2/03_assets_checkpoints_and_sampling.md",
)


class G22Blocked(RuntimeError):
    """A fail-closed formal preflight result."""


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _safe_rel(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise G22Blocked("G22_PATH_INVALID")
    path = Path(*value.split("/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise G22Blocked("G22_PATH_ESCAPE")
    return path


def _reject_symlink_chain(path: Path, root: Path) -> None:
    root = root.absolute()
    path = path.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise G22Blocked("G22_PATH_ESCAPE") from error
    current = root
    for part in (root, *relative.parts):
        current = Path(part) if part == root else current / part
        if current.is_symlink():
            raise G22Blocked(f"G22_SYMLINK_REJECTED:{current}")


def _resolve(root: Path, ref: str) -> Path:
    relative = _safe_rel(ref)
    root = root.absolute()
    path = root.joinpath(relative)
    _reject_symlink_chain(path, root)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise G22Blocked("G22_PATH_ESCAPE") from error
    return path


def _validate_dual_roots(repository_root: Path, data_root: Path) -> None:
    if repository_root == data_root or repository_root.is_relative_to(data_root) or data_root.is_relative_to(repository_root):
        raise G22Blocked("G22_DUAL_ROOTS_NESTED")
    _reject_symlink_chain(repository_root, repository_root.parent)
    _reject_symlink_chain(data_root, data_root.parent)


def _load_hashed(root: Path, ref: str, expected: str) -> dict[str, JSONValue]:
    path = _resolve(root, ref)
    if not path.is_file():
        raise G22Blocked(f"G22_MISSING:{ref}")
    size, observed = _sha256(path)
    if observed != expected:
        raise G22Blocked(f"G22_SHA256_MISMATCH:{ref}")
    value = load_canonical_json(path)
    if not isinstance(value, dict) or canonical_json_hash(value) != expected:
        raise G22Blocked(f"G22_CANONICAL_HASH_MISMATCH:{ref}")
    return value


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise G22Blocked(f"G22_GIT_FAILED:{' '.join(args)}")
    return result.stdout.strip()


def _git_identity(repository_root: Path) -> dict[str, JSONValue]:
    if not repository_root.is_dir():
        raise G22Blocked("G22_REPOSITORY_ROOT_MISSING")
    _reject_symlink_chain(repository_root, repository_root.parent)
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise G22Blocked("G22_REPOSITORY_DIRTY")
    head = _git(repository_root, "rev-parse", "HEAD")
    tree = _git(repository_root, "rev-parse", "HEAD^{tree}")
    if not _HEX40.fullmatch(head) or not _HEX40.fullmatch(tree):
        raise G22Blocked("G22_GIT_OBJECT_INVALID")
    sources: dict[str, JSONValue] = {}
    for ref in _SOURCE_PATHS + ("src/param_importance_nlp/experiments/stage2_g22_adapter.py",):
        path = _resolve(repository_root, ref)
        if not path.is_file():
            raise G22Blocked(f"G22_SOURCE_MISSING:{ref}")
        raw = path.read_bytes()
        shown = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"HEAD:{ref}"],
            check=False,
            capture_output=True,
        )
        if shown.returncode or shown.stdout != raw:
            raise G22Blocked(f"G22_SOURCE_WORKTREE_DRIFT:{ref}")
        blob = _git(repository_root, "rev-parse", f"HEAD:{ref}")
        sources[ref] = {"sha256": hashlib.sha256(raw).hexdigest(), "git_blob": blob}
    return {"head": head, "tree": tree, "sources": sources}


def _producer_identity(repository_root: Path, producer: str) -> dict[str, JSONValue]:
    if not _HEX40.fullmatch(producer):
        raise G22Blocked("G22_PRODUCER_COMMIT_INVALID")
    _git(repository_root, "cat-file", "-e", f"{producer}^{{commit}}")
    current = _git(repository_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(repository_root), "merge-base", "--is-ancestor", producer, current],
        check=False,
    )
    if ancestor.returncode:
        raise G22Blocked("G22_PRODUCER_NOT_ANCESTOR")
    source_objects: dict[str, JSONValue] = {}
    for ref in _SOURCE_PATHS:
        producer_blob = _git(repository_root, "rev-parse", f"{producer}:{ref}")
        current_blob = _git(repository_root, "rev-parse", f"HEAD:{ref}")
        if producer_blob != current_blob:
            raise G22Blocked(f"G22_PRODUCER_SOURCE_DRIFT:{ref}")
        source_objects[ref] = producer_blob
    return {"commit": producer, "mode": "same_commit" if producer == current else "ancestor", "source_git_blobs": source_objects}


def _validate_task_inputs(root: Path, refs: Mapping[str, str]) -> tuple[dict[str, LoadedTaskArtifact], str]:
    try:
        return validate_formal_s203_task_artifacts(root, refs)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise G22Blocked(f"G22_S203_FORMAL_INPUT_INVALID:{type(error).__name__}") from error


def _verify_s203_lineage(
    root: Path,
    loaded: Mapping[str, LoadedTaskArtifact],
    config: ResolvedConfigV2,
    output_dir: str,
) -> tuple[str, str]:
    """Reuse the producer's predecessor verifier for binding/prereg hashes."""
    source_refs = next(iter({item.source_refs for item in loaded.values()}), ())
    auxiliary = {ASSET_REF}
    predecessor_refs = tuple(ref for ref in source_refs if ref not in auxiliary)
    if not predecessor_refs:
        raise G22Blocked("G22_S203_PREDECESSOR_REFS_MISSING")
    orchestration = config.section("orchestration")
    if not isinstance(orchestration, Mapping):
        raise G22Blocked("G22_RESOLVED_CONFIG_ORCHESTRATION_INVALID")
    view = dict(orchestration)
    view["input_result_refs"] = list(predecessor_refs)
    fake_config = SimpleNamespace(
        section=lambda name: view if name == "orchestration" else config.section(name),
        task_id=config.task_id,
        run_intent=config.run_intent,
    )
    request = SimpleNamespace(
        config=fake_config,
        task=DEFAULT_TASK_CATALOG.get(TASK_ID),
        environment=SimpleNamespace(),
    )
    context = _predecessor_context(request, root, TaskArtifactStore(root, output_dir))
    return context.binding_hash, canonical_json_hash(context.payload("preregistration"))


def _validate_offline(root: Path, manifest: AssetResolutionManifest) -> list[dict[str, JSONValue]]:
    result: list[dict[str, JSONValue]] = []
    expected_pairs = {(model, stage): pair for (model, stage), pair in FORMAL_CHECKPOINT_SELECTION.items()}
    for item in manifest.checkpoints:
        if not item.load_evidence_ref or item.load_evidence_ref not in OFFLINE_SHA256 or item.load_evidence_sha256 != OFFLINE_SHA256[item.load_evidence_ref]:
            raise G22Blocked("G22_OFFLINE_LOAD_REF_OR_HASH_INVALID")
        value = _load_hashed(root, item.load_evidence_ref, item.load_evidence_sha256)
        required = {"architecture", "backward_smoke", "config_sha256", "cuda_device_excluded", "forward_smoke", "model_global_rng_untouched", "model_id", "model_state_hash", "model_state_unchanged", "model_type", "offline", "parameter_count", "parameter_dtypes", "parameter_registry_hash", "repository", "revision", "root_ref", "schema_version", "tokenizer_length", "tokenizer_sha256", "tokenizer_vocab_size", "training_stage", "training_step"}
        if set(value) != required or value["schema_version"] != "stage2-checkpoint-offline-load-v1" or value["offline"] is not True or value["cuda_device_excluded"] is not True or value["model_global_rng_untouched"] is not True or value["model_state_unchanged"] is not True:
            raise G22Blocked("G22_OFFLINE_LOAD_SCHEMA_OR_FLAGS_INVALID")
        pair = expected_pairs.get((item.model_id, item.training_stage))
        if pair is None or value["training_step"] != pair[0] or value["revision"] != pair[1] or value["model_id"] != item.model_id or value["repository"] != item.repository or value["training_stage"] != item.training_stage or value["root_ref"] != item.root_ref:
            raise G22Blocked("G22_OFFLINE_LOAD_IDENTITY_MISMATCH")
        for field in ("model_state_hash", "parameter_registry_hash", "config_sha256", "tokenizer_sha256"):
            if not isinstance(value[field], str) or not _HEX64.fullmatch(value[field]):
                raise G22Blocked(f"G22_OFFLINE_LOAD_DIGEST_INVALID:{field}")
        expected_count = 14067712 if item.model_id == "pythia-14m" else 30494720
        if value["architecture"] != ["GPTNeoXForCausalLM"] or value["model_type"] != "gpt_neox" or value["parameter_dtypes"] != ["torch.float32"] or value["parameter_count"] != expected_count or value["tokenizer_length"] != 50277 or value["tokenizer_vocab_size"] != 50254:
            raise G22Blocked("G22_OFFLINE_LOAD_MODEL_SCHEMA_INVALID")
        forward = value["forward_smoke"]
        backward = value["backward_smoke"]
        if not isinstance(forward, Mapping) or not isinstance(backward, Mapping) or forward.get("finite") is not True or backward.get("finite") is not True or forward.get("logits_shape") != [1, 8, 50304] or forward.get("sample_id") != 0:
            raise G22Blocked("G22_OFFLINE_LOAD_SMOKE_INVALID")
        result.append({"ref": item.load_evidence_ref, "sha256": item.load_evidence_sha256, "model": item.model_id, "stage": item.training_stage, "step": item.training_step})
    if len(result) != 6 or len({item["ref"] for item in result}) != 6:
        raise G22Blocked("G22_OFFLINE_LOAD_COUNT_INVALID")
    return result


def _validate_real_assets(root: Path) -> dict[str, JSONValue]:
    evidence = _load_hashed(root, AUTHORITY_EVIDENCE_REF, AUTHORITY_EVIDENCE_SHA256)
    expected_evidence = {"active_partial_objects_untouched", "asset_resolution_hash", "asset_resolution_ref", "checkpoint_count", "combination_smoke", "consumer_commit", "cuda_device_excluded", "data_range_hash", "data_range_ref", "execution_commit", "failed_attempts", "gate_id", "offline_load_count", "producer_commit", "schema_version", "status"}
    attempts = evidence["failed_attempts"]
    combination = evidence["combination_smoke"]
    if set(evidence) != expected_evidence or evidence["schema_version"] != "stage2-g2.2-asset-gate-evidence-v1" or evidence["status"] != "PASS" or evidence["gate_id"] != GATE_ID or evidence["checkpoint_count"] != 6 or evidence["offline_load_count"] != 6 or evidence["asset_resolution_hash"] != ASSET_DIGEST or evidence["data_range_hash"] != DATA_DIGEST or evidence["producer_commit"] != PRODUCER_COMMIT or evidence["execution_commit"] != PRODUCER_COMMIT or evidence["consumer_commit"] is not None or evidence["cuda_device_excluded"] is not True or evidence["active_partial_objects_untouched"] is not True or attempts != [{"ref": ref, "sha256": sha} for ref, sha in FAILED_ATTEMPTS]:
        raise G22Blocked("G22_AUTHORITY_EVIDENCE_STATUS_INVALID")
    if combination != {"all_registry_hashes_present": True, "cell_count": 6, "registry_hash_count": 2, "status": "PASS"}:
        raise G22Blocked("G22_AUTHORITY_COMBINATION_SMOKE_INVALID")
    asset_value = _load_hashed(root, ASSET_REF, ASSET_SHA256)
    data_value = _load_hashed(root, DATA_REF, DATA_SHA256)
    selection_value = _load_hashed(root, SELECTION_REF, SELECTION_SHA256)
    asset = AssetResolutionManifest.from_mapping(asset_value)
    data = DataRangeManifest.from_mapping(data_value)
    validate_formal_asset_identity(asset)
    if asset.digest != ASSET_DIGEST or data.digest != DATA_DIGEST or asset.data_range.digest != data.digest:
        raise G22Blocked("G22_MANIFEST_DIGEST_MISMATCH")
    if asset.producer_commit != PRODUCER_COMMIT or asset.execution_commit != PRODUCER_COMMIT or asset.consumer_commit is not None or asset.data_range != data:
        raise G22Blocked("G22_MANIFEST_PROVENANCE_INVALID")
    if selection_value.get("schema_version") != "stage2-checkpoint-selection-v1" or selection_value.get("total_training_steps") != FORMAL_TOTAL_TRAINING_STEPS or selection_value.get("selection_rule") != "nearest checkpoint to target fraction; ties choose earlier step":
        raise G22Blocked("G22_SELECTION_MANIFEST_INVALID")
    rows = selection_value.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise G22Blocked("G22_SELECTION_ROWS_INVALID")
    expected_rows = [(model, stage, step, rev) for (model, stage), (step, rev) in FORMAL_CHECKPOINT_SELECTION.items()]
    observed_rows = [(row.get("model_id"), row.get("training_stage"), row.get("training_step"), row.get("revision")) for row in rows if isinstance(row, Mapping)]
    if observed_rows != expected_rows:
        raise G22Blocked("G22_SELECTION_IDENTITY_DRIFT")
    offline = _validate_offline(root, asset)
    model_manifests: list[dict[str, JSONValue]] = []
    for checkpoint in asset.checkpoints:
        if checkpoint.manifest_ref is None or checkpoint.manifest_sha256 is None or checkpoint.revision is None or checkpoint.state != "ready":
            raise G22Blocked("G22_CHECKPOINT_MANIFEST_INCOMPLETE")
        manifest_path = _resolve(root, checkpoint.manifest_ref)
        size, digest = _sha256(manifest_path)
        if digest != checkpoint.manifest_sha256:
            raise G22Blocked("G22_CHECKPOINT_MANIFEST_SHA_MISMATCH")
        value = load_canonical_json(manifest_path)
        requested_revision = {"initialization": "step0", "early": "step1000", "mid_late": "step71000"}[checkpoint.training_stage]
        if not isinstance(value, Mapping) or value.get("schema") != "parameter-importance-model-manifest-v1" or value.get("repo") != checkpoint.repository or value.get("revision") != checkpoint.revision or value.get("requested_revision") != requested_revision:
            raise G22Blocked("G22_MODEL_MANIFEST_SCHEMA_INVALID")
        if set(value) not in ({"schema", "requested_revision", "repo", "revision", "files", "downloaded_at", "transport_endpoint"}, {"schema", "requested_revision", "repo", "revision", "files", "repair_scope"}):
            raise G22Blocked("G22_MODEL_MANIFEST_FIELDS_INVALID")
        if "transport_endpoint" in value and value["transport_endpoint"] not in {"https://huggingface.co", "https://hf-mirror.com"}:
            raise G22Blocked("G22_MODEL_MANIFEST_ENDPOINT_INVALID")
        files = value.get("files")
        if not isinstance(files, list) or len(files) != len(checkpoint.files):
            raise G22Blocked("G22_MODEL_MANIFEST_FILES_INVALID")
        expected_files = {item.path: (item.size_bytes, item.sha256) for item in checkpoint.files}
        for file in files:
            if not isinstance(file, Mapping) or set(file) != {"name", "official_lfs_sha256", "sha256", "size_bytes"}:
                raise G22Blocked("G22_MODEL_MANIFEST_FILE_SCHEMA_INVALID")
            name = file["name"]
            if name not in expected_files or file["size_bytes"] != expected_files[name][0] or file["sha256"] != expected_files[name][1]:
                raise G22Blocked("G22_CHECKPOINT_FILE_MISMATCH")
            actual = _resolve(root, checkpoint.root_ref + "/" + name)
            observed_size, observed_sha = _sha256(actual)
            if (observed_size, observed_sha) != expected_files[name]:
                raise G22Blocked("G22_CHECKPOINT_FILE_BYTES_MISMATCH")
        model_manifests.append({"ref": checkpoint.manifest_ref, "sha256": checkpoint.manifest_sha256, "size_bytes": size})
    data_root = root / "datasets" / "pile-deduped-pythia-preshuffled"
    for item in data.files:
        actual = _resolve(data_root, item.path)
        observed_size, observed_sha = _sha256(actual)
        if (observed_size, observed_sha) != (item.size_bytes, item.sha256):
            raise G22Blocked("G22_DATA_FILE_BYTES_MISMATCH")
    prefix = _resolve(root, data.manifest_ref)
    psize, psha = _sha256(prefix)
    if psha != data.manifest_sha256:
        raise G22Blocked("G22_DATA_PREFIX_MANIFEST_MISMATCH")
    return {"evidence_ref": AUTHORITY_EVIDENCE_REF, "evidence_sha256": AUTHORITY_EVIDENCE_SHA256, "asset_ref": ASSET_REF, "asset_sha256": ASSET_SHA256, "asset_digest": asset.digest, "data_ref": DATA_REF, "data_sha256": DATA_SHA256, "data_digest": data.digest, "selection_ref": SELECTION_REF, "selection_sha256": SELECTION_SHA256, "offline_loads": offline, "model_manifests": model_manifests}


def _config(root: Path, ref: str, expected_hash: str) -> ResolvedConfigV2:
    value = load_canonical_json(_resolve(root, ref))
    if not isinstance(value, Mapping):
        raise G22Blocked("G22_RESOLVED_CONFIG_NOT_OBJECT")
    config = ResolvedConfigV2.from_mapping(value)
    if config.task_id != TASK_ID or config.run_intent != "formal" or config.formal_eligible is not True or config.config_hash != expected_hash:
        raise G22Blocked("G22_RESOLVED_CONFIG_NOT_FORMAL_OR_MISMATCH")
    return config


def _gate(
    *,
    status: GateStatus,
    checked_at: str,
    measured: Mapping[str, JSONValue],
    refs: Sequence[str],
    reasons: Sequence[str] = (),
) -> GateRecord:
    return GateRecord(gate_id=GATE_ID, stage=2, status=status, checked_at=checked_at, measured=dict(measured), threshold={"authority_status": "PASS", "asset_resolution_hash": ASSET_DIGEST, "data_range_hash": DATA_DIGEST, "checkpoint_count": 6, "offline_load_count": 6, "formal_task_artifacts": list(ARTIFACT_KINDS)}, evidence_refs=tuple(refs), reasons=tuple(reasons))


def evaluate_formal_g22(
    *,
    repository_root: str | Path,
    data_root: str | Path,
    resolved_config_ref: str,
    s203_artifact_refs: Mapping[str, str],
    authority_evidence_ref: str = AUTHORITY_EVIDENCE_REF,
    offline_load_refs: Mapping[str, str] | None = None,
    manifest_refs: Mapping[str, str] | None = None,
    output_dir: str | None = None,
) -> dict[str, JSONValue]:
    """Validate S2.3 and publish exactly one formal G2.2 GateRecord commit.

    Missing or invalid inputs return a BLOCKED result without writing anything.
    A commit is written only after all four formal task commits, the resolved
    config, both clean roots, and all server bytes have passed validation.
    """
    checked_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    repository = Path(repository_root).absolute()
    data = Path(data_root).absolute()
    try:
        _validate_dual_roots(repository, data)
        if authority_evidence_ref != AUTHORITY_EVIDENCE_REF:
            raise G22Blocked("G22_AUTHORITY_REF_NOT_CANONICAL")
        if offline_load_refs is not None:
            if set(offline_load_refs) != set(OFFLINE_SHA256):
                raise G22Blocked("G22_OFFLINE_REF_SET_NOT_CANONICAL")
            if any(offline_load_refs[key] != key for key in OFFLINE_SHA256):
                raise G22Blocked("G22_OFFLINE_REF_ALIAS_REJECTED")
        if manifest_refs is not None:
            expected_refs = {"asset_resolution": ASSET_REF, "data_range": DATA_REF, "selection": SELECTION_REF}
            if dict(manifest_refs) != expected_refs:
                raise G22Blocked("G22_MANIFEST_REF_SET_NOT_CANONICAL")
        loaded, config_hash = _validate_task_inputs(data, s203_artifact_refs)
        config = _config(data, resolved_config_ref, config_hash)
        artifacts = config.section("artifacts")
        if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("output_dir"), str):
            raise G22Blocked("G22_RESOLVED_CONFIG_OUTPUT_CONTRACT_INVALID")
        configured_output_dir = str(artifacts["output_dir"])
        if output_dir is not None and output_dir != configured_output_dir:
            raise G22Blocked("G22_OUTPUT_DIR_OVERRIDE_REJECTED")
        upstream_binding_hash, preregistration_hash = _verify_s203_lineage(
            data, loaded, config, configured_output_dir
        )
        validate_formal_s203_payloads(
            {kind: item.payload for kind, item in loaded.items()},
            expected_preregistration_hash=preregistration_hash,
            expected_upstream_binding_hash=upstream_binding_hash,
        )
        repo_identity = _git_identity(repository)
        producer = _producer_identity(repository, PRODUCER_COMMIT)
        assets = _validate_real_assets(data)
        refs = tuple(s203_artifact_refs[k] for k in ARTIFACT_KINDS) + (AUTHORITY_EVIDENCE_REF, ASSET_REF, DATA_REF, SELECTION_REF) + tuple(item["ref"] for item in assets["offline_loads"])
        measured: dict[str, JSONValue] = {"adapter_schema_version": ADAPTER_SCHEMA_VERSION, "task_id": TASK_ID, "config": {"ref": resolved_config_ref, "config_hash": config.config_hash, "full_hash": config.full_hash, "run_intent": config.run_intent, "formal_eligible": config.formal_eligible}, "roots": {"repository_root": str(repository), "data_root": str(data)}, "repository": repo_identity, "producer": producer, "authority": assets, "lineage": {"upstream_binding_hash": upstream_binding_hash, "preregistration_contract_hash": preregistration_hash}, "runtime": {"runtime": "TaskRuntime", "formal_envelope": "load_committed_task_artifact", "store": "TaskArtifactStore", "gate_schema": "gate-record-v1"}, "input_artifacts": {kind: {"commit_ref": loaded[kind].identity.commit_ref, "artifact_hash": loaded[kind].identity.artifact_hash} for kind in ARTIFACT_KINDS}}
        gate = _gate(status=GateStatus.PASS, checked_at=checked_at, measured=measured, refs=refs)
        store = TaskArtifactStore(data, configured_output_dir)
        existing = store.discover_complete(task_id=TASK_ID, config_hash=config.config_hash, artifact_kinds=("gate_record",), formal_eligible=True)
        if existing:
            loaded_gate = load_committed_task_artifact(data, existing["gate_record"], require_formal=True)
            previous = GateRecord.from_mapping(dict(loaded_gate.payload))
            previous_semantic = previous.to_dict()
            current_semantic = gate.to_dict()
            previous_semantic.pop("checked_at", None)
            previous_semantic.pop("artifact_hash", None)
            current_semantic.pop("checked_at", None)
            current_semantic.pop("artifact_hash", None)
            if previous_semantic != current_semantic:
                raise G22Blocked("G22_EXISTING_GATE_SEMANTIC_DRIFT")
            return {"schema_version": ADAPTER_SCHEMA_VERSION, "gate_record": previous.to_dict(), "status": previous.status.value, "formal_eligible": True, "commit_ref": existing["gate_record"], "reused": True}
        published = store.publish(task_id=TASK_ID, artifact_kind="gate_record", config_hash=config.config_hash, run_intent="formal", payload=gate.to_dict(), formal_eligible=True, source_refs=refs)
        return {"schema_version": ADAPTER_SCHEMA_VERSION, "gate_record": gate.to_dict(), "status": "PASS", "formal_eligible": True, "commit_ref": published.commit_ref, "reused": False}
    except (G22Blocked, FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        reason = str(error) or type(error).__name__
        blocked = _gate(status=GateStatus.BLOCKED, checked_at=checked_at, measured={"adapter_schema_version": ADAPTER_SCHEMA_VERSION, "roots": {"repository_root": str(repository), "data_root": str(data)}}, refs=(), reasons=(reason,))
        return {"schema_version": ADAPTER_SCHEMA_VERSION, "gate_record": blocked.to_dict(), "status": "BLOCKED", "formal_eligible": False, "commit_ref": None, "reused": False, "reason": reason}


evaluate_g22 = evaluate_formal_g22
