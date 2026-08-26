"""Independent, fail-closed qualification of the Stage 2 G2.3 reference.

The Stage 2.04 runner intentionally produces a *candidate* reference.  This
module is the consumer which can qualify it.  It never accepts a metrics JSON
file, a caller supplied pass flag, or a caller supplied scalar as evidence.
Every scalar in the returned report is calculated from the immutable task
result, committed JSON artifacts, tensor bundles, and the two resume trees.

The implementation is deliberately conservative.  A formal result which does
not contain the raw block diagnostics needed for a check is ``BLOCKED`` (and
never ``PASS``).  Attempts are content addressed by all six inputs, so a
partial attempt does not lock out a later complete aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import random
import shutil
import subprocess
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np

from param_importance_nlp.contracts.jsonio import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.runtime.task_artifacts import (
    LoadedTaskArtifact,
    load_committed_task_artifact,
)
from param_importance_nlp.runtime.task_runtime import TaskRunResult, TaskRunStatus
from param_importance_nlp.runtime.tensor_bundle import load_tensor_bundle

from .stage2_formal import (
    ReferenceUncertainty,
    _ReferenceShardStore,
    _ReferenceSnapshotStore,
    _BoundedCheckpointStore,
    _BoundedMoments,
    _bounded_moments_digest,
    bounded_reference_numeric_diagnostics,
    estimate_reference_uncertainty_bounded,
    estimate_sequence_variance_bounded,
    _moments_from_shards,
    _vector_digest,
    estimate_reference_uncertainty_shards,
)
from .stage2_g23_contracts import (
    boundary_digest,
    unique_identity_values,
    validate_external_manifest,
    validate_generator_boundary,
    validate_resume_prefix,
    validate_sizing_plan_contract,
    validate_weighting_contract,
)
from .sampling import CANDIDATE_BATCH_SIZES, DrawStreamManifest, SamplingPlan


SCHEMA_VERSION = "stage2-g23-reference-evaluation-v1"
GATE_ID = "stage2.G2.3"
REQUIRED_CELL_COUNT = 6
EXPECTED_CELL_IDS: tuple[str, ...] = tuple(
    f"{model}:{stage}"
    for model in ("pythia-14m", "pythia-31m-deduped")
    for stage in ("initialization", "early", "mid_late")
)
THRESHOLDS: Mapping[str, float] = {
    "normalized_l1": 0.02,
    "pearson": 0.995,
    "signal_eligible_spearman": 0.995,
    "layer_module_spearman": 0.995,
    "topk_overlap": 0.98,
    "layer_module_delta": 0.01,
    "layer_module_l1_q95": 0.01,
    "h_ref_divisor": 4.0,
    "epsilon_num_divisor": 10.0,
}
CORRECTED_DELTA_SCHEMA_VERSION = "stage2-g23-corrected-delta-sci-v1"
CORRECTED_DELTA_BATCH_SIZES: tuple[int, ...] = tuple(
    int(value) for value in CANDIDATE_BATCH_SIZES
)
CORRECTED_DELTA_SIDECAR_FIELDS = frozenset(
    {
        "schema_version",
        "source_producer_schema_version",
        "source_producer_ref",
        "source_producer_artifact_hash",
        "source_producer_table_mode",
        "source_producer_commit",
        "evaluator_commit",
        "evaluator_source_sha256",
        "formula_contract_hash",
        "formula_version",
        "formula",
        "absolute_floors",
        "reference_id",
        "sizing_result_hash",
        "sizing_plan_hash",
        "registry_hash",
        "candidate_sample_counts",
        "delta_sci_batch_sizes",
        "selected_sample_count_per_stream",
        "delta_sci_by_endpoint",
        "signal_scale_by_endpoint",
        "noise_scale_by_endpoint",
        "sizing_nodes",
        "correction_reason",
        "artifact_hash",
    }
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_EVALUATOR_SOURCE_RELATIVE = "src/param_importance_nlp/experiments/stage2_g23_evaluator.py"


class G23Blocked(ValueError):
    """Raised internally for an input which cannot be formally qualified."""


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise G23Blocked(f"{field}:SHA256_REQUIRED")
    return value


def _commit(value: object, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise G23Blocked(f"{field}:PRODUCER_COMMIT_REQUIRED")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G23Blocked(f"{field}:FINITE_NUMBER_REQUIRED")
    number = float(value)
    if not math.isfinite(number):
        raise G23Blocked(f"{field}:NON_FINITE")
    return number


def _path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise G23Blocked(f"{field}:LOGICAL_PATH_REQUIRED")
    if value.startswith("/") or value.endswith("/") or "//" in value or re.match(r"^[A-Za-z]:", value):
        raise G23Blocked(f"{field}:PATH_ESCAPE")
    if any(part == "" for part in value.split("/")):
        raise G23Blocked(f"{field}:PATH_ESCAPE")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise G23Blocked(f"{field}:PATH_ESCAPE")
    return parsed.as_posix()


def _reject_symlink_chain(root: Path, logical: str, field: str) -> None:
    """Reject symlink indirection even when its resolved target stays in root."""

    current = root
    try:
        if current.is_symlink():
            raise G23Blocked(f"{field}:SYMLINK_FORBIDDEN")
    except OSError as error:
        raise G23Blocked(f"{field}:UNREADABLE") from error
    for part in PurePosixPath(logical).parts:
        current = current / part
        try:
            if current.is_symlink():
                raise G23Blocked(f"{field}:SYMLINK_FORBIDDEN")
        except OSError as error:
            raise G23Blocked(f"{field}:UNREADABLE") from error


def _reject_absolute_symlink_chain(path: Path, field: str) -> Path:
    """Reject symlinks in an absolute path without resolving them first."""

    absolute = path.absolute()
    anchor = Path(absolute.anchor)
    try:
        logical = absolute.relative_to(anchor).as_posix()
    except ValueError as error:
        raise G23Blocked(f"{field}:PATH_ESCAPE") from error
    _reject_symlink_chain(anchor, logical, field)
    return absolute


def _reject_symlinks_under(path: Path, field: str) -> None:
    try:
        if path.is_symlink():
            raise G23Blocked(f"{field}:SYMLINK_FORBIDDEN")
        if path.exists():
            for child in path.rglob("*"):
                if child.is_symlink():
                    raise G23Blocked(f"{field}:SYMLINK_FORBIDDEN")
    except OSError as error:
        raise G23Blocked(f"{field}:UNREADABLE") from error


def _canonical_payload_hash(value: Mapping[str, object], field: str) -> str:
    declared = _sha(value.get("artifact_hash"), f"{field}.artifact_hash")
    payload = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(payload) != declared:
        raise G23Blocked(f"{field}:HASH_MISMATCH")
    return declared


def _array(value: object, field: str) -> np.ndarray:
    if isinstance(value, Mapping):
        raise G23Blocked(f"{field}:ARRAY_REQUIRED")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise G23Blocked(f"{field}:ARRAY_INVALID") from error
    if not np.all(np.isfinite(array)):
        raise G23Blocked(f"{field}:NON_FINITE")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _vector(value: object, field: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise G23Blocked(f"{field}:VECTOR_REQUIRED")
    result: dict[str, np.ndarray] = {}
    for name, item in value.items():
        text = str(name)
        if not text or text in result:
            raise G23Blocked(f"{field}:PARAMETER_NAMES_INVALID")
        result[text] = _array(item, f"{field}.{text}")
    return {name: result[name] for name in sorted(result)}


def _compatible(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], field: str) -> None:
    if tuple(left) != tuple(right):
        raise G23Blocked(f"{field}:PARAMETER_SET_MISMATCH")
    for name in left:
        if left[name].shape != right[name].shape:
            raise G23Blocked(f"{field}.{name}:SHAPE_MISMATCH")


def _flat(vector: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([vector[name].reshape(-1) for name in sorted(vector)])


def _l1(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> float:
    _compatible(left, right, "normalized_l1")
    denominator = float(np.abs(_flat(right)).sum())
    numerator = float(np.abs(_flat(left) - _flat(right)).sum())
    if denominator == 0.0:
        if numerator == 0.0:
            return 0.0
        raise G23Blocked("normalized_l1:ZERO_REFERENCE_DENOMINATOR")
    return numerator / denominator


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise G23Blocked("pearson:TOO_FEW_COORDINATES")
    left_centered, right_centered = left - left.mean(), right - right.mean()
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom == 0.0:
        return 1.0 if np.array_equal(left, right) else -1.0
    return float(np.dot(left_centered, right_centered) / denom)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # Equal values receive the average rank.  This is deterministic and does
    # not require scipy (which is intentionally absent from the formal image).
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rank(left), _rank(right))


def _top_overlap(left: np.ndarray, right: np.ndarray, fraction: float) -> float:
    if left.size != right.size or left.size == 0:
        raise G23Blocked("topk:COORDINATE_MISMATCH")
    count = max(1, int(math.ceil(float(left.size) * fraction)))
    left_top = set(np.argsort(-left, kind="mergesort")[:count].tolist())
    right_top = set(np.argsort(-right, kind="mergesort")[:count].tolist())
    return len(left_top.intersection(right_top)) / float(count)


def _mean(vectors: Sequence[Mapping[str, np.ndarray]], field: str) -> dict[str, np.ndarray]:
    if not vectors:
        raise G23Blocked(f"{field}:EMPTY")
    first = vectors[0]
    total = {name: np.zeros_like(value, dtype=np.float64) for name, value in first.items()}
    for current in vectors:
        _compatible(first, current, field)
        for name in total:
            total[name] += current[name]
    count = float(len(vectors))
    return {name: value / count for name, value in total.items()}


def _u_from_moments(moment: Mapping[str, object], field: str) -> dict[str, np.ndarray]:
    if not isinstance(moment, Mapping):
        raise G23Blocked(f"{field}:MOMENTS_REQUIRED")
    try:
        g1 = _vector(moment["g1"], f"{field}.g1")
        g2 = _vector(moment["g2"], f"{field}.g2")
        n1 = _finite(moment["n1"], f"{field}.n1")
        n2 = _finite(moment["n2"], f"{field}.n2")
        count = int(moment["count"])
    except (KeyError, TypeError, ValueError) as error:
        raise G23Blocked(f"{field}:MOMENTS_FIELDS_MISSING") from error
    _compatible(g1, g2, field)
    denominator = n1 * n1 - n2
    if count < 2 or denominator <= 0.0 or not math.isfinite(denominator):
        raise G23Blocked(f"{field}:U_DENOMINATOR_INVALID")
    return {name: (g1[name] * g1[name] - g2[name]) / denominator for name in g1}


def _weighted_mean_from_moments(moment: Mapping[str, object], field: str) -> dict[str, np.ndarray]:
    if not isinstance(moment, Mapping):
        raise G23Blocked(f"{field}:MOMENTS_REQUIRED")
    n1 = _finite(moment.get("n1"), f"{field}.n1")
    if n1 <= 0.0:
        raise G23Blocked(f"{field}:WEIGHT_DENOMINATOR_INVALID")
    g1 = _vector(moment.get("g1"), f"{field}.g1")
    return {name: value / n1 for name, value in g1.items()}


def _merge_moments(left: Mapping[str, object], right: Mapping[str, object], field: str) -> dict[str, object]:
    a = _vector(left["g1"], f"{field}.left.g1")
    b = _vector(right["g1"], f"{field}.right.g1")
    a2 = _vector(left["g2"], f"{field}.left.g2")
    b2 = _vector(right["g2"], f"{field}.right.g2")
    _compatible(a, b, field)
    _compatible(a2, b2, field)
    return {
        "count": int(left["count"]) + int(right["count"]),
        "n1": _finite(left["n1"], f"{field}.left.n1") + _finite(right["n1"], f"{field}.right.n1"),
        "n2": _finite(left["n2"], f"{field}.left.n2") + _finite(right["n2"], f"{field}.right.n2"),
        "g1": {name: a[name] + b[name] for name in a},
        "g2": {name: a2[name] + b2[name] for name in a2},
    }


def _moments_from_blocks(
    blocks: Sequence[Mapping[str, np.ndarray]],
    weights: Sequence[object],
    field: str,
) -> dict[str, object]:
    """Rebuild sufficient statistics from every committed raw block."""

    if not blocks or len(blocks) != len(weights):
        raise G23Blocked(f"{field}:RAW_BLOCKS_AND_WEIGHTS_REQUIRED")
    parsed_weights = [_finite(value, f"{field}.weight") for value in weights]
    if any(value <= 0.0 for value in parsed_weights):
        raise G23Blocked(f"{field}:POSITIVE_BLOCK_WEIGHTS_REQUIRED")
    first = blocks[0]
    g1 = {name: np.zeros_like(value, dtype=np.float64) for name, value in first.items()}
    g2 = {name: np.zeros_like(value, dtype=np.float64) for name, value in first.items()}
    n1 = 0.0
    n2 = 0.0
    for block, weight in zip(blocks, parsed_weights):
        _compatible(first, block, field)
        n1 += weight
        n2 += weight * weight
        for name in g1:
            g1[name] += weight * block[name]
            g2[name] += weight * weight * np.square(block[name])
    return {
        "count": len(blocks),
        "n1": n1,
        "n2": n2,
        "g1": g1,
        "g2": g2,
    }


def _moments_equal(left: Mapping[str, object], right: Mapping[str, object], field: str) -> None:
    for key in ("count", "n1", "n2"):
        if key == "count":
            if int(left.get(key, -1)) != int(right.get(key, -2)):
                raise G23Blocked(f"{field}.{key}:MOMENTS_DRIFT")
        elif not math.isclose(_finite(left.get(key), f"{field}.{key}.left"), _finite(right.get(key), f"{field}.{key}.right"), rel_tol=1e-12, abs_tol=1e-12):
            raise G23Blocked(f"{field}.{key}:MOMENTS_DRIFT")
    for key in ("g1", "g2"):
        lv, rv = _vector(left.get(key), f"{field}.left.{key}"), _vector(right.get(key), f"{field}.right.{key}")
        _compatible(lv, rv, field)
        if any(not np.allclose(lv[name], rv[name], rtol=1e-12, atol=1e-12) for name in lv):
            raise G23Blocked(f"{field}.{key}:MOMENTS_DRIFT")


def _digest_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_evaluator_provenance(
    repo_root: Path,
    *,
    module_path: Path,
) -> tuple[str, str]:
    """Bind evaluator lineage to the checkout containing the running module.

    Producer provenance is intentionally validated by ``repo_root`` supplied
    by the CLI.  This separate check never consults that argument: the
    evaluator commit and source digest must describe the checkout from which
    this module was actually imported and executed.
    """

    repository = repo_root.resolve()
    module = module_path.resolve()
    try:
        relative = module.relative_to(repository).as_posix()
    except ValueError as error:
        raise G23Blocked("evaluator_provenance:MODULE_OUTSIDE_REPOSITORY") from error
    if relative != _EVALUATOR_SOURCE_RELATIVE or not module.is_file():
        raise G23Blocked("evaluator_provenance:MODULE_SOURCE_PATH_INVALID")
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _commit(head, "evaluator_provenance.head_commit")
        subprocess.run(
            ["git", "-C", str(repository), "cat-file", "-e", f"{head}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise G23Blocked("evaluator_provenance:TRACKED_FILES_NOT_CLEAN")
        head_blob = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        worktree_blob = subprocess.run(
            ["git", "-C", str(repository), "hash-object", "--", relative],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except G23Blocked:
        raise
    except (OSError, subprocess.SubprocessError) as error:
        raise G23Blocked("evaluator_provenance:GIT_UNAVAILABLE") from error
    if not _COMMIT.fullmatch(head_blob) or worktree_blob != head_blob:
        raise G23Blocked("evaluator_provenance:MODULE_GIT_BLOB_MISMATCH")
    try:
        source_sha256 = _digest_bytes(module)
    except OSError as error:
        raise G23Blocked("evaluator_provenance:MODULE_SOURCE_UNREADABLE") from error
    if _SHA.fullmatch(source_sha256) is None:
        raise G23Blocked("evaluator_provenance:MODULE_SOURCE_HASH_INVALID")
    return head, source_sha256


def _append_attempt_index(index: Path, artifact_hash: str) -> None:
    """Append an attempt hash under an exclusive lock and atomic replace."""

    # Validate every existing path component before mkdir/open.  Checking only
    # the final index or lock permits a symlinked parent to redirect the
    # append outside the evaluator's output root.
    current = Path(index.anchor)
    for part in index.relative_to(Path(index.anchor)).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError("G23_ATTEMPT_INDEX_PATH_SYMLINK")
    index.parent.mkdir(parents=True, exist_ok=True)
    if index.is_symlink() or index.parent.is_symlink():
        raise RuntimeError("G23_ATTEMPT_INDEX_PATH_SYMLINK")
    lock = index.with_name(index.name + ".lock")
    descriptor = -1
    for _ in range(200):
        try:
            if lock.is_symlink():
                raise RuntimeError("G23_ATTEMPT_INDEX_LOCK_SYMLINK")
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(lock, flags | nofollow, 0o600)
            break
        except FileExistsError:
            try:
                if lock.is_symlink() or time.time() - lock.stat().st_mtime > 300.0:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            time.sleep(0.005)
    if descriptor < 0:
        raise RuntimeError("G23_ATTEMPT_INDEX_LOCK_TIMEOUT")
    try:
        os.close(descriptor)
        descriptor = -1
        existing = index.read_text(encoding="utf-8") if index.exists() else ""
        line = artifact_hash + "\n"
        if line in existing.splitlines(keepends=True):
            return
        fd, temporary = tempfile.mkstemp(prefix=index.name + ".", suffix=".tmp", dir=str(index.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(existing)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, index)
            try:
                parent_fd = os.open(index.parent, os.O_RDONLY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            except OSError:
                # Windows does not expose directory fsync; the atomic replace
                # remains the commit point there.
                pass
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class CellInput:
    """A reference to one formal Stage2.04 task result.

    ``task_result_ref`` is a workspace-relative JSON path.  Optional fields are
    also paths; no field carries a metric or a qualification decision.
    """

    cell_id: str
    task_result_ref: str
    config_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.cell_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}", self.cell_id):
            raise ValueError("cell_id:INVALID")
        _path(self.task_result_ref, "task_result_ref")
        if self.config_ref is not None:
            _path(self.config_ref, "config_ref")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CellInput":
        if not isinstance(value, Mapping):
            raise TypeError("cell input must be an object")
        return cls(str(value["cell_id"]), str(value["task_result_ref"]), None if value.get("config_ref") is None else str(value["config_ref"]))


@dataclass(slots=True)
class _CellEvidence:
    source: CellInput
    workspace_root: Path | None = None
    result: TaskRunResult | None = None
    result_payload: Mapping[str, object] | None = None
    reference: LoadedTaskArtifact | None = None
    convergence: LoadedTaskArtifact | None = None
    gate: LoadedTaskArtifact | None = None
    bundle_state: Mapping[str, object] | None = None
    bundle_manifest_hash: str | None = None
    sizing_states: list[Mapping[str, object]] | None = None
    sizing_root: Path | None = None
    final_state: Mapping[str, object] | None = None
    final_states: list[Mapping[str, object]] | None = None
    final_root: Path | None = None
    external_payloads: Mapping[str, Mapping[str, object]] | None = None
    identities: dict[str, str] = None  # type: ignore[assignment]
    reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.identities = {}
        self.reasons = []


def _resolve(root: Path, reference: str) -> Path:
    logical_ref = _path(reference, "reference")
    _reject_symlink_chain(root, logical_ref, "reference")
    logical = Path(logical_ref)
    path = (root / logical).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise G23Blocked("REFERENCE_PATH_ESCAPE") from error
    return path


def _load_json(root: Path, reference: str, field: str) -> Mapping[str, object]:
    try:
        value = load_canonical_json(_resolve(root, reference))
    except (OSError, ValueError, TypeError) as error:
        raise G23Blocked(f"{field}:UNREADABLE") from error
    if not isinstance(value, Mapping):
        raise G23Blocked(f"{field}:OBJECT_REQUIRED")
    return value


def _load_task_artifact_strict(root: Path, reference: str, field: str) -> LoadedTaskArtifact:
    """Inspect the commit's object_ref before any resolver can follow it."""

    commit = _load_json(root, reference, f"{field}.commit")
    object_ref = _path(commit.get("object_ref"), f"{field}.object_ref")
    _reject_symlink_chain(root, object_ref, f"{field}.object_ref")
    _reject_symlinks_under(_resolve(root, object_ref), f"{field}.object_ref")
    try:
        return load_committed_task_artifact(root, reference, require_formal=True)
    except (OSError, ValueError, TypeError) as error:
        raise G23Blocked(f"{field}:INVALID") from error


def _task_result(root: Path, source: CellInput) -> tuple[TaskRunResult, Mapping[str, object]]:
    value = _load_json(root, source.task_result_ref, "task_result")
    if value.get("schema_version") != "task-run-result-v2":
        raise G23Blocked("task_result:SCHEMA_REQUIRED")
    try:
        result = TaskRunResult.from_mapping(value)
    except Exception as error:  # TaskRuntimeError is intentionally not leaked as qualification
        raise G23Blocked(f"task_result:INVALID:{type(error).__name__}") from error
    if result.task_id != "stage2.04_reference_target" or result.stage != 2 or result.run_intent != "formal":
        raise G23Blocked("task_result:FORMAL_STAGE204_REQUIRED")
    if result.status is not TaskRunStatus.PASS or result.formal_eligible is not True:
        raise G23Blocked("task_result:COMPLETE_FORMAL_PASS_REQUIRED")
    return result, value


def _artifact(root: Path, result: TaskRunResult, kind: str) -> LoadedTaskArtifact:
    ref = result.artifact_refs.get(kind)
    if ref is None:
        raise G23Blocked(f"artifact:{kind}:MISSING")
    _reject_symlink_chain(root, _path(ref, f"artifact.{kind}.commit_ref"), f"artifact.{kind}.commit_ref")
    loaded = _load_task_artifact_strict(root, ref, f"artifact.{kind}")
    expected_schema = {
        "reference_result": "reference-result-v1",
        "reference_convergence_report": "stage2-reference-convergence-report-v1",
        "gate_record": "stage23-task-gate-candidate-v1",
    }.get(kind)
    if (
        loaded.identity.task_id != "stage2.04_reference_target"
        or loaded.identity.artifact_kind != kind
        or loaded.identity.config_hash != result.config_hash
        or loaded.run_intent != "formal"
        or loaded.identity.formal_eligible is not True
        or loaded.payload.get("schema_version") != expected_schema
    ):
        raise G23Blocked(f"artifact:{kind}:FORMAL_IDENTITY_OR_SCHEMA_MISMATCH")
    if kind == "gate_record" and (
        loaded.payload.get("task_id") != "stage2.04_reference_target"
        or not isinstance(loaded.payload.get("gate_ids"), list)
        or GATE_ID not in loaded.payload.get("gate_ids", [])
    ):
        raise G23Blocked("artifact:gate_record:GATE_ID_MISMATCH")
    return loaded


class _ShardSequence(Sequence[dict[str, np.ndarray]]):
    """Lazy view over content-addressed block shards.

    Validation walks every reference once, but vectors are opened again only
    when a statistic/replicate needs that block.  A snapshot therefore never
    creates a second in-memory copy of the complete evidence array.
    """

    def __init__(
        self,
        store: _ReferenceShardStore,
        refs: Sequence[Mapping[str, object]],
        field: str,
        *,
        cache: Sequence[Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        self.store = store
        self.refs = tuple(dict(ref) for ref in refs)
        self.field = field
        # Only tiny local fixtures are cached.  Real 14M/31M cells remain
        # streaming so the evaluator never retains a complete evidence set.
        self._cache = None if cache is None else tuple(cache)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int | slice) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
        if isinstance(index, slice):
            return [self[index_value] for index_value in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self.refs)
        if index < 0 or index >= len(self.refs):
            raise IndexError(index)
        if self._cache is not None:
            return self._cache[index]
        try:
            vector, _, _ = self.store.load(self.refs[index])
        except (OSError, TypeError, ValueError) as error:
            raise G23Blocked(f"{self.field}[{index}]:SHARD_INVALID") from error
        return vector


class _CombinedSequence(Sequence[Mapping[str, np.ndarray]]):
    """Lazy concatenation used for diagnostics spanning A and B."""

    def __init__(self, left: Sequence[Mapping[str, np.ndarray]], right: Sequence[Mapping[str, np.ndarray]]) -> None:
        self.left, self.right = left, right

    def __len__(self) -> int:
        return len(self.left) + len(self.right)

    def __getitem__(self, index: int | slice) -> Mapping[str, np.ndarray] | list[Mapping[str, np.ndarray]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.left[index] if index < len(self.left) else self.right[index - len(self.left)]


def _load_shard_records(
    resume_root: Path,
    raw_refs: object,
    field: str,
) -> tuple[_ShardSequence, list[float], list[Mapping[str, object]]]:
    if not isinstance(raw_refs, list) or not raw_refs:
        raise G23Blocked(f"{field}:SHARD_REFS_REQUIRED")
    store = _ReferenceShardStore(resume_root)
    weights: list[float] = []
    refs: list[Mapping[str, object]] = []
    cached_vectors: list[Mapping[str, np.ndarray]] = []
    cache_enabled = True
    cache_elements = 0
    for index, raw in enumerate(raw_refs):
        if not isinstance(raw, Mapping):
            raise G23Blocked(f"{field}[{index}]:SHARD_REF_INVALID")
        try:
            vector, weight, digest = store.load(raw)
        except (OSError, TypeError, ValueError) as error:
            raise G23Blocked(f"{field}[{index}]:SHARD_INVALID") from error
        if digest != raw.get("shard_hash"):
            raise G23Blocked(f"{field}[{index}]:SHARD_HASH_MISMATCH")
        weights.append(_finite(weight, f"{field}[{index}].weight"))
        refs.append(dict(raw))
        if cache_enabled:
            cache_elements += sum(int(value.size) for value in vector.values())
            if cache_elements <= 8192:
                cached_vectors.append(vector)
            else:
                cache_enabled = False
                cached_vectors.clear()
    return _ShardSequence(store, refs, field, cache=cached_vectors if cache_enabled else None), weights, refs


def _load_resume_commits(
    root: Path,
    resume_root: Path,
    schema: str,
    *,
    identities: Mapping[str, object],
) -> list[Mapping[str, object]]:
    bounded_path = resume_root / "bounded-checkpoint"
    if bounded_path.exists():
        try:
            _reject_symlinks_under(bounded_path, f"resume.{schema}.bounded_checkpoint")
            latest, bounded_bundle = load_tensor_bundle(bounded_path)
        except (OSError, TypeError, ValueError) as error:
            raise G23Blocked(f"resume:{schema}:BOUNDED_CHECKPOINT_UNREADABLE") from error
        if not isinstance(latest, Mapping) or latest.get("checkpoint_schema") != _BoundedCheckpointStore.schema_version:
            raise G23Blocked(f"resume:{schema}:BOUNDED_CHECKPOINT_SCHEMA")
        for key, expected in identities.items():
            if key in {"final_length_required", "sizing_stream"}:
                if latest.get(key) is not expected:
                    raise G23Blocked(f"resume:{schema}:BOUNDED_IDENTITY_DRIFT:{key}")
            elif latest.get(key) != expected:
                raise G23Blocked(f"resume:{schema}:BOUNDED_IDENTITY_DRIFT:{key}")
        latest = dict(latest)
        latest["bounded_checkpoint_ref"] = "bounded-checkpoint"
        latest["bounded_checkpoint_manifest_hash"] = bounded_bundle.manifest_sha256
        if schema == "stage2-reference-progress-state-v1":
            candidates = latest.get("candidate_states")
            if not isinstance(candidates, Mapping) or not candidates:
                raise G23Blocked(f"resume:{schema}:BOUNDED_CANDIDATES_MISSING")
            states: list[Mapping[str, object]] = []
            for raw_count in sorted(candidates, key=lambda value: int(value)):
                count = int(raw_count)
                item = candidates[raw_count]
                if not isinstance(item, Mapping):
                    raise G23Blocked(f"resume:{schema}:BOUNDED_CANDIDATE_INVALID")
                state = dict(latest)
                state.update({"schema_version": schema, "processed_block_pairs": count // int(latest.get("block_size", 32) or 32), "a": item.get("a"), "b": item.get("b"), "shard_refs_a": [], "shard_refs_b": [], "shard_count": 0, "bounded_storage": True})
                states.append(state)
            return states
        state = dict(latest)
        state["schema_version"] = schema
        state["bounded_storage"] = True
        return [state]
    commits_dir = resume_root / "commits"
    if not commits_dir.is_dir():
        raise G23Blocked(f"resume:{schema}:COMMITS_MISSING")
    try:
        resume_logical = commits_dir.relative_to(root).as_posix()
    except ValueError as error:
        raise G23Blocked(f"resume:{schema}:ROOT_ESCAPE") from error
    _reject_symlink_chain(root, resume_logical, f"resume.{schema}.root")
    paths = sorted(commits_dir.glob("*.json"))
    if not paths:
        raise G23Blocked(f"resume:{schema}:COMMITS_EMPTY")
    states: list[Mapping[str, object]] = []
    previous_refs_a: list[Mapping[str, object]] | None = None
    previous_refs_b: list[Mapping[str, object]] | None = None
    for index, commit_path in enumerate(paths, start=1):
        try:
            _reject_symlink_chain(root, commit_path.relative_to(root).as_posix(), f"resume.{schema}.commit")
        except ValueError as error:
            raise G23Blocked(f"resume:{schema}:COMMIT_ESCAPE") from error
        try:
            commit = load_canonical_json(commit_path)
        except (OSError, ValueError) as error:
            raise G23Blocked(f"resume:{schema}:COMMIT_UNREADABLE") from error
        if not isinstance(commit, Mapping):
            raise G23Blocked(f"resume:{schema}:COMMIT_OBJECT_REQUIRED")
        required = {"schema_version", "sequence", "state_digest", "object_ref", "object_manifest_hash", "artifact_hash"}
        if set(commit) != required or commit.get("schema_version") != "stage2-reference-progress-commit-v1":
            raise G23Blocked(f"resume:{schema}:COMMIT_FIELDS")
        if int(commit.get("sequence", -1)) != index:
            raise G23Blocked(f"resume:{schema}:SEQUENCE_NOT_CONTIGUOUS")
        payload = {key: value for key, value in commit.items() if key != "artifact_hash"}
        if commit.get("artifact_hash") != canonical_json_hash(payload):
            raise G23Blocked(f"resume:{schema}:COMMIT_HASH")
        relative = Path(_path(str(commit["object_ref"]), f"resume.{schema}.object_ref"))
        _reject_symlink_chain(resume_root, relative.as_posix(), f"resume.{schema}.object_ref")
        _reject_symlinks_under(resume_root / relative, f"resume.{schema}.object_ref")
        try:
            state, bundle = load_tensor_bundle(resume_root / relative)
        except (OSError, ValueError, TypeError) as error:
            raise G23Blocked(f"resume:{schema}:OBJECT_UNREADABLE") from error
        if bundle.manifest_sha256 != _sha(commit.get("object_manifest_hash"), "resume.object_manifest_hash"):
            raise G23Blocked(f"resume:{schema}:MANIFEST_HASH")
        if not isinstance(state, Mapping) or state.get("schema_version") != schema:
            raise G23Blocked(f"resume:{schema}:STATE_SCHEMA")
        if state.get("snapshot_encoding") == _ReferenceSnapshotStore._COMPACT_ENCODING:
            try:
                state = _ReferenceSnapshotStore.materialize(state, resume_root)
            except (OSError, TypeError, ValueError) as error:
                raise G23Blocked(f"resume:{schema}:COMPACT_SNAPSHOT_INVALID") from error
        if "blocks_a" in state or "blocks_b" in state or "block_weights_a" in state or "block_weights_b" in state:
            raise G23Blocked(f"resume:{schema}:RAW_BLOCKS_FORBIDDEN_USE_SHARDS")
        raw_refs_a, raw_refs_b = state.get("shard_refs_a"), state.get("shard_refs_b")
        if not isinstance(raw_refs_a, list) or not isinstance(raw_refs_b, list):
            raise G23Blocked(f"resume:{schema}:SHARD_REFS_MISSING")
        if not all(isinstance(item, Mapping) for item in raw_refs_a + raw_refs_b):
            raise G23Blocked(f"resume:{schema}:SHARD_REFS_INVALID")
        try:
            validate_resume_prefix(
                previous_refs_a,
                previous_refs_b,
                raw_refs_a,
                raw_refs_b,
                field=f"resume.{schema}",
            )
        except ValueError as error:
            raise G23Blocked(str(error)) from error
        expected_pairs = int(state.get("processed_block_pairs", 0))
        if expected_pairs <= 0 or len(raw_refs_a) != expected_pairs:
            raise G23Blocked(f"resume:{schema}:SHARD_PREFIX_COUNT_MISMATCH")
        if schema == "stage2-reference-one-shot-progress-v1" and len(raw_refs_b) != expected_pairs:
            raise G23Blocked(f"resume:{schema}:SHARD_PREFIX_COUNT_MISMATCH")
        if schema == "stage2-reference-progress-state-v1" and state.get("sizing_stream") is True and raw_refs_b:
            raise G23Blocked(f"resume:{schema}:SIZING_B_SHARDS_FORBIDDEN")
        try:
            vectors_a, weights_a, _ = _load_shard_records(resume_root, raw_refs_a, f"resume.{schema}.shard_refs_a")
            vectors_b, weights_b, _ = (
                _load_shard_records(resume_root, raw_refs_b, f"resume.{schema}.shard_refs_b")
                if raw_refs_b else ([], [], [])
            )
        except G23Blocked:
            raise
        rebuilt_a = _moments_from_blocks(vectors_a, weights_a, f"resume.{schema}.moments_a")
        _moments_equal(rebuilt_a, state.get("a"), f"resume.{schema}.moments_a")
        if vectors_b:
            rebuilt_b = _moments_from_blocks(vectors_b, weights_b, f"resume.{schema}.moments_b")
            _moments_equal(rebuilt_b, state.get("b"), f"resume.{schema}.moments_b")
        if int(state.get("shard_count", -1)) != len(raw_refs_a) + len(raw_refs_b):
            raise G23Blocked(f"resume:{schema}:SHARD_COUNT_DIGEST_MISMATCH")
        try:
            state_digest = _ReferenceSnapshotStore._state_digest(state)
        except (KeyError, TypeError, ValueError) as error:
            raise G23Blocked(f"resume:{schema}:STATE_DIGEST_INVALID") from error
        if state_digest != _sha(commit.get("state_digest"), "resume.state_digest"):
            raise G23Blocked(f"resume:{schema}:STATE_DIGEST_MISMATCH")
        required_identity = {
            "plan_hash",
            "provider_state_digest",
            "registry_hash",
            "weighting_assumptions",
            "rng_state_digest",
            "rng_state",
        }
        if schema == "stage2-reference-progress-state-v1":
            required_identity |= {"sizing_draw_hash", "sizing_identity_hash", "sizing_stream"}
        else:
            required_identity |= {
                "sizing_result_hash",
                "stream_a_draw_hash",
                "stream_b_draw_hash",
                "sizing_result_identity_hash",
                "final_length_required",
            }
        if not required_identity.issubset(state):
            raise G23Blocked(f"resume:{schema}:IDENTITY_FIELDS_MISSING")
        try:
            if state.get("rng_state_digest") != boundary_digest(state.get("rng_state"), field=f"resume.{schema}.rng_state"):
                raise G23Blocked(f"resume:{schema}:RNG_STATE_DIGEST_MISMATCH")
            validate_weighting_contract(state.get("weighting_assumptions"), field=f"resume.{schema}.weighting_assumptions")
        except ValueError as error:
            raise G23Blocked(str(error)) from error
        for key, expected in identities.items():
            if key in state and state.get(key) != expected:
                raise G23Blocked(f"resume:{schema}:{key}:IDENTITY_DRIFT")
        states.append(state)
        previous_refs_a = [dict(item) for item in raw_refs_a]
        previous_refs_b = [dict(item) for item in raw_refs_b]
    return states


def _bounded_moments_strict(
    value: object,
    field: str,
    *,
    require_higher: bool,
) -> _BoundedMoments:
    if not isinstance(value, Mapping):
        raise G23Blocked(f"{field}:OBJECT_REQUIRED")
    try:
        moments = _BoundedMoments.from_state(value)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise G23Blocked(f"{field}:INVALID") from error
    if (
        moments.count <= 0
        or not math.isfinite(moments.n1)
        or not math.isfinite(moments.n2)
        or moments.n1 <= 0.0
        or moments.n2 <= 0.0
        or moments.first_weight is None
        or not math.isfinite(moments.first_weight)
        or moments.first_weight <= 0.0
        or not moments.all_equal_weights
    ):
        raise G23Blocked(f"{field}:EQUAL_WEIGHT_IDENTITY_REQUIRED")
    expected_names = set(moments.g1)
    required_vectors = (moments.g2, moments.p2)
    if not expected_names or any(set(vector) != expected_names for vector in required_vectors):
        raise G23Blocked(f"{field}:MOMENT_PARAMETER_SET_MISMATCH")
    if require_higher and (
        not moments.include_higher
        or set(moments.p3) != expected_names
        or set(moments.p4) != expected_names
    ):
        raise G23Blocked(f"{field}:HIGHER_MOMENTS_REQUIRED")
    for vector in (moments.g1, moments.g2, moments.p2, moments.p3, moments.p4):
        for item in vector.values():
            if not np.all(np.isfinite(item)):
                raise G23Blocked(f"{field}:NON_FINITE")
    weight = float(moments.first_weight)
    if not math.isclose(moments.n1, moments.count * weight, rel_tol=1e-12, abs_tol=1e-12):
        raise G23Blocked(f"{field}:N1_COUNT_WEIGHT_MISMATCH")
    if not math.isclose(moments.n2, moments.count * weight * weight, rel_tol=1e-12, abs_tol=1e-12):
        raise G23Blocked(f"{field}:N2_COUNT_WEIGHT_MISMATCH")
    for name in expected_names:
        if np.any(moments.p2[name] < 0.0) or np.any(moments.g2[name] < 0.0):
            raise G23Blocked(f"{field}:SECOND_MOMENT_NEGATIVE")
        if not np.allclose(
            moments.g2[name], weight * weight * moments.p2[name], rtol=1e-12, atol=1e-12
        ):
            raise G23Blocked(f"{field}:WEIGHTED_POWER_SUM_DRIFT")
        if np.any(np.square(moments.g1[name]) > moments.count * moments.g2[name] * (1.0 + 1e-12) + 1e-12):
            raise G23Blocked(f"{field}:CAUCHY_BOUND_FAILED")
    return moments


def _resume_roots(root: Path, result_ref: str) -> tuple[Path, Path]:
    logical_ref = _path(result_ref, "reference_result_ref")
    _reject_symlink_chain(root, logical_ref, "reference_result_ref")
    commit = Path(logical_ref)
    # .../<output_dir>/commits/reference_result.json -> .../<output_dir>
    output_dir = (root / commit.parent.parent).resolve()
    try:
        output_dir.relative_to(root.resolve())
    except ValueError as error:
        raise G23Blocked("reference_result_ref:PATH_ESCAPE") from error
    for child in (output_dir, output_dir / "resume", output_dir / "resume" / "reference-sizing", output_dir / "resume" / "reference-final"):
        try:
            _reject_symlink_chain(root, child.relative_to(root).as_posix(), "resume_root")
        except ValueError as error:
            raise G23Blocked("resume_root:ESCAPE") from error
    return output_dir / "resume" / "reference-sizing", output_dir / "resume" / "reference-final"


def _validate_six_cell_manifest(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Mapping):
        raise G23Blocked("six_cell_manifest:OBJECT_REQUIRED")
    if value.get("schema_version") != "stage2-s204-six-cell-manifest-v1" or value.get("status") != "READY" or value.get("scope") != "formal":
        raise G23Blocked("six_cell_manifest:FORMAL_READY_REQUIRED")
    declared = _sha(value.get("manifest_hash"), "six_cell_manifest.manifest_hash")
    if canonical_json_hash({key: item for key, item in value.items() if key != "manifest_hash"}) != declared:
        raise G23Blocked("six_cell_manifest:HASH_MISMATCH")
    _sha(value.get("asset_resolution_hash"), "six_cell_manifest.asset_resolution_hash")
    _commit(value.get("asset_producer_commit"), "six_cell_manifest.asset_producer_commit")
    _commit(value.get("asset_execution_commit"), "six_cell_manifest.asset_execution_commit")
    _sha(value.get("data_range_hash"), "six_cell_manifest.data_range_hash")
    rows = value.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != REQUIRED_CELL_COUNT:
        raise G23Blocked("six_cell_manifest:EXACTLY_SIX_ROWS_REQUIRED")

    # S2.3 materialization now publishes the authoritative registry identity
    # per cell.  Keep the mapping's insertion order part of the wire contract:
    # this catches a producer that silently reorders or drops a cell before a
    # consumer can bind an artifact to the wrong checkpoint.
    has_registry_map = "registry_hashes_by_cell" in value
    raw_registry_map = value.get("registry_hashes_by_cell")
    registry_hashes_by_cell: dict[str, str] | None = None
    if has_registry_map:
        if not isinstance(raw_registry_map, Mapping):
            raise G23Blocked("six_cell_manifest.registry_hashes_by_cell:OBJECT_REQUIRED")
        if tuple(raw_registry_map) != EXPECTED_CELL_IDS:
            raise G23Blocked("six_cell_manifest.registry_hashes_by_cell:CELL_ORDER_INVALID")
        registry_hashes_by_cell = {
            cell_id: _sha(
                raw_registry_map[cell_id],
                f"six_cell_manifest.registry_hashes_by_cell.{cell_id}",
            )
            for cell_id in EXPECTED_CELL_IDS
        }

    by_id: dict[str, Mapping[str, object]] = {}
    checkpoint_ids: set[object] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise G23Blocked("six_cell_manifest:ROW_OBJECT_REQUIRED")
        cell_id = row.get("cell_id")
        if cell_id not in EXPECTED_CELL_IDS or cell_id in by_id:
            raise G23Blocked("six_cell_manifest:CELL_SET_INVALID")
        model, stage = str(cell_id).split(":", 1)
        if row.get("model_id") != model or row.get("training_stage") != stage:
            raise G23Blocked("six_cell_manifest:CELL_ID_FIELDS_MISMATCH")
        if not isinstance(row.get("checkpoint_id"), str) or not row.get("checkpoint_id") or row.get("checkpoint_id") in checkpoint_ids:
            raise G23Blocked("six_cell_manifest:CHECKPOINT_IDENTITY_INVALID")
        checkpoint_ids.add(row.get("checkpoint_id"))
        if not isinstance(row.get("checkpoint_revision"), str) or not row.get("checkpoint_revision"):
            raise G23Blocked("six_cell_manifest:CHECKPOINT_REVISION_REQUIRED")
        _sha(row.get("checkpoint_hash"), f"six_cell_manifest.{cell_id}.checkpoint_hash")
        _sha(row.get("config_hash"), f"six_cell_manifest.{cell_id}.config_hash")
        _sha(row.get("registry_hash"), f"six_cell_manifest.{cell_id}.registry_hash")
        by_id[str(cell_id)] = row
    if tuple(by_id) != EXPECTED_CELL_IDS:
        raise G23Blocked("six_cell_manifest:CELL_ORDER_INVALID")
    row_registry_hashes = {
        cell_id: _sha(
            by_id[cell_id].get("registry_hash"),
            f"six_cell_manifest.{cell_id}.registry_hash",
        )
        for cell_id in EXPECTED_CELL_IDS
    }
    registry_hash = _sha(value.get("registry_hash"), "six_cell_manifest.registry_hash")
    if registry_hashes_by_cell is not None:
        if any(
            row_registry_hashes[cell_id] != registry_hashes_by_cell[cell_id]
            for cell_id in EXPECTED_CELL_IDS
        ):
            raise G23Blocked("six_cell_manifest:REGISTRY_ROW_MAP_MISMATCH")
        distinct_registry_hashes = set(registry_hashes_by_cell.values())
        expected_registry_hash = (
            next(iter(distinct_registry_hashes))
            if len(distinct_registry_hashes) == 1
            else canonical_json_hash(registry_hashes_by_cell)
        )
        if registry_hash != expected_registry_hash:
            raise G23Blocked("six_cell_manifest:REGISTRY_DIGEST_MISMATCH")
    # Before the per-cell map was materialized, the only safe legacy form was
    # a genuinely common registry hash.  Do not infer or force a common hash
    # when old rows already carry model-specific identities.
    elif any(row_registry_hashes[cell_id] != registry_hash for cell_id in EXPECTED_CELL_IDS):
        raise G23Blocked("six_cell_manifest:LEGACY_COMMON_REGISTRY_REQUIRED")
    data = value.get("data")
    if not isinstance(data, Mapping) or data.get("data_range_hash") != value.get("data_range_hash"):
        raise G23Blocked("six_cell_manifest:DATA_RANGE_DRIFT")
    return tuple(by_id[cell_id] for cell_id in EXPECTED_CELL_IDS)


def _identity_object(value: object, field: str, *, required: Sequence[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise G23Blocked(f"{field}:OBJECT_REQUIRED")
    for key in required:
        if key not in value:
            raise G23Blocked(f"{field}.{key}:MISSING")
    identity_hash = _sha(value.get("identity_hash"), f"{field}.identity_hash")
    if canonical_json_hash({key: item for key, item in value.items() if key != "identity_hash"}) != identity_hash:
        raise G23Blocked(f"{field}:HASH_MISMATCH")
    return value


def _validate_registry_artifact(value: object, registry_hash: str, vector_names: Sequence[str] | None = None) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or value.get("schema_version") != "stage2-parameter-registry-artifact-v1":
        raise G23Blocked("parameter_registry_artifact:SCHEMA_REQUIRED")
    if value.get("status") == "MISSING" or value.get("registry_hash") != registry_hash:
        raise G23Blocked("parameter_registry_artifact:REGISTRY_REQUIRED")
    _canonical_payload_hash(value, "parameter_registry_artifact")
    groups = value.get("parameter_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise G23Blocked("parameter_registry_artifact:GROUPS_REQUIRED")
    if vector_names is not None and set(str(name) for name in groups) != set(vector_names):
        raise G23Blocked("parameter_registry_artifact:PARAMETER_SET_MISMATCH")
    for name, item in groups.items():
        if not isinstance(name, str) or not isinstance(item, Mapping):
            raise G23Blocked("parameter_registry_artifact:GROUP_ENTRY_INVALID")
        if not isinstance(item.get("layer"), str) or not item.get("layer") or not isinstance(item.get("module"), str) or not item.get("module"):
            raise G23Blocked("parameter_registry_artifact:EXPLICIT_LAYER_MODULE_REQUIRED")
    return value


def _validate_producer_provenance(
    convergence: Mapping[str, object],
    *,
    repo_root: Path | None,
) -> str:
    provenance = convergence.get("producer_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("schema_version") != "stage2-reference-producer-provenance-v2":
        raise G23Blocked("producer_provenance:TRUSTED_REPOSITORY_BINDING_REQUIRED")
    head = _commit(provenance.get("head_commit"), "producer_provenance.head_commit")
    tree = provenance.get("head_tree")
    if not isinstance(tree, str) or _COMMIT.fullmatch(tree) is None:
        raise G23Blocked("producer_provenance.head_tree:GIT_OBJECT_REQUIRED")
    if provenance.get("tracked_clean") is not True:
        raise G23Blocked("producer_provenance:TRACKED_FILES_NOT_CLEAN")
    sources = provenance.get("source_bytes")
    if not isinstance(sources, list) or not sources:
        raise G23Blocked("producer_provenance.source_bytes:REQUIRED")
    source_payload: list[Mapping[str, object]] = []
    for index, item in enumerate(sources):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "git_blob"}:
            raise G23Blocked(f"producer_provenance.source_bytes[{index}]:FIELDS")
        relative = _path(item.get("path"), f"producer_provenance.source_bytes[{index}].path")
        source_payload.append(item)
        _sha(item.get("sha256"), f"producer_provenance.source_bytes[{index}].sha256")
        blob = item.get("git_blob")
        if not isinstance(blob, str) or _COMMIT.fullmatch(blob) is None:
            raise G23Blocked(f"producer_provenance.source_bytes[{index}].git_blob:GIT_OBJECT_REQUIRED")
        if repo_root is not None:
            path = _resolve(repo_root, relative)
            if not path.is_file() or _digest_bytes(path) != item.get("sha256"):
                raise G23Blocked(f"producer_provenance.source_bytes[{index}]:SOURCE_DRIFT")
            try:
                blob = subprocess.run(
                    ["git", "-C", str(repo_root), "hash-object", relative],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError) as error:
                raise G23Blocked("producer_provenance:GIT_UNAVAILABLE") from error
            if blob != item.get("git_blob"):
                raise G23Blocked(f"producer_provenance.source_bytes[{index}]:GIT_BLOB_DRIFT")
    if canonical_json_hash(
        {
            "head_commit": head,
            "head_tree": tree,
            "tracked_clean": True,
            "source_bytes": source_payload,
        }
    ) != provenance.get("provenance_hash"):
        raise G23Blocked("producer_provenance:HASH_MISMATCH")
    if repo_root is not None:
        try:
            actual_head = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            actual_tree = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD^{tree}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise G23Blocked("producer_provenance:GIT_UNAVAILABLE") from error
        if actual_head != head or actual_tree != tree or status:
            raise G23Blocked("producer_provenance:REPOSITORY_HEAD_OR_CLEAN_STATE_DRIFT")
    if convergence.get("stage2_reference_producer_commit") != head:
        raise G23Blocked("producer_commit:PROVENANCE_MISMATCH")
    return head


def _validate_external_lineage(
    root: Path,
    convergence: Mapping[str, object],
    result: TaskRunResult,
    *,
    cell_id: str,
) -> Mapping[str, Mapping[str, object]]:
    raw = convergence.get("external_lineage")
    if not isinstance(raw, Mapping):
        raise G23Blocked("external_lineage:ALL_AUTHORITATIVE_REFS_REQUIRED")
    expected = {
        "s23_asset_resolution": "asset_resolution",
        "s23_six_cell_manifest": "six_cell_manifest",
        "resolved_config": "resolved_config",
        "checkpoint_manifest": "checkpoint_manifest",
        "model_manifest": "model_manifest",
        "data_manifest": "data_manifest",
        "tokenizer_manifest": "tokenizer_manifest",
        "parameter_registry": "parameter_registry",
        "preregistration": "preregistration",
        "sizing_plan": "reference_sizing_plan",
    }
    loaded: dict[str, Mapping[str, object]] = {}
    for name, kind in expected.items():
        item = raw.get(name)
        if not isinstance(item, Mapping):
            raise G23Blocked(f"external_lineage.{name}:MISSING")
        ref = item.get("commit_ref")
        if not isinstance(ref, str):
            raise G23Blocked(f"external_lineage.{name}.commit_ref:MISSING")
        _reject_symlink_chain(root, _path(ref, f"external_lineage.{name}.commit_ref"), f"external_lineage.{name}.commit_ref")
        artifact = _load_task_artifact_strict(root, ref, f"external_lineage.{name}")
        try:
            source_manifest = validate_external_manifest(
                artifact,
                root,
                expected_kind=kind,
                declared_sources=item.get("source_manifest"),
            )
        except (OSError, TypeError, ValueError) as error:
            raise G23Blocked(f"external_lineage.{name}:SOURCE_MANIFEST_INVALID") from error
        if item.get("artifact_hash") != artifact.identity.artifact_hash or item.get("config_hash") != artifact.identity.config_hash or item.get("task_id") != artifact.identity.task_id or item.get("payload_hash") != canonical_json_hash(artifact.payload):
            raise G23Blocked(f"external_lineage.{name}:IDENTITY_MISMATCH")
        if item.get("source_refs") != list(artifact.source_refs) or item.get("source_manifest") != source_manifest:
            raise G23Blocked(f"external_lineage.{name}:SOURCE_BINDING_MISMATCH")
        loaded[name] = artifact.payload
    config = loaded["resolved_config"]
    if config.get("config_hash") != result.config_hash and config.get("resolved_config_hash") != result.config_hash:
        raise G23Blocked("external_lineage.resolved_config:CONFIG_MISMATCH")
    preregistration = loaded["preregistration"]
    if preregistration.get("schema_version") != "stage2-preregistration-v1" or preregistration.get("scope") != "formal":
        raise G23Blocked("external_lineage.preregistration:FORMAL_SCOPE_REQUIRED")
    supplied_prereg_hash = preregistration.get("preregistration_hash")
    if not isinstance(supplied_prereg_hash, str) or canonical_json_hash({key: item for key, item in preregistration.items() if key != "preregistration_hash"}) != supplied_prereg_hash:
        raise G23Blocked("external_lineage.preregistration:HASH_MISMATCH")
    precision = preregistration.get("equivalence_and_precision")
    if not isinstance(precision, Mapping) or not isinstance(precision.get("absolute_floors"), Mapping):
        raise G23Blocked("external_lineage.preregistration:FORMULA_CONTRACT_MISSING")
    manifest = loaded["s23_six_cell_manifest"]
    if manifest.get("schema_version") != "stage2-s204-six-cell-manifest-v1" or manifest.get("status") != "READY" or manifest.get("scope") != "formal":
        raise G23Blocked("external_lineage.s23_six_cell_manifest:FORMAL_READY_REQUIRED")
    rows = manifest.get("checkpoints")
    if not isinstance(rows, list) or tuple(item.get("cell_id") for item in rows if isinstance(item, Mapping)) != EXPECTED_CELL_IDS:
        raise G23Blocked("external_lineage.s23_six_cell_manifest:CELL_SET_INVALID")
    if not any(isinstance(item, Mapping) and item.get("cell_id") == cell_id for item in rows):
        raise G23Blocked("external_lineage.s23_six_cell_manifest:CELL_MISSING")
    asset_resolution = loaded["s23_asset_resolution"]
    if asset_resolution.get("schema_version") != "stage2-task-asset-resolution-v1":
        raise G23Blocked("external_lineage.s23_asset_resolution:SCHEMA_REQUIRED")
    if asset_resolution.get("scope") != "formal":
        raise G23Blocked("external_lineage.s23_asset_resolution:FORMAL_SCOPE_REQUIRED")
    if asset_resolution.get("six_cell_manifest_hash") != manifest.get("manifest_hash"):
        raise G23Blocked("external_lineage.s23_asset_resolution:MANIFEST_HASH_MISMATCH")
    _commit(asset_resolution.get("producer_commit"), "external_lineage.s23_asset_resolution.producer_commit")
    _commit(manifest.get("asset_producer_commit"), "external_lineage.s23_six_cell_manifest.asset_producer_commit")
    _commit(manifest.get("asset_execution_commit"), "external_lineage.s23_six_cell_manifest.asset_execution_commit")
    tokenizer = loaded["tokenizer_manifest"]
    if tokenizer.get("schema_version") != "tokenizer-manifest-v1":
        raise G23Blocked("external_lineage.tokenizer_manifest:SCHEMA_REQUIRED")
    for field in ("asset_id", "revision", "checkpoint_id"):
        if not isinstance(tokenizer.get(field), str) or not tokenizer.get(field):
            raise G23Blocked(f"external_lineage.tokenizer_manifest.{field}:REQUIRED")
    return loaded


def _validate_capacity_preflight(
    root: Path,
    convergence: Mapping[str, object],
    external_payloads: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    value = convergence.get("capacity_preflight")
    if not isinstance(value, Mapping) or value.get("schema_version") != "stage2-reference-capacity-preflight-v1":
        raise G23Blocked("capacity_preflight:REQUIRED")
    declared = _sha(value.get("artifact_hash"), "capacity_preflight.artifact_hash")
    if canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) != declared:
        raise G23Blocked("capacity_preflight:HASH_MISMATCH")
    model = external_payloads.get("model_manifest", {})
    parameter_count = value.get("parameter_count")
    external_count = model.get("parameter_count") if isinstance(model, Mapping) else None
    if isinstance(external_count, int) and not isinstance(external_count, bool) and parameter_count != external_count:
        raise G23Blocked("capacity_preflight:PARAMETER_COUNT_DRIFT")
    try:
        count = int(parameter_count)
        block_size = int(value.get("block_size"))
        max_sample = int(value.get("candidate_max_sample_count_per_stream"))
        max_blocks = int(value.get("max_block_count_per_stream"))
    except (TypeError, ValueError) as error:
        raise G23Blocked("capacity_preflight:FIELDS_INVALID") from error
    if count <= 0 or block_size <= 0 or max_sample <= 0 or max_blocks != max_sample // block_size:
        raise G23Blocked("capacity_preflight:FORMULA_FIELDS_INVALID")
    storage_mode = value.get("storage_mode", "raw-shards-v1")
    if storage_mode == "bounded-online-fp64-v1":
        expected_shards = 0
        expected_moments = 25 * count * 8
    elif storage_mode == "raw-shards-v1":
        expected_shards = max_blocks * 2 * count * 8
        expected_moments = max_blocks * 4 * count * 8
    else:
        raise G23Blocked("capacity_preflight:STORAGE_MODE_INVALID")
    expected_disk = int((expected_shards + expected_moments) * 1.20 + 64 * 1024**2)
    if value.get("single_copy_shard_bytes") != expected_shards or value.get("snapshot_moment_bytes") != expected_moments or value.get("estimated_disk_bytes") != expected_disk:
        raise G23Blocked("capacity_preflight:FORMULA_MISMATCH")
    if value.get("disk_ok") is not True or value.get("ram_ok") is not True:
        raise G23Blocked("capacity_preflight:FAIL_CLOSED_NOT_READY")
    if shutil.disk_usage(root).free < expected_disk:
        raise G23Blocked("capacity_preflight:CURRENT_FREE_DISK_INSUFFICIENT")
    available_ram: int | None = None
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64), ("sullAvailExtendedVirtual", ctypes.c_uint64)]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            available_ram = int(status.ullAvailPhys)
    except (AttributeError, OSError, TypeError):
        pass
    if available_ram is None:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    available_ram = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
    peak_ram = int(value.get("peak_ram_bytes", 0))
    if available_ram is None or peak_ram <= 0 or available_ram < peak_ram:
        raise G23Blocked("capacity_preflight:CURRENT_AVAILABLE_RAM_INSUFFICIENT_OR_UNKNOWN")
    return value


def _prepare_cell(root: Path, source: CellInput, *, repo_root: Path | None = None) -> _CellEvidence:
    evidence = _CellEvidence(source, workspace_root=root)
    try:
        result, result_payload = _task_result(root, source)
        evidence.result, evidence.result_payload = result, result_payload
        evidence.identities["result_hash"] = _sha(result.result_hash, "result_hash")
        evidence.identities["config_hash"] = _sha(result.config_hash, "config_hash")
        if source.config_ref is not None:
            config = _load_json(root, source.config_ref, "config")
            if config.get("schema_version") not in {"resolved-config-v1", "resolved-config-v2"} or config.get("config_hash") != result.config_hash:
                raise G23Blocked("config_hash:CONFIG_RESULT_MISMATCH")
        evidence.reference = _artifact(root, result, "reference_result")
        evidence.convergence = _artifact(root, result, "reference_convergence_report")
        evidence.gate = _artifact(root, result, "gate_record")
        rp, cp = evidence.reference.payload, evidence.convergence.payload
        if rp.get("scope") != "formal" or rp.get("formal_eligible") is not False:
            raise G23Blocked("reference:FORMAL_SCOPE_REQUIRED")
        if cp.get("formal_scope") != "formal" or cp.get("formal_eligible") is not False:
            raise G23Blocked("convergence:FORMAL_SCOPE_REQUIRED")
        if cp.get("diagnostics_schema_version") != "stage2-reference-producer-diagnostics-v1":
            raise G23Blocked("convergence:DIAGNOSTICS_SCHEMA_REQUIRED")
        producer_commit = _validate_producer_provenance(cp, repo_root=repo_root)
        six_rows = _validate_six_cell_manifest(cp.get("six_cell_manifest"))
        six_hash = _sha(cp.get("six_cell_manifest_hash"), "six_cell_manifest_hash")
        if six_hash != _sha(cp.get("six_cell_manifest", {}).get("manifest_hash"), "six_cell_manifest.manifest_hash"):  # type: ignore[union-attr]
            raise G23Blocked("six_cell_manifest_hash:MISMATCH")
        cell_id = cp.get("cell_id")
        if cell_id not in EXPECTED_CELL_IDS:
            raise G23Blocked("cell_id:FIXED_SIX_CELL_REQUIRED")
        row = next(item for item in six_rows if item.get("cell_id") == cell_id)
        if source.cell_id != cell_id:
            raise G23Blocked("cell_id:CALLER_REFERENCE_MISMATCH")
        external_payloads = _validate_external_lineage(root, cp, result, cell_id=str(cell_id))
        evidence.external_payloads = external_payloads
        _validate_capacity_preflight(root, cp, external_payloads)
        if canonical_json_hash(external_payloads["s23_six_cell_manifest"]) != canonical_json_hash(cp.get("six_cell_manifest")):
            raise G23Blocked("six_cell_manifest:EXTERNAL_TASK_ARTIFACT_MISMATCH")
        for identity_name, external_name in (
            ("model_identity", "model_manifest"),
            ("data_identity", "data_manifest"),
            ("checkpoint_identity", "checkpoint_manifest"),
        ):
            identity = cp.get(identity_name)
            external = external_payloads.get(external_name)
            if not isinstance(identity, Mapping) or not isinstance(external, Mapping):
                raise G23Blocked(f"{identity_name}:EXTERNAL_MANIFEST_MISSING")
            for field_name in (
                "asset_id", "revision", "model_id", "training_stage",
                "checkpoint_id", "checkpoint_hash", "config_hash", "registry_hash",
                "data_range_hash",
            ):
                if field_name in external and field_name in identity and external.get(field_name) != identity.get(field_name):
                    raise G23Blocked(f"{identity_name}:{field_name}:EXTERNAL_MANIFEST_DRIFT")
        metadata_bindings = result.metadata.get("identity_bindings") if isinstance(result.metadata, Mapping) else None
        if not isinstance(metadata_bindings, Mapping):
            raise G23Blocked("task_result.metadata.identity_bindings:MISSING")
        for binding_key in (
            "stage2_reference_producer_commit",
            "producer_provenance",
            "config_identity",
            "checkpoint_identity",
            "registry_identity",
            "model_identity",
            "data_identity",
            "tokenizer_identity",
            "external_lineage",
        ):
            if metadata_bindings.get(binding_key) != cp.get(binding_key):
                raise G23Blocked(f"task_result.identity_bindings.{binding_key}:DRIFT")
        config_identity = _identity_object(
            cp.get("config_identity"),
            "config_identity",
            required=("config_hash", "task_id", "checkpoint_config_hash"),
        )
        if config_identity.get("config_hash") != result.config_hash or config_identity.get("task_id") != result.task_id:
            raise G23Blocked("config_identity:RESULT_MISMATCH")
        model_identity = _identity_object(cp.get("model_identity"), "model_identity", required=("asset_id", "revision"))
        data_identity = _identity_object(cp.get("data_identity"), "data_identity", required=("asset_id", "revision"))
        if data_identity.get("data_range_hash") != cp.get("six_cell_manifest", {}).get("data_range_hash"):  # type: ignore[union-attr]
            raise G23Blocked("data_identity:DATA_RANGE_MISMATCH")
        data_manifest = cp.get("six_cell_manifest", {}).get("data")  # type: ignore[union-attr]
        if isinstance(data_manifest, Mapping) and data_manifest.get("dataset_id") is not None and data_identity.get("asset_id") != data_manifest.get("dataset_id"):
            raise G23Blocked("data_identity:DATASET_ID_MISMATCH")
        checkpoint_identity = _identity_object(
            cp.get("checkpoint_identity"),
            "checkpoint_identity",
            required=("checkpoint_id", "checkpoint_revision", "checkpoint_asset_id", "model_id", "training_stage", "checkpoint_hash", "config_hash", "registry_hash"),
        )
        if any(checkpoint_identity.get(key) != row.get(key) for key in ("cell_id", "model_id", "training_stage", "checkpoint_hash", "config_hash", "registry_hash")):
            raise G23Blocked("checkpoint_identity:SIX_CELL_ROW_MISMATCH")
        if checkpoint_identity.get("checkpoint_asset_id") != model_identity.get("asset_id"):
            raise G23Blocked("checkpoint_identity:MODEL_ASSET_MISMATCH")
        if checkpoint_identity.get("checkpoint_revision") != model_identity.get("revision"):
            raise G23Blocked("checkpoint_identity:MODEL_REVISION_MISMATCH")
        if config_identity.get("checkpoint_config_hash") != checkpoint_identity.get("config_hash"):
            raise G23Blocked("config_identity:CHECKPOINT_CONFIG_MISMATCH")
        checkpoint_lineage = cp.get("external_lineage", {}).get("checkpoint_manifest") if isinstance(cp.get("external_lineage"), Mapping) else None
        if not isinstance(checkpoint_lineage, Mapping) or not isinstance(result.checkpoint_ref, str) or result.checkpoint_ref != checkpoint_lineage.get("commit_ref"):
            raise G23Blocked("task_result.checkpoint_ref:EXTERNAL_BINDING_MISMATCH")
        provider = cp.get("provider")
        provider_checkpoint = provider.get("checkpoint_identity") if isinstance(provider, Mapping) else None
        external_checkpoint = external_payloads.get("checkpoint_manifest")
        if not isinstance(provider_checkpoint, Mapping) or not isinstance(external_checkpoint, Mapping):
            raise G23Blocked("provider.checkpoint_identity:REQUIRED")
        provider_required = (
            "model_id", "training_stage", "checkpoint_id", "revision",
            "root_ref", "manifest_ref", "manifest_sha256", "registry_hash",
            "config_hash",
        )
        if set(provider_checkpoint) != set(provider_required):
            raise G23Blocked("provider.checkpoint_identity:FIELDS_INVALID")
        provider_expected = {
            "model_id": checkpoint_identity.get("model_id"),
            "training_stage": checkpoint_identity.get("training_stage"),
            "checkpoint_id": checkpoint_identity.get("checkpoint_id"),
            "revision": checkpoint_identity.get("checkpoint_revision"),
            "root_ref": row.get("checkpoint_root_ref"),
            "manifest_ref": external_checkpoint.get("source_manifest_ref"),
            "manifest_sha256": external_checkpoint.get("source_manifest_sha256"),
            "registry_hash": checkpoint_identity.get("registry_hash"),
            "config_hash": checkpoint_identity.get("config_hash"),
        }
        if provider_checkpoint != provider_expected:
            raise G23Blocked("provider.checkpoint_identity:TOP_LEVEL_MISMATCH")
        if (
            external_checkpoint.get("checkpoint_id") != provider_checkpoint.get("checkpoint_id")
            or external_checkpoint.get("model_id") != provider_checkpoint.get("model_id")
            or external_checkpoint.get("revision") != provider_checkpoint.get("revision")
            or external_checkpoint.get("source_manifest_ref") != row.get("checkpoint_manifest_ref")
            or external_checkpoint.get("source_manifest_sha256") != provider_checkpoint.get("manifest_sha256")
        ):
            raise G23Blocked("provider.checkpoint_identity:EXTERNAL_MANIFEST_MISMATCH")
        source_manifest = external_checkpoint.get("checkpoint_manifest")
        if not isinstance(source_manifest, Mapping) or canonical_json_hash(dict(source_manifest)) != provider_checkpoint.get("manifest_sha256"):
            raise G23Blocked("provider.checkpoint_identity:SOURCE_MANIFEST_HASH_MISMATCH")
        if row.get("checkpoint_root_ref") != provider_checkpoint.get("root_ref"):
            raise G23Blocked("provider.checkpoint_identity:SIX_CELL_ROOT_MISMATCH")
        tokenizer_identity = _identity_object(
            cp.get("tokenizer_identity"),
            "tokenizer_identity",
            required=("asset_id", "revision", "checkpoint_id"),
        )
        tokenizer_manifest = external_payloads.get("tokenizer_manifest")
        if not isinstance(tokenizer_manifest, Mapping) or any(
            tokenizer_identity.get(key) != tokenizer_manifest.get(key)
            for key in ("asset_id", "revision", "checkpoint_id")
        ):
            raise G23Blocked("tokenizer_identity:EXTERNAL_MANIFEST_MISMATCH")
        if tokenizer_identity.get("checkpoint_id") != checkpoint_identity.get("checkpoint_id"):
            raise G23Blocked("tokenizer_identity:CHECKPOINT_MISMATCH")
        registry_identity = _identity_object(cp.get("registry_identity"), "registry_identity", required=("registry_hash", "parameter_registry_artifact_hash"))
        row_registry_hash = _sha(row.get("registry_hash"), f"six_cell_manifest.{cell_id}.registry_hash")
        registry_hash = _sha(registry_identity.get("registry_hash"), "registry_identity.registry_hash")
        if registry_hash != row_registry_hash or registry_hash != rp.get("registry_hash"):
            raise G23Blocked("registry_identity:DRIFT")
        registry_artifact = _validate_registry_artifact(cp.get("parameter_registry_artifact"), row_registry_hash)
        if registry_artifact.get("artifact_hash") != registry_identity.get("parameter_registry_artifact_hash"):
            raise G23Blocked("parameter_registry_artifact:IDENTITY_HASH_MISMATCH")
        external_registry = external_payloads.get("parameter_registry")
        if external_registry is None or canonical_json_hash(external_registry) != canonical_json_hash(registry_artifact):
            raise G23Blocked("parameter_registry_artifact:EXTERNAL_REF_MISMATCH")
        evidence.identities.update({
            "registry_hash": registry_hash,
            "checkpoint_hash": _sha(checkpoint_identity.get("checkpoint_hash"), "checkpoint_hash"),
            "producer_commit": producer_commit,
            "cell_id": str(cell_id),
            "six_cell_manifest_hash": six_hash,
            "sizing_plan_hash": _sha(cp.get("sizing_plan_artifact_hash"), "sizing_plan_artifact_hash"),
        })
        plan = cp.get("sizing_plan")
        if not isinstance(plan, Mapping):
            raise G23Blocked("sizing_plan:SCHEMA_REQUIRED")
        try:
            selected_candidate = cp.get("selected_sample_count_per_stream")
            if selected_candidate is not None and (
                isinstance(selected_candidate, bool) or not isinstance(selected_candidate, int)
            ):
                raise ValueError("sizing_plan:SELECTED_NODE_INVALID")
            validate_sizing_plan_contract(
                plan,
                selected_sample_count=selected_candidate,
                field="sizing_plan",
            )
        except ValueError as error:
            raise G23Blocked(str(error)) from error
        if _sha(plan.get("artifact_hash"), "sizing_plan.artifact_hash") != evidence.identities["sizing_plan_hash"]:
            raise G23Blocked("sizing_plan:HASH_MISMATCH")
        if canonical_json_hash({key: item for key, item in plan.items() if key != "artifact_hash"}) != evidence.identities["sizing_plan_hash"]:
            raise G23Blocked("sizing_plan:CONTENT_HASH_MISMATCH")
        reference_id = plan.get("reference_id")
        if not isinstance(reference_id, str) or not reference_id:
            raise G23Blocked("sizing_plan.reference_id:IDENTITY_REQUIRED")
        evidence.identities["reference_id"] = reference_id
        external_plan = external_payloads.get("sizing_plan")
        if not isinstance(external_plan, Mapping) or canonical_json_hash(external_plan) != canonical_json_hash(plan):
            raise G23Blocked("sizing_plan:EXTERNAL_TASK_ARTIFACT_MISMATCH")
        one_shot = cp.get("one_shot_result")
        if not isinstance(one_shot, Mapping):
            raise G23Blocked("one_shot_result:RAW_DIAGNOSTIC_MISSING")
        one_shot_hash = _canonical_payload_hash(one_shot, "one_shot_result")
        evidence.identities["one_shot_result_hash"] = one_shot_hash
        evidence.identities["sizing_result_hash"] = _sha(one_shot.get("sizing_result_hash"), "sizing_result_hash")
        evidence.identities["provider_state_digest"] = _sha(one_shot.get("provider_state_digest"), "provider_state_digest")
        evidence.identities["stream_a_draw_hash"] = _sha(one_shot.get("stream_a_draw_hash"), "stream_a_draw_hash")
        evidence.identities["stream_b_draw_hash"] = _sha(one_shot.get("stream_b_draw_hash"), "stream_b_draw_hash")
        if evidence.identities["stream_a_draw_hash"] == evidence.identities["stream_b_draw_hash"]:
            raise G23Blocked("draw_hash:STREAMS_NOT_INDEPENDENT")
        delta = cp.get("candidate_delta_sci")
        if not isinstance(delta, Mapping) or delta.get("schema_version") != "stage2-reference-delta-sci-v2":
            raise G23Blocked("delta_sci:SIZING_DERIVED_ARTIFACT_REQUIRED")
        delta_ref = _path(delta.get("source_ref"), "candidate_delta_sci.source_ref")
        delta_hash = _sha(delta.get("source_hash"), "candidate_delta_sci.source_hash")
        if cp.get("candidate_delta_sci_source") != delta_ref or cp.get("candidate_delta_sci_source_hash") != delta_hash:
            raise G23Blocked("delta_sci:SOURCE_BINDING_MISMATCH")
        source_value = _load_json(root, delta_ref, "candidate_delta_sci")
        source_artifact_hash = _sha(source_value.get("artifact_hash"), "candidate_delta_sci.artifact_hash")
        if source_artifact_hash != delta_hash or source_value != {
            key: item for key, item in delta.items()
            if key not in {"source_ref", "source_hash", "source_artifact_hash"}
        }:
            raise G23Blocked("delta_sci:SIZING_DERIVED_SOURCE_DRIFT")
        lineage_formula = cp.get("external_lineage", {}).get("preregistration") if isinstance(cp.get("external_lineage"), Mapping) else None
        if not isinstance(lineage_formula, Mapping) or lineage_formula.get("artifact_hash") != cp.get("formula_contract_hash"):
            raise G23Blocked("delta_sci:FORMULA_CONTRACT_BINDING_REQUIRED")
        formula_payload = external_payloads.get("preregistration")
        if not isinstance(formula_payload, Mapping) or cp.get("formula_contract") != formula_payload:
            raise G23Blocked("delta_sci:FORMULA_CONTRACT_DRIFT")
        if delta.get("formula_contract_hash") != cp.get("formula_contract_hash"):
            raise G23Blocked("delta_sci:FORMULA_HASH_MISMATCH")
        evidence.identities["sizing_delta_sci_hash"] = delta_hash
        one_shot_plan = cp.get("one_shot_plan")
        if not isinstance(one_shot_plan, Mapping) or one_shot_plan.get("schema_version") not in {
            "stage2-reference-one-shot-plan-v1",
            "stage2-reference-one-shot-plan-v2",
        }:
            raise G23Blocked("one_shot_plan:SCHEMA_REQUIRED")
        selected_candidate = cp.get("selected_sample_count_per_stream")
        selected_one_shot = one_shot_plan.get("sample_count_per_stream")
        if (
            isinstance(selected_candidate, bool)
            or not isinstance(selected_candidate, int)
            or isinstance(selected_one_shot, bool)
            or not isinstance(selected_one_shot, int)
            or selected_one_shot != selected_candidate
        ):
            raise G23Blocked("one_shot_plan:SELECTED_SIZING_NODE_MISMATCH")
        try:
            validate_sizing_plan_contract(
                plan,
                selected_sample_count=selected_one_shot,
                field="one_shot_plan.sizing_plan",
            )
        except ValueError as error:
            raise G23Blocked(str(error)) from error
        one_shot_plan_hash = _canonical_payload_hash(one_shot_plan, "one_shot_plan")
        evidence.identities["one_shot_plan_hash"] = one_shot_plan_hash
        if one_shot_plan.get("sizing_result_hash") != evidence.identities["sizing_result_hash"] or one_shot.get("plan_hash") != one_shot_plan_hash:
            raise G23Blocked("one_shot_identity:PLAN_RESULT_MISMATCH")
        if one_shot.get("provider_state_digest") != evidence.identities["provider_state_digest"] or one_shot.get("registry_hash") != evidence.identities["registry_hash"]:
            raise G23Blocked("one_shot_identity:PROVIDER_REGISTRY_MISMATCH")
        if cp.get("sizing_result_hash") != evidence.identities["sizing_result_hash"]:
            raise G23Blocked("sizing_result_hash:CONVERGENCE_MISMATCH")
        replay = cp.get("resume_replay")
        if not isinstance(replay, Mapping):
            raise G23Blocked("resume_replay:RAW_DIAGNOSTIC_MISSING")
        sizing_draw_hash = _sha(cp.get("sizing_draw_hash"), "sizing_draw_hash")
        sizing_identity_hash = _sha(cp.get("sizing_identity_hash"), "sizing_identity_hash")
        expected_sizing_identity_hash = canonical_json_hash(
            {
                "plan_hash": evidence.identities["sizing_plan_hash"],
                "provider_state_digest": evidence.identities["provider_state_digest"],
                "registry_hash": evidence.identities["registry_hash"],
                "sizing_draw_hash": sizing_draw_hash,
                "sizing_stream": "reference_sizing",
                "draw_start_position": int(plan.get("draw_start_position", 0)),
                "draw_end_position_exclusive": int(
                    plan.get("draw_end_position_exclusive", max(plan.get("candidate_sample_counts", [0])))
                ),
            }
        )
        if sizing_identity_hash != expected_sizing_identity_hash:
            raise G23Blocked("sizing_identity_hash:FORMULA_MISMATCH")
        if sizing_draw_hash in {evidence.identities["stream_a_draw_hash"], evidence.identities["stream_b_draw_hash"]}:
            raise G23Blocked("draw_hash:SIZING_FINAL_REUSE")
        if one_shot_plan.get("sizing_stream") != "reference_sizing" or one_shot_plan.get("stream_a") != "reference_A" or one_shot_plan.get("stream_b") != "reference_B":
            raise G23Blocked("one_shot_plan:STREAM_IDENTITY_INVALID")
        if one_shot_plan.get("schema_version") == "stage2-reference-one-shot-plan-v2":
            plan_final_start = int(plan.get("final_stream_start_position", plan.get("draw_start_position", 0)))
            for prefix in ("stream_a", "stream_b"):
                start = one_shot_plan.get(f"{prefix}_draw_start_position")
                end = one_shot_plan.get(f"{prefix}_draw_end_position_exclusive")
                if start != plan_final_start or end != start + int(selected_one_shot):
                    raise G23Blocked("one_shot_plan:FINAL_SEGMENT_IDENTITY_MISMATCH")
        sizing_expected = {
            "plan_hash": evidence.identities["sizing_plan_hash"],
            "provider_state_digest": evidence.identities["provider_state_digest"],
            "registry_hash": evidence.identities["registry_hash"],
            "weighting_assumptions": one_shot.get("weighting_assumptions"),
            "sizing_draw_hash": sizing_draw_hash,
            "sizing_identity_hash": sizing_identity_hash,
            "sizing_stream": True,
        }
        final_sizing_identity = _sha(cp.get("sizing_result_identity_hash"), "sizing_result_identity_hash")
        expected_final_sizing_identity = canonical_json_hash(
            {
                "sizing_result_hash": evidence.identities["sizing_result_hash"],
                "provider_state_digest": evidence.identities["provider_state_digest"],
                "registry_hash": evidence.identities["registry_hash"],
                "stream_a_draw_hash": evidence.identities["stream_a_draw_hash"],
                "stream_b_draw_hash": evidence.identities["stream_b_draw_hash"],
                "stream_a_draw_start_position": int(one_shot_plan.get("stream_a_draw_start_position", 0)),
                "stream_a_draw_end_position_exclusive": int(one_shot_plan.get("stream_a_draw_end_position_exclusive", selected_one_shot)),
                "stream_b_draw_start_position": int(one_shot_plan.get("stream_b_draw_start_position", 0)),
                "stream_b_draw_end_position_exclusive": int(one_shot_plan.get("stream_b_draw_end_position_exclusive", selected_one_shot)),
            }
        )
        if final_sizing_identity != expected_final_sizing_identity:
            raise G23Blocked("sizing_result_identity_hash:FORMULA_MISMATCH")
        final_expected = {
            "plan_hash": one_shot_plan_hash,
            "sizing_result_hash": evidence.identities["sizing_result_hash"],
            "provider_state_digest": evidence.identities["provider_state_digest"],
            "registry_hash": evidence.identities["registry_hash"],
            "weighting_assumptions": one_shot.get("weighting_assumptions"),
            "stream_a_draw_hash": evidence.identities["stream_a_draw_hash"],
            "stream_b_draw_hash": evidence.identities["stream_b_draw_hash"],
            "sizing_result_identity_hash": final_sizing_identity,
            "final_length_required": True,
        }
        bundle_ref = _path(rp.get("tensor_bundle_ref"), "tensor_bundle_ref")
        bundle_path = _resolve(root, bundle_ref)
        _reject_symlinks_under(bundle_path, "tensor_bundle")
        state, bundle = load_tensor_bundle(bundle_path)
        if not isinstance(state, Mapping):
            raise G23Blocked("tensor_bundle:OBJECT_REQUIRED")
        if bundle.manifest_sha256 != _sha(rp.get("tensor_bundle_manifest_hash"), "tensor_bundle_manifest_hash"):
            raise G23Blocked("tensor_bundle:MANIFEST_HASH_DRIFT")
        evidence.bundle_state, evidence.bundle_manifest_hash = state, bundle.manifest_sha256
        evidence.identities["bundle_manifest_hash"] = bundle.manifest_sha256
        metadata = rp.get("metadata")
        if not isinstance(metadata, Mapping):
            raise G23Blocked("reference.metadata:OBJECT_REQUIRED")
        for key, identity in (("sizing_result_hash", "sizing_result_hash"), ("sequence_variance_hash", "sequence_variance_hash")):
            if key in metadata:
                evidence.identities[identity] = _sha(metadata[key], identity)
        one_shot = cp.get("one_shot_result")
        if not isinstance(one_shot, Mapping):
            raise G23Blocked("one_shot_result:RAW_DIAGNOSTIC_MISSING")
        payload_without_hash = {key: value for key, value in one_shot.items() if key != "artifact_hash"}
        one_shot_hash = _sha(one_shot.get("artifact_hash"), "one_shot_result_hash")
        if one_shot_hash != canonical_json_hash(payload_without_hash):
            raise G23Blocked("one_shot_result_hash:HASH_MISMATCH")
        evidence.identities["one_shot_result_hash"] = one_shot_hash
        evidence.identities["sizing_result_hash"] = _sha(one_shot.get("sizing_result_hash"), "sizing_result_hash")
        evidence.identities["provider_state_digest"] = _sha(one_shot.get("provider_state_digest"), "provider_state_digest")
        evidence.identities["stream_a_draw_hash"] = _sha(one_shot.get("stream_a_draw_hash"), "stream_a_draw_hash")
        evidence.identities["stream_b_draw_hash"] = _sha(one_shot.get("stream_b_draw_hash"), "stream_b_draw_hash")
        if evidence.identities["stream_a_draw_hash"] == evidence.identities["stream_b_draw_hash"]:
            raise G23Blocked("draw_hash:STREAMS_NOT_INDEPENDENT")
        sizing_root, final_root = _resume_roots(root, result.artifact_refs["reference_result"])
        evidence.sizing_states = _load_resume_commits(
            root,
            sizing_root,
            "stage2-reference-progress-state-v1",
            identities=sizing_expected,
        )
        evidence.sizing_root = sizing_root
        final_states = _load_resume_commits(
            root,
            final_root,
            "stage2-reference-one-shot-progress-v1",
            identities=final_expected,
        )
        evidence.final_state = final_states[-1]
        evidence.final_states = final_states
        evidence.final_root = final_root
        if not bool(evidence.final_state.get("bounded_storage")) and int(evidence.final_state.get("processed_block_pairs", 0)) != len(final_states):
            raise G23Blocked("final_resume:COMMITTED_LENGTH_MISMATCH")
        if not bool(evidence.sizing_states[-1].get("bounded_storage")) and int(evidence.sizing_states[-1].get("processed_block_pairs", 0)) != len(evidence.sizing_states):
            raise G23Blocked("sizing_resume:COMMITTED_LENGTH_MISMATCH")
    except G23Blocked as error:
        evidence.reasons.append(str(error))
    except Exception as error:
        evidence.reasons.append(f"UNEXPECTED_INPUT_ERROR:{type(error).__name__}")
    return evidence


def _sizing_vectors(evidence: _CellEvidence) -> tuple[list[int], list[dict[str, np.ndarray]], Mapping[str, object]]:
    if not evidence.sizing_states:
        raise G23Blocked("sizing:RAW_DIAGNOSTICS_MISSING")
    if evidence.sizing_root is None:
        raise G23Blocked("sizing:SHARD_ROOT_MISSING")
    plan = None
    if evidence.convergence is not None:
        plan = evidence.convergence.payload.get("sizing_plan")
    if not isinstance(plan, Mapping):
        raise G23Blocked("sizing.plan:RAW_DIAGNOSTIC_MISSING")
    raw_counts = plan.get("candidate_sample_counts")
    if not isinstance(raw_counts, list) or len(raw_counts) < 2:
        raise G23Blocked("sizing.candidate_counts:RAW_DIAGNOSTIC_MISSING")
    try:
        counts = list(
            validate_sizing_plan_contract(
                plan,
                selected_sample_count=plan.get("selected_sample_count_per_stream"),
                field="sizing_plan",
            )
        )
        block_size = int(plan["block_size"])
    except (TypeError, ValueError, OverflowError) as error:
        raise G23Blocked(str(error)) from error
    states_by_count: dict[int, Mapping[str, object]] = {}
    for state in evidence.sizing_states:
        sample_count = int(state.get("processed_block_pairs", 0)) * int(plan.get("block_size", 0))
        if sample_count in counts:
            states_by_count[sample_count] = state
    if any(count not in states_by_count for count in counts[-2:]):
        raise G23Blocked("sizing:ADJACENT_FINAL_NODES_MISSING")
    selected_declared = evidence.convergence.payload.get("selected_sample_count_per_stream") if evidence.convergence is not None else None
    if isinstance(selected_declared, bool) or not isinstance(selected_declared, int):
        raise G23Blocked("sizing:selected_sample_count_per_stream:RAW_DIAGNOSTIC_MISSING")
    selected_states = {
        state.get("selected_sample_count_per_stream")
        for state in evidence.sizing_states
        if state.get("selected_sample_count_per_stream") is not None
    }
    if selected_states != {selected_declared} or selected_declared not in counts:
        raise G23Blocked("sizing:SELECTED_NODE_NOT_INDEPENDENTLY_COMMITTED")
    vectors: list[dict[str, np.ndarray]] = []
    bounded = bool(evidence.sizing_states[-1].get("bounded_storage"))
    for count in counts[-2:]:
        state = states_by_count[count]
        if bounded:
            moments = _bounded_moments_strict(
                state.get("a"), f"sizing.bounded.{count}.a", require_higher=False
            )
            expected_blocks = count // block_size
            if moments.count != expected_blocks or int(state.get("processed_block_pairs", -1)) != expected_blocks:
                raise G23Blocked("sizing:BOUNDED_COUNT_MISMATCH")
            assumptions = state.get("weighting_assumptions")
            if not isinstance(assumptions, Mapping):
                raise G23Blocked("sizing.weighting_assumptions:MISSING")
            try:
                validate_weighting_contract(assumptions, field="sizing.weighting_assumptions")
                vectors.append(moments.u(assumptions=assumptions))
            except (TypeError, ValueError) as error:
                raise G23Blocked(f"sizing:BOUNDED_MOMENTS_INVALID:{type(error).__name__}") from error
            continue
        refs_a = state.get("shard_refs_a")
        if not isinstance(refs_a, list) or not refs_a:
            raise G23Blocked("sizing.shard_refs_a:SHARDS_REQUIRED")
        store = _ReferenceShardStore(evidence.sizing_root)
        assumptions = state.get("weighting_assumptions")
        if not isinstance(assumptions, Mapping):
            raise G23Blocked("sizing.weighting_assumptions:MISSING")
        moments_a = _moments_from_shards(store, refs_a, assumptions)
        moments = moments_a
        if not isinstance(state.get("a"), Mapping) or not isinstance(state.get("b"), Mapping):
            raise G23Blocked("sizing.moments:STATE_MISSING")
        rebuilt_a = moments_a.to_state()
        _moments_equal(rebuilt_a, state["a"], "sizing.moments_a")
        vectors.append(_u_from_moments(moments.to_state(), "sizing.u"))
    return counts[-2:], vectors, plan


def _final_vectors_bounded(
    evidence: _CellEvidence,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    _BoundedMoments,
    _BoundedMoments,
    ReferenceUncertainty,
]:
    if evidence.final_state is None or evidence.bundle_state is None:
        raise G23Blocked("final:RAW_DIAGNOSTICS_MISSING")
    state = evidence.final_state
    if not bool(state.get("bounded_storage")):
        raise G23Blocked("final:BOUNDED_STORAGE_REQUIRED")
    moments_a = _bounded_moments_strict(
        state.get("a"), "final.bounded.a", require_higher=True
    )
    moments_b = _bounded_moments_strict(
        state.get("b"), "final.bounded.b", require_higher=True
    )
    processed = int(state.get("processed_block_pairs", 0))
    if processed <= 0 or moments_a.count != processed or moments_b.count != processed:
        raise G23Blocked("final:BOUNDED_COUNT_MISMATCH")
    assumptions = state.get("weighting_assumptions")
    if not isinstance(assumptions, Mapping):
        raise G23Blocked("final.weighting_assumptions:MISSING")
    try:
        validate_weighting_contract(assumptions, field="final.weighting_assumptions")
        combined = moments_a.combine(moments_b)
        bias = combined.u(assumptions=assumptions)
        mean_a, mean_b = moments_a.mean(), moments_b.mean()
        cross = {name: mean_a[name] * mean_b[name] for name in mean_a}
        ranking = {name: np.square(value) for name, value in combined.mean().items()}
        recomputed = estimate_reference_uncertainty_bounded(moments_a, moments_b)
        sequence_variance = estimate_sequence_variance_bounded(
            combined,
            block_size=int(state.get("block_size", 0)),
        )
    except (TypeError, ValueError) as error:
        raise G23Blocked(f"final:BOUNDED_RECOMPUTE_FAILED:{type(error).__name__}") from error
    bundle_state = evidence.bundle_state
    for key, value in (
        ("bias_reference", bias),
        ("cross_reference", cross),
        ("ranking_reference", ranking),
    ):
        loaded = _vector(bundle_state.get(key), f"bundle.{key}")
        _compatible(value, loaded, f"bundle.{key}")
        if not all(np.array_equal(value[name], loaded[name]) for name in value):
            raise G23Blocked(f"bundle.{key}:BOUNDED_VALUE_MISMATCH")
        if evidence.reference is not None and _sha(
            evidence.reference.payload.get(f"{key}_hash"), f"reference.{key}_hash"
        ) != _vector_digest(value):
            raise G23Blocked(f"reference.{key}_hash:VALUE_MISMATCH")
    uncertainty = bundle_state.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise G23Blocked("bundle.uncertainty:MISSING")
    for name, expected in (
        ("bias_variance", recomputed.bias_variance),
        ("cross_variance", recomputed.cross_variance),
        ("ranking_variance", recomputed.ranking_variance),
    ):
        loaded = _vector(uncertainty.get(name), f"bundle.{name}")
        _compatible(expected, loaded, f"bundle.{name}")
        if not all(np.allclose(expected[key], loaded[key], rtol=1e-12, atol=1e-12) for key in expected):
            raise G23Blocked(f"bundle.{name}:BOUNDED_JACKKNIFE_VALUE_MISMATCH")
    loaded_sequence = _vector(bundle_state.get("sequence_variance"), "bundle.sequence_variance")
    _compatible(sequence_variance, loaded_sequence, "bundle.sequence_variance")
    if not all(np.allclose(sequence_variance[name], loaded_sequence[name], rtol=1e-12, atol=1e-12) for name in sequence_variance):
        raise G23Blocked("bundle.sequence_variance:BOUNDED_VALUE_MISMATCH")
    return bias, cross, ranking, moments_a, moments_b, recomputed


def _final_vectors(
    evidence: _CellEvidence,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    Sequence[Mapping[str, np.ndarray]],
    Sequence[Mapping[str, np.ndarray]],
    ReferenceUncertainty,
]:
    if evidence.final_state is None or evidence.bundle_state is None:
        raise G23Blocked("final:RAW_DIAGNOSTICS_MISSING")
    state = evidence.final_state
    if evidence.final_root is None:
        raise G23Blocked("final:SHARD_ROOT_MISSING")
    blocks_a, weights_a, refs_a = _load_shard_records(
        evidence.final_root, state.get("shard_refs_a"), "final.shard_refs_a"
    )
    blocks_b, weights_b, refs_b = _load_shard_records(
        evidence.final_root, state.get("shard_refs_b"), "final.shard_refs_b"
    )
    if len(blocks_a) != len(blocks_b):
        raise G23Blocked("final.shard_prefix:COUNT_MISMATCH")
    for left, right in zip(blocks_a, blocks_b):
        _compatible(left, right, "final.blocks")
    moments_a = _moments_from_blocks(blocks_a, weights_a, "final.moments_a")
    moments_b = _moments_from_blocks(blocks_b, weights_b, "final.moments_b")
    mean_a, mean_b = _weighted_mean_from_moments(moments_a, "final.mean_a"), _weighted_mean_from_moments(moments_b, "final.mean_b")
    mean_all = _weighted_mean_from_moments(_merge_moments(moments_a, moments_b, "final.moments"), "final.mean_all")
    if not isinstance(state.get("a"), Mapping) or not isinstance(state.get("b"), Mapping):
        raise G23Blocked("final.moments:STATE_MISSING")
    _moments_equal(moments_a, state["a"], "final.moments_a")
    _moments_equal(moments_b, state["b"], "final.moments_b")
    bias = _u_from_moments(_merge_moments(moments_a, moments_b, "final.moments"), "final.bias")
    cross = {name: mean_a[name] * mean_b[name] for name in mean_a}
    ranking = {name: np.square(mean_all[name]) for name in mean_all}
    bstate = evidence.bundle_state
    for key, value in (("bias_reference", bias), ("cross_reference", cross), ("ranking_reference", ranking)):
        if key not in bstate:
            raise G23Blocked(f"bundle.{key}:MISSING")
        loaded = _vector(bstate[key], f"bundle.{key}")
        _compatible(value, loaded, f"bundle.{key}")
        if not all(np.array_equal(value[name], loaded[name]) for name in value):
            raise G23Blocked(f"bundle.{key}:RAW_VALUE_MISMATCH")
        if evidence.reference is not None:
            manifest_hash = evidence.reference.payload.get(f"{key}_hash")
            if _sha(manifest_hash, f"reference.{key}_hash") != _vector_digest(value):
                raise G23Blocked(f"reference.{key}_hash:VALUE_MISMATCH")
    try:
        assumptions = validate_weighting_contract(
            state.get("weighting_assumptions"), field="final.weighting_assumptions"
        )
        recomputed = estimate_reference_uncertainty_shards(
            _ReferenceShardStore(evidence.final_root), refs_a, refs_b, assumptions
        )
    except (TypeError, ValueError, OSError) as error:
        raise G23Blocked(f"final.uncertainty:RAW_JACKKNIFE_RECOMPUTE_FAILED:{type(error).__name__}") from error
    uncertainty = bstate.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise G23Blocked("bundle.uncertainty:MISSING")
    for name, expected in (
        ("bias_variance", recomputed.bias_variance),
        ("cross_variance", recomputed.cross_variance),
        ("ranking_variance", recomputed.ranking_variance),
    ):
        loaded = _vector(uncertainty.get(name), f"bundle.{name}")
        _compatible(expected, loaded, f"bundle.{name}")
        if not all(np.array_equal(expected[key], loaded[key]) for key in expected):
            raise G23Blocked(f"bundle.{name}:RAW_JACKKNIFE_VALUE_MISMATCH")
    # A producer may publish the complete uncertainty payload as well as the
    # tensor maps.  If it does, every self-reported scalar/hash is checked
    # against the object independently rebuilt above; none is an authority.
    reported_payload = recomputed.to_dict()
    for key, reported in uncertainty.items():
        if key in {"bias_variance", "cross_variance", "ranking_variance"}:
            continue
        if key in reported_payload and reported != reported_payload[key]:
            raise G23Blocked(f"bundle.uncertainty.{key}:RAW_JACKKNIFE_METADATA_MISMATCH")
    return bias, cross, ranking, blocks_a, blocks_b, recomputed


def _group_map(vector: Mapping[str, np.ndarray], evidence: _CellEvidence) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    layer: dict[str, list[str]] = {}
    module: dict[str, list[str]] = {}
    if evidence.convergence is None:
        raise G23Blocked("parameter_registry_artifact:CONVERGENCE_MISSING")
    registry_hash = evidence.identities.get("registry_hash")
    registry = _validate_registry_artifact(
        evidence.convergence.payload.get("parameter_registry_artifact"),
        _sha(registry_hash, "registry_hash"),
        tuple(vector),
    )
    groups = registry["parameter_groups"]
    assert isinstance(groups, Mapping)
    for name in vector:
        item = groups.get(name)
        if not isinstance(item, Mapping):
            raise G23Blocked(f"parameter_registry_artifact.{name}:MISSING")
        layer_name, module_name = item.get("layer"), item.get("module")
        if not isinstance(layer_name, str) or not isinstance(module_name, str) or not layer_name or not module_name:
            raise G23Blocked(f"parameter_registry_artifact.{name}:EXPLICIT_LAYER_MODULE_REQUIRED")
        layer.setdefault(layer_name, []).append(name)
        module.setdefault(module_name, []).append(name)
    return layer, module


def _aggregate(vector: Mapping[str, np.ndarray], groups: Mapping[str, Sequence[str]]) -> tuple[np.ndarray, np.ndarray]:
    totals, means = [], []
    for names in sorted(groups):
        values = np.concatenate([np.abs(vector[name]).reshape(-1) for name in groups[names]])
        totals.append(float(values.sum()))
        means.append(float(values.mean()))
    return np.asarray(totals, dtype=np.float64), np.asarray(means, dtype=np.float64)


def _sizing_sequence_variance(
    store: _ReferenceShardStore,
    refs: Sequence[Mapping[str, object]],
    assumptions: Mapping[str, object],
    block_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Recompute weighted sizing mean/sequence variance with two shard passes."""

    if block_size <= 0 or len(refs) < 2:
        raise G23Blocked("delta_sci:SIZING_SHARDS_TOO_SHORT")
    moments = _moments_from_shards(store, refs, assumptions)
    mean = moments.mean()
    variance = {name: np.zeros_like(value, dtype=np.float64) for name, value in mean.items()}
    for ref in refs:
        try:
            vector, weight, _ = store.load(ref)
        except (OSError, TypeError, ValueError) as error:
            raise G23Blocked("delta_sci:SIZING_SHARD_INVALID") from error
        for name in variance:
            variance[name] += float(weight) * np.square(vector[name] - mean[name])
    sigma2 = {name: value * float(block_size) / float(moments.n1) for name, value in variance.items()}
    return mean, sigma2


def _delta_sci(
    evidence: _CellEvidence,
    counts: Sequence[int],
    *,
    evaluator_commit: str,
) -> tuple[dict[str, dict[int, float]], dict[str, float], Mapping[str, object]]:
    """Recompute the S2.6-B margin from hash-bound sizing moments.

    ``counts`` are the S2.4 sizing/convergence nodes.  They are deliberately
    not the estimator batch-size domain consumed by S2.6.  Older r23
    producers published ``delta_sci_by_endpoint`` keyed by those sizing
    counts; the evaluator retains that artifact as immutable provenance and
    emits a corrected, content-addressed sidecar keyed by
    ``CANDIDATE_BATCH_SIZES``.  No draw, threshold, or producer artifact is
    rewritten.
    """

    if evidence.convergence is None or evidence.sizing_root is None or not evidence.sizing_states:
        raise G23Blocked("delta_sci:SIZING_RAW_DIAGNOSTICS_MISSING")
    cp = evidence.convergence.payload
    source = cp.get("candidate_delta_sci")
    if not isinstance(source, Mapping) or source.get("schema_version") != "stage2-reference-delta-sci-v2":
        raise G23Blocked("delta_sci:SIZING_DERIVED_ARTIFACT_REQUIRED")
    source_ref = _path(source.get("source_ref"), "candidate_delta_sci.source_ref")
    source_hash = _sha(source.get("source_hash"), "candidate_delta_sci.source_hash")
    if source.get("source_artifact_hash") != source_hash:
        raise G23Blocked("delta_sci:SIZING_SOURCE_ARTIFACT_HASH_MISMATCH")
    if evidence.workspace_root is None:
        raise G23Blocked("delta_sci:WORKSPACE_ROOT_MISSING")
    source_value = _load_json(evidence.workspace_root, source_ref, "candidate_delta_sci")
    if _sha(source_value.get("artifact_hash"), "candidate_delta_sci.artifact_hash") != source_hash:
        raise G23Blocked("delta_sci:SIZING_SOURCE_HASH_MISMATCH")
    expected_source = {key: item for key, item in source.items() if key not in {"source_ref", "source_hash", "source_artifact_hash"}}
    if source_value != expected_source:
        raise G23Blocked("delta_sci:SIZING_SOURCE_CONTENT_MISMATCH")
    plan = cp.get("sizing_plan")
    if not isinstance(plan, Mapping):
        raise G23Blocked("delta_sci:SIZING_PLAN_MISSING")
    block_size = int(plan.get("block_size", 0))
    if block_size <= 0:
        raise G23Blocked("delta_sci:BLOCK_SIZE_INVALID")
    external = evidence.external_payloads or {}
    prereg = external.get("preregistration")
    if not isinstance(prereg, Mapping):
        raise G23Blocked("delta_sci:FORMULA_CONTRACT_MISSING")
    precision = prereg.get("equivalence_and_precision")
    if not isinstance(precision, Mapping):
        raise G23Blocked("delta_sci:FORMULA_CONTRACT_MISSING")
    floors = precision.get("absolute_floors")
    if not isinstance(floors, Mapping) or set(floors) != {"tau_model", "tau_layer", "tau_module", "tau_coord", "tau_nmse"}:
        raise G23Blocked("delta_sci:ABSOLUTE_FLOORS_MISSING")
    floors_f = {name: _finite(floors.get(name), f"delta_sci.{name}") for name in ("tau_model", "tau_layer", "tau_module", "tau_coord", "tau_nmse")}
    if any(value <= 0.0 for value in floors_f.values()):
        raise G23Blocked("delta_sci:ABSOLUTE_FLOORS_INVALID")
    if (
        source.get("source_kind") not in {"reference_sizing_raw_shards", "reference_sizing_bounded_online"}
        or source.get("formula_version") != "stage2-reference-sizing-margin-v1"
        or source.get("formula") != "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B"
        or source.get("formula_contract_hash") != cp.get("formula_contract_hash")
        or source.get("reference_id") != plan.get("reference_id")
        or source.get("sizing_plan_hash") != cp.get("sizing_plan_artifact_hash")
        or source.get("sizing_result_hash") != cp.get("sizing_result_hash")
        or source.get("registry_hash") != evidence.identities.get("registry_hash")
        or source.get("candidate_sample_counts") != list(counts)
        or source.get("absolute_floors") != dict(floors)
    ):
        raise G23Blocked("delta_sci:SOURCE_IDENTITY_MISMATCH")
    producer_delta = source_value.get("delta_sci_by_endpoint")
    producer_signal = source_value.get("signal_scale_by_endpoint")
    producer_noise = source_value.get("noise_scale_by_endpoint")
    if not all(isinstance(value, Mapping) for value in (producer_delta, producer_signal, producer_noise)):
        raise G23Blocked("delta_sci:PRODUCER_SCALE_TABLES_MISSING")
    sizing_keys = {str(count) for count in counts}
    batch_keys = {str(batch_size) for batch_size in CORRECTED_DELTA_BATCH_SIZES}
    producer_key_sets: set[frozenset[str]] = set()
    for table in (producer_delta, producer_signal, producer_noise):
        assert isinstance(table, Mapping)
        if set(table) != {"model_total", "layer", "module"}:
            raise G23Blocked("delta_sci:PRODUCER_ENDPOINT_DOMAIN_INVALID")
        for endpoint in ("model_total", "layer", "module"):
            values = table.get(endpoint)
            if not isinstance(values, Mapping):
                raise G23Blocked(f"delta_sci:PRODUCER_ENDPOINT_TABLE_MISSING:{endpoint}")
            if any(not isinstance(key, str) for key in values):
                raise G23Blocked(f"delta_sci:PRODUCER_SCALE_KEY_INVALID:{endpoint}")
            producer_key_sets.add(frozenset(str(key) for key in values))
    if len(producer_key_sets) != 1:
        raise G23Blocked("delta_sci:PRODUCER_SCALE_KEY_DRIFT")
    producer_keys = next(iter(producer_key_sets))
    if producer_keys == frozenset(sizing_keys):
        producer_table_mode = "sizing_nodes_legacy"
    elif producer_keys == frozenset(batch_keys):
        producer_table_mode = "candidate_batch_sizes"
    else:
        raise G23Blocked("delta_sci:PRODUCER_SCALE_KEY_DOMAIN_INVALID")
    state_by_count: dict[int, Mapping[str, object]] = {}
    for state in evidence.sizing_states:
        count = int(state.get("processed_block_pairs", 0)) * block_size
        if count in counts:
            state_by_count[count] = state
    if any(count not in state_by_count for count in counts):
        raise G23Blocked("delta_sci:SIZING_CANDIDATE_STATE_MISSING")
    convergence_payload = evidence.convergence.payload
    selected_value = convergence_payload.get("selected_sample_count_per_stream")
    if isinstance(selected_value, bool) or not isinstance(selected_value, int) or selected_value not in state_by_count:
        raise G23Blocked("delta_sci:SELECTED_SIZING_NODE_MISSING")
    selected_count = int(selected_value)
    if producer_table_mode == "candidate_batch_sizes":
        if source.get("delta_sci_batch_sizes") != list(CORRECTED_DELTA_BATCH_SIZES):
            raise G23Blocked("delta_sci:PRODUCER_BATCH_DOMAIN_BINDING_MISMATCH")
        if source.get("selected_sample_count_per_stream") != selected_count:
            raise G23Blocked("delta_sci:PRODUCER_SELECTED_NODE_BINDING_MISMATCH")
    bounded = source.get("source_kind") == "reference_sizing_bounded_online"
    selected_scales: dict[str, tuple[float, float]] | None = None
    if bounded:
        selected_moments = _BoundedMoments.from_state(state_by_count[selected_count]["a"])  # type: ignore[arg-type]
        first_vector = selected_moments.g1
        store = None
    else:
        refs_selected = state_by_count[selected_count].get("shard_refs_a")
        if not isinstance(refs_selected, list) or not refs_selected:
            raise G23Blocked("delta_sci:SIZING_SHARDS_MISSING")
        store = _ReferenceShardStore(evidence.sizing_root)
        try:
            first_vector, _, _ = store.load(refs_selected[0])
        except (OSError, TypeError, ValueError) as error:
            raise G23Blocked("delta_sci:SIZING_SHARD_INVALID") from error
    names = tuple(sorted(first_vector))
    layer_groups, module_groups = _group_map(first_vector, evidence)
    computed: dict[str, dict[int, float]] = {endpoint: {} for endpoint in ("model_total", "layer", "module")}
    computed_signal: dict[str, dict[int, float]] = {endpoint: {} for endpoint in computed}
    computed_noise: dict[str, dict[int, float]] = {endpoint: {} for endpoint in computed}
    legacy_computed: dict[str, dict[int, float]] = {endpoint: {} for endpoint in computed}
    legacy_signal: dict[str, dict[int, float]] = {endpoint: {} for endpoint in computed}
    legacy_noise: dict[str, dict[int, float]] = {endpoint: {} for endpoint in computed}
    computed_nodes: list[Mapping[str, object]] = []
    for count in counts:
        state = state_by_count[count]
        assumptions = state.get("weighting_assumptions")
        if not isinstance(assumptions, Mapping):
            raise G23Blocked("delta_sci:WEIGHTING_ASSUMPTIONS_MISSING")
        if bounded:
            moments = _BoundedMoments.from_state(state["a"])  # type: ignore[arg-type]
            mean = moments.mean()
            sigma2 = estimate_sequence_variance_bounded(moments, block_size=block_size)
            refs = []
        else:
            refs = state.get("shard_refs_a")
            if not isinstance(refs, list) or len(refs) != int(state.get("processed_block_pairs", 0)):
                raise G23Blocked("delta_sci:SIZING_SHARD_PREFIX_INVALID")
            mean, sigma2 = _sizing_sequence_variance(store, refs, assumptions, block_size)
        a = {name: np.square(mean[name]) for name in mean}
        model_signal = max(abs(float(sum(np.sum(value) for value in a.values()))), floors_f["tau_model"])
        model_noise = abs(float(sum(np.sum(value) for value in sigma2.values())))
        layer_signal_values = [float(sum(np.sum(a[name]) for name in names)) for names in layer_groups.values()]
        layer_noise_values = [float(sum(np.sum(sigma2[name]) for name in names)) for names in layer_groups.values()]
        module_signal_values = [float(sum(np.sum(a[name]) for name in names)) for names in module_groups.values()]
        module_noise_values = [float(sum(np.sum(sigma2[name]) for name in names)) for names in module_groups.values()]
        endpoint_values = {
            "model_total": (max(abs(model_signal), floors_f["tau_model"]), abs(model_noise)),
            "layer": (max(float(sum(abs(value) for value in layer_signal_values)), floors_f["tau_layer"]), float(sum(abs(value) for value in layer_noise_values))),
            "module": (max(float(sum(abs(value) for value in module_signal_values)), floors_f["tau_module"]), float(sum(abs(value) for value in module_noise_values))),
        }
        for endpoint, (signal, sigma2_total) in endpoint_values.items():
            noise = float(sigma2_total) / float(count)
            legacy_signal[endpoint][count] = signal
            legacy_noise[endpoint][count] = noise
            legacy_computed[endpoint][count] = max(0.10 * noise, 0.01 * signal)
        if count == selected_count:
            selected_scales = dict(endpoint_values)
        state_digest = (
            canonical_json_hash({"checkpoint_schema": _BoundedCheckpointStore.schema_version, "plan_hash": plan.get("artifact_hash"), "sample_count": count, "moments_hash": _bounded_moments_digest(moments)})
            if bounded else _ReferenceSnapshotStore._state_digest(state)
        )
        shard_refs_hash = canonical_json_hash([]) if bounded else canonical_json_hash([
            {"shard_hash": ref.get("shard_hash"), "manifest_hash": ref.get("manifest_hash"), "weight": ref.get("weight")}
            for ref in refs if isinstance(ref, Mapping)
        ])
        computed_nodes.append({"sample_count": count, "state_digest": state_digest, "shard_refs_hash": shard_refs_hash, "mean_hash": _vector_digest(mean), "sequence_variance_hash": _vector_digest(sigma2)})
    if selected_scales is None:
        raise G23Blocked("delta_sci:SELECTED_SIZING_SCALES_MISSING")
    for batch_size in CORRECTED_DELTA_BATCH_SIZES:
        for endpoint, (signal, sigma2) in selected_scales.items():
            noise = float(sigma2) / float(batch_size)
            delta = max(0.10 * noise, 0.01 * signal)
            if not all(math.isfinite(value) and value > 0 for value in (signal, noise, delta)):
                raise G23Blocked(f"delta_sci:{endpoint}:B{batch_size}:NON_FINITE")
            computed[endpoint][batch_size] = delta
            computed_signal[endpoint][batch_size] = signal
            computed_noise[endpoint][batch_size] = noise
    expected_computed = {endpoint: {str(batch_size): value for batch_size, value in mapping.items()} for endpoint, mapping in computed.items()}
    expected_signal = {endpoint: {str(batch_size): value for batch_size, value in mapping.items()} for endpoint, mapping in computed_signal.items()}
    expected_noise = {endpoint: {str(batch_size): value for batch_size, value in mapping.items()} for endpoint, mapping in computed_noise.items()}
    expected_legacy = {endpoint: {str(count): value for count, value in mapping.items()} for endpoint, mapping in legacy_computed.items()}
    expected_legacy_signal = {endpoint: {str(count): value for count, value in mapping.items()} for endpoint, mapping in legacy_signal.items()}
    expected_legacy_noise = {endpoint: {str(count): value for count, value in mapping.items()} for endpoint, mapping in legacy_noise.items()}
    if producer_table_mode == "sizing_nodes_legacy":
        if producer_delta != expected_legacy:
            raise G23Blocked("delta_sci:LEGACY_PRODUCER_FORMULA_RECOMPUTE_MISMATCH")
        if producer_signal != expected_legacy_signal or producer_noise != expected_legacy_noise:
            raise G23Blocked("delta_sci:LEGACY_PRODUCER_SCALE_RECOMPUTE_MISMATCH")
    elif producer_table_mode == "candidate_batch_sizes":
        if producer_delta != expected_computed:
            raise G23Blocked("delta_sci:PRODUCER_FORMULA_RECOMPUTE_MISMATCH")
        if producer_signal != expected_signal or producer_noise != expected_noise:
            raise G23Blocked("delta_sci:PRODUCER_SCALE_RECOMPUTE_MISMATCH")
    if source_value.get("sizing_nodes") != computed_nodes:
        raise G23Blocked("delta_sci:SIZING_NODE_BINDING_MISMATCH")
    correction_reason = (
        "producer_delta_sci_keyed_by_sizing_nodes; S2.6_requires_candidate_batch_sizes"
        if producer_table_mode == "sizing_nodes_legacy"
        else "producer_delta_sci_candidate_batch_sizes_verified_by_evaluator"
    )
    sidecar: dict[str, object] = {
        "schema_version": CORRECTED_DELTA_SCHEMA_VERSION,
        "source_producer_schema_version": source.get("schema_version"),
        "source_producer_ref": source_ref,
        "source_producer_artifact_hash": source_hash,
        "source_producer_table_mode": producer_table_mode,
        "source_producer_commit": evidence.identities.get("producer_commit"),
        "evaluator_commit": evaluator_commit,
        "evaluator_source_sha256": _digest_bytes(Path(__file__).resolve()),
        "formula_contract_hash": source.get("formula_contract_hash"),
        "formula_version": source.get("formula_version"),
        "formula": source.get("formula"),
        "absolute_floors": dict(floors),
        "reference_id": plan.get("reference_id"),
        "sizing_result_hash": source.get("sizing_result_hash"),
        "sizing_plan_hash": source.get("sizing_plan_hash"),
        "registry_hash": source.get("registry_hash"),
        "candidate_sample_counts": list(counts),
        "delta_sci_batch_sizes": list(CORRECTED_DELTA_BATCH_SIZES),
        "selected_sample_count_per_stream": selected_count,
        "delta_sci_by_endpoint": expected_computed,
        "signal_scale_by_endpoint": expected_signal,
        "noise_scale_by_endpoint": expected_noise,
        "sizing_nodes": computed_nodes,
        "correction_reason": correction_reason,
    }
    sidecar["artifact_hash"] = canonical_json_hash(sidecar)
    return computed, {endpoint: min(mapping.values()) for endpoint, mapping in computed.items()}, sidecar


def _u_from_blocks_longdouble(
    blocks: Sequence[Mapping[str, np.ndarray]],
    weights: Sequence[object],
    field: str,
) -> dict[str, np.ndarray]:
    """Independently recompute weighted block-U values in extended precision."""

    if not blocks or len(blocks) != len(weights):
        raise G23Blocked(f"{field}:RAW_BLOCKS_AND_WEIGHTS_REQUIRED")
    parsed_weights = [np.longdouble(_finite(value, f"{field}.weight")) for value in weights]
    if any(value <= 0 for value in parsed_weights):
        raise G23Blocked(f"{field}:POSITIVE_BLOCK_WEIGHTS_REQUIRED")
    first = blocks[0]
    names = tuple(sorted(first))
    g1 = {name: np.zeros(first[name].shape, dtype=np.longdouble) for name in names}
    g2 = {name: np.zeros(first[name].shape, dtype=np.longdouble) for name in names}
    n1 = np.longdouble(0)
    n2 = np.longdouble(0)
    for block, weight in zip(blocks, parsed_weights):
        _compatible(first, block, field)
        n1 += weight
        n2 += weight * weight
        for name in names:
            value = np.asarray(block[name], dtype=np.longdouble)
            g1[name] += weight * value
            g2[name] += weight * weight * value * value
    denominator = n1 * n1 - n2
    if len(blocks) < 2 or denominator <= 0:
        raise G23Blocked(f"{field}:U_DENOMINATOR_INVALID")
    return {
        name: np.asarray((g1[name] * g1[name] - g2[name]) / denominator, dtype=np.float64)
        for name in names
    }


def _numerical_error(evidence: _CellEvidence, reference: Mapping[str, np.ndarray]) -> float:
    if evidence.convergence is None or evidence.bundle_state is None or evidence.final_state is None:
        raise G23Blocked("epsilon_num:RAW_DIAGNOSTIC_MISSING")
    source = evidence.bundle_state.get("numerical_diagnostics")
    metadata = evidence.convergence.payload.get("numerical_diagnostics")
    if not isinstance(source, Mapping) or not isinstance(metadata, Mapping):
        raise G23Blocked("epsilon_num:EXPLICIT_DIAGNOSTIC_REQUIRED")
    if bool(evidence.final_state.get("bounded_storage")):
        if (
            source.get("schema_version") != "stage2-reference-numerical-diagnostics-v2"
            or metadata.get("schema_version") != "stage2-reference-numerical-diagnostics-v2"
            or source.get("storage_mode") != "bounded-online-fp64-v1"
            or metadata.get("storage_mode") != "bounded-online-fp64-v1"
        ):
            raise G23Blocked("epsilon_num:BOUNDED_DIAGNOSTIC_SCHEMA_REQUIRED")
        _canonical_payload_hash(metadata, "epsilon_num.metadata")
        moments_a = _bounded_moments_strict(
            evidence.final_state.get("a"), "epsilon_num.bounded.a", require_higher=True
        )
        moments_b = _bounded_moments_strict(
            evidence.final_state.get("b"), "epsilon_num.bounded.b", require_higher=True
        )
        try:
            expected_high, expected_accumulated, expected_bound = bounded_reference_numeric_diagnostics(
                moments_a, moments_b
            )
        except ValueError as error:
            raise G23Blocked("epsilon_num:BOUNDED_RECOMPUTE_FAILED") from error
        high = _vector(source.get("high_precision"), "epsilon_num.high_precision")
        accumulated = _vector(source.get("accumulated"), "epsilon_num.accumulated")
        error_bound = _vector(source.get("error_bound"), "epsilon_num.error_bound")
        for observed, expected, label in (
            (high, expected_high, "HIGH_PRECISION"),
            (accumulated, expected_accumulated, "ACCUMULATED"),
            (error_bound, expected_bound, "ERROR_BOUND"),
        ):
            _compatible(observed, expected, f"epsilon_num.{label.lower()}")
            if not all(np.allclose(observed[name], expected[name], rtol=1e-12, atol=1e-12) for name in expected):
                raise G23Blocked(f"epsilon_num:{label}_RECOMPUTE_MISMATCH")
        manifest_hash = evidence.final_state.get("bounded_checkpoint_manifest_hash")
        state_digest = canonical_json_hash(
            {
                "checkpoint_schema": _BoundedCheckpointStore.schema_version,
                "object_manifest_hash": manifest_hash,
            }
        )
        required_bindings = {
            "bounded_checkpoint_ref": "bounded-checkpoint",
            "bounded_checkpoint_manifest_hash": manifest_hash,
            "bounded_checkpoint_state_digest": state_digest,
            "moments_a_hash": _bounded_moments_digest(moments_a),
            "moments_b_hash": _bounded_moments_digest(moments_b),
            "high_precision_hash": _vector_digest(expected_high),
            "accumulated_hash": _vector_digest(expected_accumulated),
            "error_bound_hash": _vector_digest(expected_bound),
        }
        for key, expected in required_bindings.items():
            if metadata.get(key) != expected or source.get(key) != expected:
                raise G23Blocked(f"epsilon_num:BOUNDED_{key.upper()}_MISMATCH")
        _compatible(expected_accumulated, reference, "epsilon_num.reference")
        if not all(np.allclose(expected_accumulated[name], reference[name], rtol=1e-12, atol=1e-12) for name in reference):
            raise G23Blocked("epsilon_num:REFERENCE_RECOMPUTE_MISMATCH")
        computed_error = max(float(np.max(value)) for value in expected_bound.values())
        if not math.isclose(
            _finite(metadata.get("max_abs_error"), "epsilon_num.max_abs_error"),
            computed_error,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise G23Blocked("epsilon_num:DECLARED_ERROR_MISMATCH")
        return computed_error
    if source.get("schema_version") != "stage2-reference-numerical-diagnostics-v1":
        raise G23Blocked("epsilon_num:RAW_DIAGNOSTIC_SCHEMA_REQUIRED")
    _canonical_payload_hash(metadata, "epsilon_num.metadata")
    if evidence.final_root is None:
        raise G23Blocked("epsilon_num:SHARD_ROOT_REQUIRED")
    blocks_a, weights_a, _ = _load_shard_records(
        evidence.final_root, evidence.final_state.get("shard_refs_a"), "epsilon_num.shard_refs_a"
    )
    blocks_b, weights_b, _ = _load_shard_records(
        evidence.final_root, evidence.final_state.get("shard_refs_b"), "epsilon_num.shard_refs_b"
    )
    blocks = _CombinedSequence(blocks_a, blocks_b)
    weights = weights_a + weights_b
    raw_rows: list[Mapping[str, object]] = []
    for block, weight in zip(blocks, weights):
        raw_rows.append({"vector_hash": _vector_digest(block), "weight": float(weight)})
    raw_digest = canonical_json_hash(raw_rows)
    if source.get("raw_block_digest") != raw_digest or metadata.get("raw_block_digest") != raw_digest:
        raise G23Blocked("epsilon_num:RAW_BLOCK_BINDING_MISMATCH")
    high = source.get("high_precision")
    accumulated = source.get("accumulated")
    if high is None or accumulated is None:
        raise G23Blocked("epsilon_num:HIGH_PRECISION_PAIR_REQUIRED")
    left, right = _vector(high, "epsilon_num.high_precision"), _vector(accumulated, "epsilon_num.accumulated")
    _compatible(left, right, "epsilon_num")
    _compatible(left, reference, "epsilon_num.reference")
    if _vector_digest(left) != _sha(source.get("high_precision_hash"), "epsilon_num.high_precision_hash") or _vector_digest(right) != _sha(source.get("accumulated_hash"), "epsilon_num.accumulated_hash"):
        raise G23Blocked("epsilon_num:VECTOR_HASH_MISMATCH")
    if metadata.get("high_precision_hash") != source.get("high_precision_hash") or metadata.get("accumulated_hash") != source.get("accumulated_hash"):
        raise G23Blocked("epsilon_num:METADATA_HASH_MISMATCH")
    if evidence.reference is None or evidence.reference.payload.get("bias_reference_hash") != _vector_digest(reference):
        raise G23Blocked("epsilon_num:REFERENCE_BINDING_MISMATCH")
    # Recompute both sides from the committed raw blocks.  The producer's
    # numerical vectors are evidence to cross-check, never an authority that
    # can be replaced by a caller-authored scalar.
    expected_high = _u_from_blocks_longdouble(blocks, weights, "epsilon_num.independent_high_precision")
    expected_accumulated = _u_from_moments(
        _moments_from_blocks(blocks, weights, "epsilon_num.independent_accumulation"),
        "epsilon_num.independent_accumulation",
    )
    for name in expected_high:
        if not np.allclose(left[name], expected_high[name], rtol=1e-12, atol=1e-12):
            raise G23Blocked("epsilon_num:HIGH_PRECISION_RECOMPUTE_MISMATCH")
        if not np.allclose(right[name], expected_accumulated[name], rtol=1e-12, atol=1e-12):
            raise G23Blocked("epsilon_num:ACCUMULATED_RECOMPUTE_MISMATCH")
    if not np.allclose(_flat(right), _flat(reference), rtol=1e-12, atol=1e-12):
        raise G23Blocked("epsilon_num:REFERENCE_RECOMPUTE_MISMATCH")
    computed_error = float(max(np.max(np.abs(expected_high[name] - expected_accumulated[name])) for name in expected_high))
    declared_error = _finite(metadata.get("max_abs_error"), "epsilon_num.max_abs_error")
    if not math.isclose(declared_error, computed_error, rel_tol=1e-12, abs_tol=1e-12):
        raise G23Blocked("epsilon_num:DECLARED_ERROR_MISMATCH")
    if evidence.final_root is None:
        raise G23Blocked("epsilon_num:RESUME_ROOT_MISSING")
    commits = sorted((evidence.final_root / "commits").glob("*.json"))
    if not commits:
        raise G23Blocked("epsilon_num:RESUME_COMMITS_MISSING")
    latest_path = commits[-1]
    _reject_symlink_chain(evidence.final_root, latest_path.relative_to(evidence.final_root).as_posix(), "epsilon_num.latest_commit")
    latest = load_canonical_json(latest_path)
    if not isinstance(latest, Mapping):
        raise G23Blocked("epsilon_num:RESUME_LATEST_INVALID")
    if metadata.get("resume_latest_commit_ref") != latest_path.relative_to(evidence.final_root).as_posix():
        raise G23Blocked("epsilon_num:RESUME_COMMIT_REF_MISMATCH")
    if metadata.get("resume_latest_commit_hash") != latest.get("artifact_hash"):
        raise G23Blocked("epsilon_num:RESUME_COMMIT_HASH_MISMATCH")
    if metadata.get("resume_latest_manifest_hash") != latest.get("object_manifest_hash"):
        raise G23Blocked("epsilon_num:RESUME_MANIFEST_HASH_MISMATCH")
    return computed_error


def _state_replay_verified(evidence: _CellEvidence) -> bool:
    if evidence.final_state is None or evidence.convergence is None or evidence.final_root is None:
        raise G23Blocked("state_replay:RAW_DIAGNOSTIC_MISSING")
    invariance = evidence.convergence.payload.get("state_invariance")
    if not isinstance(invariance, Mapping):
        raise G23Blocked("state_replay:STATE_INVARIANCE_REQUIRED")
    model_before = _sha(invariance.get("model_state_before_hash"), "state_replay.model_before")
    model_after = _sha(invariance.get("model_state_after_hash"), "state_replay.model_after")
    rng_before = _sha(invariance.get("rng_state_before_hash"), "state_replay.rng_before")
    rng_after = _sha(invariance.get("rng_state_after_hash"), "state_replay.rng_after")
    if model_before != model_after:
        return False
    before_state = invariance.get("rng_state_before")
    after_state = invariance.get("rng_state_after")
    draw_artifacts = evidence.convergence.payload.get("draw_artifacts")
    if not isinstance(before_state, Mapping) or not isinstance(after_state, Mapping) or not isinstance(draw_artifacts, Mapping):
        raise G23Blocked("state_replay:ACTUAL_RNG_AND_DRAW_ARTIFACTS_REQUIRED")
    if canonical_json_hash(before_state) != rng_before or canonical_json_hash(after_state) != rng_after:
        raise G23Blocked("state_replay:RNG_STATE_HASH_MISMATCH")

    def tupleify(value: object) -> object:
        if isinstance(value, list):
            return tuple(tupleify(item) for item in value)
        if isinstance(value, Mapping):
            return {str(key): tupleify(item) for key, item in value.items()}
        return value

    sampling_plans: dict[str, SamplingPlan] = {}
    for stream in ("reference_sizing", "reference_A", "reference_B"):
        item = draw_artifacts.get(stream)
        if not isinstance(item, Mapping):
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:MISSING")
        raw_plan, raw_manifest, raw_actual = item.get("sampling_plan"), item.get("manifest"), item.get("actual_state")
        if not isinstance(raw_plan, Mapping) or not isinstance(raw_manifest, Mapping) or not isinstance(raw_actual, Mapping):
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:FIELDS")
        try:
            plan = SamplingPlan.from_mapping(raw_plan)
            manifest = DrawStreamManifest.from_manifest(raw_manifest)
        except (TypeError, ValueError) as error:
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:INVALID") from error
        if manifest.stream_state.stream != stream or manifest.sampling_plan_hash != plan.digest:
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:IDENTITY_MISMATCH")
        expected_manifest = plan.draw_manifest(stream, len(manifest.draws)).to_manifest()
        if expected_manifest != dict(raw_manifest):
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:DRAW_REPLAY_MISMATCH")
        if raw_actual.get("stream") != stream or int(raw_actual.get("count", -1)) != len(manifest.draws):
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:STATE_BOUNDARY_MISMATCH")
        try:
            validate_generator_boundary(
                raw_actual,
                sampling=plan,
                stream=stream,
                count=len(manifest.draws),
                field=f"state_replay.draw_artifacts.{stream}",
            )
        except ValueError as error:
            raise G23Blocked(str(error)) from error
        if (
            raw_actual.get("state_before_sha256") != manifest.stream_state.state_before_sha256
            or raw_actual.get("state_after_sha256") != manifest.stream_state.state_after_sha256
        ):
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:MANIFEST_STATE_DIGEST_MISMATCH")
        state_before_obj = tupleify(raw_actual.get("state_before"))
        state_after_obj = tupleify(raw_actual.get("state_after"))
        rng = random.Random()
        try:
            rng.setstate(state_before_obj)  # type: ignore[arg-type]
            for _ in manifest.draws:
                rng.randrange(len(plan.universe.sample_ids))
            replay_after = rng.getstate()
        except (TypeError, ValueError, IndexError) as error:
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:GENERATOR_REPLAY_FAILED") from error
        if replay_after != state_after_obj:
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:GENERATOR_STATE_DRIFT")
        # ``random.setstate`` consumes tuples, while the published artifact
        # deliberately stores the JSON-safe list form.  Hash the exact
        # published representation and only tupleify for generator replay.
        if canonical_json_hash({"algorithm_version": plan.algorithm_version, "state": raw_actual.get("state_before")}) != raw_actual.get("state_before_sha256") or canonical_json_hash({"algorithm_version": plan.algorithm_version, "state": raw_actual.get("state_after")}) != raw_actual.get("state_after_sha256"):
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:GENERATOR_STATE_HASH_MISMATCH")
        if before_state.get("streams", {}).get(stream) != raw_actual.get("state_before") or after_state.get("streams", {}).get(stream) != raw_actual.get("state_after"):
            raise G23Blocked(f"state_replay.draw_artifacts.{stream}:INVARIANCE_BINDING_MISMATCH")
        sampling_plans[stream] = plan

    sizing_plan_payload = evidence.convergence.payload.get("sizing_plan")
    one_shot_plan_payload = evidence.convergence.payload.get("one_shot_plan")
    if not isinstance(sizing_plan_payload, Mapping) or not isinstance(one_shot_plan_payload, Mapping):
        raise G23Blocked("state_replay:RESUME_PLAN_MISSING")
    try:
        sizing_block_size = int(sizing_plan_payload["block_size"])
        final_block_size = int(one_shot_plan_payload["block_size"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise G23Blocked("state_replay:RESUME_BLOCK_SIZE_INVALID") from error
    if sizing_block_size <= 0 or final_block_size <= 0:
        raise G23Blocked("state_replay:RESUME_BLOCK_SIZE_INVALID")

    def validate_resume_rng_state(
        state: Mapping[str, object],
        *,
        schema: str,
        index: int,
        block_size: int,
        streams: tuple[str, ...],
    ) -> None:
        raw_count = state.get("processed_block_pairs")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
            raise G23Blocked(f"state_replay.{schema}[{index}]:COUNT_INVALID")
        count = raw_count * block_size
        raw_boundary = state.get("rng_state")
        if len(streams) == 1:
            if not isinstance(raw_boundary, Mapping):
                raise G23Blocked(f"state_replay.{schema}[{index}]:RNG_STATE_MISSING")
            try:
                start = int(
                    sizing_plan_payload.get("draw_start_position", 0)
                    if schema == "sizing"
                    else one_shot_plan_payload.get(
                        "stream_a_draw_start_position" if streams[0] == "reference_A" else "stream_b_draw_start_position",
                        0,
                    )
                )
                validate_generator_boundary(
                    raw_boundary,
                    sampling=sampling_plans[streams[0]],
                    stream=streams[0],
                    count=count,
                    # Segment boundaries are absolute stream positions.
                    # The v1 prefix remains start=0.
                    start=start,
                    field=f"state_replay.{schema}[{index}].rng_state",
                )
            except ValueError as error:
                raise G23Blocked(str(error)) from error
            return
        # Resume state uses stable endpoint keys ``a``/``b``; the sampling
        # manifests retain their semantic stream names.  Bind them by this
        # fixed pair order rather than accepting caller-chosen keys.
        endpoint_keys = ("a", "b")
        if not isinstance(raw_boundary, Mapping) or set(raw_boundary) != set(endpoint_keys):
            raise G23Blocked(f"state_replay.{schema}[{index}]:RNG_STATE_PAIR_REQUIRED")
        for endpoint_key, stream in zip(endpoint_keys, streams):
            boundary = raw_boundary.get(endpoint_key)
            if not isinstance(boundary, Mapping):
                raise G23Blocked(f"state_replay.{schema}[{index}].{stream}:RNG_STATE_MISSING")
            try:
                start = int(
                    one_shot_plan_payload.get(
                        "stream_a_draw_start_position" if stream == "reference_A" else "stream_b_draw_start_position",
                        0,
                    )
                )
                validate_generator_boundary(
                    boundary,
                    sampling=sampling_plans[stream],
                    stream=stream,
                    count=count,
                    start=start,
                    field=f"state_replay.{schema}[{index}].{stream}",
                )
            except ValueError as error:
                raise G23Blocked(str(error)) from error

    if not evidence.sizing_states or not evidence.final_states:
        raise G23Blocked("state_replay:ALL_RESUME_STATES_REQUIRED")
    sizing_states_to_validate = (
        evidence.sizing_states[-1:]
        if bool(evidence.sizing_states[-1].get("bounded_storage"))
        else evidence.sizing_states
    )
    for index, state in enumerate(sizing_states_to_validate, start=1):
        validate_resume_rng_state(
            state,
            schema="sizing",
            index=index,
            block_size=sizing_block_size,
            streams=("reference_sizing",),
        )
    for index, state in enumerate(evidence.final_states, start=1):
        validate_resume_rng_state(
            state,
            schema="final",
            index=index,
            block_size=final_block_size,
            streams=("reference_A", "reference_B"),
        )
    replay = evidence.convergence.payload.get("resume_replay")
    if not isinstance(replay, Mapping):
        raise G23Blocked("state_replay:RESUME_REPLAY_REQUIRED")
    replay_hash = _sha(replay.get("replay_hash"), "state_replay.replay_hash")
    if canonical_json_hash({key: item for key, item in replay.items() if key != "replay_hash"}) != replay_hash:
        raise G23Blocked("state_replay:REPLAY_HASH_MISMATCH")
    artifact_ref = _path(replay.get("artifact_ref"), "state_replay.artifact_ref")
    _reject_symlink_chain(evidence.final_root, artifact_ref, "state_replay.artifact_ref")
    if bool(evidence.final_state.get("bounded_storage")):
        if (
            replay.get("schema_version") != "stage2-reference-resume-replay-v2"
            or replay.get("storage_mode") != "bounded-online-fp64-v1"
            or artifact_ref != "bounded-checkpoint"
        ):
            raise G23Blocked("state_replay:BOUNDED_REPLAY_SCHEMA_REQUIRED")
        checkpoint_path = evidence.final_root / artifact_ref
        _reject_symlinks_under(checkpoint_path, "state_replay.bounded_checkpoint")
        try:
            state, bundle = load_tensor_bundle(checkpoint_path)
        except (OSError, TypeError, ValueError) as error:
            raise G23Blocked("state_replay:BOUNDED_CHECKPOINT_UNREADABLE") from error
        if not isinstance(state, Mapping) or state.get("checkpoint_schema") != _BoundedCheckpointStore.schema_version:
            raise G23Blocked("state_replay:BOUNDED_CHECKPOINT_SCHEMA")
        state_digest = canonical_json_hash(
            {
                "checkpoint_schema": _BoundedCheckpointStore.schema_version,
                "object_manifest_hash": bundle.manifest_sha256,
            }
        )
        if (
            replay.get("artifact_hash") != bundle.manifest_sha256
            or replay.get("object_manifest_hash") != bundle.manifest_sha256
            or replay.get("state_digest") != state_digest
            or evidence.final_state.get("bounded_checkpoint_manifest_hash") != bundle.manifest_sha256
        ):
            raise G23Blocked("state_replay:BOUNDED_CHECKPOINT_BINDING")
        if int(state.get("processed_block_pairs", -1)) != int(evidence.final_state.get("processed_block_pairs", -2)):
            raise G23Blocked("state_replay:BOUNDED_FINAL_COUNT_DRIFT")
        if replay.get("source_one_shot_result_hash") != replay.get("replayed_one_shot_result_hash") or replay.get("source_one_shot_result_hash") != evidence.identities.get("one_shot_result_hash"):
            raise G23Blocked("state_replay:RESULT_HASH_DRIFT")
        return True
    commit_path = evidence.final_root / artifact_ref
    try:
        commit = load_canonical_json(commit_path)
    except (OSError, ValueError, TypeError) as error:
        raise G23Blocked("state_replay:ARTIFACT_COMMIT_UNREADABLE") from error
    if not isinstance(commit, Mapping) or set(commit) != {"schema_version", "sequence", "state_digest", "object_ref", "object_manifest_hash", "artifact_hash"}:
        raise G23Blocked("state_replay:ARTIFACT_COMMIT_FIELDS")
    if commit.get("schema_version") != "stage2-reference-progress-commit-v1" or commit.get("artifact_hash") != replay.get("artifact_hash"):
        raise G23Blocked("state_replay:ARTIFACT_COMMIT_BINDING")
    commit_payload = {key: item for key, item in commit.items() if key != "artifact_hash"}
    if canonical_json_hash(commit_payload) != commit.get("artifact_hash"):
        raise G23Blocked("state_replay:ARTIFACT_COMMIT_HASH")
    object_ref = _path(commit.get("object_ref"), "state_replay.object_ref")
    _reject_symlink_chain(evidence.final_root, object_ref, "state_replay.object_ref")
    _reject_symlinks_under(evidence.final_root / object_ref, "state_replay.object_ref")
    state, bundle = load_tensor_bundle(evidence.final_root / object_ref)
    if not isinstance(state, Mapping) or state.get("schema_version") != "stage2-reference-one-shot-progress-v1":
        raise G23Blocked("state_replay:ARTIFACT_STATE_SCHEMA")
    if bundle.manifest_sha256 != commit.get("object_manifest_hash") or commit.get("state_digest") != _sha(replay.get("state_digest"), "state_replay.state_digest"):
        raise G23Blocked("state_replay:ARTIFACT_MANIFEST_BINDING")
    if _ReferenceSnapshotStore._state_digest(state) != commit.get("state_digest"):
        raise G23Blocked("state_replay:ARTIFACT_STATE_DIGEST")
    if int(commit.get("sequence", -1)) != int(state.get("processed_block_pairs", -2)):
        raise G23Blocked("state_replay:ARTIFACT_SEQUENCE")
    if (
        int(commit.get("sequence", -1)) != int(evidence.final_state.get("processed_block_pairs", -2))
        or commit.get("state_digest") != _ReferenceSnapshotStore._state_digest(evidence.final_state)
    ):
        raise G23Blocked("state_replay:NOT_FINAL_COMMITTED_STATE")
    if replay.get("object_manifest_hash") != bundle.manifest_sha256:
        raise G23Blocked("state_replay:REPLAY_MANIFEST_HASH")
    if replay.get("source_one_shot_result_hash") != replay.get("replayed_one_shot_result_hash") or replay.get("source_one_shot_result_hash") != evidence.identities.get("one_shot_result_hash"):
        raise G23Blocked("state_replay:RESULT_HASH_DRIFT")
    return True


def _bootstrap_interval(
    blocks: Sequence[Mapping[str, np.ndarray]],
    weights: Sequence[object],
    field: str,
    *,
    square: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted endpoint bootstrap; weights are resampled with their blocks."""

    if not blocks or len(blocks) != len(weights):
        raise G23Blocked(f"{field}:EMPTY_OR_WEIGHT_MISMATCH")
    parsed = [_finite(value, f"{field}.weight") for value in weights]
    rng = np.random.default_rng(2_304_230)
    # A fixed lower bound keeps the quantile deterministic while avoiding an
    # unbounded shard reread multiplier for the smallest formal cell.
    reps = max(128, min(2048, 32 * len(blocks)))
    sampled: list[np.ndarray] = []
    for _ in range(reps):
        indices = rng.integers(0, len(blocks), size=len(blocks))
        selected = [blocks[int(index)] for index in indices]
        selected_weights = [parsed[int(index)] for index in indices]
        mean = _weighted_mean_from_moments(
            _moments_from_blocks(selected, selected_weights, field), f"{field}.mean"
        )
        value = {name: np.square(item) for name, item in mean.items()} if square else mean
        sampled.append(_flat(value))
    matrix = np.stack(sampled, axis=0)
    return np.quantile(matrix, 0.025, axis=0), np.quantile(matrix, 0.975, axis=0), matrix.mean(axis=0)


def _bootstrap_u_diagnostics(
    blocks_a: Sequence[Mapping[str, np.ndarray]],
    blocks_b: Sequence[Mapping[str, np.ndarray]],
    weights_a: Sequence[object],
    weights_b: Sequence[object],
    center: Mapping[str, np.ndarray],
    layer_groups: Mapping[str, Sequence[str]],
    module_groups: Mapping[str, Sequence[str]],
) -> tuple[float, float, float, float]:
    """Bootstrap the U reference, then aggregate model/layer/module totals."""

    if not blocks_a or not blocks_b or len(blocks_a) != len(weights_a) or len(blocks_b) != len(weights_b):
        raise G23Blocked("h_ref:RAW_BLOCKS_EMPTY")
    parsed_a = [_finite(value, "h_ref.weight_a") for value in weights_a]
    parsed_b = [_finite(value, "h_ref.weight_b") for value in weights_b]
    rng_a = np.random.default_rng(2_304_231)
    rng_b = np.random.default_rng(2_304_232)
    reps = max(128, min(2048, 32 * max(len(blocks_a), len(blocks_b))))
    model_totals: list[float] = []
    layer_l1: list[float] = []
    module_l1: list[float] = []
    center_layer, center_module = _aggregate(center, layer_groups), _aggregate(center, module_groups)
    center_layer_total, center_layer_mean = center_layer
    center_module_total, center_module_mean = center_module
    for _ in range(reps):
        indices_a = rng_a.integers(0, len(blocks_a), size=len(blocks_a))
        indices_b = rng_b.integers(0, len(blocks_b), size=len(blocks_b))
        selected_a = [blocks_a[int(index)] for index in indices_a]
        selected_b = [blocks_b[int(index)] for index in indices_b]
        selected_wa = [parsed_a[int(index)] for index in indices_a]
        selected_wb = [parsed_b[int(index)] for index in indices_b]
        moments = _merge_moments(
            _moments_from_blocks(selected_a, selected_wa, "h_ref.bootstrap.a"),
            _moments_from_blocks(selected_b, selected_wb, "h_ref.bootstrap.b"),
            "h_ref.bootstrap",
        )
        u = _u_from_moments(moments, "h_ref.bootstrap.u")
        model_totals.append(float(np.abs(_flat(u)).sum()))
        layer_total, layer_mean = _aggregate(u, layer_groups)
        module_total, module_mean = _aggregate(u, module_groups)
        layer_ratio = max(
            float(np.abs(layer_total - center_layer_total).sum() / max(np.abs(center_layer_total).sum(), 1e-300)),
            float(np.abs(layer_mean - center_layer_mean).sum() / max(np.abs(center_layer_mean).sum(), 1e-300)),
        )
        module_ratio = max(
            float(np.abs(module_total - center_module_total).sum() / max(np.abs(center_module_total).sum(), 1e-300)),
            float(np.abs(module_mean - center_module_mean).sum() / max(np.abs(center_module_mean).sum(), 1e-300)),
        )
        layer_l1.append(layer_ratio)
        module_l1.append(module_ratio)
    model_array = np.asarray(model_totals, dtype=np.float64)
    model_half_width = float((np.quantile(model_array, 0.975) - np.quantile(model_array, 0.025)) / 2.0)
    layer_q95 = float(np.quantile(np.asarray(layer_l1), 0.95))
    module_q95 = float(np.quantile(np.asarray(module_l1), 0.95))
    return max(model_half_width, layer_q95, module_q95), model_half_width, layer_q95, module_q95


def _bootstrap_u_interval(
    blocks: Sequence[Mapping[str, np.ndarray]],
    weights: Sequence[object],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not blocks:
        raise G23Blocked(f"{field}:EMPTY")
    rng = np.random.default_rng(2_304_232)
    reps = max(128, min(2048, 32 * len(blocks)))
    sampled: list[np.ndarray] = []
    for _ in range(reps):
        indices = rng.integers(0, len(blocks), size=len(blocks))
        selected = [blocks[int(index)] for index in indices]
        selected_weights = [weights[int(index)] for index in indices]
        sampled.append(_flat(_u_from_moments(_moments_from_blocks(selected, selected_weights, field), f"{field}.u")))
    matrix = np.stack(sampled, axis=0)
    return np.quantile(matrix, 0.025, axis=0), np.quantile(matrix, 0.975, axis=0)


def _bootstrap_independent_cross_interval(
    blocks_a: Sequence[Mapping[str, np.ndarray]],
    weights_a: Sequence[object],
    blocks_b: Sequence[Mapping[str, np.ndarray]],
    weights_b: Sequence[object],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap two independent streams, then form the endpoint product."""

    if not blocks_a or not blocks_b:
        raise G23Blocked(f"{field}:EMPTY")
    wa = [_finite(item, f"{field}.weight_a") for item in weights_a]
    wb = [_finite(item, f"{field}.weight_b") for item in weights_b]
    if len(wa) != len(blocks_a) or len(wb) != len(blocks_b):
        raise G23Blocked(f"{field}:WEIGHT_MISMATCH")
    rng_a = np.random.default_rng(2_304_240)
    rng_b = np.random.default_rng(2_304_241)
    reps = max(128, min(2048, 32 * max(len(blocks_a), len(blocks_b))))
    rows: list[np.ndarray] = []
    for _ in range(reps):
        ia = rng_a.integers(0, len(blocks_a), size=len(blocks_a))
        ib = rng_b.integers(0, len(blocks_b), size=len(blocks_b))
        ma = _weighted_mean_from_moments(
            _moments_from_blocks([blocks_a[int(i)] for i in ia], [wa[int(i)] for i in ia], field),
            f"{field}.mean_a",
        )
        mb = _weighted_mean_from_moments(
            _moments_from_blocks([blocks_b[int(i)] for i in ib], [wb[int(i)] for i in ib], field),
            f"{field}.mean_b",
        )
        rows.append(_flat({name: ma[name] * mb[name] for name in ma}))
    matrix = np.stack(rows, axis=0)
    return np.quantile(matrix, 0.025, axis=0), np.quantile(matrix, 0.975, axis=0)


def _bootstrap_independent_bias_interval(
    blocks_a: Sequence[Mapping[str, np.ndarray]],
    weights_a: Sequence[object],
    blocks_b: Sequence[Mapping[str, np.ndarray]],
    weights_b: Sequence[object],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not blocks_a or not blocks_b:
        raise G23Blocked(f"{field}:EMPTY")
    wa = [_finite(item, f"{field}.weight_a") for item in weights_a]
    wb = [_finite(item, f"{field}.weight_b") for item in weights_b]
    if len(wa) != len(blocks_a) or len(wb) != len(blocks_b):
        raise G23Blocked(f"{field}:WEIGHT_MISMATCH")
    rng_a = np.random.default_rng(2_304_250)
    rng_b = np.random.default_rng(2_304_251)
    reps = max(128, min(2048, 32 * max(len(blocks_a), len(blocks_b))))
    rows: list[np.ndarray] = []
    for _ in range(reps):
        ia = rng_a.integers(0, len(blocks_a), size=len(blocks_a))
        ib = rng_b.integers(0, len(blocks_b), size=len(blocks_b))
        ma = _moments_from_blocks([blocks_a[int(i)] for i in ia], [wa[int(i)] for i in ia], f"{field}.a")
        mb = _moments_from_blocks([blocks_b[int(i)] for i in ib], [wb[int(i)] for i in ib], f"{field}.b")
        rows.append(_flat(_u_from_moments(_merge_moments(ma, mb, field), f"{field}.u")))
    matrix = np.stack(rows, axis=0)
    return np.quantile(matrix, 0.025, axis=0), np.quantile(matrix, 0.975, axis=0)


_CHEBYSHEV_95 = math.sqrt(20.0)


def _bounded_square_mean_variance(moment: _BoundedMoments, field: str) -> dict[str, np.ndarray]:
    """Exact delete-one jackknife variance of ``mean(block)^2`` from powers."""

    if moment.count < 3 or not moment.include_higher or not moment.p3 or not moment.p4:
        raise G23Blocked(f"{field}:HIGHER_MOMENTS_REQUIRED")
    n = float(moment.count)
    weight = float(moment.first_weight or 0.0)
    if weight <= 0.0:
        raise G23Blocked(f"{field}:WEIGHT_REQUIRED")
    result: dict[str, np.ndarray] = {}
    for name in moment.g1:
        s1 = moment.g1[name] / weight
        s2, s3, s4 = moment.p2[name], moment.p3[name], moment.p4[name]
        center = np.square(s1 / n)
        denominator = np.square(n - 1.0)
        a = np.square(s1) / denominator
        b = -2.0 * s1 / denominator
        c = np.full_like(s1, 1.0 / denominator)
        sum_theta = n * a + b * s1 + c * s2
        sum_theta2 = (
            n * np.square(a)
            + np.square(b) * s2
            + np.square(c) * s4
            + 2.0 * a * b * s1
            + 2.0 * a * c * s2
            + 2.0 * b * c * s3
        )
        variance = (n - 1.0) / n * (
            sum_theta2 - 2.0 * center * sum_theta + n * np.square(center)
        )
        tolerance = 1e-12 * np.maximum(1.0, np.abs(sum_theta2))
        if np.any(variance < -tolerance):
            raise G23Blocked(f"{field}:NEGATIVE_JACKKNIFE_VARIANCE")
        result[name] = np.maximum(variance, 0.0)
    return result


def _bounded_precision_diagnostics(
    *,
    bias: Mapping[str, np.ndarray],
    cross: Mapping[str, np.ndarray],
    moments_a: _BoundedMoments,
    moments_b: _BoundedMoments,
    uncertainty: ReferenceUncertainty,
    layer_groups: Mapping[str, Sequence[str]],
    module_groups: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Conservative 95% endpoint envelopes from exact bounded jackknife data.

    Raw-shard rounds retain their fixed bootstrap implementation.  For the
    bounded representation, a distribution-free Chebyshev multiplier is
    applied to the exact online jackknife variance.  This is deliberately
    wider (and therefore harder for the precision Gate) than a normal 95%
    interval; thresholds and margins are unchanged.
    """

    bias_half = {
        name: _CHEBYSHEV_95 * np.sqrt(np.maximum(value, 0.0))
        for name, value in uncertainty.bias_variance.items()
    }
    cross_half = {
        name: _CHEBYSHEV_95 * np.sqrt(np.maximum(value, 0.0))
        for name, value in uncertainty.cross_variance.items()
    }
    a_variance = _bounded_square_mean_variance(moments_a, "bounded.a_square")
    b_variance = _bounded_square_mean_variance(moments_b, "bounded.b_square")
    a_half_map = {name: _CHEBYSHEV_95 * np.sqrt(value) for name, value in a_variance.items()}
    b_half_map = {name: _CHEBYSHEV_95 * np.sqrt(value) for name, value in b_variance.items()}
    mean_a, mean_b = moments_a.mean(), moments_b.mean()
    a_rank_map = {name: np.square(value) for name, value in mean_a.items()}
    b_rank_map = {name: np.square(value) for name, value in mean_b.items()}

    def relative_l1_bound(groups: Mapping[str, Sequence[str]]) -> float:
        center_total, center_mean = _aggregate(bias, groups)
        total_bound = 0.0
        mean_bound = 0.0
        for names in groups.values():
            group_bound = float(sum(np.sum(bias_half[name]) for name in names))
            group_count = float(sum(bias[name].size for name in names))
            total_bound += group_bound
            mean_bound += group_bound / max(group_count, 1.0)
        return max(
            total_bound / max(float(np.abs(center_total).sum()), 1e-300),
            mean_bound / max(float(np.abs(center_mean).sum()), 1e-300),
        )

    model_half = float(sum(np.sum(value) for value in bias_half.values()))
    layer_q95 = relative_l1_bound(layer_groups)
    module_q95 = relative_l1_bound(module_groups)
    flat_bias_half = _flat(bias_half)
    flat_cross_half = _flat(cross_half)
    flat_a_half = _flat(a_half_map)
    flat_b_half = _flat(b_half_map)
    flat_a = _flat(a_rank_map)
    flat_b = _flat(b_rank_map)
    return {
        "uncertainty_method": "bounded-exact-jackknife-chebyshev95-v1",
        "h_ref": max(model_half, layer_q95, module_q95),
        "model_half_width": model_half,
        "layer_l1_q95": layer_q95,
        "module_l1_q95": module_q95,
        "bias_low": _flat(bias) - flat_bias_half,
        "bias_high": _flat(bias) + flat_bias_half,
        "cross_low": _flat(cross) - flat_cross_half,
        "cross_high": _flat(cross) + flat_cross_half,
        "a_rank": flat_a,
        "b_rank": flat_b,
        "a_half": flat_a_half,
        "b_half": flat_b_half,
    }


def _sequence_scaling_bounded(
    evidence: _CellEvidence,
    moments_a: _BoundedMoments,
    moments_b: _BoundedMoments,
    block_size: int,
) -> bool:
    if evidence.bundle_state is None:
        raise G23Blocked("variance_scaling:bundle_missing")
    stored = _vector(evidence.bundle_state.get("sequence_variance"), "bundle.sequence_variance")
    try:
        expected = estimate_sequence_variance_bounded(
            moments_a.combine(moments_b), block_size=block_size
        )
    except ValueError as error:
        raise G23Blocked("variance_scaling:BOUNDED_RECOMPUTE_FAILED") from error
    _compatible(expected, stored, "variance_scaling")
    return all(np.allclose(expected[name], stored[name], rtol=1e-12, atol=1e-12) for name in expected)


def _sequence_scaling(
    evidence: _CellEvidence,
    blocks: Sequence[Mapping[str, np.ndarray]],
    block_size: int,
    weights: Sequence[object] | None = None,
    extra_blocks: Sequence[Mapping[str, np.ndarray]] | None = None,
    extra_weights: Sequence[object] | None = None,
) -> bool:
    if evidence.bundle_state is None:
        raise G23Blocked("variance_scaling:bundle_missing")
    stored = evidence.bundle_state.get("sequence_variance")
    if stored is None:
        raise G23Blocked("variance_scaling:SEQUENCE_VARIANCE_MISSING")
    sequence = _vector(stored, "bundle.sequence_variance")
    parsed_weights = [1.0] * len(blocks) if weights is None else [_finite(item, "variance_scaling.weight") for item in weights]
    extra = () if extra_blocks is None else extra_blocks
    parsed_extra = [] if extra_weights is None else [_finite(item, "variance_scaling.extra_weight") for item in extra_weights]
    if len(parsed_weights) != len(blocks) or (extra_blocks is not None and len(parsed_extra) != len(extra)) or not blocks:
        raise G23Blocked("variance_scaling:WEIGHT_MISMATCH")
    moments = _moments_from_blocks(blocks, parsed_weights, "variance_scaling.blocks")
    if extra:
        moments = _merge_moments(
            moments,
            _moments_from_blocks(extra, parsed_extra, "variance_scaling.extra_blocks"),
            "variance_scaling.combined",
        )
    mean = _weighted_mean_from_moments(moments, "variance_scaling.mean")
    expected = {name: np.zeros_like(value) for name, value in mean.items()}
    factor = float(block_size) / max(_finite(moments.get("n1"), "variance_scaling.n1"), 1e-300)
    for block, weight in zip(blocks, parsed_weights):
        for name in expected:
            expected[name] += float(weight) * np.square(block[name] - mean[name]) * factor
    for block, weight in zip(extra, parsed_extra):
        for name in expected:
            expected[name] += float(weight) * np.square(block[name] - mean[name]) * factor
    _compatible(expected, sequence, "variance_scaling")
    return all(np.allclose(expected[name], sequence[name], rtol=1e-9, atol=1e-12) for name in expected)


def _evaluate_cell(evidence: _CellEvidence, *, evaluator_commit: str) -> dict[str, object]:
    cell: dict[str, object] = {"cell_id": evidence.source.cell_id, "status": "BLOCKED", "identities": dict(evidence.identities), "reasons": list(evidence.reasons)}
    if evidence.reasons:
        return cell
    try:
        counts, previous, plan = _sizing_vectors(evidence)
        bounded = bool(evidence.final_state and evidence.final_state.get("bounded_storage"))
        moments_a: _BoundedMoments | None = None
        moments_b: _BoundedMoments | None = None
        if bounded:
            bias, cross, ranking, moments_a, moments_b, uncertainty = _final_vectors_bounded(evidence)
            blocks_a: Sequence[Mapping[str, np.ndarray]] = ()
            blocks_b: Sequence[Mapping[str, np.ndarray]] = ()
        else:
            bias, cross, ranking, blocks_a, blocks_b, uncertainty = _final_vectors(evidence)
        previous_bias, latest_bias = previous
        flat_prev, flat_latest = _flat(previous_bias), _flat(latest_bias)
        delta_sci, min_delta_by_endpoint, corrected_delta_sci = _delta_sci(
            evidence,
            [int(item) for item in plan["candidate_sample_counts"]],
            evaluator_commit=evaluator_commit,
        )
        epsilon_num = _numerical_error(evidence, bias)
        block_size = int(plan.get("block_size", 0))
        if block_size <= 0:
            raise G23Blocked("block_size:INVALID")
        layer, module = _group_map(latest_bias, evidence)
        prev_layer, prev_module = _group_map(previous_bias, evidence)
        layer_prev_total, layer_prev_mean = _aggregate(previous_bias, prev_layer)
        layer_latest_total, layer_latest_mean = _aggregate(latest_bias, layer)
        module_prev_total, module_prev_mean = _aggregate(previous_bias, prev_module)
        module_latest_total, module_latest_mean = _aggregate(latest_bias, module)
        if layer_prev_total.size != layer_latest_total.size or module_prev_total.size != module_latest_total.size:
            raise G23Blocked("layer_module:GROUPING_DRIFT")
        layer_delta = max(float(np.max(np.abs(layer_latest_total - layer_prev_total) / np.maximum(np.abs(layer_latest_total), 1e-300))), float(np.max(np.abs(layer_latest_mean - layer_prev_mean) / np.maximum(np.abs(layer_latest_mean), 1e-300))))
        module_delta = max(float(np.max(np.abs(module_latest_total - module_prev_total) / np.maximum(np.abs(module_latest_total), 1e-300))), float(np.max(np.abs(module_latest_mean - module_prev_mean) / np.maximum(np.abs(module_latest_mean), 1e-300))))
        layer_spearman = _spearman(layer_prev_total, layer_latest_total)
        module_spearman = _spearman(module_prev_total, module_latest_total)
        numeric_bundle = evidence.bundle_state.get("numerical_diagnostics") if evidence.bundle_state is not None else None
        if not isinstance(numeric_bundle, Mapping):
            raise G23Blocked("epsilon_num:NUMERICAL_BUNDLE_MISSING")
        high_numeric = _vector(numeric_bundle.get("high_precision"), "epsilon_num.high_precision")
        accumulated_numeric = _vector(numeric_bundle.get("accumulated"), "epsilon_num.accumulated")
        _compatible(high_numeric, accumulated_numeric, "epsilon_num")
        error_vector = (
            _vector(numeric_bundle.get("error_bound"), "epsilon_num.error_bound")
            if bounded
            else {name: np.abs(high_numeric[name] - accumulated_numeric[name]) for name in high_numeric}
        )
        epsilon_by_endpoint = {
            "model_total": float(sum(np.sum(value) for value in error_vector.values())),
            "layer": float(sum(np.sum(error_vector[name]) for names in layer.values() for name in names)),
            "module": float(sum(np.sum(error_vector[name]) for names in module.values() for name in names)),
        }
        bias_variance = uncertainty.bias_variance
        cross_variance = uncertainty.cross_variance
        ranking_variance = uncertainty.ranking_variance
        _compatible(bias_variance, cross_variance, "variance")
        _compatible(bias_variance, ranking_variance, "variance")
        if evidence.final_state is None:
            raise G23Blocked("final:STATE_MISSING")
        if bounded:
            if moments_a is None or moments_b is None:
                raise G23Blocked("final:BOUNDED_MOMENTS_MISSING")
            bounded_precision = _bounded_precision_diagnostics(
                bias=bias,
                cross=cross,
                moments_a=moments_a,
                moments_b=moments_b,
                uncertainty=uncertainty,
                layer_groups=layer,
                module_groups=module,
            )
            h_ref = float(bounded_precision["h_ref"])
            model_half_width = float(bounded_precision["model_half_width"])
            layer_l1_q95 = float(bounded_precision["layer_l1_q95"])
            module_l1_q95 = float(bounded_precision["module_l1_q95"])
            bias_low = np.asarray(bounded_precision["bias_low"])
            bias_high = np.asarray(bounded_precision["bias_high"])
            cross_low = np.asarray(bounded_precision["cross_low"])
            cross_high = np.asarray(bounded_precision["cross_high"])
            a_rank = np.asarray(bounded_precision["a_rank"])
            b_rank = np.asarray(bounded_precision["b_rank"])
            a_half = np.asarray(bounded_precision["a_half"])
            b_half = np.asarray(bounded_precision["b_half"])
            weights_a: list[float] = []
            weights_b: list[float] = []
        else:
            if evidence.final_root is None:
                raise G23Blocked("final:SHARD_ROOT_MISSING")
            _, weights_a, _ = _load_shard_records(
                evidence.final_root, evidence.final_state.get("shard_refs_a"), "final.shard_refs_a"
            )
            _, weights_b, _ = _load_shard_records(
                evidence.final_root, evidence.final_state.get("shard_refs_b"), "final.shard_refs_b"
            )
            h_ref, model_half_width, layer_l1_q95, module_l1_q95 = _bootstrap_u_diagnostics(
                blocks_a, blocks_b, weights_a, weights_b, latest_bias, layer, module
            )
            bias_low, bias_high = _bootstrap_independent_bias_interval(
                blocks_a, weights_a, blocks_b, weights_b, "bias.bootstrap"
            )
            a_mean = _weighted_mean_from_moments(_moments_from_blocks(blocks_a, weights_a, "a_mean"), "a_mean")
            b_mean = _weighted_mean_from_moments(_moments_from_blocks(blocks_b, weights_b, "b_mean"), "b_mean")
            a_rank, b_rank = np.square(_flat(a_mean)), np.square(_flat(b_mean))
            a_low, a_high, _ = _bootstrap_interval(blocks_a, weights_a, "a.square.bootstrap", square=True)
            b_low, b_high, _ = _bootstrap_interval(blocks_b, weights_b, "b.square.bootstrap", square=True)
            cross_low, cross_high = _bootstrap_independent_cross_interval(
                blocks_a, weights_a, blocks_b, weights_b, "cross.bootstrap"
            )
            a_half = (a_high - a_low) / 2.0
            b_half = (b_high - b_low) / 2.0
        signal_floor = evidence.convergence.payload.get("numerical_floor") if evidence.convergence is not None else None
        if signal_floor is None:
            raise G23Blocked("signal_eligible:numerical_floor_missing")
        floor = _finite(signal_floor, "numerical_floor")
        reference_half_width = (bias_high - bias_low) / 2.0
        signal_mask = np.abs(flat_latest) > np.maximum(5.0 * reference_half_width, 10.0 * floor)
        if not np.any(signal_mask):
            raise G23Blocked("signal_eligible:EMPTY")
        metrics: dict[str, object] = {
            "normalized_l1": _l1(previous_bias, latest_bias),
            "pearson": _pearson(flat_prev, flat_latest),
            "signal_eligible_spearman": _spearman(flat_prev[signal_mask], flat_latest[signal_mask]),
            "layer_module_spearman": min(layer_spearman, module_spearman),
            "topk_overlap_0_001": _top_overlap(flat_prev, flat_latest, 0.001),
            "topk_overlap_0_01": _top_overlap(flat_prev, flat_latest, 0.01),
            "topk_overlap_0_05": _top_overlap(flat_prev, flat_latest, 0.05),
            "layer_module_delta": max(layer_delta, module_delta),
            "h_ref": h_ref,
            "layer_l1_bootstrap_q95": layer_l1_q95,
            "module_l1_bootstrap_q95": module_l1_q95,
            "min_delta_sci": min(min_delta_by_endpoint.values()),
            "min_delta_sci_by_endpoint": min_delta_by_endpoint,
            "epsilon_num": epsilon_num,
            "epsilon_num_by_endpoint": epsilon_by_endpoint,
            "h_ref_model_total": model_half_width,
            "h_ref_layer": layer_l1_q95,
            "h_ref_module": module_l1_q95,
            "a_b_spearman": _spearman(a_rank, b_rank),
            "a_b_top_overlap_0_001": _top_overlap(a_rank, b_rank, 0.001),
            "a_b_top_overlap_0_01": _top_overlap(a_rank, b_rank, 0.01),
            "a_b_top_overlap_0_05": _top_overlap(a_rank, b_rank, 0.05),
            "bias_cross_interval_max_z": float(
                np.max(
                    np.abs(_flat(cross) - _flat(bias))
                    / np.maximum((bias_high - bias_low + cross_high - cross_low) / 2.0, 1e-300)
                )
            ),
            "a_square_interval_halfwidth_max": float(np.max(a_half)),
            "b_square_interval_halfwidth_max": float(np.max(b_half)),
            "ranking_bias_direction_sum": float(np.sum(_flat(ranking) - _flat(bias))),
            "jackknife_bias_variance_trace": uncertainty.trace_bias_variance,
            "jackknife_cross_variance_trace": float(sum(np.sum(value) for value in cross_variance.values())),
            "jackknife_ranking_variance_trace": float(sum(np.sum(value) for value in ranking_variance.values())),
            "sizing_min_delta_sci": min(min_delta_by_endpoint.values()),
            "sizing_node_previous": int(counts[0]),
            "sizing_node_latest": int(counts[1]),
            "uncertainty_method": (
                bounded_precision["uncertainty_method"]
                if bounded
                else "raw-shard-fixed-bootstrap-v1"
            ),
        }
        checks: dict[str, bool] = {
            "normalized_l1": metrics["normalized_l1"] <= THRESHOLDS["normalized_l1"],
            "pearson": metrics["pearson"] >= THRESHOLDS["pearson"],
            "signal_eligible_spearman": metrics["signal_eligible_spearman"] >= THRESHOLDS["signal_eligible_spearman"],
            "layer_module_spearman": metrics["layer_module_spearman"] >= THRESHOLDS["layer_module_spearman"],
            "topk_overlap_0_001": metrics["topk_overlap_0_001"] >= THRESHOLDS["topk_overlap"],
            "topk_overlap_0_01": metrics["topk_overlap_0_01"] >= THRESHOLDS["topk_overlap"],
            "topk_overlap_0_05": metrics["topk_overlap_0_05"] >= THRESHOLDS["topk_overlap"],
            "layer_module_delta": metrics["layer_module_delta"] <= THRESHOLDS["layer_module_delta"],
            "layer_l1_bootstrap_q95": metrics["layer_l1_bootstrap_q95"] <= THRESHOLDS["layer_module_l1_q95"],
            "module_l1_bootstrap_q95": metrics["module_l1_bootstrap_q95"] <= THRESHOLDS["layer_module_l1_q95"],
            "h_ref_model_total": metrics["h_ref_model_total"] <= min_delta_by_endpoint["model_total"] / THRESHOLDS["h_ref_divisor"],
            "h_ref_layer": metrics["h_ref_layer"] <= min_delta_by_endpoint["layer"] / THRESHOLDS["h_ref_divisor"],
            "h_ref_module": metrics["h_ref_module"] <= min_delta_by_endpoint["module"] / THRESHOLDS["h_ref_divisor"],
            "h_ref": metrics["h_ref"] <= min(min_delta_by_endpoint.values()) / THRESHOLDS["h_ref_divisor"],
            "epsilon_num_model_total": metrics["epsilon_num_by_endpoint"]["model_total"] <= min_delta_by_endpoint["model_total"] / THRESHOLDS["epsilon_num_divisor"],
            "epsilon_num_layer": metrics["epsilon_num_by_endpoint"]["layer"] <= min_delta_by_endpoint["layer"] / THRESHOLDS["epsilon_num_divisor"],
            "epsilon_num_module": metrics["epsilon_num_by_endpoint"]["module"] <= min_delta_by_endpoint["module"] / THRESHOLDS["epsilon_num_divisor"],
            "epsilon_num": metrics["epsilon_num"] <= min(min_delta_by_endpoint.values()) / THRESHOLDS["epsilon_num_divisor"],
            "a_b_spearman": metrics["a_b_spearman"] >= THRESHOLDS["pearson"],
            "a_b_top_overlap_0_001": metrics["a_b_top_overlap_0_001"] >= THRESHOLDS["topk_overlap"],
            "a_b_top_overlap_0_01": metrics["a_b_top_overlap_0_01"] >= THRESHOLDS["topk_overlap"],
            "a_b_top_overlap_0_05": metrics["a_b_top_overlap_0_05"] >= THRESHOLDS["topk_overlap"],
            "bias_cross_interval_covered": bool(
                np.all(np.abs(_flat(cross) - _flat(bias)) <= (bias_high - bias_low + cross_high - cross_low) / 2.0)
            ),
            "ranking_bias_direction": metrics["ranking_bias_direction_sum"] >= 0.0,
            "jackknife_uncertainty_verified": all(
                np.all(np.isfinite(value)) and np.all(value >= 0.0)
                for vector in (bias_variance, cross_variance, ranking_variance)
                for value in vector.values()
            ),
            "variance_scaling_verified": (
                _sequence_scaling_bounded(evidence, moments_a, moments_b, block_size)
                if bounded and moments_a is not None and moments_b is not None
                else _sequence_scaling(
                    evidence, blocks_a, block_size, weights_a,
                    extra_blocks=blocks_b, extra_weights=weights_b,
                )
            ),
            "state_replay_verified": _state_replay_verified(evidence),
        }
        one_shot = evidence.convergence.payload.get("one_shot_result") if evidence.convergence else None
        plan_one_shot = evidence.convergence.payload.get("one_shot_plan") if evidence.convergence else None
        checks["one_shot_complete"] = (
            isinstance(one_shot, Mapping)
            and one_shot.get("status") == "COMPLETE"
            and one_shot.get("one_shot") is True
            and isinstance(plan_one_shot, Mapping)
            and plan_one_shot.get("one_shot") is True
            and plan_one_shot.get("sizing_stream") == "reference_sizing"
            and plan_one_shot.get("stream_a") == "reference_A"
            and plan_one_shot.get("stream_b") == "reference_B"
            and int(one_shot.get("processed_sample_count_per_stream", 0)) == int(plan_one_shot.get("sample_count_per_stream", -1))
            and (
                int(evidence.final_state.get("processed_block_pairs", 0)) * block_size
                if bounded and evidence.final_state is not None
                else len(blocks_a) * block_size
            ) == int(one_shot.get("processed_sample_count_per_stream", -1))
        )
        checks["a_b_interval_covered"] = bool(np.all(np.abs(a_rank - b_rank) <= a_half + b_half))
        checks["a_b_top_overlap"] = bool(
            checks["a_b_top_overlap_0_001"]
            and checks["a_b_top_overlap_0_01"]
            and checks["a_b_top_overlap_0_05"]
        )
        # Keep the machine-facing booleans inside the calculated metrics map as
        # well as in ``checks``.  The formal launcher can therefore bind and
        # consume this artifact without accepting a second caller-authored
        # qualification object.
        metrics.update({
            "task_result_hash": evidence.identities["result_hash"],
            "bundle_manifest_sha256": evidence.identities["bundle_manifest_hash"],
            "a_b_interval_covered": checks["a_b_interval_covered"],
            "bias_cross_interval_covered": checks["bias_cross_interval_covered"],
            "ranking_bias_direction": checks["ranking_bias_direction"],
            "variance_scaling_verified": checks["variance_scaling_verified"],
            "state_replay_verified": checks["state_replay_verified"],
            "one_shot_complete": checks["one_shot_complete"],
        })
        cell["_corrected_delta_sci"] = dict(corrected_delta_sci)
        reasons = [name + ":THRESHOLD_FAILED" for name, passed in checks.items() if not passed]
        cell.update({
            "status": "PASS" if not reasons else "BLOCKED",
            "metrics": metrics,
            "checks": checks,
            "reasons": reasons,
            "endpoints": {
                "model_total": {"metrics": metrics, "checks": checks},
                "layer": {
                    "metrics": {
                        "spearman": layer_spearman,
                        "total_delta": layer_delta,
                        "l1_bootstrap_q95": layer_l1_q95,
                    },
                    "checks": {
                        "spearman": checks["layer_module_spearman"],
                        "total_per_param_delta": checks["layer_module_delta"],
                        "l1_bootstrap_q95": checks["layer_l1_bootstrap_q95"],
                    },
                },
                "module": {
                    "metrics": {
                        "spearman": module_spearman,
                        "total_delta": module_delta,
                        "l1_bootstrap_q95": module_l1_q95,
                    },
                    "checks": {
                        "spearman": checks["layer_module_spearman"],
                        "total_per_param_delta": checks["layer_module_delta"],
                        "l1_bootstrap_q95": checks["module_l1_bootstrap_q95"],
                    },
                },
            },
        })
    except G23Blocked as error:
        cell["reasons"] = list(cell.get("reasons", [])) + [str(error)]
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as error:
        cell["reasons"] = list(cell.get("reasons", [])) + [f"RAW_DIAGNOSTIC_INVALID:{type(error).__name__}"]
    return cell


def _publish_corrected_delta_sidecars(
    root: Path,
    output_root: Path | None,
    cells: Sequence[dict[str, object]],
) -> None:
    """Atomically publish evaluator-owned corrected delta artifacts.

    The sidecar is content addressed by its own canonical payload hash.  The
    producer's original artifact is never overwritten; each PASS cell binds
    the new workspace-relative reference and hash into its identities/metrics.
    """

    pending = [
        cell
        for cell in cells
        if cell.get("status") == "PASS"
        and isinstance(cell.get("_corrected_delta_sci"), Mapping)
    ]
    if not pending:
        return
    if output_root is None:
        for cell in pending:
            cell.pop("_corrected_delta_sci", None)
            cell["status"] = "BLOCKED"
            cell["formal_eligible"] = False
            reasons = cell.setdefault("reasons", [])
            if isinstance(reasons, list):
                reasons.append("delta_sci:CORRECTED_SIDECAR_OUTPUT_REQUIRED")
        return
    try:
        output_root.relative_to(root)
    except ValueError as error:
        raise G23Blocked("delta_sci:CORRECTED_SIDECAR_OUTPUT_OUTSIDE_WORKSPACE") from error
    sidecar_root = output_root / "g2.3-corrected-delta-sci"
    _reject_absolute_symlink_chain(sidecar_root, "corrected_delta_sci_root")
    sidecar_root.mkdir(parents=True, exist_ok=True)
    for cell in pending:
        raw = cell.pop("_corrected_delta_sci")
        if not isinstance(raw, Mapping):
            continue
        sidecar = dict(raw)
        if set(sidecar) != CORRECTED_DELTA_SIDECAR_FIELDS:
            raise RuntimeError("G23_CORRECTED_DELTA_SCHEMA_FIELDS_MISMATCH")
        if sidecar.get("schema_version") != CORRECTED_DELTA_SCHEMA_VERSION:
            raise RuntimeError("G23_CORRECTED_DELTA_SCHEMA_VERSION_MISMATCH")
        if sidecar.get("delta_sci_batch_sizes") != list(CORRECTED_DELTA_BATCH_SIZES):
            raise RuntimeError("G23_CORRECTED_DELTA_BATCH_DOMAIN_MISMATCH")
        mode = sidecar.get("source_producer_table_mode")
        expected_reason = (
            "producer_delta_sci_keyed_by_sizing_nodes; S2.6_requires_candidate_batch_sizes"
            if mode == "sizing_nodes_legacy"
            else "producer_delta_sci_candidate_batch_sizes_verified_by_evaluator"
            if mode == "candidate_batch_sizes"
            else None
        )
        if expected_reason is None or sidecar.get("correction_reason") != expected_reason:
            raise RuntimeError("G23_CORRECTED_DELTA_CORRECTION_REASON_MISMATCH")
        sidecar_hash = _canonical_payload_hash(sidecar, "corrected_delta_sci")
        target = sidecar_root / f"{sidecar_hash}.json"
        _reject_absolute_symlink_chain(target, "corrected_delta_sci_target")
        if target.exists():
            try:
                existing = load_canonical_json(target)
            except (OSError, TypeError, ValueError) as error:
                raise RuntimeError("G23_CORRECTED_DELTA_CONTENT_ADDRESS_COLLISION") from error
            if existing != sidecar:
                raise RuntimeError("G23_CORRECTED_DELTA_CONTENT_ADDRESS_COLLISION")
        else:
            write_canonical_json(target, sidecar)
        try:
            reference = target.relative_to(root).as_posix()
        except ValueError as error:
            raise G23Blocked("delta_sci:CORRECTED_SIDECAR_OUTSIDE_WORKSPACE") from error
        identities = cell.setdefault("identities", {})
        metrics = cell.get("metrics")
        if not isinstance(identities, dict) or not isinstance(metrics, dict):
            cell["status"] = "BLOCKED"
            cell["formal_eligible"] = False
            reasons = cell.setdefault("reasons", [])
            if isinstance(reasons, list):
                reasons.append("delta_sci:CORRECTED_SIDECAR_BINDING_TARGET_INVALID")
            continue
        identities["corrected_delta_sci_hash"] = sidecar_hash
        identities["corrected_delta_sci_ref"] = reference
        metrics["corrected_delta_sci_hash"] = sidecar_hash
        metrics["corrected_delta_sci_ref"] = reference
        metrics["corrected_delta_sci_batch_sizes"] = list(CORRECTED_DELTA_BATCH_SIZES)
        metrics["delta_sci_source"] = "g23_output_derived_corrected_sidecar"


def _attempt_payload(cells: Sequence[Mapping[str, object]], calculator: Mapping[str, object], *, expected_cell_ids: Sequence[str]) -> dict[str, object]:
    complete = len(cells) == REQUIRED_CELL_COUNT and tuple(str(cell.get("cell_id")) for cell in cells) == tuple(expected_cell_ids) and all(cell.get("status") == "PASS" for cell in cells)
    status = "PASS" if complete else ("BLOCKED" if cells else "NOT_RUN")
    reasons = [reason for cell in cells for reason in cell.get("reasons", []) if isinstance(reason, str)]
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": status,
        "formal_eligible": complete,
        "required_cell_count": REQUIRED_CELL_COUNT,
        "complete_cell_count": sum(cell.get("status") == "PASS" for cell in cells),
        "expected_cell_ids": list(expected_cell_ids),
        "cells": list(cells),
        "calculator": dict(calculator),
        "thresholds": dict(THRESHOLDS),
        "reasons": reasons,
    }


def evaluate_formal_g23(
    workspace_root: str | Path,
    cells: Sequence[CellInput | Mapping[str, object] | str],
    *,
    expected_cell_ids: Sequence[str] | None = None,
    output_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate exactly six formal Stage2.04 outputs and publish one attempt.

    ``cells`` contains only references.  If fewer than six references are
    supplied the result is ``BLOCKED``/``NOT_RUN`` and no partial metric can
    qualify the study.
    """

    root = _reject_absolute_symlink_chain(Path(workspace_root), "workspace_root")
    module_path = Path(__file__).resolve()
    evaluator_repo = module_path.parents[3]
    evaluator_commit, evaluator_source_sha256 = _validate_evaluator_provenance(
        evaluator_repo,
        module_path=module_path,
    )
    producer_repo = None if repo_root is None else Path(repo_root).resolve()
    normalized_items: list[CellInput] = []
    for index, item in enumerate(cells):
        if isinstance(item, CellInput):
            normalized_items.append(item)
        elif isinstance(item, str):
            normalized_items.append(CellInput(f"cell-{index}", item))
        else:
            normalized_items.append(CellInput.from_mapping(item))
    normalized = tuple(normalized_items)
    # The six-cell set is a S2.3 contract, not a caller-controlled argument.
    # ``expected_cell_ids`` remains accepted for source compatibility only;
    # its values never influence the formal decision.
    expected = EXPECTED_CELL_IDS
    prepared: list[_CellEvidence] = []
    structure_reason: str | None = None
    if len(normalized) == REQUIRED_CELL_COUNT and len(set(item.cell_id for item in normalized)) == REQUIRED_CELL_COUNT:
        prepared = [_prepare_cell(root, item, repo_root=producer_repo) for item in normalized]
    elif normalized:
        structure_reason = "DUPLICATE_OR_INVALID_CELL_IDENTITIES"
    if prepared:
        derived_ids = [item.identities.get("cell_id", "") for item in prepared]
        if tuple(sorted(derived_ids, key=lambda value: EXPECTED_CELL_IDS.index(value) if value in EXPECTED_CELL_IDS else 99)) != EXPECTED_CELL_IDS:
            structure_reason = "SIX_CELL_MANIFEST_DERIVATION_MISMATCH"
        try:
            unique_identity_values(
                [item.identities for item in prepared],
                (
                    "result_hash",
                    "config_hash",
                    "checkpoint_hash",
                    "one_shot_result_hash",
                    "stream_a_draw_hash",
                    "stream_b_draw_hash",
                    "bundle_manifest_hash",
                    "sizing_plan_hash",
                    "reference_id",
                    "one_shot_plan_hash",
                ),
                field="six_cell.identities",
            )
        except ValueError as error:
            structure_reason = str(error)
        manifest_hashes = {item.identities.get("six_cell_manifest_hash", "") for item in prepared}
        if len(manifest_hashes) != 1 or "" in manifest_hashes:
            structure_reason = "SIX_CELL_MANIFEST_NOT_COMMON"
    cell_rows = [
        _evaluate_cell(item, evaluator_commit=evaluator_commit)
        for item in prepared
    ]
    if output_root is not None:
        out = _reject_absolute_symlink_chain(Path(output_root), "output_root")
    else:
        out = None
    _publish_corrected_delta_sidecars(root, out, cell_rows)
    producer_values: set[str] = set()
    for item in prepared:
        if "producer_commit" in item.identities:
            producer_values.add(item.identities["producer_commit"])
    if len(producer_values) == 1:
        producer = next(iter(producer_values))
    else:
        producer = ""
    calculator = {
        "producer_commit": producer,
        "evaluator_commit": evaluator_commit,
        "source_sha256": evaluator_source_sha256,
        "source_schema": SCHEMA_VERSION,
    }
    payload = _attempt_payload(cell_rows, calculator, expected_cell_ids=expected)
    if len(normalized) != REQUIRED_CELL_COUNT:
        payload["status"] = "NOT_RUN" if not normalized else "BLOCKED"
        payload["formal_eligible"] = False
        payload["reasons"] = ["REQUIRES_EXACTLY_SIX_FORMAL_TASK_RESULTS"]
    elif structure_reason is not None:
        payload["status"] = "BLOCKED"
        payload["formal_eligible"] = False
        payload["reasons"] = list(payload.get("reasons", [])) + [structure_reason]
    if producer == "":
        payload["formal_eligible"] = False
        if payload["status"] == "PASS":
            payload["status"] = "BLOCKED"
        payload["reasons"] = list(payload.get("reasons", [])) + ["PRODUCER_COMMIT_NOT_COMMON_OR_MISSING"]
    artifact_hash = canonical_json_hash(payload)
    payload["artifact_hash"] = artifact_hash
    if output_root is not None:
        assert out is not None
        attempt_dir = out / "g2.3-attempts" / artifact_hash
        _reject_absolute_symlink_chain(attempt_dir, "attempt_dir")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _reject_absolute_symlink_chain(attempt_dir, "attempt_dir")
        target = attempt_dir / "evaluation.json"
        _reject_absolute_symlink_chain(target, "attempt_target")
        if target.exists():
            try:
                existing = load_canonical_json(target)
            except (OSError, ValueError, TypeError) as error:
                raise RuntimeError("G23_ATTEMPT_CONTENT_ADDRESS_COLLISION") from error
            if existing != payload:
                raise RuntimeError("G23_ATTEMPT_CONTENT_ADDRESS_COLLISION")
        if not target.exists():
            write_canonical_json(target, payload)
        index = out / "g2.3-attempts.jsonl"
        _append_attempt_index(index, str(payload["artifact_hash"]))
    return payload


def evaluate_g23(workspace_root: str | Path, cells: Sequence[CellInput | Mapping[str, object] | str], **kwargs: object) -> dict[str, object]:
    """Compatibility alias for callers using the shorter Gate name."""

    return evaluate_formal_g23(workspace_root, cells, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class G23ReferenceEvaluator:
    """Small state-free façade convenient for task/CLI integrations."""

    workspace_root: str | Path
    output_root: str | Path | None = None
    expected_cell_ids: tuple[str, ...] | None = None
    repo_root: str | Path | None = None

    def evaluate(self, cells: Sequence[CellInput | Mapping[str, object] | str]) -> dict[str, object]:
        return evaluate_formal_g23(
            self.workspace_root,
            cells,
            expected_cell_ids=self.expected_cell_ids,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )


__all__ = [
    "CellInput",
    "CORRECTED_DELTA_BATCH_SIZES",
    "CORRECTED_DELTA_SIDECAR_FIELDS",
    "CORRECTED_DELTA_SCHEMA_VERSION",
    "G23Blocked",
    "G23ReferenceEvaluator",
    "GATE_ID",
    "SCHEMA_VERSION",
    "THRESHOLDS",
    "evaluate_formal_g23",
    "evaluate_g23",
]
