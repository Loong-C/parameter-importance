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
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

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

from .stage2_formal import _ReferenceSnapshotStore, _vector_digest


SCHEMA_VERSION = "stage2-g23-reference-evaluation-v1"
GATE_ID = "stage2.G2.3"
REQUIRED_CELL_COUNT = 6
THRESHOLDS: Mapping[str, float] = {
    "normalized_l1": 0.02,
    "pearson": 0.995,
    "signal_eligible_spearman": 0.995,
    "layer_module_spearman": 0.995,
    "topk_overlap": 0.98,
    "layer_module_delta": 0.01,
    "h_ref_divisor": 4.0,
    "epsilon_num_divisor": 10.0,
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


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
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise G23Blocked(f"{field}:PATH_ESCAPE")
    return parsed.as_posix()


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


def _find(mapping: object, names: Iterable[str]) -> object | None:
    wanted = set(names)
    if isinstance(mapping, Mapping):
        for key, value in mapping.items():
            if str(key) in wanted:
                return value
        for value in mapping.values():
            found = _find(value, wanted)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        for value in mapping:
            found = _find(value, wanted)
            if found is not None:
                return found
    return None


def _find_key(mapping: object, names: Iterable[str]) -> tuple[str, object] | None:
    wanted = set(names)
    if isinstance(mapping, Mapping):
        for key, value in mapping.items():
            if str(key) in wanted:
                return str(key), value
        for value in mapping.values():
            found = _find_key(value, wanted)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        for value in mapping:
            found = _find_key(value, wanted)
            if found is not None:
                return found
    return None


def _digest_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if not self.cell_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", self.cell_id):
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
    result: TaskRunResult | None = None
    result_payload: Mapping[str, object] | None = None
    reference: LoadedTaskArtifact | None = None
    convergence: LoadedTaskArtifact | None = None
    gate: LoadedTaskArtifact | None = None
    bundle_state: Mapping[str, object] | None = None
    bundle_manifest_hash: str | None = None
    sizing_states: list[Mapping[str, object]] | None = None
    final_state: Mapping[str, object] | None = None
    identities: dict[str, str] = None  # type: ignore[assignment]
    reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.identities = {}
        self.reasons = []


def _resolve(root: Path, reference: str) -> Path:
    logical = Path(_path(reference, "reference"))
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


def _task_result(root: Path, source: CellInput) -> tuple[TaskRunResult, Mapping[str, object]]:
    value = _load_json(root, source.task_result_ref, "task_result")
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
    try:
        return load_committed_task_artifact(root, ref, require_formal=True)
    except (OSError, ValueError, TypeError) as error:
        raise G23Blocked(f"artifact:{kind}:INVALID") from error


def _load_resume_commits(root: Path, resume_root: Path, schema: str) -> list[Mapping[str, object]]:
    commits_dir = resume_root / "commits"
    if not commits_dir.is_dir():
        raise G23Blocked(f"resume:{schema}:COMMITS_MISSING")
    paths = sorted(commits_dir.glob("*.json"))
    if not paths:
        raise G23Blocked(f"resume:{schema}:COMMITS_EMPTY")
    states: list[Mapping[str, object]] = []
    for index, commit_path in enumerate(paths, start=1):
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
        try:
            state, bundle = load_tensor_bundle(resume_root / relative)
        except (OSError, ValueError, TypeError) as error:
            raise G23Blocked(f"resume:{schema}:OBJECT_UNREADABLE") from error
        if bundle.manifest_sha256 != _sha(commit.get("object_manifest_hash"), "resume.object_manifest_hash"):
            raise G23Blocked(f"resume:{schema}:MANIFEST_HASH")
        if not isinstance(state, Mapping) or state.get("schema_version") != schema:
            raise G23Blocked(f"resume:{schema}:STATE_SCHEMA")
        try:
            state_digest = _ReferenceSnapshotStore._state_digest(state)
        except (KeyError, TypeError, ValueError) as error:
            raise G23Blocked(f"resume:{schema}:STATE_DIGEST_INVALID") from error
        if state_digest != _sha(commit.get("state_digest"), "resume.state_digest"):
            raise G23Blocked(f"resume:{schema}:STATE_DIGEST_MISMATCH")
        states.append(state)
    return states


def _validate_state_identity(state: Mapping[str, object], identities: Mapping[str, str], field: str) -> None:
    for state_key, identity_key in (
        ("registry_hash", "registry_hash"),
        ("provider_state_digest", "provider_state_digest"),
        ("sizing_result_hash", "sizing_result_hash"),
        ("stream_a_draw_hash", "stream_a_draw_hash"),
        ("stream_b_draw_hash", "stream_b_draw_hash"),
    ):
        if state_key in state and identity_key in identities and str(state[state_key]) != identities[identity_key]:
            raise G23Blocked(f"{field}:{state_key}:IDENTITY_DRIFT")


def _resume_roots(root: Path, result_ref: str) -> tuple[Path, Path]:
    commit = Path(_path(result_ref, "reference_result_ref"))
    # .../<output_dir>/commits/reference_result.json -> .../<output_dir>
    output_dir = (root / commit.parent.parent).resolve()
    try:
        output_dir.relative_to(root.resolve())
    except ValueError as error:
        raise G23Blocked("reference_result_ref:PATH_ESCAPE") from error
    return output_dir / "resume" / "reference-sizing", output_dir / "resume" / "reference-final"


def _extract_checkpoint_hash(reference: Mapping[str, object], convergence: Mapping[str, object], result_payload: Mapping[str, object]) -> str:
    found = _find_key({"reference": reference, "convergence": convergence, "result": result_payload}, {
        "checkpoint_hash", "checkpoint_manifest_hash", "checkpoint_sha256", "checkpoint_revision_hash",
    })
    if found is None:
        raise G23Blocked("checkpoint_hash:RAW_IDENTITY_MISSING")
    return _sha(found[1], "checkpoint_hash")


def _extract_producer_commit(reference: Mapping[str, object], convergence: Mapping[str, object], result_payload: Mapping[str, object]) -> str:
    found = _find_key({"reference": reference, "convergence": convergence, "result": result_payload}, {
        "producer_commit", "producer_git_commit", "execution_commit",
    })
    if found is None:
        raise G23Blocked("producer_commit:RAW_IDENTITY_MISSING")
    return _commit(found[1], "producer_commit")


def _prepare_cell(root: Path, source: CellInput) -> _CellEvidence:
    evidence = _CellEvidence(source)
    try:
        result, result_payload = _task_result(root, source)
        evidence.result, evidence.result_payload = result, result_payload
        evidence.identities["result_hash"] = _sha(result.result_hash, "result_hash")
        evidence.identities["config_hash"] = _sha(result.config_hash, "config_hash")
        if source.config_ref is not None:
            config = _load_json(root, source.config_ref, "config")
            if config.get("config_hash") != result.config_hash:
                raise G23Blocked("config_hash:CONFIG_RESULT_MISMATCH")
        evidence.reference = _artifact(root, result, "reference_result")
        evidence.convergence = _artifact(root, result, "reference_convergence_report")
        evidence.gate = _artifact(root, result, "gate_record")
        rp, cp = evidence.reference.payload, evidence.convergence.payload
        evidence.identities["registry_hash"] = _sha(rp.get("registry_hash"), "registry_hash")
        evidence.identities["checkpoint_hash"] = _extract_checkpoint_hash(rp, cp, result_payload)
        evidence.identities["producer_commit"] = _extract_producer_commit(rp, cp, result_payload)
        bundle_ref = _path(rp.get("tensor_bundle_ref"), "tensor_bundle_ref")
        bundle_path = _resolve(root, bundle_ref)
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
        evidence.sizing_states = _load_resume_commits(root, sizing_root, "stage2-reference-progress-state-v1")
        final_states = _load_resume_commits(root, final_root, "stage2-reference-one-shot-progress-v1")
        evidence.final_state = final_states[-1]
        _validate_state_identity(evidence.final_state, evidence.identities, "final_resume")
        if int(evidence.final_state.get("processed_block_pairs", 0)) != len(final_states):
            raise G23Blocked("final_resume:COMMITTED_LENGTH_MISMATCH")
        _validate_state_identity(evidence.sizing_states[-1], evidence.identities, "sizing_resume")
    except G23Blocked as error:
        evidence.reasons.append(str(error))
    except Exception as error:
        evidence.reasons.append(f"UNEXPECTED_INPUT_ERROR:{type(error).__name__}")
    return evidence


def _sizing_vectors(evidence: _CellEvidence) -> tuple[list[int], list[dict[str, np.ndarray]], Mapping[str, object]]:
    if not evidence.sizing_states:
        raise G23Blocked("sizing:RAW_DIAGNOSTICS_MISSING")
    latest = evidence.sizing_states[-1]
    plan = None
    if evidence.convergence is not None:
        plan = evidence.convergence.payload.get("plan")
    if not isinstance(plan, Mapping):
        raise G23Blocked("sizing.plan:RAW_DIAGNOSTIC_MISSING")
    raw_counts = plan.get("candidate_sample_counts")
    if not isinstance(raw_counts, list) or len(raw_counts) < 2:
        raise G23Blocked("sizing.candidate_counts:RAW_DIAGNOSTIC_MISSING")
    counts = [int(item) for item in raw_counts]
    states_by_count: dict[int, Mapping[str, object]] = {}
    for state in evidence.sizing_states:
        sample_count = int(state.get("processed_block_pairs", 0)) * int(plan.get("block_size", 0))
        if sample_count in counts:
            states_by_count[sample_count] = state
    if any(count not in states_by_count for count in counts[-2:]):
        raise G23Blocked("sizing:ADJACENT_FINAL_NODES_MISSING")
    vectors: list[dict[str, np.ndarray]] = []
    for count in counts[-2:]:
        state = states_by_count[count]
        a = state.get("a")
        b = state.get("b")
        if not isinstance(a, Mapping):
            raise G23Blocked("sizing.a:MOMENTS_MISSING")
        if isinstance(b, Mapping) and int(b.get("count", 0)) > 0:
            vectors.append(_u_from_moments(_merge_moments(a, b, "sizing.moments"), "sizing.u"))
        else:
            vectors.append(_u_from_moments(a, "sizing.u"))
    return counts[-2:], vectors, plan


def _final_vectors(evidence: _CellEvidence) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], Mapping[str, np.ndarray]]:
    if evidence.final_state is None or evidence.bundle_state is None:
        raise G23Blocked("final:RAW_DIAGNOSTICS_MISSING")
    state = evidence.final_state
    raw_a, raw_b = state.get("blocks_a"), state.get("blocks_b")
    if not isinstance(raw_a, list) or not isinstance(raw_b, list) or not raw_a or len(raw_a) != len(raw_b):
        raise G23Blocked("final.blocks_a_b:RAW_DIAGNOSTICS_MISSING")
    blocks_a = [_vector(item, "final.blocks_a") for item in raw_a]
    blocks_b = [_vector(item, "final.blocks_b") for item in raw_b]
    for left, right in zip(blocks_a, blocks_b):
        _compatible(left, right, "final.blocks")
    all_blocks = blocks_a + blocks_b
    mean_a, mean_b, mean_all = _mean(blocks_a, "final.mean_a"), _mean(blocks_b, "final.mean_b"), _mean(all_blocks, "final.mean_all")
    bias = _u_from_moments(_merge_moments(state["a"], state["b"], "final.moments"), "final.bias")
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
    uncertainty = bstate.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise G23Blocked("bundle.uncertainty:MISSING")
    variance = _vector(uncertainty.get("bias_variance"), "bundle.bias_variance")
    return bias, cross, ranking, blocks_a, blocks_b, variance


def _group_map(vector: Mapping[str, np.ndarray], evidence: _CellEvidence) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    layer: dict[str, list[str]] = {}
    module: dict[str, list[str]] = {}
    metadata = evidence.reference.payload.get("metadata") if evidence.reference is not None else None
    registry = _find(metadata, {"parameter_registry", "registry", "parameter_groups"})
    if isinstance(registry, Mapping):
        for name in vector:
            item = registry.get(name)
            if isinstance(item, Mapping):
                layer_name = str(item.get("layer", item.get("layer_name", name.split(".")[0])))
                module_name = str(item.get("module", item.get("module_name", name.split(".")[0])))
            else:
                layer_name = name.split(".")[0]
                module_name = layer_name
            layer.setdefault(layer_name, []).append(name)
            module.setdefault(module_name, []).append(name)
    else:
        for name in vector:
            parts = name.split(".")
            layer.setdefault(".".join(parts[:2]) if len(parts) > 1 else parts[0], []).append(name)
            module.setdefault(parts[0], []).append(name)
    return layer, module


def _aggregate(vector: Mapping[str, np.ndarray], groups: Mapping[str, Sequence[str]]) -> tuple[np.ndarray, np.ndarray]:
    totals, means = [], []
    for names in sorted(groups):
        values = np.concatenate([np.abs(vector[name]).reshape(-1) for name in groups[names]])
        totals.append(float(values.sum()))
        means.append(float(values.mean()))
    return np.asarray(totals, dtype=np.float64), np.asarray(means, dtype=np.float64)


def _delta_sci(evidence: _CellEvidence, counts: Sequence[int]) -> tuple[dict[int, float], float]:
    if evidence.convergence is None:
        raise G23Blocked("delta_sci:CONVERGENCE_MISSING")
    source = _find(evidence.convergence.payload, {"delta_sci_by_B", "scientific_delta_by_B", "delta_sci"})
    if not isinstance(source, Mapping):
        raise G23Blocked("delta_sci:RAW_DIAGNOSTIC_MISSING")
    values: dict[int, float] = {}
    for count in counts:
        raw = source.get(str(count), source.get(count))
        if raw is None:
            raise G23Blocked(f"delta_sci:{count}:MISSING")
        values[count] = _finite(raw, f"delta_sci[{count}]")
        if values[count] <= 0:
            raise G23Blocked(f"delta_sci[{count}]:POSITIVE_REQUIRED")
    return values, min(values.values())


def _numerical_error(evidence: _CellEvidence, reference: Mapping[str, np.ndarray]) -> float:
    if evidence.convergence is None:
        raise G23Blocked("epsilon_num:CONVERGENCE_MISSING")
    source = _find(evidence.convergence.payload, {"numerical_diagnostics", "numeric_diagnostics", "accumulation_diagnostics"})
    if not isinstance(source, Mapping):
        raise G23Blocked("epsilon_num:RAW_DIAGNOSTIC_MISSING")
    high = _find(source, {"high_precision", "pairwise_reference", "reference_high_precision", "fp64_pairwise"})
    accumulated = _find(source, {"accumulated", "streaming_reference", "reference_accumulated", "fp32_accumulated"})
    if high is None or accumulated is None:
        raise G23Blocked("epsilon_num:HIGH_PRECISION_PAIR_REQUIRED")
    left, right = _vector(high, "epsilon_num.high_precision"), _vector(accumulated, "epsilon_num.accumulated")
    _compatible(left, right, "epsilon_num")
    _compatible(left, reference, "epsilon_num.reference")
    return float(max(np.max(np.abs(left[name] - right[name])) for name in left))


def _state_replay_verified(evidence: _CellEvidence) -> bool:
    if evidence.final_state is None or evidence.convergence is None:
        raise G23Blocked("state_replay:RAW_DIAGNOSTIC_MISSING")
    source: object = {"final": evidence.final_state, "convergence": evidence.convergence.payload}
    model_before = _find(source, {"model_state_before_hash", "model_before_hash"})
    model_after = _find(source, {"model_state_after_hash", "model_after_hash"})
    rng_before = _find(source, {"rng_state_before_hash", "rng_before_hash"})
    rng_after = _find(source, {"rng_state_after_hash", "rng_after_hash"})
    replay = _find(source, {"resume_replay_hash", "replay_hash", "resume_equivalence_hash"})
    replay_again = _find(source, {"fresh_replay_hash", "uninterrupted_replay_hash", "replayed_hash"})
    values = (model_before, model_after, rng_before, rng_after, replay, replay_again)
    if any(value is None for value in values):
        raise G23Blocked("state_replay:MODEL_RNG_REPLAY_HASHES_REQUIRED")
    hashes = [_sha(value, "state_replay_hash") for value in values]
    if hashes[0] != hashes[1] or hashes[2] != hashes[3] or hashes[4] != hashes[5]:
        return False
    return True


def _sequence_scaling(evidence: _CellEvidence, blocks: Sequence[Mapping[str, np.ndarray]], block_size: int) -> bool:
    if evidence.bundle_state is None:
        raise G23Blocked("variance_scaling:bundle_missing")
    stored = evidence.bundle_state.get("sequence_variance")
    if stored is None:
        raise G23Blocked("variance_scaling:SEQUENCE_VARIANCE_MISSING")
    sequence = _vector(stored, "bundle.sequence_variance")
    mean = _mean(blocks, "variance_scaling.blocks")
    expected = {name: np.zeros_like(value) for name, value in mean.items()}
    factor = float(block_size) / float(len(blocks) - 1)
    for block in blocks:
        for name in expected:
            expected[name] += np.square(block[name] - mean[name]) * factor
    _compatible(expected, sequence, "variance_scaling")
    return all(np.allclose(expected[name], sequence[name], rtol=1e-9, atol=1e-12) for name in expected)


def _evaluate_cell(evidence: _CellEvidence) -> dict[str, object]:
    cell: dict[str, object] = {"cell_id": evidence.source.cell_id, "status": "BLOCKED", "identities": dict(evidence.identities), "reasons": list(evidence.reasons)}
    if evidence.reasons:
        return cell
    try:
        counts, previous, plan = _sizing_vectors(evidence)
        bias, cross, ranking, blocks_a, blocks_b, bias_variance = _final_vectors(evidence)
        previous_bias, latest_bias = previous
        flat_prev, flat_latest = _flat(previous_bias), _flat(latest_bias)
        delta_sci, min_delta = _delta_sci(evidence, [int(item) for item in plan["candidate_sample_counts"]])
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
        uncertainty = evidence.bundle_state["uncertainty"]  # type: ignore[index]
        cross_variance = _vector(uncertainty["cross_variance"], "bundle.cross_variance")
        ranking_variance = _vector(uncertainty["ranking_variance"], "bundle.ranking_variance")
        _compatible(bias_variance, cross_variance, "variance")
        _compatible(bias_variance, ranking_variance, "variance")
        variance_total = float(sum(np.sum(value) for value in bias_variance.values()))
        h_ref = 1.96 * math.sqrt(max(0.0, variance_total))
        cross_diff = _flat(cross) - _flat(bias)
        cross_sd = np.sqrt(np.maximum(0.0, _flat(bias_variance) + _flat(cross_variance)))
        a_mean, b_mean = _mean(blocks_a, "a_mean"), _mean(blocks_b, "b_mean")
        a_rank, b_rank = np.square(_flat(a_mean)), np.square(_flat(b_mean))
        signal_floor = _find(evidence.convergence.payload if evidence.convergence else {}, {"numerical_floor", "numeric_floor"})
        if signal_floor is None:
            raise G23Blocked("signal_eligible:numerical_floor_missing")
        floor = _finite(signal_floor, "numerical_floor")
        signal_mask = np.abs(flat_latest) > np.maximum(5.0 * 1.96 * np.sqrt(np.maximum(0.0, _flat(bias_variance))), 10.0 * floor)
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
            "min_delta_sci": min_delta,
            "epsilon_num": epsilon_num,
            "a_b_spearman": _spearman(a_rank, b_rank),
            "a_b_top_overlap_0_01": _top_overlap(a_rank, b_rank, 0.01),
            "bias_cross_interval_max_z": float(np.max(np.abs(cross_diff) / np.maximum(cross_sd, 1e-300))),
            "ranking_bias_direction_sum": float(np.sum(_flat(ranking) - _flat(bias))),
            "sizing_min_delta_sci": min_delta,
            "sizing_node_previous": int(counts[0]),
            "sizing_node_latest": int(counts[1]),
        }
        checks: dict[str, bool] = {
            "normalized_l1": metrics["normalized_l1"] <= 0.02,
            "pearson": metrics["pearson"] >= 0.995,
            "signal_eligible_spearman": metrics["signal_eligible_spearman"] >= 0.995,
            "layer_module_spearman": metrics["layer_module_spearman"] >= 0.995,
            "topk_overlap_0_001": metrics["topk_overlap_0_001"] >= 0.98,
            "topk_overlap_0_01": metrics["topk_overlap_0_01"] >= 0.98,
            "topk_overlap_0_05": metrics["topk_overlap_0_05"] >= 0.98,
            "layer_module_delta": metrics["layer_module_delta"] <= 0.01,
            "h_ref": metrics["h_ref"] <= min_delta / 4.0,
            "epsilon_num": metrics["epsilon_num"] <= min_delta / 10.0,
            "a_b_spearman": metrics["a_b_spearman"] >= 0.995,
            "a_b_top_overlap": metrics["a_b_top_overlap_0_01"] >= 0.98,
            "bias_cross_interval_covered": bool(np.all(np.abs(cross_diff) <= 1.96 * np.maximum(cross_sd, 0.0))),
            "ranking_bias_direction": metrics["ranking_bias_direction_sum"] >= 0.0,
            "variance_scaling_verified": _sequence_scaling(evidence, blocks_a + blocks_b, block_size),
            "state_replay_verified": _state_replay_verified(evidence),
        }
        one_shot = evidence.convergence.payload.get("one_shot_result") if evidence.convergence else None
        plan_one_shot = evidence.convergence.payload.get("one_shot_plan") if evidence.convergence else None
        checks["one_shot_complete"] = isinstance(one_shot, Mapping) and one_shot.get("status") == "COMPLETE" and one_shot.get("one_shot") is True and isinstance(plan_one_shot, Mapping) and plan_one_shot.get("one_shot") is True and int(one_shot.get("processed_sample_count_per_stream", 0)) == int(plan_one_shot.get("sample_count_per_stream", -1)) and len(blocks_a) * block_size == int(one_shot.get("processed_sample_count_per_stream", -1))
        checks["a_b_interval_covered"] = bool(np.all(np.abs(_flat(a_mean) - _flat(b_mean)) <= 1.96 * np.sqrt(np.maximum(0.0, _flat(bias_variance)))))
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
        reasons = [name + ":THRESHOLD_FAILED" for name, passed in checks.items() if not passed]
        cell.update({"status": "PASS" if not reasons else "BLOCKED", "metrics": metrics, "checks": checks, "reasons": reasons, "endpoints": {"model_total": {"metrics": metrics, "checks": checks}}})
    except G23Blocked as error:
        cell["reasons"] = list(cell.get("reasons", [])) + [str(error)]
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as error:
        cell["reasons"] = list(cell.get("reasons", [])) + [f"RAW_DIAGNOSTIC_INVALID:{type(error).__name__}"]
    return cell


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
) -> dict[str, object]:
    """Evaluate exactly six formal Stage2.04 outputs and publish one attempt.

    ``cells`` contains only references.  If fewer than six references are
    supplied the result is ``BLOCKED``/``NOT_RUN`` and no partial metric can
    qualify the study.
    """

    root = Path(workspace_root).resolve()
    normalized_items: list[CellInput] = []
    for index, item in enumerate(cells):
        if isinstance(item, CellInput):
            normalized_items.append(item)
        elif isinstance(item, str):
            normalized_items.append(CellInput(f"cell-{index}", item))
        else:
            normalized_items.append(CellInput.from_mapping(item))
    normalized = tuple(normalized_items)
    inferred = tuple(item.cell_id for item in normalized)
    expected = tuple(expected_cell_ids or inferred)
    if len(expected) != REQUIRED_CELL_COUNT or len(set(expected)) != REQUIRED_CELL_COUNT:
        expected = tuple(f"cell-{index}" for index in range(REQUIRED_CELL_COUNT))
    prepared: list[_CellEvidence] = []
    structure_reason: str | None = None
    if len(normalized) == REQUIRED_CELL_COUNT and len(set(item.cell_id for item in normalized)) == REQUIRED_CELL_COUNT:
        prepared = [_prepare_cell(root, item) for item in normalized]
    elif normalized:
        structure_reason = "DUPLICATE_OR_INVALID_CELL_IDENTITIES"
    cell_rows = [_evaluate_cell(item) for item in prepared]
    source_hashes: dict[str, str] = {}
    producer_values: set[str] = set()
    for item in prepared:
        source_hashes[item.source.cell_id] = item.identities.get("result_hash", "")
        if "producer_commit" in item.identities:
            producer_values.add(item.identities["producer_commit"])
    if len(producer_values) == 1:
        producer = next(iter(producer_values))
    else:
        producer = ""
    module_path = Path(__file__).resolve()
    calculator = {"producer_commit": producer, "source_sha256": _digest_bytes(module_path), "source_schema": SCHEMA_VERSION}
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
        out = Path(output_root).resolve()
        attempt_dir = out / "g2.3-attempts" / artifact_hash
        attempt_dir.mkdir(parents=True, exist_ok=True)
        target = attempt_dir / "evaluation.json"
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
        line = str(payload["artifact_hash"]) + "\n"
        if not index.exists() or line not in index.read_text(encoding="utf-8").splitlines(keepends=True):
            with index.open("a", encoding="utf-8") as handle:
                handle.write(line)
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

    def evaluate(self, cells: Sequence[CellInput | Mapping[str, object] | str]) -> dict[str, object]:
        return evaluate_formal_g23(
            self.workspace_root,
            cells,
            expected_cell_ids=self.expected_cell_ids,
            output_root=self.output_root,
        )


__all__ = [
    "CellInput",
    "G23Blocked",
    "G23ReferenceEvaluator",
    "GATE_ID",
    "SCHEMA_VERSION",
    "THRESHOLDS",
    "evaluate_formal_g23",
    "evaluate_g23",
]
