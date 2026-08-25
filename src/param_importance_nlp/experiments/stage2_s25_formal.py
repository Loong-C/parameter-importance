"""Strict S2.5/G2.4a production control plane.

The generic Stage 2 task runner intentionally follows the plan DAG and does
not have S2.4 as a direct predecessor.  A fresh S2.4 run therefore hands its
six cell identities to this module through the immutable S2.5 rebind plan.
This module is the detached consumer of that handoff: it revalidates the
source at launch time, loads the independent S2.4 reference views, and runs
the existing :class:`RecoverablePairedWaveRunner` without creating draws or
falling back to a fixture provider.

The output of the runner is still a candidate until all six cells have
completed the operational G2.4a checks.  Only then is the small, immutable
``stage2-g24a-formal-evaluation-v1`` object published as PASS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.task_catalog import DEFAULT_TASK_CATALOG
from ..experiments.sampling import RepetitionMapping, SamplingPlan
from ..runtime.task_artifacts import load_committed_task_artifact
from ..runtime.task_runtime import TaskExecutionRequest, TaskRuntimeEnvironment
from ..runtime.tensor_bundle import load_tensor_bundle
from .stage2_formal import RecoverablePairedWaveRunner, _vector_digest
from .stage2_s25_rebind import (
    APPROVED_GPU_UUIDS,
    CELL_COMPONENTS,
    EXPECTED_CELL_IDS,
    EXCLUDED_PCI,
)


S25_REBIND_SCHEMA = "stage2-s205-rebind-plan-v1"
S25_GATE_SCHEMA = "stage2-g24a-formal-evaluation-v1"
S25_CELL_SUMMARY_SCHEMA = "stage2-s25-formal-cell-summary-v1"
S25_STATUS_SCHEMA = "stage2-s25-formal-status-v1"
S25_TASK_ID = "stage2.05_paired_estimator_runner"
S25_G23_SCHEMA = "stage2-g23-reference-evaluation-v1"
S25_M2_TOLERANCE = 1e-10

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class S25ExecutionBlocked(RuntimeError):
    """Raised when a formal S2.5 source or recovery boundary is unsafe."""


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S25ExecutionBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _ref(root: Path, value: object, *, field: str, allow_missing: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S25ExecutionBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise S25ExecutionBlocked(f"{field}:PATH_ESCAPE")
    if root.is_symlink():
        raise S25ExecutionBlocked(f"{field}:SYMLINK_ROOT")
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise S25ExecutionBlocked(f"{field}:SYMLINK_COMPONENT")
    target = (root / Path(*logical.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise S25ExecutionBlocked(f"{field}:PATH_ESCAPE") from error
    if not allow_missing and not target.exists():
        raise S25ExecutionBlocked(f"{field}:MISSING")
    return target


def _load_object(root: Path, value: object, *, field: str) -> tuple[Path, dict[str, Any]]:
    path = _ref(root, value, field=field)
    try:
        loaded = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S25ExecutionBlocked(f"{field}:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(loaded, Mapping):
        raise S25ExecutionBlocked(f"{field}:OBJECT_REQUIRED")
    return path, dict(loaded)


def _verify_artifact(value: Mapping[str, object], *, field: str) -> str:
    declared = _sha(value.get("artifact_hash"), field=f"{field}.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(body) != declared:
        raise S25ExecutionBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return declared


def _finite(value: object, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise S25ExecutionBlocked(f"{field}:NONFINITE")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, field=f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, field=f"{field}[{index}]")
        return
    # numpy scalars may occur in only test-only loader probes; convert them
    # before applying the same finite rule used by canonical JSON artifacts.
    if isinstance(value, np.generic):
        _finite(value.item(), field=field)
        return
    raise S25ExecutionBlocked(f"{field}:NOT_JSON_VALUE")


def _validate_g23(root: Path, plan: Mapping[str, object]) -> tuple[str, dict[str, Any]]:
    ref = plan.get("g23_evaluation_ref")
    path, value = _load_object(root, ref, field="s205_plan.g23_evaluation_ref")
    digest = _verify_artifact(value, field="g23_evaluation")
    if (
        value.get("schema_version") != S25_G23_SCHEMA
        or value.get("gate_id") != "stage2.G2.3"
        or value.get("status") != "PASS"
        or value.get("formal_eligible") is not True
        or value.get("required_cell_count") != 6
        or value.get("complete_cell_count") != 6
        or digest != plan.get("g23_evaluation_hash")
    ):
        raise S25ExecutionBlocked("G2.3_STRICT_PASS_OR_HASH_INVALID")
    cells = value.get("cells")
    if not isinstance(cells, list) or tuple(
        item.get("cell_id") if isinstance(item, Mapping) else None for item in cells
    ) != EXPECTED_CELL_IDS:
        raise S25ExecutionBlocked("G2.3_CELL_ORDER_INVALID")
    for item in cells:
        if not isinstance(item, Mapping) or item.get("status") != "PASS":
            raise S25ExecutionBlocked("G2.3_CELL_PASS_REQUIRED")
        if not isinstance(item.get("identities"), Mapping):
            raise S25ExecutionBlocked("G2.3_CELL_IDENTITIES_REQUIRED")
        if not isinstance(item.get("metrics"), Mapping):
            raise S25ExecutionBlocked("G2.3_OUTPUT_METRICS_REQUIRED")
        _finite(item["metrics"], field=f"G2.3.metrics.{item.get('cell_id')}")
    return path.relative_to(root).as_posix(), value


def _validate_source_row(
    root: Path,
    row: Mapping[str, object],
    g23_cell: Mapping[str, object],
    *,
    execution_commit: str,
) -> None:
    cell_id = row.get("cell_id")
    if cell_id not in EXPECTED_CELL_IDS:
        raise S25ExecutionBlocked("S205_CELL_ID_INVALID")
    required = {
        "component", "config_ref", "environment_ref", "reference_artifact_refs",
        "config_hash", "status_artifact_hash", "result_hash", "execution_commit",
        "formal_execution_ref", "formal_execution_hash", "authorization_execution_commit",
        "authorization_producer_commit",
    }
    if not required.issubset(row):
        raise S25ExecutionBlocked(f"S205_CELL_FIELDS_MISSING:{cell_id}")
    if row.get("execution_commit") != execution_commit or _COMMIT.fullmatch(str(execution_commit)) is None:
        raise S25ExecutionBlocked(f"S205_EXECUTION_COMMIT_INVALID:{cell_id}")
    for name in ("config_hash", "status_artifact_hash", "result_hash", "formal_execution_hash"):
        _sha(row.get(name), field=f"S205.{cell_id}.{name}")
    for name in ("authorization_execution_commit", "authorization_producer_commit"):
        if _COMMIT.fullmatch(str(row.get(name, ""))) is None:
            raise S25ExecutionBlocked(f"S205.{cell_id}.{name}:COMMIT_REQUIRED")
    _ref(root, row.get("config_ref"), field=f"S205.{cell_id}.config_ref")
    environment_path = _ref(root, row.get("environment_ref"), field=f"S205.{cell_id}.environment_ref")
    formal_path = _ref(root, row.get("formal_execution_ref"), field=f"S205.{cell_id}.formal_execution_ref")
    status_path = _ref(root, row.get("task_result_status_path"), field=f"S205.{cell_id}.final_status")

    config = load_canonical_json(_ref(root, row["config_ref"], field=f"S205.{cell_id}.config").resolve())
    if not isinstance(config, Mapping) or config.get("config_hash") != row.get("config_hash"):
        raise S25ExecutionBlocked(f"S205_CONFIG_HASH_INVALID:{cell_id}")
    status = load_canonical_json(status_path)
    if (
        not isinstance(status, Mapping)
        or status.get("schema_version") != "stage2-s204-cell-final-status-v3"
        or status.get("status") != "COMPLETE"
        or status.get("formal_eligible") is not True
        or status.get("cell_id") != cell_id
        or status.get("execution_commit") != execution_commit
        or status.get("artifact_hash") != row.get("status_artifact_hash")
        or status.get("task_result_hash") != row.get("result_hash")
    ):
        raise S25ExecutionBlocked(f"S204_STATUS_DRIFT:{cell_id}")
    _verify_artifact(status, field=f"S204.{cell_id}.final_status")

    env = load_canonical_json(environment_path)
    if not isinstance(env, Mapping) or env.get("schema_version") != "task-runtime-environment-v1":
        raise S25ExecutionBlocked(f"S205_ENVIRONMENT_INVALID:{cell_id}")
    if env.get("evidence_refs", {}).get("formal_execution") != row.get("formal_execution_ref"):
        raise S25ExecutionBlocked(f"S205_FORMAL_EXECUTION_BINDING_INVALID:{cell_id}")
    formal = load_canonical_json(formal_path)
    if not isinstance(formal, Mapping) or formal.get("schema_version") != "formal-execution-evidence-v1":
        raise S25ExecutionBlocked(f"S205_FORMAL_EXECUTION_INVALID:{cell_id}")
    if _verify_artifact(formal, field=f"S205.{cell_id}.formal_execution") != row.get("formal_execution_hash"):
        raise S25ExecutionBlocked(f"S205_FORMAL_EXECUTION_HASH_INVALID:{cell_id}")
    identities = g23_cell.get("identities")
    if not isinstance(identities, Mapping) or identities.get("result_hash") != row.get("result_hash") or identities.get("config_hash") != row.get("config_hash"):
        raise S25ExecutionBlocked(f"G2.3_S204_BINDING_INVALID:{cell_id}")

    refs = row.get("reference_artifact_refs")
    if not isinstance(refs, Mapping) or set(refs) != {"reference_result", "reference_convergence_report", "gate_record"}:
        raise S25ExecutionBlocked(f"S205_REFERENCE_REFS_INVALID:{cell_id}")
    for kind, reference in refs.items():
        try:
            loaded = load_committed_task_artifact(root, str(reference), require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise S25ExecutionBlocked(f"S205_REFERENCE_COMMIT_INVALID:{cell_id}:{kind}") from error
        if (
            loaded.identity.task_id != "stage2.04_reference_target"
            or loaded.identity.artifact_kind != kind
            or loaded.identity.config_hash != row.get("config_hash")
            or loaded.identity.formal_eligible is not True
            or loaded.run_intent != "formal"
        ):
            raise S25ExecutionBlocked(f"S205_REFERENCE_IDENTITY_INVALID:{cell_id}:{kind}")


def load_s25_rebind_plan(data_root: str | Path, rebind_ref: str) -> dict[str, Any]:
    """Load and revalidate a fresh S2.4/G2.3 handoff at launch time."""

    root = Path(data_root).resolve()
    path, plan = _load_object(root, rebind_ref, field="s205_rebind_ref")
    # d9e79d5 predates an embedded plan hash.  Preserve that handoff format by
    # deriving its canonical digest, while accepting the stronger embedded
    # hash when a producer has already upgraded the control-plane envelope.
    digest = (
        _verify_artifact(plan, field="s205_rebind_plan")
        if "artifact_hash" in plan
        else canonical_json_hash(plan)
    )
    if plan.get("schema_version") != S25_REBIND_SCHEMA or plan.get("status") != "READY" or plan.get("formal_eligible") is not True:
        raise S25ExecutionBlocked("S205_REBIND_READY_FORMAL_REQUIRED")
    if not isinstance(plan.get("execution_commit"), str) or _COMMIT.fullmatch(plan["execution_commit"]) is None:
        raise S25ExecutionBlocked("S205_REBIND_EXECUTION_COMMIT_INVALID")
    rows = plan.get("cells")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_CELL_IDS):
        raise S25ExecutionBlocked("S205_REBIND_SIX_CELL_ROWS_REQUIRED")
    g23_ref, g23 = _validate_g23(root, plan)
    for index, (expected, row) in enumerate(zip(EXPECTED_CELL_IDS, rows)):
        if not isinstance(row, Mapping) or row.get("cell_id") != expected or row.get("component") != CELL_COMPONENTS[expected]:
            raise S25ExecutionBlocked(f"S205_REBIND_CELL_ORDER_INVALID:{index}")
        _validate_source_row(root, row, g23["cells"][index], execution_commit=str(plan["execution_commit"]))  # type: ignore[index]
    result = dict(plan)
    result["artifact_hash"] = digest
    result["_data_root"] = str(root)
    result["_rebind_ref"] = path.relative_to(root).as_posix()
    result["_g23_ref"] = g23_ref
    result["_g23"] = g23
    return result


def _reference_views(root: Path, row: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    refs = row["reference_artifact_refs"]
    assert isinstance(refs, Mapping)
    try:
        loaded = load_committed_task_artifact(root, str(refs["reference_result"]), require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise S25ExecutionBlocked(f"S205_REFERENCE_RESULT_INVALID:{row.get('cell_id')}") from error
    payload = dict(loaded.payload)
    if payload.get("schema_version") != "reference-result-v1" or payload.get("scope") != "formal" or payload.get("formal_eligible") is not False:
        raise S25ExecutionBlocked(f"S205_REFERENCE_RESULT_SCOPE_INVALID:{row.get('cell_id')}")
    bundle_ref = payload.get("tensor_bundle_ref")
    if not isinstance(bundle_ref, str) or not bundle_ref:
        raise S25ExecutionBlocked(f"S205_REFERENCE_BUNDLE_REF_MISSING:{row.get('cell_id')}")
    candidates: list[Path] = []
    config_path = _ref(root, row["config_ref"], field="S205.config_ref")
    config_value = load_canonical_json(config_path)
    output_dir = None
    if isinstance(config_value, Mapping):
        artifacts = config_value.get("artifacts")
        if isinstance(artifacts, Mapping) and isinstance(artifacts.get("output_dir"), str):
            output_dir = _ref(root, artifacts["output_dir"], field="S205.config.output_dir")
    bases = [root, config_path.parent]
    if output_dir is not None:
        bases.append(output_dir)
    for base in bases:
        try:
            candidates.append(_ref(base, bundle_ref, field="S205.reference_bundle"))
        except S25ExecutionBlocked:
            continue
    bundle_path = next((item for item in candidates if item.is_dir()), None)
    if bundle_path is None or bundle_path.is_symlink():
        raise S25ExecutionBlocked(f"S205_REFERENCE_BUNDLE_MISSING:{row.get('cell_id')}")
    try:
        state, bundle = load_tensor_bundle(bundle_path)
    except (OSError, TypeError, ValueError) as error:
        raise S25ExecutionBlocked(f"S205_REFERENCE_BUNDLE_INVALID:{row.get('cell_id')}") from error
    if not isinstance(state, Mapping) or bundle.manifest_sha256 != payload.get("tensor_bundle_manifest_hash"):
        raise S25ExecutionBlocked(f"S205_REFERENCE_BUNDLE_HASH_INVALID:{row.get('cell_id')}")
    views: dict[str, Mapping[str, object]] = {}
    for short, long_name in (("bias", "bias_reference"), ("cross", "cross_reference"), ("ranking", "ranking_reference")):
        value = state.get(long_name)
        declared = payload.get(f"{long_name}_hash")
        if not isinstance(value, Mapping) or not isinstance(declared, str) or _vector_digest(value) != declared:
            raise S25ExecutionBlocked(f"S205_REFERENCE_VIEW_HASH_INVALID:{row.get('cell_id')}:{short}")
        views[short] = value
    return views


def load_s25_experiment_plan(data_root: str | Path, plan_ref: str) -> dict[str, Any]:
    """Load the pre-registered pilot B/M/R plan; no draw is generated."""

    root = Path(data_root).resolve()
    _path, value = _load_object(root, plan_ref, field="s205_experiment_plan_ref")
    try:
        from .stage2_formal import FormalExperimentPlan

        plan = FormalExperimentPlan.from_mapping(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise S25ExecutionBlocked(f"S205_EXPERIMENT_PLAN_INVALID:{error}") from error
    if plan.task_id != S25_TASK_ID or plan.stream != "pilot" or plan.selection_basis != "preregistered_development":
        raise S25ExecutionBlocked("S205_EXPERIMENT_PLAN_TASK_OR_STREAM_INVALID")
    return plan.to_dict()


def _derive_s205_config(source: ResolvedConfigV2) -> ResolvedConfigV2:
    wire = source.to_dict()
    payload = {key: value for key, value in wire.items() if key not in {"config_hash", "full_hash"}}
    payload["task_id"] = S25_TASK_ID
    execution = payload.get("execution")
    artifacts = payload.get("artifacts")
    if not isinstance(execution, Mapping) or not isinstance(artifacts, Mapping):
        raise S25ExecutionBlocked("S205_SOURCE_CONFIG_SECTIONS_INVALID")
    execution_wire = dict(execution)
    execution_wire["runner_kind"] = DEFAULT_TASK_CATALOG.get(S25_TASK_ID).runner_kind.value
    payload["execution"] = execution_wire
    artifacts_wire = dict(artifacts)
    artifacts_wire["required_kinds"] = list(DEFAULT_TASK_CATALOG.get(S25_TASK_ID).artifact_kinds)
    payload["artifacts"] = artifacts_wire
    try:
        derived = ResolvedConfigV2(payload)
    except (TypeError, ValueError) as error:
        raise S25ExecutionBlocked(f"S205_CONFIG_DERIVATION_FAILED:{error}") from error
    derived_wire = derived.to_dict()
    for section in set(wire) - {"task_id", "execution", "artifacts", "config_hash", "full_hash"}:
        if derived_wire.get(section) != wire.get(section):
            raise S25ExecutionBlocked(f"S205_CONFIG_BINDING_DROPPED:{section}")
    return derived


def _build_provider(root: Path, plan: Mapping[str, object], row: Mapping[str, object]) -> tuple[Any, FormalExecutionEvidence, SamplingPlan]:
    config_value = load_canonical_json(_ref(root, row["config_ref"], field="S205.config_ref"))
    environment_value = load_canonical_json(_ref(root, row["environment_ref"], field="S205.environment_ref"))
    if not isinstance(config_value, Mapping) or not isinstance(environment_value, Mapping):
        raise S25ExecutionBlocked(f"S205_PROVIDER_INPUT_OBJECT_INVALID:{row.get('cell_id')}")
    try:
        source_config = ResolvedConfigV2.from_mapping(dict(config_value))
        source_environment = TaskRuntimeEnvironment.from_mapping(dict(environment_value))
        derived_config = _derive_s205_config(source_config)
        formal_value = load_canonical_json(_ref(root, row["formal_execution_ref"], field="S205.formal_execution_ref"))
        execution = FormalExecutionEvidence.from_mapping(dict(formal_value))
        execution.require_for_stage(2)
        from .stage23_task_runners import _formal_provider, _sampling_plan

        refs = dict(source_environment.evidence_refs)
        refs["formal_execution"] = str(row["formal_execution_ref"])
        environment = TaskRuntimeEnvironment(
            capabilities=source_environment.capabilities | frozenset({"server", "cuda", "model_assets", "data_assets"}),
            frozen_contract_stages=source_environment.frozen_contract_stages | frozenset({2}),
            passed_gate_ids=source_environment.passed_gate_ids | frozenset({"stage2.G2.3"}),
            estimator_decision_ref=source_environment.estimator_decision_ref,
            evidence_refs=refs,
        )
        request = TaskExecutionRequest(
            config=derived_config,
            task=DEFAULT_TASK_CATALOG.get(S25_TASK_ID),
            environment=environment,
        )
        context = _formal_provider(request, root)
        provider_sampling = _sampling_plan(request, context)
    except Exception as error:
        if isinstance(error, S25ExecutionBlocked):
            raise
        raise S25ExecutionBlocked(f"S205_PROVIDER_BIND_FAILED:{row.get('cell_id')}:{type(error).__name__}:{error}") from error
    return context.provider, execution, provider_sampling


def _make_mappings(sampling: SamplingPlan, plan: Mapping[str, object]) -> tuple[RepetitionMapping, ...]:
    from .stage2_formal import FormalExperimentPlan

    formal_plan = FormalExperimentPlan.from_mapping(dict(plan))
    draws = sampling.draws("pilot", formal_plan.repetitions * formal_plan.batch_size)
    return tuple(
        RepetitionMapping.create(
            repetition_id=f"rep-{index:04d}",
            draws=draws[index * formal_plan.batch_size : (index + 1) * formal_plan.batch_size],
            m_values=formal_plan.microbatch_counts,
        )
        for index in range(formal_plan.repetitions)
    )


def _cell_checks(root: Path, artifact_root: Path, expected: Sequence[str]) -> dict[str, object]:
    commits = artifact_root / "commits"
    if not commits.is_dir():
        raise S25ExecutionBlocked("S205_WAVE_COMMITS_MISSING")
    state_ok = True
    m2_ok = True
    finite_ok = True
    signed_ok = True
    for unit_id in expected:
        path = commits / f"{unit_id}.json"
        if not path.is_file():
            raise S25ExecutionBlocked(f"S205_WAVE_UNIT_MISSING:{unit_id}")
        commit = load_canonical_json(path)
        if not isinstance(commit, Mapping):
            raise S25ExecutionBlocked(f"S205_WAVE_COMMIT_INVALID:{unit_id}")
        relative = Path(str(commit.get("object_ref", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise S25ExecutionBlocked(f"S205_WAVE_OBJECT_ESCAPE:{unit_id}")
        state, _bundle = load_tensor_bundle(artifact_root / relative)
        if not isinstance(state, Mapping):
            raise S25ExecutionBlocked(f"S205_WAVE_STATE_INVALID:{unit_id}")
        state_ok &= state.get("state_digest") == state.get("state_digest_after")
        m2 = state.get("m2_double_max_abs_error")
        m2_ok &= isinstance(m2, (int, float)) and not isinstance(m2, bool) and math.isfinite(float(m2)) and float(m2) <= S25_M2_TOLERANCE
        _finite(state, field=f"S205.wave.{unit_id}")
        vectors = state.get("vectors")
        signed_ok &= isinstance(vectors, Mapping) and all("u_m" in str(name) or str(name) in {"raw", "double"} for name in vectors)
        for field in ("gradient_evaluations", "gradient_seconds", "formula_seconds", "wall_seconds", "sample_budget", "statistical_weight"):
            finite_ok &= isinstance(state.get(field), (int, float)) and not isinstance(state.get(field), bool) and math.isfinite(float(state[field]))
    return {
        "state_replay_verified": bool(state_ok),
        "m2_equivalent": bool(m2_ok),
        "finite_costs": bool(finite_ok),
        "signed_outputs": bool(signed_ok),
        "unit_count": len(expected),
    }


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_canonical_json(path)
        if existing != dict(payload):
            raise S25ExecutionBlocked(f"S205_OUTPUT_CONFLICT:{path}")
        return
    write_canonical_json(path, dict(payload))


@dataclass(frozen=True, slots=True)
class S25FormalRunner:
    """Execute one S2.5 cell or the complete six-cell G2.4a wave."""

    data_root: Path
    rebind_plan: Mapping[str, object]
    experiment_plan: Mapping[str, object]
    sampling_plan: SamplingPlan
    artifact_root: Path
    m2_tolerance: float = S25_M2_TOLERANCE

    def run_cell(self, cell_id: str) -> dict[str, object]:
        if cell_id not in EXPECTED_CELL_IDS:
            raise S25ExecutionBlocked("S205_CELL_UNKNOWN")
        rows = self.rebind_plan.get("cells")
        assert isinstance(rows, list)
        row = next(item for item in rows if isinstance(item, Mapping) and item.get("cell_id") == cell_id)
        component = CELL_COMPONENTS[cell_id]
        cell_root = self.artifact_root / "cells" / component
        summary_path = cell_root / "summary.json"
        if summary_path.exists():
            value = load_canonical_json(summary_path)
            if not isinstance(value, Mapping) or value.get("schema_version") != S25_CELL_SUMMARY_SCHEMA or value.get("status") != "PASS":
                raise S25ExecutionBlocked(f"S205_EXISTING_CELL_NOT_RECOVERABLE:{cell_id}")
            return dict(value) | {"resumed": True}
        provider, execution, provider_sampling = _build_provider(self.data_root, self.rebind_plan, row)
        try:
            from .stage23_task_runners import _project_sampling_plan_to_provider

            projected = _project_sampling_plan_to_provider(self.sampling_plan, provider_sampling)
        except Exception as error:
            raise S25ExecutionBlocked(f"S205_SAMPLING_PROVIDER_BIND_FAILED:{cell_id}:{error}") from error
        mappings = _make_mappings(projected, self.experiment_plan)
        views = _reference_views(self.data_root, row)
        runner = RecoverablePairedWaveRunner(provider, execution=execution, m2_tolerance=self.m2_tolerance)
        summary = runner.run(
            wave_id=f"s205-{component}",
            mappings=mappings,
            reference=views["bias"],
            reference_hash=_vector_digest(views["bias"]),
            references=views,
            artifact_root=cell_root / "paired-wave",
        )
        summary_payload = summary.to_dict()
        summary_output = cell_root / "paired-wave" / "summary.json"
        _write_once(summary_output, summary_payload)
        checks = _cell_checks(self.data_root, cell_root / "paired-wave", summary.completed_unit_ids)
        transitions = summary.replay_evidence.get("state_transitions")
        checks["replay_complete"] = isinstance(transitions, Mapping) and set(transitions) == set(summary.expected_unit_ids)
        checks["complete"] = summary.complete
        passed = all(bool(value) for value in checks.values() if isinstance(value, bool))
        g23_cells = self.rebind_plan["_g23"]["cells"]  # type: ignore[index]
        g23_cell = next(item for item in g23_cells if item["cell_id"] == cell_id)
        metrics = dict(g23_cell["metrics"])
        metrics.update({"runner": checks, "sample_budget": summary.cost_statistics["scientific_equal_sample_cost"].get("sample_budget")})
        payload: dict[str, object] = {
            "schema_version": S25_CELL_SUMMARY_SCHEMA,
            "cell_id": cell_id,
            "status": "PASS" if passed else "BLOCKED",
            "formal_eligible": bool(passed),
            "rebind_plan_hash": self.rebind_plan["artifact_hash"],
            "g23_evaluation_hash": self.rebind_plan["g23_evaluation_hash"],
            "experiment_plan_hash": canonical_json_hash(self.experiment_plan),
            "sampling_plan_hash": self.sampling_plan.digest,
            "summary_hash": summary.artifact_hash,
            "summary_ref": summary_output.relative_to(self.data_root).as_posix(),
            "metrics": metrics,
            "checks": checks,
            "reference_hashes": dict(summary.reference_hashes),
            "expected_unit_ids": list(summary.expected_unit_ids),
            "completed_unit_ids": list(summary.completed_unit_ids),
            "failure_evidence_dir": (cell_root / "paired-wave" / "failures").relative_to(self.data_root).as_posix(),
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        _write_once(summary_path, payload)
        return payload

    def run_all(self) -> dict[str, object]:
        # ``run_cell`` annotates a recovered row for the caller, but recovery
        # metadata is not part of the scientific Gate identity.  Strip it so
        # a second aggregation is byte-identical to the first publication.
        results = [
            {key: value for key, value in self.run_cell(cell_id).items() if key != "resumed"}
            for cell_id in EXPECTED_CELL_IDS
        ]
        passed = all(item.get("status") == "PASS" and item.get("formal_eligible") is True for item in results)
        gate: dict[str, object] = {
            "schema_version": S25_GATE_SCHEMA,
            "gate_id": "stage2.G2.4a",
            "status": "PASS" if passed else "BLOCKED",
            "formal_eligible": bool(passed),
            "cell_count": len(results),
            "results": results,
            "rebind_plan_ref": self.rebind_plan["_rebind_ref"],
            "rebind_plan_hash": self.rebind_plan["artifact_hash"],
            "g23_evaluation_ref": self.rebind_plan["_g23_ref"],
            "g23_evaluation_hash": self.rebind_plan["g23_evaluation_hash"],
            "execution_commit": self.rebind_plan["execution_commit"],
            "evidence_refs": [self.rebind_plan["_rebind_ref"], self.rebind_plan["_g23_ref"]],
            "confirmatory_draws_generated": False,
        }
        gate["artifact_hash"] = canonical_json_hash(gate)
        gate_path = self.artifact_root / "g2.4a-evaluation.json"
        _write_once(gate_path, gate)
        return gate


def preflight_s25(
    data_root: str | Path,
    *,
    rebind_ref: str,
    sampling_ref: str,
    experiment_plan_ref: str,
    artifact_root: str,
) -> dict[str, object]:
    """Read-only launch check; it never constructs a provider or draws."""

    root = Path(data_root).resolve()
    rebind = load_s25_rebind_plan(root, rebind_ref)
    _, sampling_value = _load_object(root, sampling_ref, field="sampling_plan_ref")
    try:
        sampling = SamplingPlan.from_mapping(sampling_value)
    except (TypeError, ValueError) as error:
        raise S25ExecutionBlocked(f"S205_SAMPLING_PLAN_INVALID:{error}") from error
    experiment = load_s25_experiment_plan(root, experiment_plan_ref)
    artifact = _ref(root, artifact_root, field="artifact_root", allow_missing=True)
    if artifact.exists() and not artifact.is_dir():
        raise S25ExecutionBlocked("S205_ARTIFACT_ROOT_NOT_DIRECTORY")
    from .stage2_formal import FormalExperimentPlan

    parsed = FormalExperimentPlan.from_mapping(experiment)
    if parsed.sampling_plan_hash != sampling.digest:
        raise S25ExecutionBlocked("S205_EXPERIMENT_SAMPLING_HASH_MISMATCH")
    return {
        "schema_version": "stage2-s205-formal-preflight-v1",
        "status": "READY",
        "formal_eligible": True,
        "rebind_ref": rebind["_rebind_ref"],
        "rebind_hash": rebind["artifact_hash"],
        "g23_evaluation_ref": rebind["_g23_ref"],
        "g23_evaluation_hash": rebind["g23_evaluation_hash"],
        "execution_commit": rebind["execution_commit"],
        "sampling_plan_ref": str(sampling_ref),
        "sampling_plan_hash": sampling.digest,
        "experiment_plan_ref": str(experiment_plan_ref),
        "experiment_plan_hash": canonical_json_hash(experiment),
        "artifact_root": str(artifact_root),
        "cell_count": 6,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_pci": EXCLUDED_PCI,
        "confirmatory_draws_generated": False,
        "optional_stopping": False,
        "silent_skip": False,
        "max_attempts": 1,
    }


# Keep the launcher-facing ``s205`` spelling available alongside the concise
# module-level ``s25`` API; both names refer to the same immutable contract.
load_s205_rebind_plan = load_s25_rebind_plan
load_s205_experiment_plan = load_s25_experiment_plan
preflight_s205 = preflight_s25
S205FormalRunner = S25FormalRunner


__all__ = [
    "APPROVED_GPU_UUIDS", "EXCLUDED_PCI", "EXPECTED_CELL_IDS", "S25ExecutionBlocked",
    "S25FormalRunner", "S25_GATE_SCHEMA", "load_s25_experiment_plan", "load_s25_rebind_plan",
    "preflight_s25", "S205FormalRunner", "load_s205_experiment_plan", "load_s205_rebind_plan", "preflight_s205",
]
