"""Fail-closed consumer for the evaluator-corrected G2.3 delta sidecar.

S2.4's bounded producer stores sizing diagnostics at the reference sizing
sample counts.  S2.6 must consume the evaluator amendment, which is the only
artifact allowed to expose the four preregistered pilot batch sizes.  This
module deliberately has no fallback path to the producer's old table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re
from typing import Mapping

from ..contracts.jsonio import canonical_json_hash, load_canonical_json


CORRECTED_DELTA_SCHEMA_VERSION = "stage2-g23-corrected-delta-sci-v1"
SIZING_SAMPLE_COUNTS = (131072, 262144)
PILOT_BATCH_SIZES = (32, 64, 128, 256)
ENDPOINTS = ("model_total", "layer", "module")
CORRECTED_DELTA_SOURCE = "g23_output_derived_corrected_sidecar"
SOURCE_TABLE_MODES = ("sizing_nodes_legacy", "candidate_batch_sizes")
SOURCE_SCHEMA_VERSION = "stage2-reference-delta-sci-v2"
FORMULA_VERSION = "stage2-reference-sizing-margin-v1"
FORMULA = "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B"
G23_CELL_IDS = tuple(
    f"{model}:{stage}"
    for model in ("pythia-14m", "pythia-31m-deduped")
    for stage in ("initialization", "early", "mid_late")
)
SIDECAR_FIELDS = frozenset(
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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class CorrectedDeltaRejected(ValueError):
    """Raised when a G2.3 corrected-delta binding cannot be proven."""


def _fail(reason: str) -> None:
    raise CorrectedDeltaRejected(reason)


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field}:SHA256_REQUIRED")
    return value


def _commit(value: object, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        _fail(f"{field}:COMMIT_REQUIRED")
    return value


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{field}:LOGICAL_POSIX_PATH_REQUIRED")
    if value.startswith("/") or value.endswith("/") or "//" in value:
        _fail(f"{field}:PATH_ESCAPE")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        _fail(f"{field}:PATH_ESCAPE")
    current = root.resolve()
    try:
        if current.is_symlink():
            _fail(f"{field}:SYMLINK_FORBIDDEN")
        for part in parsed.parts:
            current = current / part
            if current.is_symlink():
                _fail(f"{field}:SYMLINK_FORBIDDEN")
    except OSError as error:
        raise CorrectedDeltaRejected(f"{field}:UNREADABLE") from error
    resolved = root.resolve() / Path(*parsed.parts)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise CorrectedDeltaRejected(f"{field}:PATH_ESCAPE") from error
    return resolved


def _finite(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field}:FINITE_NUMBER_REQUIRED")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        _fail(f"{field}:INVALID_NUMBER")
    return number


def _strict_list(value: object, expected: tuple[int, ...], field: str) -> None:
    if not isinstance(value, list) or tuple(value) != expected or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        _fail(f"{field}:DOMAIN_INVALID")


def _validate_nodes(sidecar: Mapping[str, object]) -> None:
    raw = sidecar.get("sizing_nodes")
    if not isinstance(raw, list) or len(raw) != len(SIZING_SAMPLE_COUNTS):
        _fail("sizing_nodes:COUNT_INVALID")
    observed: list[int] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {
            "sample_count",
            "state_digest",
            "shard_refs_hash",
            "mean_hash",
            "sequence_variance_hash",
        }:
            _fail(f"sizing_nodes[{index}]:FIELDS_INVALID")
        count = item.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int):
            _fail(f"sizing_nodes[{index}].sample_count:INTEGER_REQUIRED")
        observed.append(count)
        for field in ("state_digest", "shard_refs_hash", "mean_hash", "sequence_variance_hash"):
            _sha(item.get(field), f"sizing_nodes[{index}].{field}")
    if tuple(observed) != SIZING_SAMPLE_COUNTS:
        _fail("sizing_nodes:SAMPLE_COUNT_ORDER_INVALID")


def _validate_table(value: object, field: str) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != set(ENDPOINTS):
        _fail(f"{field}:ENDPOINT_DOMAIN_INVALID")
    result: dict[str, dict[str, float]] = {}
    expected_keys = {str(item) for item in PILOT_BATCH_SIZES}
    for endpoint in ENDPOINTS:
        raw = value.get(endpoint)
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            _fail(f"{field}.{endpoint}:B_DOMAIN_INVALID")
        result[endpoint] = {
            key: _finite(raw[key], f"{field}.{endpoint}.{key}", positive=True)
            for key in sorted(raw)
        }
    return result


def _validate_sidecar_structure(sidecar: Mapping[str, object]) -> dict[str, dict[str, float]]:
    if set(sidecar) != SIDECAR_FIELDS:
        _fail("corrected_delta_sci:UNKNOWN_OR_MISSING_FIELDS")
    if sidecar.get("schema_version") != CORRECTED_DELTA_SCHEMA_VERSION:
        _fail("corrected_delta_sci:SCHEMA_INVALID")
    artifact_hash = _sha(sidecar.get("artifact_hash"), "corrected_delta_sci.artifact_hash")
    if canonical_json_hash({key: value for key, value in sidecar.items() if key != "artifact_hash"}) != artifact_hash:
        _fail("corrected_delta_sci:HASH_MISMATCH")
    if sidecar.get("source_producer_schema_version") != SOURCE_SCHEMA_VERSION:
        _fail("corrected_delta_sci.source_producer_schema_version:INVALID")
    _commit(sidecar.get("source_producer_commit"), "source_producer_commit")
    _commit(sidecar.get("evaluator_commit"), "evaluator_commit")
    for field in (
        "source_producer_artifact_hash",
        "evaluator_source_sha256",
        "formula_contract_hash",
        "sizing_result_hash",
        "sizing_plan_hash",
        "registry_hash",
    ):
        _sha(sidecar.get(field), field)
    source_ref = sidecar.get("source_producer_ref")
    if not isinstance(source_ref, str):
        _fail("source_producer_ref:PATH_REQUIRED")
    if sidecar.get("source_producer_table_mode") not in SOURCE_TABLE_MODES:
        _fail("source_producer_table_mode:DOMAIN_INVALID")
    if sidecar.get("formula_version") != FORMULA_VERSION or sidecar.get("formula") != FORMULA:
        _fail("corrected_delta_sci:FORMULA_CONTRACT_INVALID")
    floors = sidecar.get("absolute_floors")
    if not isinstance(floors, Mapping) or set(floors) != {"tau_model", "tau_layer", "tau_module", "tau_coord", "tau_nmse"}:
        _fail("absolute_floors:FIELDS_INVALID")
    for name, value in floors.items():
        _finite(value, f"absolute_floors.{name}", positive=True)
    reference_id = sidecar.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id:
        _fail("reference_id:REQUIRED")
    _strict_list(sidecar.get("candidate_sample_counts"), SIZING_SAMPLE_COUNTS, "candidate_sample_counts")
    _strict_list(sidecar.get("delta_sci_batch_sizes"), PILOT_BATCH_SIZES, "delta_sci_batch_sizes")
    selected = sidecar.get("selected_sample_count_per_stream")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected not in SIZING_SAMPLE_COUNTS:
        _fail("selected_sample_count_per_stream:DOMAIN_INVALID")
    _validate_nodes(sidecar)
    mode = sidecar.get("source_producer_table_mode")
    expected_reason = (
        "producer_delta_sci_keyed_by_sizing_nodes; S2.6_requires_candidate_batch_sizes"
        if mode == "sizing_nodes_legacy"
        else "producer_delta_sci_candidate_batch_sizes_verified_by_evaluator"
    )
    if sidecar.get("correction_reason") != expected_reason:
        _fail("correction_reason:DOMAIN_INVALID")
    delta = _validate_table(sidecar.get("delta_sci_by_endpoint"), "delta_sci_by_endpoint")
    signal = _validate_table(sidecar.get("signal_scale_by_endpoint"), "signal_scale_by_endpoint")
    noise = _validate_table(sidecar.get("noise_scale_by_endpoint"), "noise_scale_by_endpoint")
    for endpoint in ENDPOINTS:
        for batch_size in PILOT_BATCH_SIZES:
            key = str(batch_size)
            expected = max(0.10 * noise[endpoint][key], 0.01 * signal[endpoint][key])
            if delta[endpoint][key] != expected:
                _fail(f"delta_sci_by_endpoint.{endpoint}.{key}:FORMULA_MISMATCH")
        if len({signal[endpoint][str(size)] for size in PILOT_BATCH_SIZES}) != 1:
            _fail(f"signal_scale_by_endpoint.{endpoint}:B_INVARIANT_INVALID")
        base_product = noise[endpoint]["32"] * 32.0
        if any(noise[endpoint][str(size)] * float(size) != base_product for size in PILOT_BATCH_SIZES):
            _fail(f"noise_scale_by_endpoint.{endpoint}:B_INVARIANT_INVALID")
    return delta


def _validate_source_provenance(root: Path, sidecar: Mapping[str, object]) -> None:
    source_path = _logical_path(root, sidecar.get("source_producer_ref"), field="source_producer_ref")
    try:
        source = load_canonical_json(source_path)
    except (OSError, TypeError, ValueError) as error:
        raise CorrectedDeltaRejected("source_producer_ref:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(source, Mapping):
        _fail("source_producer_ref:OBJECT_REQUIRED")
    source = dict(source)
    source_hash = _sha(source.get("artifact_hash"), "source_producer.artifact_hash")
    if source_hash != sidecar.get("source_producer_artifact_hash"):
        _fail("source_producer_artifact_hash:MISMATCH")
    if canonical_json_hash({key: value for key, value in source.items() if key != "artifact_hash"}) != source_hash:
        _fail("source_producer_artifact_hash:CONTENT_MISMATCH")
    # These are the source-side identities which the evaluator copied into its
    # amendment.  The source's old table itself is never returned to callers.
    for field in (
        "formula_contract_hash",
        "formula_version",
        "formula",
        "absolute_floors",
        "reference_id",
        "sizing_result_hash",
        "sizing_plan_hash",
        "registry_hash",
        "candidate_sample_counts",
        "sizing_nodes",
    ):
        if source.get(field) != sidecar.get(field):
            _fail(f"source_producer.{field}:IDENTITY_MISMATCH")
    if source.get("schema_version") != sidecar.get("source_producer_schema_version"):
        _fail("source_producer.schema_version:IDENTITY_MISMATCH")
    if sidecar.get("source_producer_table_mode") == "candidate_batch_sizes" and source.get("selected_sample_count_per_stream") != sidecar.get("selected_sample_count_per_stream"):
        _fail("source_producer.selected_sample_count_per_stream:IDENTITY_MISMATCH")
    source_mode = sidecar.get("source_producer_table_mode")
    for table_name in ("delta_sci_by_endpoint", "signal_scale_by_endpoint", "noise_scale_by_endpoint"):
        table = source.get(table_name)
        if not isinstance(table, Mapping) or set(table) != set(ENDPOINTS):
            _fail(f"source_producer.{table_name}:ENDPOINT_DOMAIN_INVALID")
        expected = {str(value) for value in (SIZING_SAMPLE_COUNTS if source_mode == "sizing_nodes_legacy" else PILOT_BATCH_SIZES)}
        for endpoint in ENDPOINTS:
            values = table.get(endpoint)
            if not isinstance(values, Mapping) or set(values) != expected:
                _fail(f"source_producer.{table_name}.{endpoint}:KEY_DOMAIN_INVALID")


@dataclass(frozen=True, slots=True)
class CorrectedDeltaBinding:
    """The only delta table representation admitted by S2.6."""

    artifact_hash: str
    ref: str
    cell_id: str
    config_hash: str
    result_hash: str
    batch_sizes: tuple[int, ...]
    source: str
    delta_sci_by_endpoint: Mapping[str, Mapping[str, float]]

    def delta_for(self, batch_size: int) -> dict[str, float]:
        if batch_size not in PILOT_BATCH_SIZES:
            raise CorrectedDeltaRejected("batch_size:NOT_PREREGISTERED")
        key = str(batch_size)
        return {
            "bias": float(self.delta_sci_by_endpoint["model_total"][key]),
            "nmse": float(self.delta_sci_by_endpoint["layer"][key]),
            "rank": float(self.delta_sci_by_endpoint["module"][key]),
        }


def load_bound_corrected_delta(
    data_root: str | Path,
    *,
    g23_evaluation_ref: str,
    cell_id: str,
    expected_config_hash: str,
    expected_result_hash: str,
    expected_sizing_plan_hash: str | None = None,
    expected_sizing_result_hash: str | None = None,
    expected_reference_id: str | None = None,
    expected_registry_hash: str | None = None,
) -> CorrectedDeltaBinding:
    """Load one G23 PASS cell and its exact, hash-bound corrected sidecar."""

    root = Path(data_root).resolve()
    g23_cell_id = cell_id if cell_id in G23_CELL_IDS else cell_id.replace(".", ":", 1)
    if g23_cell_id not in G23_CELL_IDS:
        _fail("cell_id:NOT_G23_CELL")
    evaluation_path = _logical_path(root, g23_evaluation_ref, field="g23_evaluation_ref")
    try:
        evaluation = load_canonical_json(evaluation_path)
    except (OSError, TypeError, ValueError) as error:
        raise CorrectedDeltaRejected("g23_evaluation:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(evaluation, Mapping):
        _fail("g23_evaluation:OBJECT_REQUIRED")
    evaluation = dict(evaluation)
    if evaluation.get("schema_version") != "stage2-g23-reference-evaluation-v1" or evaluation.get("status") != "PASS" or evaluation.get("formal_eligible") is not True:
        _fail("g23_evaluation:PASS_REQUIRED")
    if evaluation.get("artifact_hash") != canonical_json_hash({key: value for key, value in evaluation.items() if key != "artifact_hash"}):
        _fail("g23_evaluation:HASH_MISMATCH")
    calculator = evaluation.get("calculator")
    if not isinstance(calculator, Mapping):
        _fail("g23_evaluation.calculator:REQUIRED")
    _commit(calculator.get("evaluator_commit"), "calculator.evaluator_commit")
    _sha(calculator.get("source_sha256"), "calculator.source_sha256")
    if calculator.get("source_schema") != "stage2-g23-reference-evaluation-v1":
        _fail("calculator.source_schema:INVALID")
    if calculator.get("producer_commit") is not None:
        _commit(calculator.get("producer_commit"), "calculator.producer_commit")
    cells = evaluation.get("cells")
    if (
        not isinstance(cells, list)
        or len(cells) != len(G23_CELL_IDS)
        or evaluation.get("required_cell_count") != len(G23_CELL_IDS)
        or evaluation.get("complete_cell_count") != len(G23_CELL_IDS)
        or evaluation.get("expected_cell_ids") != list(G23_CELL_IDS)
    ):
        _fail("g23_evaluation.cells:SIX_PASS_CELLS_REQUIRED")
    target: Mapping[str, object] | None = None
    observed_cell_ids: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_id"), str):
            _fail("g23_evaluation.cells:CELL_INVALID")
        observed_cell_ids.append(str(cell["cell_id"]))
        if cell.get("status") != "PASS" or cell.get("formal_eligible") not in (None, True):
            _fail("g23_evaluation.cells:PASS_REQUIRED")
        cell_identities = cell.get("identities")
        cell_metrics = cell.get("metrics")
        if not isinstance(cell_identities, Mapping) or not isinstance(cell_metrics, Mapping):
            _fail("g23_evaluation.cells:IDENTITIES_AND_METRICS_REQUIRED")
        if _sha(cell_identities.get("corrected_delta_sci_hash"), "identities.corrected_delta_sci_hash") != cell_metrics.get("corrected_delta_sci_hash"):
            _fail("g23_evaluation.cells:CORRECTED_HASH_BINDING_INVALID")
        if cell_identities.get("corrected_delta_sci_ref") != cell_metrics.get("corrected_delta_sci_ref"):
            _fail("g23_evaluation.cells:CORRECTED_REF_BINDING_INVALID")
        if cell_metrics.get("corrected_delta_sci_batch_sizes") != list(PILOT_BATCH_SIZES) or cell_metrics.get("delta_sci_source") != CORRECTED_DELTA_SOURCE:
            _fail("g23_evaluation.cells:CORRECTED_SOURCE_BINDING_INVALID")
        if cell.get("cell_id") == g23_cell_id:
            target = cell
    if tuple(observed_cell_ids) != G23_CELL_IDS:
        _fail("g23_evaluation.cells:CELL_SET_INVALID")
    if target is None:
        _fail("g23_evaluation.cells:TARGET_CELL_MISSING")
    identities = target.get("identities")
    metrics = target.get("metrics")
    if not isinstance(identities, Mapping) or not isinstance(metrics, Mapping):
        _fail("g23_evaluation.cell:IDENTITIES_AND_METRICS_REQUIRED")
    identity = dict(identities)
    metric = dict(metrics)
    if identity.get("cell_id") != g23_cell_id or identity.get("config_hash") != expected_config_hash or identity.get("result_hash") != expected_result_hash:
        _fail("g23_evaluation.cell:IDENTITY_MISMATCH")
    for field in ("config_hash", "result_hash", "sizing_plan_hash", "sizing_result_hash", "registry_hash", "reference_id", "producer_commit"):
        if field in identity and field.endswith("hash"):
            _sha(identity[field], f"identities.{field}")
    _commit(identity.get("producer_commit"), "identities.producer_commit")
    if calculator.get("producer_commit") is not None and calculator.get("producer_commit") != identity.get("producer_commit"):
        _fail("calculator.producer_commit:CELL_IDENTITY_MISMATCH")
    if expected_sizing_plan_hash is not None and identity.get("sizing_plan_hash") != expected_sizing_plan_hash:
        _fail("identities.sizing_plan_hash:MISMATCH")
    if expected_sizing_result_hash is not None and identity.get("sizing_result_hash") != expected_sizing_result_hash:
        _fail("identities.sizing_result_hash:MISMATCH")
    if expected_reference_id is not None and identity.get("reference_id") != expected_reference_id:
        _fail("identities.reference_id:MISMATCH")
    if expected_registry_hash is not None and identity.get("registry_hash") != expected_registry_hash:
        _fail("identities.registry_hash:MISMATCH")
    sidecar_hash = _sha(identity.get("corrected_delta_sci_hash"), "identities.corrected_delta_sci_hash")
    sidecar_ref = identity.get("corrected_delta_sci_ref")
    if not isinstance(sidecar_ref, str):
        _fail("identities.corrected_delta_sci_ref:REQUIRED")
    if metric.get("corrected_delta_sci_hash") != sidecar_hash or metric.get("corrected_delta_sci_ref") != sidecar_ref or metric.get("corrected_delta_sci_batch_sizes") != list(PILOT_BATCH_SIZES) or metric.get("delta_sci_source") != CORRECTED_DELTA_SOURCE:
        _fail("metrics:CORRECTED_DELTA_BINDING_MISMATCH")
    sidecar_path = _logical_path(root, sidecar_ref, field="corrected_delta_sci_ref")
    if sidecar_path.parent.name != "g2.3-corrected-delta-sci" or sidecar_path.name != f"{sidecar_hash}.json":
        _fail("corrected_delta_sci_ref:CONTENT_ADDRESS_INVALID")
    # G23 evaluates into <output_root>/g2.3-attempts/<hash>/evaluation.json;
    # the sidecar must be in that same output_root, not merely somewhere under
    # DATA_ROOT with a matching basename.
    expected_output_root = evaluation_path.parent.parent.parent
    if sidecar_path.parent.parent != expected_output_root:
        _fail("corrected_delta_sci_ref:EVALUATOR_OUTPUT_ROOT_MISMATCH")
    try:
        sidecar = load_canonical_json(sidecar_path)
    except (OSError, TypeError, ValueError) as error:
        raise CorrectedDeltaRejected("corrected_delta_sci_ref:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(sidecar, Mapping):
        _fail("corrected_delta_sci:OBJECT_REQUIRED")
    sidecar = dict(sidecar)
    if sidecar.get("artifact_hash") != sidecar_hash:
        _fail("corrected_delta_sci:BOUND_HASH_MISMATCH")
    delta = _validate_sidecar_structure(sidecar)
    _validate_source_provenance(root, sidecar)
    if sidecar.get("source_producer_commit") != identity.get("producer_commit"):
        _fail("corrected_delta_sci.source_producer_commit:CELL_IDENTITY_MISMATCH")
    if sidecar.get("evaluator_commit") != calculator.get("evaluator_commit") or sidecar.get("evaluator_source_sha256") != calculator.get("source_sha256"):
        _fail("corrected_delta_sci:EVALUATOR_BINDING_MISMATCH")
    for field in ("reference_id", "sizing_result_hash", "sizing_plan_hash", "registry_hash"):
        if sidecar.get(field) != identity.get(field):
            _fail(f"corrected_delta_sci.{field}:CELL_IDENTITY_MISMATCH")
    return CorrectedDeltaBinding(
        artifact_hash=sidecar_hash,
        ref=sidecar_ref,
        cell_id=g23_cell_id,
        config_hash=expected_config_hash,
        result_hash=expected_result_hash,
        batch_sizes=PILOT_BATCH_SIZES,
        source=CORRECTED_DELTA_SOURCE,
        delta_sci_by_endpoint=delta,
    )
