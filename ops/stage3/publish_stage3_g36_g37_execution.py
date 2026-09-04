#!/usr/bin/env python3
"""Append the independently published G3-6 and G3-7 Gates to Stage 3 evidence.

Stage3.09 deliberately evaluates against the frozen G3-0-through-G3-5
execution evidence while carrying G3-6 as an explicit authority.  Only after
the independent G3-7 publication exists may both Gates be appended, in order,
to create the execution evidence consumed by Stage3.10.  This command derives
the Gate refs and producer config identities from the two publication commits;
callers cannot relabel or reorder them.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
import sys
from typing import Any

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[2]
    for _candidate in (_repository_root, _repository_root / "src"):
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))

from ops.stage3.run_stage3_formal import (
    GateAuthorityPublisher,
    Stage3OrchestratorError,
)
from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.jsonio import (
    canonical_json_bytes,
    canonical_json_hash,
)
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.experiments.stage3_g36_publisher import (
    STAGE3_G36_TASK_ID,
    Stage3G36Publication,
)
from param_importance_nlp.experiments.stage3_g37_publisher import (
    STAGE3_G37_TASK_ID,
    Stage3G37Publication,
)
from param_importance_nlp.experiments.stage3_gate import REQUIRED_STAGE3_GATE_IDS
from param_importance_nlp.runtime import (
    load_committed_task_artifact,
    publish_canonical_immutable,
)


SCHEMA_VERSION = "stage3-g36-g37-execution-publication-v1"
_HASH_LENGTH = 64


class Stage3GateExecutionPublicationError(ValueError):
    """Raised when the two-Gate execution chain is not safe to publish."""


def _fail(
    code: str, detail: object | None = None
) -> Stage3GateExecutionPublicationError:
    return Stage3GateExecutionPublicationError(
        code if detail is None else f"{code}:{detail}"
    )


def _hash(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail("STAGE3_GATE_EXECUTION_HASH_INVALID", field)
    return value


def _ref(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        raise _fail("STAGE3_GATE_EXECUTION_REF_INVALID", field)
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise _fail("STAGE3_GATE_EXECUTION_REF_INVALID", field)
    return logical.as_posix()


def _authority_dir(value: object, *, field: str) -> str:
    ref = _ref(value, field=field)
    if PurePosixPath(ref).parts[0] != "evidence":
        raise _fail("STAGE3_GATE_EXECUTION_AUTHORITY_DIR_INVALID", field)
    return ref


def _receipt_path(root: Path, value: Path | None) -> tuple[Path | None, str | None]:
    if value is None:
        return None, None
    candidate = value if value.is_absolute() else root / value
    absolute = candidate.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise _fail("STAGE3_GATE_EXECUTION_RECEIPT_OUTSIDE_ROOT") from error
    logical = PurePosixPath(*relative.parts)
    if not logical.parts or logical.parts[0] != "results" or logical.suffix != ".json":
        raise _fail("STAGE3_GATE_EXECUTION_RECEIPT_REF_INVALID")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise _fail("STAGE3_GATE_EXECUTION_RECEIPT_SYMLINK")
    return absolute, logical.as_posix()


def _load_formal(root: Path, reference: str, *, field: str, kind: str):
    ref = _ref(reference, field=field)
    try:
        loaded = load_committed_task_artifact(root, ref, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("STAGE3_GATE_EXECUTION_FORMAL_COMMIT_INVALID", field) from error
    if loaded.identity.artifact_kind != kind:
        raise _fail("STAGE3_GATE_EXECUTION_ARTIFACT_KIND_INVALID", field)
    return loaded


def _gate_ids(execution: FormalExecutionEvidence) -> tuple[str, ...]:
    return tuple(gate.gate_id for gate in execution.prerequisite_gates)


def _validate_live_gate(gate: GateRecord, *, gate_id: str) -> None:
    if (
        gate.gate_id != gate_id
        or gate.stage != 3
        or gate.status is not GateStatus.PASS
        or gate.effective_status() is not GateStatus.PASS
        or not gate.evidence_refs
    ):
        raise _fail("STAGE3_GATE_EXECUTION_GATE_NOT_LIVE_PASS", gate_id)


def _validate_execution(
    execution: FormalExecutionEvidence,
    *,
    expected: tuple[GateRecord, ...],
    field: str,
) -> None:
    execution.require_for_stage(3)
    if _gate_ids(execution) != tuple(gate.gate_id for gate in expected):
        raise _fail("STAGE3_GATE_EXECUTION_GATE_ORDER_INVALID", field)
    if tuple(gate.artifact_hash for gate in execution.prerequisite_gates) != tuple(
        gate.artifact_hash for gate in expected
    ):
        raise _fail("STAGE3_GATE_EXECUTION_GATE_HASH_INVALID", field)


def _publication_inputs(
    root: Path,
    *,
    base_execution_ref: str,
    g36_publication_ref: str,
    g37_publication_ref: str,
) -> tuple[
    Any,
    FormalExecutionEvidence,
    Stage3G36Publication,
    Stage3G37Publication,
    GateRecord,
    GateRecord,
]:
    base_loaded = _load_formal(
        root,
        base_execution_ref,
        field="base_execution_ref",
        kind="formal_execution_evidence",
    )
    g36_loaded = _load_formal(
        root,
        g36_publication_ref,
        field="g36_publication_ref",
        kind="g36_publication",
    )
    g37_loaded = _load_formal(
        root,
        g37_publication_ref,
        field="g37_publication_ref",
        kind="g37_publication",
    )
    try:
        base = FormalExecutionEvidence.from_mapping(dict(base_loaded.payload))
        g36 = Stage3G36Publication.from_mapping(dict(g36_loaded.payload))
        g37 = Stage3G37Publication.from_mapping(dict(g37_loaded.payload))
    except (TypeError, ValueError, FormalRunRejected) as error:
        raise _fail("STAGE3_GATE_EXECUTION_PUBLICATION_PAYLOAD_INVALID") from error

    base_gates = tuple(base.prerequisite_gates)
    _validate_execution(base, expected=base_gates, field="base")
    if _gate_ids(base) != tuple(REQUIRED_STAGE3_GATE_IDS):
        raise _fail("STAGE3_GATE_EXECUTION_BASE_GATE_SET_INVALID")
    for gate, gate_id in zip(base_gates, REQUIRED_STAGE3_GATE_IDS, strict=True):
        _validate_live_gate(gate, gate_id=gate_id)

    if (
        g36.task_id != STAGE3_G36_TASK_ID
        or g36.status != "PASS"
        or g36.formal_eligible is not True
        or g36.execution_evidence_ref != base_execution_ref
        or g36.execution_evidence_hash != base.artifact_hash
        or g36_loaded.identity.config_hash != g36.config_hash
    ):
        raise _fail("STAGE3_GATE_EXECUTION_G36_PUBLICATION_INVALID")
    if (
        g37.task_id != STAGE3_G37_TASK_ID
        or g37.status != "PASS"
        or g37.formal_eligible is not True
        or g37.execution_evidence_ref != base_execution_ref
        or g37.execution_evidence_hash != base.artifact_hash
        or g37.g3_6_ref != g36.g3_6_ref
        or g37.g3_6_hash != g36.g3_6_hash
        or g37.frozen_source_table_ref != g36.frozen_source_table_ref
        or g37.formal_plan_ref != g36.formal_plan_ref
        or g37.provenance_ref != g36.provenance_ref
        or g37.evaluation_ref != g36.evaluation_ref
        or g37.recommendation_ref is None
        or g37.finalization_ref is None
        or g37_loaded.identity.config_hash != g37.publication_config_hash
    ):
        raise _fail("STAGE3_GATE_EXECUTION_G37_PUBLICATION_INVALID")
    s309_config_hash = g37.input_config_hashes.get("cost_accuracy_table")
    if (
        s309_config_hash != g37.input_config_hashes.get("quadrature_decision")
        or not isinstance(s309_config_hash, str)
    ):
        raise _fail("STAGE3_GATE_EXECUTION_S309_CONFIG_IDENTITY_INVALID")
    _hash(s309_config_hash, field="s309_config_hash")

    g36_gate_loaded = _load_formal(
        root, g36.g3_6_ref, field="g3_6_ref", kind="gate_record"
    )
    g37_gate_loaded = _load_formal(
        root, g37.g3_7_ref, field="g3_7_ref", kind="gate_record"
    )
    try:
        g36_gate = GateRecord.from_mapping(dict(g36_gate_loaded.payload))
        g37_gate = GateRecord.from_mapping(dict(g37_gate_loaded.payload))
    except (TypeError, ValueError) as error:
        raise _fail("STAGE3_GATE_EXECUTION_GATE_PAYLOAD_INVALID") from error
    _validate_live_gate(g36_gate, gate_id="stage3.G3-6")
    _validate_live_gate(g37_gate, gate_id="stage3.G3-7")
    if (
        g36_gate.artifact_hash != g36.g3_6_hash
        or g36_gate.to_dict() != g36.g3_6_gate.to_dict()
        or g37_gate.artifact_hash != g37.g3_7_hash
        or g37_gate.to_dict() != g37.g3_7_gate.to_dict()
    ):
        raise _fail("STAGE3_GATE_EXECUTION_GATE_PUBLICATION_MISMATCH")
    return base_loaded, base, g36, g37, g36_gate, g37_gate


def publish_execution_chain(arguments: argparse.Namespace) -> Mapping[str, object]:
    root = arguments.data_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise _fail("STAGE3_GATE_EXECUTION_ROOT_INVALID")
    base_ref = _ref(arguments.base_execution_ref, field="base_execution_ref")
    g36_publication_ref = _ref(
        arguments.g36_publication_ref, field="g36_publication_ref"
    )
    g37_publication_ref = _ref(
        arguments.g37_publication_ref, field="g37_publication_ref"
    )
    g36_output_dir = _authority_dir(
        arguments.g36_execution_output_dir, field="g36_execution_output_dir"
    )
    g37_output_dir = _authority_dir(
        arguments.g37_execution_output_dir, field="g37_execution_output_dir"
    )
    if g36_output_dir == g37_output_dir:
        raise _fail("STAGE3_GATE_EXECUTION_OUTPUT_DIR_COLLISION")
    receipt_path, receipt_ref = _receipt_path(root, arguments.receipt)

    base_loaded, base, g36, g37, g36_gate, g37_gate = _publication_inputs(
        root,
        base_execution_ref=base_ref,
        g36_publication_ref=g36_publication_ref,
        g37_publication_ref=g37_publication_ref,
    )
    s308_config_hash = _hash(g36.config_hash, field="s308_config_hash")
    s309_config_hash = _hash(
        g37.input_config_hashes["cost_accuracy_table"], field="s309_config_hash"
    )
    publisher = GateAuthorityPublisher(root)
    intermediate_ref, appended_g36 = publisher.publish_external(
        output_dir=g36_output_dir,
        config_hash=s308_config_hash,
        gate_ref=g36.g3_6_ref,
        previous_evidence_ref=base_ref,
        gate_id="stage3.G3-6",
    )
    final_ref, appended_g37 = publisher.publish_external(
        output_dir=g37_output_dir,
        config_hash=s309_config_hash,
        gate_ref=g37.g3_7_ref,
        previous_evidence_ref=intermediate_ref,
        gate_id="stage3.G3-7",
    )
    if (
        appended_g36.artifact_hash != g36_gate.artifact_hash
        or appended_g37.artifact_hash != g37_gate.artifact_hash
    ):
        raise _fail("STAGE3_GATE_EXECUTION_APPENDED_GATE_DRIFT")

    intermediate_loaded = _load_formal(
        root,
        intermediate_ref,
        field="intermediate_execution_ref",
        kind="formal_execution_evidence",
    )
    final_loaded = _load_formal(
        root,
        final_ref,
        field="final_execution_ref",
        kind="formal_execution_evidence",
    )
    intermediate = FormalExecutionEvidence.from_mapping(
        dict(intermediate_loaded.payload)
    )
    final = FormalExecutionEvidence.from_mapping(dict(final_loaded.payload))
    _validate_execution(
        intermediate,
        expected=(*base.prerequisite_gates, g36_gate),
        field="intermediate",
    )
    _validate_execution(
        final,
        expected=(*base.prerequisite_gates, g36_gate, g37_gate),
        field="final",
    )
    if (
        intermediate_loaded.identity.config_hash != s308_config_hash
        or final_loaded.identity.config_hash != s309_config_hash
        or tuple(intermediate_loaded.source_refs) != (base_ref, g36.g3_6_ref)
        or tuple(final_loaded.source_refs) != (intermediate_ref, g37.g3_7_ref)
        or intermediate.contract_freeze_hash != base.contract_freeze_hash
        or final.contract_freeze_hash != base.contract_freeze_hash
        or intermediate.asset_manifest_hashes != base.asset_manifest_hashes
        or final.asset_manifest_hashes != base.asset_manifest_hashes
        or dict(intermediate.metadata) != dict(base.metadata)
        or dict(final.metadata) != dict(base.metadata)
    ):
        raise _fail("STAGE3_GATE_EXECUTION_FINAL_CHAIN_INVALID")

    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scope": "formal",
        "formal_eligible": True,
        "base_execution_ref": base_ref,
        "base_execution_hash": base.artifact_hash,
        "base_execution_envelope_hash": base_loaded.identity.artifact_hash,
        "g36_publication_ref": g36_publication_ref,
        "g36_publication_hash": g36.artifact_hash,
        "g3_6_ref": g36.g3_6_ref,
        "g3_6_hash": g36_gate.artifact_hash,
        "g37_publication_ref": g37_publication_ref,
        "g37_publication_hash": g37.artifact_hash,
        "g3_7_ref": g37.g3_7_ref,
        "g3_7_hash": g37_gate.artifact_hash,
        "s308_config_hash": s308_config_hash,
        "s309_config_hash": s309_config_hash,
        "intermediate_execution_ref": intermediate_ref,
        "intermediate_execution_hash": intermediate.artifact_hash,
        "intermediate_execution_envelope_hash": intermediate_loaded.identity.artifact_hash,
        "final_execution_ref": final_ref,
        "final_execution_hash": final.artifact_hash,
        "final_execution_envelope_hash": final_loaded.identity.artifact_hash,
        "gate_ids": list(_gate_ids(final)),
    }
    if receipt_ref is not None:
        receipt["receipt_ref"] = receipt_ref
    receipt["receipt_hash"] = canonical_json_hash(receipt)
    if receipt_path is not None:
        publish_canonical_immutable(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-execution-ref", required=True)
    parser.add_argument("--g36-publication-ref", required=True)
    parser.add_argument("--g37-publication-ref", required=True)
    parser.add_argument("--g36-execution-output-dir", required=True)
    parser.add_argument("--g37-execution-output-dir", required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt = publish_execution_chain(_parser().parse_args(argv))
    except (
        OSError,
        TypeError,
        ValueError,
        FormalRunRejected,
        Stage3OrchestratorError,
    ) as error:
        blocked: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "formal_eligible": False,
            "reason": f"{type(error).__name__}:{error}",
        }
        blocked["receipt_hash"] = canonical_json_hash(blocked)
        print(canonical_json_bytes(blocked).decode("utf-8"), end="")
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "Stage3GateExecutionPublicationError",
    "main",
    "publish_execution_chain",
]
