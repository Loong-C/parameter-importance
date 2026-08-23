"""Publish the formal S2.02 contract freeze and amend S202 execution evidence.

The old S202 ``contract-freeze-s202.json`` is an authorization note, not a
``contract-freeze-v1`` payload.  This small producer turns explicitly supplied
source files into a real formal :class:`ContractFreeze` TaskArtifact and then
creates a new, append-only ``FormalExecutionEvidence`` document bound to that
artifact.  Hashes are always derived here; callers cannot supply a hash or a
status to this producer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping

from param_importance_nlp.cli import _load_mapping
from param_importance_nlp.contracts import (
    ContractFreeze,
    ContractState,
    FormalExecutionEvidence,
    canonical_json_bytes,
    canonical_json_hash,
    load_canonical_json,
    loads_strict_json,
    write_canonical_json,
)
from param_importance_nlp.runtime import TaskArtifactStore, load_committed_task_artifact
from param_importance_nlp.atomic import atomic_write_bytes


STAGE = 2
TASK_ID = "stage2_contract_freeze_s202"
ARTIFACT_KIND = "contract_freeze"
DEFAULT_OUTPUT_DIR = "evidence/stage2/s202-formal-contract"
DEFAULT_AMENDED_EVIDENCE = (
    "evidence/stage2/s202-formal-auth/formal-execution-s202-amendment-r1.json"
)
FORMULA_VERSION = "stage2-estimator-contract-v1"

# This is the Stage 2 gate family from plan/stage2/README.md.  It is a frozen
# requirement list, not a claim that all of these gates have already passed.
REQUIRED_GATE_IDS = (
    "stage2.G2.0",
    "stage2.G2.1",
    "stage2.G2.2-dev",
    "stage2.G2.2",
    "stage2.G2.3",
    "stage2.G2.4a",
    "stage2.G2.4b",
    "stage2.G2.5",
    "stage2.G2.6",
    "stage2.G2.7a",
    "stage2.G2.7b",
    "stage2.G2.8",
)


def _logical_ref(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a POSIX-relative reference")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a POSIX-relative reference")
    # materialize_s204 rejects these as formal source authorities.  Rejecting
    # them here avoids publishing a commit that the downstream loader cannot
    # consume.
    if set(path.parts).intersection({"fixtures", "src", "ops", "configs"}):
        raise ValueError(f"{field} cannot point into tracked code/config/fixture")
    return path.as_posix()


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"source is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path, ref: str, source: Path) -> str:
    """Make the source ref a real immutable DATA_ROOT file without drift."""

    logical = _logical_ref(ref, field="source_ref")
    source = source.resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"source is not a regular file: {source}")
    target = (root / Path(*PurePosixPath(logical).parts)).resolve()
    target.relative_to(root.resolve())
    payload = source.read_bytes()
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise RuntimeError(f"SOURCE_SNAPSHOT_DRIFT:{logical}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, payload)
    return logical


def _require_formal_source(ref: str, source: Path) -> None:
    """Check the identity-bearing JSON notes without trusting their hashes."""

    value = load_canonical_json(source)
    if not isinstance(value, dict):
        raise ValueError(f"formal source must be an object: {ref}")
    if ref.endswith("contract-freeze-s202.json"):
        if value.get("schema_version") != "stage2-s202-formal-contract-freeze-v1":
            raise ValueError("S202 contract explanation schema mismatch")
        if value.get("stage") != STAGE or value.get("scope") != "formal":
            raise ValueError("S202 contract explanation is not formal Stage 2")


def _schema_hash(path: Path) -> str:
    # A schema source may be pretty-printed JSON; strict parsing proves that
    # it is real JSON while the hash remains the exact source-byte identity.
    value = loads_strict_json(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"schema source must be an object: {path}")
    return _sha256(path)


def _base_config_hash(path: Path) -> str:
    value = _load_mapping(path)
    identity = value.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("base config must contain identity mapping")
    if identity.get("stage") != STAGE or identity.get("run_intent") != "formal":
        raise ValueError("base config must declare formal Stage 2")
    if identity.get("formal_eligible") is not True:
        raise ValueError("base config formal_eligible must be true")
    # This is the semantic hash of the parsed config, independent of YAML
    # whitespace.  It is intentionally not a caller-provided hash.
    return canonical_json_hash(value)


@dataclass(frozen=True, slots=True)
class S202Production:
    freeze: ContractFreeze
    contract_commit_ref: str
    amended_evidence_ref: str
    amended_evidence: FormalExecutionEvidence


def produce_s202_contract_freeze(
    *,
    data_root: str | Path,
    source_files: Mapping[str, str | Path],
    schema_files: Mapping[str, str | Path],
    base_config: str | Path,
    formal_execution_ref: str,
    amended_evidence_ref: str = DEFAULT_AMENDED_EVIDENCE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    frozen_at: str,
) -> S202Production:
    """Build, publish and round-trip the formal S202 contract/evidence pair.

    ``source_files`` maps the desired workspace-relative evidence ref to the
    actual file to read.  The source bytes are snapshotted at that ref before
    publishing, so every ``source_ref`` in the formal TaskArtifact is readable
    from the DATA_ROOT consumed by materializers.
    """

    root = Path(data_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("data_root must be an existing non-symlink directory")
    source_hashes: dict[str, str] = {}
    for raw_ref, raw_path in source_files.items():
        ref = _snapshot(root, raw_ref, Path(raw_path))
        source = root / Path(*PurePosixPath(ref).parts)
        _require_formal_source(ref, source) if ref.endswith("contract-freeze-s202.json") else None
        source_hashes[ref] = _sha256(source)
    if not source_hashes:
        raise ValueError("at least one formal source is required")

    schema_hashes = {
        _logical_schema_ref(ref): _schema_hash(Path(path).resolve())
        for ref, path in schema_files.items()
    }
    if not schema_hashes:
        raise ValueError("at least one schema source is required")

    config_path = Path(base_config).resolve()
    config_hash = _base_config_hash(config_path)
    config_ref = _logical_ref(
        "evidence/stage2/s202-formal-auth/base-config-formal-stage2-estimator.yaml",
        field="base_config_ref",
    )
    source_hashes[config_ref] = _sha256(config_path)
    # Keep a readable snapshot for the config source ref as well.  Its parsed
    # semantic hash is config_hash; the raw hash remains in source_hashes.
    _snapshot(root, config_ref, config_path)

    explanation_ref = next(
        (ref for ref in source_hashes if ref.endswith("contract-freeze-s202.json")),
        None,
    )
    if explanation_ref is None:
        raise ValueError("contract explanation source is required")

    freeze = ContractFreeze(
        contract_id="stage2.contract.s202",
        stage=STAGE,
        scope="formal",
        state=ContractState.FROZEN,
        formula_version=FORMULA_VERSION,
        config_hash=config_hash,
        schema_hashes=schema_hashes,
        source_hashes=dict(sorted(source_hashes.items())),
        required_gate_ids=REQUIRED_GATE_IDS,
        frozen_at=frozen_at,
        reason=None,
        decision_ref=explanation_ref,
    )

    store = TaskArtifactStore(root, output_dir)
    published = store.publish(
        task_id=TASK_ID,
        artifact_kind=ARTIFACT_KIND,
        config_hash=freeze.config_hash,
        run_intent="formal",
        payload=freeze.to_dict(),
        formal_eligible=True,
        source_refs=tuple(sorted(source_hashes)),
    )
    loaded = load_committed_task_artifact(root, published.commit_ref, require_formal=True)
    matches = [
        item
        for item in (loaded.payload,)
        if isinstance(item, Mapping) and item.get("schema_version") == "contract-freeze-v1"
    ]
    if len(matches) != 1 or ContractFreeze.from_mapping(dict(matches[0])) != freeze:
        raise RuntimeError("CONTRACT_FREEZE_TASK_ARTIFACT_ROUND_TRIP_FAILED")

    parent_ref = _logical_ref(formal_execution_ref, field="formal_execution_ref")
    parent_path = root / Path(*PurePosixPath(parent_ref).parts)
    parent = FormalExecutionEvidence.from_mapping(load_canonical_json(parent_path))
    metadata = dict(parent.metadata)
    amendment_fields: dict[str, Any] = {
        "amendment_parent_ref": parent_ref,
        "amendment_type": "contract-freeze-v1-task-artifact-binding",
        "contract_freeze_commit_ref": published.commit_ref,
        "contract_freeze_artifact_hash": freeze.artifact_hash,
    }
    for key, value in amendment_fields.items():
        if key in metadata and metadata[key] != value:
            raise RuntimeError(f"FORMAL_EVIDENCE_METADATA_DRIFT:{key}")
        metadata[key] = value
    amended = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash=freeze.artifact_hash,
        asset_manifest_hashes=parent.asset_manifest_hashes,
        prerequisite_gates=parent.prerequisite_gates,
        metadata=metadata,
    )
    amended_ref = _logical_ref(amended_evidence_ref, field="amended_evidence_ref")
    amended_path = root / Path(*PurePosixPath(amended_ref).parts)
    encoded = amended.to_dict()
    if amended_path.exists():
        existing = FormalExecutionEvidence.from_mapping(load_canonical_json(amended_path))
        if existing != amended:
            raise RuntimeError("FORMAL_EVIDENCE_AMENDMENT_DRIFT")
    else:
        write_canonical_json(amended_path, encoded)
    reread = FormalExecutionEvidence.from_mapping(load_canonical_json(amended_path))
    if reread != amended or reread.contract_freeze_hash != freeze.artifact_hash:
        raise RuntimeError("FORMAL_EVIDENCE_AMENDMENT_ROUND_TRIP_FAILED")
    return S202Production(
        freeze=freeze,
        contract_commit_ref=published.commit_ref,
        amended_evidence_ref=amended_ref,
        amended_evidence=reread,
    )


def _logical_schema_ref(ref: str) -> str:
    if not isinstance(ref, str) or not ref or "\\" in ref:
        raise ValueError("schema_ref must be a POSIX-relative reference")
    path = PurePosixPath(ref)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("schema_ref must be a POSIX-relative reference")
    return path.as_posix()


def _default_sources(repository: Path, data_root: Path) -> dict[str, Path]:
    return {
        "evidence/stage2/s202-formal-auth/contract-freeze-s202.json": data_root
        / "evidence/stage2/s202-formal-auth/contract-freeze-s202.json",
        "reports/stage2/s2.2/formal-authorization-amendment-20260823.json": data_root
        / "reports/stage2/s2.2/formal-authorization-amendment-20260823.json",
        "reports/stage2/s2.2/g2.1-formal-stage1-handoff-evidence-20260823-r1.json": data_root
        / "reports/stage2/s2.2/g2.1-formal-stage1-handoff-evidence-20260823-r1.json",
        (
            "evidence/stage1/s1-11-formal/3f18b04df8922be9894678ae4842bd999c7e8fd5/"
            "s1-11-r4-20260821/index.json"
        ): data_root
        / "evidence/stage1/s1-11-formal/3f18b04df8922be9894678ae4842bd999c7e8fd5/"
        / "s1-11-r4-20260821/index.json",
        "evidence/stage2/s202-formal-auth/source-snapshots/plan-s2-01.md": repository
        / "plan/stage2/01_scope_hypotheses_and_preregistration.md",
        "evidence/stage2/s202-formal-auth/source-snapshots/plan-s2-02.md": repository
        / "plan/stage2/02_stage1_handoff_and_fixed_state_contract.md",
        "evidence/stage2/s202-formal-auth/source-snapshots/mathematics.md": repository
        / "docs/mathematics.md",
    }


def _parse_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        ref, separator, path = value.partition("=")
        if not separator or not ref or not path:
            raise ValueError("source spec must be REF=PATH")
        if ref in result:
            raise ValueError(f"duplicate source ref: {ref}")
        result[ref] = Path(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--formal-execution-ref", required=True)
    parser.add_argument("--source", action="append", default=[], metavar="REF=PATH")
    parser.add_argument(
        "--schema",
        action="append",
        default=[],
        metavar="REF=PATH",
        help="schema source; defaults to the two shared Stage schemas",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--amended-evidence-ref", default=DEFAULT_AMENDED_EVIDENCE)
    parser.add_argument("--frozen-at", required=True)
    args = parser.parse_args(argv)
    root = args.data_root.resolve()
    repository = args.repository.resolve()
    sources = _parse_specs(args.source) if args.source else _default_sources(repository, root)
    schemas = _parse_specs(args.schema) if args.schema else {
        "schemas/shared/contract-freeze-v1.json": repository
        / "schemas/shared/contract-freeze-v1.json",
        "schemas/shared/formal-execution-evidence-v1.json": repository
        / "schemas/shared/formal-execution-evidence-v1.json",
    }
    result = produce_s202_contract_freeze(
        data_root=root,
        source_files=sources,
        schema_files=schemas,
        base_config=(
            args.base_config
            if args.base_config.is_absolute()
            else repository / args.base_config
        ),
        formal_execution_ref=args.formal_execution_ref,
        amended_evidence_ref=args.amended_evidence_ref,
        output_dir=args.output_dir,
        frozen_at=args.frozen_at,
    )
    print(
        canonical_json_bytes(
            {
                "contract_commit_ref": result.contract_commit_ref,
                "contract_artifact_hash": result.freeze.artifact_hash,
                "amended_evidence_ref": result.amended_evidence_ref,
                "amended_evidence_hash": result.amended_evidence.artifact_hash,
            }
        ).decode("utf-8").rstrip("\n")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
