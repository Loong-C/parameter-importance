from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

import pytest

import param_importance_nlp.assets as assets_module
from param_importance_nlp.assets import (
    AssetActorRole,
    AssetEncodingError,
    AssetFile,
    AssetNotReadyError,
    AssetState,
    AssetValidationError,
    AssetVerificationError,
    build_manifest,
    compute_asset_id,
    load_manifest,
    publish_manifest_atomic,
    resolve_ready_asset,
    transition_manifest,
    validate_asset_path,
    validate_manifest,
    validate_state_transition,
    verify_only,
)
from param_importance_nlp.atomic import sha256_file


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata(asset_type: str = "model") -> dict[str, object]:
    if asset_type == "model":
        return {
            "architecture": "FixtureLM",
            "parameter_count": 42,
            "dtype": "float32",
            "initialization_id": "seed:7",
        }
    if asset_type == "tokenizer":
        return {
            "tokenizer_class": "FixtureTokenizer",
            "vocab_size": 128,
            "special_tokens": {"pad_token": "<pad>", "unk_token": "<unk>"},
            "normalization": "NFC",
        }
    if asset_type == "dataset":
        return {
            "splits": {
                "train": {"sample_count": 10, "fields": ["text", "label"]},
                "validation": {"sample_count": 2, "fields": ["text", "label"]},
            },
            "preprocessing_version": "fixture-preprocess-v1",
        }
    if asset_type == "source":
        return {"source_kind": "git", "license": "Apache-2.0"}
    raise AssertionError(asset_type)


def _manifest(
    tmp_path: Path,
    *,
    declared_size: int | None = None,
    declared_hash: str | None = None,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "asset-root"
    root.mkdir(parents=True)
    payload = b"immutable fixture\n"
    (root / "weights.bin").write_bytes(payload)
    manifest = build_manifest(
        asset_type="model",
        name="fixture-step0",
        source="huggingface:example/fixture",
        revision="0123456789abcdef",
        files=[
            AssetFile(
                path="weights.bin",
                size_bytes=len(payload) if declared_size is None else declared_size,
                sha256=_digest(payload) if declared_hash is None else declared_hash,
                role="weights",
            )
        ],
        actor="test-fetcher",
        actor_role=AssetActorRole.FETCHER,
        evidence_ref="evidence/fetch-start.json",
        generator_version="tests/1",
        metadata=_metadata(),
        created_at="2026-07-19T00:00:00Z",
    )
    return root, manifest


def _transition_to(
    manifest: dict[str, object], target: AssetState
) -> dict[str, object]:
    sequence = [
        (
            AssetState.DOWNLOADED,
            AssetActorRole.FETCHER,
            "evidence/fetch-complete.json",
            "2026-07-19T00:00:01Z",
        ),
        (
            AssetState.VERIFIED,
            AssetActorRole.VERIFIER,
            "evidence/verification.json",
            "2026-07-19T00:00:02Z",
        ),
        (
            AssetState.READY,
            AssetActorRole.GATE,
            "evidence/gate.json",
            "2026-07-19T00:00:03Z",
        ),
    ]
    updated = manifest
    for state, actor_role, evidence_ref, at in sequence:
        updated = transition_manifest(
            updated,
            state,
            actor="test-actor",
            actor_role=actor_role,
            evidence_ref=evidence_ref,
            summary=f"entered {state.value}",
            at=at,
        )
        if state is target:
            return updated
    raise AssertionError(target)


def _manifest_directory(tmp_path: Path) -> Path:
    path = tmp_path / "manifests"
    path.mkdir(parents=True)
    return path


def test_asset_state_machine_requires_order_role_and_terminal_invalid() -> None:
    validate_state_transition(
        AssetState.DOWNLOADING,
        AssetState.DOWNLOADED,
        actor_role=AssetActorRole.FETCHER,
    )
    validate_state_transition(
        AssetState.DOWNLOADED,
        AssetState.VERIFIED,
        actor_role=AssetActorRole.VERIFIER,
    )
    validate_state_transition(
        AssetState.VERIFIED,
        AssetState.READY,
        actor_role=AssetActorRole.GATE,
    )
    with pytest.raises(AssetValidationError, match="not authorized"):
        validate_state_transition(
            AssetState.DOWNLOADING,
            AssetState.DOWNLOADED,
            actor_role=AssetActorRole.VERIFIER,
        )
    with pytest.raises(AssetValidationError, match="Forbidden"):
        validate_state_transition(
            AssetState.DOWNLOADING,
            AssetState.READY,
            actor_role=AssetActorRole.GATE,
        )
    with pytest.raises(AssetValidationError, match="Forbidden"):
        validate_state_transition(
            AssetState.INVALID,
            AssetState.DOWNLOADING,
            actor_role=AssetActorRole.FETCHER,
        )


def test_history_enforces_roles_and_evidence(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    wrong_role = deepcopy(downloaded)
    wrong_role["state_history"][-1]["actor_role"] = "gate"
    with pytest.raises(AssetValidationError, match="not authorized"):
        validate_manifest(wrong_role)

    with pytest.raises(AssetValidationError, match="evidence_ref"):
        transition_manifest(
            downloaded,
            AssetState.VERIFIED,
            actor="test-verifier",
            actor_role=AssetActorRole.VERIFIER,
            evidence_ref=None,
            summary="missing evidence",
        )

    missing_key = deepcopy(downloaded)
    del missing_key["state_history"][-1]["evidence_ref"]
    with pytest.raises(AssetValidationError, match="missing"):
        validate_manifest(missing_key)


def test_invalid_manifest_cannot_flip_back_to_success(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    invalid = transition_manifest(
        downloaded,
        AssetState.INVALID,
        actor="test-verifier",
        actor_role=AssetActorRole.VERIFIER,
        evidence_ref="evidence/hash-failure.json",
        summary="hash failed",
        at="2026-07-19T00:00:02Z",
    )
    with pytest.raises(AssetValidationError, match="Forbidden"):
        transition_manifest(
            invalid,
            AssetState.VERIFIED,
            actor="test-verifier",
            actor_role=AssetActorRole.VERIFIER,
            evidence_ref="evidence/illegal-reversal.json",
            summary="illegal reversal",
        )


def test_asset_id_is_stable_across_file_order_and_state(tmp_path: Path) -> None:
    first = AssetFile("b.bin", 2, _digest(b"bb"))
    second = AssetFile("a.bin", 1, _digest(b"a"))
    identity = {
        "asset_type": "model",
        "name": "ordered-fixture",
        "source": "huggingface:example/ordered",
        "revision": "deadbeef",
        "metadata": _metadata(),
    }
    left = compute_asset_id(**identity, files=[first, second])
    right = compute_asset_id(**identity, files=[second, first])
    assert left == right

    changed_metadata = deepcopy(_metadata())
    changed_metadata["initialization_id"] = "seed:8"
    assert left != compute_asset_id(
        **{**identity, "metadata": changed_metadata}, files=[first, second]
    )

    _, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    assert downloaded["asset_id"] == manifest["asset_id"]


@pytest.mark.parametrize("asset_type", ["model", "tokenizer", "dataset", "source"])
def test_typed_metadata_accepts_each_asset_contract(asset_type: str) -> None:
    manifest = build_manifest(
        asset_type=asset_type,
        name=f"fixture-{asset_type}",
        source=f"fixture:{asset_type}",
        revision="v1.2.3",
        files=[AssetFile("artifact.bin", 1, _digest(b"x"))],
        actor="test-fetcher",
        actor_role="fetcher",
        evidence_ref="evidence/fetch.json",
        generator_version="tests/1",
        metadata=_metadata(asset_type),
        created_at="2026-07-19T00:00:00Z",
    )
    validate_manifest(manifest)


@pytest.mark.parametrize(
    ("asset_type", "field", "bad_value"),
    [
        ("model", "parameter_count", 0),
        ("tokenizer", "special_tokens", []),
        ("dataset", "splits", {}),
        ("source", "license", ""),
    ],
)
def test_typed_metadata_rejects_missing_or_wrong_minimum_fields(
    asset_type: str,
    field: str,
    bad_value: object,
) -> None:
    metadata = _metadata(asset_type)
    metadata[field] = bad_value
    with pytest.raises(AssetValidationError, match=f"metadata.{field}"):
        build_manifest(
            asset_type=asset_type,
            name=f"bad-{asset_type}",
            source=f"fixture:{asset_type}",
            revision="v1.2.3",
            files=[AssetFile("artifact.bin", 1, _digest(b"x"))],
            actor="test-fetcher",
            actor_role="fetcher",
            evidence_ref=None,
            generator_version="tests/1",
            metadata=metadata,
        )


@pytest.mark.parametrize("revision", ["", "unknown", "LATEST", "main", " master "])
def test_manifest_rejects_generic_or_blank_revision(revision: str) -> None:
    with pytest.raises(AssetValidationError, match="revision"):
        compute_asset_id(
            asset_type="model",
            name="bad-revision",
            source="fixture:model",
            revision=revision,
            files=[AssetFile("weights.bin", 1, _digest(b"x"))],
            metadata=_metadata(),
        )


@pytest.mark.parametrize(
    "path",
    [
        "../escape.bin",
        "/absolute.bin",
        "nested\\windows.bin",
        "weights.bin.part",
        "weights.part.meta",
        "asset.lock",
        "tmp/weights.bin",
        "weights.tmp-123",
        "temp/weights.bin",
    ],
)
def test_asset_paths_reject_traversal_and_unfinished_objects(path: str) -> None:
    with pytest.raises(AssetValidationError):
        validate_asset_path(path)


def test_manifest_rejects_case_colliding_paths() -> None:
    with pytest.raises(AssetValidationError, match="case-colliding"):
        compute_asset_id(
            asset_type="dataset",
            name="collision",
            source="huggingface:example/collision",
            revision="deadbeef",
            files=[
                AssetFile("Data.bin", 1, _digest(b"a")),
                AssetFile("data.bin", 1, _digest(b"b")),
            ],
            metadata=_metadata("dataset"),
        )


def test_manifest_source_rejects_query_or_signed_url() -> None:
    with pytest.raises(AssetValidationError, match="signed URL"):
        compute_asset_id(
            asset_type="model",
            name="unsafe-source",
            source="https://example.invalid/model?token=secret",
            revision="deadbeef",
            files=[AssetFile("weights.bin", 1, _digest(b"x"))],
            metadata=_metadata(),
        )


def test_verify_only_checks_size_and_sha_without_changing_state(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    before = deepcopy(downloaded)
    report = verify_only(downloaded, root)
    assert report == {
        "schema_version": "stage0-asset-verification-v1",
        "asset_id": downloaded["asset_id"],
        "state": "downloaded",
        "files_checked": 1,
        "bytes_checked": len(b"immutable fixture\n"),
        "ok": True,
    }
    assert downloaded == before


def test_verify_only_rejects_downloading_and_invalid(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    with pytest.raises(AssetVerificationError, match="acquisition must finish"):
        verify_only(manifest, root)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    invalid = transition_manifest(
        downloaded,
        AssetState.INVALID,
        actor="test-verifier",
        actor_role="verifier",
        evidence_ref="evidence/rejection.json",
        summary="rejected",
    )
    with pytest.raises(AssetVerificationError, match="acquisition must finish"):
        verify_only(invalid, root)


def test_verify_only_reports_size_and_hash_mismatch(tmp_path: Path) -> None:
    size_root, bad_size = _manifest(tmp_path / "size", declared_size=999)
    bad_size = _transition_to(bad_size, AssetState.DOWNLOADED)
    with pytest.raises(AssetVerificationError, match="Size mismatch"):
        verify_only(bad_size, size_root)

    hash_root, bad_hash = _manifest(tmp_path / "hash", declared_hash="0" * 64)
    bad_hash = _transition_to(bad_hash, AssetState.DOWNLOADED)
    with pytest.raises(AssetVerificationError, match="SHA-256 mismatch"):
        verify_only(bad_hash, hash_root)


def test_ready_only_resolver_rejects_other_states_and_detects_drift(
    tmp_path: Path,
) -> None:
    root, manifest = _manifest(tmp_path)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    with pytest.raises(AssetNotReadyError, match="requires state=ready"):
        resolve_ready_asset(downloaded, root)

    ready = _transition_to(manifest, AssetState.READY)
    resolved = resolve_ready_asset(ready, root)
    assert resolved.path_for("weights.bin") == (root / "weights.bin").resolve()
    assert resolved.asset_id == ready["asset_id"]

    with pytest.raises(TypeError):
        resolve_ready_asset(ready, root, verify_hashes=False)  # type: ignore[call-arg]

    (root / "weights.bin").write_bytes(b"tampered fixture!\n")
    assert (root / "weights.bin").stat().st_size == len(b"immutable fixture\n")
    with pytest.raises(AssetVerificationError, match="SHA-256 mismatch"):
        resolve_ready_asset(ready, root)


def test_atomic_manifest_publication_is_canonical_and_no_clobber(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "fixture.json"
    assert publish_manifest_atomic(
        target, manifest, manifest_root=manifest_root
    ) == target
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n")
    assert load_manifest(target) == manifest
    assert not list(target.parent.glob(".*.tmp-*"))
    with pytest.raises(FileExistsError, match="already exists"):
        publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
    assert not list(target.parent.glob(".*.tmp-*"))


def test_atomic_replacement_requires_cas_and_only_advances_history(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "candidate.json"
    publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)

    with pytest.raises(AssetValidationError, match="requires expected"):
        publish_manifest_atomic(
            target,
            downloaded,
            manifest_root=manifest_root,
            allow_replace=True,
        )

    publish_manifest_atomic(
        target,
        downloaded,
        manifest_root=manifest_root,
        allow_replace=True,
        expected_previous_sha256=sha256_file(target),
    )
    assert load_manifest(target)["state"] == "downloaded"

    invalid = transition_manifest(
        downloaded,
        AssetState.INVALID,
        actor="test-verifier",
        actor_role="verifier",
        evidence_ref="evidence/invalid.json",
        summary="invalid candidate",
    )
    publish_manifest_atomic(
        target,
        invalid,
        manifest_root=manifest_root,
        allow_replace=True,
        expected_previous_sha256=sha256_file(target),
    )
    ready_from_original = _transition_to(manifest, AssetState.READY)
    with pytest.raises(AssetValidationError, match="terminal"):
        publish_manifest_atomic(
            target,
            ready_from_original,
            manifest_root=manifest_root,
            allow_replace=True,
            expected_previous_sha256=sha256_file(target),
        )


def test_ready_manifest_is_immutable_at_path_but_invalidation_gets_new_path(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    ready = _transition_to(manifest, AssetState.READY)
    invalidated = transition_manifest(
        ready,
        AssetState.INVALID,
        actor="test-gate",
        actor_role="gate",
        evidence_ref="evidence/post-admission-audit.json",
        summary="post-admission audit failed",
    )
    manifest_root = _manifest_directory(tmp_path)
    ready_path = manifest_root / "ready.json"
    publish_manifest_atomic(ready_path, ready, manifest_root=manifest_root)
    with pytest.raises(AssetValidationError, match="new path"):
        publish_manifest_atomic(
            ready_path,
            invalidated,
            manifest_root=manifest_root,
            allow_replace=True,
            expected_previous_sha256=sha256_file(ready_path),
        )
    assert load_manifest(ready_path)["state"] == "ready"

    invalidation_path = manifest_root / "ready.invalidated.json"
    publish_manifest_atomic(
        invalidation_path, invalidated, manifest_root=manifest_root
    )
    assert load_manifest(invalidation_path)["state"] == "invalid"


def test_replacement_rejects_stale_cas_digest(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "candidate.json"
    publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
    stale_digest = sha256_file(target)
    downloaded = _transition_to(manifest, AssetState.DOWNLOADED)
    publish_manifest_atomic(
        target,
        downloaded,
        manifest_root=manifest_root,
        allow_replace=True,
        expected_previous_sha256=stale_digest,
    )
    verified = _transition_to(manifest, AssetState.VERIFIED)
    with pytest.raises(AssetValidationError, match="Stale"):
        publish_manifest_atomic(
            target,
            verified,
            manifest_root=manifest_root,
            allow_replace=True,
            expected_previous_sha256=stale_digest,
        )


def test_concurrent_replacements_allow_exactly_one_cas_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "candidate.json"
    publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
    expected = sha256_file(target)
    first = transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor="fetcher-a",
        actor_role="fetcher",
        evidence_ref="evidence/fetch-a.json",
        summary="fetch A complete",
    )
    second = transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor="fetcher-b",
        actor_role="fetcher",
        evidence_ref="evidence/fetch-b.json",
        summary="fetch B complete",
    )

    original_lock = assets_module._advisory_manifest_lock
    barrier = threading.Barrier(2)

    @contextmanager
    def synchronized_lock(lock_target: Path):
        barrier.wait(timeout=5)
        with original_lock(lock_target):
            yield

    monkeypatch.setattr(
        assets_module, "_advisory_manifest_lock", synchronized_lock
    )

    def publish(candidate: dict[str, object]) -> Path:
        return publish_manifest_atomic(
            target,
            candidate,
            manifest_root=manifest_root,
            allow_replace=True,
            expected_previous_sha256=expected,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, candidate) for candidate in (first, second)]
        successes = 0
        failures: list[BaseException] = []
        for future in futures:
            try:
                future.result(timeout=10)
                successes += 1
            except BaseException as error:
                failures.append(error)

    assert successes == 1
    assert len(failures) == 1
    assert isinstance(failures[0], AssetValidationError)
    assert "Concurrent" in str(failures[0]) or "Stale" in str(failures[0])
    assert load_manifest(target)["state"] == "downloaded"


def test_manifest_advisory_lock_rejects_a_hard_link(tmp_path: Path) -> None:
    _asset_root, manifest = _manifest(tmp_path / "asset")
    manifest_root = _manifest_directory(tmp_path)
    target = manifest_root / "candidate.json"
    victim = tmp_path / "unrelated-empty-file"
    victim.write_bytes(b"")
    lock_path = manifest_root / f".{target.name}.publish.lock"
    lock_path.hardlink_to(victim)

    with pytest.raises(AssetValidationError, match="single-link regular file"):
        publish_manifest_atomic(
            target,
            manifest,
            manifest_root=manifest_root,
            allow_replace=True,
        )

    assert victim.read_bytes() == b""


def test_publication_requires_approved_existing_non_symlink_root(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    manifest_root = _manifest_directory(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(AssetValidationError, match="escapes approved"):
        publish_manifest_atomic(
            outside / "escape.json", manifest, manifest_root=manifest_root
        )
    with pytest.raises(AssetValidationError, match="parent must already exist"):
        publish_manifest_atomic(
            manifest_root / "missing" / "candidate.json",
            manifest,
            manifest_root=manifest_root,
        )

    real = manifest_root / "real"
    real.mkdir()
    link = manifest_root / "linked"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a directory symlink is not permitted on this platform")
    with pytest.raises(AssetValidationError, match="symlinks or junctions"):
        publish_manifest_atomic(
            link / "candidate.json", manifest, manifest_root=manifest_root
        )


def test_strict_loader_rejects_bom_duplicate_keys_and_non_finite_json(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + canonical)
    with pytest.raises(AssetEncodingError, match="BOM"):
        load_manifest(bom)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"key":1,"key":2}', encoding="utf-8")
    with pytest.raises(AssetEncodingError, match="Duplicate"):
        load_manifest(duplicate)

    non_finite = tmp_path / "nan.json"
    non_finite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(AssetEncodingError, match="Non-finite"):
        load_manifest(non_finite)


def test_manifest_rejects_broken_history_asset_id_and_missing_metadata(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)
    broken_history = deepcopy(manifest)
    broken_history["state"] = "ready"
    with pytest.raises(AssetValidationError, match="final state_history"):
        validate_manifest(broken_history)

    broken_id = deepcopy(manifest)
    broken_id["asset_id"] = "0" * 64
    with pytest.raises(AssetValidationError, match="asset_id mismatch"):
        validate_manifest(broken_id)

    missing_metadata = deepcopy(manifest)
    del missing_metadata["metadata"]
    with pytest.raises(AssetValidationError, match="metadata"):
        validate_manifest(missing_metadata)


def test_schema_document_captures_roles_evidence_and_typed_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "schemas" / "stage0-asset-manifest-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schema_version"]["const"] == (
        "stage0-asset-manifest-v1"
    )
    assert "metadata" in schema["required"]
    transition_required = schema["$defs"]["stateTransition"]["required"]
    assert {"actor_role", "evidence_ref"} <= set(transition_required)
    assert {"modelMetadata", "tokenizerMetadata", "datasetMetadata", "sourceMetadata"} <= set(
        schema["$defs"]
    )
