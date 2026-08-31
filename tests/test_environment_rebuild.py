from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pytest

from ops.stage0 import rebuild_environment as rebuild
from ops.stage0.rebuild_environment import (
    CANDIDATE_NAME,
    audited_command_output,
    controlled_environment,
    external_network_attempts,
    hashed_requirements_bytes,
    inventory_owned_tree,
    load_lock_provenance,
    load_gpu_baseline,
    normalize_cudnn_version,
    parse_pins,
    parse_wheel_manifest,
    read_immutable,
    remove_exact_owned_tree,
    stable_hash,
    validate_core_runtime_versions,
    validate_direct_requirement_inputs,
    wheel_entries_by_distribution,
    write_immutable,
)
from param_importance_nlp.atomic import stable_json_bytes
from param_importance_nlp.environment import (
    parse_requirements_lock,
    parse_wheelhouse_manifest,
)


def test_parse_pins_accepts_indexes_and_exact_versions(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "--extra-index-url https://example.invalid/simple\n"
        "Foo_Bar==1.2.3\n"
        "numpy==2.5.1\n",
        encoding="utf-8",
    )
    assert parse_pins(lock) == {"foo-bar": "1.2.3", "numpy": "2.5.1"}


@pytest.mark.parametrize(
    "requirement",
    ["name>=1", "name @ https://example.invalid/a.whl", "-e ./package", "git+https://x"],
)
def test_parse_pins_rejects_unlocked_or_remote_requirements(
    tmp_path: Path, requirement: str
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text(requirement + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_pins(lock)


def test_parse_wheel_manifest_handles_crlf(tmp_path: Path) -> None:
    wheel = b"wheel-fixture"
    digest = hashlib.sha256(wheel).hexdigest()
    manifest = tmp_path / "wheels.tsv"
    manifest.write_bytes(f"{digest}\t{len(wheel)}\tfixture-1-py3-none-any.whl\r\n".encode())
    assert parse_wheel_manifest(manifest) == {
        "fixture-1-py3-none-any.whl": {"sha256": digest, "size": len(wheel)}
    }


def test_required_dependency_inputs_are_bound_to_lock_provenance() -> None:
    repository = Path(__file__).resolve().parents[1]
    lock_path = repository / "environment" / "requirements.lock"
    input_paths = [
        repository / "environment" / "base-requirements.in",
        repository / "environment" / "requirements.in",
        repository / "environment" / "linux-only-requirements.in",
    ]
    input_hashes = {
        str(path.relative_to(repository)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in input_paths
    }
    lock = parse_requirements_lock(lock_path)

    provenance = load_lock_provenance(
        repository / "reports" / "stage0" / "lock-provenance-20260719.json",
        repository,
        lock=lock_path,
        lock_snapshot=lock,
        dependency_inputs=input_hashes,
    )
    audit = validate_direct_requirement_inputs(input_paths, lock)

    assert provenance["runtime_expectations"]["torch_cuda_runtime"] == "12.6"
    assert audit["status"] == "PASS"
    assert audit["direct_distribution_count"] > 10

    stale_hashes = dict(input_hashes)
    stale_hashes["environment/requirements.in"] = "0" * 64
    with pytest.raises(ValueError, match="required input hashes"):
        load_lock_provenance(
            repository / "reports" / "stage0" / "lock-provenance-20260719.json",
            repository,
            lock=lock_path,
            lock_snapshot=lock,
            dependency_inputs=stale_hashes,
        )


def test_direct_requirement_pin_must_match_lock(tmp_path: Path) -> None:
    declaration = tmp_path / "requirements.in"
    declaration.write_text("alpha==2\n", encoding="utf-8")
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("alpha==1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicts with locked"):
        validate_direct_requirement_inputs(
            [declaration], parse_requirements_lock(lock_path)
        )


def test_core_runtime_versions_must_match_lock_policy() -> None:
    repository = Path(__file__).resolve().parents[1]
    lock = parse_requirements_lock(repository / "environment" / "requirements.lock")
    expectations = json.loads(
        (
            repository
            / "reports"
            / "stage0"
            / "lock-provenance-20260719.json"
        ).read_text(encoding="utf-8")
    )["runtime_expectations"]
    observed = {
        "python": "3.12.3",
        "implementation": "CPython",
        "torch": "2.12.1+cu126",
        "torch_cuda_runtime": "12.6",
        "cudnn": 91002,
        "cudnn_distribution": "9.10.2.21",
        "nccl_distribution": "2.29.3",
        "transformers": "4.57.6",
        "datasets": "4.8.5",
        "accelerate": "1.14.0",
        "tensorboard": "2.21.0",
    }

    assert validate_core_runtime_versions(observed, lock, expectations)["status"] == "PASS"

    drifted = dict(observed)
    drifted["torch_cuda_runtime"] = "12.7"
    with pytest.raises(RuntimeError, match="torch_cuda_runtime"):
        validate_core_runtime_versions(drifted, lock, expectations)


def test_network_audit_rejects_inet_but_not_unix(tmp_path: Path) -> None:
    trace = tmp_path / "trace.txt"
    trace.write_text(
        '1 socket(AF_UNIX, SOCK_STREAM, 0) = 3\n'
        '2 connect(3, {sa_family=AF_INET, sin_port=htons(443)}, 16) = -1\n',
        encoding="utf-8",
    )
    assert external_network_attempts(trace) == [
        '2 connect(3, {sa_family=AF_INET, sin_port=htons(443)}, 16) = -1'
    ]


def test_environment_identity_hash_is_order_independent() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_controlled_environment_is_an_explicit_secret_free_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = {
        "HF_TOKEN": "hf-secret",
        "HUGGING_FACE_HUB_TOKEN": "hub-secret",
        "PIP_INDEX_URL": "https://token@example.invalid/simple",
        "PIP_EXTRA_INDEX_URL": "https://extra.example.invalid/simple",
        "PIP_TRUSTED_HOST": "example.invalid",
        "PIP_CERT": "secret-certificate.pem",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)

    data_root = tmp_path / "data-root"
    cache = data_root / "cache" / "candidate"
    temporary = data_root / "tmp" / "candidate"
    cache.mkdir(parents=True)
    temporary.mkdir(parents=True)
    environment = controlled_environment(data_root, cache, temporary)

    expected_keys = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TZ",
        "PIP_NO_INDEX",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "PIP_CONFIG_FILE",
        "PIP_CACHE_DIR",
        "PIP_REQUIRE_VIRTUALENV",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
    }
    assert set(environment) == expected_keys
    assert set(environment).isdisjoint(inherited)
    assert not set(environment.values()).intersection(inherited.values())
    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["PIP_CONFIG_FILE"]


@pytest.mark.parametrize(
    "candidate",
    ["a", "stage0-candidate-v1", "Candidate_01.release", "a" * 96],
)
def test_candidate_name_accepts_one_safe_component(candidate: str) -> None:
    assert CANDIDATE_NAME.fullmatch(candidate) is not None


@pytest.mark.parametrize(
    "candidate",
    ["", ".hidden", "-leading", "contains space", "a/b", "a\\b", "a" * 97],
)
def test_candidate_name_rejects_unsafe_or_oversized_values(candidate: str) -> None:
    assert CANDIDATE_NAME.fullmatch(candidate) is None


def test_cudnn_runtime_integer_is_rendered_as_a_semantic_version() -> None:
    assert normalize_cudnn_version(91002) == "9.10.2"
    assert normalize_cudnn_version("8907") == "8.9.7"
    assert normalize_cudnn_version(None) is None
    with pytest.raises(ValueError, match="Invalid cuDNN"):
        normalize_cudnn_version(0)


def _gpu_baseline_payload(*, schema: str = "stage0.gate-report.v1") -> dict[str, object]:
    return {
        "schema_version": schema,
        "generated_at": "2026-07-19T00:00:00+08:00",
        "status": "BLOCKED",
        "subgates": {
            "G0-C": {"status": "PASS"},
            "G0-G": {
                "status": "BLOCKED",
                "evidence": {"driver_version": "575.57.08"},
            },
        },
        "system_snapshot": {
            "hostname": "fixture-host",
            "runtime_versions": {"system_cuda_toolkit": "12.9"},
        },
    }


def test_gpu_baseline_accepts_the_blocked_gate_schema(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    report = repository / "reports" / "stage0" / "g0.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(_gpu_baseline_payload()), encoding="utf-8")

    baseline = load_gpu_baseline(
        report,
        repository,
        expected_hostname="fixture-host",
        now=datetime(2026, 7, 19, 1, tzinfo=timezone.utc),
    )

    assert baseline["g0_g_status"] == "BLOCKED"
    assert baseline["driver_version"] == "575.57.08"
    assert baseline["system_cuda_toolkit_version"] == "12.9"
    assert baseline["ref"] == "reports/stage0/g0.json"
    assert baseline["sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()


def test_gpu_baseline_rejects_wrong_schema_and_repository_escape(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    wrong_schema = repository / "wrong-schema.json"
    wrong_schema.write_text(
        json.dumps(_gpu_baseline_payload(schema="stage0.gate-report.v0")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected schema"):
        load_gpu_baseline(
            wrong_schema,
            repository,
            expected_hostname="fixture-host",
            now=datetime(2026, 7, 19, 1, tzinfo=timezone.utc),
        )

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_gpu_baseline_payload()), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gpu_baseline(
            outside,
            repository,
            expected_hostname="fixture-host",
            now=datetime(2026, 7, 19, 1, tzinfo=timezone.utc),
        )


def test_hashed_requirements_use_canonical_lock_and_manifest_wheel_mapping(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text(
        "--extra-index-url https://example.invalid/simple\n"
        "Foo.Bar==1.0\n"
        "typing-extensions==4.16.0\n",
        encoding="utf-8",
    )
    hashes = {
        "foo_bar-1.0-py3-none-any.whl": "b" * 64,
        "typing_extensions-4.16.0-py3-none-any.whl": "c" * 64,
    }
    manifest_path = tmp_path / "wheelhouse-sha256.tsv"
    manifest_path.write_text(
        "".join(
            f"{digest}\t1\t{filename}\n"
            for filename, digest in reversed(list(hashes.items()))
        ),
        encoding="ascii",
    )

    lock = parse_requirements_lock(lock_path)
    manifest = parse_wheelhouse_manifest(manifest_path)
    mapped = wheel_entries_by_distribution(manifest)
    rendered = hashed_requirements_bytes(lock, manifest)

    assert sorted(mapped) == ["foo-bar", "typing-extensions"]
    assert sorted(entry.filename for entry in mapped["foo-bar"]) == [
        "foo_bar-1.0-py3-none-any.whl",
    ]
    assert rendered == (
        f"foo-bar==1.0 --hash=sha256:{'b' * 64}\n"
        f"typing-extensions==4.16.0 --hash=sha256:{'c' * 64}\n"
    ).encode("ascii")


def test_hashed_requirements_reject_a_lock_pin_without_a_manifest_wheel(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("alpha==1\nbeta==2\n", encoding="utf-8")
    manifest_path = tmp_path / "wheelhouse-sha256.tsv"
    manifest_path.write_text(
        f"{'a' * 64}\t1\talpha-1-py3-none-any.whl\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="No manifest wheel matches beta==2"):
        hashed_requirements_bytes(
            parse_requirements_lock(lock_path),
            parse_wheelhouse_manifest(manifest_path),
        )


def test_owned_tree_cleanup_rejects_content_drift_before_exact_removal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "owned"
    nested = root / "nested"
    nested.mkdir(parents=True)
    owner = {"schema_version": "stage0.owned-temporary.v1", "token": "fixture"}
    (root / ".stage0-owner.json").write_bytes(stable_json_bytes(owner))
    payload = nested / "payload.txt"
    payload.write_text("before", encoding="utf-8")
    expected = inventory_owned_tree(root)

    payload.write_text("content-drifted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contents drifted"):
        remove_exact_owned_tree(root, owner_marker=owner, expected=expected)
    assert root.is_dir()
    assert payload.read_text(encoding="utf-8") == "content-drifted"

    remove_exact_owned_tree(
        root,
        owner_marker=owner,
        expected=inventory_owned_tree(root),
    )
    assert not root.exists()


def test_owned_tree_inventory_rejects_a_symbolic_link_root_without_os_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "owned"
    root.mkdir()
    path_type = type(root)
    original_is_symlink = path_type.is_symlink

    def pretend_root_is_symlink(path: Path) -> bool:
        return path == root or original_is_symlink(path)

    monkeypatch.setattr(path_type, "is_symlink", pretend_root_is_symlink)
    with pytest.raises(RuntimeError, match="Unsafe owned-tree root"):
        inventory_owned_tree(root)


def test_write_immutable_never_overwrites_existing_evidence(tmp_path: Path) -> None:
    target = tmp_path / "manifests" / "identity.json"
    write_immutable(target, b"first\n")

    with pytest.raises(FileExistsError):
        write_immutable(target, b"second\n")

    assert target.read_bytes() == b"first\n"
    assert not list(target.parent.glob(f".{target.name}.tmp-*"))
    if rebuild.os.name != "nt":
        assert target.stat().st_mode & 0o022 == 0


def test_read_immutable_rejects_a_hard_link(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "identity.json"
    source.write_bytes(b"evidence\n")
    target.hardlink_to(source)

    with pytest.raises(RuntimeError, match="single-link regular file"):
        read_immutable(target)


@pytest.mark.skipif(rebuild.os.name == "nt", reason="POSIX permission semantics")
def test_read_immutable_rejects_group_writable_evidence(tmp_path: Path) -> None:
    target = tmp_path / "identity.json"
    target.write_bytes(b"evidence\n")
    target.chmod(0o664)

    with pytest.raises(RuntimeError, match="group/other writable"):
        read_immutable(target)


def test_network_audit_detects_sendmmsg(tmp_path: Path) -> None:
    trace = tmp_path / "trace.txt"
    line = "9 sendmmsg(3, [{msg_hdr={msg_name={sa_family=AF_INET}}}], 1, 0) = 1"
    trace.write_text(line + "\n", encoding="utf-8")

    assert external_network_attempts(trace) == [line]


def test_failed_audited_command_still_parses_external_network_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "command-network.trace"
    log = tmp_path / "command.log"
    external_line = (
        "42 sendto(3, data, 4, 0, {sa_family=AF_INET6, sin6_port=htons(443)}, 28) = -1"
    )

    def failed_run_logged(
        command: list[str], *, environment: dict[str, str], log_path: Path
    ) -> subprocess.CompletedProcess[str]:
        del environment
        traced_path = Path(command[command.index("-o") + 1])
        traced_path.write_text(external_line + "\n", encoding="utf-8")
        log_path.write_text("exit_code=7\ncommand failed\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 7, stdout="command failed")

    monkeypatch.setattr(rebuild, "run_logged", failed_run_logged)
    with pytest.raises(RuntimeError, match="attempted external network access"):
        audited_command_output(
            ["python", "-c", "raise SystemExit(7)"],
            environment={},
            log=log,
            trace=trace,
        )
    assert external_network_attempts(trace) == [external_line]
