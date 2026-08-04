from __future__ import annotations

import argparse
import http.client
from io import BytesIO, StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any

import pytest

from ops.stage0 import dispatch_g3_relay_via_lab as dispatcher
from ops.stage0 import receive_g3_asset_stream as receiver
from ops.stage0 import relay_g3_object_from_lab as relay
from param_importance_nlp.asset_acquisition import StreamReceptionPlan


_ROOT = Path(__file__).resolve().parents[1]
_REQUIREMENTS = _ROOT / "configs/stage0/g3-asset-requirements-v1.json"
_LAYOUT = _ROOT / "configs/stage0/g3-asset-layout-v1.json"
_PLAN = _ROOT / "configs/stage0/g3-download-plan-v1.json"
_FIRST_OBJECT = "huggingface/EleutherAI/pythia-410m-deduped/config.json"


def _dispatch_args(*object_ids: str) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=_ROOT,
        requirements=_REQUIREMENTS,
        layout=_LAYOUT,
        plan=_PLAN,
        object_id=list(object_ids),
        relay_process="local",
        overall_timeout_seconds=30.0,
        endpoint_profile="official",
        lab_python_profile="path",
    )


def _receiver_args(
    tmp_path: Path,
    binding: Any,
    **changes: Any,
) -> argparse.Namespace:
    data_root = tmp_path / "data-root"
    for name in ("models", "datasets", "operations", "tmp"):
        (data_root / name).mkdir(parents=True, exist_ok=True)
    values = {
        "source_root": _ROOT,
        "data_root": data_root,
        "requirements": _REQUIREMENTS,
        "layout": _LAYOUT,
        "plan": _PLAN,
        **binding.to_dict(),
        "overall_timeout_seconds": 30.0,
        "legacy_state_action": "none",
        "expected_offset": None,
        "plan_only": False,
    }
    values.pop("schema_version")
    values.update(changes)
    return argparse.Namespace(**values)


def test_receiver_resolves_only_the_frozen_object_target(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    for name in ("models", "datasets", "operations", "tmp"):
        (data_root / name).mkdir(parents=True, exist_ok=True)
    arguments = argparse.Namespace(
        source_root=_ROOT,
        data_root=data_root,
        requirements=_REQUIREMENTS,
        layout=_LAYOUT,
        plan=_PLAN,
        object_id=_FIRST_OBJECT,
        overall_timeout_seconds=30.0,
    )

    spec, target, binding, approved_root = receiver._resolve_frozen_object(
        arguments,
        route="local-via-lab",
    )

    assert spec.source_id == _FIRST_OBJECT
    assert spec.expected_size == 570
    assert target == data_root / "models/pythia-410m-deduped-step0/config.json"
    assert target.parent.is_dir()
    assert binding.object_id == _FIRST_OBJECT
    assert approved_root == data_root


def test_receiver_protocol_emitter_writes_exactly_one_json_line() -> None:
    output = StringIO()
    receiver._emit({"phase": "READY"}, stream=output)
    assert output.getvalue().count("\n") == 1
    assert output.getvalue().splitlines() == ['{"phase":"READY"}']


def test_dispatcher_sends_script_over_stdin_and_keeps_urls_out_of_argv(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[list[str], bytes, float]] = []
    monkeypatch.setattr(
        dispatcher,
        "resolve_source_git_commit",
        lambda _root, **_kwargs: "a" * 40,
    )
    arguments = _dispatch_args(_FIRST_OBJECT)
    arguments.relay_process = "lab"
    _, _, items = dispatcher._load_specs(arguments)
    expected_binding = items[0][1]

    def fake_run(
        command: list[str],
        *,
        input: bytes,
        check: bool,
        stdout: Any,
        stderr: Any,
        timeout: float,
    ) -> Any:
        assert check is False
        assert stdout is dispatcher.subprocess.PIPE
        assert stderr is None
        calls.append((command, input, timeout))
        result = {
            "schema_version": "stage0-asset-acquisition-result-v1",
            "status": "downloaded",
            "source_id": expected_binding.object_id,
            "revision": expected_binding.revision,
            "size_bytes": expected_binding.expected_size,
            "sha256": expected_binding.expected_sha256,
            "attempts": 1,
            "resumed": False,
            "network_accessed": True,
        }
        payload = {
            "schema_version": relay.PROTOCOL_VERSION,
            "phase": "COMPLETE",
            "object_id": expected_binding.object_id,
            "binding": expected_binding.to_dict(),
            "result": result,
            "runtime_urls_persisted": False,
        }
        return SimpleNamespace(
            returncode=0,
            stdout=(json.dumps(payload, separators=(",", ":")) + "\n").encode(),
        )

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    results = dispatcher.dispatch(arguments)

    assert len(results) == 1
    assert len(calls) == 1
    command, script, timeout = calls[0]
    assert command[0] == "ssh"
    assert dispatcher.LAB_ALIAS in command
    assert _FIRST_OBJECT in command
    assert all("://" not in item and "?" not in item for item in command)
    assert b"def _runtime_url" in script
    assert b"https://huggingface.co/EleutherAI" not in script
    assert timeout <= arguments.overall_timeout_seconds
    assert results[0]["route"] == "lab-direct"
    assert results[0]["result_status"] == "downloaded"
    assert results[0]["endpoint_profile"] == "official"
    assert results[0]["lab_python_profile"] == "path"


def test_relay_endpoint_profile_is_named_and_runtime_only() -> None:
    official = relay._runtime_url(
        _FIRST_OBJECT,
        "a" * 40,
        endpoint_profile="official",
    )
    mirror = relay._runtime_url(
        _FIRST_OBJECT,
        "a" * 40,
        endpoint_profile="hf-mirror",
    )
    assert official.startswith("https://huggingface.co/")
    assert mirror.startswith("https://hf-mirror.com/")
    assert official.removeprefix("https://huggingface.co") == mirror.removeprefix(
        "https://hf-mirror.com"
    )
    with pytest.raises(relay.RelayError, match="ENDPOINT_PROFILE_INVALID"):
        relay._runtime_url(_FIRST_OBJECT, "a" * 40, endpoint_profile="invalid")


def test_dispatcher_passes_only_the_named_mirror_profile_in_argv() -> None:
    arguments = _dispatch_args(_FIRST_OBJECT)
    arguments.relay_process = "lab"
    arguments.endpoint_profile = "hf-mirror"
    _, _, items = dispatcher._load_specs(arguments)
    command = dispatcher._lab_command(
        items[0][1],
        overall_timeout_seconds=30.0,
        endpoint_profile=arguments.endpoint_profile,
    )
    profile_index = command.index("--endpoint-profile")
    assert command[profile_index + 1] == "hf-mirror"
    assert all("://" not in item and "?" not in item for item in command)


def test_dispatcher_uses_only_a_named_lab_python_profile() -> None:
    arguments = _dispatch_args(_FIRST_OBJECT)
    arguments.relay_process = "lab"
    _, _, items = dispatcher._load_specs(arguments)
    command = dispatcher._lab_command(
        items[0][1],
        overall_timeout_seconds=30.0,
        lab_python_profile="cjl-python312",
    )
    assert r"C:\Users\cjl\Apps\Python312\python.exe" in command
    assert command[command.index(dispatcher.LAB_ALIAS) + 1] == (
        r"C:\Users\cjl\Apps\Python312\python.exe"
    )
    with pytest.raises(dispatcher.G3RelayDispatchError, match="profile is invalid"):
        dispatcher._lab_command(
            items[0][1],
            overall_timeout_seconds=30.0,
            lab_python_profile="arbitrary-command",
        )


def test_receiver_expected_offset_is_checked_inside_the_locked_callback() -> None:
    plan = StreamReceptionPlan(
        source_id=_FIRST_OBJECT,
        revision="a" * 40,
        expected_size=570,
        expected_sha256="b" * 64,
        offset=17,
        already_ready=False,
    )
    receiver._check_expected_offset(SimpleNamespace(expected_offset=17), plan)
    with pytest.raises(receiver.G3StreamReceiverError, match="offset does not match"):
        receiver._check_expected_offset(SimpleNamespace(expected_offset=18), plan)


def test_lab_pipe_command_is_url_free_and_uses_only_native_ssh_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    arguments = _dispatch_args(_FIRST_OBJECT)
    arguments.relay_process = "lab-pipe"
    relay_path, _, items = dispatcher._load_specs(arguments)
    binding = items[0][1]
    command = dispatcher._pipe_command(
        relay_path,
        binding,
        offset=0,
        overall_timeout_seconds=30.0,
        endpoint_profile="hf-mirror",
        lab_python_profile="cjl-python312",
    )
    assert Path(command[0]).name.casefold() == "cmd.exe"
    assert command[1:4] == ["/d", "/s", "/c"]
    pipeline = command[4]
    assert pipeline.count(" | ") == 2
    assert "type" in pipeline
    assert dispatcher.LAB_ALIAS in pipeline
    assert dispatcher.SERVER_ALIAS in pipeline
    assert "--relay-mode emit" in pipeline
    assert "--emit-offset 0" in pipeline
    assert "--expected-offset 0" in pipeline
    assert "://" not in pipeline and "?" not in pipeline

    complete_part_command = dispatcher._pipe_command(
        relay_path,
        binding,
        offset=binding.expected_size,
        overall_timeout_seconds=30.0,
        endpoint_profile="hf-mirror",
        lab_python_profile="cjl-python312",
    )
    complete_part_pipeline = complete_part_command[4]
    assert complete_part_pipeline.startswith("type NUL | ")
    assert dispatcher.LAB_ALIAS not in complete_part_pipeline
    assert f"--expected-offset {binding.expected_size}" in complete_part_pipeline


def test_lab_pipe_plan_and_completion_transcripts_are_exact() -> None:
    arguments = _dispatch_args(_FIRST_OBJECT)
    arguments.relay_process = "lab-pipe"
    _, _, items = dispatcher._load_specs(arguments)
    binding = items[0][1]
    reception = {
        "schema_version": "stage0-asset-stream-reception-plan-v1",
        "source_id": binding.object_id,
        "revision": binding.revision,
        "expected_size": binding.expected_size,
        "expected_sha256": binding.expected_sha256,
        "offset": 0,
        "already_ready": False,
    }
    ready = {
        "schema_version": relay.PROTOCOL_VERSION,
        "phase": "READY",
        "object_id": binding.object_id,
        "binding": binding.to_dict(),
        "reception": reception,
        "runtime_urls_persisted": False,
    }
    plan_complete = {**ready, "phase": "PLAN_COMPLETE"}
    plan_payload = "".join(
        json.dumps(value, separators=(",", ":")) + "\n"
        for value in (ready, plan_complete)
    ).encode()
    assert dispatcher._parse_plan_result(
        plan_payload,
        binding=binding,
        returncode=0,
    ) == reception

    result = {
        "schema_version": "stage0-asset-acquisition-result-v1",
        "status": "downloaded",
        "source_id": binding.object_id,
        "revision": binding.revision,
        "size_bytes": binding.expected_size,
        "sha256": binding.expected_sha256,
        "attempts": 1,
        "resumed": False,
        "network_accessed": True,
    }
    complete = {
        "schema_version": relay.PROTOCOL_VERSION,
        "phase": "COMPLETE",
        "object_id": binding.object_id,
        "binding": binding.to_dict(),
        "result": result,
        "runtime_urls_persisted": False,
    }
    pipe_payload = "".join(
        json.dumps(value, separators=(",", ":")) + "\n"
        for value in (ready, complete)
    ).encode()
    assert dispatcher._parse_pipe_result(
        pipe_payload,
        binding=binding,
        returncode=0,
        expected_offset=0,
    ) == result


def test_dispatcher_runs_plan_then_native_pipe_for_lab_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    monkeypatch.setattr(
        dispatcher,
        "resolve_source_git_commit",
        lambda _root, **_kwargs: "a" * 40,
    )
    arguments = _dispatch_args(_FIRST_OBJECT)
    arguments.relay_process = "lab-pipe"
    arguments.endpoint_profile = "hf-mirror"
    arguments.lab_python_profile = "cjl-python312"
    _, _, items = dispatcher._load_specs(arguments)
    binding = items[0][1]
    reception = {
        "schema_version": "stage0-asset-stream-reception-plan-v1",
        "source_id": binding.object_id,
        "revision": binding.revision,
        "expected_size": binding.expected_size,
        "expected_sha256": binding.expected_sha256,
        "offset": 0,
        "already_ready": False,
    }
    ready = {
        "schema_version": relay.PROTOCOL_VERSION,
        "phase": "READY",
        "object_id": binding.object_id,
        "binding": binding.to_dict(),
        "reception": reception,
        "runtime_urls_persisted": False,
    }
    plan_payload = "".join(
        json.dumps(value, separators=(",", ":")) + "\n"
        for value in (ready, {**ready, "phase": "PLAN_COMPLETE"})
    ).encode()
    result = {
        "schema_version": "stage0-asset-acquisition-result-v1",
        "status": "downloaded",
        "source_id": binding.object_id,
        "revision": binding.revision,
        "size_bytes": binding.expected_size,
        "sha256": binding.expected_sha256,
        "attempts": 1,
        "resumed": False,
        "network_accessed": True,
    }
    complete = {
        "schema_version": relay.PROTOCOL_VERSION,
        "phase": "COMPLETE",
        "object_id": binding.object_id,
        "binding": binding.to_dict(),
        "result": result,
        "runtime_urls_persisted": False,
    }
    pipe_payload = "".join(
        json.dumps(value, separators=(",", ":")) + "\n"
        for value in (ready, complete)
    ).encode()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        assert kwargs["check"] is False
        assert kwargs["stdout"] is dispatcher.subprocess.PIPE
        assert kwargs["stderr"] is None
        assert kwargs["timeout"] > 0
        calls.append(command)
        payload = plan_payload if "--plan-only" in command[-1] else pipe_payload
        return SimpleNamespace(returncode=0, stdout=payload)

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    results = dispatcher.dispatch(arguments)

    assert len(calls) == 2
    assert calls[0][0] == "ssh" and "-n" in calls[0]
    assert Path(calls[1][0]).name.casefold() == "cmd.exe"
    assert results[0]["result_status"] == "downloaded"
    assert results[0]["relay_process"] == "lab-pipe"


def test_emit_mode_is_single_attempt_and_lab_direct_only() -> None:
    _, _, items = dispatcher._load_specs(_dispatch_args(_FIRST_OBJECT))
    binding = items[0][1]
    values = {
        key: value
        for key, value in binding.to_dict().items()
        if key != "schema_version"
    }
    values["route"] = "lab-direct"
    arguments = argparse.Namespace(
        **values,
        max_attempts=1,
        request_timeout_seconds=1.0,
        overall_timeout_seconds=2.0,
        chunk_size=1,
        server_alias=relay.SERVER_ALIAS,
        endpoint_profile="hf-mirror",
        relay_mode="emit",
        emit_offset=0,
    )
    relay._validate(arguments)
    arguments.max_attempts = 2
    with pytest.raises(relay.RelayError, match="EMIT_MODE_CONTRACT_INVALID"):
        relay._validate(arguments)


def test_dispatcher_default_selection_is_the_exact_thirteen_object_freeze() -> None:
    relay_path, script, specs = dispatcher._load_specs(_dispatch_args())
    assert relay_path.is_file()
    assert script
    assert len(specs) == 13
    assert len({spec.source_id for spec, _binding in specs}) == 13
    assert {binding.route for _spec, binding in specs} == {"local-via-lab"}


def test_lab_relay_receiver_command_uses_only_documented_alias_and_stable_id() -> None:
    lab_arguments = _dispatch_args(_FIRST_OBJECT)
    lab_arguments.relay_process = "lab"
    _, _, lab_items = dispatcher._load_specs(lab_arguments)
    lab_binding = lab_items[0][1]
    arguments = argparse.Namespace(
        **{
            key: value
            for key, value in lab_binding.to_dict().items()
            if key != "schema_version"
        },
        server_alias=relay.SERVER_ALIAS,
        overall_timeout_seconds=30.0,
    )
    command = relay._receiver_command(arguments)
    assert relay.SERVER_ALIAS in command
    assert _FIRST_OBJECT in command[-1]
    assert all("://" not in item and "?" not in item for item in command)
    assert "ProxyCommand" not in " ".join(command)

    _, _, local_items = dispatcher._load_specs(_dispatch_args(_FIRST_OBJECT))
    local_binding = local_items[0][1]
    local_arguments = argparse.Namespace(
        **{
            key: value
            for key, value in local_binding.to_dict().items()
            if key != "schema_version"
        },
        server_alias=relay.LOCAL_SERVER_ALIAS,
        overall_timeout_seconds=30.0,
    )
    local_command = relay._receiver_command(local_arguments)
    assert relay.LOCAL_SERVER_ALIAS in local_command
    assert all("://" not in item and "?" not in item for item in local_command)


def test_lab_relay_rejects_runtime_url_as_object_identity() -> None:
    arguments = argparse.Namespace(
        object_id="https://example.invalid/object",
        revision="a" * 40,
        expected_size=1,
        expected_sha256="b" * 64,
        max_attempts=1,
        request_timeout_seconds=1.0,
        overall_timeout_seconds=2.0,
        chunk_size=1,
        server_alias=relay.LOCAL_SERVER_ALIAS,
        requirements_sha256="c" * 64,
        layout_sha256="d" * 64,
        plan_sha256="e" * 64,
        spec_ref="configs/stage0/http-objects/object.json",
        spec_sha256="f" * 64,
        asset_root_ref="models/object",
        final_path="object.bin",
        generator_git_commit="1" * 40,
        source_git_commit="2" * 40,
        route="local-via-lab",
    )
    try:
        relay._validate(arguments)
    except relay.RelayError as error:
        assert error.code == "OBJECT_ID_INVALID"
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("runtime URL was accepted as a stable object ID")


def test_relay_route_must_match_the_documented_ssh_alias() -> None:
    _, _, items = dispatcher._load_specs(_dispatch_args(_FIRST_OBJECT))
    binding = items[0][1]
    arguments = argparse.Namespace(
        **{
            key: value
            for key, value in binding.to_dict().items()
            if key != "schema_version"
        },
        max_attempts=1,
        request_timeout_seconds=1.0,
        overall_timeout_seconds=2.0,
        chunk_size=1,
        server_alias=relay.SERVER_ALIAS,
    )
    with pytest.raises(relay.RelayError) as caught:
        relay._validate(arguments)
    assert caught.value.code == "ROUTE_ALIAS_MISMATCH"


def test_receiver_rejects_a_target_binding_drift(tmp_path: Path) -> None:
    _, _, items = dispatcher._load_specs(_dispatch_args(_FIRST_OBJECT))
    arguments = _receiver_args(tmp_path, items[0][1], final_path="other.json")
    with pytest.raises(receiver.G3StreamReceiverError, match="frozen binding"):
        receiver._resolve_request(arguments)


@pytest.mark.parametrize(
    "field",
    [
        "requirements_sha256",
        "layout_sha256",
        "plan_sha256",
        "spec_sha256",
        "source_git_commit",
    ],
)
def test_receiver_rejects_every_control_identity_drift(
    tmp_path: Path,
    field: str,
) -> None:
    _, _, items = dispatcher._load_specs(_dispatch_args(_FIRST_OBJECT))
    original = getattr(items[0][1], field)
    replacement = ("0" if original[0] != "0" else "1") + original[1:]
    arguments = _receiver_args(tmp_path, items[0][1], **{field: replacement})
    with pytest.raises(receiver.G3StreamReceiverError, match="frozen binding"):
        receiver._resolve_request(arguments)


def test_relay_preserves_incomplete_read_partial_for_the_next_resume() -> None:
    class PartialResponse:
        def read(self, _size: int) -> bytes:
            raise http.client.IncompleteRead(b"prefix", 4)

    process = SimpleNamespace(stdin=BytesIO())
    arguments = SimpleNamespace(
        object_id=_FIRST_OBJECT,
        chunk_size=64,
    )
    with pytest.raises(relay.RelayError) as caught:
        relay._stream_response(
            PartialResponse(),
            process,
            arguments=arguments,
            expected_bytes=10,
            deadline=time.monotonic() + 1.0,
        )
    assert caught.value.code == "HTTP_TRANSFER_INCOMPLETE"
    assert process.stdin.getvalue() == b"prefix"


def test_protocol_read_has_a_monotonic_deadline() -> None:
    class BlockingStream:
        def readline(self) -> bytes:
            time.sleep(0.25)
            return b""

    started = time.monotonic()
    with pytest.raises(relay.RelayError) as caught:
        relay._protocol_line(
            BlockingStream(),  # type: ignore[arg-type]
            expected_phase="READY",
            deadline=time.monotonic() + 0.02,
        )
    assert caught.value.code == "OVERALL_TIMEOUT"
    assert time.monotonic() - started < 0.2


def test_receiver_pipe_write_has_the_same_monotonic_deadline() -> None:
    class BlockingPipe:
        closed = False

        def write(self, _payload: bytes) -> int:
            time.sleep(0.25)
            return 1

        def flush(self) -> None:
            return None

    process = SimpleNamespace(stdin=BlockingPipe())
    started = time.monotonic()
    with pytest.raises(relay.RelayError) as caught:
        relay._write_receiver(
            process,
            b"x",
            deadline=time.monotonic() + 0.02,
        )
    assert caught.value.code == "OVERALL_TIMEOUT"
    assert time.monotonic() - started < 0.2


def test_dispatcher_rejects_failed_protocol_even_with_zero_exit() -> None:
    _, _, items = dispatcher._load_specs(_dispatch_args(_FIRST_OBJECT))
    binding = items[0][1]
    failed = {
        "schema_version": relay.PROTOCOL_VERSION,
        "phase": "FAILED",
        "object_id": binding.object_id,
        "failure": {"status": "failed", "code": "NETWORK_ERROR"},
        "runtime_urls_persisted": False,
    }
    with pytest.raises(dispatcher.G3RelayDispatchError, match="structured failure"):
        dispatcher._parse_relay_result(
            (json.dumps(failed) + "\n").encode(),
            binding=binding,
            returncode=0,
        )


def test_dispatcher_rejects_complete_envelope_with_failed_result_status() -> None:
    _, _, items = dispatcher._load_specs(_dispatch_args(_FIRST_OBJECT))
    binding = items[0][1]
    result = {
        "schema_version": "stage0-asset-acquisition-result-v1",
        "status": "failed",
        "source_id": binding.object_id,
        "revision": binding.revision,
        "size_bytes": binding.expected_size,
        "sha256": binding.expected_sha256,
        "attempts": 1,
        "resumed": False,
        "network_accessed": True,
    }
    complete = {
        "schema_version": relay.PROTOCOL_VERSION,
        "phase": "COMPLETE",
        "object_id": binding.object_id,
        "binding": binding.to_dict(),
        "result": result,
        "runtime_urls_persisted": False,
    }
    with pytest.raises(dispatcher.G3RelayDispatchError, match="result does not match"):
        dispatcher._parse_relay_result(
            (json.dumps(complete) + "\n").encode(),
            binding=binding,
            returncode=0,
        )


def test_dispatcher_turns_subprocess_timeout_into_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dispatcher,
        "resolve_source_git_commit",
        lambda _root, **_kwargs: "a" * 40,
    )

    def expire(command: list[str], **kwargs: Any) -> Any:
        raise dispatcher.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(dispatcher.subprocess, "run", expire)
    arguments = _dispatch_args(_FIRST_OBJECT)
    arguments.overall_timeout_seconds = 0.2
    with pytest.raises(dispatcher.G3RelayDispatchError, match="deadline expired"):
        dispatcher.dispatch(arguments)
