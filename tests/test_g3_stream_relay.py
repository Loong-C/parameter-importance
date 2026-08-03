from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ops.stage0 import dispatch_g3_relay_via_lab as dispatcher
from ops.stage0 import receive_g3_asset_stream as receiver
from ops.stage0 import relay_g3_object_from_lab as relay


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
    )


def test_receiver_resolves_only_the_frozen_object_target(tmp_path: Path) -> None:
    data_root = tmp_path / "data-root"
    (data_root / "models").mkdir(parents=True)
    (data_root / "datasets").mkdir()
    arguments = argparse.Namespace(
        source_root=_ROOT,
        data_root=data_root,
        requirements=_REQUIREMENTS,
        layout=_LAYOUT,
        plan=_PLAN,
        object_id=_FIRST_OBJECT,
    )

    spec, target = receiver._resolve_request(arguments)

    assert spec.source_id == _FIRST_OBJECT
    assert spec.expected_size == 570
    assert target == data_root / "models/pythia-410m-deduped-step0/config.json"
    assert target.parent.is_dir()


def test_receiver_protocol_emitter_writes_exactly_one_json_line() -> None:
    output = StringIO()
    receiver._emit({"phase": "READY"}, stream=output)
    assert output.getvalue().count("\n") == 1
    assert output.getvalue().splitlines() == ['{"phase":"READY"}']


def test_dispatcher_sends_script_over_stdin_and_keeps_urls_out_of_argv(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[list[str], bytes]] = []

    def fake_run(command: list[str], *, input: bytes, check: bool) -> Any:
        assert check is False
        calls.append((command, input))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    arguments = _dispatch_args(_FIRST_OBJECT)
    arguments.relay_process = "lab"
    results = dispatcher.dispatch(arguments)

    assert len(results) == 1
    assert len(calls) == 1
    command, script = calls[0]
    assert command[0] == "ssh"
    assert dispatcher.LAB_ALIAS in command
    assert _FIRST_OBJECT in command
    assert all("://" not in item and "?" not in item for item in command)
    assert b"def _runtime_url" in script
    assert b"https://huggingface.co/EleutherAI" not in script


def test_dispatcher_default_selection_is_the_exact_thirteen_object_freeze() -> None:
    relay_path, script, specs = dispatcher._load_specs(_dispatch_args())
    assert relay_path.is_file()
    assert script
    assert len(specs) == 13
    assert len({spec.source_id for spec in specs}) == 13


def test_lab_relay_receiver_command_uses_only_documented_alias_and_stable_id() -> None:
    command = relay._receiver_command(_FIRST_OBJECT)
    assert relay.SERVER_ALIAS in command
    assert _FIRST_OBJECT in command[-1]
    assert all("://" not in item and "?" not in item for item in command)
    assert "ProxyCommand" not in " ".join(command)

    local_command = relay._receiver_command(
        _FIRST_OBJECT,
        server_alias=relay.LOCAL_SERVER_ALIAS,
    )
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
    )
    try:
        relay._validate(arguments)
    except relay.RelayError as error:
        assert error.code == "OBJECT_ID_INVALID"
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("runtime URL was accepted as a stable object ID")
