"""Dispatch the URL-free G3 stream relay through the documented SSH aliases."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any

from param_importance_nlp.asset_acquisition import AssetObjectSpec
from param_importance_nlp.asset_download_plan import load_g3_download_plan
from param_importance_nlp.asset_layout import load_stage0_asset_layout
from param_importance_nlp.asset_requirements import load_stage0_asset_requirements
from param_importance_nlp.contracts import ensure_json_object, load_canonical_json
from param_importance_nlp.storage import is_within


LAB_ALIAS = "lab-pc"
_RELAY_SCRIPT_REF = "ops/stage0/relay_g3_object_from_lab.py"


class G3RelayDispatchError(ValueError):
    """Raised before dispatch when the local freeze is incomplete or unsafe."""


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument(
        "--relay-process",
        choices=("local", "lab"),
        default="local",
        help=(
            "Run the in-memory HTTP relay locally (default) or feed it to "
            "python - on lab-pc; both routes use only documented SSH aliases."
        ),
    )
    return parser


def _source_file(source_root: Path, value: str | Path, *, field: str) -> Path:
    supplied = Path(value)
    target = supplied if supplied.is_absolute() else source_root / supplied
    if _is_link_like(target):
        raise G3RelayDispatchError(f"{field} may not be link-like")
    try:
        resolved = target.resolve(strict=True)
    except OSError as error:
        raise G3RelayDispatchError(f"{field} is missing") from error
    if not resolved.is_file() or not is_within(resolved, source_root):
        raise G3RelayDispatchError(f"{field} is outside source root")
    relative = resolved.relative_to(source_root)
    current = source_root
    for part in relative.parts:
        current = current / part
        if _is_link_like(current):
            raise G3RelayDispatchError(f"{field} traverses a link")
    return resolved


def _load_specs(
    arguments: argparse.Namespace,
) -> tuple[Path, bytes, list[AssetObjectSpec]]:
    try:
        source_root = arguments.source_root.resolve(strict=True)
    except OSError as error:
        raise G3RelayDispatchError("source root is missing") from error
    if not source_root.is_dir() or _is_link_like(source_root):
        raise G3RelayDispatchError("source root must be a real directory")
    requirements_path = _source_file(
        source_root, arguments.requirements, field="requirements"
    )
    layout_path = _source_file(source_root, arguments.layout, field="layout")
    plan_path = _source_file(source_root, arguments.plan, field="plan")
    relay_path = _source_file(source_root, _RELAY_SCRIPT_REF, field="relay script")
    requirements = load_stage0_asset_requirements(requirements_path)
    layout = load_stage0_asset_layout(layout_path, requirements=requirements)
    plan = load_g3_download_plan(
        plan_path,
        requirements=requirements,
        layout=layout,
    )
    selected = list(arguments.object_id)
    if len(selected) != len(set(selected)):
        raise G3RelayDispatchError("object_id selections must be unique")
    known = {entry["object_id"] for entry in plan["entries"]}
    if any(value not in known for value in selected):
        raise G3RelayDispatchError("selected object_id is outside the frozen plan")
    entries = [
        dict(entry)
        for entry in plan["entries"]
        if not selected or entry["object_id"] in selected
    ]
    specs: list[AssetObjectSpec] = []
    for entry in entries:
        spec_path = _source_file(
            source_root,
            source_root.joinpath(*PurePosixPath(entry["spec_ref"]).parts),
            field="object spec",
        )
        value = ensure_json_object(load_canonical_json(spec_path), field="object spec")
        spec = AssetObjectSpec.from_mapping(value)
        if spec.source_id != entry["object_id"]:
            raise G3RelayDispatchError("object spec does not match the plan")
        specs.append(spec)
    if not specs:
        raise G3RelayDispatchError("relay selection is empty")
    return relay_path, relay_path.read_bytes(), specs


def _relay_arguments(spec: AssetObjectSpec) -> list[str]:
    return [
        "--object-id",
        spec.source_id,
        "--revision",
        spec.revision,
        "--expected-size",
        str(spec.expected_size),
        "--expected-sha256",
        spec.expected_sha256,
    ]


def _lab_command(spec: AssetObjectSpec) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=180",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=12",
        LAB_ALIAS,
        "python",
        "-",
        *_relay_arguments(spec),
    ]
    if any("://" in argument or "?" in argument for argument in command):
        raise G3RelayDispatchError("dispatcher argv may not contain runtime URLs")
    return command


def _local_command(relay_path: Path, spec: AssetObjectSpec) -> list[str]:
    command = [
        sys.executable,
        str(relay_path),
        *_relay_arguments(spec),
        "--server-alias",
        "sophgo13-via-lab",
    ]
    if any("://" in argument or "?" in argument for argument in command):
        raise G3RelayDispatchError("dispatcher argv may not contain runtime URLs")
    return command


def dispatch(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    relay_path, relay_source, specs = _load_specs(arguments)
    relay_process = getattr(arguments, "relay_process", "local")
    if relay_process not in {"local", "lab"}:
        raise G3RelayDispatchError("relay_process is invalid")
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"relay={index}/{len(specs)} object_id={spec.source_id} "
            f"expected_size={spec.expected_size}",
            flush=True,
        )
        if relay_process == "lab":
            completed = subprocess.run(
                _lab_command(spec),
                input=relay_source,
                check=False,
            )
        else:
            completed = subprocess.run(
                _local_command(relay_path, spec),
                check=False,
            )
        result = {
            "object_id": spec.source_id,
            "expected_size": spec.expected_size,
            "expected_sha256": spec.expected_sha256,
            "returncode": completed.returncode,
        }
        results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(
                f"G3 relay stopped at stable object_id={spec.source_id}"
            )
    return results


def main() -> int:
    arguments = _parser().parse_args()
    try:
        results = dispatch(arguments)
    except (G3RelayDispatchError, OSError, RuntimeError) as error:
        print(
            f"status=FAILED error_type={type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(f"status=PASS objects={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
