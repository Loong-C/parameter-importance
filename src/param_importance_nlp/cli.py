from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .capacity import StorageBudget, check_launch_storage
from .git_guard import format_findings, scan_repository
from .storage import REQUIRED_DIRECTORIES, StorageLayout, run_storage_canary


def _git_guard(arguments: argparse.Namespace) -> int:
    findings = scan_repository(
        arguments.repo,
        max_bytes=arguments.max_bytes,
        allowlist=arguments.allow,
    )
    if findings:
        print(format_findings(findings))
        return 1
    print("git-guard: PASS")
    return 0


def _storage_check(arguments: argparse.Namespace) -> int:
    layout = StorageLayout.from_value(arguments.data_root)
    failures = layout.validate(require_writable=arguments.require_writable)
    report: dict[str, object] = {
        "schema_version": "stage0.storage-check.v1",
        "data_root": str(layout.root),
        "directories": list(REQUIRED_DIRECTORIES),
        "failures": failures,
        "canaries": [],
    }
    if not failures and arguments.canary:
        report["canaries"] = [
            run_storage_canary(layout, name) for name in REQUIRED_DIRECTORIES
        ]
    report["ok"] = not failures and all(
        item["ok"] for item in report["canaries"]  # type: ignore[index,union-attr]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _storage_budget_check(arguments: argparse.Namespace) -> int:
    budget = StorageBudget.from_expected(arguments.name, arguments.expected_new_bytes)
    report = {
        "schema_version": "stage0.storage-launch-check.v1",
        "budget": budget.as_dict(),
        "measurement": check_launch_storage(
            data_root=arguments.data_root,
            root_filesystem=arguments.root_filesystem,
            budget=budget,
            root_minimum_free_bytes=arguments.root_minimum_free_bytes,
        ),
    }
    report["ok"] = report["measurement"]["ok"]  # type: ignore[index]
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="param-importance-stage0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    guard = subparsers.add_parser("git-guard")
    guard.add_argument("--repo", type=Path, default=Path.cwd())
    guard.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024)
    guard.add_argument("--allow", action="append", default=[])
    guard.set_defaults(handler=_git_guard)

    storage = subparsers.add_parser("storage-check")
    storage.add_argument("--data-root", type=Path)
    storage.add_argument("--require-writable", action="store_true")
    storage.add_argument("--canary", action="store_true")
    storage.set_defaults(handler=_storage_check)

    budget = subparsers.add_parser("storage-budget-check")
    budget.add_argument("--data-root", type=Path, required=True)
    budget.add_argument("--root-filesystem", type=Path, required=True)
    budget.add_argument("--name", required=True)
    budget.add_argument("--expected-new-bytes", type=int, required=True)
    budget.add_argument(
        "--root-minimum-free-bytes", type=int, default=10 * 1024 * 1024 * 1024
    )
    budget.set_defaults(handler=_storage_budget_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    sys.exit(main())
