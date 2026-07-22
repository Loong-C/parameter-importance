"""重建 Stage 0--9 本机合同冻结集合。

此脚本只读取仓库内的源码、schema、计划和 resolved fixture 配置，计算原始文件
SHA-256，再通过公共 :class:`ContractFreeze` 合同发布 canonical JSON。它不会读取
服务器、不会执行训练，也不会生成 formal Gate。时间戳必须由调用者显式传入，
避免同一源树因“当前时间”产生不同 artifact。

运行示例::

    python -m ops.local_contracts \
      --repository . \
      --config configs/local-fixtures/resolved-config-v1.json \
      --output configs/local-fixtures/local-contract-freezes-v1.json \
      --frozen-at 2026-07-22T00:00:00Z
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Final, Iterable

from param_importance_nlp.contracts import (
    ContractFreeze,
    ContractState,
    ResolvedConfig,
    canonical_json_hash,
    ensure_json_object,
    load_canonical_json,
    write_canonical_json,
)


_STAGE_SOURCE_PATTERNS: Final[dict[int, tuple[str, ...]]] = {
    0: (
        "plan/stage0/README.md",
        "src/param_importance_nlp/contracts/*.py",
        "src/param_importance_nlp/runtime/*.py",
        "src/param_importance_nlp/local_fixture.py",
        "src/param_importance_nlp/{assets,atomic,cache,capacity,cli,environment,git_guard,lifecycle,storage}.py",
    ),
    1: (
        "docs/mathematics.md",
        "plan/stage1/README.md",
        "src/param_importance_nlp/core/*.py",
    ),
    2: (
        "plan/stage2/README.md",
        "src/param_importance_nlp/experiments/sampling.py",
        "src/param_importance_nlp/experiments/stage2.py",
        "src/param_importance_nlp/providers/*.py",
    ),
    3: (
        "plan/stage3/README.md",
        "src/param_importance_nlp/core/quadrature.py",
        "src/param_importance_nlp/experiments/stage3.py",
    ),
    4: ("plan/general_plan.md", "src/param_importance_nlp/experiments/routes.py"),
    5: ("plan/general_plan.md", "src/param_importance_nlp/experiments/routes.py"),
    6: ("plan/general_plan.md", "src/param_importance_nlp/experiments/routes.py"),
    7: (
        "plan/general_plan.md",
        "src/param_importance_nlp/core/baselines.py",
        "src/param_importance_nlp/core/pruning.py",
        "src/param_importance_nlp/experiments/pruning.py",
    ),
    8: (
        "plan/general_plan.md",
        "src/param_importance_nlp/experiments/ablation.py",
    ),
    9: ("plan/general_plan.md", "src/param_importance_nlp/analysis/*.py"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_braces(pattern: str) -> tuple[str, ...]:
    """展开本脚本使用的一层 ``{a,b}``，不把 shell glob 语义带入命令行。"""

    if "{" not in pattern:
        return (pattern,)
    prefix, remainder = pattern.split("{", 1)
    options, suffix = remainder.split("}", 1)
    return tuple(f"{prefix}{option}{suffix}" for option in options.split(","))


def _resolve_files(repository: Path, patterns: Iterable[str]) -> tuple[Path, ...]:
    """解析受控仓库相对 pattern，并拒绝空匹配、symlink 与路径逃逸。"""

    root = repository.resolve(strict=True)
    resolved: dict[str, Path] = {}
    for raw_pattern in patterns:
        for pattern in _expand_braces(raw_pattern):
            matches = sorted(root.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"LOCAL_FREEZE_SOURCE_PATTERN_EMPTY:{pattern}")
            for path in matches:
                actual = path.resolve(strict=True)
                try:
                    relative = actual.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"LOCAL_FREEZE_SOURCE_ESCAPE:{path}") from exc
                if path.is_symlink() or not actual.is_file():
                    raise ValueError(f"LOCAL_FREEZE_SOURCE_NOT_REGULAR_FILE:{path}")
                logical = relative.as_posix()
                if "__pycache__" not in relative.parts and not logical.endswith(".pyc"):
                    resolved[logical] = actual
    return tuple(resolved[name] for name in sorted(resolved))


def _hash_mapping(repository: Path, files: Iterable[Path]) -> dict[str, str]:
    root = repository.resolve(strict=True)
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in files
    }


def build_local_contract_freeze_set(
    *,
    repository: str | Path,
    config_path: str | Path,
    frozen_at: str,
) -> dict[str, object]:
    """返回本机 scope 的十阶段冻结集合，不写文件。

    ``FROZEN`` 仅表示这里列出的本机数学、schema 和源码合同已内容寻址；正式
    B/M/R、阈值、资产、GPU 与服务器 Gate 不在本 artifact 中，也不会因此获得
    资格。
    """

    root = Path(repository).resolve(strict=True)
    config_target = Path(config_path)
    if not config_target.is_absolute():
        config_target = root / config_target
    raw_config = ensure_json_object(
        load_canonical_json(config_target), field="resolved fixture config"
    )
    config = ResolvedConfig.from_mapping(raw_config)
    if config.section("identity")["run_intent"] != "local_fixture":
        raise ValueError("LOCAL_FREEZE_REQUIRES_LOCAL_FIXTURE_CONFIG")

    common_schema_files = _resolve_files(root, ("schemas/shared/*.json",))
    generator = _resolve_files(root, ("ops/local_contracts.py",))
    freezes: list[dict[str, object]] = []
    for stage in range(10):
        stage_schemas = _resolve_files(root, (f"schemas/stage{stage}/*.json",))
        sources = _resolve_files(root, _STAGE_SOURCE_PATTERNS[stage])
        freeze = ContractFreeze(
            contract_id=f"stage{stage}.contract.local-v1",
            stage=stage,
            scope="local_fixture",
            state=ContractState.FROZEN,
            formula_version=f"stage{stage}-local-core-v1",
            config_hash=config.config_hash,
            schema_hashes=_hash_mapping(root, (*common_schema_files, *stage_schemas)),
            source_hashes=_hash_mapping(root, (*generator, *sources)),
            required_gate_ids=(),
            frozen_at=frozen_at,
        )
        freezes.append(freeze.to_dict())
    payload: dict[str, object] = {
        "schema_version": "local-contract-freezes-v1",
        "scope": "local_fixture",
        "formal_eligible": False,
        "freezes": freezes,
    }
    return payload | {"artifact_hash": canonical_json_hash(payload)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-at", required=True)
    arguments = parser.parse_args(argv)
    artifact = build_local_contract_freeze_set(
        repository=arguments.repository,
        config_path=arguments.config,
        frozen_at=arguments.frozen_at,
    )
    output = arguments.output
    if not output.is_absolute():
        output = arguments.repository.resolve(strict=True) / output
    write_canonical_json(output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
