"""两阶段 checkpoint 对象与权威 commit 存储。

对象目录发布只证明字节完整；只有独立、规范 JSON 的 commit 记录发布后，该对象
才会进入恢复发现集合。这样即使进程在对象 rename、commit 或 ``latest`` 更新之间
中断，恢复入口也不会误选半完成状态，派生索引则可由 commit 集合重建。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..atomic import atomic_write_json, stable_json_bytes, stable_json_hash
from ._jsonio import load_canonical_json
from .tensor_bundle import load_tensor_bundle, publish_tensor_bundle


COMMIT_SCHEMA = "runtime.checkpoint-commit.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class CheckpointCommit:
    checkpoint_id: str
    generation: int
    manifest_sha256: str
    parent_checkpoint_id: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CheckpointRetentionPolicy:
    """只决定保留集合、不执行物理删除的 retention 合同。

    最新 checkpoint 按 ``(generation, checkpoint_id)`` 排序保留；best、milestone
    和被活动 run/派生实验引用的 checkpoint 必须由调用方显式给出不可歧义 ID。
    默认不保留整个 parent 链，避免每步 checkpoint 都串联时策略退化为“全部保留”；
    如研究需要完整 lineage，可显式启用 ``keep_lineage_ancestors``。
    """

    keep_latest: int = 1
    best_checkpoint_ids: tuple[str, ...] = ()
    milestone_checkpoint_ids: tuple[str, ...] = ()
    protected_checkpoint_ids: tuple[str, ...] = ()
    keep_lineage_ancestors: bool = False
    policy_version: str = "runtime.checkpoint-retention-policy.v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.keep_latest, bool)
            or not isinstance(self.keep_latest, int)
            or self.keep_latest < 1
        ):
            raise ValueError("CHECKPOINT_RETENTION_KEEP_LATEST_INVALID")
        for field_name in (
            "best_checkpoint_ids",
            "milestone_checkpoint_ids",
            "protected_checkpoint_ids",
        ):
            values = tuple(getattr(self, field_name))
            if len(values) != len(set(values)):
                raise ValueError(f"CHECKPOINT_RETENTION_DUPLICATE_ID:{field_name}")
            for value in values:
                CheckpointStore._validate_id(value)
            object.__setattr__(self, field_name, values)
        if self.policy_version != "runtime.checkpoint-retention-policy.v1":
            raise ValueError("CHECKPOINT_RETENTION_POLICY_SCHEMA_MISMATCH")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "keep_latest": self.keep_latest,
            "best_checkpoint_ids": list(self.best_checkpoint_ids),
            "milestone_checkpoint_ids": list(self.milestone_checkpoint_ids),
            "protected_checkpoint_ids": list(self.protected_checkpoint_ids),
            "keep_lineage_ancestors": self.keep_lineage_ancestors,
        }


@dataclass(frozen=True, slots=True)
class CheckpointRetentionSelection:
    """绑定完整 commit 集合 hash 的确定性 retention 选择结果。"""

    keep_checkpoint_ids: tuple[str, ...]
    tombstone_checkpoint_ids: tuple[str, ...]
    keep_reasons: Mapping[str, tuple[str, ...]]
    source_commit_hashes: Mapping[str, str]
    policy_hash: str
    selection_hash: str
    schema_version: str = "runtime.checkpoint-retention-selection.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "runtime.checkpoint-retention-selection.v1":
            raise ValueError("CHECKPOINT_RETENTION_SELECTION_SCHEMA_MISMATCH")
        if len(self.keep_checkpoint_ids) != len(set(self.keep_checkpoint_ids)):
            raise ValueError("CHECKPOINT_RETENTION_DUPLICATE_KEEP_ID")
        if len(self.tombstone_checkpoint_ids) != len(
            set(self.tombstone_checkpoint_ids)
        ):
            raise ValueError("CHECKPOINT_RETENTION_DUPLICATE_TOMBSTONE_ID")
        keep = set(self.keep_checkpoint_ids)
        tombstone = set(self.tombstone_checkpoint_ids)
        if keep.intersection(tombstone):
            raise ValueError("CHECKPOINT_RETENTION_SELECTION_OVERLAP")
        if keep.union(tombstone) != set(self.source_commit_hashes):
            raise ValueError("CHECKPOINT_RETENTION_SELECTION_NOT_EXHAUSTIVE")
        if set(self.keep_reasons) != keep:
            raise ValueError("CHECKPOINT_RETENTION_KEEP_REASONS_MISMATCH")
        for checkpoint_id in keep.union(tombstone):
            CheckpointStore._validate_id(checkpoint_id)
        normalized_reasons: dict[str, tuple[str, ...]] = {}
        for checkpoint_id, reasons in self.keep_reasons.items():
            normalized = tuple(reasons)
            if not normalized or any(
                not isinstance(reason, str) or not reason for reason in normalized
            ):
                raise ValueError("CHECKPOINT_RETENTION_EMPTY_KEEP_REASON")
            if len(normalized) != len(set(normalized)):
                raise ValueError("CHECKPOINT_RETENTION_DUPLICATE_KEEP_REASON")
            normalized_reasons[checkpoint_id] = normalized
        for digest in (*self.source_commit_hashes.values(), self.policy_hash, self.selection_hash):
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("CHECKPOINT_RETENTION_INVALID_HASH")
        object.__setattr__(
            self, "keep_reasons", MappingProxyType(normalized_reasons)
        )
        object.__setattr__(
            self, "source_commit_hashes", MappingProxyType(dict(self.source_commit_hashes))
        )
        if stable_json_hash(self._payload_without_hash()) != self.selection_hash:
            raise ValueError("CHECKPOINT_RETENTION_SELECTION_HASH_MISMATCH")

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "keep_checkpoint_ids": list(self.keep_checkpoint_ids),
            "tombstone_checkpoint_ids": list(self.tombstone_checkpoint_ids),
            "keep_reasons": {
                checkpoint_id: list(reasons)
                for checkpoint_id, reasons in sorted(self.keep_reasons.items())
            },
            "source_commit_hashes": dict(sorted(self.source_commit_hashes.items())),
            "policy_hash": self.policy_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 canonical JSON 的完整选择 artifact。"""

        value = self._payload_without_hash()
        value["selection_hash"] = self.selection_hash
        return value


@dataclass(frozen=True, slots=True)
class CheckpointRetentionApplication:
    """一次 tombstone-only 应用结果；``objects_deleted`` 永远为零。"""

    selection_hash: str
    newly_tombstoned: tuple[str, ...]
    already_tombstoned: tuple[str, ...]
    tombstone_paths: tuple[str, ...]
    objects_deleted: int = 0
    schema_version: str = "runtime.checkpoint-retention-application.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "runtime.checkpoint-retention-application.v1":
            raise ValueError("CHECKPOINT_RETENTION_APPLICATION_SCHEMA_MISMATCH")
        if (
            not isinstance(self.selection_hash, str)
            or len(self.selection_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.selection_hash
            )
        ):
            raise ValueError("CHECKPOINT_RETENTION_INVALID_SELECTION_HASH")
        if self.objects_deleted != 0:
            raise ValueError("CHECKPOINT_RETENTION_CORE_MUST_NOT_DELETE_OBJECTS")
        if set(self.newly_tombstoned).intersection(self.already_tombstoned):
            raise ValueError("CHECKPOINT_RETENTION_APPLICATION_OVERLAP")
        if len(self.tombstone_paths) != len(self.newly_tombstoned) + len(
            self.already_tombstoned
        ):
            raise ValueError("CHECKPOINT_RETENTION_PATH_COUNT_MISMATCH")


class CheckpointStore:
    """只发现已提交 checkpoint 的本地文件存储。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.tombstones = self.root / "tombstones"
        for path in (self.objects, self.commits, self.tombstones):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(value: str) -> str:
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ValueError(f"CHECKPOINT_INVALID_ID:{value!r}")
        return value

    def publish(
        self,
        checkpoint_id: str,
        state: Any,
        *,
        generation: int,
        metadata: dict[str, Any],
        parent_checkpoint_id: str | None = None,
    ) -> CheckpointCommit:
        """发布对象并在重新加载校验后发布唯一 commit。"""

        self._validate_id(checkpoint_id)
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("CHECKPOINT_GENERATION_NOT_INTEGER")
        if generation < 0:
            raise ValueError("CHECKPOINT_NEGATIVE_GENERATION")
        if parent_checkpoint_id is not None:
            self._validate_id(parent_checkpoint_id)
            # 只检查 ``commits/<id>.json`` 是否存在仍然可能接受以下坏父节点：
            # 对象字节已损坏、bundle hash 漂移、父链断裂，或父节点已被 retention
            # tombstone。发布子提交前必须把整条父链和每个 bundle 都读完；这样失败
            # 一定发生在创建子对象之前，不会制造看似权威但不可恢复的后代。
            try:
                _, parent_value = self._load_verified_lineage(
                    parent_checkpoint_id,
                    require_target_active=True,
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    f"CHECKPOINT_PARENT_NOT_COMMITTED:{parent_checkpoint_id}"
                ) from exc
            parent_generation = int(parent_value["generation"])
            if generation <= parent_generation:
                raise ValueError(
                    "CHECKPOINT_GENERATION_NOT_STRICTLY_INCREASING:"
                    f"parent={parent_checkpoint_id}:{parent_generation}:"
                    f"child={checkpoint_id}:{generation}"
                )
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) for key in metadata
        ):
            raise ValueError("CHECKPOINT_METADATA_NOT_STRING_KEYED_OBJECT")
        # 在发布大对象之前验证 JSON 可规范编码，失败时不制造无意义孤儿。
        stable_json_bytes(metadata)
        commit_path = self.commits / f"{checkpoint_id}.json"
        if commit_path.exists():
            raise FileExistsError(f"CHECKPOINT_COMMIT_EXISTS:{checkpoint_id}")
        bundle = publish_tensor_bundle(self.objects / checkpoint_id, state)
        _, verified = load_tensor_bundle(bundle.path)
        if verified.manifest_sha256 != bundle.manifest_sha256:
            raise RuntimeError("CHECKPOINT_POST_PUBLISH_HASH_DRIFT")
        value = {
            "schema_version": COMMIT_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "generation": generation,
            "object_relative_path": f"objects/{checkpoint_id}",
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "parent_checkpoint_id": parent_checkpoint_id,
            "metadata": metadata,
            "committed_at": _now(),
        }
        atomic_write_json(commit_path, value)
        # ``latest`` 只是可重建索引，不能因为“最后发布”就覆盖为较旧 generation。
        # reconcile 会逐条验证完整 lineage，再从所有可恢复节点中选择最大者。
        self.reconcile()
        return self._to_commit(value)

    def _read_commit(self, checkpoint_id: str) -> dict[str, Any]:
        self._validate_id(checkpoint_id)
        path = self.commits / f"{checkpoint_id}.json"
        value = load_canonical_json(path)
        if not isinstance(value, dict):
            raise ValueError("CHECKPOINT_COMMIT_ROOT_NOT_OBJECT")
        self._validate_commit_value(value, checkpoint_id=checkpoint_id)
        return value

    @staticmethod
    def _validate_commit_value(value: dict[str, Any], *, checkpoint_id: str) -> None:
        expected = {
            "schema_version",
            "checkpoint_id",
            "generation",
            "object_relative_path",
            "bundle_manifest_sha256",
            "parent_checkpoint_id",
            "metadata",
            "committed_at",
        }
        if set(value) != expected:
            raise ValueError(
                f"CHECKPOINT_COMMIT_FIELDS_MISMATCH:{sorted(set(value)-expected)}"
            )
        if value.get("schema_version") != COMMIT_SCHEMA:
            raise ValueError("CHECKPOINT_COMMIT_SCHEMA_MISMATCH")
        if value.get("checkpoint_id") != checkpoint_id:
            raise ValueError("CHECKPOINT_COMMIT_ID_MISMATCH")
        if value.get("object_relative_path") != f"objects/{checkpoint_id}":
            raise ValueError("CHECKPOINT_OBJECT_PATH_MISMATCH")
        generation = value.get("generation")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("CHECKPOINT_INVALID_GENERATION")
        digest = value.get("bundle_manifest_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("CHECKPOINT_INVALID_BUNDLE_HASH")
        if not isinstance(value.get("metadata"), dict):
            raise ValueError("CHECKPOINT_METADATA_NOT_OBJECT")
        parent_checkpoint_id = value.get("parent_checkpoint_id")
        if parent_checkpoint_id is not None:
            if not isinstance(parent_checkpoint_id, str):
                raise ValueError("CHECKPOINT_PARENT_ID_NOT_STRING")
            CheckpointStore._validate_id(parent_checkpoint_id)
            if parent_checkpoint_id == checkpoint_id:
                raise ValueError("CHECKPOINT_LINEAGE_SELF_CYCLE")
        if not isinstance(value.get("committed_at"), str) or not value["committed_at"]:
            raise ValueError("CHECKPOINT_COMMITTED_AT_INVALID")

    @staticmethod
    def _to_commit(value: dict[str, Any]) -> CheckpointCommit:
        return CheckpointCommit(
            checkpoint_id=value["checkpoint_id"],
            generation=int(value["generation"]),
            manifest_sha256=value["bundle_manifest_sha256"],
            parent_checkpoint_id=value.get("parent_checkpoint_id"),
            metadata=dict(value["metadata"]),
        )

    def _read_tombstone(
        self,
        checkpoint_id: str,
        *,
        commit_value: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """读取并验证 tombstone；不存在时返回 ``None``。

        tombstone 绑定的是 canonical commit hash。调用方在遍历 lineage 时可把
        已读取的 commit 传入，避免重复 I/O；无论哪条路径都不会只凭文件名判断
        “已删除”，因为损坏的 tombstone 本身也不是权威状态。
        """

        path = self.tombstones / f"{checkpoint_id}.json"
        if not path.exists():
            return None
        value = load_canonical_json(path)
        if not isinstance(value, dict):
            raise ValueError("CHECKPOINT_TOMBSTONE_ROOT_NOT_OBJECT")
        expected = {
            "schema_version",
            "checkpoint_id",
            "commit_sha256",
            "reason",
            "recorded_at",
        }
        if set(value) != expected:
            raise ValueError("CHECKPOINT_TOMBSTONE_FIELDS_MISMATCH")
        if value.get("schema_version") != "runtime.checkpoint-tombstone.v1":
            raise ValueError("CHECKPOINT_TOMBSTONE_SCHEMA_MISMATCH")
        if value.get("checkpoint_id") != checkpoint_id:
            raise ValueError("CHECKPOINT_TOMBSTONE_ID_MISMATCH")
        digest = value.get("commit_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("CHECKPOINT_TOMBSTONE_COMMIT_HASH_INVALID")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise ValueError("CHECKPOINT_TOMBSTONE_REASON_INVALID")
        if not isinstance(value.get("recorded_at"), str) or not value["recorded_at"]:
            raise ValueError("CHECKPOINT_TOMBSTONE_RECORDED_AT_INVALID")
        commit = (
            commit_value
            if commit_value is not None
            else self._read_commit(checkpoint_id)
        )
        if value.get("commit_sha256") != stable_json_hash(commit):
            raise ValueError("CHECKPOINT_TOMBSTONE_COMMIT_HASH_MISMATCH")
        return value

    def _is_tombstoned(self, checkpoint_id: str) -> bool:
        return self._read_tombstone(checkpoint_id) is not None

    def _load_verified_lineage(
        self,
        checkpoint_id: str,
        *,
        require_target_active: bool,
    ) -> tuple[Any, dict[str, Any]]:
        """完整验证目标节点到根节点的 commit、tombstone 与 bundle。

        generation 的方向固定为 ``parent < child``，允许中间跳号但不允许相等
        或倒序。已经 tombstone 的祖先仍是合法历史节点：retention 的默认策略可
        只 tombstone 旧祖先而保留可恢复子节点；不过新子提交不能再以 tombstone
        节点为直接父节点，因此 ``publish`` 会令目标父节点必须 active。

        Args:
            checkpoint_id: 要恢复或作为父节点验证的目标 ID。
            require_target_active: 为真时目标自身存在 tombstone 就拒绝；祖先的
                tombstone 仍需完成 hash 校验，但不会使活动后代失效。

        Returns:
            目标节点的状态树与 canonical commit 对象。祖先状态只用于完整性
            验证，不会暴露给调用方。
        """

        self._validate_id(checkpoint_id)
        target_id = checkpoint_id
        current_id = checkpoint_id
        child_id: str | None = None
        child_generation: int | None = None
        visited: set[str] = set()
        path: list[str] = []
        target_state: Any = None
        target_value: dict[str, Any] | None = None

        while True:
            # 先判重再重复读取 commit，才能把人为改写出来的闭环报告为 cycle，
            # 而不是偶然由某个后续校验掩盖。
            if current_id in visited:
                cycle = "->".join((*path, current_id))
                raise ValueError(f"CHECKPOINT_LINEAGE_CYCLE:{cycle}")
            visited.add(current_id)
            path.append(current_id)

            try:
                value = self._read_commit(current_id)
            except FileNotFoundError as exc:
                if child_id is None:
                    raise
                raise ValueError(
                    "CHECKPOINT_LINEAGE_PARENT_MISSING:"
                    f"child={child_id}:parent={current_id}"
                ) from exc

            generation = int(value["generation"])
            if child_generation is not None and generation >= child_generation:
                raise ValueError(
                    "CHECKPOINT_LINEAGE_GENERATION_NOT_INCREASING:"
                    f"parent={current_id}:{generation}:"
                    f"child={child_id}:{child_generation}"
                )

            tombstone = self._read_tombstone(
                current_id,
                commit_value=value,
            )
            if (
                current_id == target_id
                and require_target_active
                and tombstone is not None
            ):
                raise ValueError(f"CHECKPOINT_TOMBSTONED:{current_id}")

            state, bundle = load_tensor_bundle(
                self.root / value["object_relative_path"]
            )
            if bundle.manifest_sha256 != value["bundle_manifest_sha256"]:
                raise ValueError(
                    f"CHECKPOINT_BUNDLE_HASH_MISMATCH:{current_id}"
                )
            if current_id == target_id:
                target_state = state
                target_value = value

            parent_id = value["parent_checkpoint_id"]
            if parent_id is None:
                break
            child_id = current_id
            child_generation = generation
            current_id = parent_id

        if target_value is None:  # pragma: no cover - 循环入口保证至少读取目标一次。
            raise RuntimeError("CHECKPOINT_LINEAGE_TARGET_NOT_LOADED")
        return target_state, target_value

    def load(
        self,
        checkpoint_id: str,
        *,
        expected_metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, CheckpointCommit]:
        """从权威 commit 加载；在验证结束前不向调用方暴露部分状态。"""

        state, value = self._load_verified_lineage(
            checkpoint_id,
            require_target_active=True,
        )
        metadata = dict(value["metadata"])
        if expected_metadata is not None:
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"CHECKPOINT_METADATA_MISMATCH:{key}")
        return state, self._to_commit(value)

    def reconcile(self) -> dict[str, Any]:
        """验证全部 commit，并从最高 generation 重建派生 ``latest``。"""

        valid: list[CheckpointCommit] = []
        invalid: list[dict[str, str]] = []
        tombstoned: list[str] = []
        for path in sorted(self.commits.glob("*.json")):
            checkpoint_id = path.stem
            try:
                is_tombstoned = self._is_tombstoned(checkpoint_id)
                # tombstoned 节点不进入恢复候选，但其 commit、对象和完整父链仍应
                # 可审计。否则损坏 tombstone 会被误报为正常的“已清理”节点。
                _, value = self._load_verified_lineage(
                    checkpoint_id,
                    require_target_active=not is_tombstoned,
                )
            except Exception as exc:  # 诊断必须保留每个坏引用，而不是静默跳过。
                invalid.append({"checkpoint_id": checkpoint_id, "reason": str(exc)})
            else:
                if is_tombstoned:
                    tombstoned.append(checkpoint_id)
                else:
                    valid.append(self._to_commit(value))
        valid.sort(key=lambda item: (item.generation, item.checkpoint_id))
        if valid:
            latest = valid[-1]
            value = self._read_commit(latest.checkpoint_id)
            atomic_write_json(
                self.root / "latest.json",
                {
                    "schema_version": "runtime.checkpoint-latest.v1",
                    "checkpoint_id": latest.checkpoint_id,
                    "commit_sha256": stable_json_hash(value),
                    "generation": latest.generation,
                },
            )
        else:
            # 索引不具有权威性；当不存在可恢复节点时删去旧索引，避免调用方把
            # 上一次 reconcile 的坏子节点误认为当前恢复点。
            (self.root / "latest.json").unlink(missing_ok=True)
        return {
            "schema_version": "runtime.checkpoint-reconcile.v1",
            "valid": [item.checkpoint_id for item in valid],
            "invalid": invalid,
            "tombstoned": tombstoned,
            "orphan_objects": sorted(
                path.name
                for path in self.objects.iterdir()
                if path.is_dir() and not (self.commits / f"{path.name}.json").exists()
            ),
        }

    def discover(self) -> tuple[CheckpointCommit, ...]:
        """返回所有未 tombstone 且可完整恢复的提交，按 generation/ID 排序。"""

        commits: list[CheckpointCommit] = []
        for path in sorted(self.commits.glob("*.json")):
            if self._is_tombstoned(path.stem):
                continue
            _, commit = self.load(path.stem)
            commits.append(commit)
        return tuple(sorted(commits, key=lambda item: (item.generation, item.checkpoint_id)))

    def load_latest(
        self,
        *,
        expected_metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, CheckpointCommit]:
        """通过派生 ``latest`` 索引加载，并重新验证全部活动候选与完整父链。

        该方法绝不把 ``latest.json`` 当作权威来源。索引的 commit hash、generation
        和最高可恢复候选必须同时一致。坏子节点会被排除在 recoverable 集合外；
        它仍会由 ``reconcile`` 报告为 invalid，但不会阻止已经重建的索引恢复到
        最近一个健康祖先。若索引自身指向坏节点，本方法直接 fail-closed。
        """

        index = load_canonical_json(self.root / "latest.json")
        if not isinstance(index, dict):
            raise ValueError("CHECKPOINT_LATEST_ROOT_NOT_OBJECT")
        expected_fields = {
            "schema_version",
            "checkpoint_id",
            "commit_sha256",
            "generation",
        }
        if set(index) != expected_fields:
            raise ValueError("CHECKPOINT_LATEST_FIELDS_MISMATCH")
        if index.get("schema_version") != "runtime.checkpoint-latest.v1":
            raise ValueError("CHECKPOINT_LATEST_SCHEMA_MISMATCH")
        checkpoint_id = index.get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            raise ValueError("CHECKPOINT_LATEST_ID_INVALID")
        self._validate_id(checkpoint_id)

        value = self._read_commit(checkpoint_id)
        if index.get("commit_sha256") != stable_json_hash(value):
            raise ValueError("CHECKPOINT_LATEST_COMMIT_HASH_MISMATCH")
        state, indexed_commit = self.load(
            checkpoint_id,
            expected_metadata=expected_metadata,
        )
        if index.get("generation") != indexed_commit.generation:
            raise ValueError("CHECKPOINT_LATEST_GENERATION_MISMATCH")

        recoverable: list[CheckpointCommit] = []
        for path in sorted(self.commits.glob("*.json")):
            try:
                if self._is_tombstoned(path.stem):
                    continue
                _, candidate = self.load(path.stem)
            except Exception:
                # 与 reconcile 一致：坏节点不是恢复候选；诊断详情由 reconcile
                # 的 invalid 数组承载，latest 入口只负责不选择它。
                continue
            recoverable.append(candidate)
        if not recoverable:
            raise ValueError("CHECKPOINT_LATEST_NO_RECOVERABLE_COMMIT")
        highest = max(
            recoverable,
            key=lambda item: (item.generation, item.checkpoint_id),
        )
        if checkpoint_id != highest.checkpoint_id:
            raise ValueError(
                "CHECKPOINT_LATEST_NOT_HIGHEST_RECOVERABLE:"
                f"indexed={checkpoint_id}:expected={highest.checkpoint_id}"
            )
        return state, indexed_commit

    def write_tombstone(self, checkpoint_id: str, *, reason: str) -> Path:
        """只发布删除意图记录；本核心接口本身不递归删除对象。"""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("CHECKPOINT_TOMBSTONE_REASON_REQUIRED")
        target = self.tombstones / f"{checkpoint_id}.json"
        if target.exists():
            raise FileExistsError(f"CHECKPOINT_TOMBSTONE_EXISTS:{checkpoint_id}")
        # 直接入口同样只能 tombstone 可完整恢复的活动节点；不能用删除意图掩盖
        # 已损坏对象或断裂 lineage。
        _, _ = self.load(checkpoint_id)
        value = self._read_commit(checkpoint_id)
        atomic_write_json(
            target,
            {
                "schema_version": "runtime.checkpoint-tombstone.v1",
                "checkpoint_id": checkpoint_id,
                "commit_sha256": stable_json_hash(value),
                "reason": reason,
                "recorded_at": _now(),
            },
        )
        return target

    def select_retention(
        self,
        policy: CheckpointRetentionPolicy,
    ) -> CheckpointRetentionSelection:
        """在完整、可恢复的活动 commit 集合上生成确定性保留选择。

        选择阶段完全只读。每个来源 commit 的 canonical SHA-256 都进入选择
        artifact，随后应用时如果集合或内容发生变化会 fail-closed。显式声明为
        best、milestone 或 protected 的 ID 必须仍是活动 checkpoint；核心层不会
        猜测“最佳”指标，也不会把已经 tombstone 的对象悄悄复活。

        Args:
            policy: 保留最新数量、显式保护集合及 lineage 祖先策略。

        Returns:
            同时穷尽 keep/tombstone 集合并绑定输入 commit hash 的不可变选择。

        Raises:
            ValueError: 无活动 checkpoint、显式 ID 不存在、lineage 不完整或
                checkpoint artifact 损坏时抛出。
        """

        if not isinstance(policy, CheckpointRetentionPolicy):
            raise TypeError("CHECKPOINT_RETENTION_POLICY_REQUIRED")
        commits = self.discover()
        if not commits:
            raise ValueError("CHECKPOINT_RETENTION_NO_ACTIVE_CHECKPOINT")

        ordered_ids = tuple(commit.checkpoint_id for commit in commits)
        active_ids = set(ordered_ids)
        values = {
            checkpoint_id: self._read_commit(checkpoint_id)
            for checkpoint_id in ordered_ids
        }
        source_hashes = {
            checkpoint_id: stable_json_hash(values[checkpoint_id])
            for checkpoint_id in ordered_ids
        }
        reasons: dict[str, set[str]] = {}

        def protect(checkpoint_id: str, reason: str) -> None:
            if checkpoint_id not in active_ids:
                raise ValueError(
                    f"CHECKPOINT_RETENTION_REFERENCED_ID_NOT_ACTIVE:{checkpoint_id}"
                )
            reasons.setdefault(checkpoint_id, set()).add(reason)

        for checkpoint_id in ordered_ids[-policy.keep_latest :]:
            protect(checkpoint_id, "latest")
        for checkpoint_id in policy.best_checkpoint_ids:
            protect(checkpoint_id, "best")
        for checkpoint_id in policy.milestone_checkpoint_ids:
            protect(checkpoint_id, "milestone")
        for checkpoint_id in policy.protected_checkpoint_ids:
            protect(checkpoint_id, "protected")

        if policy.keep_lineage_ancestors:
            # 遍历时 reasons 可能新增祖先，因此使用队列而不是对 dict 动态迭代。
            queue = list(reasons)
            visited: set[str] = set()
            while queue:
                child_id = queue.pop(0)
                if child_id in visited:
                    continue
                visited.add(child_id)
                parent_id = values[child_id].get("parent_checkpoint_id")
                if parent_id is None:
                    continue
                if not isinstance(parent_id, str) or parent_id not in active_ids:
                    raise ValueError(
                        "CHECKPOINT_RETENTION_LINEAGE_ANCESTOR_NOT_ACTIVE:"
                        f"{child_id}:{parent_id}"
                    )
                was_new = parent_id not in reasons
                reasons.setdefault(parent_id, set()).add("lineage_ancestor")
                if was_new:
                    queue.append(parent_id)

        keep_ids = tuple(
            checkpoint_id for checkpoint_id in ordered_ids if checkpoint_id in reasons
        )
        tombstone_ids = tuple(
            checkpoint_id
            for checkpoint_id in ordered_ids
            if checkpoint_id not in reasons
        )
        normalized_reasons = {
            checkpoint_id: tuple(sorted(reasons[checkpoint_id]))
            for checkpoint_id in keep_ids
        }
        policy_hash = stable_json_hash(policy.to_dict())
        payload = {
            "schema_version": "runtime.checkpoint-retention-selection.v1",
            "keep_checkpoint_ids": list(keep_ids),
            "tombstone_checkpoint_ids": list(tombstone_ids),
            "keep_reasons": {
                checkpoint_id: list(reason_values)
                for checkpoint_id, reason_values in sorted(
                    normalized_reasons.items()
                )
            },
            "source_commit_hashes": dict(sorted(source_hashes.items())),
            "policy_hash": policy_hash,
        }
        return CheckpointRetentionSelection(
            keep_checkpoint_ids=keep_ids,
            tombstone_checkpoint_ids=tombstone_ids,
            keep_reasons=normalized_reasons,
            source_commit_hashes=source_hashes,
            policy_hash=policy_hash,
            selection_hash=stable_json_hash(payload),
        )

    def apply_retention(
        self,
        selection: CheckpointRetentionSelection,
        *,
        reason: str = "checkpoint retention policy",
    ) -> CheckpointRetentionApplication:
        """幂等应用冻结选择，仅发布 tombstone，不删除 commit 或对象字节。

        应用前会重新校验所有来源 commit hash 和当前活动集合。若选择后又发布了
        新 checkpoint、已有 commit 被改写，或某个应保留 ID 被外部 tombstone，
        本方法会拒绝继续。循环中断后可重放同一选择：已经写完的 tombstone 被
        识别为 ``already_tombstoned``，其余候选继续发布。

        ``objects/`` 与 ``commits/`` 下的内容始终保持不变；物理清理由更高层、
        带额外授权和审计的工具负责，不能从本核心 API 隐式触发。
        """

        if not isinstance(selection, CheckpointRetentionSelection):
            raise TypeError("CHECKPOINT_RETENTION_SELECTION_REQUIRED")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("CHECKPOINT_RETENTION_REASON_REQUIRED")

        source_ids = set(selection.source_commit_hashes)
        for checkpoint_id, expected_hash in selection.source_commit_hashes.items():
            try:
                value = self._read_commit(checkpoint_id)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"CHECKPOINT_RETENTION_SOURCE_COMMIT_INVALID:{checkpoint_id}"
                ) from exc
            if stable_json_hash(value) != expected_hash:
                raise ValueError(
                    f"CHECKPOINT_RETENTION_SOURCE_COMMIT_DRIFT:{checkpoint_id}"
                )

        candidate_states = {
            checkpoint_id: self._is_tombstoned(checkpoint_id)
            for checkpoint_id in selection.tombstone_checkpoint_ids
        }
        for checkpoint_id in selection.keep_checkpoint_ids:
            if self._is_tombstoned(checkpoint_id):
                raise ValueError(
                    f"CHECKPOINT_RETENTION_KEEP_BECAME_TOMBSTONED:{checkpoint_id}"
                )

        # discover 会完整验证活动对象；额外 checkpoint 会使选择陈旧，必须重新选。
        active_ids = {commit.checkpoint_id for commit in self.discover()}
        expected_active = set(selection.keep_checkpoint_ids).union(
            checkpoint_id
            for checkpoint_id, tombstoned in candidate_states.items()
            if not tombstoned
        )
        if active_ids != expected_active or not active_ids.issubset(source_ids):
            raise ValueError(
                "CHECKPOINT_RETENTION_ACTIVE_SET_DRIFT:"
                f"expected={sorted(expected_active)}:observed={sorted(active_ids)}"
            )

        newly_tombstoned: list[str] = []
        already_tombstoned: list[str] = []
        paths: list[str] = []
        audit_reason = (
            f"{reason.strip()}; policy_hash={selection.policy_hash}; "
            f"selection_hash={selection.selection_hash}"
        )
        for checkpoint_id in selection.tombstone_checkpoint_ids:
            path = self.tombstones / f"{checkpoint_id}.json"
            if candidate_states[checkpoint_id]:
                already_tombstoned.append(checkpoint_id)
            else:
                path = self.write_tombstone(checkpoint_id, reason=audit_reason)
                newly_tombstoned.append(checkpoint_id)
            paths.append(path.relative_to(self.root).as_posix())

        # latest 是派生索引；完成或重放应用后都从剩余权威 commit 集合重建。
        self.reconcile()
        return CheckpointRetentionApplication(
            selection_hash=selection.selection_hash,
            newly_tombstoned=tuple(newly_tombstoned),
            already_tombstoned=tuple(already_tombstoned),
            tombstone_paths=tuple(paths),
            objects_deleted=0,
        )
