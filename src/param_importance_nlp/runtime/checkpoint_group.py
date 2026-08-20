"""Authority layer for complete single-rank and distributed checkpoints.

``CheckpointStore`` makes one rank's tensor object recoverable.  A distributed
run needs one more commit boundary: every rank object must validate before the
checkpoint is discoverable as a group.  This module provides that boundary and
keeps ``latest``, checkpoint events, and lineage as deterministic derived views.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from ..atomic import atomic_write_bytes, atomic_write_json, sha256_file, stable_json_bytes, stable_json_hash
from ._jsonio import load_canonical_json
from .checkpoint import CheckpointStore


GROUP_COMMIT_SCHEMA = "runtime.checkpoint-group-commit.v1"
GROUP_COMMIT_SCHEMA_V2 = "runtime.checkpoint-group-commit.v2"
GROUP_LATEST_SCHEMA = "runtime.checkpoint-group-latest.v1"
GROUP_EVENT_SCHEMA = "runtime.checkpoint-group-event.v1"
GROUP_LINEAGE_SCHEMA = "runtime.checkpoint-group-lineage.v1"


def _hash_is_valid(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _update_state_hash(digest: Any, value: Any) -> None:
    """Hash a trusted primitive/tensor state tree without pickle."""

    try:
        import torch
    except ImportError:  # pragma: no cover - checkpoint runtime requires torch
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(stable_json_bytes(list(tensor.shape)))
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"numpy\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(stable_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
        return
    if isinstance(value, int):
        digest.update(b"int\0" + str(value).encode("ascii") + b"\0")
        return
    if isinstance(value, float):
        digest.update(b"float\0" + value.hex().encode("ascii") + b"\0")
        return
    if isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"str\0" + str(len(payload)).encode("ascii") + b"\0" + payload)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        digest.update(str(len(value)).encode("ascii") + b"\0")
        for item in value:
            _update_state_hash(digest, item)
        return
    if isinstance(value, Mapping):
        keys = list(value)
        if any(isinstance(key, bool) or not isinstance(key, (str, int)) for key in keys):
            raise TypeError("CHECKPOINT_GROUP_STATE_KEY_UNSUPPORTED")
        ordered = sorted(keys, key=lambda key: (0 if isinstance(key, str) else 1, str(key)))
        digest.update(b"dict\0" + str(len(ordered)).encode("ascii") + b"\0")
        for key in ordered:
            _update_state_hash(digest, key)
            _update_state_hash(digest, value[key])
        return
    raise TypeError(f"CHECKPOINT_GROUP_STATE_TYPE_UNSUPPORTED:{type(value).__name__}")


def checkpoint_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_state_hash(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointGroupCommit:
    checkpoint_id: str
    generation: int
    run_id: str
    world_size: int
    parent_checkpoint_id: str | None
    commit_sha256: str
    rank_checkpoints: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]
    schema_version: str = GROUP_COMMIT_SCHEMA
    global_step: int | None = None
    successful_optimizer_step: int | None = None
    skip_count: int | None = None


class CheckpointGroupStore:
    """Validate all rank commits before publishing one recovery authority."""

    def __init__(self, workspace_root: str | Path, group_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve(strict=True)
        candidate = Path(group_root)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        self.root = candidate.resolve()
        try:
            self.root.relative_to(self.workspace_root)
        except ValueError as error:
            raise ValueError("CHECKPOINT_GROUP_ROOT_ESCAPES_WORKSPACE") from error
        self.commits = self.root / "commits"
        self.commits.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(value: object, *, field: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"CHECKPOINT_GROUP_ID_INVALID:{field}")
        CheckpointStore._validate_id(value)
        return value

    def _logical_path(self, value: object, *, field: str) -> Path:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError(f"CHECKPOINT_GROUP_REF_INVALID:{field}")
        logical = PurePosixPath(value)
        if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
            raise ValueError(f"CHECKPOINT_GROUP_REF_ESCAPE:{field}")
        path = self.workspace_root.joinpath(*logical.parts).resolve()
        try:
            path.relative_to(self.workspace_root)
        except ValueError as error:
            raise ValueError(f"CHECKPOINT_GROUP_REF_ESCAPE:{field}") from error
        return path

    @staticmethod
    def _validate_training_state(state: Any, *, expected_generation: int) -> Mapping[str, Any]:
        v1_required = {
            "schema_version",
            "run_spec_hash",
            "registry_hash",
            "optimizer_contract_hash",
            "runtime_layout_hash",
            "training_state",
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "rng",
            "cursor",
            "importance",
            "records",
            "importance_trajectory_points",
        }
        v2_required = v1_required | {"checkpoint_ids"}
        if not isinstance(state, Mapping):
            raise ValueError("CHECKPOINT_GROUP_TRAINING_STATE_FIELDS_INVALID")
        version = state.get("schema_version")
        if version == "training-checkpoint-state-v1":
            if set(state) != v1_required:
                raise ValueError("CHECKPOINT_GROUP_TRAINING_STATE_FIELDS_INVALID")
        elif version == "training-checkpoint-state-v2":
            if set(state) != v2_required:
                raise ValueError("CHECKPOINT_GROUP_TRAINING_STATE_FIELDS_INVALID")
        else:
            raise ValueError("CHECKPOINT_GROUP_TRAINING_STATE_VERSION_INVALID")
        control = state.get("training_state")
        if not isinstance(control, Mapping):
            raise ValueError("CHECKPOINT_GROUP_STEP_BOUNDARY_MISMATCH")
        if version == "training-checkpoint-state-v1" and control.get("global_step") != expected_generation:
            raise ValueError("CHECKPOINT_GROUP_STEP_BOUNDARY_MISMATCH")
        if version == "training-checkpoint-state-v2":
            global_step, attempt_index, skipped_steps = (
                control.get("global_step"),
                control.get("attempt_index"),
                control.get("skipped_steps"),
            )
            if (
                any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (global_step, attempt_index, skipped_steps))
                or global_step + skipped_steps != attempt_index
                or attempt_index != expected_generation
            ):
                raise ValueError("CHECKPOINT_GROUP_ATTEMPT_BOUNDARY_MISMATCH")
            checkpoint_ids = state.get("checkpoint_ids")
            if (
                not isinstance(checkpoint_ids, list)
                or not checkpoint_ids
                or not all(isinstance(item, str) and item for item in checkpoint_ids)
                or len(set(checkpoint_ids)) != len(checkpoint_ids)
                or control.get("last_checkpoint_id") != checkpoint_ids[-1]
            ):
                raise ValueError("CHECKPOINT_GROUP_V2_LINEAGE_INVALID")
        if not isinstance(state.get("rng"), Mapping) or not isinstance(state.get("cursor"), Mapping):
            raise ValueError("CHECKPOINT_GROUP_RNG_OR_CURSOR_MISSING")
        if not isinstance(state.get("records"), list):
            raise ValueError("CHECKPOINT_GROUP_RECORDS_INVALID")
        return state

    @staticmethod
    def _validate_rank_payload_lineage(
        store: CheckpointStore,
        state: Mapping[str, Any],
        *,
        checkpoint_id: str,
    ) -> None:
        """Bind v2's convenience ID list to the immutable rank commit chain."""

        if state.get("schema_version") != "training-checkpoint-state-v2":
            return
        declared = state.get("checkpoint_ids")
        if not isinstance(declared, list):  # guarded above; retain local proof.
            raise ValueError("CHECKPOINT_GROUP_V2_LINEAGE_INVALID")
        chain: list[str] = []
        current: str | None = checkpoint_id
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ValueError("CHECKPOINT_GROUP_RANK_LINEAGE_CYCLE")
            seen.add(current)
            store.load(current)
            commit = store._read_commit(current)  # noqa: SLF001 - exact committed-parent reconstruction
            chain.append(current)
            parent = commit.get("parent_checkpoint_id")
            if parent is not None and not isinstance(parent, str):
                raise ValueError("CHECKPOINT_GROUP_RANK_PARENT_INVALID")
            current = parent
        chain.reverse()
        if declared != chain:
            raise ValueError("CHECKPOINT_GROUP_V2_PAYLOAD_LINEAGE_MISMATCH")

    def _rank_binding(
        self,
        raw: Mapping[str, Any],
        *,
        rank: int,
        world_size: int,
        generation: int,
    ) -> tuple[dict[str, Any], Any]:
        if set(raw) != {"rank", "checkpoint_store_ref", "checkpoint_id", "event_pointer"}:
            raise ValueError("CHECKPOINT_GROUP_RANK_BINDING_FIELDS_INVALID")
        if raw.get("rank") != rank:
            raise ValueError("CHECKPOINT_GROUP_RANK_ORDER_INVALID")
        store_path = self._logical_path(raw.get("checkpoint_store_ref"), field="checkpoint_store_ref")
        checkpoint_id = self._validate_id(raw.get("checkpoint_id"), field="rank_checkpoint_id")
        store = CheckpointStore(store_path)
        state, commit = store.load(checkpoint_id, expected_metadata={"world_size": world_size})
        state = self._validate_training_state(state, expected_generation=generation)
        self._validate_rank_payload_lineage(store, state, checkpoint_id=checkpoint_id)
        if (
            state.get("schema_version") == "training-checkpoint-state-v2"
            and state["training_state"].get("last_checkpoint_id") != checkpoint_id  # type: ignore[index]
        ):
            raise ValueError("CHECKPOINT_GROUP_V2_CURRENT_COMMIT_LINEAGE_MISMATCH")
        commit_path = store.commits / f"{checkpoint_id}.json"
        event = raw.get("event_pointer")
        if not isinstance(event, Mapping) or set(event) != {
            "event_ref", "event_sha256", "checkpoint_event_sequence"
        }:
            raise ValueError("CHECKPOINT_GROUP_EVENT_POINTER_INVALID")
        event_path = self._logical_path(event.get("event_ref"), field="event_ref")
        if not event_path.is_file() or sha256_file(event_path) != event.get("event_sha256"):
            raise ValueError("CHECKPOINT_GROUP_EVENT_STREAM_HASH_MISMATCH")
        sequence = event.get("checkpoint_event_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("CHECKPOINT_GROUP_EVENT_SEQUENCE_INVALID")
        control = state["training_state"]
        if int(control["event_sequence"]) - 1 != sequence:
            raise ValueError("CHECKPOINT_GROUP_EVENT_SEQUENCE_STATE_MISMATCH")
        binding = {
            "rank": rank,
            "checkpoint_store_ref": Path(store_path).relative_to(self.workspace_root).as_posix(),
            "checkpoint_id": checkpoint_id,
            "commit_ref": commit_path.relative_to(self.workspace_root).as_posix(),
            "commit_sha256": sha256_file(commit_path),
            "bundle_manifest_sha256": commit.manifest_sha256,
            "full_state_sha256": checkpoint_state_sha256(state),
            "model_sha256": checkpoint_state_sha256(state["model"]),
            "optimizer_sha256": checkpoint_state_sha256(state["optimizer"]),
            "scheduler_sha256": checkpoint_state_sha256(state["scheduler"]),
            "scaler_sha256": checkpoint_state_sha256(state["scaler"]),
            "importance_sha256": checkpoint_state_sha256(state["importance"]),
            "rng_sha256": checkpoint_state_sha256(state["rng"]),
            "cursor_sha256": checkpoint_state_sha256(state["cursor"]),
            "event_pointer": dict(event),
        }
        return binding, state

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "config_hash",
            "environment_hash",
            "model_manifest_id",
            "data_manifest_id",
            "sampler_seed",
            "epoch",
            "committed_global_batch",
            "next_global_batch",
            "prefetch_policy",
            "snapshot_type",
            "state_extension_schema",
            "save_wall_seconds",
            "checkpoint_bytes",
            "peak_memory_bytes",
        }
        if set(metadata) != required:
            raise ValueError("CHECKPOINT_GROUP_METADATA_FIELDS_INVALID")
        for name in ("config_hash", "environment_hash"):
            if not _hash_is_valid(metadata.get(name)):
                raise ValueError(f"CHECKPOINT_GROUP_METADATA_HASH_INVALID:{name}")
        for name in ("model_manifest_id", "data_manifest_id", "prefetch_policy", "snapshot_type", "state_extension_schema"):
            if not isinstance(metadata.get(name), str) or not str(metadata[name]):
                raise ValueError(f"CHECKPOINT_GROUP_METADATA_STRING_INVALID:{name}")
        for name in ("sampler_seed", "epoch", "committed_global_batch", "next_global_batch", "checkpoint_bytes", "peak_memory_bytes"):
            value = metadata.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"CHECKPOINT_GROUP_METADATA_INTEGER_INVALID:{name}")
        if metadata["next_global_batch"] != metadata["committed_global_batch"]:
            raise ValueError("CHECKPOINT_GROUP_NEXT_BATCH_CURSOR_INVALID")
        wall = metadata.get("save_wall_seconds")
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall < 0:
            raise ValueError("CHECKPOINT_GROUP_SAVE_TIME_INVALID")
        if metadata["snapshot_type"] != "optimizer_step_checkpoint":
            raise ValueError("CHECKPOINT_GROUP_SNAPSHOT_TYPE_INVALID")
        return dict(metadata)

    def publish(
        self,
        checkpoint_id: str,
        *,
        generation: int,
        run_id: str,
        world_size: int,
        rank_checkpoints: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
        parent_checkpoint_id: str | None = None,
        derive_views: bool = True,
        commit_schema_version: str = GROUP_COMMIT_SCHEMA,
    ) -> CheckpointGroupCommit:
        checkpoint_id = self._validate_id(checkpoint_id, field="checkpoint_id")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("CHECKPOINT_GROUP_GENERATION_INVALID")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("CHECKPOINT_GROUP_RUN_ID_INVALID")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("CHECKPOINT_GROUP_WORLD_SIZE_INVALID")
        if len(rank_checkpoints) != world_size:
            raise ValueError("CHECKPOINT_GROUP_RANK_COUNT_INVALID")
        if commit_schema_version not in {GROUP_COMMIT_SCHEMA, GROUP_COMMIT_SCHEMA_V2}:
            raise ValueError("CHECKPOINT_GROUP_COMMIT_SCHEMA_VERSION_INVALID")
        if parent_checkpoint_id is not None:
            parent_checkpoint_id = self._validate_id(parent_checkpoint_id, field="parent")
            parent = self._read_and_validate(parent_checkpoint_id)
            if parent["run_id"] != run_id or parent["world_size"] != world_size:
                raise ValueError("CHECKPOINT_GROUP_PARENT_IDENTITY_MISMATCH")
            if int(parent["generation"]) >= generation:
                raise ValueError("CHECKPOINT_GROUP_GENERATION_NOT_INCREASING")
        commit_path = self.commits / f"{checkpoint_id}.json"
        if commit_path.exists():
            raise FileExistsError(f"CHECKPOINT_GROUP_COMMIT_EXISTS:{checkpoint_id}")
        bindings: list[dict[str, Any]] = []
        states: list[Mapping[str, Any]] = []
        for rank, raw in enumerate(rank_checkpoints):
            binding, state = self._rank_binding(
                raw, rank=rank, world_size=world_size, generation=generation
            )
            bindings.append(binding)
            states.append(state)
        versions = {state.get("schema_version") for state in states}
        if versions not in ({"training-checkpoint-state-v1"}, {"training-checkpoint-state-v2"}):
            raise ValueError("CHECKPOINT_GROUP_TRAINING_STATE_VERSION_DRIFT")
        legacy_equivalent_v2 = (
            versions == {"training-checkpoint-state-v2"}
            and all(
                isinstance(state.get("training_state"), Mapping)
                and state["training_state"].get("skipped_steps") == 0
                and state["training_state"].get("global_step") == generation
                and state["training_state"].get("attempt_index") == generation
                for state in states
            )
        )
        compatible = (
            versions == {"training-checkpoint-state-v2"}
            if commit_schema_version == GROUP_COMMIT_SCHEMA_V2
            else versions == {"training-checkpoint-state-v1"} or legacy_equivalent_v2
        )
        if not compatible:
            raise ValueError(
                "CHECKPOINT_GROUP_COMMIT_AND_TRAINING_STATE_VERSION_MISMATCH"
            )
        for field in (
            "model_sha256",
            "optimizer_sha256",
            "scheduler_sha256",
            "scaler_sha256",
            "importance_sha256",
        ):
            if len({binding[field] for binding in bindings}) != 1:
                raise ValueError(f"CHECKPOINT_GROUP_SHARED_STATE_DRIFT:{field}")
        normalized_metadata = self._validate_metadata(metadata)
        controls = [state["training_state"] for state in states]
        if not all(isinstance(control, Mapping) for control in controls):
            raise ValueError("CHECKPOINT_GROUP_CONTROL_STATE_INVALID")
        version = next(iter(versions))
        if parent_checkpoint_id is not None:
            parent = self._read_and_validate(parent_checkpoint_id)
            expected_parent_schema = commit_schema_version
            if parent.get("schema_version") != expected_parent_schema:
                raise ValueError("CHECKPOINT_GROUP_PARENT_VERSION_MISMATCH")
            if commit_schema_version == GROUP_COMMIT_SCHEMA_V2:
                parent_bindings = parent.get("rank_checkpoints")
                if not isinstance(parent_bindings, list) or len(parent_bindings) != world_size:
                    raise ValueError("CHECKPOINT_GROUP_PARENT_RANK_BINDINGS_INVALID")
                for rank, (state, parent_binding) in enumerate(
                    zip(states, parent_bindings, strict=True)
                ):
                    ids = state.get("checkpoint_ids")
                    parent_rank_id = (
                        parent_binding.get("checkpoint_id")
                        if isinstance(parent_binding, Mapping)
                        else None
                    )
                    if (
                        not isinstance(ids, list)
                        or len(ids) < 2
                        or ids[-2] != parent_rank_id
                        or bindings[rank]["checkpoint_id"] != ids[-1]
                    ):
                        raise ValueError("CHECKPOINT_GROUP_V2_PARENT_RANK_LINEAGE_MISMATCH")
        value: dict[str, Any] = {
            "schema_version": commit_schema_version,
            "checkpoint_id": checkpoint_id,
            "generation": generation,
            "run_id": run_id,
            "world_size": world_size,
            "parent_checkpoint_id": parent_checkpoint_id,
            "last_completed_step": generation,
            "next_step": generation + 1,
            "rank_checkpoints": bindings,
            "shared_state_sha256": {
                field: bindings[0][field]
                for field in (
                    "model_sha256",
                    "optimizer_sha256",
                    "scheduler_sha256",
                    "scaler_sha256",
                    "importance_sha256",
                )
            },
            "metadata": normalized_metadata,
        }
        if commit_schema_version == GROUP_COMMIT_SCHEMA_V2:
            first = controls[0]
            assert isinstance(first, Mapping)
            attempt_index = first.get("attempt_index")
            global_step = first.get("global_step")
            skipped_steps = first.get("skipped_steps")
            if (
                attempt_index != generation
                or any(
                    control.get("attempt_index") != attempt_index
                    or control.get("global_step") != global_step
                    or control.get("skipped_steps") != skipped_steps
                    for control in controls[1:]
                )
            ):
                raise ValueError("CHECKPOINT_GROUP_V2_CONTROL_STATE_DRIFT")
            value.update(
                {
                    "schema_version": GROUP_COMMIT_SCHEMA_V2,
                    # ``generation`` is the complete attempt boundary; it is
                    # intentionally distinct from the successful-step cursor
                    # so a SKIPPED attempt gets its own immutable group commit.
                    "attempt_index": attempt_index,
                    "global_step": global_step,
                    "successful_optimizer_step": global_step,
                    "skip_count": skipped_steps,
                    "next_attempt_index": generation + 1,
                    "last_completed_step": global_step,
                    "next_step": global_step + 1,
                }
            )
        value["commit_sha256"] = stable_json_hash(value)
        atomic_write_json(commit_path, value)
        # Every rank object was fully loaded above.  Re-open the small atomic
        # group commit to catch publication drift without loading all large
        # tensor bundles a second time here; reconcile below performs the
        # independent post-commit full replay used by discovery.
        verified = self._read(checkpoint_id)
        if verified != value:
            raise RuntimeError("CHECKPOINT_GROUP_POST_PUBLISH_DRIFT")
        if derive_views:
            self.reconcile()
        return self._to_commit(value)

    def _read(self, checkpoint_id: str) -> dict[str, Any]:
        path = self.commits / f"{self._validate_id(checkpoint_id, field='load')}.json"
        value = load_canonical_json(path)
        if not isinstance(value, dict):
            raise ValueError("CHECKPOINT_GROUP_COMMIT_NOT_OBJECT")
        return value

    def _read_and_validate(self, checkpoint_id: str) -> dict[str, Any]:
        value = self._read(checkpoint_id)
        v1_expected = {
            "schema_version", "checkpoint_id", "generation", "run_id", "world_size",
            "parent_checkpoint_id", "last_completed_step", "next_step", "rank_checkpoints",
            "shared_state_sha256", "metadata", "commit_sha256",
        }
        v2_expected = v1_expected | {
            "attempt_index", "global_step", "successful_optimizer_step", "skip_count",
            "next_attempt_index",
        }
        version = value.get("schema_version")
        if (
            (version == GROUP_COMMIT_SCHEMA and set(value) != v1_expected)
            or (version == GROUP_COMMIT_SCHEMA_V2 and set(value) != v2_expected)
            or version not in {GROUP_COMMIT_SCHEMA, GROUP_COMMIT_SCHEMA_V2}
        ):
            raise ValueError("CHECKPOINT_GROUP_COMMIT_FIELDS_OR_VERSION_INVALID")
        declared = value.pop("commit_sha256")
        if declared != stable_json_hash(value):
            raise ValueError("CHECKPOINT_GROUP_COMMIT_HASH_MISMATCH")
        value["commit_sha256"] = declared
        if value.get("checkpoint_id") != checkpoint_id:
            raise ValueError("CHECKPOINT_GROUP_COMMIT_ID_MISMATCH")
        generation = value.get("generation")
        world_size = value.get("world_size")
        if (
            isinstance(generation, bool) or not isinstance(generation, int) or generation < 0
            or isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1
        ):
            raise ValueError("CHECKPOINT_GROUP_COMMIT_NUMERIC_FIELDS_INVALID")
        if version == GROUP_COMMIT_SCHEMA:
            if (
                value.get("last_completed_step") != generation
                or value.get("next_step") != generation + 1
            ):
                raise ValueError("CHECKPOINT_GROUP_COMMIT_NUMERIC_FIELDS_INVALID")
        else:
            attempt_index = value.get("attempt_index")
            global_step = value.get("global_step")
            successful_step = value.get("successful_optimizer_step")
            skip_count = value.get("skip_count")
            next_attempt = value.get("next_attempt_index")
            if (
                any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (attempt_index, global_step, successful_step, skip_count, next_attempt))
                or attempt_index != generation
                or global_step != successful_step
                or global_step + skip_count != attempt_index
                or next_attempt != attempt_index + 1
                or value.get("last_completed_step") != global_step
                or value.get("next_step") != global_step + 1
            ):
                raise ValueError("CHECKPOINT_GROUP_V2_ATTEMPT_METADATA_INVALID")
        if not isinstance(value.get("run_id"), str) or not value["run_id"]:
            raise ValueError("CHECKPOINT_GROUP_COMMIT_RUN_ID_INVALID")
        parent_id = value.get("parent_checkpoint_id")
        parent: dict[str, Any] | None = None
        if parent_id is not None:
            parent_id = self._validate_id(parent_id, field="parent")
            parent = self._read_and_validate(parent_id)
            if (
                parent["run_id"] != value["run_id"]
                or parent["world_size"] != world_size
                or int(parent["generation"]) >= generation
                or parent.get("schema_version") != version
            ):
                raise ValueError("CHECKPOINT_GROUP_PARENT_LINEAGE_INVALID")
        raw_bindings = value.get("rank_checkpoints")
        if not isinstance(raw_bindings, list) or len(raw_bindings) != world_size:
            raise ValueError("CHECKPOINT_GROUP_RANK_BINDINGS_INVALID")
        verified_bindings: list[dict[str, Any]] = []
        verified_states: list[Any] = []
        for rank, binding in enumerate(raw_bindings):
            if not isinstance(binding, Mapping):
                raise ValueError("CHECKPOINT_GROUP_RANK_BINDING_INVALID")
            store_ref = binding.get("checkpoint_store_ref")
            reconstructed, state = self._rank_binding(
                {
                    "rank": rank,
                    "checkpoint_store_ref": store_ref,
                    "checkpoint_id": binding.get("checkpoint_id"),
                    "event_pointer": binding.get("event_pointer"),
                },
                rank=rank,
                world_size=world_size,
                generation=generation,
            )
            if reconstructed != dict(binding):
                raise ValueError("CHECKPOINT_GROUP_RANK_BINDING_DRIFT")
            if version == GROUP_COMMIT_SCHEMA_V2:
                control = state.get("training_state")
                if (
                    state.get("schema_version") != "training-checkpoint-state-v2"
                    or not isinstance(control, Mapping)
                    or control.get("attempt_index") != generation
                    or control.get("global_step") != value.get("global_step")
                    or control.get("skipped_steps") != value.get("skip_count")
                ):
                    raise ValueError("CHECKPOINT_GROUP_V2_RANK_CONTROL_DRIFT")
            verified_bindings.append(reconstructed)
            verified_states.append(state)
        if version == GROUP_COMMIT_SCHEMA:
            # A v1 group commit may bridge the v2 rank payload only when its
            # attempt and successful-step cursors are indistinguishable from
            # legacy semantics.  A skipped attempt must remain explicitly v2;
            # otherwise v1's generation-based fields would erase it.
            for state in verified_states:
                control = state["training_state"]
                if state.get("schema_version") == "training-checkpoint-state-v2":
                    if (
                        control.get("attempt_index") != generation
                        or control.get("global_step") != generation
                        or control.get("skipped_steps") != 0
                    ):
                        raise ValueError("CHECKPOINT_GROUP_V1_V2_BRIDGE_NOT_EQUIVALENT")
        if version == GROUP_COMMIT_SCHEMA_V2 and parent is not None:
            parent_bindings = parent["rank_checkpoints"]
            for rank, (state, binding, parent_binding) in enumerate(
                zip(verified_states, verified_bindings, parent_bindings, strict=True)
            ):
                checkpoint_ids = state.get("checkpoint_ids") if isinstance(state, Mapping) else None
                if (
                    not isinstance(checkpoint_ids, list)
                    or len(checkpoint_ids) < 2
                    or checkpoint_ids[-1] != binding["checkpoint_id"]
                    or checkpoint_ids[-2] != parent_binding["checkpoint_id"]
                ):
                    raise ValueError(
                        "CHECKPOINT_GROUP_V2_PARENT_RANK_LINEAGE_MISMATCH"
                    )
        shared = value.get("shared_state_sha256")
        fields = (
            "model_sha256",
            "optimizer_sha256",
            "scheduler_sha256",
            "scaler_sha256",
            "importance_sha256",
        )
        if not isinstance(shared, Mapping) or set(shared) != set(fields) or any(
            shared[field] != verified_bindings[0][field]
            or len({binding[field] for binding in verified_bindings}) != 1
            for field in fields
        ):
            raise ValueError("CHECKPOINT_GROUP_SHARED_STATE_INVALID")
        self._validate_metadata(value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {})
        return value

    @staticmethod
    def _to_commit(value: Mapping[str, Any]) -> CheckpointGroupCommit:
        return CheckpointGroupCommit(
            checkpoint_id=str(value["checkpoint_id"]),
            generation=int(value["generation"]),
            run_id=str(value["run_id"]),
            world_size=int(value["world_size"]),
            parent_checkpoint_id=value.get("parent_checkpoint_id"),
            commit_sha256=str(value["commit_sha256"]),
            rank_checkpoints=tuple(dict(item) for item in value["rank_checkpoints"]),
            metadata=dict(value["metadata"]),
            schema_version=str(value["schema_version"]),
            global_step=(
                int(value["global_step"])
                if value.get("schema_version") == GROUP_COMMIT_SCHEMA_V2
                else None
            ),
            successful_optimizer_step=(
                int(value["successful_optimizer_step"])
                if value.get("schema_version") == GROUP_COMMIT_SCHEMA_V2
                else None
            ),
            skip_count=(
                int(value["skip_count"])
                if value.get("schema_version") == GROUP_COMMIT_SCHEMA_V2
                else None
            ),
        )

    def load(
        self,
        checkpoint_id: str,
        *,
        expected_run_id: str | None = None,
        expected_world_size: int | None = None,
        expected_config_hash: str | None = None,
        expected_data_manifest_id: str | None = None,
    ) -> tuple[tuple[Any, ...], CheckpointGroupCommit]:
        commit = self.verify(
            checkpoint_id,
            expected_run_id=expected_run_id,
            expected_world_size=expected_world_size,
            expected_config_hash=expected_config_hash,
            expected_data_manifest_id=expected_data_manifest_id,
        )
        value = self._read(checkpoint_id)
        # ``verify`` has already loaded and hash-checked every rank object.  The
        # following loads return independent state trees to the caller.
        if value.get("commit_sha256") != commit.commit_sha256:
            raise RuntimeError("CHECKPOINT_GROUP_COMMIT_DRIFT_AFTER_VERIFY")
        states: list[Any] = []
        for binding in value["rank_checkpoints"]:
            store = CheckpointStore(self._logical_path(binding["checkpoint_store_ref"], field="checkpoint_store_ref"))
            state, _ = store.load(binding["checkpoint_id"], expected_metadata={"world_size": value["world_size"]})
            states.append(state)
        return tuple(states), commit

    def verify(
        self,
        checkpoint_id: str,
        *,
        expected_run_id: str | None = None,
        expected_world_size: int | None = None,
        expected_config_hash: str | None = None,
        expected_data_manifest_id: str | None = None,
    ) -> CheckpointGroupCommit:
        """Fully validate every rank object without returning duplicate states."""

        value = self._read_and_validate(checkpoint_id)
        if expected_run_id is not None and value["run_id"] != expected_run_id:
            raise ValueError("CHECKPOINT_GROUP_RUN_ID_INCOMPATIBLE")
        if expected_world_size is not None and value["world_size"] != expected_world_size:
            raise ValueError("CHECKPOINT_GROUP_WORLD_SIZE_INCOMPATIBLE")
        metadata = value["metadata"]
        if expected_config_hash is not None and metadata["config_hash"] != expected_config_hash:
            raise ValueError("CHECKPOINT_GROUP_CONFIG_INCOMPATIBLE")
        if expected_data_manifest_id is not None and metadata["data_manifest_id"] != expected_data_manifest_id:
            raise ValueError("CHECKPOINT_GROUP_DATA_MANIFEST_INCOMPATIBLE")
        return self._to_commit(value)

    def discover(self) -> tuple[CheckpointGroupCommit, ...]:
        commits: list[CheckpointGroupCommit] = []
        for path in sorted(self.commits.glob("*.json")):
            commits.append(self._to_commit(self._read_and_validate(path.stem)))
        return tuple(sorted(commits, key=lambda item: (item.generation, item.checkpoint_id)))

    def reconcile(self) -> dict[str, Any]:
        diagnostics: list[dict[str, str]] = []
        valid: list[dict[str, Any]] = []
        for path in sorted(self.commits.glob("*.json")):
            try:
                valid.append(self._read_and_validate(path.stem))
            except Exception as error:
                diagnostics.append({"checkpoint_id": path.stem, "reason": str(error)})
        valid.sort(key=lambda item: (int(item["generation"]), str(item["checkpoint_id"])))
        latest_path = self.root / "latest.json"
        events_path = self.root / "checkpoint-events.jsonl"
        lineage_path = self.root / "lineage.json"
        events: list[dict[str, Any]] = []
        for value in valid:
            event = {
                "schema_version": GROUP_EVENT_SCHEMA,
                "event_id": stable_json_hash(
                    {"checkpoint_id": value["checkpoint_id"], "commit_sha256": value["commit_sha256"]}
                )[:32],
                "checkpoint_id": value["checkpoint_id"],
                "generation": value["generation"],
                "commit_ref": (self.commits / f"{value['checkpoint_id']}.json").relative_to(self.workspace_root).as_posix(),
                "commit_sha256": value["commit_sha256"],
                "parent_checkpoint_id": value["parent_checkpoint_id"],
                "last_completed_step": value["last_completed_step"],
                "next_step": value["next_step"],
            }
            event["artifact_hash"] = stable_json_hash(event)
            events.append(event)
        event_bytes = b"".join(stable_json_bytes(event) for event in events)
        lineage = {
            "schema_version": GROUP_LINEAGE_SCHEMA,
            "commits": [
                {
                    "checkpoint_id": value["checkpoint_id"],
                    "generation": value["generation"],
                    "parent_checkpoint_id": value["parent_checkpoint_id"],
                    "commit_sha256": value["commit_sha256"],
                }
                for value in valid
            ],
            "event_ref": events_path.relative_to(self.workspace_root).as_posix(),
            "event_sha256": hashlib.sha256(event_bytes).hexdigest(),
        }
        lineage["artifact_hash"] = stable_json_hash(lineage)
        latest_value: dict[str, Any] | None = None
        if valid:
            latest = valid[-1]
            latest_value = {
                "schema_version": GROUP_LATEST_SCHEMA,
                "checkpoint_id": latest["checkpoint_id"],
                "generation": latest["generation"],
                "commit_sha256": latest["commit_sha256"],
            }
            latest_value["artifact_hash"] = stable_json_hash(latest_value)

        # Inspect existing derived views before replacement.  A syntactically
        # valid but forged/stale reference is a diagnostic, just like corrupt
        # JSON; neither form is ever used as recovery authority.
        derived_diagnostics: list[str] = []
        if latest_path.exists():
            try:
                if load_canonical_json(latest_path) != latest_value:
                    derived_diagnostics.append("latest:CONTENT_DRIFT")
            except Exception as error:
                derived_diagnostics.append(f"latest:{type(error).__name__}:{error}")
        if events_path.exists():
            try:
                if events_path.read_bytes() != event_bytes:
                    derived_diagnostics.append("events:CONTENT_DRIFT")
            except Exception as error:
                derived_diagnostics.append(f"events:{type(error).__name__}:{error}")
        if lineage_path.exists():
            try:
                if load_canonical_json(lineage_path) != lineage:
                    derived_diagnostics.append("lineage:CONTENT_DRIFT")
            except Exception as error:
                derived_diagnostics.append(f"lineage:{type(error).__name__}:{error}")

        atomic_write_bytes(events_path, event_bytes)
        atomic_write_json(lineage_path, lineage)
        if latest_value is not None:
            atomic_write_json(latest_path, latest_value)
        elif latest_path.exists():
            latest_path.unlink()
        return {
            "schema_version": "runtime.checkpoint-group-reconcile.v1",
            "valid": [value["checkpoint_id"] for value in valid],
            "invalid": diagnostics,
            "derived_diagnostics": derived_diagnostics,
            "latest_checkpoint_id": None if not valid else valid[-1]["checkpoint_id"],
            "event_ref": events_path.relative_to(self.workspace_root).as_posix(),
            "lineage_ref": lineage_path.relative_to(self.workspace_root).as_posix(),
        }


__all__ = [
    "CheckpointGroupCommit",
    "CheckpointGroupStore",
    "GROUP_COMMIT_SCHEMA",
    "GROUP_COMMIT_SCHEMA_V2",
    "GROUP_EVENT_SCHEMA",
    "GROUP_LATEST_SCHEMA",
    "GROUP_LINEAGE_SCHEMA",
    "checkpoint_state_sha256",
]
