"""有序后台预取必须保持样本次序，并可从安全 cursor 状态恢复。"""

from __future__ import annotations

import pytest
import torch

from param_importance_nlp.providers import (
    InMemoryDatasetAdapter,
    PrefetchBatchCursor,
    TrainingMicrobatch,
    configure_batch_cursor,
)


def _dataset() -> InMemoryDatasetAdapter:
    steps = []
    for index in range(8):
        batch = TrainingMicrobatch(
            f"batch-{index}",
            {
                "input_ids": torch.tensor([[index, index + 1]], dtype=torch.int64),
                "labels": torch.tensor([index % 2], dtype=torch.int64),
            },
            (f"sample-{index}",),
            {"index": index},
        )
        steps.append((batch,))
    return InMemoryDatasetAdapter("prefetch-fixture", tuple(steps))


@pytest.mark.parametrize("persistent_workers", [False, True])
def test_prefetch_cursor_preserves_order_and_replays_pending_state(
    persistent_workers: bool,
) -> None:
    first = PrefetchBatchCursor(
        _dataset().cursor(seed=11),
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=persistent_workers,
    )
    try:
        assert first.next_microbatches()[0].sample_ids == ("sample-0",)
        assert first.next_microbatches()[0].sample_ids == ("sample-1",)
        state = first.state_dict()
        # state_dict 不得继续推进 cursor；重复快照的源位置和 pending 身份完全相同。
        repeated = first.state_dict()
        assert state["source_state"] == repeated["source_state"]
        assert [
            item[0]["sample_ids"] for item in state["pending"]  # type: ignore[index]
        ] == [
            item[0]["sample_ids"] for item in repeated["pending"]  # type: ignore[index]
        ]
        expected = [
            first.next_microbatches()[0].sample_ids[0]
            for _ in range(6)
        ]
    finally:
        first.close()

    resumed = PrefetchBatchCursor(
        _dataset().cursor(seed=11),
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=persistent_workers,
    )
    try:
        resumed.load_state_dict(state)
        actual = [
            resumed.next_microbatches()[0].sample_ids[0]
            for _ in range(6)
        ]
        assert actual == expected == [f"sample-{index}" for index in range(2, 8)]
        with pytest.raises(StopIteration):
            resumed.next_microbatches()
    finally:
        resumed.close()


def test_configure_batch_cursor_keeps_zero_worker_direct_path() -> None:
    source = _dataset().cursor(seed=7)
    assert configure_batch_cursor(
        source,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
    ) is source
