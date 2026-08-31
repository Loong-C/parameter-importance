"""Stage 2 formal handoff 使用只读 offline-shaped provider 的定向测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
)
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.experiments import stage23_task_runners as stage23
from param_importance_nlp.providers import (
    InMemoryFrozenSampleResolver,
    TorchFixedStateGradientProvider,
    build_tiny_training_fixture,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _BaseConfig:
    def section(self, name: str) -> object:
        if name != "identity":
            raise KeyError(name)
        return {"master_seed": 20260722}


class _Inputs:
    predecessor_task_ids = ("stage2.01_scope_hypotheses_and_preregistration",)
    references = (
        "runs/stage2-01/commits/preregistration.json",
        "runs/stage2-01/commits/hypothesis-contract.json",
        "runs/stage2-01/commits/gate-record.json",
    )
    binding_hash = _hash("formal-handoff-predecessors")
    artifacts = tuple(
        SimpleNamespace(artifact_hash=_hash(f"source-{index}"))
        for index in range(3)
    )

    _payloads = {
        "preregistration": {
            "sampling_stream_names": list(stage23.STREAM_NAMES),
        },
        "hypothesis_contract": {
            "statistical_unit": "independent_repetition",
        },
    }

    def payload(self, artifact_kind: str) -> object:
        return self._payloads[artifact_kind]


def test_formal_handoff_runs_read_only_invariant_with_mocked_offline_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """正式 handoff 不因 adapter 未实现而阻塞，也绝不调用 synthetic fallback。"""

    fixture = build_tiny_training_fixture(
        task_type="sequence_classification",
        seed=73,
        steps=4,
        microbatches_per_step=2,
        microbatch_size=2,
    )
    samples = {
        batch.batch_id: batch
        for step in fixture.dataset.steps
        for batch in step
    }
    resolver = InMemoryFrozenSampleResolver(
        samples,
        resolver_id="mocked-offline-stage2-handoff",
        loss_unit="sample",
        statistical_unit="frozen_microbatch",
        weight_unit="effective_sample",
        sampling_design="formal_mock_uniform_with_replacement",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    provider = TorchFixedStateGradientProvider(
        fixture.model,
        resolver,
        fixed_state_id="offline-mocked-hash-bound-state",
        output_dtype=torch.float64,
    )
    asset_hashes = (_hash("model-manifest"), _hash("data-manifest"), _hash("tokenizer"))
    evidence = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_hash("contract-freeze"),
        asset_manifest_hashes=asset_hashes,
        prerequisite_gates=(
            GateRecord(
                "stage1.G1-EXIT",
                1,
                GateStatus.PASS,
                "2026-07-22T00:00:00+00:00",
                evidence_refs=("commits/stage1-exit.json",),
            ),
        ),
    )
    context = stage23._ProviderContext(
        provider=provider,
        sample_ids=resolver.sample_ids,
        evidence=evidence,
        # 与真实 _formal_provider 的返回身份完全一致；仅模型/数据工厂替换为 tiny。
        provider_kind="offline_hf_torch_fixed_state",
        asset_manifest_hashes=asset_hashes,
    )
    calls = {"formal": 0, "local": 0}

    def formal_provider(_request: object, _root: Path):
        calls["formal"] += 1
        return context

    def forbidden_local_provider(_request: object):
        calls["local"] += 1
        raise AssertionError("formal handoff 不得回退 synthetic/local provider")

    monkeypatch.setattr(stage23, "_formal_provider", formal_provider)
    monkeypatch.setattr(stage23, "_local_provider", forbidden_local_provider)
    monkeypatch.setattr(stage23, "_predecessor_context", lambda *_args: _Inputs())

    request = SimpleNamespace(
        config=SimpleNamespace(run_intent="formal", base_config=_BaseConfig()),
        task=DEFAULT_TASK_CATALOG.get(
            "stage2.02_stage1_handoff_and_fixed_state_contract"
        ),
    )
    before = provider.state_digest()
    payloads, source_refs = stage23._run_stage2_handoff_audit(
        request,
        tmp_path,
        SimpleNamespace(),
    )
    after = provider.state_digest()

    assert calls == {"formal": 1, "local": 0}
    assert before == after
    assert source_refs == _Inputs.references
    handoff = payloads["handoff_manifest"]
    fixed_state = payloads["fixed_state_contract"]
    gate = payloads["gate_record"]
    assert handoff["scope"] == "formal"
    assert handoff["status"] == "FORMAL_CANDIDATE"
    assert handoff["formal_eligible"] is False
    assert fixed_state["status"] == "FORMAL_CANDIDATE"
    assert fixed_state["validation_evidence"]["state_unchanged"] is True
    assert fixed_state["provider_state_digest"] == before
    assert gate["gate_status"] == "NOT_RUN"
    assert gate["local_validation_status"] == "NOT_RUN"
