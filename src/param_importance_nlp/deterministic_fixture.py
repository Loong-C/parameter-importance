"""Small, source-generated deterministic training fixture for Stage 0/1.

The fixture contains no opaque model binary.  A two-layer token model is
recreated from the declared seed, fixed token/label tensors are evaluated, and
the checked-in JSON stores the expected initialization, loss, and gradient
summaries.  Stage 1 can reuse the same fixture without treating it as evidence
for any importance estimator formula.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
from typing import Any, Final, Mapping

import torch

from .contracts import canonical_json_hash, load_canonical_json
from .contracts.jsonio import JSONValue


FIXTURE_SCHEMA_VERSION: Final = "stage0-deterministic-training-fixture-v1"
FIXTURE_GENERATOR_VERSION: Final = "tiny-token-lm-generator-v1"
FIXTURE_ID: Final = "stage0-tiny-token-lm-v1"


class DeterministicFixtureError(RuntimeError):
    """The checked-in deterministic fixture does not replay exactly."""


class _TinyTokenModel(torch.nn.Module):
    def __init__(self, *, seed: int, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size).double()
        self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False).double()
        # Python's MT19937 plus power-of-two denominators gives a generator
        # independent of Torch RNG/kernel releases and exactly representable
        # FP64 initialization values.
        generator = random.Random(seed)
        with torch.no_grad():
            for parameter in (self.embedding.weight, self.projection.weight):
                values = [generator.randrange(-16, 17) / 256 for _ in range(parameter.numel())]
                parameter.copy_(torch.tensor(values, dtype=torch.float64).reshape(parameter.shape))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(token_ids))


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(item) for item in tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def generate_deterministic_fixture() -> dict[str, JSONValue]:
    seed = 20260803
    vocab_size = 8
    hidden_size = 4
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    labels = torch.tensor([[2, 3, 4, -100], [3, 2, 1, -100]], dtype=torch.long)
    sample_ids = ["fixture-sequence-0000", "fixture-sequence-0001"]
    model = _TinyTokenModel(seed=seed, vocab_size=vocab_size, hidden_size=hidden_size)
    initial_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    logits = model(token_ids)
    valid = labels.ne(-100)
    selected_logits = logits[valid].gather(1, labels[valid].unsqueeze(1)).squeeze(1)
    loss_numerator = (selected_logits - 1.0).square().sum()
    effective_count = int(valid.sum().item())
    mean_loss = loss_numerator / effective_count
    mean_loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    summaries: dict[str, JSONValue] = {
        name: {
            "sum": float(tensor.double().sum().item()),
            "l2": float(torch.linalg.vector_norm(tensor.double()).item()),
            "max_abs": float(tensor.double().abs().max().item()),
        }
        for name, tensor in sorted(gradients.items())
    }
    body: dict[str, JSONValue] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_id": FIXTURE_ID,
        "generator_version": FIXTURE_GENERATOR_VERSION,
        "generator_source_ref": "src/param_importance_nlp/deterministic_fixture.py",
        "change_reason": "Initial Stage 0 G9 deterministic fixture freeze.",
        "mathematical_semantics": {
            "task": "causal_token_prediction_fixture",
            "loss": "sum_squared_error_of_labeled_token_logit_to_one_over_non_ignore_labels_divided_by_effective_count",
            "ignore_index": -100,
            "importance_mathematics_implemented": False,
        },
        "model": {
            "seed": seed,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "dtype": "float64",
            "initial_state_sha256": _tensor_digest(initial_state),
        },
        "inputs": {
            "sample_ids": sample_ids,
            "token_ids": token_ids.tolist(),
            "labels": labels.tolist(),
        },
        "expected": {
            "effective_count": effective_count,
            "loss_numerator": float(loss_numerator.detach().double().item()),
            "mean_loss": float(mean_loss.detach().double().item()),
            "gradient_state_sha256": _tensor_digest(gradients),
            "gradient_summaries": summaries,
        },
        "tolerances": {
            "absolute": 1e-7,
            "relative": 1e-6,
        },
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def validate_deterministic_fixture(path: str | Path) -> dict[str, JSONValue]:
    raw = load_canonical_json(Path(path))
    if not isinstance(raw, Mapping):
        raise DeterministicFixtureError("DETERMINISTIC_FIXTURE_OBJECT_INVALID")
    expected = dict(raw)
    declared = expected.pop("artifact_hash", None)
    if declared != canonical_json_hash(expected):
        raise DeterministicFixtureError("DETERMINISTIC_FIXTURE_HASH_INVALID")
    generated = generate_deterministic_fixture()
    if set(raw) != set(generated):
        raise DeterministicFixtureError("DETERMINISTIC_FIXTURE_FIELDS_INVALID")
    actual_expected = raw.get("expected")
    replay_expected = generated.get("expected")
    tolerances = raw.get("tolerances")
    if not isinstance(actual_expected, Mapping) or not isinstance(replay_expected, Mapping) or not isinstance(tolerances, Mapping):
        raise DeterministicFixtureError("DETERMINISTIC_FIXTURE_NUMERIC_CONTRACT_INVALID")
    absolute = float(tolerances["absolute"])
    relative = float(tolerances["relative"])
    for field in ("loss_numerator", "mean_loss"):
        if not math.isclose(
            float(actual_expected[field]),
            float(replay_expected[field]),
            abs_tol=absolute,
            rel_tol=relative,
        ):
            raise DeterministicFixtureError(f"DETERMINISTIC_FIXTURE_{field.upper()}_DRIFT")
    if actual_expected != replay_expected or dict(raw) != generated:
        raise DeterministicFixtureError("DETERMINISTIC_FIXTURE_REPLAY_DRIFT")
    return generated


__all__ = [
    "DeterministicFixtureError",
    "FIXTURE_GENERATOR_VERSION",
    "FIXTURE_ID",
    "FIXTURE_SCHEMA_VERSION",
    "generate_deterministic_fixture",
    "validate_deterministic_fixture",
]
